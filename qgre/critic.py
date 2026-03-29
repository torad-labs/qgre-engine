"""VPRM Critic — Per-Region Per-Dimension learned baseline for QGRE.

Each quality dimension gets its own small MLP that predicts the expected reward
from mean-pooled hidden states of the corresponding region. Replaces the SPO
scalar EMA baseline with a learned value function that can capture which token
patterns predict high/low quality scores.

Architecture (per quality):
    mean_pool(hidden_states[region]) → Linear(hidden_dim, 128) → ReLU → Linear(128, 128) → ReLU → Linear(128, 1)

When the segmenter finds only 1 region for a sample, that sample falls back
to SPO scalar baseline (the critic can't learn region-specific patterns
without region diversity).
"""

from __future__ import annotations

from collections import defaultdict

import torch
import torch.nn as nn


class QualityMLP(nn.Module):
    """Small MLP that predicts a single quality score from pooled hidden states."""

    def __init__(self, hidden_dim: int, intermediate_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [batch, hidden_dim] → [batch, 1]"""
        return self.net(x)


class VPRMCritic(nn.Module):
    """Per-region per-dimension critic for VPRM advantages.

    One QualityMLP per quality dimension. Each MLP takes the mean-pooled
    hidden states from its assigned region and predicts the expected reward.

    step_qualities maps step_num → [quality_names], same as the advantage estimator.
    The critic creates one MLP per unique quality name.
    """

    def __init__(
        self,
        hidden_dim: int,
        step_qualities: dict[int, list[str]],
        intermediate_dim: int = 128,
        clip_advantage: float = 5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.step_qualities = step_qualities
        self.clip_advantage = clip_advantage

        # Collect all unique quality names
        all_qualities: list[str] = []
        seen: set[str] = set()
        for step_num in sorted(step_qualities.keys()):
            for q in step_qualities[step_num]:
                if q not in seen:
                    all_qualities.append(q)
                    seen.add(q)

        self.quality_names = all_qualities

        # One MLP per quality (online heads — learn fast via MSE)
        self.heads = nn.ModuleDict({
            q: QualityMLP(hidden_dim, intermediate_dim) for q in all_qualities
        })

        # Target heads — slow-moving copy for stable advantage predictions (Polyak averaged)
        import copy
        self.target_heads = nn.ModuleDict({
            q: copy.deepcopy(self.heads[q]) for q in all_qualities
        })
        for param in self.target_heads.parameters():
            param.requires_grad = False

        # Map quality → step_num (for region assignment)
        self._quality_to_step: dict[str, int] = {}
        for step_num, qualities in step_qualities.items():
            for q in qualities:
                self._quality_to_step[q] = step_num

    @torch.no_grad()
    def update_target_network(self, tau: float = 0.01):
        """Polyak averaging: θ_target ← (1-τ)θ_target + τ*θ_online."""
        for q_name in self.quality_names:
            for op, tp in zip(self.heads[q_name].parameters(), self.target_heads[q_name].parameters()):
                if not torch.isfinite(op.data).all():
                    import warnings
                    warnings.warn(f"NaN/Inf in online head '{q_name}' — skipping Polyak update")
                    return
                tp.data.mul_(1.0 - tau).add_(op.data, alpha=tau)

    @torch.no_grad()
    def sync_target_to_online(self):
        """Hard copy online → target. Used during warmup."""
        for q_name in self.quality_names:
            for op, tp in zip(self.heads[q_name].parameters(), self.target_heads[q_name].parameters()):
                tp.data.copy_(op.data)

    def forward(
        self,
        hidden_states: torch.Tensor,
        regions: list[str],
        use_target: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict baselines for each quality from region-pooled hidden states.

        Args:
            hidden_states: [seq_len, hidden_dim] — DETACHED from training graph
            regions: [seq_len] — region label per token (from segmenter)
            use_target: if True, use slow-moving target heads for stable predictions

        Returns:
            dict mapping quality_name → predicted baseline (scalar tensor)
        """
        # Mean-pool hidden states per region (STEP_1, STEP_2, ...)
        # Extract unique step IDs in Python (avoids GPU sync from .unique().tolist())
        step_ids_present = sorted({
            int(r.split("_")[1]) for r in regions if r.startswith("STEP_")
        })
        region_ids = torch.tensor(
            [int(r.split("_")[1]) if r.startswith("STEP_") else -1 for r in regions],
            device=hidden_states.device,
        )
        region_pools: dict[str, torch.Tensor] = {}
        for step_id in step_ids_present:
            mask = (region_ids == step_id).float()
            count = mask.sum()
            if count > 0:
                pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=0) / count
                region_pools[f"STEP_{step_id}"] = pooled

        # Predict baseline for each quality using its region's pooled states
        heads = self.target_heads if use_target else self.heads
        predictions: dict[str, torch.Tensor] = {}
        for q_name in self.quality_names:
            step_num = self._quality_to_step[q_name]
            region_key = f"STEP_{step_num}"
            if region_key in region_pools:
                pooled = region_pools[region_key].unsqueeze(0)  # [1, hidden_dim]
                predictions[q_name] = heads[q_name](pooled).squeeze(0).squeeze(0)
            else:
                # Region not found — return zero baseline (SPO fallback handles this)
                predictions[q_name] = torch.tensor(0.0, device=hidden_states.device)

        return predictions

    def compute_advantages(
        self,
        hidden_states: torch.Tensor,
        regions: list[str],
        actual_rewards: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Compute per-quality advantages and critic losses.

        Args:
            hidden_states: [seq_len, hidden_dim] — DETACHED
            regions: [seq_len] — from segmenter
            actual_rewards: quality_name → actual reward score

        Returns:
            (advantages, critic_losses):
            - advantages: quality_name → clipped advantage (float)
            - critic_losses: quality_name → MSE loss tensor (for backward)
        """
        # Pool regions ONCE, then run both head sets against the same pools
        step_ids_present = sorted({
            int(r.split("_")[1]) for r in regions if r.startswith("STEP_")
        })
        region_ids = torch.tensor(
            [int(r.split("_")[1]) if r.startswith("STEP_") else -1 for r in regions],
            device=hidden_states.device,
        )
        region_pools: dict[str, torch.Tensor] = {}
        for step_id in step_ids_present:
            mask = (region_ids == step_id).float()
            count = mask.sum()
            if count > 0:
                pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=0) / count
                region_pools[f"STEP_{step_id}"] = pooled

        advantages: dict[str, float] = {}
        critic_losses: dict[str, torch.Tensor] = {}

        for q_name in self.quality_names:
            actual = actual_rewards.get(q_name, 0.0)
            step_num = self._quality_to_step[q_name]
            region_key = f"STEP_{step_num}"

            if region_key not in region_pools:
                if not hasattr(self, "_region_warned"):
                    self._region_warned = set()
                if q_name not in self._region_warned:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Critic: no {region_key} found for quality '{q_name}' — "
                        "segmenter may not be producing expected regions, critic will have zero gradient."
                    )
                    self._region_warned.add(q_name)
                advantages[q_name] = 0.0
                continue

            pooled = region_pools[region_key].unsqueeze(0)

            # Target prediction for stable advantage
            target_pred = self.target_heads[q_name](pooled).squeeze(0).squeeze(0)
            adv = actual - target_pred.detach().item()
            adv = max(-self.clip_advantage, min(self.clip_advantage, adv))
            advantages[q_name] = adv

            # Online prediction for MSE loss (online learns fast)
            online_pred = self.heads[q_name](pooled).squeeze(0).squeeze(0)
            if online_pred.requires_grad:
                reward_target = torch.tensor(actual, device=online_pred.device, dtype=online_pred.dtype)
                critic_losses[q_name] = (online_pred - reward_target) ** 2

        return advantages, critic_losses

    def compute_batch_advantages(
        self,
        batch_hidden_states: list[torch.Tensor],
        batch_regions: list[list[str]],
        batch_rewards: list[dict[str, float]],
        spo_fallback_mask: list[bool] | None = None,
    ) -> tuple[list[dict[str, float]], torch.Tensor]:
        """Batch version: compute advantages and total critic loss.

        Args:
            batch_hidden_states: list of [seq_len, hidden_dim] tensors (DETACHED)
            batch_regions: list of region label lists
            batch_rewards: list of quality_name → actual reward dicts
            spo_fallback_mask: per-sample bool — True means skip critic, use SPO

        Returns:
            (batch_advantages, total_critic_loss)
        """
        batch_advantages: list[dict[str, float]] = []
        device = batch_hidden_states[0].device if batch_hidden_states else "cpu"
        all_losses: list[torch.Tensor] = []

        for i, (hs, regions, rewards) in enumerate(
            zip(batch_hidden_states, batch_regions, batch_rewards)
        ):
            if spo_fallback_mask is not None and spo_fallback_mask[i]:
                batch_advantages.append({q: 0.0 for q in self.quality_names})
                continue

            advs, losses = self.compute_advantages(hs, regions, rewards)
            batch_advantages.append(advs)
            all_losses.extend(losses.values())

        if all_losses:
            total_loss = torch.stack(all_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=device)

        return batch_advantages, total_loss

    def state_dict_with_meta(self) -> dict:
        """Save critic state with metadata for checkpoint/resume."""
        return {
            "model_state": self.state_dict(),
            "quality_names": self.quality_names,
            "hidden_dim": self.hidden_dim,
            "step_qualities": self.step_qualities,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: str = "cpu") -> VPRMCritic:
        """Restore critic from checkpoint. Handles old checkpoints without target_heads."""
        critic = cls(
            hidden_dim=checkpoint["hidden_dim"],
            step_qualities=checkpoint["step_qualities"],
        )
        # Load with strict=False to handle old checkpoints without target_heads
        critic.load_state_dict(checkpoint["model_state"], strict=False)
        # If target_heads weren't in checkpoint, sync from online heads
        if not any(k.startswith("target_heads.") for k in checkpoint["model_state"]):
            critic.sync_target_to_online()
        critic.to(device)
        return critic
