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

from qgre.types import TrainingContext


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
        step_region_map: dict[int, int] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.step_qualities = step_qualities
        self.clip_advantage = clip_advantage
        # step_region_map: virtual_step → physical_region (e.g., {7: 2, 8: 3, 9: 6})
        self.step_region_map = step_region_map or {}

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
        # Then apply step_region_map to get the actual region step
        self._quality_to_step: dict[str, int] = {}
        self._quality_to_region: dict[str, int] = {}
        for step_num, qualities in step_qualities.items():
            for q in qualities:
                self._quality_to_step[q] = step_num
                # Apply step_region_map: virtual steps map to physical regions
                region_step = self.step_region_map.get(step_num, step_num)
                self._quality_to_region[q] = region_step

    @torch.no_grad()
    def update_target_network(self, tau: float = 0.01):
        """Polyak averaging: θ_target ← (1-τ)θ_target + τ*θ_online."""
        # RL3-009: Document that caller (compute_advantages or trainer) must call this
        for q_name in self.quality_names:
            for op, tp in zip(self.heads[q_name].parameters(), self.target_heads[q_name].parameters()):
                if not torch.isfinite(op.data).all():
                    import warnings
                    warnings.warn(f"NaN/Inf in online head '{q_name}' — skipping Polyak update")
                    return
                # Move online params to target device before Polyak update
                op_data = op.data.to(device=tp.device, dtype=tp.dtype)
                tp.data.mul_(1.0 - tau).add_(op_data, alpha=tau)

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
        ctx: TrainingContext,
        use_target: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Predict baselines for each quality from region-pooled hidden states.

        Args:
            hidden_states: [seq_len, hidden_dim] — DETACHED from training graph
            regions: [seq_len] — region label per token (from segmenter)
            ctx: TrainingContext — provides device and dtype for tensor operations
            use_target: if True, use slow-moving target heads for stable predictions

        Returns:
            dict mapping quality_name → predicted baseline (scalar tensor)
        """
        # Mean-pool hidden states per region (STEP_1, STEP_2, ...)
        # Extract unique step IDs in Python (avoids GPU sync from .unique().tolist())
        step_ids_present = sorted({
            int(r.split("_")[1]) for r in regions if r.startswith("STEP_") and "_" in r
        })
        region_ids_list = []
        for r in regions:
            if r.startswith("STEP_") and "_" in r:
                try:
                    region_ids_list.append(int(r.split("_")[1]))
                except (ValueError, IndexError):
                    region_ids_list.append(-1)
            else:
                region_ids_list.append(-1)
        region_ids = torch.tensor(region_ids_list, device=ctx.device)
        region_pools: dict[str, torch.Tensor] = {}
        for step_id in step_ids_present:
            mask = (region_ids == step_id).float()
            count = mask.sum()
            if count > 0:
                pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=0) / count
                region_pools[f"STEP_{step_id}"] = pooled

        # Predict baseline for each quality using its region's pooled states
        # Uses _quality_to_region which applies step_region_map for virtual steps
        heads = self.target_heads if use_target else self.heads
        predictions: dict[str, torch.Tensor] = {}
        for q_name in self.quality_names:
            region_step = self._quality_to_region[q_name]
            region_key = f"STEP_{region_step}"
            if region_key in region_pools:
                # C04-NUMERICAL: Cast pooled hidden states to ctx.dtype before MLP
                pooled = region_pools[region_key].to(dtype=ctx.dtype).unsqueeze(0)  # [1, hidden_dim]
                out = heads[q_name](pooled)
                predictions[q_name] = out.squeeze() if out.numel() == 1 else out.squeeze(0).squeeze(0)
            else:
                # AE5: Region missing → return None so caller skips this quality's advantage
                # (zero baseline would cause unbounded advantage = reward - 0)
                if not hasattr(self, "_region_not_found_count"):
                    self._region_not_found_count = 0
                self._region_not_found_count += 1
                import logging
                logging.getLogger(__name__).error(
                    f"RL3-003: Region {region_key} not found for quality '{q_name}'. "
                    f"Skipping advantage for this quality. Total skipped: {self._region_not_found_count}"
                )
                predictions[q_name] = None

        return predictions

    def compute_advantages(
        self,
        hidden_states: torch.Tensor,
        regions: list[str],
        actual_rewards: dict[str, float],
        ctx: TrainingContext,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Compute per-quality advantages and critic losses.

        Args:
            hidden_states: [seq_len, hidden_dim] — DETACHED
            regions: [seq_len] — from segmenter
            actual_rewards: quality_name → actual reward score
            ctx: TrainingContext — provides device and dtype for tensor operations

        Returns:
            (advantages, critic_losses):
            - advantages: quality_name → clipped advantage (float)
            - critic_losses: quality_name → MSE loss tensor (for backward)
        """
        # Pool regions ONCE, then run both head sets against the same pools
        step_ids_present = sorted({
            int(r.split("_")[1]) for r in regions if r.startswith("STEP_")
        })
        # C01-DEVICE: Use ctx.device for tensor creation
        region_ids = torch.tensor(
            [int(r.split("_")[1]) if r.startswith("STEP_") else -1 for r in regions],
            device=ctx.device,
        )
        region_pools: dict[str, torch.Tensor] = {}
        for step_id in step_ids_present:
            mask = (region_ids == step_id).float()
            count = mask.sum()
            if count > 0:
                pooled = (hidden_states * mask.unsqueeze(-1)).sum(dim=0) / count
                # RL3-004: Track NaN replacement count
                if torch.isnan(pooled).any():
                    if not hasattr(self, "_nan_replacement_count"):
                        self._nan_replacement_count = 0
                    self._nan_replacement_count += 1
                    import warnings
                    warnings.warn(f"RL3-004: NaN detected in VPRM critic region pooling for STEP_{step_id}. Returning zero. Total: {self._nan_replacement_count}")
                    pooled = torch.zeros_like(pooled)
                region_pools[f"STEP_{step_id}"] = pooled

        advantages: dict[str, float] = {}
        critic_losses: dict[str, torch.Tensor] = {}

        for q_name in self.quality_names:
            actual = actual_rewards.get(q_name, 0.0)
            # Use _quality_to_region which applies step_region_map for virtual steps
            region_step = self._quality_to_region[q_name]
            region_key = f"STEP_{region_step}"

            if region_key not in region_pools:
                # A2-5: Track occurrence count and log periodically
                if not hasattr(self, "_region_warned"):
                    self._region_warned = set()
                    self._region_warned_count = {}
                if q_name not in self._region_warned_count:
                    self._region_warned_count[q_name] = 0
                self._region_warned_count[q_name] += 1
                if q_name not in self._region_warned or self._region_warned_count[q_name] % 100 == 0:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Critic: no {region_key} found for quality '{q_name}' — "
                        f"segmenter may not be producing expected regions, critic will have zero gradient. "
                        f"(occurred {self._region_warned_count[q_name]} times)"
                    )
                    self._region_warned.add(q_name)
                advantages[q_name] = None
                continue

            # C04-NUMERICAL: Cast pooled hidden states to ctx.dtype before MLP
            pooled = region_pools[region_key].to(dtype=ctx.dtype).unsqueeze(0)

            # Target prediction for stable advantage
            target_pred = self.target_heads[q_name](pooled).squeeze(0).squeeze(0)
            # ADV-R2-5: Check for NaN before .item()
            if torch.isnan(target_pred):
                import warnings
                warnings.warn(f"NaN target prediction for quality '{q_name}' — setting advantage to 0.0")
                advantages[q_name] = 0.0
                continue
            adv = actual - target_pred.detach().item()
            adv = max(-self.clip_advantage, min(self.clip_advantage, adv))
            advantages[q_name] = adv

            # Online prediction for MSE loss (online learns fast)
            online_pred = self.heads[q_name](pooled).squeeze(0).squeeze(0)
            # ADV-R2-6: Warn when no gradient will flow
            if not online_pred.requires_grad:
                if not hasattr(self, "_no_grad_warned"):
                    self._no_grad_warned = set()
                if q_name not in self._no_grad_warned:
                    import warnings
                    warnings.warn(f"Critic head '{q_name}' has requires_grad=False — no loss recorded, critic will not learn")
                    self._no_grad_warned.add(q_name)
            if online_pred.requires_grad:
                # C01-DEVICE: Use ctx.device for tensor creation
                reward_target = torch.tensor(actual, device=ctx.device, dtype=online_pred.dtype)
                critic_losses[q_name] = (online_pred - reward_target) ** 2

        return advantages, critic_losses

    def compute_batch_advantages(
        self,
        batch_hidden_states: list[torch.Tensor],
        batch_regions: list[list[str]],
        batch_rewards: list[dict[str, float]],
        ctx: TrainingContext,
        spo_fallback_mask: list[bool] | None = None,
    ) -> tuple[list[dict[str, float]], torch.Tensor]:
        """Batch version: compute advantages and total critic loss.

        Args:
            batch_hidden_states: list of [seq_len, hidden_dim] tensors (DETACHED)
            batch_regions: list of region label lists
            batch_rewards: list of quality_name → actual reward dicts
            ctx: TrainingContext — provides device and dtype for tensor operations
            spo_fallback_mask: per-sample bool — True means skip critic, use SPO

        Returns:
            (batch_advantages, total_critic_loss)
        """
        batch_advantages: list[dict[str, float]] = []
        all_losses: list[torch.Tensor] = []

        for i, (hs, regions, rewards) in enumerate(
            zip(batch_hidden_states, batch_regions, batch_rewards)
        ):
            if spo_fallback_mask is not None and spo_fallback_mask[i]:
                batch_advantages.append({q: 0.0 for q in self.quality_names})
                continue

            advs, losses = self.compute_advantages(hs, regions, rewards, ctx)
            batch_advantages.append(advs)
            # AE-R3-02, AE-R3-04: Filter out None values from losses
            all_losses.extend([l for l in losses.values() if l is not None])

        if all_losses:
            total_loss = torch.stack(all_losses).mean()
        else:
            # C01-DEVICE: Use ctx.device for tensor creation
            total_loss = torch.tensor(0.0, device=ctx.device)

        return batch_advantages, total_loss

    def compute_advantages_from_spans(
        self,
        hidden_states: torch.Tensor,
        token_masks: dict[str, torch.Tensor],
        actual_rewards: dict[str, float],
        ctx: TrainingContext,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Compute per-quality advantages using span-based token masks.

        This is the span-aware alternative to compute_advantages(). Instead of
        pooling by STEP_X regions, it pools by each quality's token mask directly.
        This is more precise — the mask IS the quality's region.

        Args:
            hidden_states: [seq_len, hidden_dim] — DETACHED from training graph
            token_masks: quality_name → [seq_len] boolean/float mask from scored_spans
            actual_rewards: quality_name → actual reward score
            ctx: TrainingContext — provides device and dtype for tensor operations

        Returns:
            (advantages, critic_losses):
            - advantages: quality_name → clipped advantage (float)
            - critic_losses: quality_name → MSE loss tensor (for backward)
        """
        advantages: dict[str, float] = {}
        critic_losses: dict[str, torch.Tensor] = {}

        for q_name in self.quality_names:
            actual = actual_rewards.get(q_name, 0.0)

            # Get this quality's token mask
            if q_name not in token_masks:
                # Quality not in spans — skip (no gradient)
                advantages[q_name] = None
                continue

            mask = token_masks[q_name]
            # Convert to float mask for pooling (handles both bool and float masks)
            # For span masks: 1.0 = first occurrence, REPETITION_MARKER (-1.0) = repeat
            # Pool over ALL non-zero positions (both first and repeat)
            float_mask = (mask != 0).float()

            # Align mask to hidden_states length (mask may be completion-only, hidden_states may include prompt)
            hs_len = hidden_states.shape[0]
            mask_len = float_mask.shape[0]
            if mask_len < hs_len:
                # Mask is shorter — pad with zeros at start (prompt region)
                padding = torch.zeros(hs_len - mask_len, device=float_mask.device, dtype=float_mask.dtype)
                float_mask = torch.cat([padding, float_mask], dim=0)
            elif mask_len > hs_len:
                # Mask is longer — trim from end
                float_mask = float_mask[:hs_len]

            count = float_mask.sum()

            if count == 0:
                # Empty mask — skip
                advantages[q_name] = None
                continue

            # Mean-pool hidden states over this quality's span
            pooled = (hidden_states * float_mask.unsqueeze(-1)).sum(dim=0) / count

            if torch.isnan(pooled).any():
                if not hasattr(self, "_nan_span_count"):
                    self._nan_span_count = 0
                self._nan_span_count += 1
                import warnings
                warnings.warn(f"NaN in span pooling for '{q_name}'. Total: {self._nan_span_count}")
                advantages[q_name] = 0.0
                continue

            # Cast to ctx.dtype and add batch dimension
            pooled = pooled.to(dtype=ctx.dtype).unsqueeze(0)

            # Target prediction for stable advantage
            target_pred = self.target_heads[q_name](pooled).squeeze(0).squeeze(0)
            if torch.isnan(target_pred):
                import warnings
                warnings.warn(f"NaN target prediction for quality '{q_name}' (span mode)")
                advantages[q_name] = 0.0
                continue

            adv = actual - target_pred.detach().item()
            adv = max(-self.clip_advantage, min(self.clip_advantage, adv))
            advantages[q_name] = adv

            # Online prediction for MSE loss
            online_pred = self.heads[q_name](pooled).squeeze(0).squeeze(0)
            if online_pred.requires_grad:
                reward_target = torch.tensor(actual, device=ctx.device, dtype=online_pred.dtype)
                critic_losses[q_name] = (online_pred - reward_target) ** 2

        return advantages, critic_losses

    def compute_batch_advantages_from_spans(
        self,
        batch_hidden_states: list[torch.Tensor],
        batch_token_masks: list[dict[str, torch.Tensor]],
        batch_rewards: list[dict[str, float]],
        ctx: TrainingContext,
    ) -> tuple[list[dict[str, float]], torch.Tensor]:
        """Batch version of compute_advantages_from_spans.

        Args:
            batch_hidden_states: list of [seq_len, hidden_dim] tensors (DETACHED)
            batch_token_masks: list of quality_name → token_mask dicts
            batch_rewards: list of quality_name → actual reward dicts
            ctx: TrainingContext — provides device and dtype for tensor operations

        Returns:
            (batch_advantages, total_critic_loss)
        """
        batch_advantages: list[dict[str, float]] = []
        all_losses: list[torch.Tensor] = []

        for i, (hs, masks, rewards) in enumerate(
            zip(batch_hidden_states, batch_token_masks, batch_rewards)
        ):
            if not masks:
                # No masks for this sample — skip critic
                batch_advantages.append({q: None for q in self.quality_names})
                continue

            advs, losses = self.compute_advantages_from_spans(hs, masks, rewards, ctx)
            batch_advantages.append(advs)
            all_losses.extend([l for l in losses.values() if l is not None])

        if all_losses:
            total_loss = torch.stack(all_losses).mean()
        else:
            total_loss = torch.tensor(0.0, device=ctx.device)

        return batch_advantages, total_loss

    def state_dict_with_meta(self) -> dict:
        """Save critic state with metadata for checkpoint/resume."""
        return {
            "model_state": self.state_dict(),
            "quality_names": self.quality_names,
            "hidden_dim": self.hidden_dim,
            "step_qualities": self.step_qualities,
            "step_region_map": self.step_region_map,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint: dict, device: str = "cpu") -> VPRMCritic:
        """Restore critic from checkpoint. Handles old checkpoints without target_heads or step_region_map."""
        # C6: Convert step_qualities keys to int to prevent string key corruption
        step_qualities = checkpoint["step_qualities"]
        if step_qualities:
            step_qualities = {int(k): v for k, v in step_qualities.items()}

        critic = cls(
            hidden_dim=checkpoint["hidden_dim"],
            step_qualities=step_qualities,
            step_region_map=checkpoint.get("step_region_map"),  # May be None for old checkpoints
        )
        # Load with strict=False to handle old checkpoints without target_heads
        critic.load_state_dict(checkpoint["model_state"], strict=False)
        # If target_heads weren't in checkpoint, sync from online heads
        if not any(k.startswith("target_heads.") for k in checkpoint["model_state"]):
            critic.sync_target_to_online()
        critic.to(device)
        return critic
