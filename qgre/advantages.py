from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch

from qgre.segments import Segmenter, segmenter_region_count, uniform_segmenter
from qgre.types import RewardResult


def apply_frontier_amplification(
    step_advs: dict,
    step_nums: list[int],
    frontier_steps: set[int] | None,
    amplification: float,
) -> None:
    """Multiply advantages for frontier steps (blocking phase advancement).

    Modifies step_advs in place.
    """
    if frontier_steps and amplification > 0:
        for sn in step_nums:
            if sn in frontier_steps:
                step_advs[sn] = step_advs[sn] * (1.0 + amplification)


def broadcast_step_advantages_to_tokens(
    step_advs: dict[int, float | torch.Tensor],
    regions: list[str],
    region_extra_steps: dict[int, list[int]],
    sample_idx: int | None = None,
) -> torch.Tensor:
    """Broadcast per-step advantages to per-token by region label.

    Shared by SPO/GRPO path and VPRM path. Each token gets the sum of its
    region's primary step advantage + any virtual steps mapped to that region.

    Args:
        step_advs: step_num → advantage (float for VPRM, Tensor[batch] for SPO/GRPO)
        regions: per-token region labels from segmenter
        region_extra_steps: region_step → [virtual steps mapped to it]
        sample_idx: when step_advs values are batch tensors, index into them
    """
    # Pre-build label → advantage value map from unique regions (O(n_labels) string ops)
    # then do O(seq_len) dict lookups instead of per-token string parsing
    label_to_adv: dict[str, float | torch.Tensor] = {}
    for region in set(regions):
        if region.startswith("STEP_"):
            sn = int(region.split("_")[1])
            if sn in step_advs:
                val = step_advs[sn]
                primary = val[sample_idx] if sample_idx is not None else val
                contribs = [primary]
                for vs in region_extra_steps.get(sn, []):
                    if vs in step_advs:
                        v = step_advs[vs]
                        contribs.append(v[sample_idx] if sample_idx is not None else v)
                if isinstance(contribs[0], torch.Tensor):
                    label_to_adv[region] = torch.stack(contribs).sum()
                else:
                    label_to_adv[region] = sum(contribs)
        elif region == "THINK" and 0 in step_advs:
            val = step_advs[0]
            label_to_adv[region] = val[sample_idx] if sample_idx is not None else val

    seq_len = len(regions)
    token_advs = torch.zeros(seq_len)
    for t, region in enumerate(regions):
        if region in label_to_adv:
            token_advs[t] = label_to_adv[region]
    return token_advs


def build_batch_reward_tensors(
    reward_results: list[RewardResult],
) -> dict[str, torch.Tensor]:
    """Convert list[RewardResult] → dict[str, Tensor] per quality component.

    Each tensor has shape [batch_size]. Missing keys are zero-filled.
    """
    if not reward_results:
        return {}

    all_keys: set[str] = set()
    for rr in reward_results:
        all_keys.update(rr.scores.keys())

    tensors: dict[str, torch.Tensor] = {}
    for key in sorted(all_keys):
        values = [rr.scores.get(key, 0.0) for rr in reward_results]
        tensors[key] = torch.tensor(values, dtype=torch.float32)

    return tensors


def build_phase_qualities(
    step_qualities: dict[int, list[str]],
    cumulative: bool = True,
) -> dict[int, list[str]]:
    """Build phase→qualities mapping from step_qualities config.

    If cumulative=True (default QGRE behavior): phase N includes all qualities from steps 1..N.
    If cumulative=False: phase N includes only step N's qualities.
    """
    steps = sorted(step_qualities.keys())
    if cumulative:
        return {
            phase: [q for s in steps if s <= phase for q in step_qualities[s]]
            for phase in steps
        }
    else:
        return dict(step_qualities)


class QGREStepAdvantageEstimator:
    """Unified: SPO + GDPO + VPRM + QGRE phase gating.

    Configurable for any domain. Accepts:
    - step_qualities: mapping of step_num → quality names (from your reward_fn)
    - segmenter: function that splits token IDs into step regions
    - mode: "spo" (persistent value tracker) or "grpo" (group-mean baseline)
    """

    def __init__(
        self,
        lr: float = 0.1,
        mode: str = "spo",
        step_qualities: dict[int, list[str]] | None = None,
        segmenter: Segmenter | None = None,
        normalize_advantages: bool = True,
        filter_groups: bool = True,
        step_region_map: dict[int, int] | None = None,
        frontier_amplification: float = 2.0,
        var_aware: bool = True,
        var_threshold: float = 0.01,
        var_lr: float = 0.05,
        min_var_ratio: float = 0.01,
    ):
        self.lr = lr
        self.mode = mode
        self.normalize_advantages = normalize_advantages
        self.filter_groups = filter_groups
        self.frontier_amplification = frontier_amplification
        self._reward_key_checked = False
        if step_qualities is None:
            raise ValueError(
                "step_qualities is required. Pass a dict mapping step numbers to quality names, e.g.:\n"
                "  {1: ['q_format'], 2: ['q_grounding'], 3: ['q_accuracy']}\n"
                "See examples/ for domain-specific configs."
            )
        self.step_qualities = step_qualities
        self.segmenter = segmenter or uniform_segmenter
        self._step_nums = sorted(self.step_qualities.keys())
        self.V: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._step_seen: dict[int, set[int]] = defaultdict(set)
        # Target-aware aspiration gap
        self._aspiration_beta = 0.0  # Set from config via trainer
        self._aspiration_target = 0.0

        # Variance-aware baseline: track per-(prompt, step) reward variance
        self._var_aware = var_aware
        self._var_threshold = var_threshold
        self._var_lr = var_lr
        self._min_var_ratio = min_var_ratio
        self._reward_var: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(lambda: self._var_threshold))
        self._reward_mean: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        # step_region_map: virtual steps (no segmenter region) → region step whose tokens carry their advantage
        # e.g., {7: 2} means step 7's advantage is added to STEP_2 tokens
        self.step_region_map = step_region_map or {}
        # Validate: virtual steps must exist in step_qualities, region targets must too
        if self.step_region_map:
            import warnings
            for vs, rs in self.step_region_map.items():
                if vs not in self.step_qualities:
                    warnings.warn(
                        f"step_region_map key {vs} not in step_qualities — "
                        f"mapped advantage will never be computed."
                    )
                if rs not in self.step_qualities:
                    warnings.warn(
                        f"step_region_map value {rs} (target for step {vs}) not in step_qualities — "
                        f"no tokens will carry step {vs}'s advantage. Check segmenter regions."
                    )
        # Build reverse map: region_step → [virtual steps that map to it]
        self._region_extra_steps: dict[int, list[int]] = defaultdict(list)
        for virtual_step, region_step in self.step_region_map.items():
            self._region_extra_steps[region_step].append(virtual_step)

    def compute_advantages(
        self,
        batch_prompt_ids: list[int],
        batch_token_ids: list[list[int]],
        batch_reward_results: list[RewardResult],
        batch_active_qualities: list[list[str]],
        group_size: int | None = None,
        frontier_steps: set[int] | None = None,
    ) -> tuple[list[torch.Tensor], list[list[str]]]:
        """Compute per-token advantages via segment → step rewards → SPO/GRPO → GDPO → broadcast.

        Args:
            frontier_steps: Steps that block phase advancement (below mastery threshold).
                When set, these steps receive amplified advantages (frontier_amplification).
                Steps NOT in the frontier get base weight 1.0; frontier steps get
                1.0 + frontier_amplification (default: 3x total gradient pressure).

        Returns:
            (batch_advantages, batch_regions) — per-token advantages and region labels
        """
        batch_size = len(batch_token_ids)

        # First-batch invariant: check reward keys overlap with step_qualities
        if not self._reward_key_checked and batch_reward_results:
            import warnings
            all_quality_keys = set()
            for qs in self.step_qualities.values():
                all_quality_keys.update(qs)
            all_reward_keys = set()
            for rr in batch_reward_results:
                all_reward_keys.update(rr.scores.keys())
            overlap = all_quality_keys & all_reward_keys
            if not overlap:
                warnings.warn(
                    f"Reward key mismatch: reward_fn returns {sorted(all_reward_keys)} "
                    f"but step_qualities expects {sorted(all_quality_keys)}. "
                    f"All step rewards will be 0.0 — training has no signal."
                )
            self._reward_key_checked = True

        # Phase 1: Segment tokens + compute per-step rewards
        all_regions: list[list[str]] = []
        all_step_rewards: list[dict[int, float]] = []

        for i in range(batch_size):
            regions = self.segmenter(batch_token_ids[i])
            step_rews: dict[int, float] = {}
            for step_num, quality_keys in self.step_qualities.items():
                active = [k for k in quality_keys if k in batch_active_qualities[i]]
                if active:
                    vals = [batch_reward_results[i].scores.get(k, 0.0) for k in active]
                    step_rews[step_num] = sum(vals) / len(vals)
                else:
                    step_rews[step_num] = 0.0
            all_step_rewards.append(step_rews)
            all_regions.append(regions)

        # Phase 2: Per-step advantages (SPO or GRPO baseline)
        step_advs: dict[int, torch.Tensor] = {
            s: torch.zeros(batch_size) for s in self._step_nums
        }

        if self.mode == "spo":
            self._compute_spo_advantages(batch_prompt_ids, all_step_rewards, step_advs, batch_size)
        else:
            self._compute_grpo_advantages(
                batch_prompt_ids, all_step_rewards, step_advs, batch_size,
                group_size=group_size or batch_size,
            )

        # Advantage normalization — mode-dependent:
        #
        # SPO raw mode (QGRE-native): NO normalization. SPO baseline (r - V) is already
        # per-prompt centered. Adding batch normalization double-centers and erases the
        # importance hierarchy between steps. Raw advantages preserve which steps have
        # large signal (far from baseline) vs small signal (near baseline).
        #
        # GRPO+normalize: per-step mean+std normalization (GDPO-style).
        # GRPO+dr_grpo: per-step mean-only subtraction (no std division).
        for step_num in self._step_nums:
            # NaN guard: replace NaN advantages before normalization (ms-swift #8123)
            if step_advs[step_num].isnan().any():
                import warnings
                nan_count = step_advs[step_num].isnan().sum().item()
                warnings.warn(
                    f"Step {step_num}: {nan_count}/{len(step_advs[step_num])} advantages are NaN. "
                    f"Check reward_fn for NaN returns. Replacing with 0.0."
                )
                step_advs[step_num] = torch.nan_to_num(step_advs[step_num], nan=0.0)
            if self.mode == "spo":
                # SPO: use raw (r - V) advantages. The per-prompt EMA baseline IS the
                # centering mechanism. No batch normalization — this preserves the natural
                # magnitude hierarchy where bottleneck steps (far from baseline) produce
                # larger gradients than mastered steps (near baseline).
                pass
            elif self.normalize_advantages:
                mean = step_advs[step_num].mean()
                std = step_advs[step_num].std(correction=0)
                if std > 1e-8:
                    step_advs[step_num] = (step_advs[step_num] - mean) / (std + 1e-8)
                else:
                    step_advs[step_num] = step_advs[step_num] - mean
            else:
                mean = step_advs[step_num].mean()
                step_advs[step_num] = step_advs[step_num] - mean

        # Phase-aware frontier amplification: focus gradient on bottleneck steps
        apply_frontier_amplification(step_advs, self._step_nums, frontier_steps, self.frontier_amplification)

        # Phase 3: Broadcast per-step advantages to per-token by region
        batch_advantages: list[torch.Tensor] = []
        for i in range(batch_size):
            token_advs = broadcast_step_advantages_to_tokens(
                step_advs, all_regions[i], self._region_extra_steps, sample_idx=i,
            )
            batch_advantages.append(token_advs)

        return batch_advantages, all_regions

    def _compute_spo_advantages(
        self,
        batch_prompt_ids: list[int],
        all_step_rewards: list[dict[int, float]],
        step_advs: dict[int, torch.Tensor],
        batch_size: int,
    ):
        # Pre-compute batch mean per step for warm-start (PLAN.md spec: use batch mean, not sample)
        batch_means: dict[int, float] = {}
        for step_num in self._step_nums:
            rewards = [all_step_rewards[i].get(step_num, 0.0) for i in range(batch_size)]
            batch_means[step_num] = float(np.mean(rewards)) if rewards else 0.0

        for step_num in self._step_nums:
            for i in range(batch_size):
                pid = batch_prompt_ids[i]
                r = all_step_rewards[i].get(step_num, 0.0)
                v = self.V[pid][step_num]

                # Warm-start: first observation → set baseline to BATCH MEAN (not sample)
                if step_num not in self._step_seen[pid]:
                    v = batch_means[step_num]
                    self._step_seen[pid].add(step_num)

                step_advs[step_num][i] = r - v

                # Target-aware aspiration gap: preserve directional signal from shaped rewards.
                # Without this, the baseline eats the partial credit gradient (0.4 - 0.4 = 0).
                # With this, sub-target completions get proportional negative push toward target.
                if self._aspiration_beta > 0 and self._aspiration_target > 0:
                    step_advs[step_num][i] += self._aspiration_beta * (r - self._aspiration_target)

                # Variance-aware baseline: slow lr when reward is constant
                effective_lr = self.lr
                if self._var_aware:
                    # Track reward variance via running mean + EMA of squared deviation
                    r_mean = self._reward_mean[pid][step_num]
                    new_mean = r_mean + self._var_lr * (r - r_mean)
                    self._reward_mean[pid][step_num] = new_mean
                    old_var = self._reward_var[pid][step_num]
                    new_var = old_var + self._var_lr * ((r - r_mean) ** 2 - old_var)
                    self._reward_var[pid][step_num] = new_var
                    if new_var < self._var_threshold:
                        effective_lr = self.lr * max(new_var / self._var_threshold, self._min_var_ratio)

                self.V[pid][step_num] = v + effective_lr * (r - v)

    def _compute_grpo_advantages(
        self,
        batch_prompt_ids: list[int],
        all_step_rewards: list[dict[int, float]],
        step_advs: dict[int, torch.Tensor],
        batch_size: int,
        group_size: int,
    ):
        if batch_size % group_size != 0:
            raise ValueError(
                f"GRPO requires batch_size divisible by group_size, "
                f"got {batch_size} % {group_size} != 0."
            )
        num_groups = batch_size // group_size
        for step_num in self._step_nums:
            for g in range(num_groups):
                start = g * group_size
                end = start + group_size
                group_rewards = [all_step_rewards[i].get(step_num, 0.0) for i in range(start, end)]
                mean = float(np.mean(group_rewards))
                std = float(np.std(group_rewards))
                if std < 1e-8 and self.filter_groups:
                    # DAPO Dynamic Sampling: all-identical rewards → zero advantage (no signal)
                    for i in range(start, end):
                        step_advs[step_num][i] = 0.0
                else:
                    # Mean-only subtraction: outer GDPO loop handles normalization per-step.
                    # No std division here — GDPO replaces group-level std normalization.
                    for i in range(start, end):
                        step_advs[step_num][i] = all_step_rewards[i].get(step_num, 0.0) - mean

    def adapt_lr(
        self,
        kl: float,
        kl_threshold: float = 0.1,
        kl_factor: float = 2.0,
        lr_factor: float = 1.5,
        min_lr: float = 0.01,
        max_lr: float = 0.5,
    ):
        """KL-adaptive SPO learning rate (SPO paper Algorithm 1).

        Decrease lr when KL high (model drifting), increase when KL low (stagnating).
        """
        if kl > kl_factor * kl_threshold:
            self.lr = max(self.lr / lr_factor, min_lr)
        elif kl < kl_threshold / kl_factor:
            self.lr = min(self.lr * lr_factor, max_lr)

    def get_prompt_priorities(self) -> dict[int, float]:
        """Return |mean advantage| per prompt for prioritized sampling (SPO paper Section 3.2).

        Prompts with large |advantage| are sampled more often — adaptive curriculum.
        Returns dict mapping prompt_id → priority weight (higher = sample more).
        """
        priorities: dict[int, float] = {}
        for pid, steps in self.V.items():
            if not steps:
                continue
            # Use mean |V| across steps as priority proxy — prompts where the model
            # is far from baseline have high learning signal
            mean_abs_v = float(np.mean([abs(v) for v in steps.values()])) if steps else 0.0
            priorities[pid] = mean_abs_v
        return priorities

    def on_tier_advance(self, new_tier: int, prompt_tier_map: dict[int, int]):
        """Reset SPO baseline for the NEW step only — preserve learned baselines for mastered steps."""
        for pid, tier in prompt_tier_map.items():
            if tier == new_tier:
                self.V[pid][new_tier] = 0.0
                self._step_seen[pid].discard(new_tier)

    def state_dict(self) -> dict:
        return {
            "V": {pid: dict(steps) for pid, steps in self.V.items()},
            "step_seen": {pid: list(steps) for pid, steps in self._step_seen.items()},
            "reward_var": {pid: dict(steps) for pid, steps in self._reward_var.items()},
            "reward_mean": {pid: dict(steps) for pid, steps in self._reward_mean.items()},
            "lr": self.lr,
            "mode": self.mode,
        }

    def compute_advantages_with_spans(
        self,
        batch_prompt_ids: list[int],
        batch_token_ids: list[list[int]],
        batch_reward_results: list["RewardResult"],
        batch_active_qualities: list[list[str]],
        batch_token_masks: list[dict[str, torch.Tensor]],
        group_size: int | None = None,
        frontier_steps: set[int] | None = None,
    ) -> list[torch.Tensor]:
        """Compute per-token advantages using span-based token masks.

        Same per-step SPO/GRPO logic as compute_advantages(), but broadcasts
        advantages to tokens using scored_spans masks instead of section regions.

        Args:
            batch_token_masks: per-sample dict of quality_name → [seq_len] boolean mask
                (from qgre.spans.scored_spans_to_token_masks)
        """
        batch_size = len(batch_token_ids)

        # Phase 1: Compute per-step rewards (same as region-based path)
        all_step_rewards: list[dict[int, float]] = []
        for i in range(batch_size):
            step_rews: dict[int, float] = {}
            for step_num, quality_keys in self.step_qualities.items():
                active = [k for k in quality_keys if k in batch_active_qualities[i]]
                if active:
                    vals = [batch_reward_results[i].scores.get(k, 0.0) for k in active]
                    step_rews[step_num] = sum(vals) / len(vals)
                else:
                    step_rews[step_num] = 0.0
            all_step_rewards.append(step_rews)

        # Phase 2: Per-step advantages via SPO or GRPO (same as region-based path)
        step_advs: dict[int, torch.Tensor] = {
            s: torch.zeros(batch_size) for s in self._step_nums
        }
        if self.mode == "spo":
            self._compute_spo_advantages(batch_prompt_ids, all_step_rewards, step_advs, batch_size)
        else:
            self._compute_grpo_advantages(
                batch_prompt_ids, all_step_rewards, step_advs, batch_size,
                group_size=group_size or batch_size,
            )

        # Normalization (same as region-based path)
        for step_num in self._step_nums:
            if step_advs[step_num].isnan().any():
                import warnings
                nan_count = step_advs[step_num].isnan().sum().item()
                warnings.warn(f"Step {step_num}: {nan_count} advantages are NaN. Replacing with 0.0.")
                step_advs[step_num] = torch.nan_to_num(step_advs[step_num], nan=0.0)
            if self.mode == "spo":
                pass  # Raw SPO advantages
            elif self.normalize_advantages:
                mean = step_advs[step_num].mean()
                std = step_advs[step_num].std(correction=0)
                if std > 1e-8:
                    step_advs[step_num] = (step_advs[step_num] - mean) / (std + 1e-8)
                else:
                    step_advs[step_num] = step_advs[step_num] - mean
            else:
                mean = step_advs[step_num].mean()
                step_advs[step_num] = step_advs[step_num] - mean

        # Frontier amplification (same as region-based path)
        apply_frontier_amplification(step_advs, self._step_nums, frontier_steps, self.frontier_amplification)

        # Phase 3: Broadcast using span masks instead of region labels
        batch_advantages: list[torch.Tensor] = []
        for i in range(batch_size):
            seq_len = len(batch_token_ids[i])
            token_advs = torch.zeros(seq_len)
            masks = batch_token_masks[i]

            for step_num, quality_keys in self.step_qualities.items():
                step_adv = step_advs[step_num][i].item()
                if abs(step_adv) < 1e-10:
                    continue
                # Apply this step's advantage to all tokens covered by ANY of its qualities' spans
                for q_name in quality_keys:
                    if q_name in masks:
                        token_advs += step_adv * masks[q_name]

            batch_advantages.append(token_advs)

        return batch_advantages

    def load_state_dict(self, state: dict):
        self.lr = state.get("lr", self.lr)
        self.mode = state.get("mode", self.mode)
        self.V = defaultdict(lambda: defaultdict(float))
        for pid, steps in state.get("V", {}).items():
            for step_num, val in steps.items():
                self.V[int(pid)][int(step_num)] = float(val)
        self._step_seen = defaultdict(set)
        for pid, steps in state.get("step_seen", {}).items():
            self._step_seen[int(pid)] = set(int(s) for s in steps)
        self._reward_var = defaultdict(lambda: defaultdict(lambda: self._var_threshold))
        for pid, steps in state.get("reward_var", {}).items():
            for step_num, val in steps.items():
                self._reward_var[int(pid)][int(step_num)] = float(val)
        self._reward_mean = defaultdict(lambda: defaultdict(float))
        for pid, steps in state.get("reward_mean", {}).items():
            for step_num, val in steps.items():
                self._reward_mean[int(pid)][int(step_num)] = float(val)


def compute_advantages_vprm(
    critic,  # VPRMCritic
    hidden_states: torch.Tensor,
    regions: list[str],
    reward_result: "RewardResult",
    step_qualities: dict[int, list[str]],
    active_qualities: list[str],
    step_region_map: dict[int, int] | None = None,
    frontier_steps: set[int] | None = None,
    frontier_amplification: float = 2.0,
    min_regions: int = 2,
    aspiration_beta: float = 0.0,
    aspiration_target: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute per-token advantages using VPRM critic for a single sample.

    Returns:
        (token_advantages, critic_loss, used_critic):
        - token_advantages: [seq_len] per-token advantages
        - critic_loss: scalar MSE loss for critic training
        - used_critic: True if critic was used, False if SPO fallback
    """
    seq_len = hidden_states.shape[0]
    device = hidden_states.device

    # Check if enough regions for critic — else SPO fallback
    n_regions = segmenter_region_count(regions)
    if n_regions < min_regions:
        return (
            torch.zeros(seq_len, device=device),
            torch.tensor(0.0, device=device),
            False,
        )

    # Get actual rewards per quality
    actual_rewards = {k: reward_result.scores.get(k, 0.0) for k in active_qualities}

    # Compute advantages via critic (hidden states must be DETACHED)
    advs_dict, critic_losses = critic.compute_advantages(
        hidden_states.detach(), regions, actual_rewards,
    )

    # Build reverse map: region_step → [virtual steps that map to it]
    region_extra_steps: dict[int, list[int]] = defaultdict(list)
    if step_region_map:
        for vs, rs in step_region_map.items():
            region_extra_steps[rs].append(vs)

    # Broadcast per-quality advantages to per-token by region
    step_nums = sorted(step_qualities.keys())
    # Build per-step advantages from quality advantages
    step_advs: dict[int, float] = {}
    for step_num in step_nums:
        qualities = [q for q in step_qualities[step_num] if q in active_qualities]
        if qualities:
            vals = [advs_dict.get(q, 0.0) for q in qualities]
            step_advs[step_num] = sum(vals) / len(vals)
        else:
            step_advs[step_num] = 0.0

    # Aspiration gap: preserve shaped reward gradient even when VPRM replaces SPO advantages
    if aspiration_beta > 0 and aspiration_target > 0:
        for step_num in step_nums:
            qualities = [q for q in step_qualities[step_num] if q in active_qualities]
            if qualities:
                step_reward = sum(reward_result.scores.get(q, 0.0) for q in qualities) / len(qualities)
                step_advs[step_num] += aspiration_beta * (step_reward - aspiration_target)

    apply_frontier_amplification(step_advs, step_nums, frontier_steps, frontier_amplification)

    # Broadcast to tokens
    token_advantages = broadcast_step_advantages_to_tokens(
        step_advs, regions, region_extra_steps,
    ).to(device)

    # Total critic loss
    if critic_losses:
        total_critic_loss = torch.stack(list(critic_losses.values())).mean()
    else:
        total_critic_loss = torch.tensor(0.0, device=device)

    return token_advantages, total_critic_loss, True
