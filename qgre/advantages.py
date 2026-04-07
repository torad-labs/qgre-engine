from __future__ import annotations

import logging
import math
import warnings
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
import torch

from qgre.attention_bonds import (
    apply_importance_constraint,
    compute_confidence_gate,
)
from qgre.segments import Segmenter, segmenter_region_count, uniform_segmenter
from qgre.spans import REPETITION_MARKER


if TYPE_CHECKING:
    from qgre.types import PromptContext, RewardResult, TrainingContext


_logger = logging.getLogger(__name__)

# Penalty multiplier for repeated spans (applied to abs(q_adv))
# 1.5 means repeating gives -1.5 * |q_adv| regardless of sign
REPETITION_PENALTY_MULTIPLIER = 1.5


# =============================================================================
# EGRS: Entropy-Gated Reinforcement System
# =============================================================================


def compute_span_correctness(
    reward_result: RewardResult,
    step_qualities: dict[int, list[str]],
    threshold: float = 0.5,
) -> dict[int, bool]:
    """Map step_num -> is_correct based on quality scores.

    A span is "correct" if ALL its quality scores meet or exceed the threshold.
    This matches the QGRE philosophy: we want mastery, not partial credit.

    Args:
        reward_result: Reward function output with per-quality scores.
        step_qualities: Mapping of step_num -> list of quality names.
        threshold: Score threshold for "correct" classification.

    Returns:
        Dict mapping step_num -> bool (True if correct).
    """
    result = {}
    for step_num, qualities in step_qualities.items():
        if not qualities:
            # No qualities defined = assume correct (no signal)
            # Warn once per step to surface configuration issues
            if not hasattr(compute_span_correctness, "_empty_warned"):
                compute_span_correctness._empty_warned = set()  # type: ignore[attr-defined]
            if step_num not in compute_span_correctness._empty_warned:  # type: ignore[attr-defined]
                compute_span_correctness._empty_warned.add(step_num)  # type: ignore[attr-defined]
                warnings.warn(
                    f"EGRS: step {step_num} has empty qualities list in step_qualities. "
                    "Step will always be treated as 'correct' (Q2 or Q1). Check config.",
                    stacklevel=2,
                )
            result[step_num] = True
            continue
        scores = [reward_result.scores.get(q, 0.0) for q in qualities]
        # All scores must meet threshold
        result[step_num] = all(s >= threshold for s in scores)
    return result


def apply_egrs_matrix(
    token_advantages: torch.Tensor,
    regions: list[str],
    token_entropy: torch.Tensor,
    step_correctness: dict[int, bool],
    entropy_threshold: float = 0.5,
    gate_temperature: float = 0.1,
    exploration_weight: float = 0.1,
    importance: torch.Tensor | None = None,
    eric_strength: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, set[tuple[int, int]]]:
    """Apply EGRS 2x2 matrix to token advantages.

    Classifies each token into one of 4 quadrants and applies appropriate treatment:
    - Q1 (uncertain+correct): Scale advantage by confidence_gate, then apply ERIC dampening
    - Q2 (confident+correct): Zero advantage (already learned)
    - Q3 (confident+wrong): Zero advantage, set entropy adjustment
    - Q4 (uncertain+wrong): Zero advantage, flag for hint

    Args:
        token_advantages: Per-token advantages [seq_len].
        regions: Per-token region labels from segmenter.
        token_entropy: Normalized entropy [seq_len] in [0, 1].
        step_correctness: step_num -> is_correct mapping.
        entropy_threshold: Threshold for confident/uncertain classification.
        gate_temperature: Sigmoid temperature for soft gating.
        exploration_weight: Lambda for entropy bonus (Q3).
        importance: Optional per-token importance for ERIC dampening [seq_len].
                   If provided, Q1 advantages are dampened by (1 + eric_strength * importance).
        eric_strength: ERIC dampening multiplier (1.0 = standard).

    Returns:
        Tuple of:
        - Modified token advantages [seq_len]
        - Entropy adjustments [seq_len] (non-zero for Q3 tokens)
        - Hint flags: set of (step_num, token_idx) for Q4 tokens
    """
    seq_len = len(regions)
    device = token_advantages.device
    dtype = token_advantages.dtype

    # Compute confidence gate: ~0 when confident, ~1 when uncertain
    confidence_gate = compute_confidence_gate(
        token_entropy,
        threshold=entropy_threshold,
        temperature=gate_temperature,
    )

    # Initialize outputs
    modified_advs = token_advantages.clone()
    entropy_adjustments = torch.zeros(seq_len, device=device, dtype=dtype)
    hint_flags: set[tuple[int, int]] = set()

    # A-1: Clamp iteration to min of all tensor lengths to prevent IndexError
    max_iter = min(len(regions), len(token_entropy), len(token_advantages))
    if max_iter < seq_len:
        warnings.warn(
            f"A-1: Region/entropy/advantage length mismatch. "
            f"regions={len(regions)}, entropy={len(token_entropy)}, advs={len(token_advantages)}. "
            f"Processing only first {max_iter} tokens.",
            stacklevel=2,
        )

    for t in range(max_iter):
        region = regions[t]
        # Parse step number from region label
        if region.startswith("STEP_"):
            try:
                step_num = int(region.split("_")[1])
            except (IndexError, ValueError):
                # Track malformed regions - surface issue without spamming
                if not hasattr(apply_egrs_matrix, "_malformed_warned"):
                    apply_egrs_matrix._malformed_warned = set()  # type: ignore[attr-defined]
                if region not in apply_egrs_matrix._malformed_warned:  # type: ignore[attr-defined]
                    apply_egrs_matrix._malformed_warned.add(region)  # type: ignore[attr-defined]
                    warnings.warn(
                        f"EGRS: Malformed region label '{region}' at token {t}. "
                        "Token will receive no EGRS treatment. Check segmenter output.",
                        stacklevel=2,
                    )
                continue
        elif region == "THINK":
            step_num = 0
        else:
            continue  # FORMAT or other regions - no EGRS treatment

        # Look up correctness - warn if step not found (silent default is dangerous)
        if step_num not in step_correctness:
            warnings.warn(
                f"EGRS: step {step_num} not in step_correctness dict. "
                f"Defaulting to correct=True. Check step_qualities config.",
                stacklevel=2,
            )
        correct = step_correctness.get(step_num, True)
        # Confidence from gate: low gate = confident, high gate = uncertain
        gate_val = confidence_gate[t].item()
        confident = gate_val < 0.5

        if correct:
            if confident:
                # Q2: Confident + Correct → Already learned, zero advantage
                modified_advs[t] = 0.0
            else:
                # Q1: Uncertain + Correct → Reinforce
                # Gate: ~1 when uncertain (should reinforce), ~0 when confident (don't reinforce)
                # Scale advantage by gate value
                scaled_adv = token_advantages[t] * gate_val
                # Apply ERIC dampening if importance provided (prevents cascade destabilization)
                if importance is not None:
                    clamped_strength = min(eric_strength, 10.0)
                    dampening = 1.0 + clamped_strength * importance[t].item()
                    scaled_adv = scaled_adv / dampening
                modified_advs[t] = scaled_adv
        else:
            # Wrong answer - no reinforcement
            modified_advs[t] = 0.0
            if confident:
                # Q3: Confident + Wrong → Shake confidence via entropy boost
                entropy_adjustments[t] = exploration_weight
            else:
                # Q4: Uncertain + Wrong → Flag for hint injection
                hint_flags.add((step_num, t))

    return modified_advs, entropy_adjustments, hint_flags


def _validate_region_step_coverage(
    regions: list[str],
    step_qualities: dict[int, list[str]],
    sample_idx: int | None = None,
) -> None:
    """Validate that all STEP_N regions in output have corresponding step_qualities entries.

    Warns if a step region has no qualities defined — this means the segmenter found a step
    that won't receive any advantage signal. This is usually a config mismatch but may be
    intentional for partial step coverage.
    """
    step_regions = {r for r in set(regions) if r.startswith("STEP_")}
    for region in step_regions:
        try:
            step_num = int(region.split("_")[1])
        except (IndexError, ValueError):
            continue  # Malformed region name — skip
        if step_num not in step_qualities:
            sample_info = f" (sample {sample_idx})" if sample_idx is not None else ""
            warnings.warn(
                f"Region '{region}' found in segmenter output{sample_info} but step {step_num} "
                f"not in step_qualities. Available steps: {sorted(step_qualities.keys())}. "
                f"This step will receive 0.0 advantage.",
                stacklevel=2,
            )


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
            # ADV-R1-7: Check if step exists in step_advs before amplifying
            if sn in frontier_steps and sn in step_advs:
                step_advs[sn] = step_advs[sn] * (1.0 + amplification)


def broadcast_step_advantages_to_tokens(
    step_advs: dict[int, float | torch.Tensor],
    regions: list[str],
    region_extra_steps: dict[int, list[int]],
    sample_idx: int | None = None,
    ctx: TrainingContext | None = None,
    bond_strength: torch.Tensor | None = None,
    constraint_strength: float = 1.0,
) -> torch.Tensor:
    """Broadcast per-step advantages to per-token by region label.

    Shared by SPO/GRPO path and VPRM path. Each token gets the sum of its
    region's primary step advantage + any virtual steps mapped to that region.

    Args:
        step_advs: step_num → advantage (float for VPRM, Tensor[batch] for SPO/GRPO)
        regions: per-token region labels from segmenter
        region_extra_steps: region_step → [virtual steps mapped to it]
        sample_idx: when step_advs values are batch tensors, index into them
        ctx: TrainingContext for device and dtype
        bond_strength: per-token attention bond strength (0-1), constrains advantages
        constraint_strength: multiplier for bond strength effect (default 1.0)
    """
    # Pre-build label → advantage value map from unique regions (O(n_labels) string ops)
    # then do O(seq_len) dict lookups instead of per-token string parsing
    label_to_adv: dict[str, float | torch.Tensor] = {}
    for region in set(regions):
        if region.startswith("STEP_"):
            sn = int(region.split("_")[1])
            if sn in step_advs:
                val = step_advs[sn]
                # A-2: Bounds check before indexing
                if sample_idx is not None:
                    if isinstance(val, torch.Tensor) and sample_idx >= val.shape[0]:
                        warnings.warn(
                            f"A-2: sample_idx {sample_idx} >= step_advs[{sn}].shape[0] ({val.shape[0]}). Skipping.",
                            stacklevel=2,
                        )
                        continue
                    primary = val[sample_idx]  # type: ignore[index]
                else:
                    primary = val
                contribs = [primary]
                for vs in region_extra_steps.get(sn, []):
                    if vs in step_advs:
                        v = step_advs[vs]
                        # A-2: Bounds check for virtual steps too
                        if sample_idx is not None:
                            if isinstance(v, torch.Tensor) and sample_idx >= v.shape[0]:
                                warnings.warn(
                                    f"A-2: sample_idx {sample_idx} >= step_advs[{vs}].shape[0] ({v.shape[0]}). Skipping.",
                                    stacklevel=2,
                                )
                                continue
                            contribs.append(v[sample_idx])  # type: ignore[index]
                        else:
                            contribs.append(v)
                # Convert all to tensors on consistent device before torch.stack
                if not contribs:
                    continue
                if isinstance(contribs[0], torch.Tensor):
                    ref_device = contribs[0].device
                    contribs_tensors = [
                        c.to(ref_device)
                        if isinstance(c, torch.Tensor)
                        else torch.tensor(c, device=ref_device)
                        for c in contribs
                    ]
                    label_to_adv[region] = torch.stack(contribs_tensors).sum()
                else:
                    ref_device = ctx.device if ctx is not None else "cpu"
                    contribs_tensors = [
                        c if isinstance(c, torch.Tensor) else torch.tensor(c, device=ref_device)
                        for c in contribs
                    ]
                    label_to_adv[region] = torch.stack(contribs_tensors).sum()
        elif region == "THINK" and 0 in step_advs:
            val = step_advs[0]
            label_to_adv[region] = val[sample_idx] if sample_idx is not None else val  # type: ignore[index]

    seq_len = len(regions)
    # Use ctx.device if available, otherwise infer from step_advs tensors
    device = ctx.device if ctx is not None else None
    if device is None:
        for val in step_advs.values():
            if isinstance(val, torch.Tensor):
                device = val.device
                break
        if device is None:
            device = "cpu"

    # C05-SHAPE: Validate ctx.device matches input tensor device before ops
    if ctx is not None:
        for step_num, val in step_advs.items():
            if isinstance(val, torch.Tensor) and val.device != ctx.device:
                raise ValueError(
                    f"C05-SHAPE: step_advs[{step_num}] on {val.device} but ctx.device={ctx.device}",
                )

    # C04-NUMERICAL: Use ctx.dtype for advantage tensor creation
    dtype = ctx.dtype if ctx is not None else torch.float32
    token_advs = torch.zeros(seq_len, device=device, dtype=dtype)
    for t, region in enumerate(regions):
        if region in label_to_adv:
            token_advs[t] = label_to_adv[region]

    # Apply importance constraint if bond_strength (importance) provided
    if bond_strength is not None:
        # A-4: Move to correct device before validation
        if bond_strength.device != device:
            bond_strength = bond_strength.to(device)
        # Validate shape consistency
        if bond_strength.shape != token_advs.shape:
            raise ValueError(
                f"C05-SHAPE: bond_strength shape {bond_strength.shape} does not match token_advs shape {token_advs.shape}",
            )
        # Apply advantage-gated importance constraint:
        # - Positive advantage + high importance → dampen (protect correct anchors)
        # - Negative advantage + high importance → NO dampen (correct confident mistakes)
        token_advs = apply_importance_constraint(token_advs, bond_strength, constraint_strength)

    return token_advs


def build_batch_reward_tensors(
    reward_results: list[RewardResult],
    device: str | torch.device | None = None,
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
        tensors[key] = torch.tensor(values, dtype=torch.float32, device=device)

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
            phase: [q for s in steps if s <= phase for q in step_qualities[s]] for phase in steps
        }
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
        staleness_window: int = 50,
        baseline_prior: float = 0.5,
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
                "See examples/ for domain-specific configs.",
            )
        self.step_qualities = step_qualities
        self.segmenter = segmenter or uniform_segmenter
        self._step_nums = sorted(self.step_qualities.keys())
        # Per-quality SPO baselines (keyed by quality name, not step number)
        self.V: dict[int, dict[int | str, float]] = defaultdict(lambda: defaultdict(float))
        self.V_last_seen: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._quality_seen: dict[int, set[str]] = defaultdict(set)
        # Legacy step-seen for backward compat (deprecated)
        self._step_seen: dict[int, set[int]] = defaultdict(set)
        # Target-aware aspiration gap
        self._aspiration_beta = 0.0  # Set from config via trainer
        self._aspiration_target = 0.0
        self._advantage_scale = 1.0  # Set from config via trainer
        # AE-R3-01: Initialize clip_advantage for token-level clipping
        self._clip_advantage = 10.0  # Default, overridden by config

        # Variance-aware baseline: track per-(prompt, quality) reward variance
        self._var_aware = var_aware
        self._var_threshold = var_threshold
        self._var_lr = var_lr
        self._min_var_ratio = min_var_ratio
        self._reward_var: dict[int, dict[int | str, float]] = defaultdict(
            lambda: defaultdict(lambda: self._var_threshold)
        )
        self._reward_mean: dict[int, dict[int | str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # Staleness decay for sparse qualities
        self._staleness_window = staleness_window
        self._baseline_prior = baseline_prior
        self._current_step = 0  # Updated externally by trainer

        # Auto-reset on distribution shift: track divergence per (prompt, step)
        # When |r - baseline| > threshold for N consecutive observations, reset baseline
        self._divergence_window: dict[int, dict[int, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._divergence_threshold = 0.3  # Absolute divergence threshold (0.85→0.2 = 0.65)
        self._divergence_window_size = 3  # Consecutive observations before reset
        self._divergence_cleanup_interval = 100  # Steps between cleanup
        self._divergence_last_cleanup = 0

        # step_region_map: virtual steps (no segmenter region) → region step whose tokens carry their advantage
        # e.g., {7: 2} means step 7's advantage is added to STEP_2 tokens
        self.step_region_map = step_region_map or {}
        # Validate: virtual steps must exist in step_qualities, region targets must too
        if self.step_region_map:
            for vs, rs in self.step_region_map.items():
                if vs not in self.step_qualities:
                    warnings.warn(
                        f"step_region_map key {vs} not in step_qualities — "
                        f"mapped advantage will never be computed.",
                        stacklevel=2,
                    )
                if rs not in self.step_qualities:
                    warnings.warn(
                        f"step_region_map value {rs} (target for step {vs}) not in step_qualities — "
                        f"no tokens will carry step {vs}'s advantage. Check segmenter regions.",
                        stacklevel=2,
                    )
            # Additional validation: warn if mapped regions don't exist in actual segmenter output
            # This is done per-sample in compute_advantages when regions are available
        # Build reverse map: region_step → [virtual steps that map to it]
        self._region_extra_steps: dict[int, list[int]] = defaultdict(list)
        for virtual_step, region_step in self.step_region_map.items():
            self._region_extra_steps[region_step].append(virtual_step)

    def get_baseline(self, prompt_id: int, quality_name: str) -> float:
        """Get baseline with staleness decay for sparse qualities.

        Qualities that haven't been seen in staleness_window steps decay
        toward baseline_prior to prevent stale baselines from anchoring advantage.
        """
        V = self.V[prompt_id][quality_name]
        last_seen = self.V_last_seen[prompt_id][quality_name]

        if last_seen == 0:
            return self._baseline_prior  # Never seen → use prior

        # Guard: if current_step not yet set or negative staleness, use prior
        if self._current_step == 0:
            return self._baseline_prior
        steps_since = self._current_step - last_seen
        if steps_since < 0:
            # Checkpoint restored with future last_seen — use prior
            warnings.warn(
                f"Negative staleness {steps_since} for prompt {prompt_id} quality {quality_name}. "
                f"current_step={self._current_step}, last_seen={last_seen}. Using prior.",
                stacklevel=2,
            )
            return self._baseline_prior

        if steps_since > self._staleness_window:
            # Continuous exponential decay (no discrete jumps)
            decay = 0.9 ** (steps_since / self._staleness_window)
            return V * decay + self._baseline_prior * (1 - decay)

        return V

    def update_baseline(
        self,
        prompt_id: int,
        quality_name: str,
        reward: float,
        effective_lr: float,
    ) -> None:
        """Update baseline and track last-seen step."""
        V = self.V[prompt_id][quality_name]
        self.V[prompt_id][quality_name] = V + effective_lr * (reward - V)
        self.V_last_seen[prompt_id][quality_name] = self._current_step

    def set_current_step(self, step: int) -> None:
        """Called by trainer at start of each step."""
        self._current_step = step
        # Periodic cleanup of stale divergence_window entries
        if step - self._divergence_last_cleanup >= self._divergence_cleanup_interval:
            self._cleanup_divergence_window()
            self._divergence_last_cleanup = step

    def _cleanup_divergence_window(self) -> None:
        """Remove stale entries from divergence_window to prevent memory leak."""
        stale_pids = []
        for pid in self._divergence_window:
            # Remove prompt if it hasn't been seen in staleness_window steps
            if pid not in self.V or not self.V[pid]:
                stale_pids.append(pid)
        for pid in stale_pids:
            del self._divergence_window[pid]
        if stale_pids:
            _logger.info(f"Cleaned {len(stale_pids)} stale prompts from divergence_window")

    def compute_advantages(
        self,
        batch_prompt_ids: list[int],
        batch_token_ids: list[list[int]],
        batch_reward_results: list[RewardResult],
        batch_active_qualities: list[list[str]],
        group_size: int | None = None,
        frontier_steps: set[int] | None = None,
        batch_contexts: list[PromptContext] | None = None,
        ctx: TrainingContext | None = None,
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
                    f"All step rewards will be 0.0 — training has no signal.",
                    stacklevel=2,
                )
            self._reward_key_checked = True

        # Phase 1: Segment tokens + compute per-step rewards
        all_regions: list[list[str]] = []
        all_step_rewards: list[dict[int, float]] = []

        for i in range(batch_size):
            regions = self.segmenter(batch_token_ids[i])
            # Validate segmenter output matches step_qualities config (once per init, not per batch)
            if not getattr(self, "_region_validated", False):
                _validate_region_step_coverage(regions, self.step_qualities, sample_idx=i)
                self._region_validated = True
            step_rews: dict[int, float] = {}
            for step_num, quality_keys in self.step_qualities.items():
                active = [k for k in quality_keys if k in batch_active_qualities[i]]
                if active:
                    vals = [batch_reward_results[i].scores.get(k, 0.0) for k in active]
                    step_rews[step_num] = sum(vals) / max(len(vals), 1)
                # else: skip — inactive qualities produce NO advantage signal,
                # not 0.0 vs old baseline which generates catastrophic negatives
            all_step_rewards.append(step_rews)
            all_regions.append(regions)

        # Phase 2: Per-step advantages (SPO or GRPO baseline)
        # C05-SHAPE: Create on ctx.device to avoid device mismatch in broadcast
        device = ctx.device if ctx is not None else "cpu"
        step_advs: dict[int, torch.Tensor] = {
            s: torch.zeros(batch_size, device=device) for s in self._step_nums
        }

        if self.mode == "spo":
            self._compute_spo_advantages(
                batch_prompt_ids,
                all_step_rewards,
                step_advs,
                batch_size,
                batch_contexts=batch_contexts,
            )
        else:
            self._compute_grpo_advantages(
                batch_prompt_ids,
                all_step_rewards,
                step_advs,
                batch_size,
                group_size=group_size or batch_size,
            )

        # Advantage normalization — mode-dependent:
        # SPO raw: no normalization (EMA baseline is the centering).
        # GRPO+normalize: per-step mean+std (GDPO-style).
        # GRPO+dr_grpo: per-step mean-only (no std division).
        self._normalize_step_advantages(step_advs)

        # Phase-aware frontier amplification: focus gradient on bottleneck steps
        apply_frontier_amplification(
            step_advs, self._step_nums, frontier_steps, self.frontier_amplification
        )

        # Phase 3: Broadcast per-step advantages to per-token by region
        batch_advantages: list[torch.Tensor] = []
        for i in range(batch_size):
            token_advs = broadcast_step_advantages_to_tokens(
                step_advs,  # type: ignore[arg-type]
                all_regions[i],
                self._region_extra_steps,
                sample_idx=i,
                ctx=ctx,
            )
            batch_advantages.append(token_advs)

        return batch_advantages, all_regions

    def _normalize_step_advantages(self, step_advs: dict[int, torch.Tensor]):
        """NaN guard + mode-dependent normalization. Shared by region and span paths."""
        for step_num in self._step_nums:
            # NaN guard (ms-swift #8123): reward_fn can return NaN on malformed completions
            if step_advs[step_num].isnan().any():
                nan_count = step_advs[step_num].isnan().sum().item()
                warnings.warn(
                    f"Step {step_num}: {nan_count}/{len(step_advs[step_num])} advantages are NaN. "
                    f"Check reward_fn for NaN returns. Replacing with 0.0.",
                    stacklevel=2,
                )
                step_advs[step_num] = torch.nan_to_num(step_advs[step_num], nan=0.0)
            if self.mode == "spo":
                # SPO raw: no normalization. The per-prompt EMA baseline IS the centering
                # mechanism. Batch normalization would double-center and erase the importance
                # hierarchy between steps (bottleneck steps should produce larger gradients).
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

    def _compute_spo_advantages(
        self,
        batch_prompt_ids: list[int],
        all_step_rewards: list[dict[int, float]],
        step_advs: dict[int, torch.Tensor],
        batch_size: int,
        batch_contexts: list[PromptContext] | None = None,
    ):
        # ADV-R1-2: Validate batch_contexts length if provided
        if batch_contexts is not None and len(batch_contexts) < batch_size:
            raise ValueError(
                f"batch_contexts length ({len(batch_contexts)}) < batch_size ({batch_size}). "
                "Must provide context for all samples.",
            )
        # Pre-compute batch mean per step for warm-start (PLAN.md spec: use batch mean, not sample)
        # Only include samples that HAVE this step (not inactive/skipped)
        batch_means: dict[int, float] = {}
        for step_num in self._step_nums:
            rewards = [
                all_step_rewards[i][step_num]
                for i in range(batch_size)
                if step_num in all_step_rewards[i]
            ]
            batch_means[step_num] = float(np.mean(rewards)) if rewards else 0.0

        for step_num in self._step_nums:
            for i in range(batch_size):
                # A-6: Bounds check BEFORE any array access
                if i >= len(batch_prompt_ids):
                    raise IndexError(
                        f"A-6: batch index {i} out of range for batch_prompt_ids (len={len(batch_prompt_ids)}). "
                        f"batch_size={batch_size}, step_num={step_num}",
                    )

                # Skip inactive qualities — no advantage signal, no baseline update
                if step_num not in all_step_rewards[i]:
                    step_advs[step_num][i] = 0.0
                    continue

                ctx = batch_contexts[i] if batch_contexts and i < len(batch_contexts) else None
                pid = ctx.prompt_id if ctx is not None else batch_prompt_ids[i]
                r = all_step_rewards[i][step_num]
                v = self.V[pid][step_num]

                # Warm-start: first observation → set baseline to BATCH MEAN (not sample)
                is_first_observation = step_num not in self._step_seen[pid]
                if is_first_observation:
                    v = batch_means[step_num]
                    self._step_seen[pid].add(step_num)

                # Perfect score (1.0) = zero advantage. Nothing to learn — don't waste
                # gradient reinforcing specific tokens. Imperfect = push toward 1.0.
                if r >= 1.0:
                    step_advs[step_num][i] = 0.0
                else:
                    step_advs[step_num][i] = r - v
                    # Aspiration gap: push toward perfection (1.0).
                    # ADV-R2-1: Safe attribute access with getattr
                    if self._aspiration_beta > 0:
                        ctx = (
                            batch_contexts[i]
                            if batch_contexts
                            and i < len(batch_contexts)
                            and batch_contexts[i] is not None
                            else None
                        )
                        warmup = getattr(ctx, "aspiration_warmup", 1.0) if ctx else 1.0
                        warmup = max(
                            0.0, min(1.0, warmup)
                        )  # Clamp to [0, 1] to prevent double application
                        step_advs[step_num][i] += self._aspiration_beta * warmup * (r - 1.0)

                # Auto-reset on distribution shift: detect sustained divergence
                # Skip on first observation (baseline not calibrated yet)
                if not is_first_observation:
                    divergence = abs(r - v)
                    self._divergence_window[pid][step_num].append(divergence)
                    # Keep only last N observations
                    if len(self._divergence_window[pid][step_num]) > self._divergence_window_size:
                        self._divergence_window[pid][step_num].pop(0)

                    # Check if all recent observations exceed threshold
                    window = self._divergence_window[pid][step_num]
                    if len(window) >= self._divergence_window_size and all(
                        d > self._divergence_threshold for d in window
                    ):
                        # Distribution shift detected — reset this prompt's baselines
                        import logging

                        _logger = logging.getLogger("qgre.advantages")
                        _logger.warning(
                            f"[AUTO-RESET] Distribution shift detected for prompt {pid}, step {step_num}: "
                            f"divergence {window} all > {self._divergence_threshold}. "
                            f"Resetting baselines (baseline={v:.3f}, new_score={r:.3f})",
                        )
                        self._reset_prompt_baselines(pid)
                        # Clear divergence window after reset
                        if pid in self._divergence_window:
                            self._divergence_window[pid].clear()

                # Variance-aware baseline: slow lr when reward is constant
                effective_lr = self.lr
                if self._var_aware:
                    # Track reward variance via running mean + EMA of squared deviation
                    r_mean = self._reward_mean[pid][step_num]
                    new_mean = r_mean + self._var_lr * (r - r_mean)
                    self._reward_mean[pid][step_num] = new_mean
                    old_var = self._reward_var[pid][step_num]
                    # ADV-R2-2: Use new_mean for variance (not stale r_mean)
                    # Clamp variance BEFORE computing effective_lr
                    new_var = old_var + self._var_lr * ((r - new_mean) ** 2 - old_var)
                    if new_var < 0:
                        # RSP-005: Negative variance indicates numerical instability in EMA
                        warnings.warn(
                            f"RSP-005: Negative variance {new_var:.6f} for prompt {pid} step {step_num}. "
                            f"old_var={old_var:.6f}, r={r:.4f}, r_mean={r_mean:.4f}, new_mean={new_mean:.4f}. "
                            "Clamping to 0 — baseline may drift.",
                            stacklevel=2,
                        )
                        new_var = 0.0
                    self._reward_var[pid][step_num] = max(0.0, new_var)
                    # Skip variance-aware LR on first sample (when old_var == threshold and r_mean == 0)
                    if is_first_observation:
                        effective_lr = self.lr
                    elif new_var < self._var_threshold:
                        effective_lr = self.lr * max(
                            new_var / max(self._var_threshold, 1e-8), self._min_var_ratio
                        )

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
                f"got {batch_size} % {group_size} != 0.",
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

    def _reset_prompt_baselines(self, prompt_id: int):
        """Reset all baseline state for a single prompt.

        Called when distribution shift is detected (auto-reset mechanism).
        Clears baselines, variance, and observation tracking to force recalibration.
        """
        if prompt_id in self.V:
            self.V[prompt_id].clear()
        if prompt_id in self.V_last_seen:
            self.V_last_seen[prompt_id].clear()
        if prompt_id in self._quality_seen:
            self._quality_seen[prompt_id].clear()
        if prompt_id in self._step_seen:
            self._step_seen[prompt_id].clear()
        if prompt_id in self._reward_var:
            self._reward_var[prompt_id].clear()
        if prompt_id in self._reward_mean:
            self._reward_mean[prompt_id].clear()

    def on_tier_advance(self, new_tier: int, prompt_tier_map: dict[int, int]):
        """Full reset of baselines for affected prompts on tier/phase advance.

        When the reward distribution shifts (tier advance, phase advance), stale baselines
        anchored to the old distribution produce catastrophic negative advantages.
        Reset V, _reward_mean, _reward_var, _quality_seen, V_last_seen, and _step_seen
        for ALL qualities/steps of affected prompts — not just one key.

        This forces the warm-start path (is_first_observation) to recalibrate baselines
        from the first batch at the new difficulty level.
        """
        _logger.warning(
            f"on_tier_advance called: new_tier={new_tier}, n_prompts={len(prompt_tier_map)}, pids={list(prompt_tier_map.keys())[:5]}"
        )
        n_cleared = 0
        for pid in prompt_tier_map:
            # Reset ALL per-quality baselines for this prompt
            if pid in self.V:
                n_qualities = len(self.V[pid])
                self.V[pid].clear()
                n_cleared += 1
                if n_cleared <= 2:  # Log first 2 prompts
                    _logger.warning(f"  Cleared {n_qualities} qualities for prompt {pid}")
            if pid in self.V_last_seen:
                self.V_last_seen[pid].clear()
            if pid in self._quality_seen:
                self._quality_seen[pid].clear()
            if pid in self._step_seen:
                self._step_seen[pid].clear()
            # Reset variance tracking — stale variance locks EMA lr to near-zero
            if pid in self._reward_var:
                self._reward_var[pid].clear()
            if pid in self._reward_mean:
                self._reward_mean[pid].clear()
            # Clear divergence tracking — fresh start after manual reset
            if pid in self._divergence_window:
                self._divergence_window[pid].clear()
        _logger.warning(
            f"on_tier_advance: cleared baselines for {n_cleared}/{len(prompt_tier_map)} prompts"
        )

    def state_dict(self) -> dict:
        return {
            # Per-quality baselines (keyed by quality name string)
            "V": {pid: dict(qualities) for pid, qualities in self.V.items()},
            "V_last_seen": {pid: dict(qualities) for pid, qualities in self.V_last_seen.items()},
            "quality_seen": {pid: list(qualities) for pid, qualities in self._quality_seen.items()},
            # Legacy step_seen for backward compat
            "step_seen": {pid: list(steps) for pid, steps in self._step_seen.items()},
            # Per-quality variance tracking
            "reward_var": {pid: dict(qualities) for pid, qualities in self._reward_var.items()},
            "reward_mean": {pid: dict(qualities) for pid, qualities in self._reward_mean.items()},
            # Divergence tracking for auto-reset
            "divergence_window": {
                pid: {step: list(window) for step, window in steps.items()}
                for pid, steps in self._divergence_window.items()
            },
            "lr": self.lr,
            "mode": self.mode,
            "current_step": self._current_step,
            "divergence_last_cleanup": self._divergence_last_cleanup,
        }

    def compute_advantages_with_spans(
        self,
        batch_prompt_ids: list[int],
        batch_token_ids: list[list[int]],
        batch_reward_results: list[RewardResult],
        batch_active_qualities: list[list[str]],
        batch_token_masks: list[dict[str, torch.Tensor]],
        group_size: int | None = None,
        frontier_steps: set[int] | None = None,
        batch_contexts: list[PromptContext] | None = None,
        ctx: TrainingContext | None = None,
    ) -> tuple[list[torch.Tensor], dict[str, dict[str, float]]]:
        """Compute per-token advantages using PER-QUALITY span-based token masks.

        Unlike step-level averaging, this computes independent advantages per quality
        and broadcasts each quality's advantage to only its own span tokens.

        Returns:
            (batch_advantages, batch_quality_metrics):
            - batch_advantages: per-token advantage tensors
            - batch_quality_metrics: per-sample dict of quality_name → {reward, baseline, advantage}
        """
        batch_size = len(batch_token_ids)

        # Per-quality advantages for each sample
        all_quality_advs: list[dict[str, float]] = []
        batch_quality_metrics: dict[str, dict[str, float]] = {}

        # Phase 1+2: Compute per-quality advantages directly (no step averaging)
        for i in range(batch_size):
            prompt_ctx = batch_contexts[i] if batch_contexts and i < len(batch_contexts) else None
            pid = prompt_ctx.prompt_id if prompt_ctx is not None else batch_prompt_ids[i]
            warmup = getattr(prompt_ctx, "aspiration_warmup", 1.0) if prompt_ctx else 1.0
            warmup = max(0.0, min(1.0, warmup))  # Clamp to [0, 1] to prevent double application

            quality_advs: dict[str, float] = {}

            for quality_name in batch_active_qualities[i]:
                r = batch_reward_results[i].scores.get(quality_name, 0.0)
                v = self.get_baseline(pid, quality_name)

                # Warm-start: first observation → use prior
                if quality_name not in self._quality_seen[pid]:
                    v = self._baseline_prior
                    self._quality_seen[pid].add(quality_name)

                # Perfect score (1.0) = zero advantage — nothing to learn
                if r >= 1.0:
                    quality_advs[quality_name] = 0.0
                else:
                    adv = r - v
                    # Aspiration gap: push toward perfection (1.0, not mastery_threshold)
                    if self._aspiration_beta > 0:
                        adv += self._aspiration_beta * warmup * (r - 1.0)
                    quality_advs[quality_name] = adv

                # Variance-aware baseline learning rate
                effective_lr = self.lr
                if self._var_aware:
                    r_mean = self._reward_mean[pid][quality_name]
                    new_mean = r_mean + self._var_lr * (r - r_mean)
                    self._reward_mean[pid][quality_name] = new_mean
                    old_var = self._reward_var[pid][quality_name]
                    new_var = old_var + self._var_lr * ((r - new_mean) ** 2 - old_var)
                    if new_var < 0:
                        warnings.warn(
                            f"Negative variance {new_var:.6f} for prompt {pid} quality {quality_name}. "
                            f"old_var={old_var:.6f}, r={r:.4f}, r_mean={r_mean:.4f}. Clamping to 0.",
                            stacklevel=2,
                        )
                    self._reward_var[pid][quality_name] = max(0.0, new_var)
                    if new_var < self._var_threshold:
                        effective_lr = self.lr * max(
                            new_var / self._var_threshold, self._min_var_ratio
                        )

                # Update baseline using new per-quality method
                self.update_baseline(pid, quality_name, r, effective_lr)

                # Collect metrics for logging
                batch_quality_metrics[f"sample_{i}/{quality_name}"] = {
                    "reward": r,
                    "baseline": v,
                    "advantage": quality_advs[quality_name],
                }

            all_quality_advs.append(quality_advs)

        # Phase 3: Broadcast per-quality advantages to tokens (additive + normalized)
        batch_advantages: list[torch.Tensor] = []
        # A5: Validate batch_active_qualities and batch_token_masks have same length
        if len(batch_active_qualities) != len(batch_token_masks):
            raise ValueError(
                f"A5: Shape mismatch — batch_active_qualities has {len(batch_active_qualities)} entries, "
                f"batch_token_masks has {len(batch_token_masks)} entries. Lengths must match.",
            )

        for i in range(batch_size):
            seq_len = len(batch_token_ids[i])
            # Use ctx.device if available, otherwise infer from batch_token_masks
            device = ctx.device if ctx is not None else "cpu"
            if ctx is None and batch_token_masks[i]:
                first_mask = next(iter(batch_token_masks[i].values()), None)
                if first_mask is not None and isinstance(first_mask, torch.Tensor):
                    device = first_mask.device

            # C05-SHAPE: Validate ctx.device matches input tensor device before ops
            if ctx is not None and batch_token_masks[i]:
                first_mask = next(iter(batch_token_masks[i].values()), None)
                if first_mask is not None and isinstance(first_mask, torch.Tensor):
                    if first_mask.device != ctx.device:
                        raise ValueError(
                            f"C05-SHAPE: Input mask device {first_mask.device} does not match ctx.device {ctx.device}",
                        )

            # C04-NUMERICAL: Use ctx.dtype for advantage tensor creation
            dtype = ctx.dtype if ctx is not None else torch.float32
            token_advs = torch.zeros(seq_len, device=device, dtype=dtype)
            overlap_count = torch.zeros(seq_len, device=device, dtype=dtype)
            # R3-RSP-009: Track first-occurrence advantages to cap repetition penalty accumulation
            first_occurrence_advs = torch.zeros(seq_len, device=device, dtype=dtype)
            masks = batch_token_masks[i]

            # RL3-008: Track skipped qualities and raise if all skipped due to SHAPE MISMATCH
            skipped_count = 0
            shape_mismatch_count = 0
            for quality_name in batch_active_qualities[i]:
                if quality_name not in masks:
                    # A-5: Log warning when quality missing from masks
                    if not hasattr(self, "_quality_mask_mismatch_count"):
                        self._quality_mask_mismatch_count = 0
                    self._quality_mask_mismatch_count += 1
                    if self._quality_mask_mismatch_count <= 5:
                        _logger.warning(
                            f"A-5: Quality '{quality_name}' in active_qualities but not in token_masks "
                            f"for sample {i}. Skipping. Total occurrences: {self._quality_mask_mismatch_count}",
                        )
                    skipped_count += 1
                    continue
                q_adv = all_quality_advs[i].get(quality_name, 0.0)
                if abs(q_adv) < 1e-10:
                    continue
                q_mask = masks[quality_name]
                # Graceful handling instead of assert — don't crash on data-dependent mismatch
                if q_mask.shape[0] != seq_len:
                    warnings.warn(
                        f"Mask shape mismatch for quality '{quality_name}': "
                        f"mask has {q_mask.shape[0]} tokens but sequence has {seq_len}. "
                        f"Skipping — check reward_fn scored_spans and tokenizer consistency.",
                        stacklevel=2,
                    )
                    skipped_count += 1
                    shape_mismatch_count += 1
                    continue
                # Sign-aware repetition penalty:
                # - Mask value 1.0 = first occurrence → normal advantage
                # - Mask value REPETITION_MARKER = repeat → penalty (always negative effect)
                # Split mask into first-occurrence and repetition components
                first_mask = (q_mask == 1.0).float()
                repeat_mask = (q_mask == REPETITION_MARKER).float()

                # First occurrence: normal q_adv (positive for correct, negative for wrong)
                # Repetition: always penalize with -|q_adv| * multiplier
                # This ensures repeating correct = net negative, repeating wrong = even more negative
                token_advs += q_adv * first_mask
                # R3-RSP-009: Track first-occurrence advantage magnitude for capping repetition penalty
                first_occurrence_advs = torch.maximum(
                    first_occurrence_advs, torch.abs(q_adv * first_mask)
                )
                token_advs += -abs(q_adv) * REPETITION_PENALTY_MULTIPLIER * repeat_mask

                # For overlap normalization, count both first and repeat as participating
                overlap_count += first_mask + repeat_mask

            # RL3-008: Raise if ALL qualities were skipped AND at least one was due to shape mismatch
            # (Empty masks dict is backward compat, not an error)
            if shape_mismatch_count > 0 and skipped_count == len(batch_active_qualities[i]):
                raise RuntimeError(
                    f"RL3-008: All {skipped_count} qualities skipped for sample {i} due to mask shape mismatch. "
                    "Check reward_fn scored_spans and tokenizer consistency.",
                )

            # Normalize: tokens in multiple quality spans get their advantage divided by overlap count.
            # Example: token in both q_format and q_correct_H spans → advantage / 2
            # Tokens with zero overlap (thinking, whitespace) get advantage = 0 (no training signal).
            # R3-RSP-006: Log warning if any tokens have zero overlap count before clamping
            zero_overlap_mask = overlap_count == 0.0
            if zero_overlap_mask.any():
                zero_count = zero_overlap_mask.sum().item()
                warnings.warn(
                    f"R3-RSP-006: {zero_count} tokens in sample {i} have zero overlap "
                    f"(missing from all quality masks). These tokens get zero advantage silently.",
                    stacklevel=2,
                )
            overlap_count = torch.clamp(overlap_count, min=1.0)
            token_advs = token_advs / overlap_count

            # R3-RSP-009: Cap accumulated repetition penalty to not exceed first-occurrence magnitude
            # This prevents multiple repeat annotations from training correct token as wrong.
            # For tokens with first_occurrence_advs > 0, ensure token_advs >= -first_occurrence_advs.
            token_advs = torch.where(
                first_occurrence_advs > 0,
                torch.maximum(token_advs, -first_occurrence_advs),
                token_advs,
            )

            # Scale advantages to fit model's logit resolution
            if self._advantage_scale != 1.0:
                token_advs = token_advs * self._advantage_scale

            # Clip to prevent unbounded repetition penalties
            token_advs = torch.clamp(
                token_advs, min=-self._clip_advantage, max=self._clip_advantage
            )

            batch_advantages.append(token_advs)

        return batch_advantages, batch_quality_metrics

    def load_state_dict(self, state: dict):
        self.lr = state.get("lr", self.lr)
        self.mode = state.get("mode", self.mode)
        self._current_step = state.get("current_step", 0)
        # Restore cleanup tracking
        self._divergence_last_cleanup = state.get("divergence_last_cleanup", 0)

        # Per-quality baselines — preserve original key type (int for legacy step-based,
        # string for new per-quality span-based). torch.save/load preserves Python types.
        self.V = defaultdict(lambda: defaultdict(float))  # type: ignore[misc]
        for pid, qualities in state.get("V", {}).items():
            for key, val in qualities.items():
                # Preserve original key type (int or str)
                # Check for NaN and warn/skip
                val_float = float(val)
                if math.isnan(val_float):
                    warnings.warn(
                        f"NaN detected in advantage baseline V[{pid}][{key}]. Skipping entry.",
                        stacklevel=2,
                    )
                    continue
                self.V[int(pid)][key] = val_float

        self.V_last_seen = defaultdict(lambda: defaultdict(int))
        for pid, qualities in state.get("V_last_seen", {}).items():
            for key, val in qualities.items():
                self.V_last_seen[int(pid)][key] = int(val)

        self._quality_seen = defaultdict(set)
        for pid, qualities in state.get("quality_seen", {}).items():
            self._quality_seen[int(pid)] = set(qualities)

        # Legacy step_seen for backward compat
        self._step_seen = defaultdict(set)
        for pid, steps in state.get("step_seen", {}).items():
            self._step_seen[int(pid)] = {int(s) for s in steps}

        # Per-quality variance tracking — preserve original key type
        for pid, qualities in state.get("reward_var", {}).items():
            for key, val in qualities.items():
                self._reward_var[int(pid)][key] = float(val)

        self._reward_mean = defaultdict(lambda: defaultdict(float))
        for pid, qualities in state.get("reward_mean", {}).items():
            for key, val in qualities.items():
                self._reward_mean[int(pid)][key] = float(val)

        # Divergence tracking for auto-reset
        self._divergence_window = defaultdict(lambda: defaultdict(list))
        for pid, steps in state.get("divergence_window", {}).items():
            for step, window in steps.items():
                self._divergence_window[int(pid)][int(step)] = list(window)


def compute_advantages_vprm(
    critic,  # VPRMCritic
    hidden_states: torch.Tensor,
    regions: list[str],
    reward_result: RewardResult,
    step_qualities: dict[int, list[str]],
    active_qualities: list[str],
    step_region_map: dict[int, int] | None = None,
    frontier_steps: set[int] | None = None,
    frontier_amplification: float = 2.0,
    min_regions: int = 2,
    aspiration_beta: float = 0.0,
    aspiration_target: float = 0.0,
    clip_advantage: float | None = None,
    ctx: TrainingContext | None = None,
    token_masks: dict[str, torch.Tensor] | None = None,  # Span-based masks (if available)
    bond_strength: torch.Tensor | None = None,  # Per-token attention bond strength (0-1)
    constraint_strength: float = 1.0,  # Multiplier for bond strength effect
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Compute per-token advantages using VPRM critic for a single sample.

    Args:
        bond_strength: Per-token attention bond strength [seq_len], range [0, 1].
            High-bond tokens get constrained advantages to prevent cascade.
        constraint_strength: Multiplier for bond strength effect (default 1.0).

    Returns:
        (token_advantages, critic_loss, used_critic):
        - token_advantages: [seq_len] per-token advantages
        - critic_loss: scalar MSE loss for critic training
        - used_critic: True if critic was used, False if SPO fallback
    """
    seq_len = hidden_states.shape[0]
    # C01-DEVICE: Use ctx.device when available, fallback to hidden_states.device
    # This prevents device mismatch when hidden_states is on different device than ctx
    device = ctx.device if ctx is not None else hidden_states.device

    # C05-SHAPE: Validate hidden_states device matches ctx.device when ctx provided
    if ctx is not None and hidden_states.device != ctx.device:
        warnings.warn(
            f"C05-SHAPE: hidden_states on {hidden_states.device} but ctx.device={ctx.device}. "
            "Moving to ctx.device. Check upstream tensor placement.",
            stacklevel=2,
        )
        hidden_states = hidden_states.to(ctx.device)

    # Check if enough regions for critic — else SPO fallback
    # Skip this check when using spans (token_masks) since spans don't use STEP_N regions
    if not token_masks:
        n_regions = segmenter_region_count(regions)
        if n_regions < min_regions:
            # RL3-010: Return flag + log metric for SPO fallback
            if not hasattr(compute_advantages_vprm, "_spo_fallback_count"):
                compute_advantages_vprm._spo_fallback_count = 0  # type: ignore[attr-defined]
            compute_advantages_vprm._spo_fallback_count += 1  # type: ignore[attr-defined]
            if compute_advantages_vprm._spo_fallback_count <= 3:  # type: ignore[attr-defined]
                import logging

                logging.getLogger(__name__).info(
                    f"RL3-010: SPO fallback (regions={n_regions} < min_regions={min_regions}). "
                    f"Total fallbacks: {compute_advantages_vprm._spo_fallback_count}",  # type: ignore[attr-defined]
                )
            return (
                torch.zeros(seq_len, device=device),
                torch.tensor(0.0, device=device),
                False,
            )

    # Get actual rewards per quality
    actual_rewards = {k: reward_result.scores.get(k, 0.0) for k in active_qualities}

    # Compute advantages via critic (hidden states must be DETACHED)
    # Use span-based method if token_masks available (not None), otherwise fall back to region-based
    # Note: empty dict {} is valid (means no spans for this sample) - use `is not None` check
    if token_masks is not None and hasattr(critic, "compute_advantages_from_spans"):
        advs_dict, critic_losses = critic.compute_advantages_from_spans(
            hidden_states.detach(),
            token_masks,
            actual_rewards,
            ctx=ctx,
        )
    else:
        advs_dict, critic_losses = critic.compute_advantages(
            hidden_states.detach(),
            regions,
            actual_rewards,
            ctx=ctx,
        )

    # Build reverse map: region_step → [virtual steps that map to it]
    region_extra_steps: dict[int, list[int]] = defaultdict(list)
    if step_region_map:
        for vs, rs in step_region_map.items():
            region_extra_steps[rs].append(vs)

    # Broadcast per-quality advantages to per-token by region
    step_nums = sorted(step_qualities.keys())
    # R2-RSP-003: Initialize step_advs with ALL step_nums including those with virtual steps mapped
    step_advs: dict[int, float] = dict.fromkeys(step_nums, 0.0)
    if step_region_map:
        for vs in step_region_map:
            if vs not in step_advs:
                step_advs[vs] = 0.0
    # Build per-step advantages from quality advantages
    for step_num in step_nums:
        qualities = [q for q in step_qualities[step_num] if q in active_qualities]
        if qualities:
            # RSP-001: Warn when quality returns None from critic (silent advantage erasure)
            quality_advs = [(q, advs_dict.get(q, 0.0)) for q in qualities]
            dropped = [q for q, v in quality_advs if v is None]
            vals = [v for _q, v in quality_advs if v is not None]
            if dropped:
                import logging

                logging.getLogger(__name__).warning(
                    f"RSP-001: {len(dropped)} qualities returned None from critic for step {step_num}. "
                    f"Dropped qualities: {dropped}. This erases advantage signal for this step.",
                )
            step_advs[step_num] = sum(vals) / len(vals) if vals else 0.0

    # RL3-007: Document virtual step behavior and add metric
    if step_region_map:
        virtual_steps_used = []
        for vs in step_region_map:
            if vs not in step_advs:
                step_advs[vs] = 0.0
                virtual_steps_used.append(vs)
        if virtual_steps_used and not hasattr(compute_advantages_vprm, "_virtual_logged"):
            import logging

            logging.getLogger(__name__).info(
                f"RL3-007: Virtual steps initialized: {virtual_steps_used}. "
                "These are frontier amplification targets that don't have direct quality assignments.",
            )
            compute_advantages_vprm._virtual_logged = True  # type: ignore[attr-defined]

    # Perfect score = zero advantage. Imperfect = push toward 1.0.
    # Include virtual steps (generated by frontier amplification) in aspiration bonus loop
    all_step_keys = list(step_advs.keys())
    for step_num in all_step_keys:
        if step_num not in step_qualities:
            continue  # Virtual step, no qualities to check
        qualities = [q for q in step_qualities[step_num] if q in active_qualities]
        if qualities:
            step_reward = sum(reward_result.scores.get(q, 0.0) for q in qualities) / max(
                len(qualities), 1
            )
            if step_reward >= 1.0:
                step_advs[step_num] = 0.0
            elif aspiration_beta > 0:
                step_advs[step_num] += aspiration_beta * (step_reward - 1.0)

    apply_frontier_amplification(step_advs, step_nums, frontier_steps, frontier_amplification)

    # Broadcast to tokens
    token_advantages = broadcast_step_advantages_to_tokens(
        step_advs,  # type: ignore[arg-type]
        regions,
        region_extra_steps,
        ctx=ctx,
        bond_strength=bond_strength,
        constraint_strength=constraint_strength,
    ).to(device)

    # Clip VPRM advantages after broadcast
    if clip_advantage is not None:
        token_advantages = torch.clamp(token_advantages, min=-clip_advantage, max=clip_advantage)

    # Total critic loss
    if critic_losses:
        losses = [loss for loss in critic_losses.values() if not torch.isnan(loss)]
        if len(losses) < len(critic_losses):
            warnings.warn(
                f"Filtered {len(critic_losses) - len(losses)} NaN critic losses before aggregation",
                stacklevel=2,
            )
        if losses:
            total_critic_loss = torch.stack(losses).mean()
        else:
            # All losses were NaN — return zero with requires_grad=True to preserve gradient flow
            total_critic_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        # No critic losses available — return zero with requires_grad=True to preserve gradient flow
        total_critic_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return token_advantages, total_critic_loss, True
