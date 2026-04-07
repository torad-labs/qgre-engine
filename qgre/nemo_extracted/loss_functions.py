# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Extracted from NeMo RL v0.5.0 for QGRE Engine (Step 0b).
# Stripped: Ray, Megatron, DTensor, vocab-parallel, context-parallel, sequence-packing.
# Kept: ClippedPGLossFn core math (single-GPU, single-process).

from typing import TypedDict

import torch
from typing_extensions import NotRequired

from qgre.nemo_extracted.kl import calculate_kl, masked_mean


def apply_eligibility_traces(
    advantages: torch.Tensor,
    lambda_val: float,
) -> torch.Tensor:
    """GRPO-λ: λ-return approximation for per-token credit assignment.

    Backward-accumulates advantages with decay factor λ, giving earlier tokens
    credit for downstream correct reasoning (critic-free TD(λ)).

    Args:
        advantages: [batch, seq] per-token advantages (from VPRM step-level)
        lambda_val: trace decay factor (0=no traces, 0.95=typical)

    Returns:
        [batch, seq] λ-return weighted advantages
    """
    batch, seq = advantages.shape
    traces = torch.zeros_like(advantages)
    # Backward pass: accumulate eligibility trace from end of sequence
    trace = torch.zeros(batch, device=advantages.device)
    for t in range(seq - 1, -1, -1):
        # RL3-001: Check for NaN before accumulation
        adv_t = advantages[:, t]
        if torch.isnan(adv_t).any():
            import warnings

            warnings.warn(
                f"NaN detected in eligibility trace at position {t}. Replacing with 0.0",
                stacklevel=2,
            )
            adv_t = torch.where(torch.isnan(adv_t), torch.zeros_like(adv_t), adv_t)
        trace = adv_t + lambda_val * trace
        traces[:, t] = trace
    return traces


class ClippedPGLossConfig(TypedDict):
    reference_policy_kl_penalty: float
    reference_policy_kl_type: str
    kl_input_clamp_value: float | None
    kl_output_clamp_value: float | None
    ratio_clip_min: float
    ratio_clip_max: float
    ratio_clip_c: float | None
    use_on_policy_kl_approximation: bool
    use_importance_sampling_correction: bool
    truncated_importance_sampling_ratio: float | None
    token_level_loss: bool
    force_on_policy_ratio: NotRequired[bool]
    remove_length_normalization: NotRequired[bool]  # Dr.GRPO: skip horizon division
    lambda_return: NotRequired[float]  # GRPO-λ: eligibility trace decay factor


class ClippedPGLossFn:
    """Clipped Policy Gradient loss — extracted from NeMo RL.

    Supports GRPO, PPO, REINFORCE/RLOO, and DAPO-style asymmetric clipping.
    Single-GPU only — no distributed, no vocab-parallel.
    """

    def __init__(self, cfg: ClippedPGLossConfig):
        self.ratio_clip_min = cfg["ratio_clip_min"]
        self.ratio_clip_max = cfg["ratio_clip_max"]
        self.ratio_clip_c = cfg["ratio_clip_c"]
        self.reference_policy_kl_penalty = cfg["reference_policy_kl_penalty"]
        self.reference_policy_kl_type = cfg["reference_policy_kl_type"]
        self.kl_input_clamp_value = cfg["kl_input_clamp_value"]
        self.kl_output_clamp_value = cfg["kl_output_clamp_value"]
        self.force_on_policy_ratio = cfg.get("force_on_policy_ratio", False)
        self.use_on_policy_kl_approximation = cfg["use_on_policy_kl_approximation"]
        self.use_importance_sampling_correction = cfg["use_importance_sampling_correction"]
        self.truncated_importance_sampling_ratio = cfg["truncated_importance_sampling_ratio"]
        self.token_level_loss = cfg["token_level_loss"]
        self.remove_length_normalization = cfg.get("remove_length_normalization", False)
        self.lambda_return = cfg.get("lambda_return", 0.0)

    def __call__(
        self,
        curr_logprobs: torch.Tensor,
        prev_logprobs: torch.Tensor,
        advantages: torch.Tensor,
        mask: torch.Tensor,
        reference_logprobs: torch.Tensor | None = None,
        kl_region_weights: torch.Tensor | None = None,
        return_per_token_loss: bool = False,
    ) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, torch.Tensor]:
        """Compute clipped PG loss.

        Args:
            curr_logprobs: Log probs from current policy [batch, seq]
            prev_logprobs: Log probs from generation policy [batch, seq]
            advantages: Per-token advantages [batch, seq]
            mask: Token mask (1 = valid, 0 = padding) [batch, seq]
            reference_logprobs: Log probs from reference policy [batch, seq] (for KL)

        Returns:
            (loss, metrics_dict)
        """
        global_valid_toks = mask.sum()

        # Probability ratio
        if self.force_on_policy_ratio:
            log_ratios = curr_logprobs - curr_logprobs.detach()
            ratios = log_ratios.exp()
            ratios_clamped = ratios
        else:
            log_ratios = curr_logprobs - prev_logprobs
            ratios = log_ratios.exp()
            ratios_clamped = ratios.clamp(
                1.0 - self.ratio_clip_min,
                1.0 + self.ratio_clip_max,
            )

        # GRPO-λ eligibility traces: weight advantages by accumulated log-prob traces
        # λ-return approximation using token-level log-probabilities as eligibility traces
        # (ICLR 2026 under review). Gives per-TOKEN credit within steps.
        if self.lambda_return > 0:
            advantages = apply_eligibility_traces(advantages, self.lambda_return)

        # Clipped surrogate loss
        loss1 = -advantages * ratios
        loss2 = -advantages * ratios_clamped
        clip_loss = torch.max(loss1, loss2)

        # Dual-clipping (DAPO-style)
        if self.ratio_clip_c is not None and self.ratio_clip_c > 1:
            loss3 = -advantages * self.ratio_clip_c
            clip_loss = torch.where(advantages < 0, torch.min(clip_loss, loss3), clip_loss)

        # Importance sampling correction (ratio: π_current / π_generation)
        if self.use_importance_sampling_correction:
            importance_weights = torch.exp(curr_logprobs.detach() - prev_logprobs).detach()
            importance_weights = torch.nan_to_num(
                importance_weights, nan=0.0, posinf=0.0, neginf=0.0
            )
            # Clamp to prevent underflow (silent zero gradients)
            # RL3-006: Track when clamping occurs
            clamped_mask = importance_weights < 1e-8
            if clamped_mask.any():
                import warnings

                warnings.warn(
                    f"RL3-006: {clamped_mask.sum().item()} importance weights clamped to minimum (1e-8). Gradients may be suppressed.",
                    stacklevel=2,
                )
            importance_weights = importance_weights.clamp(min=1e-8)
            if self.truncated_importance_sampling_ratio is not None:
                importance_weights = importance_weights.clamp(
                    max=self.truncated_importance_sampling_ratio
                )
        else:
            importance_weights = torch.ones_like(prev_logprobs)

        # Aggregate loss — seq-mean-token-sum-norm (verl core_algos.py lines 1172-1184)
        # Equal weight to each sequence, then normalize by horizon length
        weighted_loss = importance_weights * clip_loss
        if self.token_level_loss:
            seq_losses = torch.sum(weighted_loss * mask, dim=-1)
            seq_mask = (mask.sum(dim=-1) > 0).float()
            n_valid_seqs = seq_mask.sum().clamp(min=1)
            if self.remove_length_normalization:
                # Dr.GRPO (arXiv:2503.20783): no horizon division — unbiased gradients
                actor_loss = (seq_losses * seq_mask).sum() / n_valid_seqs
            else:
                # seq-mean-token-sum-norm: sum per-token, mean per-seq, normalize by horizon
                actor_loss = (seq_losses * seq_mask).sum() / n_valid_seqs / max(mask.shape[-1], 1)
        else:
            actor_loss = masked_mean(
                masked_mean(weighted_loss, mask, dim=-1),
                (mask.sum(dim=-1) > 0).float(),
                global_normalization_factor=(mask.sum(dim=-1) > 0).float().sum(),
            )

        # KL regularization
        if self.reference_policy_kl_penalty != 0 and reference_logprobs is not None:
            if self.use_on_policy_kl_approximation:
                kl_weights = torch.exp(curr_logprobs - prev_logprobs).detach()
                kl_weights = torch.nan_to_num(kl_weights, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                kl_weights = torch.ones_like(curr_logprobs)

            # Region-specific KL: scale penalty per-token by region weights (THR-style)
            # THINK=0.1 (explore), FORMAT=2.0 (lock), STEP=1.0 (normal)
            region_scale = kl_region_weights if kl_region_weights is not None else 1.0
            kl = (
                kl_weights
                * self.reference_policy_kl_penalty
                * region_scale
                * calculate_kl(
                    logprobs=curr_logprobs,
                    logprobs_reference=reference_logprobs,
                    kl_type=self.reference_policy_kl_type,
                    input_clamp_value=self.kl_input_clamp_value,
                    output_clamp_value=self.kl_output_clamp_value,
                )
            )
            kl_loss = masked_mean(kl, mask, global_normalization_factor=global_valid_toks)
        else:
            kl_loss = torch.tensor(0.0, device=actor_loss.device)

        loss = actor_loss + kl_loss

        with torch.no_grad():
            ratio_mean = masked_mean(
                ratios.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            ratio_clamped_mean = masked_mean(
                ratios_clamped.detach(),
                mask,
                global_normalization_factor=global_valid_toks,
            ).item()
            metrics = {
                "loss": loss.item(),
                "actor_loss": actor_loss.item(),
                "kl_penalty": kl_loss.item(),
                "probs_ratio_mean": ratio_mean,
                "probs_ratio_clamped_mean": ratio_clamped_mean,
            }

        if return_per_token_loss:
            # Return per-token loss for per-quality loss computation
            # weighted_loss is [batch, seq] — the per-token contribution before reduction
            return loss, metrics, weighted_loss.detach()
        return loss, metrics
