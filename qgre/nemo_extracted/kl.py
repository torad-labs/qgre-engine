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
# Extracted from NeMo RL v0.5.0 algorithms/utils.py for QGRE Engine (Step 0b).
# Contains: calculate_kl (Schulman 2020 approximation) and masked_mean.

from __future__ import annotations

import torch


def calculate_kl(
    logprobs: torch.Tensor,
    logprobs_reference: torch.Tensor,
    kl_type: str = "k3",
    input_clamp_value: float | None = 20.0,
    output_clamp_value: float | None = 10.0,
) -> torch.Tensor:
    """Per-token KL divergence estimate between two log-prob tensors.

    From Schulman 2020: http://joschu.net/blog/kl-approx.html

    Args:
        logprobs: [batch, seq] log probs from current policy
        logprobs_reference: [batch, seq] log probs from reference policy
        kl_type: "k1" (linear), "k2" (squared), or "k3" (exponential, default)
        input_clamp_value: Clamp log-ratio to prevent numerical instability
        output_clamp_value: Clamp KL output to prevent numerical instability

    Returns:
        [batch, seq] per-token KL penalty values
    """
    logr = logprobs_reference - logprobs
    if input_clamp_value is not None:
        logr = logr.clamp(min=-input_clamp_value, max=input_clamp_value)

    if kl_type == "k1":
        kl = -logr
    elif kl_type == "k2":
        kl = torch.square(logr) / 2
    elif kl_type == "k3":
        # H1-001: exp(logr) can produce huge values (exp(20)=4.85e8) even with input clamp.
        # Standard clamp on output doesn't clamp gradients — grad(exp(x)) = exp(x) backprops
        # the same huge value. Use detach trick: clamp forward, pass gradient through clamped value.
        exp_logr = torch.exp(logr)
        # Gradient-safe clamp: if exp_logr > max, use max for forward AND backward
        exp_max = 1e4  # exp(~9.2), safe margin below exp(20)
        exp_logr_safe = torch.where(
            exp_logr > exp_max,
            torch.full_like(exp_logr, exp_max),
            exp_logr,
        )
        kl = exp_logr_safe - 1 - logr
    else:
        raise ValueError(f"Invalid KL type: {kl_type}")

    if output_clamp_value is not None:
        kl = kl.clamp(min=0, max=output_clamp_value)

    return kl


def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    global_normalization_factor: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Masked mean with optional global normalization factor.

    For microbatch training, pass global_normalization_factor = total valid tokens
    across all microbatches to get correct global mean.
    """
    normalization_factor = (
        torch.sum(mask, dim=dim)
        if global_normalization_factor is None
        else global_normalization_factor
    )
    # TRN-R1-5: Always return tensor (not scalar 0.0) for consistent type
    if isinstance(normalization_factor, torch.Tensor):
        if (normalization_factor == 0).all():
            return torch.zeros_like(torch.sum(values * mask, dim=dim))
        return torch.sum(values * mask, dim=dim) / normalization_factor.clamp(min=1e-6)
    if normalization_factor == 0:
        return torch.tensor(0.0, device=values.device, dtype=values.dtype)
    return torch.sum(values * mask, dim=dim) / max(normalization_factor, 1e-6)
