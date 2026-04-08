# LLDS — Lazy Likelihood Displacement Stabilization (arXiv:2512.04220)
#
# Auxiliary loss that penalizes silent log-prob decay on correct/positive completions.
# Three-level gate: trajectory (response-level decline) + token (displaced) + action (adv ≥ 0).
# Addresses training collapse precursor where reward looks stable but correct-answer
# log-probs silently decrease.
#
# Extracted from torad-labs/verl fork (verl/trainer/ppo/core_algos.py lines 2486-2526).
# Config: algorithm.llds_coef (default 0.05).

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch


def compute_llds_loss(
    log_prob: torch.Tensor,
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LLDS auxiliary loss — prevents Lazy Likelihood Displacement death spiral.

    When GRPO training is working, correct-response log-probs should increase.
    LLD occurs when they silently decrease — a precursor to policy collapse.
    This loss penalizes that decay with a three-level gate.

    Args:
        log_prob: [batch, seq] current policy log probs
        old_log_prob: [batch, seq] rollout-time log probs
        advantages: [batch, seq] per-token advantages
        response_mask: [batch, seq] mask (1 = response token)

    Returns:
        (loss, llds_mask) — loss scalar and gate mask for metrics
    """
    eps = 1e-8
    mask_sum = response_mask.sum(-1, keepdim=True).clamp(min=1)

    # 1. Trajectory gate: mean log-prob decreased for this response
    mean_new = (log_prob * response_mask).sum(-1, keepdim=True) / mask_sum
    mean_old = (old_log_prob * response_mask).sum(-1, keepdim=True) / mask_sum
    traj_gate = (mean_new < mean_old - eps).float()

    # 2. Token gate: this token displaced downward
    token_gate = (log_prob < old_log_prob - eps).float()

    # 3. Action gate: non-negative advantage (correct completions only)
    resp_adv = (advantages * response_mask).sum(-1, keepdim=True) / mask_sum
    action_gate = (resp_adv >= 0).float()

    # Combined gate: only penalize tokens on correct responses that are declining
    llds_mask = traj_gate * token_gate * action_gate * response_mask
    mask_count = llds_mask.sum()
    if mask_count == 0:
        import torch

        loss = torch.tensor(0.0, device=log_prob.device, dtype=log_prob.dtype, requires_grad=True)
    else:
        displacement = (old_log_prob - log_prob) * llds_mask
        loss = displacement.sum() / mask_count

    return loss, llds_mask
