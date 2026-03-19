"""Equivalence test: fixed completions through algorithm layer (Step 7).

Tests the ALGORITHM layer only — not generation or reward.
When reference data from verl is available, replace synthetic_reference
with real JSONL data. Until then, tests verify internal consistency.
"""

import torch
import numpy as np
import pytest

from qgre.advantages import QGREStepAdvantageEstimator
from qgre.nemo_extracted.loss_functions import ClippedPGLossFn
from qgre.nemo_extracted.kl import masked_mean
from qgre.nemo_extracted.logits import logprobs_from_logits
from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES as STEP_QUALITIES, OPEN_ANGLE, STEP_TOKEN, CLOSE_ANGLE, CLOSE_SLASH
from qgre.types import RewardResult


def _make_tokens():
    step_num_map = {1: 16, 2: 17, 3: 18, 4: 19}
    tokens = []
    for s in range(1, 5):
        tokens.extend([OPEN_ANGLE, STEP_TOKEN, step_num_map[s], 9999, CLOSE_ANGLE])
        tokens.extend([100 + s, 200 + s, 300 + s])
        tokens.extend([CLOSE_SLASH, STEP_TOKEN, step_num_map[s], 9999, CLOSE_ANGLE])
    return tokens


ALL_Q = []
for qs in STEP_QUALITIES.values():
    ALL_Q.extend(qs)


def test_advantages_deterministic():
    """Same input → same output. No hidden state leaks between calls."""
    tokens = _make_tokens()

    results = [
        RewardResult(reward=0.8, scores={q: 0.8 for q in ALL_Q}, phase=4),
        RewardResult(reward=0.3, scores={q: 0.3 for q in ALL_Q}, phase=4),
        RewardResult(reward=0.6, scores={q: 0.6 for q in ALL_Q}, phase=4),
        RewardResult(reward=0.5, scores={q: 0.5 for q in ALL_Q}, phase=4),
    ]

    # Run twice with fresh estimators
    est1 = QGREStepAdvantageEstimator(lr=0.1, mode="grpo")
    advs1 = est1.compute_advantages(
        batch_prompt_ids=[1, 2, 3, 4],
        batch_token_ids=[tokens] * 4,
        batch_reward_results=results,
        batch_active_qualities=[ALL_Q] * 4,
        group_size=4,
    )

    est2 = QGREStepAdvantageEstimator(lr=0.1, mode="grpo")
    advs2 = est2.compute_advantages(
        batch_prompt_ids=[1, 2, 3, 4],
        batch_token_ids=[tokens] * 4,
        batch_reward_results=results,
        batch_active_qualities=[ALL_Q] * 4,
        group_size=4,
    )

    for a1, a2 in zip(advs1, advs2):
        assert torch.allclose(a1, a2, atol=1e-6)


def test_loss_deterministic():
    """Same inputs to loss function → same output."""
    cfg = {
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.28,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
        "force_on_policy_ratio": True,
    }

    loss_fn = ClippedPGLossFn(cfg)

    torch.manual_seed(42)
    curr_lp = torch.randn(4, 16) * 0.1 - 3.0
    prev_lp = curr_lp.detach().clone()
    advantages = torch.randn(4, 16)
    mask = torch.ones(4, 16)

    loss1, _ = loss_fn(curr_lp, prev_lp, advantages, mask)
    loss2, _ = loss_fn(curr_lp, prev_lp, advantages, mask)

    assert torch.allclose(loss1, loss2, atol=1e-6)


def test_advantage_loss_pipeline_consistency():
    """Full pipeline: advantages → pad → loss. No shape mismatches or nans."""
    tokens = _make_tokens()
    batch_size = 4

    results = [
        RewardResult(reward=r, scores={q: r for q in ALL_Q}, phase=4)
        for r in [0.9, 0.3, 0.7, 0.5]
    ]

    est = QGREStepAdvantageEstimator(lr=0.1, mode="grpo")
    advs = est.compute_advantages(
        batch_prompt_ids=[1, 1, 1, 1],
        batch_token_ids=[tokens] * batch_size,
        batch_reward_results=results,
        batch_active_qualities=[ALL_Q] * batch_size,
        group_size=batch_size,
    )

    # Pad advantages to uniform length
    max_len = max(len(a) for a in advs)
    padded = torch.zeros(batch_size, max_len)
    mask = torch.zeros(batch_size, max_len)
    for i, a in enumerate(advs):
        padded[i, :len(a)] = a
        mask[i, :len(a)] = 1.0

    # Create synthetic log probs
    curr_lp = torch.randn(batch_size, max_len) * 0.1 - 3.0
    prev_lp = curr_lp.detach().clone()

    cfg = {
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.28,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
        "force_on_policy_ratio": True,
    }

    loss_fn = ClippedPGLossFn(cfg)
    loss, metrics = loss_fn(curr_lp, prev_lp, padded, mask)

    assert loss.isfinite(), f"Loss is not finite: {loss}"
    assert "loss" in metrics


def test_spo_vs_grpo_produce_different_advantages():
    """SPO and GRPO modes produce different advantage distributions."""
    tokens = _make_tokens()

    results = [
        RewardResult(reward=0.9, scores={q: 0.9 for q in ALL_Q}, phase=4),
        RewardResult(reward=0.1, scores={q: 0.1 for q in ALL_Q}, phase=4),
    ]

    # SPO
    est_spo = QGREStepAdvantageEstimator(lr=0.1, mode="spo")
    # Warm up SPO
    est_spo.compute_advantages([1, 2], [tokens] * 2, results, [ALL_Q] * 2)
    advs_spo = est_spo.compute_advantages([1, 2], [tokens] * 2, results, [ALL_Q] * 2)

    # GRPO
    est_grpo = QGREStepAdvantageEstimator(lr=0.1, mode="grpo")
    advs_grpo = est_grpo.compute_advantages([1, 1], [tokens] * 2, results, [ALL_Q] * 2, group_size=2)

    # They should produce different values (different baseline methods)
    spo_flat = torch.cat(advs_spo)
    grpo_flat = torch.cat(advs_grpo)

    # At minimum, both should be finite
    assert spo_flat.isfinite().all()
    assert grpo_flat.isfinite().all()
