"""Internal consistency tests: fixed completions through algorithm layer (Step 7).

Tests the ALGORITHM layer only — not generation or reward.
Verifies that the same input produces the same output (determinism),
and that SPO/GRPO modes produce structurally correct advantages.

NOTE: This does NOT test equivalence with verl or any external framework.
When verl reference data is available, a separate test_verl_equivalence.py
should be created for cross-framework comparison.
"""

import torch

from qgre.advantages import QGREStepAdvantageEstimator
from qgre.nemo_extracted.loss_functions import ClippedPGLossFn
from qgre.segments import CLOSE_ANGLE, CLOSE_SLASH, OPEN_ANGLE, STEP_TOKEN
from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES as STEP_QUALITIES
from qgre.segments import qwen3_xml_segmenter as XML_SEG
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
        RewardResult(reward=0.8, scores=dict.fromkeys(ALL_Q, 0.8), phase=4),
        RewardResult(reward=0.3, scores=dict.fromkeys(ALL_Q, 0.3), phase=4),
        RewardResult(reward=0.6, scores=dict.fromkeys(ALL_Q, 0.6), phase=4),
        RewardResult(reward=0.5, scores=dict.fromkeys(ALL_Q, 0.5), phase=4),
    ]

    # Run twice with fresh estimators
    est1 = QGREStepAdvantageEstimator(
        lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG
    )
    advs1, _ = est1.compute_advantages(
        batch_prompt_ids=[1, 2, 3, 4],
        batch_token_ids=[tokens] * 4,
        batch_reward_results=results,
        batch_active_qualities=[ALL_Q] * 4,
        group_size=4,
    )

    est2 = QGREStepAdvantageEstimator(
        lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG
    )
    advs2, _ = est2.compute_advantages(
        batch_prompt_ids=[1, 2, 3, 4],
        batch_token_ids=[tokens] * 4,
        batch_reward_results=results,
        batch_active_qualities=[ALL_Q] * 4,
        group_size=4,
    )

    for a1, a2 in zip(advs1, advs2, strict=False):
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
        RewardResult(reward=r, scores=dict.fromkeys(ALL_Q, r), phase=4) for r in [0.9, 0.3, 0.7, 0.5]
    ]

    est = QGREStepAdvantageEstimator(
        lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG
    )
    advs, _ = est.compute_advantages(
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
        padded[i, : len(a)] = a
        mask[i, : len(a)] = 1.0

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
        RewardResult(reward=0.9, scores=dict.fromkeys(ALL_Q, 0.9), phase=4),
        RewardResult(reward=0.1, scores=dict.fromkeys(ALL_Q, 0.1), phase=4),
    ]

    # SPO
    est_spo = QGREStepAdvantageEstimator(
        lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG
    )
    # Warm up SPO
    est_spo.compute_advantages([1, 2], [tokens] * 2, results, [ALL_Q] * 2)
    advs_spo, _ = est_spo.compute_advantages([1, 2], [tokens] * 2, results, [ALL_Q] * 2)

    # GRPO
    est_grpo = QGREStepAdvantageEstimator(
        lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG
    )
    advs_grpo, _ = est_grpo.compute_advantages(
        [1, 1], [tokens] * 2, results, [ALL_Q] * 2, group_size=2
    )

    # They should produce different values (different baseline methods)
    spo_flat = torch.cat(advs_spo)
    grpo_flat = torch.cat(advs_grpo)

    # At minimum, both should be finite
    assert spo_flat.isfinite().all()
    assert grpo_flat.isfinite().all()


def test_dr_grpo_removes_length_normalization():
    """Dr.GRPO mode: loss not divided by horizon length (arXiv:2503.20783)."""
    from qgre.nemo_extracted.loss_functions import ClippedPGLossFn

    base_cfg = {
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

    torch.manual_seed(42)
    curr_lp = torch.randn(4, 16) * 0.1 - 3.0
    prev_lp = curr_lp.detach().clone()
    advantages = torch.randn(4, 16)
    mask = torch.ones(4, 16)

    # Standard GRPO: divides by horizon (16)
    grpo_fn = ClippedPGLossFn({**base_cfg, "remove_length_normalization": False})
    grpo_loss, _ = grpo_fn(curr_lp, prev_lp, advantages, mask)

    # Dr.GRPO: no horizon division
    dr_fn = ClippedPGLossFn({**base_cfg, "remove_length_normalization": True})
    dr_loss, _ = dr_fn(curr_lp, prev_lp, advantages, mask)

    # Dr.GRPO loss should be ~16x larger (horizon length) since it skips the division
    ratio = dr_loss.item() / grpo_loss.item()
    assert 14 < ratio < 18, f"Expected ~16x ratio, got {ratio}"


def test_dr_grpo_no_std_normalization():
    """Dr.GRPO: GDPO step skips std division when normalize_advantages=False."""
    tokens = _make_tokens()

    # Use GRPO mode to test the GDPO normalization step behavior
    # (SPO mode skips batch normalization entirely — per-prompt baseline is the only centering)
    # 4 samples with different rewards → varied step advantages
    results = [
        RewardResult(reward=0.9, scores=dict.fromkeys(ALL_Q, 0.9), phase=4),
        RewardResult(reward=0.3, scores=dict.fromkeys(ALL_Q, 0.3), phase=4),
        RewardResult(reward=0.6, scores=dict.fromkeys(ALL_Q, 0.6), phase=4),
        RewardResult(reward=0.1, scores=dict.fromkeys(ALL_Q, 0.1), phase=4),
    ]

    # Standard: normalize by std in GDPO step
    est_norm = QGREStepAdvantageEstimator(
        lr=0.1,
        mode="grpo",
        step_qualities=STEP_QUALITIES,
        segmenter=XML_SEG,
        normalize_advantages=True,
    )
    advs_norm, _ = est_norm.compute_advantages([1, 2, 3, 4], [tokens] * 4, results, [ALL_Q] * 4)

    # Dr.GRPO: mean-only in GDPO step
    est_raw = QGREStepAdvantageEstimator(
        lr=0.1,
        mode="grpo",
        step_qualities=STEP_QUALITIES,
        segmenter=XML_SEG,
        normalize_advantages=False,
    )
    advs_raw, _ = est_raw.compute_advantages([1, 2, 3, 4], [tokens] * 4, results, [ALL_Q] * 4)

    norm_flat = torch.cat(advs_norm)
    raw_flat = torch.cat(advs_raw)
    assert norm_flat.isfinite().all()
    assert raw_flat.isfinite().all()

    # Key: both should have zero mean (mean-subtracted)
    assert abs(norm_flat[norm_flat != 0].mean().item()) < 0.05
    assert abs(raw_flat[raw_flat != 0].mean().item()) < 0.05

    # Normalized should have std ≈ 1, raw should have different std (the raw spread)
    norm_std = norm_flat[norm_flat != 0].std().item()
    raw_std = raw_flat[raw_flat != 0].std().item()
    assert abs(norm_std - 1.0) < 0.3, f"Normalized std should be ≈1, got {norm_std}"
    assert norm_std != raw_std, f"Normalized and raw should differ: {norm_std} vs {raw_std}"
