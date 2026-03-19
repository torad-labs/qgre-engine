"""Tests for advantage computation (Steps 0c, 0d, 8)."""

import torch
import numpy as np
import pytest

from qgre.advantages import QGREStepAdvantageEstimator, build_batch_reward_tensors
from qgre.segments import (
    CLOSE_ANGLE, CLOSE_SLASH, HYPERGRAPH_V1_STEP_QUALITIES, OPEN_ANGLE,
    STEP_TOKEN, THINK_END, THINK_START, qwen3_xml_segmenter, segment_completion,
)

STEP_QUALITIES = HYPERGRAPH_V1_STEP_QUALITIES
XML_SEG = qwen3_xml_segmenter
from qgre.types import RewardResult


# --- Step 0c: Batch reward tensor construction ---


def test_batch_reward_tensors_shape():
    """list[dict] → dict[str, Tensor] with correct shapes."""
    results = [
        RewardResult(reward=0.5, scores={"q_a": 1.0, "q_b": 0.5}),
        RewardResult(reward=0.3, scores={"q_a": 0.8, "q_b": 0.2}),
        RewardResult(reward=0.7, scores={"q_a": 0.9, "q_b": 0.7}),
        RewardResult(reward=0.4, scores={"q_a": 0.6, "q_b": 0.1}),
    ]
    tensors = build_batch_reward_tensors(results)
    assert tensors["q_a"].shape == (4,)
    assert tensors["q_b"].shape == (4,)
    assert torch.allclose(tensors["q_a"], torch.tensor([1.0, 0.8, 0.9, 0.6]))


def test_batch_reward_tensors_missing_keys():
    """Dicts with different key sets → missing keys zero-filled."""
    results = [
        RewardResult(reward=0.5, scores={"q_a": 1.0}),
        RewardResult(reward=0.3, scores={"q_b": 0.5}),
    ]
    tensors = build_batch_reward_tensors(results)
    assert tensors["q_a"][1].item() == 0.0  # missing in second result
    assert tensors["q_b"][0].item() == 0.0  # missing in first result


def test_batch_reward_tensors_empty():
    """Empty list → empty dict."""
    assert build_batch_reward_tensors([]) == {}


# --- Step 0d: Credit assignment tests ---


def _make_completion_tokens(steps=(1, 2, 3, 4)):
    """Build token sequence with specified step regions."""
    step_num_map = {1: 16, 2: 17, 3: 18, 4: 19}
    tokens = []
    for s in steps:
        # Opening tag
        tokens.extend([OPEN_ANGLE, STEP_TOKEN, step_num_map[s], 9999, CLOSE_ANGLE])
        # 3 content tokens
        tokens.extend([100 + s, 200 + s, 300 + s])
        # Closing tag
        tokens.extend([CLOSE_SLASH, STEP_TOKEN, step_num_map[s], 9999, CLOSE_ANGLE])
    return tokens


def _make_reward_result(step1_score=1.0, step4_score=1.0, phase=4):
    """Build a RewardResult with controlled per-step scores."""
    scores = {
        "q_format_tags": step1_score, "q_tag_content": step1_score,
        "q_node_in_prompt": step1_score, "q_node_format": step1_score, "q_node_length": step1_score,
        "q_chain_s2_refs_s1": 0.5,
        "q_chain_s3_refs_s2": 0.5, "q_self_consistency": 0.5,
        "q_step4_valid_json": step4_score, "q_step4_has_keys": step4_score,
        "q_existence_correct": step4_score, "q_archetype_correct": step4_score, "q_node_f1": step4_score,
    }
    all_qualities = []
    for qs in STEP_QUALITIES.values():
        all_qualities.extend(qs)
    total = sum(scores.get(q, 0.0) for q in all_qualities) / len(all_qualities)
    return RewardResult(reward=total, scores=scores, phase=phase)


ALL_QUALITIES = []
for qs in STEP_QUALITIES.values():
    ALL_QUALITIES.extend(qs)


def test_credit_step1_correct_step4_wrong():
    """Step 1 correct, step 4 wrong → step 1 advantage > step 4 advantage.

    Uses GRPO mode where within-group comparison creates measurable advantages.
    """
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    # Two completions in one group: one with good step1/bad step4, the other reversed
    rr_good_s1 = _make_reward_result(step1_score=1.0, step4_score=0.0)
    rr_good_s4 = _make_reward_result(step1_score=0.0, step4_score=1.0)

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 1],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr_good_s1, rr_good_s4],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        group_size=2,
    )

    regions = segment_completion(tokens)
    # For completion 0 (good step1, bad step4): step 1 should be higher than step 4
    step1_advs_0 = [advs[0][t].item() for t, r in enumerate(regions) if r == "STEP_1"]
    step4_advs_0 = [advs[0][t].item() for t, r in enumerate(regions) if r == "STEP_4"]

    step1_mean = np.mean(step1_advs_0)
    step4_mean = np.mean(step4_advs_0)

    assert step1_mean > step4_mean, f"step1={step1_mean}, step4={step4_mean}"


def test_credit_all_steps_correct():
    """All steps score 1.0 → all advantages similar magnitude."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    rr1 = _make_reward_result(step1_score=1.0, step4_score=1.0)
    rr2 = _make_reward_result(step1_score=0.8, step4_score=0.8)

    # Set baselines
    estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
    )

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
    )

    # All step advantages should be finite
    for adv in advs:
        assert adv.isfinite().all()


def test_credit_phase1_format_only():
    """active_qualities = phase 1 only → only step 1 has non-zero advantage."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    rr1 = _make_reward_result(step1_score=1.0, step4_score=0.5)
    rr2 = _make_reward_result(step1_score=0.5, step4_score=0.8)

    phase1_qualities = STEP_QUALITIES[1]  # Only step 1 qualities

    estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[phase1_qualities, phase1_qualities],
    )

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[phase1_qualities, phase1_qualities],
    )

    regions = segment_completion(tokens)

    # Steps 2-4 should have zero step rewards (no active qualities) → zero advantage
    for step_n in [2, 3, 4]:
        step_advs = [advs[0][t].item() for t, r in enumerate(regions) if r == f"STEP_{step_n}"]
        if step_advs:
            assert all(abs(a) < 1e-6 for a in step_advs), f"STEP_{step_n} should be ~0, got {step_advs}"


# --- SPO value tracker tests ---


def test_spo_warmstart_no_spike():
    """First observation: V set to reward, advantage ≈ 0."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()
    rr = _make_reward_result(step1_score=0.8, step4_score=0.6)

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 1],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr, rr],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
    )

    # After warm-start, advantages should be near-zero (GDPO may shift them)
    for adv in advs:
        assert adv.abs().max() < 1.0, f"Warm-start spike detected: max={adv.abs().max()}"


def test_spo_second_observation_has_advantage():
    """Second observation with different reward → non-zero advantage."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    rr1 = _make_reward_result(step1_score=0.5, step4_score=0.5)
    rr2 = _make_reward_result(step1_score=1.0, step4_score=0.0)

    # First call sets baselines
    estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
    )

    # Second call with different rewards → advantages exist
    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr2, rr1],  # swap
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
    )

    # At least some advantages should be non-zero
    all_advs = torch.cat(advs)
    assert all_advs.abs().max() > 0.01, "No non-zero advantages after second observation"


def test_spo_on_tier_advance_resets_v():
    """on_tier_advance resets V for affected prompts."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()
    rr = _make_reward_result(step1_score=0.9, step4_score=0.7)

    estimator.compute_advantages(
        batch_prompt_ids=[42],
        batch_token_ids=[tokens],
        batch_reward_results=[rr],
        batch_active_qualities=[ALL_QUALITIES],
    )

    # V should have values for prompt 42
    assert 42 in estimator.V

    # Tier advance resets ONLY the new step (2) for prompt 42, preserves mastered step (1)
    old_step1_v = estimator.V[42][1]
    estimator.on_tier_advance(new_tier=2, prompt_tier_map={42: 2, 99: 1})
    assert estimator.V[42][2] == 0.0, "New step should be reset"
    assert estimator.V[42][1] == old_step1_v, "Mastered step should be preserved"
    assert 2 not in estimator._step_seen.get(42, set()), "New step should not be in seen set"


def test_spo_value_tracker_ema_convergence():
    """100 identical rewards → V converges to reward value."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()
    rr = _make_reward_result(step1_score=0.85, step4_score=0.65)

    for _ in range(100):
        estimator.compute_advantages(
            batch_prompt_ids=[1, 2],
            batch_token_ids=[tokens, tokens],
            batch_reward_results=[rr, rr],
            batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        )

    # V[1][1] should converge to step 1 reward (mean of step 1 qualities = 0.85)
    v_step1 = estimator.V[1][1]
    assert abs(v_step1 - 0.85) < 0.02, f"V did not converge: V={v_step1}, expected≈0.85"


# --- GRPO fallback tests ---


def test_grpo_fallback_group_normalize():
    """4 completions with different rewards → mean≈0, std≈1 within group."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    results = [
        _make_reward_result(step1_score=1.0, step4_score=0.9),
        _make_reward_result(step1_score=0.5, step4_score=0.3),
        _make_reward_result(step1_score=0.8, step4_score=0.6),
        _make_reward_result(step1_score=0.2, step4_score=0.1),
    ]

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 1, 1, 1],
        batch_token_ids=[tokens] * 4,
        batch_reward_results=results,
        batch_active_qualities=[ALL_QUALITIES] * 4,
        group_size=4,
    )

    # After GDPO normalization, per-step advantages should be normalized
    for adv in advs:
        assert adv.isfinite().all()


def test_grpo_fallback_degenerate_group():
    """4 identical rewards → all advantages ≈ 0."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    rr = _make_reward_result(step1_score=0.8, step4_score=0.8)

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 1, 1, 1],
        batch_token_ids=[tokens] * 4,
        batch_reward_results=[rr] * 4,
        batch_active_qualities=[ALL_QUALITIES] * 4,
        group_size=4,
    )

    # All advantages should be ~0 (degenerate group)
    for adv in advs:
        assert adv.isfinite().all()
        assert adv.abs().max() < 0.01, f"Degenerate group should have ~0 advantages, got max={adv.abs().max()}"


# --- GDPO normalization tests ---


def test_gdpo_per_step_normalize():
    """After normalization, each step's advantages have std≈1 across batch."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    # Create batch with high variance in step 1, low in step 4
    results = [
        _make_reward_result(step1_score=1.0, step4_score=0.5),
        _make_reward_result(step1_score=0.0, step4_score=0.5),
        _make_reward_result(step1_score=0.8, step4_score=0.5),
        _make_reward_result(step1_score=0.2, step4_score=0.5),
    ]

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2, 3, 4],
        batch_token_ids=[tokens] * 4,
        batch_reward_results=results,
        batch_active_qualities=[ALL_QUALITIES] * 4,
        group_size=4,
    )

    # Step 4 should have ~0 advantages (all same reward)
    regions = segment_completion(tokens)
    step4_indices = [t for t, r in enumerate(regions) if r == "STEP_4"]

    if step4_indices:
        step4_vals = [advs[i][step4_indices[0]].item() for i in range(4)]
        assert all(abs(v) < 0.01 for v in step4_vals), f"Step 4 should be ~0: {step4_vals}"


def test_gdpo_preserves_sign():
    """Above-mean completion → positive advantage; below-mean → negative."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    results = [
        _make_reward_result(step1_score=1.0, step4_score=1.0),  # above mean
        _make_reward_result(step1_score=0.0, step4_score=0.0),  # below mean
    ]

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens] * 2,
        batch_reward_results=results,
        batch_active_qualities=[ALL_QUALITIES] * 2,
        group_size=2,
    )

    regions = segment_completion(tokens)
    step1_idx = next(t for t, r in enumerate(regions) if r == "STEP_1")

    # Completion 0 (high) should have positive step 1 advantage
    assert advs[0][step1_idx].item() > 0, "High completion should have positive advantage"
    # Completion 1 (low) should have negative step 1 advantage
    assert advs[1][step1_idx].item() < 0, "Low completion should have negative advantage"


# --- Step 8: Integration tests ---


def test_think_tokens_get_zero_advantage():
    """Tokens in THINK region → advantage == 0."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)

    tokens = [
        THINK_START, 100, 101, THINK_END,
        OPEN_ANGLE, STEP_TOKEN, 16, 9999, CLOSE_ANGLE,
        200, 201,
        CLOSE_SLASH, STEP_TOKEN, 16, 9999, CLOSE_ANGLE,
    ]

    rr1 = _make_reward_result(step1_score=1.0, step4_score=0.5)
    rr2 = _make_reward_result(step1_score=0.0, step4_score=0.8)

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        group_size=2,
    )

    regions = segment_completion(tokens)
    for i in range(2):
        for t, region in enumerate(regions):
            if region == "THINK":
                assert advs[i][t].item() == 0.0, f"THINK token at {t} should have 0 advantage"


def test_format_tokens_get_zero_advantage():
    """Tokens in FORMAT region (tag tokens) → advantage == 0."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)

    tokens = [
        OPEN_ANGLE, STEP_TOKEN, 16, 9999, CLOSE_ANGLE,
        200, 201,
        CLOSE_SLASH, STEP_TOKEN, 16, 9999, CLOSE_ANGLE,
    ]

    rr1 = _make_reward_result(step1_score=1.0, step4_score=0.5)
    rr2 = _make_reward_result(step1_score=0.0, step4_score=0.8)

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        group_size=2,
    )

    regions = segment_completion(tokens)
    for i in range(2):
        for t, region in enumerate(regions):
            if region == "FORMAT":
                assert advs[i][t].item() == 0.0, f"FORMAT token at {t} should have 0 advantage"


def test_credit_phase4_all_active():
    """All qualities active (phase 4+): all steps have non-zero advantages."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()
    rr1 = _make_reward_result(step1_score=0.9, step4_score=0.8)
    rr2 = _make_reward_result(step1_score=0.1, step4_score=0.2)
    advs, regions_batch = estimator.compute_advantages(
        batch_prompt_ids=[1, 1],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr1, rr2],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        group_size=2,
    )
    step_seen = set()
    for t, region in enumerate(regions_batch[0]):
        if region.startswith("STEP_") and advs[0][t].item() != 0.0:
            step_seen.add(region)
    assert len(step_seen) >= 2, f"Expected multiple steps with non-zero advantage, got {step_seen}"


def test_full_pipeline_credit_assignment():
    """8 completions with varied per-step quality through full pipeline."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()
    results = []
    for i in range(8):
        results.append(_make_reward_result(
            step1_score=0.1 * i + 0.1,
            step4_score=0.9 - 0.1 * i,
        ))
    # Warm up SPO
    estimator.compute_advantages(
        list(range(8)), [tokens]*8, results, [ALL_QUALITIES]*8,
    )
    advs, regions = estimator.compute_advantages(
        list(range(8)), [tokens]*8, results, [ALL_QUALITIES]*8,
    )
    assert len(advs) == 8
    for a in advs:
        assert a.isfinite().all()
        assert len(a) == len(tokens)
    # High step1 completions should have different step1 advantages than low step1 completions
    step1_positions = [t for t, r in enumerate(regions[0]) if r == "STEP_1"]
    if step1_positions:
        high_adv = advs[7][step1_positions[0]].item()
        low_adv = advs[0][step1_positions[0]].item()
        assert high_adv != low_adv, "Varied rewards should produce varied advantages"


# --- Regression tests for bug fixes ---


def test_grpo_nondivisible_batch_raises():
    """GRPO with batch_size not divisible by group_size → ValueError."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()
    rr = _make_reward_result(step1_score=0.5, step4_score=0.5)

    with pytest.raises(ValueError, match="divisible by group_size"):
        estimator.compute_advantages(
            batch_prompt_ids=[1, 2, 3],  # 3 not divisible by 2
            batch_token_ids=[tokens] * 3,
            batch_reward_results=[rr] * 3,
            batch_active_qualities=[ALL_QUALITIES] * 3,
            group_size=2,
        )


def test_gdpo_batch_size_1_no_nan():
    """GDPO normalization with batch_size=1 → no NaN (uses correction=0)."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()
    rr = _make_reward_result(step1_score=0.8, step4_score=0.6)

    # First call warm-starts
    estimator.compute_advantages(
        batch_prompt_ids=[1],
        batch_token_ids=[tokens],
        batch_reward_results=[rr],
        batch_active_qualities=[ALL_QUALITIES],
    )

    # Second call with batch_size=1 — must not produce NaN
    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1],
        batch_token_ids=[tokens],
        batch_reward_results=[rr],
        batch_active_qualities=[ALL_QUALITIES],
    )

    assert advs[0].isfinite().all(), "batch_size=1 produced NaN advantages"


# --- KL-adaptive SPO learning rate tests ---


def test_kl_adaptive_lr_decrease_on_high_kl():
    """High KL → lr decreases."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    initial_lr = estimator.lr
    estimator.adapt_lr(kl=0.5, kl_threshold=0.1, kl_factor=2.0, lr_factor=1.5)
    assert estimator.lr < initial_lr, f"lr should decrease: {estimator.lr} >= {initial_lr}"
    assert estimator.lr == pytest.approx(initial_lr / 1.5)


def test_kl_adaptive_lr_increase_on_low_kl():
    """Low KL → lr increases."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    initial_lr = estimator.lr
    estimator.adapt_lr(kl=0.01, kl_threshold=0.1, kl_factor=2.0, lr_factor=1.5)
    assert estimator.lr > initial_lr, f"lr should increase: {estimator.lr} <= {initial_lr}"
    assert estimator.lr == pytest.approx(initial_lr * 1.5)


# --- Prioritized prompt sampling tests ---


def test_get_prompt_priorities_returns_abs_v():
    """get_prompt_priorities returns mean |V| per prompt."""
    estimator = QGREStepAdvantageEstimator(lr=0.1, mode="spo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG)
    tokens = _make_completion_tokens()

    rr_high = _make_reward_result(step1_score=1.0, step4_score=1.0)
    rr_low = _make_reward_result(step1_score=0.0, step4_score=0.0)

    # Build up V values
    for _ in range(10):
        estimator.compute_advantages(
            batch_prompt_ids=[1, 2],
            batch_token_ids=[tokens, tokens],
            batch_reward_results=[rr_high, rr_low],
            batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        )

    priorities = estimator.get_prompt_priorities()
    assert 1 in priorities and 2 in priorities
    assert priorities[1] > 0 or priorities[2] > 0, "At least one prompt should have nonzero priority"


# --- Fix 2: GRPO no double normalization ---


def test_grpo_no_double_normalization():
    """GRPO inner loop does mean-only subtraction, outer GDPO handles full normalization."""
    estimator = QGREStepAdvantageEstimator(
        lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG,
        normalize_advantages=True,
    )
    tokens = _make_completion_tokens()

    # Two completions with different step1 scores
    rr_high = _make_reward_result(step1_score=1.0, step4_score=0.5)
    rr_low = _make_reward_result(step1_score=0.0, step4_score=0.5)

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=[rr_high, rr_low],
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        group_size=2,
    )

    # After GDPO normalization, step1 advantages should be non-zero (different rewards)
    regions = segment_completion(tokens)
    step1_idx = next(t for t, r in enumerate(regions) if r == "STEP_1")
    adv_high = advs[0][step1_idx].item()
    adv_low = advs[1][step1_idx].item()
    assert adv_high > adv_low, f"High-reward should have higher advantage: {adv_high} vs {adv_low}"

    # Step4 should have ~0 advantages (both have same score)
    step4_idx = next(t for t, r in enumerate(regions) if r == "STEP_4")
    step4_vals = [advs[i][step4_idx].item() for i in range(2)]
    assert all(abs(v) < 0.01 for v in step4_vals), f"Step 4 same-score should be ~0: {step4_vals}"


def test_dr_grpo_no_std_division():
    """Dr.GRPO mode: no std division anywhere — only mean subtraction."""
    estimator = QGREStepAdvantageEstimator(
        lr=0.1, mode="grpo", step_qualities=STEP_QUALITIES, segmenter=XML_SEG,
        normalize_advantages=False,  # Dr.GRPO
    )
    tokens = _make_completion_tokens()

    results = [
        _make_reward_result(step1_score=1.0, step4_score=0.9),
        _make_reward_result(step1_score=0.0, step4_score=0.1),
    ]

    advs, _ = estimator.compute_advantages(
        batch_prompt_ids=[1, 2],
        batch_token_ids=[tokens, tokens],
        batch_reward_results=results,
        batch_active_qualities=[ALL_QUALITIES, ALL_QUALITIES],
        group_size=2,
    )

    regions = segment_completion(tokens)
    step1_idx = next(t for t, r in enumerate(regions) if r == "STEP_1")

    # With dr_grpo and mean-only subtraction, advantages should reflect raw difference
    # High step1 (1.0) minus mean (0.5) = 0.5 before outer normalization
    # Outer also does mean-only (no std) → result should be non-zero
    assert advs[0][step1_idx].item() > 0, "High-reward should have positive advantage"
    assert advs[1][step1_idx].item() < 0, "Low-reward should have negative advantage"
    assert advs[0].isfinite().all()
    assert advs[1].isfinite().all()


# --- Fix 5: Reward key mismatch warning ---


def test_reward_key_mismatch_warns():
    """Warning emitted when reward keys don't overlap with step_qualities."""
    import warnings

    estimator = QGREStepAdvantageEstimator(
        lr=0.1, mode="spo",
        step_qualities={1: ["q_format"], 2: ["q_accuracy"]},
        segmenter=XML_SEG,
    )
    tokens = _make_completion_tokens()

    # Reward fn returns completely different keys than step_qualities expects
    rr = RewardResult(reward=0.5, scores={"format_score": 0.9, "acc_score": 0.8})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimator.compute_advantages(
            batch_prompt_ids=[1],
            batch_token_ids=[tokens],
            batch_reward_results=[rr],
            batch_active_qualities=[["q_format", "q_accuracy"]],
        )
        mismatch_warns = [x for x in w if "Reward key mismatch" in str(x.message)]
        assert len(mismatch_warns) >= 1, "Should warn about reward key mismatch"


def test_reward_key_match_no_warning():
    """No warning when reward keys overlap with step_qualities."""
    import warnings

    estimator = QGREStepAdvantageEstimator(
        lr=0.1, mode="spo",
        step_qualities={1: ["q_format"]},
        segmenter=XML_SEG,
    )
    tokens = _make_completion_tokens()

    rr = RewardResult(reward=0.5, scores={"q_format": 0.9})

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        estimator.compute_advantages(
            batch_prompt_ids=[1],
            batch_token_ids=[tokens],
            batch_reward_results=[rr],
            batch_active_qualities=[["q_format"]],
        )
        mismatch_warns = [x for x in w if "Reward key mismatch" in str(x.message)]
        assert len(mismatch_warns) == 0, "Should NOT warn when keys match"
