"""Test EGRS 2x2 matrix logic."""

import torch

from qgre.advantages import apply_egrs_matrix, compute_span_correctness
from qgre.types import RewardResult


def test_compute_span_correctness_all_correct():
    """Test span correctness when all scores meet threshold."""
    reward_result = RewardResult(
        reward=0.9,
        scores={"q_format": 0.8, "q_accuracy": 0.9, "q_grounding": 0.7},
    )
    step_qualities = {1: ["q_format"], 2: ["q_accuracy", "q_grounding"]}

    correctness = compute_span_correctness(reward_result, step_qualities, threshold=0.5)

    assert correctness[1] is True, "Step 1 should be correct (0.8 >= 0.5)"
    assert correctness[2] is True, "Step 2 should be correct (0.9 >= 0.5 and 0.7 >= 0.5)"


def test_compute_span_correctness_partial():
    """Test span correctness with partial failures."""
    reward_result = RewardResult(
        reward=0.5,
        scores={"q_format": 0.8, "q_accuracy": 0.3, "q_grounding": 0.2},
    )
    step_qualities = {1: ["q_format"], 2: ["q_accuracy", "q_grounding"]}

    correctness = compute_span_correctness(reward_result, step_qualities, threshold=0.5)

    assert correctness[1] is True, "Step 1 should be correct (0.8 >= 0.5)"
    assert correctness[2] is False, "Step 2 should be wrong (0.3 < 0.5)"


def test_compute_span_correctness_missing_quality():
    """Test handling of missing quality keys."""
    reward_result = RewardResult(
        reward=0.5,
        scores={"q_format": 0.8},  # Missing q_accuracy
    )
    step_qualities = {1: ["q_format"], 2: ["q_accuracy"]}

    correctness = compute_span_correctness(reward_result, step_qualities, threshold=0.5)

    assert correctness[1] is True
    assert correctness[2] is False, "Missing quality defaults to 0.0"


def test_apply_egrs_matrix_q1_uncertain_correct():
    """Q1: Uncertain + Correct → Reinforce with scaling."""
    seq_len = 4
    token_advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])
    regions = ["STEP_1", "STEP_1", "STEP_1", "STEP_1"]
    # High entropy = uncertain
    token_entropy = torch.tensor([0.8, 0.8, 0.8, 0.8])
    step_correctness = {1: True}  # Correct

    modified, entropy_adj, hints = apply_egrs_matrix(
        token_advantages, regions, token_entropy, step_correctness,
        entropy_threshold=0.5, gate_temperature=0.1,
    )

    # Q1: advantage should be scaled by gate (~1 for uncertain)
    assert modified[0].item() > 0.5, f"Q1 should retain most advantage, got {modified[0]}"
    # No entropy adjustment for Q1
    assert entropy_adj[0].item() == 0.0, "Q1 should have no entropy adjustment"
    # No hints for Q1
    assert len(hints) == 0, "Q1 should not flag hints"


def test_apply_egrs_matrix_q2_confident_correct():
    """Q2: Confident + Correct → Zero advantage (already learned)."""
    seq_len = 4
    token_advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])
    regions = ["STEP_1", "STEP_1", "STEP_1", "STEP_1"]
    # Low entropy = confident
    token_entropy = torch.tensor([0.1, 0.1, 0.1, 0.1])
    step_correctness = {1: True}  # Correct

    modified, entropy_adj, hints = apply_egrs_matrix(
        token_advantages, regions, token_entropy, step_correctness,
        entropy_threshold=0.5, gate_temperature=0.1,
    )

    # Q2: advantage should be 0
    assert modified[0].item() == 0.0, f"Q2 should have 0 advantage, got {modified[0]}"
    # No entropy adjustment
    assert entropy_adj[0].item() == 0.0
    # No hints
    assert len(hints) == 0


def test_apply_egrs_matrix_q3_confident_wrong():
    """Q3: Confident + Wrong → Zero advantage, entropy boost."""
    seq_len = 4
    token_advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])
    regions = ["STEP_1", "STEP_1", "STEP_1", "STEP_1"]
    # Low entropy = confident
    token_entropy = torch.tensor([0.1, 0.1, 0.1, 0.1])
    step_correctness = {1: False}  # Wrong

    modified, entropy_adj, hints = apply_egrs_matrix(
        token_advantages, regions, token_entropy, step_correctness,
        entropy_threshold=0.5, gate_temperature=0.1, exploration_weight=0.2,
    )

    # Q3: advantage should be 0
    assert modified[0].item() == 0.0, f"Q3 should have 0 advantage, got {modified[0]}"
    # Q3: entropy adjustment should be exploration_weight
    assert abs(entropy_adj[0].item() - 0.2) < 0.001, f"Q3 should have entropy adj=0.2, got {entropy_adj[0]}"
    # No hints for Q3 (will get hint after entropy boost makes it uncertain)
    assert len(hints) == 0


def test_apply_egrs_matrix_q4_uncertain_wrong():
    """Q4: Uncertain + Wrong → Zero advantage, flag for hint."""
    seq_len = 4
    token_advantages = torch.tensor([1.0, 1.0, 1.0, 1.0])
    regions = ["STEP_1", "STEP_1", "STEP_1", "STEP_1"]
    # High entropy = uncertain
    token_entropy = torch.tensor([0.8, 0.8, 0.8, 0.8])
    step_correctness = {1: False}  # Wrong

    modified, entropy_adj, hints = apply_egrs_matrix(
        token_advantages, regions, token_entropy, step_correctness,
        entropy_threshold=0.5, gate_temperature=0.1,
    )

    # Q4: advantage should be 0
    assert modified[0].item() == 0.0, f"Q4 should have 0 advantage, got {modified[0]}"
    # Q4: no entropy adjustment (already uncertain)
    assert entropy_adj[0].item() == 0.0
    # Q4: should be flagged for hint
    assert len(hints) == 4, f"All 4 Q4 tokens should be flagged, got {len(hints)}"
    assert (1, 0) in hints, "Token 0 should be flagged"


def test_apply_egrs_matrix_mixed_quadrants():
    """Test sequence with tokens in different quadrants."""
    # 8 tokens: 2 per quadrant
    token_advantages = torch.ones(8)
    regions = ["STEP_1", "STEP_1", "STEP_2", "STEP_2", "STEP_3", "STEP_3", "STEP_4", "STEP_4"]
    # Entropy pattern: step 1,3 uncertain (0.8), step 2,4 confident (0.2)
    token_entropy = torch.tensor([0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0.2, 0.2])
    # Correctness: step 1,2 correct, step 3,4 wrong
    step_correctness = {1: True, 2: True, 3: False, 4: False}

    modified, entropy_adj, hints = apply_egrs_matrix(
        token_advantages, regions, token_entropy, step_correctness,
        entropy_threshold=0.5, gate_temperature=0.1, exploration_weight=0.15,
    )

    # Step 1 (uncertain + correct) → Q1: scaled advantage
    assert modified[0].item() > 0.5, f"Q1 should have scaled advantage, got {modified[0]}"
    assert modified[1].item() > 0.5, f"Q1 should have scaled advantage, got {modified[1]}"

    # Step 2 (confident + correct) → Q2: zero advantage
    assert modified[2].item() == 0.0, f"Q2 should have 0 advantage, got {modified[2]}"
    assert modified[3].item() == 0.0, f"Q2 should have 0 advantage, got {modified[3]}"

    # Step 3 (uncertain + wrong) → Q4: zero advantage, hint flag
    assert modified[4].item() == 0.0, f"Q4 should have 0 advantage, got {modified[4]}"
    assert (3, 4) in hints, "Token 4 (step 3) should be flagged for hint"
    assert (3, 5) in hints, "Token 5 (step 3) should be flagged for hint"

    # Step 4 (confident + wrong) → Q3: zero advantage, entropy boost
    assert modified[6].item() == 0.0, f"Q3 should have 0 advantage, got {modified[6]}"
    assert abs(entropy_adj[6].item() - 0.15) < 0.001, f"Q3 should have entropy adj, got {entropy_adj[6]}"
    assert abs(entropy_adj[7].item() - 0.15) < 0.001, f"Q3 should have entropy adj, got {entropy_adj[7]}"


def test_apply_egrs_matrix_format_region_ignored():
    """FORMAT regions should not get EGRS treatment."""
    token_advantages = torch.tensor([1.0, 1.0, 1.0])
    regions = ["FORMAT", "STEP_1", "FORMAT"]
    token_entropy = torch.tensor([0.1, 0.1, 0.1])  # All confident
    step_correctness = {1: True}

    modified, entropy_adj, hints = apply_egrs_matrix(
        token_advantages, regions, token_entropy, step_correctness,
    )

    # FORMAT tokens should be unchanged
    assert modified[0].item() == 1.0, "FORMAT token should be unchanged"
    assert modified[2].item() == 1.0, "FORMAT token should be unchanged"
    # STEP_1 (confident + correct) → Q2: zero
    assert modified[1].item() == 0.0, "STEP_1 Q2 should be zero"


def test_apply_egrs_matrix_think_region():
    """THINK region should map to step 0."""
    token_advantages = torch.tensor([1.0, 1.0])
    regions = ["THINK", "THINK"]
    token_entropy = torch.tensor([0.8, 0.8])  # Uncertain
    step_correctness = {0: False}  # THINK is wrong

    modified, entropy_adj, hints = apply_egrs_matrix(
        token_advantages, regions, token_entropy, step_correctness,
    )

    # THINK (uncertain + wrong) → Q4: hint flag
    assert (0, 0) in hints, "THINK token should be flagged for hint"
    assert (0, 1) in hints, "THINK token should be flagged for hint"


if __name__ == "__main__":
    test_compute_span_correctness_all_correct()
    test_compute_span_correctness_partial()
    test_compute_span_correctness_missing_quality()
    test_apply_egrs_matrix_q1_uncertain_correct()
    test_apply_egrs_matrix_q2_confident_correct()
    test_apply_egrs_matrix_q3_confident_wrong()
    test_apply_egrs_matrix_q4_uncertain_wrong()
    test_apply_egrs_matrix_mixed_quadrants()
    test_apply_egrs_matrix_format_region_ignored()
    test_apply_egrs_matrix_think_region()
    print("\n" + "=" * 50)
    print("All EGRS matrix tests PASSED")
