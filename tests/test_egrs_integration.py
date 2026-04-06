"""Integration test for EGRS (Entropy-Gated Reinforcement System)."""

import torch

from qgre.advantages import apply_egrs_matrix, compute_span_correctness
from qgre.attention_bonds import compute_confidence_gate, compute_normalized_entropy
from qgre.config import EGRSConfig, QGREConfig
from qgre.hints import HintRegistry
from qgre.types import RewardResult


def test_egrs_config_defaults():
    """Verify EGRS config has sensible defaults."""
    cfg = QGREConfig()
    assert cfg.egrs.enabled is False
    assert cfg.egrs.reward_threshold == 0.5
    assert cfg.egrs.entropy_threshold == 0.5
    assert cfg.egrs.gate_temperature == 0.1
    assert cfg.egrs.exploration_weight == 0.1
    assert cfg.egrs.hint_enabled is True
    assert cfg.egrs.mastery_threshold == 0.8


def test_egrs_full_pipeline():
    """Test complete EGRS pipeline: logits → entropy → matrix → loss components."""
    batch, seq, vocab = 1, 9, 1000
    egrs_cfg = EGRSConfig(enabled=True)

    # Create mock logits with specific confidence patterns
    logits = torch.zeros(batch, seq, vocab)
    # Tokens 0-2: STEP_1 - confident (peaked)
    logits[:, 0:3, 0] = 15.0
    # Tokens 3-5: STEP_2 - also confident (peaked at different token)
    logits[:, 3:6, 5] = 15.0
    # Tokens 6-8: STEP_3 - uncertain (uniform)
    # Already zero = uniform

    # Step 1: Compute normalized entropy
    token_entropy = compute_normalized_entropy(logits)
    assert token_entropy.shape == (batch, seq)
    assert token_entropy[0, 0].item() < 0.1, "Peaked should have low entropy"
    assert token_entropy[0, 8].item() > 0.9, "Uniform should have high entropy"

    # Step 2: Create reward result and compute span correctness
    # STEP_1: correct, STEP_2: wrong, STEP_3: correct
    reward_result = RewardResult(
        reward=0.7,
        scores={"q_step1": 0.8, "q_step2": 0.3, "q_step3": 0.9},
    )
    step_qualities = {1: ["q_step1"], 2: ["q_step2"], 3: ["q_step3"]}
    span_correctness = compute_span_correctness(
        reward_result, step_qualities, threshold=egrs_cfg.reward_threshold
    )
    assert span_correctness[1] is True, "Step 1 correct (0.8 >= 0.5)"
    assert span_correctness[2] is False, "Step 2 wrong (0.3 < 0.5)"
    assert span_correctness[3] is True, "Step 3 correct (0.9 >= 0.5)"

    # Step 3: Create regions and advantages
    # Tokens 0-2: STEP_1 (correct, confident) → Q2
    # Tokens 3-5: STEP_2 (wrong, confident) → Q3
    # Tokens 6-8: STEP_3 (correct, uncertain) → Q1
    regions = ["STEP_1"] * 3 + ["STEP_2"] * 3 + ["STEP_3"] * 3
    token_advantages = torch.ones(seq)

    # Step 4: Apply EGRS matrix
    modified_advs, entropy_adj, hint_flags = apply_egrs_matrix(
        token_advantages,
        regions,
        token_entropy[0],  # First batch item
        span_correctness,
        entropy_threshold=egrs_cfg.entropy_threshold,
        gate_temperature=egrs_cfg.gate_temperature,
        exploration_weight=egrs_cfg.exploration_weight,
    )

    # Step 5: Verify quadrant behavior
    # Tokens 0-2: STEP_1 (correct), confident → Q2 (zero advantage)
    assert modified_advs[0].item() == 0.0, f"Q2 should have 0 adv, got {modified_advs[0]}"
    assert modified_advs[1].item() == 0.0, f"Q2 should have 0 adv, got {modified_advs[1]}"

    # Tokens 3-5: STEP_2 (wrong), confident → Q3 (entropy boost)
    assert entropy_adj[3].item() > 0, f"Q3 should have entropy adj, got {entropy_adj[3]}"
    assert entropy_adj[4].item() > 0, f"Q3 should have entropy adj, got {entropy_adj[4]}"
    assert modified_advs[3].item() == 0.0, f"Q3 should have 0 adv, got {modified_advs[3]}"

    # Tokens 6-8: STEP_3 (correct), uncertain → Q1 (scaled advantage)
    assert modified_advs[8].item() > 0.5, f"Q1 should have scaled adv, got {modified_advs[8]}"
    assert entropy_adj[8].item() == 0.0, f"Q1 should have no entropy adj"

    # No hint flags (Q3 doesn't flag, only Q4 does)
    assert len(hint_flags) == 0, f"No Q4 tokens expected, got {hint_flags}"

    print(f"Token entropy: {[f'{e:.2f}' for e in token_entropy[0].tolist()]}")
    print(f"Modified advantages: {[f'{a:.2f}' for a in modified_advs.tolist()]}")
    print(f"Entropy adjustments: {[f'{e:.2f}' for e in entropy_adj.tolist()]}")
    print(f"Hint flags: {hint_flags}")


def test_egrs_with_hint_registry():
    """Test EGRS integration with hint registry."""
    registry = HintRegistry(mastery_threshold=0.8)
    seq = 6
    regions = ["STEP_1", "STEP_1", "STEP_2", "STEP_2", "STEP_3", "STEP_3"]
    token_advantages = torch.ones(seq)
    # All uncertain
    token_entropy = torch.full((seq,), 0.8)
    # Step 1: correct, Step 2: wrong, Step 3: wrong
    span_correctness = {1: True, 2: False, 3: False}

    modified_advs, entropy_adj, hint_flags = apply_egrs_matrix(
        token_advantages,
        regions,
        token_entropy,
        span_correctness,
    )

    # Q4 tokens (uncertain + wrong) should be flagged
    # Step 2 and Step 3 are wrong and uncertain
    assert (2, 2) in hint_flags, "STEP_2 token 2 should be flagged"
    assert (2, 3) in hint_flags, "STEP_2 token 3 should be flagged"
    assert (3, 4) in hint_flags, "STEP_3 token 4 should be flagged"
    assert (3, 5) in hint_flags, "STEP_3 token 5 should be flagged"

    # Step 1 (correct) should not be flagged
    assert (1, 0) not in hint_flags
    assert (1, 1) not in hint_flags

    # Register hints
    for step_num, t in hint_flags:
        span_id = f"STEP_{step_num}"
        registry.flag_for_hint(
            prompt_id=1,
            span_id=span_id,
            hint_tokens=[100, 101],  # Mock tokens
            current_mastery=0.2,
            current_step=10,
        )

    # Verify registry state
    assert len(registry) == 2, f"Should have 2 unique (prompt, span) entries, got {len(registry)}"
    assert (1, "STEP_2") in registry
    assert (1, "STEP_3") in registry


def test_egrs_entropy_loss_computation():
    """Test entropy loss computation matches expected formula."""
    batch, seq = 2, 5
    # Mock entropy adjustments (only Q3 tokens have non-zero)
    entropy_adj = torch.tensor([
        [0.0, 0.0, 0.1, 0.1, 0.0],  # Q3 at positions 2,3
        [0.1, 0.0, 0.0, 0.0, 0.1],  # Q3 at positions 0,4
    ])
    # Mock token entropy
    token_entropy = torch.tensor([
        [0.2, 0.5, 0.3, 0.4, 0.8],
        [0.1, 0.6, 0.7, 0.2, 0.3],
    ])
    # Mask (all tokens valid)
    mask = torch.ones(batch, seq)

    # EGRS loss = -sum(adj * entropy * mask)
    egrs_loss = -(entropy_adj * token_entropy * mask).sum()

    # Manual calculation:
    # Sample 0: -0.1*0.3 + -0.1*0.4 = -0.07
    # Sample 1: -0.1*0.1 + -0.1*0.3 = -0.04
    # Total: -0.11
    expected = -(0.1*0.3 + 0.1*0.4 + 0.1*0.1 + 0.1*0.3)
    assert abs(egrs_loss.item() - expected) < 0.001, f"Expected {expected}, got {egrs_loss.item()}"


def test_egrs_mastery_decay_with_game_state():
    """Test that hint decay uses GameState tier_mastery for per-span decay."""
    from qgre.types import GameState
    from qgre.hints import HintRegistry

    # Create GameState with mastery data
    game_state = GameState()
    # Record mastery: STEP_1 mastered (0.9), STEP_2 not mastered (0.2)
    for _ in range(10):
        game_state.record_tier_step_score("default", 1, 0.9)
        game_state.record_tier_step_score("default", 2, 0.2)

    # Verify mastery values
    step1_mastery = game_state.get_tier_step_mastery("default", 1)
    step2_mastery = game_state.get_tier_step_mastery("default", 2)
    assert abs(step1_mastery - 0.9) < 0.01, f"STEP_1 mastery should be ~0.9, got {step1_mastery}"
    assert abs(step2_mastery - 0.2) < 0.01, f"STEP_2 mastery should be ~0.2, got {step2_mastery}"

    # Create registry with hints for both spans
    # R3-MIO-002: Pass seed for deterministic test behavior
    registry = HintRegistry(mastery_threshold=0.8, seed=42)
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)
    registry.flag_for_hint(1, "STEP_2", [200], 0.0, 0)

    # Create mastery lookup function (same pattern as trainer.py)
    def make_mastery_fn(tier: str):
        def mastery_fn(span_id: str) -> float:
            if span_id.startswith("STEP_"):
                try:
                    step_num = int(span_id.split("_")[1])
                    return game_state.get_tier_step_mastery(tier, step_num)
                except (IndexError, ValueError):
                    return 0.0
            return 0.0
        return mastery_fn

    # Get hints with mastery decay (seed=42 ensures deterministic random)
    hints = registry.get_hints_for_prompt(1, mastery_fn=make_mastery_fn("default"))

    # STEP_1 (mastery=0.9 > threshold=0.8) should NOT inject
    # STEP_2 (mastery=0.2 < threshold=0.8) should inject
    assert "STEP_1" not in hints, "STEP_1 should be suppressed (mastery > threshold)"
    assert "STEP_2" in hints, "STEP_2 should inject (mastery < threshold)"

    print(f"STEP_1 mastery: {step1_mastery:.2f} → suppressed")
    print(f"STEP_2 mastery: {step2_mastery:.2f} → injected")


def test_n_completions_aggregation():
    """Test that hint success tracking aggregates correctly with n_completions > 1.

    When the same prompt_id appears multiple times (n_completions > 1),
    we aggregate results before updating the registry:
    - ANY failure → record_failure (model hasn't learned)
    - ALL succeeded WITHOUT hint → can graduate
    - ALL succeeded but some WITH hint → don't graduate yet
    """
    from qgre.hints import HintRegistry

    registry = HintRegistry(mastery_threshold=0.8, success_streak_to_clear=2)
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)

    # Simulate n_completions=4 for same prompt
    # 3 succeed without hint, 1 fails
    outcomes = {
        (1, "STEP_1"): {"success_no_hint": 3, "success_with_hint": 0, "failure": 1}
    }

    # ANY failure should trigger record_failure
    for (pid, span_id), counts in outcomes.items():
        if counts["failure"] > 0:
            registry.record_failure(pid, span_id)

    # Streak should be reset to 0
    assert registry._hints[(1, "STEP_1")].success_count == 0, "Failure should reset streak"

    # Now simulate all 4 succeeding without hint
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)  # Re-flag
    outcomes2 = {
        (1, "STEP_1"): {"success_no_hint": 4, "success_with_hint": 0, "failure": 0}
    }

    for (pid, span_id), counts in outcomes2.items():
        if counts["failure"] == 0 and counts["success_no_hint"] > 0 and counts["success_with_hint"] == 0:
            registry.record_success(pid, span_id, hint_was_used=False)

    assert registry._hints[(1, "STEP_1")].success_count == 1, "Should increment streak"

    # Another round of all succeeding without hint → should graduate (streak=2)
    graduated = registry.record_success(1, "STEP_1", hint_was_used=False)
    assert graduated, "Should graduate after 2 consecutive successes without hint"
    assert (1, "STEP_1") not in registry, "Entry should be removed after graduation"


def test_hint_registry_checkpoint_roundtrip():
    """Test that hint registry survives checkpoint save/restore."""
    from qgre.hints import HintRegistry

    # Create registry with some state
    registry = HintRegistry(mastery_threshold=0.75, success_streak_to_clear=3)
    registry.flag_for_hint(1, "STEP_1", [100, 101, 102], 0.2, 10)
    registry.flag_for_hint(1, "STEP_2", [200, 201], 0.3, 15)
    registry.flag_for_hint(2, "STEP_1", [300], 0.1, 20)

    # Simulate some usage
    registry._hints[(1, "STEP_1")].success_count = 2
    registry._hints[(1, "STEP_1")].total_uses = 5

    # Serialize
    data = registry.to_dict()

    # Verify serialized structure
    assert data["mastery_threshold"] == 0.75
    assert data["success_streak_to_clear"] == 3
    assert len(data["hints"]) == 3

    # Deserialize
    restored = HintRegistry.from_dict(data)

    # Verify restored state
    assert restored.mastery_threshold == 0.75
    assert restored.success_streak_to_clear == 3
    assert len(restored) == 3
    assert (1, "STEP_1") in restored
    assert (1, "STEP_2") in restored
    assert (2, "STEP_1") in restored

    # Verify entry details preserved
    entry = restored._hints[(1, "STEP_1")]
    assert entry.hint_tokens == [100, 101, 102]
    assert entry.success_count == 2
    assert entry.total_uses == 5
    assert entry.mastery_at_flag == 0.2
    assert entry.flagged_step == 10


def test_hint_extractor_with_tokenizer():
    """Test hint extraction produces valid tokens that can be decoded."""
    from qgre.hints import make_hamiltonian_hint_extractor
    from transformers import AutoTokenizer

    # Use a real tokenizer (or mock if not available)
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    except Exception:
        # Skip if tokenizer not available
        import pytest
        pytest.skip("Tokenizer not available")

    extractor = make_hamiltonian_hint_extractor()

    metadata = {
        "T_expr": "p**2/(2*m)",
        "V_expr": "m*g*x",
        "H_expr": "p**2/(2*m) + m*g*x",
        "ground_truth": "H = p**2/(2*m) + m*g*x; dq/dt = p/m; dp/dt = -m*g",
    }

    # Extract and tokenize STEP_5 (Hamiltonian)
    hint_text = extractor("STEP_5", metadata)
    assert hint_text == "H = p**2/(2*m) + m*g*x"

    # Tokenize
    hint_tokens = tokenizer.encode(hint_text, add_special_tokens=False)
    assert len(hint_tokens) > 0, "Should produce tokens"

    # Decode back
    decoded = tokenizer.decode(hint_tokens, skip_special_tokens=True)
    assert "H = " in decoded, "Decoded should contain hint"

    # Test STEP_6 (equations)
    eqn_text = extractor("STEP_6", metadata)
    eqn_tokens = tokenizer.encode(eqn_text, add_special_tokens=False)
    eqn_decoded = tokenizer.decode(eqn_tokens, skip_special_tokens=True)
    assert "dq/dt" in eqn_decoded or "dp/dt" in eqn_decoded


def test_egrs_config_hint_extractor_options():
    """Test EGRS config parses hint_extractor options correctly."""
    from qgre.config import EGRSConfig

    # Default config
    cfg = EGRSConfig()
    assert cfg.hint_extractor == "none"
    assert cfg.hint_extractor_mapping == {}

    # Hamiltonian extractor
    cfg2 = EGRSConfig(hint_extractor="hamiltonian")
    assert cfg2.hint_extractor == "hamiltonian"

    # Generic extractor with mapping
    cfg3 = EGRSConfig(
        hint_extractor="generic",
        hint_extractor_mapping={"STEP_1": "answer_1", "STEP_2": "answer_2"}
    )
    assert cfg3.hint_extractor == "generic"
    assert cfg3.hint_extractor_mapping["STEP_1"] == "answer_1"


def test_apply_egrs_matrix_all_quadrants():
    """Test that apply_egrs_matrix correctly handles all 4 quadrants."""
    from qgre.advantages import apply_egrs_matrix
    import torch

    seq_len = 8
    token_advantages = torch.ones(seq_len)

    # Set up regions: 2 tokens per step
    regions = ["STEP_1", "STEP_1", "STEP_2", "STEP_2", "STEP_3", "STEP_3", "STEP_4", "STEP_4"]

    # Set up entropy: STEP_1,3 confident (low), STEP_2,4 uncertain (high)
    token_entropy = torch.tensor([0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9])

    # Set up correctness: STEP_1,2 correct, STEP_3,4 wrong
    step_correctness = {1: True, 2: True, 3: False, 4: False}

    modified_advs, entropy_adj, hint_flags = apply_egrs_matrix(
        token_advantages,
        regions,
        token_entropy,
        step_correctness,
        entropy_threshold=0.5,
        gate_temperature=0.1,
        exploration_weight=0.1,
    )

    # Q2: STEP_1 (confident + correct) → zero advantage
    assert modified_advs[0].item() == 0.0, "Q2 should have zero advantage"
    assert modified_advs[1].item() == 0.0, "Q2 should have zero advantage"
    assert entropy_adj[0].item() == 0.0, "Q2 should have no entropy adjustment"

    # Q1: STEP_2 (uncertain + correct) → scaled advantage
    assert modified_advs[2].item() > 0.5, "Q1 should have positive scaled advantage"
    assert modified_advs[3].item() > 0.5, "Q1 should have positive scaled advantage"
    assert entropy_adj[2].item() == 0.0, "Q1 should have no entropy adjustment"

    # Q3: STEP_3 (confident + wrong) → zero advantage + entropy boost
    assert modified_advs[4].item() == 0.0, "Q3 should have zero advantage"
    assert modified_advs[5].item() == 0.0, "Q3 should have zero advantage"
    assert entropy_adj[4].item() > 0.0, "Q3 should have entropy adjustment"
    assert entropy_adj[5].item() > 0.0, "Q3 should have entropy adjustment"

    # Q4: STEP_4 (uncertain + wrong) → zero advantage + hint flag
    assert modified_advs[6].item() == 0.0, "Q4 should have zero advantage"
    assert modified_advs[7].item() == 0.0, "Q4 should have zero advantage"
    assert entropy_adj[6].item() == 0.0, "Q4 should have no entropy adjustment"

    # Check hint flags for Q4
    q4_flags = [(s, t) for s, t in hint_flags if s == 4]
    assert len(q4_flags) == 2, f"Q4 should flag 2 tokens, got {len(q4_flags)}"


def test_compute_span_correctness_edge_cases():
    """Test span correctness computation with edge cases."""
    from qgre.advantages import compute_span_correctness
    from qgre.types import RewardResult

    # Normal case
    rr = RewardResult(
        reward=0.7,
        scores={"q_step1": 0.8, "q_step2": 0.3, "q_step3": 0.6}
    )
    step_qualities = {1: ["q_step1"], 2: ["q_step2"], 3: ["q_step3"]}

    result = compute_span_correctness(rr, step_qualities, threshold=0.5)
    assert result[1] is True, "Step 1 should be correct (0.8 >= 0.5)"
    assert result[2] is False, "Step 2 should be wrong (0.3 < 0.5)"
    assert result[3] is True, "Step 3 should be correct (0.6 >= 0.5)"

    # Multiple qualities per step (all must pass)
    rr2 = RewardResult(
        reward=0.5,
        scores={"q_a": 0.8, "q_b": 0.4, "q_c": 0.9}
    )
    step_qualities2 = {1: ["q_a", "q_b"]}  # q_b fails
    result2 = compute_span_correctness(rr2, step_qualities2, threshold=0.5)
    assert result2[1] is False, "Step 1 should be wrong (q_b=0.4 < 0.5)"

    # Missing quality key defaults to 0.0
    rr3 = RewardResult(reward=0.5, scores={"q_exists": 0.8})
    step_qualities3 = {1: ["q_exists", "q_missing"]}
    result3 = compute_span_correctness(rr3, step_qualities3, threshold=0.5)
    assert result3[1] is False, "Should be wrong (missing key defaults to 0.0)"


if __name__ == "__main__":
    test_egrs_config_defaults()
    test_egrs_full_pipeline()
    test_egrs_with_hint_registry()
    test_egrs_entropy_loss_computation()
    test_egrs_mastery_decay_with_game_state()
    test_n_completions_aggregation()
    test_hint_registry_checkpoint_roundtrip()
    test_hint_extractor_with_tokenizer()
    test_egrs_config_hint_extractor_options()
    test_apply_egrs_matrix_all_quadrants()
    test_compute_span_correctness_edge_cases()
    print("\n" + "=" * 50)
    print("All EGRS integration tests PASSED")
