"""Test HintRegistry for EGRS Q4 directional guidance."""

import random

from qgre.hints import HintRegistry


def test_flag_and_get_hint():
    """Test basic flag and get operations."""
    registry = HintRegistry(mastery_threshold=0.8)

    # Flag a hint
    registry.flag_for_hint(
        prompt_id=1,
        span_id="STEP_1",
        hint_tokens=[100, 101, 102],
        current_mastery=0.0,
        current_step=10,
    )

    # Should have one hint
    assert len(registry) == 1
    assert (1, "STEP_1") in registry

    # Get hint with low mastery (should always return)
    random.seed(42)  # For reproducibility
    hint = registry.get_hint(1, "STEP_1", current_mastery=0.0)
    assert hint == [100, 101, 102]


def test_hint_probability_decay():
    """Test that hint probability decays with mastery."""
    registry = HintRegistry(mastery_threshold=0.8)
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)

    # Test probability at different mastery levels
    # mastery=0 → prob=1.0
    prob_0 = registry._compute_hint_probability(0.0)
    assert prob_0 == 1.0, f"Expected 1.0 at mastery=0, got {prob_0}"

    # mastery=0.4 → prob=0.5
    prob_mid = registry._compute_hint_probability(0.4)
    assert abs(prob_mid - 0.5) < 0.001, f"Expected 0.5 at mastery=0.4, got {prob_mid}"

    # mastery=0.8 → prob=0.0
    prob_full = registry._compute_hint_probability(0.8)
    assert prob_full == 0.0, f"Expected 0.0 at mastery=0.8, got {prob_full}"

    # mastery>threshold → prob=0.0
    prob_over = registry._compute_hint_probability(1.0)
    assert prob_over == 0.0, f"Expected 0.0 at mastery>threshold, got {prob_over}"


def test_per_span_mastery_decay():
    """Test that each span uses its own mastery for decay."""
    registry = HintRegistry(mastery_threshold=0.8)

    # Flag two spans with different mastery levels
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)  # Will have high mastery (0.8)
    registry.flag_for_hint(1, "STEP_2", [200], 0.0, 0)  # Will have low mastery (0.0)

    # Create mastery_fn that returns different values per span
    def mastery_fn(span_id: str) -> float:
        if span_id == "STEP_1":
            return 0.8  # Mastered → no hint
        if span_id == "STEP_2":
            return 0.0  # Not mastered → always hint
        return 0.0

    # Get hints - only STEP_2 should return (STEP_1 mastered)
    random.seed(42)
    hints = registry.get_hints_for_prompt(1, mastery_fn=mastery_fn)

    assert "STEP_1" not in hints, "STEP_1 should be suppressed due to high mastery"
    assert "STEP_2" in hints, "STEP_2 should inject due to low mastery"
    assert hints["STEP_2"] == [200]


def test_success_without_hint_clears():
    """Test that consecutive successes without hint clear the entry."""
    registry = HintRegistry(mastery_threshold=0.8, success_streak_to_clear=2)
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)

    # Success with hint doesn't count
    cleared = registry.record_success(1, "STEP_1", hint_was_used=True)
    assert not cleared
    assert (1, "STEP_1") in registry

    # First success without hint
    cleared = registry.record_success(1, "STEP_1", hint_was_used=False)
    assert not cleared
    assert (1, "STEP_1") in registry

    # Second success without hint → cleared
    cleared = registry.record_success(1, "STEP_1", hint_was_used=False)
    assert cleared
    assert (1, "STEP_1") not in registry


def test_failure_resets_streak():
    """Test that failure resets the success streak."""
    registry = HintRegistry(mastery_threshold=0.8, success_streak_to_clear=2)
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)

    # One success without hint
    registry.record_success(1, "STEP_1", hint_was_used=False)
    assert registry._hints[(1, "STEP_1")].success_count == 1

    # Failure resets
    registry.record_failure(1, "STEP_1")
    assert registry._hints[(1, "STEP_1")].success_count == 0

    # Still in registry
    assert (1, "STEP_1") in registry


def test_reflag_resets_streak():
    """Test that re-flagging resets the success streak."""
    registry = HintRegistry(mastery_threshold=0.8, success_streak_to_clear=2)
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)

    # One success without hint
    registry.record_success(1, "STEP_1", hint_was_used=False)
    assert registry._hints[(1, "STEP_1")].success_count == 1

    # Re-flag (maybe wrong again after being right)
    registry.flag_for_hint(1, "STEP_1", [200, 201], 0.2, 50)

    # Streak reset, tokens updated
    assert registry._hints[(1, "STEP_1")].success_count == 0
    assert registry._hints[(1, "STEP_1")].hint_tokens == [200, 201]


def test_get_hints_for_prompt():
    """Test getting all hints for a specific prompt."""
    registry = HintRegistry(mastery_threshold=0.8)

    # Flag multiple spans for prompt 1
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)
    registry.flag_for_hint(1, "STEP_2", [200], 0.0, 0)
    # Flag for different prompt
    registry.flag_for_hint(2, "STEP_1", [300], 0.0, 0)

    # Get hints for prompt 1 (mastery_fn returns 0.0 → always inject)
    random.seed(42)
    hints = registry.get_hints_for_prompt(1, mastery_fn=lambda span_id: 0.0)

    assert "STEP_1" in hints
    assert "STEP_2" in hints
    assert hints["STEP_1"] == [100]
    assert hints["STEP_2"] == [200]


def test_serialization_roundtrip():
    """Test checkpoint serialization and deserialization."""
    registry = HintRegistry(mastery_threshold=0.8, success_streak_to_clear=3)
    registry.flag_for_hint(1, "STEP_1", [100, 101], 0.2, 10)
    registry.flag_for_hint(2, "STEP_2", [200], 0.3, 20)

    # Record some activity
    registry.record_success(1, "STEP_1", hint_was_used=False)
    registry._hints[(1, "STEP_1")].total_uses = 5

    # Serialize
    data = registry.to_dict()

    # Deserialize
    restored = HintRegistry.from_dict(data)

    # Verify
    assert restored.mastery_threshold == 0.8
    assert restored.success_streak_to_clear == 3
    assert len(restored) == 2
    assert (1, "STEP_1") in restored
    assert (2, "STEP_2") in restored
    assert restored._hints[(1, "STEP_1")].hint_tokens == [100, 101]
    assert restored._hints[(1, "STEP_1")].success_count == 1
    assert restored._hints[(1, "STEP_1")].total_uses == 5


def test_clear_all():
    """Test clearing all hints."""
    registry = HintRegistry()
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)
    registry.flag_for_hint(2, "STEP_2", [200], 0.0, 0)

    cleared = registry.clear_all()
    assert cleared == 2
    assert len(registry) == 0


def test_get_stats():
    """Test statistics collection."""
    registry = HintRegistry()
    registry.flag_for_hint(1, "STEP_1", [100], 0.0, 0)
    registry.flag_for_hint(2, "STEP_2", [200], 0.0, 0)
    registry._hints[(1, "STEP_1")].total_uses = 3
    registry._hints[(2, "STEP_2")].total_uses = 2
    registry._hints[(1, "STEP_1")].success_count = 1

    stats = registry.get_stats()
    assert stats["hint_count"] == 2
    assert stats["total_hint_uses"] == 5
    assert stats["avg_success_streak"] == 0.5  # (1 + 0) / 2


def test_hint_not_found():
    """Test behavior when hint not found."""
    registry = HintRegistry()

    hint = registry.get_hint(1, "STEP_1", 0.0)
    assert hint is None

    cleared = registry.record_success(1, "STEP_1", hint_was_used=False)
    assert not cleared

    # record_failure on non-existent should not crash
    registry.record_failure(1, "STEP_1")


def test_hamiltonian_hint_extractor():
    """Test Hamiltonian-specific hint extractor."""
    from qgre.hints import make_hamiltonian_hint_extractor

    extractor = make_hamiltonian_hint_extractor()

    metadata = {
        "T_expr": "p**2/(2*m)",
        "V_expr": "m*g*x",
        "H_expr": "p**2/(2*m) + m*g*x",
        "ground_truth": "H = p**2/(2*m) + m*g*x; dq/dt = p/m; dp/dt = -m*g",
    }

    # STEP_3 (KINETIC) should return T_expr
    assert extractor("STEP_3", metadata) == "T = p**2/(2*m)"

    # STEP_4 (POTENTIAL) should return V_expr
    assert extractor("STEP_4", metadata) == "V = m*g*x"

    # STEP_5 (HAMILTONIAN) should return H_expr
    assert extractor("STEP_5", metadata) == "H = p**2/(2*m) + m*g*x"

    # STEP_6 (EQUATIONS) should extract dq/dt and dp/dt
    eqns = extractor("STEP_6", metadata)
    assert "dq/dt = p/m" in eqns
    assert "dp/dt = -m*g" in eqns

    # STEP_1, STEP_2 return None (coordinates/momentum don't need hints)
    assert extractor("STEP_1", metadata) is None
    assert extractor("STEP_2", metadata) is None

    # Empty metadata returns None
    assert extractor("STEP_3", {}) is None
    assert extractor("STEP_3", None) is None


def test_generic_hint_extractor():
    """Test configurable generic hint extractor."""
    from qgre.hints import make_generic_hint_extractor

    extractor = make_generic_hint_extractor(
        span_to_field={"STEP_1": "answer_1", "STEP_2": "answer_2"},
        field_format="Answer: {value}",
    )

    metadata = {"answer_1": "42", "answer_2": "hello"}

    assert extractor("STEP_1", metadata) == "Answer: 42"
    assert extractor("STEP_2", metadata) == "Answer: hello"
    assert extractor("STEP_3", metadata) is None  # Not in mapping
    assert extractor("STEP_1", {}) is None  # Missing field


if __name__ == "__main__":
    test_flag_and_get_hint()
    test_hint_probability_decay()
    test_success_without_hint_clears()
    test_failure_resets_streak()
    test_reflag_resets_streak()
    test_get_hints_for_prompt()
    test_serialization_roundtrip()
    test_clear_all()
    test_get_stats()
    test_hint_not_found()
    print("\n" + "=" * 50)
    print("All HintRegistry tests PASSED")
