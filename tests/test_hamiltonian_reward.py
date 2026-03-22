"""Tests for Hamiltonian reward function — catch scoring bugs before training."""

import pytest
from examples.hamiltonian.reward_fn import hamiltonian_reward


# ─── Test completions that simulate real model output ─────────────────────────

SPRING_PROMPT = (
    "A block of mass 3 kg is attached to a spring with spring constant "
    "k = 6 N/m on a frictionless surface. Let x be the displacement from "
    "equilibrium.\n\nDerive the Hamiltonian H(x, p) from first principles "
    "and find Hamilton's equations of motion."
)

SPRING_META = {
    "ground_truth": "H = p**2/6 + 3*x**2; dx/dt = p/3; dp/dt = -6*x",
    "H_expr": "p**2/6 + 3*x**2",
    "T_expr": "p**2/6",
    "V_expr": "3*x**2",
    "dqdt": "p/3",
    "dpdt": "-6*x",
    "coordinates": "x",
    "difficulty": "tier1",
    "system": "spring",
}


class TestExtractionAndScoring:
    """Test that the reward function correctly extracts and scores equations."""

    def test_perfect_completion_scores_high(self):
        """Model writes perfect answer in trained format → should score ~1.0."""
        completion = """
The generalized coordinate is x (displacement from equilibrium).
The conjugate momentum is p = m*dx/dt = 3*dx/dt.

Kinetic energy: T = p²/(2m) = p²/6
Potential energy: V = (k/2)x² = 3x²

HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6x
"""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        assert result.scores["q_format"] >= 0.7
        assert result.scores["q_has_math"] >= 0.7
        assert result.scores["q_identifies_T"] >= 0.5
        assert result.scores["q_identifies_V"] >= 0.5
        assert result.scores["q_correct_dqdt"] >= 0.9, f"dqdt should be ~1.0, got {result.scores['q_correct_dqdt']}"
        assert result.scores["q_correct_dpdt"] >= 0.9, f"dpdt should be ~1.0, got {result.scores['q_correct_dpdt']}"
        assert result.scores["q_correct_H"] >= 0.9, f"H should be ~1.0, got {result.scores['q_correct_H']}"

    def test_wrong_coefficient_scores_lower(self):
        """Model writes -3x instead of -6x → should score significantly lower than correct."""
        correct_completion = """
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6x
"""
        wrong_completion = """
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -3x
"""
        correct_result = hamiltonian_reward(SPRING_PROMPT, correct_completion, SPRING_META)
        wrong_result = hamiltonian_reward(SPRING_PROMPT, wrong_completion, SPRING_META)

        # Wrong coefficient MUST score lower than correct
        assert wrong_result.scores["q_correct_dpdt"] < correct_result.scores["q_correct_dpdt"], \
            f"Wrong coeff ({wrong_result.scores['q_correct_dpdt']}) should be < correct ({correct_result.scores['q_correct_dpdt']})"
        # And the gap should be meaningful (not 0.7 vs 0.7)
        gap = correct_result.scores["q_correct_dpdt"] - wrong_result.scores["q_correct_dpdt"]
        assert gap >= 0.15, f"Gap between correct and wrong coefficient should be >= 0.15, got {gap}"

    def test_wrong_sign_scores_very_low(self):
        """Model writes +6x instead of -6x → should score very low."""
        completion = """
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = 6x
"""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        assert result.scores["q_correct_dpdt"] < 0.5, \
            f"Wrong sign should score < 0.5, got {result.scores['q_correct_dpdt']}"

    def test_no_labels_still_extracts(self):
        """Model writes equations without HAMILTONIAN:/EQUATIONS: labels → should still extract."""
        completion = """
The kinetic energy is T = p²/6 and the potential energy is V = 3x².

So H = p²/6 + 3x²

Hamilton's equations give us:
  dx/dt = p/3
  dp/dt = -6x
"""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        # Should still find the equations via fallback regex
        assert result.scores["q_correct_dqdt"] >= 0.5, \
            f"Should extract dqdt without labels, got {result.scores['q_correct_dqdt']}"
        assert result.scores["q_correct_H"] >= 0.5, \
            f"Should extract H without labels, got {result.scores['q_correct_H']}"

    def test_latex_format_extracts(self):
        """Model writes in LaTeX-ish format → should still extract."""
        completion = """
Kinetic energy T = p^2/(2*3) = p^2/6
Potential energy V = (6/2)*x^2 = 3*x^2

HAMILTONIAN: H = p^2/6 + 3*x^2
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6*x
"""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        assert result.scores["q_correct_dqdt"] >= 0.9
        assert result.scores["q_correct_dpdt"] >= 0.9

    def test_symbolic_not_numerical_scores_lower(self):
        """Model writes p/m instead of p/3 → should score lower (didn't substitute numbers)."""
        completion = """
HAMILTONIAN: H = p²/(2m) + (k/2)x²
EQUATIONS:
  dq/dt = p/m
  dp/dt = -kx
"""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        # Should score lower than numerical answer because sympy can't match symbols
        assert result.scores["q_correct_dqdt"] < 0.9, \
            f"Symbolic (not numerical) should score < 0.9, got {result.scores['q_correct_dqdt']}"

    def test_empty_completion_scores_zero(self):
        """Empty or garbage completion → should score near 0."""
        result = hamiltonian_reward(SPRING_PROMPT, "", SPRING_META)
        assert result.reward < 0.2

        result2 = hamiltonian_reward(SPRING_PROMPT, "I don't know how to do this.", SPRING_META)
        assert result2.reward < 0.3

    def test_thinking_block_with_answer(self):
        """Model has <think> block followed by answer → should extract from answer part."""
        completion = """<think>
Let me work through this step by step. Mass is 3kg, spring constant is 6 N/m.
T = p²/(2m) = p²/6
V = (k/2)x² = 3x²
So H = T + V = p²/6 + 3x²
dq/dt = ∂H/∂p = p/3
dp/dt = -∂H/∂q = -6x
</think>

HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6x
"""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        assert result.scores["q_correct_dqdt"] >= 0.9
        assert result.scores["q_correct_dpdt"] >= 0.9
        assert result.scores["q_correct_H"] >= 0.9

    def test_thinking_only_no_answer(self):
        """Model only has <think> block, no final answer → should score low on equations."""
        completion = """<think>
Let me work through this. The mass is 3kg and spring constant is 6.
So T = p²/6 and V = 3x². The Hamiltonian would be H = p²/6 + 3x².
Then dq/dt = p/3 and dp/dt = -6x. Let me verify...
"""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        # Should still get SOME credit from regex fallback finding equations in think block
        # But less than a properly formatted answer
        assert result.scores["q_correct_dqdt"] <= 0.9  # Not perfect without labels


class TestNumericalEquivalence:
    """Test that symbolically equivalent expressions score 1.0."""

    def test_equivalent_forms_of_H(self):
        """p²/6 + 3x² == p**2/6 + 3*x**2 == (1/6)*p**2 + 3*x**2"""
        for H_form in ["p**2/6 + 3*x**2", "(1/6)*p**2 + 3*x**2", "3*x**2 + p**2/6"]:
            completion = f"""
HAMILTONIAN: H = {H_form}
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6*x
"""
            result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
            assert result.scores["q_correct_H"] >= 0.9, \
                f"H form '{H_form}' should score >= 0.9, got {result.scores['q_correct_H']}"


class TestGrading:
    """Test that grading produces meaningful gradient signal."""

    def test_score_ordering(self):
        """Better answers must score higher. This is the gradient signal."""
        completions = {
            "perfect": """
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6x
""",
            "wrong_coeff": """
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -3x
""",
            "wrong_sign": """
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = 6x
""",
            "garbage": "The Hamiltonian is a thing in physics.",
        }

        scores = {}
        for name, comp in completions.items():
            result = hamiltonian_reward(SPRING_PROMPT, comp, SPRING_META)
            scores[name] = result.reward

        assert scores["perfect"] > scores["wrong_coeff"], \
            f"perfect ({scores['perfect']}) should > wrong_coeff ({scores['wrong_coeff']})"
        assert scores["wrong_coeff"] > scores["wrong_sign"], \
            f"wrong_coeff ({scores['wrong_coeff']}) should > wrong_sign ({scores['wrong_sign']})"
        assert scores["wrong_sign"] > scores["garbage"], \
            f"wrong_sign ({scores['wrong_sign']}) should > garbage ({scores['garbage']})"
