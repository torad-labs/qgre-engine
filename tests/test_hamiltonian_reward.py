"""Tests for Hamiltonian reward function — catch scoring bugs before training."""

import pytest
from examples.hamiltonian.reward_fn import hamiltonian_reward


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

PERFECT_STRUCTURED = """
COORDINATES: q = x
MOMENTUM: p = m*dx/dt = 3*dx/dt
KINETIC: T = p²/(2*3) = p²/6
POTENTIAL: V = (6/2)*x² = 3x²
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6x
"""

VELOCITY_FORM = """
COORDINATES: q = x
MOMENTUM: p = 3*dx/dt
KINETIC: T = (1/2)*3*(dx/dt)² = (3/2)*(dx/dt)²
POTENTIAL: V = 3x²
HAMILTONIAN: H = (3/2)*(dx/dt)² + 3x²
EQUATIONS:
  dq/dt = dx/dt
  dp/dt = -6x
"""

WRONG_COEFFICIENT = """
COORDINATES: q = x
MOMENTUM: p = 3*dx/dt
KINETIC: T = p²/6
POTENTIAL: V = 3x²
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -3x
"""

NO_LABELS = """
The kinetic energy is T = p²/6 and the potential energy is V = 3x².
So H = p²/6 + 3x²
Hamilton's equations give us:
  dx/dt = p/3
  dp/dt = -6x
"""


PENDULUM_PROMPT = (
    "A simple pendulum of mass 2 kg and length 0.5 m swings under gravity. "
    "Let θ be the angle from vertical.\n\nDerive the Hamiltonian H(θ, p) from "
    "first principles and find Hamilton's equations of motion."
)

# Physics verification:
#   m=2, L=0.5, g=9.8
#   T = (1/2)*m*L²*θ̇² = (1/2)*2*(0.25)*θ̇² = 0.25*θ̇²
#   p_theta = m*L²*θ̇ = 0.5*θ̇  →  θ̇ = 2*p_theta
#   T = 0.25*(2*p_theta)² = p_theta²
#   V = -m*g*L*cos(θ) = -2*9.8*0.5*cos(θ) = -9.8*cos(θ) = -49*cos(θ)/5
#   H = p_theta² - 49*cos(theta)/5
#   dθ/dt = ∂H/∂p_theta = 2*p_theta
#   dp_theta/dt = -∂H/∂θ = -49*sin(theta)/5
PENDULUM_META = {
    "ground_truth": "H = p_theta**2 - 49*cos(theta)/5; dtheta/dt = 2*p_theta; dp/dt = -49*sin(theta)/5",
    "H_expr": "p_theta**2 - 49*cos(theta)/5",
    "T_expr": "p_theta**2",
    "V_expr": "-49*cos(theta)/5",
    "dqdt": "2*p_theta",
    "dpdt": "-49*sin(theta)/5",
    "coordinates": "theta",
    "difficulty": "tier2",
    "system": "pendulum",
}

# Explicit Lagrangian notation: ∂L/∂θ̇ is stated. T is in p form. Correct answer.
PENDULUM_EXPLICIT_MOMENTUM = """
COORDINATES: q = θ
MOMENTUM: p = ∂L/∂θ̇ = mL²θ̇ = 2(0.25)θ̇ = 0.5θ̇
KINETIC: T = p²/(2mL²) = p²/(2·0.5) = p²
POTENTIAL: V = -mgLcos(θ) = -9.8cos(θ)
HAMILTONIAN: H = p² - 9.8cos(θ)
EQUATIONS:
  dθ/dt = 2p
  dp/dt = -9.8sin(θ)
"""

# Implicit definition: states p = 0.5θ̇ without deriving via ∂L/∂θ̇. T in p form.
PENDULUM_IMPLICIT_MOMENTUM = """
COORDINATES: q = θ
MOMENTUM: p = 0.5θ̇
KINETIC: T = p²
POTENTIAL: V = -9.8cos(θ)
HAMILTONIAN: H = p² - 9.8cos(θ)
EQUATIONS:
  dθ/dt = 2p
  dp/dt = -9.8sin(θ)
"""

# Failure mode: T and H written in velocity form (θ̇) rather than momentum form (p).
PENDULUM_VELOCITY_FORM = """
COORDINATES: q = θ
MOMENTUM: p = 0.5θ̇
KINETIC: T = 0.25θ̇²
POTENTIAL: V = -9.8cos(θ)
HAMILTONIAN: H = 0.25θ̇² - 9.8cos(θ)
EQUATIONS:
  dθ/dt = θ̇
  dp/dt = -9.8sin(θ)
"""

# Failure mode: wrong coefficient on dp/dt.
# Correct: dp/dt = -9.8sin(θ). Model writes -4.9sin(θ) (off by factor of 2).
PENDULUM_WRONG_DERIVATIVE = """
COORDINATES: q = θ
MOMENTUM: p = ∂L/∂θ̇ = 0.5θ̇
KINETIC: T = p²
POTENTIAL: V = -9.8cos(θ)
HAMILTONIAN: H = p² - 9.8cos(θ)
EQUATIONS:
  dθ/dt = 2p
  dp/dt = -4.9sin(θ)
"""


class TestStructuredFormat:
    """Test that labeled sections are correctly extracted and scored."""

    def test_perfect_structured_scores_high(self):
        result = hamiltonian_reward(SPRING_PROMPT, PERFECT_STRUCTURED, SPRING_META)
        assert result.scores["q_format"] >= 0.9
        assert result.scores["q_momentum_defined"] >= 0.7
        assert result.scores["q_T_uses_p"] >= 0.7
        assert result.scores["q_V_correct"] >= 0.9
        assert result.scores["q_correct_dqdt"] >= 0.9
        assert result.scores["q_correct_dpdt"] >= 0.9
        assert result.scores["q_correct_H"] >= 0.9
        assert result.reward >= 0.8

    def test_velocity_form_T_scores_low(self):
        """T in velocity form (dx/dt)² instead of momentum form (p²) → q_T_uses_p should be low."""
        result = hamiltonian_reward(SPRING_PROMPT, VELOCITY_FORM, SPRING_META)
        assert result.scores["q_T_uses_p"] <= 0.5, \
            f"Velocity form T should score <= 0.5, got {result.scores['q_T_uses_p']}"

    def test_wrong_coefficient_gradient(self):
        """dp/dt = -3x instead of -6x → should score lower than correct."""
        correct = hamiltonian_reward(SPRING_PROMPT, PERFECT_STRUCTURED, SPRING_META)
        wrong = hamiltonian_reward(SPRING_PROMPT, WRONG_COEFFICIENT, SPRING_META)
        gap = correct.scores["q_correct_dpdt"] - wrong.scores["q_correct_dpdt"]
        assert gap >= 0.15, f"Gap should be >= 0.15, got {gap}"

    def test_no_labels_still_works(self):
        """Without structured labels, fallback extraction should still work."""
        result = hamiltonian_reward(SPRING_PROMPT, NO_LABELS, SPRING_META)
        assert result.scores["q_correct_dqdt"] >= 0.5
        assert result.scores["q_correct_H"] >= 0.5

    def test_format_rewards_labels(self):
        """q_format should reward structured labels over free-form."""
        structured = hamiltonian_reward(SPRING_PROMPT, PERFECT_STRUCTURED, SPRING_META)
        freeform = hamiltonian_reward(SPRING_PROMPT, NO_LABELS, SPRING_META)
        assert structured.scores["q_format"] > freeform.scores["q_format"]


class TestMomentumForm:
    """Test that momentum-form qualities distinguish p from q̇."""

    def test_momentum_with_numbers(self):
        text = "MOMENTUM: p = 3*dx/dt"
        result = hamiltonian_reward(SPRING_PROMPT, f"COORDINATES: q = x\n{text}\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x", SPRING_META)
        assert result.scores["q_momentum_defined"] >= 0.7

    def test_no_momentum_section(self):
        text = "H = p²/6 + 3x²"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_momentum_defined"] <= 0.5


class TestNumericalEquivalence:
    """Test that symbolically equivalent expressions score 1.0."""

    def test_equivalent_H_forms(self):
        for H_form in ["p**2/6 + 3*x**2", "(1/6)*p**2 + 3*x**2", "3*x**2 + p**2/6"]:
            text = f"COORDINATES: q = x\nMOMENTUM: p = 3v\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = {H_form}\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6*x"
            result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
            assert result.scores["q_correct_H"] >= 0.9, \
                f"H form '{H_form}' should score >= 0.9, got {result.scores['q_correct_H']}"


class TestScoreOrdering:
    """Test that better answers score higher — this IS the gradient signal."""

    def test_overall_ordering(self):
        scores = {}
        completions = {
            "perfect": PERFECT_STRUCTURED,
            "wrong_coeff": WRONG_COEFFICIENT,
            "velocity_form": VELOCITY_FORM,
            "garbage": "The Hamiltonian is a thing in physics.",
        }
        for name, comp in completions.items():
            result = hamiltonian_reward(SPRING_PROMPT, comp, SPRING_META)
            scores[name] = result.reward

        assert scores["perfect"] > scores["wrong_coeff"], \
            f"perfect ({scores['perfect']:.2f}) > wrong_coeff ({scores['wrong_coeff']:.2f})"
        assert scores["wrong_coeff"] > scores["velocity_form"], \
            f"wrong_coeff ({scores['wrong_coeff']:.2f}) > velocity_form ({scores['velocity_form']:.2f})"
        assert scores["velocity_form"] > scores["garbage"], \
            f"velocity_form ({scores['velocity_form']:.2f}) > garbage ({scores['garbage']:.2f})"

    def test_empty_scores_zero(self):
        result = hamiltonian_reward(SPRING_PROMPT, "", SPRING_META)
        assert result.reward < 0.1

    def test_garbage_scores_low(self):
        result = hamiltonian_reward(SPRING_PROMPT, "I don't know.", SPRING_META)
        assert result.reward < 0.2


class TestEdgeCases:
    """Test that reward function never crashes on real model output patterns."""

    def test_numeric_H_scores_low(self):
        """Model plugs in numbers instead of keeping symbolic → distinct low score."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3v\nKINETIC: T = 4\nPOTENTIAL: V = 12\nHAMILTONIAN: H = 16\nEQUATIONS:\n  dq/dt = 0\n  dp/dt = -19.6"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_correct_H"] <= 0.2

    def test_velocity_form_H_partial_credit(self):
        """H in velocity form (ẋ) gets more than unparseable (0.2) but less than correct."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3*dx/dt\nKINETIC: T = 3/2 * ẋ²\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = 3/2 * ẋ² + 3x²\nEQUATIONS:\n  dq/dt = ẋ\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_correct_H"] > 0.2, "Velocity-form H should score above 0.2"
        assert result.scores["q_correct_H"] < 0.8, "Velocity-form H should score below correct"

    def test_latex_wrapped_H(self):
        """Model wraps answer in $$ delimiters — should still parse."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3v\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: $$ H = p^2/6 + 3x^2 $$\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_correct_H"] >= 0.9

    def test_latex_dot_notation(self):
        """Model uses \\dot{x} LaTeX — should not crash."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3*\\dot{x}\nKINETIC: T = \\frac{p^2}{6}\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = \\frac{\\dot{x}^2}{2} + 3x²\nEQUATIONS:\n  dq/dt = \\dot{x}\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        # Should not crash, and velocity-form H should get partial credit
        assert result.scores["q_correct_H"] > 0.1

    def test_comma_separated_expressions(self):
        """Sympy can return a list for comma-separated input — must not crash."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3v\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x², where x is displacement\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        # Must not crash — score doesn't matter as long as no exception
        assert isinstance(result.reward, float)

    def test_multiline_derivation_chain(self):
        """Model writes T = p²/(2m) = (2ẋ)²/4 = ẋ² — final form is velocity, no credit."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3*dx/dt\nKINETIC: T = p²/(2*3) = (3*ẋ)²/6 = 3ẋ²/2\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        # T starts with p but final form is velocity — same score as pure velocity.
        # Only the final form matters. No credit for "knowing p" if you undo it.
        assert result.scores["q_T_uses_p"] <= 0.2


class TestGranularMomentumQualities:
    """Tests for q_defines_momentum and q_T_in_momentum and q_H_in_momentum.

    These qualities do not exist in the reward function yet — tests are written
    RED first per C07 (test-first). They will pass once the scorer is implemented.
    """

    # ── q_defines_momentum ────────────────────────────────────────────────────

    def test_explicit_lagrangian_notation_scores_high(self):
        """∂L/∂θ̇ notation with numeric derivation → q_defines_momentum >= 0.8."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        score = result.scores["q_defines_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score >= 0.8, (
            f"Explicit Lagrangian notation should score >= 0.8, got {score}"
        )

    def test_implicit_definition_partial_credit(self):
        """p = 0.5θ̇ with no ∂L/∂θ̇ → q_defines_momentum partial credit [0.3, 0.7]."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_IMPLICIT_MOMENTUM, PENDULUM_META)
        score = result.scores["q_defines_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert 0.3 <= score <= 0.7, (
            f"Implicit definition should score between 0.3 and 0.7, got {score}"
        )

    def test_no_momentum_scores_zero(self):
        """Completion with no p definition → q_defines_momentum <= 0.1."""
        no_momentum = "COORDINATES: q = θ\nKINETIC: T = 0.25θ̇²\nPOTENTIAL: V = -9.8cos(θ)\nHAMILTONIAN: H = 0.25θ̇² - 9.8cos(θ)\nEQUATIONS:\n  dθ/dt = θ̇\n  dp/dt = -9.8sin(θ)"
        result = hamiltonian_reward(PENDULUM_PROMPT, no_momentum, PENDULUM_META)
        score = result.scores["q_defines_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score <= 0.1, (
            f"No momentum definition should score <= 0.1, got {score}"
        )

    # ── q_T_in_momentum ───────────────────────────────────────────────────────

    def test_T_momentum_form_scores_high(self):
        """T = p² (momentum form) → q_T_in_momentum >= 0.8."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        score = result.scores["q_T_in_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score >= 0.8, (
            f"T in momentum form should score >= 0.8, got {score}"
        )

    def test_T_velocity_form_scores_low(self):
        """T = 0.25θ̇² (velocity form) → q_T_in_momentum <= 0.3."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_VELOCITY_FORM, PENDULUM_META)
        score = result.scores["q_T_in_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score <= 0.3, (
            f"T in velocity form should score <= 0.3, got {score}"
        )

    def test_T_mixed_partial(self):
        """T contains both p and θ̇ terms → q_T_in_momentum is partial (0.3, 0.8)."""
        mixed_T = (
            "COORDINATES: q = θ\nMOMENTUM: p = 0.5θ̇\n"
            "KINETIC: T = p²/2 + 0.1θ̇²\n"
            "POTENTIAL: V = -9.8cos(θ)\n"
            "HAMILTONIAN: H = p²/2 + 0.1θ̇² - 9.8cos(θ)\n"
            "EQUATIONS:\n  dθ/dt = 2p\n  dp/dt = -9.8sin(θ)"
        )
        result = hamiltonian_reward(PENDULUM_PROMPT, mixed_T, PENDULUM_META)
        score = result.scores["q_T_in_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert 0.3 < score < 0.8, (
            f"Mixed T (p² and θ̇²) should score between 0.3 and 0.8, got {score}"
        )

    # ── q_H_in_momentum ───────────────────────────────────────────────────────

    def test_H_momentum_form_scores_high(self):
        """H = p² - 9.8cos(θ) (momentum form) → q_H_in_momentum >= 0.8."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        score = result.scores["q_H_in_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score >= 0.8, (
            f"H in momentum form should score >= 0.8, got {score}"
        )

    def test_H_velocity_form_scores_low(self):
        """H = 0.25θ̇² - 9.8cos(θ) (velocity form) → q_H_in_momentum <= 0.3."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_VELOCITY_FORM, PENDULUM_META)
        score = result.scores["q_H_in_momentum"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score <= 0.3, (
            f"H in velocity form should score <= 0.3, got {score}"
        )


class TestGranularCoefficientQualities:
    """Tests for q_correct_coefficient and q_derivative_correct.

    These qualities do not exist yet — tests are written RED first.
    """

    # ── q_correct_coefficient ─────────────────────────────────────────────────

    def test_correct_coefficients_score_high(self):
        """All coefficients correct (p², -9.8cos(θ), 2p, -9.8sin(θ)) → q_correct_coefficient >= 0.7."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        score = result.scores["q_correct_coefficient"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score >= 0.7, (
            f"Correct coefficients should score >= 0.7, got {score}"
        )

    def test_wrong_coefficient_scores_lower(self):
        """dp/dt = -4.9sin(θ) instead of -9.8sin(θ) → q_correct_coefficient < correct.
        H and dq/dt are still correct (2/3 components), so score is partial, not zero."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_WRONG_DERIVATIVE, PENDULUM_META)
        score = result.scores["q_correct_coefficient"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score < 1.0, (
            f"Wrong dp/dt coefficient should score < 1.0, got {score}"
        )

    def test_coefficient_gradient(self):
        """Correct coefficients must score at least 0.2 higher than wrong coefficients."""
        correct = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        wrong = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_WRONG_DERIVATIVE, PENDULUM_META)
        gap = correct.scores["q_correct_coefficient"] - wrong.scores["q_correct_coefficient"]
        assert gap >= 0.2, (
            f"Correct ({correct.scores['q_correct_coefficient']:.2f}) must beat "
            f"wrong ({wrong.scores['q_correct_coefficient']:.2f}) by >= 0.2, gap={gap:.2f}"
        )

    # ── q_derivative_correct ──────────────────────────────────────────────────

    def test_correct_derivative_scores_high(self):
        """Both dθ/dt = 2p and dp/dt = -9.8sin(θ) correct → q_derivative_correct >= 0.8."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        score = result.scores["q_derivative_correct"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        assert score >= 0.8, (
            f"Correct derivatives should score >= 0.8, got {score}"
        )

    def test_factor_of_two_error_distinct_signal(self):
        """dp/dt off by factor of 2 → q_derivative_correct gives a distinct mid-range signal."""
        result = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_WRONG_DERIVATIVE, PENDULUM_META)
        score = result.scores["q_derivative_correct"]
        assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        # Half the derivatives are right (dθ/dt is correct), half wrong (dp/dt off by 2)
        # Score should be in the middle range — not 0, not 1
        assert 0.3 <= score <= 0.7, (
            f"Factor-of-2 error should produce mid-range signal [0.3, 0.7], got {score}"
        )

    def test_derivative_gradient(self):
        """Correct derivatives must score higher than wrong derivatives."""
        correct = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        wrong = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_WRONG_DERIVATIVE, PENDULUM_META)
        assert correct.scores["q_derivative_correct"] > wrong.scores["q_derivative_correct"], (
            f"Correct ({correct.scores['q_derivative_correct']:.2f}) should beat "
            f"wrong ({wrong.scores['q_derivative_correct']:.2f})"
        )


class TestGranularOrdering:
    """End-to-end ordering tests: better physics → higher reward.

    These tests verify the gradient signal that GRPO/SPO relies on to improve.
    A reward function with no ordering signal cannot train anything.
    """

    def test_momentum_form_beats_velocity_form(self):
        """Momentum form (p²) must score higher overall than velocity form (θ̇²)."""
        momentum = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        velocity = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_VELOCITY_FORM, PENDULUM_META)
        assert momentum.reward > velocity.reward, (
            f"Momentum form ({momentum.reward:.3f}) should beat velocity form ({velocity.reward:.3f})"
        )

    def test_correct_derivative_beats_wrong(self):
        """Correct dp/dt must score higher overall than off-by-factor-of-2 dp/dt."""
        correct = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_EXPLICIT_MOMENTUM, PENDULUM_META)
        wrong = hamiltonian_reward(PENDULUM_PROMPT, PENDULUM_WRONG_DERIVATIVE, PENDULUM_META)
        assert correct.reward > wrong.reward, (
            f"Correct derivatives ({correct.reward:.3f}) should beat wrong ({wrong.reward:.3f})"
        )


class TestNeverCrash:
    """Fuzz test: reward function must NEVER crash, no matter what the model outputs.

    Every test here asserts isinstance(result.reward, float) — the score doesn't matter,
    only that the function returns without exception. These patterns come from real model
    outputs that caused crashes during training.
    """

    GARBAGE_COMPLETIONS = [
        # Comma-separated expressions (sympify returns list)
        "HAMILTONIAN: H = p²/6 + 3x², where x is displacement",
        "KINETIC: T = p²/6, V = 3x²",
        "HAMILTONIAN: H = a, b, c",
        # Pure numbers
        "HAMILTONIAN: H = 16",
        "KINETIC: T = 4\nPOTENTIAL: V = 12\nHAMILTONIAN: H = 16",
        # LaTeX heavy
        "$$ H = \\frac{p^2}{6} + 3x^2 $$",
        "HAMILTONIAN: $$ H = \\frac{\\dot{x}^2}{2} + 3x^2 $$",
        "KINETIC: $T = \\frac{p^2}{2m}$\nHAMILTONIAN: $H = \\frac{p^2}{2m} + V$",
        # Unicode mess
        "HAMILTONIAN: H = ẋ² + θ̇² + ṙ²",
        "KINETIC: T = ½mv²\nHAMILTONIAN: H = ½mv² + mgh",
        # Empty / whitespace
        "",
        "   ",
        "\n\n\n",
        # No structure at all
        "I don't know how to solve this problem.",
        "The answer is 42.",
        # Malformed labels
        "HAMILTONIAN:",
        "HAMILTONIAN: H =",
        "HAMILTONIAN: H = = = ",
        "KINETIC: \nPOTENTIAL: \nHAMILTONIAN: ",
        # Nested parentheses / brackets
        "HAMILTONIAN: H = ((((p²/6)))) + [3x²]",
        "KINETIC: T = {p²}/{6}",
        # Very long expression
        "HAMILTONIAN: H = " + " + ".join([f"{i}*x**{i}" for i in range(50)]),
        # Model talks instead of answering
        "Let me think about this step by step. First, we need to consider the Lagrangian...",
        # Mixed correct and garbage
        "COORDINATES: q = x\nMOMENTUM: p = ???\nKINETIC: T = idk\nPOTENTIAL: V = maybe 3x²\nHAMILTONIAN: H = who knows\nEQUATIONS:\n  dq/dt = ¯\\_(ツ)_/¯\n  dp/dt = lol",
        # Repeated labels
        "HAMILTONIAN: H = p²/6\nHAMILTONIAN: H = 3x²\nHAMILTONIAN: H = p²/6 + 3x²",
        # Code blocks
        "```python\nH = p**2/6 + 3*x**2\n```",
        "```\nCOORDINATES: q = x\nHAMILTONIAN: H = p²/6 + 3x²\n```",
        # Pendulum-specific garbage patterns (from Immune Memory: Qwen3-1.7B output formats)
        "MOMENTUM: p = ∂L/∂???",
        "KINETIC: T = p^2/(2*0)",
        "HAMILTONIAN: H = θ̇² + ṙ² / idk",
        "∂(ax²)/∂x = ???",
        "MOMENTUM: p = ∂L/∂θ̇ = undefined",
    ]

    @pytest.mark.parametrize("completion", GARBAGE_COMPLETIONS)
    def test_never_crash(self, completion):
        """Reward function must return a float for ANY input, never raise."""
        result = hamiltonian_reward(SPRING_PROMPT, completion, SPRING_META)
        assert isinstance(result.reward, float), f"Expected float, got {type(result.reward)}"
        assert 0.0 <= result.reward <= 1.0, f"Reward {result.reward} out of [0, 1] range"


class TestRealModelOutputFormats:
    """Tests built from REAL model completions that caused extraction failures.

    Every test case here is a format the model actually produced during training.
    If the model invents a new format, add it here.
    """

    def test_bullet_latex_equations(self):
        """Model writes equations as markdown bullets with LaTeX fractions."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3*dx/dt\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEQUATIONS:\n- $ \\frac{dq}{dt} = \\frac{p}{3} $\n- $ \\frac{dp}{dt} = -6x $"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_correct_dqdt"] >= 0.7, f"Bullet LaTeX dqdt should extract, got {result.scores['q_correct_dqdt']}"
        assert result.scores["q_correct_dpdt"] >= 0.7, f"Bullet LaTeX dpdt should extract, got {result.scores['q_correct_dpdt']}"

    def test_indented_latex_equations(self):
        """Model writes equations as indented LaTeX."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3*dx/dt\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEQUATIONS:\n  $ \\frac{dq}{dt} = \\frac{p}{3} $\n  $ \\frac{dp}{dt} = -6x $"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_correct_dqdt"] >= 0.7, f"Indented LaTeX dqdt should extract, got {result.scores['q_correct_dqdt']}"
        assert result.scores["q_correct_dpdt"] >= 0.7, f"Indented LaTeX dpdt should extract, got {result.scores['q_correct_dpdt']}"

    def test_double_dollar_latex_equations(self):
        """Model writes equations in $$ display math blocks."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3*dx/dt\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEquations:\n$$\n\\frac{dq}{dt} = \\frac{p}{3}\n$$\n$$\n\\frac{dp}{dt} = -6x\n$$"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_correct_dqdt"] >= 0.5, f"Display math dqdt should extract, got {result.scores['q_correct_dqdt']}"
        assert result.scores["q_correct_dpdt"] >= 0.5, f"Display math dpdt should extract, got {result.scores['q_correct_dpdt']}"

    def test_bold_markdown_labels(self):
        """Model wraps labels in ** bold markers."""
        text = "**COORDINATES:** q = x\n**MOMENTUM:** p = 3*dx/dt\n**KINETIC:** T = p²/6\n**POTENTIAL:** V = 3x²\n**HAMILTONIAN:** H = p²/6 + 3x²\n**EQUATIONS:**\n  dq/dt = p/3\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_T_uses_p"] >= 0.7, f"Bold labels T should extract, got {result.scores['q_T_uses_p']}"
        assert result.scores["q_correct_H"] >= 0.7, f"Bold labels H should extract, got {result.scores['q_correct_H']}"

    def test_hash_header_labels(self):
        """Model uses ### headers instead of plain labels."""
        text = "### COORDINATES: q = x\n### MOMENTUM: p = 3*dx/dt\n### KINETIC: T = p²/6\n### POTENTIAL: V = 3x²\n### HAMILTONIAN: H = p²/6 + 3x²\n### EQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_T_uses_p"] >= 0.7, f"Hash labels T should extract, got {result.scores['q_T_uses_p']}"
        assert result.scores["q_correct_H"] >= 0.7, f"Hash labels H should extract, got {result.scores['q_correct_H']}"

    def test_derivation_then_labels(self):
        """Model writes long derivation first, then structured labels at the end.
        Extractors must take LAST match, not first."""
        text = """### 3. **Kinetic Energy**
The kinetic energy is T = (1/2)*m*v² = (3/2)*v²

### 5. **Hamiltonian**
H = T + V = (3/2)*v² + 3x²

---

### Final Output:

COORDINATES: q = x
MOMENTUM: p = 3*dx/dt
KINETIC: T = p²/6
POTENTIAL: V = 3x²
HAMILTONIAN: H = p²/6 + 3x²
EQUATIONS:
  dq/dt = p/3
  dp/dt = -6x"""
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        # Must extract from final labels, not derivation headers
        assert result.scores["q_T_uses_p"] >= 0.7, f"Should extract last KINETIC (p form), got {result.scores['q_T_uses_p']}"
        assert result.scores["q_correct_H"] >= 0.7, f"Should extract last HAMILTONIAN, got {result.scores['q_correct_H']}"

    def test_incline_coordinate_s(self):
        """Model uses s instead of x for incline problems — extractors must handle."""
        text = "COORDINATES: q = s\nMOMENTUM: p = 2*ds/dt\nKINETIC: T = p²/4\nPOTENTIAL: V = -9.8*s\nHAMILTONIAN: H = p²/4 - 9.8*s\nEQUATIONS:\n  ds/dt = p/2\n  dp/dt = 9.8"
        meta = {**SPRING_META, "H_expr": "p**2/4 - 49*s/5", "dqdt": "p/2", "dpdt": "49/5",
                "T_expr": "p**2/4", "V_expr": "-49*s/5"}
        result = hamiltonian_reward(SPRING_PROMPT, text, meta)
        assert result.scores["q_correct_dqdt"] >= 0.7, f"s-coordinate dqdt should extract, got {result.scores['q_correct_dqdt']}"

    def test_latex_frac_in_kinetic(self):
        """Model writes KINETIC with LaTeX \\frac{}{}."""
        text = "COORDINATES: q = x\nMOMENTUM: p = 3*dx/dt\nKINETIC: $ T = \\frac{p^2}{6} $\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x"
        result = hamiltonian_reward(SPRING_PROMPT, text, SPRING_META)
        assert result.scores["q_T_uses_p"] >= 0.7, f"LaTeX frac T should parse, got {result.scores['q_T_uses_p']}"
        assert result.scores["q_correct_H"] >= 0.7, f"H should still work, got {result.scores['q_correct_H']}"
