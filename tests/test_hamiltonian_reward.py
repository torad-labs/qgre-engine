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
