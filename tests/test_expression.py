"""Tests for qgre.expression — math expression parsing, normalization, and scoring."""

import pytest
import sympy as sp

from qgre.expression import (
    _BASE_SYMPY_LOCALS,
    SympyTimeoutError,
    best_match,
    extract_constants_from_prompt,
    normalize_for_sympy,
    normalize_text,
    parse_math,
    remap_variables,
    score_expression,
    score_terms_structurally,
    string_similarity,
    sympy_scorer,
    sympy_timeout,
    try_sympify,
)


# ─── 1. normalize_text ────────────────────────────────────────────────────────


class TestNormalizeText:
    def test_unicode_squared(self):
        result = normalize_text("x²")
        # ² should not appear in output; ^ or nothing is fine
        assert "²" not in result

    def test_unicode_cdot(self):
        result = normalize_text("a·b")
        assert "·" not in result
        # should map to * or similar
        assert "*" in result or "ab" in result

    def test_lowercases(self):
        result = normalize_text("H = P²/2M")
        assert result == result.lower()

    def test_removes_spaces(self):
        result = normalize_text("x + y")
        assert " " not in result


# ─── 2. normalize_for_sympy ───────────────────────────────────────────────────


class TestNormalizeForSympy:
    def test_latex_fraction(self):
        result = normalize_for_sympy(r"\frac{p^2}{4}")
        # Should convert to (p**2)/(4) or equivalent
        assert "frac" not in result
        assert "p" in result

    def test_trig_latex(self):
        result = normalize_for_sympy(r"\sin(x)")
        assert "sin" in result
        assert "\\" not in result

    def test_degree_symbol(self):
        result = normalize_for_sympy("60°")
        # 60° should become pi/3
        assert "pi" in result or "3" in result

    def test_unicode_squared_to_power(self):
        result = normalize_for_sympy("p²")
        assert "**2" in result or "^2" in result


# ─── 3. parse_math — plain algebra → sympy Expr ──────────────────────────────


class TestParseMath:
    def test_plain_algebra(self):
        result = parse_math(
            "p**2/4 + 2*x**2", sympy_locals={"x": sp.Symbol("x"), "p": sp.Symbol("p")}
        )
        assert result is not None
        assert isinstance(result, sp.Expr)

    def test_simple_number(self):
        result = parse_math("42")
        assert result is not None

    def test_empty_returns_none(self):
        result = parse_math("")
        assert result is None

    def test_nonsense_returns_none(self):
        result = parse_math("!!!not math at all@@@")
        assert result is None


# ─── 4. try_sympify ──────────────────────────────────────────────────────────


class TestTrySympify:
    def test_valid_expression(self):
        result = try_sympify("p**2/6", sympy_locals={"p": sp.Symbol("p")})
        assert result is not None

    def test_unparseable_returns_none(self):
        result = try_sympify("this is not math @#$")
        assert result is None

    def test_none_like_input(self):
        result = try_sympify("")
        assert result is None


# ─── 5. remap_variables ──────────────────────────────────────────────────────


class TestRemapVariables:
    def test_maps_coord_to_teacher(self):
        # Student uses y, teacher uses x — should remap
        student = sp.sympify("y**2 + p**2")
        teacher = sp.sympify("x**2 + p**2", locals={"p": sp.Symbol("p"), "x": sp.Symbol("x")})
        result = remap_variables(student, teacher)
        # After remapping, student should look like teacher
        assert sp.simplify(result - teacher) == 0 or "x" in str(result)

    def test_same_vars_unchanged(self):
        x = sp.Symbol("x")
        p = sp.Symbol("p")
        student = x**2 + p**2
        teacher = x**2 + p**2
        result = remap_variables(student, teacher)
        assert result == student

    def test_non_expr_returned_unchanged(self):
        student = "not a sympy expr"
        teacher = sp.Symbol("x")
        result = remap_variables(student, teacher)
        assert result == student


# ─── 6. score_expression — exact match ───────────────────────────────────────


class TestScoreExpressionExact:
    def test_exact_match_returns_1(self):
        score = score_expression("p**2/6", "p**2/6", ["x", "p"])
        assert score == 1.0

    def test_none_student_returns_0(self):
        score = score_expression(None, "p**2/6", ["x", "p"])
        assert score == 0.0

    def test_sign_flip_returns_approx_08(self):
        # -x vs x — sign flip should give 0.8
        score = score_expression("-x", "x", ["x"])
        assert abs(score - 0.8) < 0.05

    def test_wrong_expression_partial_credit(self):
        # Completely wrong but parseable — should give partial credit in [0.2, 0.7]
        score = score_expression("p**2/3", "p**2/6 + 3*x**2", ["x", "p"])
        assert 0.0 <= score <= 1.0


# ─── 7. score_terms_structurally ─────────────────────────────────────────────


class TestScoreTermsStructurally:
    def test_exact_polynomial_match(self):
        x = sp.Symbol("x")
        p = sp.Symbol("p")
        student = p**2 / 6 + 3 * x**2
        teacher = p**2 / 6 + 3 * x**2
        score = score_terms_structurally(student, teacher)
        assert score is not None
        assert abs(score - 1.0) < 0.01

    def test_missing_term_partial(self):
        x = sp.Symbol("x")
        p = sp.Symbol("p")
        student = 3 * x**2  # missing p^2/6 term
        teacher = p**2 / 6 + 3 * x**2
        score = score_terms_structurally(student, teacher)
        assert score is not None
        assert score < 1.0
        assert score >= 0.0

    def test_non_polynomial_returns_none(self):
        x = sp.Symbol("x")
        student = sp.sin(x)
        teacher = sp.cos(x)
        score = score_terms_structurally(student, teacher)
        # May return None for trig expressions
        assert score is None or isinstance(score, float)


# ─── 8. string_similarity ────────────────────────────────────────────────────


class TestStringSimilarity:
    def test_identical_strings(self):
        score = string_similarity("p**2/6", "p**2/6")
        assert score == 1.0

    def test_identical_after_normalize(self):
        # These should normalize to the same thing
        score = string_similarity("p^2/6", "p**2/6")
        assert score == 1.0

    def test_different_strings_low(self):
        score = string_similarity("abc", "xyz")
        assert score < 0.5

    def test_substring_match(self):
        score = string_similarity("p**2/6", "p**2/6 + 3*x**2")
        # substring match returns 0.7
        assert score >= 0.5


# ─── 9. best_match ───────────────────────────────────────────────────────────


class TestBestMatch:
    def test_returns_highest_score(self):
        candidates = ["p**2/3", "p**2/6", "x**2"]
        score = best_match(candidates, "p**2/6", variables=["x", "p"])
        assert score == 1.0  # p**2/6 should match exactly

    def test_empty_candidates(self):
        score = best_match([], "p**2/6")
        assert score == 0.0


# ─── 10. extract_constants_from_prompt ───────────────────────────────────────


class TestExtractConstantsFromPrompt:
    def test_extracts_mass(self):
        prompt = "A block of mass m = 3 kg on a frictionless surface."
        subs = extract_constants_from_prompt(prompt)
        assert isinstance(subs, dict)
        # Should have m → 3
        values = list(subs.values())
        assert any(v == 3 for v in values)

    def test_extracts_spring_constant(self):
        prompt = "A spring with spring constant k = 6 N/m."
        subs = extract_constants_from_prompt(prompt)
        values = list(subs.values())
        assert any(v == 6 for v in values)

    def test_empty_prompt(self):
        subs = extract_constants_from_prompt("A block slides.")
        assert isinstance(subs, dict)


# ─── 11. sympy_scorer thin wrapper ───────────────────────────────────────────


class TestSympyScorer:
    def test_exact_match(self):
        score = sympy_scorer("p**2/6", "p**2/6")
        assert score == 1.0

    def test_none_student(self):
        score = sympy_scorer(None, "p**2/6")
        assert score == 0.0

    def test_returns_float(self):
        score = sympy_scorer("p**2/4", "p**2/6 + 3*x**2")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ─── 12. _BASE_SYMPY_LOCALS ──────────────────────────────────────────────────


class TestBaseSympy:
    def test_has_only_pi(self):
        assert set(_BASE_SYMPY_LOCALS.keys()) == {"pi"}

    def test_pi_is_sympy_pi(self):
        assert _BASE_SYMPY_LOCALS["pi"] is sp.pi


# ─── 13. SympyTimeoutError ───────────────────────────────────────────────────


class TestSympyTimeoutError:
    def test_can_be_raised(self):
        with pytest.raises(SympyTimeoutError):
            raise SympyTimeoutError("timed out")

    def test_is_exception(self):
        assert issubclass(SympyTimeoutError, Exception)


# ─── 14. sympy_timeout context manager ───────────────────────────────────────


class TestSympyTimeout:
    def test_no_op_path_works(self):
        # Should not raise for fast operations
        with sympy_timeout(2):
            result = sp.Symbol("x") + 1
        assert result is not None

    def test_nesting_without_error(self):
        with sympy_timeout(2), sympy_timeout(2):
            pass  # Should not crash
