"""General-purpose math expression parsing, normalization, and scoring.

Extracted from examples/hamiltonian/reward_fn.py and reward_fn_v2.py.
Provides two scoring strategies:
  - sympy_scorer: symbolic equivalence with partial credit (0.0–1.0)
  - math_verify_scorer: HuggingFace math-verify binary verification (0.0 or 1.0)
"""

from __future__ import annotations

import logging
import re
import signal
from contextlib import contextmanager

import sympy as sp


try:
    from latex2sympy2_extended import latex2sympy as _latex2sympy
except ImportError:
    _latex2sympy = None

from typing import Any as _Any

from qgre.reward_parsing import _COMMON_DEGREES, _COMMON_DEGREES_LATEX


_mv_parse: _Any = None
_mv_verify: _Any = None
_MATH_VERIFY_AVAILABLE = False
try:
    from math_verify import parse as _mv_parse  # type: ignore[no-redef]
    from math_verify import verify as _mv_verify

    _MATH_VERIFY_AVAILABLE = True  # type: ignore[reportConstantRedefinition]
except ImportError:
    pass

logger = logging.getLogger("qgre.expression")


# ─── Timeout ───────────────────────────────────────────────────────────────────


class SympyTimeoutError(Exception):
    """Raised when sympy operation exceeds time limit."""


@contextmanager
def sympy_timeout(seconds: int = 2):
    """Context manager for timing out sympy operations.

    Uses SIGALRM — only works on Unix, not in threaded contexts.
    Falls back to no timeout on Windows or if signal is unavailable.

    Usage:
        with sympy_timeout(2):
            result = sp.simplify(expr)
    """
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise SympyTimeoutError(f"Sympy operation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ─── Private constants ────────────────────────────────────────────────────────

# PRIVATE: do not mutate. All functions accept sympy_locals and merge internally.
_BASE_SYMPY_LOCALS: dict = {"pi": sp.pi}

# Symbol map for math-verify / build_substitutions / velocity_form_gold
_VERIFY_SYMBOLS: dict[str, sp.Basic] = {
    str(s): s for s in sp.symbols("x y r theta q p t m k g", real=True)
}


# ─── Normalization ────────────────────────────────────────────────────────────


def normalize_text(s: str) -> str:
    """Normalize text for fuzzy matching."""
    s = s.lower()
    s = s.replace("**", "^").replace("\\frac", "").replace("\\cdot", "*")
    s = s.replace("·", "*").replace("×", "*").replace(" ", "")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"(\d)\*([a-z])", r"\1\2", s)
    return s


def normalize_for_sympy(expr_str: str) -> str:
    """Minimal normalization for plain-algebra sympify fallback.

    Only handles what sympy.sympify can't parse natively: Unicode, degree symbols,
    implicit multiplication, and basic cleanup. LaTeX is handled by latex2sympy.
    """
    s = expr_str.strip()
    # Strip trailing incomplete LaTeX/markdown: **, $, \, etc.
    s = re.sub(r"[\*$\\]+$", "", s).strip()
    # Strip markdown bold/italic at boundaries
    s = re.sub(r"^\*{2,3}|(?<!\w)\*{2,3}$|\*{2,3}$", "", s).strip()
    # Strip LaTeX delimiters: $$ ... $$, $ ... $
    s = re.sub(r"\$+", "", s).strip()
    # Strip trailing LaTeX line breaks \\
    s = re.sub(r"\\\\$", "", s).strip()
    # Degree symbols → radians (latex2sympy doesn't convert degrees)
    s = re.sub(r"(\d+)\^\\circ", lambda m: _COMMON_DEGREES.get(m.group(1), m.group(1)), s)
    s = re.sub(r"(\d+)\^\{\\circ\}", lambda m: _COMMON_DEGREES.get(m.group(1), m.group(1)), s)
    s = re.sub(r"(\d+)°", lambda m: _COMMON_DEGREES.get(m.group(1), m.group(1)), s)

    # Bare sin(60) etc — assume degrees for common physics angles
    def _replace_trig(match, fn_name):
        angle = match.group(1)
        if angle in _COMMON_DEGREES:
            return f"{fn_name}({_COMMON_DEGREES[angle]})"
        return match.group(0)

    s = re.sub(r"sin\((\d+)\)", lambda m: _replace_trig(m, "sin"), s)
    s = re.sub(r"cos\((\d+)\)", lambda m: _replace_trig(m, "cos"), s)
    # LaTeX velocity markers → _VDOT_ (velocity form marker)
    s = re.sub(r"\\dot\{([a-zA-Z])\}", r"_VDOT_", s)
    # LaTeX \cdot → *
    s = s.replace("\\cdot", "*")
    # LaTeX \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # LaTeX trig → plain names
    s = re.sub(r"(?<=[a-zA-Z0-9)])\\(sin|cos|tan|exp|log|ln)\b", r"*\1", s)
    s = re.sub(r"\\(sin|cos|tan|exp|log|ln)\b", r"\1", s)
    # Strip LaTeX text commands
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(left|right|,|;|quad|qquad|displaystyle)", "", s)
    # LaTeX fractions: \frac{a}{b} → (a)/(b) — pattern handles one level of nesting
    s = re.sub(
        r"\\frac\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", r"(\1)/(\2)", s
    )
    # Strip remaining LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Strip leftover braces
    s = s.replace("{", "").replace("}", "")
    # Unicode
    s = s.replace("^", "**")
    s = s.replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    for vc in ("ẋ", "ẏ", "ṙ"):
        s = s.replace(vc, "_VDOT_")
    s = s.replace("θ", "theta").replace("ω", "omega")
    s = s.replace("p_θ", "p_theta")
    s = re.sub(r"\u0307", "", s)  # combining dot above
    # Implicit multiplication
    s = re.sub(r"(\d)([a-zA-Z_(])", r"\1*\2", s)
    s = re.sub(r"([A-Z])([a-z])", r"\1*\2", s)
    _KNOWN_NAMES = {
        "sin",
        "cos",
        "tan",
        "exp",
        "log",
        "sqrt",
        "ln",
        "pi",
        "theta",
        "omega",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "theta1",
        "theta2",
        "p_theta",
        "p_r",
        "p_s",
        "p_x",
        "p_y",
    }

    def _split_vars(match):
        word = match.group(0)
        if word.lower() in _KNOWN_NAMES:
            return word
        return "*".join(word)

    s = re.sub(r"[a-z]{2,}", _split_vars, s)
    s = re.sub(r"(\))\s*([a-zA-Z_(])", r"\1*\2", s)
    s = re.sub(r"([a-zA-Z0-9_])\s*(\()", r"\1*\2", s)
    for fn in ("sin", "cos", "tan", "exp", "log", "sqrt", "ln"):
        s = s.replace(f"{fn}*(", f"{fn}(")
    s = re.sub(r"(\d)\s+([a-zA-Z_(])", r"\1*\2", s)
    # Strip leading "H =", "T =", etc.
    s = re.sub(r"^[A-Za-z_]+\s*=\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── Parsing ──────────────────────────────────────────────────────────────────


def parse_math(expr_str: str, sympy_locals: dict | None = None) -> sp.Basic | None:
    """Parse a math expression (LaTeX or plain) to sympy. Primary parser.

    Tries latex2sympy first (handles \\frac, \\sin, \\dot, derivatives, etc.),
    then falls back to sympify with normalization for plain algebra.

    Args:
        sympy_locals: additional symbol definitions merged with _BASE_SYMPY_LOCALS.
    """
    if not expr_str or expr_str.strip() == "":
        return None

    raw = expr_str.strip()
    # Strip wrapping $ delimiters for latex2sympy
    cleaned = re.sub(r"^\$+|\$+$", "", raw).strip()

    # Pre-process degree notation before any parser
    cleaned = re.sub(
        r"(\d+)\^\s*\\circ",
        lambda m: _COMMON_DEGREES_LATEX.get(m.group(1), m.group(1)),
        cleaned,
    )
    cleaned = re.sub(
        r"(\d+)\^\s*\{\\circ\}",
        lambda m: _COMMON_DEGREES_LATEX.get(m.group(1), m.group(1)),
        cleaned,
    )

    # Try latex2sympy first — handles all LaTeX natively
    if _latex2sympy is not None and "\\" in cleaned:
        latex_cleaned = cleaned
        for fn in ("sin", "cos", "tan", "exp", "log", "ln", "sqrt"):
            latex_cleaned = re.sub(rf"(?<!\\)\b{fn}\s*\(", rf"\\{fn}(", latex_cleaned)
        try:
            result = _latex2sympy(latex_cleaned)
            if isinstance(result, sp.Expr):
                return result
        except Exception as exc:
            logger.debug("latex2sympy failed on %r: %s", cleaned, exc)

    # Fallback: normalize and sympify
    normed = ""
    _locals = {**_BASE_SYMPY_LOCALS, **(sympy_locals or {})}
    try:
        normed = normalize_for_sympy(expr_str)
        result = sp.sympify(normed, locals=_locals)
        if isinstance(result, sp.Expr):
            return result
        return None
    except Exception as exc:
        logger.debug("sympify fallback failed on %r (normalized: %r): %s", expr_str, normed, exc)
        return None


def try_sympify(expr_str: str, sympy_locals: dict | None = None) -> sp.Basic | None:
    """Try to parse expression string into sympy. Uses parse_math (latex2sympy + fallback).

    Returns None if parsing fails. Callers MUST check for None before calling .subs() or
    other methods.
    """
    result = parse_math(expr_str, sympy_locals=sympy_locals)
    if result is None:
        logger.debug("parse_math returned None for: %r", expr_str)
    return result


# ─── Constant extraction ──────────────────────────────────────────────────────


def extract_constants_from_prompt(prompt: str) -> dict:
    """Extract physical constants (m, k, F, l) from the prompt text.

    Returns a sympy substitution dict mapping symbols → values.
    Used by scorers to evaluate symbolic student expressions against
    ground truth that has constants evaluated.
    """
    subs = {}
    # Mass
    m_match = re.search(r"mass\s*(?:m\s*)?=?\s*(\d+(?:\.\d+)?)", prompt)
    if m_match:
        val = sp.nsimplify(m_match.group(1))
        subs[sp.Symbol("m", positive=True)] = val
        subs[sp.Symbol("m")] = val
    # Spring constant
    k_match = re.search(r"(?:spring\s+)?constant\s*k?\s*=\s*(\d+(?:\.\d+)?)", prompt)
    if k_match:
        val = sp.nsimplify(k_match.group(1))
        subs[sp.Symbol("k", positive=True)] = val
        subs[sp.Symbol("k")] = val
    # Force
    f_match = re.search(r"force\s*(?:F\s*)?=?\s*(\d+(?:\.\d+)?)", prompt)
    if f_match:
        val = sp.nsimplify(f_match.group(1))
        subs[sp.Symbol("F")] = val
    # Length/radius (match both lowercase l and uppercase L)
    l_match = re.search(r"(?:length|radius)\s*(?:[lLR]\s*)?=?\s*(\d+(?:\.\d+)?)", prompt)
    if l_match:
        val = sp.nsimplify(l_match.group(1))
        subs[sp.Symbol("l", positive=True)] = val
        subs[sp.Symbol("l")] = val
    return subs


# ─── Variable remapping ───────────────────────────────────────────────────────


def remap_variables(student: sp.Basic, teacher: sp.Basic) -> sp.Basic:
    """Remap student variables to match teacher's coordinate AND momentum names.

    The model may use x where ground truth uses y, or p where ground truth
    uses p_theta. Both are valid — the names are arbitrary. This function
    finds a consistent mapping from student symbols to teacher symbols.

    Remaps coordinates (x, y, s, r, q, theta, ...) AND momenta (p, p_theta, ...).
    Does NOT remap constants (m, g, k).
    """
    if not isinstance(student, sp.Expr) or not isinstance(teacher, sp.Expr):
        return student

    _COORDS = {"x", "y", "s", "r", "q", "theta", "theta1", "theta2", "x1", "x2"}
    _MOMENTA = {"p", "p_x", "p_y", "p_r", "p_s", "p_theta", "p1", "p2"}
    _REMAP = _COORDS | _MOMENTA

    student_remap = {s for s in student.free_symbols if s.name in _REMAP}
    teacher_remap = {s for s in teacher.free_symbols if s.name in _REMAP}

    if teacher_remap <= student_remap:
        return student

    student_coords = {s for s in student_remap if s.name in _COORDS}
    teacher_coords = {s for s in teacher_remap if s.name in _COORDS}
    student_momenta = {s for s in student_remap if s.name in _MOMENTA}
    teacher_momenta = {s for s in teacher_remap if s.name in _MOMENTA}

    subs = {}

    if len(student_coords) == len(teacher_coords) and student_coords != teacher_coords:
        s_sorted = sorted(student_coords, key=lambda s: s.name)
        t_sorted = sorted(teacher_coords, key=lambda s: s.name)
        for sv, tv in zip(s_sorted, t_sorted, strict=False):
            if sv != tv:
                subs[sv] = tv
    elif student_coords != teacher_coords:
        logger.debug(
            "remap_variables: coord count mismatch (student=%d, teacher=%d), skipping remap",
            len(student_coords),
            len(teacher_coords),
        )

    if len(student_momenta) == len(teacher_momenta) and student_momenta != teacher_momenta:
        s_sorted = sorted(student_momenta, key=lambda s: s.name)
        t_sorted = sorted(teacher_momenta, key=lambda s: s.name)
        for sv, tv in zip(s_sorted, t_sorted, strict=False):
            if sv != tv:
                subs[sv] = tv
    elif student_momenta != teacher_momenta:
        logger.debug(
            "remap_variables: momenta count mismatch (student=%d, teacher=%d), skipping remap",
            len(student_momenta),
            len(teacher_momenta),
        )

    if subs:
        return student.subs(subs)

    return student


# ─── Structural term scoring ──────────────────────────────────────────────────


def score_terms_structurally(student: sp.Basic, teacher: sp.Basic) -> float | None:
    """Score expression by matching polynomial terms structurally.

    Returns fraction of teacher terms matched (same monomial, coefficient within 10%).
    Returns None if expressions aren't polynomial-like (trig, exp, etc.).

    This catches missing terms that numerical closeness misses:
    - V = 3x² + 19.6x (teacher) vs V = 19.6x (student)
    - Numerical: student/teacher ≈ 0.94 at x=0.4 → 0.85 partial credit
    - Structural: 1/2 terms matched → 0.50 partial credit
    """
    try:
        student_exp = sp.expand(student)
        teacher_exp = sp.expand(teacher)

        def get_terms(expr):
            terms = {}
            if expr.is_Add:
                for term in expr.args:
                    coeff, base = term.as_coeff_Mul()
                    if base == 1:
                        base = sp.Integer(1)
                    terms[base] = float(coeff)
            elif expr.is_Mul or expr.is_Pow or expr.is_Symbol:
                coeff, base = expr.as_coeff_Mul()
                if base == 1:
                    base = sp.Integer(1)
                terms[base] = float(coeff)
            elif expr.is_Number:
                terms[sp.Integer(1)] = float(expr)
            else:
                return None
            return terms

        teacher_terms = get_terms(teacher_exp)
        student_terms = get_terms(student_exp)

        if teacher_terms is None or student_terms is None:
            return None
        if not teacher_terms:
            return None

        matched = 0
        coeff_accuracy_sum = 0.0
        for base, t_coeff in teacher_terms.items():
            if base in student_terms:
                s_coeff = student_terms[base]
                if abs(t_coeff) < 1e-10:
                    if abs(s_coeff) < 1e-10:
                        matched += 1
                        coeff_accuracy_sum += 1.0
                else:
                    ratio = s_coeff / t_coeff
                    if 0.9 <= ratio <= 1.1:
                        matched += 1
                        coeff_accuracy_sum += 1.0 - abs(1.0 - ratio)
                    elif 0.5 <= ratio <= 2.0:
                        matched += 0.5
                        coeff_accuracy_sum += 0.5 * (1.0 - min(abs(1.0 - ratio), 1.0))

        if len(teacher_terms) == 0:
            return None
        term_coverage = matched / len(teacher_terms)
        avg_coeff_acc = coeff_accuracy_sum / max(matched, 1)

        return term_coverage * avg_coeff_acc

    except Exception as exc:
        logger.debug("score_terms_structurally failed: %s", exc)
        return None


# ─── Expression scoring ───────────────────────────────────────────────────────


def score_expression(
    student_str: str | None,
    teacher_str: str,
    variables: list[str],
    constant_subs: dict | None = None,
    sympy_locals: dict | None = None,
) -> float:
    """Score a mathematical expression against ground truth.

    Args:
        constant_subs: optional sympy substitution dict (from extract_constants_from_prompt)
            to evaluate symbolic constants (m, k, F) in student expression before comparing.
        sympy_locals: additional symbol definitions merged with _BASE_SYMPY_LOCALS.

    Returns 0.0-1.0:
    - 1.0: exact symbolic or numerical match (including after variable remapping)
    - 0.8: correct up to sign (V vs -V — valid physics, different reference frame)
    - 0.2-0.7: partial credit based on structural term matching (robust)
    - 0.2: attempted but unparseable
    - 0.0: not found
    """
    if student_str is None:
        return 0.0

    # Handle chained equalities: "(6/2)*x² = 3x²" → try final expression first
    if "=" in student_str:
        parts = [p.strip() for p in student_str.split("=")]
        student = try_sympify(parts[-1], sympy_locals=sympy_locals)
        if student is None and len(parts) > 1:
            student = try_sympify(parts[0], sympy_locals=sympy_locals)
    else:
        student = try_sympify(student_str, sympy_locals=sympy_locals)
    teacher = try_sympify(teacher_str, sympy_locals=sympy_locals)

    # Substitute known constants (m, k, F from prompt) into student expression
    if constant_subs and student is not None and isinstance(student, sp.Expr):
        try:
            student = student.subs(constant_subs)
        except Exception as exc:
            logger.debug("constant substitution failed on %s: %s", student, exc)

    if teacher is None:
        return string_similarity(student_str, teacher_str)
    if student is None:
        return 0.2

    if not isinstance(student, sp.Basic) or isinstance(student, sp.logic.boolalg.BooleanAtom):
        return 0.2
    if not isinstance(teacher, sp.Basic) or isinstance(teacher, sp.logic.boolalg.BooleanAtom):
        return string_similarity(student_str, teacher_str)

    # Variable remapping: student may use x where teacher uses y (both valid)
    student_remapped = remap_variables(student, teacher)

    candidates = [student_remapped]
    if student_remapped != student:
        candidates.append(student)

    for candidate in candidates:
        # Exact symbolic match — try multiple simplification strategies with timeout
        for simplifier in [sp.simplify, sp.trigsimp, sp.ratsimp, sp.nsimplify]:
            try:
                with sympy_timeout(2):
                    if simplifier(candidate - teacher) == 0:
                        return 1.0
            except (Exception, SympyTimeoutError) as exc:
                logger.debug("simplifier %s failed: %s", simplifier.__name__, exc)
                continue

        try:
            with sympy_timeout(2):
                if sp.simplify(sp.expand(candidate) - sp.expand(teacher)) == 0:
                    return 1.0
        except (Exception, SympyTimeoutError) as exc:
            logger.debug("expand+simplify failed: %s", exc)

        try:
            with sympy_timeout(2):
                if sp.trigsimp(sp.expand_trig(candidate - teacher)) == 0:
                    return 1.0
        except (Exception, SympyTimeoutError) as exc:
            logger.debug("trigsimp failed: %s", exc)

        # Sign convention check
        for simplifier in [sp.simplify, sp.expand]:
            try:
                with sympy_timeout(2):
                    if simplifier(candidate + teacher) == 0:
                        return 0.8
            except (Exception, SympyTimeoutError) as exc:
                logger.debug("sign-check %s failed: %s", simplifier.__name__, exc)
                continue

    # Numerical equivalence check at two points (use remapped candidate)
    candidate = candidates[0]
    try:
        free = (candidate.free_symbols | teacher.free_symbols) - {sp.Symbol("pi")}
        if free:
            _probes = [
                sp.Rational(3, 7),
                sp.Rational(5, 11),
                sp.Rational(7, 13),
                sp.Rational(11, 17),
                sp.Rational(13, 19),
                sp.Rational(17, 23),
            ]
            free_list = sorted(free, key=str)
            probe_set_1 = {s: _probes[j % len(_probes)] for j, s in enumerate(free_list)}
            probe_set_2 = {s: _probes[(j + 1) % len(_probes)] for j, s in enumerate(free_list)}
            for test_vals in [probe_set_1, probe_set_2]:
                s_val = float(candidate.subs(test_vals))
                t_val = float(teacher.subs(test_vals))
                if abs(t_val) <= 1e-10:
                    if abs(s_val) > 1e-6:
                        break
                    continue
                if abs(s_val - t_val) > 1e-6 * max(abs(t_val), 1):
                    break
            else:
                return 1.0

            # Check sign-flipped numerical match
            for test_vals in [probe_set_1, probe_set_2]:
                s_val = float(candidate.subs(test_vals))
                t_val = float(teacher.subs(test_vals))
                if abs(t_val) <= 1e-10:
                    continue
                if abs(s_val + t_val) > 1e-6 * max(abs(t_val), 1):
                    break
            else:
                return 0.8
        else:
            s_val = float(candidate)
            t_val = float(teacher)
            if abs(s_val - t_val) < 1e-8 * max(abs(t_val), 1):
                return 1.0
            if abs(s_val + t_val) < 1e-8 * max(abs(t_val), 1):
                return 0.8
    except Exception as exc:
        logger.debug("numerical equivalence check failed: %s", exc)

    # Partial credit: structural term matching > numerical closeness
    try:
        structural_score = score_terms_structurally(candidate, teacher)
        if structural_score is not None:
            return 0.2 + 0.5 * structural_score

        # Fallback: numerical closeness for non-polynomial expressions
        free = (candidate.free_symbols | teacher.free_symbols) - {sp.Symbol("pi")}
        numerical_score = 0.0
        if free:
            test_point = {s: sp.Rational(3, 7) for s in free}
            try:
                s_val = float(candidate.subs(test_point))
                t_val = float(teacher.subs(test_point))
                if abs(t_val) > 1e-10:
                    ratio = s_val / t_val
                    numerical_score = max(0.0, 1.0 - abs(1.0 - ratio))
                elif abs(s_val) < 1e-10:
                    numerical_score = 1.0
            except Exception as exc:
                logger.debug("partial credit numerical eval failed: %s", exc)

        student_syms = candidate.free_symbols
        teacher_syms = teacher.free_symbols
        if not teacher_syms:
            sym_overlap = 0.0 if student_syms else numerical_score
        else:
            sym_overlap = len(student_syms & teacher_syms) / len(teacher_syms)

        partial = 0.2 + 0.4 * numerical_score + 0.1 * sym_overlap
        return min(partial, 0.70)
    except Exception as exc:
        logger.debug("partial credit calculation failed: %s", exc)
        return 0.2


# ─── String similarity ────────────────────────────────────────────────────────


def string_similarity(a: str, b: str) -> float:
    """Fuzzy string similarity for math expressions."""
    na = normalize_text(a)
    nb = normalize_text(b)
    if na == nb:
        return 1.0
    if nb in na or na in nb:
        return 0.7
    tokens_a = set(re.findall(r"[a-z_]+|\d+", na))
    tokens_b = set(re.findall(r"[a-z_]+|\d+", nb))
    if not tokens_b:
        return 0.0
    return 0.3 * len(tokens_a & tokens_b) / len(tokens_b)


# ─── Best match ───────────────────────────────────────────────────────────────


def best_match(
    candidates: list[str],
    teacher_str: str,
    variables: list[str] | None = None,
    constant_subs: dict | None = None,
    sympy_locals: dict | None = None,
) -> float:
    """Score the best-matching candidate expression against ground truth.

    Tries each candidate, returns the highest score.
    """
    if not candidates:
        return 0.0
    best = 0.0
    for expr in candidates:
        score = score_expression(
            expr,
            teacher_str,
            variables or [],
            constant_subs=constant_subs,
            sympy_locals=sympy_locals,
        )
        best = max(best, score)
        if best >= 1.0:
            break
    return best


# ─── math-verify functions (guarded) ─────────────────────────────────────────


def gold_parse(sympy_str: str) -> list | None:
    """Convert sympy-format ground truth to math-verify parsed form.

    Strategy: sympify → LaTeX → math-verify parse. This ensures the gold
    goes through the same LaTeX parser as the model's output.

    Raises ImportError if math_verify is not installed.
    """
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError(
            "math_verify is required for gold_parse. Install with: pip install math-verify"
        )
    try:
        expr = sp.sympify(sympy_str, locals=_VERIFY_SYMBOLS)
        latex = sp.latex(expr)
        result = _mv_parse(f"${latex}$")
        return result if result else None
    except Exception as exc:
        logger.debug("Cannot parse ground truth %r: %s", sympy_str, exc)
        return None


def find_correct(
    expressions: list[tuple[str, str, int, int]],
    gold_parsed: list,
    lhs_patterns: list[str] | None = None,
    substitutions: dict | None = None,
) -> tuple[float, list]:
    """Find the first expression that matches gold via math-verify.

    Returns (1.0, [(span)]) on match, (0.0, []) otherwise.
    If substitutions is provided (e.g., {m: 2}), tries substituting free
    variables in the model's answer before comparing — handles generic
    formulas like p²/(2m) when gold has numeric values like p²/4.

    Raises ImportError if math_verify is not installed.
    """
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError(
            "math_verify is required for find_correct. Install with: pip install math-verify"
        )
    for lhs, rhs, char_start, char_end in expressions:
        if lhs_patterns is not None:
            lhs_clean = lhs.lower().strip("*$\\")
            if not any(lhs_clean == p.lower() for p in lhs_patterns):
                continue

        try:
            answer_parsed = _mv_parse(f"${rhs}$")
        except Exception as exc:
            logger.debug("math-verify parse failed on rhs %r: %s", rhs, exc)
            continue

        if not answer_parsed:
            continue

        try:
            if _mv_verify(gold_parsed, answer_parsed):
                return 1.0, [(char_start, char_end)]
        except Exception as exc:
            logger.debug("math-verify verify failed: %s", exc)

        if substitutions and answer_parsed:
            try:
                expr = answer_parsed[0] if isinstance(answer_parsed, list) else answer_parsed
                if hasattr(expr, "free_symbols") and expr.free_symbols:
                    subbed = expr.subs(substitutions)
                    with sympy_timeout(2):
                        subbed_simplified = sp.simplify(subbed)
                    subbed_latex = sp.latex(subbed_simplified)
                    subbed_parsed = _mv_parse(f"${subbed_latex}$")
                    if subbed_parsed and _mv_verify(gold_parsed, subbed_parsed):
                        return 1.0, [(char_start, char_end)]
            except (Exception, SympyTimeoutError) as exc:
                logger.debug("substitution fallback failed: %s", exc)

    return 0.0, []


def find_derivative(
    expressions: list[tuple[str, str, int, int]],
    gold_parsed: list,
    var: str,
    substitutions: dict | None = None,
) -> tuple[float, list]:
    """Find a correct Hamilton equation dVAR/dt = gold.

    Raises ImportError if math_verify is not installed.
    """
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError(
            "math_verify is required for find_derivative. Install with: pip install math-verify"
        )
    coord_aliases: dict[str, list[str]] = {"q": ["x", "y", "r", "theta", "s"], "p": ["p"]}
    all_vars = [var, *coord_aliases.get(var, [])]

    lhs_markers: set[str] = set()
    for v in all_vars:
        lhs_markers.add(f"d{v}/dt")
        lhs_markers.add(f"d{v}")
        lhs_markers.add(f"dot{{{v}}}")
        lhs_markers.add(f"frac{{d{v}}}{{dt}}")

    for lhs, rhs, char_start, char_end in expressions:
        lhs_clean = lhs.lower().strip("*$\\").replace(" ", "")
        if not any(m in lhs_clean for m in lhs_markers):
            continue

        try:
            answer_parsed = _mv_parse(f"${rhs}$")
        except Exception as exc:
            logger.debug("math-verify parse failed on rhs %r: %s", rhs, exc)
            continue

        if not answer_parsed:
            continue

        try:
            if _mv_verify(gold_parsed, answer_parsed):
                return 1.0, [(char_start, char_end)]
        except Exception as exc:
            logger.debug("math-verify verify failed: %s", exc)

        if substitutions and answer_parsed:
            try:
                expr = answer_parsed[0] if isinstance(answer_parsed, list) else answer_parsed
                if hasattr(expr, "free_symbols") and expr.free_symbols:
                    with sympy_timeout(2):
                        subbed = sp.simplify(expr.subs(substitutions))
                    subbed_parsed = _mv_parse(f"${sp.latex(subbed)}$")
                    if subbed_parsed and _mv_verify(gold_parsed, subbed_parsed):
                        return 1.0, [(char_start, char_end)]
            except (Exception, SympyTimeoutError) as exc:
                logger.debug("substitution fallback failed: %s", exc)

    return 0.0, []


# ─── Substitution helpers ─────────────────────────────────────────────────────


def build_substitutions(meta: dict) -> dict:
    """Extract known constants from metadata for substitution into generic formulas.

    The model often writes correct formulas with symbolic variables (p²/(2m))
    instead of substituting numeric values (p²/4 when m=2). This builds a
    substitution dict so we can verify equivalence after plugging in constants.

    Only substitutes values that are DERIVABLE from the problem metadata —
    never introduces assumptions. Wrong formulas still score 0.
    """
    subs: dict = {}
    T_str = meta.get("T_expr")
    if T_str:
        try:
            p = _VERIFY_SYMBOLS["p"]
            T_sym = sp.sympify(T_str, locals=_VERIFY_SYMBOLS)
            coeff = T_sym.coeff(p, 2)
            if coeff and coeff != 0:
                mass = sp.Rational(1, 2) / coeff
                subs[_VERIFY_SYMBOLS["m"]] = mass
        except Exception as exc:
            logger.debug("build_substitutions: T_expr mass extraction failed: %s", exc)

    V_str = meta.get("V_expr")
    coord = meta.get("coordinates", "x")
    if V_str and not subs.get(_VERIFY_SYMBOLS["m"]):
        logger.debug("build_substitutions: skipping g extraction from V_expr — mass not available")
    if V_str and subs.get(_VERIFY_SYMBOLS["m"]):
        try:
            q_sym = _VERIFY_SYMBOLS.get(coord, sp.Symbol(coord, real=True))
            V_sym = sp.sympify(V_str, locals=_VERIFY_SYMBOLS)
            v_coeff = V_sym.coeff(q_sym, 1)
            if v_coeff and v_coeff != 0:
                mass = subs[_VERIFY_SYMBOLS["m"]]
                g_val = v_coeff / mass
                if g_val.is_number:
                    subs[_VERIFY_SYMBOLS["g"]] = g_val
        except Exception as exc:
            logger.debug("build_substitutions: V_expr g extraction failed: %s", exc)

    if V_str:
        try:
            q_sym = _VERIFY_SYMBOLS.get(coord, sp.Symbol(coord, real=True))
            V_sym = sp.sympify(V_str, locals=_VERIFY_SYMBOLS)
            v_coeff2 = V_sym.coeff(q_sym, 2)
            if v_coeff2 and v_coeff2 != 0:
                k_val = 2 * v_coeff2
                if k_val.is_number:
                    subs[_VERIFY_SYMBOLS["k"]] = k_val
        except Exception as exc:
            logger.debug("build_substitutions: V_expr k extraction failed: %s", exc)

    return subs


def velocity_form_gold(meta: dict) -> dict[str, list | None]:
    """Generate velocity-form gold expressions by substituting p → m*coord_dot.

    Hamiltonian formalism uses (q, p) but students often write T = ½mv².
    Both are correct. This creates alternate gold expressions in velocity form
    so math-verify can match either.

    Returns dict mapping meta_key → parsed gold, for T_expr and H_expr only.

    Raises ImportError if math_verify is not installed.
    """
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError(
            "math_verify is required for velocity_form_gold. Install with: pip install math-verify"
        )
    result: dict[str, list | None] = {}
    T_str = meta.get("T_expr")
    if not T_str:
        return result

    try:
        p = _VERIFY_SYMBOLS["p"]
        T_sym = sp.sympify(T_str, locals=_VERIFY_SYMBOLS)

        coeff = T_sym.coeff(p, 2)
        if not coeff or coeff == 0:
            return result
        mass = sp.Rational(1, 2) / coeff

        coord = meta.get("coordinates", "x")
        q_dot = sp.Symbol(f"dot{{{coord}}}")

        p_sub = mass * q_dot
        T_vel = T_sym.subs(p, p_sub)
        T_vel_simplified = sp.simplify(T_vel)
        T_latex = sp.latex(T_vel_simplified)
        t_parsed = _mv_parse(f"${T_latex}$")
        if t_parsed is None:
            logger.debug(
                "velocity_form_gold: _mv_parse returned None for T_expr (latex=%r)", T_latex
            )
        result["T_expr"] = t_parsed or None

        H_str = meta.get("H_expr")
        if H_str:
            H_sym = sp.sympify(H_str, locals=_VERIFY_SYMBOLS)
            H_vel = H_sym.subs(p, p_sub)
            H_vel_simplified = sp.simplify(H_vel)
            H_latex = sp.latex(H_vel_simplified)
            h_parsed = _mv_parse(f"${H_latex}$")
            if h_parsed is None:
                logger.debug(
                    "velocity_form_gold: _mv_parse returned None for H_expr (latex=%r)", H_latex
                )
            result["H_expr"] = h_parsed or None
    except (SympyTimeoutError, Exception) as exc:
        logger.debug("velocity_form_gold failed: %s", exc)

    return result


# ─── Thin scorer wrappers ─────────────────────────────────────────────────────


def sympy_scorer(
    student_str: str,
    teacher_str: str,
    variables: list[str] | None = None,
    constant_subs: dict | None = None,
    sympy_locals: dict | None = None,
) -> float:
    """Score via sympy equivalence. Returns 0.0-1.0 with partial credit."""
    return score_expression(
        student_str,
        teacher_str,
        variables or [],
        constant_subs=constant_subs,
        sympy_locals=sympy_locals,
    )


def math_verify_scorer(
    student_str: str,
    teacher_str: str,
    substitutions: dict | None = None,
) -> float:
    """Score via HuggingFace math-verify. Returns 0.0 or 1.0 (binary)."""
    if not _MATH_VERIFY_AVAILABLE:
        raise ImportError(
            "math_verify is required for math_verify_scorer. Install with: pip install math-verify"
        )
    gold = gold_parse(teacher_str)
    if gold is None:
        return 0.0
    from qgre.reward_parsing import extract_rhs_expressions

    expressions = extract_rhs_expressions(student_str)
    if not expressions:
        try:
            answer = _mv_parse(f"${student_str}$")
            if answer and _mv_verify(gold, answer):
                return 1.0
        except Exception as exc:
            logger.debug("math_verify_scorer direct parse failed: %s", exc)
        return 0.0
    score, _ = find_correct(expressions, gold, substitutions=substitutions)
    return score
