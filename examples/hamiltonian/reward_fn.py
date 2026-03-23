"""Granular progressive reward function for Hamiltonian mechanics training.

Phase 1: q_format, q_has_math — structured response with labeled sections
Phase 2: q_momentum_defined, q_T_uses_p, q_V_correct — momentum form + physics
Phase 3: q_correct_dqdt, q_correct_dpdt — Hamilton's equations match ground truth
Phase 4: q_correct_H, q_consistency — full Hamiltonian + internal consistency

Each quality targets a specific labeled section in the structured output format:
  COORDINATES: q = ...
  MOMENTUM: p = ...
  KINETIC: T = ...
  POTENTIAL: V = ...
  HAMILTONIAN: H = ...
  EQUATIONS:
    dq/dt = ...
    dp/dt = ...
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import sympy as sp

from qgre.types import RewardResult

logger = logging.getLogger("qgre.hamiltonian_reward")

_DIAG_PATH = Path("output/hamiltonian/diagnostics.jsonl")


# ─── Expression normalization ─────────────────────────────────────────────────

def _normalize_text(s: str) -> str:
    """Normalize text for fuzzy matching."""
    s = s.lower()
    s = s.replace("**", "^").replace("\\frac", "").replace("\\cdot", "*")
    s = s.replace("·", "*").replace("×", "*").replace(" ", "")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"(\d)\*([a-z])", r"\1\2", s)
    return s


def _normalize_for_sympy(expr_str: str) -> str:
    """Clean up expression string for sympy parsing."""
    s = expr_str.strip()
    # Strip markdown bold/italic at boundaries only (not multiplication *)
    s = re.sub(r"^\*{2,3}|(?<!\w)\*{2,3}$|\*{2,3}$", "", s).strip()
    # Strip LaTeX delimiters: $$ ... $$, $ ... $
    s = re.sub(r"\$+", "", s).strip()
    # Strip trailing LaTeX line breaks \\
    s = re.sub(r"\\\\$", "", s).strip()
    # LaTeX \cdot → * (before stripping LaTeX commands)
    s = s.replace("\\cdot", "*")
    # LaTeX \sqrt{x} → sqrt(x)
    s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
    # LaTeX \sin, \cos etc → sin, cos
    s = re.sub(r"\\(sin|cos|tan|exp|log|ln)\b", r"\1", s)
    # Degree symbol: sin(30°) → sin(pi/6), sin(45°) → sin(pi/4), sin(60°) → sin(pi/3)
    s = re.sub(r"(\d+)°", lambda m: {"30": "pi/6", "45": "pi/4", "60": "pi/3", "90": "pi/2"}.get(m.group(1), m.group(1)), s)
    # Strip LaTeX \text{}, \left, \right etc
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\(left|right|,|;|quad|qquad|displaystyle)", "", s)
    # LaTeX \dot{x} → _VDOT_ (velocity form marker) — BEFORE stripping LaTeX commands
    s = re.sub(r"\\dot\{([a-zA-Z])\}", r"_VDOT_", s)
    # LaTeX fractions: \frac{a}{b} → (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
    # Strip remaining LaTeX commands
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    # Strip leftover braces from LaTeX
    s = s.replace("{", "").replace("}", "")
    # Unicode superscripts
    s = s.replace("^", "**")
    s = s.replace("²", "**2").replace("³", "**3").replace("⁴", "**4")
    # Unicode velocity symbols (precomposed) → _VDOT_
    # NOTE: ẋ ẏ ṙ are precomposed single chars. θ̇ and q̇ are base+combining dot.
    # Handle precomposed first, then combining dot separately.
    for vc in ("ẋ", "ẏ", "ṙ"):
        s = s.replace(vc, "_VDOT_")
    s = s.replace("θ", "theta").replace("ω", "omega")
    s = s.replace("p_θ", "p_theta").replace("p_r", "p_r").replace("p_s", "p_s")
    # Strip combining characters (dots above letters)
    s = re.sub(r"\u0307", "", s)  # combining dot above
    # digit * letter: 3x → 3*x
    s = re.sub(r"(\d)([a-zA-Z_(])", r"\1*\2", s)
    # letter/paren * letter: )x → )*x, ) x → )* x, x( → x*(
    s = re.sub(r"(\))\s*([a-zA-Z_(])", r"\1*\2", s)
    # letter/number before paren: x( → x*(
    s = re.sub(r"([a-zA-Z0-9_])\s*(\()", r"\1*\2", s)
    # Restore function calls broken by above: sin*( → sin(
    for fn in ("sin", "cos", "tan", "exp", "log", "sqrt", "ln"):
        s = s.replace(f"{fn}*(", f"{fn}(")
    # fraction-space-variable: 1/2 x → 1/2*x
    s = re.sub(r"(\d)\s+([a-zA-Z_(])", r"\1*\2", s)
    # Strip leading "H =", "T =", "V =", etc. (leftover from label extraction)
    s = re.sub(r"^[A-Za-z_]+\s*=\s*", "", s)
    # Clean up whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ─── Sympy helpers ────────────────────────────────────────────────────────────

_SYMPY_LOCALS = {
    "x": sp.Symbol("x"), "y": sp.Symbol("y"), "r": sp.Symbol("r", positive=True),
    "s": sp.Symbol("s"), "q": sp.Symbol("q"),
    "p": sp.Symbol("p"), "p_x": sp.Symbol("p_x"), "p_y": sp.Symbol("p_y"),
    "p_r": sp.Symbol("p_r", positive=True), "p_s": sp.Symbol("p_s"),
    "p_theta": sp.Symbol("p_theta"), "p1": sp.Symbol("p1"), "p2": sp.Symbol("p2"),
    "theta": sp.Symbol("theta"), "theta1": sp.Symbol("theta1"), "theta2": sp.Symbol("theta2"),
    "x1": sp.Symbol("x1"), "x2": sp.Symbol("x2"),
    "m": sp.Symbol("m", positive=True),
    "g": sp.Rational(98, 10),
    "pi": sp.pi,
}


def _try_sympify(expr_str: str) -> sp.Basic | None:
    """Try to parse expression string into sympy, with normalization."""
    if not expr_str or expr_str.strip() == "":
        return None
    try:
        normed = _normalize_for_sympy(expr_str)
        result = sp.sympify(normed, locals=_SYMPY_LOCALS)
        if isinstance(result, sp.logic.boolalg.BooleanAtom):
            return None
        # sympify can return list/tuple for comma-separated exprs — reject those
        if not isinstance(result, sp.Basic):
            return None
        return result
    except Exception:
        return None


def _score_expression(student_str: str | None, teacher_str: str, variables: list[str]) -> float:
    """Score a mathematical expression against ground truth.

    Returns 0.0-1.0:
    - 1.0: exact symbolic or numerical match
    - 0.2-0.85: partial credit based on numerical closeness
    - 0.2: attempted but unparseable
    - 0.0: not found
    """
    if student_str is None:
        return 0.0

    student = _try_sympify(student_str)
    teacher = _try_sympify(teacher_str)

    if teacher is None:
        return _string_similarity(student_str, teacher_str)
    if student is None:
        return 0.2

    if not isinstance(student, sp.Basic) or isinstance(student, sp.logic.boolalg.BooleanAtom):
        return 0.2
    if not isinstance(teacher, sp.Basic) or isinstance(teacher, sp.logic.boolalg.BooleanAtom):
        return _string_similarity(student_str, teacher_str)

    # Exact symbolic match — try multiple simplification strategies
    for simplifier in [sp.simplify, sp.trigsimp, sp.ratsimp, sp.nsimplify]:
        try:
            if simplifier(student - teacher) == 0:
                return 1.0
        except Exception:
            continue

    try:
        if sp.simplify(sp.expand(student) - sp.expand(teacher)) == 0:
            return 1.0
    except Exception:
        pass

    try:
        if sp.trigsimp(sp.expand_trig(student - teacher)) == 0:
            return 1.0
    except Exception:
        pass

    # Numerical equivalence check at two points
    try:
        free = (student.free_symbols | teacher.free_symbols) - {sp.Symbol('pi')}
        if free:
            for test_vals in [{s: sp.Rational(3, 7) for s in free}, {s: sp.Rational(5, 11) for s in free}]:
                s_val = float(student.subs(test_vals))
                t_val = float(teacher.subs(test_vals))
                if abs(t_val) > 1e-10 and abs(s_val - t_val) > 1e-6 * max(abs(t_val), 1):
                    break
            else:
                return 1.0  # Both points matched
    except Exception:
        pass

    # Partial credit based on numerical closeness
    try:
        free = (student.free_symbols | teacher.free_symbols) - {sp.Symbol('pi')}
        numerical_score = 0.0
        if free:
            test_point = {s: sp.Rational(3, 7) for s in free}
            try:
                s_val = float(student.subs(test_point))
                t_val = float(teacher.subs(test_point))
                if abs(t_val) > 1e-10:
                    ratio = s_val / t_val
                    numerical_score = max(0.0, 1.0 - abs(1.0 - ratio))
                elif abs(s_val) < 1e-10:
                    numerical_score = 1.0
            except Exception:
                pass

        student_syms = student.free_symbols
        teacher_syms = teacher.free_symbols
        sym_overlap = len(student_syms & teacher_syms) / len(teacher_syms) if teacher_syms else 1.0

        partial = 0.2 + 0.5 * numerical_score + 0.2 * sym_overlap
        return min(partial, 0.85)
    except Exception:
        return 0.2


def _string_similarity(a: str, b: str) -> float:
    na = _normalize_text(a)
    nb = _normalize_text(b)
    if na == nb:
        return 1.0
    if nb in na or na in nb:
        return 0.7
    tokens_a = set(re.findall(r"[a-z_]+|\d+", na))
    tokens_b = set(re.findall(r"[a-z_]+|\d+", nb))
    if not tokens_b:
        return 0.0
    return 0.3 * len(tokens_a & tokens_b) / len(tokens_b)


# ─── Structured section extractors ────────────────────────────────────────────

def _extract_labeled(text: str, label: str) -> str | None:
    """Extract the value after a labeled line like 'KINETIC: T = ...'

    Takes the LAST match — the model writes derivation headers early
    (### 2. **Kinetic Energy**) and structured labels at the end
    (**KINETIC:** T = p²/6). We want the final labeled answer.
    """
    patterns = [
        rf"{label}:\s*[A-Za-z_]*\s*=\s*([^\n]+)",  # KINETIC: T = ...
        rf"{label}:\s*([^\n]+)",                      # KINETIC: p²/6
    ]
    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            expr = matches[-1].group(1).strip()  # Last match
            # Take the last = if there are multiple (e.g. "T = p²/(2m) = p²/6")
            parts = expr.rsplit("=", 1)
            return parts[-1].strip() if len(parts) > 1 else expr
    return None


def _extract_equations_block(text: str) -> list[str]:
    """Extract equations from EQUATIONS: block or scattered patterns.

    Handles multiple model output formats:
    - Indented: '  dq/dt = p/3'
    - Bulleted: '- dq/dt = p/3'
    - LaTeX: '$ \\frac{dq}{dt} = p $'
    - Mixed: '- $ \\frac{dp}{dt} = -6x $'
    """
    results = []

    # Try EQUATIONS: block — grab everything after the label until next section or end
    block_m = re.search(r"EQUATIONS[:\s]*\n((?:[\s\-*$]+[^\n]+\n?)+)", text, re.IGNORECASE)
    if block_m:
        block_text = block_m.group(1)
        # Normalize LaTeX fractions first
        block_text = re.sub(r"\\frac\{d([a-z_]+)\}\{dt\}", r"d\1/dt", block_text)
        block_text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", block_text)
        block_text = re.sub(r"\\[a-zA-Z]+", "", block_text)
        block_text = re.sub(r"[$*]", "", block_text)
        for line in block_text.strip().split("\n"):
            line = line.strip().lstrip("-").strip()
            eq_m = re.search(r"=\s*([^\n;]+)", line)
            if eq_m:
                results.append(eq_m.group(1).strip())
        if results:
            return results

    # Fallback: scattered equation patterns (also handle LaTeX)
    # Normalize LaTeX in full text for fallback
    norm_text = re.sub(r"\\frac\{d([a-z_]+)\}\{dt\}", r"d\1/dt", text)
    norm_text = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", norm_text)
    norm_text = re.sub(r"\\[a-zA-Z]+", "", norm_text)
    norm_text = re.sub(r"[$*]", "", norm_text)
    for pat in [r"d[a-z_]+/dt\s*=\s*([^\n;,]+)", r"[∂d]H/[∂d][a-z_]+\s*=\s*([^\n;,]+)"]:
        for m in re.finditer(pat, norm_text):
            results.append(m.group(1).strip())
    return results


def _extract_H(text: str) -> str | None:
    """Extract Hamiltonian expression."""
    result = _extract_labeled(text, "HAMILTONIAN")
    if result:
        return result
    # Fallback: H = ... anywhere
    m = re.search(r"H\s*=\s*([^\n;]+)", text)
    if m:
        expr = m.group(1).strip()
        parts = expr.rsplit("=", 1)
        return parts[-1].strip() if len(parts) > 1 else expr
    return None


def _extract_numbers_from_prompt(prompt: str) -> set[str]:
    nums = set()
    for m in re.finditer(r"[=]\s*(\d+(?:\.\d+)?)", prompt):
        nums.add(m.group(1))
    for m in re.finditer(r"(?:mass|constant|length|radius|velocity|charge|field)\s+.*?(\d+(?:\.\d+)?)", prompt):
        nums.add(m.group(1))
    return nums


# ─── Quality scorers ──────────────────────────────────────────────────────────

def _score_format(text: str) -> float:
    """q_format: structured response with labeled sections."""
    labels_found = 0
    for label in ["COORDINATES", "MOMENTUM", "KINETIC", "POTENTIAL", "HAMILTONIAN", "EQUATIONS"]:
        if re.search(rf"{label}\s*:", text, re.IGNORECASE):
            labels_found += 1

    if labels_found >= 5:
        return 1.0
    if labels_found >= 3:
        return 0.7
    # Fallback: check for basic physics content
    has_math = any(s in text for s in ["=", "H ", "T ", "V "])
    has_length = len(text.strip()) > 100
    if has_length and has_math:
        return 0.4
    if has_length:
        return 0.2
    return 0.0


def _score_has_math(text: str) -> float:
    """q_has_math: has mathematical content."""
    indicators = ["=", "**2", "^2", "²", "/2", "p^2", "p²", "p**2",
                   "cos(", "sin(", "exp(", "sqrt(", "H =", "T =", "V ="]
    count = sum(1 for p in indicators if p.lower() in text.lower())
    return min(1.0, count / 3)


def _score_momentum_defined(text: str) -> float:
    """q_momentum_defined: MOMENTUM section defines p in terms of q̇."""
    # Check for MOMENTUM: label with content (use last match via _extract_labeled)
    momentum_str = _extract_labeled(text, "MOMENTUM")
    has_label = momentum_str is not None

    if has_label and momentum_str:
        has_numbers = bool(re.search(r"\d", momentum_str))
        has_expression = bool(re.search(r"[a-z*/+\-]", momentum_str))
        if has_numbers and has_expression:
            return 1.0
        if has_expression:
            return 0.7
        return 0.5

    # Fallback: check for momentum definition anywhere
    if re.search(r"p\s*=\s*m\s*[*·]?\s*[a-zθ]|p\s*=\s*\d+\s*[*·]?\s*d[a-z]", text, re.IGNORECASE):
        return 0.5
    if re.search(r"conjugate\s+momentum|p\s*=\s*\d", text, re.IGNORECASE):
        return 0.3
    return 0.0


def _score_T_uses_p(text: str, meta: dict) -> float:
    """q_T_uses_p: KINETIC section has T in terms of momentum p, NOT velocity q̇.

    Checks the FULL kinetic expression chain, not just the final simplification.
    Model often writes T = p²/(2m) = (mẋ)²/(2m) = ẋ²/2 — the p² at the start matters.
    """
    # Get full kinetic line (before rsplit), not just final = part
    # Use LAST match — model writes headers early, structured labels at end
    kinetic_full = None
    for pat in [r"KINETIC:\s*[A-Za-z_]*\s*=\s*([^\n]+)", r"KINETIC:\s*([^\n]+)"]:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            kinetic_full = matches[-1].group(1).strip()
            break
    if kinetic_full is None:
        m = re.search(r"T\s*=\s*([^\n;]+)", text)
        if m:
            kinetic_full = m.group(1).strip()
        else:
            return 0.0

    # Also get the final form for strict checking
    parts = kinetic_full.rsplit("=", 1)
    kinetic_final = parts[-1].strip() if len(parts) > 1 else kinetic_full

    # Check for momentum variables (good)
    has_p = bool(re.search(r"p[_²^2\s/(*]|p\*\*2|p\^2", kinetic_full))
    # Check for velocity in the FINAL form (bad — means final answer uses velocity)
    has_velocity_final = bool(re.search(
        r"[ẋẏṙ]|\\dot|d[a-z]/dt|_VDOT_|q̇|θ̇", kinetic_final, re.IGNORECASE
    ))

    if has_p and not has_velocity_final:
        # Final answer is in p form — check correctness
        expected_T = meta.get("T_expr", "")
        if expected_T:
            score = _score_expression(kinetic_final, expected_T, [])
            return max(0.7, score)  # At least 0.7 for having p form
        return 0.8
    if has_p and has_velocity_final:
        # Started with p but simplified back to velocity in final form.
        # Score same as pure velocity — only the FINAL form matters.
        # No credit for "knowing p" if you undo it. This kills the
        # local attractor where the model oscillates at 0.5.
        return 0.2
    if has_velocity_final:
        return 0.2  # Wrote T but in velocity form
    return 0.3  # Wrote something but unclear


def _score_V_correct(text: str, meta: dict) -> float:
    """q_V_correct: POTENTIAL section matches ground truth V."""
    expected_V = meta.get("V_expr", "")
    if not expected_V or expected_V == "none":
        return 0.0

    potential_str = _extract_labeled(text, "POTENTIAL")
    if potential_str is None:
        # Fallback
        m = re.search(r"V\s*=\s*([^\n;]+)", text)
        if m:
            potential_str = m.group(1).strip()
        else:
            return 0.0

    return _score_expression(potential_str, expected_V, [])


def _score_grounding(text: str, prompt: str) -> float:
    """q_grounding: completion uses actual numerical values from the prompt."""
    prompt_nums = _extract_numbers_from_prompt(prompt)
    if not prompt_nums:
        return 1.0
    specific_nums = {n for n in prompt_nums if n not in ("0", "1", "2")}
    if not specific_nums:
        return 1.0
    found = sum(1 for n in specific_nums if n in text)
    return found / len(specific_nums)


def _score_dqdt(text: str, meta: dict) -> float:
    """q_correct_dqdt: Hamilton's first equation matches ground truth."""
    expected = meta.get("dqdt", "")
    if not expected or expected == "none":
        return 0.0

    expected_parts = [e.strip() for e in expected.split(";")]
    extracted = _extract_equations_block(text)

    if not extracted:
        if re.search(r"dq/dt|∂H/∂p|dx/dt|dtheta/dt|dr/dt|ds/dt", text):
            return 0.2
        return 0.0

    scores = []
    for exp_part in expected_parts:
        best = 0.0
        for ext in extracted:
            score = _score_expression(ext, exp_part, [])
            best = max(best, score)
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0


def _score_dpdt(text: str, meta: dict) -> float:
    """q_correct_dpdt: Hamilton's second equation matches ground truth."""
    expected = meta.get("dpdt", "")
    if not expected or expected == "none":
        return 0.0

    expected_parts = [e.strip() for e in expected.split(";")]
    extracted = _extract_equations_block(text)

    if not extracted:
        if re.search(r"dp/dt|-∂H/∂q|-dH/dq|dp_r/dt|dp_theta/dt", text):
            return 0.2
        return 0.0

    scores = []
    for exp_part in expected_parts:
        best = 0.0
        for ext in extracted:
            score = _score_expression(ext, exp_part, [])
            best = max(best, score)
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0


def _score_correct_H(text: str, meta: dict) -> float:
    """q_correct_H: HAMILTONIAN section matches ground truth.

    Differentiated scoring so the model gets distinct signals:
    - 1.0: correct H in momentum form
    - 0.5-0.7: correct structure but velocity form (ẋ instead of p)
    - 0.3-0.5: partially correct (some terms right)
    - 0.2: attempted but unparseable
    - 0.1: wrote a number instead of symbolic expression
    - 0.0: nothing found
    """
    expected_H = meta.get("H_expr", "")
    if not expected_H or expected_H == "none":
        return 0.0

    extracted_H = _extract_H(text)
    if extracted_H is None:
        return 0.0

    # Check if model evaluated to a number instead of keeping symbolic
    normed = _normalize_for_sympy(extracted_H)
    parsed = _try_sympify(extracted_H)
    if parsed is not None and isinstance(parsed, sp.Basic) and parsed.is_number:
        return 0.1  # Distinct signal: "you plugged in numbers, keep it symbolic"

    # Try direct match first
    direct_score = _score_expression(extracted_H, expected_H, [])
    if direct_score >= 0.7:
        return direct_score

    # Check if it's velocity form of the correct H — substitute _VDOT_ → p/m
    # If H has _VDOT_ terms, try replacing with p to see if structure matches
    if "_VDOT_" in normed:
        # The model wrote the right structure but in velocity form
        # Give partial credit: 0.5 for correct structure, wrong variables
        # Try to check if V term (non-velocity part) is correct
        expected_sym = _try_sympify(expected_H)
        if expected_sym is not None:
            # Extract just the potential part from expected (terms without p)
            p_sym = sp.Symbol("p")
            expected_V_terms = expected_sym.subs(p_sym, 0)
            # Check if the model's non-VDOT terms match expected V terms
            vdot_sym = sp.Symbol("_VDOT_")
            if parsed is not None:
                student_V_terms = parsed.subs(vdot_sym, 0)
                try:
                    if sp.simplify(student_V_terms - expected_V_terms) == 0:
                        return 0.6  # V correct, T in wrong form
                except Exception:
                    pass
        return 0.4  # Has velocity form — recognizably an H, just wrong variables

    return direct_score


def _score_consistency(text: str, meta: dict) -> float:
    """q_consistency: stated H's derivatives match stated equations."""
    extracted_H_str = _extract_H(text)
    if extracted_H_str is None:
        return 0.0

    H_sym = _try_sympify(extracted_H_str)
    if H_sym is None or not isinstance(H_sym, sp.Expr):
        return 0.2

    coords = meta.get("coordinates", "x")
    coord_list = [c.strip() for c in coords.split(",")]

    extracted_eqs = _extract_equations_block(text)
    if not extracted_eqs:
        return 0.3

    consistency_scores = []
    for coord in coord_list:
        p_name = f"p_{coord}" if coord not in ("x", "y", "s", "q") else "p"
        if coord in ("x1", "x2"):
            p_name = coord.replace("x", "p")

        p_sym = _SYMPY_LOCALS.get(p_name) or sp.Symbol(p_name)
        q_sym = _SYMPY_LOCALS.get(coord) or sp.Symbol(coord)

        try:
            expected_dqdt = sp.diff(H_sym, p_sym)
            expected_dpdt = -sp.diff(H_sym, q_sym)

            for eq_str in extracted_eqs:
                eq_sym = _try_sympify(eq_str)
                if eq_sym is not None:
                    try:
                        if sp.simplify(eq_sym - expected_dqdt) == 0:
                            consistency_scores.append(1.0)
                            break
                        if sp.simplify(eq_sym - expected_dpdt) == 0:
                            consistency_scores.append(1.0)
                            break
                    except Exception:
                        pass
            else:
                consistency_scores.append(0.3)
        except Exception:
            consistency_scores.append(0.2)

    return sum(consistency_scores) / max(len(consistency_scores), 1)


# ─── Main reward function ─────────────────────────────────────────────────────

def hamiltonian_reward(
    prompt: str,
    completion: str,
    metadata: dict | None = None,
) -> RewardResult:
    """Score a Hamiltonian derivation with granular per-section qualities.

    Phase 1: q_format, q_has_math — structured output with labeled sections
    Phase 2: q_momentum_defined, q_T_uses_p, q_V_correct — momentum form + physics
    Phase 3: q_correct_dqdt, q_correct_dpdt — Hamilton's equations match ground truth
    Phase 4: q_correct_H, q_consistency — full Hamiltonian + internal consistency
    """
    meta = metadata or {}
    text = completion
    scores: dict[str, float] = {}

    # ── Phase 1: Format ──
    scores["q_format"] = _score_format(text)
    scores["q_has_math"] = _score_has_math(text)

    # ── Phase 2: Physics — granular per-section ──
    scores["q_momentum_defined"] = _score_momentum_defined(text)
    scores["q_T_uses_p"] = _score_T_uses_p(text, meta)
    scores["q_V_correct"] = _score_V_correct(text, meta)

    # ── Phase 3: Equation correctness ──
    scores["q_correct_dqdt"] = _score_dqdt(text, meta)
    scores["q_correct_dpdt"] = _score_dpdt(text, meta)

    # ── Phase 4: Full Hamiltonian + consistency ──
    scores["q_correct_H"] = _score_correct_H(text, meta)
    scores["q_consistency"] = _score_consistency(text, meta)

    total = sum(scores.values()) / max(len(scores), 1)

    # ── Diagnostic logging ──
    try:
        _DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        diag = {
            "system": meta.get("system", "unknown"),
            "difficulty": meta.get("difficulty", "unknown"),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "total": round(total, 3),
            "completion_len": len(text),
        }
        with open(_DIAG_PATH, "a") as f:
            f.write(json.dumps(diag) + "\n")
    except Exception:
        pass

    return RewardResult(reward=total, scores=scores)
