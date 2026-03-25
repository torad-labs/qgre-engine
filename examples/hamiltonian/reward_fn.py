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
    # Strip trailing incomplete LaTeX/markdown: **, $, \, etc.
    s = re.sub(r"[\*$\\]+$", "", s).strip()
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
    # Insert * only if preceded by a letter/digit (not by * or operator)
    s = re.sub(r"(?<=[a-zA-Z0-9)])\\(sin|cos|tan|exp|log|ln)\b", r"*\1", s)
    s = re.sub(r"\\(sin|cos|tan|exp|log|ln)\b", r"\1", s)
    # Degree symbols: handle ALL forms BEFORE ^ → ** conversion
    # LaTeX: 60^\circ, 60^{\circ}, 60°
    s = re.sub(r"(\d+)\^\\circ", lambda m: {"30": "pi/6", "45": "pi/4", "60": "pi/3", "90": "pi/2"}.get(m.group(1), m.group(1)), s)
    s = re.sub(r"(\d+)\^\{\\circ\}", lambda m: {"30": "pi/6", "45": "pi/4", "60": "pi/3", "90": "pi/2"}.get(m.group(1), m.group(1)), s)
    s = re.sub(r"(\d+)°", lambda m: {"30": "pi/6", "45": "pi/4", "60": "pi/3", "90": "pi/2"}.get(m.group(1), m.group(1)), s)
    # Bare sin(60), sin(30) etc without degree symbol — assume degrees for common physics angles
    s = re.sub(r"sin\((\d+)\)", lambda m: {"30": "sin(pi/6)", "45": "sin(pi/4)", "60": "sin(pi/3)", "90": "sin(pi/2)"}.get(m.group(1), f"sin({m.group(1)})"), s)
    s = re.sub(r"cos\((\d+)\)", lambda m: {"30": "cos(pi/6)", "45": "cos(pi/4)", "60": "cos(pi/3)", "90": "cos(pi/2)"}.get(m.group(1), f"cos({m.group(1)})"), s)
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
    # uppercase * lowercase: Fx → F*x, Ky → K*y (constant × coordinate)
    s = re.sub(r"([A-Z])([a-z])", r"\1*\2", s)
    # Consecutive single lowercase letters: mgy → m*g*y, mgh → m*g*h
    # Only split known physics variable sequences (avoid breaking 'sin', 'cos', 'theta', etc)
    # Strategy: if a sequence of 2+ lowercase letters is NOT a known name, split it
    _KNOWN_NAMES = {"sin", "cos", "tan", "exp", "log", "sqrt", "ln", "pi",
                    "theta", "omega", "alpha", "beta", "gamma", "delta",
                    "theta1", "theta2", "p_theta", "p_r", "p_s", "p_x", "p_y"}
    def _split_vars(match):
        word = match.group(0)
        if word.lower() in _KNOWN_NAMES:
            return word
        # Split into single chars with * between: mgy → m*g*y
        return "*".join(word)
    s = re.sub(r"[a-z]{2,}", _split_vars, s)
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


def _remap_variables(student: sp.Basic, teacher: sp.Basic) -> sp.Basic:
    """Remap student variables to match teacher's coordinate names.

    The model may use x where ground truth uses y, or s where it uses x.
    Both are valid — the coordinate name is arbitrary. This function finds
    a consistent mapping from student symbols to teacher symbols.

    Only remaps single-letter coordinate variables (x, y, s, r, q, theta).
    Does NOT remap p (momentum) or constants (m, g, k).
    """
    _COORDS = {"x", "y", "s", "r", "q", "theta", "theta1", "theta2",
                "x1", "x2", "p_x", "p_y", "p_r", "p_s", "p_theta"}
    _MOMENTA = {"p", "p_x", "p_y", "p_r", "p_s", "p_theta", "p1", "p2"}

    student_syms = {s for s in student.free_symbols if s.name in _COORDS}
    teacher_syms = {s for s in teacher.free_symbols if s.name in _COORDS}

    # If symbols already match, no remapping needed
    if student_syms & teacher_syms == teacher_syms:
        return student

    # Simple case: one coordinate variable each — direct map
    student_coords = student_syms - {s for s in student.free_symbols if s.name in _MOMENTA}
    teacher_coords = teacher_syms - {s for s in teacher.free_symbols if s.name in _MOMENTA}

    if len(student_coords) == 1 and len(teacher_coords) == 1:
        s_var = next(iter(student_coords))
        t_var = next(iter(teacher_coords))
        if s_var != t_var:
            return student.subs(s_var, t_var)

    # Multi-variable: try positional mapping (sorted by name)
    if len(student_coords) == len(teacher_coords) and len(student_coords) > 0:
        s_sorted = sorted(student_coords, key=lambda s: s.name)
        t_sorted = sorted(teacher_coords, key=lambda s: s.name)
        subs = {s: t for s, t in zip(s_sorted, t_sorted) if s != t}
        if subs:
            return student.subs(subs)

    return student  # Can't remap — return as-is


def _score_expression(student_str: str | None, teacher_str: str, variables: list[str]) -> float:
    """Score a mathematical expression against ground truth.

    Returns 0.0-1.0:
    - 1.0: exact symbolic or numerical match (including after variable remapping)
    - 0.8: correct up to sign (V vs -V — valid physics, different reference frame)
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

    # Variable remapping: student may use x where teacher uses y (both valid)
    student_remapped = _remap_variables(student, teacher)

    # Try both original and remapped student against teacher
    candidates = [student_remapped]
    if student_remapped != student:
        candidates.append(student)

    for candidate in candidates:
        # Exact symbolic match — try multiple simplification strategies
        for simplifier in [sp.simplify, sp.trigsimp, sp.ratsimp, sp.nsimplify]:
            try:
                if simplifier(candidate - teacher) == 0:
                    return 1.0
            except Exception:
                continue

        try:
            if sp.simplify(sp.expand(candidate) - sp.expand(teacher)) == 0:
                return 1.0
        except Exception:
            pass

        try:
            if sp.trigsimp(sp.expand_trig(candidate - teacher)) == 0:
                return 1.0
        except Exception:
            pass

        # Sign convention check: student = -teacher is valid physics (different reference frame)
        for simplifier in [sp.simplify, sp.expand]:
            try:
                if simplifier(candidate + teacher) == 0:
                    return 0.8  # Correct magnitude, opposite sign — partial credit
            except Exception:
                continue

    # Numerical equivalence check at two points (use remapped candidate)
    candidate = candidates[0]
    try:
        free = (candidate.free_symbols | teacher.free_symbols) - {sp.Symbol('pi')}
        if free:
            _probes = [sp.Rational(3, 7), sp.Rational(5, 11), sp.Rational(7, 13),
                       sp.Rational(11, 17), sp.Rational(13, 19), sp.Rational(17, 23)]
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
                return 0.8  # Numerical sign-flip match
        else:
            s_val = float(candidate)
            t_val = float(teacher)
            if abs(s_val - t_val) < 1e-8 * max(abs(t_val), 1):
                return 1.0
            if abs(s_val + t_val) < 1e-8 * max(abs(t_val), 1):
                return 0.8
    except Exception:
        pass

    # Partial credit based on numerical closeness
    try:
        free = (candidate.free_symbols | teacher.free_symbols) - {sp.Symbol('pi')}
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
            except Exception:
                pass

        student_syms = candidate.free_symbols
        teacher_syms = teacher.free_symbols
        if not teacher_syms:
            sym_overlap = 0.0 if student_syms else numerical_score
        else:
            sym_overlap = len(student_syms & teacher_syms) / len(teacher_syms)

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

_LABEL_ALIASES = {
    "COORDINATES": ["coordinates", "coordinate", "generalized coordinate"],
    "MOMENTUM": ["momentum", "conjugate momentum"],
    "KINETIC": ["kinetic", "kinetic energy"],
    "POTENTIAL": ["potential", "potential energy"],
    "HAMILTONIAN": ["hamiltonian"],
    "EQUATIONS": ["equations", "equations of motion", "hamilton's equations"],
}


def _extract_labeled(text: str, label: str) -> str | None:
    """Extract the value after a labeled line like 'KINETIC: T = ...'

    Takes the LAST match — the model writes derivation headers early
    (### 2. **Kinetic Energy**) and structured labels at the end
    (**KINETIC:** T = p²/6). We want the final labeled answer.

    Also matches bold markdown headers: **Kinetic Energy:** T = ...
    """
    # Build alias patterns for this label
    aliases = _LABEL_ALIASES.get(label.upper(), [label.lower()])
    all_labels = [label] + aliases

    patterns = []
    for lbl in all_labels:
        esc = re.escape(lbl)
        patterns.extend([
            rf"\*{{0,3}}{esc}\*{{0,3}}[:\s]+[A-Za-z_]*\s*=\s*([^\n]+)",  # **KINETIC:** T = ... or KINETIC: T = ...
            rf"\*{{0,3}}{esc}\*{{0,3}}[:\s]+\$?\s*([^\n]+)",              # **Kinetic Energy:** $ T = p²/6 $
        ])

    for pat in patterns:
        matches = list(re.finditer(pat, text, re.IGNORECASE))
        if matches:
            expr = matches[-1].group(1).strip()  # Last match
            # Strip trailing LaTeX/markdown artifacts: **, $, \
            expr = re.sub(r"[\*$\\]+$", "", expr).strip()
            # Strip leading $ from LaTeX inline
            expr = re.sub(r"^\$\s*", "", expr).strip()
            # Take the last = if there are multiple (e.g. "T = p²/(2m) = p²/6")
            # But only if the part after = looks like a math expression, not prose
            parts = expr.rsplit("=", 1)
            if len(parts) > 1:
                rhs = parts[-1].strip()
                # If RHS looks like prose (contains common words), use the full expr instead
                if re.search(r"\b(the|was|is|from|and|for|with|where|since|given)\b", rhs, re.IGNORECASE):
                    return expr  # Full expression, not the prose after last =
                return rhs
            return expr
    return None


# ─── Section-agnostic expression finder ──────────────────────────────────────

def _find_all_expressions(text: str) -> dict[str, list[str]]:
    """Extract ALL 'X = expr' patterns from the text, grouped by variable name.

    Returns dict mapping variable letter (T, V, H, p, q, etc.) to list of
    expression strings found. Takes the RAW text from each match — no label
    matching, no section detection. Finds math wherever it appears.
    """
    results: dict[str, list[str]] = {}

    # Strip LaTeX display delimiters so we can find expressions inside $$ blocks
    cleaned = text
    cleaned = re.sub(r"\$\$\s*", " ", cleaned)
    cleaned = re.sub(r"\$", " ", cleaned)

    # Pattern: letter(s) = expression (on same line or after = )
    for m in re.finditer(r'([A-Za-z_][A-Za-z_0-9]*)\s*=\s*([^\n,;]{3,})', cleaned):
        var = m.group(1).strip()
        expr = m.group(2).strip()
        # Strip trailing markdown/LaTeX artifacts
        expr = re.sub(r"[\*$\\]+$", "", expr).strip()
        # Skip prose (has common English words)
        if re.search(r"\b(the|was|is|from|and|for|with|where|since|given|that|this)\b", expr, re.IGNORECASE):
            continue
        # Skip very short or empty
        if len(expr) < 1:
            continue
        results.setdefault(var, []).append(expr)

    # Also find dX/dt = expr patterns (for Hamilton's equations)
    for m in re.finditer(r'd([a-z_]+)/dt\s*=\s*([^\n,;]{2,})', cleaned):
        var = f"d{m.group(1)}/dt"
        expr = m.group(2).strip()
        expr = re.sub(r"[\*$\\]+$", "", expr).strip()
        if len(expr) < 1:
            continue
        results.setdefault(var, []).append(expr)

    # LaTeX fraction derivatives: \frac{dq}{dt} = expr
    for m in re.finditer(r'\\frac\{d([a-z_]+)\}\{dt\}\s*=\s*([^\n,;]{2,})', text):
        var = f"d{m.group(1)}/dt"
        expr = m.group(2).strip()
        expr = re.sub(r"[\*$\\]+$", "", expr).strip()
        results.setdefault(var, []).append(expr)

    return results


def _best_match(candidates: list[str], teacher_str: str, variables: list[str] | None = None) -> float:
    """Score the best-matching candidate expression against ground truth.

    Tries each candidate, returns the highest score. This is the core of
    section-agnostic scoring — instead of requiring the expression to be
    in a specific labeled section, we find it anywhere in the text.
    """
    if not candidates:
        return 0.0
    best = 0.0
    for expr in candidates:
        score = _score_expression(expr, teacher_str, variables or [])
        if score > best:
            best = score
        if best >= 1.0:
            break
    return best


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
        aliases = _LABEL_ALIASES.get(label, [label.lower()])
        for lbl in [label] + aliases:
            if re.search(rf"\*{{0,3}}{re.escape(lbl)}\*{{0,3}}\s*:", text, re.IGNORECASE):
                labels_found += 1
                break

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
        has_expression = bool(re.search(r"[a-z\u0370-\u03FF\u0300-\u036F*/+\-]", momentum_str))
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
    """q_T_uses_p: T is expressed in terms of momentum p, matching ground truth.

    Section-agnostic: finds ALL 'T = expr' anywhere in the text, checks if
    any contain p and match the ground truth. No label required.
    """
    expected_T = meta.get("T_expr", "")
    if not expected_T:
        return 0.0

    # Gather all T expressions from the text
    all_exprs = _find_all_expressions(text)
    t_candidates = all_exprs.get("T", [])

    # Filter to candidates that contain p (the whole point)
    p_candidates = [e for e in t_candidates if re.search(r"p[_²^2\s/(*]|p\*\*2|p\^2|\bp\b", e)]

    if p_candidates:
        score = _best_match(p_candidates, expected_T)
        return max(0.7, score)  # At least 0.7 for having p form

    # No T = p... found. Check if ANY expression in the text matches ground truth T
    # (model might write it as part of H derivation without a separate T = line)
    if t_candidates:
        # T expressions exist but none have p — velocity form
        return 0.0

    return 0.0  # No T expression found at all


def _score_T_in_momentum(text: str) -> float:
    """Check if kinetic energy T is expressed in momentum form (p²) vs velocity form.

    Section-agnostic: finds ALL 'T = expr' anywhere in the text.
    Returns 1.0 if ANY T expression contains p without velocity markers.
    """
    try:
        all_exprs = _find_all_expressions(text)
        t_candidates = all_exprs.get("T", [])
        if not t_candidates:
            return 0.0

        for expr in t_candidates:
            has_p = bool(re.search(r"p[_²/(*]|p\*\*|p\^|p_[a-z]|\bp\d|\bp\b", expr))
            has_velocity = bool(re.search(
                r"[ẋẏṙ]|\\dot|_VDOT_|\u0307|d[a-zθ]/dt|\\frac\{d[a-zθ]", expr
            ))
            if has_p and not has_velocity:
                return 1.0
            if has_p and has_velocity:
                return 0.5

        return 0.0  # No T expression with p found
    except Exception:
        return 0.0


def _score_H_in_momentum(text: str) -> float:
    """Check if Hamiltonian H is expressed using momentum variable p.

    Section-agnostic: finds ALL 'H = expr' anywhere in the text.
    Returns 1.0 if ANY H expression contains p without velocity markers.
    """
    try:
        all_exprs = _find_all_expressions(text)
        h_candidates = all_exprs.get("H", [])
        if not h_candidates:
            return 0.0

        for expr in h_candidates:
            has_p = bool(re.search(r"p[_²/(*+\-]|p\*\*|p\^|p_[a-z]|\bp\d|\bp\b", expr))
            has_velocity = bool(re.search(
                r"[ẋẏṙ]|\\dot|_VDOT_|\u0307|d[a-zθ]/dt|\\frac\{d[a-zθ]", expr
            ))
            if has_p and not has_velocity:
                return 1.0
            if has_p and has_velocity:
                return 0.5

        return 0.0
    except Exception:
        return 0.0


def _score_V_correct(text: str, meta: dict) -> float:
    """q_V_correct: potential energy matches ground truth V.

    Section-agnostic: finds ALL 'V = expr' anywhere in the text.
    """
    expected_V = meta.get("V_expr", "")
    if not expected_V or expected_V == "none":
        return 0.0

    # Section-agnostic: find all V expressions
    all_exprs = _find_all_expressions(text)
    v_candidates = all_exprs.get("V", [])

    # Also try label-based extraction as fallback
    labeled_v = _extract_labeled(text, "POTENTIAL")
    if labeled_v and labeled_v not in v_candidates:
        v_candidates.append(labeled_v)

    if not v_candidates:
        return 0.0

    # Score best candidate
    score = _best_match(v_candidates, expected_V)
    if score >= 0.9:
        return score

    # Fallback: model may write symbolic form (mgy, Fx, kx²) while ground truth
    # has constants evaluated (49*y/5, -3*x, 3*x**2). Try substituting known
    # constants from the problem into the student expression via sympy.
    # Use the best-scoring candidate from section-agnostic extraction.
    potential_str = v_candidates[0]  # Best candidate for detailed analysis
    student_sym = _try_sympify(potential_str)
    teacher_sym = _try_sympify(expected_V)
    if student_sym is not None and teacher_sym is not None:
        # Build substitution dict from problem metadata
        subs = {}
        # Extract mass from prompt
        prompt = meta.get("prompt", "")
        m_match = re.search(r"mass\s*(?:m\s*)?=?\s*(\d+(?:\.\d+)?)", prompt)
        if m_match:
            subs[sp.Symbol("m", positive=True)] = sp.nsimplify(m_match.group(1))
            subs[sp.Symbol("m")] = sp.nsimplify(m_match.group(1))
        # Extract spring constant
        k_match = re.search(r"(?:spring\s+)?constant\s*k?\s*=\s*(\d+(?:\.\d+)?)", prompt)
        if k_match:
            subs[sp.Symbol("k", positive=True)] = sp.nsimplify(k_match.group(1))
            subs[sp.Symbol("k")] = sp.nsimplify(k_match.group(1))
        # Extract force
        f_match = re.search(r"force\s*(?:F\s*)?=?\s*(\d+(?:\.\d+)?)", prompt)
        if f_match:
            subs[sp.Symbol("F", positive=True)] = sp.nsimplify(f_match.group(1))
            subs[sp.Symbol("F")] = sp.nsimplify(f_match.group(1))

        if subs:
            try:
                student_eval = student_sym.subs(subs)
                diff = sp.simplify(student_eval - teacher_sym)
                if diff == 0:
                    return 1.0
                # Try numerical check after substitution
                free = (student_eval.free_symbols | teacher_sym.free_symbols) - {sp.Symbol("pi")}
                if free:
                    test_point = {s: sp.Rational(3, 7) for s in free}
                    s_val = float(student_eval.subs(test_point))
                    t_val = float(teacher_sym.subs(test_point))
                    if abs(t_val) > 1e-10 and abs(s_val - t_val) < 1e-4 * max(abs(t_val), 1):
                        return 1.0
            except Exception:
                pass

    return score


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
    """q_correct_dqdt: Hamilton's first equation matches ground truth.

    Section-agnostic: finds ALL 'dq/dt = expr' patterns anywhere in the text.
    """
    expected = meta.get("dqdt", "")
    if not expected or expected == "none":
        return 0.0

    expected_parts = [e.strip() for e in expected.split(";")]

    # Section-agnostic: find all derivative expressions
    all_exprs = _find_all_expressions(text)
    # Collect all dX/dt expressions (dq/dt, dx/dt, dy/dt, ds/dt, dtheta/dt)
    extracted = []
    for key, vals in all_exprs.items():
        if key.startswith("d") and "/dt" in key:
            extracted.extend(vals)
    # Also try old block extraction as fallback
    block_extracted = _extract_equations_block(text)
    for e in block_extracted:
        if e not in extracted:
            extracted.append(e)

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
    """q_correct_dpdt: Hamilton's second equation matches ground truth.

    Section-agnostic: finds ALL equation patterns anywhere in the text.
    """
    expected = meta.get("dpdt", "")
    if not expected or expected == "none":
        return 0.0

    expected_parts = [e.strip() for e in expected.split(";")]

    # Section-agnostic + block extraction
    all_exprs = _find_all_expressions(text)
    extracted = []
    for key, vals in all_exprs.items():
        if key.startswith("d") and "/dt" in key:
            extracted.extend(vals)
    block_extracted = _extract_equations_block(text)
    for e in block_extracted:
        if e not in extracted:
            extracted.append(e)

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
    """q_correct_H: Hamiltonian matches ground truth.

    Section-agnostic: finds ALL 'H = expr' anywhere in the text,
    scores the best match against ground truth.
    """
    expected_H = meta.get("H_expr", "")
    if not expected_H or expected_H == "none":
        return 0.0

    # Section-agnostic: find all H expressions
    all_exprs = _find_all_expressions(text)
    h_candidates = all_exprs.get("H", [])

    # Also try label-based extraction as fallback
    extracted_H = _extract_H(text)
    if extracted_H and extracted_H not in h_candidates:
        h_candidates.append(extracted_H)

    if not h_candidates:
        return 0.0

    # Score best candidate
    best = _best_match(h_candidates, expected_H)
    if best >= 0.7:
        return best

    # Use the first candidate for detailed analysis below
    extracted_H = h_candidates[0]

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


def _score_correct_coefficient(text: str, meta: dict) -> float:
    """Quality scorer: checks numerical coefficients in H and equations against ground truth.

    Targets the failure where the model writes correct structure but wrong coefficients
    (e.g., dp/dt = -3x instead of -6x).

    Returns:
        1.0 — all coefficients in H and equations match ground truth
        0.7 — H coefficients correct but equation coefficients wrong (or vice versa)
        0.4 — some coefficients correct
        0.2 — expressions found but unparseable or all wrong
        0.0 — no H_expr/dqdt/dpdt in meta, or nothing extractable
    """
    try:
        has_H_expr = bool(meta.get("H_expr", ""))
        has_dqdt = bool(meta.get("dqdt", ""))
        has_dpdt = bool(meta.get("dpdt", ""))

        if not has_H_expr and not has_dqdt and not has_dpdt:
            return 0.0

        component_scores: list[float] = []
        any_extracted = False

        # ── Score H ──────────────────────────────────────────────────────────
        h_score: float | None = None
        if has_H_expr:
            extracted_H = _extract_H(text)
            if extracted_H is not None:
                any_extracted = True
                h_score = _score_expression(extracted_H, meta["H_expr"], [])
                component_scores.append(h_score)

        # ── Score equations ──────────────────────────────────────────────────
        eq_score: float | None = None
        if has_dqdt or has_dpdt:
            extracted_eqs = _extract_equations_block(text)
            eq_component_scores: list[float] = []

            if extracted_eqs:
                any_extracted = True

                if has_dqdt:
                    expected_parts = [e.strip() for e in meta["dqdt"].split(";") if e.strip()]
                    part_scores = []
                    for exp_part in expected_parts:
                        best = max(
                            (_score_expression(ext, exp_part, []) for ext in extracted_eqs),
                            default=0.0,
                        )
                        part_scores.append(best)
                    if part_scores:
                        eq_component_scores.append(sum(part_scores) / len(part_scores))

                if has_dpdt:
                    expected_parts = [e.strip() for e in meta["dpdt"].split(";") if e.strip()]
                    part_scores = []
                    for exp_part in expected_parts:
                        best = max(
                            (_score_expression(ext, exp_part, []) for ext in extracted_eqs),
                            default=0.0,
                        )
                        part_scores.append(best)
                    if part_scores:
                        eq_component_scores.append(sum(part_scores) / len(part_scores))

            if eq_component_scores:
                eq_score = sum(eq_component_scores) / len(eq_component_scores)
                component_scores.append(eq_score)

        if not any_extracted:
            return 0.0

        if not component_scores:
            return 0.2

        avg = sum(component_scores) / len(component_scores)

        # Tier assignment:
        # 1.0 requires ALL components to be >= 0.9 (all coefficients correct)
        if all(s >= 0.9 for s in component_scores):
            return 1.0

        # 0.7 tier: one side clearly correct, the other wrong (or vice versa)
        if h_score is not None and eq_score is not None:
            if (h_score >= 0.85 and eq_score < 0.6) or (eq_score >= 0.85 and h_score < 0.6):
                return 0.7

        # General average-based tiers
        if avg >= 0.75:
            return 0.7
        if avg >= 0.35:
            return 0.4
        return max(0.2, avg)
    except Exception:
        return 0.0


def _score_derivative_correct(text: str, meta: dict) -> float:
    """Quality scorer: checks if dp/dt derivative is correct, with a DISTINCT signal (0.5)
    for factor-of-2 errors.

    Targets the observed failure where ∂(ax²)/∂x = ax instead of 2ax.

    Returns:
        1.0 — dp/dt matches ground truth exactly (sympy or numerical)
        0.5 — dp/dt is exactly half or double of ground truth (factor-of-2 error)
        0.2 — dp/dt found but doesn't match in any way
        0.0 — no dp/dt found or no dpdt in meta
    """
    try:
        expected = meta.get("dpdt", "")
        if not expected or expected == "none":
            return 0.0

        expected_parts = [e.strip() for e in expected.split(";") if e.strip()]

        extracted = _extract_equations_block(text)

        if not extracted:
            # Check if dp/dt pattern exists anywhere for minimal partial credit
            if re.search(r"dp/dt|dp_[a-z]+/dt|-∂H/∂[a-z]|-dH/d[a-z]", text):
                return 0.1
            return 0.0

        # Find best matching equation across all expected parts
        best_score = 0.0
        best_student_str: str | None = None
        best_teacher_str: str | None = None

        for exp_part in expected_parts:
            for ext in extracted:
                score = _score_expression(ext, exp_part, [])
                if score > best_score:
                    best_score = score
                    best_student_str = ext
                    best_teacher_str = exp_part

        # Exact match
        if best_score >= 0.9:
            return 1.0

        # Factor-of-2 detection — the key differentiator for missing chain-rule coefficient
        if best_student_str is not None and best_teacher_str is not None:
            student_sym = _try_sympify(best_student_str)
            teacher_sym = _try_sympify(best_teacher_str)

            if student_sym is not None and teacher_sym is not None:
                try:
                    free = (student_sym.free_symbols | teacher_sym.free_symbols) - {sp.Symbol("pi")}
                    test_point = {s: sp.Rational(3, 7) for s in free}
                    student_val = float(student_sym.subs(test_point))
                    teacher_val = float(teacher_sym.subs(test_point))

                    if abs(teacher_val) > 1e-10:
                        ratio = student_val / teacher_val
                        if abs(ratio - 0.5) < 0.05:
                            return 0.5  # Student wrote half the derivative (forgot factor of 2)
                        if abs(ratio - 2.0) < 0.05:
                            return 0.5  # Student doubled the derivative
                except Exception:
                    pass

        # Partial credit capped at 0.4
        if best_score > 0.0:
            return min(best_score, 0.4)

        # dp/dt found in extracted equations but matched nothing
        return 0.2

    except Exception:
        return 0.0


def _score_defines_momentum(text: str) -> float:
    """Quality scorer: checks if completion explicitly defines momentum p via
    the Lagrangian partial derivative p = ∂L/∂q̇, not just incidentally mentions p.

    Returns:
        1.0 — explicit ∂L/∂q̇ notation found (LaTeX, Unicode, or text forms)
        0.7 — MOMENTUM section has a numeric expression for p but no ∂L notation
        0.3 — p = appears somewhere without a MOMENTUM label
        0.0 — no p = found anywhere
    """
    if not text:
        return 0.0

    # ── 1. Check for explicit partial derivative of L w.r.t. generalized velocity ──
    _PARTIAL_PATTERNS = [
        # LaTeX: \frac{\partial L}{\partial \dot{q}} or \partial L / \partial \dot{q}
        r"\\frac\s*\{\\partial\s*L\}\s*\{\\partial\s*\\dot",
        r"\\partial\s*L\s*/\s*\\partial\s*\\dot",
        r"\\partial\s*L\s*/\s*\\partial",
        # Unicode: ∂L/∂q̇  ∂L/∂θ̇  ∂L/∂ẋ  ∂L/∂ṙ
        r"∂L\s*/\s*∂",
        r"∂\s*L\s*/\s*∂",
        # Text forms: "partial L / partial", "partial L/partial", "dL/d"
        r"partial\s+L\s*/\s*partial",
        r"dL\s*/\s*d",
    ]
    try:
        for pat in _PARTIAL_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                return 1.0
    except Exception:
        pass

    # ── 2. MOMENTUM section with numeric expression but no ∂L ──
    try:
        momentum_str = _extract_labeled(text, "MOMENTUM")
        if momentum_str is not None and momentum_str.strip():
            has_numbers = bool(re.search(r"\d", momentum_str))
            has_p_eq = bool(re.search(r"p\s*=", text, re.IGNORECASE))
            if has_numbers and has_p_eq:
                return 0.7
    except Exception:
        pass

    # ── 3. p = appears anywhere, no MOMENTUM label ──
    try:
        if re.search(r"p\s*=", text, re.IGNORECASE):
            return 0.3
    except Exception:
        pass

    return 0.0


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
    meta = dict(metadata) if metadata else {}  # Shallow copy — don't mutate shared dataloader dict
    meta["prompt"] = prompt  # Make prompt available to scoring functions (e.g., V_correct constant substitution)
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

    # ── Phase 5: Granular momentum/variable/derivative checks ──
    scores["q_defines_momentum"] = _score_defines_momentum(text)
    scores["q_T_in_momentum"] = _score_T_in_momentum(text)
    scores["q_H_in_momentum"] = _score_H_in_momentum(text)
    scores["q_correct_coefficient"] = _score_correct_coefficient(text, meta)
    scores["q_derivative_correct"] = _score_derivative_correct(text, meta)

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
