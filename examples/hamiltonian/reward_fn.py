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

import copy
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import sympy as sp

from qgre.expression import (
    _BASE_SYMPY_LOCALS,
    extract_constants_from_prompt,
    normalize_for_sympy,
    parse_math,
    score_expression,
    try_sympify,
)

# Imports from qgre modules (extracted code)
from qgre.reward_parsing import (
    StructuredOutputParser,
    parse_structured_output,
)
from qgre.types import RewardResult


logger = logging.getLogger("qgre.hamiltonian_reward")

# ─── Hamiltonian-specific symbol table ───────────────────────────────────────

HAMILTONIAN_LOCALS = {
    **_BASE_SYMPY_LOCALS,
    "x": sp.Symbol("x"),
    "y": sp.Symbol("y"),
    "r": sp.Symbol("r", positive=True),
    "s": sp.Symbol("s"),
    "q": sp.Symbol("q"),
    "p": sp.Symbol("p"),
    "p_x": sp.Symbol("p_x"),
    "p_y": sp.Symbol("p_y"),
    "p_r": sp.Symbol("p_r", positive=True),
    "p_s": sp.Symbol("p_s"),
    "p_theta": sp.Symbol("p_theta"),
    "p1": sp.Symbol("p1"),
    "p2": sp.Symbol("p2"),
    "theta": sp.Symbol("theta"),
    "theta1": sp.Symbol("theta1"),
    "theta2": sp.Symbol("theta2"),
    "x1": sp.Symbol("x1"),
    "x2": sp.Symbol("x2"),
    "m": sp.Symbol("m", positive=True),
    "k": sp.Symbol("k", positive=True),
    "F": sp.Symbol("F"),
    "l": sp.Symbol("l", positive=True),
    "g": sp.Rational(98, 10),
}

_DIAG_PATH = Path("output/hamiltonian/diagnostics.jsonl")


class ScorerResult(NamedTuple):
    """Score + the character spans the scorer actually evaluated.

    Returned by migrated scorers so the span and the score come from the
    same data path. Replaces the old pattern where _find_expression_spans
    independently located text regions for each quality.

    Using NamedTuple (not dataclass) because:
      - zero allocation overhead in the reward hot path
      - tuple unpacking works naturally: score, spans = _score_X(pc, meta)
      - field names prevent index-confusion bugs
    """

    score: float
    spans: list[tuple[int, int]]


@dataclass(frozen=True)
class ParsedCompletion:
    """Pre-computed parser results for a single completion.

    Created ONCE per hamiltonian_reward() call and threaded to all scorers.
    The StructuredOutputParser has internal caches, but those are per-instance.
    Previously, each scorer created its own instance (7+ per completion),
    defeating the cache entirely. This dataclass ensures ONE parse pass.

    `expression_spans` is the unified data path — each expression is paired
    with its char position. Scorers that consume it get span alignment for
    free, eliminating divergence with the legacy _find_expression_spans.
    """

    text: str
    parser: StructuredOutputParser
    all_expressions: dict[str, list[str]]
    equations: list[str]
    section_spans: dict[str, list[tuple[int, int]]]
    expression_spans: dict[str, list[tuple[str, int, int]]]

    @staticmethod
    def from_text(text: str) -> ParsedCompletion:
        parser = parse_structured_output(text)
        return ParsedCompletion(
            text=text,
            parser=parser,
            all_expressions=parser.get_all_expressions(),
            equations=parser.get_equations(),
            section_spans=parser.get_section_spans(),
            expression_spans=parser.get_all_expressions_with_spans(),
        )

    def get_labeled(self, label: str) -> str | None:
        return self.parser.get_labeled(label)

    def get_equations_rhs(self) -> list[str]:
        """Extract RHS from each equation (e.g., 'p/2' from 'dq/dt = p/2')."""
        rhs_list = []
        for eq in self.equations:
            if "=" in eq:
                rhs = eq.split("=", 1)[-1].strip()
                if rhs:
                    rhs_list.append(rhs)
            else:
                rhs_list.append(eq)
        return rhs_list


# ─── Quality scorers ──────────────────────────────────────────────────────────


def _labeled_section_spans(pc: ParsedCompletion) -> list[tuple[int, int]]:
    """Union of all labeled-section spans, sorted by start position.

    Used by format-style scorers that target "the labeled output region".
    Deduplicates identical spans.
    """
    all_spans: list[tuple[int, int]] = []
    for label in ["COORDINATES", "MOMENTUM", "KINETIC", "POTENTIAL", "HAMILTONIAN", "EQUATIONS"]:
        all_spans.extend(pc.section_spans.get(label, []))
    return sorted(set(all_spans), key=lambda x: x[0])


def _score_format(pc: ParsedCompletion) -> ScorerResult:
    """q_format: structured response with labeled sections.

    Span: labeled sections if present (reward the structured output region);
    otherwise the full completion (so negative signal can train away from bad format).
    """
    labels_found = 0
    for label in ["COORDINATES", "MOMENTUM", "KINETIC", "POTENTIAL", "HAMILTONIAN", "EQUATIONS"]:
        if pc.get_labeled(label) is not None:
            labels_found += 1

    labeled_spans = _labeled_section_spans(pc)
    full_span: list[tuple[int, int]] = [(0, len(pc.text))] if pc.text else []
    spans = labeled_spans if labeled_spans else full_span

    if labels_found >= 5:
        return ScorerResult(1.0, spans)
    if labels_found >= 3:
        return ScorerResult(0.7, spans)

    text = pc.text
    has_math = any(s in text for s in ["=", "H ", "T ", "V "])
    has_length = len(text.strip()) > 100
    if has_length and has_math:
        return ScorerResult(0.4, spans)
    if has_length:
        return ScorerResult(0.2, spans)
    return ScorerResult(0.0, [])


def _score_has_math(pc: ParsedCompletion) -> ScorerResult:
    """q_has_math: has mathematical content.

    Span: labeled sections if present, otherwise the full completion.
    """
    indicators = [
        "=",
        "**2",
        "^2",
        "²",
        "/2",
        "p^2",
        "p²",
        "p**2",
        "cos(",
        "sin(",
        "exp(",
        "sqrt(",
        "H =",
        "T =",
        "V =",
    ]
    text = pc.text
    count = sum(1 for p in indicators if p.lower() in text.lower())
    score = min(1.0, count / 3)

    labeled_spans = _labeled_section_spans(pc)
    full_span: list[tuple[int, int]] = [(0, len(pc.text))] if pc.text else []
    spans = labeled_spans if labeled_spans else full_span
    return ScorerResult(score, spans if score > 0 else [])


def _score_momentum_defined(pc: ParsedCompletion) -> ScorerResult:
    """q_momentum_defined: MOMENTUM section defines p in terms of q̇."""
    momentum_str = pc.get_labeled("MOMENTUM")

    if momentum_str:
        # Span: use MOMENTUM section span
        momentum_spans = pc.section_spans.get("MOMENTUM", [])
        spans = [momentum_spans[0]] if momentum_spans else []

        has_numbers = any(c.isdigit() for c in momentum_str)
        has_expression = any(c.isalpha() or c in "*/+-" for c in momentum_str)
        if has_numbers and has_expression:
            return ScorerResult(1.0, spans)
        if has_expression:
            return ScorerResult(0.7, spans)
        return ScorerResult(0.5, spans)

    # Fallback: p = ... expression anywhere in final output
    p_entries = pc.expression_spans.get("p", [])
    for expr, start, end in p_entries:
        lower = expr.lower()
        if "d" in lower or any(c.isdigit() for c in expr):
            return ScorerResult(0.5, [(start, end)])

    # Weak signal: "conjugate momentum" text present — no localizable span
    if "conjugate momentum" in pc.text.lower():
        return ScorerResult(0.3, [])

    return ScorerResult(0.0, [])


def _score_T_uses_p(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    """q_T_uses_p: T is expressed in terms of momentum p, matching ground truth.

    Uses pc.expression_spans so the winning T expression's span is returned.
    """
    expected_T = meta.get("T_expr", "")
    if not expected_T:
        return ScorerResult(0.0, [])

    # Position-aware T entries
    t_entries = pc.expression_spans.get("T", [])

    # Filter to entries whose expression contains p
    p_entries = [
        (e, s, end) for e, s, end in t_entries if re.search(r"p[_²^2\s/(*]|p\*\*2|p\^2|\bp\b", e)
    ]

    if p_entries:
        # Score best p-containing candidate, track its span
        best_score = 0.0
        best_span: tuple[int, int] | None = None
        for expr, s, end in p_entries:
            cs = score_expression(expr, expected_T, [], constant_subs=meta.get("_constant_subs"))
            if cs > best_score:
                best_score = cs
                best_span = (s, end)
            if best_score >= 1.0:
                break
        final = max(0.7, best_score)  # At least 0.7 for having p form
        spans = [best_span] if best_span else [(p_entries[0][1], p_entries[0][2])]
        return ScorerResult(final, spans)

    # No T = p... found — velocity form or no T at all
    return ScorerResult(0.0, [])


def _has_velocity_form(expr_str: str) -> bool:
    """Check if expression contains velocity markers (derivatives, dot notation).

    Fast regex check first (catches >95% of cases), then sympy parse only
    when regex says no — avoids expensive latex2sympy/sympify in the common case.
    """
    # Fast path: regex catches most velocity forms
    if re.search(r"[ẋẏṙ]|\\dot|_VDOT_|\u0307|d[a-zθ]/dt|\\frac\{d[a-zθ]", expr_str):
        return True
    # Slow path: parse only when regex misses (e.g. latex2sympy Derivative detection)
    if "\\" in expr_str:
        parsed = parse_math(expr_str, sympy_locals=HAMILTONIAN_LOCALS)
        if parsed is not None:
            if parsed.atoms(sp.Derivative):
                return True
            if any(s.name.startswith("dot{") for s in parsed.free_symbols):
                return True
    return False


def _has_momentum_var(expr_str: str) -> bool:
    """Check if expression contains momentum variable p."""
    return bool(re.search(r"p[_²/(*+\-]|p\*\*|p\^|p_[a-z]|\bp\d|\bp\b", expr_str))


def _score_in_momentum_form(pc: ParsedCompletion, var_name: str) -> ScorerResult:
    """Check if variable is expressed in momentum form (p) vs velocity form.

    Uses pc.expression_spans so the span points at the exact line that was
    evaluated. Returns 1.0 if ANY expression contains p without velocity markers.
    """
    try:
        entries = pc.expression_spans.get(var_name, [])
        if not entries:
            return ScorerResult(0.0, [])

        for expr, start, end in entries:
            has_p = _has_momentum_var(expr)
            has_velocity = _has_velocity_form(expr)
            if has_p and not has_velocity:
                return ScorerResult(1.0, [(start, end)])
            if has_p and has_velocity:
                return ScorerResult(0.5, [(start, end)])

        # Found candidates but none had p — empty gradient (model wrote something,
        # but the diagnostic is "wrong form" not "missing"; other scorers handle it)
        return ScorerResult(0.0, [])
    except Exception as exc:
        logger.debug("_score_%s_in_momentum failed: %s", var_name, exc)
        return ScorerResult(0.0, [])


def _score_T_in_momentum(pc: ParsedCompletion) -> ScorerResult:
    return _score_in_momentum_form(pc, "T")


def _score_H_in_momentum(pc: ParsedCompletion) -> ScorerResult:
    return _score_in_momentum_form(pc, "H")


def _score_V_correct(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    """q_V_correct: potential energy matches ground truth V.

    Uses pc.expression_spans for position-aware candidates, with label-based
    fallback via POTENTIAL section span.
    """
    expected_V = meta.get("V_expr", "")
    if not expected_V or expected_V == "none":
        return ScorerResult(0.0, [])

    # Primary: position-aware V entries
    v_entries: list[tuple[str, int, int]] = list(pc.expression_spans.get("V", []))

    # Fallback: label-based extraction (POTENTIAL: expr without =)
    labeled_v = pc.get_labeled("POTENTIAL")
    if labeled_v and not any(expr == labeled_v for expr, _, _ in v_entries):
        section_spans_v = pc.section_spans.get("POTENTIAL", [])
        if section_spans_v:
            s0, e0 = section_spans_v[0]
            v_entries.append((labeled_v, s0, e0))

    if not v_entries:
        return ScorerResult(0.0, [])

    # Score each candidate, track the winner with its source span
    best_score = 0.0
    best_entry = v_entries[0]
    for entry in v_entries:
        expr = entry[0]
        s = score_expression(expr, expected_V, [], constant_subs=meta.get("_constant_subs"))
        if s > best_score:
            best_score = s
            best_entry = entry
        if best_score >= 1.0:
            break

    best_expr, best_start, best_end = best_entry
    best_spans = [(best_start, best_end)]

    if best_score >= 0.9:
        return ScorerResult(best_score, best_spans)

    # Fallback: symbolic form (mgy, Fx, kx²) vs evaluated ground truth (49y/5).
    # Substitute known prompt constants into student expression before comparing.
    student_sym = try_sympify(best_expr, sympy_locals=HAMILTONIAN_LOCALS)
    teacher_sym = try_sympify(expected_V, sympy_locals=HAMILTONIAN_LOCALS)
    subs = meta.get("_constant_subs", {})
    if student_sym is not None and teacher_sym is not None and subs:
        try:
            student_eval = student_sym.subs(subs)
            diff = sp.simplify(student_eval - teacher_sym)
            if diff == 0:
                return ScorerResult(1.0, best_spans)
            free = (student_eval.free_symbols | teacher_sym.free_symbols) - {sp.Symbol("pi")}
            if free:
                test_point = {s: sp.Rational(3, 7) for s in free}
                s_val = float(student_eval.subs(test_point))
                t_val = float(teacher_sym.subs(test_point))
                if abs(t_val) > 1e-10 and abs(s_val - t_val) < 1e-4 * max(abs(t_val), 1):
                    return ScorerResult(1.0, best_spans)
        except Exception as exc:
            logger.debug("V_correct sympy substitution failed: %s", exc)

    return ScorerResult(best_score, best_spans)


def _score_equation(
    pc: ParsedCompletion, meta: dict, meta_key: str, fallback_pattern: str
) -> ScorerResult:
    """Score a Hamilton equation (dq/dt or dp/dt) against ground truth.

    Uses pc.expression_spans for all derivative keys (dq/dt, dp/dt, dx/dt, ...).
    Falls back to the EQUATIONS section span if the block has extra RHS text not
    captured by the derivative-key extraction.
    """
    expected = meta.get(meta_key, "")
    if not expected or expected == "none":
        return ScorerResult(0.0, [])

    expected_parts = [e.strip() for e in expected.split(";")]

    # Primary: position-aware derivative entries (dq/dt, dp/dt, dx/dt, ...)
    extracted_entries: list[tuple[str, int, int]] = []
    for key, entries in pc.expression_spans.items():
        if key.startswith("d") and "/dt" in key:
            extracted_entries.extend(entries)

    # Block fallback: RHS from the EQUATIONS section that weren't caught by
    # the derivative-key extraction (e.g., multiline LaTeX). Use EQUATIONS
    # section span as the position.
    seen_exprs = {e for e, _, _ in extracted_entries}
    block_rhs = pc.get_equations_rhs()
    eq_section_spans = pc.section_spans.get("EQUATIONS", [])
    if eq_section_spans:
        fallback_start, fallback_end = eq_section_spans[0]
        for expr in block_rhs:
            if expr not in seen_exprs:
                extracted_entries.append((expr, fallback_start, fallback_end))
                seen_exprs.add(expr)

    if not extracted_entries:
        # Weak signal: pattern appears in text but no parseable equations
        if re.search(fallback_pattern, pc.text):
            # Can't localize without expressions — leave spans empty
            return ScorerResult(0.2, [])
        return ScorerResult(0.0, [])

    # For each expected part, find the best-matching extracted entry.
    # Aggregate score = average across parts; aggregate spans = union of winners.
    part_scores: list[float] = []
    winning_spans: list[tuple[int, int]] = []
    for exp_part in expected_parts:
        best = 0.0
        best_entry: tuple[str, int, int] | None = None
        for entry in extracted_entries:
            expr = entry[0]
            s = score_expression(expr, exp_part, [], constant_subs=meta.get("_constant_subs"))
            if s > best:
                best = s
                best_entry = entry
        part_scores.append(best)
        if best_entry is not None:
            _, start, end = best_entry
            span = (start, end)
            if span not in winning_spans:
                winning_spans.append(span)

    final_score = sum(part_scores) / len(part_scores) if part_scores else 0.0
    return ScorerResult(final_score, winning_spans)


def _score_dqdt(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    return _score_equation(pc, meta, "dqdt", r"dq/dt|∂H/∂p|dx/dt|dtheta/dt|dr/dt|ds/dt")


def _score_dpdt(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    return _score_equation(pc, meta, "dpdt", r"dp/dt|-∂H/∂q|-dH/dq|dp_r/dt|dp_theta/dt")


def _score_correct_H(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    """q_correct_H: Hamiltonian matches ground truth.

    Finds all H expressions via pc.expression_spans (position-aware), with a
    label-based fallback for cases like "HAMILTONIAN: p²/4" where no `H =`
    pattern exists. Returns the score and the span of the winning expression.
    """
    expected_H = meta.get("H_expr", "")
    if not expected_H or expected_H == "none":
        return ScorerResult(0.0, [])

    # Primary: position-aware expressions from FINAL OUTPUT
    h_entries: list[tuple[str, int, int]] = list(pc.expression_spans.get("H", []))

    # Fallback: label-based extraction (HAMILTONIAN: expr without =)
    labeled_h = pc.get_labeled("HAMILTONIAN")
    if labeled_h and not any(expr == labeled_h for expr, _, _ in h_entries):
        section_spans_h = pc.section_spans.get("HAMILTONIAN", [])
        if section_spans_h:
            s0, e0 = section_spans_h[0]
            h_entries.append((labeled_h, s0, e0))

    # Split chained equalities — sub-candidates inherit the source line's span.
    # "T + V = p²/2 + 0 = p²/2" → also score "p²/2" but span stays at source line.
    expanded: list[tuple[str, int, int]] = []
    seen_exprs: set[str] = set()
    for expr, start, end in h_entries:
        if expr not in seen_exprs:
            expanded.append((expr, start, end))
            seen_exprs.add(expr)
        if "=" in expr:
            last_rhs = expr.rsplit("=", 1)[-1].strip()
            if last_rhs and last_rhs not in seen_exprs:
                expanded.append((last_rhs, start, end))
                seen_exprs.add(last_rhs)
    h_entries = expanded

    if not h_entries:
        return ScorerResult(0.0, [])

    # Score each candidate, track the winner with its source span
    best_score = 0.0
    best_entry = h_entries[0]
    for entry in h_entries:
        expr = entry[0]
        s = score_expression(expr, expected_H, [], constant_subs=meta.get("_constant_subs"))
        if s > best_score:
            best_score = s
            best_entry = entry
        if best_score >= 1.0:
            break

    best_expr, best_start, best_end = best_entry
    best_spans = [(best_start, best_end)]

    if best_score >= 0.7:
        return ScorerResult(best_score, best_spans)

    # Detailed analysis on the BEST candidate
    normed = normalize_for_sympy(best_expr)
    parsed = try_sympify(best_expr, sympy_locals=HAMILTONIAN_LOCALS)
    if parsed is not None and isinstance(parsed, sp.Basic) and parsed.is_number:
        # Distinct signal: "you plugged in numbers, keep it symbolic"
        return ScorerResult(0.1, best_spans)

    direct_score = score_expression(
        best_expr, expected_H, [], constant_subs=meta.get("_constant_subs")
    )
    if direct_score >= 0.7:
        return ScorerResult(direct_score, best_spans)

    # Velocity form check — correct structure but wrong variables
    has_velocity = "_VDOT_" in normed or (parsed is not None and parsed.atoms(sp.Derivative))
    if has_velocity:
        expected_sym = try_sympify(expected_H, sympy_locals=HAMILTONIAN_LOCALS)
        if expected_sym is not None:
            p_sym = sp.Symbol("p")
            expected_V_terms = expected_sym.subs(p_sym, 0)
            vdot_sym = sp.Symbol("_VDOT_")
            if parsed is not None:
                student_V_terms = parsed.subs(vdot_sym, 0)
                try:
                    if sp.simplify(student_V_terms - expected_V_terms) == 0:
                        return ScorerResult(0.6, best_spans)  # V correct, T wrong form
                except Exception as exc:
                    logger.debug("velocity form V-check failed: %s", exc)
        return ScorerResult(0.4, best_spans)  # Velocity form — recognizably an H

    return ScorerResult(direct_score, best_spans)


def _score_consistency(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    """q_consistency: stated H's derivatives match stated equations.

    Returns spans covering the H expression AND the equation expressions —
    the tokens where consistency is evaluated span both.
    """
    # Build spans from H + equations that the scorer examines
    scoring_spans: list[tuple[int, int]] = []

    # H span
    h_entries = list(pc.expression_spans.get("H", []))
    extracted_H_str: str | None = None
    if h_entries:
        # Use the last H expression (model's final answer)
        extracted_H_str, hs, he = h_entries[-1]
        scoring_spans.append((hs, he))
    else:
        labeled_h = pc.get_labeled("HAMILTONIAN")
        if labeled_h:
            extracted_H_str = labeled_h
            section_spans_h = pc.section_spans.get("HAMILTONIAN", [])
            if section_spans_h:
                scoring_spans.append(section_spans_h[0])

    if extracted_H_str is None:
        return ScorerResult(0.0, [])

    H_sym = try_sympify(extracted_H_str, sympy_locals=HAMILTONIAN_LOCALS)
    if H_sym is None or not isinstance(H_sym, sp.Expr):
        return ScorerResult(0.2, scoring_spans)

    coords = meta.get("coordinates", "x")
    coord_list = [c.strip() for c in coords.split(",") if c.strip()]
    if not coord_list:
        logger.debug("_score_consistency: empty coord_list, returning 0.2")
        return ScorerResult(0.2, scoring_spans)

    # Build equation entries with spans (same pattern as _score_equation)
    equation_entries: list[tuple[str, int, int]] = []
    for key, entries in pc.expression_spans.items():
        if key.startswith("d") and "/dt" in key:
            equation_entries.extend(entries)
    seen_exprs = {e for e, _, _ in equation_entries}
    eq_section_spans = pc.section_spans.get("EQUATIONS", [])
    if eq_section_spans:
        fs, fe = eq_section_spans[0]
        for block_expr in pc.get_equations_rhs():
            if block_expr not in seen_exprs:
                equation_entries.append((block_expr, fs, fe))
                seen_exprs.add(block_expr)

    if not equation_entries:
        return ScorerResult(0.3, scoring_spans)

    # Add equation spans to the output spans
    for _, s, e in equation_entries:
        span = (s, e)
        if span not in scoring_spans:
            scoring_spans.append(span)

    h_sym_names = {s.name for s in H_sym.free_symbols}

    consistency_scores: list[float] = []
    for coord in coord_list:
        candidates = [
            f"p_{coord}",
            "p",
            coord.replace("x", "p") if coord.startswith("x") else None,
            f"p{coord[-1]}" if coord[-1].isdigit() else None,
        ]
        candidates = [c for c in candidates if c is not None]
        p_name = next((c for c in candidates if c in h_sym_names), candidates[0])

        p_sym = HAMILTONIAN_LOCALS.get(p_name) or sp.Symbol(p_name)
        q_sym = HAMILTONIAN_LOCALS.get(coord) or sp.Symbol(coord)

        try:
            expected_dqdt = sp.diff(H_sym, p_sym)
            expected_dpdt = -sp.diff(H_sym, q_sym)

            for eq_str, _, _ in equation_entries:
                eq_sym = try_sympify(eq_str, sympy_locals=HAMILTONIAN_LOCALS)
                if eq_sym is not None:
                    try:
                        if sp.simplify(eq_sym - expected_dqdt) == 0:
                            consistency_scores.append(1.0)
                            break
                        if sp.simplify(eq_sym - expected_dpdt) == 0:
                            consistency_scores.append(1.0)
                            break
                    except Exception as exc:
                        logger.debug("consistency eq-check failed: %s", exc)
            else:
                consistency_scores.append(0.3)
        except Exception as exc:
            logger.debug("consistency diff failed: %s", exc)
            consistency_scores.append(0.2)

    if not consistency_scores:
        return ScorerResult(0.3, scoring_spans)
    final_score = sum(consistency_scores) / len(consistency_scores)
    return ScorerResult(final_score, scoring_spans)


def _score_correct_coefficient(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    """Quality scorer: checks numerical coefficients in H and equations against ground truth.

    Returns spans of the H expression and equation expressions that were scored.
    """
    try:
        has_H_expr = bool(meta.get("H_expr", ""))
        has_dqdt = bool(meta.get("dqdt", ""))
        has_dpdt = bool(meta.get("dpdt", ""))

        if not has_H_expr and not has_dqdt and not has_dpdt:
            return ScorerResult(0.0, [])

        component_scores: list[float] = []
        any_extracted = False
        winning_spans: list[tuple[int, int]] = []

        def _add_span(span: tuple[int, int]) -> None:
            if span not in winning_spans:
                winning_spans.append(span)

        # ── Score H ──────────────────────────────────────────────────────────
        h_score: float | None = None
        if has_H_expr:
            # Find H via expression_spans (primary) or HAMILTONIAN label (fallback)
            h_entries = list(pc.expression_spans.get("H", []))
            if not h_entries:
                labeled_h = pc.get_labeled("HAMILTONIAN")
                if labeled_h:
                    section_spans_h = pc.section_spans.get("HAMILTONIAN", [])
                    if section_spans_h:
                        s0, e0 = section_spans_h[0]
                        h_entries.append((labeled_h, s0, e0))
            if h_entries:
                any_extracted = True
                # Use the last H expression (model's final answer)
                expr, h_start, h_end = h_entries[-1]
                h_score = score_expression(
                    expr,
                    meta.get("H_expr", ""),
                    [],
                    constant_subs=meta.get("_constant_subs"),
                )
                component_scores.append(h_score)
                _add_span((h_start, h_end))

        # ── Score equations ──────────────────────────────────────────────────
        eq_score: float | None = None
        if has_dqdt or has_dpdt:
            # Build (expr, span) list for all derivative keys + EQUATIONS block fallback
            extracted_entries: list[tuple[str, int, int]] = []
            for key, entries in pc.expression_spans.items():
                if key.startswith("d") and "/dt" in key:
                    extracted_entries.extend(entries)
            seen_exprs = {e for e, _, _ in extracted_entries}
            eq_section_spans = pc.section_spans.get("EQUATIONS", [])
            if eq_section_spans:
                fs, fe = eq_section_spans[0]
                for block_expr in pc.get_equations_rhs():
                    if block_expr not in seen_exprs:
                        extracted_entries.append((block_expr, fs, fe))
                        seen_exprs.add(block_expr)

            eq_component_scores: list[float] = []

            if extracted_entries:
                any_extracted = True
                csubs = meta.get("_constant_subs")

                for meta_key, active in [("dqdt", has_dqdt), ("dpdt", has_dpdt)]:
                    if not active:
                        continue
                    expected_str = meta.get(meta_key, "")
                    if not expected_str:
                        continue
                    expected_parts = [e.strip() for e in expected_str.split(";") if e.strip()]
                    part_scores: list[float] = []
                    for exp_part in expected_parts:
                        best = 0.0
                        best_entry: tuple[str, int, int] | None = None
                        for entry in extracted_entries:
                            ext_expr = entry[0]
                            s = score_expression(ext_expr, exp_part, [], constant_subs=csubs)
                            if s > best:
                                best = s
                                best_entry = entry
                        part_scores.append(best)
                        if best_entry is not None:
                            _add_span((best_entry[1], best_entry[2]))
                    if part_scores:
                        eq_component_scores.append(sum(part_scores) / len(part_scores))

            if eq_component_scores:
                eq_score = sum(eq_component_scores) / len(eq_component_scores)
                component_scores.append(eq_score)

        if not any_extracted:
            return ScorerResult(0.0, [])

        if not component_scores:
            return ScorerResult(0.2, winning_spans)

        avg = sum(component_scores) / len(component_scores)

        # Tier assignment: 1.0 requires ALL components >= 0.9
        if all(s >= 0.9 for s in component_scores):
            return ScorerResult(1.0, winning_spans)

        # 0.7 tier: one side clearly correct, other wrong
        if h_score is not None and eq_score is not None:
            if (h_score >= 0.85 and eq_score < 0.6) or (eq_score >= 0.85 and h_score < 0.6):
                return ScorerResult(0.7, winning_spans)

        if avg >= 0.75:
            return ScorerResult(0.7, winning_spans)
        if avg >= 0.35:
            return ScorerResult(0.4, winning_spans)
        return ScorerResult(max(0.2, avg), winning_spans)
    except Exception as exc:
        logger.debug("_score_correct_coefficient failed: %s", exc)
        return ScorerResult(0.0, [])


def _score_derivative_correct(pc: ParsedCompletion, meta: dict) -> ScorerResult:
    """Quality scorer: checks if dp/dt derivative is correct, with a DISTINCT signal (0.5)
    for factor-of-2 errors.

    Returns the span of the equation expression that was matched.
    """
    try:
        expected = meta.get("dpdt", "")
        if not expected or expected == "none":
            return ScorerResult(0.0, [])

        expected_parts = [e.strip() for e in expected.split(";") if e.strip()]

        # Build position-aware derivative entries (same pattern as _score_equation)
        extracted_entries: list[tuple[str, int, int]] = []
        for key, entries in pc.expression_spans.items():
            if key.startswith("d") and "/dt" in key:
                extracted_entries.extend(entries)
        seen_exprs = {e for e, _, _ in extracted_entries}
        eq_section_spans = pc.section_spans.get("EQUATIONS", [])
        if eq_section_spans:
            fs, fe = eq_section_spans[0]
            for block_expr in pc.get_equations_rhs():
                if block_expr not in seen_exprs:
                    extracted_entries.append((block_expr, fs, fe))
                    seen_exprs.add(block_expr)

        if not extracted_entries:
            # Weak signal: pattern exists in text but no parseable equations
            if re.search(r"dp/dt|dp_[a-z]+/dt|-∂H/∂[a-z]|-dH/d[a-z]", pc.text):
                return ScorerResult(0.1, [])
            return ScorerResult(0.0, [])

        # Find best matching entry
        best_score = 0.0
        best_entry: tuple[str, int, int] | None = None
        best_teacher_str: str | None = None

        for exp_part in expected_parts:
            for entry in extracted_entries:
                ext = entry[0]
                s = score_expression(ext, exp_part, [], constant_subs=meta.get("_constant_subs"))
                if s > best_score:
                    best_score = s
                    best_entry = entry
                    best_teacher_str = exp_part

        best_spans: list[tuple[int, int]] = (
            [(best_entry[1], best_entry[2])] if best_entry is not None else []
        )

        if best_score >= 0.9:
            return ScorerResult(1.0, best_spans)

        # Factor-of-2 detection
        if best_entry is not None and best_teacher_str is not None:
            best_student_str = best_entry[0]
            student_sym = try_sympify(best_student_str, sympy_locals=HAMILTONIAN_LOCALS)
            teacher_sym = try_sympify(best_teacher_str, sympy_locals=HAMILTONIAN_LOCALS)

            if student_sym is not None and teacher_sym is not None:
                try:
                    free = (student_sym.free_symbols | teacher_sym.free_symbols) - {sp.Symbol("pi")}
                    test_point = {s: sp.Rational(3, 7) for s in free}
                    student_val = float(student_sym.subs(test_point))
                    teacher_val = float(teacher_sym.subs(test_point))

                    if abs(teacher_val) > 1e-10:
                        ratio = student_val / teacher_val
                        if abs(ratio - 0.5) < 0.05:
                            return ScorerResult(0.5, best_spans)  # Forgot factor of 2
                        if abs(ratio - 2.0) < 0.05:
                            return ScorerResult(0.5, best_spans)  # Doubled
                except Exception as exc:
                    logger.debug("factor-of-2 check failed: %s", exc)

        if best_score > 0.0:
            return ScorerResult(min(best_score, 0.4), best_spans)

        return ScorerResult(0.2, best_spans)

    except Exception as exc:
        logger.debug("_score_derivative_correct failed: %s", exc)
        return ScorerResult(0.0, [])


def _score_defines_momentum(pc: ParsedCompletion) -> ScorerResult:
    """Quality scorer: checks if completion explicitly defines momentum p via
    the Lagrangian partial derivative p = ∂L/∂q̇, not just incidentally mentions p.

    Returns:
        1.0 — explicit ∂L/∂q̇ notation found (LaTeX, Unicode, or text forms)
        0.7 — MOMENTUM section has a numeric expression for p but no ∂L notation
        0.3 — p = appears somewhere without a MOMENTUM label
        0.0 — no p = found anywhere
    """
    text = pc.text
    if not text:
        return ScorerResult(0.0, [])

    momentum_spans = pc.section_spans.get("MOMENTUM", [])
    momentum_span = [momentum_spans[0]] if momentum_spans else []

    # ── 1. Explicit partial derivative of L w.r.t. generalized velocity ──
    _PARTIAL_PATTERNS = [
        r"\\frac\s*\{\\partial\s*L\}\s*\{\\partial\s*\\dot",
        r"\\partial\s*L\s*/\s*\\partial\s*\\dot",
        r"\\partial\s*L\s*/\s*\\partial",
        r"∂L\s*/\s*∂",
        r"∂\s*L\s*/\s*∂",
        r"partial\s+L\s*/\s*partial",
        r"dL\s*/\s*d",
    ]
    try:
        for pat in _PARTIAL_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                # Precise span around the ∂L/∂... notation
                return ScorerResult(1.0, [(m.start(), m.end())])
    except Exception as exc:
        logger.debug("_score_defines_momentum pattern failed: %s", exc)

    # ── 2. MOMENTUM section with numeric expression but no ∂L ──
    try:
        momentum_str = pc.get_labeled("MOMENTUM")
        if momentum_str is not None and momentum_str.strip():
            has_numbers = bool(re.search(r"\d", momentum_str))
            has_p_eq = bool(re.search(r"p\s*=", text, re.IGNORECASE))
            if has_numbers and has_p_eq:
                return ScorerResult(0.7, momentum_span)
    except Exception as exc:
        logger.debug("_score_defines_momentum MOMENTUM check failed: %s", exc)

    # ── 3. p = appears anywhere, no MOMENTUM label ──
    try:
        m = re.search(r"p\s*=", text, re.IGNORECASE)
        if m:
            # Use expression_spans["p"] if we have one; otherwise just the match span
            p_entries = pc.expression_spans.get("p", [])
            if p_entries:
                start, end = p_entries[0][1], p_entries[0][2]
                return ScorerResult(0.3, [(start, end)])
            return ScorerResult(0.3, [(m.start(), m.end())])
    except Exception as exc:
        logger.debug("_score_defines_momentum p= check failed: %s", exc)

    return ScorerResult(0.0, [])


# ─── Main reward function ─────────────────────────────────────────────────────


def hamiltonian_reward(
    prompt: str,
    completion: str,
    metadata: dict | None = None,
    scorer=None,
) -> RewardResult:
    """Score a Hamiltonian derivation with granular per-section qualities.

    Phase 1: q_format, q_has_math — structured output with labeled sections
    Phase 2: q_momentum_defined, q_T_uses_p, q_V_correct — momentum form + physics
    Phase 3: q_correct_dqdt, q_correct_dpdt — Hamilton's equations match ground truth
    Phase 4: q_correct_H, q_consistency — full Hamiltonian + internal consistency

    Args:
        scorer: optional callable(student_str, teacher_str, ...) → float.
                When None, uses the existing v1 sympy scoring behavior.
    """
    meta = copy.deepcopy(metadata) if metadata else {}  # Deep copy to avoid mutation
    meta["prompt"] = prompt  # Make prompt available to scoring functions
    # Extract physical constants from prompt once — shared by all scorers
    meta["_constant_subs"] = extract_constants_from_prompt(prompt)
    scores: dict[str, float] = {}
    scored_spans: dict[str, list[tuple[int, int]]] = {}

    # Parse completion ONCE — all scorers share this pre-computed result.
    pc = ParsedCompletion.from_text(completion)

    # ── Phase 1: Format ──
    # Migrated: _score_format and _score_has_math return ScorerResult
    r = _score_format(pc)
    scores["q_format"] = r.score
    scored_spans["q_format"] = r.spans

    r = _score_has_math(pc)
    scores["q_has_math"] = r.score
    scored_spans["q_has_math"] = r.spans

    # ── Phase 2: Physics — granular per-section ──
    # Migrated: _score_momentum_defined returns ScorerResult
    r = _score_momentum_defined(pc)
    scores["q_momentum_defined"] = r.score
    scored_spans["q_momentum_defined"] = r.spans

    # Migrated: _score_T_uses_p returns ScorerResult
    r = _score_T_uses_p(pc, meta)
    scores["q_T_uses_p"] = r.score
    scored_spans["q_T_uses_p"] = r.spans

    # Migrated: _score_V_correct returns ScorerResult
    r = _score_V_correct(pc, meta)
    scores["q_V_correct"] = r.score
    scored_spans["q_V_correct"] = r.spans

    # ── Phase 3: Equation correctness ──
    # Migrated: _score_dqdt and _score_dpdt return ScorerResult
    r = _score_dqdt(pc, meta)
    scores["q_correct_dqdt"] = r.score
    scored_spans["q_correct_dqdt"] = r.spans

    r = _score_dpdt(pc, meta)
    scores["q_correct_dpdt"] = r.score
    scored_spans["q_correct_dpdt"] = r.spans

    # ── Phase 4: Full Hamiltonian + consistency ──
    # Migrated: _score_correct_H returns ScorerResult
    r = _score_correct_H(pc, meta)
    scores["q_correct_H"] = r.score
    scored_spans["q_correct_H"] = r.spans

    # Migrated: _score_consistency returns ScorerResult
    r = _score_consistency(pc, meta)
    scores["q_consistency"] = r.score
    scored_spans["q_consistency"] = r.spans

    # ── Phase 5: Granular momentum/variable/derivative checks ──
    # Migrated: _score_defines_momentum returns ScorerResult
    r = _score_defines_momentum(pc)
    scores["q_defines_momentum"] = r.score
    scored_spans["q_defines_momentum"] = r.spans

    # Migrated: _score_T_in_momentum and _score_H_in_momentum return ScorerResult
    r = _score_T_in_momentum(pc)
    scores["q_T_in_momentum"] = r.score
    scored_spans["q_T_in_momentum"] = r.spans

    r = _score_H_in_momentum(pc)
    scores["q_H_in_momentum"] = r.score
    scored_spans["q_H_in_momentum"] = r.spans

    # Migrated: _score_correct_coefficient returns ScorerResult
    r = _score_correct_coefficient(pc, meta)
    scores["q_correct_coefficient"] = r.score
    scored_spans["q_correct_coefficient"] = r.spans

    # Migrated: _score_derivative_correct returns ScorerResult (FINAL scorer migration)
    r = _score_derivative_correct(pc, meta)
    scores["q_derivative_correct"] = r.score
    scored_spans["q_derivative_correct"] = r.spans

    # All scorers now return ScorerResult. No legacy _find_expression_spans fallback needed.

    total = sum(scores.values()) / max(len(scores), 1)

    # ── Diagnostic logging ──
    try:
        _DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        diag = {
            "system": meta.get("system", "unknown"),
            "difficulty": meta.get("difficulty", "unknown"),
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "total": round(total, 3),
            "completion_len": len(completion),
        }
        with open(_DIAG_PATH, "a") as f:
            f.write(json.dumps(diag) + "\n")
    except Exception as exc:
        logger.debug("scorer exception: %s", exc)

    return RewardResult(reward=total, scores=scores, scored_spans=scored_spans)
