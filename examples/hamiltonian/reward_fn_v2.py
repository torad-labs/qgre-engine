"""Correctness-only reward function for Hamiltonian mechanics training.

Scores mathematical correctness via sympy equivalence using HuggingFace's
math-verify library (the same verification system used by TRL and the RLVR
community for math benchmarks). No format scoring.

The model can write in any format — LaTeX, ASCII, unicode. The reward function
extracts all math expressions and checks each against ground truth.

Design principles (from April 2026 analysis + 25-paper survey):
- RL teaches WHAT to think (correctness). SFT teaches HOW to present (format).
- Format scoring in RL is the #1 reward hacking vector (arxiv:2602.18037).
- Score the first correct occurrence, ignore duplicates.
- Use math-verify for parsing + verification (battle-tested, highest accuracy on MATH).

Qualities:
  q_kinetic:     T expression matches ground truth
  q_potential:   V expression matches ground truth
  q_hamiltonian: H expression matches ground truth
  q_dqdt:        dq/dt Hamilton equation matches ground truth
  q_dpdt:        dp/dt Hamilton equation matches ground truth
  q_consistency: model's equations are consistent with its own H
"""

from __future__ import annotations

import logging
import signal
from contextlib import contextmanager

import sympy as sp
from math_verify import parse, verify

from qgre.types import RewardResult


logger = logging.getLogger("qgre.hamiltonian_reward")

# Shared symbols for consistency check (sp.diff needs the same symbol objects)
_SYMBOL_MAP = {str(s): s for s in sp.symbols("x y r theta q p t m k g", real=True)}


# ─── Timeout ───────────────────────────────────────────────────────────────────


class SympyTimeoutError(Exception):
    pass


@contextmanager
def sympy_timeout(seconds: int = 3):
    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise SympyTimeoutError(f"Sympy timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ─── Normalization ─────────────────────────────────────────────────────────────


def _normalize_text(text: str) -> str:
    """Normalize unicode artifacts that math-verify can't handle natively."""
    return (
        text.replace("\u00b2", "^2")
        .replace("\u00b3", "^3")
        .replace("\u00b7", "*")
        .replace("\u00d7", "*")
    )


# ─── Ground truth parsing ──────────────────────────────────────────────────────


def _gold_parse(sympy_str: str) -> list | None:
    """Convert sympy-format ground truth to math-verify parsed form.

    Strategy: sympify → LaTeX → math-verify parse. This ensures the gold
    goes through the same LaTeX parser as the model's output.
    """
    try:
        expr = sp.sympify(sympy_str, locals=_SYMBOL_MAP)
        latex = sp.latex(expr)
        result = parse(f"${latex}$")
        return result if result else None
    except Exception:
        logger.warning(f"Cannot parse ground truth: {sympy_str}")
        return None


# ─── Expression extraction ─────────────────────────────────────────────────────


def _extract_rhs_expressions(
    text: str,
) -> list[tuple[str, str, int, int]]:
    """Extract all (lhs, rhs, char_start, char_end) from text.

    Format-agnostic: finds expressions regardless of whether they're wrapped
    in $ delimiters, $$ blocks, bold markers, or plain text. The model can
    write in ANY format — we scan every line for '=' and extract.

    Strategy: split text into lines, for each line containing '=':
    1. Strip all formatting artifacts ($, $$, *, #, labels with colons)
    2. Split on '=' to get LHS/RHS pairs
    3. Return the cleaned LHS, raw RHS, and character offsets into original text

    No dependency on specific delimiters. math-verify handles LaTeX vs plain
    text internally when parsing the RHS.
    """
    results: list[tuple[str, str, int, int]] = []
    normalized = _normalize_text(text)

    char_pos = 0
    for line in normalized.split("\n"):
        line_start = char_pos
        line_end = char_pos + len(line)
        char_pos = line_end + 1  # +1 for the \n

        if "=" not in line:
            continue

        # Strip ALL formatting from the line before splitting on =
        cleaned = line.strip()
        # Strip $$ and $ delimiters
        cleaned = cleaned.replace("$$", "").replace("$", "")
        # Strip markdown bold/heading markers at boundaries
        cleaned = cleaned.lstrip("*#").rstrip("*").strip()
        # Strip LaTeX alignment markers
        cleaned = cleaned.lstrip("&").strip()

        if "=" not in cleaned or len(cleaned) < 3:
            continue

        parts = cleaned.split("=")
        if len(parts) < 2:
            continue

        # Extract LHS: take last token before first =
        lhs_raw = parts[0].strip()
        # Strip label prefixes (KINETIC:, **MOMENTUM**:, etc.)
        if ":" in lhs_raw:
            lhs_raw = lhs_raw.split(":")[-1].strip()
        # Strip remaining markdown/LaTeX at boundaries
        lhs_raw = lhs_raw.lstrip("*#").rstrip("*").strip()
        # Take last whitespace-separated token as variable name
        tokens = lhs_raw.split()
        lhs = tokens[-1].strip("*$\\()") if tokens else ""

        if not lhs:
            continue

        # Each part after the first = is a potential RHS
        for rhs_raw in parts[1:]:
            rhs = rhs_raw.strip()
            if not rhs or len(rhs) < 1:
                continue
            # Skip prose: if RHS starts with a common English word, skip
            first_word = rhs.split()[0].lower() if rhs.split() else ""
            if first_word in {
                "the",
                "is",
                "was",
                "from",
                "and",
                "for",
                "with",
                "where",
                "since",
                "given",
                "that",
                "this",
                "which",
                "here",
                "we",
            }:
                continue
            results.append((lhs, rhs, line_start, line_end))

    return results


# ─── Scoring ───────────────────────────────────────────────────────────────────


def _find_correct(
    expressions: list[tuple[str, str, int, int]],
    gold_parsed: list,
    lhs_patterns: list[str] | None = None,
) -> tuple[float, list[tuple[int, int]]]:
    """Find the first expression that matches gold via math-verify.

    Returns (1.0, [(span)]) on match, (0.0, []) otherwise.
    """
    for lhs, rhs, char_start, char_end in expressions:
        # Filter by LHS if specified
        if lhs_patterns is not None:
            lhs_clean = lhs.lower().strip("*$\\")
            if not any(lhs_clean == p.lower() for p in lhs_patterns):
                continue

        # Parse RHS via math-verify
        try:
            answer_parsed = parse(f"${rhs}$")
        except Exception:
            continue

        if not answer_parsed:
            continue

        # Verify equivalence
        try:
            if verify(gold_parsed, answer_parsed):
                return 1.0, [(char_start, char_end)]
        except Exception:
            continue

    return 0.0, []


def _find_derivative(
    expressions: list[tuple[str, str, int, int]],
    gold_parsed: list,
    var: str,
) -> tuple[float, list[tuple[int, int]]]:
    """Find a correct Hamilton equation dVAR/dt = gold."""
    # All ways the model might write the LHS of a derivative
    coord_aliases = {"q": ["x", "y", "r", "theta", "s"], "p": ["p"]}
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
            answer_parsed = parse(f"${rhs}$")
        except Exception:
            continue

        if not answer_parsed:
            continue

        try:
            if verify(gold_parsed, answer_parsed):
                return 1.0, [(char_start, char_end)]
        except Exception:
            continue

    return 0.0, []


def _score_consistency(
    expressions: list[tuple[str, str, int, int]],
    meta: dict,
) -> float:
    """Check if the model's dq/dt and dp/dt are consistent with its H.

    dq/dt should equal dH/dp, dp/dt should equal -dH/dq.
    Uses sympy differentiation (not math-verify) because we need to
    compute derivatives of the model's own H expression.
    """
    # Find model's H
    h_expr = None
    for lhs, rhs, _, _ in expressions:
        if lhs.lower().strip("*$\\") in ("h", "hamiltonian"):
            try:
                answer = parse(f"${rhs}$")
                if answer:
                    h_expr = answer[0]  # First parsed result (sympy expr)
                    break
            except Exception:
                continue

    if h_expr is None or not hasattr(h_expr, "free_symbols"):
        return 0.0

    # Find model's dq/dt and dp/dt
    coord = meta.get("coordinates", "x")
    p = _SYMBOL_MAP.get("p", sp.Symbol("p"))
    q = _SYMBOL_MAP.get(coord, sp.Symbol(coord))

    # Substitute model's symbols to match our symbol map
    subs = {}
    for sym in h_expr.free_symbols:
        name = str(sym)
        if name in _SYMBOL_MAP:
            subs[sym] = _SYMBOL_MAP[name]
    if subs:
        h_expr = h_expr.subs(subs)

    dqdt_expr = None
    dpdt_expr = None
    all_coord_names = [coord, "q"]

    for lhs, rhs, _, _ in expressions:
        lhs_clean = lhs.lower().strip("*$\\").replace(" ", "")

        if dqdt_expr is None:
            for v in all_coord_names:
                if f"d{v}" in lhs_clean and (
                    "/dt" in lhs_clean or "frac" in lhs_clean or "dot" in lhs_clean
                ):
                    try:
                        a = parse(f"${rhs}$")
                        if a:
                            dqdt_expr = a[0]
                            if hasattr(dqdt_expr, "free_symbols"):
                                s = {
                                    sym: _SYMBOL_MAP[str(sym)]
                                    for sym in dqdt_expr.free_symbols
                                    if str(sym) in _SYMBOL_MAP
                                }
                                if s:
                                    dqdt_expr = dqdt_expr.subs(s)
                    except Exception:
                        pass
                    break

        if dpdt_expr is None:
            if "dp" in lhs_clean and (
                "/dt" in lhs_clean or "frac" in lhs_clean or "dot" in lhs_clean
            ):
                try:
                    a = parse(f"${rhs}$")
                    if a:
                        dpdt_expr = a[0]
                        if hasattr(dpdt_expr, "free_symbols"):
                            s = {
                                sym: _SYMBOL_MAP[str(sym)]
                                for sym in dpdt_expr.free_symbols
                                if str(sym) in _SYMBOL_MAP
                            }
                            if s:
                                dpdt_expr = dpdt_expr.subs(s)
                except Exception:
                    pass

    if dqdt_expr is None or dpdt_expr is None:
        return 0.0

    try:
        with sympy_timeout(3):
            expected_dqdt = sp.diff(h_expr, p)
            expected_dpdt = -sp.diff(h_expr, q)
            dqdt_ok = sp.simplify(dqdt_expr - expected_dqdt) == 0
            dpdt_ok = sp.simplify(dpdt_expr - expected_dpdt) == 0
            if dqdt_ok and dpdt_ok:
                return 1.0
            if dqdt_ok or dpdt_ok:
                return 0.5
    except (SympyTimeoutError, Exception):
        pass
    return 0.0


# ─── Main reward function ──────────────────────────────────────────────────────


def hamiltonian_reward(
    prompt: str,
    completion: str,
    metadata: dict | None = None,
) -> RewardResult:
    """Score a Hamiltonian derivation by mathematical correctness only.

    No format scoring. Uses math-verify for parsing and verification.
    Extracts ALL expressions and checks each against ground truth.
    """
    meta = metadata or {}
    text = completion
    scores: dict[str, float] = {}
    scored_spans: dict[str, list[tuple[int, int]]] = {}

    # Extract all expressions from the completion
    expressions = _extract_rhs_expressions(text)

    # Score each quality
    for quality, meta_key, lhs_patterns in [
        ("q_kinetic", "T_expr", ["T", "t", "kinetic"]),
        ("q_potential", "V_expr", ["V", "v", "potential"]),
        ("q_hamiltonian", "H_expr", ["H", "h", "hamiltonian"]),
    ]:
        if meta.get(meta_key):
            gold = _gold_parse(meta[meta_key])
            if gold:
                score, spans = _find_correct(expressions, gold, lhs_patterns)
                scores[quality] = score
                if spans:
                    scored_spans[quality] = spans
            else:
                scores[quality] = 0.0
        else:
            scores[quality] = 0.0

    # Derivatives
    if meta.get("dqdt"):
        gold = _gold_parse(meta["dqdt"])
        if gold:
            coord = meta.get("coordinates", "x")
            score, spans = _find_derivative(expressions, gold, var=coord)
            if score == 0.0 and coord != "q":
                score, spans = _find_derivative(expressions, gold, var="q")
            scores["q_dqdt"] = score
            if spans:
                scored_spans["q_dqdt"] = spans
        else:
            scores["q_dqdt"] = 0.0

    if meta.get("dpdt"):
        gold = _gold_parse(meta["dpdt"])
        if gold:
            score, spans = _find_derivative(expressions, gold, var="p")
            scores["q_dpdt"] = score
            if spans:
                scored_spans["q_dpdt"] = spans
        else:
            scores["q_dpdt"] = 0.0

    # Consistency
    scores["q_consistency"] = _score_consistency(expressions, meta)

    # Aggregate (floor at 0.01 for Dr.GRPO)
    total = max(sum(scores.values()) / max(len(scores), 1), 0.01)

    return RewardResult(
        reward=total,
        scores=scores,
        scored_spans=scored_spans,
    )
