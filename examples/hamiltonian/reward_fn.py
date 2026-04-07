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
import signal
from contextlib import contextmanager
from pathlib import Path

import sympy as sp


# ─── Timeout utility for sympy operations ─────────────────────────────────────


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
        # Windows or signal not available — no timeout
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


try:
    from latex2sympy2_extended import latex2sympy as _latex2sympy
except ImportError:
    _latex2sympy = None  # Fallback: regex-only path

from qgre.segments import HAMILTONIAN_LABEL_ALIASES
from qgre.types import RewardResult


logger = logging.getLogger("qgre.hamiltonian_reward")


# ─── Structured Output Parser (regex-free extraction) ─────────────────────────


class StructuredOutputParser:
    """Line-based parser for Hamiltonian structured output.

    Parses the expected format:
        COORDINATES: q = x
        MOMENTUM: p = 2*dx/dt
        KINETIC: T = p²/4
        POTENTIAL: V = 2*x²
        HAMILTONIAN: H = p²/4 + 2*x²
        EQUATIONS:
          dq/dt = p/2
          dp/dt = -4*x

    Uses exact string matching instead of regex for label detection.
    More robust and predictable than regex-based extraction.
    """

    # Canonical labels and their aliases (case-insensitive)
    LABELS = {
        "COORDINATES": {"coordinates", "coordinate", "generalized coordinate", "coords"},
        "MOMENTUM": {"momentum", "conjugate momentum", "momenta"},
        "KINETIC": {"kinetic", "kinetic energy", "t"},
        "POTENTIAL": {"potential", "potential energy", "v"},
        "HAMILTONIAN": {"hamiltonian", "h"},
        "EQUATIONS": {"equations", "equations of motion", "hamilton's equations", "eom"},
    }

    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self._cache: dict[str, str | None] = {}
        self._equations_cache: list[str] | None = None
        self._all_expressions_cache: dict[str, list[str]] | None = None

    def _clean_line(self, line: str) -> str:
        """Strip markdown artifacts from a line."""
        s = line.strip()
        # Strip markdown bold/italic markers from start
        while s.startswith("*") or s.startswith("#"):
            s = s.lstrip("*#").strip()
        # Strip trailing * markers
        while s.endswith("*"):
            s = s.rstrip("*").strip()
        # Strip **LABEL**: pattern — bold markers around label before colon
        # Handle cases like "MOMENTUM**: p = ..." → "MOMENTUM: p = ..."
        for i in range(len(s)):
            if s[i : i + 3] == "**:" or s[i : i + 2] == "*:":
                s = s[:i] + s[i + 2 :] if s[i : i + 3] == "**:" else s[:i] + s[i + 1 :]
                break
        # Strip LaTeX delimiters
        if s.startswith("$"):
            s = s[1:].strip()
        if s.endswith("$"):
            s = s[:-1].strip()
        return s

    def _extract_expression(self, content: str) -> str:
        """Extract the mathematical expression from content after label."""
        s = content.strip()
        # Strip leading variable name and = (e.g., "T = p²/4" → "p²/4")
        if "=" in s:
            parts = s.split("=", 1)
            rhs = parts[-1].strip()
            # Check if RHS looks like math (not prose)
            prose_words = {
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
            }
            words = rhs.lower().split()
            if not any(w in prose_words for w in words[:3]):  # Check first 3 words
                return self._clean_expression(rhs)
        return self._clean_expression(s)

    def _clean_expression(self, expr: str) -> str:
        """Clean trailing artifacts from expression."""
        s = expr.strip()
        # Strip trailing markdown/LaTeX artifacts
        while s and s[-1] in "*$\\":
            s = s[:-1].strip()
        # Strip trailing LaTeX line break
        if s.endswith("\\\\"):
            s = s[:-2].strip()
        return s

    def _match_label(self, line: str) -> tuple[str, str] | None:
        """Check if line starts with a known label. Returns (canonical_label, rest_of_line) or None."""
        cleaned = self._clean_line(line).lower()

        for canonical, aliases in self.LABELS.items():
            all_names = {canonical.lower()} | aliases
            for name in all_names:
                # Check for "LABEL:" or "LABEL :" pattern
                if cleaned.startswith(name):
                    rest = cleaned[len(name) :].lstrip()
                    if rest.startswith(":") or rest.startswith("="):
                        content = rest[1:].strip()
                        return (canonical, content)
                    # Also match if there's content directly after (no colon)
                    if rest and rest[0] in "=":
                        return (canonical, rest)
        return None

    def get_labeled(self, label: str) -> str | None:
        """Get the expression for a labeled section. Returns the LAST match."""
        if label in self._cache:
            return self._cache[label]

        result = None
        canonical = label.upper()

        for line in self.lines:
            match = self._match_label(line)
            if match and match[0] == canonical:
                expr = self._extract_expression(match[1])
                if expr:  # Take last non-empty match
                    result = expr

        self._cache[label] = result
        return result

    def get_equations(self) -> list[str]:
        """Get all equation expressions (dq/dt = ..., dp/dt = ...)."""
        if self._equations_cache is not None:
            return self._equations_cache

        equations = []
        in_equations_section = False

        for line in self.lines:
            match = self._match_label(line)
            if match:
                if match[0] == "EQUATIONS":
                    in_equations_section = True
                    # Check if there's content on the same line
                    if match[1]:
                        equations.append(self._clean_expression(match[1]))
                else:
                    in_equations_section = False
                continue

            # In EQUATIONS section, look for indented equations
            if in_equations_section:
                cleaned = self._clean_line(line)
                if cleaned and "=" in cleaned:
                    # Check for derivative patterns: dq/dt, dp/dt, \frac{dq}{dt}
                    lower = cleaned.lower()
                    if "d" in lower and ("dt" in lower or "/dt" in lower or "{dt}" in lower):
                        equations.append(self._clean_expression(cleaned))

        # Also scan entire text for derivative expressions outside EQUATIONS section
        # But skip lines that are labeled (those are other sections, not equations)
        for line in self.lines:
            # Skip if this line has a label
            if self._match_label(line):
                continue
            cleaned = self._clean_line(line)
            lower = cleaned.lower()
            # Match dX/dt = ... or \frac{dX}{dt} = ... (actual derivative equations)
            # Must start with d or \frac{d to be a Hamilton equation
            if "=" in cleaned:
                # Check for dq/dt or dp/dt pattern at start of expression
                if (lower.startswith("d") and "/dt" in lower) or (
                    "\\frac{d" in cleaned.lower() or "frac{d" in lower
                ):
                    expr = self._clean_expression(cleaned)
                    if expr not in equations:
                        equations.append(expr)

        self._equations_cache = equations
        return equations

    def get_all_expressions(self) -> dict[str, list[str]]:
        """Get all 'VAR = expr' patterns, grouped by variable name.

        Also captures derivative expressions like dq/dt = ..., dp/dt = ...
        which get keyed as 'dq/dt', 'dp/dt', etc.
        """
        if self._all_expressions_cache is not None:
            return self._all_expressions_cache

        results: dict[str, list[str]] = {}

        for line in self.lines:
            cleaned = self._clean_line(line)
            if "=" not in cleaned:
                continue

            # Split on first =
            parts = cleaned.split("=", 1)
            if len(parts) != 2:
                continue

            lhs = parts[0].strip()
            rhs = parts[1].strip()

            # Skip if RHS looks like prose
            prose_words = {
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
            }
            rhs_words = rhs.lower().split()
            if any(w in prose_words for w in rhs_words[:3]):
                continue

            # Skip empty or very short expressions
            if len(rhs) < 1:
                continue

            # Check for derivative expressions: dq/dt, dp/dt, \frac{dq}{dt}
            lower_lhs = lhs.lower()
            if "/dt" in lower_lhs or "{dt}" in lower_lhs:
                # Normalize derivative key to "dX/dt" format
                # Handle various patterns
                for var in ["q", "p", "x", "y", "r", "s", "theta", "p_theta", "p_r"]:
                    if f"d{var}" in lower_lhs or var in lower_lhs:
                        key = f"d{var}/dt"
                        expr = self._clean_expression(rhs)
                        if expr:
                            results.setdefault(key, []).append(expr)
                        break
                continue

            # Extract variable name (last word/symbol before =)
            # Handle cases like "T", "p", "H = ...", "KINETIC: T = ..."
            var_parts = lhs.split()
            if var_parts:
                var = var_parts[-1].strip("*:$")
                # Filter to single letters or known physics variables
                if len(var) <= 3 or var.lower() in {
                    "theta",
                    "omega",
                    "alpha",
                    "p_theta",
                    "p_x",
                    "p_y",
                }:
                    expr = self._clean_expression(rhs)
                    if expr:
                        results.setdefault(var, []).append(expr)

        self._all_expressions_cache = results
        return results

    def _find_final_output_boundary(self) -> int:
        """Find the character position where FINAL OUTPUT begins.

        The model is instructed to "always end with these labeled lines".
        FINAL OUTPUT is the LAST contiguous block of labeled sections.

        Strategy: scan backwards from end to find where labeled sections cluster.
        The boundary is the START of the LAST labeled section block.

        Returns character position, or 0 if no labeled sections found.
        """
        # Collect all labeled section positions
        labeled_positions: list[tuple[int, int, str]] = []  # (line_idx, char_start, label)
        char_pos = 0
        for i, line in enumerate(self.lines):
            line_start = char_pos
            match = self._match_label(line)
            if match:
                labeled_positions.append((i, line_start, match[0]))
            char_pos += len(line) + 1  # +1 for newline

        if not labeled_positions:
            return 0  # No labels found — everything is reasoning

        # Find the LAST contiguous block of labeled sections
        # "Contiguous" means labels appear within N lines of each other
        MAX_GAP = 5  # Allow up to 5 lines between labeled sections in final output

        # Work backwards: find where the final block starts
        final_block_start_idx = len(labeled_positions) - 1
        for i in range(len(labeled_positions) - 1, 0, -1):
            prev_line = labeled_positions[i - 1][0]
            curr_line = labeled_positions[i][0]
            if curr_line - prev_line > MAX_GAP:
                # Gap too large — this is where the final block starts
                final_block_start_idx = i
                break
        else:
            # All labels are in one contiguous block
            final_block_start_idx = 0

        # The boundary is the char position of the first label in the final block
        return labeled_positions[final_block_start_idx][1]

    def get_section_spans(self) -> dict[str, list[tuple[int, int]]]:
        """Get character spans for each labeled section in FINAL OUTPUT only.

        Returns:
            {canonical_label: [(char_start, char_end), ...], ...}

        CRITICAL: Only spans from the FINAL OUTPUT region are returned.
        Reasoning text (before the final output boundary) is EXCLUDED to prevent
        training signal from tentative expressions like "H = p²/4 maybe?".

        Finds in FINAL OUTPUT only:
        1. Labeled sections (HAMILTONIAN: ..., KINETIC: ..., etc.)
        2. VAR = expr patterns (H = ..., V = ..., etc.)
        3. Equation patterns (dX/dt = ..., frac{dX}{dt} = ...)
        """
        spans: dict[str, list[tuple[int, int]]] = {
            "COORDINATES": [],
            "MOMENTUM": [],
            "KINETIC": [],
            "POTENTIAL": [],
            "HAMILTONIAN": [],
            "EQUATIONS": [],
        }

        # Find where FINAL OUTPUT begins — everything before is reasoning
        final_output_boundary = self._find_final_output_boundary()

        char_pos = 0
        current_label: str | None = None
        current_start: int = 0

        for i, line in enumerate(self.lines):
            line_start = char_pos
            line_end = char_pos + len(line)

            # SKIP lines before the final output boundary
            if line_start < final_output_boundary:
                char_pos = line_end + 1
                continue

            match = self._match_label(line)
            if match:
                # Close previous section
                if current_label and current_start < line_start:
                    spans[current_label].append((current_start, line_start))

                # Start new section
                current_label = match[0]
                current_start = line_start

            # Also find VAR = expr patterns in the line (FINAL OUTPUT only)
            # Skip if line was already matched as a labeled section (avoid duplicates)
            cleaned = self._clean_line(line)
            if "=" in cleaned and not match:
                # Find VAR = patterns by looking for single-letter vars followed by =
                for var, label in [
                    ("H", "HAMILTONIAN"),
                    ("V", "POTENTIAL"),
                    ("T", "KINETIC"),
                    ("p", "MOMENTUM"),
                ]:
                    # Look for "VAR =" or "VAR=" pattern anywhere in the line
                    # Use exact string search, not regex
                    idx = 0
                    while idx < len(cleaned):
                        # Find next occurrence of var
                        found = cleaned.find(var, idx)
                        if found == -1:
                            break
                        # Check if followed by = (possibly with spaces)
                        after = cleaned[found + len(var) :].lstrip()
                        if after.startswith("="):
                            # Check it's a standalone var (not part of a word)
                            is_standalone = found == 0 or not cleaned[found - 1].isalnum()
                            if is_standalone:
                                spans[label].append((line_start, line_end))
                                break  # One match per line per var is enough
                        idx = found + 1
                    else:
                        continue
                    break  # Found a match for this var, stop checking other vars

                # Check for equation patterns: dX/dt = ... or frac{dX}{dt} = ...
                lower = cleaned.lower()
                has_derivative = False
                # Check for dX/dt pattern
                if "d" in lower:
                    d_idx = lower.find("d")
                    while d_idx >= 0 and d_idx < len(lower) - 3:
                        # Look for /dt after d + letter
                        if d_idx + 1 < len(lower) and lower[d_idx + 1].isalpha():
                            rest = lower[d_idx + 2 :]
                            if "/dt" in rest:
                                has_derivative = True
                                break
                        d_idx = lower.find("d", d_idx + 1)
                # Check for frac{dX}{dt} pattern
                if "frac{d" in lower and "dt}" in lower:
                    has_derivative = True

                if has_derivative:
                    spans["EQUATIONS"].append((line_start, line_end))

            char_pos = line_end + 1  # +1 for newline

        # Close final section
        if current_label and current_start < len(self.text):
            spans[current_label].append((current_start, len(self.text)))

        return spans


# ─── Parser factory (for consistent usage) ────────────────────────────────────


def parse_structured_output(text: str) -> StructuredOutputParser:
    """Create a parser for structured Hamiltonian output."""
    return StructuredOutputParser(text)


# Common physics degree → radian mappings (shared by parser and normalizer)
_COMMON_DEGREES = {"30": "pi/6", "45": "pi/4", "60": "pi/3", "90": "pi/2"}
_COMMON_DEGREES_LATEX = {
    "30": r"\frac{\pi}{6}",
    "45": r"\frac{\pi}{4}",
    "60": r"\frac{\pi}{3}",
    "90": r"\frac{\pi}{2}",
}

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
        else:
            return match.group(0)  # Return original unchanged

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
    # LaTeX fractions: \frac{a}{b} → (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
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


def _parse_math(expr_str: str) -> sp.Basic | None:
    """Parse a math expression (LaTeX or plain) to sympy. Primary parser.

    Tries latex2sympy first (handles \\frac, \\sin, \\dot, derivatives, etc.),
    then falls back to sympify with normalization for plain algebra.
    """
    if not expr_str or expr_str.strip() == "":
        return None

    raw = expr_str.strip()
    # Strip wrapping $ delimiters for latex2sympy
    cleaned = re.sub(r"^\$+|\$+$", "", raw).strip()

    # Pre-process degree notation before any parser (neither handles it correctly)
    cleaned = re.sub(
        r"(\d+)\^\s*\\circ", lambda m: _COMMON_DEGREES_LATEX.get(m.group(1), m.group(1)), cleaned
    )
    cleaned = re.sub(
        r"(\d+)\^\s*\{\\circ\}",
        lambda m: _COMMON_DEGREES_LATEX.get(m.group(1), m.group(1)),
        cleaned,
    )

    # Try latex2sympy first — handles all LaTeX natively
    if _latex2sympy is not None and "\\" in cleaned:
        # Ensure bare trig names have LaTeX prefix (latex2sympy treats "sin" as s*i*n)
        # Only do this when we're already in the latex2sympy path (has backslashes)
        latex_cleaned = cleaned
        for fn in ("sin", "cos", "tan", "exp", "log", "ln", "sqrt"):
            latex_cleaned = re.sub(rf"(?<!\\)\b{fn}\s*\(", rf"\\{fn}(", latex_cleaned)
        try:
            result = _latex2sympy(latex_cleaned)
            # Only accept sp.Expr — the one type safe for .subs()/.simplify()/etc.
            # Rejects: Equality (sp.Rel), And/Or (sp.Boolean), BooleanAtom, custom types.
            # Known issue: latex2sympy returns And() for chained equalities like
            # "T = p²/2m = p²/6" which crashes .subs() (HuggingFace Math-Verify #24).
            # Validate output before passing to fallback normalizer
            if isinstance(result, sp.Expr):
                return result
        except Exception as exc:
            logger.debug("latex2sympy failed on %r: %s", cleaned, exc)

    # Fallback: normalize and sympify (plain algebra like p**2/6, 3*x**2)
    # NOTE: degree handling in _normalize_for_sympy is the fallback for when
    # _parse_math's latex2sympy path fails — do not remove it.
    normed = ""
    try:
        normed = _normalize_for_sympy(expr_str)
        result = sp.sympify(normed, locals=_SYMPY_LOCALS)
        if isinstance(result, sp.Expr):
            return result
        return None
    except Exception as exc:
        logger.debug("sympify fallback failed on %r (normalized: %r): %s", expr_str, normed, exc)
        return None


# ─── Sympy helpers ────────────────────────────────────────────────────────────

_SYMPY_LOCALS = {
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
    "l": sp.Symbol("l", positive=True),  # length
    "g": sp.Rational(98, 10),
    "pi": sp.pi,
}


def _extract_constants_from_prompt(prompt: str) -> dict:
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


def _try_sympify(expr_str: str) -> sp.Basic | None:
    """Try to parse expression string into sympy. Uses _parse_math (latex2sympy + fallback).

    Returns None if parsing fails. Callers MUST check for None before calling .subs() or other methods.
    """
    result = _parse_math(expr_str)
    if result is None:
        logger.debug("_parse_math returned None for: %r", expr_str)
    return result


def _remap_variables(student: sp.Basic, teacher: sp.Basic) -> sp.Basic:
    """Remap student variables to match teacher's coordinate AND momentum names.

    The model may use x where ground truth uses y, or p where ground truth
    uses p_theta. Both are valid — the names are arbitrary. This function
    finds a consistent mapping from student symbols to teacher symbols.

    Remaps coordinates (x, y, s, r, q, theta, ...) AND momenta (p, p_theta, ...).
    Does NOT remap constants (m, g, k).
    """
    # Only sp.Expr supports .subs() safely. Reject Equality, And, Boolean, etc.
    if not isinstance(student, sp.Expr) or not isinstance(teacher, sp.Expr):
        return student

    _COORDS = {"x", "y", "s", "r", "q", "theta", "theta1", "theta2", "x1", "x2"}
    _MOMENTA = {"p", "p_x", "p_y", "p_r", "p_s", "p_theta", "p1", "p2"}
    _REMAP = _COORDS | _MOMENTA
    _CONSTANTS = {"m", "g", "k", "pi", "omega", "alpha", "beta", "gamma", "delta"}

    student_remap = {s for s in student.free_symbols if s.name in _REMAP}
    teacher_remap = {s for s in teacher.free_symbols if s.name in _REMAP}

    # If all teacher symbols already present in student, no remapping needed
    if teacher_remap <= student_remap:
        return student

    # Split into coords and momenta for independent remapping
    student_coords = {s for s in student_remap if s.name in _COORDS}
    teacher_coords = {s for s in teacher_remap if s.name in _COORDS}
    student_momenta = {s for s in student_remap if s.name in _MOMENTA}
    teacher_momenta = {s for s in teacher_remap if s.name in _MOMENTA}

    subs = {}

    # Remap coordinates
    if len(student_coords) == len(teacher_coords) and student_coords != teacher_coords:
        s_sorted = sorted(student_coords, key=lambda s: s.name)
        t_sorted = sorted(teacher_coords, key=lambda s: s.name)
        for sv, tv in zip(s_sorted, t_sorted, strict=False):
            if sv != tv:
                subs[sv] = tv

    # Remap momenta (model uses p, ground truth uses p_theta, p_s, etc.)
    if len(student_momenta) == len(teacher_momenta) and student_momenta != teacher_momenta:
        s_sorted = sorted(student_momenta, key=lambda s: s.name)
        t_sorted = sorted(teacher_momenta, key=lambda s: s.name)
        for sv, tv in zip(s_sorted, t_sorted, strict=False):
            if sv != tv:
                subs[sv] = tv

    if subs:
        return student.subs(subs)

    return student


def _score_terms_structurally(student: sp.Basic, teacher: sp.Basic) -> float | None:
    """Score expression by matching polynomial terms structurally.

    Returns fraction of teacher terms matched (same monomial, coefficient within 10%).
    Returns None if expressions aren't polynomial-like (trig, exp, etc.).

    This catches missing terms that numerical closeness misses:
    - V = 3x² + 19.6x (teacher) vs V = 19.6x (student)
    - Numerical: student/teacher ≈ 0.94 at x=0.4 → 0.85 partial credit
    - Structural: 1/2 terms matched → 0.50 partial credit
    """
    try:
        # Expand to standard polynomial form
        student_exp = sp.expand(student)
        teacher_exp = sp.expand(teacher)

        # Get terms as {monomial_base: coefficient} dict
        # e.g., 3*x**2 + 19.6*x → {x**2: 3, x: 19.6}
        def get_terms(expr):
            terms = {}
            if expr.is_Add:
                for term in expr.args:
                    coeff, base = term.as_coeff_Mul()
                    if base == 1:  # constant term
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
                return None  # Not a simple polynomial
            return terms

        teacher_terms = get_terms(teacher_exp)
        student_terms = get_terms(student_exp)

        if teacher_terms is None or student_terms is None:
            return None
        if not teacher_terms:
            return None

        # Match terms: same base, coefficient within 10%
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
                    if 0.9 <= ratio <= 1.1:  # Within 10%
                        matched += 1
                        coeff_accuracy_sum += 1.0 - abs(1.0 - ratio)
                    elif 0.5 <= ratio <= 2.0:  # Within 2x — partial term credit
                        matched += 0.5
                        coeff_accuracy_sum += 0.5 * (1.0 - min(abs(1.0 - ratio), 1.0))

        # Score = (matched_terms / total_terms) * avg_coefficient_accuracy
        if len(teacher_terms) == 0:
            return None  # Avoid division by zero
        term_coverage = matched / len(teacher_terms)
        avg_coeff_acc = coeff_accuracy_sum / max(matched, 1)

        return term_coverage * avg_coeff_acc

    except Exception:
        return None


def _score_expression(
    student_str: str | None,
    teacher_str: str,
    variables: list[str],
    constant_subs: dict | None = None,
) -> float:
    """Score a mathematical expression against ground truth.

    Args:
        constant_subs: optional sympy substitution dict (from _extract_constants_from_prompt)
            to evaluate symbolic constants (m, k, F) in student expression before comparing.

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
    # Many derivations show work via "expr1 = expr2 = ... = final". We want the final.
    if "=" in student_str:
        parts = [p.strip() for p in student_str.split("=")]
        # Try the last part first (most simplified form)
        student = _try_sympify(parts[-1])
        # If that fails, try the first part (may be more explicit)
        if student is None and len(parts) > 1:
            student = _try_sympify(parts[0])
    else:
        student = _try_sympify(student_str)
    teacher = _try_sympify(teacher_str)

    # Substitute known constants (m, k, F from prompt) into student expression
    if constant_subs and student is not None and isinstance(student, sp.Expr):
        try:
            student = student.subs(constant_subs)
        except Exception as exc:
            logger.debug("constant substitution failed on %s: %s", student, exc)

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
        # Exact symbolic match — try multiple simplification strategies with timeout
        for simplifier in [sp.simplify, sp.trigsimp, sp.ratsimp, sp.nsimplify]:
            try:
                with sympy_timeout(2):
                    if simplifier(candidate - teacher) == 0:
                        return 1.0
            except (Exception, SympyTimeoutError):
                continue

        try:
            with sympy_timeout(2):
                if sp.simplify(sp.expand(candidate) - sp.expand(teacher)) == 0:
                    return 1.0
        except (Exception, SympyTimeoutError):
            pass

        try:
            with sympy_timeout(2):
                if sp.trigsimp(sp.expand_trig(candidate - teacher)) == 0:
                    return 1.0
        except (Exception, SympyTimeoutError):
            pass

        # Sign convention check: student = -teacher is valid physics (different reference frame)
        for simplifier in [sp.simplify, sp.expand]:
            try:
                with sympy_timeout(2):
                    if simplifier(candidate + teacher) == 0:
                        return 0.8  # Correct magnitude, opposite sign — partial credit
            except (Exception, SympyTimeoutError):
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

    # Partial credit: structural term matching (robust) > numerical closeness (fallback)
    # Structural matching catches missing terms that numerical closeness misses.
    try:
        structural_score = _score_terms_structurally(candidate, teacher)
        if structural_score is not None:
            # Scale structural score: 0.0 (no terms) to 0.7 (all terms, slight coeff error)
            # Missing terms get proportionally lower scores (e.g., 1/2 terms → 0.35 max)
            return 0.2 + 0.5 * structural_score

        # Fallback: numerical closeness for non-polynomial expressions (trig, exp, etc.)
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
            except Exception:
                pass

        student_syms = candidate.free_symbols
        teacher_syms = teacher.free_symbols
        if not teacher_syms:
            sym_overlap = 0.0 if student_syms else numerical_score
        else:
            sym_overlap = len(student_syms & teacher_syms) / len(teacher_syms)

        # Cap at 0.70 — partial credit should not exceed mastery threshold
        partial = 0.2 + 0.4 * numerical_score + 0.1 * sym_overlap
        return min(partial, 0.70)
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

# Label aliases imported from qgre.segments (single source of truth)
_LABEL_ALIASES = HAMILTONIAN_LABEL_ALIASES


def _extract_labeled(text: str, label: str) -> str | None:
    """Extract the value after a labeled line like 'KINETIC: T = ...'

    Uses the StructuredOutputParser for robust line-based extraction.
    Takes the LAST match — the model writes derivation headers early
    and structured labels at the end. We want the final labeled answer.
    """
    parser = parse_structured_output(text)
    return parser.get_labeled(label)


# ─── Section-agnostic expression finder ──────────────────────────────────────


def _find_all_expressions(text: str) -> dict[str, list[str]]:
    """Extract ALL 'X = expr' patterns from the text, grouped by variable name.

    Uses the StructuredOutputParser for robust line-based extraction.
    Returns dict mapping variable letter (T, V, H, p, q, etc.) to list of
    expression strings found.
    """
    parser = parse_structured_output(text)
    return parser.get_all_expressions()


def _best_match(
    candidates: list[str],
    teacher_str: str,
    variables: list[str] | None = None,
    constant_subs: dict | None = None,
) -> float:
    """Score the best-matching candidate expression against ground truth.

    Tries each candidate, returns the highest score. This is the core of
    section-agnostic scoring — instead of requiring the expression to be
    in a specific labeled section, we find it anywhere in the text.
    """
    if not candidates:
        return 0.0
    best = 0.0
    for expr in candidates:
        score = _score_expression(expr, teacher_str, variables or [], constant_subs=constant_subs)
        best = max(best, score)
        if best >= 1.0:
            break
    return best


def _extract_equations_block(text: str) -> list[str]:
    """Extract RHS of Hamilton's equations from EQUATIONS: block.

    Uses the StructuredOutputParser for robust extraction.
    Returns the RIGHT-HAND SIDE only (e.g., "p/2" not "dq/dt = p/2").
    """
    parser = parse_structured_output(text)
    equations = parser.get_equations()

    # Extract RHS from each equation
    rhs_list = []
    for eq in equations:
        if "=" in eq:
            rhs = eq.split("=", 1)[-1].strip()
            if rhs:
                rhs_list.append(rhs)
        else:
            # If no =, assume it's already the RHS
            rhs_list.append(eq)
    return rhs_list


def _extract_H(text: str) -> str | None:
    """Extract Hamiltonian expression."""
    parser = parse_structured_output(text)

    # Try labeled extraction first
    result = parser.get_labeled("HAMILTONIAN")
    if result:
        return result

    # Fallback: find H in all expressions
    all_exprs = parser.get_all_expressions()
    h_candidates = all_exprs.get("H", [])
    if h_candidates:
        return h_candidates[-1]  # Return last match

    return None


def _extract_numbers_from_prompt(prompt: str) -> set[str]:
    nums = set()
    for m in re.finditer(r"[=]\s*(\d+(?:\.\d+)?)", prompt):
        nums.add(m.group(1))
    for m in re.finditer(
        r"(?:mass|constant|length|radius|velocity|charge|field)\s+.*?(\d+(?:\.\d+)?)", prompt
    ):
        nums.add(m.group(1))
    return nums


# ─── Quality scorers ──────────────────────────────────────────────────────────


def _score_format(text: str) -> float:
    """q_format: structured response with labeled sections."""
    parser = parse_structured_output(text)

    # Count how many expected labels are present
    labels_found = 0
    for label in ["COORDINATES", "MOMENTUM", "KINETIC", "POTENTIAL", "HAMILTONIAN", "EQUATIONS"]:
        if parser.get_labeled(label) is not None:
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
    count = sum(1 for p in indicators if p.lower() in text.lower())
    return min(1.0, count / 3)


def _score_momentum_defined(text: str) -> float:
    """q_momentum_defined: MOMENTUM section defines p in terms of q̇."""
    parser = parse_structured_output(text)

    # Check for MOMENTUM: label with content
    momentum_str = parser.get_labeled("MOMENTUM")

    if momentum_str:
        # Check if expression has numbers and variables
        has_numbers = any(c.isdigit() for c in momentum_str)
        has_expression = any(c.isalpha() or c in "*/+-" for c in momentum_str)
        if has_numbers and has_expression:
            return 1.0
        if has_expression:
            return 0.7
        return 0.5

    # Fallback: check for momentum definition in all expressions
    all_exprs = parser.get_all_expressions()
    p_candidates = all_exprs.get("p", [])
    if p_candidates:
        # Check if any p = ... expression contains mass * velocity pattern
        for expr in p_candidates:
            lower = expr.lower()
            if "d" in lower or any(c.isdigit() for c in expr):
                return 0.5

    # Check for "conjugate momentum" text
    if "conjugate momentum" in text.lower():
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
        score = _best_match(p_candidates, expected_T, constant_subs=meta.get("_constant_subs"))
        return max(0.7, score)  # At least 0.7 for having p form

    # No T = p... found. Check if ANY expression in the text matches ground truth T
    # (model might write it as part of H derivation without a separate T = line)
    if t_candidates:
        # T expressions exist but none have p — velocity form
        return 0.0

    return 0.0  # No T expression found at all


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
        parsed = _parse_math(expr_str)
        if parsed is not None:
            if parsed.atoms(sp.Derivative):
                return True
            if any(s.name.startswith("dot{") for s in parsed.free_symbols):
                return True
    return False


def _has_momentum_var(expr_str: str) -> bool:
    """Check if expression contains momentum variable p."""
    return bool(re.search(r"p[_²/(*+\-]|p\*\*|p\^|p_[a-z]|\bp\d|\bp\b", expr_str))


def _score_in_momentum_form(text: str, var_name: str) -> float:
    """Check if variable is expressed in momentum form (p) vs velocity form.

    Section-agnostic: finds ALL 'VAR = expr' anywhere in the text.
    Returns 1.0 if ANY expression contains p without velocity markers.
    """
    try:
        all_exprs = _find_all_expressions(text)
        candidates = all_exprs.get(var_name, [])
        if not candidates:
            return 0.0

        for expr in candidates:
            has_p = _has_momentum_var(expr)
            has_velocity = _has_velocity_form(expr)
            if has_p and not has_velocity:
                return 1.0
            if has_p and has_velocity:
                return 0.5

        return 0.0
    except Exception as exc:
        logger.debug("_score_%s_in_momentum failed: %s", var_name, exc)
        return 0.0


def _score_T_in_momentum(text: str) -> float:
    return _score_in_momentum_form(text, "T")


def _score_H_in_momentum(text: str) -> float:
    return _score_in_momentum_form(text, "H")


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

    # Score best candidate and track which one scored highest
    score = 0.0
    best_v_candidate = v_candidates[0]
    for cand in v_candidates:
        s = _score_expression(cand, expected_V, [], constant_subs=meta.get("_constant_subs"))
        if s > score:
            score = s
            best_v_candidate = cand
        if score >= 1.0:
            break
    if score >= 0.9:
        return score

    # Fallback: model may write symbolic form (mgy, Fx, kx²) while ground truth
    # has constants evaluated (49*y/5, -3*x, 3*x**2). Try substituting known
    # constants from the problem into the student expression via sympy.
    # Uses centralized _constant_subs (from _extract_constants_from_prompt) to avoid
    # symbol assumption mismatches from duplicate extraction.
    potential_str = best_v_candidate
    student_sym = _try_sympify(potential_str)
    teacher_sym = _try_sympify(expected_V)
    subs = meta.get("_constant_subs", {})
    if student_sym is not None and teacher_sym is not None and subs:
        try:
            student_eval = student_sym.subs(subs)
            diff = sp.simplify(student_eval - teacher_sym)
            if diff == 0:
                return 1.0
            free = (student_eval.free_symbols | teacher_sym.free_symbols) - {sp.Symbol("pi")}
            if free:
                test_point = {s: sp.Rational(3, 7) for s in free}
                s_val = float(student_eval.subs(test_point))
                t_val = float(teacher_sym.subs(test_point))
                if abs(t_val) > 1e-10 and abs(s_val - t_val) < 1e-4 * max(abs(t_val), 1):
                    return 1.0
        except Exception as exc:
            logger.debug("V_correct sympy substitution failed: %s", exc)

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


def _score_equation(text: str, meta: dict, meta_key: str, fallback_pattern: str) -> float:
    """Score a Hamilton equation (dq/dt or dp/dt) against ground truth.

    Section-agnostic: finds ALL derivative expressions anywhere in the text.
    """
    expected = meta.get(meta_key, "")
    if not expected or expected == "none":
        return 0.0

    expected_parts = [e.strip() for e in expected.split(";")]

    # Section-agnostic: find all derivative expressions + block extraction
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
        if re.search(fallback_pattern, text):
            return 0.2
        return 0.0

    scores = []
    for exp_part in expected_parts:
        best = 0.0
        for ext in extracted:
            score = _score_expression(ext, exp_part, [], constant_subs=meta.get("_constant_subs"))
            best = max(best, score)
        scores.append(best)

    return sum(scores) / len(scores) if scores else 0.0


def _score_dqdt(text: str, meta: dict) -> float:
    return _score_equation(text, meta, "dqdt", r"dq/dt|∂H/∂p|dx/dt|dtheta/dt|dr/dt|ds/dt")


def _score_dpdt(text: str, meta: dict) -> float:
    return _score_equation(text, meta, "dpdt", r"dp/dt|-∂H/∂q|-dH/dq|dp_r/dt|dp_theta/dt")


def _score_correct_H(text: str, meta: dict) -> float:
    """q_correct_H: Hamiltonian matches ground truth.

    Section-agnostic: finds ALL 'H = expr' anywhere in the text,
    scores the best match against ground truth.
    """
    expected_H = meta.get("H_expr", "")
    # Check for sentinel value "none" or empty string before processing
    if not expected_H or expected_H == "none":
        return 0.0

    # Section-agnostic: find all H expressions
    all_exprs = _find_all_expressions(text)
    h_candidates = all_exprs.get("H", [])

    # Also try label-based extraction as fallback
    extracted_H = _extract_H(text)
    if extracted_H and extracted_H not in h_candidates:
        h_candidates.append(extracted_H)

    # Split chained equalities: "T + V = p²/2 + 0 = p²/2" → also try "p²/2" (last RHS)
    expanded = []
    for cand in h_candidates:
        expanded.append(cand)
        if "=" in cand:
            last_rhs = cand.rsplit("=", 1)[-1].strip()
            if last_rhs and last_rhs not in expanded:
                expanded.append(last_rhs)
    h_candidates = expanded

    if not h_candidates:
        return 0.0

    # Score best candidate and track which one it was
    best = 0.0
    best_candidate = h_candidates[0]
    for cand in h_candidates:
        score = _score_expression(cand, expected_H, [], constant_subs=meta.get("_constant_subs"))
        if score > best:
            best = score
            best_candidate = cand
        if best >= 1.0:
            break
    if best >= 0.7:
        return best

    # Use the BEST candidate (not first) for detailed analysis below
    extracted_H = best_candidate

    # Check if model evaluated to a number instead of keeping symbolic
    normed = _normalize_for_sympy(extracted_H)
    parsed = _try_sympify(extracted_H)
    if parsed is not None and isinstance(parsed, sp.Basic) and parsed.is_number:
        return 0.1  # Distinct signal: "you plugged in numbers, keep it symbolic"

    # Try direct match first
    direct_score = _score_expression(
        extracted_H, expected_H, [], constant_subs=meta.get("_constant_subs")
    )
    if direct_score >= 0.7:
        return direct_score

    # Check if it's velocity form of the correct H — contains derivatives or _VDOT_
    # latex2sympy produces Derivative objects, _normalize_for_sympy produces _VDOT_ markers
    has_velocity = "_VDOT_" in normed or (parsed is not None and parsed.atoms(sp.Derivative))
    if has_velocity:
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
    coord_list = [c.strip() for c in coords.split(",") if c.strip()]
    # Return early with partial credit if coord_list is empty
    if not coord_list:
        return 0.2

    extracted_eqs = _extract_equations_block(text)
    if not extracted_eqs:
        return 0.3

    # Find the actual momentum symbols used in H (don't guess from coordinate names)
    h_sym_names = {s.name for s in H_sym.free_symbols}

    consistency_scores = []
    for coord in coord_list:
        # Try candidate momentum names in priority order
        candidates = [
            f"p_{coord}",
            "p",
            coord.replace("x", "p") if coord.startswith("x") else None,
            f"p{coord[-1]}" if coord[-1].isdigit() else None,
        ]
        candidates = [c for c in candidates if c is not None]
        p_name = next((c for c in candidates if c in h_sym_names), candidates[0])

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
                # CFG-R1-3: Use .get() consistently to avoid KeyError
                h_score = _score_expression(
                    extracted_H,
                    meta.get("H_expr", ""),
                    [],
                    constant_subs=meta.get("_constant_subs"),
                )
                component_scores.append(h_score)

        # ── Score equations ──────────────────────────────────────────────────
        eq_score: float | None = None
        if has_dqdt or has_dpdt:
            extracted_eqs = _extract_equations_block(text)
            eq_component_scores: list[float] = []

            if extracted_eqs:
                any_extracted = True

                csubs = meta.get("_constant_subs")
                if has_dqdt:
                    dqdt_str = meta.get("dqdt", "")
                    if dqdt_str:
                        expected_parts = [e.strip() for e in dqdt_str.split(";") if e.strip()]
                        part_scores = []
                        for exp_part in expected_parts:
                            best = max(
                                (
                                    _score_expression(ext, exp_part, [], constant_subs=csubs)
                                    for ext in extracted_eqs
                                ),
                                default=0.0,
                            )
                            part_scores.append(best)
                        if part_scores:
                            eq_component_scores.append(sum(part_scores) / len(part_scores))

                if has_dpdt:
                    # CFG-R1-3: Use .get() consistently to avoid KeyError
                    dpdt_str = meta.get("dpdt", "")
                    if dpdt_str:
                        expected_parts = [e.strip() for e in dpdt_str.split(";") if e.strip()]
                        part_scores = []
                        for exp_part in expected_parts:
                            best = max(
                                (
                                    _score_expression(ext, exp_part, [], constant_subs=csubs)
                                    for ext in extracted_eqs
                                ),
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
    except Exception as exc:
        logger.debug("_score_correct_coefficient failed: %s", exc)
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
                score = _score_expression(
                    ext, exp_part, [], constant_subs=meta.get("_constant_subs")
                )
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

    except Exception as exc:
        logger.debug("_score_derivative_correct failed: %s", exc)
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


# ─── Span finder (runs on RAW text, positions valid for char→token mapping) ───


def _find_expression_spans(text: str) -> dict[str, list[tuple[int, int]]]:
    """Find character spans of scored expression patterns in RAW completion text.

    Uses the StructuredOutputParser for exact string matching — NO REGEX.
    Runs on the ORIGINAL text with NO cleaning so character offsets are valid
    for the char→token mapping in qgre/spans.py.

    Returns quality_name → [(char_start, char_end), ...].
    """
    parser = StructuredOutputParser(text)
    section_spans = parser.get_section_spans()

    spans: dict[str, list[tuple[int, int]]] = {}

    # Map section spans to quality names
    spans["q_correct_H"] = section_spans.get("HAMILTONIAN", [])
    spans["q_V_correct"] = section_spans.get("POTENTIAL", [])

    t_spans = section_spans.get("KINETIC", [])
    spans["q_T_uses_p"] = t_spans
    spans["q_T_in_momentum"] = t_spans  # Same spans, different quality

    spans["q_H_in_momentum"] = spans["q_correct_H"]

    eq_spans = section_spans.get("EQUATIONS", [])
    spans["q_correct_dqdt"] = eq_spans
    spans["q_correct_dpdt"] = eq_spans
    spans["q_consistency"] = eq_spans
    spans["q_correct_coefficient"] = eq_spans
    spans["q_derivative_correct"] = eq_spans

    momentum_spans = section_spans.get("MOMENTUM", [])
    spans["q_momentum_defined"] = momentum_spans
    spans["q_defines_momentum"] = momentum_spans

    # Format/has_math qualities: target labeled sections when present.
    # FALLBACK: If NO labeled sections found, span the FULL completion so the model
    # gets negative training signal for bad format. Without this, garbage output
    # (no labels at all) would get zero gradient.
    all_labeled_spans = []
    for label in ["COORDINATES", "MOMENTUM", "KINETIC", "POTENTIAL", "HAMILTONIAN", "EQUATIONS"]:
        all_labeled_spans.extend(section_spans.get(label, []))
    # Deduplicate and sort by start position
    all_labeled_spans = sorted(set(all_labeled_spans), key=lambda x: x[0])
    if all_labeled_spans:
        # Good structure found — train on labeled sections only
        spans["q_format"] = all_labeled_spans
        spans["q_has_math"] = all_labeled_spans
    elif len(text) > 0:
        # No structure found — train on full completion so bad output gets negative signal
        # Skip if text is empty (zero-width span is meaningless for training)
        spans["q_format"] = [(0, len(text))]
        spans["q_has_math"] = [(0, len(text))]
    else:
        # Empty text — no spans
        spans["q_format"] = []
        spans["q_has_math"] = []

    # Grounding: target sections with numerical values (T, V, H, equations)
    grounding_spans = []
    for label in ["KINETIC", "POTENTIAL", "HAMILTONIAN", "EQUATIONS"]:
        grounding_spans.extend(section_spans.get(label, []))
    spans["q_grounding"] = sorted(set(grounding_spans), key=lambda x: x[0])

    return spans


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
    import copy

    meta = copy.deepcopy(metadata) if metadata else {}  # Deep copy to avoid mutation
    meta["prompt"] = prompt  # Make prompt available to scoring functions
    # Extract physical constants from prompt once — shared by all scorers
    meta["_constant_subs"] = _extract_constants_from_prompt(prompt)
    text = completion
    scores: dict[str, float] = {}

    # Find expression spans on RAW text (before any scoring/cleaning)
    expression_spans = _find_expression_spans(text)

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

    return RewardResult(reward=total, scores=scores, scored_spans=expression_spans)
