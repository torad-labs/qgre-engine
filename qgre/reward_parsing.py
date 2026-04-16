"""Structured output parsing utilities for Hamiltonian reward functions.

Provides line-based parsing of model outputs containing labeled physics sections
(COORDINATES, MOMENTUM, KINETIC, POTENTIAL, HAMILTONIAN, EQUATIONS), plus
expression extraction helpers used by reward scoring.
"""

from __future__ import annotations


# Common physics degree → radian mappings (shared by parser and normalizer)
_COMMON_DEGREES: dict[str, str] = {"30": "pi/6", "45": "pi/4", "60": "pi/3", "90": "pi/2"}
_COMMON_DEGREES_LATEX: dict[str, str] = {
    "30": r"\frac{\pi}{6}",
    "45": r"\frac{\pi}{4}",
    "60": r"\frac{\pi}{3}",
    "90": r"\frac{\pi}{2}",
}


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

    def __init__(self, text: str, labels: dict | None = None):
        self.text = text
        self.lines = text.split("\n")
        self._cache: dict[str, str | None] = {}
        self._equations_cache: list[str] | None = None
        self._all_expressions_cache: dict[str, list[str]] | None = None
        self._all_expressions_with_spans_cache: dict[str, list[tuple[str, int, int]]] | None = None
        if labels is not None:
            self._labels = labels
        else:
            self._labels = self.__class__.LABELS

    def _clean_line(self, line: str) -> str:
        """Strip markdown artifacts from a line."""
        s = line.strip()
        # Strip markdown bold/italic markers from start
        while s.startswith(("*", "#")):
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

        for canonical, aliases in self._labels.items():
            all_names = {canonical.lower()} | aliases
            for name in all_names:
                # Check for "LABEL:" or "LABEL :" pattern
                if cleaned.startswith(name):
                    rest = cleaned[len(name) :].lstrip()
                    if rest.startswith((":", "=")):
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

    def get_all_expressions_with_spans(self) -> dict[str, list[tuple[str, int, int]]]:
        """Get all 'VAR = expr' patterns with character positions.

        Same extraction logic as get_all_expressions() but:
          - tracks character position during extraction
          - respects the FINAL OUTPUT boundary (excludes reasoning/scratch-work)

        Returns: {var_key: [(expr_str, char_start, char_end), ...]}

        The character range (char_start, char_end) covers the whole source line —
        sub-expression granularity is not tracked (scorers may split chained
        equalities; the span still points at the source line that contained the
        expression).
        """
        if self._all_expressions_with_spans_cache is not None:
            return self._all_expressions_with_spans_cache

        final_output_boundary = self._find_final_output_boundary()

        results: dict[str, list[tuple[str, int, int]]] = {}
        char_pos = 0

        for line in self.lines:
            line_start = char_pos
            line_end = char_pos + len(line)
            char_pos = line_end + 1  # +1 for newline

            # Skip lines before FINAL OUTPUT — scoring scratch-work is a leak
            if line_start < final_output_boundary:
                continue

            cleaned = self._clean_line(line)
            if "=" not in cleaned:
                continue

            parts = cleaned.split("=", 1)
            if len(parts) != 2:
                continue

            lhs = parts[0].strip()
            rhs = parts[1].strip()

            # Skip prose RHS (same filter as get_all_expressions)
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

            if len(rhs) < 1:
                continue

            # Derivative keys: normalize to "dX/dt" format
            lower_lhs = lhs.lower()
            if "/dt" in lower_lhs or "{dt}" in lower_lhs:
                for var in ["q", "p", "x", "y", "r", "s", "theta", "p_theta", "p_r"]:
                    if f"d{var}" in lower_lhs or var in lower_lhs:
                        key = f"d{var}/dt"
                        expr = self._clean_expression(rhs)
                        if expr:
                            results.setdefault(key, []).append((expr, line_start, line_end))
                        break
                continue

            # Standard VAR = expr (same var extraction as get_all_expressions)
            var_parts = lhs.split()
            if var_parts:
                var = var_parts[-1].strip("*:$")
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
                        results.setdefault(var, []).append((expr, line_start, line_end))

        self._all_expressions_with_spans_cache = results
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
        max_gap = 5  # Allow up to 5 lines between labeled sections in final output

        # Work backwards: find where the final block starts
        final_block_start_idx = len(labeled_positions) - 1
        for i in range(len(labeled_positions) - 1, 0, -1):
            prev_line = labeled_positions[i - 1][0]
            curr_line = labeled_positions[i][0]
            if curr_line - prev_line > max_gap:
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

        for _i, line in enumerate(self.lines):
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
                    ("p_x", "MOMENTUM"),
                    ("p_theta", "MOMENTUM"),
                    ("p_r", "MOMENTUM"),
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


def _normalize_unicode(text: str) -> str:
    """Normalize unicode artifacts (², ³, ·, ×)."""
    return (
        text.replace("\u00b2", "^2")
        .replace("\u00b3", "^3")
        .replace("\u00b7", "*")
        .replace("\u00d7", "*")
    )


# Alias used by extract_rhs_expressions (matches reward_fn_v2 naming)
_normalize_text = _normalize_unicode


def extract_rhs_expressions(
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
    normalized = _normalize_unicode(text)

    char_pos = 0
    for line in normalized.split("\n"):
        line_start = char_pos
        line_end = char_pos + len(line)
        char_pos = line_end + 1  # +1 for the \n

        if "=" not in line:
            continue

        # Strip ALL formatting from the line before splitting on =
        cleaned = line.strip()
        cleaned = cleaned.replace("$$", "").replace("$", "")
        cleaned = cleaned.lstrip("*#").rstrip("*").strip()
        cleaned = cleaned.lstrip("&").strip()

        if "=" not in cleaned or len(cleaned) < 3:
            continue

        parts = cleaned.split("=")
        if len(parts) < 2:
            continue

        lhs_raw = parts[0].strip()
        if ":" in lhs_raw:
            lhs_raw = lhs_raw.split(":")[-1].strip()
        lhs_raw = lhs_raw.lstrip("*#").rstrip("*").strip()
        tokens = lhs_raw.split()
        lhs = tokens[-1].strip("*$\\()") if tokens else ""

        if not lhs:
            continue

        for rhs_raw in parts[1:]:
            rhs = rhs_raw.strip()
            if not rhs or len(rhs) < 1:
                continue
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
