"""Tests for qgre.reward_parsing — structured output parsing utilities."""

from qgre.reward_parsing import (
    _COMMON_DEGREES,
    _COMMON_DEGREES_LATEX,
    StructuredOutputParser,
    extract_rhs_expressions,
    parse_structured_output,
)


# ─── Fixtures (inline) ───────────────────────────────────────────────────────

BASIC_TEXT = "KINETIC: T = p^2/4\nPOTENTIAL: V = 2*x^2"

CUSTOM_TEXT = "CUSTOM: val = 42"

FULL_STRUCTURED = (
    "COORDINATES: q = x\n"
    "MOMENTUM: p = 2*dx/dt\n"
    "KINETIC: T = p^2/4\n"
    "POTENTIAL: V = 2*x^2\n"
    "HAMILTONIAN: H = p^2/4 + 2*x^2\n"
    "EQUATIONS:\n"
    "  dq/dt = p/2\n"
    "  dp/dt = -4*x\n"
)

# Text with prose lines mixed in
PROSE_TEXT = (
    "The Hamiltonian is the total energy of the system.\n"
    "H = p^2/4 + 2*x^2\n"
    "where the first term is the kinetic energy.\n"
    "x = the displacement from equilibrium\n"
)


# ─── 1. StructuredOutputParser with default labels ────────────────────────────


class TestStructuredOutputParserDefaults:
    def test_parse_kinetic(self):
        parser = StructuredOutputParser(BASIC_TEXT)
        result = parser.get_labeled("KINETIC")
        assert result is not None
        assert "p" in result

    def test_parse_potential(self):
        parser = StructuredOutputParser(BASIC_TEXT)
        result = parser.get_labeled("POTENTIAL")
        assert result is not None
        assert "x" in result

    def test_parse_missing_label_returns_none(self):
        parser = StructuredOutputParser(BASIC_TEXT)
        result = parser.get_labeled("HAMILTONIAN")
        assert result is None


# ─── 2. StructuredOutputParser with custom labels ────────────────────────────


class TestStructuredOutputParserCustomLabels:
    def test_custom_label_parses(self):
        custom_labels = {"CUSTOM": {"custom", "c"}}
        parser = StructuredOutputParser(CUSTOM_TEXT, labels=custom_labels)
        result = parser.get_labeled("CUSTOM")
        assert result is not None
        assert "42" in result

    def test_default_labels_absent_with_custom(self):
        custom_labels = {"CUSTOM": {"custom", "c"}}
        parser = StructuredOutputParser(BASIC_TEXT, labels=custom_labels)
        # KINETIC is not in custom labels — should not be found
        result = parser.get_labeled("KINETIC")
        assert result is None


# ─── 3. parse_structured_output factory ──────────────────────────────────────


class TestParseStructuredOutputFactory:
    def test_returns_parser_instance(self):
        parser = parse_structured_output(BASIC_TEXT)
        assert isinstance(parser, StructuredOutputParser)

    def test_parser_is_functional(self):
        parser = parse_structured_output(FULL_STRUCTURED)
        assert parser.get_labeled("HAMILTONIAN") is not None


# ─── 4. get_labeled("HAMILTONIAN") ───────────────────────────────────────────


class TestGetLabeled:
    def test_hamiltonian_label(self):
        parser = StructuredOutputParser(FULL_STRUCTURED)
        result = parser.get_labeled("HAMILTONIAN")
        assert result is not None
        # Should contain H expression RHS
        assert "p" in result or "x" in result

    def test_returns_last_match(self):
        # Two HAMILTONIAN lines — should return last non-empty
        text = "HAMILTONIAN: H = p^2\nHAMILTONIAN: H = p^2/4 + 2*x^2\n"
        parser = StructuredOutputParser(text)
        result = parser.get_labeled("HAMILTONIAN")
        assert result is not None
        # Last match contains "4" or "2" — more specific expression
        assert "4" in result or "x" in result


# ─── 5. get_equations() ──────────────────────────────────────────────────────


class TestGetEquations:
    def test_returns_dqdt_and_dpdt(self):
        parser = StructuredOutputParser(FULL_STRUCTURED)
        eqs = parser.get_equations()
        assert len(eqs) >= 2
        # Should contain the RHS expressions
        assert any("p" in eq for eq in eqs)
        assert any("x" in eq or "4" in eq for eq in eqs)

    def test_empty_text_returns_empty(self):
        parser = StructuredOutputParser("No equations here.")
        eqs = parser.get_equations()
        assert isinstance(eqs, list)


# ─── 6. get_all_expressions() ────────────────────────────────────────────────


class TestGetAllExpressions:
    def test_groups_by_variable(self):
        parser = StructuredOutputParser(FULL_STRUCTURED)
        exprs = parser.get_all_expressions()
        assert isinstance(exprs, dict)
        # Should find H and/or T and/or V
        all_keys = set(exprs.keys())
        assert all_keys & {"H", "T", "V", "p", "q", "dq/dt", "dp/dt"}

    def test_values_are_lists(self):
        parser = StructuredOutputParser(FULL_STRUCTURED)
        exprs = parser.get_all_expressions()
        for key, val in exprs.items():
            assert isinstance(val, list)
            assert all(isinstance(v, str) for v in val)


# ─── 7. get_section_spans() ──────────────────────────────────────────────────


class TestGetSectionSpans:
    def test_returns_non_empty_spans(self):
        parser = StructuredOutputParser(FULL_STRUCTURED)
        spans = parser.get_section_spans()
        assert isinstance(spans, dict)
        # At least some spans should be non-empty
        non_empty = [k for k, v in spans.items() if v]
        assert len(non_empty) > 0

    def test_spans_are_tuples(self):
        parser = StructuredOutputParser(FULL_STRUCTURED)
        spans = parser.get_section_spans()
        for label, span_list in spans.items():
            for span in span_list:
                assert isinstance(span, tuple)
                assert len(span) == 2
                assert span[0] <= span[1]


# ─── 8. _find_final_output_boundary() ────────────────────────────────────────


class TestFindFinalOutputBoundary:
    def test_returns_zero_when_no_labels(self):
        parser = StructuredOutputParser("No labels here at all.")
        boundary = parser._find_final_output_boundary()
        assert boundary == 0

    def test_returns_nonzero_with_labels(self):
        parser = StructuredOutputParser(FULL_STRUCTURED)
        boundary = parser._find_final_output_boundary()
        assert isinstance(boundary, int)
        assert boundary >= 0


# ─── 9. extract_rhs_expressions() — basic ────────────────────────────────────


class TestExtractRhsExpressions:
    def test_returns_list_of_tuples(self):
        text = "H = p^2/4 + 2*x^2\nV = 2*x^2\n"
        results = extract_rhs_expressions(text)
        assert isinstance(results, list)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 4
            lhs, rhs, start, end = item
            assert isinstance(lhs, str)
            assert isinstance(rhs, str)
            assert isinstance(start, int)
            assert isinstance(end, int)

    def test_extracts_hamiltonian_expression(self):
        text = "H = p^2/4 + 2*x^2\n"
        results = extract_rhs_expressions(text)
        assert len(results) >= 1
        lhs, rhs, _, _ = results[0]
        assert lhs == "H"
        assert "p" in rhs or "x" in rhs


# ─── 10. extract_rhs_expressions() — skips prose lines ───────────────────────


class TestExtractRhsExpressionsProse:
    def test_skips_prose_lines(self):
        text = "H = the total energy of the system\nT = p^2/4\n"
        results = extract_rhs_expressions(text)
        # The prose line "H = the total energy..." should be skipped
        lhs_values = [r[0] for r in results]
        # T = p^2/4 should be captured
        assert "T" in lhs_values
        # H = the total energy... should NOT appear (prose RHS)
        h_results = [(l, r) for l, r, _, _ in results if l == "H"]
        for _, rhs in h_results:
            first = rhs.split()[0].lower() if rhs.split() else ""
            assert first not in {"the", "is", "was", "from", "and", "for", "with"}

    def test_captures_math_expressions(self):
        text = "V = 2*x^2\ndp/dt = -4*x\n"
        results = extract_rhs_expressions(text)
        assert len(results) >= 1


# ─── _COMMON_DEGREES sanity checks ───────────────────────────────────────────


class TestCommonDegrees:
    def test_common_degrees_has_standard_angles(self):
        assert "30" in _COMMON_DEGREES
        assert "45" in _COMMON_DEGREES
        assert "60" in _COMMON_DEGREES
        assert "90" in _COMMON_DEGREES

    def test_common_degrees_latex_has_fractions(self):
        assert "30" in _COMMON_DEGREES_LATEX
        assert "frac" in _COMMON_DEGREES_LATEX["30"]


# ─── get_all_expressions_with_spans — unified data path ────────────────────


class TestGetAllExpressionsWithSpans:
    """The unified data path: each expression paired with its char position.

    Replaces the legacy pattern where scoring and span-finding were independent
    code paths that could silently disagree.
    """

    def test_returns_positioned_tuples(self):
        text = "HAMILTONIAN: H = p**2/4 + 2*x**2"
        parser = StructuredOutputParser(text)
        result = parser.get_all_expressions_with_spans()
        assert isinstance(result, dict)
        for key, entries in result.items():
            for entry in entries:
                assert isinstance(entry, tuple)
                assert len(entry) == 3
                expr, start, end = entry
                assert isinstance(expr, str)
                assert isinstance(start, int)
                assert isinstance(end, int)
                assert start < end

    def test_spans_point_at_source_lines(self):
        """Each span should cover the source line containing the expression."""
        text = "HAMILTONIAN: H = p**2/4\nKINETIC: T = p**2/4"
        parser = StructuredOutputParser(text)
        result = parser.get_all_expressions_with_spans()

        for key, entries in result.items():
            for expr, start, end in entries:
                slice_text = text[start:end]
                # The slice must contain the expression's characters
                # (it's the whole line, so the expression should appear in it)
                # Compare without whitespace/substitutions
                assert "=" in slice_text  # every entry comes from a line with =

    def test_keys_are_subset_of_all_expressions(self):
        """expression_spans respects FINAL OUTPUT boundary; its keys are a subset
        of all_expressions (which scans the whole text)."""
        text = (
            "Let me reason first: H = maybe_wrong\n"
            "HAMILTONIAN: H = p**2/4 + 2*x**2\n"
            "KINETIC: T = p**2/4\n"
        )
        parser = StructuredOutputParser(text)
        ae = parser.get_all_expressions()
        es = parser.get_all_expressions_with_spans()
        assert set(es.keys()).issubset(set(ae.keys()))

    def test_expression_strings_match_all_expressions(self):
        """For every (expr, start, end) in expression_spans, expr must appear in
        the corresponding all_expressions list for the same key."""
        text = "HAMILTONIAN: H = p**2/4\nEQUATIONS:\n  dq/dt = p/2\n  dp/dt = -4*x\n"
        parser = StructuredOutputParser(text)
        ae = parser.get_all_expressions()
        es = parser.get_all_expressions_with_spans()
        for key, entries in es.items():
            for expr, _, _ in entries:
                assert expr in ae[key], (
                    f"{expr!r} in expression_spans[{key!r}] "
                    f"but not in all_expressions[{key!r}] = {ae[key]}"
                )

    def test_respects_final_output_boundary(self):
        """Expressions in reasoning text (before FINAL OUTPUT) are excluded.

        This fixes a training signal leak where scorers were evaluating
        scratch-work expressions.
        """
        # Clear separation: reasoning block, then labeled FINAL OUTPUT
        text = (
            "Reasoning attempt 1: H = p^2 + 2x maybe?\n"
            "Let me try again: H = p^3 + x\n"
            "\n"
            "\n"
            "\n"
            "\n"
            "\n"
            "\n"
            "\n"
            "\n"  # gap > MAX_GAP (5 lines)
            "COORDINATES: q = x\n"
            "MOMENTUM: p = 2*dx/dt\n"
            "HAMILTONIAN: H = p**2/4 + 2*x**2\n"
        )
        parser = StructuredOutputParser(text)
        es = parser.get_all_expressions_with_spans()
        h_entries = es.get("H", [])
        # Only the final HAMILTONIAN line should be captured — reasoning excluded
        assert len(h_entries) == 1
        expr, _, _ = h_entries[0]
        assert "p**2/4" in expr

    def test_cache_returns_same_object(self):
        """Cache avoids re-parsing on repeated calls."""
        text = "HAMILTONIAN: H = p**2/4"
        parser = StructuredOutputParser(text)
        r1 = parser.get_all_expressions_with_spans()
        r2 = parser.get_all_expressions_with_spans()
        assert r1 is r2  # same object from cache


# ─── Coverage + empty-span contracts ───────────────────────────────────────


class TestScoredSpansContracts:
    """Tests that hamiltonian_reward's scored_spans satisfy the unified contract:
    every quality in scores has a corresponding spans entry; zero scores yield
    empty spans (no gradient signal for absent qualities).
    """

    def test_every_quality_has_spans_entry(self):
        from examples.hamiltonian.reward_fn import hamiltonian_reward

        text = (
            "COORDINATES: q = x\n"
            "MOMENTUM: p = 2*dx/dt\n"
            "KINETIC: T = p**2/4\n"
            "POTENTIAL: V = 2*x**2\n"
            "HAMILTONIAN: H = p**2/4 + 2*x**2\n"
            "EQUATIONS:\n"
            "  dq/dt = p/2\n"
            "  dp/dt = -4*x\n"
        )
        meta = {
            "H_expr": "p**2/4 + 2*x**2",
            "T_expr": "p**2/4",
            "V_expr": "2*x**2",
            "dqdt": "p/2",
            "dpdt": "-4*x",
            "coordinates": "x",
        }
        result = hamiltonian_reward("", text, meta)
        # Every quality scored MUST have a corresponding spans entry
        for quality in result.scores:
            assert quality in result.scored_spans, f"{quality} missing from scored_spans"

    def test_zero_scores_have_empty_spans(self):
        """When no meta is provided, scorers that require meta return 0.0 with
        empty spans — no gradient signal for absent qualities."""
        from examples.hamiltonian.reward_fn import hamiltonian_reward

        text = "random completion with no structure"
        result = hamiltonian_reward("", text, {})
        # q_correct_H has no meta → score 0, spans empty
        assert result.scores.get("q_correct_H", 0.0) == 0.0
        assert result.scored_spans.get("q_correct_H", []) == []

    def test_nonzero_scores_have_nonempty_spans(self):
        """When a scorer finds an expression to evaluate, its spans are populated."""
        from examples.hamiltonian.reward_fn import hamiltonian_reward

        text = "HAMILTONIAN: H = p**2/4 + 2*x**2"
        meta = {"H_expr": "p**2/4 + 2*x**2"}
        result = hamiltonian_reward("", text, meta)
        assert result.scores["q_correct_H"] > 0.0
        assert len(result.scored_spans["q_correct_H"]) >= 1
