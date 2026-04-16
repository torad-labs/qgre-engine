# Eli Review: Unified Expression-Span Data Path

## Verdict

The plan is structurally sound and targets the right thing. It has one real flaw and two minor concerns. Here is the analysis.

---

## 1. Does it actually solve the leverage point?

Yes. The diagnosis is precise and the evidence is strong (6/10 fixed bugs trace to this root cause).

Here is the actual shape of the problem. Two completely independent code paths decide what the model wrote:

- **Scoring path**: `_find_all_expressions(pc)` calls `pc.all_expressions`, which comes from `parser.get_all_expressions()`. This iterates `self.lines`, splits on `=`, extracts variable names and RHS expressions. It operates on ALL lines. It determines what the model said.

- **Span path**: `_find_expression_spans(pc)` reads `pc.section_spans`, which comes from `parser.get_section_spans()`. This iterates the SAME `self.lines` but with completely different logic: it looks for labeled sections (`HAMILTONIAN:`, etc.) and `VAR = expr` patterns, computing character positions. It determines WHERE the model said it.

These two functions agree on nothing except the raw text. `get_all_expressions` returns `{"H": ["p^2/2 + V"]}` meaning "the model wrote H = p^2/2 + V." `get_section_spans` returns `{"HAMILTONIAN": [(450, 490)]}` meaning "the HAMILTONIAN section spans characters 450-490." The scorer evaluates the first. The span finder returns the second. If `get_all_expressions` picks up an `H = ...` expression from line 12 but `get_section_spans` puts the HAMILTONIAN label at line 25, the gradient hits the wrong tokens. The model gets punished (or rewarded) for text it did not write.

The plan fixes this by computing expression identity and position in the same pass. `get_all_expressions_with_spans()` returns `{"H": [("p^2/2 + V", 450, 478)]}` -- the expression AND where it lives, extracted together. When the scorer picks the best H candidate, the span comes for free because it was extracted alongside.

This is not moving the problem. This is eliminating the structural possibility of disagreement.

---

## 2. Does it follow conservative decomposition?

Yes. This is well-designed incremental migration:

- Phase 1 adds the new method and field without changing any behavior
- Phase 2-3 migrate scorers one at a time, each independently testable
- Phase 4 deletes the legacy path only after everything is migrated
- The legacy fallback in `hamiltonian_reward()` keeps unmigrated scorers working during the transition

The choice to extend `ParsedCompletion` rather than creating a bridge module is correct. The dataclass already holds `all_expressions` and `section_spans` -- adding `expression_spans` that unifies them is natural extension, not architectural invention. Fascia score is high: the new field connects directly to the existing `from_text()` factory and every scorer that reads from `pc`.

---

## 3. Structural issues the planning agents missed

### 3a. The real flaw: span granularity mismatch (medium severity)

The plan says scorers will return `list[tuple[int, int]]` spans alongside their scores. But look at how `_score_correct_H` actually works (lines 495-566):

```python
all_exprs = _find_all_expressions(pc)
h_candidates = all_exprs.get("H", [])
# Also try label-based extraction as fallback
extracted_H = _extract_H(pc)
if extracted_H and extracted_H not in h_candidates:
    h_candidates.append(extracted_H)
# Split chained equalities: "T + V = p^2/2 + 0 = p^2/2"
expanded = []
for cand in h_candidates:
    expanded.append(cand)
    if "=" in cand:
        last_rhs = cand.rsplit("=", 1)[-1].strip()
```

The scorer doesn't just read from `pc.all_expressions`. It also:
1. Falls back to `_extract_H(pc)` (label-based extraction)
2. Splits chained equalities to create sub-candidates
3. Picks the BEST match from the expanded set

If the best match is a sub-candidate created by splitting `"T + V = p^2/2 + 0 = p^2/2"` into `"p^2/2"`, what span do you return? The span from `expression_spans` covers the full `H = T + V = p^2/2 + 0 = p^2/2` line. The scorer actually matched against just `p^2/2`. The span is wider than what was scored.

This is not a showstopper -- a wider span still hits the right neighborhood of tokens, which is far better than hitting the wrong section entirely. But the plan presents it as though each scorer naturally produces precise spans, when in reality the expression-to-score mapping has these expansion/fallback steps that break the clean correspondence.

**Recommendation**: Accept this for Phase 2. Document that spans point to the source expression, not the sub-expression the scorer ultimately matched. This is honest and still a major improvement over the current section-level spans. If sub-expression precision matters later, that is a separate refinement.

### 3b. The `_extract_H` fallback creates a second extraction path (low severity)

`_score_correct_H` calls `_extract_H(pc)` as a fallback alongside `all_expressions`. If you migrate the scorer to use `pc.expression_spans`, you still need this fallback or you risk regression. But `_extract_H` is its own extraction logic -- it might find things that `get_all_expressions_with_spans` does not.

The plan does not mention this. It should explicitly decide: either fold `_extract_H`'s logic into the new method (making it comprehensive enough to not need fallbacks), or keep the fallback and accept that one scorer has a secondary extraction path.

### 3c. `get_all_expressions` operates on ALL lines; `get_section_spans` only on FINAL OUTPUT (low severity)

Look at `get_section_spans` line 368: `if line_start < final_output_boundary: continue`. It skips everything before FINAL OUTPUT. But `get_all_expressions` (lines 219-286) iterates ALL `self.lines` with no such boundary check.

When you unify them, you must decide: does `get_all_expressions_with_spans` respect the FINAL OUTPUT boundary or not? The scoring path currently does not (it finds expressions everywhere). The span path currently does (it only spans final output). Unifying them forces a choice.

**Recommendation**: Respect the FINAL OUTPUT boundary in the unified method. Scoring expressions from reasoning/scratch work was always a leak -- the model should be rewarded for its final answer, not its intermediate thoughts. This is a bonus fix that comes naturally from the unification.

---

## 4. Does it feel right?

Yes. Here is why, beyond the architecture.

The current `_find_expression_spans` (lines 902-969) is a lookup table. It maps section labels to quality names with zero intelligence:

```python
spans["q_correct_H"] = section_spans.get("HAMILTONIAN", [])
spans["q_V_correct"] = section_spans.get("POTENTIAL", [])
```

It does not know what the scorer found. It does not know which expression won. It does not even know if the scorer found anything at all. It just says "HAMILTONIAN section exists at these chars, so if any quality mentions H, point there." This is the kind of code that gets written when two systems need to agree but have no shared interface -- you build a manual mapping and hope.

The plan replaces hope with structure. Each scorer will know what it scored and where that thing lives, because both facts come from the same data. The mapping becomes unnecessary because the information flows through a single path.

The phased approach is also right for the project's current state. There is no deadline pressure (per the memory notes). The incremental migration means each phase can be tested and the system stays deployable throughout.

---

## Summary

- The diagnosis is correct. The plan solves the actual problem, not a proxy.
- The decomposition is conservative and well-phased.
- Address the `_extract_H` fallback explicitly before starting Phase 2.
- Accept that scorer spans will be expression-level, not sub-expression-level. Document this honestly.
- Consider respecting the FINAL OUTPUT boundary in the unified method -- it is a natural consequence of unification and fixes a pre-existing leak.
- The plan is the right move. Execute it.

73.
