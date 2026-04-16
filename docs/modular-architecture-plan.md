# QGRE Modular Architecture Plan

> Crystallized 2026-04-16. ELBO 0.60. Zero regressions. No shortcuts. Consult on every decision.

## Decisions (Locked)

1. **v1/v2 → Composable**: One `reward_fn.py`, verification strategy injectable, `reward_fn_v2.py` deleted
2. **Module name → `reward_parsing.py`**
3. **Scope → Includes trainer.py decomposition** (conservative: curriculum first, evaluate, then decide)
4. **Constants → In `expression.py`** (angle conversions available to any physics domain)
5. **SYMPY_LOCALS → Private base + merge param** (base has only `pi`; domain symbols passed at call time)
6. **v2's `_score_consistency` → Quality scorer in `reward_fn.py`** (domain logic, not general utility)
7. **Checkpoint → NO modification needed** (saves GameState data; curriculum functions are logic, not state)
8. **No `diagnostics.py`** — failed fascia check (F ≈ 0, decorative). Keep JSONL writer inline.
9. **No line count target for reward_fn.py** — the file is whatever size remains after extraction. "~800L" was an arbitrary constraint that adds coordination cost without value. (Pressure test finding)
10. **velocity_form_gold + build_substitutions → expression.py** — hidden dependencies of math_verify_scorer. Must ship together in Bundle B. (Pressure test finding)

## Resolved Questions

### SYMPY_LOCALS: private base + merge parameter

`_SYMPY_LOCALS` contains 24 symbols. Only `pi` is universally mathematical. The other 23 (`x, y, r, p, theta, m, k, g, ...`) are Hamiltonian-specific. A public mutable dict would cause cross-domain collisions.

```python
# qgre/expression.py
_BASE_SYMPY_LOCALS = {"pi": sp.pi}

def parse_math(expr_str: str, sympy_locals: dict | None = None) -> sp.Basic | None:
    merged = {**_BASE_SYMPY_LOCALS, **(sympy_locals or {})}
    ...

# examples/hamiltonian/reward_fn.py
HAMILTONIAN_LOCALS = {
    "x": sp.Symbol("x"), "y": sp.Symbol("y"), "p": sp.Symbol("p"),
    "theta": sp.Symbol("theta"), "m": sp.Symbol("m", positive=True),
    "k": sp.Symbol("k", positive=True), "g": sp.Rational(98, 10),
    # ... all 23 physics symbols
}
score = score_expression(student, teacher, sympy_locals=HAMILTONIAN_LOCALS)
```

### v2's `_score_consistency`: quality scorer in `reward_fn.py`

Both v1 (line 1577) and v2 (line 300) check internal consistency: "does dH/dp equal the model's stated dq/dt?" This encodes Hamilton's equations — domain physics, not general math. The TOOLS it uses (`sympy.diff`, `parse_math`) come from `expression.py`. The LOGIC (what to differentiate, what constitutes "consistency") is Hamiltonian-specific. Keep v1's version (takes `text`). v2's version (takes pre-extracted `expressions`) becomes unnecessary once v2 is deleted.

### Checkpoint: no modification needed

`save()` serializes `self.game_state` (the data) and `self.advantage_estimator.state_dict()` (the baselines). It does NOT serialize `self._tier_order`, `self._difficulty_column`, or `self._tier_advance_threshold` — those are config values set in `__init__`. Curriculum functions READ these. Extracting them to standalone functions that take `game_state` as a parameter changes nothing about serialization.

### After curriculum extraction: step() is untouched

The curriculum methods (`_record_mastery_and_advance`, `_apply_difficulty_gate`, `_get_prompt_tier`) live in `train()`, NOT in `step()`. Extracting them slims `train()` by ~220 lines (803→~580) but `step()` at 1565 lines is completely untouched. The next extraction target must come from WITHIN `step()` — requires studying its internal sections to find clean boundaries. This is Phase 2c work.

---

## Phase 1: Extract Reward Infrastructure

### 1A. Create `qgre/reward_parsing.py` (~400 lines)

**Extract from** `examples/hamiltonian/reward_fn.py`:

| What | Lines (approx) | Notes |
|------|---------------|-------|
| `StructuredOutputParser` class | 80-496 | Make `LABELS` a constructor parameter (default `None`) instead of hardcoded Hamiltonian labels |
| `parse_structured_output()` factory | 501-503 | Thin wrapper |
| Line-cleaning utilities | `_clean_line`, `_extract_expression`, `_clean_expression` | Methods on the parser |
| Expression extraction | `get_all_expressions`, `get_equations` | Methods on the parser |
| Section span detection | `get_section_spans`, `_find_final_output_boundary` | Methods on the parser |

**Also extract from** `examples/hamiltonian/reward_fn_v2.py`:

| What | Notes |
|------|-------|
| `_extract_rhs_expressions()` | Format-agnostic expression extractor — scans ANY line for `LHS = RHS`. More general than StructuredOutputParser. Standalone function, not a method. |

**Contract**: Two extraction strategies in one module:
- `StructuredOutputParser(text, labels)` — label-based, for structured output with sections
- `extract_rhs_expressions(text)` — format-agnostic, for free-form math

**Hamiltonian label config stays in `qgre/segments.py`** where `HAMILTONIAN_LABEL_ALIASES` already lives.

### 1B. Create `qgre/expression.py` (~500 lines)

**Extract from** `examples/hamiltonian/reward_fn.py`:

| What | Current name | Notes |
|------|-------------|-------|
| Timeout utility | `SympyTimeoutError`, `sympy_timeout()` | General-purpose |
| Text normalization | `_normalize_text()` → `normalize_text()` | Public, simple Unicode cleanup |
| Sympy normalization | `_normalize_for_sympy()` → `normalize_for_sympy()` | Public. Accepts `sympy_locals` merge param. |
| Math parser | `_parse_math()` → `parse_math()` | Public. Accepts `sympy_locals` merge param. |
| Safe parse wrapper | `_try_sympify()` → `try_sympify()` | Public. |
| Variable remapping | `_remap_variables()` → `remap_variables()` | Public. Coord/momentum name sets as params. |
| Expression comparison | `_score_expression()` → `score_expression()` | Core: student vs teacher, returns 0.0-1.0 |
| Structural term matching | `_score_terms_structurally()` → `score_terms_structurally()` | Public. |
| String similarity fallback | `_string_similarity()` → `string_similarity()` | Public. |
| Best match from candidates | `_best_match()` → `best_match()` | Public. |
| Constant extraction | `_extract_constants_from_prompt()` → `extract_constants_from_prompt()` | Public. |
| Base symbol table | `_SYMPY_LOCALS` → `_BASE_SYMPY_LOCALS` | Private. Contains only `{"pi": sp.pi}`. |
| Angle constants | `_COMMON_DEGREES`, `_COMMON_DEGREES_LATEX` | Public. Used by normalize_for_sympy. |

**Also extract from** `examples/hamiltonian/reward_fn_v2.py`:

| What | Notes |
|------|-------|
| `_gold_parse()` → `gold_parse()` | math-verify gold parsing |
| `_find_correct()` → `find_correct()` | math-verify expression matching |
| `_find_derivative()` → `find_derivative()` | math-verify derivative matching |
| `_build_substitutions()` → `build_substitutions()` | Build substitution dict from metadata. **REQUIRED** — math_verify_scorer hidden dependency. |
| `_velocity_form_gold()` → `velocity_form_gold()` | Velocity-form gold expressions. **REQUIRED** — math_verify_scorer hidden dependency. |
| `_normalize_text()` (v2 version) | Merge into `normalize_text()` |

**Hidden dependency (from pressure test):** `math_verify_scorer` silently fails on velocity-form expressions (e.g., `p²/(2m)` vs ground truth `p²/4` when m=2) without `build_substitutions` and `velocity_form_gold`. These are NOT optional utilities — they are required for math_verify_scorer to produce correct results. Ship together.

**Two scorer interfaces:**

```python
def sympy_scorer(student_str: str, teacher_str: str,
                 variables: list[str] | None = None,
                 constant_subs: dict | None = None,
                 sympy_locals: dict | None = None) -> float:
    """Score via sympy equivalence. Returns 0.0-1.0 with partial credit."""

def math_verify_scorer(student_str: str, teacher_str: str,
                       substitutions: dict | None = None) -> float:
    """Score via HuggingFace math-verify. Returns 0.0 or 1.0 (binary)."""
```

### ~~1C. Create `qgre/diagnostics.py`~~ — CANCELLED

Failed fascia check: F ≈ 0 (Δbelief=0, T=low, V=internal, Z=loose). A 50-line module with one JSONL writer function is decorative — it doesn't change system behavior when removed. Keep the 3-line diagnostic logging inline in `hamiltonian_reward()`.

### 1D. Restructure `examples/hamiltonian/reward_fn.py`

**No line count target.** The file is whatever size remains after replacing inlined code with imports. Don't force a number — the extraction is mechanical (move code, fix imports), not creative (restructure to hit a target).

After extraction, this file contains ONLY:
- `HAMILTONIAN_LOCALS` — 23 physics-specific sympy symbols (moved from `_SYMPY_LOCALS`)
- 15 domain-specific `_score_*` functions (including `_score_consistency` — domain logic)
- `_find_expression_spans()` — Hamiltonian-specific quality→section mapping
- `hamiltonian_reward()` — the entry point, with injectable `scorer` parameter
- Thin wrappers: `_extract_labeled()`, `_extract_H()`, `_extract_equations_block()`
- Imports from `qgre.reward_parsing`, `qgre.expression`
- Inline JSONL diagnostic logging (no diagnostics module)

### 1E. Delete `examples/hamiltonian/reward_fn_v2.py`

Its math-verify logic moves to `qgre/expression.py`. Its `_score_consistency()` is superseded by v1's version in `reward_fn.py`. Its `_extract_rhs_expressions()` moves to `reward_parsing.py`. The file is deleted.

### 1F. Tests for extracted modules

| New test file | What it tests |
|--------------|--------------|
| `tests/test_reward_parsing.py` | StructuredOutputParser with custom label configs, extract_rhs_expressions, expression extraction, section spans, final output boundary detection |
| `tests/test_expression.py` | sympy_scorer, math_verify_scorer, normalize_for_sympy, parse_math, remap_variables, score_expression partial credit, sympy_timeout, extract_constants_from_prompt, sympy_locals merge behavior |

**Ported from**: `tests/test_hamiltonian_reward.py` (parsing/expression tests that are actually general-purpose)

**`tests/test_hamiltonian_reward.py` remains**: tests domain-specific quality scorers and the composed `hamiltonian_reward()` function.

---

## Phase 2: Trainer Decomposition (Conservative)

### Phase 2a: Extract `qgre/curriculum.py` (~300 lines)

**Extract from `trainer.py`:**

| Method | Lines | What it becomes |
|--------|-------|----------------|
| `_get_prompt_tier` | 2619-2624 (5L) | `get_prompt_tier(metadata, difficulty_column)` |
| `_record_mastery_and_advance` | 2625-2773 (148L) | `record_mastery_and_advance(game_state, reward_results, active_qualities, batch_contexts, step_qualities, advantage_estimator, dataloader, difficulty_column, global_step)` |
| `_apply_difficulty_gate` | 2774-2839 (65L) | `apply_difficulty_gate(dataloader, game_state, difficulty_column)` |

**Parameters:** These functions currently access `self.game_state`, `self._tier_order`, `self._difficulty_column`, `self._dataloader`, `self.step_qualities`, `self.advantage_estimator`, `self.global_step`. All passed explicitly as parameters. No `self` coupling.

**Checkpoint impact:** NONE. `save()` serializes `game_state` (the data). These functions are logic, not state.

### Phase 2b: Evaluate `step()` (1565 lines, UNTOUCHED by 2a)

After curriculum extraction, `train()` drops to ~580 lines. `step()` remains 1565 lines.

**Do NOT pre-commit to loss.py, weight_sync.py, or training_loop.py.** Study `step()`'s internal sections:
- Map tensor data flow through step()
- Identify which sections share tensors (forward pass → logprobs → advantages AND loss)
- Find boundaries where parameters can be passed cleanly vs where coupling is fundamental
- Decide next extraction based on evidence, not plan

### Phase 2c: Next extraction (TBD after evaluation)

Candidates from crystallization (σ = 0.5, boundaries unclear):
- `loss.py` — if loss computation can separate from advantage estimation at the logprob level
- `weight_sync.py` — if orchestration layer isn't redundant with existing weight_bus.py
- `training_loop.py` — if train()'s remaining ~580 lines justify extraction

Decision made AFTER Phase 2b evaluation, not before.

---

## Phase 3: Integration & Verification

| Step | What |
|------|------|
| 3A | Update `qgre/__init__.py` with new exports |
| 3B | Run full test suite (38 test files) — ZERO regressions |
| 3C | Run Hamiltonian reward function on sample data — verify identical scores |
| 3D | Grep for all `from examples.hamiltonian.reward_fn import` — fix every one |
| 3E | Verify `python -m qgre train` still works end-to-end |

---

## Final File Tree

```
qgre/
├── __init__.py              (updated exports)
├── __main__.py              (unchanged)
├── reward_parsing.py        (NEW ~400L — StructuredOutputParser + extract_rhs_expressions)
├── expression.py            (NEW ~500L — sympy_scorer, math_verify_scorer, normalization)
├── curriculum.py            (NEW ~300L — tier advancement, mastery, difficulty gating)
├── trainer.py               (SLIMMED — curriculum methods removed, step() untouched)
├── advantages.py            (unchanged)
├── attention_bonds.py       (unchanged)
├── autograd_4bit.py         (unchanged)
├── checkpoint.py            (unchanged)
├── config.py                (unchanged)
├── critic.py                (unchanged)
├── data.py                  (unchanged)
├── generation.py            (unchanged)
├── hints.py                 (unchanged)
├── logging.py               (unchanged)
├── segments.py              (unchanged)
├── spans.py                 (unchanged)
├── sync_state.py            (unchanged)
├── types.py                 (unchanged)
├── weight_bus.py            (unchanged)
├── weight_export.py         (unchanged)
├── weight_load.py           (unchanged)
└── ...

examples/hamiltonian/
├── reward_fn.py             (SLIMMED ~800L — domain scorers, HAMILTONIAN_LOCALS, injectable scorer)
├── generate_data.py         (unchanged)
├── system_prompts.py        (unchanged)
└── (reward_fn_v2.py DELETED)

tests/
├── test_reward_parsing.py   (NEW)
├── test_expression.py       (NEW)
├── ... (all existing test files unchanged)
```

## Execution Order

```
Phase 1A (reward_parsing.py) ─┐
Phase 1B (expression.py)      ┤ parallel
                               ┘
         │
Phase 1F (tests for 1A-1B)
         │
Phase 1D (slim reward_fn.py)
         │
Phase 1E (delete v2)
         │
    ═══ TEST GATE ═══  (all 38 test files must pass)
         │
Phase 2A (curriculum.py)
         │
    ═══ TEST GATE ═══
         │
Phase 2B (evaluate step() — what next?)
         │
Phase 2C (TBD based on evidence)
         │
    ═══ TEST GATE ═══
         │
Phase 3A-3E (integration & verification)
```

## Phase 4: Second Domain Example (follow-up)

After Phases 1-3 are complete, the general-purpose claims of `reward_parsing.py` and `expression.py` are untested. A second domain example (chemistry, code review, essay grading, or similar) that imports from `qgre.reward_parsing` and `qgre.expression` would:

- Validate that the extraction is genuinely reusable (not just Hamiltonian with the labels swapped)
- Demonstrate `_find_final_output_boundary` outside Hamiltonian context
- Demonstrate `extract_rhs_expressions` as the format-agnostic alternative
- Provide a second data point for the `sympy_locals` merge pattern
- Give open-source users a template that ISN'T physics

This is a separate effort from the decomposition. Not blocking. But without it, the modularity is a claim, not a demonstration.

---

## Rules

1. **Zero regressions** — every existing test must pass after every phase
2. **No new bugs** — run tests after each extraction, not just at the end
3. **No shortcuts** — extract everything listed, don't skip "small" items
4. **No autonomous decisions** — consult Marcos on any architectural ambiguity
5. **Mechanical extraction first** — move code, fix imports, verify tests. Refactor AFTER it works.
6. **No decorative modules** — if F ≈ 0 (removing it changes nothing), don't create it
7. **Conservative trainer** — extract clean boundaries, evaluate, then decide next
