# QGRE Modular Architecture — Rejection Framework

> What NOT to do. Every item here was considered, measured, and rejected with evidence.
> Violating these boundaries re-introduces the structural problems this refactoring solves.

---

## Rejected Patterns

### 1. Sub-packages inside `qgre/`

**What training wants:** `qgre/reward_utils/`, `qgre/reward/`, or `qgre/scoring/` with `__init__.py`.

**Why it's wrong:** `qgre/` has 29 flat .py files. Zero sub-packages. Adding nesting creates the first architectural precedent — every future contributor will ask "should this be a sub-package?" The flat pattern IS the architecture. `spans.py`, `segments.py`, `advantages.py` all coexist at the same level. New modules follow the same pattern.

**What to do instead:** Flat .py files alongside existing modules. `qgre/reward_parsing.py`, `qgre/expression.py`, `qgre/curriculum.py`.

**Immune class:** `nesting-without-precedent`

---

### 2. Abstract base classes for reward functions

**What training wants:** `AbstractRewardParser`, `AbstractScorer`, `RewardPipeline(ABC)` with abstract methods that domains override.

**Why it's wrong:** `RewardResult` IS the contract. A reward function is any callable that returns `RewardResult`. Adding ABCs creates an inheritance hierarchy that constrains more than it enables. The chemistry researcher doesn't want to subclass `AbstractRewardParser` — they want to call `from qgre.reward_parsing import StructuredOutputParser` and pass their own labels. Composition via imports, not inheritance via override.

**Evidence:** The math example (`examples/math/reward_fn.py`) is 28 lines. It returns `RewardResult`. That's the entire contract. An ABC would make the simplest case harder without making the complex case easier.

**What to do instead:** Functions that take parameters and return typed results. `score_expression(student, teacher, sympy_locals=MY_LOCALS) -> float`. Compose by calling, not by inheriting.

**Immune class:** `abc-over-composition`

---

### 3. One file per pipeline stage (5+ thin modules)

**What training wants:** `qgre/lexer.py`, `qgre/normalizer.py`, `qgre/comparator.py`, `qgre/span_finder.py`, `qgre/scorer.py` — one module per pipeline concept.

**Why it's wrong:** Some stages are too thin to justify their own module. `sections→expressions` is a method on the parser (not a separate module). `expressions→sympy` is a function in the expression module (not its own file). The codebase median module size is 300-600 lines. A 50-line module is overhead, not architecture.

**Evidence:** `diagnostics.py` was planned as a 50-line module. Fascia measurement: F ≈ 0. Removing it changes nothing about system behavior. It's decorative. The same applies to any module below ~100 lines that could live as a function in a load-bearing module.

**What to do instead:** Two modules at natural thickness boundaries: `reward_parsing.py` (~400L, parsing + extraction) and `expression.py` (~500L, comparison + normalization). Plus `curriculum.py` (~300L) for trainer. Three total, each with real weight.

**Immune class:** `one-concept-one-file-always`

---

### 4. Decorative modules (F ≈ 0)

**What training wants:** "Clean separation" — every distinct concern gets its own file, regardless of size or coupling.

**Why it's wrong:** A module is architectural only if removing it changes system behavior. The fascia test: F(i→j) = |Δbelief| × T(i) × V(i) × Z(i,j). If Δbelief = 0 (removing it changes nothing), the module is decorative. It adds import overhead, file navigation cost, and a false sense of organization without structural value.

**Evidence:** `qgre/diagnostics.py` was planned at 50 lines (one JSONL writer function). Δbelief = 0 (removing it doesn't change training behavior). T = low (barely tested). V = internal (only called by reward functions). Z = loose (one function call). F ≈ 0. Rejected.

**What to do instead:** Keep small utilities inline in their load-bearing parent module. The JSONL diagnostic writer stays in `hamiltonian_reward()` as 3 lines. If a utility grows large enough to warrant its own module (>100L, multiple callers, tested independently), promote it then — not preemptively.

**Immune class:** `decorative-modules`

---

### 5. Full trainer decomposition upfront (without studying the boundaries)

**What training wants:** Plan all 4 trainer modules now (loss.py, weight_sync.py, training_loop.py, curriculum.py), commit to extracting them all, execute in parallel.

**Why it's wrong:** `step()` is 1565 lines with shared tensor state between advantage estimation and loss computation. The boundaries WITHIN step() are genuinely unclear from outside. Planning modules for code you haven't deeply studied creates architectural fiction — names and line counts that look specific but aren't grounded in evidence. This is NOT about priority or demand — both reward and trainer decomposition will happen. It's about SEQUENCING: extract what has clear boundaries first, study what remains, then extract the next piece with evidence.

**Evidence:** The curriculum methods (`_record_mastery_and_advance`, `_apply_difficulty_gate`) live in `train()`, not `step()`. Extracting them doesn't touch step() at all. trainer.py drops from 3745 to ~3530 — modest. The HARD problem (step() at 1565L) requires implementation-level study, not planning-level commitment. After curriculum extraction, the remaining boundaries become visible from inside.

**What to do instead:** Extract `curriculum.py` (clean GameState boundary). Run tests. Study step()'s internal sections with the clean parts removed. Then extract the next piece with real evidence about where tensors are shared, what can be passed as parameters, and what's genuinely coupled.

**Immune class:** `decompose-everything-now`

---

### 6. Public mutable symbol tables

**What training wants:** `DEFAULT_SYMPY_LOCALS` as a public dict that users extend by mutation: `DEFAULT_SYMPY_LOCALS["element"] = sp.Symbol("element")`.

**Why it's wrong:** Module-level dict mutation is a side-effect collision. If chemistry adds `{"H": sp.Symbol("H")}` for hydrogen and Hamiltonian adds `{"H": sp.Symbol("H")}` for the Hamiltonian, they stomp each other. The mutation persists across imports — any code that imports the module gets the mutated dict.

**Evidence:** `sympify()` already accepts a `locals` parameter that takes a dict merge. The safe mechanism EXISTS in sympy's own API. The base dict should contain only universally mathematical symbols (`pi`). Domain-specific symbols are passed at call time and merged locally.

**What to do instead:** Private `_BASE_SYMPY_LOCALS = {"pi": sp.pi}`. All functions accept `sympy_locals: dict | None = None` and merge internally: `{**_BASE_SYMPY_LOCALS, **(sympy_locals or {})}`. Each domain defines its own symbol dict (`HAMILTONIAN_LOCALS`, `CHEMISTRY_LOCALS`) and passes it explicitly.

**Immune class:** `mutable-module-state`

---

### 7. v2 as a separate file after refactoring

**What training wants:** Keep `reward_fn_v2.py` as an alternative implementation alongside v1.

**Why it's wrong:** After extracting shared code to `qgre/`, v2 becomes ~200 lines of glue code that calls the same utilities as v1 but in a slightly different order. Two files that do 90% the same thing using the same underlying modules is maintenance overhead, not clean separation. The composable scorer injection (`scorer=sympy_scorer` vs `scorer=math_verify_scorer`) gives users the same choice without a second file.

**Evidence:** v2's `_score_consistency` (line 300) does the same thing as v1's `_score_consistency` (line 1577) — check model's H against its own stated equations. v2's `_extract_rhs_expressions` is actually MORE general than v1's parser — it should be promoted to `reward_parsing.py`, not kept in a secondary file. v2's math-verify integration is a SCORER STRATEGY, not a separate reward function.

**What to do instead:** One `reward_fn.py` with injectable scorer parameter. math-verify logic lives in `expression.py` as `math_verify_scorer()`. `extract_rhs_expressions` lives in `reward_parsing.py`. v2 is deleted.

**Immune class:** `parallel-implementations`

---

### 8. Documentation before separation

**What training wants:** Write comprehensive docstrings, README sections, and tutorial notebooks before or instead of extracting modules.

**Why it's wrong:** When parsing and scoring live in the same 2028-line file, no amount of documentation helps the chemistry researcher find what they need. They'd have to read 2028 lines of Hamiltonian physics to discover that `_score_expression` on line 896 is reusable. Separation IS documentation. A 400-line file named `reward_parsing.py` is more teachable than a 2028-line file with perfect docstrings.

**Evidence:** The teachability test from crystallization: "Can you explain `expression.py` without explaining `reward_parsing.py`?" Yes — comparing `p²/4` to `p**2/(2*2)` is purely mathematical, no parsing context needed. "Can you explain `reward_parsing.py` without `expression.py`?" Yes — extracting "KINETIC: T = p²/4" from text is pure text processing. Independently teachable = correctly separated. This is better documentation than any README.

**What to do instead:** Separate first. The module names, function signatures, and import paths ARE the documentation. Add docstrings to public functions AFTER extraction — they describe the module's API, not an implementation buried in a god object.

**Immune class:** `docs-over-structure`

---

### 9. Arbitrary line count targets

**What training wants:** "Slim reward_fn.py to ~800L" — a specific number that sounds like a concrete goal.

**Why it's wrong:** The number 800 has no structural basis. It's an aesthetic guess. The file should be whatever size remains after replacing inlined code with imports. If that's 600L, great. If that's 1000L, also fine — the remaining code is domain-specific quality scorers that BELONG together. Forcing a line count target turns a mechanical extraction (move code, fix imports) into a creative coordination problem (what do I cut/restructure to hit 800?). The constraint adds cost without value.

**Evidence (from pressure test):** With a ~800L target, the "slim reward_fn" feature had I(x)=0.40 and barely positive Net even with all dependencies resolved. When redefined as "replace inlined code with imports" (no target), I(x) dropped to 0.25 and Net became solidly positive (+0.403). The target itself was the constraint.

**What to do instead:** Extract code mechanically. The file size is an OUTPUT of the extraction, not an INPUT.

**Immune class:** `arbitrary-targets`

---

## Boundary Summary

| If you're tempted to... | Stop and check | Because |
|------------------------|----------------|---------|
| Create a sub-package | Is `qgre/` still flat? | First nesting sets precedent |
| Write an abstract base class | Is `RewardResult` sufficient? | The contract already exists |
| Create a <100L module | Does removing it change behavior? (F ≈ 0?) | Decorative modules are overhead |
| Plan all trainer modules now | Have you read step()'s 1565 lines? | Don't plan what you haven't studied |
| Expose a mutable module-level dict | Can you pass it as a parameter instead? | Module state causes cross-domain collisions |
| Keep v2 alongside v1 | Are they >10% different after extraction? | Parallel implementations diverge |
| Write docs instead of extracting | Can someone find the reusable code? | Separation is documentation |
| Set a line count target | Is the number structurally derived? | Arbitrary targets add cost without value |
