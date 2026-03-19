---
description: >-
  Plan-driven development loop for QGRE engine. Reads plan docs, compares against
  what's built, checks API correctness, finds gaps, builds the next piece, runs
  tests, and repeats until the plan is fully implemented with zero drift.
argument-hint: "[step name — e.g. 'step-0a', 'step-0d', 'step-1', 'all']"
allowed-tools: Read, Grep, Glob, Bash, Agent, Edit, Write, WebFetch, WebSearch
---

# Feature Loop — Plan-Driven Iterative Development (QGRE Engine)

You are a plan-driven development loop. Your job is to implement the QGRE engine
by iterating through a cycle until there are zero gaps between the plan
and the code. You do NOT invent. You do NOT improvise. You execute the plan.

## Input

The user provides a build step name (e.g. `step-0d`, `step-1`) or `all`.

Plan docs live at:
- `docs/PLAN.md` — the master plan (build order, algorithm design, architecture, risks, issue resolutions, Exa findings)
- `docs/PILLARS.md` — six pillars decomposition (components per pillar, research findings, dependency graph)
- `docs/SPECIAL-TOKENS-SUPERPOWER.md` — VPRM spec (segment_completion, step-level rewards, token ID patterns)

If a requested step's spec is not in the plan docs, STOP and tell the user.

## Project Structure

```
qgre/
  __init__.py          — package root, exports RewardResult
  types.py             — RewardResult dataclass
  config.py            — engine config dataclass (Step 1)
  segments.py          — segment_completion() + STEP_QUALITIES (Step 0d)
  advantages.py        — QGREStepAdvantageEstimator (Step 0d)
  data.py              — DataLoader: parquet → tokenize → pad → batch (Step 0e)
  checkpoint.py        — GameState serializer + checkpoint resume (Steps 0a, 0f)
  logging.py           — MLflow tracking + JSONL dump (Steps 4, 6)
  trainer.py           — QGRETrainer (Step 1)
  nemo_extracted/
    __init__.py        — NeMo RL attribution
    loss_functions.py  — ClippedPGLossFn (Step 0b)
    kl.py              — KL calculation + kl_cov (Step 0b)
    logits.py          — Log prob computation (Step 0b)
    LICENSE            — Apache-2.0 attribution
examples/
  hypergraph/
    config.yaml        — full QGRE config (model, data, generation, algorithm, training, logging)
    reward_fn.py       — stub reward function returning RewardResult
  math/
    config.yaml        — minimal GRPO config
    reward_fn.py       — stub scalar reward
tests/
  conftest.py          — fixtures (synthetic batches, known token IDs, mock models)
  test_checkpoint.py   — Steps 0a, 0f, 5
  test_nemo_extracted.py — Step 0b
  test_segments.py     — Step 0d (segmentation)
  test_advantages.py   — Steps 0c, 0d, 8 (advantages + credit assignment)
  test_data.py         — Step 0e
  test_trainer.py      — Step 1
  test_wiring.py       — Steps 2-3 (GPU required)
  test_logging.py      — Steps 4, 6
  test_equivalence.py  — Step 7
  test_smoke.py        — GPU smoke test
```

## Build Sequence (from PLAN.md)

```
Prerequisites:
  0a: GameState serializer (checkpoint.py)
  0b: NeMo RL extraction (nemo_extracted/*.py)
  0c: Batch reward tensor construction (advantages.py — partial)
  0d: QGREStepAdvantageEstimator (segments.py + advantages.py) — CORE ALGORITHM
  0e: DataLoader (data.py)
  0f: Checkpoint resume (checkpoint.py)
  0g: LoRA verification harness (new module)

Assembly:
  1: QGRETrainer (trainer.py + config.py)
  2: Wire Unsloth + vLLM generation
  3: Wire reward function
  4: Wire MLflow tracking
  5: Wire checkpoint save/resume
  6: Wire JSONL dump
  7: Equivalence test
  8: Credit assignment test
```

## The Loop

Execute this cycle. Each iteration does ALL 6 phases. After phase 6,
if gaps remain, start phase 1 again. Continue until phase 6 finds zero gaps.

```
PHASE 1: READ THE PLAN
    ↓
PHASE 2: READ WHAT'S BUILT
    ↓
PHASE 3: CHECK API CORRECTNESS
    ↓
PHASE 4: FIND GAPS (plan vs built)
    ↓
PHASE 5: BUILD THE NEXT GAP
    ↓
PHASE 6: VERIFY + TEST + LOOP OR DONE
    ↓
 gaps > 0? → PHASE 1
 gaps = 0? → COMPLETE
```

---

### PHASE 1: Read the Plan

Read the relevant plan sections for the requested step. Extract:

1. **Build sequence** — ordered steps and their dependencies
2. **File manifest** — every file that should exist when this step is done
3. **Architecture** — what imports what, what depends on what
4. **Data types** — RewardResult, config dataclasses, STEP_QUALITIES mapping
5. **Algorithm spec** — pseudocode from PLAN.md for this step's component
6. **Test spec** — tests listed in "Verifiable Tests Per Deliverable" section of PLAN.md
7. **Risks** — relevant risks from the risk table and Exa findings

Hold this as your source of truth. The plan is the contract.

---

### PHASE 2: Read What's Built

Scan the actual codebase to see what exists RIGHT NOW:

1. `glob` for all `.py` files in `qgre/`, `tests/`, `examples/`
2. `git diff --name-only` to see what's changed recently
3. Read key files that should exist per the step's file manifest
4. Check if files are stubs (1-3 line comments) or have real implementation
5. `python -c "import qgre"` — does the package import?

Produce a status checklist:
```
[x] qgre/types.py — exists, RewardResult defined
[x] qgre/__init__.py — exists, exports RewardResult
[ ] qgre/segments.py — STUB (comment only, no implementation)
[ ] qgre/advantages.py — STUB (comment only, no implementation)
[ ] tests/test_segments.py — STUB (comment only, no tests)
...
```

---

### PHASE 3: Check API Correctness

For every file that has REAL implementation (not stubs), verify correctness:

**PyTorch / Training:**
- `torch.Tensor` operations use correct dtypes (float32 for advantages)
- No in-place operations on tensors that require grad
- `torch.no_grad()` where appropriate
- Loss reduction matches plan (token-level vs sequence-level)
- Gradient accumulation: `loss /= accumulation_steps` before backward

**QGRE-Specific:**
- `RewardResult` is imported from `qgre.types`, not redefined
- `STEP_QUALITIES` mapping matches SPECIAL-TOKENS-SUPERPOWER.md exactly
- Token IDs match Qwen3 verified values (THINK_START=151667, THINK_END=151668, etc.)
- `segment_completion()` uses token ID pattern matching, NOT decoded-text regex
- SPO value tracker: EMA update `V = V + lr * (r - V)`, NOT replacement
- GDPO normalization: per-step across batch, NOT per-sequence
- Phase gating: only active qualities contribute to step rewards

**NeMo RL Extracted:**
- Apache-2.0 headers preserved on all extracted files
- No imports from `nemo_rl.*` (all deps stripped)
- No imports from `ray`, `megatron`, `nemo_rl.distributed`
- `masked_mean` handles zero-mask edge case (no division by zero)

**Config:**
- `generation` section includes temperature, top_p, stop_token_ids
- `algorithm.mode` is either "spo" or "grpo"
- SPO config has `lr` and `n=1`; GRPO config has `n` and `filter_groups`

**Architecture boundaries:**
- `qgre/` modules import only from `qgre/` and standard libs (torch, numpy)
- `examples/` import from `qgre` package
- `tests/` import from `qgre` package
- No circular imports within `qgre/`

If you find violations, fix them immediately before proceeding to phase 4.

---

### PHASE 4: Find Gaps

Compare the plan's spec for this step against what exists. Produce a gap list:

```
GAP LIST (iteration N):
1. [MISSING] qgre/segments.py — segment_completion() not implemented
2. [INCOMPLETE] qgre/advantages.py — SPO warm-start logic missing
3. [DRIFT] examples/hypergraph/config.yaml — clip_ratio_high is 0.28, plan says 0.28 ✓
4. [API] qgre/nemo_extracted/loss_functions.py — still imports nemo_rl.algorithms.interfaces
5. [TEST] tests/test_segments.py — stub only, no test functions
```

Gap types:
- **MISSING** — file doesn't exist or is a stub
- **INCOMPLETE** — implementation exists but doesn't match plan spec
- **DRIFT** — values, structure, or behavior diverged from plan
- **API** — using wrong API, wrong imports, wrong patterns
- **TEST** — test from "Verifiable Tests Per Deliverable" not implemented
- **RISK** — known risk from Exa findings not mitigated

Rank gaps by dependency order (from PLAN.md build sequence).
The next gap to build is the FIRST one that unblocks other gaps.

---

### PHASE 5: Build the Next Gap

Pick the highest-priority gap. Build it:

1. **Read the plan section** for this specific component
2. **Read adjacent files** that this component depends on or is depended on by
3. **Write or edit the file** — match the plan exactly
4. **Run the step's tests:** `pytest tests/test_{module}.py -v`
5. **Run import check:** `python -c "from qgre.{module} import {class}"`

Rules:
- Build ONE gap per iteration, not all of them
- Match the plan's types, names, and algorithm exactly
- Use pseudocode from PLAN.md and SPECIAL-TOKENS-SUPERPOWER.md as implementation spec
- Token IDs must match the verified Qwen3 values in the plan
- Do NOT add features, refactors, or improvements not in the plan
- Do NOT add comments unless the logic is non-obvious
- If the plan specifies test cases, implement them in the corresponding test file
- Smoke test model is Qwen3-1.7B (NOT 8B). Config defaults: unsloth/Qwen3-1.7B-unsloth-bnb-4bit
- Do NOT copy patterns from training-dojo v1. This engine is a clean rewrite.

**MANDATORY: Exa search before ANY bug fix.**
When a test fails, an error occurs, or unexpected behavior is observed:
1. STOP — do not guess the fix
2. Search Exa for the specific error message, library version, or API in question
3. Read the search results to understand the root cause
4. ONLY THEN apply a fix based on evidence, not training data assumptions
5. Cite the source (GitHub issue #, docs URL, or paper) in a comment if non-obvious

This rule exists because training data is stale. Unsloth, vLLM, and PyTorch
APIs change faster than any model's knowledge cutoff. The cost of one Exa search
is seconds. The cost of a wrong fix is hours of debugging.

If building this gap reveals a plan deficiency, NOTE IT but don't fix the plan.
Report it in phase 6 so the user can decide.

---

### PHASE 6: Verify + Test + Loop or Done

After building the gap:

1. **Test:** `pytest tests/test_{module}.py -v` — must pass
2. **Import check:** `python -c "import qgre"` — must succeed
3. **Re-read the file** — verify it matches the plan
4. **Check imports** — no boundary violations introduced
5. **Re-run phase 4** mentally — how many gaps remain?

**If a test FAILS:** Do NOT guess the fix. Follow the mandatory Exa search
protocol from Phase 5. Search for the error message on Exa FIRST, understand
the root cause from real sources, THEN fix. No exceptions.

If gaps remain:
```
ITERATION N COMPLETE
Built: [what was built]
Tests: PASS/FAIL (N passed, M failed)
Remaining gaps: N
Next gap: [description]
Continuing...
```
→ Go back to PHASE 1.

If zero gaps remain for this step:
```
STEP COMPLETE: [step name]

Files created/modified:
  [list every file touched]

Plan compliance:
  Algorithm: matches PLAN.md pseudocode
  Token IDs: verified against SPECIAL-TOKENS-SUPERPOWER.md
  Architecture boundaries: CLEAN
  Tests: ALL PASSING

Remaining concerns:
  [any plan deficiencies noted during build]

Ready for: next step / integration test / PR
```

---

## Autonomous Continuation

When a step completes, DO NOT STOP AND WAIT for user input. Immediately
proceed to the next gap. The loop is designed to run unattended. The only
reasons to pause:

1. A test fails and Exa search doesn't resolve it — ask the user
2. A plan deficiency is found that requires a design decision — ask the user
3. ALL gaps are closed — report completion

For GPU tests: run them with `pytest --gpu` if the GPU is available
(`nvidia-smi` shows free memory). If GPU is busy, skip GPU tests and
continue with CPU-testable work.

## Current Work Queue (auto-updated, 2026-03-18)

Engine is FEATURE LOOP COMPLETE. 85 CPU tests + 3 GPU smoke tests pass.
Three rounds of review completed (code-reviewer + silent-failure-hunter + simplifier × 2).
13 bugs found in round 1, all fixed. 2 issues found in round 2 (config _pick, missing regression tests), both fixed.
5 regression tests added for the most critical fixes.

### Session Build Log

```
Round 1 — Build (8 iterations):
  0a: GameState serializer → checkpoint.py (11 tests)
  0b: NeMo RL extraction → loss_functions.py, kl.py, logits.py (10 tests)
  0c+0d: Advantage estimator → segments.py + advantages.py (22 tests)
  0e: DataLoader → data.py (9 tests)
  0f: Checkpoint resume → checkpoint.py (extends 0a tests)
  0g: LoRA verifier → lora_verify.py (7 tests)
  1+4+6: Trainer + config + logging → trainer.py, config.py, logging.py (12 tests)
  2: Generation backend → generation.py (GPU tests)

Round 2 — Review + Fix:
  Code reviewer found: 4 critical (IS weight no-op, all-ones mask, kl_loss device, GRPO remainder)
  Silent failure hunter found: 6 critical + 7 high (NaN guard, GDPO NaN, DataLoader zero, etc.)
  Simplifier applied: comment cleanup, dict comprehensions, extracted helpers
  → 13 bugs fixed, tests re-verified on GPU

Round 3 — Review of fixes:
  Code reviewer: all 13 fixes verified, found config._pick missing on algorithm.spo/grpo + 7 untested fixes
  Silent failure hunter: found 2 CRITICAL (prompt_lengths on completion-only tensor, double [:, 1:] shift) + 4 HIGH
  Simplifier: cosmetic cleanups (comment trimming, variable extraction)
  → config._pick fixed, 5 regression tests added, 2 critical alignment bugs fixed (85 CPU + 3 GPU pass)

Fixes applied from round 3:
  - prompt_lengths=[0] for completion-only tensors (vLLM returns response-only)
  - advantages[:, :-1] instead of [:, 1:] (truncate, not shift — logprobs already shifted)
  - config._pick applied to algorithm.spo/grpo sub-dicts

Known deferred items (not bugs, design decisions for v2):
  - DataLoader state not saved in checkpoint (resume replays from epoch 0)
  - GRPO last-batch crash if dataset not divisible (add drop_last option)
  - SHA-256 prompt_id breaks compatibility with hypothetical old checkpoints (no old checkpoints exist)
  - CompletionLogger file handle leak on crash (flush() mitigates, __del__ would be cleaner)
```

### COMPLETED: Generalize engine — committed de2a74c through 548005a

### ACTIVE WORK: Wire GameState into engine for phase advancement

The QGRE superpower (from SPECIAL-TOKENS-SUPERPOWER.md): the engine has direct access to
token-level structure via the segmenter. It can assign per-step rewards at the token level.
This is what makes QGRE different from generic GRPO.

Currently broken: GameState is created and checkpointed but NEVER READ by the engine.
Phase advancement is entirely the reward_fn's responsibility. But the PLAN says
"phase-gated by GameState" — the engine should manage phase transitions.

The architecture should be:
- reward_fn: scores per-quality (text-level verification: parse XML, check JSON, verify grounding)
- engine: uses quality scores + GameState to determine phase, track mastery, advance phases
- engine: uses segmenter + step_qualities + phase to compute per-token advantages

**Changes needed:**

```
Step P1: Engine-managed phase advancement
  - QGRETrainer.step() updates GameState after scoring:
    * Records quality scores per archetype via add_quality_score()
    * Checks mastery windows via get_quality_mean()
    * Advances game_state.phase when thresholds are met
  - Phase threshold config in AlgorithmConfig (e.g., mastery_threshold: 0.8)
  - RewardResult.phase is SET BY THE ENGINE (from game_state.phase), not by reward_fn
  - reward_fn returns scores only — engine determines phase
  - Test: quality scores accumulate, phase advances when threshold met

Step P2: GameState → advantage estimator wiring
  - QGRETrainer passes game_state.phase to compute_advantages (not rr.phase)
  - on_tier_advance() called when phase advances
  - SPO V-tracker resets for newly-active qualities
  - Test: phase advance → V-tracker reset → no spike

Step P3: Mastery tracking per step (not per archetype)
  - GameState tracks quality means PER STEP, not just per archetype
  - Phase N advances when step N's quality mean exceeds threshold
  - This is the QGRE curriculum: step 1 mastery unlocks step 2 qualities
  - Test: step 1 mastery → phase 2 → step 2 qualities activate

Step P4: Remove phase from RewardResult
  - RewardResult becomes: reward + scores only
  - Phase lives in GameState (engine-managed)
  - Update all tests, examples, conftest
  - Backward compat: if RewardResult.phase is set, warn and ignore

Step P5: Update README + docs
  - Document the engine-managed phase model
  - Show how mastery thresholds work
  - Update "Bring Your Own Domain" section
```

Step P6: Full train() loop in trainer.py
  - Add train() method that loops: for batch in dataloader → generate → score → step()
  - Calls generation_backend.generate() for completions
  - Calls reward_fn for scoring
  - Calls step() for algorithm + backward
  - Integrates phase advancement after step()
  - Records per-step scores to GameState
  - Checks phase advance, calls on_tier_advance if advanced
  - Saves checkpoints every save_freq steps
  - Logs to MLflow via log_step_metrics
  - Test: mock trainer runs full loop for 3 steps

Step P7: Fix checkpoint serialization
  - gamestate_to_dict / gamestate_from_dict must match new GameState fields
  - step_mastery: dict of deque → dict of list (with maxlen preserved)
  - phase_history: list of ints
  - mastery_threshold: float
  - Remove references to old fields (quality_windows, elo_ratings, mastery_counts, max_active_tier)
  - Test: round-trip new GameState through checkpoint

Step P8: Wire MLflow + LR scheduler
  - trainer.train() calls log_step_metrics after each step
  - setup_optimizer creates scheduler from config
  - Scheduler steps after each optimizer step
  - Test: mock mlflow receives expected metrics

Step P9: SPO warm-start fix — use batch mean not current sample
  - Change v = r to v = batch_mean for first observation (matches PLAN.md spec)
  - Test: first observation advantage ≈ 0 regardless of individual reward value

**Execution rules:**
- Build ONE step per iteration
- After each step: run pytest — must pass
- Exa search before any bug fix
- Commit and push after each step passes

The engine is currently hardcoded to 4 XML steps with v1-specific quality names.
This must be generalized so any domain can use the engine.

**Changes needed (in order):**

```
Step G1: Make STEP_QUALITIES configurable (segments.py + advantages.py)
  - Remove hardcoded STEP_QUALITIES dict from segments.py
  - Accept step_qualities as a parameter to QGREStepAdvantageEstimator.__init__()
  - Accept step_qualities in QGREConfig (algorithm section of YAML)
  - Replace all range(1, 5) with range based on len(step_qualities)
  - PHASE_QUALITIES in trainer.py derived from config, not hardcoded
  - Update examples/hypergraph/config.yaml with step_qualities mapping
  - Test: existing tests pass with step_qualities passed explicitly

Step G2: Make segment_completion() pluggable (segments.py + trainer.py)
  - Define a Segmenter protocol: Callable[[list[int]], list[str]]
  - Move current segment_completion to qwen3_xml_segmenter() as one implementation
  - QGREStepAdvantageEstimator accepts segmenter as __init__ parameter
  - QGRETrainer accepts segmenter in config or constructor
  - Default: qwen3_xml_segmenter (backward compatible)
  - Add a simple json_key_segmenter() as a second implementation for JSON-based outputs
  - Test: both segmenters produce valid region labels

Step G3: Make phase count configurable (advantages.py + trainer.py)
  - Phase count = max phase in step_qualities config, not hardcoded 4
  - PHASE_QUALITIES built dynamically: phase N includes all qualities from steps 1..N
  - Allow custom phase→qualities mapping in config for non-cumulative phase structures
  - Test: 5-phase config works, 3-phase config works

Step G4: Bump max_tokens default to 4096 (config.py + examples)
  - GenerationConfig.max_tokens default: 2048 → 4096
  - Update example configs
  - Test: config loads with 4096

Step G5: Update README and tests
  - Update README "Bring Your Own Domain" section with new config examples
  - Update test fixtures to pass step_qualities explicitly
  - Verify all 85+ CPU tests pass
  - Run GPU smoke test
```

**Execution rules:**
- Build ONE step per iteration
- After each step: run `python -m pytest tests/ -q` — must pass
- If test fails → MANDATORY Exa search before fix
- After all G steps: run GPU smoke test if GPU free
- When all pass → commit and push → report FEATURE LOOP COMPLETE

### Post-completion work (when user requests):
- `/commit` — commit all changes
- Move to training-dojo for hypergraph-scan-v2 run planning
- Create training data prompts for the new run
- The engine is pip-installable: `pip install -e .` from training-dojo

### Known constraints:
- GPU smoke tests must run individually — 16GB RTX 5080 can't load 3 models in sequence
- Primary GPU test: `pytest tests/test_smoke.py::test_three_steps_no_crash --gpu -v`
- Qwen3-1.7B for smoke tests, NOT 8B
- Never copy patterns from training-dojo v1
- Always search Exa before any bug fix
- `force_on_policy_ratio=True` disables ratio clipping (by design for on-policy, documented)

## Important: What This Command Does NOT Do

- Does NOT modify plan docs. The plan is the contract. If you find issues,
  report them to the user — don't silently "fix" the plan.
- Does NOT commit to git. The user decides when to commit.
- Does NOT make architectural decisions. If the plan doesn't specify
  something, ask the user rather than inventing.
- Does NOT skip phases. Every iteration runs all 6 phases. Phase 3 (API check)
  catches drift that accumulates silently.
- Does NOT touch files outside the current step's scope. One step at a time.

## Completion Promise

**NEVER report FEATURE LOOP COMPLETE without running this checklist:**

```
COMPLETION CHECKLIST (run every time before reporting done):
1. Read docs/PLAN.md — list every algorithm/feature described. Check each is implemented.
2. Read docs/SPECIAL-TOKENS-SUPERPOWER.md — list every technique. Check each is implemented or explicitly deferred.
3. Read docs/PILLARS.md — list every component per pillar. Check each exists.
4. grep for TODO/FIXME/STUB/placeholder in qgre/*.py — must be zero.
5. Run pytest — all tests must pass.
6. Check GameState is USED (not just stored) — engine must manage phase advancement.
7. Check reward scores flow into per-step advantages (not just sequence-level).
8. Check segmenter is called and regions affect advantage computation.
9. If ANY item fails → there are gaps. Do NOT report complete.
```

When ALL gaps are closed and the checklist passes:

```
FEATURE LOOP COMPLETE — zero gaps remaining
```

$ARGUMENTS
