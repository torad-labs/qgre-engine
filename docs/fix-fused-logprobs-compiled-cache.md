# Fix: Fused Logprobs Compiled Cache Issue

## Date: 2026-03-23
## Updated: 2026-03-24 (code scan + deep analysis + tech scan)

## Problem

The fused logprobs path crashes with `RuntimeError: Fused logprobs diverge from standard path (max diff: 0.250000)` on every training restart.

## Root Cause

**UNSLOTH_RETURN_HIDDEN_STATES is evaluated at COMPILE TIME, not runtime.**

Unsloth's `compiler.py` generates compiled forward modules (stored in `unsloth_compiled_cache/`). When the module is generated, it reads `os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES")` and bakes the result into a static variable `NOT_RETURN_LOGITS` in the generated Python code. Toggling the env var after compilation has ZERO effect — the compiled graph is frozen.

### Evidence

- Unsloth issue #3126 (open, Aug 2025), user @sbhavani: *"I don't think the UNSLOTH_RETURN_LOGITS=1 env var is used because unsloth's compiled modules evaluate the env var at compile time, not runtime."* **Still unfixed upstream as of 2026-03-24.**
- Issue #3000 (open, Jul 2025): The compiled code has `NOT_RETURN_LOGITS` as a static variable. The `if not NOT_RETURN_LOGITS: logits = self.lm_head(hidden_states...)` branch is hardcoded at compile time.
- Our standalone test (fresh model load) returns hidden states correctly because no cached compiled module exists.
- Training (checkpoint resume) uses the pre-compiled module from `unsloth_compiled_cache/` which was compiled with `RETURN_HIDDEN_STATES=0`.
- `rm -rf unsloth_compiled_cache` is the standard Unsloth maintainer recommendation for cache issues (seen in issues #4181, #4294, #3763). Our Step 2 is validated by upstream practice.

### What Unsloth's GRPO Trainer Does

1. Sets `os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"` **BEFORE** any Unsloth import
2. The compiled module is generated WITH the flag enabled
3. ALL forward calls permanently return hidden states — no toggling
4. Gets `lm_head` via `model.get_output_embeddings()`
5. Uses `chunked_hidden_states_selective_log_softmax(hidden_states, lm_head, ...)` for chunked logprobs

### Current Code State (from code scan 2026-03-24)

- `__main__.py` does NOT set `UNSLOTH_RETURN_HIDDEN_STATES` before imports. The env var is only toggled at runtime inside the context manager in `fused_logprobs.py` — which has zero effect on the compiled cache.
- `fused_logprobs.py:33-46` — `unsloth_hidden_states_mode()` context manager toggles the env var at runtime. This is dead code on resume because the compiled cache already has the flag baked in.
- `trainer.py:389-410` — validation does a full `self.model(mb_ids)` standard forward to compare against fused. Once hidden states mode is global, this call ALSO returns hidden states, making the comparison meaningless.
- `trainer.py:412-426` — **latent bug in fallback path**: line 423 assigns `mb_output = self.model(mb_ids)` then deletes it, but line 425 references `mb_logits` which was never assigned. This would crash if the fallback ever triggers. Must fix.
- `generation.py` — vLLM path uses `model.fast_generate()` which goes through vLLM's own inference engine, completely separate from Unsloth's compiled forward. Confirmed clean.
- No compiled cache deletion exists anywhere in the codebase currently.

## Architectural Consequence: Single-Path Forward

This fix permanently eliminates the standard forward path from the engine. Once `UNSLOTH_RETURN_HIDDEN_STATES=1` is global, every `model(input_ids)` call returns hidden states — there is no way to get logits from a forward call. This is intentional and matches what Unsloth's own GRPO trainer does. The consequence is that every safety mechanism downstream (validation, fallback) must be designed for a single-path world where hidden states are the only forward output. There is no "standard path" to compare against or fall back to.

## Solution

### Step 1: Set env var before imports in `__main__.py`

```python
import os
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"  # MUST be before any Unsloth import
```

**Import safety check:** The first Unsloth import in the codebase is `from unsloth import FastLanguageModel` inside `generation.py:UnslothBackend.load()` — a lazy import inside a method, not a module-level import. `__main__.py` imports `qgre.config` (no Unsloth) before calling `cmd_train()` which lazy-imports `qgre.generation`. So setting the env var at the top of `__main__.py` is safe — it fires before any Unsloth code loads.

### Step 2: Delete compiled cache on startup

```python
import shutil
cache_dir = "unsloth_compiled_cache"
if os.path.isdir(cache_dir):
    shutil.rmtree(cache_dir)
```

This forces Unsloth to recompile the forward module with `RETURN_HIDDEN_STATES=1` baked in. The 10-30s recompile cost is noise against a training run.

### Step 3: Simplify fused_logprobs.py

Remove the `unsloth_hidden_states_mode()` context manager entirely. ALL forward calls now return hidden states. The function becomes:

```python
def get_hidden_states_and_lm_head(model, input_ids, **kwargs):
    # No env var toggling needed — it's set globally at startup
    #
    # CRITICAL: Do NOT pass `labels` in kwargs. Unsloth issue #3000 (open):
    # Qwen3Moe (and possibly other architectures) has a code path where
    # `labels is not None` bypasses the NOT_RETURN_LOGITS check entirely,
    # jumping straight to fused CE loss. The env var is ignored in that branch.
    # QGRE never passes labels to the forward call — this must stay that way.
    assert "labels" not in kwargs, (
        "Do not pass labels to forward when using hidden states mode. "
        "See Unsloth issue #3000: labels bypass UNSLOTH_RETURN_HIDDEN_STATES."
    )
    output = model(input_ids, **kwargs)
    hidden_states = output.logits  # Returns hidden states when env var is set at compile time

    # Model-agnostic shape check: use lm_head dimensions as ground truth.
    # lm_head is nn.Linear(hidden_dim, vocab_size). If output matches
    # lm_head.out_features → got logits. If matches in_features → got hidden states.
    # (Improved from original model.config.vocab_size approach during harden R1 —
    # lm_head dims are authoritative, config can be missing or wrong.)
    last_dim = hidden_states.shape[-1]
    if last_dim == lm_head.out_features:
        return None, None  # Got logits, not hidden states
    if last_dim != lm_head.in_features:
        return None, None  # Unexpected dim — corrupted or unsupported architecture

    return hidden_states, lm_head
```

### Step 4: Update validation in trainer.py

The standard forward can't produce logits anymore (all forwards return hidden states). The validation needs to compute logits manually from the same hidden states + lm_head:

```python
if not self._fused_validated:
    # Compute logits manually from hidden states + lm_head
    with torch.no_grad():
        manual_logits = lm_head(hidden_states[:, :-1, :]).float()
        std_lp = logprobs_from_logits(manual_logits, mb_ids[:, 1:])
        # Compare fused chunked logprobs vs manual full logprobs
        if not torch.allclose(mb_lp.detach(), std_lp, atol=1e-3):
            raise RuntimeError(...)
```

This compares chunked logprobs (from fused path) against full logprobs (from same hidden states + lm_head) — both use the same precision, so divergence should be ~0.

**Validation scope (explicit):** This validation checks **chunking correctness only** — that splitting hidden states into chunks and projecting each through lm_head produces the same result as projecting the full sequence at once. It does NOT validate model-level correctness (wrong hidden states, dtype mismatch, corrupted weights). Model-level correctness is validated by the training loop itself — NaN loss, zero rewards, or diverging metrics would catch those failures. This is a deliberate scope reduction from the original cross-path validation. The original could catch both chunking and model errors but relied on a standard forward that no longer exists.

### Step 5: Replace fallback path in trainer.py

The current fallback (trainer.py:412-426) has two problems: a variable reference bug (`mb_logits` never assigned), and a conceptual one — with global hidden states mode, `self.model(mb_ids)` returns hidden states, not logits, so the fallback can't produce logits via standard forward either. There is no standard path to fall back to.

**Decision: the fallback raises RuntimeError with diagnostic info.**

```python
# Fused path returned None — model did not return hidden states
raise RuntimeError(
    f"Step {self.global_step}: fused logprobs unavailable — "
    f"UNSLOTH_RETURN_HIDDEN_STATES did not take effect. "
    f"Delete unsloth_compiled_cache/ and restart. "
    f"To disable fused logprobs entirely, set algorithm.use_fused_logprobs=false."
)
```

A clear crash with actionable diagnostics is better than silent degradation. There is no independent forward path to fall back to — attempting one produces the same hidden states. The `use_fused_logprobs=false` config flag remains as the manual escape hatch to the standard (full logits) path, which requires the env var to NOT be set (i.e., a code change to `__main__.py`).

### Step 6: Handle vLLM generation

vLLM uses its own inference engine (`model.fast_generate()` in `generation.py`) and doesn't go through Unsloth's compiled forward. Confirmed clean from code scan — vLLM loads base model weights directly through its own engine.

## Files to Modify

- `qgre/__main__.py` — set env var + delete compiled cache (before any imports that touch Unsloth)
- `qgre/fused_logprobs.py` — remove context manager, simplify to always-hidden-states, model-agnostic shape check
- `qgre/trainer.py` — update validation to use manual logits from hidden_states, fix fallback path bug
- `tests/test_fused_logprobs.py` — update tests

## Verification

1. Delete `unsloth_compiled_cache/` directory
2. Start training — compiled module should be regenerated with hidden states mode
3. Step-1 validation passes (chunked logprobs ≈ manual logprobs within 1e-3)
4. No 0.25 divergence
5. vLLM generation still works normally
6. Memory savings: peak VRAM should be ~2GB lower than standard path

## Risk

- Deleting compiled cache adds ~10-30s cold start on first training step (recompilation). This is standard Unsloth practice — maintainers recommend `rm -rf unsloth_compiled_cache` as first troubleshooting step.
- If Unsloth removes UNSLOTH_RETURN_HIDDEN_STATES env var in a future version, the lm_head dimension check (`last_dim == lm_head.out_features`) will catch it and trigger the RuntimeError with diagnostics. PR #2772 shows Unsloth internally moving toward CCE-based logprob computation that may eventually replace the env var mechanism — the shape check is our insurance.
- vLLM generation confirmed unaffected — uses its own engine, never touches Unsloth compiled forward.
- **Labels constraint (issue #3000, open):** If `labels` are ever passed to the forward call, Qwen3Moe (and possibly other MoE architectures) bypasses the `NOT_RETURN_LOGITS` check and returns logits regardless of the env var. QGRE currently never passes labels — this is enforced by an assert in `get_hidden_states_and_lm_head`. If future QGRE features require labels in the forward call, this constraint must be revisited.
- Compiled cache can occasionally generate syntactically invalid Python (issue #3763). Unconditional cache deletion on startup prevents corrupted modules from persisting.

## Tech Scan Validation (2026-03-24)

Exa live search across GitHub issues, Unsloth docs, and Chinese ML forums. Key findings:

- **#3126 (open):** Compile-time env var baking confirmed by multiple users. Still unfixed upstream. Our fix is the correct and only workaround.
- **#3000 (open):** `UNSLOTH_RETURN_HIDDEN_STATES` ignored when `labels` passed to forward on Qwen3Moe. QGRE is safe (no labels in forward), but the constraint is now enforced with an assert.
- **PR #2772:** Unsloth internally exploring CCE-based logprob computation that may replace the env var mechanism. Our shape check provides resilience against this change.
- **#3763:** Compiled cache can generate syntactically invalid Python. Unconditional deletion prevents corrupted modules.
- **#4181, #4294:** `rm -rf unsloth_compiled_cache` is standard Unsloth maintainer advice. Our Step 2 is validated.
- **Chinese forums (CSDN):** No unique findings beyond English sources. Consistently recommend cache deletion after Unsloth upgrades.
- **No blockers found.** Fix plan validated. One addition: `labels` assert in `get_hidden_states_and_lm_head`.

## References

- Unsloth issue #3126 (compile-time env var, open): https://github.com/unslothai/unsloth/issues/3126
- Unsloth issue #3000 (labels bypass hidden states, open): https://github.com/unslothai/unsloth/issues/3000
- Unsloth issue #3763 (corrupted compiled cache): https://github.com/unslothai/unsloth/issues/3763
- Unsloth issue #4181 (cache deletion as standard fix): https://github.com/unslothai/unsloth/issues/4181
- Unsloth PR #2772 (CCE-based logprobs, env var removal): https://github.com/unslothai/unsloth/pull/2772
- Unsloth PR #1831 (Mistral GRPO fix): https://github.com/unslothai/unsloth/pull/1831
- Unsloth GRPO long context docs: https://unsloth.ai/docs/new/grpo-long-context
- Unsloth Memory Efficient RL docs: https://docs.unsloth.ai/basics/memory-efficient-rl
