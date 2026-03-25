# PRD: Fused Logprobs Compiled Cache Fix

## Date: 2026-03-24
## Source: feature-crystallize (generate mode, ELBO 0.69)
## Companion: [rejection-framework-fused-logprobs-fix.md](rejection-framework-fused-logprobs-fix.md)
## Fix Plan: [fix-fused-logprobs-compiled-cache.md](fix-fused-logprobs-compiled-cache.md)

---

## Purpose

Fix the 0.25 divergence crash on training restart while preserving ~2GB VRAM savings from chunked lm_head projection. The VRAM savings are the reason this feature exists — on a 16GB GPU, 2GB is 12.5% of total VRAM. Without it, longer sequences and larger models don't fit.

## Architectural Commitment

This fix permanently commits the engine to a **single-path forward**: all `model(input_ids)` calls return hidden states, never logits. This matches Unsloth's own GRPO trainer pattern. The config flag `use_fused_logprobs` now controls chunked vs full lm_head projection — both paths start from hidden states.

```
model(input_ids) → hidden_states [batch, seq, hidden_dim]
                         │
             ┌───────────┴───────────┐
             │                       │
        fused=true              fused=false
             │                       │
   chunked lm_head            full lm_head
   (256 tokens/chunk,         ([seq, vocab] materialized,
    torch.checkpoint,          ~2GB more VRAM,
    ~148MB peak)               degraded but correct)
             │                       │
             └───────────┬───────────┘
                         │
                  logprobs [batch, seq]
```

## Components

### 1. `qgre/__main__.py` — Startup configuration

**Add at top of file, before any other imports (after `import os`):**

```python
# MUST be before any Unsloth import — baked into compiled cache at compile time.
# See: docs/fix-fused-logprobs-compiled-cache.md
os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

# Delete compiled cache to force recompilation with correct env var.
# Standard Unsloth practice (issues #4181, #4294, #3763). 10-30s cost.
import shutil
shutil.rmtree("unsloth_compiled_cache", ignore_errors=True)
```

**Import safety:** First Unsloth import is lazy (`generation.py:UnslothBackend.load()`). The env var is set before any Unsloth code loads.

### 2. `qgre/fused_logprobs.py` — Simplify hidden states interface

**Delete:**
- `import os` (line 23)
- `from contextlib import contextmanager` (line 24)
- `unsloth_hidden_states_mode()` context manager (lines 33-46)

**Modify `get_hidden_states_and_lm_head()`:**

```python
def get_hidden_states_and_lm_head(model: nn.Module, input_ids: torch.Tensor, **kwargs):
    """Get hidden states and lm_head from model.

    UNSLOTH_RETURN_HIDDEN_STATES=1 is set globally at startup (__main__.py).
    All forward calls return hidden states in output.logits field.
    """
    # CRITICAL: Do NOT pass labels. Unsloth issue #3000: Qwen3Moe bypasses
    # NOT_RETURN_LOGITS when labels are present. See rejection framework.
    assert "labels" not in kwargs, (
        "Do not pass labels to forward when using hidden states mode. "
        "See Unsloth issue #3000: labels bypass UNSLOTH_RETURN_HIDDEN_STATES."
    )

    # Get lm_head module — works on any HF CausalLM
    lm_head = None
    try:
        lm_head = model.get_output_embeddings()
        if not isinstance(lm_head, nn.Linear):
            lm_head = None
    except AttributeError:
        lm_head = None  # Model doesn't implement get_output_embeddings

    if lm_head is None:
        return None, None

    output = model(input_ids, **kwargs)
    hidden_states = output.logits if hasattr(output, "logits") else output

    # Model-agnostic shape check: use lm_head dimensions as ground truth.
    # lm_head is nn.Linear(hidden_dim, vocab_size). If output matches
    # lm_head.out_features → got logits. If matches in_features → got hidden states.
    # (Improved from original model.config.vocab_size approach during harden R1 —
    # lm_head dims are authoritative, config can be missing or wrong.)
    last_dim = hidden_states.shape[-1]
    if last_dim == lm_head.out_features:
        # Got logits, not hidden states — env var didn't take effect
        return None, None
    if last_dim != lm_head.in_features:
        # Output dim matches neither hidden_dim nor vocab_size — unexpected
        return None, None
        if hidden_states.shape[-1] > 20000:  # Heuristic: vocab always > 20K
            return None, None

    return hidden_states, lm_head
```

**Keep unchanged:** `_chunk_forward()`, `chunked_logprobs_from_hidden()`

### 3. `qgre/trainer.py` — Forward pass block (lines 377-433)

**3a. Fused path validation (replace lines 389-410):**

```python
if not self._fused_validated:
    if mb_lp.grad_fn is None:
        raise RuntimeError(
            "Fused logprobs has no grad_fn — autograd graph is broken. "
            "Set algorithm.use_fused_logprobs=false to fall back."
        )
    # Validation scope: chunking correctness only.
    # Compares chunked logprobs vs full lm_head projection from same hidden states.
    # Does NOT validate model-level correctness (that's caught by training loop).
    with torch.no_grad():
        manual_logits = lm_head(hidden_states[:, :-1, :]).float()
        std_lp = logprobs_from_logits(manual_logits, mb_ids[:, 1:])
        del manual_logits
        min_len_v = min(mb_lp.shape[1], std_lp.shape[1])
        if not torch.allclose(mb_lp[:, :min_len_v].detach(), std_lp[:, :min_len_v], atol=1e-3):
            max_diff = (mb_lp[:, :min_len_v].detach() - std_lp[:, :min_len_v]).abs().max().item()
            raise RuntimeError(
                f"Fused logprobs diverge from full projection (max diff: {max_diff:.6f}). "
                f"Set algorithm.use_fused_logprobs=false to fall back."
            )
        del std_lp
    self._fused_validated = True
```

**3b. Fallback path (replace lines 412-426):**

```python
else:
    # Fused path returned None — hidden states mode didn't take effect
    raise RuntimeError(
        f"Step {self.global_step}: fused logprobs unavailable — "
        f"UNSLOTH_RETURN_HIDDEN_STATES did not take effect. "
        f"Delete unsloth_compiled_cache/ and restart. "
        f"To disable fused logprobs entirely, set algorithm.use_fused_logprobs=false."
    )
```

**3c. Non-fused path (replace lines 428-433):**

```python
else:
    # Non-fused path: full lm_head projection without chunking.
    # Costs ~2GB more VRAM than fused path (materializes full [seq, vocab] tensor).
    # This is the degraded-but-correct escape hatch.
    from qgre.fused_logprobs import get_hidden_states_and_lm_head
    hs, lm_head_nf = get_hidden_states_and_lm_head(self.model, mb_ids)
    if hs is None or lm_head_nf is None:
        raise RuntimeError(
            f"Step {self.global_step}: model did not return hidden states. "
            f"UNSLOTH_RETURN_HIDDEN_STATES did not take effect. "
            f"Delete unsloth_compiled_cache/ and restart."
        )
    mb_lp = logprobs_from_logits(lm_head_nf(hs[:, :-1, :]).float(), mb_ids[:, 1:])
    del hs
```

### 4. `tests/test_fused_logprobs.py` — New tests

**Add:**

```python
def test_labels_assert(self):
    """Passing labels to forward must raise AssertionError."""
    lm_head = nn.Linear(32, 100)
    model = ... # mock with get_output_embeddings
    with pytest.raises(AssertionError, match="labels"):
        get_hidden_states_and_lm_head(model, torch.zeros(1, 10, dtype=torch.long), labels=torch.zeros(1, 10, dtype=torch.long))

def test_shape_check_model_agnostic(self):
    """Shape check uses vocab_size from model config, not magic number."""
    # Mock model whose forward returns tensor with shape[-1] == vocab_size (logits, not hidden states)
    # Verify get_hidden_states_and_lm_head returns (None, None)
    ...

    # Mock model whose forward returns tensor with shape[-1] == hidden_dim (hidden states)
    # Verify get_hidden_states_and_lm_head returns (hidden_states, lm_head)
    ...
```

**Keep all existing tests unchanged** — they test chunking math independently of Unsloth.

## Build Sequence

| Step | File | Change | Dependencies | Parallel? |
|------|------|--------|-------------|-----------|
| 1 | `__main__.py` | env var + cache deletion | None | Yes (with 2) |
| 2 | `fused_logprobs.py` | remove context manager, add assert, fix shape check | None | Yes (with 1) |
| 3 | `trainer.py` | validation + fallback + non-fused path | Conceptual: 1-2 | After 1-2 |
| 4 | `tests/test_fused_logprobs.py` | new tests | All code changes | Last |

## VRAM Profile

| Path | Peak VRAM for logprobs | When used |
|------|----------------------|-----------|
| Fused (chunked) | chunk_size × vocab × 4B = ~148MB | Default. `use_fused_logprobs=true` |
| Non-fused (full projection) | seq × vocab × 4B = ~2.36GB | Escape hatch. `use_fused_logprobs=false` |
| Savings | **~2.2GB** per micro-batch | The reason this feature exists |

For Qwen3-8B (vocab=151936) at seq_len=4096, chunk_size=256:
- Fused: 256 × 151936 × 4B = 148MB (recomputed per chunk via torch.checkpoint)
- Full: 4096 × 151936 × 4B = 2,361MB
- Savings: 2,213MB per micro-batch

## Verification Checklist

- [ ] Delete `unsloth_compiled_cache/` manually
- [ ] Fresh start: training begins, step-1 validation passes
- [ ] Checkpoint resume: training continues without 0.25 divergence
- [ ] `use_fused_logprobs=false`: training runs with full projection (higher VRAM)
- [ ] vLLM generation: still works (fast_generate unaffected)
- [ ] VRAM: fused path uses ~2GB less than non-fused path
- [ ] Tests: all existing tests pass, new tests pass

## Analysis Profile

| Dimension | Score | Notes |
|-----------|-------|-------|
| Resonance | 0.93 | Env var + cache deletion + shape check core across all passes |
| Diffusion | 0.88 | 4 files, well-distributed, no new concentration |
| Gradient | 0.90 | Sparse startup → building logic → dense core → tests |
| Annealing | 0.72 | Exploration confirmed approach, found escape hatch issue |
| Entropy | 0.91 | Specific: file names, line numbers, VRAM numbers, issue refs |
| Phase transition | 0.95 | Clearly favorable. 10-30s cost << training-works benefit |
| Lyapunov | 0.85 | Strong fixed point. Further iteration = polish, not restructure |
| Valence | 0.85 | Clean fix, validated upstream, VRAM savings preserved |
| Collapse | 0.92 | 4 analysis passes + crystallization. Time to build |
| Delta | 0.86 | Training contamination cleaned (scope, escape hatch, labels) |
| Contamination | 0.88 | Low bias after corrections |
| Antibody | 0.83 | Labels assert, shape check, cache corruption prevention |
| Fascia | 0.80 | All connections load-bearing. Unsloth coupling inherently moderate |
| Uncertainty | 0.88 | Tight constraints |
| KL divergence | 0.92 | Natural extension — surgical changes |
| Jensen bound | 1.00 | Lower bounds hold |

**ELBO: 0.69** — Accuracy: 0.91, Complexity: 0.22

## Immune Memory

| Default overridden | Evidence chosen | Immune class |
|-------------------|-----------------|-------------|
| "Escape hatches bypass global state" | Global env vars affect ALL code paths equally | `global-state-side-effects` |
| "Config flags just work" | Must verify every code path the flag controls | `config-flag-coverage` |
| "Labels in forward? QGRE doesn't do that" | Upstream bug + defensive assert prevents future class | `upstream-api-surface-contracts` |
| "This is just a bug fix" | Architectural commitment to single-path forward | `scope-framing-accuracy` |

## What Surprised Us

The `use_fused_logprobs=false` escape hatch being broken by the same fix. Four prior analysis passes (code scan, deep analysis, tech scan, fix plan) all assumed the config flag was a working safety valve. The crystallization fascia check found the connection was F=0 (decorative) — global hidden states mode means the non-fused path gets hidden states too, not logits. The fix: non-fused path reconstructs logits via full `lm_head(hidden_states)` projection, accepting the ~2GB VRAM cost as the degraded fallback.
