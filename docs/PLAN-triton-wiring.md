# Plan: Wire Triton Fused Logprobs Into Training Loop

**Priority:** #1 for next run
**Expected savings:** ~1.5 GB GPU memory, ~2-3x faster logprob forward pass
**Target:** 9.5 GB training process (down from 11 GB)
**Risk:** Low — kernel written and tested, autograd.Function wrapper verified

## Background

`qgre/triton_logprobs.py` has a complete Triton kernel (`_fused_logprob_kernel`)
that computes logprobs from hidden states in a single GPU launch. It tiles
along the vocab dimension (BLOCK_V=128), never materializes the full
[seq, vocab] tensor, and runs entirely on-GPU with no Python round-trips.

The kernel was written in commit `ed5c5f6` (March 19, 2026) but was NEVER
enabled because it doesn't build a PyTorch autograd graph. The training
loss needs autograd for `curr_logprobs` (current policy) to flow gradients
back to the model. The kernel was disabled with a hardcoded
`self._use_triton_logprobs = False`.

The current path (`qgre/fused_logprobs.py`) uses PyTorch chunked lm_head
projection with `torch.checkpoint` per chunk. This works but has overhead:
- 16 Python → CUDA round-trips per forward pass (256-token chunks × 4096 seq)
- 16 MORE during backward (torch.checkpoint recomputes)
- ~500 MB for checkpoint metadata + 16 autograd CheckpointFunction nodes
- ~400 MB for fp32 intermediate tensors per chunk

## The Insight (Revised)

The plan originally proposed splitting old_logprobs from curr_logprobs.
On code review (April 2026), old_logprobs are ALREADY handled:
- **gen_logprobs from vLLM**: Captured during generation, stored per-sample,
  padded in trainer. No model forward pass needed. Already detached.
- **Fallback**: `mb_lp.detach()` — zero-cost detach of curr_logprobs.

The real savings come from replacing `chunked_logprobs_from_hidden` for
**curr_logprobs** with a Triton forward + PyTorch backward hybrid.

## Implementation (DONE — April 8, 2026)

### Step 1: `_TritonLogprobAutograd` (torch.autograd.Function)

`triton_logprobs.py` now has `_TritonLogprobAutograd`:
- **Forward**: Triton kernel — single GPU launch per batch element, zero [seq, vocab]
  allocation. Replaces 16 Python→CUDA round-trips.
- **Backward**: Chunked PyTorch matmul. Recomputes softmax per chunk (64 positions,
  peak 37 MB per chunk). Same memory profile as checkpoint recomputation.
- **Config**: `algorithm.use_triton_logprobs: bool = True` (default on)
- **Fallback**: Chunked path when Triton unavailable or on error (logged once)

### Step 1 verification (all pass):
- `test_triton_with_grad_matches_pytorch` — forward values match PyTorch reference
- `test_triton_with_grad_has_grad_fn` — autograd graph intact
- `test_triton_with_grad_backward_hidden` — non-zero, finite hidden gradients
- `test_triton_with_grad_backward_weight` — non-zero, finite lm_head.weight gradients
- `test_triton_with_grad_gradient_numerical` — gradient values match PyTorch reference
- `test_triton_with_grad_with_bias` — bias path works end-to-end
- `test_triton_with_grad_empty_sequence` — edge case handled

### Step 2: Evaluate whether LLDS old_logprobs can also use Triton

The LLDS (Log-Likelihood Displacement Score) computation also uses
generation-time logprobs. If these are also detached, they can go through
the Triton path too.

Check: `qgre/nemo_extracted/llds.py` — does `compute_llds_loss` need
autograd on the old logprobs? If not, switch to Triton.

### Step 3: Investigate Triton backward kernel for curr_logprobs

### Step 2: Triton backward kernel (future optimization)

The current backward uses chunked PyTorch matmul. A Triton backward kernel
could tile the `W^T @ (softmax - one_hot)` computation the same way the
forward tiles the logit computation. This would eliminate the [chunk, vocab]
intermediate tensor in backward too.

Math: `grad_hidden[t] = grad_out[t] * W^T @ (one_hot(label[t]) - softmax(W @ h[t]))`

The challenge: backward also needs `grad_weight` (lm_head is in modules_to_save).
`grad_weight += (grad_out * diff).T @ h_chunk` is a [vocab, hidden] accumulation
that's harder to tile without atomics. Likely needs `tl.atomic_add` or a
reduction tree.

**Decision**: Defer until profiling shows backward is the bottleneck. The
forward savings alone justify Step 1.

### Step 3: Wire the four QGRE-specific Triton kernels (concept 312)

After logprobs are on Triton, the next targets are:

1. Fused segment_completion — parallel token ID pattern matching
2. Fused advantage_broadcast — region→advantage lookup table
3. Fused LLDS gate — three-level element-wise ops
4. Fused region-KL-weighted loss — region lookup + KL + masking

Each replaces a Python loop with a GPU-parallel kernel.

### Step 5: MLX backend for Mac training

Same algorithm layer, MLX tensor ops instead of PyTorch/Triton. This is
a separate backend, not a modification of the Triton path.

## Memory Budget After Triton Wiring

### Step 1 only (old_logprobs via Triton):
```
Current:    7,700 MiB training process
Step 1:    ~6,500 MiB (-1,200 MiB from no autograd on old_logprobs)
```

### Steps 1-3 (both logprob paths via Triton):
```
Step 1-3:  ~5,000 MiB (-2,700 MiB total)
```

### With LoRA-Pro at rank 16:
```
Step 1-3 + LoRA-Pro rank 16: ~7,000 MiB (5,000 + 2,000 for LoRA-Pro)
```

Still fits in 16 GB with desktop overhead.

## Files Modified (Step 1)

- `qgre/triton_logprobs.py` — added `_TritonLogprobAutograd`, `triton_logprobs_with_grad`,
  refactored validation/forward into shared helpers
- `qgre/trainer.py` — conditional Triton path in fused logprobs section, lazy detection,
  one-time validation against full lm_head projection
- `qgre/config.py` — added `use_triton_logprobs: bool = True`
- `tests/test_triton_logprobs.py` — 7 new tests for differentiable wrapper

## Verification

1. Numerical equivalence: `torch.allclose(triton, pytorch, atol=1e-3)` ✓ (10/10 tests pass)
2. Gradient correctness: hidden and weight gradients match PyTorch reference ✓
3. Memory measurement: `torch.cuda.max_memory_allocated()` before and after (TODO: next run)
4. Speed measurement: wall-clock per step comparison (TODO: next run)
5. Training correctness: run with `use_triton_logprobs: true`, verify reward trajectory (TODO: next run)

## Dependencies

- Triton 3.5+ (installed ✓)
- CUDA compute capability 12.0+ (RTX 5080 ✓)
- `qgre/triton_logprobs.py` forward kernel (written ✓)
- `_TritonLogprobAutograd` backward (written ✓, 7 tests passing ✓)
