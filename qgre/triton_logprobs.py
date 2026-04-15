"""Triton kernel: fused lm_head matmul + selective log_softmax + gather.

Computes logprobs from hidden states without materializing the full [seq, vocab]
logits tensor. Tiles along the vocab dimension (BLOCK_V=128) to compute:
  hidden[t] @ lm_head.T → logsumexp over vocab → gather at label[t] → logprob[t]

Peak VRAM: hidden_dim × BLOCK_V per thread block (not seq × vocab).
CRITICAL: BLOCK_V must be ≤ 128 for Qwen3 (vocab 151936 is divisible by 128 but NOT 256).

Reference: verl #2899 (wrong results when vocab % tile_size != 0).
Reference: Pablo Miralles (pablomirallesg.com/blog/fused-matmul-logsumexp).
Reference: Liger-Kernel PR #672 (fused GRPO loss, 46GB savings).
"""

from __future__ import annotations

import torch
from torch import nn


try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _fused_logprob_kernel(
        hidden_ptr,  # [seq, hidden_dim]
        weight_ptr,  # [vocab, hidden_dim] (lm_head.weight)
        label_ptr,  # [seq]
        output_ptr,  # [seq] logprobs
        lse_ptr,  # [seq] logsumexp (saved for backward)
        bias_ptr,  # [vocab] or null
        seq_len,
        hidden_dim: tl.constexpr,
        vocab_size,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,  # Must divide vocab_size (128 for Qwen3)
    ):
        """One program per sequence position. Tiles over vocab to compute logprob.

        Saves logsumexp per position for backward pass (Triton backward recomputes
        softmax from lse, ensuring self-consistency with forward).
        """
        pid = tl.program_id(0)
        if pid >= seq_len:
            return

        label = tl.load(label_ptr + pid)

        if label >= vocab_size or label < 0:
            # Invalid label: write -inf logprob, and lse=+inf so that
            # softmax=exp(logit-inf)=0 in the backward pass — safely zeros
            # out all contributions from this position. Using lse=0.0 would
            # cause exp(logit-0) on real logits to produce astronomical
            # wrong gradients in the backward kernel.
            tl.store(output_ptr + pid, float("-inf"))
            tl.store(lse_ptr + pid, float("inf"))
            return

        h_offs = tl.arange(0, hidden_dim)
        h = tl.load(hidden_ptr + pid * hidden_dim + h_offs).to(tl.float32)

        max_logit = float("-inf")
        label_logit = float("-inf")

        # First pass: find max logit + label logit
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            mask = v_offs < vocab_size
            w = tl.load(
                weight_ptr + v_offs[:, None] * hidden_dim + h_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            if HAS_BIAS:
                b = tl.load(bias_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)
                logits = logits + b
            tile_max = tl.max(tl.where(mask, logits, float("-inf")))
            max_logit = tl.maximum(max_logit, tile_max)
            label_in_tile = (label >= v_start) & (label < v_start + BLOCK_V)
            if label_in_tile:
                label_logit = tl.load(
                    weight_ptr + label * hidden_dim + h_offs,
                    mask=None,
                ).to(tl.float32)
                label_logit = tl.sum(label_logit * h)
                if HAS_BIAS:
                    label_logit = label_logit + tl.load(bias_ptr + label).to(tl.float32)

        # Second pass: compute sum(exp(logit - max))
        sum_exp = tl.zeros([], dtype=tl.float32)
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            mask = v_offs < vocab_size
            w = tl.load(
                weight_ptr + v_offs[:, None] * hidden_dim + h_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)
            logits = tl.sum(w * h[None, :], axis=1)
            if HAS_BIAS:
                b = tl.load(bias_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)
                logits = logits + b
            exp_vals = tl.exp(tl.where(mask, logits - max_logit, float("-inf")))
            sum_exp += tl.sum(exp_vals)

        sum_exp = tl.maximum(sum_exp, 1e-30)
        lse = max_logit + tl.log(sum_exp)
        log_prob = label_logit - lse
        log_prob = tl.minimum(log_prob, 0.0)
        tl.store(output_ptr + pid, log_prob)
        tl.store(lse_ptr + pid, lse)

    # ─── Backward kernels ─────────────────────────────────────────────────────

    @triton.jit
    def _fused_logprob_grad_hidden_kernel(
        hidden_ptr,  # [seq, hidden_dim]
        weight_ptr,  # [vocab, hidden_dim]
        label_ptr,  # [seq]
        lse_ptr,  # [seq] saved from forward
        grad_output_ptr,  # [seq]
        grad_hidden_ptr,  # [seq, hidden_dim] output
        bias_ptr,  # [vocab] or null
        seq_len,
        hidden_dim: tl.constexpr,
        vocab_size,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,
    ):
        """One program per position. Tiles over vocab to compute grad_hidden.

        grad_hidden[t] = grad_out[t] * (W[label,:] - sum_v softmax(v) * W[v,:])
        Recomputes softmax from saved lse — self-consistent with forward.
        """
        pid = tl.program_id(0)
        if pid >= seq_len:
            return

        h_offs = tl.arange(0, hidden_dim)
        h = tl.load(hidden_ptr + pid * hidden_dim + h_offs).to(tl.float32)
        g = tl.load(grad_output_ptr + pid).to(tl.float32)
        label = tl.load(label_ptr + pid)
        lse = tl.load(lse_ptr + pid).to(tl.float32)

        # Accumulate grad_hidden by tiling over vocab
        grad_h = tl.zeros([hidden_dim], dtype=tl.float32)

        for v_start in range(0, vocab_size, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            mask = v_offs < vocab_size

            w = tl.load(
                weight_ptr + v_offs[:, None] * hidden_dim + h_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)

            # Recompute logits (same path as forward)
            logits = tl.sum(w * h[None, :], axis=1)
            if HAS_BIAS:
                b = tl.load(bias_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)
                logits = logits + b

            # Softmax from saved lse (self-consistent with forward)
            softmax = tl.exp(tl.where(mask, logits - lse, float("-inf")))

            # diff = (one_hot - softmax): -softmax everywhere, +1 at label
            diff = -softmax
            label_in_tile = (label >= v_start) & (label < v_start + BLOCK_V)
            if label_in_tile:
                diff = tl.where(v_offs == label, diff + 1.0, diff)

            # grad_h += g * diff @ W  = g * sum_v(diff[v] * W[v, :])
            scaled_diff = g * diff  # [BLOCK_V]
            grad_h += tl.sum(scaled_diff[:, None] * w, axis=0)  # [hidden_dim]

        tl.store(grad_hidden_ptr + pid * hidden_dim + h_offs, grad_h)

    @triton.jit
    def _fused_logprob_grad_weight_kernel(
        hidden_ptr,  # [total_positions, hidden_dim]
        weight_ptr,  # [vocab, hidden_dim]
        label_ptr,  # [total_positions]
        lse_ptr,  # [total_positions] saved from forward
        grad_output_ptr,  # [total_positions]
        grad_weight_ptr,  # [vocab, hidden_dim] output
        bias_ptr,  # [vocab] or null
        total_positions,
        hidden_dim,
        vocab_size,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,  # Tile hidden dim to keep accumulator in registers
    ):
        """One program per (vocab_tile, hidden_tile). Iterates over positions.

        Accumulator: [BLOCK_V, BLOCK_D] in fp32 = 128×128×4 = 64 KB.
        Fits in registers. No spill to global memory.
        2D grid: (n_vocab_tiles, n_hidden_tiles).
        """
        vid = tl.program_id(0)
        did = tl.program_id(1)
        v_start = vid * BLOCK_V
        d_start = did * BLOCK_D
        v_offs = v_start + tl.arange(0, BLOCK_V)
        d_offs = d_start + tl.arange(0, BLOCK_D)
        v_mask = v_offs < vocab_size
        d_mask = d_offs < hidden_dim

        # Accumulator: [BLOCK_V, BLOCK_D] — 64 KB, fits in registers
        grad_w = tl.zeros([BLOCK_V, BLOCK_D], dtype=tl.float32)

        for t in range(total_positions):
            g = tl.load(grad_output_ptr + t).to(tl.float32)
            label = tl.load(label_ptr + t)
            lse = tl.load(lse_ptr + t).to(tl.float32)

            # Load h[t, d_start:d_start+BLOCK_D] for the outer product
            h_tile = tl.load(
                hidden_ptr + t * hidden_dim + d_offs,
                mask=d_mask,
                other=0.0,
            ).to(tl.float32)  # [BLOCK_D]

            # Recompute logits for this vocab tile — need full hidden dim
            # We compute logit[v] = sum_d(W[v,d] * h[t,d]) over ALL d
            # This requires loading W and h in full hidden_dim chunks
            logit_accum = tl.zeros([BLOCK_V], dtype=tl.float32)
            for k_start in range(0, hidden_dim, BLOCK_D):
                k_offs = k_start + tl.arange(0, BLOCK_D)
                k_mask = k_offs < hidden_dim
                w_chunk = tl.load(
                    weight_ptr + v_offs[:, None] * hidden_dim + k_offs[None, :],
                    mask=v_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)  # [BLOCK_V, BLOCK_D]
                h_k = tl.load(
                    hidden_ptr + t * hidden_dim + k_offs,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)  # [BLOCK_D]
                logit_accum += tl.sum(w_chunk * h_k[None, :], axis=1)

            if HAS_BIAS:
                b = tl.load(bias_ptr + v_offs, mask=v_mask, other=0.0).to(tl.float32)
                logit_accum = logit_accum + b

            # Softmax from saved lse
            softmax_v = tl.exp(tl.where(v_mask, logit_accum - lse, float("-inf")))

            # diff = (one_hot - softmax) * grad_out
            diff = -softmax_v
            diff = tl.where(v_offs == label, diff + 1.0, diff)
            scaled_diff = g * diff  # [BLOCK_V]

            # Outer product tile: grad_w += scaled_diff[:, None] * h_tile[None, :]
            grad_w += scaled_diff[:, None] * h_tile[None, :]  # [BLOCK_V, BLOCK_D]

        # Store the tile
        tl.store(
            grad_weight_ptr + v_offs[:, None] * hidden_dim + d_offs[None, :],
            grad_w,
            mask=v_mask[:, None] & d_mask[None, :],
        )

    @triton.jit
    def _fused_logprob_grad_bias_kernel(
        hidden_ptr,  # [total_positions, hidden_dim]
        weight_ptr,  # [vocab, hidden_dim]
        label_ptr,  # [total_positions]
        lse_ptr,  # [total_positions] saved from forward
        grad_output_ptr,  # [total_positions]
        grad_bias_ptr,  # [vocab] output
        bias_ptr,  # [vocab]
        total_positions,
        hidden_dim,
        vocab_size,
        BLOCK_V: tl.constexpr,
        BLOCK_D: tl.constexpr,  # Tile hidden dim for logit recomputation
    ):
        """One program per vocab tile. Iterates over positions.

        grad_bias[v] = sum_t (g[t] * (one_hot(label[t], v) - softmax[t, v]))

        Recomputes softmax from saved lse using the same element-wise path as
        forward/grad_hidden — self-consistent. No outer product with h, just
        a reduction over positions, so the accumulator is [BLOCK_V] only.
        1D grid: (n_vocab_tiles,).
        """
        vid = tl.program_id(0)
        v_start = vid * BLOCK_V
        v_offs = v_start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < vocab_size

        # Accumulator: [BLOCK_V] fp32, tiny.
        grad_b = tl.zeros([BLOCK_V], dtype=tl.float32)

        # Bias values for this tile — loaded once, reused across positions
        # inside the logit recomputation.
        b_tile = tl.load(bias_ptr + v_offs, mask=v_mask, other=0.0).to(tl.float32)

        for t in range(total_positions):
            g = tl.load(grad_output_ptr + t).to(tl.float32)
            label = tl.load(label_ptr + t)
            lse = tl.load(lse_ptr + t).to(tl.float32)

            # Recompute logits for this vocab tile — tile over hidden dim.
            logit_accum = tl.zeros([BLOCK_V], dtype=tl.float32)
            for k_start in range(0, hidden_dim, BLOCK_D):
                k_offs = k_start + tl.arange(0, BLOCK_D)
                k_mask = k_offs < hidden_dim
                w_chunk = tl.load(
                    weight_ptr + v_offs[:, None] * hidden_dim + k_offs[None, :],
                    mask=v_mask[:, None] & k_mask[None, :],
                    other=0.0,
                ).to(tl.float32)  # [BLOCK_V, BLOCK_D]
                h_k = tl.load(
                    hidden_ptr + t * hidden_dim + k_offs,
                    mask=k_mask,
                    other=0.0,
                ).to(tl.float32)  # [BLOCK_D]
                logit_accum += tl.sum(w_chunk * h_k[None, :], axis=1)

            logit_accum = logit_accum + b_tile

            # Softmax from saved lse
            softmax_v = tl.exp(tl.where(v_mask, logit_accum - lse, float("-inf")))

            # diff = one_hot(label) - softmax
            diff = -softmax_v
            diff = tl.where(v_offs == label, diff + 1.0, diff)

            grad_b += g * diff

        tl.store(grad_bias_ptr + v_offs, grad_b, mask=v_mask)


# Cached zero tensors used as the bias pointer argument when HAS_BIAS=False.
# Triton dead-code eliminates the bias loads (HAS_BIAS is tl.constexpr), but
# the kernel still needs a valid pointer argument. Cache is keyed by
# (vocab_size, device_str). Sized to vocab_size so speculative prefetch on
# any Triton build stays in-bounds.
_DUMMY_BIAS_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _dummy_bias_ptr(vocab_size: int, device: torch.device) -> torch.Tensor:
    """Return a cached zero tensor of shape [vocab_size] on the given device.

    Used as the bias pointer argument when HAS_BIAS=False, instead of
    aliasing hidden states (which can be smaller than vocab_size for short
    sequences and would read out-of-bounds under speculative prefetch).
    """
    key = (vocab_size, str(device))
    cached = _DUMMY_BIAS_CACHE.get(key)
    if cached is None:
        cached = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        _DUMMY_BIAS_CACHE[key] = cached
    return cached


def _validate_triton_inputs(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
) -> tuple[int, int, int, int, bool]:
    """Shared validation for Triton logprob paths. Returns (batch, seq_len, hidden_dim, vocab_size, has_bias)."""
    batch, seq_len, hidden_dim = hidden_states.shape
    vocab_size = weight.shape[0]
    has_bias = bias is not None

    if labels.numel() > 0:
        label_min = labels.min().item()
        label_max = labels.max().item()
        if label_min < 0 or label_max >= vocab_size:
            raise ValueError(
                f"Triton logprob: label out of bounds [{label_min}, {label_max}] "
                f"vs vocab_size={vocab_size}.",
            )

    BLOCK_V = 128
    if vocab_size % BLOCK_V != 0:
        raise ValueError(
            f"Triton logprob: vocab_size={vocab_size} not divisible by BLOCK_V={BLOCK_V}.",
        )

    ref_device = hidden_states.device
    if weight.device != ref_device:
        raise RuntimeError(
            f"Device mismatch: hidden_states on {ref_device}, weight on {weight.device}.",
        )
    if has_bias and bias.device != ref_device:
        raise RuntimeError(
            f"Device mismatch: hidden_states on {ref_device}, bias on {bias.device}.",
        )

    return batch, seq_len, hidden_dim, vocab_size, has_bias


def _run_triton_forward(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    labels: torch.Tensor,
    batch: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    has_bias: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run Triton forward. Returns (logprobs [batch, seq], lse [batch, seq])."""
    result = torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)
    lse_out = torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)
    BLOCK_V = 128
    # Shared sentinel when HAS_BIAS=False — sized to vocab_size and cached.
    bias_arg = bias if has_bias else _dummy_bias_ptr(vocab_size, hidden_states.device)

    for b in range(batch):
        h = hidden_states[b].contiguous()
        lab = labels[b].contiguous()

        _fused_logprob_kernel[(seq_len,)](
            h,
            weight,
            lab,
            result[b],
            lse_out[b],
            bias_arg,
            seq_len,
            hidden_dim,
            vocab_size,
            HAS_BIAS=has_bias,
            BLOCK_V=BLOCK_V,
        )

        if torch.isnan(result[b]).any():
            raise RuntimeError(
                f"NaN in Triton logprob output for batch {b}. "
                "Check model stability (inf/nan in hidden states).",
            )

    return result, lse_out


def triton_logprobs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute logprobs using Triton fused kernel — zero vocab-tensor allocation.

    No autograd graph — output is detached. Use triton_logprobs_with_grad
    for the differentiable version.

    Args:
        hidden_states: [batch, seq, hidden_dim] from model body
        lm_head: nn.Linear(hidden_dim, vocab_size)
        labels: [batch, seq] token IDs

    Returns:
        [batch, seq] log probabilities (float32), no grad_fn
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "triton_logprobs_from_hidden called but triton is not installed. "
            "Use chunked_logprobs_from_hidden for the cuBLAS path — the "
            "numerics differ, so the caller must choose explicitly.",
        )

    batch, seq_len, hidden_dim, vocab_size, has_bias = _validate_triton_inputs(
        hidden_states,
        lm_head.weight,
        lm_head.bias,
        labels,
    )

    if seq_len == 0 or batch == 0:
        return torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)

    logprobs, _lse = _run_triton_forward(
        hidden_states,
        lm_head.weight,
        lm_head.bias,
        labels,
        batch,
        seq_len,
        hidden_dim,
        vocab_size,
        has_bias,
    )
    return logprobs


# ─── Differentiable Triton logprobs (autograd.Function) ──────────────────────


class _TritonLogprobAutograd(torch.autograd.Function):
    """Full Triton forward + backward for differentiable logprob computation.

    Forward: Triton kernel tiles over vocab — zero [seq, vocab] allocation.
    Saves lse (logsumexp) per position for backward.

    Backward: Two Triton kernels, both tiling over vocab:
      1. grad_hidden: one program per position, tiles over vocab
      2. grad_weight: one program per vocab tile, iterates over positions

    Self-consistent: backward recomputes softmax from saved lse using the
    SAME element-wise path as forward. No cuBLAS. No dtype fights. No
    cross-implementation validation needed.

    No [seq, vocab] intermediate. No weight.float() allocation (1.2 GB saved).
    """

    @staticmethod
    def forward(ctx, hidden_states, weight, bias, labels) -> torch.Tensor:
        has_bias = bias is not None and bias.numel() > 0
        batch, seq_len, hidden_dim = hidden_states.shape
        vocab_size = weight.shape[0]

        result, lse = _run_triton_forward(
            hidden_states,
            weight,
            bias if has_bias else None,
            labels,
            batch,
            seq_len,
            hidden_dim,
            vocab_size,
            has_bias,
        )

        ctx.save_for_backward(hidden_states, weight, bias, labels, lse)
        ctx.has_bias = has_bias
        return result

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        hidden, weight, bias, labels, lse = ctx.saved_tensors
        has_bias = ctx.has_bias
        batch, seq_len, hidden_dim = hidden.shape
        vocab_size = weight.shape[0]
        BLOCK_V = 128

        # Bias gradient path: computed by a dedicated kernel below when the
        # bias both exists and was asked for. If bias is absent or the
        # autograd engine didn't mark it as needing grad, we return None.
        compute_grad_bias = has_bias and bias is not None and ctx.needs_input_grad[2]
        grad_bias: torch.Tensor | None = None

        # Empty sequence: no kernels to launch, return zero gradients.
        # Must respect needs_input_grad for all three learnable inputs —
        # returning None when the engine expects a tensor breaks accumulation.
        if batch == 0 or seq_len == 0:
            return (
                torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None,
                torch.zeros_like(weight) if ctx.needs_input_grad[1] else None,
                torch.zeros_like(bias) if compute_grad_bias else None,
                None,
            )

        # Flatten batch × seq so kernels process all positions in one launch.
        # No per-batch allocation — saves 1.2 GB vs looping per batch element.
        total_positions = batch * seq_len
        h_flat = hidden.reshape(total_positions, hidden_dim).contiguous()
        lab_flat = labels.reshape(total_positions).contiguous()
        lse_flat = lse.reshape(total_positions).contiguous()
        g_flat = grad_output.reshape(total_positions).contiguous()

        # Dummy pointer for the bias ptr arg when HAS_BIAS=False. Use a
        # cached vocab_size-sized sentinel instead of aliasing h_flat —
        # prevents speculative prefetch from reading past the hidden states
        # buffer for short sequences.
        bias_ptr = bias if has_bias else _dummy_bias_ptr(vocab_size, hidden.device)

        # Kernel 1: grad_hidden — Triton, one program per position.
        # Only allocate the output tensor if the gradient is actually needed.
        grad_hidden: torch.Tensor | None = None
        if ctx.needs_input_grad[0]:
            grad_hidden = torch.empty_like(hidden)
            grad_hidden_flat = grad_hidden.reshape(total_positions, hidden_dim)
            _fused_logprob_grad_hidden_kernel[(total_positions,)](
                h_flat,
                weight,
                lab_flat,
                lse_flat,
                g_flat,
                grad_hidden_flat,
                bias_ptr,
                total_positions,
                hidden_dim,
                vocab_size,
                HAS_BIAS=has_bias,
                BLOCK_V=BLOCK_V,
            )
            grad_hidden = grad_hidden.reshape(batch, seq_len, hidden_dim)

        # Kernel 2: grad_weight — 2D grid (vocab_tiles × hidden_tiles).
        # Accumulator per program: [BLOCK_V, BLOCK_D] = 64 KB fp32, fits in registers.
        grad_weight: torch.Tensor | None = None
        BLOCK_D = 128
        n_vocab_tiles = (vocab_size + BLOCK_V - 1) // BLOCK_V
        if ctx.needs_input_grad[1]:
            # weight.dtype (bf16 for Qwen3): kernel accumulates fp32 in
            # registers, tl.store truncates to output dtype. Saves 594 MB
            # vs fp32 allocation. SGD noise >> bf16 rounding; optimizer
            # upscales internally.
            grad_weight = torch.zeros(
                weight.shape,
                dtype=weight.dtype,
                device=weight.device,
            )
            n_hidden_tiles = (hidden_dim + BLOCK_D - 1) // BLOCK_D
            _fused_logprob_grad_weight_kernel[(n_vocab_tiles, n_hidden_tiles)](
                h_flat,
                weight,
                lab_flat,
                lse_flat,
                g_flat,
                grad_weight,
                bias_ptr,
                total_positions,
                hidden_dim,
                vocab_size,
                HAS_BIAS=has_bias,
                BLOCK_V=BLOCK_V,
                BLOCK_D=BLOCK_D,
            )

        # Kernel 3: grad_bias — 1D grid (vocab_tiles,). Only launched when a
        # real trainable bias exists and is needed. Accumulator per program
        # is [BLOCK_V] fp32 — tiny. Separate kernel instead of folding into
        # grad_weight because grad_weight has n_hidden_tiles programs per
        # vocab tile and bias is one-per-vocab; folding would compute it
        # n_hidden_tiles times per vocab element.
        if compute_grad_bias:
            # fp32 unconditionally — the kernel accumulates in fp32, and the
            # tensor is tiny ([vocab] = ~600 KB for 151K vocab). bf16 would
            # truncate for zero VRAM savings. Optimizer upscasts internally.
            grad_bias = torch.zeros(
                bias.shape,
                dtype=torch.float32,
                device=bias.device,
            )
            _fused_logprob_grad_bias_kernel[(n_vocab_tiles,)](
                h_flat,
                weight,
                lab_flat,
                lse_flat,
                g_flat,
                grad_bias,
                bias,  # Real bias — required; this path is only hit when has_bias=True
                total_positions,
                hidden_dim,
                vocab_size,
                BLOCK_V=BLOCK_V,
                BLOCK_D=BLOCK_D,
            )

        # H1-NUM003: Detect inf/nan in grad_weight before returning.
        # bf16 accumulation can overflow on extreme logits; inf gradients cause dead zones.
        if grad_weight is not None and not torch.isfinite(grad_weight).all():
            import warnings

            inf_count = (~torch.isfinite(grad_weight)).sum().item()
            warnings.warn(
                f"NUM003: grad_weight has {inf_count} non-finite values "
                f"(of {grad_weight.numel()} total). This may indicate logit explosion. "
                "Clamping to ±65504 (bf16 max) to prevent optimizer corruption.",
                stacklevel=3,
            )
            grad_weight = grad_weight.clamp(min=-65504, max=65504)

        return grad_hidden, grad_weight, grad_bias, None


def triton_logprobs_with_grad(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Differentiable logprobs via full Triton forward + backward.

    Drop-in replacement for chunked_logprobs_from_hidden. Three Triton
    kernels (grad_hidden, grad_weight, grad_bias) — no cuBLAS, no
    [seq, vocab] intermediate, self-consistent softmax recomputation
    from saved lse in both directions.

    Args:
        hidden_states: [batch, seq, hidden_dim] from model body
        lm_head: nn.Linear(hidden_dim, vocab_size)
        labels: [batch, seq] token IDs

    Returns:
        [batch, seq] log probabilities (float32) WITH grad_fn
    """
    if not HAS_TRITON:
        raise RuntimeError(
            "triton_logprobs_with_grad called but triton is not installed. "
            "Use chunked_logprobs_from_hidden for the cuBLAS path — the "
            "numerics differ, so the caller must choose explicitly.",
        )

    _validate_triton_inputs(hidden_states, lm_head.weight, lm_head.bias, labels)

    batch, seq_len = hidden_states.shape[:2]
    if seq_len == 0 or batch == 0:
        return torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)

    # Pass bias as empty tensor if None — autograd.Function needs consistent arg count.
    # Cached to avoid per-microbatch allocator traffic (common path for Qwen3).
    bias = lm_head.bias if lm_head.bias is not None else _dummy_bias_ptr(0, hidden_states.device)

    return _TritonLogprobAutograd.apply(hidden_states, lm_head.weight, bias, labels)
