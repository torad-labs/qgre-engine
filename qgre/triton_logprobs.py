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
        output_ptr,  # [seq]
        bias_ptr,  # [vocab] or null
        seq_len,
        hidden_dim: tl.constexpr,
        vocab_size,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,  # Must divide vocab_size (128 for Qwen3)
    ):
        """One program per sequence position. Tiles over vocab to compute logprob."""
        pid = tl.program_id(0)
        if pid >= seq_len:
            return

        # Load the label for this position
        label = tl.load(label_ptr + pid)

        # GEN-R2-1: Validate label bounds
        if label >= vocab_size or label < 0:
            # Out-of-bounds label → return -inf (zero probability)
            tl.store(output_ptr + pid, float("-inf"))
            return

        # Load hidden state for this position: [hidden_dim]
        h_offs = tl.arange(0, hidden_dim)
        h = tl.load(hidden_ptr + pid * hidden_dim + h_offs).to(tl.float32)

        # Compute logsumexp and gather the label logit in one pass over vocab tiles
        max_logit = float("-inf")
        label_logit = float("-inf")

        # First pass: find max logit (for numerical stability) + label logit
        for v_start in range(0, vocab_size, BLOCK_V):
            v_offs = v_start + tl.arange(0, BLOCK_V)
            mask = v_offs < vocab_size

            # Load weight tile: [BLOCK_V, hidden_dim] — transposed matmul
            # w[v, d] = weight_ptr[v * hidden_dim + d]
            w = tl.load(
                weight_ptr + v_offs[:, None] * hidden_dim + h_offs[None, :],
                mask=mask[:, None],
                other=0.0,
            ).to(tl.float32)

            # Dot product: logits[v] = sum(h * w[v]) for v in [v_start, v_start+BLOCK_V)
            logits = tl.sum(w * h[None, :], axis=1)  # [BLOCK_V]

            if HAS_BIAS:
                b = tl.load(bias_ptr + v_offs, mask=mask, other=0.0).to(tl.float32)
                logits = logits + b

            # Update running max
            tile_max = tl.max(tl.where(mask, logits, float("-inf")))
            max_logit = tl.maximum(max_logit, tile_max)

            # Gather label logit if in this tile
            label_in_tile = (label >= v_start) & (label < v_start + BLOCK_V)
            if label_in_tile:
                # GB3-003: Explicit -inf assignment if label not found (was implicit)
                label_logit = tl.load(
                    weight_ptr + label * hidden_dim + h_offs,
                    mask=None,
                ).to(tl.float32)
                label_logit = tl.sum(label_logit * h)
                if HAS_BIAS:
                    label_logit = label_logit + tl.load(bias_ptr + label).to(tl.float32)

        # Second pass: compute sum(exp(logit - max)) for logsumexp
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

        # GEN-R2-3: Clamp sum_exp to prevent log(0) = -inf → result = +inf
        sum_exp = tl.maximum(sum_exp, 1e-30)

        # log_softmax(label) = label_logit - max_logit - log(sum_exp)
        log_prob = label_logit - max_logit - tl.log(sum_exp)
        # LC5: Clamp to ≤0 — logprobs must be in [-inf, 0]; positive values indicate numerical issues
        log_prob = tl.minimum(log_prob, 0.0)
        tl.store(output_ptr + pid, log_prob)


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
) -> torch.Tensor:
    """Run the Triton kernel over all batches. Returns [batch, seq] logprobs."""
    result = torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)
    BLOCK_V = 128

    for b in range(batch):
        h = hidden_states[b].contiguous()
        lab = labels[b].contiguous()
        out = result[b]

        _fused_logprob_kernel[(seq_len,)](
            h,
            weight,
            lab,
            out,
            bias if has_bias else h,  # dummy ptr when no bias
            seq_len,
            hidden_dim,
            vocab_size,
            HAS_BIAS=has_bias,
            BLOCK_V=BLOCK_V,
        )

        if torch.isnan(out).any():
            raise RuntimeError(
                f"NaN in Triton logprob output for batch {b}. "
                "Check model stability (inf/nan in hidden states).",
            )

    return result


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
        from qgre.fused_logprobs import chunked_logprobs_from_hidden

        return chunked_logprobs_from_hidden(hidden_states, lm_head, labels)

    batch, seq_len, hidden_dim, vocab_size, has_bias = _validate_triton_inputs(
        hidden_states,
        lm_head.weight,
        lm_head.bias,
        labels,
    )

    if seq_len == 0 or batch == 0:
        return torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)

    return _run_triton_forward(
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


# ─── Differentiable Triton logprobs (autograd.Function) ──────────────────────


class _TritonLogprobAutograd(torch.autograd.Function):
    """Triton forward + PyTorch backward for differentiable logprob computation.

    Forward: fused Triton kernel — one GPU launch per batch element, zero [seq, vocab]
    allocation. Replaces 16 Python→CUDA round-trips from chunked lm_head + checkpoint.

    Backward: chunked PyTorch matmul. Recomputes softmax per chunk (same as
    torch.checkpoint recomputation in the chunked path). Peak per chunk:
    chunk_size × vocab × 4 bytes (e.g., 64 × 151936 × 4 = 37 MB).

    Net savings vs chunked_logprobs_from_hidden:
    - Forward: single Triton launch vs 16 Python→CUDA round-trips
    - No torch.checkpoint metadata (~500 MB)
    - No 16 autograd CheckpointFunction nodes
    - Backward: same memory profile as checkpoint recomputation
    """

    BACKWARD_CHUNK_SIZE = 64

    @staticmethod
    def forward(ctx, hidden_states, weight, bias, labels) -> torch.Tensor:
        has_bias = bias is not None and bias.numel() > 0
        batch, seq_len, hidden_dim = hidden_states.shape
        vocab_size = weight.shape[0]

        result = _run_triton_forward(
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

        ctx.save_for_backward(hidden_states, weight, bias, labels)
        ctx.has_bias = has_bias
        return result

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, None]:
        hidden, weight, bias, labels = ctx.saved_tensors
        has_bias = ctx.has_bias
        batch, seq_len, _hidden_dim = hidden.shape
        chunk_size = _TritonLogprobAutograd.BACKWARD_CHUNK_SIZE

        grad_hidden = torch.zeros_like(hidden)
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        grad_bias = (
            torch.zeros(bias.shape, dtype=torch.float32, device=bias.device)
            if has_bias and bias is not None and bias.requires_grad
            else None
        )

        for b in range(batch):
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                c = end - start
                h_chunk = hidden[b, start:end].float()  # [C, D]

                # Recompute logits (same as Triton kernel, but in PyTorch for backward)
                logits = h_chunk @ weight.T  # [C, V]
                if has_bias and bias is not None:
                    logits = logits + bias.unsqueeze(0)

                # diff = one_hot(label) - softmax(logits)
                probs = torch.softmax(logits, dim=-1)  # [C, V]
                del logits
                probs.neg_()  # -softmax
                lab = labels[b, start:end]
                probs.scatter_add_(
                    1,
                    lab.unsqueeze(1),
                    torch.ones(c, 1, device=probs.device, dtype=probs.dtype),
                )
                # probs is now (one_hot - softmax)

                g = grad_output[b, start:end].unsqueeze(1)  # [C, 1]
                g_diff = g * probs  # [C, V] — scaled gradient

                # grad_hidden: [C, V] @ [V, D] → [C, D]
                grad_hidden[b, start:end] = (g_diff @ weight).to(hidden.dtype)

                # grad_weight: [V, D] += [V, C] @ [C, D]
                grad_weight.addmm_(g_diff.T, h_chunk)

                # grad_bias: [V] += sum([C, V], dim=0)
                if grad_bias is not None:
                    grad_bias.add_(g_diff.sum(0))

                del probs, g_diff

        return grad_hidden, grad_weight.to(weight.dtype), grad_bias, None


def triton_logprobs_with_grad(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Differentiable logprobs via Triton forward + PyTorch backward.

    Drop-in replacement for chunked_logprobs_from_hidden. Uses the Triton kernel
    for the forward pass (zero vocab-tensor allocation, single GPU launch), and
    chunked PyTorch matmul for backward (same memory profile as checkpoint path).

    Args:
        hidden_states: [batch, seq, hidden_dim] from model body
        lm_head: nn.Linear(hidden_dim, vocab_size)
        labels: [batch, seq] token IDs

    Returns:
        [batch, seq] log probabilities (float32) WITH grad_fn
    """
    if not HAS_TRITON:
        from qgre.fused_logprobs import chunked_logprobs_from_hidden

        return chunked_logprobs_from_hidden(hidden_states, lm_head, labels)

    _validate_triton_inputs(hidden_states, lm_head.weight, lm_head.bias, labels)

    batch, seq_len = hidden_states.shape[:2]
    if seq_len == 0 or batch == 0:
        return torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)

    # Pass bias as empty tensor if None — autograd.Function needs consistent arg count
    bias = (
        lm_head.bias
        if lm_head.bias is not None
        else torch.empty(
            0,
            device=hidden_states.device,
        )
    )

    return _TritonLogprobAutograd.apply(hidden_states, lm_head.weight, bias, labels)
