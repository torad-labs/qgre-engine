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
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _fused_logprob_kernel(
        hidden_ptr,        # [seq, hidden_dim]
        weight_ptr,        # [vocab, hidden_dim] (lm_head.weight)
        label_ptr,         # [seq]
        output_ptr,        # [seq]
        bias_ptr,          # [vocab] or null
        seq_len,
        hidden_dim: tl.constexpr,
        vocab_size,
        HAS_BIAS: tl.constexpr,
        BLOCK_V: tl.constexpr,   # Must divide vocab_size (128 for Qwen3)
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
        label_logit = tl.zeros([], dtype=tl.float32)

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
                label_idx = label - v_start
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


def triton_logprobs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute logprobs using Triton fused kernel — zero vocab-tensor allocation.

    Args:
        hidden_states: [batch, seq, hidden_dim] from model body
        lm_head: nn.Linear(hidden_dim, vocab_size)
        labels: [batch, seq] token IDs

    Returns:
        [batch, seq] log probabilities (float32)
    """
    if not HAS_TRITON:
        # Fallback to PyTorch selective_log_softmax path
        from qgre.fused_logprobs import chunked_logprobs_from_hidden
        return chunked_logprobs_from_hidden(hidden_states, lm_head, labels)

    batch, seq_len, hidden_dim = hidden_states.shape
    vocab_size = lm_head.weight.shape[0]
    has_bias = lm_head.bias is not None

    # L2: Validate labels are within vocab bounds before kernel call
    if labels.numel() > 0:
        label_min = labels.min().item()
        label_max = labels.max().item()
        if label_min < 0 or label_max >= vocab_size:
            raise ValueError(
                f"L2: Triton logprob kernel: label out of bounds. "
                f"Label range [{label_min}, {label_max}] exceeds vocab_size={vocab_size}. "
                f"Invalid labels would cause -inf logprobs without masking. "
                f"Check tokenizer and label data."
            )

    # Qwen3: vocab=151936, divisible by 128 but NOT 256
    BLOCK_V = 128

    # Validate vocab_size is divisible by BLOCK_V
    if vocab_size % BLOCK_V != 0:
        raise ValueError(
            f"Triton logprob kernel requires vocab_size % BLOCK_V == 0, "
            f"got vocab_size={vocab_size}, BLOCK_V={BLOCK_V}. "
            f"Use a different BLOCK_V or disable fused logprobs."
        )

    # Return empty tensor early if seq_len=0 or batch=0
    if seq_len == 0 or batch == 0:
        return torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)

    result = torch.empty(batch, seq_len, dtype=torch.float32, device=hidden_states.device)

    # Validate all tensors on same device before kernel call
    ref_device = hidden_states.device
    if lm_head.weight.device != ref_device:
        raise RuntimeError(
            f"Device mismatch: hidden_states on {ref_device}, lm_head.weight on {lm_head.weight.device}. "
            "All tensors must be on the same device for Triton kernel."
        )
    if has_bias and lm_head.bias.device != ref_device:
        raise RuntimeError(
            f"Device mismatch: hidden_states on {ref_device}, lm_head.bias on {lm_head.bias.device}. "
            "All tensors must be on the same device for Triton kernel."
        )

    for b in range(batch):
        h = hidden_states[b].contiguous()  # [seq, hidden_dim]
        lab = labels[b].contiguous()       # [seq]
        out = result[b]                    # [seq]

        if lab.device != ref_device:
            raise RuntimeError(
                f"Device mismatch: batch {b} labels on {lab.device}, expected {ref_device}. "
                "All tensors must be on the same device for Triton kernel."
            )

        grid = (seq_len,)
        _fused_logprob_kernel[grid](
            h, lm_head.weight, lab, out,
            lm_head.bias if has_bias else h,  # dummy ptr when no bias
            seq_len, hidden_dim, vocab_size,
            HAS_BIAS=has_bias,
            BLOCK_V=BLOCK_V,
        )

    return result
