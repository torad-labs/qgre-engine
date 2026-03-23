"""Fused chunked logprob computation — Phase 4 of PLAN.md.

Computes log probabilities WITHOUT materializing the full [seq, vocab] logits tensor.
Uses Unsloth's UNSLOTH_RETURN_HIDDEN_STATES env var to get hidden states from the
full patched forward (preserving LoRA, gradient checkpointing, inplace attention),
then processes lm_head in chunks.

Peak VRAM: chunk_size × vocab_size × dtype_bytes (e.g., 256 × 151936 × 2 = 74MB)
vs full:   seq_len × vocab_size × dtype_bytes   (e.g., 4096 × 151936 × 2 = 1.17GB)

CRITICAL implementation notes (from deep analysis + tech scan 2026-03-23):
1. torch.checkpoint per chunk — without it, autograd stores ALL chunk logits (2.37GB)
2. torch.cat(chunks) — NOT in-place assignment to torch.zeros (breaks autograd graph)
3. .float() cast before selective_log_softmax — bf16 path is 10-50× slower
4. Use UNSLOTH_RETURN_HIDDEN_STATES, NOT body splitting — body bypass causes 0.25 divergence

Inspired by Liger Kernel's fused linear cross entropy approach (linkedin/Liger-Kernel).
Unsloth's own GRPO trainer uses the same UNSLOTH_RETURN_HIDDEN_STATES pattern.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from qgre.nemo_extracted.logits import selective_log_softmax


@contextmanager
def unsloth_hidden_states_mode():
    """Context manager to toggle Unsloth's hidden states return mode.

    When active, model(input_ids).logits returns hidden_states [batch, seq, hidden_dim]
    instead of logits [batch, seq, vocab_size]. This is Unsloth's internal mechanism
    for avoiding full logit materialization in their GRPO trainer.
    """
    prev = os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0")
    os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
    try:
        yield
    finally:
        os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = prev


def get_hidden_states_and_lm_head(model: nn.Module, input_ids: torch.Tensor, **kwargs):
    """Get hidden states and lm_head from model using Unsloth's env var mechanism.

    Uses UNSLOTH_RETURN_HIDDEN_STATES=1 to make the Unsloth-patched forward
    return hidden states instead of logits. This preserves ALL Unsloth optimizations
    (LoRA, gradient checkpointing, inplace attention) unlike body-only splitting.

    Args:
        model: The Unsloth-wrapped language model
        input_ids: [batch, seq] token IDs
        **kwargs: Additional kwargs passed to model forward (e.g., attention_mask)

    Returns:
        (hidden_states, lm_head) — hidden_states [batch, seq, hidden], lm_head nn.Linear
        Returns (None, None) if Unsloth env var mechanism doesn't work.
    """
    # Get lm_head module — works on any HF CausalLM
    lm_head = None
    try:
        lm_head = model.get_output_embeddings()
        if not isinstance(lm_head, nn.Linear):
            lm_head = None
    except (AttributeError, TypeError):
        pass

    if lm_head is None:
        return None, None

    # Get hidden states via Unsloth's env var mechanism
    with unsloth_hidden_states_mode():
        output = model(input_ids, **kwargs)

    hidden_states = output.logits if hasattr(output, "logits") else output

    # Shape assertion: hidden_dim should be << vocab_size
    # If Unsloth didn't honor the env var, we'd get [batch, seq, 151936] instead of [batch, seq, 2048]
    if hidden_states.shape[-1] > 10000:
        # Env var didn't take effect — got logits instead of hidden states
        return None, None

    return hidden_states, lm_head


def _chunk_forward(chunk_hidden: torch.Tensor, lm_head: nn.Linear, chunk_labels: torch.Tensor) -> torch.Tensor:
    """Forward one chunk through lm_head + selective_log_softmax.

    Separated into its own function for torch.checkpoint compatibility.
    torch.checkpoint requires a function (not inline code) to wrap.
    """
    chunk_logits = lm_head(chunk_hidden).float()  # Cast to fp32 — bf16 path is 10-50× slower
    result = selective_log_softmax(chunk_logits, chunk_labels)
    return result


def chunked_logprobs_from_hidden(
    hidden_states: torch.Tensor,
    lm_head: nn.Linear,
    labels: torch.Tensor,
    chunk_size: int = 256,
    use_checkpoint: bool = True,
) -> torch.Tensor:
    """Compute log probs from hidden states via chunked lm_head projection.

    Uses selective_log_softmax: never materializes a [chunk, vocab] log-prob tensor.
    For fp32: uses logsumexp identity (zero vocab-sized allocations).

    CRITICAL: Uses torch.checkpoint per chunk to prevent autograd from storing
    all chunk logits for backward. Without checkpoint, autograd stores 16 chunks
    × 148MB = 2.37GB — same as the full logit tensor. Checkpoint recomputes each
    chunk during backward instead (2× lm_head compute, but saves 2.2GB).

    CRITICAL: Uses torch.cat instead of in-place assignment to torch.zeros.
    torch.zeros creates a leaf tensor with no grad_fn. In-place slice assignment
    copies values but severs the autograd graph. torch.cat creates a proper graph
    node that connects all chunks back to hidden_states → model parameters.

    Args:
        hidden_states: [batch, seq, hidden] — output of model body (before lm_head)
        lm_head: nn.Linear(hidden, vocab) — the language model head
        labels: [batch, seq] — next-token labels to gather log probs for
        chunk_size: tokens per chunk (lower = less memory)
        use_checkpoint: if True, use torch.checkpoint per chunk (saves ~2GB VRAM)

    Returns:
        [batch, seq] log probs (float32) — gathered at label positions only
        WITH grad_fn connected to hidden_states → model parameters
    """
    batch, seq_len, hidden = hidden_states.shape
    chunks = []

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_hidden = hidden_states[:, start:end, :]
        chunk_labels = labels[:, start:end]

        if use_checkpoint:
            chunk_lp = torch_checkpoint(
                _chunk_forward, chunk_hidden, lm_head, chunk_labels,
                use_reentrant=False,
            )
        else:
            chunk_lp = _chunk_forward(chunk_hidden, lm_head, chunk_labels)

        chunks.append(chunk_lp)

    # torch.cat preserves the autograd graph — each chunk's grad_fn connects
    # back through lm_head → hidden_states → model parameters.
    # DO NOT replace with torch.zeros + in-place slice assignment.
    return torch.cat(chunks, dim=1)
