"""Fused chunked logprob computation — Phase 4 of PLAN.md.

Computes log probabilities WITHOUT materializing the full [seq, vocab] logits tensor.
Instead, processes lm_head in chunks of `chunk_size` tokens at a time:
  hidden_states[:, chunk] @ lm_head.weight.T → [chunk, vocab] → log_softmax → gather → discard

Peak VRAM: chunk_size × vocab_size × dtype_bytes (e.g., 256 × 151936 × 2 = 74MB)
vs full:   seq_len × vocab_size × dtype_bytes   (e.g., 4096 × 151936 × 2 = 1.17GB)

CRITICAL implementation notes (from deep analysis 2026-03-23):
1. torch.checkpoint per chunk — without it, autograd stores ALL chunk logits (2.37GB)
2. torch.cat(chunks) — NOT in-place assignment to torch.zeros (breaks autograd graph)
3. .float() cast before selective_log_softmax — bf16 path is 10-50× slower

Inspired by Liger Kernel's fused linear cross entropy approach (linkedin/Liger-Kernel).
Reference: "Cutting LLM Memory by 84%" (Medium, Feb 2026).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from qgre.nemo_extracted.logits import selective_log_softmax


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
            # torch.checkpoint: don't save chunk_logits for backward — recompute them.
            # This is the key memory optimization. Without it, autograd stores ALL
            # chunk logits (16 × 148MB = 2.37GB) in saved_tensors for backward.
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


def get_hidden_states_and_lm_head(
    model: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
):
    """Extract hidden states (before lm_head) and the lm_head linear layer.

    Works with Unsloth PeftModel, HuggingFace CausalLM, or any model with
    a `.model` body and `.lm_head` attribute.

    Args:
        model: The language model
        input_ids: [batch, seq] token IDs
        attention_mask: [batch, seq] attention mask (CRITICAL: must be passed
            to body forward to avoid attending to padding tokens)

    Returns:
        (hidden_states, lm_head) — hidden_states [batch, seq, hidden], lm_head nn.Linear
        Returns (None, None) if body/lm_head split cannot be resolved.
    """
    # Navigate through Unsloth/PEFT wrappers to find the model body and lm_head
    inner = model
    while hasattr(inner, "model") and not hasattr(inner, "lm_head"):
        inner = inner.model
    if hasattr(inner, "base_model"):
        inner = inner.base_model
    while hasattr(inner, "model") and not hasattr(inner, "lm_head"):
        inner = inner.model

    if not hasattr(inner, "lm_head"):
        return None, None

    # Descend past CausalLM wrappers: if inner.model itself has both .model and
    # .lm_head, it's a CausalLM (e.g., Qwen3ForCausalLM), not the transformer body.
    # Keep going deeper until we find the module that OWNS lm_head — where .model
    # is the actual body (no .lm_head of its own).
    causal_lm = inner.model if hasattr(inner, "model") else None
    while causal_lm is not None and hasattr(causal_lm, "lm_head") and hasattr(causal_lm, "model"):
        inner = causal_lm
        causal_lm = inner.model if hasattr(inner, "model") else None

    lm_head = inner.lm_head

    # Get the model body (everything except lm_head)
    body = inner.model if hasattr(inner, "model") else None
    if body is None:
        return None, None

    # Safety: body must NOT have its own lm_head — if it does, the descent
    # didn't go deep enough and body is still a CausalLM, not the transformer body
    if hasattr(body, "lm_head"):
        return None, None

    # Forward through body only — pass attention_mask to avoid padding corruption
    kwargs = {}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask

    try:
        body_output = body(input_ids, **kwargs)
    except TypeError as e:
        if attention_mask is not None and "unexpected keyword argument" in str(e) and "attention_mask" in str(e):
            import warnings
            warnings.warn(
                f"Model body does not accept attention_mask — falling back without it. "
                f"Padded tokens may corrupt hidden states. Error: {e}"
            )
            body_output = body(input_ids)
        else:
            raise  # Re-raise unexpected TypeErrors — don't swallow real bugs

    # Handle various output formats
    if hasattr(body_output, "last_hidden_state"):
        hidden_states = body_output.last_hidden_state
    elif isinstance(body_output, tuple) and len(body_output) > 0:
        hidden_states = body_output[0]
    elif isinstance(body_output, torch.Tensor):
        hidden_states = body_output
    else:
        return None, None

    return hidden_states, lm_head
