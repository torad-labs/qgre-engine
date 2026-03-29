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

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from qgre.nemo_extracted.logits import selective_log_softmax


def get_hidden_states_and_lm_head(model: nn.Module, input_ids: torch.Tensor, **kwargs):
    """Get hidden states and lm_head from model.

    UNSLOTH_RETURN_HIDDEN_STATES=1 is set globally at startup (__main__.py).
    All forward calls return hidden states in output.logits field.

    Args:
        model: The Unsloth-wrapped language model
        input_ids: [batch, seq] token IDs
        **kwargs: Additional kwargs passed to model forward (e.g., attention_mask)

    Returns:
        (hidden_states, lm_head) — hidden_states [batch, seq, hidden], lm_head nn.Linear
        Returns (None, None) if hidden states mode didn't take effect.
    """
    # CRITICAL: Do NOT pass labels. Unsloth issue #3000 (open): Qwen3Moe
    # bypasses NOT_RETURN_LOGITS when labels are present. See rejection framework.
    assert "labels" not in kwargs, (
        "Do not pass labels to forward when using hidden states mode. "
        "See Unsloth issue #3000: labels bypass UNSLOTH_RETURN_HIDDEN_STATES."
    )

    # Get lm_head module — uses WeightExporter for PEFT ModulesToSaveWrapper unwrapping
    from qgre.weight_export import WeightExporter
    lm_head = WeightExporter().get_lm_head(model)
    if lm_head is None:
        # Fallback: try direct access without unwrapping
        try:
            lm_head = model.get_output_embeddings()
            if not isinstance(lm_head, nn.Linear):
                lm_head = None
        except AttributeError:
            lm_head = None  # Model doesn't implement get_output_embeddings

    if lm_head is None:
        return None, None

    # No env var toggling needed — set globally at startup
    output = model(input_ids, **kwargs)
    hidden_states = output.logits if hasattr(output, "logits") else output

    # Model-agnostic shape check: use lm_head dimensions as ground truth.
    # lm_head is nn.Linear(hidden_dim, vocab_size). If output matches
    # lm_head.out_features → got logits. If matches in_features → got hidden states.
    last_dim = hidden_states.shape[-1]
    if last_dim == lm_head.out_features:
        import warnings
        warnings.warn(
            f"get_hidden_states_and_lm_head: output dim {last_dim} matches vocab_size "
            f"(lm_head.out_features={lm_head.out_features}). Model returned logits, "
            f"not hidden states. UNSLOTH_RETURN_HIDDEN_STATES may not have taken effect."
        )
        return None, None
    if last_dim != lm_head.in_features:
        import warnings
        warnings.warn(
            f"get_hidden_states_and_lm_head: output dim {last_dim} matches neither "
            f"hidden_dim ({lm_head.in_features}) nor vocab_size ({lm_head.out_features}). "
            f"Model output is corrupted or architecture is unsupported."
        )
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
    if seq_len == 0:
        return torch.zeros(batch, 0, dtype=torch.float32, device=hidden_states.device)
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
