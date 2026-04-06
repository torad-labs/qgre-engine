# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Extracted from NeMo RL v0.5.0 for QGRE Engine (Step 0b).
# Log prob computation utilities — single-GPU, no distributed.

import torch


def selective_log_softmax(
    logits: torch.Tensor,
    index: torch.Tensor,
) -> torch.Tensor:
    """Compute log softmax probabilities for selected tokens only.

    Uses the identity log_softmax(x_i) = x_i - logsumexp(x) to avoid
    materializing a full [batch, seq, vocab] log-probability tensor.
    Peak memory: [batch, seq] instead of [batch, seq, vocab].

    For bf16/fp16: falls back to per-row log_softmax (logsumexp is numerically
    unstable in half precision). Still avoids the full vocab tensor by looping.

    Ported from TRL PR #2799 (Tyler Romero, Feb 2025).
    Source: https://www.tylerromero.com/posts/2025-02-selective-log-softmax/

    Args:
        logits: [..., vocab] logit tensor (any dtype)
        index: [...] token IDs to gather log probs for

    Returns:
        [...] gathered log probabilities (same shape as index)
    """
    # Validate dtype before gather
    if index.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"selective_log_softmax: labels must be int32 or int64, got {index.dtype}. "
            "torch.gather requires integer index tensor.",
        )
    # GB3-002: Add shape assertion before gather
    if logits.shape[0] != index.shape[0]:
        raise ValueError(
            f"GB3-002: selective_log_softmax batch size mismatch. "
            f"logits.shape[0]={logits.shape[0]} != index.shape[0]={index.shape[0]}",
        )
    if len(logits.shape) < 2 or len(index.shape) < 1:
        raise ValueError(
            f"GB3-002: selective_log_softmax shape error. "
            f"logits.shape={logits.shape} (expected [..., vocab]), "
            f"index.shape={index.shape} (expected [...]).",
        )

    if logits.dtype in (torch.float32, torch.float64):
        # logsumexp identity: log_softmax(x_i) = x_i - logsumexp(x)
        # Loop over batch to avoid materializing full [batch, seq, vocab]
        # LP-R3-01: Add bounds validation for fp32 path
        vocab_size = logits.shape[-1]
        if (index < 0).any() or (index >= vocab_size).any():
            raise ValueError(
                f"LP-R3-01: Index out of bounds in fp32 path. "
                f"index range [{index.min().item()}, {index.max().item()}] vs vocab_size={vocab_size}",
            )
        lse = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        selected = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        return (selected - lse).to(torch.float32)
    # GEN-R1-5: bf16/fp16: convert to FP32 BEFORE log_softmax for stability
    # logsumexp in BF16 loses precision — compute log_softmax in FP32.
    # GB2-004: Use functional approach to preserve gradients (no in-place)
    logprobs_list = []
    for logits_row, index_row in zip(logits, index, strict=False):
        # LP-R2-04: Validate bounds before bf16 path
        vocab_size = logits_row.shape[-1]
        if (index_row < 0).any() or (index_row >= vocab_size).any():
            raise ValueError(
                f"LP-R2-04: Index out of bounds in bf16 path. "
                f"index range [{index_row.min().item()}, {index_row.max().item()}] vs vocab_size={vocab_size}",
            )
        logprobs_row = logits_row.float().log_softmax(dim=-1)
        selected = torch.gather(
            logprobs_row,
            dim=-1,
            index=index_row.unsqueeze(-1),
        ).squeeze(-1)
        logprobs_list.append(selected)
    return torch.stack(logprobs_list).to(torch.float32)


def logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute log probabilities of labels given logits — memory efficient.

    Uses selective_log_softmax per chunk: never materializes [batch, chunk, vocab]
    as a log-probability tensor. For fp32, uses logsumexp identity (no log_softmax
    allocation at all). For bf16, loops per row within each chunk.

    Peak memory: batch × chunk_size (not batch × chunk_size × vocab).
    For Qwen3 vocab (151936): 37,000× less memory per chunk vs naive approach.

    Args:
        logits: [batch, seq, vocab] raw model output (any dtype)
        labels: [batch, seq] token IDs to gather log probs for
        chunk_size: tokens per chunk (lower = less memory, slightly slower)

    Returns:
        [batch, seq] log probabilities for each token (float32)
    """
    batch, seq_len, _vocab = logits.shape
    # Use torch.cat to preserve autograd graph — NOT torch.zeros + in-place assignment.
    # torch.zeros creates a leaf tensor; in-place slice assignment severs the graph.
    # Same pattern as chunked_logprobs_from_hidden in fused_logprobs.py.
    chunks = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunks.append(
            selective_log_softmax(
                logits[:, start:end, :],
                labels[:, start:end],
            )
        )

    if not chunks:
        return torch.zeros(batch, 0, dtype=torch.float32, device=logits.device)
    return torch.cat(chunks, dim=1)


def compute_response_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute log probs for response tokens only (shifted by 1 for next-token prediction).

    Args:
        logits: [batch, seq, vocab] model output
        input_ids: [batch, seq] full sequence (prompt + response)
        response_mask: [batch, seq] mask where 1 = response token

    Returns:
        [batch, seq-1] log probs for next-token prediction, masked to response only
    """
    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = response_mask[:, 1:]

    log_probs = logprobs_from_logits(shift_logits, shift_labels)
    # LP-R2-08: Avoid -inf × 0 = NaN by using torch.where
    return torch.where(shift_mask.bool(), log_probs, 0.0)
