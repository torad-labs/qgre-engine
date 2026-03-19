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


def logprobs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Compute log probabilities of labels given logits — memory efficient.

    Chunks along the sequence dimension to avoid materializing a full
    [batch, seq, vocab] tensor. Each chunk: log_softmax → gather → discard.
    Peak memory: batch × chunk_size × vocab (not batch × full_seq × vocab).

    For 1 × 256 × 151936 in bf16 = 74MB per chunk (vs 1.2GB for full 4096 seq).
    Inspired by Liger Kernel's chunked cross entropy approach (linkedin/Liger-Kernel).

    Args:
        logits: [batch, seq, vocab] raw model output (any dtype)
        labels: [batch, seq] token IDs to gather log probs for
        chunk_size: tokens per chunk (lower = less memory, slightly slower)

    Returns:
        [batch, seq] log probabilities for each token (float32)
    """
    batch, seq_len, vocab = logits.shape
    result = torch.zeros(batch, seq_len, dtype=torch.float32, device=logits.device)

    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_logits = logits[:, start:end, :]
        chunk_labels = labels[:, start:end]

        chunk_lp = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
        result[:, start:end] = chunk_lp.gather(
            dim=-1, index=chunk_labels.unsqueeze(-1)
        ).squeeze(-1).to(torch.float32)

    return result


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
    return log_probs * shift_mask
