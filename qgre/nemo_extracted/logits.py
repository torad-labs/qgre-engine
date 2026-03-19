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


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute log probabilities of labels given logits.

    Memory-efficient: log_softmax in native dtype (bf16), gather first (tiny),
    then cast gathered values to float32. Avoids allocating a full float32 copy
    of the vocab-sized logits tensor (2.3GB per seq × 4096 × 151K vocab).

    Args:
        logits: [batch, seq, vocab] raw model output
        labels: [batch, seq] token IDs to gather log probs for

    Returns:
        [batch, seq] log probabilities for each token (float32)
    """
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    gathered = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return gathered.to(torch.float32)


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
