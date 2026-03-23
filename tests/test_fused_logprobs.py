"""Tests for fused logprobs — chunked lm_head projection with torch.checkpoint.

Tests the core chunking + autograd mechanics. The Unsloth env var integration
(UNSLOTH_RETURN_HIDDEN_STATES) is tested via live model tests, not mocks.
"""

import torch
import torch.nn as nn
import pytest

from qgre.fused_logprobs import chunked_logprobs_from_hidden


class TestChunkedLogprobs:
    """Test chunked logprobs with torch.checkpoint and torch.cat."""

    def test_grad_fn_preserved(self):
        """Result must have grad_fn (CatBackward0) — autograd graph intact."""
        lm_head = nn.Linear(32, 100)
        hidden = torch.randn(1, 20, 32, requires_grad=True)
        labels = torch.randint(0, 100, (1, 20))
        result = chunked_logprobs_from_hidden(hidden, lm_head, labels, chunk_size=8)
        assert result.grad_fn is not None, "Must have grad_fn — autograd graph must be preserved"

    def test_backward_produces_gradients(self):
        """Backward through chunked path must produce non-zero gradients."""
        lm_head = nn.Linear(32, 100)
        hidden = torch.randn(1, 20, 32, requires_grad=True)
        labels = torch.randint(0, 100, (1, 20))
        result = chunked_logprobs_from_hidden(hidden, lm_head, labels, chunk_size=8)
        loss = result.sum()
        loss.backward()
        assert hidden.grad is not None, "hidden_states must have grad"
        assert lm_head.weight.grad is not None, "lm_head must have grad"
        assert hidden.grad.abs().sum() > 0, "Gradients must be non-zero"

    def test_checkpoint_vs_no_checkpoint_equivalent(self):
        """Checkpoint and non-checkpoint paths should produce same logprobs."""
        torch.manual_seed(42)
        lm_head = nn.Linear(32, 100)
        hidden = torch.randn(1, 20, 32, requires_grad=True)
        labels = torch.randint(0, 100, (1, 20))

        with_ckpt = chunked_logprobs_from_hidden(hidden, lm_head, labels, chunk_size=8, use_checkpoint=True)
        without_ckpt = chunked_logprobs_from_hidden(hidden, lm_head, labels, chunk_size=8, use_checkpoint=False)
        assert torch.allclose(with_ckpt, without_ckpt, atol=1e-5), "Checkpoint must not change values"

    def test_single_chunk_equivalent_to_full(self):
        """When chunk_size >= seq_len, result should match non-chunked."""
        lm_head = nn.Linear(32, 100)
        hidden = torch.randn(1, 10, 32, requires_grad=True)
        labels = torch.randint(0, 100, (1, 10))

        chunked = chunked_logprobs_from_hidden(hidden, lm_head, labels, chunk_size=1000)
        # Compare against direct computation
        with torch.no_grad():
            logits = lm_head(hidden).float()
            from qgre.nemo_extracted.logits import selective_log_softmax
            direct = selective_log_softmax(logits, labels)
        assert torch.allclose(chunked.detach(), direct, atol=1e-5)

    def test_output_shape(self):
        """Output shape should be [batch, seq]."""
        lm_head = nn.Linear(32, 100)
        hidden = torch.randn(2, 15, 32, requires_grad=True)
        labels = torch.randint(0, 100, (2, 15))
        result = chunked_logprobs_from_hidden(hidden, lm_head, labels, chunk_size=4)
        assert result.shape == (2, 15)

    def test_lm_head_with_bias(self):
        """Should work with lm_head that has bias."""
        lm_head = nn.Linear(32, 100, bias=True)
        hidden = torch.randn(1, 10, 32, requires_grad=True)
        labels = torch.randint(0, 100, (1, 10))
        result = chunked_logprobs_from_hidden(hidden, lm_head, labels, chunk_size=4)
        assert result.grad_fn is not None
        result.sum().backward()
        assert lm_head.bias.grad is not None, "Bias grad must flow"
