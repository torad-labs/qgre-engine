"""Tests for fused logprobs — chunked lm_head projection with torch.checkpoint.

Tests the core chunking + autograd mechanics. The Unsloth env var integration
(UNSLOTH_RETURN_HIDDEN_STATES) is tested via live model tests, not mocks.
"""

import torch
import torch.nn as nn
import pytest

from qgre.fused_logprobs import chunked_logprobs_from_hidden, get_hidden_states_and_lm_head


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


class TestGetHiddenStatesAndLmHead:
    """Test hidden states extraction with model-agnostic shape check and labels guard."""

    def _make_stub_model(self, hidden_dim, vocab_size, return_dim=None):
        """Create a stub model that returns tensors of specified shape."""
        out_dim = return_dim if return_dim is not None else hidden_dim

        class StubModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

                class _Config:
                    pass
                self.config = _Config()
                self.config.vocab_size = vocab_size

            def get_output_embeddings(self):
                return self._lm_head

            def forward(self, input_ids, **kwargs):
                batch, seq = input_ids.shape
                class Out:
                    pass
                result = Out()
                result.logits = torch.randn(batch, seq, out_dim)
                return result

        return StubModel()

    def test_labels_assert_rejects_labels_in_kwargs(self):
        """Passing labels in kwargs must raise AssertionError."""
        model = self._make_stub_model(32, 100)
        input_ids = torch.randint(0, 100, (1, 10))
        with pytest.raises(AssertionError, match="labels"):
            get_hidden_states_and_lm_head(model, input_ids, labels=torch.zeros(1, 10))

    def test_shape_check_returns_none_when_output_matches_vocab(self):
        """When output last dim == vocab_size, function raises RuntimeError (GB3-005)."""
        # Model returns tensor with shape[-1] == vocab_size (got logits, not hidden states)
        model = self._make_stub_model(32, 100, return_dim=100)
        input_ids = torch.randint(0, 100, (1, 10))
        # GB3-005: Should raise RuntimeError instead of returning None
        import pytest
        with pytest.raises(RuntimeError, match="GB3-005.*logits.*not hidden states"):
            get_hidden_states_and_lm_head(model, input_ids)

    def test_shape_check_succeeds_when_output_is_hidden_dim(self):
        """When output last dim < vocab_size, function returns hidden states."""
        model = self._make_stub_model(32, 100, return_dim=32)
        input_ids = torch.randint(0, 100, (1, 10))
        hs, lm = get_hidden_states_and_lm_head(model, input_ids)
        assert hs is not None, "Should succeed when hidden_dim < vocab_size"
        assert hs.shape[-1] == 32
        assert isinstance(lm, nn.Linear)

    def test_model_without_config_uses_lm_head_dims(self):
        """When model has no config attribute, shape check uses lm_head.in_features."""
        class NoConfigModel(nn.Module):
            def __init__(self):
                super().__init__()
                self._lm_head = nn.Linear(32, 100, bias=False)

            def get_output_embeddings(self):
                return self._lm_head

            def forward(self, input_ids, **kwargs):
                batch, seq = input_ids.shape
                class Out:
                    pass
                result = Out()
                result.logits = torch.randn(batch, seq, 32)
                return result

        model = NoConfigModel()
        input_ids = torch.randint(0, 100, (1, 10))
        hs, lm = get_hidden_states_and_lm_head(model, input_ids)
        assert hs is not None, "Should succeed — output dim 32 matches lm_head.in_features"

    def test_no_lm_head_returns_none(self):
        """When model has no get_output_embeddings or it returns non-Linear."""
        class NoEmbModel(nn.Module):
            def get_output_embeddings(self):
                return None  # Not nn.Linear
            def forward(self, input_ids, **kwargs):
                return torch.randn(1, 10, 32)

        model = NoEmbModel()
        input_ids = torch.randint(0, 100, (1, 10))
        hs, lm = get_hidden_states_and_lm_head(model, input_ids)
        assert hs is None
        assert lm is None
