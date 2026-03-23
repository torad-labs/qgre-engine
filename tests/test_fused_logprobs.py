"""Tests for fused logprobs wrapper chain resolution.

Mock the Unsloth PeftModel → LoraModel → CausalLM → Body chain
to verify get_hidden_states_and_lm_head resolves correctly without GPU.
"""

import torch
import torch.nn as nn
import pytest

from qgre.fused_logprobs import get_hidden_states_and_lm_head, chunked_logprobs_from_hidden


class MockBody(nn.Module):
    """The actual transformer body (e.g., Qwen3Model). Has .model but NO .lm_head."""
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.layers = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, **kwargs):
        batch, seq = input_ids.shape
        return (torch.randn(batch, seq, 32, requires_grad=True),)


class MockCausalLM(nn.Module):
    """CausalLM wrapper (e.g., Qwen3ForCausalLM). Has BOTH .model (body) and .lm_head."""
    def __init__(self):
        super().__init__()
        self.model = MockBody()
        self.lm_head = nn.Linear(32, 100)


class MockLoraModel(nn.Module):
    """LoRA wrapper. Has .model (CausalLM) and delegates .lm_head from CausalLM."""
    def __init__(self):
        super().__init__()
        self.model = MockCausalLM()
        self.lm_head = self.model.lm_head  # Delegation — same object


class MockPeftModel(nn.Module):
    """Top-level PeftModel. Has .model and .base_model."""
    def __init__(self):
        super().__init__()
        self.model = MockLoraModel()
        self.base_model = self.model
        self.lm_head = self.model.lm_head  # Delegation chain


class TestWrapperChainResolution:
    """Verify get_hidden_states_and_lm_head resolves through Unsloth-style wrapping."""

    def test_peft_model_resolves_to_body(self):
        """Full chain: PeftModel → LoraModel → CausalLM → Body."""
        model = MockPeftModel()
        ids = torch.randint(0, 100, (1, 10))
        h, lm = get_hidden_states_and_lm_head(model, ids)
        assert h is not None, "hidden_states should not be None"
        assert lm is not None, "lm_head should not be None"
        # lm_head should be the Linear from CausalLM, not a wrapper
        assert isinstance(lm, nn.Linear)
        assert lm.weight.shape == (100, 32)
        # hidden_states should come from MockBody
        assert h.shape == (1, 10, 32)

    def test_lm_head_is_causal_lm_head(self):
        """lm_head should be the CausalLM's lm_head, not a wrapper's delegation."""
        model = MockPeftModel()
        ids = torch.randint(0, 100, (1, 10))
        _, lm = get_hidden_states_and_lm_head(model, ids)
        # The lm_head should be the same object as MockCausalLM.lm_head
        assert lm is model.model.model.lm_head

    def test_body_has_no_lm_head(self):
        """The resolved body (MockBody) must NOT have lm_head."""
        model = MockPeftModel()
        inner_body = model.model.model.model  # MockBody
        assert not hasattr(inner_body, "lm_head")

    def test_bare_causal_lm(self):
        """Direct CausalLM (no PEFT wrapping) should also resolve."""
        model = MockCausalLM()
        ids = torch.randint(0, 100, (1, 10))
        h, lm = get_hidden_states_and_lm_head(model, ids)
        assert h is not None
        assert lm is not None
        assert lm is model.lm_head

    def test_model_without_lm_head_returns_none(self):
        """Model without lm_head should return (None, None)."""
        model = nn.Linear(32, 100)
        ids = torch.randint(0, 100, (1, 10))
        h, lm = get_hidden_states_and_lm_head(model, ids)
        assert h is None
        assert lm is None

    def test_double_wrapped_causal_lm(self):
        """CausalLM wrapped twice should still resolve to the inner body."""
        class DoubleWrapped(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = MockCausalLM()  # First CausalLM
                self.model.model = MockCausalLM()  # body is ANOTHER CausalLM
                self.model.model.model = MockBody()  # actual body
                self.lm_head = self.model.lm_head

        model = DoubleWrapped()
        ids = torch.randint(0, 100, (1, 10))
        h, lm = get_hidden_states_and_lm_head(model, ids)
        assert h is not None, "Should resolve through double-wrapped CausalLM"
        assert h.shape == (1, 10, 32)


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
