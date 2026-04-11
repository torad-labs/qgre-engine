"""Tests for Triton fused logprobs kernel (Phase 4) and differentiable autograd wrapper."""

import pytest
import torch

from qgre.triton_logprobs import HAS_TRITON


pytestmark = pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="Triton + CUDA required",
)


@pytest.mark.gpu
def test_triton_logprobs_small():
    """Triton kernel matches PyTorch reference on small vocab."""
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 16, 2
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_from_hidden(hidden, lm_head, labels)
    expected = (
        torch.nn.functional.log_softmax(lm_head(hidden), dim=-1)
        .gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    assert torch.allclose(result, expected, atol=1e-3), (
        f"max diff: {(result - expected).abs().max()}"
    )
    assert (result <= 0).all()


@pytest.mark.gpu
def test_triton_logprobs_non_power_of_2_vocab():
    """Triton kernel correct with vocab not power of 2 (like Qwen3 151936 = 128*1187)."""
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 128 * 5, 8, 1  # 640 = non-power-of-2
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_from_hidden(hidden, lm_head, labels)
    expected = (
        torch.nn.functional.log_softmax(lm_head(hidden), dim=-1)
        .gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    assert torch.allclose(result, expected, atol=1e-3), (
        f"max diff: {(result - expected).abs().max()}"
    )
    assert result.isfinite().all()


@pytest.mark.gpu
def test_triton_logprobs_with_bias():
    """Triton kernel works when lm_head has bias."""
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=True, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_from_hidden(hidden, lm_head, labels)
    expected = (
        torch.nn.functional.log_softmax(lm_head(hidden), dim=-1)
        .gather(
            dim=-1,
            index=labels.unsqueeze(-1),
        )
        .squeeze(-1)
    )

    assert torch.allclose(result, expected, atol=1e-3), (
        f"max diff: {(result - expected).abs().max()}"
    )


# ─── Differentiable Triton logprobs (autograd.Function) ──────────────────────


@pytest.mark.gpu
def test_triton_with_grad_matches_pytorch():
    """Triton autograd wrapper matches PyTorch reference numerically."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 16, 2
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    expected = (
        torch.nn.functional.log_softmax(lm_head(hidden.detach().requires_grad_(True)), dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1)
    )

    assert torch.allclose(result.detach(), expected.detach(), atol=1e-3), (
        f"max diff: {(result.detach() - expected.detach()).abs().max()}"
    )


@pytest.mark.gpu
def test_triton_with_grad_has_grad_fn():
    """Triton autograd wrapper produces a tensor with grad_fn (autograd graph intact)."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    assert result.grad_fn is not None, "Triton logprobs must have grad_fn for training"


@pytest.mark.gpu
def test_triton_with_grad_backward_hidden():
    """Backward through Triton autograd produces non-zero gradients for hidden states."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 2
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    loss = result.sum()
    loss.backward()

    assert hidden.grad is not None, "hidden.grad must not be None after backward"
    assert hidden.grad.abs().sum() > 0, "hidden.grad must be non-zero"
    assert hidden.grad.isfinite().all(), "hidden.grad must be finite"


@pytest.mark.gpu
def test_triton_with_grad_backward_weight():
    """Backward through Triton autograd produces non-zero gradients for lm_head weight."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    loss = result.sum()
    loss.backward()

    assert lm_head.weight.grad is not None, "lm_head.weight.grad must not be None"
    assert lm_head.weight.grad.abs().sum() > 0, "lm_head.weight.grad must be non-zero"
    assert lm_head.weight.grad.isfinite().all(), "lm_head.weight.grad must be finite"


@pytest.mark.gpu
def test_triton_with_grad_gradient_numerical():
    """Triton autograd gradients match PyTorch reference gradients numerically."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 32, 128, 4, 1

    # Triton path
    hidden_t = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result_t = triton_logprobs_with_grad(hidden_t, lm_head, labels)
    result_t.sum().backward()

    # PyTorch reference path (same weights, same inputs)
    hidden_p = hidden_t.detach().clone().requires_grad_(True)
    logits_p = hidden_p @ lm_head.weight.T
    log_probs_p = torch.nn.functional.log_softmax(logits_p, dim=-1)
    result_p = log_probs_p.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    result_p.sum().backward()

    assert torch.allclose(hidden_t.grad, hidden_p.grad, atol=1e-3), (
        f"hidden grad max diff: {(hidden_t.grad - hidden_p.grad).abs().max()}"
    )


@pytest.mark.gpu
def test_triton_with_grad_with_bias():
    """Triton autograd wrapper works with lm_head bias and produces correct gradients."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=True, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)

    expected = (
        torch.nn.functional.log_softmax(lm_head(hidden.detach()), dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.allclose(result.detach(), expected, atol=1e-3)

    result.sum().backward()
    assert hidden.grad is not None and hidden.grad.isfinite().all()
    assert lm_head.weight.grad is not None and lm_head.weight.grad.isfinite().all()
    assert lm_head.bias.grad is not None and lm_head.bias.grad.isfinite().all()


@pytest.mark.gpu
def test_triton_with_grad_empty_sequence():
    """Triton autograd wrapper handles empty sequence gracefully."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    hidden_dim, vocab_size = 64, 256
    hidden = torch.randn(1, 0, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (1, 0), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    assert result.shape == (1, 0)
