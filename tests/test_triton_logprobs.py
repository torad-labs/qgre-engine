"""Tests for Triton fused logprobs kernel (Phase 4)."""

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

    assert torch.allclose(
        result, expected, atol=1e-3
    ), f"max diff: {(result - expected).abs().max()}"
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

    assert torch.allclose(
        result, expected, atol=1e-3
    ), f"max diff: {(result - expected).abs().max()}"
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

    assert torch.allclose(
        result, expected, atol=1e-3
    ), f"max diff: {(result - expected).abs().max()}"
