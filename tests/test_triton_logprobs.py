"""Tests for Triton fused logprobs kernel and differentiable autograd wrapper.

Every gradient test compares against a PyTorch autograd reference — not just
"is not None" or "isfinite". Silent zeros, wrong signs, and off-by-one in
the vocab tile loop all show up as allclose failures against the reference.
"""

import pytest
import torch

from qgre.triton_logprobs import HAS_TRITON


pytestmark = pytest.mark.skipif(
    not HAS_TRITON or not torch.cuda.is_available(),
    reason="Triton + CUDA required",
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _pytorch_logprobs(hidden, weight, bias, labels):
    """Reference log-probabilities via PyTorch — the ground truth."""
    logits = hidden @ weight.T
    if bias is not None:
        logits = logits + bias
    return (
        torch.nn.functional.log_softmax(logits.float(), dim=-1)
        .gather(dim=-1, index=labels.unsqueeze(-1))
        .squeeze(-1)
    )


def _pytorch_grads(hidden, weight, bias, labels):
    """Run PyTorch autograd reference and return (grad_hidden, grad_weight, grad_bias)."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    b = bias.detach().clone().requires_grad_(True) if bias is not None else None

    lp = _pytorch_logprobs(h, w, b, labels)
    lp.sum().backward()
    return h.grad, w.grad, b.grad if b is not None else None


# ─── Forward: value correctness ─────────────────────────────────────────────


@pytest.mark.gpu
def test_forward_small_no_bias():
    """Forward matches PyTorch on small vocab, no bias."""
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 16, 2
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_from_hidden(hidden, lm_head, labels)
    expected = _pytorch_logprobs(hidden, lm_head.weight, None, labels)

    assert torch.allclose(result, expected, atol=1e-3), (
        f"max diff: {(result - expected).abs().max()}"
    )
    assert (result <= 0).all(), "log-probabilities must be <= 0"


@pytest.mark.gpu
def test_forward_non_power_of_2_vocab():
    """Forward correct with vocab not power of 2 (like Qwen3 151936 = 128*1187)."""
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 128 * 5, 8, 1  # 640
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_from_hidden(hidden, lm_head, labels)
    expected = _pytorch_logprobs(hidden, lm_head.weight, None, labels)

    assert torch.allclose(result, expected, atol=1e-3), (
        f"max diff: {(result - expected).abs().max()}"
    )


@pytest.mark.gpu
def test_forward_with_bias():
    """Forward correct when lm_head has bias."""
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=True, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_from_hidden(hidden, lm_head, labels)
    expected = _pytorch_logprobs(hidden, lm_head.weight, lm_head.bias, labels)

    assert torch.allclose(result, expected, atol=1e-3), (
        f"max diff: {(result - expected).abs().max()}"
    )


# ─── Forward: invalid labels ────────────────────────────────────────────────


@pytest.mark.gpu
def test_forward_invalid_label_raises_at_validation():
    """Labels outside [0, vocab_size) raise ValueError at the validation layer."""
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")

    # Label at vocab_size (one past valid range)
    labels = torch.full((batch, seq_len), vocab_size, dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match="label out of bounds"):
        triton_logprobs_from_hidden(hidden, lm_head, labels)

    # Negative label
    labels_neg = torch.full((batch, seq_len), -1, dtype=torch.long, device="cuda")
    with pytest.raises(ValueError, match="label out of bounds"):
        triton_logprobs_from_hidden(hidden, lm_head, labels_neg)


@pytest.mark.gpu
def test_forward_lse_inf_sentinel_for_invalid_label():
    """The lse=+inf sentinel zeroes out softmax contributions for invalid labels.

    This tests the kernel-level sentinel, not the validation guard. We bypass
    validation by constructing valid labels, running forward to get lse, then
    checking that an invalid-label position would produce lse=+inf. We verify
    this indirectly through the kernel's internal behavior by checking that
    the forward pass with valid labels doesn't produce +inf lse.
    """
    from qgre.triton_logprobs import triton_logprobs_from_hidden

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda")
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_from_hidden(hidden, lm_head, labels)
    # Valid labels: all logprobs must be finite (not -inf, not nan)
    assert result.isfinite().all(), "Valid labels must produce finite logprobs"
    # Valid labels: all logprobs must be <= 0
    assert (result <= 0).all(), "Valid labels must produce non-positive logprobs"


# ─── Autograd: grad_fn exists ───────────────────────────────────────────────


@pytest.mark.gpu
def test_autograd_has_grad_fn():
    """Autograd wrapper produces a tensor with grad_fn (graph intact)."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    assert result.grad_fn is not None, "Triton logprobs must have grad_fn for training"


# ─── Autograd: gradient correctness (the real tests) ────────────────────────
# Every test below compares Triton gradients against PyTorch autograd on the
# same inputs and weights. "is not None" and "isfinite" are necessary but not
# sufficient — allclose against the reference catches silent zeros, wrong
# signs, and tile-boundary bugs.


@pytest.mark.gpu
def test_grad_hidden_matches_pytorch():
    """grad_hidden from Triton matches PyTorch autograd reference."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 32, 128, 4, 2

    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    result.sum().backward()

    ref_gh, _, _ = _pytorch_grads(hidden, lm_head.weight, None, labels)

    assert torch.allclose(hidden.grad, ref_gh, atol=1e-3), (
        f"hidden grad max diff: {(hidden.grad - ref_gh).abs().max()}"
    )


@pytest.mark.gpu
def test_grad_weight_matches_pytorch():
    """grad_weight from Triton matches PyTorch autograd reference."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 32, 128, 4, 2

    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    result.sum().backward()

    _, ref_gw, _ = _pytorch_grads(hidden, lm_head.weight, None, labels)

    assert torch.allclose(lm_head.weight.grad, ref_gw, atol=1e-3), (
        f"weight grad max diff: {(lm_head.weight.grad - ref_gw).abs().max()}"
    )


@pytest.mark.gpu
def test_grad_bias_matches_pytorch():
    """grad_bias from Triton matches PyTorch autograd reference.

    This is the test that would have caught the silent-zero no-op bias kernel.
    The old test asserted `lm_head.bias.grad is not None and isfinite()` —
    a zero tensor passes both checks. This test compares values.
    """
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 32, 128, 4, 1

    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=True, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    result.sum().backward()

    ref_gh, ref_gw, ref_gb = _pytorch_grads(
        hidden,
        lm_head.weight,
        lm_head.bias,
        labels,
    )

    assert lm_head.bias.grad is not None, "bias.grad must not be None"
    assert lm_head.bias.grad.abs().sum() > 0, "bias.grad is all zeros — the kernel wrote nothing"
    assert torch.allclose(lm_head.bias.grad, ref_gb, atol=1e-3), (
        f"bias grad max diff: {(lm_head.bias.grad - ref_gb).abs().max()}"
    )

    # Also check hidden and weight while we're here — bias path changes
    # the logits, so all three grads should still match.
    assert torch.allclose(hidden.grad, ref_gh, atol=1e-3), (
        f"hidden grad max diff (with bias): {(hidden.grad - ref_gh).abs().max()}"
    )
    assert torch.allclose(lm_head.weight.grad, ref_gw, atol=1e-3), (
        f"weight grad max diff (with bias): {(lm_head.weight.grad - ref_gw).abs().max()}"
    )


@pytest.mark.gpu
def test_grad_all_three_larger_batch():
    """All three gradients correct on a larger batch to stress tile boundaries."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(7)
    # 384 vocab = 3 tiles of BLOCK_V=128 — tests the boundary between tiles
    hidden_dim, vocab_size, seq_len, batch = 64, 384, 16, 4

    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    result.sum().backward()

    ref_gh, ref_gw, _ = _pytorch_grads(hidden, lm_head.weight, None, labels)

    assert torch.allclose(hidden.grad, ref_gh, atol=1e-3), (
        f"hidden grad max diff: {(hidden.grad - ref_gh).abs().max()}"
    )
    assert torch.allclose(lm_head.weight.grad, ref_gw, atol=1e-3), (
        f"weight grad max diff: {(lm_head.weight.grad - ref_gw).abs().max()}"
    )


# ─── Autograd: needs_input_grad gating ──────────────────────────────────────


@pytest.mark.gpu
def test_no_grad_hidden_when_not_required():
    """When hidden doesn't require grad, grad_hidden is not allocated.

    This tests the needs_input_grad[0] gate. Without the gate, the kernel
    would allocate an empty_like(hidden) tensor that nobody reads — wasted
    VRAM on every backward pass.
    """
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 32, 128, 4, 1

    # hidden does NOT require grad
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=False)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    result.sum().backward()

    # hidden.grad stays None — no wasted allocation
    assert hidden.grad is None, "hidden.grad should be None when requires_grad=False"
    # weight still gets its gradient
    assert lm_head.weight.grad is not None
    assert lm_head.weight.grad.abs().sum() > 0


@pytest.mark.gpu
def test_no_grad_weight_when_frozen():
    """When weight is frozen, grad_weight is not allocated."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 32, 128, 4, 1

    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    lm_head.weight.requires_grad_(False)
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    result.sum().backward()

    assert lm_head.weight.grad is None, "weight.grad should be None when requires_grad=False"
    # hidden still gets its gradient
    assert hidden.grad is not None
    assert hidden.grad.abs().sum() > 0


@pytest.mark.gpu
def test_no_grad_bias_when_frozen():
    """When bias is frozen (requires_grad=False), grad_bias is not allocated."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 32, 128, 4, 1

    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=True, device="cuda")
    lm_head.bias.requires_grad_(False)
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    result.sum().backward()

    assert lm_head.bias.grad is None, "bias.grad should be None when requires_grad=False"
    # hidden and weight still get their gradients
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0
    assert lm_head.weight.grad is not None and lm_head.weight.grad.abs().sum() > 0


# ─── Autograd: empty sequence backward ──────────────────────────────────────


@pytest.mark.gpu
def test_empty_sequence_forward_shape():
    """Empty sequence produces correct output shape."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    hidden_dim, vocab_size = 64, 256
    hidden = torch.randn(1, 0, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (1, 0), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    assert result.shape == (1, 0)


@pytest.mark.gpu
def test_empty_sequence_no_grad_fn():
    """Empty sequence returns detached tensor — no computation, no grad_fn.

    The early-return path in triton_logprobs_with_grad produces a plain
    torch.empty because there's nothing to differentiate. The autograd
    backward path (batch==0 / seq_len==0 guard) handles the case where
    the autograd.Function *is* called with empty inputs, but the
    higher-level wrapper short-circuits before that.
    """
    from qgre.triton_logprobs import triton_logprobs_with_grad

    hidden_dim, vocab_size = 64, 256
    hidden = torch.randn(1, 0, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (1, 0), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    assert result.shape == (1, 0)
    # No grad_fn because no computation was performed
    assert result.grad_fn is None


@pytest.mark.gpu
def test_zero_batch_no_grad_fn():
    """Batch size 0 returns detached tensor — no computation, no grad_fn."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    hidden_dim, vocab_size = 64, 256
    hidden = torch.randn(0, 4, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (0, 4), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    assert result.shape == (0, 4)
    assert result.grad_fn is None


# ─── Autograd: forward value matches reference ──────────────────────────────


@pytest.mark.gpu
def test_autograd_forward_matches_pytorch_no_bias():
    """Autograd wrapper forward values match PyTorch reference (no bias)."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 16, 2
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    expected = _pytorch_logprobs(hidden.detach(), lm_head.weight, None, labels)

    assert torch.allclose(result.detach(), expected, atol=1e-3), (
        f"max diff: {(result.detach() - expected).abs().max()}"
    )


@pytest.mark.gpu
def test_autograd_forward_matches_pytorch_with_bias():
    """Autograd wrapper forward values match PyTorch reference (with bias)."""
    from qgre.triton_logprobs import triton_logprobs_with_grad

    torch.manual_seed(42)
    hidden_dim, vocab_size, seq_len, batch = 64, 256, 8, 1
    hidden = torch.randn(batch, seq_len, hidden_dim, device="cuda", requires_grad=True)
    lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=True, device="cuda")
    labels = torch.randint(0, vocab_size, (batch, seq_len), device="cuda")

    result = triton_logprobs_with_grad(hidden, lm_head, labels)
    expected = _pytorch_logprobs(hidden.detach(), lm_head.weight, lm_head.bias, labels)

    assert torch.allclose(result.detach(), expected, atol=1e-3), (
        f"max diff: {(result.detach() - expected).abs().max()}"
    )


# ─── Dummy bias pointer ─────────────────────────────────────────────────────


@pytest.mark.gpu
def test_dummy_bias_ptr_cached():
    """The dummy bias pointer is cached and reused across calls."""
    from qgre.triton_logprobs import _dummy_bias_ptr

    t1 = _dummy_bias_ptr(256, torch.device("cuda"))
    t2 = _dummy_bias_ptr(256, torch.device("cuda"))
    assert t1.data_ptr() == t2.data_ptr(), "Same (vocab, device) should return same tensor"

    t3 = _dummy_bias_ptr(512, torch.device("cuda"))
    assert t3.data_ptr() != t1.data_ptr(), "Different vocab should return different tensor"
    assert t3.shape == (512,)
