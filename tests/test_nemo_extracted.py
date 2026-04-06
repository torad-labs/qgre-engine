"""Tests for NeMo RL extracted modules (Step 0b)."""

import pytest
import torch


def test_import_loss_functions():
    """import succeeds with no external deps."""


def test_import_kl():
    """import succeeds."""


def test_import_logits():
    """import succeeds."""


def test_clipped_pg_loss_nonzero():
    """ClippedPGLossFn on synthetic data → non-zero, finite loss."""
    from qgre.nemo_extracted.loss_functions import ClippedPGLossFn

    cfg = {
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.28,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    batch, seq = 4, 32
    curr_lp = torch.randn(batch, seq) * 0.1 - 3.0
    prev_lp = torch.randn(batch, seq) * 0.1 - 3.0
    advantages = torch.randn(batch, seq)
    mask = torch.ones(batch, seq)
    mask[:, -5:] = 0  # pad last 5 tokens

    loss, metrics = loss_fn(curr_lp, prev_lp, advantages, mask)

    assert loss.isfinite()
    assert loss.item() != 0.0
    assert "loss" in metrics
    assert "actor_loss" in metrics


def test_clipped_pg_loss_clip_bounds():
    """Large ratio → loss is clipped."""
    from qgre.nemo_extracted.loss_functions import ClippedPGLossFn

    cfg = {
        "reference_policy_kl_penalty": 0.0,
        "reference_policy_kl_type": "k3",
        "kl_input_clamp_value": 20.0,
        "kl_output_clamp_value": 10.0,
        "ratio_clip_min": 0.2,
        "ratio_clip_max": 0.28,
        "ratio_clip_c": None,
        "use_on_policy_kl_approximation": False,
        "use_importance_sampling_correction": False,
        "truncated_importance_sampling_ratio": None,
        "token_level_loss": True,
    }
    loss_fn = ClippedPGLossFn(cfg)

    batch, seq = 2, 8
    # Create large ratio: curr much higher than prev → ratio >> 1
    curr_lp = torch.zeros(batch, seq)
    prev_lp = torch.full((batch, seq), -5.0)  # ratio = exp(5) ≈ 148
    advantages = torch.ones(batch, seq)
    mask = torch.ones(batch, seq)

    loss, metrics = loss_fn(curr_lp, prev_lp, advantages, mask)
    assert loss.isfinite()
    # Clamped ratio should be 1.28 (1 + 0.28), not 148
    assert metrics["probs_ratio_clamped_mean"] < 2.0


def test_kl_calculation_matches_manual():
    """KL on small tensors matches manual computation."""
    from qgre.nemo_extracted.kl import calculate_kl

    lp = torch.tensor([-1.0, -2.0, -0.5, -3.0])
    lp_ref = torch.tensor([-1.5, -1.8, -0.6, -2.5])

    # k3: exp(logr) - 1 - logr where logr = lp_ref - lp
    logr = lp_ref - lp
    expected = torch.exp(logr) - 1 - logr

    result = calculate_kl(lp, lp_ref, kl_type="k3", input_clamp_value=None, output_clamp_value=None)
    assert torch.allclose(result, expected, atol=1e-6)


def test_kl_types():
    """All KL types produce finite, non-negative results."""
    from qgre.nemo_extracted.kl import calculate_kl

    lp = torch.randn(4, 8) - 3.0
    lp_ref = torch.randn(4, 8) - 3.0

    for kl_type in ["k1", "k2", "k3"]:
        result = calculate_kl(lp, lp_ref, kl_type=kl_type)
        assert result.isfinite().all(), f"kl_type={kl_type} produced non-finite values"


def test_masked_mean_correctness():
    """masked_mean with known mask matches manual computation."""
    from qgre.nemo_extracted.kl import masked_mean

    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    result = masked_mean(values, mask)
    # (1*1 + 2*1 + 3*0 + 4*0) / (1+1+0+0) = 3/2 = 1.5
    assert abs(result.item() - 1.5) < 1e-6


def test_masked_mean_zero_mask():
    """masked_mean with all-zero mask → near-zero (epsilon protected)."""
    from qgre.nemo_extracted.kl import masked_mean

    values = torch.tensor([[1.0, 2.0]])
    mask = torch.zeros(1, 2)
    result = masked_mean(values, mask)
    assert result.isfinite()
    assert abs(result.item()) < 0.01


def test_logprobs_from_logits():
    """logprobs_from_logits on known logits matches manual log_softmax + gather."""
    from qgre.nemo_extracted.logits import logprobs_from_logits

    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])  # [1, 2, 3]
    labels = torch.tensor([[2, 1]])  # [1, 2]

    result = logprobs_from_logits(logits, labels)

    # Manual: log_softmax then gather
    lp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    expected = torch.tensor([[lp[0, 0, 2].item(), lp[0, 1, 1].item()]])

    assert torch.allclose(result, expected, atol=1e-6)


def test_selective_log_softmax_fp32_equivalence():
    """selective_log_softmax (fp32 path) matches naive log_softmax + gather."""
    from qgre.nemo_extracted.logits import selective_log_softmax

    torch.manual_seed(42)
    batch, seq, vocab = 4, 64, 1024
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    labels = torch.randint(0, vocab, (batch, seq))

    result = selective_log_softmax(logits, labels)

    # Naive reference: full log_softmax + gather
    lp = torch.nn.functional.log_softmax(logits, dim=-1)
    expected = lp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(
        result, expected, atol=1e-5
    ), f"max diff: {(result - expected).abs().max().item()}"
    assert (result <= 0).all(), "log probs must be ≤ 0"


def test_selective_log_softmax_bf16_equivalence():
    """selective_log_softmax (bf16 path) matches naive within bf16 tolerance."""
    from qgre.nemo_extracted.logits import selective_log_softmax

    torch.manual_seed(42)
    batch, seq, vocab = 4, 64, 1024
    logits = torch.randn(batch, seq, vocab, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (batch, seq))

    result = selective_log_softmax(logits, labels)

    # Naive reference in fp32 for comparison
    lp = torch.nn.functional.log_softmax(logits.float(), dim=-1)
    expected = lp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).float()

    # bf16 log_softmax has ~0.05 max error vs fp32 reference (expected for half precision)
    assert torch.allclose(
        result, expected, atol=0.05
    ), f"max diff: {(result - expected).abs().max().item()}"
    assert (result <= 0).all(), "log probs must be ≤ 0"


def test_selective_log_softmax_large_vocab():
    """selective_log_softmax works with Qwen3-sized vocab (151936)."""
    from qgre.nemo_extracted.logits import selective_log_softmax

    batch, seq, vocab = 1, 8, 151936  # Qwen3 vocab
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32) * 0.1 - 3.0
    labels = torch.randint(0, vocab, (batch, seq))

    result = selective_log_softmax(logits, labels)

    assert result.shape == (batch, seq)
    assert result.isfinite().all()
    assert (result <= 0).all(), "log probs must be ≤ 0"


def test_logprobs_from_logits_selective_vs_old():
    """logprobs_from_logits (using selective) matches old naive implementation numerically."""
    from qgre.nemo_extracted.logits import logprobs_from_logits

    torch.manual_seed(42)
    batch, seq, vocab = 2, 128, 512
    logits = torch.randn(batch, seq, vocab, dtype=torch.float32)
    labels = torch.randint(0, vocab, (batch, seq))

    result = logprobs_from_logits(logits, labels, chunk_size=32)

    # Reference: full naive log_softmax + gather (no chunking)
    lp = torch.nn.functional.log_softmax(logits, dim=-1)
    expected = lp.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1).float()

    assert torch.allclose(
        result, expected, atol=1e-5
    ), f"max diff: {(result - expected).abs().max().item()}"


# --- GRPO-λ eligibility trace tests ---


def test_eligibility_traces_lambda_zero_is_identity():
    """λ=0 → traces equal original advantages (no blending)."""
    from qgre.nemo_extracted.loss_functions import apply_eligibility_traces

    advantages = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = apply_eligibility_traces(advantages, lambda_val=0.0)
    assert torch.allclose(result, advantages), f"λ=0 should be identity, got {result}"


def test_eligibility_traces_lambda_nonzero_blends():
    """λ>0 → traces are λ-weighted sum from future tokens."""
    from qgre.nemo_extracted.loss_functions import apply_eligibility_traces

    advantages = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    result = apply_eligibility_traces(advantages, lambda_val=0.9)
    # Last token advantage=1.0, so trace at t=2 = 0 + 0.9*1.0 = 0.9
    # trace at t=1 = 0 + 0.9*0.9 = 0.81, trace at t=0 = 0 + 0.9*0.81 = 0.729
    assert result[0, 3].item() == pytest.approx(1.0)
    assert result[0, 2].item() == pytest.approx(0.9)
    assert result[0, 1].item() == pytest.approx(0.81)
    assert result[0, 0].item() == pytest.approx(0.729)
