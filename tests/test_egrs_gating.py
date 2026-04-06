"""Test EGRS soft gating functions."""

import torch

from qgre.attention_bonds import compute_confidence_gate, compute_normalized_entropy


def test_normalized_entropy_uniform_distribution():
    """Uniform distribution should have entropy ~1.0."""
    batch, seq, vocab = 2, 10, 1000

    # Uniform logits = uniform distribution = max entropy
    logits = torch.zeros(batch, seq, vocab)

    entropy = compute_normalized_entropy(logits, vocab_size=vocab)

    assert entropy.shape == (batch, seq)
    # Uniform should be very close to 1.0
    assert entropy.min() > 0.95, f"Uniform entropy should be ~1.0, got min={entropy.min():.4f}"
    assert entropy.max() <= 1.0, f"Entropy should not exceed 1.0, got max={entropy.max():.4f}"
    print(f"Uniform entropy: mean={entropy.mean():.4f}")


def test_normalized_entropy_peaked_distribution():
    """Peaked distribution should have entropy ~0.0."""
    batch, seq, vocab = 2, 10, 1000

    # Peaked logits: one token has very high logit
    logits = torch.zeros(batch, seq, vocab)
    logits[:, :, 0] = 20.0  # Very peaked at first token

    entropy = compute_normalized_entropy(logits, vocab_size=vocab)

    assert entropy.shape == (batch, seq)
    # Peaked should be very close to 0.0
    assert entropy.max() < 0.05, f"Peaked entropy should be ~0.0, got max={entropy.max():.4f}"
    assert entropy.min() >= 0.0, f"Entropy should not be negative, got min={entropy.min():.4f}"
    print(f"Peaked entropy: mean={entropy.mean():.4f}")


def test_normalized_entropy_infer_vocab_size():
    """Test that vocab_size can be inferred from logits."""
    batch, seq, vocab = 1, 5, 500

    logits = torch.randn(batch, seq, vocab)

    # Without explicit vocab_size
    entropy = compute_normalized_entropy(logits)

    assert entropy.shape == (batch, seq)
    assert (entropy >= 0.0).all() and (entropy <= 1.0).all()


def test_confidence_gate_boundaries():
    """Test that confidence gate gives expected values at boundaries."""
    # Create entropy values spanning the range
    entropy = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    threshold = 0.5
    temperature = 0.1

    gate = compute_confidence_gate(entropy, threshold=threshold, temperature=temperature)

    # At threshold, gate should be 0.5
    assert abs(gate[2].item() - 0.5) < 0.01, f"Gate at threshold should be 0.5, got {gate[2]:.4f}"

    # Below threshold (low entropy = confident), gate should be ~0
    assert gate[0].item() < 0.01, f"Gate at entropy=0 should be ~0, got {gate[0]:.4f}"
    assert gate[1].item() < 0.1, f"Gate at entropy=0.25 should be <0.1, got {gate[1]:.4f}"

    # Above threshold (high entropy = uncertain), gate should be ~1
    assert gate[4].item() > 0.99, f"Gate at entropy=1 should be ~1, got {gate[4]:.4f}"
    assert gate[3].item() > 0.9, f"Gate at entropy=0.75 should be >0.9, got {gate[3]:.4f}"

    print(f"Confidence gate values: {[f'{g:.4f}' for g in gate.tolist()]}")


def test_confidence_gate_temperature_effect():
    """Test that temperature controls sharpness of transition."""
    entropy = torch.tensor([0.4, 0.45, 0.5, 0.55, 0.6])
    threshold = 0.5

    # Sharp transition (low temperature)
    gate_sharp = compute_confidence_gate(entropy, threshold=threshold, temperature=0.01)
    # Soft transition (high temperature)
    gate_soft = compute_confidence_gate(entropy, threshold=threshold, temperature=0.5)

    # Sharp should have bigger difference between 0.4 and 0.6
    sharp_range = gate_sharp[4].item() - gate_sharp[0].item()
    soft_range = gate_soft[4].item() - gate_soft[0].item()

    assert (
        sharp_range > soft_range
    ), f"Sharp gate should have larger range than soft: {sharp_range:.4f} vs {soft_range:.4f}"

    print(f"Sharp (T=0.01): {[f'{g:.4f}' for g in gate_sharp.tolist()]}")
    print(f"Soft (T=0.5): {[f'{g:.4f}' for g in gate_soft.tolist()]}")


def test_confidence_gate_monotonic():
    """Gate should be monotonically increasing with entropy."""
    entropy = torch.linspace(0, 1, 100)
    gate = compute_confidence_gate(entropy, threshold=0.5, temperature=0.1)

    # Check monotonicity
    for i in range(len(gate) - 1):
        assert (
            gate[i] <= gate[i + 1]
        ), f"Gate should be monotonic: gate[{i}]={gate[i]:.4f} > gate[{i+1}]={gate[i+1]:.4f}"

    print("Confidence gate is monotonically increasing")


def test_confidence_gate_batch():
    """Test that confidence gate works with batched input."""
    batch, seq = 4, 20
    entropy = torch.rand(batch, seq)

    gate = compute_confidence_gate(entropy, threshold=0.5, temperature=0.1)

    assert gate.shape == (batch, seq)
    assert (gate >= 0.0).all() and (gate <= 1.0).all()


def test_egrs_quadrant_classification():
    """Test that we can classify tokens into the 4 quadrants."""
    # Simulate a batch with varying entropy and correctness
    batch, seq = 1, 8
    vocab = 1000

    # Create logits with known entropy patterns:
    # Tokens 0-1: low entropy (confident)
    # Tokens 2-3: medium entropy (boundary)
    # Tokens 4-5: high entropy (uncertain)
    # Tokens 6-7: very high entropy (very uncertain)
    logits = torch.zeros(batch, seq, vocab)
    logits[:, 0:2, 0] = 15.0  # Peaked (confident)
    logits[:, 2:4, :100] = 2.0  # Somewhat peaked
    logits[:, 4:6, :500] = 1.0  # Spread out
    # tokens 6-7 stay uniform (max entropy)

    entropy = compute_normalized_entropy(logits, vocab_size=vocab)
    gate = compute_confidence_gate(entropy, threshold=0.5, temperature=0.1)

    # Simulate span correctness (step-level, not token-level)
    step_correctness = {0: True, 1: False}  # step 0 correct, step 1 wrong
    token_to_step = [0, 0, 0, 0, 1, 1, 1, 1]  # first 4 tokens = step 0, last 4 = step 1

    # Classify into quadrants
    quadrants = []
    for t in range(seq):
        step = token_to_step[t]
        correct = step_correctness[step]
        confident = gate[0, t].item() < 0.5  # Low gate = confident

        if correct and not confident:
            q = "Q1"  # uncertain+correct → reinforce
        elif correct and confident:
            q = "Q2"  # confident+correct → do nothing
        elif not correct and confident:
            q = "Q3"  # confident+wrong → entropy boost
        else:
            q = "Q4"  # uncertain+wrong → hint

        quadrants.append(q)

    print(f"Entropy: {[f'{e:.3f}' for e in entropy[0].tolist()]}")
    print(f"Gate: {[f'{g:.3f}' for g in gate[0].tolist()]}")
    print(f"Quadrants: {quadrants}")

    # Verify expected quadrants (approximate due to soft gating)
    # Tokens 0-1 (confident, correct) should be Q2
    assert quadrants[0] == "Q2", f"Token 0 should be Q2, got {quadrants[0]}"
    assert quadrants[1] == "Q2", f"Token 1 should be Q2, got {quadrants[1]}"
    # Tokens 4-5 (uncertain, wrong) should be Q4
    assert quadrants[4] == "Q4", f"Token 4 should be Q4, got {quadrants[4]}"


if __name__ == "__main__":
    test_normalized_entropy_uniform_distribution()
    test_normalized_entropy_peaked_distribution()
    test_normalized_entropy_infer_vocab_size()
    test_confidence_gate_boundaries()
    test_confidence_gate_temperature_effect()
    test_confidence_gate_monotonic()
    test_confidence_gate_batch()
    test_egrs_quadrant_classification()
    print("\n" + "=" * 50)
    print("All EGRS gating tests PASSED")
