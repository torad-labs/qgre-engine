"""Test attention-constrained advantage - both attention and ERIC (entropy) modes."""

import torch

from qgre.attention_bonds import (
    apply_importance_constraint,
    compute_bond_strength,
    compute_causal_decay,
    compute_entropy_importance,
    select_attention_layer,
)


def test_bond_strength_computation():
    """Verify bond strength is computed correctly from attention."""
    batch, heads, seq = 2, 4, 10

    # Create mock attention: later tokens attend to earlier tokens
    # attention[b, h, i, j] = attention from position i to position j
    attention = torch.zeros(batch, heads, seq, seq)

    # Token 5 attends strongly to token 2 (high bond for token 2)
    attention[:, :, 5, 2] = 0.8
    # Token 7 attends to token 3
    attention[:, :, 7, 3] = 0.6
    # Token 9 attends to token 0
    attention[:, :, 9, 0] = 0.9

    bond = compute_bond_strength(attention, seq_len=seq, mode="max_received")

    # Token 2 should have high bond (attended by token 5)
    assert bond[0, 2].item() >= 0.7, f"Token 2 bond should be high, got {bond[0, 2]}"
    # Token 3 should have medium bond
    assert bond[0, 3].item() >= 0.5, f"Token 3 bond should be medium, got {bond[0, 3]}"
    # Token 0 should have high bond (attended by token 9)
    assert bond[0, 0].item() >= 0.8, f"Token 0 bond should be high, got {bond[0, 0]}"
    # Last token should have 0 bond (nothing attends to it)
    assert bond[0, -1].item() == 0.0, f"Last token should have 0 bond, got {bond[0, -1]}"

    print(f"Bond strength: {bond[0].tolist()}")
    print("test_bond_strength_computation PASSED")


def test_advantage_dampening():
    """Verify high-importance tokens get dampened POSITIVE advantages only."""
    seq_len = 10

    # Importance: token 3 has high importance (0.8), others have low (0.1)
    importance = torch.full((seq_len,), 0.1)
    importance[3] = 0.8

    # Test 1: POSITIVE advantages should be dampened
    positive_advs = torch.ones(seq_len)
    dampened_pos = apply_importance_constraint(positive_advs, importance, strength=1.0)

    # Token 3 should have lower advantage (dampened)
    assert dampened_pos[3] < dampened_pos[0], (
        "High-importance token should have lower positive advantage"
    )
    assert dampened_pos[3] < 0.6, (
        f"Token 3 positive advantage should be dampened below 0.6, got {dampened_pos[3]}"
    )

    # Test 2: NEGATIVE advantages should NOT be dampened (asymmetric, sign-gated)
    # AC8 fix: negative advantage + high importance → NO DAMPEN (preserve correction signals)
    negative_advs = -torch.ones(seq_len)
    dampened_neg = apply_importance_constraint(negative_advs, importance, strength=1.0)

    # Token 3 negative advantage should be UNCHANGED (no dampening for corrections)
    assert dampened_neg[3] == -1.0, (
        f"Negative advantages should NOT be dampened, got {dampened_neg[3]} instead of -1.0"
    )
    # Positive should still be dampened
    assert dampened_pos[3] < 1.0, f"Positive advantages SHOULD be dampened, got {dampened_pos[3]}"

    print(f"Positive advantages (dampened): {[f'{x:.3f}' for x in dampened_pos.tolist()]}")
    print(f"Negative advantages (dampened): {[f'{x:.3f}' for x in dampened_neg.tolist()]}")
    print("test_advantage_dampening PASSED")


def test_causal_decay_computation():
    """Verify causal decay is computed correctly from sequence length.

    Formula: decay = 1.0 / (1 + log2(seq_len / 128))
    - seq=128: log2(1)=0, decay=1.0
    - seq=256: log2(2)=1, decay=0.5
    - seq=512: log2(4)=2, decay=0.33
    - seq=2048: log2(16)=4, decay=0.2
    """
    # Short sequences: decay = 1.0 (linear, strong early protection)
    decay_128 = compute_causal_decay(128)
    assert 0.95 <= decay_128 <= 1.0, f"decay(128) should be 1.0, got {decay_128}"

    # Medium sequences: decay ~0.33 (512 tokens = 2 doublings from 128)
    decay_512 = compute_causal_decay(512)
    assert 0.30 <= decay_512 <= 0.36, f"decay(512) should be ~0.33, got {decay_512}"

    # Long sequences: decay = 0.2 (clamped floor)
    decay_2048 = compute_causal_decay(2048)
    assert 0.19 <= decay_2048 <= 0.21, f"decay(2048) should be 0.2, got {decay_2048}"

    # Monotonic: longer sequences should have lower decay
    assert decay_128 >= decay_512 >= decay_2048, (
        f"Decay should be monotonically decreasing: {decay_128} >= {decay_512} >= {decay_2048}"
    )

    print(
        f"Causal decay: seq=128 → {decay_128:.3f}, seq=512 → {decay_512:.3f}, seq=2048 → {decay_2048:.3f}"
    )
    print("test_causal_decay_computation PASSED")


def test_select_attention_layer():
    """Verify layer selection works."""
    n_layers = 28
    batch, heads, seq = 1, 4, 10

    # Mock attentions tuple
    attentions = tuple(torch.randn(batch, heads, seq, seq) * (i + 1) for i in range(n_layers))

    # Select last layer (-1)
    last = select_attention_layer(attentions, -1)
    assert torch.allclose(last, attentions[-1])

    # Select middle layer (-2)
    middle = select_attention_layer(attentions, -2)
    assert torch.allclose(middle, attentions[n_layers // 2])

    # Select explicit layer
    explicit = select_attention_layer(attentions, 5)
    assert torch.allclose(explicit, attentions[5])

    print("test_select_attention_layer PASSED")


def test_entropy_importance():
    """Test ERIC (Entropy-Regulated Importance Constraint)."""
    batch, seq, vocab = 2, 20, 1000

    # Create mock logits with varying entropy
    logits = torch.randn(batch, seq, vocab)

    # Make early tokens (0-5) confident (low entropy = peaked distribution)
    logits[:, 0:5, 0] = 15.0  # Strong peak at vocab[0]

    # Make later tokens (15-20) uncertain (high entropy = flat distribution)
    # Random logits already have high entropy, so no change needed

    # Compute importance with entropy_position mode
    importance = compute_entropy_importance(
        logits,
        seq_len=seq,
        mode="entropy_position",
        position_decay=0.5,
    )

    assert importance.shape == (
        batch,
        seq,
    ), f"Expected shape {(batch, seq)}, got {importance.shape}"
    assert importance.min() >= 0.0, "Importance should be >= 0"
    assert importance.max() <= 1.0, "Importance should be <= 1"

    # Early tokens with low entropy should have HIGH importance (anchor protection)
    early_importance = importance[:, 0:5].mean().item()
    # Late tokens with high entropy should have LOW importance (decision points)
    late_importance = importance[:, 15:20].mean().item()

    assert early_importance > late_importance, (
        f"Early/low-entropy tokens should have higher importance: {early_importance:.4f} > {late_importance:.4f}"
    )

    print(f"Early tokens (0-5) importance: {early_importance:.4f}")
    print(f"Late tokens (15-20) importance: {late_importance:.4f}")
    print("test_entropy_importance PASSED")


def test_eric_modes():
    """Test different ERIC computation modes."""
    batch, seq, vocab = 1, 10, 100

    logits = torch.randn(batch, seq, vocab)
    # Make first token confident for clear test signal
    logits[:, 0, 0] = 10.0

    # Test all modes
    for mode in ["entropy", "position", "entropy_position"]:
        importance = compute_entropy_importance(
            logits,
            seq_len=seq,
            mode=mode,
            position_decay=0.5,
        )
        assert importance.shape == (batch, seq), f"Mode {mode}: wrong shape"
        assert importance.min() >= 0.0, f"Mode {mode}: negative importance"
        assert importance.max() <= 1.0, f"Mode {mode}: importance > 1"
        print(
            f"Mode '{mode}': min={importance.min():.3f}, max={importance.max():.3f}, mean={importance.mean():.3f}"
        )

    # Verify position-only mode creates monotonic decay
    position_importance = compute_entropy_importance(
        logits,
        seq_len=seq,
        mode="position",
        position_decay=1.0,
    )
    for i in range(seq - 1):
        assert position_importance[0, i] >= position_importance[0, i + 1], (
            f"Position mode should be monotonically decreasing: pos {i} ({position_importance[0, i]:.3f}) < pos {i + 1} ({position_importance[0, i + 1]:.3f})"
        )

    print("test_eric_modes PASSED")


if __name__ == "__main__":
    test_bond_strength_computation()
    test_advantage_dampening()
    test_causal_decay_computation()
    test_select_attention_layer()
    test_entropy_importance()
    test_eric_modes()
    print("\n" + "=" * 50)
    print("All attention constraint tests PASSED")
