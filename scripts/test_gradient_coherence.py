"""Quick test of gradient coherence measurement to validate near-zero cosine."""

import torch
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qgre.gradient_coherence import compute_gradient_coherence


def test_with_dummy_model():
    """Test with aligned vs orthogonal gradients to validate measurement."""

    print("=" * 80)
    print("GRADIENT COHERENCE VALIDATION TEST")
    print("=" * 80)

    # Create a simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.Linear(50, 25),
        torch.nn.Linear(25, 10),
    )

    # Test 1: All gradients pointing in same direction (should have high cosine)
    print("\n[TEST 1] All gradients aligned (same direction)")
    for param in model.parameters():
        param.grad = torch.ones_like(param)

    stats = compute_gradient_coherence(model)
    print(f"  Mean cosine: {stats['mean_cosine']:.6f} (expect ~1.0)")
    print(f"  Norm ratio: {stats['norm_ratio']:.2f}")
    print(f"  n_layers: {stats['n_layers']}, n_comparisons: {stats['n_comparisons']}")

    # Test 2: Random orthogonal gradients (should have near-zero cosine)
    print("\n[TEST 2] Random gradients (orthogonal)")
    for param in model.parameters():
        param.grad = torch.randn_like(param)

    stats = compute_gradient_coherence(model)
    print(f"  Mean cosine: {stats['mean_cosine']:.6f} (expect ~0.0)")
    print(f"  Norm ratio: {stats['norm_ratio']:.2f}")
    print(f"  Per-layer cosines: {stats['per_layer_cosines']}")

    # Test 3: Opposite directions (should have negative cosine)
    print("\n[TEST 3] Alternating directions (opposing)")
    for i, param in enumerate(model.parameters()):
        if i % 2 == 0:
            param.grad = torch.ones_like(param)
        else:
            param.grad = -torch.ones_like(param)

    stats = compute_gradient_coherence(model)
    print(f"  Mean cosine: {stats['mean_cosine']:.6f} (expect ~-1.0 or mixed)")
    print(f"  Norm ratio: {stats['norm_ratio']:.2f}")
    print(f"  Per-layer cosines: {stats['per_layer_cosines']}")

    # Test 4: LoRA-like structure (low-rank)
    print("\n[TEST 4] LoRA-like low-rank gradients")
    for param in model.parameters():
        # Create low-rank gradient (rank 5)
        u = torch.randn(param.shape[0], 5) if param.dim() == 2 else torch.randn_like(param).unsqueeze(-1).expand(-1, 5)
        v = torch.randn(5, param.shape[-1]) if param.dim() == 2 else torch.randn(5, *param.shape[1:]) if param.dim() > 1 else torch.randn(5)
        if param.dim() == 2:
            param.grad = u @ v
        else:
            param.grad = torch.randn_like(param)  # Fallback for 1D

    stats = compute_gradient_coherence(model)
    print(f"  Mean cosine: {stats['mean_cosine']:.6f}")
    print(f"  Norm ratio: {stats['norm_ratio']:.2f}")
    print(f"  First 3 cosines: {stats['per_layer_cosines'][:3]}")

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  - If real training shows cosine ~1.0 → layers aligned (healthy)")
    print("  - If real training shows cosine ~0.0 → layers orthogonal (eddies forming)")
    print("  - If real training shows cosine ~-1.0 → layers opposing (serious conflict)")
    print("  - LoRA low-rank structure can naturally produce low cosines")
    print("=" * 80)


if __name__ == "__main__":
    test_with_dummy_model()
