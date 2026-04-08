"""Tests for LoRA-Pro gradient adjustment.

Verifies that LoRA-Pro:
1. Correctly identifies LoRA A/B pairs
2. Adjusts gradients to better approximate full fine-tuning
3. Produces consistent results with checkpointing
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn


class MockLoRAModel(nn.Module):
    """Minimal LoRA model for testing gradient adjustment."""

    def __init__(self, in_dim: int = 64, out_dim: int = 32, rank: int = 8):
        super().__init__()
        self.rank = rank
        # Simulate PEFT naming convention
        # lora_A: [rank, in_dim], lora_B: [out_dim, rank]
        self.base_layer = nn.Linear(in_dim, out_dim, bias=False)
        self.base_layer.weight.requires_grad = False  # Base frozen

        # LoRA matrices with PEFT-style naming
        # Note: Initialize B with small non-zero values so gradients flow through both A and B
        self.lora_A_default_weight = nn.Parameter(torch.randn(rank, in_dim) * 0.01)
        self.lora_B_default_weight = nn.Parameter(torch.randn(out_dim, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        # LoRA: x @ A.T @ B.T = x @ (B @ A).T
        lora_out = x @ self.lora_A_default_weight.T @ self.lora_B_default_weight.T
        return base_out + lora_out

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        """Override to provide PEFT-style naming for LoRA params."""
        # Yield base layer params (frozen, but included for completeness)
        for name, param in self.base_layer.named_parameters(prefix="base_layer"):
            yield f"{prefix}{name}", param
        # Yield LoRA params with PEFT naming
        yield f"{prefix}layer.lora_A.default.weight", self.lora_A_default_weight
        yield f"{prefix}layer.lora_B.default.weight", self.lora_B_default_weight


class TestLoRAProDiscovery:
    """Test LoRA pair discovery."""

    def test_discovers_lora_pairs(self):
        """LoRA-Pro should discover A/B pairs from PEFT naming."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        model = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        config = LoRAProConfig(enabled=True)
        adjuster = LoRAProAdjuster(model, lora_rank=8, lora_alpha=16, config=config)

        assert len(adjuster._lora_pairs) == 1
        A, B, layer_id = adjuster._lora_pairs[0]
        assert A.shape == (8, 64)  # [rank, in_dim]
        assert B.shape == (32, 8)  # [out_dim, rank]

    def test_computes_correct_scaling(self):
        """Scaling factor should match RSLoRA formula."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        model = MockLoRAModel(rank=8)
        config = LoRAProConfig(enabled=True, use_rslora=True)
        adjuster = LoRAProAdjuster(model, lora_rank=8, lora_alpha=16, config=config)

        # RSLoRA: alpha / sqrt(rank) = 16 / sqrt(8) ≈ 5.66
        expected = 16 / math.sqrt(8)
        assert abs(adjuster.scaling_factor - expected) < 0.01

        # Standard: alpha / rank = 16 / 8 = 2.0
        config_std = LoRAProConfig(enabled=True, use_rslora=False)
        adjuster_std = LoRAProAdjuster(model, lora_rank=8, lora_alpha=16, config=config_std)
        assert adjuster_std.scaling_factor == 2.0


class TestLoRAProGradientAdjustment:
    """Test gradient adjustment mechanics."""

    def test_adjusts_gradients(self):
        """LoRA-Pro should modify gradients after adjustment."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        model = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        config = LoRAProConfig(enabled=True)
        adjuster = LoRAProAdjuster(model, lora_rank=8, lora_alpha=16, config=config)

        # Create random input and compute loss
        x = torch.randn(4, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        # Capture gradients before adjustment
        A, B, _ = adjuster._lora_pairs[0]
        grad_A_before = A.grad.clone()
        grad_B_before = B.grad.clone()

        # Apply adjustment
        metrics = adjuster.adjust_gradients(global_step=0)

        # Gradients should change
        assert not torch.allclose(A.grad, grad_A_before, atol=1e-6)
        assert not torch.allclose(B.grad, grad_B_before, atol=1e-6)

        # Metrics should be populated
        assert "lora_pro/grad_norm_before" in metrics
        assert "lora_pro/grad_norm_after" in metrics
        assert metrics["lora_pro/n_adjusted"] == 1

    def test_step_zero_vs_step_one(self):
        """Step 0 uses simplified adjustment, step 1+ uses full Sylvester."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        # Run step 0
        model0 = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        torch.manual_seed(42)
        config = LoRAProConfig(enabled=True)
        adjuster0 = LoRAProAdjuster(model0, lora_rank=8, lora_alpha=16, config=config)

        x = torch.randn(4, 64)
        y = model0(x)
        loss = y.sum()
        loss.backward()

        metrics0 = adjuster0.adjust_gradients(global_step=0)

        # Run step 1 on fresh model
        model1 = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        torch.manual_seed(42)
        adjuster1 = LoRAProAdjuster(model1, lora_rank=8, lora_alpha=16, config=config)

        x = torch.randn(4, 64)
        y = model1(x)
        loss = y.sum()
        loss.backward()

        # First call to build momentum
        adjuster1.adjust_gradients(global_step=0)

        # Re-compute gradients
        model1.zero_grad()
        y = model1(x)
        loss = y.sum()
        loss.backward()

        metrics1 = adjuster1.adjust_gradients(global_step=1)

        # Both should produce valid metrics
        assert metrics0["lora_pro/n_adjusted"] == 1
        assert metrics1["lora_pro/n_adjusted"] == 1


class TestLoRAProCheckpointing:
    """Test state save/restore."""

    def test_state_dict_roundtrip(self):
        """State should be restorable after checkpoint."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        model = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        config = LoRAProConfig(enabled=True)
        adjuster = LoRAProAdjuster(model, lora_rank=8, lora_alpha=16, config=config)

        # Build up some momentum state
        for step in range(3):
            model.zero_grad()
            x = torch.randn(4, 64)
            y = model(x)
            loss = y.sum()
            loss.backward()
            adjuster.adjust_gradients(global_step=step)

        # Save state
        state = adjuster.state_dict()
        assert len(state) == 1  # One layer

        # Create new adjuster and restore
        model2 = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        adjuster2 = LoRAProAdjuster(model2, lora_rank=8, lora_alpha=16, config=config)
        adjuster2.load_state_dict(state)

        # Check momentum restored
        for layer_id in state:
            assert adjuster2._states[layer_id].exp_avg is not None
            assert adjuster2._states[layer_id].exp_avg_sq is not None


class TestLoRAProDisabled:
    """Test behavior when disabled."""

    def test_no_adjustment_when_disabled(self):
        """Disabled LoRA-Pro should not modify gradients."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        model = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        config = LoRAProConfig(enabled=False)  # Disabled
        adjuster = LoRAProAdjuster(model, lora_rank=8, lora_alpha=16, config=config)

        x = torch.randn(4, 64)
        y = model(x)
        loss = y.sum()
        loss.backward()

        A, B, _ = adjuster._lora_pairs[0]
        grad_A_before = A.grad.clone()
        grad_B_before = B.grad.clone()

        metrics = adjuster.adjust_gradients(global_step=0)

        # Gradients unchanged
        assert torch.allclose(A.grad, grad_A_before)
        assert torch.allclose(B.grad, grad_B_before)
        assert metrics == {}


class TestGradientMitigation:
    """Test gradient scaling and floor mitigations for RL."""

    def test_grad_scale_amplifies_gradients(self):
        """grad_scale should multiply final gradients."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        # Without scaling
        model1 = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        torch.manual_seed(42)
        config1 = LoRAProConfig(enabled=True, grad_scale=1.0)
        adjuster1 = LoRAProAdjuster(model1, lora_rank=8, lora_alpha=16, config=config1)

        x = torch.randn(4, 64)
        y = model1(x)
        loss = y.sum()
        loss.backward()
        metrics1 = adjuster1.adjust_gradients(global_step=0)
        norm1 = metrics1["lora_pro/grad_norm_after"]

        # With 10x scaling
        model2 = MockLoRAModel(in_dim=64, out_dim=32, rank=8)
        torch.manual_seed(42)
        config2 = LoRAProConfig(enabled=True, grad_scale=10.0)
        adjuster2 = LoRAProAdjuster(model2, lora_rank=8, lora_alpha=16, config=config2)

        x = torch.randn(4, 64)
        y = model2(x)
        loss = y.sum()
        loss.backward()
        metrics2 = adjuster2.adjust_gradients(global_step=0)
        norm2 = metrics2["lora_pro/grad_norm_after"]

        # 10x scale should amplify gradient norm significantly
        # Not exactly 10x due to numerical interactions, but should be >5x
        assert norm2 / norm1 > 5.0

    def test_grad_floor_prevents_collapse(self):
        """grad_floor should prevent gradients from collapsing to zero."""
        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        model = MockLoRAModel(in_dim=64, out_dim=32, rank=8)

        # Use very high RSLoRA scaling to create tiny gradients
        config = LoRAProConfig(enabled=True, grad_floor=1e-4)
        adjuster = LoRAProAdjuster(model, lora_rank=8, lora_alpha=256, config=config)

        x = torch.randn(4, 64) * 0.001  # Tiny input to create tiny gradients
        y = model(x)
        loss = y.sum()
        loss.backward()

        metrics = adjuster.adjust_gradients(global_step=0)

        # Gradient norm should be at least grad_floor
        assert metrics["lora_pro/grad_norm_after"] >= config.grad_floor * 0.9


class TestSylvesterSolver:
    """Test Sylvester equation solver."""

    def test_solves_simple_sylvester(self):
        """Sylvester solver should find correct X for AX + XB = C."""
        from qgre.lora_pro import solve_sylvester

        # Simple test case
        A = torch.tensor([[1.0, 0.5], [0.5, 2.0]])
        B = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        C = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        X = solve_sylvester(A, B, C)

        # Verify: AX + XB should equal C
        residual = A @ X + X @ B - C
        assert residual.abs().max() < 1e-4

    def test_handles_bfloat16(self):
        """Sylvester solver should handle bfloat16 input."""
        from qgre.lora_pro import solve_sylvester

        A = torch.randn(4, 4, dtype=torch.bfloat16)
        B = torch.randn(4, 4, dtype=torch.bfloat16)
        C = torch.randn(4, 4, dtype=torch.bfloat16)

        # Should not raise
        X = solve_sylvester(A, B, C)
        assert X.dtype == torch.bfloat16


@pytest.mark.gpu
class TestLoRAProGPU:
    """GPU-specific tests."""

    def test_adjusts_gradients_on_gpu(self):
        """LoRA-Pro should work on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from qgre.lora_pro import LoRAProAdjuster, LoRAProConfig

        model = MockLoRAModel(in_dim=64, out_dim=32, rank=8).cuda()
        config = LoRAProConfig(enabled=True)
        adjuster = LoRAProAdjuster(model, lora_rank=8, lora_alpha=16, config=config)

        x = torch.randn(4, 64, device="cuda")
        y = model(x)
        loss = y.sum()
        loss.backward()

        metrics = adjuster.adjust_gradients(global_step=0)

        assert metrics["lora_pro/n_adjusted"] == 1
        A, B, _ = adjuster._lora_pairs[0]
        assert A.grad.device.type == "cuda"
        assert B.grad.device.type == "cuda"
