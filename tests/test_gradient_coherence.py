"""Tests for temporal gradient cosine and turbulence detection."""

import torch
from torch import nn

from qgre.gradient_coherence import (
    TurbulenceDetector,
    compute_gradient_coherence,
    reset_gradient_cache,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(8, 4)
        self.linear2 = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear2(self.linear1(x))


def _backward_step(model: nn.Module, x: torch.Tensor) -> None:
    """Run a forward + backward pass to populate .grad on each parameter."""
    model.zero_grad()
    out = model(x)
    loss = out.sum()
    loss.backward()


def test_temporal_cosine_first_step_is_zero():
    """First step has no cached gradient → temporal cosine should be 0."""
    reset_gradient_cache()
    model = TinyModel()
    _backward_step(model, torch.randn(2, 8))
    stats = compute_gradient_coherence(model)
    assert stats["temporal_cosine"] == 0.0
    assert stats["n_comparisons"] == 0


def test_temporal_cosine_identical_gradients_is_one():
    """Same input twice → identical gradients → temporal cosine ≈ 1.0."""
    reset_gradient_cache()
    model = TinyModel()
    x = torch.randn(2, 8)
    _backward_step(model, x)
    compute_gradient_coherence(model)  # caches step 1

    _backward_step(model, x)  # identical input → identical gradients
    stats = compute_gradient_coherence(model)  # compares to cached step 1

    assert stats["temporal_cosine"] > 0.95, (
        f"Identical inputs should give temporal cosine ~1.0, got {stats['temporal_cosine']}"
    )
    assert stats["n_comparisons"] > 0


def test_temporal_cosine_different_gradients_differs_from_identical():
    """Different inputs should give lower temporal cosine than identical inputs."""
    reset_gradient_cache()
    torch.manual_seed(0)
    model = TinyModel()

    x1 = torch.randn(2, 8)
    _backward_step(model, x1)
    compute_gradient_coherence(model)

    x2 = torch.randn(2, 8) * 10  # different input
    _backward_step(model, x2)
    stats_different = compute_gradient_coherence(model)

    # Now identical input twice
    reset_gradient_cache()
    _backward_step(model, x1)
    compute_gradient_coherence(model)
    _backward_step(model, x1)
    stats_identical = compute_gradient_coherence(model)

    assert stats_identical["temporal_cosine"] > stats_different["temporal_cosine"], (
        f"Identical inputs ({stats_identical['temporal_cosine']:.3f}) should give "
        f"higher cosine than different inputs ({stats_different['temporal_cosine']:.3f})"
    )


def test_spatial_cosine_is_near_zero_for_unrelated_params():
    """Adjacent unrelated parameters should have spatial cosine ≈ 0."""
    reset_gradient_cache()
    model = TinyModel()
    _backward_step(model, torch.randn(2, 8))
    stats = compute_gradient_coherence(model)
    # Spatial cosine between linear1.weight and linear2.weight
    assert abs(stats["spatial_cosine"]) < 0.5


def test_lora_weight_norm_computed():
    """LoRA weight norm should be computed when params have 'lora_A' or 'lora_B' in name."""
    reset_gradient_cache()

    # Build a model with '.lora_A.' and '.lora_B.' in parameter names
    # via nested modules (matching real PEFT naming convention)
    parent = nn.Module()
    parent.base = nn.Linear(8, 4)
    layer = nn.Module()
    layer.lora_A = nn.Linear(8, 4, bias=False)
    layer.lora_B = nn.Linear(4, 4, bias=False)
    parent.layer = layer

    def _forward(x: torch.Tensor) -> torch.Tensor:
        return parent.base(x)

    parent.forward = _forward  # type: ignore[assignment]

    # Verify naming matches pattern
    names = [n for n, _ in parent.named_parameters()]
    assert any(".lora_A." in n for n in names), f"Expected .lora_A. in names: {names}"

    parent.zero_grad()
    parent.base(torch.randn(2, 8)).sum().backward()
    stats = compute_gradient_coherence(parent)
    assert stats["lora_weight_norm"] > 0


def test_reset_gradient_cache_clears_state():
    """reset_gradient_cache should clear cached gradients."""
    reset_gradient_cache()
    model = TinyModel()
    _backward_step(model, torch.randn(2, 8))
    compute_gradient_coherence(model)  # populates cache

    reset_gradient_cache()

    _backward_step(model, torch.randn(2, 8))
    stats = compute_gradient_coherence(model)
    # After reset, first step has no comparison → 0 temporal cosine
    assert stats["temporal_cosine"] == 0.0
    assert stats["n_comparisons"] == 0


def test_turbulence_detector_calibration_and_transitions():
    """Detector should calibrate, then transition based on cosine values."""
    detector = TurbulenceDetector(
        calibration_steps=3,
        cosine_threshold_low=0.05,
        cosine_threshold_high=0.20,
        transition_window=2,
    )

    # Calibration phase
    assert detector.update(0, {"temporal_cosine": 0.15}) == "CALIBRATING"
    assert detector.update(1, {"temporal_cosine": 0.12}) == "CALIBRATING"
    assert detector.update(2, {"temporal_cosine": 0.18}) == "LAMINAR"  # calibration complete

    # Stay laminar with high cosine
    assert detector.update(3, {"temporal_cosine": 0.10}) == "LAMINAR"

    # Drop below threshold → transitional
    assert detector.update(4, {"temporal_cosine": 0.01}) == "TRANSITIONAL"

    # Second drop → turbulent (transition_window=2)
    assert detector.update(5, {"temporal_cosine": 0.01}) == "TURBULENT"

    # Recovery above threshold
    assert detector.update(6, {"temporal_cosine": 0.25}) == "LAMINAR"


def test_backward_compat_mean_cosine_alias():
    """mean_cosine should be an alias for temporal_cosine."""
    reset_gradient_cache()
    model = TinyModel()
    x = torch.randn(2, 8)
    _backward_step(model, x)
    compute_gradient_coherence(model)
    _backward_step(model, x)
    stats = compute_gradient_coherence(model)
    assert stats["mean_cosine"] == stats["temporal_cosine"]


class ModelWithModulesToSave(nn.Module):
    """Mimics PEFT ModulesToSaveWrapper naming: 'modules_to_save' in param path."""

    def __init__(self, vocab_size: int = 64, hidden: int = 8):
        super().__init__()
        self.lora_layer = nn.Linear(hidden, hidden, bias=False)
        # Simulate PEFT's ModulesToSaveWrapper — the key is that
        # "modules_to_save" appears in the parameter name path.
        self.modules_to_save = nn.ModuleDict({"default": nn.Linear(hidden, vocab_size, bias=False)})

    def forward(self, x):
        return self.modules_to_save["default"](self.lora_layer(x))


def test_modules_to_save_included_in_norms():
    """modules_to_save grads should appear in layer_norms (via scalar cache)."""
    reset_gradient_cache()
    model = ModelWithModulesToSave()
    model.zero_grad()
    model(torch.randn(2, 8)).sum().backward()
    stats = compute_gradient_coherence(model)

    # Model has 2 params with grads: lora_layer.weight + modules_to_save.default.weight
    # Both should contribute to layer_norms
    assert len(stats["per_layer_norms"]) >= 2, (
        f"Expected >=2 norms (lora + mts), got {len(stats['per_layer_norms'])}"
    )


def test_modules_to_save_temporal_cosine():
    """modules_to_save grads should contribute to temporal cosine across steps."""
    reset_gradient_cache()
    model = ModelWithModulesToSave()
    x = torch.randn(2, 8)

    # Step 1: populate cache
    model.zero_grad()
    model(x).sum().backward()
    compute_gradient_coherence(model)

    # Step 2: identical input → identical grads → high cosine
    model.zero_grad()
    model(x).sum().backward()
    stats = compute_gradient_coherence(model)

    assert stats["n_comparisons"] >= 2, (
        f"Expected >=2 temporal comparisons (lora + mts), got {stats['n_comparisons']}"
    )
    assert stats["temporal_cosine"] > 0.9, (
        f"Identical inputs should give high temporal cosine, got {stats['temporal_cosine']}"
    )


def test_modules_to_save_no_gpu_allocation():
    """modules_to_save path must not store float32 flattened grads on GPU."""
    reset_gradient_cache()
    model = ModelWithModulesToSave()
    model.zero_grad()
    model(torch.randn(2, 8)).sum().backward()
    stats = compute_gradient_coherence(model)

    # Verify the function returned valid stats (didn't crash)
    assert stats["n_layers"] >= 1

    # Check that _prev_grads values are all on CPU
    from qgre.gradient_coherence import _prev_grads

    for name, tensor in _prev_grads.items():
        assert tensor.device.type == "cpu", f"_prev_grads[{name}] on {tensor.device}, expected cpu"
