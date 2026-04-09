"""Gradient coherence analysis for detecting laminar→turbulent transitions.

Two measurements:

1. **Temporal cosine** (the convergence signal):
   Same parameter across consecutive steps. High temporal cosine = gradients
   pointing in a consistent direction = model converging on something.
   Low temporal cosine = each step pulls randomly = exploration or instability.
   This is the metric the literature uses for pre-grokking detection.

2. **Spatial cosine** (the layer-disagreement signal):
   Adjacent parameters within the same step. High spatial cosine = layers
   pulling together. Low = layers pulling in different directions.
   Mathematically expected to be ~0 for unrelated high-dim vectors
   (concentration of measure), so only deviations from 0 are informative.

The TurbulenceDetector operates on temporal cosine (the convergence signal).
Spatial cosine is logged as a secondary diagnostic.

Memory cost: one cached copy of LoRA gradients (~100 MB for rank 32).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


# Cache for previous step's gradients, keyed by parameter name.
# Module-level so it persists across calls within the same training run.
_prev_grads: dict[str, torch.Tensor] = {}


def compute_gradient_coherence(model: torch.nn.Module) -> dict[str, Any]:
    """Compute gradient coherence metrics: temporal + spatial + norms.

    Call AFTER backward + gradient clipping, BEFORE optimizer.step().

    Args:
        model: The model with gradients computed.

    Returns:
        Dictionary with:
        - temporal_cosine: mean cosine between this step's and previous step's
          gradients for each parameter (the convergence signal).
        - spatial_cosine: mean cosine between adjacent parameters within this
          step (layer-disagreement signal, expected ~0 for unrelated params).
        - mean_cosine: alias for temporal_cosine (backward compat with trainer).
        - min_cosine / max_cosine: extremes of temporal cosine across params.
        - norm_ratio: ratio of max to min gradient norm across parameters.
        - mean_grad_norm: average gradient L2 norm across parameters.
        - lora_weight_norm: total L2 norm of LoRA adapter weights (grokking signal —
          decreasing weight norm during flat loss predicts breakthrough).
        - per_layer_norms: list of gradient norms per parameter.
        - per_layer_cosines: list of temporal cosines per parameter.
        - n_layers, n_comparisons, nonzero_norms, n_weights, n_biases: diagnostics.
    """
    global _prev_grads  # noqa: PLW0603

    # Collect current gradients
    current_grads: dict[str, torch.Tensor] = {}
    weight_grads: list[torch.Tensor] = []
    all_grads: list[torch.Tensor] = []
    layer_names: list[str] = []
    lora_weight_norm_sq = 0.0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Track LoRA weight norm (grokking signal)
        if ".lora_A." in name or ".lora_B." in name:
            lora_weight_norm_sq += param.data.float().norm().item() ** 2
        if param.grad is None:
            continue
        grad_flat = param.grad.flatten().float()
        current_grads[name] = grad_flat
        all_grads.append(grad_flat)
        layer_names.append(name)
        if param.dim() >= 2:
            weight_grads.append(grad_flat)

    n_biases = len(all_grads) - len(weight_grads)

    if len(all_grads) < 1:
        # Do NOT update _prev_grads — leave the previous valid cache intact
        # so the next call with gradients can compute meaningful temporal cosine.
        return _empty_stats()

    # ── Gradient norms ──
    layer_norms = [g.norm().item() for g in all_grads]
    mean_norm = float(np.mean(layer_norms))
    max_norm = max(layer_norms)
    min_norm = min(layer_norms) if min(layer_norms) > 0 else 1e-10
    norm_ratio = min(max_norm / min_norm, 1e6)  # Clamp to 1M to avoid Inf from frozen params

    # ── Temporal cosine (convergence signal) ──
    # Compare each parameter's gradient to its cached value from previous step.
    temporal_cosines: list[float] = []
    for name, grad in current_grads.items():
        if name not in _prev_grads:
            continue
        prev = _prev_grads[name]
        if prev.shape != grad.shape:
            continue  # Shape changed (e.g., model restructured) — skip
        n1 = grad.norm().item()
        n2 = prev.norm().item()
        if n1 > 1e-10 and n2 > 1e-10:
            cos = torch.dot(grad, prev).item() / (n1 * n2)
            temporal_cosines.append(cos)

    temporal_mean = float(np.mean(temporal_cosines)) if temporal_cosines else 0.0
    temporal_min = float(min(temporal_cosines)) if temporal_cosines else 0.0
    temporal_max = float(max(temporal_cosines)) if temporal_cosines else 0.0

    # ── Spatial cosine (layer-disagreement signal) ──
    # Compare adjacent weight-gradient pairs within this step.
    # Expected ~0 for unrelated high-dim vectors; deviations are signal.
    spatial_cosines: list[float] = []
    comparison_grads = weight_grads if len(weight_grads) >= 2 else all_grads
    for i in range(len(comparison_grads) - 1):
        g1 = comparison_grads[i]
        g2 = comparison_grads[i + 1]
        min_len = min(g1.numel(), g2.numel())
        if min_len == 0:
            continue
        g1s = g1[:min_len]
        g2s = g2[:min_len]
        n1 = g1s.norm().item()
        n2 = g2s.norm().item()
        if n1 > 1e-10 and n2 > 1e-10:
            cos = torch.dot(g1s, g2s).item() / (n1 * n2)
            spatial_cosines.append(cos)

    spatial_mean = float(np.mean(spatial_cosines)) if spatial_cosines else 0.0

    # Cache current gradients for next step's temporal comparison.
    # Clone to avoid holding references to the autograd graph.
    _prev_grads = {name: grad.detach().clone() for name, grad in current_grads.items()}

    return {
        # Primary signal (temporal — what the detector uses)
        "temporal_cosine": temporal_mean,
        "mean_cosine": temporal_mean,  # backward compat alias
        "min_cosine": temporal_min,
        "max_cosine": temporal_max,
        # Secondary signal (spatial — layer disagreement)
        "spatial_cosine": spatial_mean,
        # Norms
        "norm_ratio": norm_ratio,
        "mean_grad_norm": mean_norm,
        "lora_weight_norm": lora_weight_norm_sq**0.5,
        # Per-parameter detail
        "per_layer_norms": layer_norms,
        "per_layer_cosines": [float(c) for c in temporal_cosines],
        # Diagnostics
        "n_layers": len(all_grads),
        "n_comparisons": len(temporal_cosines),
        "nonzero_norms": sum(1 for n in layer_norms if n > 1e-10),
        "n_weights": len(weight_grads),
        "n_biases": n_biases,
    }


def _empty_stats() -> dict[str, Any]:
    """Return zero-valued stats when there's nothing to compute."""
    return {
        "temporal_cosine": 0.0,
        "mean_cosine": 0.0,
        "min_cosine": 0.0,
        "max_cosine": 0.0,
        "spatial_cosine": 0.0,
        "norm_ratio": 1.0,
        "mean_grad_norm": 0.0,
        "lora_weight_norm": 0.0,
        "per_layer_norms": [],
        "per_layer_cosines": [],
        "n_layers": 0,
        "n_comparisons": 0,
        "nonzero_norms": 0,
        "n_weights": 0,
        "n_biases": 0,
    }


def reset_gradient_cache() -> None:
    """Clear the cached previous-step gradients.

    Call on checkpoint resume or engine recreation to avoid comparing
    against stale gradients from a different training state.
    """
    global _prev_grads  # noqa: PLW0603
    _prev_grads = {}


class TurbulenceDetector:
    """Stateful detector for laminar→turbulent transitions.

    Operates on TEMPORAL cosine (same parameter across steps).
    Thresholds calibrated for RL + LoRA training:
    - During active dropout (>10%): temporal cosine expected 0.02-0.15
    - Post-dropout: temporal cosine expected 0.05-0.40
    - SFT baseline: temporal cosine expected 0.3-0.8
    """

    def __init__(
        self,
        calibration_steps: int = 30,
        cosine_threshold_low: float = 0.02,
        cosine_threshold_high: float = 0.10,
        transition_window: int = 5,
    ):
        """
        Args:
            calibration_steps: Number of initial steps to calibrate baseline.
                First step has no cached gradient, so effective calibration
                starts at step 2.
            cosine_threshold_low: Below this = turbulent (gradients uncorrelated).
                0.02 for RL+LoRA with dropout; raise to 0.05 post-dropout.
            cosine_threshold_high: Above this = laminar (gradients aligned).
                0.10 for RL+LoRA with dropout; raise to 0.20 post-dropout.
            transition_window: Consecutive turbulent steps to confirm transition.
        """
        self.calibration_steps = calibration_steps
        self.cosine_threshold_low = cosine_threshold_low
        self.cosine_threshold_high = cosine_threshold_high
        self.transition_window = transition_window

        self.state = "CALIBRATING"
        self.calibration_data: list[float] = []
        self.turbulent_count = 0
        self.transition_step: int | None = None

    def update(self, step: int, coherence_stats: dict[str, float]) -> str:
        """Update detector with new coherence measurement.

        Args:
            step: Current training step
            coherence_stats: Output from compute_gradient_coherence()

        Returns:
            Current state: CALIBRATING, LAMINAR, TRANSITIONAL, TURBULENT
        """
        # Use temporal_cosine if available, fall back to mean_cosine for compat
        mean_cosine = coherence_stats.get(
            "temporal_cosine", coherence_stats.get("mean_cosine", 0.0)
        )

        if self.state == "CALIBRATING":
            self.calibration_data.append(mean_cosine)
            if len(self.calibration_data) >= self.calibration_steps:
                baseline_cosine = float(np.mean(self.calibration_data))
                print(
                    f"Gradient coherence calibrated: baseline temporal_cosine = {baseline_cosine:.4f}"
                )
                # Adapt thresholds based on calibration baseline. For dropout-heavy
                # regimes where baseline is near-zero, use tighter thresholds.
                if baseline_cosine > 0.01:
                    self.cosine_threshold_low = max(0.01, baseline_cosine * 0.3)
                    self.cosine_threshold_high = max(0.03, baseline_cosine * 1.5)
                else:
                    # Near-zero baseline (likely heavy LoRA dropout). Use ultra-tight
                    # thresholds so the detector doesn't immediately flag TURBULENT.
                    self.cosine_threshold_low = 0.001
                    self.cosine_threshold_high = 0.01
                self.state = "LAMINAR"

        elif self.state == "LAMINAR":
            if mean_cosine < self.cosine_threshold_low:
                self.turbulent_count += 1
                if self.turbulent_count >= self.transition_window:
                    self.state = "TURBULENT"
                    self.transition_step = step
                else:
                    self.state = "TRANSITIONAL"
            else:
                self.turbulent_count = 0

        elif self.state == "TRANSITIONAL":
            if mean_cosine < self.cosine_threshold_low:
                self.turbulent_count += 1
                if self.turbulent_count >= self.transition_window:
                    self.state = "TURBULENT"
                    self.transition_step = step
            elif mean_cosine > self.cosine_threshold_high:
                self.state = "LAMINAR"
                self.turbulent_count = 0
            else:
                # Hysteresis zone: cosine is between thresholds. Reset the
                # turbulent counter so a single recovery into the middle band
                # prevents false-positive turbulence from accumulating across
                # interleaved signals.
                self.turbulent_count = 0

        elif self.state == "TURBULENT":
            if mean_cosine > self.cosine_threshold_high:
                self.state = "LAMINAR"
                self.turbulent_count = 0

        return self.state
