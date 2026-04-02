"""Gradient coherence analysis for detecting laminar→turbulent transitions.

Measures:
- Inter-layer gradient alignment (cosine similarity)
- Batch gradient variance (chaos detector)
- Per-layer gradient norm ratios (instability index)

These metrics directly capture the "eddies forming" phenomenon:
- Laminar flow: gradients aligned, low variance, stable norms
- Turbulent flow: gradients misaligned, high variance, unstable norms
- Bifurcation: sudden transition between states

This approach measures what actually happens during Phase 7 collapse:
layers start pulling in different directions, energy dissipates into noise.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


def compute_gradient_coherence(model: torch.nn.Module) -> Dict[str, float]:
    """Compute gradient coherence metrics across model layers.

    Args:
        model: The model with gradients computed (after backward, before optimizer.step)

    Returns:
        Dictionary with:
        - mean_cosine: average cosine similarity between adjacent layers
        - min_cosine: minimum cosine (most misaligned pair)
        - max_cosine: maximum cosine (most aligned pair)
        - norm_ratio: ratio of max to min gradient norm across layers
        - mean_grad_norm: average gradient norm across layers
        - per_layer_norms: list of gradient norms per layer
        - per_layer_cosines: list of cosine similarities between adjacent layers
    """
    # Collect gradients per layer, grouped by type (weights vs biases)
    weight_grads = []  # Only weight matrices
    bias_grads = []    # Only bias vectors
    all_grads = []     # All parameters
    layer_names = []

    for name, param in model.named_parameters():
        if param.grad is not None and param.requires_grad:
            # Flatten gradient to 1D vector
            grad_flat = param.grad.flatten()
            all_grads.append(grad_flat)
            layer_names.append(name)

            # Separate weights (2D) from biases (1D) for like-to-like comparison
            if param.dim() >= 2:
                weight_grads.append(grad_flat)
            elif param.dim() == 1:
                bias_grads.append(grad_flat)

    if len(all_grads) < 2:
        # Not enough layers to compute coherence
        return {
            "mean_cosine": 0.0,
            "min_cosine": 0.0,
            "max_cosine": 0.0,
            "norm_ratio": 1.0,
            "mean_grad_norm": 0.0,
            "per_layer_norms": [],
            "per_layer_cosines": [],
        }

    # Compute gradient norms per layer
    layer_norms = [grad.norm().item() for grad in all_grads]
    mean_norm = np.mean(layer_norms)
    max_norm = max(layer_norms)
    min_norm = min(layer_norms) if min(layer_norms) > 0 else 1e-10
    norm_ratio = max_norm / min_norm

    # Strategy: Compare like-to-like (weights-to-weights) for meaningful cosine
    # Use weight gradients for comparison (they're the main learning signal)
    comparison_grads = weight_grads if len(weight_grads) >= 2 else all_grads

    layer_cosines = []
    for i in range(len(comparison_grads) - 1):
        grad1 = comparison_grads[i]
        grad2 = comparison_grads[i + 1]

        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        # Flatten to same size by taking min length and comparing prefixes
        # This allows comparison even with different sizes
        min_len = min(grad1.numel(), grad2.numel())
        if min_len == 0:
            continue

        g1 = grad1[:min_len].float()  # Cast to float32 for consistent computation
        g2 = grad2[:min_len].float()

        dot_product = torch.dot(g1, g2).item()
        norm1 = g1.norm().item()
        norm2 = g2.norm().item()

        if norm1 > 0 and norm2 > 0:
            cosine = dot_product / (norm1 * norm2)
        else:
            cosine = 0.0

        layer_cosines.append(cosine)

    # Aggregate statistics
    mean_cosine = np.mean(layer_cosines) if layer_cosines else 0.0
    min_cosine = min(layer_cosines) if layer_cosines else 0.0
    max_cosine = max(layer_cosines) if layer_cosines else 0.0

    return {
        "mean_cosine": mean_cosine,
        "min_cosine": min_cosine,
        "max_cosine": max_cosine,
        "norm_ratio": norm_ratio,
        "mean_grad_norm": mean_norm,
        "per_layer_norms": layer_norms,
        "per_layer_cosines": layer_cosines,
        "n_layers": len(all_grads),
        "n_comparisons": len(layer_cosines),
        "nonzero_norms": sum(1 for n in layer_norms if n > 1e-10),
        "n_weights": len(weight_grads),
        "n_biases": len(bias_grads),
    }


def compute_batch_gradient_variance(
    model: torch.nn.Module,
    loss_fn,
    batch_inputs: List[torch.Tensor],
    batch_targets: List[torch.Tensor],
) -> Dict[str, float]:
    """Compute gradient variance across batch samples.

    High variance = samples pulling in different directions = turbulence.

    Args:
        model: The model
        loss_fn: Loss function
        batch_inputs: List of input tensors (one per sample)
        batch_targets: List of target tensors (one per sample)

    Returns:
        Dictionary with:
        - mean_variance: average gradient variance across parameters
        - max_variance: maximum gradient variance
        - variance_to_norm_ratio: variance / mean norm (normalized chaos metric)
    """
    # Collect per-sample gradients
    sample_grads = []

    for input_tensor, target_tensor in zip(batch_inputs, batch_targets):
        model.zero_grad()
        output = model(input_tensor)
        loss = loss_fn(output, target_tensor)
        loss.backward()

        # Collect flattened gradient vector for this sample
        grad_vector = []
        for param in model.parameters():
            if param.grad is not None:
                grad_vector.append(param.grad.flatten())

        sample_grads.append(torch.cat(grad_vector))

    if len(sample_grads) < 2:
        return {
            "mean_variance": 0.0,
            "max_variance": 0.0,
            "variance_to_norm_ratio": 0.0,
        }

    # Stack into [n_samples, n_params] tensor
    grad_matrix = torch.stack(sample_grads)

    # Compute variance across samples for each parameter
    grad_variance = torch.var(grad_matrix, dim=0)  # [n_params]

    # Statistics
    mean_variance = grad_variance.mean().item()
    max_variance = grad_variance.max().item()

    # Normalize by gradient magnitude
    mean_norm = grad_matrix.abs().mean().item()
    variance_to_norm_ratio = mean_variance / max(mean_norm, 1e-10)

    return {
        "mean_variance": mean_variance,
        "max_variance": max_variance,
        "variance_to_norm_ratio": variance_to_norm_ratio,
    }


class TurbulenceDetector:
    """Stateful detector for laminar→turbulent transitions.

    Calibrates thresholds from initial training steps, then monitors
    for sudden drops in gradient coherence that signal turbulence.
    """

    def __init__(
        self,
        calibration_steps: int = 50,
        cosine_threshold_low: float = 0.3,
        cosine_threshold_high: float = 0.6,
        transition_window: int = 3,
    ):
        """
        Args:
            calibration_steps: Number of initial steps to calibrate baseline
            cosine_threshold_low: Below this = turbulent
            cosine_threshold_high: Above this = laminar
            transition_window: Number of consecutive turbulent steps to confirm transition
        """
        self.calibration_steps = calibration_steps
        self.cosine_threshold_low = cosine_threshold_low
        self.cosine_threshold_high = cosine_threshold_high
        self.transition_window = transition_window

        self.state = "CALIBRATING"  # CALIBRATING, LAMINAR, TRANSITIONAL, TURBULENT
        self.calibration_data = []
        self.turbulent_count = 0
        self.transition_step = None

    def update(self, step: int, coherence_stats: Dict[str, float]) -> str:
        """Update detector with new coherence measurement.

        Args:
            step: Current training step
            coherence_stats: Output from compute_gradient_coherence()

        Returns:
            Current state: CALIBRATING, LAMINAR, TRANSITIONAL, TURBULENT
        """
        mean_cosine = coherence_stats["mean_cosine"]

        if self.state == "CALIBRATING":
            self.calibration_data.append(mean_cosine)

            if len(self.calibration_data) >= self.calibration_steps:
                # Calibration complete
                baseline_cosine = np.mean(self.calibration_data)
                print(f"Gradient coherence calibrated: baseline mean_cosine = {baseline_cosine:.3f}")
                self.state = "LAMINAR"

        elif self.state == "LAMINAR":
            if mean_cosine < self.cosine_threshold_low:
                self.turbulent_count += 1
                if self.turbulent_count >= self.transition_window:
                    # Transition confirmed
                    self.state = "TURBULENT"
                    self.transition_step = step
                    print(f"⚠️  TURBULENCE DETECTED at step {step}: mean_cosine={mean_cosine:.3f}")
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
                    print(f"⚠️  TURBULENCE DETECTED at step {step}: mean_cosine={mean_cosine:.3f}")
            elif mean_cosine > self.cosine_threshold_high:
                # Recovered to laminar
                self.state = "LAMINAR"
                self.turbulent_count = 0
            # else: stay in transitional

        elif self.state == "TURBULENT":
            if mean_cosine > self.cosine_threshold_high:
                # Recovered from turbulence
                print(f"✓ Turbulence resolved at step {step}: mean_cosine={mean_cosine:.3f}")
                self.state = "LAMINAR"
                self.turbulent_count = 0

        return self.state
