"""LoRA-Pro: Optimal gradient adjustment for low-rank adapters.

Implements the LoRA-Pro algorithm from ICLR 2025:
"LoRA-Pro: Are Low-Rank Adapters Properly Optimized?"
Paper: https://arxiv.org/abs/2407.18242

Core insight: LoRA optimization is equivalent to full fine-tuning with a low-rank
gradient. The low-rank gradient can be expressed as g̃ = s*B*∇A + s*∇B*A where s
is the scaling factor. LoRA-Pro adjusts ∇A and ∇B so that g̃ better approximates
the full fine-tuning gradient, closing the performance gap.

Memory overhead: ~4GB for storing momentum terms on equivalent gradients.
Training time: Same as standard LoRA (closed-form solution, no extra forward pass).
Mergeability: Unchanged — still standard LoRA weights after training.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch


if TYPE_CHECKING:
    from torch import nn


logger = logging.getLogger(__name__)


def solve_sylvester(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    """Solve the Sylvester equation AX + XB = C for X.

    Uses eigendecomposition method from:
    https://stackoverflow.com/questions/73713072/solving-sylvester-equations-in-pytorch

    Args:
        A: Matrix of shape [n, n]
        B: Matrix of shape [m, m]
        C: Matrix of shape [n, m]

    Returns:
        X: Solution matrix of shape [n, m]
    """
    orig_dtype = A.dtype
    # Eigendecomposition requires float32 precision
    if A.dtype in (torch.bfloat16, torch.float16):
        A = A.to(torch.float32)
        B = B.to(torch.float32)
        C = C.to(torch.float32)

    # Sylvester: AX + XB = C  =>  AX - X(-B) = C
    B = -B
    m = B.shape[-1]
    n = A.shape[-1]

    # Eigendecomposition
    R, U = torch.linalg.eig(A)
    S, V = torch.linalg.eig(B)

    # Solve in eigenspace
    F = torch.linalg.solve(U, (C + 0j) @ V)
    W = R[..., :, None] - S[..., None, :]
    Y = F / W

    # Transform back
    X = U[..., :n, :n] @ Y[..., :n, :m] @ torch.linalg.inv(V)[..., :m, :m]

    # Return real part if inputs were real (check ALL elements, not just first)
    result = X.real if all(torch.isreal(x).all() for x in [A, B, C]) else X
    return result.to(orig_dtype)


@dataclass
class LoRAProState:
    """Per-layer state for LoRA-Pro momentum tracking."""

    exp_avg: torch.Tensor | None = None  # EMA of equivalent gradient
    exp_avg_sq: torch.Tensor | None = None  # EMA of squared equivalent gradient


@dataclass
class LoRAProConfig:
    """Configuration for LoRA-Pro gradient adjustment."""

    enabled: bool = False
    beta1: float = 0.9  # Adam beta1 for equivalent gradient momentum
    beta2: float = 0.999  # Adam beta2 for equivalent gradient momentum
    eps: float = 1e-8  # Epsilon for numerical stability
    delta: float = 1e-8  # Regularization for matrix inversion
    use_rslora: bool = True  # RSLoRA scaling (alpha/sqrt(r)) vs standard (alpha/r)
    grad_scale: float = 1.0  # Post-adjustment gradient multiplier (counteracts 1/s² attenuation)
    grad_floor: float = 1e-7  # Minimum gradient norm (prevents numerical collapse)


class LoRAProAdjuster:
    """Adjusts LoRA gradients to better approximate full fine-tuning.

    Usage:
        adjuster = LoRAProAdjuster(model, config)

        for step in training_loop:
            loss.backward()
            adjuster.adjust_gradients(step)  # <-- Insert before optimizer.step()
            optimizer.step()

    The adjuster identifies LoRA A and B matrices by name pattern (lora_A, lora_B)
    and applies the closed-form optimal gradient adjustment from the LoRA-Pro paper.
    """

    def __init__(
        self,
        model: nn.Module,
        lora_rank: int,
        lora_alpha: int,
        config: LoRAProConfig | None = None,
    ):
        """Initialize LoRA-Pro adjuster.

        Args:
            model: The PEFT model with LoRA layers.
            lora_rank: LoRA rank (r).
            lora_alpha: LoRA alpha scaling parameter.
            config: LoRA-Pro configuration. Defaults to enabled with RSLoRA scaling.
        """
        self.model = model
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.config = config or LoRAProConfig(enabled=True)

        # Compute scaling factor
        if self.config.use_rslora:
            self.scaling_factor = lora_alpha / math.sqrt(lora_rank)
        else:
            self.scaling_factor = lora_alpha / lora_rank

        # Find and pair LoRA A/B matrices
        self._lora_pairs: list[tuple[nn.Parameter, nn.Parameter, str]] = []
        self._states: dict[str, LoRAProState] = {}
        self._sylvester_failures = 0
        self._discover_lora_pairs()

        logger.info(
            f"LoRA-Pro initialized: {len(self._lora_pairs)} layer pairs, "
            f"rank={lora_rank}, alpha={lora_alpha}, scaling={self.scaling_factor:.4f}, "
            f"rslora={self.config.use_rslora}"
        )

    def _discover_lora_pairs(self) -> None:
        """Find all LoRA A/B matrix pairs in the model."""
        lora_a_params: dict[str, nn.Parameter] = {}
        lora_b_params: dict[str, nn.Parameter] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # PEFT naming: base_model.model.layers.X.self_attn.q_proj.lora_A.default.weight
            if "lora_A" in name and "weight" in name:
                # Extract layer identifier (everything before lora_A)
                layer_id = name.split("lora_A")[0]
                lora_a_params[layer_id] = param
            elif "lora_B" in name and "weight" in name:
                layer_id = name.split("lora_B")[0]
                lora_b_params[layer_id] = param

        # Match A and B matrices
        for layer_id in lora_a_params:
            if layer_id in lora_b_params:
                A = lora_a_params[layer_id]
                B = lora_b_params[layer_id]
                self._lora_pairs.append((A, B, layer_id))
                self._states[layer_id] = LoRAProState()

        if not self._lora_pairs:
            logger.warning(
                "LoRA-Pro: No LoRA A/B pairs found. "
                "Ensure model has PEFT LoRA layers with lora_A/lora_B naming."
            )

    @torch.no_grad()
    def adjust_gradients(self, global_step: int) -> dict[str, float]:
        """Apply LoRA-Pro gradient adjustment to all LoRA layers.

        Call this AFTER backward() but BEFORE optimizer.step().

        Args:
            global_step: Current training step (0-indexed).

        Returns:
            Metrics dict with adjustment statistics.
        """
        if not self.config.enabled:
            return {}

        if not self._lora_pairs:
            return {}

        metrics: dict[str, float] = {}
        total_grad_norm_before = 0.0
        total_grad_norm_after = 0.0
        n_adjusted = 0

        s = self.scaling_factor
        delta = self.config.delta
        beta1 = self.config.beta1
        beta2 = self.config.beta2
        eps = self.config.eps

        for A, B, layer_id in self._lora_pairs:
            if A.grad is None or B.grad is None:
                continue

            grad_A_orig = A.grad.clone()
            grad_B_orig = B.grad.clone()
            state = self._states[layer_id]

            # Track norm before adjustment
            total_grad_norm_before += grad_A_orig.norm().item() ** 2
            total_grad_norm_before += grad_B_orig.norm().item() ** 2

            # A: [rank, in_dim], B: [out_dim, rank]
            # AA_T: [rank, rank], B_TB: [rank, rank]
            AA_T = A @ A.T
            B_TB = B.T @ B

            # Compute pseudo-inverses with regularization
            eye_r = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
            AA_T_inv = torch.linalg.pinv(AA_T + delta * eye_r).to(A.dtype)
            B_TB_inv = torch.linalg.pinv(B_TB + delta * eye_r).to(A.dtype)

            if global_step == 0:
                # Cold start: simplified adjustment (no Sylvester)
                grad_A = grad_A_orig
                grad_B = (1 / s**2) * grad_B_orig @ AA_T_inv
            else:
                # Full adjustment with Sylvester equation
                grad_A = (1 / s**2) * B_TB_inv @ grad_A_orig
                eye_out = torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
                proj = eye_out - B @ B_TB_inv @ B.T
                grad_B = (1 / s**2) * proj @ grad_B_orig @ AA_T_inv

            # Compute equivalent gradient for Adam momentum
            equiv_grad = s * B @ grad_A + s * grad_B @ A

            # Initialize or update momentum
            if state.exp_avg is None:
                state.exp_avg = (1 - beta1) * equiv_grad
                state.exp_avg_sq = (1 - beta2) * (equiv_grad * equiv_grad)
            else:
                state.exp_avg.lerp_(equiv_grad, 1 - beta1)
                state.exp_avg_sq.mul_(beta2).addcmul_(equiv_grad, equiv_grad, value=1 - beta2)

            # Bias correction
            step = global_step + 1
            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            bias_correction2_sqrt = math.sqrt(bias_correction2)

            # Adam-style normalized gradient
            denom = (state.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            g = (state.exp_avg / bias_correction1) / denom
            g = g.to(B.dtype)

            # Back-project to A and B gradients
            grad_A_adj = s * B.T @ g
            grad_B_adj = s * g @ A.T

            if global_step == 0:
                # Cold start: use back-projected gradients directly
                final_grad_A = grad_A_adj
                final_grad_B = (1 / s**2) * grad_B_adj @ AA_T_inv
            else:
                # Full adjustment with Sylvester equation
                C = -(1 / s**2) * B_TB_inv @ grad_A_adj @ A.T
                try:
                    X = solve_sylvester(B_TB, AA_T, C)
                except Exception as e:
                    self._sylvester_failures += 1
                    import warnings
                    warnings.warn(
                        f"LoRA-Pro: Sylvester solve failed for {layer_id}: {e}. "
                        f"Using simplified adjustment (zero matrix). Total failures: {self._sylvester_failures}",
                        stacklevel=2,
                    )
                    X = torch.zeros_like(C)

                final_grad_A = (1 / s**2) * B_TB_inv @ grad_A_adj + X @ A
                eye_out = torch.eye(B.shape[0], device=B.device, dtype=B.dtype)
                proj = eye_out - B @ B_TB_inv @ B.T
                final_grad_B = (1 / s**2) * proj @ grad_B_adj @ AA_T_inv - B @ X

            # Apply gradient scaling to counteract 1/s² attenuation
            if self.config.grad_scale != 1.0:
                final_grad_A = final_grad_A * self.config.grad_scale
                final_grad_B = final_grad_B * self.config.grad_scale

            # Apply gradient floor to prevent numerical collapse
            combined_norm = math.sqrt(
                final_grad_A.norm().item() ** 2 + final_grad_B.norm().item() ** 2
            )
            if combined_norm < self.config.grad_floor and combined_norm > 0:
                scale_up = self.config.grad_floor / combined_norm
                final_grad_A = final_grad_A * scale_up
                final_grad_B = final_grad_B * scale_up
                logger.debug(
                    f"LoRA-Pro: Gradient floor triggered for {layer_id}, "
                    f"scaled {combined_norm:.2e} -> {self.config.grad_floor:.2e}"
                )

            # Apply adjusted gradients
            A.grad.copy_(final_grad_A)
            B.grad.copy_(final_grad_B)

            # Track norm after adjustment
            total_grad_norm_after += final_grad_A.norm().item() ** 2
            total_grad_norm_after += final_grad_B.norm().item() ** 2
            n_adjusted += 1

        if n_adjusted > 0:
            metrics["lora_pro/grad_norm_before"] = math.sqrt(total_grad_norm_before)
            metrics["lora_pro/grad_norm_after"] = math.sqrt(total_grad_norm_after)
            metrics["lora_pro/n_adjusted"] = n_adjusted
            ratio = math.sqrt(total_grad_norm_after) / (math.sqrt(total_grad_norm_before) + 1e-8)
            metrics["lora_pro/norm_ratio"] = ratio

        return metrics

    def get_equivalent_gradient_stats(self) -> dict[str, float]:
        """Get statistics about equivalent gradient magnitudes."""
        stats = {}
        for layer_id, state in self._states.items():
            if state.exp_avg is not None:
                stats[f"lora_pro/{layer_id}/exp_avg_norm"] = state.exp_avg.norm().item()
            if state.exp_avg_sq is not None:
                stats[f"lora_pro/{layer_id}/exp_avg_sq_mean"] = state.exp_avg_sq.mean().item()
        return stats

    def state_dict(self) -> dict:
        """Return state for checkpointing."""
        return {
            layer_id: {
                "exp_avg": state.exp_avg.cpu() if state.exp_avg is not None else None,
                "exp_avg_sq": state.exp_avg_sq.cpu() if state.exp_avg_sq is not None else None,
            }
            for layer_id, state in self._states.items()
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state from checkpoint."""
        for layer_id, saved_state in state_dict.items():
            if layer_id in self._states:
                state = self._states[layer_id]
                if saved_state["exp_avg"] is not None:
                    # Find device from corresponding parameter
                    device_found = False
                    for A, B, lid in self._lora_pairs:
                        if lid == layer_id:
                            state.exp_avg = saved_state["exp_avg"].to(A.device)
                            state.exp_avg_sq = saved_state["exp_avg_sq"].to(A.device)
                            device_found = True
                            break
                    if not device_found:
                        import warnings
                        warnings.warn(
                            f"LoRA-Pro: Could not find device for layer {layer_id}. "
                            "State may be on wrong device.",
                            stacklevel=2,
                        )


def compute_gradient_approximation_error(
    model: nn.Module,
    lora_rank: int,
    lora_alpha: int,
    use_rslora: bool = True,
) -> dict[str, float]:
    """Compute how well current LoRA gradients approximate full fine-tuning.

    This is a diagnostic tool to verify LoRA-Pro is working. Call AFTER backward()
    but BEFORE any gradient adjustment.

    Returns:
        Dict with approximation error metrics per layer.
    """
    if use_rslora:
        s = lora_alpha / math.sqrt(lora_rank)
    else:
        s = lora_alpha / lora_rank

    errors = {}

    for name, param in model.named_parameters():
        if "lora_A" not in name or "weight" not in name or param.grad is None:
            continue

        layer_id = name.split("lora_A")[0]
        A = param

        # Find corresponding B
        B = None
        for n, p in model.named_parameters():
            if layer_id in n and "lora_B" in n and "weight" in n:
                B = p
                break

        if B is None or B.grad is None:
            continue

        # Equivalent gradient: g̃ = s*B*∇A + s*∇B*A
        equiv_grad = s * B @ A.grad + s * B.grad @ A

        # The "ideal" gradient would have rank = min(out_dim, in_dim)
        # We can measure how much information is lost by comparing norms
        full_rank = min(equiv_grad.shape)
        current_rank = lora_rank

        # Compute singular values to see energy distribution
        try:
            _, singular_values, _ = torch.linalg.svd(equiv_grad.float())
            total_energy = (singular_values**2).sum()
            captured_energy = (singular_values[:current_rank] ** 2).sum()
            energy_ratio = (captured_energy / (total_energy + 1e-8)).item()
            errors[f"{layer_id}energy_captured"] = energy_ratio
        except Exception:
            pass

        errors[f"{layer_id}equiv_grad_norm"] = equiv_grad.norm().item()

    return errors
