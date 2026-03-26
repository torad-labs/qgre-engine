"""LoRA dropout during generation — surface base model knowledge for exploration.

During RL rollout generation, apply Bernoulli dropout to LoRA A matrices.
This partially reverts the model to base model behavior, letting suppressed
knowledge (e.g., gravity in vertical spring problems) surface through the
hidden representations. Different dropout masks = different internal reasoning
paths = diverse completions.

Based on: NoisyGRPO (NeurIPS 2025), NoisyRollout (NeurIPS 2025),
"Noise Injection Reveals Hidden Capabilities" (2024).

Only LoRA A (input projection) is dropped — LoRA B (output projection) is
preserved to maintain output space structure and formatting.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


def apply_lora_dropout(model: nn.Module, dropout_rate: float) -> Callable[[], None]:
    """Apply Bernoulli dropout to LoRA A matrices. Returns a restore function.

    Args:
        model: The model with LoRA adapters (Unsloth/PEFT format)
        dropout_rate: Probability of zeroing each element in LoRA A weights.
            0.0 = no dropout (passthrough), 0.15 = recommended starting point.

    Returns:
        restore(): Call this after generation to restore original weights.
    """
    if dropout_rate <= 0.0:
        return lambda: None  # No-op

    saved: dict[str, torch.Tensor] = {}

    for name, param in model.named_parameters():
        # Match LoRA A matrices (input projection) — not B (output projection)
        if "lora_A" in name and param.requires_grad:
            saved[name] = param.data.clone()
            mask = torch.bernoulli(
                torch.ones_like(param.data) * (1.0 - dropout_rate)
            )
            # Scale by 1/(1-p) to maintain expected magnitude (inverted dropout)
            param.data.mul_(mask / (1.0 - dropout_rate))

    def restore():
        for name, original in saved.items():
            # Walk the model to find the parameter by name
            parts = name.split(".")
            obj = model
            for part in parts[:-1]:
                obj = getattr(obj, part)
            getattr(obj, parts[-1]).data.copy_(original)

    return restore


def compute_dropout_rate(
    initial_rate: float,
    anneal_steps: int,
    current_step: int,
) -> float:
    """Compute annealed dropout rate. Linear decay to 0.

    Args:
        initial_rate: Starting dropout rate (e.g., 0.15)
        anneal_steps: Steps over which to anneal to 0 (e.g., 500)
        current_step: Current training step

    Returns:
        Current dropout rate (0.0 when current_step >= anneal_steps)
    """
    if anneal_steps <= 0 or initial_rate <= 0.0:
        return 0.0
    progress = min(current_step / anneal_steps, 1.0)
    return initial_rate * (1.0 - progress)
