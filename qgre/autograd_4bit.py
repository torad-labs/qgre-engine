"""Custom autograd for Linear4bit: re-dequantize in backward to save ~1.6 GB.

bitsandbytes' default saves dequantized bf16 weights in autograd for each layer.
28 layers x ~58 MiB = ~1.6 GB wasted on frozen weights that can be re-dequantized.

This module patches Linear4bit.forward with a custom autograd.Function that:
1. Dequantizes 4-bit -> bf16 in forward (same as stock)
2. Does NOT save the dequantized weight for backward
3. Re-dequantizes from the tiny 4-bit data during backward
4. Does NOT save input (gradient checkpointing handles it at layer level)

Cost: ~0.2ms per layer re-dequantization in backward.
Savings: ~1.6 GB GPU VRAM.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import torch


if TYPE_CHECKING:
    from collections.abc import Callable

    import bitsandbytes as bnb

logger = logging.getLogger(__name__)


class _Redequantize4bit(torch.autograd.Function):
    """Linear4bit forward with re-dequantization in backward.

    Saves only a reference to the existing quantized weight data (zero copy cost)
    and re-dequantizes during backward instead of holding the bf16 copy.
    Does NOT save input -- gradient checkpointing handles that at the layer level.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        x: torch.Tensor,
        weight_data: torch.Tensor,
        quant_state: Any,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        import bitsandbytes.functional as bnb_fn

        # Dequantize 4-bit -> bf16 (same as stock bitsandbytes)
        weight_bf16 = bnb_fn.dequantize_4bit(weight_data, quant_state)
        output = torch.nn.functional.linear(x, weight_bf16, bias)

        # Save ONLY the quantized weight reference (already in model memory, zero extra cost).
        # NOT the dequantized bf16 (~58 MiB per layer x 28 = 1.6 GB).
        # NOT the input (gradient checkpointing handles it at layer level).
        ctx._weight_data = weight_data
        ctx._quant_state = quant_state
        ctx._has_bias = bias is not None

        return output

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, torch.Tensor | None]:
        import bitsandbytes.functional as bnb_fn

        # Re-dequantize in backward (~0.2ms per layer, saves ~58 MiB per layer)
        weight_bf16 = bnb_fn.dequantize_4bit(ctx._weight_data, ctx._quant_state)

        # grad_input = grad_output @ weight (propagate gradients backward)
        grad_input = grad_output @ weight_bf16

        # 4-bit weights are frozen -- no grad_weight
        grad_bias = None
        if ctx._has_bias:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)

        return grad_input, None, None, grad_bias


def _make_patched_forward(module: bnb.nn.Linear4bit) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a patched forward bound to a specific module instance.

    Factory function avoids the closure capture bug where all modules
    share the same reference if you capture in a loop.
    """
    weight = module.weight  # bnb.nn.Params4bit
    bias = module.bias

    def patched_forward(x: torch.Tensor) -> torch.Tensor:
        return cast(
            "torch.Tensor",
            _Redequantize4bit.apply(
                x,
                weight.data,  # Quantized uint8 data (already in model memory)
                weight.quant_state,  # Quantization metadata (tiny)  # pyright: ignore[reportAttributeAccessIssue]
                bias,
            ),
        )

    return patched_forward


def patch_model_autograd_4bit(model: torch.nn.Module) -> int:
    """Patch all Linear4bit modules to re-dequantize in backward.

    Call AFTER model load + PEFT setup. LoRA layers call
    self.base_layer(x) which hits the patched forward.

    Returns number of modules patched.
    """
    try:
        import bitsandbytes as bnb
    except ImportError:
        logger.warning("bitsandbytes not installed, skipping autograd 4-bit patch")
        return 0

    count = 0
    for _name, mod in model.named_modules():
        if isinstance(mod, bnb.nn.Linear4bit):
            mod.forward = _make_patched_forward(mod)  # type: ignore[method-assign]
            count += 1

    if count > 0:
        estimated_savings_gb = count * 58 / 1024
        logger.info(
            f"Patched {count} Linear4bit modules: re-dequantize in backward "
            f"(saves ~{estimated_savings_gb:.1f} GB)",
        )

    return count
