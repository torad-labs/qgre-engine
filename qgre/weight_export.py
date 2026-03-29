"""WeightExporter — PEFT-aware weight extraction.

Node 1 of the Weight Sync Bus. Knows PEFT model structure, ModulesToSaveWrapper,
LoRA A/B. Does NOT know vLLM, engine internals, or buffer formats.

Replaceable: if we switch from PEFT to a different adapter framework, only this
file changes. WeightBus and WeightLoader are untouched.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class WeightExporter:
    """Extract trainable weights from a PEFT model."""

    def merge_lora(self, model: nn.Module) -> None:
        """Merge LoRA A/B into base weights in-place.

        WARNING: On 4-bit quantized models, merge creates NEW tensors (dequant→add→requant).
        This breaks shared memory with vLLM. Only use for full-precision models or deployment.
        """
        model.merge_adapter()

    def unmerge_lora(self, model: nn.Module) -> None:
        """Restore LoRA A/B as separate adapters (reverse of merge)."""
        model.unmerge_adapter()

    def get_modules_to_save(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Extract lm_head/embed_tokens trainable weights from ModulesToSaveWrapper.

        Returns dict mapping module name to weight tensor (views, not copies).
        These are the active adapter weights, not the frozen originals.
        """
        weights = {}
        state_dict = model.state_dict()
        for key, tensor in state_dict.items():
            if "modules_to_save" not in key or "weight" not in key:
                continue
            if "lm_head" in key:
                weights["lm_head"] = tensor
            elif "embed_tokens" in key:
                weights["embed_tokens"] = tensor
        return weights

    def get_lora_tensors(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Extract LoRA A/B tensors from model state_dict.

        Returns filtered state_dict with only .lora_A. and .lora_B. keys.
        Values are views of the live training parameters — updated by optimizer.step().
        """
        state_dict = model.state_dict()
        return {
            k: v for k, v in state_dict.items()
            if ".lora_A." in k or ".lora_B." in k
        }

    def get_lm_head(self, model: nn.Module) -> nn.Linear | None:
        """Get the unwrapped lm_head nn.Linear (for fused logprobs).

        PEFT wraps lm_head in ModulesToSaveWrapper when it's in modules_to_save.
        This returns the inner nn.Linear from the active adapter.
        """
        try:
            lm_head = model.get_output_embeddings()
            # Unwrap PEFT ModulesToSaveWrapper → get active adapter's nn.Linear
            if hasattr(lm_head, "modules_to_save"):
                lm_head = lm_head.modules_to_save["default"]
            if isinstance(lm_head, nn.Linear):
                return lm_head
        except (AttributeError, KeyError):
            pass
        return None
