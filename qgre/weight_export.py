"""WeightExporter — PEFT-aware weight extraction.

Node 1 of the Weight Sync Bus. Knows PEFT model structure, ModulesToSaveWrapper,
LoRA A/B. Does NOT know vLLM, engine internals, or buffer formats.

Replaceable: if we switch from PEFT to a different adapter framework, only this
file changes. WeightBus and WeightLoader are untouched.
"""

from __future__ import annotations

import torch
from torch import nn


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

    def get_modules_to_save(
        self,
        model: nn.Module,
        expected: list[str] | None = None,
        strict: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Extract lm_head/embed_tokens trainable weights from ModulesToSaveWrapper.

        Args:
            model: PEFT-wrapped model
            expected: Optional list of expected module names (e.g., ["lm_head"]).
                      If provided, warns when expected modules are missing from state_dict.
            strict: If True and expected modules are missing, raise error instead of warning.

        Returns dict mapping module name to weight tensor (views, not copies).
        These are the active adapter weights, not the frozen originals.
        """
        import warnings

        weights = {}
        # WARNING: Concurrent state_dict access race condition.
        # If optimizer.step() runs during this call, tensors may be in inconsistent state.
        # This is hard to fix without locks. Document the risk.
        # Call state_dict() inside sync function to avoid stale tensor references
        import warnings as _warnings

        if model.training:
            _warnings.warn(
                "get_modules_to_save called during training (model.training=True). "
                "Race condition possible if optimizer.step() runs concurrently. "
                "Call sync only between training steps.",
                stacklevel=2,
            )
        for key, tensor in model.state_dict().items():
            if "modules_to_save" not in key or "weight" not in key:
                continue
            # WS3-008: Force .clone() on state_dict tensors before sync
            if "lm_head" in key:
                weights["lm_head"] = tensor.clone()
            elif "embed_tokens" in key:
                weights["embed_tokens"] = tensor.clone()

        # Warn if expected modules are missing or unexpected modules present
        if expected:
            expected_set = set(expected)
            found_set = set(weights.keys())
            missing = expected_set - found_set
            unexpected = found_set - expected_set
            if missing:
                msg = (
                    f"get_modules_to_save: expected {missing} but not found in state_dict. "
                    f"Check modules_to_save config matches PEFT wrapper. Found: {list(weights.keys())}"
                )
                if strict:
                    raise RuntimeError(msg)
                warnings.warn(msg, stacklevel=2)
            if unexpected:
                warnings.warn(
                    f"get_modules_to_save: found unexpected modules {unexpected}. "
                    f"Expected {expected}, got {list(weights.keys())}. "
                    f"Config may be out of sync with PEFT wrapper — check modules_to_save.",
                    stacklevel=2,
                )
        return weights

    def get_lora_tensors(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Extract LoRA A/B tensors from model state_dict.

        Returns filtered state_dict with only .lora_A. and .lora_B. keys.
        Values are views of the live training parameters — updated by optimizer.step().
        """
        # Call state_dict() inside sync function to avoid stale tensor references
        return {k: v for k, v in model.state_dict().items() if ".lora_A." in k or ".lora_B." in k}

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
