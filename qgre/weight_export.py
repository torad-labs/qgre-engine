"""WeightExporter — PEFT-aware weight extraction.

Node 1 of the Weight Sync Bus. Knows PEFT model structure, ModulesToSaveWrapper,
LoRA A/B. Does NOT know vLLM, engine internals, or buffer formats.

Replaceable: if we switch from PEFT to a different adapter framework, only this
file changes. WeightBus and WeightLoader are untouched.

NOTE on Params4bit monkey-patch (April 2026):
    PEFT's Linear4bit.merge (peft/tuners/lora/bnb.py:402) does
        kwargs = weight.__dict__
        bnb.nn.Params4bit(w_data.to("cpu"), **kwargs)
    to rebuild the quantized weight after adding LoRA deltas. But on current
    bitsandbytes (0.49.2) + unsloth (2026.4.4), weight.__dict__ contains
    attributes that are NOT valid Params4bit constructor kwargs (e.g. 'to'),
    causing a TypeError on every merge_adapter() call. Known PEFT issue #2501.
    We wrap Params4bit.__new__ once at import time to silently drop unknown
    kwargs. This restores 4-bit merge_adapter functionality so MERGE weight
    sync strategy works.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _patch_params4bit_accept_unknown_kwargs() -> None:
    """Wrap bitsandbytes.nn.Params4bit.__new__ to drop unknown kwargs.

    Idempotent: safe to call multiple times. Only wraps the original __new__
    once, marked by a _qgre_patched attribute on the class.
    """
    try:
        import bitsandbytes as bnb  # type: ignore[import-untyped]
    except ImportError:
        return  # No bnb, no Params4bit, no patch needed.

    p4b_class = bnb.nn.Params4bit
    if getattr(p4b_class, "_qgre_patched", False):
        return

    # Valid constructor kwargs for Params4bit (from bitsandbytes 0.49.2 source).
    # Anything else in weight.__dict__ (e.g. bound method caches, torch-compile
    # attrs, Unsloth instrumentation) must be filtered out before construction.
    valid_kwargs = {
        "data",
        "requires_grad",
        "quant_state",
        "blocksize",
        "compress_statistics",
        "quant_type",
        "quant_storage",
        "module",
        "bnb_quantized",
    }
    original_new = p4b_class.__new__

    def patched_new(cls: type, *args: Any, **kwargs: Any) -> Any:
        filtered = {k: v for k, v in kwargs.items() if k in valid_kwargs}
        return original_new(cls, *args, **filtered)

    p4b_class.__new__ = staticmethod(patched_new)  # type: ignore[method-assign]
    p4b_class._qgre_patched = True  # type: ignore[attr-defined]


# Apply the patch at module import so any downstream PEFT merge_adapter call
# benefits automatically. Downstream callers do not need to know this exists.
_patch_params4bit_accept_unknown_kwargs()


class WeightExporter:
    """Extract trainable weights from a PEFT model."""

    def merge_lora(self, model: nn.Module) -> None:
        """Merge LoRA A/B into base weights in-place.

        On 4-bit quantized models, PEFT's merge_adapter dequantizes, adds LoRA deltas,
        and requantizes. The Params4bit monkey-patch (applied at import time by this
        module) ensures the requantization succeeds on current bitsandbytes versions.
        Under Unsloth's fast_inference colocation, the requantized weights remain
        accessible to vLLM via shared memory. Tested working at 9.8 GB VRAM on
        RTX 5080 with Qwen3-1.7B 4-bit + MERGE strategy (April 2026).
        """
        model.merge_adapter()  # type: ignore[attr-defined]

    def unmerge_lora(self, model: nn.Module) -> None:
        """Restore LoRA A/B as separate adapters (reverse of merge)."""
        model.unmerge_adapter()  # type: ignore[attr-defined]

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
            lm_head = model.get_output_embeddings()  # type: ignore[attr-defined]
            # Unwrap PEFT ModulesToSaveWrapper → get active adapter's nn.Linear
            if hasattr(lm_head, "modules_to_save"):
                lm_head = lm_head.modules_to_save["default"]
            if isinstance(lm_head, nn.Linear):
                return lm_head
        except (AttributeError, KeyError):
            pass
        return None
