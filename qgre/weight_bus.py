"""WeightBus — strategy dispatcher for weight sync.

Node 2 of the Weight Sync Bus. Knows NOTHING about PEFT or vLLM — pure coordination.
Dispatches sync operations between WeightExporter and WeightLoader based on strategy.

Strategies:
- DIRECT_COPY: load_lora_directly copies LoRA A/B into vLLM stacked buffers.
  Default for 4-bit quantized training. ~microseconds per sync.
- MERGE: merge_adapter bakes LoRA into base weights (shared memory with vLLM).
  Only for full-precision models. NOT viable for 4-bit (creates new tensor, breaks pointer).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch.nn as nn
    from qgre.weight_export import WeightExporter
    from qgre.weight_load import WeightLoader


class SyncStrategy(Enum):
    DIRECT_COPY = "direct_copy"  # Default: GPU tensor copy into vLLM LoRA buffers
    MERGE = "merge"              # Full-precision only: merge LoRA into base weights


class WeightBus:
    """Coordinate weight sync between exporter and loader.

    Each sync() call pushes updated training weights to the inference engine.
    Each restore_for_training() call reverses any inference-time modifications.

    The bus itself holds no model references — it receives them per call.
    This makes it stateless and testable in isolation.
    """

    def __init__(self, strategy: SyncStrategy = SyncStrategy.DIRECT_COPY):
        self.strategy = strategy
        self._initialized = False  # True after first successful sync

    def sync(
        self,
        exporter: WeightExporter,
        loader: WeightLoader,
        model: nn.Module,
        modules_to_save: list[str] | None = None,
    ) -> None:
        """Push updated weights from training model to inference engine.

        Called after optimizer.step() (step-end sync) and after LoRA dropout
        (pre-generate sync with noisy weights).

        Args:
            exporter: WeightExporter instance
            loader: WeightLoader instance
            model: PEFT-wrapped training model
            modules_to_save: Expected modules (e.g., ["lm_head"]). Warns if missing.
        """
        if self.strategy == SyncStrategy.MERGE:
            exporter.merge_lora(model)
            loader.sync_modules_to_save(exporter.get_modules_to_save(model, expected=modules_to_save))
            # MERGE modifies base weights in-place — flush vLLM KV cache to prevent stale keys/values
            loader.flush_kv_cache()
        elif self.strategy == SyncStrategy.DIRECT_COPY:
            loader.sync_lora_direct(model, first_call=not self._initialized)
            loader.sync_modules_to_save(exporter.get_modules_to_save(model, expected=modules_to_save))

        self._initialized = True

    def restore_for_training(
        self,
        exporter: WeightExporter,
        model: nn.Module,
    ) -> None:
        """Restore model state for training after inference.

        For MERGE: unmerges LoRA from base weights.
        For DIRECT_COPY: no-op (LoRA params are separate, nothing to restore).
        """
        if self.strategy == SyncStrategy.MERGE:
            exporter.unmerge_lora(model)
