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

import threading
from enum import Enum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from torch import nn

    from qgre.sync_state import SyncState
    from qgre.types import TrainingContext
    from qgre.weight_export import WeightExporter
    from qgre.weight_load import WeightLoader


class SyncStrategy(Enum):
    DIRECT_COPY = "direct_copy"  # Default: GPU tensor copy into vLLM LoRA buffers
    MERGE = "merge"  # Full-precision only: merge LoRA into base weights


class WeightBus:
    """Coordinate weight sync between exporter and loader.

    Each sync() call pushes updated training weights to the inference engine.
    Each restore_for_training() call reverses any inference-time modifications.

    The bus itself holds no model references — it receives them per call.
    This makes it stateless and testable in isolation.
    """

    def __init__(self, state: SyncState, strategy: SyncStrategy = SyncStrategy.DIRECT_COPY):
        self._state = state
        self.strategy = strategy
        self._engine_id: int | None = None  # W5: Track engine identity to detect recreation
        self._sync_lock = threading.Lock()  # Protect engine_id checks and _initialized updates

    def reset_on_engine_recreate(self):
        """W5: Reset state when vLLM engine is recreated."""
        self._state.reset_for_engine_recreate()
        self._engine_id = None

    def sync(
        self,
        exporter: WeightExporter,
        loader: WeightLoader,
        model: nn.Module,
        ctx: TrainingContext,
        modules_to_save: list[str] | None = None,
    ) -> None:
        """Push updated weights from training model to inference engine.

        Called after optimizer.step() (step-end sync) and after LoRA dropout
        (pre-generate sync with noisy weights).

        Args:
            exporter: WeightExporter instance
            loader: WeightLoader instance
            model: PEFT-wrapped training model
            ctx: TrainingContext for device/dtype validation
            modules_to_save: Expected modules (e.g., ["lm_head"]). Warns if missing.
        """
        with self._sync_lock:
            # R10: Handle None engine (lazy init) - skip engine_id tracking until engine exists
            engine = loader.engine
            engine_id = id(engine) if engine is not None else None
            if (
                engine_id is not None
                and self._engine_id is not None
                and self._engine_id != engine_id
            ):
                self._state.reset_for_engine_recreate()
                self._engine_id = None
            first_call = not self._state.initialized
        try:
            if self.strategy == SyncStrategy.MERGE:
                exporter.merge_lora(model)
                loader.sync_modules_to_save(
                    exporter.get_modules_to_save(model, expected=modules_to_save), ctx
                )
                # MERGE modifies base weights in-place — flush vLLM KV cache to prevent stale keys/values
                loader.flush_kv_cache()
                # Fortify-1: MERGE strategy must also mark sync complete; the legacy
                # path only set _initialized=True in DIRECT_COPY, leaving MERGE users
                # permanently at first_call=True.
                self._state.complete_sync(first_call=first_call)
                with self._sync_lock:
                    self._engine_id = engine_id
            elif self.strategy == SyncStrategy.DIRECT_COPY:
                # Dropout check moved inside WeightLoader.sync_lora_direct (inside lock)
                loader.sync_lora_direct(model, ctx, first_call=first_call)
                # Get fresh state_dict inside sync_modules_to_save to avoid stale tensor references
                loader.sync_modules_to_save(
                    exporter.get_modules_to_save(model, expected=modules_to_save), ctx
                )
                # Flush KV cache after DIRECT_COPY modules_to_save (embed_tokens, lm_head)
                loader.flush_kv_cache()
                self._state.complete_sync(first_call=first_call)
                with self._sync_lock:
                    self._engine_id = engine_id
        except Exception as e:
            # Clear dropout state in exception handler to prevent stuck state
            self._state.fail_sync()
            # Don't set initialized=True on failure — next sync will retry first_call path
            raise RuntimeError(f"Weight sync failed: {e}") from e

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
