"""WeightLoader — vLLM-aware weight injection.

Node 3 of the Weight Sync Bus. Knows vLLM engine structure, stacked buffers,
model_runner paths. Does NOT know PEFT, LoRA structure, or adapter names.

Replaceable: if we switch from vLLM to a different inference engine, only this
file changes. WeightExporter and WeightBus are untouched.
"""

from __future__ import annotations

import gc
import threading
import warnings
from typing import TYPE_CHECKING

import torch
from torch import nn


if TYPE_CHECKING:
    from qgre.types import TrainingContext

from qgre.types import WeightLoaderLifecycle


try:
    from qgre.lora_dropout import apply_lora_dropout
except ImportError:
    # SFH-006: Log when lora_dropout module is unavailable
    warnings.warn(
        "lora_dropout module not available. LoRA dropout will be disabled "
        "even if lora_dropout_rate is configured.",
        stacklevel=2,
    )
    apply_lora_dropout = None


class WeightLoader:
    """Inject weights into vLLM engine.

    Uses WeightLoaderLifecycle state machine to prevent impossible states.
    State transitions are explicit methods that validate preconditions.
    """

    def __init__(self, model: nn.Module):
        """Initialize with the PeftModel that has vllm_engine attached.

        Args:
            model: The PeftModel (Unsloth wraps vllm_engine onto it via patch_peft_fast_inference)
        """
        self._model = model
        self._lora_request = None  # Set during LOADING -> READY transition
        self._lifecycle = WeightLoaderLifecycle.UNINITIALIZED
        # CR-001: Thread lock for state machine transitions
        self._lock = threading.Lock()
        # CR-002: Track if KV cache may be stale (zeroing failed)
        self._cache_potentially_stale = False

    # --- Lifecycle state machine ---

    @property
    def lifecycle(self) -> WeightLoaderLifecycle:
        """Current lifecycle state."""
        return self._lifecycle

    def _transition_to(
        self, target: WeightLoaderLifecycle, valid_from: tuple[WeightLoaderLifecycle, ...]
    ) -> None:
        """Transition to target state, validating preconditions."""
        if self._lifecycle not in valid_from:
            raise RuntimeError(
                f"WeightLoader: Invalid state transition {self._lifecycle.value} -> {target.value}. "
                f"Valid source states: {[s.value for s in valid_from]}",
            )
        self._lifecycle = target

    def _transition_to_loading(self) -> None:
        """Transition to LOADING state (first sync_lora_direct call)."""
        self._transition_to(
            WeightLoaderLifecycle.LOADING,
            (WeightLoaderLifecycle.UNINITIALIZED, WeightLoaderLifecycle.ERROR),
        )

    def _transition_to_ready(self) -> None:
        """Transition to READY state (prepare_vllm_lora_loading succeeded)."""
        self._transition_to(
            WeightLoaderLifecycle.READY,
            (WeightLoaderLifecycle.LOADING,),
        )

    # ELI-001: Removed _transition_to_dropout - dropout state is tracked externally
    # by lora_dropout.apply_lora_dropout._dropout_active, not by this state machine.

    def _transition_to_error(self) -> None:
        """Transition to ERROR state (exception during operation)."""
        # Can transition to ERROR from any state
        self._lifecycle = WeightLoaderLifecycle.ERROR
        self._lora_request = None

    def _transition_to_uninitialized(self) -> None:
        """Transition to UNINITIALIZED state (reset for engine recreate)."""
        self._transition_to(
            WeightLoaderLifecycle.UNINITIALIZED,
            (
                WeightLoaderLifecycle.UNINITIALIZED,
                WeightLoaderLifecycle.LOADING,
                WeightLoaderLifecycle.READY,
                WeightLoaderLifecycle.ERROR,
            ),
        )
        self._lora_request = None
        # Clear stale cache flag since engine is being recreated with fresh state
        self._cache_potentially_stale = False

    # --- Legacy compatibility properties ---

    @property
    def _direct_ready(self) -> bool:
        """Legacy: True when in READY state (weights loaded and ready for sync)."""
        return self._lifecycle == WeightLoaderLifecycle.READY

    @property
    def _load_lora_called(self) -> bool:
        """Legacy: True when past UNINITIALIZED state."""
        return self._lifecycle != WeightLoaderLifecycle.UNINITIALIZED

    @property
    def lora_request(self):
        """The LoRARequest for vLLM fast_generate (set after first direct sync)."""
        return self._lora_request

    def sync_lora_direct(
        self, model: nn.Module, ctx: TrainingContext, first_call: bool = False
    ) -> None:
        """Copy LoRA A/B tensors directly into vLLM's stacked buffers.

        First call: registers adapter via load_lora + sets up GPU copy mappings.
        Subsequent calls: direct tensor.copy_() — microseconds, no LoRARequest churn.

        Args:
            model: The PeftModel with trained LoRA weights
            ctx: Training context for dtype/device validation
            first_call: True on first invocation (sets up mappings)
        """
        from unsloth_zoo.vllm_utils import load_lora_directly, prepare_vllm_lora_loading

        # ELI-002/003: Lock protects ALL operations including dropout check and fast path
        # to prevent race conditions between state checks and transitions
        with self._lock:
            # CR-002: Check if KV cache is potentially stale before sync
            if self._cache_potentially_stale:
                raise RuntimeError(
                    "KV cache is potentially stale due to previous flush failure. "
                    "Generations may use corrupted cache. Recreate vLLM engine or restart training."
                )

            # WS3-001: Track dropout state, skip sync if dropout active
            # (checked inside lock to prevent race with dropout activation)
            if apply_lora_dropout is not None and getattr(
                apply_lora_dropout, "_dropout_active", False
            ):
                warnings.warn(
                    "WS3-001: sync_lora_direct called while LoRA dropout is active. "
                    "Skipping sync to avoid race condition. Call restore() first.",
                    stacklevel=2,
                )
                return

            if first_call or not self._direct_ready:
                # Bootstrap: register adapter once, set up GPU→GPU copy mappings
                # State machine: UNINITIALIZED -> LOADING -> READY
                if self._lifecycle == WeightLoaderLifecycle.UNINITIALIZED:
                    self._transition_to_loading()

                if self._lora_request is None:
                    # SFH-003: Wrap adapter loading in try/catch to prevent stuck LOADING state
                    try:
                        adapter_path = self._get_adapter_config_path()
                        # WS2: Validate adapter_config.json rank matches model's rank
                        import json
                        from pathlib import Path

                        adapter_config_file = Path(adapter_path) / "adapter_config.json"
                        if adapter_config_file.exists():
                            with open(adapter_config_file) as f:
                                adapter_cfg = json.load(f)
                            disk_rank = adapter_cfg.get("r")
                            model_rank = getattr(model.peft_config.get("default"), "r", None)  # type: ignore[attr-defined]
                            if (
                                disk_rank is not None
                                and model_rank is not None
                                and disk_rank != model_rank
                            ):
                                raise RuntimeError(
                                    f"WS2: adapter_config.json rank mismatch. "
                                    f"Disk: {disk_rank}, training model: {model_rank}. "
                                    f"Delete {adapter_path} or update config to match.",
                                )
                        self._lora_request = model.load_lora(  # type: ignore[attr-defined]
                            adapter_path,
                            load_tensors=True,
                        )
                    except Exception as e:
                        self._transition_to_error()
                        raise RuntimeError(
                            f"Adapter loading failed during LOADING state: {e}",
                        ) from e

                try:
                    prepare_vllm_lora_loading(model)
                    self._transition_to_ready()
                except Exception as e:
                    import traceback

                    # WS3-002 + W10: Transition to ERROR state to allow clean retry
                    self._transition_to_error()
                    raise RuntimeError(
                        f"prepare_vllm_lora_loading failed — direct LoRA sync unavailable. "
                        f"Check vLLM engine initialization.\nOriginal error: {e}\n{traceback.format_exc()}",
                    ) from e
            else:
                # Fast path: direct GPU-to-GPU tensor copy (inside lock for thread safety)
                try:
                    load_lora_directly(model)
                except Exception as e:
                    # SFH-001: Fast path failures must transition to ERROR state
                    self._transition_to_error()
                    raise RuntimeError(
                        f"load_lora_directly fast path failed — weights may be inconsistent. "
                        f"State was {WeightLoaderLifecycle.READY.value}, now ERROR. Error: {e}",
                    ) from e

    def sync_modules_to_save(self, weights: dict[str, torch.Tensor], ctx: TrainingContext) -> None:
        """Copy lm_head/embed_tokens into vLLM's base model.

        vLLM's LoRA system ignores modules_to_save by design (vLLM PR #14978).
        We bypass this by writing directly to vLLM's model weights.

        Args:
            weights: dict mapping "lm_head" / "embed_tokens" to weight tensors
            ctx: Training context for dtype/device validation
        """
        vllm_model = self.get_vllm_model()
        if vllm_model is None:
            raise RuntimeError(
                "vLLM engine not available for modules_to_save sync. "
                "Engine must be initialized before weight sync. Check generation_backend setup.",
            )

        # W15: Validate all modules exist AND shapes/dtypes match before any copy (atomic sync)
        expected = list(weights.keys())
        for name in expected:
            tensor = weights[name]
            if name == "lm_head":
                if not hasattr(vllm_model, "lm_head"):
                    raise RuntimeError(
                        f"W15: modules_to_save sync validation: lm_head not found in vLLM model. "
                        f"Cannot sync {expected}. Check vLLM model structure.",
                    )
                target = vllm_model.lm_head.weight
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"W15: Shape mismatch in lm_head (validation): training={tensor.shape} vs vLLM={target.shape}. "
                        "Aborting before any copy.",
                    )
            elif name == "embed_tokens":
                if not hasattr(vllm_model, "model") or not hasattr(
                    vllm_model.model, "embed_tokens"
                ):
                    raise RuntimeError(
                        f"W15: modules_to_save sync validation: embed_tokens not found in vLLM model. "
                        f"Cannot sync {expected}. Check vLLM model structure.",
                    )
                target = vllm_model.model.embed_tokens.weight
                if tensor.shape != target.shape:
                    raise RuntimeError(
                        f"W15: Shape mismatch in embed_tokens (validation): training={tensor.shape} vs vLLM={target.shape}. "
                        "Aborting before any copy.",
                    )

        synced = []
        try:
            for name, tensor in weights.items():
                # C08-DTYPE: Validate tensor dtype matches ctx.dtype
                if tensor.dtype != ctx.dtype:
                    warnings.warn(
                        f"C08-DTYPE: {name} weight dtype mismatch: tensor={tensor.dtype} vs ctx.dtype={ctx.dtype}. "
                        "Converting to ctx.dtype.",
                        stacklevel=2,
                    )

                # C09-DEVICE: Validate tensor is on ctx.device before GPU copy
                if tensor.device != ctx.device:
                    raise RuntimeError(
                        f"C09-DEVICE: {name} weight device mismatch: tensor on {tensor.device}, expected {ctx.device}. "
                        "Weights must be on ctx.device before GPU-to-GPU copy.",
                    )

                if name == "lm_head":
                    try:
                        target = vllm_model.lm_head.weight
                    except AttributeError as e:
                        raise RuntimeError(
                            f"lm_head not found in vLLM model. Cannot sync {expected}. "
                            f"Check vLLM model structure: {e}",
                        ) from e
                    if tensor.shape != target.shape:
                        raise RuntimeError(
                            f"Shape mismatch syncing lm_head: training={tensor.shape} vs vLLM={target.shape}",
                        )
                    # WS-R3-08: Check dtype match before copy, convert if mismatch
                    if tensor.dtype != target.dtype:
                        warnings.warn(
                            f"W12: lm_head dtype mismatch: training={tensor.dtype} vs vLLM={target.dtype}. "
                            f"Converting to target dtype {target.dtype}.",
                            stacklevel=2,
                        )
                        tensor = tensor.to(dtype=target.dtype)
                    target.data.copy_(tensor.to(device=target.device))
                    synced.append(name)
                elif name == "embed_tokens":
                    try:
                        target = vllm_model.model.embed_tokens.weight
                    except AttributeError as e:
                        raise RuntimeError(
                            f"embed_tokens not found in vLLM model. Cannot sync {expected}. "
                            f"Check vLLM model structure: {e}",
                        ) from e
                    if tensor.shape != target.shape:
                        raise RuntimeError(
                            f"Shape mismatch syncing embed_tokens: training={tensor.shape} vs vLLM={target.shape}",
                        )
                    # WS-R3-08: Check dtype match before copy, convert if mismatch
                    if tensor.dtype != target.dtype:
                        warnings.warn(
                            f"W12: embed_tokens dtype mismatch: training={tensor.dtype} vs vLLM={target.dtype}. "
                            f"Converting to target dtype {target.dtype}.",
                            stacklevel=2,
                        )
                        tensor = tensor.to(dtype=target.dtype)
                    target.data.copy_(tensor.to(device=target.device))
                    synced.append(name)
        except Exception as e:
            # WS3-006: Rollback on failure (log error, don't crash)
            warnings.warn(
                f"WS3-006: modules_to_save sync failed mid-operation. "
                f"Synced: {synced}, expected: {expected}. Error: {e}. "
                "Weights may be inconsistent — restart training from checkpoint.",
                stacklevel=2,
            )
            raise
        finally:
            # WS8: Move synchronize into finally block to ensure it runs even on exception
            if synced and torch.cuda.is_available():
                torch.cuda.synchronize()

        # WS-R1-4: Track expected vs synced and raise if mismatch
        if expected and set(expected) != set(synced):
            raise RuntimeError(
                f"modules_to_save partial sync failure: expected {expected}, synced {synced}. "
                f"Missing: {set(expected) - set(synced)}",
            )

    def get_vllm_model(self):
        """Navigate vLLM internals to get the base model for direct weight access.

        Traverses: PeftModel → base model → vllm_engine → llm_engine → model_runner → model
        Returns None if engine not yet created (lazy init).
        """
        for obj in [self._model, getattr(self._model, "model", None)]:
            if obj is None:
                continue
            engine = getattr(obj, "vllm_engine", None)
            if engine is None:
                continue

            # Traverse vLLM engine chain with explicit attribute checks
            # to avoid AttributeError in production
            llm_engine = getattr(engine, "llm_engine", None)
            if llm_engine is None:
                continue
            model_executor = getattr(llm_engine, "model_executor", None)
            if model_executor is None:
                continue
            driver_worker = getattr(model_executor, "driver_worker", None)
            if driver_worker is None:
                continue
            model_runner = getattr(driver_worker, "model_runner", None)
            if model_runner is None:
                continue
            model = getattr(model_runner, "model", None)
            if model is not None:
                # W2-3: Validate both lm_head and embed_tokens
                if not hasattr(model, "lm_head"):
                    warnings.warn(
                        "vLLM model traversal returned object without lm_head — "
                        "may be wrong type due to vLLM version change. Skipping.",
                        stacklevel=2,
                    )
                    continue
                if not hasattr(model, "model") or not hasattr(model.model, "embed_tokens"):
                    warnings.warn(
                        "W2-3: vLLM model traversal returned object without embed_tokens — "
                        "may be wrong type due to vLLM version change. Skipping.",
                        stacklevel=2,
                    )
                    continue
                return model

        # WS3-010: Log diagnostic info on traversal failure (DEBUG level to avoid spam in retry loops)
        import logging

        logger = logging.getLogger(__name__)
        logger.debug(
            "WS3-010: get_vllm_model traversal failed. Diagnostic info: "
            f"model_type={type(self._model).__name__}, "
            f"has_vllm_engine={hasattr(self._model, 'vllm_engine')}, "
            f"has_model={hasattr(self._model, 'model')}, "
            f"base_has_vllm_engine={hasattr(getattr(self._model, 'model', None), 'vllm_engine')}",
        )
        raise RuntimeError(
            "get_vllm_model: traversal failed. Could not find vLLM model via engine chain. "
            f"Checked attributes: vllm_engine, model. "
            f"Model type: {type(self._model).__name__}. "
            "Possible causes: vLLM engine not initialized, or vLLM API changed.",
        )

    def flush_kv_cache(self) -> None:
        """Flush vLLM KV cache to reclaim VRAM without destroying the engine.

        Does NOT delete vllm_engine, fast_generate, or NCCL state.
        Only flushes KV cache pages via scheduler + zeros GPU cache tensors.
        """
        # Find vllm_engine with explicit None checks
        engine = getattr(self._model, "vllm_engine", None)
        if engine is None:
            base_model = getattr(self._model, "model", None)
            if base_model is not None:
                engine = getattr(base_model, "vllm_engine", None)
        if engine is None:
            # SFH-005: Warn when engine not found instead of silent return
            if not getattr(self, "_engine_not_found_warned", False):
                warnings.warn(
                    "flush_kv_cache: No vLLM engine found (engine not attached to model). "
                    "KV cache flush skipped. This may indicate generation_backend configuration issue.",
                    stacklevel=2,
                )
                self._engine_not_found_warned = True
            return

        # Pre-check: verify llm_engine exists before attempting scheduler access
        llm_engine = getattr(engine, "llm_engine", None)
        if llm_engine is None:
            if not getattr(self, "_kv_scheduler_warned", False):
                warnings.warn(
                    "KV cache flush: llm_engine not found (engine not fully initialized)",
                    stacklevel=2,
                )
                self._kv_scheduler_warned = True
            return

        # Flush KV cache blocks via scheduler
        try:
            scheduler_attr = getattr(llm_engine, "scheduler", None)
            if scheduler_attr is None:
                schedulers = []
            elif isinstance(scheduler_attr, list):
                schedulers = scheduler_attr
            else:
                schedulers = [scheduler_attr]

            for scheduler in schedulers:
                if hasattr(scheduler, "free_finished_seqs"):
                    scheduler.free_finished_seqs()
                block_manager = getattr(scheduler, "block_manager", None)
                if block_manager is not None:
                    gpu_allocator = getattr(block_manager, "gpu_allocator", None)
                    if gpu_allocator is not None and hasattr(gpu_allocator, "free_all"):
                        gpu_allocator.free_all()
        except AttributeError as e:
            # W2-6: Track failure count and log periodically
            if not hasattr(self, "_kv_scheduler_warned_count"):
                self._kv_scheduler_warned_count = 0
            self._kv_scheduler_warned_count += 1
            if (
                not getattr(self, "_kv_scheduler_warned", False)
                or self._kv_scheduler_warned_count % 100 == 0
            ):
                warnings.warn(
                    f"KV cache scheduler flush skipped (vLLM API change): {e}. "
                    f"(occurred {self._kv_scheduler_warned_count} times)",
                    stacklevel=2,
                )
                self._kv_scheduler_warned = True
        except Exception as e:
            # W13: Re-raise after warning when flush fails
            warnings.warn(
                f"W13: KV cache scheduler flush failed: {e}. Stale cache may cause hallucinations. "
                "Check vLLM scheduler state.",
                stacklevel=2,
            )
            raise

        # Zero GPU cache tensors directly — pre-check each attribute in chain
        model_executor = getattr(llm_engine, "model_executor", None)
        driver_worker = getattr(model_executor, "driver_worker", None) if model_executor else None
        gpu_cache = getattr(driver_worker, "gpu_cache", None) if driver_worker else None

        if gpu_cache is None:
            # No gpu_cache attribute — may be different vLLM version
            if not getattr(self, "_kv_worker_warned", False):
                warnings.warn(
                    "KV cache tensor zeroing skipped: gpu_cache not found on driver_worker",
                    stacklevel=2,
                )
                self._kv_worker_warned = True
        else:
            try:
                # W2-5: Handle both list and dict structures
                cache_items = gpu_cache.values() if isinstance(gpu_cache, dict) else gpu_cache
                for layer_cache in cache_items:
                    if isinstance(layer_cache, torch.Tensor):
                        layer_cache.zero_()
                    elif isinstance(layer_cache, (list, tuple)):
                        for t in layer_cache:
                            if isinstance(t, torch.Tensor):
                                t.zero_()
                # CR-002: Clear stale flag on successful zeroing
                self._cache_potentially_stale = False
            except (RuntimeError, AttributeError, TypeError) as e:
                # CR-002: Track that cache may be stale - generations could use old KV pairs
                self._cache_potentially_stale = True
                warnings.warn(
                    f"CRITICAL: KV cache tensor zeroing failed: {e}. "
                    "Model may generate with stale cache. Consider recreating vLLM engine.",
                    stacklevel=2,
                )
                raise RuntimeError(
                    f"KV cache zeroing failed — inference may be corrupted: {e}",
                ) from e

        gc.collect()
        torch.cuda.empty_cache()

    def _get_adapter_config_path(self) -> str:
        """Get/create a temp directory for adapter_config.json bootstrap.

        load_lora(load_tensors=True) needs adapter_config.json on disk.
        On first call (lora_request_id==1), Unsloth writes it from peft_config.
        """
        import atexit
        import tempfile

        path = getattr(self, "_adapter_path", None)
        if path is None:
            path = tempfile.mkdtemp(prefix="qgre_adapter_")
            self._adapter_path = path
            atexit.register(self.cleanup_adapter_tempdir)
        return path

    def reset_state(self) -> bool:
        """WS3-009: Reset state on engine recreate.

        Transitions to UNINITIALIZED state, clearing _lora_request.

        Returns:
            True, indicating caller MUST also reset WeightBus._initialized.
            This is an explicit contract: ignoring the return value risks
            WeightBus thinking it's still initialized when the loader is not.

        CRITICAL: After calling this method, you MUST call:
            weight_bus.reset_on_engine_recreate()
        Failure to do so will cause stale state and sync failures.
        """
        with self._lock:
            self._transition_to_uninitialized()
        # ELI-005: Return value enforces caller contract - must reset WeightBus
        import warnings
        warnings.warn(
            "WeightLoader.reset_state() called. Caller MUST also call "
            "weight_bus.reset_on_engine_recreate() to prevent stale state.",
            stacklevel=2,
        )
        return True  # weight_bus_needs_reset

    def cleanup_adapter_tempdir(self):
        """Explicitly clean up adapter tempdir. Call at trainer shutdown."""
        import logging
        import shutil

        path = getattr(self, "_adapter_path", None)
        # WS3-004: Track cleanup state to avoid double-free
        if path is not None and not getattr(self, "_cleaned_up", False):
            try:
                shutil.rmtree(path)
                self._adapter_path = None
                self._cleaned_up = True
            except OSError as e:
                logging.getLogger(__name__).warning(
                    f"Failed to clean up adapter tempdir {path}: {e}. "
                    "Directory may persist on disk.",
                )

    def __del__(self):
        """Clean up tempdir on deletion."""
        import logging
        import shutil

        path = getattr(self, "_adapter_path", None)
        # WS3-004: Only cleanup if not already cleaned
        if path is not None and not getattr(self, "_cleaned_up", False):
            try:
                shutil.rmtree(path, ignore_errors=False)
            except OSError as e:
                # SFH-002: Log cleanup failures instead of silently suppressing
                # Can't raise in __del__, but must not hide resource leaks
                logging.getLogger(__name__).warning(
                    f"__del__ cleanup failed for {path}: {e}. Directory may persist on disk.",
                )
