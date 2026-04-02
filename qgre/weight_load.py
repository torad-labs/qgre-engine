"""WeightLoader — vLLM-aware weight injection.

Node 3 of the Weight Sync Bus. Knows vLLM engine structure, stacked buffers,
model_runner paths. Does NOT know PEFT, LoRA structure, or adapter names.

Replaceable: if we switch from vLLM to a different inference engine, only this
file changes. WeightExporter and WeightBus are untouched.
"""

from __future__ import annotations

import gc
import warnings
from typing import Any

import torch
import torch.nn as nn

from qgre.types import TrainingContext

try:
    from qgre.lora_dropout import apply_lora_dropout
except ImportError:
    apply_lora_dropout = None


class WeightLoader:
    """Inject weights into vLLM engine."""

    def __init__(self, model: nn.Module):
        """Initialize with the PeftModel that has vllm_engine attached.

        Args:
            model: The PeftModel (Unsloth wraps vllm_engine onto it via patch_peft_fast_inference)
        """
        self._model = model
        self._lora_request = None  # Set on first sync_lora_direct call
        self._direct_ready = False  # True after prepare_vllm_lora_loading succeeds

    @property
    def lora_request(self):
        """The LoRARequest for vLLM fast_generate (set after first direct sync)."""
        return self._lora_request

    def sync_lora_direct(self, model: nn.Module, ctx: TrainingContext, first_call: bool = False) -> None:
        """Copy LoRA A/B tensors directly into vLLM's stacked buffers.

        First call: registers adapter via load_lora + sets up GPU copy mappings.
        Subsequent calls: direct tensor.copy_() — microseconds, no LoRARequest churn.

        Args:
            model: The PeftModel with trained LoRA weights
            ctx: Training context for dtype/device validation
            first_call: True on first invocation (sets up mappings)
        """
        from unsloth_zoo.vllm_utils import prepare_vllm_lora_loading, load_lora_directly

        # WS3-001: Track dropout state, skip sync if dropout active
        if apply_lora_dropout is not None and getattr(apply_lora_dropout, '_dropout_active', False):
            import warnings
            warnings.warn(
                "WS3-001: sync_lora_direct called while LoRA dropout is active. "
                "Skipping sync to avoid race condition. Call restore() first."
            )
            return

        if first_call or not self._direct_ready:
            # Bootstrap: register adapter once, set up GPU→GPU copy mappings
            # W2: Track whether load_lora was called to prevent double-load on recovery
            if self._lora_request is None:
                if not getattr(self, "_load_lora_called", False):
                    self._lora_request = model.load_lora(
                        self._get_adapter_config_path(), load_tensors=True
                    )
                    self._load_lora_called = True
            try:
                prepare_vllm_lora_loading(model)
                self._direct_ready = True
            except Exception as e:
                import traceback
                # WS3-002: Reset _direct_ready on exception
                self._direct_ready = False
                self._lora_request = None
                self._load_lora_called = False
                raise RuntimeError(
                    f"prepare_vllm_lora_loading failed — direct LoRA sync unavailable. "
                    f"Check vLLM engine initialization.\nOriginal error: {e}\n{traceback.format_exc()}"
                ) from e
        else:
            # Fast path: direct GPU-to-GPU tensor copy
            load_lora_directly(model)

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
                "Engine must be initialized before weight sync. Check generation_backend setup."
            )

        # W3: Validate all modules exist before any copy (atomic sync)
        expected = list(weights.keys())
        for name in expected:
            if name == "lm_head":
                if not hasattr(vllm_model, "lm_head"):
                    raise RuntimeError(
                        f"W3: modules_to_save sync validation: lm_head not found in vLLM model. "
                        f"Cannot sync {expected}. Check vLLM model structure."
                    )
            elif name == "embed_tokens":
                if not hasattr(vllm_model, "model") or not hasattr(vllm_model.model, "embed_tokens"):
                    raise RuntimeError(
                        f"W3: modules_to_save sync validation: embed_tokens not found in vLLM model. "
                        f"Cannot sync {expected}. Check vLLM model structure."
                    )

        synced = []
        try:
            for name, tensor in weights.items():
                # C08-DTYPE: Validate tensor dtype matches ctx.dtype
                if tensor.dtype != ctx.dtype:
                    warnings.warn(
                        f"C08-DTYPE: {name} weight dtype mismatch: tensor={tensor.dtype} vs ctx.dtype={ctx.dtype}. "
                        "Converting to ctx.dtype."
                    )

                # C09-DEVICE: Validate tensor is on ctx.device before GPU copy
                if tensor.device != ctx.device:
                    raise RuntimeError(
                        f"C09-DEVICE: {name} weight device mismatch: tensor on {tensor.device}, expected {ctx.device}. "
                        "Weights must be on ctx.device before GPU-to-GPU copy."
                    )

                if name == "lm_head":
                    try:
                        target = vllm_model.lm_head.weight
                    except AttributeError:
                        warnings.warn("lm_head not found in vLLM model — skipping")
                        continue
                    if tensor.shape != target.shape:
                        raise RuntimeError(
                            f"Shape mismatch syncing lm_head: training={tensor.shape} vs vLLM={target.shape}"
                        )
                    # Check dtype match before copy, warn if mismatch
                    if tensor.dtype != target.dtype:
                        import warnings
                        warnings.warn(f"lm_head dtype mismatch: training={tensor.dtype} vs vLLM={target.dtype}. Converting.")
                    target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
                    synced.append(name)
                elif name == "embed_tokens":
                    try:
                        target = vllm_model.model.embed_tokens.weight
                    except AttributeError:
                        warnings.warn("embed_tokens not found in vLLM model — skipping")
                        continue
                    if tensor.shape != target.shape:
                        raise RuntimeError(
                            f"Shape mismatch syncing embed_tokens: training={tensor.shape} vs vLLM={target.shape}"
                        )
                    # Check dtype match before copy, warn if mismatch
                    if tensor.dtype != target.dtype:
                        import warnings
                        warnings.warn(f"embed_tokens dtype mismatch: training={tensor.dtype} vs vLLM={target.dtype}. Converting.")
                    target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
                    synced.append(name)
        except Exception as e:
            # WS3-006: Rollback on failure (log error, don't crash)
            import warnings
            warnings.warn(
                f"WS3-006: modules_to_save sync failed mid-operation. "
                f"Synced: {synced}, expected: {expected}. Error: {e}. "
                "Weights may be inconsistent — restart training from checkpoint."
            )
            raise

        # WS-R1-4: Track expected vs synced and raise if mismatch
        if expected and set(expected) != set(synced):
            raise RuntimeError(
                f"modules_to_save partial sync failure: expected {expected}, synced {synced}. "
                f"Missing: {set(expected) - set(synced)}"
            )
        # Only sync if we actually copied modules_to_save (restores pre-harden behavior)
        if synced and torch.cuda.is_available():
            torch.cuda.synchronize()

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
                        "may be wrong type due to vLLM version change. Skipping."
                    )
                    continue
                if not hasattr(model, "model") or not hasattr(model.model, "embed_tokens"):
                    warnings.warn(
                        "W2-3: vLLM model traversal returned object without embed_tokens — "
                        "may be wrong type due to vLLM version change. Skipping."
                    )
                    continue
                return model

        # WS3-010: Log diagnostic info on traversal failure
        import logging
        logging.getLogger(__name__).error(
            "WS3-010: get_vllm_model traversal failed. Diagnostic info: "
            f"model_type={type(self._model).__name__}, "
            f"has_vllm_engine={hasattr(self._model, 'vllm_engine')}, "
            f"has_model={hasattr(self._model, 'model')}, "
            f"base_has_vllm_engine={hasattr(getattr(self._model, 'model', None), 'vllm_engine')}"
        )
        raise RuntimeError(
            "get_vllm_model: traversal failed. Could not find vLLM model via engine chain. "
            f"Checked attributes: vllm_engine, model. "
            f"Model type: {type(self._model).__name__}. "
            "Possible causes: vLLM engine not initialized, or vLLM API changed."
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
            return

        # Pre-check: verify llm_engine exists before attempting scheduler access
        llm_engine = getattr(engine, "llm_engine", None)
        if llm_engine is None:
            if not getattr(self, "_kv_scheduler_warned", False):
                warnings.warn("KV cache flush: llm_engine not found (engine not fully initialized)")
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
            if not getattr(self, "_kv_scheduler_warned", False) or self._kv_scheduler_warned_count % 100 == 0:
                warnings.warn(
                    f"KV cache scheduler flush skipped (vLLM API change): {e}. "
                    f"(occurred {self._kv_scheduler_warned_count} times)"
                )
                self._kv_scheduler_warned = True
        except Exception as e:
            # W4: Raise error instead of warning when flush fails
            raise RuntimeError(
                f"W4: KV cache scheduler flush failed: {e}. Stale cache may cause hallucinations. "
                "Check vLLM scheduler state."
            ) from e

        # Zero GPU cache tensors directly — pre-check each attribute in chain
        model_executor = getattr(llm_engine, "model_executor", None)
        driver_worker = getattr(model_executor, "driver_worker", None) if model_executor else None
        gpu_cache = getattr(driver_worker, "gpu_cache", None) if driver_worker else None

        if gpu_cache is None:
            # No gpu_cache attribute — may be different vLLM version
            if not getattr(self, "_kv_worker_warned", False):
                warnings.warn("KV cache tensor zeroing skipped: gpu_cache not found on driver_worker")
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
            except Exception as e:
                warnings.warn(f"KV cache tensor zeroing failed: {e}")

        gc.collect()
        torch.cuda.empty_cache()

    def _get_adapter_config_path(self) -> str:
        """Get/create a temp directory for adapter_config.json bootstrap.

        load_lora(load_tensors=True) needs adapter_config.json on disk.
        On first call (lora_request_id==1), Unsloth writes it from peft_config.
        """
        import tempfile
        path = getattr(self, "_adapter_path", None)
        if path is None:
            path = tempfile.mkdtemp(prefix="qgre_adapter_")
            self._adapter_path = path
        return path

    def reset_state(self):
        """WS3-009: Reset state on engine recreate."""
        self._direct_ready = False
        self._lora_request = None
        # W2: Reset load_lora tracking on state reset
        self._load_lora_called = False

    def cleanup_adapter_tempdir(self):
        """Explicitly clean up adapter tempdir. Call at trainer shutdown."""
        import shutil
        path = getattr(self, "_adapter_path", None)
        # WS3-004: Track cleanup state to avoid double-free
        if path is not None and not getattr(self, "_cleaned_up", False):
            try:
                shutil.rmtree(path, ignore_errors=True)
                self._adapter_path = None
                self._cleaned_up = True
            except Exception as e:
                warnings.warn(f"Failed to clean up adapter tempdir {path}: {e}")

    def __del__(self):
        """Clean up tempdir on deletion."""
        import shutil
        path = getattr(self, "_adapter_path", None)
        # WS3-004: Only cleanup if not already cleaned
        if path is not None and not getattr(self, "_cleaned_up", False):
            try:
                shutil.rmtree(path, ignore_errors=True)
            except Exception:
                pass
