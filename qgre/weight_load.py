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

    def sync_lora_direct(self, model: nn.Module, first_call: bool = False) -> None:
        """Copy LoRA A/B tensors directly into vLLM's stacked buffers.

        First call: registers adapter via load_lora + sets up GPU copy mappings.
        Subsequent calls: direct tensor.copy_() — microseconds, no LoRARequest churn.

        Args:
            model: The PeftModel with trained LoRA weights
            first_call: True on first invocation (sets up mappings)
        """
        from unsloth_zoo.vllm_utils import prepare_vllm_lora_loading, load_lora_directly

        if first_call or not self._direct_ready:
            # Bootstrap: register adapter once, set up GPU→GPU copy mappings
            self._lora_request = model.load_lora(
                self._get_adapter_config_path(), load_tensors=True
            )
            try:
                prepare_vllm_lora_loading(model)
                self._direct_ready = True
            except Exception as e:
                import traceback
                raise RuntimeError(
                    f"prepare_vllm_lora_loading failed — direct LoRA sync unavailable. "
                    f"Check vLLM engine initialization.\nOriginal error: {e}\n{traceback.format_exc()}"
                ) from e
        else:
            # Fast path: direct GPU-to-GPU tensor copy
            load_lora_directly(model)

    def sync_modules_to_save(self, weights: dict[str, torch.Tensor]) -> None:
        """Copy lm_head/embed_tokens into vLLM's base model.

        vLLM's LoRA system ignores modules_to_save by design (vLLM PR #14978).
        We bypass this by writing directly to vLLM's model weights.

        Args:
            weights: dict mapping "lm_head" / "embed_tokens" to weight tensors
        """
        vllm_model = self.get_vllm_model()
        if vllm_model is None:
            raise RuntimeError(
                "vLLM engine not available for modules_to_save sync. "
                "Engine must be initialized before weight sync. Check generation_backend setup."
            )

        synced = []
        for name, tensor in weights.items():
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
                target.data.copy_(tensor.to(target.dtype))
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
                target.data.copy_(tensor.to(target.dtype))
                synced.append(name)

        if weights and not synced:
            raise RuntimeError(
                f"modules_to_save sync failed: expected to sync {list(weights.keys())} "
                "but none matched vLLM model structure."
            )
        if synced:
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
                return model

        return None

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
            # Expected if vLLM API changed. Warn once so user knows flush is skipped.
            if not getattr(self, "_kv_scheduler_warned", False):
                warnings.warn(f"KV cache scheduler flush skipped (vLLM API change): {e}")
                self._kv_scheduler_warned = True
        except Exception as e:
            warnings.warn(f"KV cache scheduler flush failed: {e}")

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
                for layer_cache in gpu_cache:
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
