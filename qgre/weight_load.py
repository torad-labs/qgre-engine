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
                warnings.warn(
                    f"prepare_vllm_lora_loading failed: {e}\n{traceback.format_exc()}"
                    f"Falling back to LoRARequest per step (slower)."
                )
                # Fallback: create new LoRARequest each step (slow but correct)
                return
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
            warnings.warn(
                "vLLM engine not available for modules_to_save sync. "
                "Will sync on next call after engine creation."
            )
            return

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
            engine = getattr(obj, "vllm_engine", None) if obj is not None else None
            if engine is not None:
                try:
                    return engine.llm_engine.model_executor.driver_worker.model_runner.model
                except AttributeError:
                    continue
        return None

    def flush_kv_cache(self) -> None:
        """Flush vLLM KV cache to reclaim VRAM without destroying the engine.

        Does NOT delete vllm_engine, fast_generate, or NCCL state.
        Only flushes KV cache pages via scheduler + zeros GPU cache tensors.
        """
        engine = (
            getattr(self._model, "vllm_engine", None)
            or getattr(getattr(self._model, "model", None), "vllm_engine", None)
        )
        if engine is None:
            return

        # Flush KV cache blocks via scheduler
        try:
            llm_engine = engine.llm_engine
            schedulers = getattr(llm_engine, "scheduler", [llm_engine.scheduler]) \
                if hasattr(llm_engine, "scheduler") else []
            for scheduler in schedulers:
                if hasattr(scheduler, "free_finished_seqs"):
                    scheduler.free_finished_seqs()
                if hasattr(scheduler, "block_manager"):
                    bm = scheduler.block_manager
                    if hasattr(bm, "gpu_allocator") and hasattr(bm.gpu_allocator, "free_all"):
                        bm.gpu_allocator.free_all()
        except Exception:
            pass

        # Zero GPU cache tensors directly
        try:
            worker = engine.llm_engine.model_executor.driver_worker
            if hasattr(worker, "gpu_cache"):
                for layer_cache in worker.gpu_cache:
                    if isinstance(layer_cache, torch.Tensor):
                        layer_cache.zero_()
                    elif isinstance(layer_cache, (list, tuple)):
                        for t in layer_cache:
                            if isinstance(t, torch.Tensor):
                                t.zero_()
        except Exception:
            pass

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
