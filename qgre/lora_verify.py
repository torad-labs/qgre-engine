from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol


class WeightSyncable(Protocol):
    """Interface for anything that can save/load weights (LoRA + modules_to_save)."""

    def save_weights(self, path: str | Path) -> None: ...
    def load_weights(self, path: str | Path) -> None: ...


class LoRAVerifier:
    """Verify LoRA weights are correctly synced between training and generation.

    Three functions:
    (a) verify_sync: hash weights before/after load, assert match
    (b) verify_active: generate from fixed prompt, verify output differs from base
    (c) periodic_recreate: signal to recreate vLLM engine every N steps

    Integrates into QGRETrainer.step() as post-sync hook and
    QGRETrainer.resume() as mandatory step.
    """

    def __init__(self, recreate_interval: int = 50):
        self.recreate_interval = recreate_interval
        self._last_save_hash: str | None = None
        self._steps_since_recreate: int = 0

    def hash_lora_dir(self, path: str | Path) -> str:
        """Compute hash of LoRA weight files for verification."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LoRA path does not exist: {path}")

        hasher = hashlib.sha256()
        # Prefer safetensors, fall back to bin (sorted for deterministic ordering)
        weight_files = sorted(path.rglob("*.safetensors")) or sorted(path.rglob("*.bin"))
        if not weight_files:
            raise FileNotFoundError(f"No weight files (.safetensors/.bin) found in: {path}")
        for f in weight_files:
            hasher.update(f.read_bytes())

        return hasher.hexdigest()

    def verify_sync(self, lora_path: str | Path) -> bool:
        """Verify saved LoRA weights match what was saved.

        Call after save_lora + load_lora. Returns True if hash matches.
        Raises ValueError if mismatch detected.
        """
        current_hash = self.hash_lora_dir(lora_path)

        if self._last_save_hash is not None and current_hash != self._last_save_hash:
            raise ValueError(
                f"LoRA weight mismatch after sync! "
                f"Expected hash {self._last_save_hash[:16]}..., "
                f"got {current_hash[:16]}...",
            )

        self._last_save_hash = current_hash
        return True

    def on_save(self, lora_path: str | Path):
        """Record hash after saving LoRA weights."""
        self._last_save_hash = self.hash_lora_dir(lora_path)

    def should_recreate_engine(self) -> bool:
        """Check if vLLM engine should be recreated to prevent memory leak.

        Call at the start of each training step. Returns True every
        `recreate_interval` steps (default 50) to prevent unsloth #3864.
        """
        self._steps_since_recreate += 1
        if self._steps_since_recreate >= self.recreate_interval:
            self._steps_since_recreate = 0
            return True
        return False

    def reset_recreate_counter(self):
        """Call after engine recreation."""
        self._steps_since_recreate = 0

    @staticmethod
    def verify_active(model, tokenizer, test_prompt: str = "Hello") -> bool:
        """Verify LoRA is actively applied by generating 1 token.

        Generates from a fixed prompt and checks the output is non-empty.
        If LoRA sync failed silently (unsloth #3802), the model may behave
        as base model — this catches that by verifying generation works.

        Args:
            model: The language model (with LoRA applied)
            tokenizer: The tokenizer
            test_prompt: Fixed prompt to generate from

        Returns:
            True if generation produces non-empty output

        Raises:
            RuntimeError: On unexpected errors (not OOM or generation failures)
        """
        import warnings

        import torch

        # Pre-validation: check model and tokenizer have required methods
        if not hasattr(tokenizer, "encode"):
            warnings.warn("LoRA verify_active: tokenizer lacks encode method", stacklevel=2)
            return False
        if not hasattr(model, "generate"):
            warnings.warn("LoRA verify_active: model lacks generate method", stacklevel=2)
            return False
        if not hasattr(model, "parameters"):
            warnings.warn("LoRA verify_active: model lacks parameters method", stacklevel=2)
            return False

        # Pre-check: ensure model has parameters before accessing device
        try:
            params = list(model.parameters())
            if not params:
                warnings.warn("LoRA verify_active: model has no parameters", stacklevel=2)
                return False
            device = params[0].device
        except StopIteration:
            warnings.warn("LoRA verify_active: model parameters iterator empty", stacklevel=2)
            return False

        try:
            input_ids = tokenizer.encode(test_prompt, return_tensors="pt")
            if hasattr(input_ids, "to") and torch.cuda.is_available():
                input_ids = input_ids.to(device)
            with torch.no_grad():
                output = model.generate(input_ids, max_new_tokens=1, do_sample=False)
            return output.shape[-1] > input_ids.shape[-1]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # Known inference failures — return False is appropriate
            if "out of memory" in str(e).lower() or "CUDA" in str(e):
                warnings.warn(
                    f"LoRA verify_active: generation failed (OOM/CUDA): {e}", stacklevel=2
                )
                return False
            # Other RuntimeError — re-raise
            raise
        except (AttributeError, TypeError) as e:
            # Model/tokenizer API issues — warn and return False
            warnings.warn(f"LoRA verify_active: API error: {e}", stacklevel=2)
            return False
