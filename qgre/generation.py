from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import torch

from qgre.weight_bus import SyncStrategy
from qgre.weight_export import WeightExporter
from qgre.weight_load import WeightLoader


if TYPE_CHECKING:
    from qgre.config import GenerationConfig, ModelConfig


@dataclass
class GenerationOutput:
    """Output from a single generation call."""

    token_ids: list[list[int]]  # [batch_size] × variable length
    texts: list[str]  # decoded completions
    logprobs: list[list[float]] | None = (
        None  # [batch_size] × [seq_len] per-token log probs from generation
    )
    hints_used: list[dict[str, bool]] | None = (
        None  # [batch_size] — {span_id: was_injected} per sample
    )


class HintInjector(Protocol):
    """Protocol for domain-specific hint injection.

    Implementations extract hint text from ground truth/metadata and format it
    for injection into the prompt. The hint guides the model toward the correct
    answer for a specific span without giving away the complete solution.
    """

    def extract_hint(
        self,
        span_id: str,
        metadata: dict[str, Any],
    ) -> str | None:
        """Extract hint text for a span from metadata.

        Args:
            span_id: Span identifier (e.g., "STEP_1", "STEP_2").
            metadata: Prompt metadata containing ground truth.

        Returns:
            Hint text to inject, or None if no hint available.
        """
        ...


def make_hamiltonian_hint_injector() -> HintInjector:
    """Create hint injector for Hamiltonian mechanics domain.

    Extracts hints from ground truth expressions in metadata.
    Hint format: "Hint for {label}: {start_of_expression}"
    """

    class HamiltonianHintInjector:
        # Map step IDs to Hamiltonian labels
        STEP_TO_LABEL = {
            "STEP_1": "COORDINATES",
            "STEP_2": "MOMENTUM",
            "STEP_3": "KINETIC",
            "STEP_4": "POTENTIAL",
            "STEP_5": "HAMILTONIAN",
        }

        def extract_hint(
            self,
            span_id: str,
            metadata: dict[str, Any],
        ) -> str | None:
            label = self.STEP_TO_LABEL.get(span_id)
            if not label:
                return None

            # Try to get ground truth from metadata
            gt = metadata.get("ground_truth", {})
            if isinstance(gt, str):
                import json

                try:
                    gt = json.loads(gt)
                except json.JSONDecodeError:
                    return None

            # Look for the expression in ground truth
            expr = gt.get(label.lower()) or gt.get(label)
            if not expr:
                return None

            # Extract first few characters as hint (up to equals or first term)
            hint_text = str(expr)
            # Truncate to reasonable hint length (don't give away full answer)
            if len(hint_text) > 20:
                # Find a good truncation point
                for i, c in enumerate(hint_text[5:], start=5):
                    if c in "+-*/^" and i > 5:
                        hint_text = hint_text[:i]
                        break
                else:
                    hint_text = hint_text[:15] + "..."

            return f"Hint for {label}: start with {hint_text}"

    return HamiltonianHintInjector()


class UnslothBackend:
    """Unsloth FastLanguageModel + vLLM fast_generate backend.

    Implements the GenerationBackend protocol from trainer.py.
    Single-GPU, single-process. No Ray.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        generation_config: GenerationConfig,
        max_prompt_length: int = 3200,
    ):
        self.model_config = model_config
        self.generation_config = generation_config
        self.max_prompt_length = max_prompt_length
        self.model = None
        self.tokenizer = None
        self.weight_exporter = WeightExporter()
        self.weight_loader: WeightLoader | None = None  # Created after model load
        self._sync_strategy = SyncStrategy(model_config.weight_sync_strategy)

    def load(self) -> tuple[Any, Any]:
        """Load model and tokenizer. Returns (model, tokenizer)."""
        import unsloth.models.llama as _llama_mod
        from unsloth import FastLanguageModel

        # Prevent Unsloth's patch_tokenizer from modifying our tokenizer/model.
        # It runs unconditionally (ignoring fix_tokenizer) and would:
        # 1. Add <|PAD_TOKEN|> via add_special_tokens (may resize embeddings)
        # 2. Set model.config.pad_token_id to wrong value
        # 3. Set model.generation_config.pad_token_id to wrong value
        # We configure pad_token from YAML config below (single source of truth).
        _orig_patch = _llama_mod.patch_tokenizer
        _llama_mod.patch_tokenizer = lambda model, tokenizer: (model, tokenizer)

        # Suppress vLLM's env var validation during from_pretrained.
        # Unsloth sets VLLM_ATTENTION_BACKEND inside from_pretrained, but vLLM V1
        # no longer reads it (uses internal config) and warns "Unknown vLLM environment
        # variable". The seed=0 warning is inherent to VLLM_ENABLE_V1_MULTIPROCESSING=0
        # which Unsloth requires for weight access. Both are expected and non-actionable
        # during init — we restore random state after load.
        import vllm.envs as _vllm_envs

        _orig_validate = _vllm_envs.validate_environ
        _vllm_envs.validate_environ = lambda hard_fail=False: None
        # Also patch on arg_utils module where it's imported by name
        from vllm.engine import arg_utils as _arg_utils

        if hasattr(_arg_utils, "envs"):
            _arg_utils.envs.validate_environ = lambda hard_fail=False: None
        # Suppress seed=0 warning from arg_utils (inherent to MULTIPROCESSING=0)
        import logging

        _arg_utils_logger = logging.getLogger("vllm.engine.arg_utils")
        _arg_utils_level = _arg_utils_logger.level
        _arg_utils_logger.setLevel(logging.ERROR)

        try:
            max_lora_rank_val = self.model_config.max_lora_rank or self.model_config.lora_rank
            lora_rank_val = self.model_config.lora_rank
            if lora_rank_val > max_lora_rank_val:
                raise ValueError(
                    f"lora_rank ({lora_rank_val}) exceeds max_lora_rank ({max_lora_rank_val}). "
                    "vLLM will silently ignore the adapter. Increase max_lora_rank in config.",
                )
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_config.path,
                max_seq_length=self.generation_config.max_tokens + self.max_prompt_length,
                load_in_4bit=self.model_config.load_in_4bit,
                fast_inference=self.model_config.fast_inference,
                gpu_memory_utilization=self.model_config.gpu_memory_utilization,
                max_lora_rank=max_lora_rank_val,
            )
        finally:
            _llama_mod.patch_tokenizer = _orig_patch
            _vllm_envs.validate_environ = _orig_validate
            if hasattr(_arg_utils, "envs"):
                _arg_utils.envs.validate_environ = _orig_validate
            _arg_utils_logger.setLevel(_arg_utils_level)

        # Clean up VLLM_ATTENTION_BACKEND env var that Unsloth sets.
        # Not needed for vLLM V1 (internal config), prevents future warnings.
        import os

        os.environ.pop("VLLM_ATTENTION_BACKEND", None)

        # Untie word embeddings BEFORE get_peft_model to prevent PEFT from
        # warning about tie_word_embeddings + modules_to_save (PEFT #2777).
        # If both lm_head and embed_tokens are in modules_to_save (not the default),
        # they must be independent — tied storage means training one silently
        # overwrites the other. Default is ["lm_head"] only (Phase 2 optimization).
        #
        # Three things must happen:
        # 1. Set config flag (controls future model behavior)
        # 2. Clear _tied_weights_keys (PEFT reads this class attr, ignores config)
        # 3. Actually untie the weight tensors (clone lm_head.weight so it's independent)
        if getattr(model.config, "tie_word_embeddings", False):
            model.config.tie_word_embeddings = False
        # PEFT checks _tied_weights_keys (class attr on Qwen3ForCausalLM) regardless
        # of tie_word_embeddings config flag. Clear it on the instance to suppress.
        if getattr(model, "_tied_weights_keys", None):
            model._tied_weights_keys = []
        # Ensure lm_head.weight is physically independent from embed_tokens.weight
        lm_head = getattr(model, "lm_head", None)
        embed = getattr(model.model, "embed_tokens", None) if hasattr(model, "model") else None
        if lm_head is not None and embed is not None:
            if lm_head.weight.data_ptr() == embed.weight.data_ptr():
                lm_head.weight = torch.nn.Parameter(lm_head.weight.clone())
                model.config.tie_word_embeddings = False

        # Suppress expected warnings during PEFT adapter initialization:
        # - "should probably TRAIN" (logging.warning from transformers) — modules_to_save
        #   creates random adapter copies. We ARE about to train.
        # - "tie_word_embeddings" (warnings.warn from PEFT) — already handled above.
        import logging
        import warnings

        _tf_logger = logging.getLogger("transformers.modeling_utils")
        _tf_level = _tf_logger.level
        _tf_logger.setLevel(logging.ERROR)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
            model = FastLanguageModel.get_peft_model(
                model,
                r=self.model_config.lora_rank,
                lora_alpha=self.model_config.lora_alpha,
                target_modules=self.model_config.lora_target_modules,
                modules_to_save=self.model_config.modules_to_save,
                lora_dropout=0.0,
                use_gradient_checkpointing="unsloth",
            )
        _tf_logger.setLevel(_tf_level)

        # Null modules_to_save in PEFT config object so that when Unsloth's load_lora
        # bootstraps adapter_config.json (on first call, lora_request_id==1), it writes
        # the config WITHOUT modules_to_save. vLLM rejects non-null modules_to_save.
        # We handle modules_to_save ourselves via direct tensor copy (_sync_modules_to_save).
        peft_cfg = model.peft_config.get("default")
        if peft_cfg and getattr(peft_cfg, "modules_to_save", None):
            peft_cfg.modules_to_save = None

        # Configure tokenizer from YAML config — single source of truth for all models.
        # fix_tokenizer=False above prevents Unsloth from overriding with <|PAD_TOKEN|>.
        pad_token = self.model_config.pad_token
        pad_token_id = self.model_config.pad_token_id
        if not pad_token or pad_token_id < 0:
            raise ValueError(
                "model.pad_token and model.pad_token_id must be set in YAML config. "
                "Qwen3 example: pad_token='<|fim_pad|>', pad_token_id=151662",
            )
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = pad_token_id
        model.config.pad_token_id = pad_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = pad_token_id

        # Validate PAD token
        resolved_id = tokenizer.convert_tokens_to_ids(pad_token)
        assert resolved_id == pad_token_id, (
            f"{pad_token!r} resolves to {resolved_id}, not {pad_token_id} — wrong model or tokenizer"
        )
        vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is not None:
            assert pad_token_id < vocab_size, (
                f"PAD token ID {pad_token_id} >= vocab_size {vocab_size} — token doesn't exist"
            )
        assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
            f"PAD ({tokenizer.pad_token_id}) == EOS ({tokenizer.eos_token_id}) — model will never learn to stop"
        )
        assert tokenizer.pad_token_id not in self.generation_config.stop_token_ids, (
            f"PAD ({tokenizer.pad_token_id}) is a stop token — loss will mask stop signals"
        )

        print(
            f"Tokenizer: PAD={tokenizer.pad_token!r} (ID:{tokenizer.pad_token_id}), "
            f"EOS={tokenizer.eos_token!r} (ID:{tokenizer.eos_token_id}), "
            f"Stop tokens: {self.generation_config.stop_token_ids}"
        )

        self.model = model
        self.tokenizer = tokenizer
        self._FastLanguageModel = FastLanguageModel
        self.weight_loader = WeightLoader(model)

        # WS3-005: Log warning if patch fails, don't fail silently
        # Patch vLLM max_logprobs: Unsloth sets max_logprobs=0 by default,
        # but LLDS needs logprobs=1. Find the vLLM engine and set max_logprobs.
        try:
            llm = getattr(model, "_vllm_engine", None) or getattr(model, "llm", None)
            if llm is None:
                # Unsloth stores the LLM on the model — traverse attributes
                for attr_name in dir(model):
                    attr = getattr(model, attr_name, None)
                    if hasattr(attr, "llm_engine"):
                        llm = attr
                        break
            if llm is not None:
                engine = getattr(llm, "llm_engine", llm)
                if hasattr(engine, "model_config"):
                    engine.model_config.max_logprobs = self.generation_config.max_logprobs
                    print(
                        f"vLLM max_logprobs patched to {self.generation_config.max_logprobs} for LLDS logprob extraction"
                    )
                else:
                    import warnings

                    warnings.warn(
                        "WS3-005: vLLM max_logprobs patch failed: engine has no model_config attribute. "
                        "LLDS requires max_logprobs >= logprobs. Check vLLM version compatibility.",
                        stacklevel=2,
                    )
                    raise RuntimeError(
                        "vLLM max_logprobs patch failed: engine has no model_config attribute. "
                        "LLDS requires max_logprobs >= logprobs. Check vLLM version compatibility.",
                    )
            else:
                import warnings

                warnings.warn(
                    "WS3-005: vLLM max_logprobs patch failed: could not find vLLM engine. "
                    "LLDS requires max_logprobs >= logprobs. Check Unsloth integration.",
                    stacklevel=2,
                )
                raise RuntimeError(
                    "vLLM max_logprobs patch failed: could not find vLLM engine. "
                    "LLDS requires max_logprobs >= logprobs. Check Unsloth integration.",
                )
        except Exception as e:
            import warnings

            warnings.warn(f"WS3-005: vLLM max_logprobs patch failed: {e}", stacklevel=2)
            if self.generation_config.max_logprobs > 0:
                raise RuntimeError(
                    f"Could not patch vLLM max_logprobs: {e}. "
                    "LLDS is enabled but vLLM cannot return sufficient logprobs for all continuation tokens.",
                ) from e

        return model, tokenizer

    def restore_random_state(self, seed: int = -1) -> None:
        """Restore global random state after vLLM init.

        vLLM's gpu_worker calls set_random_seed(0) during init, which resets
        random/numpy/torch global state in our process (VLLM_ENABLE_V1_MULTIPROCESSING=0).
        This makes all training runs identically seeded from 0 — same dropout masks,
        same data ordering if using global state. Call this after load() to restore
        proper stochastic behavior.

        Args:
            seed: -1 = time-based (non-reproducible), 0+ = fixed seed.
        """
        import random
        import time

        import numpy as np

        if seed < 0:
            seed = int(time.time() * 1000) % (2**32)

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random state restored: seed={seed}")

    def set_training_mode(self):
        """Switch to training mode — disables Unsloth inplace optimizations.

        Must call before forward+backward. Fixes inplace op autograd error.
        Ref: unsloth #895, #2434
        """
        self._FastLanguageModel.for_training(self.model)

    def set_inference_mode(self):
        """Switch to inference mode — enables Unsloth fast kernels.

        Must call before fast_generate.
        """
        self._FastLanguageModel.for_inference(self.model)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_hints: dict[int, dict[str, str]] | None = None,
        **kwargs,
    ) -> GenerationOutput:
        """Generate completions using vLLM fast_generate.

        Args:
            input_ids: [batch, prompt_len] — left-padded prompt tokens
            attention_mask: [batch, prompt_len]
            prompt_hints: Optional hints per prompt index.
                Format: {batch_idx: {span_id: hint_text, ...}, ...}
                Hints are appended to the prompt as guidance text.

        Returns:
            GenerationOutput with token IDs, decoded texts, and hints_used flags
        """
        from vllm import SamplingParams

        requested_logprobs = 1  # Return chosen token logprob at each position (for LLDS)

        # GB3-001: Define llds_enabled from generation_config.max_logprobs > 0
        llds_enabled = self.generation_config.max_logprobs > 0

        # Verify max_logprobs >= requested_logprobs before generation
        if hasattr(self.model, "vllm_engine") or hasattr(
            getattr(self.model, "model", None), "vllm_engine"
        ):
            engine = getattr(self.model, "vllm_engine", None) or getattr(
                getattr(self.model, "model", None), "vllm_engine", None
            )
            if engine is not None:
                llm_engine = getattr(engine, "llm_engine", engine)
                if hasattr(llm_engine, "model_config"):
                    max_lp = getattr(llm_engine.model_config, "max_logprobs", 0)
                    if max_lp == 0 and llds_enabled:
                        raise ValueError(
                            "max_logprobs=0 with LLDS enabled. LLDS requires max_logprobs >= 1. "
                            "Set max_logprobs in vLLM engine config."
                        )
                    if max_lp < requested_logprobs:
                        # GB2-001: Raise error instead of warning if LLDS enabled and max_logprobs insufficient
                        if llds_enabled:
                            raise ValueError(
                                f"max_logprobs={max_lp} < requested logprobs={requested_logprobs}. "
                                "LLDS requires max_logprobs >= 1. Set max_logprobs in vLLM engine config.",
                            )
                        import warnings

                        warnings.warn(
                            f"max_logprobs={max_lp} < requested logprobs={requested_logprobs}. "
                            "vLLM will truncate logprobs output.",
                            stacklevel=2,
                        )

        temperature = self.generation_config.temperature
        top_p = self.generation_config.top_p
        top_k = self.generation_config.top_k if self.generation_config.top_k > 0 else -1
        if temperature == 0:
            top_p = 1.0
            top_k = -1
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=self.generation_config.min_p,
            max_tokens=self.generation_config.max_tokens,
            repetition_penalty=self.generation_config.repetition_penalty,
            stop_token_ids=self.generation_config.stop_token_ids,
            logprobs=requested_logprobs,
        )

        # Decode prompts for fast_generate (it takes text, not tensors)
        prompts = []
        hints_used = []
        for i in range(input_ids.shape[0]):
            # GN5: Validate attention_mask shape matches input_ids
            if attention_mask.shape != input_ids.shape:
                raise RuntimeError(
                    f"GN5: attention_mask shape {attention_mask.shape} != input_ids shape {input_ids.shape}. "
                    "Check dataloader output.",
                )
            mask = attention_mask[i].bool()
            tokens = input_ids[i][mask].tolist()
            prompt_token_count = len(tokens)  # Use original token count instead of round-trip
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)  # type: ignore[union-attr]
            # Force disable thinking mode: append </think> if not already present
            # This ensures model generates direct answers, not <think> blocks
            if "</think>" not in text:
                template = "<think>\n</think>\n\n"
                text_with_template = text.rstrip() + template
                # Validate total length after template injection using original token count
                template_token_count = len(self.tokenizer.encode(template, add_special_tokens=False))  # type: ignore[union-attr]
                total_tokens = prompt_token_count + template_token_count
                max_seq_length = self.max_prompt_length + self.generation_config.max_tokens
                if total_tokens > max_seq_length:
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Thinking template injection exceeds budget ({total_tokens} > {max_seq_length}). "
                        f"Truncating prompt to fit within budget.",
                    )
                    # Truncate original text to make room for template
                    budget_for_text = max_seq_length - template_token_count
                    if budget_for_text > 0:
                        truncated_tokens = self.tokenizer.encode(text, add_special_tokens=False)[:budget_for_text]  # type: ignore[union-attr]
                        text = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)  # type: ignore[union-attr]
                        text = text.rstrip() + template
                    else:
                        # Template itself exceeds budget - skip template
                        logging.getLogger(__name__).warning(
                            f"Thinking template itself exceeds budget. Skipping template injection.",
                        )
                else:
                    text = text_with_template

            # Inject hints if available for this prompt (EGRS Phase 5)
            hints_used_dict: dict[str, bool] = {}
            if prompt_hints and i in prompt_hints:
                sample_hints = prompt_hints[i]
                if sample_hints:
                    # Format hints as guidance text appended to prompt
                    hint_lines = []
                    injected_span_ids = []
                    for span_id, hint in sample_hints.items():
                        if hint:
                            # MIO-003: Decode and validate hint is not empty
                            try:
                                hint_str = str(hint).strip()
                                if hint_str:
                                    hint_lines.append(f"[{hint_str}]")
                                    injected_span_ids.append(span_id)
                                    # H-2: Don't mark as used yet - wait until after overflow check
                                else:
                                    import logging

                                    logging.getLogger(__name__).warning(
                                        f"MIO-003: Empty hint text after decoding for span {span_id} in sample {i}",
                                    )
                                    hints_used_dict[span_id] = False
                            except (UnicodeDecodeError, ValueError) as e:
                                import logging

                                logging.getLogger(__name__).warning(
                                    f"MIO-003: Failed to decode hint for span {span_id} in sample {i}: {e}. Skipping hint.",
                                )
                                hints_used_dict[span_id] = False
                        else:
                            hints_used_dict[span_id] = False
                    if hint_lines:
                        hint_text = "\n".join(hint_lines)
                        combined_text = text.rstrip() + f"\n\n{hint_text}\n\n"
                        # GN3: Check if combined length would exceed max_seq_length (prompt + generation budget)
                        combined_tokens = self.tokenizer.encode(  # type: ignore[union-attr]
                            combined_text, add_special_tokens=False
                        )
                        max_seq_length = self.max_prompt_length + self.generation_config.max_tokens
                        if len(combined_tokens) > max_seq_length:
                            import logging

                            logging.getLogger(__name__).warning(
                                f"GN3: Hint injection would exceed max_seq_length ({len(combined_tokens)} > {max_seq_length}). "
                                "Skipping hint injection for this sample.",
                            )
                            # H-2: Mark all hints as NOT used since injection was skipped
                            for span_id in injected_span_ids:
                                hints_used_dict[span_id] = False
                        else:
                            text = combined_text
                            # H-2: NOW mark hints as successfully used (AFTER successful injection)
                            for span_id in injected_span_ids:
                                hints_used_dict[span_id] = True

            prompts.append(text)
            hints_used.append(hints_used_dict or {})

        lora_req = self.weight_loader.lora_request if self.weight_loader else None
        if lora_req is None:
            raise RuntimeError(
                "LoRA request is None — cannot generate with base model. "
                "Ensure weight_loader.load_lora() was called before generation.",
            )
        outputs = self.model.fast_generate(  # type: ignore[union-attr]
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_req,
        )

        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"vLLM output count mismatch: {len(outputs)} outputs != {len(prompts)} prompts"
            )
        token_ids = []
        texts = []
        all_logprobs = []
        for idx, output in enumerate(outputs):
            if not output.outputs:
                raise RuntimeError(f"vLLM returned no outputs for prompt {idx}")
            completion_out = output.outputs[0]
            completion_ids = completion_out.token_ids
            if len(completion_ids) == 0:
                raise RuntimeError(f"vLLM returned empty completion for prompt {idx}")
            token_ids.append(list(completion_ids))
            texts.append(completion_out.text)

            # Extract per-token logprobs: vLLM returns list[dict[token_id, Logprob]]
            # We need the SAMPLED token's logprob at each position (not top-1).
            # With logprobs=1, vLLM always includes the sampled token plus up to 1 top token.
            sample_lps = []
            if completion_out.logprobs is not None and len(completion_out.logprobs) > 0 and len(completion_ids) > 0:
                if len(completion_out.logprobs) != len(completion_ids):
                    import warnings

                    warnings.warn(
                        f"vLLM logprobs length ({len(completion_out.logprobs)}) != "
                        f"completion length ({len(completion_ids)}) for prompt {idx}. "
                        f"Setting this sample's logprobs to None (other samples may still have valid logprobs).",
                        stacklevel=2,
                    )
                    # G6: Don't skip appending — append None so indexing stays correct
                    all_logprobs.append(None)
                    continue
                # GB3-006: Filter None values before length check
                for t, pos_dict in enumerate(completion_out.logprobs):
                    if pos_dict is None:
                        import warnings

                        warnings.warn(
                            f"GB3-006: vLLM returned None logprobs at position {t} for prompt {idx}. "
                            f"Using -inf as placeholder.",
                            stacklevel=2,
                        )
                        sample_lps.append(float("-inf"))
                        continue
                    # GB2-002: Continue loop on empty dict, fill with None placeholder
                    if not pos_dict:
                        import warnings

                        warnings.warn(
                            f"vLLM returned empty logprobs at position {t} for prompt {idx}. "
                            f"Using -inf as placeholder (token was below cutoff).",
                            stacklevel=2,
                        )
                        sample_lps.append(float("-inf"))
                        continue
                    # Extract by the actual sampled token_id — NOT by dict iteration order.
                    # With temperature > 0, the sampled token may differ from top-1.
                    sampled_id = int(completion_ids[t])
                    if sampled_id in pos_dict:
                        sample_lps.append(pos_dict[sampled_id].logprob)
                    else:
                        # GEN-R1-3: vLLM should always include sampled token with logprobs >= 1.
                        # If missing, token was truncated from top-k — return -inf instead of wrong value.
                        import warnings

                        warnings.warn(
                            f"Sampled token {sampled_id} not in logprobs dict at position {t} "
                            f"(keys: {list(pos_dict.keys())}). Returning -inf (token was below top-k cutoff).",
                            stacklevel=2,
                        )
                        sample_lps.append(float("-inf"))
            # Handle empty logprobs or accept partial logprobs (first N tokens)
            if len(sample_lps) == 0:
                # Empty logprobs — append None to maintain indexing
                all_logprobs.append(None)
            elif len(sample_lps) < len(completion_ids):
                # Partial logprobs (e.g., stop token truncated) — accept first N
                import warnings
                warnings.warn(
                    f"Partial logprobs: {len(sample_lps)} < {len(completion_ids)} for prompt {idx}. "
                    f"Accepting first {len(sample_lps)} logprobs (stop token or early truncation).",
                    stacklevel=2,
                )
                # FIX #1: Filter None values before padding - don't extend with None
                # Trainer expects list[float], not list[float | None]
                all_logprobs.append(sample_lps)
            elif len(sample_lps) > len(completion_ids):
                # More logprobs than tokens — error condition
                raise RuntimeError(
                    f"Logprobs length mismatch: {len(sample_lps)} > {len(completion_ids)} for prompt {idx}"
                )
            else:
                all_logprobs.append(sample_lps)

        # G6: Support partial logprobs — return valid logprobs for samples that passed, None for failed
        # has_logprobs = True if ALL samples have logprobs (strict mode for LLDS), False otherwise
        has_logprobs = (
            all(lps is not None and len(lps) > 0 for lps in all_logprobs) if all_logprobs else False
        )
        if not has_logprobs and any(lps is not None and len(lps) > 0 for lps in all_logprobs):
            if llds_enabled:
                raise RuntimeError(
                    f"Partial logprobs with LLDS enabled: {sum(1 for lps in all_logprobs if lps is not None and len(lps) > 0)}"
                    f"/{len(all_logprobs)} samples have logprobs. LLDS requires consistent logprobs across batch."
                )
            import logging

            # DP-R2-07: Log at WARNING level instead of DEBUG
            logging.getLogger(__name__).warning(
                f"DP-R2-07: Partial logprobs: {sum(1 for lps in all_logprobs if lps is not None and len(lps) > 0)}"
                f"/{len(all_logprobs)} samples have logprobs. LLDS will be disabled for this batch.",
            )
        # MIO-006: Warn if hint_registry exists but hint_enabled is False
        if not prompt_hints and hasattr(self, "hint_registry") and self.hint_registry is not None:  # type: ignore[attr-defined]
            import logging

            logging.getLogger(__name__).warning(
                "MIO-006: hint_registry exists but hint_enabled is False. "
                "Failures will clear hints that are never injected.",
            )
        # R2-MIO-001: Return empty list [] instead of None for hints_used when no hints injected
        return GenerationOutput(
            token_ids=token_ids,
            texts=texts,
            logprobs=all_logprobs if has_logprobs else None,
            hints_used=hints_used or [],
        )

    # Weight sync methods moved to Weight Sync Bus:
    # - WeightExporter (qgre/weight_export.py)
    # - WeightBus (qgre/weight_bus.py)
    # - WeightLoader (qgre/weight_load.py)
    # Trainer calls weight_bus.sync() instead of backend.save_weights/load_weights.
