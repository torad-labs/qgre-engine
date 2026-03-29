from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from qgre.config import GenerationConfig, ModelConfig
from qgre.weight_bus import SyncStrategy
from qgre.weight_export import WeightExporter
from qgre.weight_load import WeightLoader

# Modules trained as full weights alongside LoRA (not low-rank decomposed)
MODULES_TO_SAVE = ["lm_head", "embed_tokens"]


@dataclass
class GenerationOutput:
    """Output from a single generation call."""

    token_ids: list[list[int]]   # [batch_size] × variable length
    texts: list[str]             # decoded completions
    logprobs: list[list[float]] | None = None  # [batch_size] × [seq_len] per-token log probs from generation


class UnslothBackend:
    """Unsloth FastLanguageModel + vLLM fast_generate backend.

    Implements the GenerationBackend protocol from trainer.py.
    Single-GPU, single-process. No Ray.
    """

    def __init__(self, model_config: ModelConfig, generation_config: GenerationConfig, max_prompt_length: int = 3200):
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
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_config.path,
            max_seq_length=self.generation_config.max_tokens + self.max_prompt_length,
            load_in_4bit=self.model_config.load_in_4bit,
            fast_inference=self.model_config.fast_inference,
            gpu_memory_utilization=self.model_config.gpu_memory_utilization,
            max_lora_rank=self.model_config.max_lora_rank or self.model_config.lora_rank,
            fix_tokenizer=False,  # We configure tokenizer from our config, not Unsloth defaults
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=self.model_config.lora_rank,
            lora_alpha=self.model_config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            modules_to_save=MODULES_TO_SAVE,
            lora_dropout=0.0,
            use_gradient_checkpointing="unsloth",
        )

        # Null modules_to_save in PEFT config object so that when Unsloth's load_lora
        # bootstraps adapter_config.json (on first call, lora_request_id==1), it writes
        # the config WITHOUT modules_to_save. vLLM rejects non-null modules_to_save.
        # We handle modules_to_save ourselves via direct tensor copy (_sync_modules_to_save).
        peft_cfg = model.peft_config.get("default")
        if peft_cfg and getattr(peft_cfg, "modules_to_save", None):
            peft_cfg.modules_to_save = None

        # PEFT should untie embeddings when both lm_head and embed_tokens are in
        # modules_to_save (PEFT PR #2777). Verify — if tied, they share storage and
        # training one silently overwrites the other.
        if getattr(model.config, "tie_word_embeddings", False):
            import warnings
            warnings.warn(
                "tie_word_embeddings=True after get_peft_model with modules_to_save. "
                "PEFT should have untied them. Setting to False manually."
            )
            model.config.tie_word_embeddings = False

        # Configure tokenizer from YAML config — single source of truth for all models.
        # fix_tokenizer=False above prevents Unsloth from overriding with <|PAD_TOKEN|>.
        pad_token = self.model_config.pad_token
        pad_token_id = self.model_config.pad_token_id
        if not pad_token or pad_token_id < 0:
            raise ValueError(
                "model.pad_token and model.pad_token_id must be set in YAML config. "
                "Qwen3 example: pad_token='<|fim_pad|>', pad_token_id=151662"
            )
        tokenizer.pad_token = pad_token
        tokenizer.pad_token_id = pad_token_id
        model.config.pad_token_id = pad_token_id

        # Validate PAD token
        resolved_id = tokenizer.convert_tokens_to_ids(pad_token)
        assert resolved_id == pad_token_id, \
            f"{pad_token!r} resolves to {resolved_id}, not {pad_token_id} — wrong model or tokenizer"
        vocab_size = getattr(model.config, "vocab_size", None)
        if vocab_size is not None:
            assert pad_token_id < vocab_size, \
                f"PAD token ID {pad_token_id} >= vocab_size {vocab_size} — token doesn't exist"
        assert tokenizer.pad_token_id != tokenizer.eos_token_id, \
            f"PAD ({tokenizer.pad_token_id}) == EOS ({tokenizer.eos_token_id}) — model will never learn to stop"
        assert tokenizer.pad_token_id not in self.generation_config.stop_token_ids, \
            f"PAD ({tokenizer.pad_token_id}) is a stop token — loss will mask stop signals"

        print(f"Tokenizer: PAD={tokenizer.pad_token!r} (ID:{tokenizer.pad_token_id}), "
              f"EOS={tokenizer.eos_token!r} (ID:{tokenizer.eos_token_id}), "
              f"Stop tokens: {self.generation_config.stop_token_ids}")

        self.model = model
        self.tokenizer = tokenizer
        self._FastLanguageModel = FastLanguageModel
        self.weight_loader = WeightLoader(model)

        # Patch vLLM max_logprobs: Unsloth sets max_logprobs=0 by default,
        # but LLDS needs logprobs=1. Find the vLLM engine and set max_logprobs.
        try:
            llm = getattr(model, '_vllm_engine', None) or getattr(model, 'llm', None)
            if llm is None:
                # Unsloth stores the LLM on the model — traverse attributes
                for attr_name in dir(model):
                    attr = getattr(model, attr_name, None)
                    if hasattr(attr, 'llm_engine'):
                        llm = attr
                        break
            if llm is not None:
                engine = getattr(llm, 'llm_engine', llm)
                if hasattr(engine, 'model_config'):
                    engine.model_config.max_logprobs = 5
                    import warnings
                    warnings.warn("Patched vLLM max_logprobs=5 for LLDS logprob extraction")
        except Exception as e:
            import warnings
            warnings.warn(f"Could not patch vLLM max_logprobs: {e}. LLDS may not receive logprobs.")

        return model, tokenizer

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
        **kwargs,
    ) -> GenerationOutput:
        """Generate completions using vLLM fast_generate.

        Args:
            input_ids: [batch, prompt_len] — left-padded prompt tokens
            attention_mask: [batch, prompt_len]

        Returns:
            GenerationOutput with token IDs and decoded texts
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.generation_config.temperature,
            top_p=self.generation_config.top_p,
            top_k=self.generation_config.top_k if self.generation_config.top_k > 0 else -1,
            min_p=self.generation_config.min_p,
            max_tokens=self.generation_config.max_tokens,
            stop_token_ids=self.generation_config.stop_token_ids,
            logprobs=1,  # Return chosen token logprob at each position (for LLDS)
        )

        # Decode prompts for fast_generate (it takes text, not tensors)
        prompts = []
        for i in range(input_ids.shape[0]):
            mask = attention_mask[i].bool()
            tokens = input_ids[i][mask].tolist()
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            prompts.append(text)

        lora_req = self.weight_loader.lora_request if self.weight_loader else None
        outputs = self.model.fast_generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_req,
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
            if completion_out.logprobs is not None and len(completion_out.logprobs) > 0:
                if len(completion_out.logprobs) != len(completion_ids):
                    import warnings
                    warnings.warn(
                        f"vLLM logprobs length ({len(completion_out.logprobs)}) != "
                        f"completion length ({len(completion_ids)}) for prompt {idx}. "
                        f"Discarding logprobs for this batch."
                    )
                    all_logprobs = []
                    break
                for t, pos_dict in enumerate(completion_out.logprobs):
                    if not pos_dict:
                        import warnings
                        warnings.warn(
                            f"vLLM returned empty logprobs at position {t} for prompt {idx}. "
                            f"Discarding logprobs for this batch."
                        )
                        sample_lps = []
                        break
                    # Extract by the actual sampled token_id — NOT by dict iteration order.
                    # With temperature > 0, the sampled token may differ from top-1.
                    sampled_id = completion_ids[t]
                    if sampled_id in pos_dict:
                        sample_lps.append(pos_dict[sampled_id].logprob)
                    else:
                        # vLLM should always include sampled token with logprobs >= 1.
                        # If missing, fall back to first entry but warn.
                        import warnings
                        warnings.warn(
                            f"Sampled token {sampled_id} not in logprobs dict at position {t} "
                            f"(keys: {list(pos_dict.keys())}). Using first entry."
                        )
                        sample_lps.append(next(iter(pos_dict.values())).logprob)
            all_logprobs.append(sample_lps)

        has_logprobs = all(len(lps) > 0 for lps in all_logprobs) if all_logprobs else False
        if not has_logprobs and any(len(lps) > 0 for lps in all_logprobs):
            import warnings
            warnings.warn(
                f"Partial logprobs: {sum(1 for lps in all_logprobs if len(lps) > 0)}"
                f"/{len(all_logprobs)} samples have logprobs. LLDS disabled for this batch."
            )
        return GenerationOutput(
            token_ids=token_ids,
            texts=texts,
            logprobs=all_logprobs if has_logprobs else None,
        )

    # Weight sync methods moved to Weight Sync Bus:
    # - WeightExporter (qgre/weight_export.py)
    # - WeightBus (qgre/weight_bus.py)
    # - WeightLoader (qgre/weight_load.py)
    # Trainer calls weight_bus.sync() instead of backend.save_weights/load_weights.
