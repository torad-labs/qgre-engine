from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch


@dataclass
class PromptBatch:
    """A batch of tokenized, padded prompts ready for generation."""

    input_ids: torch.Tensor       # [batch_size, max_prompt_length] — left-padded
    attention_mask: torch.Tensor   # [batch_size, max_prompt_length]
    prompt_ids: list[int]          # hash IDs for SPO value tracker
    raw_prompts: list[str]         # original prompt text
    metadata: list[dict[str, Any]]  # ground_truth, extra_info, etc.


class QGREDataLoader:
    """DataLoader for QGRE training: parquet → tokenize → pad → batch → expand.

    Handles:
    - Load prompts from parquet or list of dicts
    - Apply chat template via tokenizer
    - Left-pad to max_prompt_length
    - Filter overlong prompts
    - Shuffle per epoch
    - Batch into train_batch_size chunks
    - Expand each prompt × n for rollout generation
    - Track epoch/step for resume
    """

    def __init__(
        self,
        prompts: list[dict[str, Any]],
        tokenizer: Any,
        max_prompt_length: int,
        train_batch_size: int,
        n_completions: int = 1,
        seed: int = 42,
        prompt_column: str = "prompt",
        metadata_columns: list[str] | None = None,
        system_prompt_column: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.train_batch_size = train_batch_size
        self.n_completions = n_completions
        self.seed = seed
        self.prompt_column = prompt_column
        self.metadata_columns = metadata_columns or []
        self.system_prompt_column = system_prompt_column

        # Tokenize and filter
        self.items = self._prepare(prompts)
        self.total_prompts = len(self.items)
        filtered = len(prompts) - self.total_prompts
        if filtered > 0:
            import warnings
            warnings.warn(
                f"Filtered {filtered}/{len(prompts)} prompts exceeding "
                f"max_prompt_length={max_prompt_length}"
            )
        if self.total_prompts == 0:
            raise ValueError(
                f"All {len(prompts)} prompts filtered by max_prompt_length={max_prompt_length}. "
                f"No training data. Increase max_prompt_length or check prompts."
            )

        # Epoch tracking
        self.epoch = 0
        self.step_in_epoch = 0
        self.total_steps = 0

        # Priority-weighted sampling (SPO paper Section 3.2)
        self._priorities: dict[int, float] | None = None
        # Difficulty-gated curriculum
        self._difficulty_gate: tuple[set[str], str] | None = None

    def _prepare(self, prompts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Tokenize, filter overlong, store results."""
        items = []
        for row in prompts:
            text = row[self.prompt_column]

            # Apply chat template if tokenizer supports it
            if hasattr(self.tokenizer, "apply_chat_template"):
                messages = []
                # Use separate system message if system_prompt_column is configured
                if self.system_prompt_column and row.get(self.system_prompt_column):
                    messages.append({"role": "system", "content": row[self.system_prompt_column]})
                messages.append({"role": "user", "content": text})

                # Qwen3 models support enable_thinking via **kwargs in their chat template.
                # We always want enable_thinking=False for training (no <think> blocks).
                # Since it's kwargs-passed, signature inspection won't detect it — we must
                # try the call and cache whether it succeeded.
                if not hasattr(self, "_enable_thinking_supported"):
                    self._enable_thinking_supported = None  # Unknown until first call

                if self._enable_thinking_supported is True:
                    token_ids = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        enable_thinking=False,
                    )
                elif self._enable_thinking_supported is False:
                    token_ids = self.tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                    )
                else:
                    # First call: try with enable_thinking=False, cache result
                    try:
                        token_ids = self.tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True,
                            enable_thinking=False,
                        )
                        self._enable_thinking_supported = True
                    except TypeError as e:
                        # Only catch if the error is specifically about enable_thinking
                        if "enable_thinking" not in str(e) and "unexpected keyword" not in str(e):
                            raise
                        self._enable_thinking_supported = False
                        token_ids = self.tokenizer.apply_chat_template(
                            messages, tokenize=True, add_generation_prompt=True,
                        )
                # transformers 5.x returns BatchEncoding, extract input_ids
                if hasattr(token_ids, "input_ids"):
                    token_ids = token_ids.input_ids
                    if isinstance(token_ids, list) and len(token_ids) == 1:
                        token_ids = token_ids[0]
            else:
                token_ids = self.tokenizer.encode(text)

            if len(token_ids) > self.max_prompt_length:
                continue  # Filter overlong

            metadata = {col: row.get(col) for col in self.metadata_columns}
            prompt_id = int.from_bytes(
                hashlib.sha256(text.encode()).digest()[:8], "big"
            )
            items.append({
                "token_ids": token_ids,
                "text": text,
                "prompt_id": prompt_id,
                "metadata": metadata,
            })

        return items

    def set_priorities(self, priorities: dict[int, float]):
        """Set per-prompt priority weights for prioritized sampling.

        Args:
            priorities: dict mapping prompt_id → priority weight (higher = sample more)
        """
        self._priorities = priorities

    def set_difficulty_gate(self, allowed_difficulties: set[str], difficulty_column: str = "difficulty"):
        """Gate prompts by difficulty: only prompts with matching difficulty get sampled.

        Prompts outside the allowed set get zero priority weight.
        Called by the trainer on phase advancement to gradually introduce harder problems.
        """
        self._difficulty_gate = (allowed_difficulties, difficulty_column)

    def _shuffle(self, epoch: int) -> list[dict[str, Any]]:
        """Deterministic shuffle per epoch, with optional priority-weighted sampling."""
        gen = torch.Generator()
        gen.manual_seed(self.seed + epoch)

        # Start with base weights
        weights = torch.tensor(
            [self._priorities.get(item["prompt_id"], 1.0) if self._priorities else 1.0
             for item in self.items],
            dtype=torch.float64,
        )

        # Apply difficulty gate: zero out prompts above the current phase
        if hasattr(self, "_difficulty_gate") and self._difficulty_gate is not None:
            allowed, col = self._difficulty_gate
            for i, item in enumerate(self.items):
                difficulty = item["metadata"].get(col, "")
                if difficulty not in allowed:
                    weights[i] = 0.0

        if weights.sum() == 0:
            # Fallback: if all weights are zero (misconfigured gate), use uniform
            weights = torch.ones(len(self.items), dtype=torch.float64)

        if self._priorities is not None or (hasattr(self, "_difficulty_gate") and self._difficulty_gate is not None):
            # Add epsilon only to non-zero weights (preserve hard zeros from difficulty gate)
            mask = weights > 0
            weights[mask] = weights[mask] + 1e-8
            weights = weights / weights.sum()
            indices = torch.multinomial(
                weights, len(self.items), replacement=True, generator=gen,
            ).tolist()
        else:
            indices = torch.randperm(len(self.items), generator=gen).tolist()

        return [self.items[i] for i in indices]

    def _left_pad(self, token_ids_list: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad a batch of token ID lists to max_prompt_length."""
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        batch_size = len(token_ids_list)
        input_ids = torch.full(
            (batch_size, self.max_prompt_length), pad_id, dtype=torch.long,
        )
        attention_mask = torch.zeros(batch_size, self.max_prompt_length, dtype=torch.long)

        for i, ids in enumerate(token_ids_list):
            length = min(len(ids), self.max_prompt_length)
            input_ids[i, -length:] = torch.tensor(ids[-length:], dtype=torch.long)
            attention_mask[i, -length:] = 1

        return input_ids, attention_mask

    def __iter__(self) -> Iterator[PromptBatch]:
        """Iterate batches for one epoch."""
        shuffled = self._shuffle(self.epoch)
        num_batches = math.ceil(len(shuffled) / self.train_batch_size)
        self.step_in_epoch = 0

        for b in range(num_batches):
            start = b * self.train_batch_size
            end = min(start + self.train_batch_size, len(shuffled))
            batch_items = shuffled[start:end]

            # Expand each prompt × n for rollout generation
            expanded_items = [item for item in batch_items for _ in range(self.n_completions)]

            token_ids_list = [item["token_ids"] for item in expanded_items]
            input_ids, attention_mask = self._left_pad(token_ids_list)

            yield PromptBatch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                prompt_ids=[item["prompt_id"] for item in expanded_items],
                raw_prompts=[item["text"] for item in expanded_items],
                metadata=[item["metadata"] for item in expanded_items],
            )

            self.step_in_epoch += 1
            self.total_steps += 1

        self.epoch += 1

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return math.ceil(len(self.items) / self.train_batch_size)

    def state_dict(self) -> dict:
        """For checkpoint resume."""
        return {
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, state: dict):
        """Resume from checkpoint."""
        self.epoch = state.get("epoch", 0)
        self.step_in_epoch = state.get("step_in_epoch", 0)
        self.total_steps = state.get("total_steps", 0)


def load_prompts_from_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Load prompts from a parquet file. Returns list of dicts."""
    import pandas as pd
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")
