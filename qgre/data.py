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
        # DP-R2-05: Validate n_completions >= 1
        if n_completions < 1:
            raise ValueError(
                f"DP-R2-05: n_completions must be >= 1, got {n_completions}. "
                "Cannot generate rollouts with n_completions=0."
            )
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
        # Validate ALL rows for metadata columns (not just first 10)
        if self.metadata_columns and prompts:
            for idx, row in enumerate(prompts):
                missing = [col for col in self.metadata_columns if col not in row]
                if missing:
                    raise ValueError(
                        f"Required metadata_columns {missing} not found in prompt data at index {idx}. "
                        f"Available columns: {list(row.keys())}. "
                        "Update metadata_columns in config or add missing columns to training data."
                    )
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
                # After unwrap, verify token_ids is still a sequence before len() check
                if not isinstance(token_ids, (list, tuple)):
                    import warnings
                    warnings.warn(
                        f"DP-R2-UNWRAP: token_ids after unwrap is scalar {type(token_ids).__name__}, "
                        f"wrapping back to list. Prompt text: {text[:100]}"
                    )
                    token_ids = [token_ids]
            else:
                token_ids = self.tokenizer.encode(text)

            # DP-R2-02: Filter out empty token_ids
            if len(token_ids) == 0:
                import warnings
                warnings.warn(
                    f"DP-R2-02: Empty token_ids after encoding, skipping prompt. "
                    f"Prompt text: {text[:100]}"
                )
                continue

            if len(token_ids) > self.max_prompt_length:
                import warnings
                warnings.warn(
                    f"Prompt truncated: {len(token_ids)} tokens > max_prompt_length={self.max_prompt_length}. "
                    "This prompt will be skipped. Increase max_prompt_length if this is unintended."
                )
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
        import math
        for prompt_id, weight in priorities.items():
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"Invalid priority weight for prompt_id {prompt_id}: {weight}. "
                    "Weights must be non-negative and finite."
                )
        self._priorities = priorities

    def set_difficulty_gate(self, allowed_difficulties: set[str], difficulty_column: str = "difficulty"):
        """Gate prompts by difficulty: only prompts with matching difficulty get sampled.

        Prompts outside the allowed set get zero priority weight.
        Called by the trainer on phase advancement to gradually introduce harder problems.
        """
        if difficulty_column not in self.metadata_columns:
            raise ValueError(
                f"set_difficulty_gate: difficulty_column='{difficulty_column}' "
                f"not in metadata_columns={self.metadata_columns}. "
                "Cannot gate by difficulty without the column. "
                "Add difficulty_column to metadata_columns in config."
            )
        # DP-R2-06: Validate difficulty_column exists in actual metadata keys, not just metadata_columns
        if self.items:
            sample_item = self.items[0]
            if difficulty_column not in sample_item["metadata"]:
                raise ValueError(
                    f"DP3-001: difficulty_column='{difficulty_column}' not found in actual data metadata. "
                    f"Available keys: {list(sample_item['metadata'].keys())}. "
                    "Check that difficulty_column is in metadata_columns and present in training data."
                )
        # DP3-003: Warn if metadata_columns is empty but difficulty_column is set
        if not self.metadata_columns:
            import warnings
            warnings.warn(
                "DP3-003: metadata_columns is empty but difficulty_column is set. "
                "Difficulty gating will not work. Add difficulty_column to metadata_columns."
            )
            # Prevent gate from being set if metadata_columns is empty
            return
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
        if self._difficulty_gate is not None:
            allowed, col = self._difficulty_gate
            filtered_count = 0
            none_count = 0
            for i, item in enumerate(self.items):
                difficulty = item["metadata"].get(col)
                # DP3-010: Treat None same as missing, don't convert to empty string
                if difficulty is None:
                    none_count += 1
                    weights[i] = 0.0
                    filtered_count += 1
                    if none_count == 1:
                        import warnings
                        warnings.warn(
                            f"Difficulty gate: prompt {item['prompt_id']} has difficulty=None and will be filtered. "
                            f"Check difficulty_column '{col}' data."
                        )
                    continue
                # DP3-010: Explicit check — don't convert None to ""
                if difficulty not in allowed:
                    weights[i] = 0.0
                    filtered_count += 1
            if none_count > 0:
                import warnings
                warnings.warn(
                    f"DP3-010: Difficulty gate: {none_count}/{len(self.items)} prompts have difficulty=None "
                    f"and were filtered. Check difficulty_column '{col}' data."
                )

        # DP-R3-03: Log at ERROR level with context about which gate caused zero weights
        if weights.sum() == 0:
            import logging
            logger = logging.getLogger(__name__)
            gate_context = []
            if self._priorities is not None:
                gate_context.append("priority weights")
            if hasattr(self, "_difficulty_gate") and self._difficulty_gate is not None:
                gate_context.append(f"difficulty_gate (allowed: {self._difficulty_gate.get('allowed_difficulties', [])})")
            logger.error(
                f"DP-R3-03: ALL weights are zero after filtering by {', '.join(gate_context)}. "
                "Falling back to uniform sampling. This indicates a configuration error: "
                "no prompts pass the active gates. Check: (1) difficulty_gate allowed_difficulties, "
                "(2) priority weights, (3) metadata column values."
            )
            weights = torch.ones(len(self.items), dtype=torch.float64)

        if self._priorities is not None or (hasattr(self, "_difficulty_gate") and self._difficulty_gate is not None):
            # DP2-007: Filter out zero-weight prompts before multinomial sampling
            mask = weights > 0
            if mask.sum() == 0:
                import warnings
                warnings.warn("All weights are zero after filtering — falling back to uniform sampling")
                indices = torch.randperm(len(self.items), generator=gen).tolist()
            else:
                # Only sample from non-zero weights
                nonzero_indices = torch.nonzero(mask, as_tuple=True)[0]
                nonzero_weights = weights[mask]
                nonzero_weights = nonzero_weights + 1e-8
                nonzero_weights = nonzero_weights / nonzero_weights.sum()
                sampled = torch.multinomial(
                    nonzero_weights, len(self.items), replacement=True, generator=gen,
                )
                # DP3-008: Add assertion that remapped indices are valid
                indices = nonzero_indices[sampled].tolist()
                if any(idx >= len(self.items) or idx < 0 for idx in indices):
                    raise RuntimeError(
                        f"DP3-008: Multinomial sampling produced invalid indices. "
                        f"Max index: {max(indices)}, items: {len(self.items)}"
                    )
        else:
            indices = torch.randperm(len(self.items), generator=gen).tolist()

        return [self.items[i] for i in indices]

    def _left_pad(self, token_ids_list: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Left-pad a batch of token ID lists to max_prompt_length."""
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        # DP2-006: Validate pad_token_id is within vocab bounds
        # Use len(tokenizer) not vocab_size — special tokens (e.g., <|fim_pad|>)
        # are valid but may exceed the base vocab_size property
        vocab_size = len(self.tokenizer) if hasattr(self.tokenizer, "__len__") else None
        if vocab_size is not None and pad_id >= vocab_size:
            raise ValueError(
                f"pad_token_id={pad_id} >= vocab_size={vocab_size}. "
                "Invalid tokenizer configuration. Set tokenizer.pad_token_id to a valid token ID."
            )
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
            expanded_items = []
            for orig_idx, item in enumerate(batch_items):
                for _ in range(self.n_completions):
                    # DP-R2-01: Use deepcopy for nested metadata
                    import copy
                    meta = copy.deepcopy(item["metadata"])
                    # DP-R3-02: Use prefixed name to avoid collision with user metadata
                    meta["_batch_prompt_idx"] = orig_idx
                    expanded_items.append({
                        "token_ids": item["token_ids"],
                        "text": item["text"],
                        "prompt_id": item["prompt_id"],
                        "metadata": meta,
                    })

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
        state = {
            "epoch": self.epoch,
            "step_in_epoch": self.step_in_epoch,
            "total_steps": self.total_steps,
            "priority_weights": self._priorities,
        }
        if self._difficulty_gate is not None:
            allowed, col = self._difficulty_gate
            state["difficulty_gate"] = {"allowed_difficulties": list(allowed), "difficulty_column": col}
        return state

    def load_state_dict(self, state: dict):
        """Resume from checkpoint."""
        from qgre.schema import validate_schema, FieldSpec, Required
        import math

        # Validate state dict structure
        validated = validate_schema(state, {
            "epoch": FieldSpec(int, Required.NO, default=0),
            "step_in_epoch": FieldSpec(int, Required.NO, default=0),
            "total_steps": FieldSpec(int, Required.NO, default=0),
            "priority_weights": FieldSpec((dict, type(None)), Required.NO, default=None),
            "difficulty_gate": FieldSpec((dict, type(None)), Required.NO, default=None),
        }, "dataloader_state")

        self.epoch = validated["epoch"]
        self.step_in_epoch = validated["step_in_epoch"]
        self.total_steps = validated["total_steps"]

        # Validate priority_weights if present
        if validated.get("priority_weights"):
            weights = validated["priority_weights"]
            if not isinstance(weights, dict):
                import warnings
                warnings.warn(
                    f"SCHEMA: priority_weights expected dict, got {type(weights).__name__}. Skipping."
                )
            else:
                # Validate each weight is non-negative and finite
                # SFH-002: Track which entries are invalid for debugging
                invalid_entries = []
                for prompt_id, weight in weights.items():
                    if not isinstance(weight, (int, float)) or weight < 0 or not math.isfinite(weight):
                        invalid_entries.append((prompt_id, weight))
                if invalid_entries:
                    import warnings
                    sample = invalid_entries[:3]  # Show first 3
                    sample_str = ", ".join(f"{pid}={w}" for pid, w in sample)
                    warnings.warn(
                        f"SFH-002: {len(invalid_entries)} invalid priority weights. "
                        f"Examples: [{sample_str}]. Skipping all priority_weights."
                    )
                else:
                    self._priorities = weights

        # Restore difficulty_gate if present
        # SFH-001: Add explicit warnings for each validation failure branch
        if validated.get("difficulty_gate"):
            gate = validated["difficulty_gate"]
            if not isinstance(gate, dict):
                import warnings
                warnings.warn(
                    f"SFH-001: difficulty_gate expected dict, got {type(gate).__name__}. "
                    "Difficulty gate will not be restored — curriculum learning disabled."
                )
            elif "allowed_difficulties" not in gate or "difficulty_column" not in gate:
                import warnings
                warnings.warn(
                    f"SFH-001: difficulty_gate missing required keys. Got keys: {list(gate.keys())}. "
                    "Difficulty gate will not be restored — curriculum learning disabled."
                )
            else:
                allowed = gate["allowed_difficulties"]
                col = gate["difficulty_column"]
                if not isinstance(allowed, list):
                    import warnings
                    warnings.warn(
                        f"SFH-001: allowed_difficulties expected list, got {type(allowed).__name__}. "
                        "Difficulty gate will not be restored — curriculum learning disabled."
                    )
                elif not isinstance(col, str):
                    import warnings
                    warnings.warn(
                        f"SFH-001: difficulty_column expected str, got {type(col).__name__}. "
                        "Difficulty gate will not be restored — curriculum learning disabled."
                    )
                else:
                    self._difficulty_gate = (set(allowed), col)


def load_prompts_from_parquet(path: str | Path) -> list[dict[str, Any]]:
    """Load prompts from a parquet file. Returns list of dicts."""
    import pandas as pd
    # CFG-R2-3: Add context on FileNotFoundError
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Training data file not found: {path}\n"
            f"Check data.train_files in config YAML and verify paths are correct.\n"
            f"Original error: {e}"
        ) from e
    return df.to_dict(orient="records")
