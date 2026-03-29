"""Span-based advantage assignment — maps reward function character spans to token indices.

The reward function identifies WHERE in the completion text each quality was scored
(character offsets). This module converts those character spans to per-token boolean
masks so the advantage estimator can target the exact tokens that express each quality.

This replaces the section-based segmenter for reward functions that return scored_spans.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch


def build_char_to_token_map(
    token_ids: list[int],
    tokenizer: Any,
    completion_text: str | None = None,
) -> list[int] | None:
    """Build a character-offset → token-index mapping.

    Uses the tokenizer's offset_mapping (re-encoding the text) when available —
    this is the authoritative mapping from the tokenizer itself, reliable across
    BPE and SentencePiece models. Falls back to per-token decode if unavailable.

    Returns a list where char_to_token[char_idx] = token_idx.
    Returns None if mapping cannot be built reliably.
    """
    if not token_ids:
        return []

    # Pre-validation: tokenizer must have decode method
    if not hasattr(tokenizer, "decode"):
        warnings.warn("build_char_to_token_map: tokenizer lacks decode method — returning None")
        return None

    # Strategy 1: Use tokenizer's offset_mapping (authoritative, model-agnostic)
    if completion_text is not None:
        try:
            encoding = tokenizer(completion_text, return_offsets_mapping=True, add_special_tokens=False)
            offsets = encoding.get("offset_mapping")
            if offsets and len(offsets) == len(token_ids):
                text_len = len(completion_text)
                char_to_token: list[int] = [-1] * text_len
                for tok_idx, (start, end) in enumerate(offsets):
                    for c in range(start, min(end, text_len)):
                        char_to_token[c] = tok_idx
                # Fill gaps (chars not covered by any token) with nearest token
                last_valid = 0
                for c in range(text_len):
                    if char_to_token[c] >= 0:
                        last_valid = char_to_token[c]
                    else:
                        char_to_token[c] = last_valid
                return char_to_token
        except Exception as e:
            warnings.warn(f"Span mapping Strategy 1 (offset_mapping) failed: {e}. Trying Strategy 2.")
            pass  # Fall through to Strategy 2

    # Strategy 2: Per-token decode (works for BPE, may fail for SentencePiece)
    token_texts = []
    decode_failures = 0
    for tid in token_ids:
        try:
            token_texts.append(tokenizer.decode([tid], skip_special_tokens=False))
        except Exception:
            token_texts.append("")
            decode_failures += 1
    if decode_failures > 0:
        warnings.warn(
            f"Span mapping: {decode_failures}/{len(token_ids)} per-token decode failures. "
            "Character-to-token mapping may be inaccurate."
        )

    per_token_len = sum(len(tt) for tt in token_texts)
    try:
        full_text = tokenizer.decode(token_ids, skip_special_tokens=False)
        full_len = len(full_text)
    except Exception as e:
        warnings.warn(f"Span mapping full decode failed: {e}. Using per-token length.")
        full_len = per_token_len

    if per_token_len != full_len:
        # Tolerate small mismatches (BPE merge boundaries add/remove chars
        # when tokens are decoded individually vs together)
        mismatch = abs(per_token_len - full_len)
        if mismatch > max(3, full_len * 0.01):
            warnings.warn(
                f"Span mapping: per-token decode length ({per_token_len}) != "
                f"full decode length ({full_len}), mismatch={mismatch}. "
                f"Token IDs: {token_ids[:10]}..., full text: {full_text[:50]}... "
                "Falling back to segmenter."
            )
            return None

    # Build char→token map, truncate or pad to match full_len
    char_to_token = []
    for i, tt in enumerate(token_texts):
        for _ in range(len(tt)):
            char_to_token.append(i)

    if len(char_to_token) > full_len:
        char_to_token = char_to_token[:full_len]
    elif len(char_to_token) < full_len:
        last_tok = char_to_token[-1] if char_to_token else 0
        char_to_token.extend([last_tok] * (full_len - len(char_to_token)))

    return char_to_token


def scored_spans_to_token_masks(
    scored_spans: dict[str, list[tuple[int, int]]],
    char_to_token: list[int],
    seq_len: int,
) -> dict[str, torch.Tensor]:
    """Convert character-based scored_spans to per-token boolean masks.

    Args:
        scored_spans: quality_name → [(char_start, char_end), ...]
        char_to_token: char_idx → token_idx mapping (from build_char_to_token_map)
        seq_len: number of tokens in the completion

    Returns:
        dict mapping quality_name → torch.Tensor of shape [seq_len] with 1.0
        at token positions covered by that quality's spans, 0.0 elsewhere.
    """
    masks: dict[str, torch.Tensor] = {}
    max_char = len(char_to_token)

    for quality_name, spans in scored_spans.items():
        mask = torch.zeros(seq_len)
        for char_start, char_end in spans:
            # Clamp to valid range
            cs = max(0, min(char_start, max_char - 1))
            ce = max(0, min(char_end, max_char))
            if cs != char_start or ce != char_end:
                import warnings
                warnings.warn(
                    f"Span offset clamped for quality '{quality_name}': "
                    f"original ({char_start}, {char_end}) → clamped ({cs}, {ce}). "
                    f"max_char={max_char}. Final tokens may lose advantage signal."
                )
            # Map char range → token indices and set mask
            for c in range(cs, ce):
                tok_idx = char_to_token[c]
                if tok_idx < seq_len:
                    mask[tok_idx] = 1.0
        masks[quality_name] = mask

    return masks
