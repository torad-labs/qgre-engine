"""Span-based advantage assignment — maps reward function character spans to token indices.

The reward function identifies WHERE in the completion text each quality was scored
(character offsets). This module converts those character spans to per-token boolean
masks so the advantage estimator can target the exact tokens that express each quality.

This replaces the section-based segmenter for reward functions that return scored_spans.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch


if TYPE_CHECKING:
    from qgre.types import TrainingContext


# Marker value for repeated spans. In spans.py, we use this as a marker.
# The actual sign-aware penalty is applied in advantages.py.
REPETITION_MARKER = -1.0


def build_char_to_token_map(
    token_ids: list[int],
    tokenizer: Any,
    completion_text: str | None = None,
) -> list[int] | None:
    """Build a character-offset → token-index mapping from ORIGINAL token_ids.

    Decodes the full sequence once, then uses convert_ids_to_tokens + offset tracking
    to map each character position to its source token. No re-encoding — uses the
    actual tokens that were generated.

    Returns a list where char_to_token[char_idx] = token_idx.
    Returns None if mapping cannot be built reliably.
    """
    if not token_ids:
        return []

    # Pre-validation: tokenizer must have decode method
    if not hasattr(tokenizer, "decode"):
        warnings.warn(
            "build_char_to_token_map: tokenizer lacks decode method — returning None", stacklevel=2
        )
        return None

    # Get the full decoded text (this is what the reward function scored)
    # CRITICAL: vLLM's output.text uses skip_special_tokens=True, so we must match
    try:
        full_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    except (ValueError, TypeError, RuntimeError) as e:
        warnings.warn(f"build_char_to_token_map: full decode failed: {e}", stacklevel=2)
        return None

    full_len = len(full_text)
    if full_len == 0:
        return None  # Return None instead of [] for empty decode

    # Build offset map by decoding each token and tracking cumulative position
    # This uses the ORIGINAL token_ids — no re-encoding
    char_to_token: list[int] = [-1] * full_len
    char_pos = 0
    decode_failures = 0

    for tok_idx, tid in enumerate(token_ids):
        try:
            # Decode this single token
            tok_text = tokenizer.decode([tid], skip_special_tokens=True)
            tok_len = len(tok_text)

            # Find where this token's text appears in full_text starting from char_pos
            # Usually it's exactly at char_pos, but BPE merges can shift things
            if tok_len > 0:
                # Try exact match first
                if full_text[char_pos : char_pos + tok_len] == tok_text:
                    for c in range(char_pos, min(char_pos + tok_len, full_len)):
                        char_to_token[c] = tok_idx
                    char_pos += tok_len
                else:
                    # BPE merge caused text difference — search nearby
                    found = False
                    for offset in range(min(5, full_len - char_pos)):
                        if full_text[char_pos + offset : char_pos + offset + tok_len] == tok_text:
                            for c in range(
                                char_pos + offset, min(char_pos + offset + tok_len, full_len)
                            ):
                                char_to_token[c] = tok_idx
                            char_pos = char_pos + offset + tok_len
                            found = True
                            break
                    if not found:
                        # Can't find exact match — assign remaining chars proportionally
                        # This handles cases where individual decode differs from joint decode
                        remaining_tokens = len(token_ids) - tok_idx
                        remaining_chars = full_len - char_pos
                        if remaining_tokens > 0 and remaining_chars > 0:
                            chars_for_this = max(1, remaining_chars // remaining_tokens)
                            for c in range(char_pos, min(char_pos + chars_for_this, full_len)):
                                char_to_token[c] = tok_idx
                            char_pos += chars_for_this
        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise  # Never swallow OOM — let it crash so we can diagnose
        except (UnicodeDecodeError, ValueError, RuntimeError) as e:
            # Known decode failure types — track and continue
            decode_failures += 1
            if decode_failures == 1:
                warnings.warn(
                    f"build_char_to_token_map: per-token decode failed at tok_idx {tok_idx}: {e}",
                    stacklevel=2,
                )
        except (TypeError, AttributeError, KeyError) as e:
            # Other decode failure types — warn but continue
            decode_failures += 1
            if decode_failures == 1:
                warnings.warn(
                    f"build_char_to_token_map: decode error at tok_idx {tok_idx}: {type(e).__name__}: {e}",
                    stacklevel=2,
                )

    # Safety check: if per-token decode covered way less than full decode, the
    # mapping is unreliable. Return None to trigger segmenter fallback.
    # DP2-010: Raise threshold to 80% (was 50%) for better reliability
    if full_len > 0 and char_pos < full_len * 0.8:
        warnings.warn(
            f"build_char_to_token_map: per-token decode covered {char_pos}/{full_len} chars "
            f"({100*char_pos/full_len:.1f}%). Mapping unreliable — returning None. "
            "Threshold is 80% coverage.",
            stacklevel=2,
        )
        return None

    # DP3-006: Track gap-fill count, warn if high percentage
    gap_count = sum(1 for c in char_to_token if c < 0)
    if gap_count > 0 and gap_count / full_len > 0.1:
        warnings.warn(
            f"DP3-006: char_to_token gap-fill: {gap_count}/{full_len} chars ({100*gap_count/full_len:.1f}%) "
            "have no direct token mapping. Filling with nearest valid. "
            "High gap percentage may indicate tokenizer decode mismatch.",
            stacklevel=2,
        )

    # Check if all tokens decode to -1 (complete mapping failure)
    if gap_count == full_len:
        warnings.warn(
            f"CRITICAL: All {full_len} characters map to -1. Complete token mapping failure. "
            "Entire batch will lose advantage signal. Check tokenizer decode behavior.",
            stacklevel=2,
        )

    # Fill any remaining gaps with nearest valid token
    last_valid = -1
    for c in range(full_len):
        if char_to_token[c] >= 0:
            last_valid = char_to_token[c]
        else:
            char_to_token[c] = last_valid if last_valid >= 0 else -1

    return char_to_token


def scored_spans_to_token_masks(
    scored_spans: dict[str, list[tuple[int, int]]],
    char_to_token: list[int],
    seq_len: int,
    ctx: TrainingContext,
    _clamped_count: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Convert character-based scored_spans to per-token boolean masks.

    Args:
        scored_spans: quality_name → [(char_start, char_end), ...]
        char_to_token: char_idx → token_idx mapping (from build_char_to_token_map)
        seq_len: number of tokens in the completion
        ctx: TrainingContext for device placement
        _clamped_count: Optional list to track clamped span count (mutable int workaround)

    Returns:
        dict mapping quality_name → torch.Tensor of shape [seq_len] with 1.0
        at token positions covered by that quality's spans, 0.0 elsewhere.
    """
    # Move counter to function scope via mutable default argument
    if _clamped_count is None:
        _clamped_count = [0]

    masks: dict[str, torch.Tensor] = {}
    max_char = len(char_to_token)

    for quality_name, spans in scored_spans.items():
        mask = torch.zeros(seq_len, device=ctx.device)
        # Reward FIRST span for each quality — later repetitions get penalized
        # The original answer is typically the correct one; repeats are noise
        # First span gets +1.0, later spans get REPETITION_MARKER penalty
        len(spans)
        for span_idx, (char_start, char_end) in enumerate(spans):
            # AE-R2-03: Skip span if completely outside bounds (don't relocate)
            if char_start >= max_char:
                import warnings

                warnings.warn(
                    f"Span completely outside bounds for quality '{quality_name}': "
                    f"char_start={char_start} >= max_char={max_char}. Skipping span.",
                    stacklevel=2,
                )
                continue
            # Clamp to valid range
            cs = max(0, min(char_start, max_char - 1))
            ce = max(0, min(char_end, max_char))
            if cs != char_start or ce != char_end:
                import warnings

                warnings.warn(
                    f"Span offset clamped for quality '{quality_name}': "
                    f"original ({char_start}, {char_end}) → clamped ({cs}, {ce}). "
                    f"max_char={max_char}. Final tokens may lose advantage signal.",
                    stacklevel=2,
                )
                # Track clamped spans for metrics
                _clamped_count[0] += 1
            # Warn if clamping caused zero-length span
            if cs >= ce:
                import warnings

                warnings.warn(
                    f"Span became zero-length after clamping for quality '{quality_name}': "
                    f"original ({char_start}, {char_end}) → clamped ({cs}, {ce}). "
                    "Advantage signal lost for this span.",
                    stacklevel=2,
                )
            # Map char range → token indices and set mask
            # First span: +1.0 (reward original answer), later spans: REPETITION_MARKER (penalize repeats)
            # The marker is detected in advantages.py where sign-aware penalty is applied
            is_first_span = span_idx == 0
            span_value = 1.0 if is_first_span else REPETITION_MARKER
            for c in range(cs, ce):
                tok_idx = char_to_token[c]
                # AE-R2-02: Skip assignment when tok_idx == -1 (invalid mapping)
                if tok_idx == -1:
                    continue
                # DP3-005: Add assertion that tok_idx < seq_len with informative error
                if tok_idx >= seq_len:
                    raise RuntimeError(
                        f"DP3-005: Span mapping exceeds seq_len. "
                        f"tok_idx={tok_idx} >= seq_len={seq_len} for quality '{quality_name}'. "
                        f"char_range=({cs}, {ce}), max_char={max_char}. "
                        "Tokenizer or span detection is inconsistent.",
                    )
                mask[tok_idx] = span_value
        masks[quality_name] = mask

    return masks
