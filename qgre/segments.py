from __future__ import annotations

import re
from collections.abc import Callable
from functools import partial
from typing import Any

import torch


# --- Segmenter protocol ---
# A segmenter takes token IDs and returns region labels (same length).
# Region labels: "THINK", "STEP_1", "STEP_2", ..., "FORMAT", "OTHER"
# The engine uses these to assign per-step advantages.
Segmenter = Callable[[list[int]], list[str]]


# --- Qwen3 XML segmenter (default) ---

# Qwen3 verified token IDs (from SPECIAL-TOKENS-SUPERPOWER.md, 2026-03-18)
THINK_START = 151667
THINK_END = 151668
STEP_TOKEN = 9520
OPEN_ANGLE = 27
CLOSE_SLASH = 522
CLOSE_ANGLE = 29
STEP_NUM_TOKENS = {16: 1, 17: 2, 18: 3, 19: 4, 20: 5, 21: 6, 22: 7, 23: 8, 24: 9}


def qwen3_xml_segmenter(token_ids: list[int]) -> list[str]:
    """Segment Qwen3 completions with <stepN_...> XML tags.

    Assigns each token to: THINK, STEP_1..N, FORMAT, or OTHER.
    Uses token ID patterns, not decoded text.
    """
    regions = ["OTHER"] * len(token_ids)
    current = "OTHER"
    n = len(token_ids)

    i = 0
    while i < n:
        tid = token_ids[i]

        if tid == THINK_START:
            current = "THINK"
            regions[i] = "THINK"
            i += 1
            continue

        if tid == THINK_END:
            regions[i] = "THINK"
            current = "OTHER"
            i += 1
            continue

        if tid == OPEN_ANGLE and i + 2 < n:
            if token_ids[i + 1] == STEP_TOKEN and token_ids[i + 2] in STEP_NUM_TOKENS:
                step = STEP_NUM_TOKENS[token_ids[i + 2]]
                current = f"STEP_{step}"
                j = i
                while j < n and token_ids[j] != CLOSE_ANGLE:
                    regions[j] = "FORMAT"
                    j += 1
                if j < n:
                    regions[j] = "FORMAT"
                i = j + 1
                continue

        if tid == CLOSE_SLASH and i + 2 < n:
            if token_ids[i + 1] == STEP_TOKEN and token_ids[i + 2] in STEP_NUM_TOKENS:
                j = i
                while j < n and token_ids[j] != CLOSE_ANGLE:
                    regions[j] = "FORMAT"
                    j += 1
                if j < n:
                    regions[j] = "FORMAT"
                current = "OTHER"
                i = j + 1
                continue

        regions[i] = current
        i += 1

    return regions


def uniform_segmenter(token_ids: list[int]) -> list[str]:
    """Trivial segmenter: all tokens get STEP_1 (uniform advantages).

    Use for domains without step structure (e.g., single-turn Q&A, math).
    The advantage estimator gives every token the same step-1 advantage.
    """
    return ["STEP_1"] * len(token_ids)


# --- HIF JSON segmenter (decode-and-regex) ---

# Default section patterns for HIF v2 JSON output.
# Maps regex patterns (matched against decoded text) to step numbers.
# Order matters — first match wins for overlapping regions.
HIF_SECTION_PATTERNS: list[tuple[str, str]] = [
    (r'"(?:network-type|metadata)"', "STEP_1"),
    (r'"nodes"', "STEP_2"),
    (r'"(?:edges|incidences)"', "STEP_3"),
    (r'"(?:scan-results|hamiltonian-score)"', "STEP_4"),
]


def _hif_json_segmenter_impl(token_ids: list[int], tokenizer: Any) -> list[str]:
    """Segment HIF JSON completions via decoded text + regex.

    Decodes token IDs to text, uses regex to find JSON section boundaries,
    maps character offsets back to token positions. Handles <think> blocks
    via the same THINK_START/THINK_END token IDs as the XML segmenter.

    Tokens not matching any section → STEP_5 (catch-all for overall quality).
    """
    if not token_ids:
        return []

    # Pre-validation: check tokenizer has decode method
    if not hasattr(tokenizer, "decode"):
        import warnings

        warnings.warn(
            "Segmenter: tokenizer lacks decode method — returning default regions", stacklevel=2
        )
        return ["STEP_5"] * len(token_ids)

    n = len(token_ids)
    regions = ["STEP_5"] * n  # Default: last step (overall quality)

    # Pass 1: Handle <think> tokens by token ID (same as XML segmenter, no decode needed)
    in_think = False
    for i, tid in enumerate(token_ids):
        if tid == THINK_START:
            in_think = True
            regions[i] = "THINK"
            continue
        if tid == THINK_END:
            regions[i] = "THINK"
            in_think = False
            continue
        if in_think:
            regions[i] = "THINK"

    # Pass 2: Decode non-think tokens and map JSON sections via regex
    # Build character→token index mapping
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
    except (ValueError, TypeError, RuntimeError) as e:
        import warnings

        warnings.warn(
            f"Segmenter full decode failed for {len(token_ids)} tokens: {e}. "
            f"Falling back to think-annotated + STEP_5 default.",
            stacklevel=2,
        )
        return regions  # If decode fails, return think-annotated + STEP_5 default

    # Get per-token text lengths for char→token mapping
    token_texts = []
    decode_failures = 0
    for tid in token_ids:
        try:
            token_texts.append(tokenizer.decode([tid], skip_special_tokens=False))
        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise  # Never swallow OOM — let it crash so we can diagnose
        except (ValueError, TypeError, RuntimeError):
            token_texts.append("")
            decode_failures += 1
    # DP2-009: Track decode failure count, warn if > threshold
    if decode_failures > 0:
        import warnings

        failure_rate = decode_failures / len(token_ids) if token_ids else 0
        if failure_rate > 0.05:  # Warn if >5% failures
            warnings.warn(
                f"Segmenter per-token decode: {decode_failures}/{len(token_ids)} failures "
                f"({failure_rate * 100:.1f}%). Region mapping may be inaccurate. "
                "Check tokenizer compatibility.",
                stacklevel=2,
            )

    # Build char_offset → token_index mapping
    char_to_token = []
    offset = 0
    for i, tt in enumerate(token_texts):
        for _ in range(len(tt)):
            char_to_token.append(i)
        offset += len(tt)

    # Find section boundaries in decoded text
    section_spans: list[tuple[int, int, str]] = []  # (start_char, end_char, region)
    for pattern, region in HIF_SECTION_PATTERNS:
        for m in re.finditer(pattern, text):
            section_spans.append((m.start(), m.end(), region))

    # Sort by position and assign regions between section markers
    section_spans.sort(key=lambda x: x[0])

    for idx, (start_char, _end_char, region) in enumerate(section_spans):
        # Region extends from this key to the next key (or end of text)
        region_end_char = section_spans[idx + 1][0] if idx + 1 < len(section_spans) else len(text)

        # Map char range to token indices
        truncation_occurred = region_end_char > len(char_to_token)
        for c in range(start_char, min(region_end_char, len(char_to_token))):
            tok_idx = char_to_token[c]
            if regions[tok_idx] != "THINK":  # Don't overwrite think tokens
                regions[tok_idx] = region
        # AE-R2-04: Log warning when truncation happens
        if truncation_occurred:
            import logging

            logging.getLogger(__name__).warning(
                f"Region '{region}' truncated: region_end_char={region_end_char} > len(char_to_token)={len(char_to_token)}. "
                "Some region tokens may be lost.",
            )

    return regions


def make_hif_json_segmenter(tokenizer: Any) -> Segmenter:
    """Create an HIF JSON segmenter bound to a specific tokenizer.

    Usage in trainer.py registration:
        segmenter = make_hif_json_segmenter(tokenizer)
    """
    return partial(_hif_json_segmenter_impl, tokenizer=tokenizer)


# --- Hamiltonian structured-label segmenter (exact string matching, no regex) ---

# --- Shared Hamiltonian label configuration (single source of truth) ---
# Used by both the segmenter AND reward_fn.py for label extraction.

# Canonical label names → list of aliases (case-insensitive matching)
HAMILTONIAN_LABEL_ALIASES: dict[str, list[str]] = {
    "COORDINATES": ["coordinates", "coordinate", "generalized coordinate"],
    "MOMENTUM": ["momentum", "conjugate momentum"],
    "KINETIC": ["kinetic", "kinetic energy"],
    "POTENTIAL": ["potential", "potential energy"],
    "HAMILTONIAN": ["hamiltonian"],
    "EQUATIONS": ["equations", "equations of motion", "hamilton's equations"],
}

# Maps canonical labels to step regions
HAMILTONIAN_LABEL_TO_STEP: dict[str, str] = {
    "COORDINATES": "STEP_1",
    "MOMENTUM": "STEP_2",
    "KINETIC": "STEP_3",
    "POTENTIAL": "STEP_4",
    "HAMILTONIAN": "STEP_5",
    "EQUATIONS": "STEP_6",
}


def _strip_markdown_prefix(line: str) -> str:
    """Strip markdown formatting from start of line (headers, bold, etc.)."""
    s = line.lstrip()
    # Strip leading # headers
    while s.startswith("#"):
        s = s[1:]
    s = s.lstrip()
    # Strip leading ** or * (bold/italic)
    while s.startswith("*"):
        s = s[1:]
    return s.lstrip()


def _match_hamiltonian_label(line: str) -> tuple[str, int] | None:
    """Check if line starts with a Hamiltonian label. Returns (canonical_label, colon_pos) or None.

    Uses exact string matching — no regex.
    """
    cleaned = _strip_markdown_prefix(line).lower()

    for canonical, aliases in HAMILTONIAN_LABEL_ALIASES.items():
        # Check canonical name first
        all_names = [canonical.lower()] + [a.lower() for a in aliases]
        for name in all_names:
            if cleaned.startswith(name):
                # Must be followed by : (possibly with trailing ** from markdown)
                rest = cleaned[len(name) :].lstrip("*").lstrip()
                if rest.startswith(":"):
                    # Find colon position in original line
                    colon_idx = line.find(":")
                    if colon_idx >= 0:
                        return (canonical, colon_idx)
    return None


def _hamiltonian_segmenter_impl(token_ids: list[int], tokenizer: Any) -> list[str]:
    """Segment Hamiltonian structured output by label.

    Each label (COORDINATES, MOMENTUM, KINETIC, POTENTIAL, HAMILTONIAN, EQUATIONS)
    maps to its own STEP region so each section gets its own quality advantage signal.
    Derivation text before any label gets STEP_1 (format/math quality).
    """
    if not token_ids:
        return []

    # Pre-validation: check tokenizer has decode method
    if not hasattr(tokenizer, "decode"):
        import warnings

        warnings.warn(
            "Hamiltonian segmenter: tokenizer lacks decode method — returning default regions",
            stacklevel=2,
        )
        return ["STEP_1"] * len(token_ids)

    n = len(token_ids)
    regions = ["STEP_1"] * n  # Default: derivation text → format quality

    # Pass 1: Handle <think> tokens by ID
    in_think = False
    for i, tid in enumerate(token_ids):
        if tid == THINK_START:
            in_think = True
            regions[i] = "THINK"
            continue
        if tid == THINK_END:
            regions[i] = "THINK"
            in_think = False
            continue
        if in_think:
            regions[i] = "THINK"

    # Pass 2: Decode and find label positions
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
    except (ValueError, TypeError, RuntimeError) as e:
        import warnings

        warnings.warn(
            f"Hamiltonian segmenter full decode failed for {len(token_ids)} tokens: {e}. "
            f"Falling back to think-annotated + STEP_1 default.",
            stacklevel=2,
        )
        return regions

    # Build char→token index mapping
    token_texts = []
    decode_failures = 0
    for tid in token_ids:
        try:
            token_texts.append(tokenizer.decode([tid], skip_special_tokens=False))
        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise  # Never swallow OOM — let it crash so we can diagnose
        except (ValueError, TypeError, RuntimeError):
            token_texts.append("")
            decode_failures += 1
    if decode_failures > 0:
        import warnings

        warnings.warn(
            f"Hamiltonian segmenter per-token decode: {decode_failures}/{len(token_ids)} failures.",
            stacklevel=2,
        )

    char_to_token = []
    for i, tt in enumerate(token_texts):
        for _ in range(len(tt)):
            char_to_token.append(i)

    # Find label positions using exact string matching (no regex)
    label_spans: list[tuple[int, str]] = []  # (start_char, region)
    char_pos = 0
    for line in text.split("\n"):
        match = _match_hamiltonian_label(line)
        if match:
            canonical_label, _ = match
            region = HAMILTONIAN_LABEL_TO_STEP.get(canonical_label, "STEP_1")
            label_spans.append((char_pos, region))
        char_pos += len(line) + 1  # +1 for newline

    if not label_spans:
        return regions  # No labels found → all STEP_1

    # Sort by position, assign regions from each label to the next
    label_spans.sort(key=lambda x: x[0])

    for idx, (start_char, region) in enumerate(label_spans):
        end_char = label_spans[idx + 1][0] if idx + 1 < len(label_spans) else len(text)
        for c in range(start_char, min(end_char, len(char_to_token))):
            tok_idx = char_to_token[c]
            if regions[tok_idx] != "THINK":
                regions[tok_idx] = region

    return regions


def segmenter_region_count(regions: list[str]) -> int:
    """Count distinct non-THINK, non-OTHER step regions in segmenter output.

    Used by VPRM to decide per-sample fallback: if only 1 region found,
    fall back to SPO scalar baseline for that sample.
    """
    return len({r for r in regions if r.startswith("STEP_")})


def make_hamiltonian_segmenter(tokenizer: Any) -> Segmenter:
    """Create a Hamiltonian segmenter bound to a specific tokenizer."""
    return partial(_hamiltonian_segmenter_impl, tokenizer=tokenizer)


# --- Generic label segmenter (config-driven) ---


def _label_segmenter_impl(
    token_ids: list[int],
    tokenizer: Any,
    patterns: list[tuple[str, str]],
    default_region: str = "STEP_1",
    ignore_case: bool = False,
) -> list[str]:
    """Generic label-based segmenter — config-driven, no domain-specific code.

    Any domain can define regex→step mappings in YAML and get per-section
    credit assignment automatically.
    """
    if not token_ids:
        return []

    # Pre-validation: check tokenizer has decode method
    if not hasattr(tokenizer, "decode"):
        import warnings

        warnings.warn(
            f"Label segmenter: tokenizer lacks decode method — returning {default_region} for all tokens",
            stacklevel=2,
        )
        return [default_region] * len(token_ids)

    n = len(token_ids)
    regions = [default_region] * n

    # Pass 1: Handle <think> tokens by ID
    in_think = False
    for i, tid in enumerate(token_ids):
        if tid == THINK_START:
            in_think = True
            regions[i] = "THINK"
            continue
        if tid == THINK_END:
            regions[i] = "THINK"
            in_think = False
            continue
        if in_think:
            regions[i] = "THINK"

    # Pass 2: Decode and find label positions
    try:
        text = tokenizer.decode(token_ids, skip_special_tokens=False)
    except (ValueError, TypeError, RuntimeError) as e:
        import warnings

        warnings.warn(
            f"Label segmenter full decode failed for {len(token_ids)} tokens: {e}. "
            f"Falling back to think-annotated + {default_region} default.",
            stacklevel=2,
        )
        return regions

    # Build char→token index mapping
    token_texts = []
    decode_failures = 0
    for tid in token_ids:
        try:
            token_texts.append(tokenizer.decode([tid], skip_special_tokens=False))
        except (torch.cuda.OutOfMemoryError, MemoryError):
            raise  # Never swallow OOM — let it crash so we can diagnose
        except (ValueError, TypeError, RuntimeError):
            token_texts.append("")
            decode_failures += 1
    if decode_failures > 0:
        import warnings

        warnings.warn(
            f"Label segmenter per-token decode: {decode_failures}/{len(token_ids)} failures.",
            stacklevel=2,
        )

    char_to_token = []
    for i, tt in enumerate(token_texts):
        for _ in range(len(tt)):
            char_to_token.append(i)

    # Find label positions
    flags = re.IGNORECASE if ignore_case else 0
    label_spans: list[tuple[int, str]] = []
    for pattern, region in patterns:
        for m in re.finditer(pattern, text, flags):
            label_spans.append((m.start(), region))

    if not label_spans:
        return regions

    # Sort by position, assign regions from each label to the next
    label_spans.sort(key=lambda x: x[0])

    for idx, (start_char, region) in enumerate(label_spans):
        end_char = label_spans[idx + 1][0] if idx + 1 < len(label_spans) else len(text)
        for c in range(start_char, min(end_char, len(char_to_token))):
            tok_idx = char_to_token[c]
            if regions[tok_idx] != "THINK":
                regions[tok_idx] = region

    return regions


def make_label_segmenter(
    tokenizer: Any,
    label_config: Any,  # LabelSegmenterConfig from config.py
) -> Segmenter:
    """Create a generic label segmenter from config.

    Usage in YAML:
        algorithm:
          segmenter: label
          label_segmenter:
            default_region: STEP_1
            ignore_case: true
            patterns:
              - pattern: '(?:^|\\n)\\s*KINETIC\\s*:'
                region: STEP_3
    """
    patterns = [(p.pattern, p.region) for p in label_config.patterns]
    return partial(
        _label_segmenter_impl,
        tokenizer=tokenizer,
        patterns=patterns,
        default_region=label_config.default_region,
        ignore_case=label_config.ignore_case,
    )


# Backward compatibility alias
segment_completion = qwen3_xml_segmenter


# --- Example step_qualities configs ---

HYPERGRAPH_V1_STEP_QUALITIES: dict[int, list[str]] = {
    1: ["q_format_tags", "q_tag_content", "q_node_in_prompt", "q_node_format", "q_node_length"],
    2: ["q_chain_s2_refs_s1"],
    3: ["q_chain_s3_refs_s2", "q_self_consistency"],
    4: [
        "q_step4_valid_json",
        "q_step4_has_keys",
        "q_existence_correct",
        "q_archetype_correct",
        "q_node_f1",
    ],
}

# Global qualities: not step-specific, contribute to overall sequence reward only.
# These are tracked in RewardResult.scores but not assigned to per-token advantages.
# (SPECIAL-TOKENS-SUPERPOWER.md line 104-106)
HYPERGRAPH_V1_GLOBAL_QUALITIES: list[str] = ["q_eos_correct"]

# Backward compatibility alias
STEP_QUALITIES = HYPERGRAPH_V1_STEP_QUALITIES
