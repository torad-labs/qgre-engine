from __future__ import annotations

from typing import Callable

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
STEP_NUM_TOKENS = {16: 1, 17: 2, 18: 3, 19: 4}


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


# Backward compatibility alias
segment_completion = qwen3_xml_segmenter


# --- Example step_qualities configs ---

HYPERGRAPH_V1_STEP_QUALITIES: dict[int, list[str]] = {
    1: ["q_format_tags", "q_tag_content", "q_node_in_prompt", "q_node_format", "q_node_length"],
    2: ["q_chain_s2_refs_s1"],
    3: ["q_chain_s3_refs_s2", "q_self_consistency"],
    4: ["q_step4_valid_json", "q_step4_has_keys", "q_existence_correct",
        "q_archetype_correct", "q_node_f1"],
}

HIF_V2_STEP_QUALITIES: dict[int, list[str]] = {
    1: ["q_valid_json", "q_hif_schema"],
    2: ["q_node_grounding", "q_node_verbatim"],
    3: ["q_incidence_refs_nodes", "q_internal_consistency"],
    4: ["q_existence_correct", "q_archetype_correct"],
    5: ["q_node_f1", "q_edge_f1"],
}

MATH_STEP_QUALITIES: dict[int, list[str]] = {
    1: ["q_correct_answer"],
}

# Backward compatibility alias
STEP_QUALITIES = HYPERGRAPH_V1_STEP_QUALITIES
