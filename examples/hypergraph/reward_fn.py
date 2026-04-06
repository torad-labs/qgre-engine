# Hypergraph decomposability reward function — STUB for testing
#
# This stub returns synthetic RewardResult with per-quality scores,
# enough to run all engine tests and the trainer without real domain logic.
#
# Full implementation: training-dojo/runs/hypergraph-scan-v1/reward_fn.py
# Replace this file with the real one for actual training.

from qgre.types import RewardResult


# Quality keys matching STEP_QUALITIES in the engine
ALL_QUALITIES = [
    # Step 1
    "q_format_tags",
    "q_tag_content",
    "q_node_in_prompt",
    "q_node_format",
    "q_node_length",
    # Step 2
    "q_chain_s2_refs_s1",
    # Step 3
    "q_chain_s3_refs_s2",
    "q_self_consistency",
    # Step 4
    "q_step4_valid_json",
    "q_step4_has_keys",
    "q_existence_correct",
    "q_archetype_correct",
    "q_node_f1",
    # Global (not step-specific, contributes to overall reward only)
    "q_eos_correct",
]

# Phase → active qualities (progressive gating)
PHASE_QUALITIES = {
    1: ["q_format_tags", "q_tag_content", "q_node_in_prompt", "q_node_format", "q_node_length"],
    2: [
        "q_format_tags",
        "q_tag_content",
        "q_node_in_prompt",
        "q_node_format",
        "q_node_length",
        "q_chain_s2_refs_s1",
    ],
    3: [
        "q_format_tags",
        "q_tag_content",
        "q_node_in_prompt",
        "q_node_format",
        "q_node_length",
        "q_chain_s2_refs_s1",
        "q_chain_s3_refs_s2",
        "q_self_consistency",
    ],
    4: ALL_QUALITIES,
}


def reward_fn(prompt: str, completion: str, meta: dict | None = None) -> RewardResult:
    """Stub reward function returning synthetic scores.

    Always returns phase 1 with perfect format scores and zero content scores.
    Replace with real reward_fn for actual training.
    """
    scores = dict.fromkeys(ALL_QUALITIES, 0.0)

    # Step 1 qualities: check if completion has any step tags (simple heuristic)
    has_step1 = "<step1_extraction>" in completion
    has_step4 = "<step4_output>" in completion

    if has_step1:
        scores["q_format_tags"] = 1.0
        scores["q_tag_content"] = 0.8
        scores["q_node_in_prompt"] = 0.7
        scores["q_node_format"] = 1.0
        scores["q_node_length"] = 0.9

    if has_step4:
        scores["q_step4_valid_json"] = 0.5
        scores["q_step4_has_keys"] = 0.5

    phase = 1
    active = PHASE_QUALITIES[phase]
    active_scores = [scores[q] for q in active if q in scores]
    total = sum(active_scores) / max(len(active_scores), 1)

    return RewardResult(reward=total, scores=scores, phase=phase)
