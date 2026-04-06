"""Test fixtures for QGRE engine tests."""

import pytest

from qgre.types import GameState, RewardResult


# --- GPU test gating (from: stackoverflow.com/questions/47559524) ---
# Tests marked @pytest.mark.gpu are SKIPPED by default.
# Pass --gpu to run them: pytest tests/ --gpu


def pytest_addoption(parser):
    parser.addoption("--gpu", action="store_true", default=False, help="run GPU tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--gpu"):
        return  # --gpu given: run everything
    skip_gpu = pytest.mark.skip(reason="need --gpu option to run")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
        if "slow" in item.keywords and not config.getoption("--gpu"):
            item.add_marker(skip_gpu)


@pytest.fixture
def mock_game_state() -> GameState:
    """GameState with non-trivial state for serialization tests."""
    gs = GameState(step_count=150, mastery_threshold=0.8)
    gs.tier_phases = {"default": 3}
    gs.phase_history = [(50, "default", 1, 2), (100, "default", 2, 3)]

    # Per-step mastery scores
    gs.record_step_score(1, 0.95)
    gs.record_step_score(1, 0.90)
    gs.record_step_score(1, 0.92)
    gs.record_step_score(2, 0.85)
    gs.record_step_score(2, 0.80)
    gs.record_step_score(3, 0.60)

    return gs


@pytest.fixture
def synthetic_reward_results() -> list[RewardResult]:
    """List of RewardResult with controlled per-quality scores."""
    return [
        RewardResult(
            reward=0.85,
            scores={
                "q_format_tags": 1.0,
                "q_tag_content": 0.9,
                "q_node_in_prompt": 0.8,
                "q_node_format": 1.0,
                "q_node_length": 0.9,
                "q_chain_s2_refs_s1": 0.7,
                "q_chain_s3_refs_s2": 0.6,
                "q_self_consistency": 0.8,
                "q_step4_valid_json": 1.0,
                "q_step4_has_keys": 1.0,
                "q_existence_correct": 0.5,
                "q_archetype_correct": 0.0,
                "q_node_f1": 0.3,
            },
            phase=4,
        ),
        RewardResult(
            reward=0.4,
            scores={
                "q_format_tags": 1.0,
                "q_tag_content": 1.0,
                "q_node_in_prompt": 0.5,
                "q_node_format": 0.0,
                "q_node_length": 0.0,
                "q_chain_s2_refs_s1": 0.0,
                "q_chain_s3_refs_s2": 0.0,
                "q_self_consistency": 0.0,
                "q_step4_valid_json": 0.0,
                "q_step4_has_keys": 0.0,
                "q_existence_correct": 0.0,
                "q_archetype_correct": 0.0,
                "q_node_f1": 0.0,
            },
            phase=4,
        ),
    ]


# Qwen3 verified token IDs from SPECIAL-TOKENS-SUPERPOWER.md
THINK_START = 151667
THINK_END = 151668
STEP_TOKEN = 9520
OPEN_ANGLE = 27
CLOSE_SLASH = 522
STEP_1_NUM = 16  # '1'
STEP_2_NUM = 17  # '2'
STEP_3_NUM = 18  # '3'
STEP_4_NUM = 19  # '4'
CLOSE_ANGLE = 29  # '>'


@pytest.fixture
def known_token_ids() -> list[int]:
    """Hand-crafted token IDs with known step boundaries.

    Structure: <think>..think..</think><step1_X>..content..</step1_X><step2_Y>..content..</step2_Y>
    Using real Qwen3 token IDs.
    """
    # Simplified: we use placeholder content tokens (100-110)
    return [
        # <think> block
        THINK_START,
        100,
        101,
        102,
        THINK_END,
        # <step1_extraction> (opening tag: < step 1 _ extraction >)
        OPEN_ANGLE,
        STEP_TOKEN,
        STEP_1_NUM,
        94842,
        CLOSE_ANGLE,
        # step 1 content
        103,
        104,
        105,
        # </step1_extraction> (closing tag: </ step 1 _ extraction >)
        CLOSE_SLASH,
        STEP_TOKEN,
        STEP_1_NUM,
        94842,
        CLOSE_ANGLE,
        # <step2_shared_context>
        OPEN_ANGLE,
        STEP_TOKEN,
        STEP_2_NUM,
        20405,
        8467,
        CLOSE_ANGLE,
        # step 2 content
        106,
        107,
        # </step2_shared_context>
        CLOSE_SLASH,
        STEP_TOKEN,
        STEP_2_NUM,
        20405,
        8467,
        CLOSE_ANGLE,
    ]
