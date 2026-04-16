"""QGRE Engine — Quality-Gated Reward Escalation for novel-domain GRPO training."""

__version__ = "0.1.0"

from qgre.curriculum import (
    apply_difficulty_gate,
    get_prompt_tier,
    record_mastery_and_advance,
)
from qgre.expression import (
    build_substitutions,
    math_verify_scorer,
    score_expression,
    sympy_scorer,
)
from qgre.reward_parsing import (
    StructuredOutputParser,
    extract_rhs_expressions,
    parse_structured_output,
)
from qgre.segments import Segmenter, qwen3_xml_segmenter, uniform_segmenter
from qgre.types import GameState, RewardResult


__all__ = [
    "GameState",
    "RewardResult",
    "Segmenter",
    "StructuredOutputParser",
    "apply_difficulty_gate",
    "build_substitutions",
    "extract_rhs_expressions",
    "get_prompt_tier",
    "math_verify_scorer",
    "parse_structured_output",
    "qwen3_xml_segmenter",
    "record_mastery_and_advance",
    "score_expression",
    "sympy_scorer",
    "uniform_segmenter",
]
