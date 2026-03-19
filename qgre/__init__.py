"""QGRE Engine — Quality-Gated Reward Escalation for novel-domain GRPO training."""

__version__ = "0.1.0"

from qgre.segments import Segmenter, qwen3_xml_segmenter, uniform_segmenter
from qgre.types import GameState, RewardResult

__all__ = [
    "GameState",
    "RewardResult",
    "Segmenter",
    "qwen3_xml_segmenter",
    "uniform_segmenter",
]
