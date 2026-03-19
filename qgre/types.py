"""Core types for QGRE Engine."""

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass
class RewardResult:
    """Output of a reward function evaluation.

    The reward_fn scores completions and returns per-quality scores.
    The engine uses .scores to compute per-step advantages and manage phase gating.
    Phase is engine-managed via GameState. reward_fn should NOT set phase.
    """

    reward: float
    scores: dict = field(default_factory=dict)  # {"quality_name": float, ...}
    phase: int = 1  # Engine-managed — set by GameState, not reward_fn


QUALITY_WINDOW_SIZE = 20


@dataclass
class GameState:
    """QGRE curriculum state — engine-managed phase advancement.

    The engine tracks quality scores per step, computes mastery, and advances
    phases when thresholds are met. This is the QGRE curriculum: step 1 mastery
    unlocks step 2 qualities.

    phase: current curriculum phase (1 = only step 1 qualities active)
    step_mastery: per-step quality windows for mastery tracking
    mastery_threshold: quality mean required to advance to next phase
    """

    phase: int = 1
    step_count: int = 0
    mastery_threshold: float = 0.8
    step_mastery: dict = field(default_factory=dict)
    # {step_num: deque([mean_quality_scores...], maxlen=QUALITY_WINDOW_SIZE)}
    phase_history: list = field(default_factory=list)

    def record_step_score(self, step_num: int, score: float):
        """Record a quality score for a step. Used by engine after each training step."""
        if step_num not in self.step_mastery:
            self.step_mastery[step_num] = deque(maxlen=QUALITY_WINDOW_SIZE)
        self.step_mastery[step_num].append(score)

    def get_step_mastery(self, step_num: int) -> float:
        """Get the mean quality score for a step over the mastery window."""
        window = self.step_mastery.get(step_num)
        if not window:
            return 0.0
        return sum(window) / len(window)

    def check_phase_advance(self, max_phase: int) -> bool:
        """Check if the current phase's step has mastered. Advance if so.

        QGRE rule: phase N requires step N mastery >= mastery_threshold.
        Returns True if phase advanced.
        """
        if self.phase >= max_phase:
            return False

        current_mastery = self.get_step_mastery(self.phase)
        if current_mastery >= self.mastery_threshold:
            self.phase += 1
            self.phase_history.append(self.step_count)
            return True
        return False
