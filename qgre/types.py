"""Core types for QGRE Engine."""

from __future__ import annotations

import logging
import random
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

import torch

logger = logging.getLogger(__name__)


CHECKPOINT_SCHEMA_VERSION: int = 1


@dataclass
class TrainingContext:
    """Training context — device, dtype, step counter, and checkpoint schema version.

    Passed explicitly through the pipeline (no thread-local/global state).
    Provides device/dtype for tensor construction and tracks training progress.
    """

    device: torch.device
    dtype: torch.dtype = torch.float32
    step: int = 0
    checkpoint_schema_version: int = CHECKPOINT_SCHEMA_VERSION

    @classmethod
    def from_config(cls, config, device: str = "cuda") -> TrainingContext:
        """Factory method to construct TrainingContext from QGREConfig.

        Args:
            config: QGREConfig instance (unused in v1, reserved for future dtype config)
            device: Device string ("cuda", "cpu", etc.)

        Returns:
            TrainingContext instance

        Raises:
            ValueError: If device string is invalid or unavailable
        """
        try:
            device_obj = torch.device(device)
        except RuntimeError as e:
            raise ValueError(f"Invalid device string '{device}': {e}") from e

        return cls(
            device=device_obj,
            dtype=torch.float32,
            step=0,
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict:
        """Serialize to dict. Device is converted to string for JSON compatibility."""
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "step": self.step,
            "checkpoint_schema_version": self.checkpoint_schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingContext:
        """Deserialize from dict. Device string is converted back to torch.device.

        Raises:
            ValueError: If required keys are missing or device/dtype strings are invalid
        """
        # Validate required keys
        required_keys = ["device", "dtype", "step"]
        missing_keys = [key for key in required_keys if key not in d]
        if missing_keys:
            raise ValueError(f"Missing required keys in TrainingContext dict: {missing_keys}")

        # Parse dtype string (e.g. "torch.float32" → torch.float32)
        dtype_str = d["dtype"]
        if dtype_str.startswith("torch."):
            try:
                dtype = getattr(torch, dtype_str.split(".", 1)[1])
            except AttributeError as e:
                raise ValueError(f"Invalid dtype string '{dtype_str}': {e}") from e
        else:
            raise ValueError(f"Invalid dtype string '{dtype_str}': expected 'torch.<dtype>' format")

        # Validate device string
        try:
            device = torch.device(d["device"])
        except RuntimeError as e:
            raise ValueError(f"Invalid device string '{d['device']}': {e}") from e

        return cls(
            device=device,
            dtype=dtype,
            step=d["step"],
            checkpoint_schema_version=d.get("checkpoint_schema_version", CHECKPOINT_SCHEMA_VERSION),
        )


# ── StateSpec Dataclasses ──


@dataclass
class TrainerState:
    """All mutable trainer state in one place.

    Tracks accumulation progress, resumption flags, and sync requirements.
    Serializable checkpoint component.
    """
    global_step: int = 0
    accumulated_loss: float = 0.0
    accumulation_count: int = 0
    accumulated_samples: int = 0
    resumed_mid_accumulation: bool = False
    fused_validated: bool = False
    needs_weight_sync: bool = False


@dataclass
class DataLoaderState:
    """Epoch tracking, sampling weights, curriculum gating.

    Manages iteration state and dynamic priority scheduling.
    Serializable checkpoint component.
    """
    epoch: int = 0
    step_in_epoch: int = 0
    total_steps: int = 0
    priority_weights: list[float] | None = None
    difficulty_gate: tuple[set[str], str] | None = None


@dataclass
class AdvantageEstimatorState:
    """Estimator config and accumulated statistics.

    Tracks clip parameters and region mapping for advantage computation.
    Serializable checkpoint component.
    """
    clip_advantage: float = 5.0
    step_region_map: dict | None = None


@dataclass
class WeightLoaderState:
    """LoRA initialization flags and cleanup tracking.

    Prevents double-initialization and tracks resource lifecycle.
    Serializable checkpoint component.
    """
    load_lora_called: bool = False
    initialized: bool = False
    cleaned_up: bool = False


@dataclass
class CheckpointState:
    """Master container holding all component states.

    Provides validation on construction and forward compatibility
    for schema evolution.
    """
    trainer: TrainerState
    dataloader: DataLoaderState
    advantage_estimator: AdvantageEstimatorState
    weight_loader: WeightLoaderState
    game_state: GameState
    vprm_critic_state: dict | None = None
    vprm_optimizer_state: dict | None = None
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self):
        """Validate field presence, types, and schema compatibility.

        Raises:
            ValueError: If required fields are missing
            TypeError: If field types are incorrect
        """
        # Validate required fields
        required_fields = [
            "trainer", "dataloader", "advantage_estimator",
            "weight_loader", "game_state"
        ]
        for field_name in required_fields:
            if not hasattr(self, field_name):
                raise ValueError(f"Missing required field: {field_name}")
            if getattr(self, field_name, None) is None:
                raise ValueError(f"Required field cannot be None: {field_name}")

        # Validate field types
        type_checks = [
            ("trainer", TrainerState),
            ("dataloader", DataLoaderState),
            ("advantage_estimator", AdvantageEstimatorState),
            ("weight_loader", WeightLoaderState),
            ("game_state", GameState),
        ]
        for field_name, expected_type in type_checks:
            value = getattr(self, field_name)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Field '{field_name}' has incorrect type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}"
                )

        # Validate optional field types
        if self.vprm_critic_state is not None and not isinstance(self.vprm_critic_state, dict):
            raise TypeError(
                f"Field 'vprm_critic_state' must be dict or None, "
                f"got {type(self.vprm_critic_state).__name__}"
            )
        if self.vprm_optimizer_state is not None and not isinstance(self.vprm_optimizer_state, dict):
            raise TypeError(
                f"Field 'vprm_optimizer_state' must be dict or None, "
                f"got {type(self.vprm_optimizer_state).__name__}"
            )

        # Check for extra fields (forward compatibility warning)
        expected_fields = {
            "trainer", "dataloader", "advantage_estimator", "weight_loader",
            "game_state", "vprm_critic_state", "vprm_optimizer_state", "schema_version"
        }
        actual_fields = set(self.__dict__.keys())
        extra_fields = actual_fields - expected_fields
        if extra_fields:
            logger.warning(
                f"CheckpointState contains unexpected fields (forward compatibility): "
                f"{sorted(extra_fields)}"
            )

        # Validate schema version
        if not isinstance(self.schema_version, int):
            raise TypeError(
                f"Field 'schema_version' must be int, got {type(self.schema_version).__name__}"
            )


class SkillStatus(Enum):
    LOCKED = "locked"
    ACTIVE = "active"
    MASTERED = "mastered"


@dataclass
class PromptContext:
    """First-class prompt identity — computed once at batch boundary, used everywhere.

    Eliminates stringly-typed prompt resolution scattered across subsystems.
    Built by the trainer from batch + GameState, then flows through
    advantage computation, VPRM, tutorial recording, and gate composition.
    """
    prompt_id: int              # Original hash ID from dataloader
    skill_key: str | None       # Tutorial skill this prompt belongs to (None if untracked)
    tier: str                   # Difficulty tier from metadata (e.g. "tutorial_gravity")
    aspiration_target: float    # Per-skill mastery_threshold or global default
    aspiration_warmup: float    # 0→1 ramp factor for recently unlocked skills
    is_active: bool             # Passes both tier gate AND skill gate

    @property
    def prompt_id_str(self) -> str:
        return str(self.prompt_id)


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
    scored_spans: dict = field(default_factory=dict)
    # scored_spans: {"quality_name": [(char_start, char_end), ...], ...}
    # Character offsets into the completion text. When populated, the engine
    # uses these for per-token advantage assignment instead of the segmenter.
    # Reward functions that don't populate this field get the legacy segmenter path.


@dataclass
class SkillNode:
    """A node in the tutorial skill dependency DAG."""
    name: str
    prompts: list[str]
    prerequisites: list[str]
    mastery_threshold: float = 0.8
    regression_threshold: float = 0.6
    mastery_window: int = 20
    review_probability: float = 0.15
    score_key: str | None = None  # Quality key from RewardResult.scores; None = overall reward
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=20))
    _was_mastered: bool = field(default=False, repr=False)
    _status: SkillStatus = field(default=SkillStatus.LOCKED, repr=False)
    _total_completions: int = field(default=0, repr=False)
    _initial_scores: list = field(default_factory=list, repr=False)  # First 5 scores after unlock
    _initial_mastery_logged: bool = field(default=False, repr=False)
    _unlock_step: int | None = field(default=None, repr=False)  # Step when skill was unlocked
    _pre_unlock_baseline: float | None = field(default=None, repr=False)  # Mastery score of prerequisites at unlock time
    aspiration_warmup_steps: int = 20  # Ramp aspiration from 0→full over N steps after unlock
    # Learnability-based advancement: advance only when variance collapses
    learnability_threshold: float = 0.10  # Advance when p(1-p) < this

    def __post_init__(self):
        if not self.prerequisites:
            self._status = SkillStatus.ACTIVE
        # Always sync deque maxlen with mastery_window
        self.recent_scores = deque(self.recent_scores, maxlen=self.mastery_window)

    @property
    def mastered(self) -> bool:
        if len(self.recent_scores) < self.mastery_window:
            return False
        score = self.mastery_score
        if self._was_mastered:
            return score >= self.regression_threshold
        return score >= self.mastery_threshold

    @property
    def mastery_score(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    @property
    def learnability(self) -> float:
        """Bernoulli variance = p(1-p). Maximized at p=0.5, zero at p=0 or p=1.

        High learnability = skill still has variance = still learning.
        Low learnability = skill is stable = ready to advance.
        """
        if len(self.recent_scores) < 5:
            # Not enough data — return max learnability to prevent premature advancement
            # 0.25 = variance at p=0.5 (maximum uncertainty)
            return 0.25
        p = self.mastery_score
        return p * (1.0 - p)

    @property
    def ready_to_advance(self) -> bool:
        """Ready when: mastery > threshold AND learnability < stale_threshold.

        High mastery + low learnability = skill is learned, move on.
        High mastery + high learnability = skill still has variance, stay.
        """
        if len(self.recent_scores) < self.mastery_window:
            return False
        if self.mastery_score < self.mastery_threshold:
            return False
        # Only advance when variance has collapsed (skill is stable)
        return self.learnability < self.learnability_threshold

    def unlocked(self, skill_tree: dict[str, SkillNode]) -> bool:
        return all(skill_tree[pre].mastered for pre in self.prerequisites)

    @property
    def status(self) -> SkillStatus:
        return self._status

    def record_score(self, v_correct: float):
        self.recent_scores.append(v_correct)
        self._total_completions += 1
        # Track initial mastery (first 5 scores after unlock) for transfer_lift
        if len(self._initial_scores) < 5:
            self._initial_scores.append(v_correct)
        if self.mastered:
            self._was_mastered = True
        elif self._was_mastered and not self.mastered:
            self._was_mastered = False

    @property
    def initial_mastery(self) -> float | None:
        """Mastery score on first 5 completions after unlock. None if < 5 completions."""
        if len(self._initial_scores) < 5:
            return None
        return sum(self._initial_scores) / len(self._initial_scores)


QUALITY_WINDOW_SIZE = 20


class StagnationStatus(Enum):
    NORMAL = "normal"
    STAGNATING = "stagnating"
    STUCK = "stuck"


@dataclass
class GameState:
    """QGRE 2D mastery matrix — tracks quality × difficulty independently.

    Two axes:
    - Quality phases (1→N): format → identification → equations → full derivation
    - Difficulty tiers: tier1 → tier2 → tier3 → ... (domain-specific)

    Each cell mastery[tier][phase] has its own rolling window. Quality phases
    advance per-tier. Tier N+1 unlocks when tier N reaches a configurable
    quality phase threshold.

    When no tiers are configured, all prompts map to a single "default" tier,
    making this equivalent to the original 1D phase system.
    """

    step_count: int = 0
    mastery_threshold: float = 0.8
    stagnation_timeout: int = 200
    plateau_window: int = 50
    plateau_threshold: float = 0.02
    quality_window_size: int = 20  # Rolling window for mastery score tracking

    # Tutorial skill tree
    tutorial_enabled: bool = False
    skill_tree: dict = field(default_factory=dict)  # str → SkillNode
    all_prompts: list = field(default_factory=list)
    post_mastery_behavior: str = "review_only"
    untracked_always_active: bool = True
    _sequential_mastery: bool = False
    default_aspiration_target: float = 0.8
    _prompt_to_skill: dict = field(default_factory=dict, repr=False)
    _tier_to_skills: dict = field(default_factory=dict, repr=False)  # tier_name → [skill_keys]
    _active_base_pool: list = field(default_factory=list, repr=False)
    _mastered_pool: list = field(default_factory=list, repr=False)
    _active_prompts_dirty: bool = field(default=True, repr=False)
    _pool_version: int = field(default=0, repr=False)

    # 2D mastery matrix
    tier_mastery: dict = field(default_factory=dict)
    # tier_mastery[tier][step_num] = deque of scores
    tier_phases: dict = field(default_factory=lambda: {"default": 1})
    # Per-tier quality phase
    active_tiers: list = field(default_factory=lambda: ["default"])
    # Currently unlocked tiers
    tier_steps_at_phase_start: dict = field(default_factory=dict)
    # tier_steps_at_phase_start[tier] = step_count when tier's current phase started
    phase_history: list = field(default_factory=list)
    # [(step_count, tier, old_phase, new_phase), ...]

    # ── Tutorial skill tree ──

    def init_tutorial(self, tutorial_config, all_prompt_ids: list[str] | None = None,
                      dataloader_items: list[dict] | None = None,
                      difficulty_column: str | None = None):
        """Initialize skill tree from TutorialConfig. Call after construction.

        Args:
            tutorial_config: TutorialConfig from parsed YAML.
            all_prompt_ids: List of all prompt IDs (as strings) in the dataset.
            dataloader_items: Raw dataloader items for metadata-based prompt matching.
                Each item has 'prompt_id' (int) and 'metadata' (dict).
            difficulty_column: Metadata column for tier mapping (e.g. "difficulty").
        """
        from qgre.config import TutorialConfig
        if not isinstance(tutorial_config, TutorialConfig) or not tutorial_config.enabled:
            self.tutorial_enabled = False
            return

        if not tutorial_config.skill_tree:
            raise ValueError(
                "tutorial.enabled=true but skill_tree is empty. "
                "Define at least one skill with prompts or match_metadata."
            )

        self.tutorial_enabled = True
        self.post_mastery_behavior = tutorial_config.post_mastery_behavior
        self.untracked_always_active = tutorial_config.untracked_always_active
        self._sequential_mastery = tutorial_config.sequential_mastery
        self.all_prompts = list(all_prompt_ids) if all_prompt_ids else []

        self.skill_tree = {}
        for key, sc in tutorial_config.skill_tree.items():
            # Resolve prompts from match_metadata if configured
            prompts = list(sc.prompts)
            if sc.match_metadata and not dataloader_items:
                warnings.warn(
                    f"Skill '{key}' has match_metadata={sc.match_metadata} but no "
                    f"dataloader_items provided. Metadata resolution skipped — "
                    f"skill will have only explicit prompts: {sc.prompts}"
                )
            if sc.match_metadata and dataloader_items:
                for item in dataloader_items:
                    meta = item.get("metadata", {})
                    if all(meta.get(col) == val for col, val in sc.match_metadata.items()):
                        pid_str = str(item["prompt_id"])
                        if pid_str not in prompts:
                            prompts.append(pid_str)

            self.skill_tree[key] = SkillNode(
                name=key,
                prompts=prompts,
                prerequisites=list(sc.prerequisites),
                mastery_threshold=sc.mastery_threshold,
                regression_threshold=sc.regression_threshold,
                mastery_window=sc.mastery_window,
                review_probability=sc.review_probability,
                score_key=sc.score_key,
                aspiration_warmup_steps=sc.aspiration_warmup_steps,
                learnability_threshold=sc.learnability_threshold,
                recent_scores=deque(maxlen=sc.mastery_window),
            )

            if not prompts:
                # Check if this skill blocks any dependents — deadlock if so
                dependents = [k for k, s in tutorial_config.skill_tree.items() if key in s.prerequisites]
                if dependents:
                    raise ValueError(
                        f"Skill '{key}' has no prompts after metadata resolution and blocks {dependents}. "
                        f"Training would deadlock. match_metadata={sc.match_metadata}, "
                        f"explicit prompts={sc.prompts}"
                    )
                warnings.warn(
                    f"Skill '{key}' has no prompts after metadata resolution. "
                    f"match_metadata={sc.match_metadata}, explicit prompts={sc.prompts}"
                )

        # Build O(1) prompt → skill lookup
        self._prompt_to_skill = {}
        for key, node in self.skill_tree.items():
            for prompt_id in node.prompts:
                self._prompt_to_skill[prompt_id] = key

        # Build tier → skills mapping (which skills have prompts in which tiers)
        self._tier_to_skills = {}
        if dataloader_items and difficulty_column:
            for item in dataloader_items:
                pid_str = str(item["prompt_id"])
                tier = item.get("metadata", {}).get(difficulty_column, "default")
                skill_key = self._prompt_to_skill.get(pid_str)
                if skill_key is not None:
                    self._tier_to_skills.setdefault(tier, set()).add(skill_key)
            # Convert sets to lists for serialization
            self._tier_to_skills = {k: list(v) for k, v in self._tier_to_skills.items()}
            logger.info(f"[TUTORIAL] Tier→skill mapping: {self._tier_to_skills}")

        self._active_prompts_dirty = True

        # Validate
        self.validate_skill_tree()

    def validate_skill_tree(self):
        """Validate DAG structure. Raise ValueError on any violation."""
        # 1. Cycle detection (topological sort)
        visited = set()
        temp = set()

        def visit(key):
            if key in temp:
                raise ValueError(f"Cycle detected in skill tree involving: {key}")
            if key in visited:
                return
            temp.add(key)
            for pre in self.skill_tree[key].prerequisites:
                if pre not in self.skill_tree:
                    raise ValueError(
                        f"Skill '{key}' has prerequisite '{pre}' "
                        f"which does not exist in skill_tree"
                    )
                visit(pre)
            temp.remove(key)
            visited.add(key)

        for key in self.skill_tree:
            visit(key)

        # 2. Duplicate prompts
        seen_prompts = {}
        for key, node in self.skill_tree.items():
            for prompt_id in node.prompts:
                if prompt_id in seen_prompts:
                    raise ValueError(
                        f"Prompt '{prompt_id}' appears in both "
                        f"'{seen_prompts[prompt_id]}' and '{key}'"
                    )
                seen_prompts[prompt_id] = key

        # 3. At least one root skill
        roots = [k for k, n in self.skill_tree.items() if not n.prerequisites]
        if not roots:
            raise ValueError(
                "No root skills (skills with empty prerequisites). "
                "Training cannot start."
            )

        # 4. regression_threshold < mastery_threshold
        for key, node in self.skill_tree.items():
            if node.regression_threshold >= node.mastery_threshold:
                raise ValueError(
                    f"Skill '{key}': regression_threshold ({node.regression_threshold}) "
                    f"must be less than mastery_threshold ({node.mastery_threshold})"
                )

        logger.info(f"Skill tree validated: {len(self.skill_tree)} skills, "
                    f"{len(roots)} roots, {len(seen_prompts)} prompts mapped")

    def _invalidate_prompt_cache(self):
        self._active_prompts_dirty = True
        self._pool_version += 1

    def snapshot_pool_version(self) -> int:
        """Snapshot pool version. Call before recording, compare after."""
        return self._pool_version

    def did_prompt_pool_change(self, snapshot: int) -> bool:
        """Check if prompt pool changed since snapshot."""
        return self._pool_version != snapshot

    def get_active_prompts(self) -> list[str]:
        """Return prompt IDs eligible for sampling this step."""
        if not self.tutorial_enabled:
            return self.all_prompts

        # Rebuild deterministic pool only on state change
        if self._active_prompts_dirty:
            self._active_base_pool = []
            self._mastered_pool = []

            # Collect active (unlocked + not mastered) and mastered skills
            active_skills = []
            for key, node in self.skill_tree.items():
                is_unlocked = node.unlocked(self.skill_tree)
                if is_unlocked and not node.mastered:
                    active_skills.append(node)
                elif node.mastered:
                    self._mastered_pool.append(node)

            if self._sequential_mastery and active_skills:
                # One skill at a time: only the first active skill's prompts
                self._active_base_pool = list(active_skills[0].prompts)
            else:
                for node in active_skills:
                    self._active_base_pool.extend(node.prompts)

            if self.untracked_always_active:
                tracked = set(self._prompt_to_skill.keys())
                untracked = [p for p in self.all_prompts if p not in tracked]
                self._active_base_pool.extend(untracked)

            self._active_prompts_dirty = False

        active = list(self._active_base_pool)

        # Review sampling for mastered skills
        for node in self._mastered_pool:
            if random.random() < node.review_probability:
                active.extend(node.prompts)

        # Post-mastery behavior: all skills mastered, no active base pool
        if not self._active_base_pool:
            if self.post_mastery_behavior == 'pause' and not active:
                logger.info("[TUTORIAL] All skills mastered. post_mastery_behavior=pause. "
                           "Returning empty pool — trainer should stop.")
                return []
            elif self.post_mastery_behavior == 'continue_all':
                logger.info("[TUTORIAL] All skills mastered. post_mastery_behavior=continue_all.")
                return self.all_prompts
            # review_only: return whatever review sampling produced (may be empty this step)
            if active:
                return active
            # review_only but no review samples this step — return mastered prompts directly
            # rather than falling back to all (which would include locked skills)
            review_fallback = []
            for node in self._mastered_pool:
                review_fallback.extend(node.prompts)
            if review_fallback:
                return review_fallback
            # review_only with nothing to review — return empty, trainer handles it
            logger.info("[TUTORIAL] review_only: no review prompts this step. Returning empty pool.")
            return []

        if not active:
            logger.warning("Tutorial system: no active prompts this step. Falling back to all.")
            return self.all_prompts

        return active

    def record_completion(self, prompt_id: str, v_correct: float):
        """Route a completion score to the appropriate skill node.

        Uses learnability-based advancement: skills only transition to MASTERED
        when ready_to_advance=True (mastery threshold + low variance). This
        prevents premature advancement when variance is still high.

        Regression still uses mastered (with hysteresis) for stability.
        """
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is None:
            return  # Untracked prompt

        node = self.skill_tree[skill_key]
        was_mastered_before = node.mastered  # For regression hysteresis
        was_ready_before = node.ready_to_advance  # For advancement gating
        node.record_score(v_correct)
        is_mastered_now = node.mastered
        is_ready_now = node.ready_to_advance

        # Advancement: uses ready_to_advance (mastery + low learnability variance)
        # This prevents advancing when skill is still high-variance (in ZPD)
        if is_ready_now and not was_ready_before:
            node._status = SkillStatus.MASTERED
            logger.info(f"[TUTORIAL] SKILL MASTERED: {node.name} "
                       f"(mastery={node.mastery_score:.2f}, "
                       f"learnability={node.learnability:.3f}, "
                       f"threshold={node.mastery_threshold}, "
                       f"learnability_threshold={node.learnability_threshold})")
            self._invalidate_prompt_cache()
            self._check_unlocks()

        # Regression: uses mastered (hysteresis threshold) for stability
        # Don't relock just because variance spiked temporarily
        elif was_mastered_before and not is_mastered_now:
            node._status = SkillStatus.ACTIVE
            logger.warning(f"[TUTORIAL] MASTERY REGRESSION: {node.name} "
                          f"(mastery={node.mastery_score:.2f}, "
                          f"regression_threshold={node.regression_threshold})")
            self._invalidate_prompt_cache()
            self._check_relocks()

    def _check_unlocks(self):
        """After a mastery event, check if any locked skills should unlock."""
        for key, node in self.skill_tree.items():
            if node.status == SkillStatus.LOCKED and node.unlocked(self.skill_tree):
                node._status = SkillStatus.ACTIVE
                node._unlock_step = self.step_count
                # Capture pre-unlock baseline: mean mastery of prerequisites at unlock time
                pre_scores = [self.skill_tree[pre].mastery_score for pre in node.prerequisites]
                node._pre_unlock_baseline = sum(pre_scores) / len(pre_scores) if pre_scores else 0.0
                self._invalidate_prompt_cache()
                logger.info(f"[TUTORIAL] SKILL UNLOCKED: {node.name} "
                           f"(prerequisites met: {node.prerequisites}, "
                           f"step={self.step_count}, "
                           f"pre_unlock_baseline={node._pre_unlock_baseline:.3f})")

    def _check_relocks(self):
        """After a regression, cascade re-lock to dependents with full mastery reset."""
        changed = True
        while changed:
            changed = False
            for key, node in self.skill_tree.items():
                if node.status in (SkillStatus.ACTIVE, SkillStatus.MASTERED) and not node.unlocked(self.skill_tree):
                    if node.prerequisites:
                        prev_status = node.status
                        node._status = SkillStatus.LOCKED
                        node._was_mastered = False
                        node.recent_scores.clear()
                        node._total_completions = 0
                        node._initial_scores.clear()
                        node._initial_mastery_logged = False
                        node._unlock_step = None
                        node._pre_unlock_baseline = None
                        changed = True
                        self._invalidate_prompt_cache()
                        logger.warning(f"[TUTORIAL] CASCADE RE-LOCK: {node.name} "
                                      f"(was {prev_status}, prerequisite lost mastery, "
                                      f"mastery state reset)")

    def resolve_mastery_score(self, prompt_id: str, reward_result) -> float:
        """Extract the mastery-tracking score for a prompt from a RewardResult.

        Uses the skill's score_key if configured, otherwise falls back to overall reward.
        """
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is not None:
            node = self.skill_tree[skill_key]
            if node.score_key is not None:
                if node.score_key in reward_result.scores:
                    return reward_result.scores[node.score_key]
                warnings.warn(
                    f"Skill '{node.name}' score_key='{node.score_key}' not in "
                    f"reward_result.scores (keys: {list(reward_result.scores.keys())}). "
                    f"Falling back to overall reward."
                )
        return reward_result.reward

    def build_prompt_contexts(
        self,
        prompt_ids: list[int],
        metadata: list[dict],
        difficulty_column: str | None = None,
        active_tiers: set[str] | None = None,
    ) -> list[PromptContext]:
        """Build PromptContext for each prompt in a batch. Computed once, used everywhere."""
        active_prompt_set = set(self.get_active_prompts()) if self.tutorial_enabled else None
        contexts = []
        for i, pid in enumerate(prompt_ids):
            pid_str = str(pid)
            skill_key = self._prompt_to_skill.get(pid_str) if self.tutorial_enabled else None
            meta = metadata[i] if i < len(metadata) else {}
            tier = meta.get(difficulty_column, "default") if difficulty_column else "default"

            # Aspiration target: per-skill if tutorial enabled, else global
            if skill_key is not None:
                asp_target = self.skill_tree[skill_key].mastery_threshold
            else:
                asp_target = self.default_aspiration_target

            # Active: tutorial-tracked prompts in active skills bypass tier gate.
            # Untracked prompts respect tier gate. Locked skill prompts are always inactive.
            if skill_key is not None and active_prompt_set is not None:
                # Tutorial-tracked: skill gate is the authority
                is_active = pid_str in active_prompt_set
            else:
                # Untracked: tier gate decides
                is_active = active_tiers is None or tier in active_tiers

            # Aspiration warmup: dampen after tutorial skill unlock OR tier phase advance
            warmup = self.get_aspiration_warmup_factor(pid_str) if self.tutorial_enabled else 1.0
            # Also dampen after tier phase advance — prevents gradient shock on new phases
            tier_warmup = self._get_tier_phase_warmup(tier)
            warmup = min(warmup, tier_warmup)

            contexts.append(PromptContext(
                prompt_id=pid,
                skill_key=skill_key,
                tier=tier,
                aspiration_target=asp_target,
                aspiration_warmup=warmup,
                is_active=is_active,
            ))
        return contexts

    def can_tier_unlock(self, tier_name: str) -> bool:
        """Check if tutorial prerequisites allow this tier to unlock.

        A tier can unlock only if all PREREQUISITE skills of skills in that tier
        are MASTERED. This prevents premature tier advancement before the model
        has demonstrated consistent mastery on the prerequisite tutorial skills.

        For example, to unlock "combined" tier (which has gravity_spring prompts),
        the prerequisites of gravity_spring (freefall, spring_only) must be mastered.

        Mastery requires mastery_window completions (default 20) at or above
        mastery_threshold. This ensures the model has seen enough examples before
        advancing to harder tiers.

        Uses _tier_to_skills mapping built during init_tutorial.
        """
        if not self.tutorial_enabled:
            return True
        skills_in_tier = self._tier_to_skills.get(tier_name, [])
        if not skills_in_tier:
            return True

        # Collect all prerequisite skills that must be mastered
        prereq_skills = set()
        for skill_key in skills_in_tier:
            node = self.skill_tree[skill_key]
            for prereq in node.prerequisites:
                prereq_skills.add(prereq)

        # Check each prerequisite is mastered
        for prereq_key in prereq_skills:
            prereq_node = self.skill_tree[prereq_key]
            if not prereq_node.mastered:
                completions = len(prereq_node.recent_scores)
                needed = prereq_node.mastery_window
                score = prereq_node.mastery_score
                threshold = prereq_node.mastery_threshold
                print(f"  │ ⏳ Tier '{tier_name}' blocked — prerequisite skill '{prereq_key}' not mastered "
                      f"({completions}/{needed} completions, score {score:.2f}/{threshold:.2f})")
                return False
        return True

    def get_aspiration_target(self, prompt_id: str) -> float:
        """Return the mastery_threshold for the skill this prompt belongs to."""
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is not None:
            return self.skill_tree[skill_key].mastery_threshold
        return self.default_aspiration_target

    def _get_tier_phase_warmup(self, tier: str, warmup_steps: int = 20) -> float:
        """Return aspiration warmup for recently-advanced tier phases.

        Ramps from 0→1 over warmup_steps after a tier phase advance.
        Prevents gradient shock from full aspiration pressure when the model
        encounters harder quality requirements at the new phase.
        """
        if tier not in self.tier_steps_at_phase_start:
            return 1.0  # No record → full aspiration
        steps_since = self.step_count - self.tier_steps_at_phase_start[tier]
        if steps_since >= warmup_steps:
            return 1.0
        return max(0.0, steps_since / warmup_steps)

    def get_aspiration_warmup_factor(self, prompt_id: str) -> float:
        """Return aspiration warmup multiplier (0→1) for recently unlocked skills.

        Ramps aspiration beta linearly from 0 to 1 over aspiration_warmup_steps
        after a skill unlocks. Prevents gradient shock from full aspiration pressure
        on prompts the model has never seen.

        Returns 1.0 for root skills, mastered skills, and untracked prompts.
        """
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is None:
            return 1.0
        node = self.skill_tree[skill_key]
        if node._unlock_step is None:
            return 1.0  # Root skill (never locked) or not yet tracked
        steps_since_unlock = self.step_count - node._unlock_step
        if steps_since_unlock >= node.aspiration_warmup_steps:
            return 1.0
        return max(0.0, steps_since_unlock / node.aspiration_warmup_steps)

    def get_tutorial_metrics(self) -> dict:
        """Return per-step tutorial metrics for logging."""
        if not self.tutorial_enabled:
            return {}
        metrics = {}
        active = sum(1 for n in self.skill_tree.values() if n.status == SkillStatus.ACTIVE)
        mastered = sum(1 for n in self.skill_tree.values() if n.status == SkillStatus.MASTERED)
        locked = sum(1 for n in self.skill_tree.values() if n.status == SkillStatus.LOCKED)
        metrics['tutorial/active_skills'] = active
        metrics['tutorial/mastered_skills'] = mastered
        metrics['tutorial/locked_skills'] = locked
        metrics['tutorial/total_skills'] = len(self.skill_tree)
        # Active prompt pool size (deterministic — rebuild cache if needed, but don't include random review sampling)
        if self._active_prompts_dirty:
            self.get_active_prompts()  # Rebuild cache (result discarded — we only want the deterministic pools)
        metrics['tutorial/active_prompt_pool_size'] = len(self._active_base_pool)
        metrics['tutorial/mastered_review_pool_size'] = sum(len(n.prompts) for n in self._mastered_pool)
        for key, node in self.skill_tree.items():
            metrics[f'tutorial/skill/{key}/mastery'] = node.mastery_score
            metrics[f'tutorial/skill/{key}/status'] = node.status.value
            metrics[f'tutorial/skill/{key}/completions'] = len(node.recent_scores)
            metrics[f'tutorial/skill/{key}/aspiration_target'] = node.mastery_threshold
            # Aspiration warmup factor (visible when ramping)
            if node._unlock_step is not None:
                steps_since = self.step_count - node._unlock_step
                if steps_since < node.aspiration_warmup_steps:
                    metrics[f'tutorial/skill/{key}/aspiration_warmup'] = steps_since / node.aspiration_warmup_steps
            # Transfer lift: initial mastery after unlock (logged once at 5 completions)
            if node.initial_mastery is not None and not node._initial_mastery_logged:
                metrics[f'tutorial/skill/{key}/initial_mastery'] = node.initial_mastery
                # Compute transfer_lift: how much did prerequisites help?
                transfer_lift = node.initial_mastery - (node._pre_unlock_baseline or 0.0)
                metrics[f'tutorial/skill/{key}/transfer_lift'] = transfer_lift
                node._initial_mastery_logged = True
                # Log prominently — this is the key metric for validating the tutorial system
                logger.info(
                    f"[TUTORIAL] *** TRANSFER LIFT: {key} ***\n"
                    f"  initial_mastery = {node.initial_mastery:.3f} (first 5 completions)\n"
                    f"  pre_unlock_baseline = {node._pre_unlock_baseline or 0:.3f} (prerequisite mastery at unlock)\n"
                    f"  transfer_lift = {transfer_lift:+.3f} "
                    f"({'POSITIVE — prerequisites helped' if transfer_lift > 0 else 'ZERO/NEGATIVE — check prerequisite structure'})"
                )
        return metrics

    def tutorial_state_dict(self) -> dict:
        """Serialize tutorial state for checkpointing."""
        if not self.tutorial_enabled:
            return {}
        return {
            'skill_tracker': {
                key: {
                    'scores': list(node.recent_scores),
                    'was_mastered': node._was_mastered,
                    'status': node._status.value,
                    'total_completions': node._total_completions,
                    'initial_scores': list(node._initial_scores),
                    'initial_mastery_logged': node._initial_mastery_logged,
                    'unlock_step': node._unlock_step,
                    'pre_unlock_baseline': node._pre_unlock_baseline,
                }
                for key, node in self.skill_tree.items()
            }
        }

    def load_tutorial_state_dict(self, state: dict):
        """Restore tutorial state from checkpoint."""
        if 'skill_tracker' not in state:
            if self.tutorial_enabled:
                logger.warning("[TUTORIAL] No tutorial state in checkpoint — starting fresh")
            return
        for key, data in state['skill_tracker'].items():
            if key not in self.skill_tree:
                logger.warning(f"[TUTORIAL] Checkpoint skill '{key}' not in current config — dropping")
                continue
            if isinstance(data, list):
                scores, was_mastered, status = data, False, None
            else:
                scores = data['scores']
                was_mastered = data.get('was_mastered', False)
                status = data.get('status', None)
            self.skill_tree[key].recent_scores = deque(
                scores, maxlen=self.skill_tree[key].mastery_window
            )
            self.skill_tree[key]._was_mastered = was_mastered
            if status is not None:
                self.skill_tree[key]._status = SkillStatus(status) if isinstance(status, str) else status
            if isinstance(data, dict):
                self.skill_tree[key]._total_completions = data.get('total_completions', 0)
                self.skill_tree[key]._initial_scores = data.get('initial_scores', [])
                self.skill_tree[key]._initial_mastery_logged = data.get('initial_mastery_logged', False)
                self.skill_tree[key]._unlock_step = data.get('unlock_step', None)
                self.skill_tree[key]._pre_unlock_baseline = data.get('pre_unlock_baseline', None)
        self._active_prompts_dirty = True

    # ── 1D backward compat properties ──

    @property
    def phase(self) -> int:
        """Global phase = min phase across active tiers. For 1D compat."""
        if not self.tier_phases:
            return 1
        active = [self.tier_phases[t] for t in self.active_tiers if t in self.tier_phases]
        return min(active) if active else 1

    @property
    def step_mastery(self) -> dict:
        """Legacy: return default tier's mastery windows."""
        return self.tier_mastery.get("default", {})

    # ── 2D mastery tracking ──

    def record_tier_step_score(self, tier: str, step_num: int, score: float):
        """Record a quality score for a tier+step cell."""
        if tier not in self.tier_mastery:
            self.tier_mastery[tier] = {}
        if step_num not in self.tier_mastery[tier]:
            self.tier_mastery[tier][step_num] = deque(maxlen=self.quality_window_size)
        self.tier_mastery[tier][step_num].append(score)

    def get_tier_step_mastery(self, tier: str, step_num: int) -> float:
        """Get the mean quality score for a tier+step cell."""
        windows = self.tier_mastery.get(tier, {})
        window = windows.get(step_num)
        if not window:
            return 0.0
        return sum(window) / len(window)

    min_observations_before_advance: int = 10

    def check_tier_phase_advance(self, tier: str, max_phase: int) -> bool:
        """Check if a tier's current quality phase is mastered. Advance if so."""
        current_phase = self.tier_phases.get(tier, 1)
        if current_phase >= max_phase:
            return False

        observations = self.tier_mastery.get(tier, {}).get(current_phase)
        not_enough_observations = observations is not None and len(observations) < self.min_observations_before_advance
        if not_enough_observations:
            return False

        mastery = self.get_tier_step_mastery(tier, current_phase)
        if mastery >= self.mastery_threshold:
            import logging
            _logger = logging.getLogger("qgre.types")
            old_phase = current_phase
            self.tier_phases[tier] = current_phase + 1
            self.phase_history.append((self.step_count, tier, old_phase, current_phase + 1))
            self.tier_steps_at_phase_start[tier] = self.step_count
            _logger.warning(f"[PHASE ADVANCE] tier={tier}, {old_phase}→{current_phase+1}, step={self.step_count}, mastery={mastery:.3f}")
            return True
        return False

    def check_tier_unlock(
        self, tier_order: list[str], unlock_phase: int, unlock_threshold: float,
    ) -> str | None:
        """Check if the next tier should be unlocked.

        Finds the first inactive tier in tier_order. Checks if ALL active tiers
        before it have reached unlock_phase with mastery >= unlock_threshold.
        Returns the newly unlocked tier name, or None.
        """
        active_set = set(self.active_tiers)

        # Find first inactive tier in tier_order
        next_tier = None
        next_idx = -1
        for i, t in enumerate(tier_order):
            if t not in active_set:
                next_tier = t
                next_idx = i
                break

        if next_tier is None:
            return None  # All tiers already unlocked

        # All active tiers before next_tier must have reached unlock_phase with threshold
        for t in tier_order[:next_idx]:
            if t not in active_set:
                continue  # Skip tiers not in active set (shouldn't happen in ordered unlock)
            phase = self.tier_phases.get(t, 1)
            if phase < unlock_phase:
                return None  # This tier hasn't reached the required quality phase
            mastery = self.get_tier_step_mastery(t, unlock_phase)
            if mastery < unlock_threshold:
                return None  # This tier hasn't mastered the required phase

        # All active tiers before next_tier are ready — unlock it
        self.active_tiers.append(next_tier)
        self.tier_phases[next_tier] = 1
        self.tier_steps_at_phase_start[next_tier] = self.step_count
        return next_tier

    def check_tier_stagnation(self, tier: str) -> StagnationStatus:
        """Check if a specific tier is stagnating in its current phase."""
        start = self.tier_steps_at_phase_start.get(tier, 0)
        steps_in_phase = self.step_count - start
        if steps_in_phase >= self.stagnation_timeout:
            return StagnationStatus.STUCK

        current_phase = self.tier_phases.get(tier, 1)
        windows = self.tier_mastery.get(tier, {})
        window = windows.get(current_phase)
        if window and len(window) >= self.plateau_window:
            recent = list(window)[-self.plateau_window:]
            half = len(recent) // 2
            first_half_mean = sum(recent[:half]) / half
            second_half_mean = sum(recent[half:]) / (len(recent) - half)
            if abs(second_half_mean - first_half_mean) < self.plateau_threshold:
                return StagnationStatus.STAGNATING

        return StagnationStatus.NORMAL

    # ── 1D compat methods (delegate to "default" tier) ──

    def record_step_score(self, step_num: int, score: float):
        """Legacy 1D: record to default tier."""
        self.record_tier_step_score("default", step_num, score)

    def get_step_mastery(self, step_num: int) -> float:
        """Legacy 1D: read from default tier."""
        return self.get_tier_step_mastery("default", step_num)

    def check_phase_advance(self, max_phase: int) -> bool:
        """Legacy 1D: advance default tier's phase."""
        return self.check_tier_phase_advance("default", max_phase)

    def check_stagnation(self) -> StagnationStatus:
        """Legacy 1D: check default tier stagnation."""
        return self.check_tier_stagnation("default")
