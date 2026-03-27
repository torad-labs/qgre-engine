"""Core types for QGRE Engine."""

from __future__ import annotations

import logging
import random
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


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

    # Tutorial skill tree
    tutorial_enabled: bool = False
    skill_tree: dict = field(default_factory=dict)  # str → SkillNode
    all_prompts: list = field(default_factory=list)
    post_mastery_behavior: str = "review_only"
    untracked_always_active: bool = True
    default_aspiration_target: float = 0.8
    _prompt_to_skill: dict = field(default_factory=dict, repr=False)
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
                      dataloader_items: list[dict] | None = None):
        """Initialize skill tree from TutorialConfig. Call after construction.

        Args:
            tutorial_config: TutorialConfig from parsed YAML.
            all_prompt_ids: List of all prompt IDs (as strings) in the dataset.
            dataloader_items: Raw dataloader items for metadata-based prompt matching.
                Each item has 'prompt_id' (int) and 'metadata' (dict).
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

            for key, node in self.skill_tree.items():
                is_unlocked = node.unlocked(self.skill_tree)
                if is_unlocked and not node.mastered:
                    self._active_base_pool.extend(node.prompts)
                elif node.mastered:
                    self._mastered_pool.append(node)

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
        """Route a completion score to the appropriate skill node."""
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is None:
            return  # Untracked prompt

        node = self.skill_tree[skill_key]
        was_mastered_before = node.mastered
        node.record_score(v_correct)
        is_mastered_now = node.mastered

        if is_mastered_now and not was_mastered_before:
            node._status = SkillStatus.MASTERED
            logger.info(f"[TUTORIAL] SKILL MASTERED: {node.name} "
                       f"(mastery={node.mastery_score:.2f}, "
                       f"threshold={node.mastery_threshold})")
            self._invalidate_prompt_cache()
            self._check_unlocks()

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
                self._invalidate_prompt_cache()
                logger.info(f"[TUTORIAL] SKILL UNLOCKED: {node.name} "
                           f"(prerequisites met: {node.prerequisites})")

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

            # Active: must pass tier gate AND skill gate
            tier_active = active_tiers is None or tier in active_tiers
            skill_active = active_prompt_set is None or pid_str in active_prompt_set
            is_active = tier_active and skill_active

            contexts.append(PromptContext(
                prompt_id=pid,
                skill_key=skill_key,
                tier=tier,
                aspiration_target=asp_target,
                is_active=is_active,
            ))
        return contexts

    def get_aspiration_target(self, prompt_id: str) -> float:
        """Return the mastery_threshold for the skill this prompt belongs to."""
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is not None:
            return self.skill_tree[skill_key].mastery_threshold
        return self.default_aspiration_target

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
            # Transfer lift: initial mastery after unlock (logged once at 5 completions)
            if node.initial_mastery is not None and not node._initial_mastery_logged:
                metrics[f'tutorial/skill/{key}/initial_mastery'] = node.initial_mastery
                node._initial_mastery_logged = True
                logger.info(f"[TUTORIAL] INITIAL MASTERY: {key} = {node.initial_mastery:.3f} "
                           f"(first 5 completions after unlock)")
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
            self.tier_mastery[tier][step_num] = deque(maxlen=QUALITY_WINDOW_SIZE)
        self.tier_mastery[tier][step_num].append(score)

    def get_tier_step_mastery(self, tier: str, step_num: int) -> float:
        """Get the mean quality score for a tier+step cell."""
        windows = self.tier_mastery.get(tier, {})
        window = windows.get(step_num)
        if not window:
            return 0.0
        return sum(window) / len(window)

    def check_tier_phase_advance(self, tier: str, max_phase: int) -> bool:
        """Check if a tier's current quality phase is mastered. Advance if so."""
        current_phase = self.tier_phases.get(tier, 1)
        if current_phase >= max_phase:
            return False

        mastery = self.get_tier_step_mastery(tier, current_phase)
        if mastery >= self.mastery_threshold:
            old_phase = current_phase
            self.tier_phases[tier] = current_phase + 1
            self.phase_history.append((self.step_count, tier, old_phase, current_phase + 1))
            self.tier_steps_at_phase_start[tier] = self.step_count
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
