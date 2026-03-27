# QGRE Tutorial System — Implementation Specification

**Version:** 1.1 (DNA equation review applied)
**Author:** Marcos Damasceno, Torad Labs
**Date:** March 26, 2026
**Target:** QGRE Engine v0.3, commit c4740ba
**Audience:** Claude Code implementing agent

---

## 1. What This Does

The tutorial system transforms QGRE from "train on all prompts simultaneously" to "master prerequisites before encountering compound tasks." It controls which prompts the model sees and when, based on demonstrated mastery of prerequisite skills.

**The result:** gravity_spring (V = kx²/2 + mgx) enters the training pool only after the model has independently mastered freefall (V = mgx) and spring_only (V = kx²/2). The model arrives at the compound task already holding both component creases in its LoRA weights. The learning distance from 3.1% to 80%+ shrinks from hundreds of steps to tens.

**What this is NOT:**

- NOT a replacement for the phase gating system. Phase gates control which quality dimensions unlock (correct_H, correct_V, etc.). The tutorial system controls which prompts appear within a phase. They compose: phase gates are vertical (quality progression), the skill tree is horizontal (task prerequisite ordering).
- NOT a curriculum difficulty sorter. Standard curriculum learning sorts by difficulty. This system sorts by prerequisite structure — a DAG, not a linear difficulty scale. A skill can be easy but locked because its prerequisites aren't met.
- NOT a data augmentation system. It does not generate new prompts. It filters the existing prompt pool.

---

## 2. The Problem (Quantified)

gravity_spring has been stuck at 3.1% V_correct for 417 completions. The model writes V = kx²/2 every time. It drops the gravity term mgx.

The base model (Qwen3-1.7B, no LoRA) writes V = kx²/2 + mgx correctly. The knowledge exists. The LoRA learned to suppress it because partial credit (0.4) for the spring-only answer was consistently reinforced. The LoRA overfit to a local optimum.

Three mechanisms address this at the gradient level: aspiration gap (directional pressure), LoRA dropout (exploration), variance-aware baseline (signal preservation). The tutorial system addresses it at the curriculum level: don't show the compound task until the model has mastered the components.

Without the tutorial system, the gradient mechanisms fight a curriculum sequencing problem. The model encounters gravity_spring before it has proven it can produce mgx independently. Even with aspiration pushing and dropout exploring, the model is attempting a compound fold without clean component folds.

With the tutorial system, the model masters freefall (V = mgx, threshold 0.85) and spring_only (V = kx²/2, threshold 0.85) independently. When gravity_spring unlocks, the LoRA already encodes both component capabilities. The aspiration gap and dropout amplify rather than carry.

---

## 3. Architecture

Three components. Total implementation: ~180 lines of new code + config.

### 3.1 Skill Dependency Graph (DAG)

A directed acyclic graph declared in YAML config. Each node is a skill. Each edge is a prerequisite relationship.

**Properties per skill node:**

| Property | Type | Description |
|----------|------|-------------|
| `name` | str | Human-readable skill name |
| `prompts` | list[str] | Prompt IDs belonging to this skill |
| `prerequisites` | list[str] | Skill keys that must be mastered before this skill unlocks |
| `mastery_threshold` | float | Rolling average V_correct score required for mastery (default: 0.8) |
| `regression_threshold` | float | Rolling average below which mastery is LOST (default: mastery_threshold - 0.2). Creates hysteresis to prevent cascade oscillation from noisy review completions. |
| `mastery_window` | int | Number of recent completions to average over (default: 20) |
| `review_probability` | float | Sampling probability after mastery (default: 0.15) |

**DAG constraints:**

- No cycles. Validate at config load time. If a cycle exists, raise a configuration error with the cycle path.
- Every prompt ID in a skill's `prompts` list must exist in the engine's prompt registry. Validate at config load time.
- A prompt ID must not appear in more than one skill. One prompt, one skill. Validate at config load time.
- Skills with empty `prerequisites` lists are root skills. They are active from step 0.

**Config schema:**

```yaml
tutorial:
  enabled: true
  
  # Post-mastery behavior when ALL skills are mastered:
  #   review_only — sample only from mastered skills at review_probability (default)
  #   pause — stop training, log curriculum complete
  #   continue_all — re-enable all prompts at full sampling weight
  post_mastery_behavior: review_only
  
  skill_tree:
    # HYSTERESIS TUNING NOTE: The gap between mastery_threshold and 
    # regression_threshold controls sensitivity to noise. Start with 
    # a 0.2 gap (e.g., 0.85/0.65). If you see false regressions on 
    # compound skills with high reward variance, tighten to 0.10-0.15 
    # (e.g., 0.75/0.60). If you see skills staying mastered when the 
    # model has clearly forgotten, widen the gap.
    
    freefall:
      prompts: [freefall_1, freefall_2, freefall_3]
      prerequisites: []
      mastery_threshold: 0.85
      regression_threshold: 0.65
      mastery_window: 20
      review_probability: 0.15

    spring_only:
      prompts: [spring_1, spring_2, spring_3]
      prerequisites: []
      mastery_threshold: 0.85
      regression_threshold: 0.65
      mastery_window: 20
      review_probability: 0.15

    gravity_spring:
      prompts: [gravity_spring_1, gravity_spring_2]
      prerequisites: [freefall, spring_only]
      mastery_threshold: 0.75
      regression_threshold: 0.55
      mastery_window: 20
      review_probability: 0.15

    damped_spring:
      prompts: [damped_1, damped_2]
      prerequisites: [spring_only]
      mastery_threshold: 0.75
      regression_threshold: 0.55
      mastery_window: 20
      review_probability: 0.15

    driven_oscillator:
      prompts: [driven_1]
      prerequisites: [gravity_spring, damped_spring]
      mastery_threshold: 0.70
      regression_threshold: 0.50
      mastery_window: 20
      review_probability: 0.15

  # Fallback: if tutorial.enabled but a prompt isn't in any skill,
  # it goes into an "untracked" pool that is always sampled.
  untracked_always_active: true
```

### 3.2 Mastery Tracker

Lives in `GameState`. Tracks per-skill mastery through a rolling deque of V_correct scores.

**Data structure:**

```python
from collections import deque
from dataclasses import dataclass, field

@dataclass
class SkillNode:
    name: str
    prompts: list[str]
    prerequisites: list[str]
    mastery_threshold: float = 0.8
    regression_threshold: float = 0.6  # hysteresis: lower bar to LOSE mastery
    mastery_window: int = 20
    review_probability: float = 0.15
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=20))
    _was_mastered: bool = field(default=False, repr=False)
    _status: str = field(default='locked', repr=False)
    
    def __post_init__(self):
        """Root skills (no prerequisites) start as 'active'. Others start 'locked'."""
        if not self.prerequisites:
            self._status = 'active'
        # Ensure deque has correct maxlen if loaded from config
        if not isinstance(self.recent_scores, deque):
            self.recent_scores = deque(self.recent_scores, maxlen=self.mastery_window)
    
    @property
    def mastered(self) -> bool:
        if len(self.recent_scores) < self.mastery_window:
            return False
        score = self.mastery_score
        if self._was_mastered:
            # Hysteresis: once mastered, use lower threshold to LOSE mastery.
            # Prevents cascade oscillation from noisy review completions.
            # Three bad review completions should not collapse the skill tree.
            return score >= self.regression_threshold
        return score >= self.mastery_threshold
    
    @property 
    def mastery_score(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)
    
    def unlocked(self, skill_tree: dict) -> bool:
        """All prerequisites must be mastered."""
        return all(skill_tree[pre].mastered for pre in self.prerequisites)
    
    @property
    def status(self) -> str:
        """locked | active | mastered"""
        # Note: 'unlocked' is computed dynamically via unlocked() method
        # status is stored as 'locked', 'active', or 'mastered'
        return self._status
    
    def record_score(self, v_correct: float):
        """Record a V_correct score for this skill."""
        self.recent_scores.append(v_correct)
        # Track mastery state for hysteresis
        if self.mastered:
            self._was_mastered = True
        elif self._was_mastered and not self.mastered:
            # Dropped below regression_threshold — actually lost mastery
            self._was_mastered = False
```

**State persistence:** The mastery tracker state (recent_scores deques for all skills) must be saved and restored across checkpoints. Add to `GameState.state_dict()` and `GameState.load_state_dict()`.

```python
# In GameState.state_dict()
def state_dict(self):
    d = super().state_dict()
    d['skill_tracker'] = {
        key: {
            'scores': list(node.recent_scores),
            'was_mastered': node._was_mastered,
            'status': node._status,
        }
        for key, node in self.skill_tree.items()
    }
    return d

# In GameState.load_state_dict()
def load_state_dict(self, state):
    super().load_state_dict(state)
    if 'skill_tracker' in state:
        for key, data in state['skill_tracker'].items():
            if key in self.skill_tree:
                # Handle both old format (list) and new format (dict)
                if isinstance(data, list):
                    scores = data
                    was_mastered = False
                    status = None
                else:
                    scores = data['scores']
                    was_mastered = data.get('was_mastered', False)
                    status = data.get('status', None)
                self.skill_tree[key].recent_scores = deque(
                    scores, maxlen=self.skill_tree[key].mastery_window
                )
                self.skill_tree[key]._was_mastered = was_mastered
                if status is not None:
                    self.skill_tree[key]._status = status
```

### 3.3 Dataloader Filter

The prompt sampler queries the skill tree before each batch and returns only eligible prompts.

**Sampling logic:**

**IMPORTANT: Call `get_active_prompts()` once per step and cache the result.** The review_probability check uses `random.random()`, so each call produces a different pool. Do not call multiple times per step.

**Performance: Event-driven prompt pool caching.** The active prompt pool only changes when a mastery state transition fires (mastery gained, mastery lost, skill unlocked, skill re-locked). Between those events, the pool is static. Rebuild the cached pool only on state transitions, not every step.

**Performance: Build a prompt-to-skill lookup dict at init time.** Both `get_aspiration_target()` and `record_completion()` need to find which skill a prompt belongs to. Linear scan is O(N*M). Build `self._prompt_to_skill: dict[str, str]` mapping prompt_id → skill_key during skill tree initialization. Use it in both methods for O(1) lookup.

```python
# In GameState.__init__ after building skill_tree:
self._prompt_to_skill = {}
for key, node in self.skill_tree.items():
    for prompt_id in node.prompts:
        self._prompt_to_skill[prompt_id] = key

# Cache for active prompts — rebuilt only on state transitions
self._active_prompts_cache = None
self._active_prompts_dirty = True  # starts dirty, forces first build
```

```python
def _invalidate_prompt_cache(self):
    """Mark prompt cache as dirty. Called on any mastery state transition."""
    self._active_prompts_dirty = True

def get_active_prompts(self) -> list[str]:
    """Return prompt IDs eligible for sampling this step.
    Uses cached pool between mastery state transitions.
    Review sampling is re-rolled each call for mastered skills."""
    if not self.tutorial_enabled:
        return self.all_prompts
    
    # Rebuild deterministic pool only when state changed
    if self._active_prompts_dirty:
        self._active_base_pool = []  # skills that are active (full weight)
        self._mastered_pool = []     # skills that are mastered (review weight)
        
        for key, node in self.skill_tree.items():
            is_unlocked = node.unlocked(self.skill_tree)
            
            if is_unlocked and not node.mastered:
                self._active_base_pool.extend(node.prompts)
            elif node.mastered:
                self._mastered_pool.append(node)
        
        # Include untracked prompts if configured
        if self.untracked_always_active:
            tracked = set(self._prompt_to_skill.keys())
            untracked = [p for p in self.all_prompts if p not in tracked]
            self._active_base_pool.extend(untracked)
        
        self._active_prompts_dirty = False
    
    # Start with the deterministic base pool
    active = list(self._active_base_pool)
    
    # Add mastered skills at review probability (re-rolled each step)
    for node in self._mastered_pool:
        if random.random() < node.review_probability:
            active.extend(node.prompts)
    
    # Handle post_mastery_behavior when nothing is in active base pool
    if not self._active_base_pool and not active:
        if self.post_mastery_behavior == 'pause':
            logger.info("[TUTORIAL] All skills mastered. post_mastery_behavior=pause. "
                       "Returning empty pool — trainer should stop.")
            return []
        elif self.post_mastery_behavior == 'continue_all':
            logger.info("[TUTORIAL] All skills mastered. post_mastery_behavior=continue_all.")
            return self.all_prompts
        # review_only: active will contain review samples (or empty if unlucky)
    
    # Safety fallback
    if not active:
        logger.warning("Tutorial system: no active prompts this step. "
                      "Falling back to all.")
        return self.all_prompts
    
    return active
```

**Invalidation integration:** Call `_invalidate_prompt_cache()` in every state transition method:

```python
# Add self._invalidate_prompt_cache() after EVERY status change:
# - In _check_unlocks(), after setting node._status = 'active'
# - In _check_relocks(), after setting node._status = 'locked'
# - In record_completion(), after setting node._status = 'mastered'
# - In record_completion(), after regression (node._status = 'active')
```

**Scoring integration:** After each completion is scored, the V_correct score routes to the appropriate skill node:

```python
def record_completion(self, prompt_id: str, v_correct: float):
    """Route a completion score to the appropriate skill node."""
    for key, node in self.skill_tree.items():
        if prompt_id in node.prompts:
            was_mastered_before = node.mastered
            node.record_score(v_correct)
            is_mastered_now = node.mastered
            
            # Check for mastery GAIN
            if is_mastered_now and not was_mastered_before:
                node._status = 'mastered'
                logger.info(f"[TUTORIAL] SKILL MASTERED: {node.name} "
                          f"(mastery={node.mastery_score:.2f}, "
                          f"threshold={node.mastery_threshold})")
                self._check_unlocks()
            
            # Check for mastery LOSS (regression through hysteresis)
            elif was_mastered_before and not is_mastered_now:
                node._status = 'active'
                logger.warning(f"[TUTORIAL] MASTERY REGRESSION: {node.name} "
                             f"(mastery={node.mastery_score:.2f}, "
                             f"regression_threshold={node.regression_threshold})")
                self._check_relocks()
            
            return
    
    # Prompt not in any skill — untracked, no action needed

def _check_unlocks(self):
    """After a mastery event, check if any locked skills should unlock."""
    for key, node in self.skill_tree.items():
        if node.status == 'locked' and node.unlocked(self.skill_tree):
            node._status = 'active'
            logger.info(f"[TUTORIAL] SKILL UNLOCKED: {node.name} "
                       f"(prerequisites met: {node.prerequisites})")

def _check_relocks(self):
    """After a regression event, cascade re-lock to dependents.
    CRITICAL: re-locked skills get full mastery reset — _was_mastered 
    and recent_scores are cleared. The model must re-demonstrate mastery 
    from scratch. Old scores are stale because LoRA weights changed 
    during the prerequisite's regression/re-mastery period."""
    changed = True
    while changed:
        changed = False
        for key, node in self.skill_tree.items():
            if node.status in ('active', 'mastered') and not node.unlocked(self.skill_tree):
                if node.prerequisites:  # don't re-lock root skills
                    prev_status = node.status
                    node._status = 'locked'
                    node._was_mastered = False  # reset hysteresis
                    node.recent_scores.clear()  # clear stale scores
                    changed = True
                    logger.warning(f"[TUTORIAL] CASCADE RE-LOCK: {node.name} "
                                 f"(was {prev_status}, prerequisite lost mastery, "
                                 f"mastery state reset)")
```

---

## 4. Integration Points

### 4.1 Where it connects to existing code

| File | Change | Description |
|------|--------|-------------|
| `qgre/config.py` | ADD | `TutorialConfig` dataclass with skill_tree schema (including `regression_threshold`) |
| `qgre/game_state.py` | MODIFY | Add `SkillNode` class, skill tree initialization, `get_active_prompts()`, `record_completion()`, `get_aspiration_target()`, state persistence (including `_was_mastered` hysteresis state) |
| `qgre/trainer.py` | MODIFY | Call `get_active_prompts()` before batch construction. Call `record_completion()` after reward scoring. Route `get_aspiration_target()` to advantage computation. |
| `config.yaml` | MODIFY | Add `tutorial:` section with differentiated thresholds |
| `tests/test_tutorial.py` | ADD | Unit tests for DAG validation, mastery tracking, hysteresis, unlock logic, prompt filtering, aspiration routing |

### 4.2 Trainer integration (trainer.py)

The trainer calls the tutorial system at two points:

**Point 1: Before batch construction (prompt selection)**

```python
# In the training loop, before selecting prompts for the batch
if self.game_state.tutorial_enabled:
    eligible_prompts = self.game_state.get_active_prompts()
else:
    eligible_prompts = self.all_prompts

# Sample from eligible_prompts for this batch
batch_prompts = random.choices(eligible_prompts, k=self.batch_size)
```

**Point 2: After reward scoring (score recording)**

```python
# After rewards are computed and spans are scored
for prompt_id, scores in zip(batch_prompt_ids, batch_scores):
    v_correct = scores.get('V_correct', 0.0)
    if self.game_state.tutorial_enabled:
        self.game_state.record_completion(prompt_id, v_correct)
```

**Point 3: During advantage computation (skill-aware aspiration target)**

The aspiration gap formula uses a `target` value. When the tutorial system is enabled, this target must come from the skill's `mastery_threshold`, not from the global SPOConfig. Different skills have different difficulty levels and different thresholds. A global target miscalibrates the directional pressure.

```python
# In trainer.py, when computing advantages for a prompt
# Add a helper to GameState:
def get_aspiration_target(self, prompt_id: str) -> float:
    """Return the mastery_threshold for the skill this prompt belongs to.
    Falls back to global SPO target for untracked prompts."""
    for key, node in self.skill_tree.items():
        if prompt_id in node.prompts:
            return node.mastery_threshold
    return self.default_aspiration_target  # from SPOConfig

# In the advantage computation loop in trainer.py:
if self.game_state.tutorial_enabled:
    target = self.game_state.get_aspiration_target(prompt_id)
else:
    target = self.spo_config.aspiration_target
```

This ensures freefall pushes toward 0.85 (easy, hold to high standard) while driven_oscillator pushes toward 0.70 (hard, accept lower mastery). The aspiration pressure is calibrated per-skill.

### 4.3 Interaction with existing mechanisms

**Phase gating:** The tutorial system operates within a phase. Phase gates control which quality dimensions are active (correct_H, correct_V, etc.). The skill tree controls which prompts are sampled. A prompt can be in a skill node AND gated by a phase. Both conditions must be met: the skill must be unlocked AND the phase must be active. The skill tree does not override phase gating.

**Aspiration gap:** **MODIFIED.** The aspiration formula `A = (r - V) + β(r - target)` now uses a per-skill `target` instead of a global value. When the tutorial system is enabled, `target` is the `mastery_threshold` of the skill the prompt belongs to. Root skills (freefall at 0.85) get higher aspiration targets than compound skills (gravity_spring at 0.75). This prevents miscalibrated pressure: without per-skill routing, a compound task at threshold 0.75 would receive excess negative aspiration from a global 0.85 target, potentially overpunishing correct-but-imperfect completions. See Section 4.2, Point 3 for implementation. For untracked prompts, falls back to the global SPOConfig target.

**LoRA dropout:** No change. Dropout applies during generation regardless of which prompts are active. On simpler tasks (freefall, spring_only), dropout is less critical because the model can learn them without exploration assistance. On compound tasks (gravity_spring), dropout surfaces the combined answer. The tutorial system ensures dropout has maximum impact: when the model encounters gravity_spring, it already has both component capabilities, so dropout only needs to surface the combination, not discover both components.

**Variance-aware baseline:** No change. Per-prompt baselines track independently. When a new skill unlocks and new prompts enter the pool, their baselines start fresh (no history). The variance-aware system won't trigger slowdown on new prompts until enough completions have accumulated.

**VPRM:** No change. The critic reads hidden states from whatever completions are generated. Tutorial system controls prompt selection upstream; VPRM operates on the generated tokens downstream.

**Spans:** No change. Span extraction runs on whatever completions are produced. The span finder doesn't care which skill a prompt belongs to.

---

## 5. Logging and Metrics

The tutorial system logs to the same MLflow/logger infrastructure as the rest of the engine.

**Per-step metrics:**

| Metric | Type | Description |
|--------|------|-------------|
| `tutorial/active_skills` | int | Number of skills currently active (unlocked + not mastered) |
| `tutorial/mastered_skills` | int | Number of skills mastered |
| `tutorial/locked_skills` | int | Number of skills still locked |
| `tutorial/total_skills` | int | Total skill count |
| `tutorial/active_prompt_pool_size` | int | Number of prompts eligible for sampling this step |

**Per-skill metrics (logged every N steps, e.g., every 10):**

| Metric | Type | Description |
|--------|------|-------------|
| `tutorial/skill/{name}/mastery` | float | Current rolling mastery score |
| `tutorial/skill/{name}/status` | str | locked / active / mastered |
| `tutorial/skill/{name}/completions` | int | Total completions recorded |
| `tutorial/skill/{name}/aspiration_target` | float | The mastery_threshold used as aspiration target for this skill |
| `tutorial/skill/{name}/initial_mastery` | float | Mastery score on first 5 completions after unlock (logged once at unlock+5). Measures how much prerequisite training helped. |
| `tutorial/skill/{name}/transfer_lift` | float | initial_mastery minus pre-unlock baseline (if pre-unlock scores are available). Positive = prerequisites helped. Zero/negative = prerequisite structure needs redesigning for this skill. |

**Event logging (on state transitions):**

```
[TUTORIAL] SKILL MASTERED: freefall (mastery=0.85, threshold=0.85, step=87)
[TUTORIAL] SKILL UNLOCKED: gravity_spring (prerequisites: [freefall, spring_only], step=104)
[TUTORIAL] SKILL MASTERED: gravity_spring (mastery=0.78, threshold=0.75, step=178)
[TUTORIAL] MASTERY REGRESSION: freefall (mastery=0.58, regression_threshold=0.65, step=312)
[TUTORIAL] CASCADE RE-LOCK: gravity_spring (prerequisite freefall lost mastery, step=312)
```

---

## 6. DAG Validation

At config load time, validate the skill tree:

```python
def validate_skill_tree(skill_tree: dict, prompt_registry: list[str]):
    """Validate DAG structure. Raise ConfigError on any violation."""
    
    # 1. Check for cycles (topological sort)
    visited = set()
    temp = set()
    
    def visit(key):
        if key in temp:
            raise ConfigError(f"Cycle detected in skill tree involving: {key}")
        if key in visited:
            return
        temp.add(key)
        for pre in skill_tree[key].prerequisites:
            if pre not in skill_tree:
                raise ConfigError(
                    f"Skill '{key}' has prerequisite '{pre}' "
                    f"which does not exist in skill_tree"
                )
            visit(pre)
        temp.remove(key)
        visited.add(key)
    
    for key in skill_tree:
        visit(key)
    
    # 2. Check for duplicate prompts across skills
    seen_prompts = {}
    for key, node in skill_tree.items():
        for prompt_id in node.prompts:
            if prompt_id in seen_prompts:
                raise ConfigError(
                    f"Prompt '{prompt_id}' appears in both "
                    f"'{seen_prompts[prompt_id]}' and '{key}'"
                )
            seen_prompts[prompt_id] = key
    
    # 3. Check all prompts exist in registry
    for key, node in skill_tree.items():
        for prompt_id in node.prompts:
            if prompt_id not in prompt_registry:
                raise ConfigError(
                    f"Prompt '{prompt_id}' in skill '{key}' "
                    f"not found in prompt registry"
                )
    
    # 4. Check at least one root skill exists
    roots = [k for k, n in skill_tree.items() if not n.prerequisites]
    if not roots:
        raise ConfigError(
            "No root skills (skills with empty prerequisites). "
            "Training cannot start."
        )
    
    # 5. Check regression_threshold < mastery_threshold for all skills
    for key, node in skill_tree.items():
        if node.regression_threshold >= node.mastery_threshold:
            raise ConfigError(
                f"Skill '{key}': regression_threshold ({node.regression_threshold}) "
                f"must be less than mastery_threshold ({node.mastery_threshold}). "
                f"Hysteresis requires a gap between gaining and losing mastery."
            )
    
    logger.info(f"Skill tree validated: {len(skill_tree)} skills, "
                f"{len(roots)} roots, {len(seen_prompts)} prompts mapped")
```

---

## 7. Test Requirements

### 7.1 Unit tests (test_tutorial.py)

**DAG validation tests:**

```
test_valid_dag_passes_validation
test_cycle_raises_error
test_missing_prerequisite_raises_error
test_duplicate_prompt_raises_error
test_missing_prompt_raises_error
test_no_root_skills_raises_error
test_regression_threshold_gte_mastery_raises_error
  → regression_threshold=0.9, mastery_threshold=0.8 → ConfigError
  → regression_threshold=0.8, mastery_threshold=0.8 → ConfigError (equal not allowed)
```

**Mastery tracking tests:**

```
test_mastery_requires_full_window
  → 19 scores of 1.0, mastery still False (window=20)
  → 20th score, mastery becomes True

test_mastery_rolling_window
  → 20 scores of 1.0 (mastered), then 5 scores of 0.0
  → mastery drops below mastery_threshold but stays above regression_threshold
  → mastery still True (hysteresis protects)

test_hysteresis_prevents_oscillation
  → 20 scores of 0.9 → mastered (above 0.85 threshold)
  → 3 scores of 0.3 → rolling average drops to ~0.81
  → still mastered (above regression_threshold 0.65)
  → dependents remain unlocked

test_hysteresis_allows_genuine_regression
  → 20 scores of 0.9 → mastered
  → 15 scores of 0.2 → rolling average drops to ~0.41
  → mastery LOST (below regression_threshold 0.65)
  → _was_mastered resets to False
  → dependents re-lock

test_remastery_after_regression
  → skill mastered, then regression (drops below regression_threshold)
  → skill reverts to active
  → to re-master, must cross mastery_threshold again (0.85), not regression_threshold
  → _was_mastered only becomes True again after crossing mastery_threshold

test_score_routing
  → record_completion routes to correct skill by prompt_id
  → unknown prompt_id does not raise (untracked)
```

**Aspiration target routing tests:**

```
test_aspiration_target_per_skill
  → get_aspiration_target('freefall_1') returns 0.85 (freefall threshold)
  → get_aspiration_target('gravity_spring_1') returns 0.75 (gravity_spring threshold)
  → get_aspiration_target('driven_1') returns 0.70 (driven_oscillator threshold)

test_aspiration_target_untracked_fallback
  → get_aspiration_target('unknown_prompt') returns global SPO target

test_aspiration_target_matches_mastery_threshold
  → for every skill, get_aspiration_target(skill.prompts[0]) == skill.mastery_threshold
```

**Unlock logic tests:**

```
test_root_skills_active_at_init
  → skills with no prerequisites start as 'active'

test_locked_until_prerequisites_mastered
  → gravity_spring locked until both freefall and spring_only mastered

test_single_prerequisite_unlock
  → damped_spring unlocks when spring_only mastered (not before)

test_multi_prerequisite_unlock
  → gravity_spring requires BOTH freefall AND spring_only

test_cascading_unlock
  → master freefall + spring_only → gravity_spring unlocks
  → master spring_only → damped_spring unlocks
  → master gravity_spring + damped_spring → driven_oscillator unlocks

test_cascading_relock
  → full tree mastered and unlocked
  → freefall regresses below regression_threshold
  → gravity_spring re-locks (prerequisite lost)
  → driven_oscillator re-locks (transitive — gravity_spring re-locked)
  → damped_spring NOT re-locked (only depends on spring_only, still mastered)
  → spring_only NOT affected (root skill, no prerequisites)

test_relock_resets_mastery_state
  → gravity_spring mastered (_was_mastered=True, recent_scores full of 0.78)
  → freefall regresses → gravity_spring cascade re-locks
  → verify gravity_spring._was_mastered == False
  → verify gravity_spring.recent_scores is empty
  → freefall re-masters → gravity_spring unlocks
  → verify gravity_spring.mastered == False (no scores, can't be mastered)
  → gravity_spring must re-demonstrate mastery from scratch
  THIS TEST CATCHES: stale scores causing immediate false mastery after re-unlock
```

**Prompt filtering tests:**

```
test_only_active_skills_sampled
  → get_active_prompts returns only prompts from active skills

test_mastered_skills_review_sampling
  → mastered skills appear in output with probability ~ review_probability
  → run 1000 iterations, verify proportional representation

test_locked_skills_never_sampled
  → locked skill prompts never appear in get_active_prompts

test_untracked_prompts_always_active
  → prompts not in any skill appear when untracked_always_active=True
  → prompts not in any skill excluded when untracked_always_active=False

test_empty_active_pool_fallback
  → if somehow nothing is active, falls back to all prompts with warning

test_post_mastery_behavior_review_only
  → all skills mastered, post_mastery_behavior='review_only'
  → get_active_prompts returns only review-probability samples
  → pool is small but non-empty (statistically, over 100 calls)

test_post_mastery_behavior_pause
  → all skills mastered, post_mastery_behavior='pause'
  → get_active_prompts returns empty list
  → trainer should detect this and stop

test_post_mastery_behavior_continue_all
  → all skills mastered, post_mastery_behavior='continue_all'
  → get_active_prompts returns all prompts at full weight

test_prompt_cache_invalidation
  → get_active_prompts returns pool A
  → master a skill (state transition fires)
  → get_active_prompts returns pool B (different — newly unlocked skill included)
  → without the mastery event, pool stays at A (cache not dirty)
```

**State persistence tests:**

```
test_state_dict_roundtrip
  → add scores, save state_dict, create new GameState, load_state_dict
  → mastery scores preserved, deques match

test_checkpoint_resume_mastery
  → skill mastered before checkpoint, still mastered after resume
```

### 7.2 Integration tests

```
test_tutorial_with_aspiration_gap
  → full training step with tutorial enabled + aspiration
  → freefall prompt: verify aspiration target = 0.85 (freefall's mastery_threshold)
  → gravity_spring prompt: verify aspiration target = 0.75 (gravity_spring's mastery_threshold)
  → untracked prompt: verify aspiration target = global SPOConfig value
  → advantage magnitude changes with per-skill target

test_tutorial_with_phase_gating
  → verify both phase gate AND skill unlock conditions are checked
  → a prompt that passes skill unlock but fails phase gate is NOT sampled

test_tutorial_disabled_no_effect
  → tutorial.enabled=False, all prompts sampled normally
  → no SkillNode objects created, no filtering applied
  → aspiration uses global target (no per-skill routing)

test_hysteresis_state_persistence
  → skill mastered (_was_mastered=True), save checkpoint
  → load checkpoint, verify _was_mastered=True
  → three bad scores, verify hysteresis still protects mastery

test_status_state_persistence
  → freefall mastered, gravity_spring active (unlocked, training mid-progress)
  → save checkpoint
  → load checkpoint
  → verify freefall._status == 'mastered'
  → verify gravity_spring._status == 'active' (NOT reset to 'locked')
  → verify gravity_spring.recent_scores preserved
```

### 7.3 Existing test compatibility

All 239 existing tests must pass with `tutorial.enabled: false` (default). The tutorial system is additive. It does not modify any existing code path when disabled.

---

## 8. Build Sequence

Ordered by dependency. Each step is independently testable.

| Step | File | Lines | Risk | Description |
|------|------|-------|------|-------------|
| 1 | `qgre/config.py` | ~40 | Low | Add `TutorialConfig` dataclass (includes `post_mastery_behavior: str = "review_only"`), `SkillConfig` dataclass (includes `regression_threshold`) |
| 2 | `qgre/game_state.py` | ~80 | Medium | Add `SkillNode` class with hysteresis, DAG validation, mastery tracking, prompt filtering, `get_aspiration_target()`, state persistence (including `_was_mastered`) |
| 3 | `tests/test_tutorial.py` | ~300 | Low | All unit tests from Section 7.1 (including hysteresis, aspiration routing, cascade re-lock, cache invalidation, and post_mastery_behavior tests) |
| 4 | `qgre/trainer.py` | ~20 | Low | Wire `get_active_prompts()` before batch, `record_completion()` after scoring, `get_aspiration_target()` into advantage computation |
| 5 | `config.yaml` | ~40 | Low | Add tutorial section with differentiated thresholds and regression thresholds |
| 6 | Run existing tests | 0 | Low | Verify all 239 existing tests pass |
| 7 | Training run | 0 | Medium | Full run with tutorial enabled, watch skill progression |

**Step 1 first.** Config is pure data. No side effects. Test by importing.

**Step 2 is the core.** All logic lives here. Test with unit tests in Step 3 before wiring to trainer.

**Step 4 is three touch points.** Minimal surface area in trainer.py. The trainer calls three functions and routes data. No logic in the trainer — all logic in GameState.

---

## 9. Edge Cases

### 9.1 All skills mastered

When every skill reaches mastery, behavior is controlled by the `post_mastery_behavior` config option:

- **`review_only`** (default): `get_active_prompts()` returns only review-probability samples from mastered skills. Training continues at reduced intensity. This is maintenance mode — the model keeps its skills warm without active learning. If review sampling produces no prompts in a step (bad luck on the random roll), falls back to all prompts with a warning.
- **`pause`**: `get_active_prompts()` returns an empty list. The trainer should detect this and stop training. Log: `[TUTORIAL] All skills mastered at step {N}. Curriculum complete. post_mastery_behavior=pause.`
- **`continue_all`**: All prompts re-enter the pool at full sampling weight. The model trains on everything with no filtering. Useful when the tutorial system is used for bootstrapping and the user wants unconstrained training afterward.

If phase gating has remaining phases, phase advance may introduce new quality dimensions that reset mastery (scores drop because new quality requirements aren't met). The skill nodes track V_correct, which is one quality dimension. Phase advance adding new requirements (e.g., correct_coefficients) doesn't reset V_correct mastery but may require new skills to be added for the new quality dimensions.

### 9.2 Mastery regression

A mastered skill can lose mastery, but **hysteresis prevents oscillation**. Once mastered (crossed `mastery_threshold` going up), the skill only loses mastery if the rolling average drops below `regression_threshold` (a lower bar, typically mastery_threshold - 0.2). This prevents three noisy review completions from collapsing the entire downstream skill tree.

Example with freefall (mastery_threshold=0.85, regression_threshold=0.65):
- Mastery achieved at score 0.86. `_was_mastered = True`.
- A few bad review completions drop score to 0.72. Still above 0.65. **Mastery retained.**
- Score drops to 0.62. Below regression_threshold. **Mastery lost.** `_was_mastered = False`.
- Dependent skills (gravity_spring) re-check prerequisites. Freefall no longer mastered. **Dependents re-lock.**

The system handles regression as follows:

- If `mastered` property returns False after `_was_mastered` was True, the skill reverts to `active` and `_was_mastered` resets to False.
- Dependent skills that were unlocked because of this skill's mastery must re-check. If the prerequisite is no longer mastered, dependents re-lock.
- This cascading re-lock is handled by `_check_unlocks()` running after every score recording.
- Log the regression: `[TUTORIAL] MASTERY REGRESSION: {name} (mastery={score:.2f}, regression_threshold={threshold})`
- Log the cascade: `[TUTORIAL] CASCADE RE-LOCK: {dependent_name} (prerequisite {name} lost mastery)`

**Why hysteresis, not hard threshold:** Without hysteresis, the review sampling (15% probability) becomes a destabilizing force. Mastered skills are sampled infrequently, so a few bad completions can dominate the rolling window. Three scores of 0.3 in a row on a window of 20 (where 17 previous scores were 0.9) drops the average from 0.85 to 0.76 — below 0.8 but well above a reasonable "actually forgot" threshold. Hysteresis treats this as noise, not regression. Only genuine forgetting (sustained poor performance across many completions) triggers re-lock.

### 9.3 Prompt added to skill mid-training

If the config is updated mid-training to add prompts to a skill:
- New prompts enter the skill's prompt list.
- Mastery tracking continues with the existing rolling window (adding prompts doesn't reset mastery).
- New prompts are immediately eligible for sampling if the skill is active.

### 9.4 Tutorial enabled mid-training

If tutorial is toggled from `false` to `true` mid-training:
- Skill nodes are created with empty score deques.
- All root skills start as `active`.
- All non-root skills start as `locked`.
- The model must re-demonstrate mastery from scratch. No mastery is assumed from pre-tutorial training.

---

## 10. Configuration Recommendations

For the current physics domain:

```yaml
tutorial:
  enabled: true
  skill_tree:
    # --- ROOT SKILLS (no prerequisites, higher threshold — easy tasks held to high standard) ---
    freefall:
      prompts: [freefall_1, freefall_2, freefall_3]
      prerequisites: []
      mastery_threshold: 0.85
      regression_threshold: 0.65
      mastery_window: 20
      review_probability: 0.15

    spring_only:
      prompts: [spring_1, spring_2, spring_3]
      prerequisites: []
      mastery_threshold: 0.85
      regression_threshold: 0.65
      mastery_window: 20
      review_probability: 0.15

    kinetic_energy:
      prompts: [ke_1, ke_2]
      prerequisites: []
      mastery_threshold: 0.85
      regression_threshold: 0.65
      mastery_window: 20
      review_probability: 0.15

    # --- COMPOUND SKILLS (lower threshold — harder tasks, accept lower mastery) ---
    gravity_spring:
      prompts: [gravity_spring_1, gravity_spring_2]
      prerequisites: [freefall, spring_only]
      mastery_threshold: 0.75
      regression_threshold: 0.55
      mastery_window: 20
      review_probability: 0.15

    full_hamiltonian:
      prompts: [hamiltonian_1, hamiltonian_2]
      prerequisites: [gravity_spring, kinetic_energy]
      mastery_threshold: 0.75
      regression_threshold: 0.55
      mastery_window: 20
      review_probability: 0.15

    damped_spring:
      prompts: [damped_1, damped_2]
      prerequisites: [spring_only]
      mastery_threshold: 0.75
      regression_threshold: 0.55
      mastery_window: 20
      review_probability: 0.15

    # --- BOSS LEVEL (lowest threshold — multiple interacting components) ---
    driven_oscillator:
      prompts: [driven_1]
      prerequisites: [full_hamiltonian, damped_spring]
      mastery_threshold: 0.70
      regression_threshold: 0.50
      mastery_window: 20
      review_probability: 0.15

  untracked_always_active: true
```

**Why differentiated thresholds:** Root skills (single-term expressions) are easy. The model should demonstrate strong mastery (0.85) before moving on — partial mastery on foundations creates compounding errors downstream. Compound skills are harder because they require combining multiple creases simultaneously. A 0.75 threshold prevents permanent blocking: if the model consistently reaches 0.76 on gravity_spring but never 0.80, a uniform threshold means the curriculum stalls forever. The aspiration gap still pushes toward improvement (the target IS the threshold), but the unlock gate is calibrated to difficulty. Boss-level skills (driven_oscillator) accept 0.70 — three simultaneous folds at high accuracy is substantially harder than any individual fold.

**Training configuration for the run:**

```yaml
# Existing mechanisms — all enabled
spo:
  target_aware: true
  aspiration_beta: 0.5
  var_aware: true
  var_threshold: 0.01

generation:
  top_p: 1.0
  lora_dropout_rate: 0.15
  lora_dropout_anneal_steps: 2000
```

---

## 11. Success Criteria

The tutorial system succeeds if:

1. **gravity_spring V_correct rises above 0.75** within 200 steps of unlocking (compared to 417 steps of stagnation at 3.1% without the tutorial system).
2. **Freefall and spring_only master independently** in under 100 steps each (mastery_threshold: 0.85).
3. **No false regression on mastered skills** during compound skill training — hysteresis prevents noisy review completions from triggering cascade re-lock.
4. **Aspiration targets are skill-specific** — freefall aspiration pushes toward 0.85, gravity_spring toward 0.75, driven_oscillator toward 0.70. Verified in training logs.
5. **Transfer lift is positive** for compound skills — gravity_spring's initial mastery after unlock is measurably higher than its pre-unlock baseline. The prerequisite structure causes real transfer, not just sequencing.
6. **All 239 existing tests pass** with tutorial disabled.
7. **New unit tests all pass** (Section 7.1, ~34 unit tests + 5 integration tests).
8. **Cascading unlock fires correctly** when monitored in training logs.

---

## 12. Future Extensions (Not In This Spec)

These are planned but NOT part of this implementation. Do not build them now.

- **Token injection (Phase 2):** When variance detects a stuck skill, generate reference completions from the base model and inject them into the RL pipeline. Requires the tutorial system to identify stuck skills. The variance-aware detection per-skill is the trigger.
- **Adaptive mastery threshold:** Lower the threshold for skills that have been active for too long without mastery. If a skill hasn't mastered in 200 steps, reduce threshold from 0.8 to 0.7. Prevents permanent blocking.
- **Skill-specific dropout rates:** Higher LoRA dropout for locked skills approaching unlock (the model needs more exploration when a new task is about to appear). Requires per-skill dropout rate in config.
- **Cross-skill transfer analysis:** The `transfer_lift` metric (Section 5) captures raw data. Future work: automated analysis that flags skills where transfer_lift is zero or negative, suggesting the prerequisite structure needs redesigning. Also: track whether mastery on one skill accelerates mastery on a related skill over time, not just at unlock. Evidence for the "origami crease" hypothesis — if learning V = mgx makes learning V = kx²/2 + mgx faster, the folds compose.
- **Automatic skill tree construction:** Given a set of prompts and reward functions, automatically discover the prerequisite structure by measuring transfer. Fully automated Game Engine construction.
- **System prompt fading:** Full instructions at tutorial phase, progressively reduced instructions as skills master. The model graduates from guided to autonomous reasoning. The system prompt IS the Vygotskian scaffold — its withdrawal as mastery is demonstrated is the scaffold removal. This is part of Marcos's original game engine vision and connects directly to the token injection architecture.
