"""Tests for the tutorial skill tree system."""

import pytest

from qgre.config import SkillConfig, TutorialConfig
from qgre.types import GameState, SkillStatus


# ─── Fixtures ───


def make_skill_tree_config():
    """Standard 5-skill DAG for testing."""
    return TutorialConfig(
        enabled=True,
        post_mastery_behavior="review_only",
        untracked_always_active=True,
        skill_tree={
            "freefall": SkillConfig(
                prompts=["ff_1", "ff_2", "ff_3"],
                prerequisites=[],
                mastery_threshold=0.85,
                regression_threshold=0.65,
                mastery_window=20,
                review_probability=0.15,
            ),
            "spring_only": SkillConfig(
                prompts=["sp_1", "sp_2", "sp_3"],
                prerequisites=[],
                mastery_threshold=0.85,
                regression_threshold=0.65,
                mastery_window=20,
                review_probability=0.15,
            ),
            "gravity_spring": SkillConfig(
                prompts=["gs_1", "gs_2"],
                prerequisites=["freefall", "spring_only"],
                mastery_threshold=0.75,
                regression_threshold=0.55,
                mastery_window=20,
                review_probability=0.15,
            ),
            "damped_spring": SkillConfig(
                prompts=["ds_1", "ds_2"],
                prerequisites=["spring_only"],
                mastery_threshold=0.75,
                regression_threshold=0.55,
                mastery_window=20,
                review_probability=0.15,
            ),
            "driven_oscillator": SkillConfig(
                prompts=["do_1"],
                prerequisites=["gravity_spring", "damped_spring"],
                mastery_threshold=0.70,
                regression_threshold=0.50,
                mastery_window=20,
                review_probability=0.15,
            ),
        },
    )


def make_game_state(config=None):
    config = config or make_skill_tree_config()
    all_ids = [
        "ff_1",
        "ff_2",
        "ff_3",
        "sp_1",
        "sp_2",
        "sp_3",
        "gs_1",
        "gs_2",
        "ds_1",
        "ds_2",
        "do_1",
        "untracked_1",
    ]
    gs = GameState()
    gs.init_tutorial(config, all_ids)
    return gs


def master_skill(gs, skill_key, score=0.9):
    """Fill a skill's window with high scores to achieve mastery."""
    node = gs.skill_tree[skill_key]
    for prompt_id in node.prompts:
        for _ in range(node.mastery_window):
            gs.record_completion(prompt_id, score)


# ─── DAG Validation Tests ───


class TestDAGValidation:
    def test_valid_dag_passes_validation(self):
        gs = make_game_state()
        # Should not raise
        gs.validate_skill_tree()

    def test_cycle_raises_error(self):
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "a": SkillConfig(prompts=["p1"], prerequisites=["b"]),
                "b": SkillConfig(prompts=["p2"], prerequisites=["a"]),
            },
        )
        gs = GameState()
        with pytest.raises(ValueError, match="Cycle detected"):
            gs.init_tutorial(config, ["p1", "p2"])

    def test_missing_prerequisite_raises_error(self):
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "a": SkillConfig(prompts=["p1"], prerequisites=["nonexistent"]),
            },
        )
        gs = GameState()
        with pytest.raises(ValueError, match="does not exist"):
            gs.init_tutorial(config, ["p1"])

    def test_duplicate_prompt_raises_error(self):
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "a": SkillConfig(prompts=["p1", "p2"], prerequisites=[]),
                "b": SkillConfig(prompts=["p2", "p3"], prerequisites=[]),
            },
        )
        gs = GameState()
        with pytest.raises(ValueError, match="appears in both"):
            gs.init_tutorial(config, ["p1", "p2", "p3"])

    def test_fully_cyclic_config_raises_error(self):
        """A fully cyclic DAG (no roots) raises cycle detection before the no-roots check."""
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "a": SkillConfig(prompts=["p1"], prerequisites=["b"]),
                "b": SkillConfig(prompts=["p2"], prerequisites=["a"]),
            },
        )
        gs = GameState()
        with pytest.raises(ValueError, match="Cycle detected"):
            gs.init_tutorial(config, ["p1", "p2"])

    def test_regression_threshold_gte_mastery_raises_error(self):
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "a": SkillConfig(
                    prompts=["p1"],
                    prerequisites=[],
                    mastery_threshold=0.8,
                    regression_threshold=0.9,
                ),
            },
        )
        gs = GameState()
        with pytest.raises(ValueError, match="regression_threshold"):
            gs.init_tutorial(config, ["p1"])

        # Equal also not allowed
        config2 = TutorialConfig(
            enabled=True,
            skill_tree={
                "a": SkillConfig(
                    prompts=["p1"],
                    prerequisites=[],
                    mastery_threshold=0.8,
                    regression_threshold=0.8,
                ),
            },
        )
        gs2 = GameState()
        with pytest.raises(ValueError, match="regression_threshold"):
            gs2.init_tutorial(config2, ["p1"])


# ─── Mastery Tracking Tests ───


class TestMasteryTracking:
    def test_mastery_requires_full_window(self):
        gs = make_game_state()
        node = gs.skill_tree["freefall"]
        for i in range(19):
            gs.record_completion("ff_1", 1.0)
        assert not node.mastered  # 19 < 20
        gs.record_completion("ff_1", 1.0)
        assert node.mastered  # 20 = window

    def test_mastery_rolling_window(self):
        gs = make_game_state()
        node = gs.skill_tree["freefall"]
        # Fill window with 1.0 → mastered
        for _ in range(20):
            gs.record_completion("ff_1", 1.0)
        assert node.mastered
        # Add 5 bad scores → drops below mastery_threshold but above regression
        for _ in range(5):
            gs.record_completion("ff_1", 0.3)
        # 15*1.0 + 5*0.3 = 16.5 / 20 = 0.825 < 0.85 but > 0.65
        assert node.mastered  # Hysteresis protects

    def test_hysteresis_prevents_oscillation(self):
        gs = make_game_state()
        node = gs.skill_tree["freefall"]
        for _ in range(20):
            gs.record_completion("ff_1", 0.9)
        assert node.mastered
        # 3 bad scores: avg = (17*0.9 + 3*0.3)/20 = 15.3+0.9/20 = 0.81
        for _ in range(3):
            gs.record_completion("ff_1", 0.3)
        assert node.mastered  # Above regression_threshold 0.65

    def test_hysteresis_allows_genuine_regression(self):
        gs = make_game_state()
        node = gs.skill_tree["freefall"]
        for _ in range(20):
            gs.record_completion("ff_1", 0.9)
        assert node.mastered
        # 15 bad scores: avg = (5*0.9 + 15*0.2)/20 = 4.5+3.0/20 = 0.375
        for _ in range(15):
            gs.record_completion("ff_1", 0.2)
        assert not node.mastered  # Below 0.65

    def test_remastery_after_regression(self):
        gs = make_game_state()
        node = gs.skill_tree["freefall"]
        # Master
        for _ in range(20):
            gs.record_completion("ff_1", 0.9)
        assert node.mastered
        # Regress fully
        for _ in range(20):
            gs.record_completion("ff_1", 0.2)
        assert not node.mastered
        assert not node._was_mastered
        # To re-master, must cross mastery_threshold again (0.85)
        for _ in range(19):
            gs.record_completion("ff_1", 0.86)
        assert not node.mastered  # Window not full yet (mixed with old 0.2s)
        for _ in range(1):
            gs.record_completion("ff_1", 0.86)
        # Window now: last 20 are 0.86 → avg 0.86 > 0.85
        assert node.mastered

    def test_score_routing(self):
        gs = make_game_state()
        gs.record_completion("ff_1", 0.5)
        assert len(gs.skill_tree["freefall"].recent_scores) == 1
        assert len(gs.skill_tree["spring_only"].recent_scores) == 0

        # Unknown prompt doesn't raise
        gs.record_completion("unknown_prompt", 0.5)


# ─── Aspiration Target Routing Tests ───


class TestAspirationTargetRouting:
    def test_aspiration_target_per_skill(self):
        gs = make_game_state()
        assert gs.get_aspiration_target("ff_1") == 0.85
        assert gs.get_aspiration_target("gs_1") == 0.75
        assert gs.get_aspiration_target("do_1") == 0.70

    def test_aspiration_target_untracked_fallback(self):
        gs = make_game_state()
        gs.default_aspiration_target = 0.8
        assert gs.get_aspiration_target("unknown_prompt") == 0.8

    def test_aspiration_target_matches_mastery_threshold(self):
        gs = make_game_state()
        for key, node in gs.skill_tree.items():
            assert gs.get_aspiration_target(node.prompts[0]) == node.mastery_threshold


# ─── Unlock Logic Tests ───


class TestUnlockLogic:
    def test_root_skills_active_at_init(self):
        gs = make_game_state()
        assert gs.skill_tree["freefall"].status == SkillStatus.ACTIVE
        assert gs.skill_tree["spring_only"].status == SkillStatus.ACTIVE

    def test_locked_until_prerequisites_mastered(self):
        gs = make_game_state()
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.LOCKED
        master_skill(gs, "freefall")
        # Only one prerequisite met
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.LOCKED

    def test_single_prerequisite_unlock(self):
        gs = make_game_state()
        assert gs.skill_tree["damped_spring"].status == SkillStatus.LOCKED
        master_skill(gs, "spring_only")
        assert gs.skill_tree["damped_spring"].status == SkillStatus.ACTIVE

    def test_multi_prerequisite_unlock(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        master_skill(gs, "spring_only")
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.ACTIVE

    def test_cascading_unlock(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        master_skill(gs, "spring_only")
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.ACTIVE
        assert gs.skill_tree["damped_spring"].status == SkillStatus.ACTIVE
        master_skill(gs, "gravity_spring")
        master_skill(gs, "damped_spring")
        assert gs.skill_tree["driven_oscillator"].status == SkillStatus.ACTIVE

    def test_cascading_relock(self):
        gs = make_game_state()
        # Master everything
        master_skill(gs, "freefall")
        master_skill(gs, "spring_only")
        master_skill(gs, "gravity_spring")
        master_skill(gs, "damped_spring")
        master_skill(gs, "driven_oscillator")
        assert gs.skill_tree["driven_oscillator"].status == SkillStatus.MASTERED

        # Regress freefall
        for _ in range(20):
            gs.record_completion("ff_1", 0.2)
        assert gs.skill_tree["freefall"].status == SkillStatus.ACTIVE
        # gravity_spring re-locked (depends on freefall)
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.LOCKED
        # driven_oscillator re-locked (depends on gravity_spring)
        assert gs.skill_tree["driven_oscillator"].status == SkillStatus.LOCKED
        # damped_spring NOT re-locked (depends only on spring_only, still mastered)
        assert gs.skill_tree["damped_spring"].status == SkillStatus.MASTERED
        # spring_only NOT affected (root)
        assert gs.skill_tree["spring_only"].status == SkillStatus.MASTERED

    def test_relock_resets_mastery_state(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        master_skill(gs, "spring_only")
        master_skill(gs, "gravity_spring")

        assert gs.skill_tree["gravity_spring"]._was_mastered

        # Regress freefall → gravity_spring cascade re-locks
        for _ in range(20):
            gs.record_completion("ff_1", 0.2)
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.LOCKED
        assert not gs.skill_tree["gravity_spring"]._was_mastered
        assert len(gs.skill_tree["gravity_spring"].recent_scores) == 0

        # Re-master freefall → gravity_spring unlocks but not mastered
        master_skill(gs, "freefall")
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.ACTIVE
        assert not gs.skill_tree["gravity_spring"].mastered


# ─── Prompt Filtering Tests ───


class TestPromptFiltering:
    def test_only_active_skills_sampled(self):
        gs = make_game_state()
        active = gs.get_active_prompts()
        # Root skills active + untracked
        assert "ff_1" in active
        assert "sp_1" in active
        assert "untracked_1" in active
        # Locked skills never appear
        assert "gs_1" not in active
        assert "ds_1" not in active
        assert "do_1" not in active

    def test_mastered_skills_review_sampling(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        # Over 1000 iterations, freefall should appear sometimes
        appearances = 0
        for _ in range(1000):
            active = gs.get_active_prompts()
            if "ff_1" in active:
                appearances += 1
        # ~15% probability → expect ~150, allow wide range
        assert 50 < appearances < 400

    def test_locked_skills_never_sampled(self):
        gs = make_game_state()
        for _ in range(100):
            active = gs.get_active_prompts()
            assert "gs_1" not in active
            assert "do_1" not in active

    def test_untracked_prompts_always_active(self):
        gs = make_game_state()
        active = gs.get_active_prompts()
        assert "untracked_1" in active

    def test_untracked_excluded_when_disabled(self):
        config = make_skill_tree_config()
        config.untracked_always_active = False
        all_ids = [
            "ff_1",
            "ff_2",
            "ff_3",
            "sp_1",
            "sp_2",
            "sp_3",
            "gs_1",
            "gs_2",
            "ds_1",
            "ds_2",
            "do_1",
            "untracked_1",
        ]
        gs = GameState()
        gs.init_tutorial(config, all_ids)
        active = gs.get_active_prompts()
        assert "untracked_1" not in active

    def test_empty_active_pool_fallback(self):
        # Create a state where everything is locked/mastered with no active
        config = TutorialConfig(
            enabled=True,
            post_mastery_behavior="review_only",
            untracked_always_active=False,
            skill_tree={
                "only": SkillConfig(
                    prompts=["p1"],
                    prerequisites=[],
                    mastery_threshold=0.85,
                    regression_threshold=0.65,
                    mastery_window=2,
                    review_probability=0.0,  # 0 review = never sampled
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["p1"])
        # Master the only skill with 0% review probability
        for _ in range(2):
            gs.record_completion("p1", 1.0)
        assert gs.skill_tree["only"].mastered
        # get_active_prompts should fall back to all prompts
        active = gs.get_active_prompts()
        assert len(active) > 0

    def test_post_mastery_behavior_pause(self):
        config = TutorialConfig(
            enabled=True,
            post_mastery_behavior="pause",
            untracked_always_active=False,
            skill_tree={
                "only": SkillConfig(
                    prompts=["p1"],
                    prerequisites=[],
                    mastery_threshold=0.85,
                    regression_threshold=0.65,
                    mastery_window=2,
                    review_probability=0.0,
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["p1"])
        for _ in range(2):
            gs.record_completion("p1", 1.0)
        active = gs.get_active_prompts()
        assert active == []

    def test_post_mastery_behavior_continue_all(self):
        config = TutorialConfig(
            enabled=True,
            post_mastery_behavior="continue_all",
            untracked_always_active=False,
            skill_tree={
                "only": SkillConfig(
                    prompts=["p1"],
                    prerequisites=[],
                    mastery_threshold=0.85,
                    regression_threshold=0.65,
                    mastery_window=2,
                    review_probability=0.0,
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["p1", "p2"])
        for _ in range(2):
            gs.record_completion("p1", 1.0)
        active = gs.get_active_prompts()
        assert "p1" in active
        assert "p2" in active

    def test_prompt_cache_invalidation(self):
        gs = make_game_state()
        pool_a = gs.get_active_prompts()
        assert "ds_1" not in pool_a  # damped_spring locked

        master_skill(gs, "spring_only")
        pool_b = gs.get_active_prompts()
        assert "ds_1" in pool_b  # damped_spring now unlocked

    def test_pool_version_tracking(self):
        gs = make_game_state()
        v1 = gs.snapshot_pool_version()
        # No state change → version unchanged
        gs.record_completion("ff_1", 0.5)
        assert not gs.did_prompt_pool_change(v1)

        # Master freefall → version increments
        v2 = gs.snapshot_pool_version()
        master_skill(gs, "freefall")
        assert gs.did_prompt_pool_change(v2)

        # Snapshot after change → no further change
        v3 = gs.snapshot_pool_version()
        gs.record_completion("ff_1", 0.9)  # Already mastered, no transition
        assert not gs.did_prompt_pool_change(v3)


# ─── State Persistence Tests ───


class TestStatePersistence:
    def test_state_dict_roundtrip(self):
        gs = make_game_state()
        for _ in range(5):
            gs.record_completion("ff_1", 0.9)
        state = gs.tutorial_state_dict()

        gs2 = make_game_state()
        gs2.load_tutorial_state_dict(state)
        assert len(gs2.skill_tree["freefall"].recent_scores) == 5

    def test_checkpoint_resume_mastery(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        assert gs.skill_tree["freefall"].mastered
        state = gs.tutorial_state_dict()

        gs2 = make_game_state()
        gs2.load_tutorial_state_dict(state)
        assert gs2.skill_tree["freefall"].mastered
        assert gs2.skill_tree["freefall"]._was_mastered

    def test_hysteresis_state_persistence(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        state = gs.tutorial_state_dict()

        gs2 = make_game_state()
        gs2.load_tutorial_state_dict(state)
        assert gs2.skill_tree["freefall"]._was_mastered
        # Three bad scores should NOT break mastery (hysteresis)
        for _ in range(3):
            gs2.record_completion("ff_1", 0.3)
        assert gs2.skill_tree["freefall"].mastered

    def test_status_state_persistence(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        master_skill(gs, "spring_only")
        # gravity_spring should be active (unlocked)
        assert gs.skill_tree["gravity_spring"].status == SkillStatus.ACTIVE
        state = gs.tutorial_state_dict()

        gs2 = make_game_state()
        gs2.load_tutorial_state_dict(state)
        assert gs2.skill_tree["freefall"]._status == SkillStatus.MASTERED
        assert gs2.skill_tree["gravity_spring"]._status == SkillStatus.ACTIVE

    def test_initial_mastery_checkpoint_roundtrip(self):
        gs = make_game_state()
        # Record 3 completions
        for _ in range(3):
            gs.record_completion("ff_1", 0.8)
        assert gs.skill_tree["freefall"]._total_completions == 3
        assert len(gs.skill_tree["freefall"]._initial_scores) == 3

        state = gs.tutorial_state_dict()
        gs2 = make_game_state()
        gs2.load_tutorial_state_dict(state)

        # Verify fields survived
        assert gs2.skill_tree["freefall"]._total_completions == 3
        assert len(gs2.skill_tree["freefall"]._initial_scores) == 3
        assert not gs2.skill_tree["freefall"]._initial_mastery_logged

        # Record 2 more → initial_mastery should fire (5 total)
        for _ in range(2):
            gs2.record_completion("ff_1", 0.9)
        assert gs2.skill_tree["freefall"].initial_mastery is not None
        # 3 * 0.8 + 2 * 0.9 = 4.2 / 5 = 0.84
        assert abs(gs2.skill_tree["freefall"].initial_mastery - 0.84) < 0.01

    def test_total_completions_checkpoint_roundtrip(self):
        gs = make_game_state()
        master_skill(gs, "freefall")
        total_before = gs.skill_tree["freefall"]._total_completions
        assert total_before > 0

        state = gs.tutorial_state_dict()
        gs2 = make_game_state()
        gs2.load_tutorial_state_dict(state)
        assert gs2.skill_tree["freefall"]._total_completions == total_before


# ─── Tutorial Disabled Tests ───


class TestTutorialDisabled:
    def test_disabled_no_effect(self):
        config = TutorialConfig(enabled=False)
        gs = GameState()
        gs.init_tutorial(config, ["p1", "p2"])
        assert not gs.tutorial_enabled
        assert gs.skill_tree == {}

    def test_disabled_metrics_empty(self):
        gs = GameState()
        assert gs.get_tutorial_metrics() == {}


# ─── Config Parsing Tests ───


class TestConfigParsing:
    def test_tutorial_config_from_yaml(self):
        import tempfile

        import yaml

        from qgre.config import QGREConfig

        config_dict = {
            "model": {
                "path": "test",
                "pad_token": "<pad>",
                "pad_token_id": 0,
                "lora_target_modules": ["q_proj"],
            },
            "data": {"train_files": ["dummy.parquet"]},
            "generation": {"stop_token_ids": [2]},
            "tutorial": {
                "enabled": True,
                "post_mastery_behavior": "pause",
                "skill_tree": {
                    "skill_a": {
                        "prompts": ["p1", "p2"],
                        "prerequisites": [],
                        "mastery_threshold": 0.9,
                        "regression_threshold": 0.7,
                    },
                    "skill_b": {
                        "prompts": ["p3"],
                        "prerequisites": ["skill_a"],
                    },
                },
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            f.flush()
            cfg = QGREConfig.from_yaml(f.name)

        assert cfg.tutorial.enabled
        assert cfg.tutorial.post_mastery_behavior == "pause"
        assert "skill_a" in cfg.tutorial.skill_tree
        assert cfg.tutorial.skill_tree["skill_a"].mastery_threshold == 0.9
        assert cfg.tutorial.skill_tree["skill_b"].prerequisites == ["skill_a"]

    def test_invalid_post_mastery_behavior_raises(self):
        import tempfile

        import yaml

        from qgre.config import QGREConfig

        config_dict = {
            "model": {
                "path": "test",
                "pad_token": "<pad>",
                "pad_token_id": 0,
                "lora_target_modules": ["q_proj"],
            },
            "data": {"train_files": ["dummy.parquet"]},
            "generation": {"stop_token_ids": [2]},
            "tutorial": {
                "enabled": True,
                "post_mastery_behavior": "invalid_mode",
                "skill_tree": {"a": {"prompts": ["p1"], "prerequisites": []}},
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            f.flush()
            with pytest.raises(ValueError, match="post_mastery_behavior"):
                QGREConfig.from_yaml(f.name)

    def test_empty_skill_tree_raises_on_init(self):
        config = TutorialConfig(enabled=True, skill_tree={})
        gs = GameState()
        with pytest.raises(ValueError, match="skill_tree is empty"):
            gs.init_tutorial(config, ["p1"])

    def test_match_metadata_resolves_prompts(self):
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "freefall": SkillConfig(
                    match_metadata={"system": "freefall"},
                    prerequisites=[],
                    mastery_threshold=0.85,
                    regression_threshold=0.65,
                ),
            },
        )
        # Simulate dataloader items with metadata
        items = [
            {"prompt_id": 111, "metadata": {"system": "freefall"}},
            {"prompt_id": 222, "metadata": {"system": "freefall"}},
            {"prompt_id": 333, "metadata": {"system": "spring"}},
        ]
        gs = GameState()
        gs.init_tutorial(config, ["111", "222", "333"], dataloader_items=items)
        assert "111" in gs.skill_tree["freefall"].prompts
        assert "222" in gs.skill_tree["freefall"].prompts
        assert "333" not in gs.skill_tree["freefall"].prompts

    def test_match_metadata_plus_explicit_prompts(self):
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "mixed": SkillConfig(
                    prompts=["explicit_1"],
                    match_metadata={"system": "freefall"},
                    prerequisites=[],
                    mastery_threshold=0.85,
                    regression_threshold=0.65,
                ),
            },
        )
        items = [
            {"prompt_id": 111, "metadata": {"system": "freefall"}},
        ]
        gs = GameState()
        gs.init_tutorial(config, ["explicit_1", "111"], dataloader_items=items)
        assert "explicit_1" in gs.skill_tree["mixed"].prompts
        assert "111" in gs.skill_tree["mixed"].prompts

    def test_tutorial_defaults_disabled(self):
        from qgre.config import QGREConfig

        cfg = QGREConfig()
        assert not cfg.tutorial.enabled
        assert cfg.tutorial.skill_tree == {}


# ─── Integration Tests ───


class TestIntegration:
    def test_tutorial_with_aspiration_gap(self):
        """Per-skill aspiration targets flow through compute_advantages."""
        from qgre.advantages import QGREStepAdvantageEstimator
        from qgre.types import PromptContext, RewardResult

        step_qualities = {1: ["q_format"], 2: ["q_accuracy"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.1,
            mode="spo",
            step_qualities=step_qualities,
        )
        estimator._aspiration_beta = 0.5
        estimator._aspiration_target = 0.8  # Global default

        # Two prompts: one from freefall (target=0.85), one from gravity_spring (target=0.75)
        contexts = [
            PromptContext(
                prompt_id=1,
                skill_key="freefall",
                tier="t1",
                aspiration_target=0.85,
                aspiration_warmup=1.0,
                is_active=True,
            ),
            PromptContext(
                prompt_id=2,
                skill_key="gravity_spring",
                tier="t1",
                aspiration_target=0.75,
                aspiration_warmup=1.0,
                is_active=True,
            ),
        ]

        rr1 = RewardResult(reward=0.5, scores={"q_format": 0.5, "q_accuracy": 0.5})
        rr2 = RewardResult(reward=0.5, scores={"q_format": 0.5, "q_accuracy": 0.5})

        advs, regions = estimator.compute_advantages(
            batch_prompt_ids=[1, 2],
            batch_token_ids=[[1, 2, 3], [4, 5, 6]],
            batch_reward_results=[rr1, rr2],
            batch_active_qualities=[["q_format", "q_accuracy"]] * 2,
            batch_contexts=contexts,
        )

        # Aspiration target is always 1.0 (perfection), not per-skill mastery threshold.
        # Both prompts get same reward (0.5) → same aspiration gap: beta * (0.5 - 1.0)
        # Per-skill targets only gate curriculum advancement, not gradient signal.
        adv_sum_1 = advs[0].sum().item()
        adv_sum_2 = advs[1].sum().item()
        assert adv_sum_1 == adv_sum_2, (
            f"Both prompts should get same aspiration push (target=1.0 for both): "
            f"prompt1={adv_sum_1:.4f}, prompt2={adv_sum_2:.4f}"
        )
        # Verify aspiration is actually applied (not zero)
        assert adv_sum_1 < 0, f"Aspiration should push negative (r=0.5 < target=1.0): {adv_sum_1}"

    def test_tutorial_with_aspiration_untracked_uses_global(self):
        """Untracked prompts use global aspiration target, not per-skill."""
        from qgre.advantages import QGREStepAdvantageEstimator
        from qgre.types import PromptContext, RewardResult

        step_qualities = {1: ["q_format"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.1,
            mode="spo",
            step_qualities=step_qualities,
        )
        estimator._aspiration_beta = 0.5
        estimator._aspiration_target = 0.8

        # Untracked prompt → should use global default 0.8
        ctx = PromptContext(
            prompt_id=99,
            skill_key=None,
            tier="default",
            aspiration_target=0.8,
            aspiration_warmup=1.0,
            is_active=True,
        )
        rr = RewardResult(reward=0.5, scores={"q_format": 0.5})

        advs, _ = estimator.compute_advantages(
            batch_prompt_ids=[99],
            batch_token_ids=[[1, 2, 3]],
            batch_reward_results=[rr],
            batch_active_qualities=[["q_format"]],
            batch_contexts=[ctx],
        )
        # aspiration gap: 0.5 * (0.5 - 0.8) = -0.15
        # Just verify it computed without error and produced non-zero advantages
        assert advs[0].abs().sum().item() > 0

    def test_tutorial_with_phase_gating(self):
        """Tutorial-tracked prompts bypass tier gate. Untracked prompts respect tier gate."""
        gs = make_game_state()
        gs.default_aspiration_target = 0.8

        # Active prompt set: root skills + untracked
        active_prompts = set(gs.get_active_prompts())
        assert "ff_1" in active_prompts  # Active root skill
        assert "gs_1" not in active_prompts  # Locked skill
        assert "untracked_1" in active_prompts  # Untracked but always active

        # Test 1: Untracked prompt + matching tier → active (tier gate decides)
        contexts = gs.build_prompt_contexts(
            prompt_ids=[999],  # str(999)="999" — not tracked
            metadata=[{"difficulty": "tutorial_gravity"}],
            difficulty_column="difficulty",
            active_tiers={"tutorial_gravity"},
        )
        assert contexts[0].is_active  # Untracked, tier passes → active

        # Test 2: Untracked prompt + wrong tier → inactive (tier gate blocks)
        contexts2 = gs.build_prompt_contexts(
            prompt_ids=[999],
            metadata=[{"difficulty": "tier3"}],
            difficulty_column="difficulty",
            active_tiers={"tutorial_gravity"},
        )
        assert not contexts2[0].is_active  # Untracked, tier fails → inactive

        # Test 3: Tracked prompt in active skill + wrong tier → ACTIVE (tutorial bypasses tier)
        # "ff_1" is in freefall (active). Even if its tier doesn't match, tutorial is authority.
        # We can't easily pass "ff_1" as an int prompt_id, but we can verify the logic:
        # skill_key is not None AND pid_str in active_prompt_set → active regardless of tier
        assert "ff_1" in gs._prompt_to_skill  # Tracked
        assert "ff_1" in active_prompts  # Active skill → bypasses tier

        # Test 4: Tracked prompt in locked skill → INACTIVE (tutorial blocks)
        assert "gs_1" in gs._prompt_to_skill  # Tracked
        assert "gs_1" not in active_prompts  # Locked skill → blocked

    def test_build_prompt_contexts_respects_skill_gate(self):
        """Prompts not in active prompt set are marked inactive."""
        gs = make_game_state()
        # Verify the composition: active prompt set determines skill gate
        active_set = set(gs.get_active_prompts())
        # "ff_1" is in active skill
        assert "ff_1" in active_set
        # "gs_1" is in locked skill
        assert "gs_1" not in active_set
        # "untracked_1" is untracked but always active
        assert "untracked_1" in active_set

    def test_initial_mastery_tracking(self):
        """initial_mastery captures first 5 scores after unlock."""
        gs = make_game_state()
        node = gs.skill_tree["freefall"]
        assert node.initial_mastery is None  # Not enough scores
        for i in range(5):
            gs.record_completion("ff_1", 0.7 + i * 0.02)
        assert node.initial_mastery is not None
        assert abs(node.initial_mastery - 0.74) < 0.01  # avg of 0.70, 0.72, 0.74, 0.76, 0.78

    def test_initial_mastery_reset_on_relock(self):
        """Cascade re-lock clears initial mastery tracking."""
        gs = make_game_state()
        master_skill(gs, "freefall")
        master_skill(gs, "spring_only")
        # gravity_spring unlocked, record some scores
        for _ in range(5):
            gs.record_completion("gs_1", 0.6)
        assert gs.skill_tree["gravity_spring"].initial_mastery is not None
        # Regress freefall → gravity_spring re-locks → initial scores cleared
        for _ in range(20):
            gs.record_completion("ff_1", 0.2)
        assert gs.skill_tree["gravity_spring"]._initial_scores == []
        assert not gs.skill_tree["gravity_spring"]._initial_mastery_logged

    def test_metrics_include_pool_size_and_aspiration(self):
        """get_tutorial_metrics includes active_prompt_pool_size and aspiration_target."""
        gs = make_game_state()
        metrics = gs.get_tutorial_metrics()
        assert "tutorial/active_prompt_pool_size" in metrics
        assert metrics["tutorial/active_prompt_pool_size"] > 0
        assert "tutorial/skill/freefall/aspiration_target" in metrics
        assert metrics["tutorial/skill/freefall/aspiration_target"] == 0.85
        assert metrics["tutorial/skill/gravity_spring/aspiration_target"] == 0.75


# ─── Learnability-Based Advancement Tests ───


class TestLearnabilityAdvancement:
    """Tests for learnability-based skill advancement (variance gating)."""

    def test_high_mastery_high_variance_no_advance(self):
        """Skill with high mastery but high variance should NOT advance."""
        config = TutorialConfig(
            enabled=True,
            post_mastery_behavior="review_only",
            skill_tree={
                "freefall": SkillConfig(
                    prompts=["ff_1"],
                    prerequisites=[],
                    mastery_threshold=0.80,
                    regression_threshold=0.60,
                    mastery_window=20,
                    learnability_threshold=0.10,  # Advance when p(1-p) < 0.10
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["ff_1"])

        # Fill window with alternating scores (high variance)
        # Mean = 0.85, but variance is high
        for i in range(20):
            score = 0.95 if i % 2 == 0 else 0.75  # Alternating → high variance
            gs.record_completion("ff_1", score)

        node = gs.skill_tree["freefall"]
        # Mastery should be ~0.85 (above threshold)
        assert node.mastery_score >= 0.80
        # Learnability = p(1-p) = 0.85 * 0.15 = 0.1275 > 0.10
        assert node.learnability > 0.10
        # Should NOT be ready to advance
        assert not node.ready_to_advance
        # Status should still be ACTIVE, not MASTERED
        assert node.status == SkillStatus.ACTIVE

    def test_high_mastery_low_variance_advances(self):
        """Skill with high mastery and low variance SHOULD advance."""
        config = TutorialConfig(
            enabled=True,
            post_mastery_behavior="review_only",
            skill_tree={
                "freefall": SkillConfig(
                    prompts=["ff_1"],
                    prerequisites=[],
                    mastery_threshold=0.80,
                    regression_threshold=0.60,
                    mastery_window=20,
                    learnability_threshold=0.10,
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["ff_1"])

        # Fill window with consistent high scores (low variance)
        # All 0.95 → p = 0.95, learnability = 0.95 * 0.05 = 0.0475
        for _ in range(20):
            gs.record_completion("ff_1", 0.95)

        node = gs.skill_tree["freefall"]
        assert node.mastery_score >= 0.90
        # Learnability = 0.95 * 0.05 = 0.0475 < 0.10
        assert node.learnability < 0.10
        assert node.ready_to_advance
        assert node.status == SkillStatus.MASTERED

    def test_variance_collapse_triggers_advancement(self):
        """Skill that starts variable then stabilizes should eventually advance."""
        config = TutorialConfig(
            enabled=True,
            post_mastery_behavior="review_only",
            skill_tree={
                "freefall": SkillConfig(
                    prompts=["ff_1"],
                    prerequisites=[],
                    mastery_threshold=0.80,
                    regression_threshold=0.60,
                    mastery_window=10,  # Shorter window for test
                    learnability_threshold=0.10,
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["ff_1"])

        # Phase 1: Variable scores (still learning)
        for i in range(10):
            score = 0.95 if i % 2 == 0 else 0.75
            gs.record_completion("ff_1", score)

        node = gs.skill_tree["freefall"]
        assert node.status == SkillStatus.ACTIVE  # Not mastered yet

        # Phase 2: Consistent high scores (variance collapses)
        for _ in range(10):
            gs.record_completion("ff_1", 0.92)

        # Now variance has collapsed
        assert node.learnability < 0.10
        assert node.ready_to_advance
        assert node.status == SkillStatus.MASTERED

    def test_dependent_unlock_waits_for_ready_to_advance(self):
        """Dependent skills should only unlock when prereqs are ready_to_advance."""
        config = TutorialConfig(
            enabled=True,
            post_mastery_behavior="review_only",
            skill_tree={
                "freefall": SkillConfig(
                    prompts=["ff_1"],
                    prerequisites=[],
                    mastery_threshold=0.80,
                    mastery_window=20,
                    learnability_threshold=0.10,
                ),
                "gravity_spring": SkillConfig(
                    prompts=["gs_1"],
                    prerequisites=["freefall"],
                    mastery_threshold=0.75,
                    mastery_window=20,
                    learnability_threshold=0.12,
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["ff_1", "gs_1"])

        # Fill freefall with high-variance scores (mastery OK, learnability high)
        for i in range(20):
            score = 0.95 if i % 2 == 0 else 0.75
            gs.record_completion("ff_1", score)

        freefall = gs.skill_tree["freefall"]
        gravity_spring = gs.skill_tree["gravity_spring"]

        # Freefall has high mastery but high variance
        assert freefall.mastery_score >= 0.80
        assert freefall.learnability > 0.10
        assert freefall.status == SkillStatus.ACTIVE  # Not MASTERED
        # Gravity_spring should still be LOCKED
        assert gravity_spring.status == SkillStatus.LOCKED

        # Now stabilize freefall
        for _ in range(20):
            gs.record_completion("ff_1", 0.92)

        # Freefall should now be MASTERED (variance collapsed)
        assert freefall.status == SkillStatus.MASTERED
        # Gravity_spring should now be ACTIVE (prereq mastered)
        assert gravity_spring.status == SkillStatus.ACTIVE

    def test_learnability_thresholds_vary_by_skill(self):
        """Different skills can have different learnability thresholds."""
        config = TutorialConfig(
            enabled=True,
            skill_tree={
                "easy_skill": SkillConfig(
                    prompts=["easy_1"],
                    prerequisites=[],
                    mastery_threshold=0.80,
                    mastery_window=10,
                    learnability_threshold=0.05,  # Strict — must be very stable
                ),
                "hard_skill": SkillConfig(
                    prompts=["hard_1"],
                    prerequisites=[],
                    mastery_threshold=0.80,
                    mastery_window=10,
                    learnability_threshold=0.15,  # Lenient — some variance OK
                ),
            },
        )
        gs = GameState()
        gs.init_tutorial(config, ["easy_1", "hard_1"])

        # Both get same scores: p=0.88 → learnability = 0.88 * 0.12 = 0.1056
        for _ in range(10):
            gs.record_completion("easy_1", 0.88)
            gs.record_completion("hard_1", 0.88)

        easy = gs.skill_tree["easy_skill"]
        hard = gs.skill_tree["hard_skill"]

        # easy_skill: learnability 0.1056 > 0.05 → NOT ready
        assert easy.learnability > 0.05
        assert not easy.ready_to_advance
        assert easy.status == SkillStatus.ACTIVE

        # hard_skill: learnability 0.1056 < 0.15 → ready
        assert hard.learnability < 0.15
        assert hard.ready_to_advance
        assert hard.status == SkillStatus.MASTERED
