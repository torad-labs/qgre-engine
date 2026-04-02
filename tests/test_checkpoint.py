"""Tests for GameState serialization (Step 0a) and checkpoint resume (Step 0f)."""

import json
import tempfile
from collections import defaultdict, deque
from pathlib import Path

import pytest

from qgre.checkpoint import (
    discover_latest_checkpoint,
    gamestate_from_dict,
    gamestate_to_dict,
    load_checkpoint,
    save_checkpoint,
)
from qgre.types import QUALITY_WINDOW_SIZE, GameState


# --- Step 0a: GameState Serializer ---


def test_gamestate_roundtrip(mock_game_state):
    """to_dict() → from_dict() → all fields equal."""
    d = gamestate_to_dict(mock_game_state)
    restored = gamestate_from_dict(d)

    assert restored.phase == mock_game_state.phase
    assert restored.step_count == mock_game_state.step_count
    assert restored.mastery_threshold == mock_game_state.mastery_threshold
    assert restored.phase_history == mock_game_state.phase_history

    # Step mastery values match
    for step_num in mock_game_state.step_mastery:
        assert step_num in restored.step_mastery
        assert list(restored.step_mastery[step_num]) == list(mock_game_state.step_mastery[step_num])


def test_gamestate_json_serializable(mock_game_state):
    """to_dict() output passes json.dumps without error."""
    d = gamestate_to_dict(mock_game_state)
    result = json.dumps(d)
    assert isinstance(result, str)
    assert len(result) > 10

    parsed = json.loads(result)
    restored = gamestate_from_dict(parsed)
    assert restored.phase == mock_game_state.phase


def test_gamestate_preserves_deque_maxlen(mock_game_state):
    """After round-trip, step_mastery deques preserve maxlen."""
    d = gamestate_to_dict(mock_game_state)
    restored = gamestate_from_dict(d)

    for step_num, dq in restored.step_mastery.items():
        assert isinstance(dq, deque)
        assert dq.maxlen == QUALITY_WINDOW_SIZE


def test_gamestate_empty_roundtrip():
    """Empty GameState survives round-trip."""
    gs = GameState()
    d = gamestate_to_dict(gs)
    restored = gamestate_from_dict(d)

    assert restored.phase == 1
    assert restored.step_count == 0
    assert restored.step_mastery == {}
    assert restored.phase_history == []
    assert restored.mastery_threshold == 0.8


def test_gamestate_stagnation_fields_roundtrip():
    """Stagnation detection fields survive round-trip and default correctly from old checkpoints."""
    gs = GameState(stagnation_timeout=150, plateau_window=30, plateau_threshold=0.05)
    gs.tier_steps_at_phase_start = {"default": 42}
    d = gamestate_to_dict(gs)
    restored = gamestate_from_dict(d)

    assert restored.stagnation_timeout == 150
    assert restored.plateau_window == 30
    assert restored.plateau_threshold == 0.05
    assert restored.tier_steps_at_phase_start == {"default": 42}

    # Old checkpoint without stagnation fields should use defaults
    old = gamestate_from_dict({"phase": 2, "step_count": 100})
    assert old.stagnation_timeout == 200
    assert old.plateau_window == 50
    assert old.plateau_threshold == 0.02
    # DP-R3-01: Missing tier_steps_at_phase_start entries are initialized from step_count
    assert old.tier_steps_at_phase_start == {"default": 100}


def test_gamestate_phase_advance():
    """Phase advances when step mastery exceeds threshold."""
    gs = GameState(mastery_threshold=0.8)

    # Step 1 not mastered yet
    for _ in range(20):
        gs.record_step_score(1, 0.7)
    assert not gs.check_phase_advance(max_phase=4)
    assert gs.phase == 1

    # Step 1 mastered
    for _ in range(20):
        gs.record_step_score(1, 0.9)
    assert gs.check_phase_advance(max_phase=4)
    assert gs.phase == 2

    # Step 2 mastered
    for _ in range(20):
        gs.record_step_score(2, 0.85)
    assert gs.check_phase_advance(max_phase=4)
    assert gs.phase == 3


def test_gamestate_no_advance_past_max():
    """Phase cannot advance past max_phase."""
    gs = GameState(mastery_threshold=0.5)
    gs.tier_phases = {"default": 4}
    gs.record_step_score(4, 1.0)
    assert not gs.check_phase_advance(max_phase=4)
    assert gs.phase == 4


# --- Step 0f: Checkpoint Resume ---


def test_checkpoint_save_load_roundtrip(mock_game_state):
    """Save full state dict → load → all fields match."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_50.pt"

        rng_state = torch.get_rng_state()
        save_checkpoint(
            path=path,
            global_step=50,
            model_state_dict={"weight": torch.randn(4, 4)},
            optimizer_state_dict={"lr": 5e-6},
            game_state=mock_game_state,
            advantage_estimator_state={"V": {1: {1: 0.5}}},
            rng_state=rng_state,
        )

        loaded = load_checkpoint(path)
        # CheckpointState: access via attributes
        assert loaded.trainer.global_step == 50
        assert isinstance(loaded.game_state, GameState)
        assert loaded.game_state.phase == mock_game_state.phase
        # AdvantageEstimatorState wraps full state_dict — check it was preserved
        assert loaded.advantage_estimator.state_dict is not None
        assert loaded.advantage_estimator.state_dict.get("V", {}).get(1, {}).get(1) == 0.5
        assert torch.equal(loaded.trainer.rng_state, rng_state)


def test_checkpoint_discovery_finds_latest():
    """Create dir with global_step_10, _50, _30 → returns _50."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        for step in [10, 50, 30]:
            path = Path(tmpdir) / f"global_step_{step}.pt"
            torch.save({"global_step": step}, path)

        latest = discover_latest_checkpoint(tmpdir)
        assert latest is not None
        assert "global_step_50" in latest.name


def test_checkpoint_discovery_empty_dir():
    """Empty dir → returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert discover_latest_checkpoint(tmpdir) is None


def test_checkpoint_discovery_nonexistent_dir():
    """Nonexistent dir → returns None."""
    assert discover_latest_checkpoint("/tmp/nonexistent_qgre_test_dir") is None


def test_checkpoint_rng_state_restored():
    """Save RNG → generate random → restore RNG → same random sequence."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_1.pt"
        rng_state = torch.get_rng_state()
        save_checkpoint(path=path, global_step=1, rng_state=rng_state)

        # Generate some random numbers
        seq1 = torch.randn(5)

        # Restore RNG state — CheckpointState has rng_state in trainer
        loaded = load_checkpoint(path)
        torch.set_rng_state(loaded.trainer.rng_state)

        # Generate again — should match
        seq2 = torch.randn(5)
        assert torch.equal(seq1, seq2)


def test_checkpoint_includes_advantage_estimator_state(mock_game_state):
    """V tracker and _step_seen persist through save/load."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_100.pt"

        adv_state = {
            "V": {42: {1: 0.85, 2: 0.6, 3: 0.0, 4: 0.3}},
            "step_seen": {42: [1, 2, 4]},
        }

        save_checkpoint(
            path=path,
            global_step=100,
            game_state=mock_game_state,
            advantage_estimator_state=adv_state,
        )

        loaded = load_checkpoint(path)
        # CheckpointState: advantage_estimator wraps state_dict
        assert loaded.advantage_estimator.state_dict["V"][42][1] == 0.85
        assert loaded.advantage_estimator.state_dict["step_seen"][42] == [1, 2, 4]


def test_gamestate_preserves_defaultdict_behavior():
    """After from_dict(), accessing missing key returns default, not KeyError."""
    gs = GameState()
    gs.record_step_score(1, 0.9)
    d = gamestate_to_dict(gs)
    restored = gamestate_from_dict(d)

    # step_mastery is a plain dict after from_dict, but GameState.record_step_score
    # handles missing keys internally, so accessing via get_step_mastery should work
    assert restored.get_step_mastery(99) == 0.0  # missing step → 0.0, not KeyError

    # record_step_score on a new step should also work without KeyError
    restored.record_step_score(99, 0.5)
    assert restored.get_step_mastery(99) == 0.5


# --- Step 5: Trainer checkpoint wiring ---


def test_trainer_save_load_step_counter(mock_game_state):
    """Train 3 steps → save → new trainer → load → step counter == 3."""
    import torch.nn as nn
    from unittest.mock import MagicMock
    from qgre.config import QGREConfig
    from qgre.trainer import QGRETrainer
    from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return MagicMock(logits=self.linear(torch.randn(1, 4).expand(x.shape[0], -1).unsqueeze(1).expand(-1, x.shape[1], -1)))

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QGREConfig()
        cfg.algorithm.step_qualities = HYPERGRAPH_V1_STEP_QUALITIES
        cfg.logging.checkpoint_dir = tmpdir
        cfg.logging.completion_dir = str(Path(tmpdir) / "completions")

        model = TinyModel()
        trainer = QGRETrainer(
            model=model, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
            game_state=mock_game_state,
        )
        trainer.setup_optimizer()
        trainer.global_step = 3
        trainer.save()

        # New trainer, load checkpoint
        model2 = TinyModel()
        trainer2 = QGRETrainer(
            model=model2, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
        )
        trainer2.setup_optimizer()
        resumed = trainer2.resume(tmpdir)

        assert resumed is True
        assert trainer2.global_step == 3
        assert trainer2.game_state.phase == mock_game_state.phase


def test_v_tracker_persists_across_checkpoint():
    """SPO V tracker values present after save/load cycle."""
    import torch.nn as nn
    from unittest.mock import MagicMock
    from qgre.config import QGREConfig
    from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES
    from qgre.trainer import QGRETrainer

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return MagicMock(logits=self.linear(torch.randn(1, 4)))

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QGREConfig()
        cfg.algorithm.step_qualities = HYPERGRAPH_V1_STEP_QUALITIES
        cfg.logging.checkpoint_dir = tmpdir
        cfg.logging.completion_dir = str(Path(tmpdir) / "completions")

        model = TinyModel()
        trainer = QGRETrainer(
            model=model, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
        )
        trainer.setup_optimizer()

        # Set some V tracker state
        trainer.advantage_estimator.V[42][1] = 0.85
        trainer.advantage_estimator.V[42][2] = 0.6
        trainer.advantage_estimator._step_seen[42] = {1, 2}
        trainer.global_step = 10
        trainer.save()

        # Restore
        model2 = TinyModel()
        trainer2 = QGRETrainer(
            model=model2, tokenizer=None,
            reward_fn=lambda *a: None, config=cfg,
        )
        trainer2.setup_optimizer()
        trainer2.resume(tmpdir)

        assert trainer2.advantage_estimator.V[42][1] == 0.85
        assert trainer2.advantage_estimator.V[42][2] == 0.6
        assert 1 in trainer2.advantage_estimator._step_seen[42]


# --- Old-format checkpoint migration ---


def test_old_format_checkpoint_migration():
    """Old checkpoint format (flat keys) migrates to StateSpec format."""
    import torch
    from qgre.checkpoint import load_checkpoint
    from qgre.types import CheckpointState, TrainerState

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_100.pt"

        # Create old-format checkpoint (no "trainer" key, flat structure)
        old_checkpoint = {
            "global_step": 100,
            "accumulated_loss": 5.5,
            "accumulated_samples": 32,
            "accumulation_count": 4,
            "model_state_dict": {"weight": torch.randn(4, 4)},
            "optimizer_state_dict": {"lr": 1e-5},
            "game_state": {"phase": 2, "step_count": 100},
            "rng_state": torch.get_rng_state(),
            # No schema_version — old format
        }
        torch.save(old_checkpoint, path)

        # Load and verify migration
        loaded = load_checkpoint(path)

        # Should be CheckpointState
        assert isinstance(loaded, CheckpointState)

        # TrainerState should be populated from flat keys
        assert loaded.trainer.global_step == 100
        assert loaded.trainer.accumulated_loss == 5.5
        assert loaded.trainer.accumulated_samples == 32
        assert loaded.trainer.accumulation_count == 4

        # Model state preserved
        assert loaded.model_state_dict is not None
        assert "weight" in loaded.model_state_dict

        # GameState migrated
        assert loaded.game_state.phase == 2


def test_trainer_state_full_roundtrip():
    """All TrainerState fields survive checkpoint round-trip."""
    import torch

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "global_step_50.pt"

        # Save with all TrainerState fields populated
        rng = torch.get_rng_state()
        save_checkpoint(
            path=path,
            global_step=50,
            model_state_dict={"w": torch.randn(2, 2)},
            accumulated_loss=2.5,
            accumulated_samples=16,
            accumulation_count=2,
        )

        # Load checkpoint manually to set additional fields
        ckpt = torch.load(path)
        ckpt["trainer"]["resumed_mid_accumulation"] = True
        ckpt["trainer"]["fused_validated"] = True
        ckpt["trainer"]["needs_weight_sync"] = True
        torch.save(ckpt, path)

        # Load and verify all fields
        loaded = load_checkpoint(path)

        assert loaded.trainer.global_step == 50
        assert loaded.trainer.accumulated_loss == 2.5
        assert loaded.trainer.accumulated_samples == 16
        assert loaded.trainer.accumulation_count == 2
        assert loaded.trainer.resumed_mid_accumulation is True
        assert loaded.trainer.fused_validated is True
        assert loaded.trainer.needs_weight_sync is True
