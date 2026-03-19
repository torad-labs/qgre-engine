"""Tests for QGRETrainer and config (Step 1)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
import pytest

from qgre.config import QGREConfig
from qgre.data import PromptBatch
from qgre.advantages import build_phase_qualities
from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES
from qgre.trainer import QGRETrainer
from qgre.types import RewardResult
from qgre.segments import OPEN_ANGLE, STEP_TOKEN, CLOSE_ANGLE, CLOSE_SLASH


class MockModel(nn.Module):
    """Minimal model that returns random logits."""

    def __init__(self, vocab_size=160000, hidden=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        logits = self.head(x)
        return MagicMock(logits=logits)


def _make_tokens(n=32):
    """Simple token sequence with step 1 structure."""
    return [
        OPEN_ANGLE, STEP_TOKEN, 16, 9999, CLOSE_ANGLE,  # <step1>
        *[100 + i for i in range(n - 10)],               # content
        CLOSE_SLASH, STEP_TOKEN, 16, 9999, CLOSE_ANGLE,  # </step1>
    ]


def _make_batch(n_completions=2, seq_len=32):
    """Create a PromptBatch for testing."""
    return PromptBatch(
        input_ids=torch.randint(0, 100, (n_completions, 16)),
        attention_mask=torch.ones(n_completions, 16, dtype=torch.long),
        prompt_ids=[hash(f"prompt_{i}") & 0x7FFFFFFF for i in range(n_completions)],
        raw_prompts=[f"Test prompt {i}" for i in range(n_completions)],
        metadata=[{} for _ in range(n_completions)],
    )


# --- Config tests ---


def test_config_from_yaml():
    """Load config from YAML file."""
    cfg = QGREConfig.from_yaml("examples/hypergraph/config.yaml")
    assert cfg.algorithm.mode == "spo"
    assert cfg.algorithm.spo.n == 1
    assert cfg.generation.temperature == 1.0
    assert cfg.algorithm.clip_ratio_low == 0.2
    assert cfg.algorithm.clip_ratio_high == 0.28


def test_config_defaults():
    """Default config has sensible values."""
    cfg = QGREConfig()
    assert cfg.algorithm.mode == "spo"
    assert cfg.training.lr == 5e-6
    assert cfg.generation.temperature == 1.0


def test_config_math_example():
    """Load math config."""
    cfg = QGREConfig.from_yaml("examples/math/config.yaml")
    assert cfg.algorithm.mode == "grpo"
    assert cfg.algorithm.grpo.n == 4


# --- Trainer tests ---


def test_trainer_forward_finite_loss():
    """Synthetic batch → loss is finite, non-zero."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QGREConfig()
        cfg.logging.completion_dir = str(Path(tmpdir) / "completions")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "checkpoints")

        model = MockModel()
        trainer = QGRETrainer(
            model=model,
            tokenizer=None,
            reward_fn=lambda *a, **k: RewardResult(reward=0.5, scores={"q_format_tags": 1.0}, phase=1),
            config=cfg,
        )
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        completions = [tokens, tokens]
        reward_results = [
            RewardResult(reward=0.8, scores={"q_format_tags": 1.0, "q_tag_content": 0.9}, phase=1),
            RewardResult(reward=0.3, scores={"q_format_tags": 0.5, "q_tag_content": 0.2}, phase=1),
        ]

        metrics = trainer.step(batch, completions, reward_results)
        assert "loss" in metrics
        assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_response_mask_masks_padding():
    """Response mask: 0 for prompt, 1 for response, 0 after EOS."""
    cfg = QGREConfig()
    model = MockModel()
    trainer = QGRETrainer(model=model, tokenizer=None, reward_fn=lambda *a: None, config=cfg)

    # Sequence: [pad, pad, prompt, prompt, response, response, EOS, pad]
    input_ids = torch.tensor([[0, 0, 10, 11, 20, 21, 151643, 0]])
    mask = trainer.compute_response_mask(input_ids, prompt_lengths=[4], eos_token_id=151643)

    assert mask[0, 0].item() == 0.0  # pad
    assert mask[0, 3].item() == 0.0  # prompt
    assert mask[0, 4].item() == 1.0  # response
    assert mask[0, 5].item() == 1.0  # response
    assert mask[0, 6].item() == 1.0  # EOS (included)
    assert mask[0, 7].item() == 0.0  # after EOS


def test_mode_switch_spo_vs_grpo():
    """Config mode='spo' vs 'grpo' → different estimator mode."""
    cfg_spo = QGREConfig()
    cfg_spo.algorithm.mode = "spo"
    trainer_spo = QGRETrainer(model=MockModel(), tokenizer=None, reward_fn=lambda *a: None, config=cfg_spo)
    assert trainer_spo.advantage_estimator.mode == "spo"

    cfg_grpo = QGREConfig()
    cfg_grpo.algorithm.mode = "grpo"
    trainer_grpo = QGRETrainer(model=MockModel(), tokenizer=None, reward_fn=lambda *a: None, config=cfg_grpo)
    assert trainer_grpo.advantage_estimator.mode == "grpo"


def test_phase_qualities_mapping():
    """build_phase_qualities produces correct progressive gating."""
    pq = build_phase_qualities(HYPERGRAPH_V1_STEP_QUALITIES)
    assert len(pq[1]) == 5   # Step 1 only
    assert len(pq[2]) == 6   # Step 1 + 2
    assert len(pq[3]) == 8   # Step 1 + 2 + 3
    assert len(pq[4]) == 13  # All steps


def test_phase_qualities_5_steps():
    """5-step config produces 5 phases."""
    sq = {1: ["a"], 2: ["b"], 3: ["c"], 4: ["d"], 5: ["e"]}
    pq = build_phase_qualities(sq)
    assert len(pq) == 5
    assert pq[5] == ["a", "b", "c", "d", "e"]


def test_phase_qualities_non_cumulative():
    """Non-cumulative mode: each phase has only its own qualities."""
    sq = {1: ["a", "b"], 2: ["c"], 3: ["d"]}
    pq = build_phase_qualities(sq, cumulative=False)
    assert pq[1] == ["a", "b"]
    assert pq[2] == ["c"]
    assert pq[3] == ["d"]


def test_trainer_accepts_custom_step_qualities():
    """QGRETrainer accepts step_qualities parameter."""
    custom_sq = {1: ["q_json_valid"], 2: ["q_grounding"], 3: ["q_accuracy"]}
    cfg = QGREConfig()
    trainer = QGRETrainer(
        model=MockModel(), tokenizer=None, reward_fn=lambda *a: None,
        config=cfg, step_qualities=custom_sq,
    )
    assert trainer.step_qualities == custom_sq
    assert len(trainer.phase_qualities) == 3


def test_trainer_accepts_custom_segmenter():
    """QGRETrainer accepts segmenter parameter."""
    from qgre.segments import uniform_segmenter

    cfg = QGREConfig()
    trainer = QGRETrainer(
        model=MockModel(), tokenizer=None, reward_fn=lambda *a: None,
        config=cfg, segmenter=uniform_segmenter,
    )
    assert trainer.advantage_estimator.segmenter is uniform_segmenter


# --- Regression tests for bug fixes ---


def test_resume_without_model_state_raises():
    """Resume from checkpoint missing model_state_dict → RuntimeError."""
    import tempfile
    from pathlib import Path
    import torch as _torch

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save a checkpoint WITHOUT model_state_dict
        path = Path(tmpdir) / "global_step_5.pt"
        _torch.save({"global_step": 5, "model_state_dict": None}, path)

        cfg = QGREConfig()
        cfg.logging.checkpoint_dir = tmpdir
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None, reward_fn=lambda *a: None, config=cfg)
        trainer.setup_optimizer()

        with pytest.raises(RuntimeError, match="missing model_state_dict"):
            trainer.resume(tmpdir)


def test_config_unknown_key_warns():
    """Unknown YAML key in config → warning emitted."""
    import warnings
    from qgre.config import QGREConfig

    raw = {
        "model": {"path": "test", "typo_key": "oops"},
        "algorithm": {"mode": "spo"},
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = QGREConfig._from_dict(raw)
        warns = [x for x in w if "Unknown" in str(x.message)]
        assert len(warns) >= 1, "No warning for unknown config key 'typo_key'"
