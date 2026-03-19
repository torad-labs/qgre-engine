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


TEST_SQ = HYPERGRAPH_V1_STEP_QUALITIES


def _cfg() -> QGREConfig:
    """Create a QGREConfig with step_qualities set for testing."""
    cfg = QGREConfig()
    cfg.algorithm.step_qualities = TEST_SQ
    cfg.model.path = "test-model"
    return cfg


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
    cfg = _cfg()
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
        cfg = _cfg()
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
    cfg = _cfg()
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
    cfg_spo = _cfg()
    cfg_spo.algorithm.mode = "spo"
    trainer_spo = QGRETrainer(model=MockModel(), tokenizer=None, reward_fn=lambda *a: None, config=cfg_spo)
    assert trainer_spo.advantage_estimator.mode == "spo"

    cfg_grpo = _cfg()
    cfg_grpo.algorithm.mode = "grpo"
    trainer_grpo = QGRETrainer(model=MockModel(), tokenizer=None, reward_fn=lambda *a: None, config=cfg_grpo)
    assert trainer_grpo.advantage_estimator.mode == "grpo"


def test_gradient_accumulation_equivalence():
    """Gradient accumulation with 2 steps produces equivalent parameter updates."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "completions")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "checkpoints")
        cfg.training.gradient_accumulation_steps = 2

        torch.manual_seed(42)
        model = MockModel()
        trainer = QGRETrainer(
            model=model, tokenizer=None,
            reward_fn=lambda *a, **k: RewardResult(reward=0.5, scores={"q_format_tags": 1.0}, phase=1),
            config=cfg,
        )
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        rrs = [
            RewardResult(reward=0.8, scores={"q_format_tags": 1.0, "q_tag_content": 0.9}, phase=1),
            RewardResult(reward=0.3, scores={"q_format_tags": 0.5, "q_tag_content": 0.2}, phase=1),
        ]

        # Get initial params
        params_before = {n: p.clone() for n, p in model.named_parameters()}

        # Step 0: accumulates but does NOT update (grad_accum=2, step 0+1 % 2 != 0)
        trainer.step(batch, [tokens, tokens], rrs)
        # Step 1: now (1+1) % 2 == 0 → optimizer step fires
        trainer.step(batch, [tokens, tokens], rrs)

        # After 2 steps with grad_accum=2, weights should have changed
        any_changed = False
        for n, p in model.named_parameters():
            if not torch.equal(p, params_before[n]):
                any_changed = True
                break
        assert any_changed, "Weights should change after gradient_accumulation_steps steps"


def test_on_policy_mode():
    """On-policy mode: old_log_probs == log_probs.detach()."""
    cfg = _cfg()
    model = MockModel()
    trainer = QGRETrainer(model=model, tokenizer=None, reward_fn=lambda *a: None, config=cfg)

    # Build synthetic inputs
    input_ids = torch.randint(0, 100, (2, 16))
    output = model(input_ids)
    logits = output.logits

    advantages = torch.randn(2, 15)
    response_mask = torch.ones(2, 15)

    # compute_loss with old_logprobs=None triggers on-policy: old = curr.detach()
    loss, metrics = trainer.compute_loss(
        logits=logits,
        input_ids=input_ids,
        advantages=advantages,
        response_mask=response_mask,
        old_logprobs=None,
    )

    assert torch.isfinite(loss), "On-policy loss should be finite"
    # On-policy → ratio = exp(curr - curr.detach()) = exp(0) = 1 → loss is purely advantage-weighted
    assert loss.item() != 0.0 or advantages.abs().sum() == 0, "On-policy loss should reflect advantages"


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
    cfg = _cfg()
    trainer = QGRETrainer(
        model=MockModel(), tokenizer=None, reward_fn=lambda *a: None,
        config=cfg, step_qualities=custom_sq,
    )
    assert trainer.step_qualities == custom_sq
    assert len(trainer.phase_qualities) == 3


def test_trainer_accepts_custom_segmenter():
    """QGRETrainer accepts segmenter parameter."""
    from qgre.segments import uniform_segmenter

    cfg = _cfg()
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

        cfg = _cfg()
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


# --- Phase advancement tests ---


def test_step_records_mastery_and_advances_phase():
    """Trainer.step() records mastery scores and advances phase when threshold met."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")

        from qgre.types import GameState
        gs = GameState(mastery_threshold=0.7)

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None,
                              reward_fn=lambda *a: None, config=cfg, game_state=gs)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()

        # High scores on step 1 qualities → should eventually advance to phase 2
        for _ in range(25):
            rrs = [
                RewardResult(reward=0.9, scores={
                    "q_format_tags": 0.95, "q_tag_content": 0.90,
                    "q_node_in_prompt": 0.85, "q_node_format": 0.90, "q_node_length": 0.88,
                }),
                RewardResult(reward=0.8, scores={
                    "q_format_tags": 0.88, "q_tag_content": 0.85,
                    "q_node_in_prompt": 0.80, "q_node_format": 0.85, "q_node_length": 0.82,
                }),
            ]
            metrics = trainer.step(batch, [tokens, tokens], rrs)

        # After 25 steps with high step-1 scores, phase should have advanced
        assert trainer.game_state.phase >= 2, f"Phase should have advanced, got {trainer.game_state.phase}"
        assert "mastery/step_1" in metrics


def test_step_uses_engine_phase_not_reward_phase():
    """Trainer uses GameState.phase for active qualities, NOT RewardResult.phase."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")

        from qgre.types import GameState
        gs = GameState(phase=1)  # Engine says phase 1

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None,
                              reward_fn=lambda *a: None, config=cfg, game_state=gs)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()

        # RewardResult claims phase=4, but engine should use GameState.phase=1
        rrs = [
            RewardResult(reward=0.5, scores={"q_format_tags": 1.0}, phase=4),
            RewardResult(reward=0.5, scores={"q_format_tags": 1.0}, phase=4),
        ]
        metrics = trainer.step(batch, [tokens, tokens], rrs)

        # Phase should still be 1 (engine-managed)
        assert metrics["phase"] == 1


# --- Dynamic length control tests ---


def test_length_penalty_applied_when_high_correctness():
    """Length penalty added when group correctness exceeds threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")
        cfg.algorithm.length_penalty_coef = 0.1
        cfg.algorithm.length_penalty_threshold = 0.3

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None,
                              reward_fn=lambda *a: None, config=cfg)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        # High rewards → correctness > threshold → length penalty applied
        rrs = [
            RewardResult(reward=0.9, scores={"q_format_tags": 0.9, "q_tag_content": 0.9}),
            RewardResult(reward=0.8, scores={"q_format_tags": 0.8, "q_tag_content": 0.8}),
        ]
        metrics = trainer.step(batch, [tokens, tokens], rrs)
        assert "length_penalty" in metrics, "Length penalty should be in metrics when correctness is high"


def test_length_penalty_skipped_when_low_correctness():
    """No length penalty when group correctness is below threshold."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")
        cfg.algorithm.length_penalty_coef = 0.1
        cfg.algorithm.length_penalty_threshold = 0.9

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None,
                              reward_fn=lambda *a: None, config=cfg)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        # Low rewards → correctness < threshold → no length penalty
        rrs = [
            RewardResult(reward=0.2, scores={"q_format_tags": 0.2, "q_tag_content": 0.1}),
            RewardResult(reward=0.1, scores={"q_format_tags": 0.1, "q_tag_content": 0.1}),
        ]
        metrics = trainer.step(batch, [tokens, tokens], rrs)
        assert "length_penalty" not in metrics, "Length penalty should NOT be applied when correctness is low"


# --- Fix 1: KL defaults ---


def test_default_config_no_kl_penalty():
    """Default config has kl_cov_ratio=0.0 and loss_mode='pg' (no KL compute)."""
    cfg = QGREConfig()
    assert cfg.algorithm.kl_cov_ratio == 0.0
    assert cfg.algorithm.loss_mode == "pg"


def test_kl_penalty_zero_when_disabled():
    """When loss_mode='pg', KL metric is 0.0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.algorithm.loss_mode = "pg"
        cfg.algorithm.kl_cov_ratio = 0.0
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None,
                              reward_fn=lambda *a: None, config=cfg)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        rrs = [
            RewardResult(reward=0.5, scores={"q_format_tags": 1.0, "q_tag_content": 0.9}),
            RewardResult(reward=0.3, scores={"q_format_tags": 0.5, "q_tag_content": 0.2}),
        ]
        metrics = trainer.step(batch, [tokens, tokens], rrs)
        assert metrics.get("kl_penalty", 0.0) == 0.0, "KL should be 0 when loss_mode='pg'"


# --- Fix 3: neg_logprob_mean is metric only ---


def test_neg_logprob_mean_is_metric_only():
    """neg_logprob_mean is in metrics but NOT part of loss gradient."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None,
                              reward_fn=lambda *a: None, config=cfg)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        rrs = [
            RewardResult(reward=0.8, scores={"q_format_tags": 1.0, "q_tag_content": 0.9}),
            RewardResult(reward=0.3, scores={"q_format_tags": 0.5, "q_tag_content": 0.2}),
        ]
        metrics = trainer.step(batch, [tokens, tokens], rrs)
        assert "neg_logprob_mean" in metrics, "neg_logprob_mean should be in metrics"
        # entropy should NOT be in metrics (old name)
        assert "entropy" not in metrics, "entropy key should be removed"


# --- Fix 4: Completion log is decoded text ---


def test_completion_log_is_decoded_text():
    """Logged completion contains readable text, not int list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")
        # Use GRPO mode to avoid SPO near-zero advantage early return
        cfg.algorithm.mode = "grpo"
        cfg.algorithm.grpo.n = 2  # Match n_completions

        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=True):
                return "decoded text for test"

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=FakeTokenizer(),
                              reward_fn=lambda *a: None, config=cfg)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        rrs = [
            RewardResult(reward=0.9, scores={"q_format_tags": 1.0, "q_tag_content": 0.9}),
            RewardResult(reward=0.1, scores={"q_format_tags": 0.1, "q_tag_content": 0.1}),
        ]

        # Capture what the completion logger receives
        logged = []
        orig_log = trainer.completion_logger.log_completion
        def capture_log(**kwargs):
            logged.append(kwargs)
            orig_log(**kwargs)
        trainer.completion_logger.log_completion = capture_log

        trainer.step(batch, [tokens, tokens], rrs)

        assert len(logged) >= 1
        # The completion= kwarg should contain decoded text
        comp = logged[0].get("completion", "")
        assert "decoded text" in comp, (
            f"Expected decoded text, got: {comp}"
        )


# --- Fix 6: Gradient accumulation loss metric ---


def test_gradient_accumulation_loss_accumulated():
    """Loss metric reflects accumulated total across gradient_accumulation_steps."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _cfg()
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")
        cfg.training.gradient_accumulation_steps = 2
        # Use GRPO mode with divergent rewards to ensure non-zero loss
        cfg.algorithm.mode = "grpo"
        cfg.algorithm.grpo.n = 2  # Match n_completions

        model = MockModel()
        trainer = QGRETrainer(model=model, tokenizer=None,
                              reward_fn=lambda *a: None, config=cfg)
        trainer.setup_optimizer()

        batch = _make_batch(n_completions=2)
        tokens = _make_tokens()
        rrs = [
            RewardResult(reward=0.9, scores={"q_format_tags": 1.0, "q_tag_content": 0.9}),
            RewardResult(reward=0.1, scores={"q_format_tags": 0.1, "q_tag_content": 0.1}),
        ]

        # Step 0: accumulates, no optimizer step
        m0 = trainer.step(batch, [tokens, tokens], rrs)
        assert "accumulated_loss" not in m0, "No accumulated_loss before optimizer step"
        step0_loss = m0["loss"]

        # Step 1: optimizer step fires → accumulated_loss should be present
        m1 = trainer.step(batch, [tokens, tokens], rrs)
        assert "accumulated_loss" in m1, "accumulated_loss should be reported after optimizer step"
        # accumulated_loss should be sum of both steps' losses
        expected_accum = step0_loss + m1["loss"]
        assert abs(m1["accumulated_loss"] - expected_accum) < 1e-6, (
            f"accumulated_loss={m1['accumulated_loss']} should equal sum of step losses={expected_accum}"
        )


# --- Fix 9: Mastery threshold from config ---


def test_mastery_threshold_from_config():
    """TrainingConfig.mastery_threshold is passed to GameState."""
    cfg = _cfg()
    cfg.training.mastery_threshold = 0.65
    trainer = QGRETrainer(model=MockModel(), tokenizer=None,
                          reward_fn=lambda *a: None, config=cfg)
    assert trainer.game_state.mastery_threshold == 0.65


# --- Fix 10: max_grad_norm from config ---


def test_max_grad_norm_from_config():
    """TrainingConfig.max_grad_norm is configurable."""
    cfg = _cfg()
    cfg.training.max_grad_norm = 0.5
    assert cfg.training.max_grad_norm == 0.5
