"""Tests for span-based advantage assignment."""

import pytest
import torch

from qgre.spans import build_char_to_token_map, scored_spans_to_token_masks
from qgre.types import RewardResult, TrainingContext


# --- Mock tokenizer for testing ---


class MockTokenizer:
    """Minimal tokenizer that splits on characters for deterministic testing."""

    def __init__(self, vocab: dict[str, int] | None = None):
        self._vocab = vocab or {}
        self._id_to_str = {v: k for k, v in self._vocab.items()}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self._id_to_str.get(t, "?") for t in token_ids)

    def encode(self, text, add_special_tokens=False):
        return [self._vocab.get(c, 0) for c in text]


@pytest.fixture
def simple_tokenizer():
    """Tokenizer where each character is one token."""
    vocab = {chr(i): i for i in range(32, 127)}  # ASCII printable
    return MockTokenizer(vocab)


@pytest.fixture
def training_ctx():
    """Minimal TrainingContext for testing."""
    return TrainingContext(device=torch.device("cpu"))


# --- build_char_to_token_map tests ---


class TestCharToTokenMap:
    def test_simple_mapping(self, simple_tokenizer):
        text = "H = p"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)
        assert char_map is not None
        assert len(char_map) == 5  # 5 chars
        # Each char maps to its own token
        assert char_map == [0, 1, 2, 3, 4]

    def test_empty_input(self, simple_tokenizer):
        assert build_char_to_token_map([], simple_tokenizer) == []

    def test_returns_none_on_large_length_mismatch(self):
        """If per-token decode is way off from full decode, return None."""

        class BadTokenizer:
            def decode(self, ids, skip_special_tokens=False):
                if len(ids) == 1:
                    return "a"
                return "a" * 20  # 20 chars vs 2 per-token — huge mismatch

        tok = BadTokenizer()
        result = build_char_to_token_map([1, 2], tok)
        assert result is None

    def test_tolerates_small_length_mismatch(self):
        """Small BPE merge boundary mismatches (1-2 chars) are tolerated."""

        class SlightlyOffTokenizer:
            def decode(self, ids, skip_special_tokens=False):
                if len(ids) == 1:
                    return "abc"
                return "abcab"  # 5 chars, per-token would be 6 — off by 1

        tok = SlightlyOffTokenizer()
        result = build_char_to_token_map([1, 2], tok)
        assert result is not None
        assert len(result) == 5  # Truncated to full decode length


# --- scored_spans_to_token_masks tests ---


class TestScoredSpansToTokenMasks:
    def test_single_span(self, simple_tokenizer, training_ctx):
        text = "V = kx + mgx"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)
        assert char_map is not None

        spans = {"q_V_correct": [(0, 6)]}  # "V = kx"
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids), training_ctx)

        assert "q_V_correct" in masks
        assert masks["q_V_correct"].shape == (len(token_ids),)
        # First 6 tokens should be 1.0
        assert masks["q_V_correct"][:6].sum() == 6.0
        assert masks["q_V_correct"][6:].sum() == 0.0

    def test_multiple_spans_same_quality(self, simple_tokenizer, training_ctx):
        text = "H = a, then H = b"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        # Two H expressions — first gets +1.0, second gets REPETITION_MARKER (-1.0)
        # Design: reward original answer, penalize repetitions
        spans = {"q_correct_H": [(0, 5), (12, 18)]}  # "H = a" and " H = b"
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids), training_ctx)

        mask = masks["q_correct_H"]
        assert mask[0:5].sum() == 5.0  # First span: +1.0 per token
        # Second span gets REPETITION_MARKER = -1.0 (penalized repetition)
        # Span (12, 17) after clamping = 5 tokens with -1.0 each
        assert mask[12:17].sum() <= -4.0  # At least 4 penalized tokens
        assert mask[5:12].sum() == 0.0  # Gap between

    def test_full_completion_span(self, simple_tokenizer, training_ctx):
        text = "full text here"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        spans = {"q_format": [(0, len(text))]}
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids), training_ctx)

        # All tokens should be 1.0
        assert masks["q_format"].sum() == len(token_ids)

    def test_empty_spans(self, simple_tokenizer, training_ctx):
        text = "test"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        spans = {"q_correct_H": []}
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids), training_ctx)
        assert masks["q_correct_H"].sum() == 0.0

    def test_overlapping_qualities(self, simple_tokenizer, training_ctx):
        text = "H = T + V"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        spans = {
            "q_correct_H": [(0, 9)],  # Whole expression
            "q_T_uses_p": [(4, 5)],  # Just "T"
        }
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids), training_ctx)

        # Token 4 ("T") should be in BOTH masks
        assert masks["q_correct_H"][4] == 1.0
        assert masks["q_T_uses_p"][4] == 1.0


# --- Integration with advantage estimator ---


class TestSpanBasedAdvantages:
    def test_advantages_reach_all_expression_tokens(self):
        """The core test: advantages should reach expression tokens in derivation."""
        from qgre.advantages import QGREStepAdvantageEstimator

        step_qualities = {
            1: ["q_format"],
            5: ["q_correct_H"],
        }
        estimator = QGREStepAdvantageEstimator(
            lr=0.1,
            mode="spo",
            step_qualities=step_qualities,
        )

        seq_len = 20
        token_masks = {
            "q_format": torch.ones(seq_len),
            "q_correct_H": torch.zeros(seq_len),
        }
        token_masks["q_correct_H"][3:8] = 1.0  # First H in derivation
        token_masks["q_correct_H"][15:19] = 1.0  # Second H in label

        # First call: SPO warm-start sets baseline = batch mean → advantage ~0
        reward1 = RewardResult(reward=0.8, scores={"q_format": 1.0, "q_correct_H": 0.4})
        estimator.compute_advantages_with_spans(
            batch_prompt_ids=[0],
            batch_token_ids=[list(range(seq_len))],
            batch_reward_results=[reward1],
            batch_active_qualities=[["q_format", "q_correct_H"]],
            batch_token_masks=[token_masks],
        )

        # Second call: same prompt, DIFFERENT score → real advantage (score - baseline)
        reward2 = RewardResult(reward=0.9, scores={"q_format": 1.0, "q_correct_H": 0.9})
        advantages, quality_metrics = estimator.compute_advantages_with_spans(
            batch_prompt_ids=[0],
            batch_token_ids=[list(range(seq_len))],
            batch_reward_results=[reward2],
            batch_active_qualities=[["q_format", "q_correct_H"]],
            batch_token_masks=[token_masks],
        )

        advs = advantages[0]
        assert advs.shape == (seq_len,)

        # Tokens 3-7 should have H advantage (derivation H gets signal!)
        assert advs[3].item() != 0.0, "Derivation H tokens should get correct_H signal"
        assert advs[15].item() != 0.0, "Label H tokens should get correct_H signal"

        # H tokens should have MORE signal than non-H tokens (format + H vs format only)
        # Token 10 only has format mask, tokens 3 has format + H
        assert abs(advs[3].item()) > abs(
            advs[10].item()
        ), "H tokens should have stronger signal than non-expression tokens"

    def test_backward_compat_empty_masks(self):
        """When token_masks are empty, all advantages should be zero."""
        from qgre.advantages import QGREStepAdvantageEstimator

        step_qualities = {1: ["q_format"], 5: ["q_correct_H"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.1,
            mode="spo",
            step_qualities=step_qualities,
        )

        reward = RewardResult(reward=0.5, scores={"q_format": 1.0, "q_correct_H": 0.5})
        advantages, _ = estimator.compute_advantages_with_spans(
            batch_prompt_ids=[0],
            batch_token_ids=[list(range(10))],
            batch_reward_results=[reward],
            batch_active_qualities=[["q_format", "q_correct_H"]],
            batch_token_masks=[{}],  # No masks
        )
        # All zeros — no masks means no signal
        assert advantages[0].sum().item() == 0.0


# --- Expression span finder test ---


class TestFindExpressionSpans:
    def test_finds_H_spans(self):
        from examples.hamiltonian.reward_fn import _find_expression_spans

        # _find_expression_spans uses StructuredOutputParser section headers,
        # not raw "H = " patterns. Only "HAMILTONIAN:" sections are matched.
        text = "derivation: H = p²/6 + 3x²\nmore text\nHAMILTONIAN: H = p²/6 + 3x²"
        spans = _find_expression_spans(text)

        assert "q_correct_H" in spans
        assert len(spans["q_correct_H"]) == 1  # Only HAMILTONIAN: section matched

    def test_finds_V_spans(self):
        from examples.hamiltonian.reward_fn import _find_expression_spans

        # "V = ..." matches POTENTIAL alias "v", and "POTENTIAL:" matches directly
        text = "V = kx² + mgx\nPOTENTIAL: V = 3x²"
        spans = _find_expression_spans(text)

        assert len(spans["q_V_correct"]) == 2  # Both lines match POTENTIAL

    def test_format_targets_labeled_sections_not_full_completion(self):
        """Format spans should target labeled sections, OR full completion for negative signal."""
        from examples.hamiltonian.reward_fn import _find_expression_spans

        # Text with no labels - full completion span for negative training signal
        # Without this, garbage output (no labels) would get zero gradient
        text = "some completion text"
        spans = _find_expression_spans(text)
        assert spans["q_format"] == [(0, len(text))]  # Full span for negative signal

        # Text with labels - format should target those sections only
        text_with_labels = "COORDINATES: q = x\nHAMILTONIAN: H = p²"
        spans = _find_expression_spans(text_with_labels)
        assert len(spans["q_format"]) >= 1  # At least one labeled section
        # Should NOT be full completion - only labeled sections get format signal
        assert spans["q_format"] != [(0, len(text_with_labels))]

    def test_equation_spans(self):
        from examples.hamiltonian.reward_fn import _find_expression_spans

        text = "dq/dt = p/3\ndp/dt = -6x"
        spans = _find_expression_spans(text)

        assert len(spans["q_correct_dqdt"]) >= 2

    def test_empty_text(self):
        from examples.hamiltonian.reward_fn import _find_expression_spans

        spans = _find_expression_spans("")
        assert spans["q_format"] == []
        assert spans["q_correct_H"] == []


# --- RewardResult with scored_spans ---


class TestRewardResultSpans:
    def test_default_empty(self):
        rr = RewardResult(reward=0.5)
        assert rr.scored_spans == {}

    def test_with_spans(self):
        rr = RewardResult(
            reward=0.8,
            scores={"q_correct_H": 1.0},
            scored_spans={"q_correct_H": [(10, 25)]},
        )
        assert rr.scored_spans["q_correct_H"] == [(10, 25)]

    def test_hamiltonian_reward_returns_spans(self):
        from examples.hamiltonian.reward_fn import hamiltonian_reward

        text = "COORDINATES: q = x\nMOMENTUM: p = 3*dx/dt\nKINETIC: T = p²/6\nPOTENTIAL: V = 3x²\nHAMILTONIAN: H = p²/6 + 3x²\nEQUATIONS:\n  dq/dt = p/3\n  dp/dt = -6x"
        meta = {
            "H_expr": "p**2/6 + 3*x**2",
            "T_expr": "p**2/6",
            "V_expr": "3*x**2",
            "dqdt": "p/3",
            "dpdt": "-6*x",
            "coordinates": "x",
        }
        result = hamiltonian_reward("test prompt", text, meta)

        assert result.scored_spans, "hamiltonian_reward should return scored_spans"
        assert "q_correct_H" in result.scored_spans
        assert "q_format" in result.scored_spans
        assert len(result.scored_spans["q_correct_H"]) >= 1
        # Format targets labeled sections only - NOT full completion
        # This prevents rewarding arbitrary text outside labeled regions
        assert result.scored_spans["q_format"] != [
            (0, len(text))
        ], "q_format must NOT span full completion - only labeled sections"
        assert (
            len(result.scored_spans["q_format"]) >= 1
        ), "q_format should have labeled section spans"


# --- Per-quality advantage edge cases ---


class TestPerQualityAdvantageEdgeCases:
    """Edge case tests for per-quality advantage computation."""

    def test_perfect_score_zero_advantage(self):
        """r=1.0 should produce zero advantage (SPO r>=1.0 gate)."""
        from qgre.advantages import QGREStepAdvantageEstimator

        step_qualities = {1: ["q_format"], 5: ["q_correct_H"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.1,
            mode="spo",
            step_qualities=step_qualities,
        )
        estimator.set_current_step(1)

        seq_len = 10
        masks = {
            "q_format": torch.ones(seq_len),
            "q_correct_H": torch.zeros(seq_len),
        }
        masks["q_correct_H"][3:7] = 1.0

        # First pass: establish baseline
        reward1 = RewardResult(reward=0.5, scores={"q_format": 0.5, "q_correct_H": 0.5})
        estimator.compute_advantages_with_spans(
            batch_prompt_ids=[0],
            batch_token_ids=[list(range(seq_len))],
            batch_reward_results=[reward1],
            batch_active_qualities=[["q_format", "q_correct_H"]],
            batch_token_masks=[masks],
        )

        # Second pass: perfect H score
        estimator.set_current_step(2)
        reward2 = RewardResult(reward=0.9, scores={"q_format": 0.8, "q_correct_H": 1.0})
        advs, metrics = estimator.compute_advantages_with_spans(
            batch_prompt_ids=[0],
            batch_token_ids=[list(range(seq_len))],
            batch_reward_results=[reward2],
            batch_active_qualities=[["q_format", "q_correct_H"]],
            batch_token_masks=[masks],
        )

        # Perfect score (1.0) should have zero advantage
        assert metrics["sample_0/q_correct_H"]["advantage"] == 0.0
        # Format is not perfect, should have non-zero advantage
        assert metrics["sample_0/q_format"]["advantage"] != 0.0

    def test_staleness_decay_to_prior(self):
        """Baseline should decay toward prior after staleness_window steps."""
        from qgre.advantages import QGREStepAdvantageEstimator

        step_qualities = {1: ["q_format"], 5: ["q_sparse"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.3,
            mode="spo",
            step_qualities=step_qualities,  # Higher LR for faster convergence
            staleness_window=10,
            baseline_prior=0.5,
        )

        seq_len = 10
        masks = {"q_sparse": torch.ones(seq_len)}

        # Establish baseline with multiple observations to converge
        for step in range(1, 11):
            estimator.set_current_step(step)
            reward = RewardResult(reward=0.9, scores={"q_sparse": 0.9})
            estimator.compute_advantages_with_spans(
                batch_prompt_ids=[0],
                batch_token_ids=[list(range(seq_len))],
                batch_reward_results=[reward],
                batch_active_qualities=[["q_sparse"]],
                batch_token_masks=[masks],
            )

        # Check baseline after convergence (should be close to 0.9)
        baseline_fresh = estimator.get_baseline(0, "q_sparse")
        assert baseline_fresh > 0.7, f"Fresh baseline should be high: {baseline_fresh}"

        # Simulate 100 steps passing (10x staleness window)
        estimator.set_current_step(110)
        baseline_stale = estimator.get_baseline(0, "q_sparse")

        # Should decay toward prior (0.5)
        assert baseline_stale < baseline_fresh, "Stale baseline should decay"
        assert (
            baseline_stale > 0.4 and baseline_stale < 0.8
        ), f"Stale baseline should be closer to prior (0.5): {baseline_stale}"

    def test_overlapping_spans_normalized(self):
        """Tokens covered by multiple qualities should get normalized average."""
        from qgre.advantages import QGREStepAdvantageEstimator

        step_qualities = {1: ["q_a", "q_b"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.1,
            mode="spo",
            step_qualities=step_qualities,
            baseline_prior=0.0,  # Zero prior for cleaner math
        )
        estimator.set_current_step(1)

        seq_len = 10
        masks = {
            "q_a": torch.zeros(seq_len),
            "q_b": torch.zeros(seq_len),
        }
        # q_a covers 0-5, q_b covers 3-8 → overlap at 3-5
        masks["q_a"][0:6] = 1.0
        masks["q_b"][3:9] = 1.0

        # Both qualities have high scores (so non-zero advantages from prior=0)
        reward = RewardResult(reward=0.8, scores={"q_a": 0.8, "q_b": 0.6})
        advs, metrics = estimator.compute_advantages_with_spans(
            batch_prompt_ids=[0],
            batch_token_ids=[list(range(seq_len))],
            batch_reward_results=[reward],
            batch_active_qualities=[["q_a", "q_b"]],
            batch_token_masks=[masks],
        )

        a_adv = metrics["sample_0/q_a"]["advantage"]
        b_adv = metrics["sample_0/q_b"]["advantage"]

        # Token 1 (only q_a): should get q_a advantage
        token_1_adv = advs[0][1].item()
        assert (
            abs(token_1_adv - a_adv) < 1e-4
        ), f"Token 1 should match q_a adv: {token_1_adv} vs {a_adv}"

        # Token 7 (only q_b): should get q_b advantage
        token_7_adv = advs[0][7].item()
        assert (
            abs(token_7_adv - b_adv) < 1e-4
        ), f"Token 7 should match q_b adv: {token_7_adv} vs {b_adv}"

        # Token 4 (overlap q_a + q_b): should get normalized average
        token_4_adv = advs[0][4].item()
        expected_overlap = (a_adv + b_adv) / 2.0
        assert (
            abs(token_4_adv - expected_overlap) < 1e-4
        ), f"Overlap token should get average: {token_4_adv} vs {expected_overlap}"

    def test_learnability_variance_thresholds(self):
        """Test learnability = p(1-p) at various mastery levels."""

        # At p=0.5: learnability = 0.25 (maximum)
        p_half = 0.5
        learnability_half = p_half * (1.0 - p_half)
        assert abs(learnability_half - 0.25) < 1e-6

        # At p=0.8: learnability = 0.16
        p_high = 0.8
        learnability_high = p_high * (1.0 - p_high)
        assert abs(learnability_high - 0.16) < 1e-6

        # At p=0.9: learnability = 0.09 (below 0.10 threshold)
        p_very_high = 0.9
        learnability_very_high = p_very_high * (1.0 - p_very_high)
        assert learnability_very_high < 0.10

        # At p=0.95: learnability = 0.0475 (definitely ready)
        p_mastered = 0.95
        learnability_mastered = p_mastered * (1.0 - p_mastered)
        assert learnability_mastered < 0.05

    def test_first_observation_uses_prior(self):
        """First observation for a quality should use prior as baseline."""
        from qgre.advantages import QGREStepAdvantageEstimator

        step_qualities = {1: ["q_new"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.1,
            mode="spo",
            step_qualities=step_qualities,
            baseline_prior=0.5,
        )
        estimator.set_current_step(1)

        seq_len = 5
        masks = {"q_new": torch.ones(seq_len)}

        # First observation: r=0.8, V=prior=0.5 → advantage = 0.3
        reward = RewardResult(reward=0.8, scores={"q_new": 0.8})
        advs, metrics = estimator.compute_advantages_with_spans(
            batch_prompt_ids=[0],
            batch_token_ids=[list(range(seq_len))],
            batch_reward_results=[reward],
            batch_active_qualities=[["q_new"]],
            batch_token_masks=[masks],
        )

        # Advantage should be r - prior = 0.8 - 0.5 = 0.3
        advantage = metrics["sample_0/q_new"]["advantage"]
        assert abs(advantage - 0.3) < 1e-4, f"First observation advantage: {advantage}"

    def test_loss_fn_returns_per_token_loss(self):
        """ClippedPGLossFn should return per-token loss when requested."""
        from qgre.nemo_extracted.loss_functions import ClippedPGLossFn

        cfg = {
            "reference_policy_kl_penalty": 0.0,
            "reference_policy_kl_type": "kl",
            "kl_input_clamp_value": None,
            "kl_output_clamp_value": None,
            "ratio_clip_min": 0.2,
            "ratio_clip_max": 0.2,
            "ratio_clip_c": None,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "truncated_importance_sampling_ratio": None,
            "token_level_loss": True,
        }
        loss_fn = ClippedPGLossFn(cfg)

        batch, seq = 2, 10
        curr_lp = torch.randn(batch, seq)
        prev_lp = curr_lp.clone()  # On-policy
        advantages = torch.randn(batch, seq)
        mask = torch.ones(batch, seq)

        # Without per-token loss
        result = loss_fn(curr_lp, prev_lp, advantages, mask)
        assert len(result) == 2
        loss, metrics = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

        # With per-token loss
        result = loss_fn(curr_lp, prev_lp, advantages, mask, return_per_token_loss=True)
        assert len(result) == 3
        loss, metrics, per_token_loss = result
        assert per_token_loss.shape == (batch, seq)
        assert not per_token_loss.requires_grad  # Should be detached

    def test_per_quality_loss_computation(self):
        """Per-quality loss should be computed from per-token loss × quality mask."""
        from qgre.nemo_extracted.loss_functions import ClippedPGLossFn

        cfg = {
            "reference_policy_kl_penalty": 0.0,
            "reference_policy_kl_type": "kl",
            "kl_input_clamp_value": None,
            "kl_output_clamp_value": None,
            "ratio_clip_min": 0.2,
            "ratio_clip_max": 0.2,
            "ratio_clip_c": None,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "truncated_importance_sampling_ratio": None,
            "token_level_loss": True,
        }
        loss_fn = ClippedPGLossFn(cfg)

        batch, seq = 1, 20
        curr_lp = torch.randn(batch, seq)
        prev_lp = curr_lp.clone()
        advantages = torch.randn(batch, seq)
        mask = torch.ones(batch, seq)

        _, _, per_token_loss = loss_fn(
            curr_lp, prev_lp, advantages, mask, return_per_token_loss=True
        )

        # Create quality masks
        q_a_mask = torch.zeros(seq)
        q_a_mask[0:10] = 1.0  # First half
        q_b_mask = torch.zeros(seq)
        q_b_mask[10:20] = 1.0  # Second half

        # Compute per-quality loss
        q_a_loss = (per_token_loss[0] * q_a_mask).sum() / q_a_mask.sum()
        q_b_loss = (per_token_loss[0] * q_b_mask).sum() / q_b_mask.sum()

        # Should be different (different token regions)
        assert (
            q_a_loss.item() != q_b_loss.item()
        ), "Per-quality losses should differ for different regions"

        # Full mask should match mean of per-token loss
        full_loss = per_token_loss[0].mean()
        weighted_avg = (q_a_loss * 10 + q_b_loss * 10) / 20
        assert (
            abs(full_loss.item() - weighted_avg.item()) < 1e-5
        ), "Weighted average should match full mean"
