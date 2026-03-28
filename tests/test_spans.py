"""Tests for span-based advantage assignment."""

import torch
import pytest

from qgre.spans import build_char_to_token_map, scored_spans_to_token_masks
from qgre.types import RewardResult


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
    def test_single_span(self, simple_tokenizer):
        text = "V = kx + mgx"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)
        assert char_map is not None

        spans = {"q_V_correct": [(0, 6)]}  # "V = kx"
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids))

        assert "q_V_correct" in masks
        assert masks["q_V_correct"].shape == (len(token_ids),)
        # First 6 tokens should be 1.0
        assert masks["q_V_correct"][:6].sum() == 6.0
        assert masks["q_V_correct"][6:].sum() == 0.0

    def test_multiple_spans_same_quality(self, simple_tokenizer):
        text = "H = a, then H = b"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        # Two H expressions
        spans = {"q_correct_H": [(0, 5), (12, 18)]}  # "H = a" and " H = b"
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids))

        mask = masks["q_correct_H"]
        assert mask[0:5].sum() == 5.0  # First span
        assert mask[12:18].sum() >= 5.0  # Second span (boundary token may vary)
        assert mask[5:12].sum() == 0.0  # Gap between

    def test_full_completion_span(self, simple_tokenizer):
        text = "full text here"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        spans = {"q_format": [(0, len(text))]}
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids))

        # All tokens should be 1.0
        assert masks["q_format"].sum() == len(token_ids)

    def test_empty_spans(self, simple_tokenizer):
        text = "test"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        spans = {"q_correct_H": []}
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids))
        assert masks["q_correct_H"].sum() == 0.0

    def test_overlapping_qualities(self, simple_tokenizer):
        text = "H = T + V"
        token_ids = simple_tokenizer.encode(text)
        char_map = build_char_to_token_map(token_ids, simple_tokenizer)

        spans = {
            "q_correct_H": [(0, 9)],  # Whole expression
            "q_T_uses_p": [(4, 5)],   # Just "T"
        }
        masks = scored_spans_to_token_masks(spans, char_map, len(token_ids))

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
            lr=0.1, mode="spo", step_qualities=step_qualities,
        )

        seq_len = 20
        token_masks = {
            "q_format": torch.ones(seq_len),
            "q_correct_H": torch.zeros(seq_len),
        }
        token_masks["q_correct_H"][3:8] = 1.0   # First H in derivation
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
        advantages = estimator.compute_advantages_with_spans(
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
        assert abs(advs[3].item()) > abs(advs[10].item()), \
            "H tokens should have stronger signal than non-expression tokens"

    def test_backward_compat_empty_masks(self):
        """When token_masks are empty, all advantages should be zero."""
        from qgre.advantages import QGREStepAdvantageEstimator

        step_qualities = {1: ["q_format"], 5: ["q_correct_H"]}
        estimator = QGREStepAdvantageEstimator(
            lr=0.1, mode="spo", step_qualities=step_qualities,
        )

        reward = RewardResult(reward=0.5, scores={"q_format": 1.0, "q_correct_H": 0.5})
        advantages = estimator.compute_advantages_with_spans(
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

        text = "derivation: H = p²/6 + 3x²\nmore text\nHAMILTONIAN: H = p²/6 + 3x²"
        spans = _find_expression_spans(text)

        assert "q_correct_H" in spans
        assert len(spans["q_correct_H"]) == 2  # Two H = ... occurrences

    def test_finds_V_spans(self):
        from examples.hamiltonian.reward_fn import _find_expression_spans

        text = "V = kx² + mgx\nPOTENTIAL: V = 3x²"
        spans = _find_expression_spans(text)

        assert len(spans["q_V_correct"]) == 2

    def test_format_is_full_completion(self):
        from examples.hamiltonian.reward_fn import _find_expression_spans

        text = "some completion text"
        spans = _find_expression_spans(text)

        assert spans["q_format"] == [(0, len(text))]
        assert spans["q_has_math"] == [(0, len(text))]

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
        meta = {"H_expr": "p**2/6 + 3*x**2", "T_expr": "p**2/6", "V_expr": "3*x**2",
                "dqdt": "p/3", "dpdt": "-6*x", "coordinates": "x"}
        result = hamiltonian_reward("test prompt", text, meta)

        assert result.scored_spans, "hamiltonian_reward should return scored_spans"
        assert "q_correct_H" in result.scored_spans
        assert "q_format" in result.scored_spans
        assert len(result.scored_spans["q_correct_H"]) >= 1
        # Format should be full completion
        assert result.scored_spans["q_format"] == [(0, len(text))]
