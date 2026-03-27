"""Tests for LLDS (Lazy Likelihood Displacement Stabilization) integration."""

import torch
import pytest
from qgre.nemo_extracted.llds import compute_llds_loss
from qgre.generation import GenerationOutput


class TestLLDSLoss:
    """Test the LLDS loss function directly."""

    def test_no_displacement_zero_loss(self):
        """When old_logprob == log_prob, LLDS loss should be zero."""
        lp = torch.tensor([[-1.0, -2.0, -1.5]])
        mask = torch.ones(1, 3)
        advantages = torch.tensor([[0.5, 0.3, 0.2]])
        loss, llds_mask = compute_llds_loss(lp, lp, advantages, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_displacement_triggers_loss(self):
        """When correct completions lose logprob, LLDS should fire."""
        old_lp = torch.tensor([[-1.0, -1.0, -1.0]])
        new_lp = torch.tensor([[-2.0, -2.0, -2.0]])  # All tokens displaced down
        mask = torch.ones(1, 3)
        advantages = torch.tensor([[1.0, 1.0, 1.0]])  # Positive advantage (correct)

        loss, llds_mask = compute_llds_loss(new_lp, old_lp, advantages, mask)
        assert loss.item() > 0, "LLDS should penalize downward displacement on correct completions"
        assert llds_mask.sum().item() == 3, "All 3 tokens should be gated"

    def test_negative_advantage_no_penalty(self):
        """LLDS should NOT fire on incorrect completions (negative advantage)."""
        old_lp = torch.tensor([[-1.0, -1.0, -1.0]])
        new_lp = torch.tensor([[-2.0, -2.0, -2.0]])
        mask = torch.ones(1, 3)
        advantages = torch.tensor([[-1.0, -1.0, -1.0]])  # Negative advantage (incorrect)

        loss, llds_mask = compute_llds_loss(new_lp, old_lp, advantages, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)
        assert llds_mask.sum().item() == 0

    def test_upward_displacement_no_penalty(self):
        """When logprobs increase, no LLDS penalty needed."""
        old_lp = torch.tensor([[-2.0, -2.0, -2.0]])
        new_lp = torch.tensor([[-1.0, -1.0, -1.0]])  # Improved
        mask = torch.ones(1, 3)
        advantages = torch.tensor([[1.0, 1.0, 1.0]])

        loss, llds_mask = compute_llds_loss(new_lp, old_lp, advantages, mask)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mask_respected(self):
        """Padding tokens should not contribute to LLDS loss."""
        old_lp = torch.tensor([[-1.0, -1.0, -1.0]])
        new_lp = torch.tensor([[-2.0, -2.0, -2.0]])
        mask = torch.tensor([[1.0, 1.0, 0.0]])  # Third token is padding
        advantages = torch.tensor([[1.0, 1.0, 1.0]])

        loss, llds_mask = compute_llds_loss(new_lp, old_lp, advantages, mask)
        assert llds_mask[0, 2].item() == 0, "Padding token should not be gated"

    def test_mixed_batch(self):
        """Batch with one correct (displaced) and one incorrect completion."""
        old_lp = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
        new_lp = torch.tensor([[-2.0, -2.0], [-2.0, -2.0]])
        mask = torch.ones(2, 2)
        advantages = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])

        loss, llds_mask = compute_llds_loss(new_lp, old_lp, advantages, mask)
        # Only first sample should contribute (positive advantage)
        assert llds_mask[0].sum().item() == 2
        assert llds_mask[1].sum().item() == 0


class TestGenerationOutputLogprobs:
    """Test GenerationOutput logprobs field."""

    def test_logprobs_default_none(self):
        out = GenerationOutput(token_ids=[[1, 2]], texts=["hi"])
        assert out.logprobs is None

    def test_logprobs_stored(self):
        lps = [[-1.0, -2.0, -1.5]]
        out = GenerationOutput(token_ids=[[1, 2, 3]], texts=["hello"], logprobs=lps)
        assert out.logprobs == lps
        assert len(out.logprobs[0]) == 3

    def test_logprobs_batch(self):
        lps = [[-1.0, -2.0], [-0.5, -1.0]]
        out = GenerationOutput(
            token_ids=[[1, 2], [3, 4]],
            texts=["a", "b"],
            logprobs=lps,
        )
        assert len(out.logprobs) == 2


class TestLogprobExtraction:
    """Test vLLM logprob extraction logic in UnslothBackend.generate()."""

    def test_extract_by_sampled_token_id(self):
        """Logprobs should be extracted by the actual sampled token_id, not dict iteration order."""
        from dataclasses import dataclass

        @dataclass
        class MockLogprob:
            logprob: float
            rank: int | None = None
            decoded_token: str | None = None

        # Simulate: top-1 token is 99 (logprob=-0.1), sampled token is 42 (logprob=-2.5)
        pos_dict = {99: MockLogprob(logprob=-0.1), 42: MockLogprob(logprob=-2.5)}
        sampled_id = 42

        # The correct behavior: extract by sampled_id, not by iteration order
        if sampled_id in pos_dict:
            result = pos_dict[sampled_id].logprob
        else:
            result = next(iter(pos_dict.values())).logprob

        assert result == -2.5, "Should extract sampled token's logprob, not top-1"

    def test_iteration_order_gives_wrong_answer(self):
        """Demonstrate why next(iter()) is wrong when sampled != top-1."""
        from dataclasses import dataclass

        @dataclass
        class MockLogprob:
            logprob: float

        pos_dict = {99: MockLogprob(logprob=-0.1), 42: MockLogprob(logprob=-2.5)}
        # next(iter()) gives the first inserted key's value
        wrong = next(iter(pos_dict.values())).logprob
        assert wrong == -0.1, "iter() grabs top-1, not sampled"
        # Correct: extract by token_id
        correct = pos_dict[42].logprob
        assert correct == -2.5
