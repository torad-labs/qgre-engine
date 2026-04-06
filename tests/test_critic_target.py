"""Tests for VPRM Critic Polyak-averaged target network."""

import pytest
import torch

from qgre.critic import VPRMCritic
from qgre.types import TrainingContext


STEP_QUALITIES = {1: ["q_format"], 5: ["q_correct_H"]}
HIDDEN_DIM = 64


@pytest.fixture
def ctx():
    """TrainingContext for tests — CPU device."""
    return TrainingContext(device=torch.device("cpu"), dtype=torch.float32, step=0)


@pytest.fixture
def critic():
    torch.manual_seed(42)
    return VPRMCritic(hidden_dim=HIDDEN_DIM, step_qualities=STEP_QUALITIES)


class TestTargetNetworkInit:
    def test_target_heads_exist(self, critic):
        assert hasattr(critic, "target_heads")
        assert set(critic.target_heads.keys()) == set(critic.heads.keys())

    def test_target_heads_match_online_at_init(self, critic):
        for q in critic.quality_names:
            for op, tp in zip(
                critic.heads[q].parameters(), critic.target_heads[q].parameters(), strict=False
            ):
                assert torch.allclose(op.data, tp.data)

    def test_target_heads_no_grad(self, critic):
        for param in critic.target_heads.parameters():
            assert not param.requires_grad


class TestPolyakAveraging:
    def test_polyak_moves_target(self, critic):
        # Modify online heads
        for param in critic.heads["q_format"].parameters():
            param.data.fill_(1.0)

        # Before update: target still at init values
        target_before = [p.data.clone() for p in critic.target_heads["q_format"].parameters()]

        critic.update_target_network(tau=0.5)

        # After update: target should be midpoint between old target and online (1.0)
        for tp, before in zip(
            critic.target_heads["q_format"].parameters(), target_before, strict=False
        ):
            expected = 0.5 * before + 0.5 * torch.ones_like(before)
            assert torch.allclose(tp.data, expected, atol=1e-6)

    def test_tau_zero_no_change(self, critic):
        before = [p.data.clone() for p in critic.target_heads["q_format"].parameters()]
        for param in critic.heads["q_format"].parameters():
            param.data.fill_(99.0)
        critic.update_target_network(tau=0.0)
        for tp, b in zip(critic.target_heads["q_format"].parameters(), before, strict=False):
            assert torch.allclose(tp.data, b)

    def test_tau_one_full_copy(self, critic):
        for param in critic.heads["q_format"].parameters():
            param.data.fill_(42.0)
        critic.update_target_network(tau=1.0)
        for tp in critic.target_heads["q_format"].parameters():
            assert torch.allclose(tp.data, torch.full_like(tp.data, 42.0))


class TestSyncTargetToOnline:
    def test_hard_sync(self, critic):
        for param in critic.heads["q_format"].parameters():
            param.data.fill_(7.0)
        critic.sync_target_to_online()
        for tp in critic.target_heads["q_format"].parameters():
            assert torch.allclose(tp.data, torch.full_like(tp.data, 7.0))


class TestTargetForAdvantages:
    def test_target_predictions_stable_after_online_update(self, critic, ctx):
        hs = torch.randn(50, HIDDEN_DIM)
        regions = ["STEP_1"] * 25 + ["STEP_5"] * 25

        # Get target predictions before
        preds_before = critic.forward(hs, regions, ctx=ctx, use_target=True)
        val_before = preds_before["q_format"].item()

        # Train online heads (simulate optimizer step)
        for param in critic.heads["q_format"].parameters():
            param.data += 0.5

        # Target predictions should barely change (tau=0.01)
        critic.update_target_network(tau=0.01)
        preds_after = critic.forward(hs, regions, ctx=ctx, use_target=True)
        val_after = preds_after["q_format"].item()

        assert abs(val_after - val_before) < 0.1, "Target should move slowly"

    def test_online_loss_not_from_target(self, critic, ctx):
        hs = torch.randn(50, HIDDEN_DIM)
        regions = ["STEP_1"] * 25 + ["STEP_5"] * 25
        rewards = {"q_format": 0.9, "q_correct_H": 0.4}

        advs, losses = critic.compute_advantages(hs, regions, rewards, ctx=ctx)

        # Losses should have requires_grad (from online heads)
        for loss in losses.values():
            assert loss.requires_grad, "MSE loss should come from online heads (trainable)"


class TestCheckpointRoundTrip:
    def test_state_dict_includes_target(self, critic):
        state = critic.state_dict_with_meta()
        target_keys = [k for k in state["model_state"] if "target_heads" in k]
        assert len(target_keys) > 0, "target_heads should be in state_dict"

    def test_roundtrip_preserves_target(self, critic):
        # Modify target via Polyak
        for param in critic.heads["q_format"].parameters():
            param.data.fill_(3.0)
        critic.update_target_network(tau=0.5)

        # Save and restore
        state = critic.state_dict_with_meta()
        restored = VPRMCritic.from_checkpoint(state)

        # Target should match
        for tp_orig, tp_rest in zip(
            critic.target_heads["q_format"].parameters(),
            restored.target_heads["q_format"].parameters(),
            strict=False,
        ):
            assert torch.allclose(tp_orig.data, tp_rest.data)

    def test_old_checkpoint_without_target(self, critic):
        """Old checkpoint (no target_heads) should sync from online."""
        state = critic.state_dict_with_meta()
        # Remove target_heads keys to simulate old checkpoint
        state["model_state"] = {
            k: v for k, v in state["model_state"].items() if "target_heads" not in k
        }

        restored = VPRMCritic.from_checkpoint(state)

        # Target should match online (synced in from_checkpoint)
        for q in restored.quality_names:
            for op, tp in zip(
                restored.heads[q].parameters(), restored.target_heads[q].parameters(), strict=False
            ):
                assert torch.allclose(op.data, tp.data)
