"""Tests for VPRM Critic — per-region per-dimension learned baseline."""

import torch
import pytest

from qgre.critic import VPRMCritic, QualityMLP
from qgre.config import QGREConfig, VPRMConfig


# --- Fixtures ---

STEP_QUALITIES = {
    1: ["q_format", "q_has_math"],
    2: ["q_momentum_defined", "q_T_uses_p"],
    3: ["q_correct_dqdt", "q_correct_dpdt"],
    4: ["q_correct_H", "q_consistency"],
}

HIDDEN_DIM = 256  # Small for fast tests


@pytest.fixture
def critic():
    return VPRMCritic(hidden_dim=HIDDEN_DIM, step_qualities=STEP_QUALITIES)


@pytest.fixture
def sample_regions():
    """Regions with 4 steps — mimics Hamiltonian structured output."""
    return (
        ["STEP_1"] * 20 +
        ["STEP_2"] * 20 +
        ["STEP_3"] * 30 +
        ["STEP_4"] * 30
    )


@pytest.fixture
def sample_hidden_states():
    return torch.randn(100, HIDDEN_DIM)


# --- QualityMLP tests ---

class TestQualityMLP:
    def test_output_shape(self):
        mlp = QualityMLP(hidden_dim=HIDDEN_DIM)
        x = torch.randn(4, HIDDEN_DIM)
        out = mlp(x)
        assert out.shape == (4, 1)

    def test_gradient_flows(self):
        mlp = QualityMLP(hidden_dim=HIDDEN_DIM)
        x = torch.randn(1, HIDDEN_DIM)
        out = mlp(x)
        out.sum().backward()
        assert all(p.grad is not None for p in mlp.parameters())


# --- VPRMCritic construction tests ---

class TestCriticConstruction:
    def test_creates_one_head_per_quality(self, critic):
        all_qualities = []
        for qs in STEP_QUALITIES.values():
            all_qualities.extend(qs)
        assert len(critic.heads) == len(all_qualities)
        for q in all_qualities:
            assert q in critic.heads

    def test_quality_to_step_mapping(self, critic):
        assert critic._quality_to_step["q_format"] == 1
        assert critic._quality_to_step["q_correct_H"] == 4

    def test_param_count_reasonable(self, critic):
        n_params = sum(p.numel() for p in critic.parameters())
        n_qualities = len(critic.quality_names)
        # Each MLP: hidden_dim*128 + 128 + 128*128 + 128 + 128*1 + 1
        # x2 for online + target heads
        expected_per_mlp = HIDDEN_DIM * 128 + 128 + 128 * 128 + 128 + 128 + 1
        assert abs(n_params - n_qualities * expected_per_mlp * 2) < 10


# --- Forward pass tests ---

class TestCriticForward:
    def test_returns_prediction_per_quality(self, critic, sample_hidden_states, sample_regions):
        preds = critic(sample_hidden_states, sample_regions)
        assert set(preds.keys()) == set(critic.quality_names)

    def test_predictions_are_scalar(self, critic, sample_hidden_states, sample_regions):
        preds = critic(sample_hidden_states, sample_regions)
        for v in preds.values():
            assert v.dim() == 0  # scalar

    def test_missing_region_returns_none(self, critic):
        """When a quality's region is missing, prediction should be None (AE5)."""
        hs = torch.randn(50, HIDDEN_DIM)
        regions = ["STEP_1"] * 50  # Only STEP_1, missing 2-4
        preds = critic(hs, regions)
        # Qualities mapped to STEP_2, 3, 4 should get None (missing region)
        assert preds["q_momentum_defined"] is None
        assert preds["q_correct_H"] is None
        # STEP_1 qualities should produce predictions (may be zero by chance at init)

    def test_think_and_other_regions_ignored(self, critic):
        """THINK and OTHER regions should not affect predictions."""
        hs = torch.randn(60, HIDDEN_DIM)
        regions = ["THINK"] * 10 + ["STEP_1"] * 20 + ["OTHER"] * 10 + ["STEP_2"] * 20
        preds = critic(hs, regions)
        # Should still produce predictions for STEP_1 and STEP_2
        assert "q_format" in preds
        assert "q_momentum_defined" in preds


# --- Advantage computation tests ---

class TestCriticAdvantages:
    def test_advantage_shape(self, critic, sample_hidden_states, sample_regions):
        rewards = {q: 0.8 for q in critic.quality_names}
        advs, losses = critic.compute_advantages(sample_hidden_states, sample_regions, rewards)
        assert set(advs.keys()) == set(critic.quality_names)
        assert all(isinstance(v, float) for v in advs.values())
        assert all(isinstance(v, torch.Tensor) for v in losses.values())

    def test_advantage_clipping(self, critic, sample_hidden_states, sample_regions):
        """Advantages should be clipped to [-clip, +clip]."""
        # Use extreme reward to trigger clipping
        rewards = {q: 100.0 for q in critic.quality_names}
        advs, _ = critic.compute_advantages(sample_hidden_states, sample_regions, rewards)
        for q, adv in advs.items():
            assert abs(adv) <= critic.clip_advantage + 0.01

    def test_critic_loss_is_mse(self, critic, sample_hidden_states, sample_regions):
        """Critic loss should be MSE between prediction and actual reward."""
        rewards = {"q_format": 0.9, "q_has_math": 0.5}
        _, losses = critic.compute_advantages(sample_hidden_states, sample_regions, rewards)
        # Loss for each quality should be non-negative
        for loss in losses.values():
            assert loss.item() >= 0.0

    def test_critic_loss_decreases_with_training(self, critic, sample_hidden_states, sample_regions):
        """Critic loss should decrease after gradient updates."""
        torch.manual_seed(42)
        rewards = {q: 0.7 for q in critic.quality_names}
        optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

        # Measure initial loss
        _, initial_losses = critic.compute_advantages(sample_hidden_states, sample_regions, rewards)
        initial_total = sum(l.item() for l in initial_losses.values())

        # Train for a few steps
        for _ in range(50):
            _, losses = critic.compute_advantages(sample_hidden_states, sample_regions, rewards)
            total = sum(losses.values())
            total.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Measure final loss
        _, final_losses = critic.compute_advantages(sample_hidden_states, sample_regions, rewards)
        final_total = sum(l.item() for l in final_losses.values())

        assert final_total < initial_total, f"Critic loss should decrease: {initial_total:.4f} → {final_total:.4f}"


# --- Batch computation tests ---

class TestBatchAdvantages:
    def test_batch_returns_correct_count(self, critic, sample_hidden_states, sample_regions):
        batch_hs = [sample_hidden_states] * 4
        batch_regions = [sample_regions] * 4
        batch_rewards = [{q: 0.5 for q in critic.quality_names}] * 4
        advs, loss = critic.compute_batch_advantages(batch_hs, batch_regions, batch_rewards)
        assert len(advs) == 4

    def test_spo_fallback_mask(self, critic, sample_hidden_states, sample_regions):
        """SPO fallback mask should zero out advantages for masked samples."""
        batch_hs = [sample_hidden_states] * 3
        batch_regions = [sample_regions] * 3
        batch_rewards = [{q: 0.5 for q in critic.quality_names}] * 3
        fallback = [False, True, False]  # Sample 1 uses SPO fallback
        advs, _ = critic.compute_batch_advantages(batch_hs, batch_regions, batch_rewards, fallback)
        # Sample 1 should have all-zero advantages
        assert all(v == 0.0 for v in advs[1].values())
        # Samples 0 and 2 should have critic-computed advantages (may be zero by chance)


# --- Checkpoint / state dict tests ---

class TestCriticCheckpoint:
    def test_state_dict_roundtrip(self, critic, sample_hidden_states, sample_regions):
        """Save and restore should produce identical predictions."""
        preds_before = critic(sample_hidden_states, sample_regions)
        state = critic.state_dict_with_meta()
        restored = VPRMCritic.from_checkpoint(state)
        preds_after = restored(sample_hidden_states, sample_regions)
        for q in critic.quality_names:
            assert torch.allclose(preds_before[q], preds_after[q])


# --- Config tests ---

class TestVPRMConfig:
    def test_default_disabled(self):
        cfg = QGREConfig()
        assert cfg.vprm.enabled is False

    def test_yaml_parsing(self, tmp_path):
        yaml_content = """
model:
  path: test
  pad_token: "<pad>"
  pad_token_id: 0
data:
  train_files: ["dummy.parquet"]
algorithm:
  step_qualities:
    1: ["q_format"]
generation:
  stop_token_ids: [2]
vprm:
  enabled: true
  intermediate_dim: 64
  lr: 0.001
  clip_advantage: 3.0
  spo_fallback_min_regions: 3
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)
        cfg = QGREConfig.from_yaml(yaml_path)
        assert cfg.vprm.enabled is True
        assert cfg.vprm.intermediate_dim == 64
        assert cfg.vprm.lr == 0.001
        assert cfg.vprm.clip_advantage == 3.0
        assert cfg.vprm.spo_fallback_min_regions == 3


# --- Integration with advantages.py ---

class TestVPRMAdvantagesIntegration:
    def test_compute_advantages_vprm(self, critic, sample_hidden_states, sample_regions):
        """Test the standalone compute_advantages_vprm function."""
        from qgre.advantages import compute_advantages_vprm
        from qgre.types import RewardResult

        rewards = RewardResult(
            reward=0.7,
            scores={q: 0.7 for q in critic.quality_names},
        )

        advs, loss, used = compute_advantages_vprm(
            critic=critic,
            hidden_states=sample_hidden_states,
            regions=sample_regions,
            reward_result=rewards,
            step_qualities=STEP_QUALITIES,
            active_qualities=list(critic.quality_names),
        )

        assert advs.shape == (100,)
        assert used is True
        assert loss.item() >= 0.0

    def test_spo_fallback_when_single_region(self, critic):
        """When only 1 region exists, should fall back to SPO."""
        from qgre.advantages import compute_advantages_vprm
        from qgre.types import RewardResult

        hs = torch.randn(50, HIDDEN_DIM)
        regions = ["STEP_1"] * 50  # Only one region
        rewards = RewardResult(reward=0.5, scores={"q_format": 0.5})

        advs, loss, used = compute_advantages_vprm(
            critic=critic,
            hidden_states=hs,
            regions=regions,
            reward_result=rewards,
            step_qualities=STEP_QUALITIES,
            active_qualities=["q_format"],
            min_regions=2,
        )

        assert used is False
        assert advs.sum().item() == 0.0
