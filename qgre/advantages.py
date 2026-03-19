from __future__ import annotations

from collections import defaultdict
from typing import Callable

import numpy as np
import torch

from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES, Segmenter, qwen3_xml_segmenter
from qgre.types import RewardResult


def build_batch_reward_tensors(
    reward_results: list[RewardResult],
) -> dict[str, torch.Tensor]:
    """Convert list[RewardResult] → dict[str, Tensor] per quality component.

    Each tensor has shape [batch_size]. Missing keys are zero-filled.
    """
    if not reward_results:
        return {}

    all_keys: set[str] = set()
    for rr in reward_results:
        all_keys.update(rr.scores.keys())

    tensors: dict[str, torch.Tensor] = {}
    for key in sorted(all_keys):
        values = [rr.scores.get(key, 0.0) for rr in reward_results]
        tensors[key] = torch.tensor(values, dtype=torch.float32)

    return tensors


def build_phase_qualities(
    step_qualities: dict[int, list[str]],
    cumulative: bool = True,
) -> dict[int, list[str]]:
    """Build phase→qualities mapping from step_qualities config.

    If cumulative=True (default QGRE behavior): phase N includes all qualities from steps 1..N.
    If cumulative=False: phase N includes only step N's qualities.
    """
    steps = sorted(step_qualities.keys())
    if cumulative:
        return {
            phase: [q for s in steps if s <= phase for q in step_qualities[s]]
            for phase in steps
        }
    else:
        return dict(step_qualities)


class QGREStepAdvantageEstimator:
    """Unified: SPO + GDPO + VPRM + QGRE phase gating.

    Configurable for any domain. Accepts:
    - step_qualities: mapping of step_num → quality names (from your reward_fn)
    - segmenter: function that splits token IDs into step regions
    - mode: "spo" (persistent value tracker) or "grpo" (group-mean baseline)
    """

    def __init__(
        self,
        lr: float = 0.1,
        mode: str = "spo",
        step_qualities: dict[int, list[str]] | None = None,
        segmenter: Segmenter | None = None,
    ):
        self.lr = lr
        self.mode = mode
        self.step_qualities = step_qualities or HYPERGRAPH_V1_STEP_QUALITIES
        self.segmenter = segmenter or qwen3_xml_segmenter
        self._step_nums = sorted(self.step_qualities.keys())
        self.V: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self._step_seen: dict[int, set[int]] = defaultdict(set)

    def compute_advantages(
        self,
        batch_prompt_ids: list[int],
        batch_token_ids: list[list[int]],
        batch_reward_results: list[RewardResult],
        batch_active_qualities: list[list[str]],
        group_size: int | None = None,
    ) -> list[torch.Tensor]:
        """Compute per-token advantages via segment → step rewards → SPO/GRPO → GDPO → broadcast."""
        batch_size = len(batch_token_ids)

        # Phase 1: Segment tokens + compute per-step rewards
        all_regions: list[list[str]] = []
        all_step_rewards: list[dict[int, float]] = []

        for i in range(batch_size):
            regions = self.segmenter(batch_token_ids[i])
            step_rews: dict[int, float] = {}
            for step_num, quality_keys in self.step_qualities.items():
                active = [k for k in quality_keys if k in batch_active_qualities[i]]
                if active:
                    step_rews[step_num] = float(np.mean([
                        batch_reward_results[i].scores.get(k, 0.0) for k in active
                    ]))
                else:
                    step_rews[step_num] = 0.0
            all_step_rewards.append(step_rews)
            all_regions.append(regions)

        # Phase 2: Per-step advantages (SPO or GRPO baseline)
        step_advs: dict[int, torch.Tensor] = {
            s: torch.zeros(batch_size) for s in self._step_nums
        }

        if self.mode == "spo":
            self._compute_spo_advantages(batch_prompt_ids, all_step_rewards, step_advs, batch_size)
        else:
            self._compute_grpo_advantages(
                batch_prompt_ids, all_step_rewards, step_advs, batch_size,
                group_size=group_size or batch_size,
            )

        # GDPO-style: normalize each step's advantages across the batch
        for step_num in self._step_nums:
            mean = step_advs[step_num].mean()
            std = step_advs[step_num].std(correction=0)
            if std > 1e-8:
                step_advs[step_num] = (step_advs[step_num] - mean) / (std + 1e-8)
            else:
                step_advs[step_num] = step_advs[step_num] - mean

        # Phase 3: Broadcast per-step advantages to per-token by region
        batch_advantages: list[torch.Tensor] = []
        for i in range(batch_size):
            token_advs = torch.zeros(len(batch_token_ids[i]))
            for t, region in enumerate(all_regions[i]):
                if region.startswith("STEP_"):
                    sn = int(region.split("_")[1])
                    if sn in step_advs:
                        token_advs[t] = step_advs[sn][i]
            batch_advantages.append(token_advs)

        return batch_advantages

    def _compute_spo_advantages(
        self,
        batch_prompt_ids: list[int],
        all_step_rewards: list[dict[int, float]],
        step_advs: dict[int, torch.Tensor],
        batch_size: int,
    ):
        for step_num in self._step_nums:
            for i in range(batch_size):
                pid = batch_prompt_ids[i]
                r = all_step_rewards[i].get(step_num, 0.0)
                v = self.V[pid][step_num]

                if step_num not in self._step_seen[pid]:
                    v = r
                    self._step_seen[pid].add(step_num)

                step_advs[step_num][i] = r - v
                self.V[pid][step_num] = v + self.lr * (r - v)

    def _compute_grpo_advantages(
        self,
        batch_prompt_ids: list[int],
        all_step_rewards: list[dict[int, float]],
        step_advs: dict[int, torch.Tensor],
        batch_size: int,
        group_size: int,
    ):
        if batch_size % group_size != 0:
            raise ValueError(
                f"GRPO requires batch_size divisible by group_size, "
                f"got {batch_size} % {group_size} != 0."
            )
        num_groups = batch_size // group_size
        for step_num in self._step_nums:
            for g in range(num_groups):
                start = g * group_size
                end = start + group_size
                group_rewards = [all_step_rewards[i].get(step_num, 0.0) for i in range(start, end)]
                mean = float(np.mean(group_rewards))
                std = float(np.std(group_rewards)) + 1e-8
                for i in range(start, end):
                    step_advs[step_num][i] = (all_step_rewards[i].get(step_num, 0.0) - mean) / std

    def on_tier_advance(self, new_tier: int, prompt_tier_map: dict[int, int]):
        for pid, tier in prompt_tier_map.items():
            if tier == new_tier:
                self.V[pid] = defaultdict(float)
                self._step_seen[pid] = set()

    def state_dict(self) -> dict:
        return {
            "V": {pid: dict(steps) for pid, steps in self.V.items()},
            "step_seen": {pid: list(steps) for pid, steps in self._step_seen.items()},
            "lr": self.lr,
            "mode": self.mode,
        }

    def load_state_dict(self, state: dict):
        self.lr = state.get("lr", self.lr)
        self.mode = state.get("mode", self.mode)
        self.V = defaultdict(lambda: defaultdict(float))
        for pid, steps in state.get("V", {}).items():
            for step_num, val in steps.items():
                self.V[int(pid)][int(step_num)] = float(val)
        self._step_seen = defaultdict(set)
        for pid, steps in state.get("step_seen", {}).items():
            self._step_seen[int(pid)] = set(int(s) for s in steps)
