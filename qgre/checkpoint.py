from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

from qgre.types import QUALITY_WINDOW_SIZE, GameState


def gamestate_to_dict(gs: GameState) -> dict:
    """Serialize GameState to a plain dict safe for json.dumps and torch.save.

    Converts: deque → {values, maxlen} for safe round-trip.
    """
    # Serialize 2D tier_mastery: {tier: {step_num: {values, maxlen}}}
    tm = {}
    for tier, step_windows in gs.tier_mastery.items():
        tm[tier] = {}
        for step_num, dq in step_windows.items():
            tm[tier][step_num] = {"values": list(dq), "maxlen": dq.maxlen}

    return {
        "step_count": gs.step_count,
        "mastery_threshold": gs.mastery_threshold,
        "tier_mastery": tm,
        "tier_phases": dict(gs.tier_phases),
        "active_tiers": list(gs.active_tiers),
        "tier_steps_at_phase_start": dict(gs.tier_steps_at_phase_start),
        "phase_history": list(gs.phase_history),
        "stagnation_timeout": gs.stagnation_timeout,
        "plateau_window": gs.plateau_window,
        "plateau_threshold": gs.plateau_threshold,
    }


def gamestate_from_dict(d: dict) -> GameState:
    """Reconstruct GameState from a plain dict.

    Restores: list → deque (with maxlen). Handles both old 1D and new 2D format.
    """
    gs = GameState()
    gs.step_count = d.get("step_count", 0)
    gs.mastery_threshold = d.get("mastery_threshold", 0.8)
    gs.phase_history = list(d.get("phase_history", []))
    gs.stagnation_timeout = d.get("stagnation_timeout", 200)
    gs.plateau_window = d.get("plateau_window", 50)
    gs.plateau_threshold = d.get("plateau_threshold", 0.02)

    # 2D tier fields
    gs.tier_phases = d.get("tier_phases", {"default": d.get("phase", 1)})
    gs.active_tiers = d.get("active_tiers", ["default"])
    gs.tier_steps_at_phase_start = d.get("tier_steps_at_phase_start", {})

    # Restore tier_mastery from serialized format
    tm = d.get("tier_mastery", {})
    gs.tier_mastery = {}
    for tier, step_windows in tm.items():
        gs.tier_mastery[tier] = {}
        for step_num, window_data in step_windows.items():
            maxlen = window_data.get("maxlen", QUALITY_WINDOW_SIZE)
            values = window_data.get("values", [])
            gs.tier_mastery[tier][int(step_num)] = deque(values, maxlen=maxlen)

    # Backward compat: migrate old 1D step_mastery to "default" tier
    if "step_mastery" in d and "tier_mastery" not in d:
        gs.tier_mastery["default"] = {}
        for step_num, window_data in d["step_mastery"].items():
            maxlen = window_data.get("maxlen", QUALITY_WINDOW_SIZE)
            values = window_data.get("values", [])
            gs.tier_mastery["default"][int(step_num)] = deque(values, maxlen=maxlen)

    return gs


def save_checkpoint(
    path: str | Path,
    global_step: int,
    model_state_dict: dict | None = None,
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
    game_state: GameState | None = None,
    advantage_estimator_state: dict | None = None,
    rng_state=None,
    cuda_rng_state=None,
    vprm_critic_state: dict | None = None,
    vprm_optimizer_state: dict | None = None,
):
    """Save full training state to a checkpoint file."""
    import torch

    checkpoint = {
        "global_step": global_step,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
        "scheduler_state_dict": scheduler_state_dict,
        "game_state": gamestate_to_dict(game_state) if game_state else None,
        "advantage_estimator_state": advantage_estimator_state,
        "rng_state": rng_state if rng_state is not None else torch.get_rng_state(),
        "cuda_rng_state": cuda_rng_state,
        "vprm_critic_state": vprm_critic_state,
        "vprm_optimizer_state": vprm_optimizer_state,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: str | Path) -> dict:
    """Load checkpoint from file. Returns raw dict — caller restores state."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("game_state") is not None:
        checkpoint["game_state"] = gamestate_from_dict(checkpoint["game_state"])
    return checkpoint


def discover_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Scan directory for global_step_N checkpoints, return path to latest."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    pattern = re.compile(r"global_step_(\d+)")
    candidates = []
    for entry in checkpoint_dir.iterdir():
        match = pattern.search(entry.name)
        if match:
            candidates.append((int(match.group(1)), entry))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]
