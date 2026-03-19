from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

from qgre.types import QUALITY_WINDOW_SIZE, GameState


def gamestate_to_dict(gs: GameState) -> dict:
    """Serialize GameState to a plain dict safe for json.dumps and torch.save.

    Converts: deque → {values, maxlen} for safe round-trip.
    """
    sm = {}
    for step_num, dq in gs.step_mastery.items():
        sm[step_num] = {"values": list(dq), "maxlen": dq.maxlen}

    return {
        "phase": gs.phase,
        "step_count": gs.step_count,
        "mastery_threshold": gs.mastery_threshold,
        "step_mastery": sm,
        "phase_history": list(gs.phase_history),
    }


def gamestate_from_dict(d: dict) -> GameState:
    """Reconstruct GameState from a plain dict.

    Restores: list → deque (with maxlen).
    """
    gs = GameState()
    gs.phase = d.get("phase", 1)
    gs.step_count = d.get("step_count", 0)
    gs.mastery_threshold = d.get("mastery_threshold", 0.8)
    gs.phase_history = list(d.get("phase_history", []))

    sm = d.get("step_mastery", {})
    gs.step_mastery = {}
    for step_num, window_data in sm.items():
        maxlen = window_data.get("maxlen", QUALITY_WINDOW_SIZE)
        values = window_data.get("values", [])
        gs.step_mastery[int(step_num)] = deque(values, maxlen=maxlen)

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
