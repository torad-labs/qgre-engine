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
        "quality_window_size": gs.quality_window_size,
    }


def gamestate_from_dict(d: dict) -> GameState:
    """Reconstruct GameState from a plain dict.

    Restores: list → deque (with maxlen). Handles both old 1D and new 2D format.
    """
    # CP3-001: Add type checking before accessing dict methods
    if not isinstance(d, dict):
        raise TypeError(
            f"CP3-001: gamestate_from_dict expects dict, got {type(d).__name__}. "
            "Checkpoint may be corrupted."
        )
    gs = GameState()
    gs.step_count = d.get("step_count", 0)
    gs.mastery_threshold = d.get("mastery_threshold", 0.8)
    gs.phase_history = list(d.get("phase_history", []))
    gs.stagnation_timeout = d.get("stagnation_timeout", 200)
    gs.plateau_window = d.get("plateau_window", 50)
    gs.plateau_threshold = d.get("plateau_threshold", 0.02)
    gs.quality_window_size = d.get("quality_window_size", QUALITY_WINDOW_SIZE)

    # 2D tier fields
    # CP3-007: Validate tier_phases is dict before using
    tier_phases_raw = d.get("tier_phases", {"default": d.get("phase", 1)})
    if not isinstance(tier_phases_raw, dict):
        raise TypeError(
            f"CP3-007: tier_phases expected dict, got {type(tier_phases_raw).__name__}. "
            "Checkpoint may be corrupted."
        )
    # C2: Cast tier_phases values to int to prevent float corruption
    gs.tier_phases = {k: int(v) for k, v in tier_phases_raw.items()}
    gs.active_tiers = d.get("active_tiers", ["default"])
    gs.tier_steps_at_phase_start = d.get("tier_steps_at_phase_start", {})

    # C3: Initialize missing tier_mastery entries for active_tiers
    for tier in gs.active_tiers:
        if tier not in gs.tier_mastery:
            gs.tier_mastery[tier] = {}

    # Restore tier_mastery from serialized format
    tm = d.get("tier_mastery", {})
    gs.tier_mastery = {}
    for tier, step_windows in tm.items():
        gs.tier_mastery[tier] = {}
        for step_num, window_data in step_windows.items():
            # CP3-001: Add isinstance check on window_data before .get()
            if not isinstance(window_data, dict):
                raise TypeError(
                    f"CP3-001: window_data for tier '{tier}' step {step_num} expected dict, "
                    f"got {type(window_data).__name__}. Checkpoint may be corrupted."
                )
            # C2-1: Use current quality_window_size when restoring deques
            maxlen = gs.quality_window_size
            values = window_data.get("values", [])
            # C2-3: Filter out NaN/Inf values when restoring
            import math
            filtered_values = [v for v in values if math.isfinite(v)]
            if len(filtered_values) < len(values):
                import warnings
                warnings.warn(
                    f"C2-3: Filtered {len(values) - len(filtered_values)} NaN/Inf values "
                    f"from deque for tier '{tier}' step {step_num}"
                )
            # CP3-006: Wrap int() in try-except with informative error
            try:
                step_key = int(step_num)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"CP3-006: Cannot convert step_num '{step_num}' to int for tier '{tier}'. "
                    f"Checkpoint may be corrupted. Error: {e}"
                ) from e
            gs.tier_mastery[tier][step_key] = deque(filtered_values, maxlen=maxlen)

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
    accumulated_loss: float = 0.0,
    dataloader_state: dict | None = None,
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
        "accumulated_loss": accumulated_loss,
        "dataloader_state": dataloader_state,
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename to avoid partial writes
    temp_path = path.with_suffix('.tmp')
    try:
        # Delete stale temp file if exists
        temp_path.unlink(missing_ok=True)
        torch.save(checkpoint, temp_path)
        # Atomic rename
        temp_path.replace(path)
    except Exception as e:
        # Clean up temp file on failure
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError(f"Failed to save checkpoint to {path}: {e}") from e


def load_checkpoint(path: str | Path) -> dict:
    """Load checkpoint from file. Returns raw dict — caller restores state.

    Attempts to load from the specified path. If loading fails (corrupted checkpoint),
    tries loading from the previous checkpoint in the directory.
    """
    import torch
    import warnings
    from pathlib import Path

    path = Path(path)
    try:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        # Validate required keys (but allow None values — trainer will check)
        required_keys = ["global_step"]
        for key in required_keys:
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required key: {key}")
        if not isinstance(checkpoint["global_step"], int):
            raise ValueError(f"Invalid type for global_step: {type(checkpoint['global_step'])}")

        # CP2-002: Wrap gamestate_from_dict in try-except with informative error
        if checkpoint.get("game_state") is not None:
            try:
                checkpoint["game_state"] = gamestate_from_dict(checkpoint["game_state"])
            except Exception as e:
                raise RuntimeError(
                    f"Failed to restore game_state from checkpoint {path}. "
                    f"Checkpoint may be corrupted or from incompatible version. "
                    f"Original error: {e}"
                ) from e
        return checkpoint
    except Exception as e:
        warnings.warn(
            f"Failed to load checkpoint from {path}: {e}. "
            "Attempting to load previous checkpoint..."
        )
        # Try to find previous checkpoint
        checkpoint_dir = Path(path).parent
        import re
        pattern = re.compile(r"global_step_(\d+)")
        match = pattern.search(Path(path).name)
        if match:
            current_step = int(match.group(1))
            # Look for any checkpoint with lower step number
            candidates = []
            for entry in checkpoint_dir.iterdir():
                if entry.exists() and entry.is_file():
                    m = pattern.search(entry.name)
                    if m and int(m.group(1)) < current_step:
                        candidates.append((int(m.group(1)), entry))
            if candidates:
                prev_path = max(candidates, key=lambda x: x[0])[1]
                warnings.warn(f"Loading previous checkpoint: {prev_path}")
                checkpoint = torch.load(prev_path, map_location="cpu", weights_only=False)
                # C2-4: Validate fallback checkpoint
                required_keys = ["global_step"]
                for key in required_keys:
                    if key not in checkpoint:
                        raise ValueError(f"Fallback checkpoint missing required key: {key}")
                if not isinstance(checkpoint["global_step"], int):
                    raise ValueError(f"Invalid type for global_step in fallback: {type(checkpoint['global_step'])}")
                if checkpoint.get("game_state") is not None:
                    try:
                        checkpoint["game_state"] = gamestate_from_dict(checkpoint["game_state"])
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to restore game_state from fallback checkpoint {prev_path}. "
                            f"Original error: {e}"
                        ) from e
                return checkpoint
        # No previous checkpoint found — re-raise original error
        raise RuntimeError(f"Checkpoint load failed and no previous checkpoint found: {e}") from e


def discover_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Scan directory for global_step_N checkpoints, return path to latest."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    pattern = re.compile(r"global_step_(\d+)")
    candidates = []
    for entry in checkpoint_dir.iterdir():
        if entry.is_file():
            match = pattern.search(entry.name)
            if match:
                candidates.append((int(match.group(1)), entry))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]
