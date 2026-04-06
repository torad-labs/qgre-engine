from __future__ import annotations

import re
from collections import defaultdict, deque
from pathlib import Path

from dataclasses import asdict

from qgre.types import (
    CHECKPOINT_SCHEMA_VERSION,
    QUALITY_WINDOW_SIZE,
    GameState,
    CheckpointState,
    TrainerState,
    DataLoaderState,
    AdvantageEstimatorState,
    WeightLoaderState,
)


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

    # DP-R3-01: Initialize missing tier_steps_at_phase_start entries for active_tiers
    for tier in gs.active_tiers:
        if tier not in gs.tier_steps_at_phase_start:
            gs.tier_steps_at_phase_start[tier] = gs.step_count

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
            # CSM-006: Warn when checkpoint maxlen differs from config quality_window_size
            ckpt_maxlen = window_data.get("maxlen")
            if ckpt_maxlen is not None and ckpt_maxlen != maxlen:
                import warnings
                warnings.warn(
                    f"CSM-006: Checkpoint maxlen={ckpt_maxlen} differs from config quality_window_size={maxlen} "
                    f"for tier '{tier}' step {step_num}. Using config value (checkpoint ignored)."
                )
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
    accumulated_samples: int = 0,
    accumulation_count: int = 0,
    dataloader_state: dict | None = None,
    training_context: dict | None = None,
    hint_registry_state: dict | None = None,
    # New: accept TrainerState directly (preferred)
    trainer_state: TrainerState | None = None,
):
    """Save full training state to a checkpoint file.

    Builds CheckpointState from parameters and serializes via asdict().
    Supports both legacy parameter interface and new TrainerState interface.
    """
    import torch

    # Build TrainerState from parameters or use provided one
    if trainer_state is not None:
        trainer = trainer_state
    else:
        trainer = TrainerState(
            global_step=global_step,
            accumulated_loss=accumulated_loss,
            accumulation_count=accumulation_count,
            accumulated_samples=accumulated_samples,
            resumed_mid_accumulation=False,
            fused_validated=False,
            needs_weight_sync=False,
            rng_state=rng_state if rng_state is not None else torch.get_rng_state(),
            cuda_rng_state=cuda_rng_state,
        )

    # Build DataLoaderState from dict or default
    if dataloader_state is not None:
        dataloader = DataLoaderState(
            epoch=dataloader_state.get("epoch", 0),
            step_in_epoch=dataloader_state.get("step_in_epoch", 0),
            total_steps=dataloader_state.get("total_steps", 0),
            priority_weights=dataloader_state.get("priority_weights"),
            difficulty_gate=dataloader_state.get("difficulty_gate"),
        )
    else:
        dataloader = DataLoaderState()

    # Build AdvantageEstimatorState — wraps the full state_dict
    advantage_estimator = AdvantageEstimatorState(state_dict=advantage_estimator_state)

    # Build WeightLoaderState (VPRM states go here)
    weight_loader = WeightLoaderState(
        load_lora_called=False,
        initialized=False,
        cleaned_up=False,
    )

    # Build CheckpointState
    checkpoint_state = CheckpointState(
        trainer=trainer,
        dataloader=dataloader,
        advantage_estimator=advantage_estimator,
        weight_loader=weight_loader,
        game_state=game_state if game_state is not None else GameState(),
        model_state_dict=model_state_dict,
        optimizer_state_dict=optimizer_state_dict,
        scheduler_state_dict=scheduler_state_dict,
        vprm_critic_state=vprm_critic_state,
        vprm_optimizer_state=vprm_optimizer_state,
        hint_registry_state=hint_registry_state,
        training_context=training_context,
        schema_version=CHECKPOINT_SCHEMA_VERSION,
    )

    # Serialize CheckpointState
    checkpoint = asdict(checkpoint_state)

    # Special handling for GameState: convert deques to serializable format
    if game_state is not None:
        checkpoint["game_state"] = gamestate_to_dict(game_state)

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
        except Exception as cleanup_err:
            warnings.warn(
                f"Failed to clean up temp file {temp_path} after checkpoint save error: {cleanup_err}"
            )
        raise RuntimeError(f"Failed to save checkpoint to {path}: {e}") from e


def load_checkpoint(path: str | Path) -> CheckpointState:
    """Load checkpoint from file. Returns CheckpointState with validated structure.

    Handles both old (flat) and new (StateSpec) checkpoint formats via migration.
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
        raw_checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # C06-SCHEMA: Validate schema_version exists
        checkpoint_schema = raw_checkpoint.get("schema_version")
        if checkpoint_schema is None:
            warnings.warn(
                f"Checkpoint {path} missing schema_version. "
                f"Assuming old format, will migrate to StateSpec.",
                UserWarning
            )
            raw_checkpoint["schema_version"] = 1  # Old format

        # C07-TYPE: Explicit type coercion with validation
        try:
            checkpoint_schema = int(raw_checkpoint.get("schema_version", 1))
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Checkpoint {path} has invalid schema_version type: {type(checkpoint_schema).__name__}. "
                f"Expected int. Cannot safely restore checkpoint. Error: {e}"
            ) from e

        # Schema version mismatch is now a warning, not an error (migration handles it)
        if checkpoint_schema != CHECKPOINT_SCHEMA_VERSION:
            warnings.warn(
                f"Checkpoint schema version mismatch: checkpoint has version {checkpoint_schema}, "
                f"current code expects version {CHECKPOINT_SCHEMA_VERSION}. "
                f"Migration will be attempted.",
                UserWarning
            )

        # Validate required keys for old format
        # New format has "trainer" key; old format has "global_step" at top level
        is_old_format = "trainer" not in raw_checkpoint
        if is_old_format:
            required_keys = ["global_step"]
            for key in required_keys:
                if key not in raw_checkpoint:
                    raise ValueError(f"Checkpoint missing required key: {key}")

            # C07-TYPE: Explicit type coercion for global_step
            try:
                raw_checkpoint["global_step"] = int(raw_checkpoint["global_step"])
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Checkpoint {path} has invalid global_step type: {type(raw_checkpoint['global_step']).__name__}. "
                    f"Expected int. Error: {e}"
                ) from e

        # CP2-002: Convert game_state dict to GameState before CheckpointState.from_dict()
        if raw_checkpoint.get("game_state") is not None:
            game_state_raw = raw_checkpoint["game_state"]
            if isinstance(game_state_raw, dict):
                try:
                    raw_checkpoint["game_state"] = gamestate_from_dict(game_state_raw)
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to restore game_state from checkpoint {path}. "
                        f"Checkpoint may be corrupted or from incompatible version. "
                        f"Original error: {e}"
                    ) from e

        # Convert to CheckpointState (handles migration from old format)
        checkpoint_state = CheckpointState.from_dict(raw_checkpoint)
        return checkpoint_state
    except (FileNotFoundError, ValueError, TypeError, RuntimeError, KeyError, EOFError) as e:
        # Catch checkpoint-specific errors only — let system errors (MemoryError,
        # KeyboardInterrupt, SystemExit, ImportError) propagate
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
                raw_checkpoint = torch.load(prev_path, map_location="cpu", weights_only=False)

                # C06-SCHEMA: Handle missing schema_version (old format)
                checkpoint_schema = raw_checkpoint.get("schema_version")
                if checkpoint_schema is None:
                    warnings.warn(
                        f"Fallback checkpoint {prev_path} missing schema_version. "
                        f"Assuming old format, will migrate.",
                        UserWarning
                    )
                    raw_checkpoint["schema_version"] = 1

                # C07-TYPE: Explicit type coercion with validation
                try:
                    checkpoint_schema = int(raw_checkpoint.get("schema_version", 1))
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Fallback checkpoint {prev_path} has invalid schema_version type: {type(checkpoint_schema).__name__}. "
                        f"Expected int. Cannot safely restore checkpoint. Error: {e}"
                    ) from e

                # Schema mismatch is warning, not error (migration handles it)
                if checkpoint_schema != CHECKPOINT_SCHEMA_VERSION:
                    warnings.warn(
                        f"Fallback checkpoint schema version mismatch: checkpoint has version {checkpoint_schema}, "
                        f"current code expects version {CHECKPOINT_SCHEMA_VERSION}. "
                        f"Migration will be attempted.",
                        UserWarning
                    )

                # C2-4: Validate fallback checkpoint (old format)
                is_old_format = "trainer" not in raw_checkpoint
                if is_old_format:
                    required_keys = ["global_step"]
                    for key in required_keys:
                        if key not in raw_checkpoint:
                            raise ValueError(f"Fallback checkpoint missing required key: {key}")

                    # C07-TYPE: Explicit type coercion for global_step
                    try:
                        raw_checkpoint["global_step"] = int(raw_checkpoint["global_step"])
                    except (ValueError, TypeError) as e:
                        raise TypeError(
                            f"Fallback checkpoint {prev_path} has invalid global_step type: {type(raw_checkpoint['global_step']).__name__}. "
                            f"Expected int. Error: {e}"
                        ) from e

                if raw_checkpoint.get("game_state") is not None:
                    game_state_raw = raw_checkpoint["game_state"]
                    if isinstance(game_state_raw, dict):
                        try:
                            raw_checkpoint["game_state"] = gamestate_from_dict(game_state_raw)
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to restore game_state from fallback checkpoint {prev_path}. "
                                f"Original error: {e}"
                            ) from e

                # Convert to CheckpointState (handles migration)
                checkpoint_state = CheckpointState.from_dict(raw_checkpoint)
                return checkpoint_state
        # No previous checkpoint found — re-raise with context
        raise RuntimeError(
            f"Checkpoint load failed for {path} and no previous checkpoint found: {e}"
        ) from e


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
