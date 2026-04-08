from __future__ import annotations

import re
import warnings
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import torch

from qgre.types import (
    CHECKPOINT_SCHEMA_VERSION,
    AdvantageEstimatorState,
    CheckpointState,
    DataLoaderState,
    GameState,
    TrainerState,
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

    Uses schema validation to replace scattered isinstance checks with a single
    validation pass. Restores: list → deque (with maxlen). Handles both old 1D
    and new 2D format.
    """

    from qgre.schema import GAME_STATE_SCHEMA, validate_schema

    # Schema validation: one pass, all type checks
    validated = validate_schema(d, GAME_STATE_SCHEMA, "game_state")

    gs = GameState()
    gs.step_count = validated["step_count"]
    gs.mastery_threshold = validated["mastery_threshold"]
    gs.phase_history = list(validated["phase_history"])
    gs.stagnation_timeout = validated["stagnation_timeout"]
    gs.plateau_window = validated["plateau_window"]
    gs.plateau_threshold = validated["plateau_threshold"]
    gs.quality_window_size = validated["quality_window_size"]

    # tier_phases with int coercion (schema validates dict, we coerce values)
    # Handle migration: if tier_phases is empty/None but "phase" exists (old format), use phase
    tier_phases_raw = validated["tier_phases"]
    if not tier_phases_raw:  # None or empty dict
        tier_phases_raw = {"default": validated.get("phase", 1)}
    try:
        gs.tier_phases = {k: int(v) for k, v in tier_phases_raw.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid tier_phases values (expected int-coercible, got non-numeric): {tier_phases_raw}. Error: {e}"
        ) from e
    gs.active_tiers = validated["active_tiers"]
    tier_steps_raw = (
        validated["tier_steps_at_phase_start"]
        if validated["tier_steps_at_phase_start"] is not None
        else {}
    )
    try:
        gs.tier_steps_at_phase_start = {k: int(v) for k, v in tier_steps_raw.items()}
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid tier_steps_at_phase_start values (expected int-coercible, got non-numeric): {tier_steps_raw}. Error: {e}"
        ) from e

    # Initialize missing tier_steps_at_phase_start entries for active_tiers
    for tier in gs.active_tiers:
        if tier not in gs.tier_steps_at_phase_start:
            gs.tier_steps_at_phase_start[tier] = gs.step_count

    # Restore tier_mastery from serialized format with NaN filtering
    tm = validated["tier_mastery"] or {}
    gs.tier_mastery = _restore_tier_mastery(tm, gs.quality_window_size)

    # Backward compat: migrate old 1D step_mastery to "default" tier
    step_mastery_raw = validated["step_mastery"]
    if step_mastery_raw is not None and not tm:
        gs.tier_mastery["default"] = _restore_step_mastery(
            step_mastery_raw,
            gs.quality_window_size,
        )

    # Initialize missing tier_mastery entries for active_tiers
    for tier in gs.active_tiers:
        if tier not in gs.tier_mastery:
            gs.tier_mastery[tier] = {}
        if tier not in gs.tier_phases:
            gs.tier_phases[tier] = 1

    return gs


def _restore_tier_mastery(tm: dict, quality_window_size: int) -> dict:
    """Restore tier_mastery dict with deques, filtering NaN values."""
    import math

    result = {}
    for tier, step_windows in tm.items():
        if not isinstance(step_windows, dict):
            raise TypeError(
                f"SCHEMA: tier_mastery['{tier}'] expected dict, got {type(step_windows).__name__}",
            )
        result[tier] = {}
        for step_num, window_data in step_windows.items():
            if not isinstance(window_data, dict):
                raise TypeError(
                    f"SCHEMA: tier_mastery['{tier}'][{step_num}] expected dict, "
                    f"got {type(window_data).__name__}",
                )
            maxlen = quality_window_size
            values = window_data.get("values", [])
            ckpt_maxlen = window_data.get("maxlen")
            if ckpt_maxlen is not None and ckpt_maxlen != maxlen:
                values_len = len(values)
                data_loss = max(0, values_len - maxlen) if values_len > maxlen else 0
                warnings.warn(
                    f"SCHEMA: Checkpoint maxlen={ckpt_maxlen} differs from config "
                    f"quality_window_size={maxlen} for tier '{tier}' step {step_num}. "
                    f"Data loss: {data_loss} entries.",
                    stacklevel=2,
                )
            # CK2: Coerce values to float before isfinite check
            try:
                coerced_values = [float(v) for v in values]
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"CK2: Cannot convert tier_mastery['{tier}'][{step_num}] values to float: {e}. "
                    f"Got values: {values[:5]}...",
                ) from e
            # Filter NaN/Inf
            filtered = [v for v in coerced_values if math.isfinite(v)]
            if len(filtered) < len(coerced_values):
                warnings.warn(
                    f"SCHEMA: Filtered {len(coerced_values) - len(filtered)} NaN/Inf values "
                    f"from tier_mastery['{tier}'][{step_num}]",
                    stacklevel=2,
                )
            try:
                step_key = int(step_num)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"SCHEMA: Cannot convert step_num '{step_num}' to int for tier '{tier}': {e}",
                ) from e
            # CK1: Warn about truncation when window_size decreased
            if len(filtered) > maxlen:
                dropped_count = len(filtered) - maxlen
                warnings.warn(
                    f"CK1: quality_window_size decreased for tier '{tier}' step {step_key}. "
                    f"Truncating {dropped_count} oldest scores (checkpoint had {len(filtered)}, config allows {maxlen}).",
                    stacklevel=2,
                )
            result[tier][step_key] = deque(filtered, maxlen=maxlen)
    return result


def _restore_step_mastery(step_mastery: dict, quality_window_size: int) -> dict:
    """Restore old format step_mastery to deques."""
    import math

    if not isinstance(step_mastery, dict):
        raise TypeError(
            f"SCHEMA: step_mastery expected dict, got {type(step_mastery).__name__}",
        )
    result = {}
    for step_num, window_data in step_mastery.items():
        if not isinstance(window_data, dict):
            raise TypeError(
                f"SCHEMA: step_mastery[{step_num}] expected dict, got {type(window_data).__name__}",
            )
        maxlen = window_data.get("maxlen", quality_window_size)
        values = window_data.get("values", [])
        # CK2: Coerce values to float before isfinite check
        try:
            coerced_values = [float(v) for v in values]
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"CK2: Cannot convert step_mastery[{step_num}] values to float: {e}. "
                f"Got values: {values[:5]}...",
            ) from e
        filtered = [v for v in coerced_values if math.isfinite(v)]
        if len(filtered) < len(coerced_values):
            warnings.warn(
                f"SCHEMA: Filtered {len(coerced_values) - len(filtered)} NaN/Inf values "
                f"from step_mastery[{step_num}]",
                stacklevel=2,
            )
        # CK3: Use quality_window_size consistently, not checkpoint's maxlen
        if maxlen != quality_window_size:
            warnings.warn(
                f"CK3: step_mastery[{step_num}] checkpoint maxlen ({maxlen}) != config quality_window_size ({quality_window_size}). "
                f"Using config value ({quality_window_size}).",
                stacklevel=2,
            )
        result[int(step_num)] = deque(filtered, maxlen=quality_window_size)
    return result


def save_checkpoint(
    path: str | Path,
    global_step: int,
    model_state_dict: dict | None = None,
    optimizer_state_dict: dict | None = None,
    scheduler_state_dict: dict | None = None,
    game_state: GameState | None = None,
    advantage_estimator_state: dict | None = None,
    rng_state: tuple | None = None,
    cuda_rng_state: torch.Tensor | None = None,
    vprm_critic_state: dict | None = None,
    vprm_optimizer_state: dict | None = None,
    accumulated_loss: float = 0.0,
    accumulated_samples: int = 0,
    accumulation_count: int = 0,
    dataloader_state: dict | None = None,
    training_context: dict | None = None,
    hint_registry_state: dict | None = None,
    lora_pro_state: dict | None = None,
    # New: accept TrainerState directly (preferred)
    trainer_state: TrainerState | None = None,
    weight_loader_state: WeightLoaderState | None = None,
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

    # Build DataLoaderState from dict or default (uses schema validation + NaN filtering)
    dataloader = (
        DataLoaderState.from_dict(dataloader_state) if dataloader_state else DataLoaderState()
    )

    # Build AdvantageEstimatorState — wraps the full state_dict
    advantage_estimator = AdvantageEstimatorState(state_dict=advantage_estimator_state)

    # Build WeightLoaderState — use provided state if available, otherwise default
    weight_loader = (
        weight_loader_state
        if weight_loader_state is not None
        else WeightLoaderState(
            load_lora_called=False,
            initialized=False,
            cleaned_up=False,
        )
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
        lora_pro_state=lora_pro_state,
        training_context=training_context,
        schema_version=CHECKPOINT_SCHEMA_VERSION,
    )

    # Serialize CheckpointState
    checkpoint = asdict(checkpoint_state)

    # C1: Always convert game_state using gamestate_to_dict (asdict doesn't handle deques)
    # Use checkpoint_state.game_state (the assigned value) not the parameter
    checkpoint["game_state"] = gamestate_to_dict(checkpoint_state.game_state)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic write: write to temp file, then rename to avoid partial writes
    temp_path = path.with_suffix(".tmp")
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
        except OSError as cleanup_err:
            warnings.warn(
                f"Failed to clean up temp file {temp_path} after checkpoint save error: {cleanup_err}",
                stacklevel=2,
            )
        raise RuntimeError(f"Failed to save checkpoint to {path}: {e}") from e


def load_checkpoint(path: str | Path) -> CheckpointState:
    """Load checkpoint from file. Returns CheckpointState with validated structure.

    Handles both old (flat) and new (StateSpec) checkpoint formats via migration.
    Attempts to load from the specified path. If loading fails (corrupted checkpoint),
    tries loading from the previous checkpoint in the directory.
    """
    import warnings
    from pathlib import Path

    import torch

    path = Path(path)
    try:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        # weights_only=False required for optimizer/RNG state.
        # Safe: checkpoints are local-only, never downloaded from untrusted sources.
        raw_checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # nosec B614

        # FIX 11: Validate raw_checkpoint is a dict
        if not isinstance(raw_checkpoint, dict):
            raise TypeError(
                f"Checkpoint file {path} contains a {type(raw_checkpoint).__name__}, "
                "not a dict. File is corrupted or was not saved by qgre.save_checkpoint."
            )

        # C06-SCHEMA: Validate schema_version exists
        checkpoint_schema = raw_checkpoint.get("schema_version")
        if checkpoint_schema is None:
            warnings.warn(
                f"Checkpoint {path} missing schema_version. "
                f"Assuming old format, will migrate to StateSpec.",
                UserWarning,
                stacklevel=2,
            )
            raw_checkpoint["schema_version"] = 1  # Old format

        # C07-TYPE: Explicit type coercion with validation
        try:
            checkpoint_schema = int(raw_checkpoint.get("schema_version", 1))
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Checkpoint {path} has invalid schema_version type: {type(checkpoint_schema).__name__}. "
                f"Expected int. Cannot safely restore checkpoint. Error: {e}",
            ) from e

        # Schema version mismatch is now a warning, not an error (migration handles it)
        if checkpoint_schema != CHECKPOINT_SCHEMA_VERSION:
            warnings.warn(
                f"Checkpoint schema version mismatch: checkpoint has version {checkpoint_schema}, "
                f"current code expects version {CHECKPOINT_SCHEMA_VERSION}. "
                f"Migration will be attempted.",
                UserWarning,
                stacklevel=2,
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
                    f"Expected int. Error: {e}",
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
                        f"Original error: {e}",
                    ) from e

        # Convert to CheckpointState (handles migration from old format)
        return CheckpointState.from_dict(raw_checkpoint)
    except (FileNotFoundError, ValueError, TypeError, RuntimeError, KeyError, EOFError) as e:
        # Catch checkpoint-specific errors only — let system errors (MemoryError,
        # KeyboardInterrupt, SystemExit, ImportError) propagate
        warnings.warn(
            f"CHECKPOINT CORRUPTION DETECTED: Failed to load checkpoint from {path}: {e}. "
            "Falling back to previous checkpoint. This may result in data loss (lost training progress). "
            "Attempting to load previous checkpoint...",
            stacklevel=2,
        )
        # Try to find previous checkpoint
        checkpoint_dir = Path(path).parent
        import re

        pattern = re.compile(r"global_step_(\d+)\.pt$")
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
                warnings.warn(
                    f"FALLBACK CHECKPOINT LOADED: {prev_path}. "
                    f"Original checkpoint {path} was corrupted. Training progress may be lost.",
                    stacklevel=2,
                )
                # See comment above about weights_only=False
                raw_checkpoint = torch.load(prev_path, map_location="cpu", weights_only=False)  # nosec B614

                # FIX: Validate fallback checkpoint is also a dict (same check as primary)
                if not isinstance(raw_checkpoint, dict):
                    raise TypeError(
                        f"Fallback checkpoint file {prev_path} contains a {type(raw_checkpoint).__name__}, "
                        "not a dict. File is corrupted or was not saved by qgre.save_checkpoint."
                    )

                # C06-SCHEMA: Handle missing schema_version (old format)
                checkpoint_schema = raw_checkpoint.get("schema_version")
                if checkpoint_schema is None:
                    warnings.warn(
                        f"Fallback checkpoint {prev_path} missing schema_version. "
                        f"Assuming old format, will migrate.",
                        UserWarning,
                        stacklevel=2,
                    )
                    raw_checkpoint["schema_version"] = 1

                # C07-TYPE: Explicit type coercion with validation
                try:
                    checkpoint_schema = int(raw_checkpoint.get("schema_version", 1))
                except (ValueError, TypeError) as e:
                    raise TypeError(
                        f"Fallback checkpoint {prev_path} has invalid schema_version type: {type(checkpoint_schema).__name__}. "
                        f"Expected int. Cannot safely restore checkpoint. Error: {e}",
                    ) from e

                # Schema mismatch is warning, not error (migration handles it)
                if checkpoint_schema != CHECKPOINT_SCHEMA_VERSION:
                    warnings.warn(
                        f"Fallback checkpoint schema version mismatch: checkpoint has version {checkpoint_schema}, "
                        f"current code expects version {CHECKPOINT_SCHEMA_VERSION}. "
                        f"Migration will be attempted.",
                        UserWarning,
                        stacklevel=2,
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
                            f"Expected int. Error: {e}",
                        ) from e

                if raw_checkpoint.get("game_state") is not None:
                    game_state_raw = raw_checkpoint["game_state"]
                    if isinstance(game_state_raw, dict):
                        try:
                            raw_checkpoint["game_state"] = gamestate_from_dict(game_state_raw)
                        except Exception as e:
                            raise RuntimeError(
                                f"Failed to restore game_state from fallback checkpoint {prev_path}. "
                                f"Original error: {e}",
                            ) from e

                # Convert to CheckpointState (handles migration)
                return CheckpointState.from_dict(raw_checkpoint)
        # No previous checkpoint found — re-raise with context
        raise RuntimeError(
            f"Checkpoint load failed for {path} and no previous checkpoint found: {e}",
        ) from e


def discover_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    """Scan directory for global_step_N checkpoints, return path to latest."""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    pattern = re.compile(r"^global_step_(\d+)\.pt$")
    candidates = []
    for entry in checkpoint_dir.iterdir():
        if entry.is_file() and entry.suffix == ".pt":
            match = pattern.match(entry.name)
            if match:
                candidates.append((int(match.group(1)), entry))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]
