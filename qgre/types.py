"""Core types for QGRE Engine."""

from __future__ import annotations

import logging
import random
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

import torch


logger = logging.getLogger(__name__)


CHECKPOINT_SCHEMA_VERSION: int = 1


@dataclass
class TrainingContext:
    """Training context — device, dtype, step counter, and checkpoint schema version.

    Passed explicitly through the pipeline (no thread-local/global state).
    Provides device/dtype for tensor construction and tracks training progress.
    """

    device: torch.device
    dtype: torch.dtype = torch.float32
    step: int = 0
    checkpoint_schema_version: int = CHECKPOINT_SCHEMA_VERSION

    @classmethod
    def from_config(cls, _config, device: str = "cuda") -> TrainingContext:
        """Factory method to construct TrainingContext from QGREConfig.

        Args:
            config: QGREConfig instance (unused in v1, reserved for future dtype config)
            device: Device string ("cuda", "cpu", etc.)

        Returns:
            TrainingContext instance

        Raises:
            ValueError: If device string is invalid or unavailable
        """
        try:
            device_obj = torch.device(device)
        except RuntimeError as e:
            raise ValueError(f"Invalid device string '{device}': {e}") from e

        return cls(
            device=device_obj,
            dtype=torch.float32,
            step=0,
            checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        )

    def to_dict(self) -> dict:
        """Serialize to dict. Device is converted to string for JSON compatibility."""
        return {
            "device": str(self.device),
            "dtype": str(self.dtype),
            "step": self.step,
            "checkpoint_schema_version": self.checkpoint_schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingContext:
        """Deserialize from dict. Device string is converted back to torch.device.

        Raises:
            ValueError: If required keys are missing or device/dtype strings are invalid
        """
        # Validate required keys
        required_keys = ["device", "dtype", "step"]
        missing_keys = [key for key in required_keys if key not in d]
        if missing_keys:
            raise ValueError(f"Missing required keys in TrainingContext dict: {missing_keys}")

        # Parse dtype string (e.g. "torch.float32" → torch.float32)
        dtype_str = d["dtype"]
        if dtype_str.startswith("torch."):
            try:
                dtype = getattr(torch, dtype_str.split(".", 1)[1])
            except AttributeError as e:
                raise ValueError(f"Invalid dtype string '{dtype_str}': {e}") from e
        else:
            raise ValueError(f"Invalid dtype string '{dtype_str}': expected 'torch.<dtype>' format")

        # Validate device string
        try:
            device = torch.device(d["device"])
        except RuntimeError as e:
            raise ValueError(f"Invalid device string '{d['device']}': {e}") from e

        return cls(
            device=device,
            dtype=dtype,
            step=d["step"],
            checkpoint_schema_version=d.get("checkpoint_schema_version", CHECKPOINT_SCHEMA_VERSION),
        )


# ── StateSpec Dataclasses ──


@dataclass
class TrainerState:
    """All mutable trainer state in one place.

    Tracks accumulation progress, resumption flags, sync requirements,
    and RNG state for reproducibility.
    Serializable checkpoint component.
    """

    global_step: int = 0
    accumulated_loss: float = 0.0
    accumulation_count: int = 0
    accumulated_samples: int = 0
    resumed_mid_accumulation: bool = False
    fused_validated: bool = False
    needs_weight_sync: bool = False
    # RNG state for reproducibility across resume
    rng_state: object | None = None  # torch.get_rng_state() output
    cuda_rng_state: object | None = None  # torch.cuda.get_rng_state() output
    # MLflow run persistence for cross-resume continuity (R3-T1)
    mlflow_run_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> TrainerState:
        """Create TrainerState from dict with schema validation.

        This is the ONLY path for reconstructing TrainerState from checkpoint.
        Validates all fields, applies defaults, and raises on missing required fields.
        """
        from qgre.schema import TRAINER_STATE_SCHEMA, validate_schema

        validated = validate_schema(d, TRAINER_STATE_SCHEMA, "trainer")
        return cls(**validated)


@dataclass
class DataLoaderState:
    """Epoch tracking, sampling weights, curriculum gating.

    Manages iteration state and dynamic priority scheduling.
    Serializable checkpoint component.
    """

    epoch: int = 0
    step_in_epoch: int = 0
    total_steps: int = 0
    priority_weights: list[float] | dict[str, float] | None = None
    difficulty_gate: tuple[set[str], str] | None = None

    @classmethod
    def from_dict(cls, d: dict) -> DataLoaderState:
        """Create DataLoaderState from dict with schema validation.

        Schema handles NaN filtering via filter_nan=True on priority_weights.
        Uses convert_difficulty_gate utility for dict→tuple migration.
        """
        from qgre.schema import (
            DATALOADER_STATE_SCHEMA,
            convert_difficulty_gate,
            validate_schema,
        )

        validated = validate_schema(d, DATALOADER_STATE_SCHEMA, "dataloader")

        return cls(
            epoch=validated["epoch"],
            step_in_epoch=validated["step_in_epoch"],
            total_steps=validated["total_steps"],
            priority_weights=validated.get("priority_weights"),
            difficulty_gate=convert_difficulty_gate(validated.get("difficulty_gate")),
        )


@dataclass
class AdvantageEstimatorState:
    """Estimator config and accumulated statistics.

    Wraps the advantage estimator's full state_dict for checkpoint serialization.
    The actual state structure is managed by QGREStepAdvantageEstimator.
    """

    # Full state dict from QGREStepAdvantageEstimator.state_dict()
    # Contains: V, V_last_seen, quality_seen, step_seen, reward_var, reward_mean, etc.
    state_dict: dict | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> AdvantageEstimatorState:
        """Create AdvantageEstimatorState from dict with schema validation."""
        from qgre.schema import ADVANTAGE_ESTIMATOR_STATE_SCHEMA, validate_schema

        if d is None:
            return cls(state_dict=None)
        validated = validate_schema(d, ADVANTAGE_ESTIMATOR_STATE_SCHEMA, "advantage_estimator")
        return cls(**validated)


class WeightLoaderLifecycle(Enum):
    """State machine for WeightLoader lifecycle (legacy serialization enum).

    Used by WeightLoaderState (the on-disk checkpoint dataclass) for backwards
    compatibility with older checkpoints. The live WeightLoader now reads its
    lifecycle from the injected SyncState (which uses qgre.sync_state.SyncLifecycle);
    this string-valued enum exists only to keep checkpoint round-trips stable.

    Tracks weight loading states. Does NOT track dropout state — that lives in
    SyncState alongside lifecycle, cache_stale, and initialized, and is managed
    via the dropout_context() context manager in lora_dropout.apply_lora_dropout.

    Valid transitions:
        UNINITIALIZED -> LOADING       (first sync_lora_direct call)
        LOADING -> READY               (prepare_vllm_lora_loading succeeds)
        LOADING -> ERROR               (prepare fails)
        READY -> UNINITIALIZED         (reset_state for engine recreate)
        ERROR -> LOADING               (retry after failure)
        ERROR -> UNINITIALIZED         (explicit recovery)
        any -> ERROR                   (exception during operation)
    """

    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"


@dataclass
class WeightLoaderState:
    """LoRA initialization flags and cleanup tracking.

    Prevents double-initialization and tracks resource lifecycle.
    Serializable checkpoint component.

    Note: lifecycle field is the source of truth. Boolean fields are kept
    for backward compatibility with existing checkpoints and computed from lifecycle.
    """

    # Legacy boolean fields (for checkpoint backward compatibility)
    load_lora_called: bool = False
    initialized: bool = False
    cleaned_up: bool = False
    # W1: Store lora_request_id to detect re-registration on resume
    lora_request_id: int | None = None
    # New: lifecycle enum as string (source of truth)
    lifecycle: str = "uninitialized"
    # SyncState.restore_failed — sticky safety flag set when a LoRA dropout
    # restore() failed mid-run. MUST round-trip through checkpoints so a
    # corrupted-weights state cannot silently re-enter dropout after restart.
    # Default False for back-compat with older checkpoints.
    restore_failed: bool = False

    def __post_init__(self):
        """Sync legacy fields from lifecycle if lifecycle is explicitly set."""
        # ELI-001: Migrate old "dropout_active" checkpoints to "ready"
        # (dropout was removed from state machine - it's tracked externally)
        if self.lifecycle == "dropout_active":
            self.lifecycle = "ready"

        # If lifecycle was explicitly set to non-default, update legacy fields
        if self.lifecycle != "uninitialized":
            # SFH-004: Validate lifecycle value with actionable error message
            try:
                lc = WeightLoaderLifecycle(self.lifecycle)
            except ValueError as e:
                valid_values = [v.value for v in WeightLoaderLifecycle]
                raise ValueError(
                    f"Invalid lifecycle value '{self.lifecycle}' in WeightLoaderState. "
                    f"Valid values: {valid_values}. "
                    "This may indicate checkpoint corruption or schema mismatch.",
                ) from e
            expected_initialized = lc == WeightLoaderLifecycle.READY
            expected_load_lora_called = lc in (
                WeightLoaderLifecycle.LOADING,
                WeightLoaderLifecycle.READY,
                WeightLoaderLifecycle.ERROR,
            )
            if self.initialized != expected_initialized:
                warnings.warn(
                    f"WeightLoaderState inconsistency: lifecycle={self.lifecycle} implies initialized={expected_initialized}, "
                    f"but checkpoint has initialized={self.initialized}. Using lifecycle value.",
                    stacklevel=2,
                )
            if self.load_lora_called != expected_load_lora_called:
                warnings.warn(
                    f"WeightLoaderState inconsistency: lifecycle={self.lifecycle} implies load_lora_called={expected_load_lora_called}, "
                    f"but checkpoint has load_lora_called={self.load_lora_called}. Using lifecycle value.",
                    stacklevel=2,
                )
            # ELI-001: initialized = True only when READY (ERROR means failed)
            self.initialized = expected_initialized
            # load_lora_called = True when we've attempted loading (includes ERROR)
            self.load_lora_called = expected_load_lora_called
        # If legacy fields set but lifecycle is default, infer lifecycle
        elif self.initialized:
            self.lifecycle = WeightLoaderLifecycle.READY.value
        elif self.load_lora_called:
            self.lifecycle = WeightLoaderLifecycle.LOADING.value

    def get_lifecycle(self) -> WeightLoaderLifecycle:
        """Get current lifecycle state as enum."""
        return WeightLoaderLifecycle(self.lifecycle)

    @classmethod
    def from_lifecycle(cls, lifecycle: WeightLoaderLifecycle) -> WeightLoaderState:
        """Create state from enum value."""
        return cls(lifecycle=lifecycle.value)

    @classmethod
    def from_dict(cls, d: dict | None) -> WeightLoaderState:
        """Create WeightLoaderState from dict with schema validation.

        Handles both new (lifecycle) and legacy (boolean flags) checkpoint formats.
        """
        from qgre.schema import WEIGHT_LOADER_STATE_SCHEMA, validate_schema

        if d is None:
            return cls()
        validated = validate_schema(d, WEIGHT_LOADER_STATE_SCHEMA, "weight_loader")
        return cls(**validated)


@dataclass
class CheckpointState:
    """Master container holding all component states.

    The single source of truth for checkpoint serialization.
    Provides validation on construction and forward compatibility
    for schema evolution.

    STATE REGISTRY PATTERN:
    To add a new optional checkpoint field:
    1. Add the field to this dataclass with default None
    2. Add the field name to OPTIONAL_FIELDS below
    That's it — from_dict will automatically handle serialization/deserialization.
    This prevents the bug class where new fields are saved but not restored.
    """

    # Component states (required)
    trainer: TrainerState
    dataloader: DataLoaderState
    advantage_estimator: AdvantageEstimatorState
    weight_loader: WeightLoaderState
    game_state: GameState
    # PyTorch model states (optional - None when saving state-only checkpoint)
    model_state_dict: dict | None = None
    optimizer_state_dict: dict | None = None
    scheduler_state_dict: dict | None = None
    # VPRM critic states (optional)
    vprm_critic_state: dict | None = None
    vprm_optimizer_state: dict | None = None
    # EGRS hint registry state (optional)
    hint_registry_state: dict | None = None
    # LoRA-Pro momentum state (optional)
    lora_pro_state: dict | None = None
    # Training context (optional)
    training_context: dict | None = None
    # Schema version for migration
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    # Registry of optional fields — from_dict iterates over this instead of hardcoding.
    # Adding a field here ensures it's restored on checkpoint load.
    # Use ClassVar so it's not a dataclass field.
    OPTIONAL_FIELDS: ClassVar[tuple[str, ...]] = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "vprm_critic_state",
        "vprm_optimizer_state",
        "hint_registry_state",
        "lora_pro_state",
        "training_context",
    )

    def __post_init__(self):
        """Validate field presence, types, and schema compatibility.

        Raises:
            ValueError: If required fields are missing
            TypeError: If field types are incorrect
        """
        # Validate required fields
        required_fields = [
            "trainer",
            "dataloader",
            "advantage_estimator",
            "weight_loader",
            "game_state",
        ]
        for field_name in required_fields:
            if not hasattr(self, field_name):
                raise ValueError(f"Missing required field: {field_name}")
            if getattr(self, field_name, None) is None:
                raise ValueError(f"Required field cannot be None: {field_name}")

        # Validate field types for component states
        type_checks = [
            ("trainer", TrainerState),
            ("dataloader", DataLoaderState),
            ("advantage_estimator", AdvantageEstimatorState),
            ("weight_loader", WeightLoaderState),
        ]
        for field_name, expected_type in type_checks:
            value = getattr(self, field_name)
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"Field '{field_name}' has incorrect type: "
                    f"expected {expected_type.__name__}, got {type(value).__name__}",
                )

        # game_state can be GameState or dict (migration path — dict converted by checkpoint.py)
        if not isinstance(self.game_state, (GameState, dict)):
            raise TypeError(
                f"Field 'game_state' has incorrect type: "
                f"expected GameState or dict, got {type(self.game_state).__name__}",
            )

        # Validate optional field types
        if self.vprm_critic_state is not None and not isinstance(self.vprm_critic_state, dict):
            raise TypeError(
                f"Field 'vprm_critic_state' must be dict or None, "
                f"got {type(self.vprm_critic_state).__name__}",
            )
        if self.vprm_optimizer_state is not None and not isinstance(
            self.vprm_optimizer_state, dict
        ):
            raise TypeError(
                f"Field 'vprm_optimizer_state' must be dict or None, "
                f"got {type(self.vprm_optimizer_state).__name__}",
            )

        # Validate optional dict fields using registry
        for field_name in self.OPTIONAL_FIELDS:
            value = getattr(self, field_name, None)
            if value is not None and not isinstance(value, dict):
                raise TypeError(
                    f"Field '{field_name}' must be dict or None, got {type(value).__name__}",
                )

        # STATE REGISTRY VALIDATION: Ensure OPTIONAL_FIELDS is complete
        # This catches the bug where a field is added to the dataclass but not to OPTIONAL_FIELDS
        required_fields_set = {
            "trainer",
            "dataloader",
            "advantage_estimator",
            "weight_loader",
            "game_state",
        }
        expected_fields = required_fields_set | set(self.OPTIONAL_FIELDS) | {"schema_version"}
        actual_fields = {
            f.name for f in self.__dataclass_fields__.values() if f.name != "OPTIONAL_FIELDS"
        }
        missing_from_registry = actual_fields - expected_fields
        if missing_from_registry:
            raise RuntimeError(
                f"STATE REGISTRY ERROR: Fields {sorted(missing_from_registry)} exist in CheckpointState "
                f"but are not in OPTIONAL_FIELDS. Add them to OPTIONAL_FIELDS to ensure "
                f"they are restored on checkpoint load.",
            )

        # Validate schema version
        if not isinstance(self.schema_version, int):
            raise TypeError(
                f"Field 'schema_version' must be int, got {type(self.schema_version).__name__}",
            )

    @classmethod
    def from_dict(cls, d: dict) -> CheckpointState:
        """Reconstruct CheckpointState from dict (e.g. from asdict() or torch.load()).

        Handles both:
        1. New format: nested StateSpec dicts (trainer: {...}, dataloader: {...})
        2. Old format: flat fields (global_step, accumulated_loss, etc.)

        Migration: Old format is detected by absence of "trainer" key.
        When migrating, fills defaults and logs warning.

        Args:
            d: Dictionary from asdict() or torch.load()

        Returns:
            CheckpointState instance with all nested dataclasses reconstructed

        Raises:
            ValueError: If required keys are missing
            TypeError: If reconstruction fails due to type mismatch
        """
        import warnings

        # Detect old format: has global_step at top level but no trainer key
        is_old_format = "global_step" in d and "trainer" not in d

        if is_old_format:
            warnings.warn(
                "Loading old checkpoint format (schema_version 1). "
                "Migrating to StateSpec format with defaults.",
                UserWarning,
                stacklevel=2,
            )
            # Validate required field exists (silent default to 0 would be catastrophic)
            if "global_step" not in d:
                raise ValueError(
                    "Old format checkpoint missing required key 'global_step'. "
                    "Cannot safely restore checkpoint — training step is unknown.",
                )
            # Warn for optional fields that affect training behavior
            missing_fields = []
            if "accumulated_loss" not in d:
                missing_fields.append("accumulated_loss")
            if "accumulated_samples" not in d:
                missing_fields.append("accumulated_samples")
            if missing_fields:
                warnings.warn(
                    f"Old checkpoint missing fields {missing_fields} — using defaults. "
                    "Loss averaging may be incorrect for this step.",
                    UserWarning,
                    stacklevel=2,
                )
            # Migrate: build StateSpec dicts from flat fields, then use schema-validated from_dict.
            # This ensures all paths go through schema validation.
            trainer_d = {
                "global_step": d["global_step"],  # Required, validated above
                "accumulated_loss": d.get("accumulated_loss", 0.0),
                "accumulation_count": d.get("accumulation_count", 0),
                "accumulated_samples": d.get("accumulated_samples", 0),
                "resumed_mid_accumulation": False,  # Computed at resume time
                "fused_validated": False,  # Re-validate after migration
                "needs_weight_sync": False,  # Set at resume time
                "rng_state": d.get("rng_state"),
                "cuda_rng_state": d.get("cuda_rng_state"),
            }
            trainer = TrainerState.from_dict(trainer_d)

            # Old format uses advantage_estimator_state (dict), wrap in StateSpec
            ae_d = {"state_dict": d.get("advantage_estimator_state")}
            advantage_estimator = AdvantageEstimatorState.from_dict(ae_d)

            # Old format uses dataloader_state (dict), wrap in StateSpec
            dl_state = d.get("dataloader_state") or {}
            dataloader = DataLoaderState.from_dict(dl_state)

            # VPRM states go into WeightLoaderState — use defaults via from_dict
            weight_loader = WeightLoaderState.from_dict({})
            # GameState uses gamestate_from_dict in checkpoint.py — here we just pass through
            game_state_raw = d.get("game_state")
            if game_state_raw is None:
                warnings.warn(
                    "Checkpoint missing 'game_state' — creating fresh GameState with phase=1. "
                    "All mastery progress and tier phases will be reset. "
                    "If unexpected, checkpoint may be corrupted.",
                    UserWarning,
                    stacklevel=2,
                )
                game_state = GameState()
            elif isinstance(game_state_raw, GameState):
                game_state = game_state_raw
            elif isinstance(game_state_raw, dict):
                from qgre.checkpoint import gamestate_from_dict

                game_state = gamestate_from_dict(game_state_raw)
            else:
                raise TypeError(
                    f"Expected dict or GameState for 'game_state', got {type(game_state_raw).__name__}. "
                    "Checkpoint may be corrupted.",
                )
        else:
            # New format: validate required keys exist
            required_keys = ["trainer"]
            missing = [k for k in required_keys if k not in d]
            if missing:
                raise ValueError(
                    f"New-format checkpoint missing required keys: {missing}. "
                    "Checkpoint may be corrupted or truncated.",
                )

            # Reconstruct nested dataclasses using schema-validated from_dict methods.
            # This is the ONLY path for StateSpec reconstruction — no direct **kwargs.
            trainer_d = d["trainer"]  # Required, validated above
            if not isinstance(trainer_d, dict):
                raise TypeError(
                    f"Expected dict for 'trainer', got {type(trainer_d).__name__}. "
                    "Checkpoint may be corrupted.",
                )
            trainer = TrainerState.from_dict(trainer_d)

            dataloader_d = d.get("dataloader", {})
            dataloader = (
                DataLoaderState.from_dict(dataloader_d)
                if isinstance(dataloader_d, dict)
                else dataloader_d
            )

            ae_d = d.get("advantage_estimator", {})
            advantage_estimator = (
                AdvantageEstimatorState.from_dict(ae_d) if isinstance(ae_d, dict) else ae_d
            )

            wl_d = d.get("weight_loader", {})
            weight_loader = WeightLoaderState.from_dict(wl_d) if isinstance(wl_d, dict) else wl_d

            game_state_raw = d.get("game_state")
            if game_state_raw is None:
                warnings.warn(
                    "Checkpoint missing 'game_state' — creating fresh GameState with phase=1. "
                    "All mastery progress and tier phases will be reset. "
                    "If unexpected, checkpoint may be corrupted.",
                    UserWarning,
                    stacklevel=2,
                )
                game_state = GameState()
            elif isinstance(game_state_raw, dict):
                # Convert dict to GameState using gamestate_from_dict
                from qgre.checkpoint import gamestate_from_dict

                game_state = gamestate_from_dict(game_state_raw)
            elif isinstance(game_state_raw, GameState):
                game_state = game_state_raw
            else:
                raise TypeError(
                    f"Expected dict or GameState for 'game_state', got {type(game_state_raw).__name__}. "
                    "Checkpoint may be corrupted.",
                )

        # Build optional fields from registry — no hardcoding field names
        optional_kwargs = {field: d.get(field) for field in cls.OPTIONAL_FIELDS}

        return cls(
            trainer=trainer,
            dataloader=dataloader,
            advantage_estimator=advantage_estimator,
            weight_loader=weight_loader,
            game_state=game_state,
            schema_version=d.get("schema_version", CHECKPOINT_SCHEMA_VERSION),
            **optional_kwargs,
        )


class SkillStatus(Enum):
    LOCKED = "locked"
    ACTIVE = "active"
    MASTERED = "mastered"


@dataclass
class PromptContext:
    """First-class prompt identity — computed once at batch boundary, used everywhere.

    Eliminates stringly-typed prompt resolution scattered across subsystems.
    Built by the trainer from batch + GameState, then flows through
    advantage computation, VPRM, tutorial recording, and gate composition.
    """

    prompt_id: int  # Original hash ID from dataloader
    skill_key: str | None  # Tutorial skill this prompt belongs to (None if untracked)
    tier: str  # Difficulty tier from metadata (e.g. "tutorial_gravity")
    aspiration_target: float  # Per-skill mastery_threshold or global default
    aspiration_warmup: float  # 0→1 ramp factor for recently unlocked skills
    is_active: bool  # Passes both tier gate AND skill gate

    @property
    def prompt_id_str(self) -> str:
        return str(self.prompt_id)


@dataclass(frozen=True)
class RewardResult:
    """Output of a reward function evaluation.

    FROZEN: instances cannot be mutated after construction. This prevents the
    "mutate for one consumer, break another" bug class where the trainer modifies
    reward/scores for one purpose (e.g., min_completion_tokens floor) and downstream
    consumers (display, mastery tracking, JSONL logging) see the corrupted values.

    Use with_floor() to create a penalized copy when the trainer needs to override.

    The reward_fn scores completions and returns per-quality scores.
    The engine uses .scores to compute per-step advantages and manage phase gating.
    Phase is engine-managed via GameState. reward_fn should NOT set phase.

    Contract:
    - scores keys MUST match quality names in step_qualities config
    - scored_spans char offsets are into the RAW completion text from vLLM decode
    - Score range: 0.0 = completely wrong, 1.0 = completely right
    - Overlapping spans are allowed (engine handles overlap normalization)
    - First occurrence per quality gets positive advantage, repeats get penalty
    """

    reward: float
    scores: dict = field(default_factory=dict)  # {"quality_name": float, ...}
    phase: int = 1  # Engine-managed — set by GameState, not reward_fn
    scored_spans: dict = field(default_factory=dict)
    # scored_spans: {"quality_name": [(char_start, char_end), ...], ...}
    # Character offsets into the completion text. When populated, the engine
    # uses these for per-token advantage assignment instead of the segmenter.
    # Reward functions that don't populate this field get the legacy segmenter path.

    def with_floor(self, reward: float) -> RewardResult:
        """Return a new RewardResult with floored reward, zeroed scores, and empty spans.

        Used by the trainer's min_completion_tokens guard to penalize empty/short
        completions without mutating the original reward function output.
        The original instance is preserved for display and JSONL logging.

        scored_spans are cleared (not preserved) so the advantage engine doesn't
        apply positive per-token advantages to tokens from a penalized completion.
        """
        return RewardResult(
            reward=reward,
            scores=dict.fromkeys(self.scores, 0.0),
            phase=self.phase,
            scored_spans={},
        )


@dataclass
class SampleData:
    """Per-sample data bundle — eliminates parallel list reindexing bugs.

    All per-sample state is stored in a single object. SPO filter reindexing
    becomes `samples = [samples[i] for i in idx]` — no way to miss a field.
    Downstream code accesses samples[i].field instead of mapping filtered
    indices back to original batch indices.

    Fields:
        completion: Token IDs for this sample's generated text
        reward_result: Reward function output (scores, spans, etc.)
        context: Prompt identity and metadata (tier, skill, aspiration)
        active_qualities: Quality keys active for this sample's tier/phase
        regions: Segmenter output (optional, None when using span-based advantages)
        token_masks: Per-quality token masks from scored_spans (optional)
        kl_region_weights: Per-token KL multipliers from region segmentation (optional)
        gen_logprobs: Generation-time logprobs for LLDS (optional, pre-padding)
    """

    completion: list[int]
    reward_result: RewardResult
    context: PromptContext
    active_qualities: list[str]
    regions: list[str] | None = None
    token_masks: dict[str, torch.Tensor] | None = None
    kl_region_weights: torch.Tensor | None = None
    gen_logprobs: torch.Tensor | None = None


@dataclass
class TrainingStep:
    """Step-level data bundle — eliminates cross-list reindexing bugs.

    All mutable step state is stored in a single object. SPO filter reindexing
    becomes `step.filter(idx)` which atomically reindexes ALL fields.

    This is the step-level analog of SampleData. SampleData bundles per-sample
    fields; TrainingStep bundles the lists/tensors that span samples.

    Invariant: All list fields have the same length (batch_size).
    Invariant: All tensor fields have batch dimension matching list lengths.
    """

    # List-based state (indexed by sample)
    samples: list[SampleData]
    reward_results: list  # list[RewardResult] - avoiding circular import
    active_qualities: list[list[str]]
    batch_regions: list[list[str] | None]
    batch_contexts: list  # list[PromptContext] - avoiding circular import
    completions: list[list[int]]

    # Tensor state (batch dimension matches list length)
    padded_advs: torch.Tensor
    comp_tensor: torch.Tensor
    comp_attention_mask: torch.Tensor
    gen_logprobs_padded: torch.Tensor | None = None
    kl_region_weights: torch.Tensor | None = None

    # Filter tracking
    filter_idx: list[int] | None = None  # Maps filtered → original indices

    def __post_init__(self) -> None:
        """Validate all lists have same length and tensors match."""
        list_lens = [
            len(self.samples),
            len(self.reward_results),
            len(self.active_qualities),
            len(self.batch_regions),
            len(self.batch_contexts),
            len(self.completions),
        ]
        if len(set(list_lens)) != 1:
            raise ValueError(
                f"TrainingStep list length mismatch: samples={len(self.samples)}, "
                f"reward_results={len(self.reward_results)}, "
                f"active_qualities={len(self.active_qualities)}, "
                f"batch_regions={len(self.batch_regions)}, "
                f"batch_contexts={len(self.batch_contexts)}, "
                f"completions={len(self.completions)}",
            )
        batch_size = list_lens[0]
        if self.padded_advs.shape[0] != batch_size:
            raise ValueError(
                f"TrainingStep tensor mismatch: padded_advs batch={self.padded_advs.shape[0]} "
                f"vs list batch={batch_size}",
            )
        # CT-7: Validate optional tensor shapes if present
        if self.gen_logprobs_padded is not None and self.gen_logprobs_padded.shape[0] != batch_size:
            raise ValueError(
                f"CT-7: gen_logprobs_padded batch={self.gen_logprobs_padded.shape[0]} vs list batch={batch_size}",
            )
        if self.kl_region_weights is not None and self.kl_region_weights.shape[0] != batch_size:
            raise ValueError(
                f"CT-7: kl_region_weights batch={self.kl_region_weights.shape[0]} vs list batch={batch_size}",
            )

    def __len__(self) -> int:
        """Return batch size."""
        return len(self.samples)

    def filter(self, idx: torch.Tensor) -> TrainingStep:
        """Atomically reindex ALL fields by idx. Returns new TrainingStep.

        Args:
            idx: 1D tensor of indices to keep (e.g., from SPO filter)

        Returns:
            New TrainingStep with all fields reindexed

        This is the key method that prevents reindexing bugs — there is no way
        to forget a field because ALL fields are filtered in one place.
        """
        idx_list = idx.tolist()

        # Validate indices at source — catch invalid indices here, not at read sites
        if idx_list:
            max_idx = max(idx_list)
            if max_idx >= len(self.samples):
                raise IndexError(
                    f"TrainingStep.filter() received index {max_idx} "
                    f"but len(samples)={len(self.samples)}. SPO filter produced invalid indices.",
                )

        # Reindex all list fields
        new_samples = [self.samples[i] for i in idx_list]
        new_reward_results = [self.reward_results[i] for i in idx_list]
        new_active_qualities = [self.active_qualities[i] for i in idx_list]
        new_batch_regions = [self.batch_regions[i] for i in idx_list]
        new_batch_contexts = [self.batch_contexts[i] for i in idx_list]
        new_completions = [self.completions[i] for i in idx_list]

        # Reindex all tensor fields
        new_padded_advs = self.padded_advs[idx]
        new_comp_tensor = self.comp_tensor[idx]
        new_comp_attention_mask = self.comp_attention_mask[idx]
        new_gen_logprobs = (
            self.gen_logprobs_padded[idx] if self.gen_logprobs_padded is not None else None
        )
        new_kl_weights = self.kl_region_weights[idx] if self.kl_region_weights is not None else None

        # Track original indices for downstream mapping (VPRM, EGRS)
        if self.filter_idx is not None:
            # Compose filters: new_filter[i] = old_filter[idx[i]]
            new_filter_idx = [self.filter_idx[i] for i in idx_list]
            # CT-4: Validate composition - all new indices must be valid in old filter
            max_old_idx = len(self.filter_idx) - 1
            for i, old_i in enumerate(idx_list):
                if old_i > max_old_idx:
                    raise IndexError(
                        f"CT-4: Filter composition error - idx[{i}]={old_i} exceeds old filter length {len(self.filter_idx)}",
                    )
        else:
            new_filter_idx = idx_list

        return TrainingStep(
            samples=new_samples,
            reward_results=new_reward_results,
            active_qualities=new_active_qualities,
            batch_regions=new_batch_regions,
            batch_contexts=new_batch_contexts,
            completions=new_completions,
            padded_advs=new_padded_advs,
            comp_tensor=new_comp_tensor,
            comp_attention_mask=new_comp_attention_mask,
            gen_logprobs_padded=new_gen_logprobs,
            kl_region_weights=new_kl_weights,
            filter_idx=new_filter_idx,
        )

    def get_original_idx(self, filtered_idx: int) -> int:
        """Map filtered batch index to original batch index.

        Used by VPRM and EGRS which need to access per-prompt state
        (mastery, baselines) using the original prompt index.
        """
        if self.filter_idx is not None:
            return self.filter_idx[filtered_idx]
        return filtered_idx


# ─── Aligned tensor frames for loss computation ──────────────────────────────


@dataclass
class MicrobatchFrame:
    """Microbatch tensors aligned to loss_len, ready for loss_fn.

    All tensor seq dimensions equal loss_len. Logprob tensors (mb_lp,
    mb_old_lp) must be truncated to [:, :loss_len] by the caller.
    """

    mask: torch.Tensor  # [mb, loss_len]
    advantages: torch.Tensor  # [mb, loss_len]
    kl_weights: torch.Tensor | None  # [mb, loss_len]
    loss_len: int


@dataclass
class AlignedLossFrame:
    """Pre-shifted tensors in logprob coordinate space.

    After build(), all tensors share the same coordinate system as mb_lp:
    position t corresponds to the prediction of token t+1.

    Coordinate algebra:
      - Raw token space:  advantages[t] = advantage for generating token t
      - Logprob space:    mb_lp[t] = log P(token[t+1] | tokens[0..t])
      - After [:, 1:] shift: advantages_shifted[t] = advantages[t+1]
        = advantage for generating token t+1 → matches mb_lp[t]

    All inputs to build() are in raw token space. build() applies the [:, 1:]
    shift to align everything with logprob space. No further shifts are needed.

    IMPORTANT: Code that mutates frame.advantages (e.g., VPRM) must write
    values in logprob space, NOT raw token space. Position t must contain
    the advantage for generating token t+1.

    Usage::

        frame = AlignedLossFrame.build(response_mask, advs, gen_lp, kl_w)
        # In microbatch loop:
        mb_advs = frame.advantages[mb_start:mb_end]  # for EGRS mutation
        # ... EGRS mutates mb_advs in-place (already in logprob space) ...
        mb = frame.slice_for_microbatch(mb_start, mb_end)
        loss = loss_fn(lp[:, : mb.loss_len], advantages=mb.advantages, mask=mb.mask)
    """

    response_mask: torch.Tensor  # [batch, loss_len]
    advantages: torch.Tensor  # [batch, loss_len]
    gen_logprobs: torch.Tensor | None  # [batch, loss_len]
    kl_weights: torch.Tensor | None  # [batch, loss_len]

    @staticmethod
    def build(
        response_mask: torch.Tensor,
        padded_advs: torch.Tensor,
        gen_logprobs_padded: torch.Tensor | None,
        kl_region_weights: torch.Tensor | None,
    ) -> AlignedLossFrame:
        """Shift from raw token space to logprob space and validate alignment.

        All inputs: [batch, max_comp_len] in raw token space.
        All outputs: [batch, max_comp_len - 1] in logprob space.
        The [:, 1:] shift aligns advantages[t+1] with logprob[t].
        """
        mask = response_mask[:, 1:]
        advs = padded_advs[:, 1:]
        gen_lp = gen_logprobs_padded[:, 1:] if gen_logprobs_padded is not None else None
        kl_w = kl_region_weights[:, 1:] if kl_region_weights is not None else None

        if advs.shape != mask.shape:
            raise RuntimeError(
                f"AlignedLossFrame: shape mismatch after shift: "
                f"advantages={advs.shape} vs response_mask={mask.shape}"
            )
        if gen_lp is not None and gen_lp.shape != mask.shape:
            raise RuntimeError(
                f"AlignedLossFrame: gen_logprobs shape {gen_lp.shape} != "
                f"response_mask shape {mask.shape}"
            )
        if kl_w is not None and kl_w.shape != mask.shape:
            raise RuntimeError(
                f"AlignedLossFrame: kl_weights shape {kl_w.shape} != "
                f"response_mask shape {mask.shape}"
            )

        return AlignedLossFrame(
            response_mask=mask,
            advantages=advs,
            gen_logprobs=gen_lp,
            kl_weights=kl_w,
        )

    def slice_for_microbatch(self, start: int, end: int) -> MicrobatchFrame:
        """Slice batch dimension. All tensors are already in logprob space."""
        mb_advs = self.advantages[start:end]
        mb_kl = self.kl_weights[start:end] if self.kl_weights is not None else None
        loss_len = mb_advs.shape[1]
        mb_mask = self.response_mask[start:end]

        return MicrobatchFrame(
            mask=mb_mask,
            advantages=mb_advs,
            kl_weights=mb_kl,
            loss_len=loss_len,
        )


@dataclass
class SkillNode:
    """A node in the tutorial skill dependency DAG."""

    name: str
    prompts: list[str]
    prerequisites: list[str]
    mastery_threshold: float = 0.8
    regression_threshold: float = 0.6
    mastery_window: int = 20
    review_probability: float = 0.15
    score_key: str | None = None  # Quality key from RewardResult.scores; None = overall reward
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=20))
    _was_mastered: bool = field(default=False, repr=False)
    _status: SkillStatus = field(default=SkillStatus.LOCKED, repr=False)
    _total_completions: int = field(default=0, repr=False)
    _initial_scores: list = field(default_factory=list, repr=False)  # First 5 scores after unlock
    _initial_mastery_logged: bool = field(default=False, repr=False)
    _unlock_step: int | None = field(default=None, repr=False)  # Step when skill was unlocked
    _pre_unlock_baseline: float | None = field(
        default=None, repr=False
    )  # Mastery score of prerequisites at unlock time
    aspiration_warmup_steps: int = 20  # Ramp aspiration from 0→full over N steps after unlock
    # Learnability-based advancement: advance only when variance collapses
    learnability_threshold: float = 0.10  # Advance when p(1-p) < this

    def __post_init__(self):
        if not self.prerequisites:
            self._status = SkillStatus.ACTIVE
        # Always sync deque maxlen with mastery_window
        self.recent_scores = deque(self.recent_scores, maxlen=self.mastery_window)

    @property
    def mastered(self) -> bool:
        if len(self.recent_scores) < self.mastery_window:
            return False
        score = self.mastery_score
        if self._was_mastered:
            return score >= self.regression_threshold
        return score >= self.mastery_threshold

    @property
    def mastery_score(self) -> float:
        if not self.recent_scores:
            return 0.0
        return sum(self.recent_scores) / len(self.recent_scores)

    @property
    def learnability(self) -> float:
        """Bernoulli variance = p(1-p). Maximized at p=0.5, zero at p=0 or p=1.

        High learnability = skill still has variance = still learning.
        Low learnability = skill is stable = ready to advance.
        """
        if len(self.recent_scores) < 5:
            # Not enough data — return max learnability to prevent premature advancement
            # 0.25 = variance at p=0.5 (maximum uncertainty)
            return 0.25
        p = self.mastery_score
        return p * (1.0 - p)

    @property
    def ready_to_advance(self) -> bool:
        """Ready when: mastery > threshold AND learnability < stale_threshold.

        High mastery + low learnability = skill is learned, move on.
        High mastery + high learnability = skill still has variance, stay.
        """
        if len(self.recent_scores) < self.mastery_window:
            return False
        if self.mastery_score < self.mastery_threshold:
            return False
        # Only advance when variance has collapsed (skill is stable)
        return self.learnability < self.learnability_threshold

    def unlocked(self, skill_tree: dict[str, SkillNode]) -> bool:
        return all(skill_tree[pre].mastered for pre in self.prerequisites)

    @property
    def status(self) -> SkillStatus:
        return self._status

    def record_score(self, v_correct: float):
        self.recent_scores.append(v_correct)
        self._total_completions += 1
        # Track initial mastery (first 5 scores after unlock) for transfer_lift
        if len(self._initial_scores) < 5:
            self._initial_scores.append(v_correct)
        if self.mastered:
            self._was_mastered = True
        elif self._was_mastered and not self.mastered:
            self._was_mastered = False

    @property
    def initial_mastery(self) -> float | None:
        """Mastery score on first 5 completions after unlock. None if < 5 completions."""
        if len(self._initial_scores) < 5:
            return None
        return sum(self._initial_scores) / len(self._initial_scores)


QUALITY_WINDOW_SIZE = 20


class StagnationStatus(Enum):
    NORMAL = "normal"
    STAGNATING = "stagnating"
    STUCK = "stuck"


@dataclass
class GameState:
    """QGRE 2D mastery matrix — tracks quality × difficulty independently.

    Two axes:
    - Quality phases (1→N): format → identification → equations → full derivation
    - Difficulty tiers: tier1 → tier2 → tier3 → ... (domain-specific)

    Each cell mastery[tier][phase] has its own rolling window. Quality phases
    advance per-tier. Tier N+1 unlocks when tier N reaches a configurable
    quality phase threshold.

    When no tiers are configured, all prompts map to a single "default" tier,
    making this equivalent to the original 1D phase system.
    """

    step_count: int = 0
    mastery_threshold: float = 0.8
    stagnation_timeout: int = 200
    plateau_window: int = 50
    plateau_threshold: float = 0.02
    quality_window_size: int = 20  # Rolling window for mastery score tracking

    # Tutorial skill tree
    tutorial_enabled: bool = False
    skill_tree: dict = field(default_factory=dict)  # str → SkillNode
    all_prompts: list = field(default_factory=list)
    post_mastery_behavior: str = "review_only"
    untracked_always_active: bool = True
    _sequential_mastery: bool = False
    default_aspiration_target: float = 0.8
    _prompt_to_skill: dict = field(default_factory=dict, repr=False)
    _tier_to_skills: dict = field(default_factory=dict, repr=False)  # tier_name → [skill_keys]
    _active_base_pool: list = field(default_factory=list, repr=False)
    _mastered_pool: list = field(default_factory=list, repr=False)
    _active_prompts_dirty: bool = field(default=True, repr=False)
    _pool_version: int = field(default=0, repr=False)

    # 2D mastery matrix
    tier_mastery: dict = field(default_factory=dict)
    # tier_mastery[tier][step_num] = deque of scores
    tier_phases: dict = field(default_factory=lambda: {"default": 1})
    # Per-tier quality phase
    active_tiers: list = field(default_factory=lambda: ["default"])
    # Currently unlocked tiers
    tier_steps_at_phase_start: dict = field(default_factory=dict)
    # tier_steps_at_phase_start[tier] = step_count when tier's current phase started
    phase_history: list = field(default_factory=list)
    # [(step_count, tier, old_phase, new_phase), ...]

    # ── Tutorial skill tree ──

    def init_tutorial(
        self,
        tutorial_config,
        all_prompt_ids: list[str] | None = None,
        dataloader_items: list[dict] | None = None,
        difficulty_column: str | None = None,
    ):
        """Initialize skill tree from TutorialConfig. Call after construction.

        Args:
            tutorial_config: TutorialConfig from parsed YAML.
            all_prompt_ids: List of all prompt IDs (as strings) in the dataset.
            dataloader_items: Raw dataloader items for metadata-based prompt matching.
                Each item has 'prompt_id' (int) and 'metadata' (dict).
            difficulty_column: Metadata column for tier mapping (e.g. "difficulty").
        """
        from qgre.config import TutorialConfig

        if not isinstance(tutorial_config, TutorialConfig) or not tutorial_config.enabled:
            self.tutorial_enabled = False
            return

        if not tutorial_config.skill_tree:
            raise ValueError(
                "tutorial.enabled=true but skill_tree is empty. "
                "Define at least one skill with prompts or match_metadata.",
            )

        self.tutorial_enabled = True
        self.post_mastery_behavior = tutorial_config.post_mastery_behavior
        self.untracked_always_active = tutorial_config.untracked_always_active
        self._sequential_mastery = tutorial_config.sequential_mastery
        self.all_prompts = list(all_prompt_ids) if all_prompt_ids else []

        self.skill_tree = {}
        for key, sc in tutorial_config.skill_tree.items():
            # Resolve prompts from match_metadata if configured
            prompts = list(sc.prompts)
            if sc.match_metadata and not dataloader_items:
                warnings.warn(
                    f"Skill '{key}' has match_metadata={sc.match_metadata} but no "
                    f"dataloader_items provided. Metadata resolution skipped — "
                    f"skill will have only explicit prompts: {sc.prompts}",
                    stacklevel=2,
                )
            if sc.match_metadata and dataloader_items:
                for item in dataloader_items:
                    meta = item.get("metadata", {})
                    if all(meta.get(col) == val for col, val in sc.match_metadata.items()):
                        pid_str = str(item["prompt_id"])
                        if pid_str not in prompts:
                            prompts.append(pid_str)

            self.skill_tree[key] = SkillNode(
                name=key,
                prompts=prompts,
                prerequisites=list(sc.prerequisites),
                mastery_threshold=sc.mastery_threshold,
                regression_threshold=sc.regression_threshold,
                mastery_window=sc.mastery_window,
                review_probability=sc.review_probability,
                score_key=sc.score_key,
                aspiration_warmup_steps=sc.aspiration_warmup_steps,
                learnability_threshold=sc.learnability_threshold,
                recent_scores=deque(maxlen=sc.mastery_window),
            )

            if prompts:
                # Check if any skill prompts are not in the dataloader
                missing_prompts = set(prompts) - set(self.all_prompts)
                if missing_prompts:
                    raise ValueError(
                        f"Skill '{key}' specifies prompt_ids not in dataloader: {missing_prompts}. "
                        f"Available prompts: {len(self.all_prompts)}. "
                        f"Skill would never receive completions and block dependent skills. "
                        f"Check skill_tree config or dataloader filtering."
                    )

            if not prompts:
                # Check if this skill blocks any dependents — deadlock if so
                dependents = [
                    k for k, s in tutorial_config.skill_tree.items() if key in s.prerequisites
                ]
                if dependents:
                    raise ValueError(
                        f"Skill '{key}' has no prompts after metadata resolution and blocks {dependents}. "
                        f"Training would deadlock. match_metadata={sc.match_metadata}, "
                        f"explicit prompts={sc.prompts}",
                    )
                warnings.warn(
                    f"Skill '{key}' has no prompts after metadata resolution. "
                    f"match_metadata={sc.match_metadata}, explicit prompts={sc.prompts}",
                    stacklevel=2,
                )

        # Build O(1) prompt → skill lookup
        self._prompt_to_skill = {}
        for key, node in self.skill_tree.items():
            for prompt_id in node.prompts:
                self._prompt_to_skill[prompt_id] = key

        # Build tier → skills mapping (which skills have prompts in which tiers)
        self._tier_to_skills = {}
        if dataloader_items and difficulty_column:
            for item in dataloader_items:
                pid_str = str(item["prompt_id"])
                tier = item.get("metadata", {}).get(difficulty_column, "default")
                skill_key = self._prompt_to_skill.get(pid_str)
                if skill_key is not None:
                    self._tier_to_skills.setdefault(tier, set()).add(skill_key)
            # Convert sets to lists for serialization
            self._tier_to_skills = {k: list(v) for k, v in self._tier_to_skills.items()}
            logger.info(f"[TUTORIAL] Tier→skill mapping: {self._tier_to_skills}")

        self._active_prompts_dirty = True

        # Validate
        self.validate_skill_tree()

    def validate_skill_tree(self):
        """Validate DAG structure. Raise ValueError on any violation."""
        # 1. Cycle detection (topological sort)
        visited = set()
        temp = set()

        def visit(key):
            if key in temp:
                raise ValueError(f"Cycle detected in skill tree involving: {key}")
            if key in visited:
                return
            temp.add(key)
            for pre in self.skill_tree[key].prerequisites:
                if pre not in self.skill_tree:
                    raise ValueError(
                        f"Skill '{key}' has prerequisite '{pre}' "
                        f"which does not exist in skill_tree",
                    )
                visit(pre)
            temp.remove(key)
            visited.add(key)

        for key in self.skill_tree:
            visit(key)

        # 2. Duplicate prompts
        seen_prompts = {}
        for key, node in self.skill_tree.items():
            for prompt_id in node.prompts:
                if prompt_id in seen_prompts:
                    raise ValueError(
                        f"Prompt '{prompt_id}' appears in both "
                        f"'{seen_prompts[prompt_id]}' and '{key}'",
                    )
                seen_prompts[prompt_id] = key

        # 3. At least one root skill
        roots = [k for k, n in self.skill_tree.items() if not n.prerequisites]
        if not roots:
            raise ValueError(
                "No root skills (skills with empty prerequisites). Training cannot start.",
            )

        # 4. regression_threshold < mastery_threshold
        for key, node in self.skill_tree.items():
            if node.regression_threshold >= node.mastery_threshold:
                raise ValueError(
                    f"Skill '{key}': regression_threshold ({node.regression_threshold}) "
                    f"must be less than mastery_threshold ({node.mastery_threshold})",
                )

        logger.info(
            f"Skill tree validated: {len(self.skill_tree)} skills, "
            f"{len(roots)} roots, {len(seen_prompts)} prompts mapped"
        )

    def _invalidate_prompt_cache(self):
        self._active_prompts_dirty = True
        self._pool_version += 1

    def snapshot_pool_version(self) -> int:
        """Snapshot pool version. Call before recording, compare after."""
        return self._pool_version

    def did_prompt_pool_change(self, snapshot: int) -> bool:
        """Check if prompt pool changed since snapshot."""
        return self._pool_version != snapshot

    def get_active_prompts(self) -> list[str]:
        """Return prompt IDs eligible for sampling this step."""
        if not self.tutorial_enabled:
            return self.all_prompts

        # Rebuild deterministic pool only on state change
        if self._active_prompts_dirty:
            self._active_base_pool = []
            self._mastered_pool = []

            # Collect active (unlocked + not mastered) and mastered skills
            active_skills = []
            for node in self.skill_tree.values():
                is_unlocked = node.unlocked(self.skill_tree)
                if is_unlocked and not node.mastered:
                    active_skills.append(node)
                elif node.mastered:
                    self._mastered_pool.append(node)

            if self._sequential_mastery and active_skills:
                # One skill at a time: only the first active skill's prompts
                self._active_base_pool = list(active_skills[0].prompts)
            else:
                for node in active_skills:
                    self._active_base_pool.extend(node.prompts)

            if self.untracked_always_active:
                tracked = set(self._prompt_to_skill.keys())
                untracked = [p for p in self.all_prompts if p not in tracked]
                self._active_base_pool.extend(untracked)

            self._active_prompts_dirty = False

        active = list(self._active_base_pool)

        # Review sampling for mastered skills
        for node in self._mastered_pool:
            if random.random() < node.review_probability:
                active.extend(node.prompts)

        # Post-mastery behavior: all skills mastered, no active base pool
        if not self._active_base_pool:
            if self.post_mastery_behavior == "pause" and not active:
                logger.info(
                    "[TUTORIAL] All skills mastered. post_mastery_behavior=pause. "
                    "Returning empty pool — trainer should stop."
                )
                return []
            if self.post_mastery_behavior == "continue_all":
                logger.info("[TUTORIAL] All skills mastered. post_mastery_behavior=continue_all.")
                return self.all_prompts
            # review_only: return whatever review sampling produced (may be empty this step)
            if active:
                return active
            # review_only but no review samples this step — return mastered prompts directly
            # rather than falling back to all (which would include locked skills)
            review_fallback = []
            for node in self._mastered_pool:
                review_fallback.extend(node.prompts)
            if review_fallback:
                return review_fallback
            # review_only with nothing to review — return empty, trainer handles it
            logger.info(
                "[TUTORIAL] review_only: no review prompts this step. Returning empty pool."
            )
            return []

        if not active:
            logger.warning("Tutorial system: no active prompts this step. Falling back to all.")
            return self.all_prompts

        return active

    def record_completion(self, prompt_id: str, v_correct: float):
        """Route a completion score to the appropriate skill node.

        Uses learnability-based advancement: skills only transition to MASTERED
        when ready_to_advance=True (mastery threshold + low variance). This
        prevents premature advancement when variance is still high.

        Regression still uses mastered (with hysteresis) for stability.
        """
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is None:
            return  # Untracked prompt

        node = self.skill_tree[skill_key]
        was_mastered_before = node.mastered  # For regression hysteresis
        was_ready_before = node.ready_to_advance  # For advancement gating
        node.record_score(v_correct)
        is_mastered_now = node.mastered
        is_ready_now = node.ready_to_advance

        # Advancement: uses ready_to_advance (mastery + low learnability variance)
        # This prevents advancing when skill is still high-variance (in ZPD)
        if is_ready_now and not was_ready_before:
            node._status = SkillStatus.MASTERED
            logger.info(
                f"[TUTORIAL] SKILL MASTERED: {node.name} "
                f"(mastery={node.mastery_score:.2f}, "
                f"learnability={node.learnability:.3f}, "
                f"threshold={node.mastery_threshold}, "
                f"learnability_threshold={node.learnability_threshold})"
            )
            self._invalidate_prompt_cache()
            self._check_unlocks()

        # Regression: uses mastered (hysteresis threshold) for stability
        # Don't relock just because variance spiked temporarily
        elif was_mastered_before and not is_mastered_now:
            node._status = SkillStatus.ACTIVE
            logger.warning(
                f"[TUTORIAL] MASTERY REGRESSION: {node.name} "
                f"(mastery={node.mastery_score:.2f}, "
                f"regression_threshold={node.regression_threshold})"
            )
            self._invalidate_prompt_cache()
            self._check_relocks()

    def _check_unlocks(self):
        """After a mastery event, check if any locked skills should unlock."""
        for node in self.skill_tree.values():
            if node.status == SkillStatus.LOCKED and node.unlocked(self.skill_tree):
                node._status = SkillStatus.ACTIVE
                node._unlock_step = self.step_count
                # Capture pre-unlock baseline: mean mastery of prerequisites at unlock time
                pre_scores = [self.skill_tree[pre].mastery_score for pre in node.prerequisites]
                node._pre_unlock_baseline = sum(pre_scores) / len(pre_scores) if pre_scores else 0.0
                self._invalidate_prompt_cache()
                logger.info(
                    f"[TUTORIAL] SKILL UNLOCKED: {node.name} "
                    f"(prerequisites met: {node.prerequisites}, "
                    f"step={self.step_count}, "
                    f"pre_unlock_baseline={node._pre_unlock_baseline:.3f})"
                )

    def _check_relocks(self):
        """After a regression, cascade re-lock to dependents with full mastery reset."""
        changed = True
        while changed:
            changed = False
            for node in self.skill_tree.values():
                if node.status in (SkillStatus.ACTIVE, SkillStatus.MASTERED) and not node.unlocked(
                    self.skill_tree
                ):
                    if node.prerequisites:
                        prev_status = node.status
                        node._status = SkillStatus.LOCKED
                        node._was_mastered = False
                        node.recent_scores.clear()
                        node._total_completions = 0
                        node._initial_scores.clear()
                        node._initial_mastery_logged = False
                        node._unlock_step = None
                        node._pre_unlock_baseline = None
                        changed = True
                        self._invalidate_prompt_cache()
                        logger.warning(
                            f"[TUTORIAL] CASCADE RE-LOCK: {node.name} "
                            f"(was {prev_status}, prerequisite lost mastery, "
                            f"mastery state reset)"
                        )

    def resolve_mastery_score(self, prompt_id: str, reward_result) -> float:
        """Extract the mastery-tracking score for a prompt from a RewardResult.

        Uses the skill's score_key if configured, otherwise falls back to overall reward.
        """
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is not None:
            node = self.skill_tree[skill_key]
            if node.score_key is not None:
                if node.score_key in reward_result.scores:
                    return reward_result.scores[node.score_key]
                warnings.warn(
                    f"Skill '{node.name}' score_key='{node.score_key}' not in "
                    f"reward_result.scores (keys: {list(reward_result.scores.keys())}). "
                    f"Falling back to overall reward.",
                    stacklevel=2,
                )
        return reward_result.reward

    def build_prompt_contexts(
        self,
        prompt_ids: list[int],
        metadata: list[dict],
        difficulty_column: str | None = None,
        active_tiers: set[str] | None = None,
    ) -> list[PromptContext]:
        """Build PromptContext for each prompt in a batch. Computed once, used everywhere."""
        active_prompt_set = set(self.get_active_prompts()) if self.tutorial_enabled else None
        contexts = []
        for i, pid in enumerate(prompt_ids):
            pid_str = str(pid)
            skill_key = self._prompt_to_skill.get(pid_str) if self.tutorial_enabled else None
            meta = metadata[i] if i < len(metadata) else {}
            tier = meta.get(difficulty_column, "default") if difficulty_column else "default"

            # Aspiration target: per-skill if tutorial enabled, else global
            if skill_key is not None:
                asp_target = self.skill_tree[skill_key].mastery_threshold
            else:
                asp_target = self.default_aspiration_target

            # Active: tutorial-tracked prompts in active skills bypass tier gate.
            # Untracked prompts respect tier gate. Locked skill prompts are always inactive.
            if skill_key is not None and active_prompt_set is not None:
                # Tutorial-tracked: skill gate is the authority
                is_active = pid_str in active_prompt_set
            else:
                # Untracked: tier gate decides
                is_active = active_tiers is None or tier in active_tiers

            # Aspiration warmup: dampen after tutorial skill unlock OR tier phase advance
            warmup = self.get_aspiration_warmup_factor(pid_str) if self.tutorial_enabled else 1.0
            # Also dampen after tier phase advance — prevents gradient shock on new phases
            tier_warmup = self._get_tier_phase_warmup(tier)
            warmup = min(warmup, tier_warmup)

            contexts.append(
                PromptContext(
                    prompt_id=pid,
                    skill_key=skill_key,
                    tier=tier,
                    aspiration_target=asp_target,
                    aspiration_warmup=warmup,
                    is_active=is_active,
                )
            )
        return contexts

    def can_tier_unlock(self, tier_name: str) -> bool:
        """Check if tutorial prerequisites allow this tier to unlock.

        A tier can unlock only if all PREREQUISITE skills of skills in that tier
        are MASTERED. This prevents premature tier advancement before the model
        has demonstrated consistent mastery on the prerequisite tutorial skills.

        For example, to unlock "combined" tier (which has gravity_spring prompts),
        the prerequisites of gravity_spring (freefall, spring_only) must be mastered.

        Mastery requires mastery_window completions (default 20) at or above
        mastery_threshold. This ensures the model has seen enough examples before
        advancing to harder tiers.

        Uses _tier_to_skills mapping built during init_tutorial.
        """
        if not self.tutorial_enabled:
            return True
        skills_in_tier = self._tier_to_skills.get(tier_name, [])
        if not skills_in_tier:
            return True

        # Collect all prerequisite skills that must be mastered
        prereq_skills = set()
        for skill_key in skills_in_tier:
            node = self.skill_tree[skill_key]
            for prereq in node.prerequisites:
                prereq_skills.add(prereq)

        # Check each prerequisite is mastered
        for prereq_key in prereq_skills:
            prereq_node = self.skill_tree[prereq_key]
            if not prereq_node.mastered:
                completions = len(prereq_node.recent_scores)
                needed = prereq_node.mastery_window
                score = prereq_node.mastery_score
                threshold = prereq_node.mastery_threshold
                print(
                    f"  │ ⏳ Tier '{tier_name}' blocked — prerequisite skill '{prereq_key}' not mastered "
                    f"({completions}/{needed} completions, score {score:.2f}/{threshold:.2f})"
                )
                return False
        return True

    def get_aspiration_target(self, prompt_id: str) -> float:
        """Return the mastery_threshold for the skill this prompt belongs to."""
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is not None:
            return self.skill_tree[skill_key].mastery_threshold
        return self.default_aspiration_target

    def _get_tier_phase_warmup(self, tier: str, warmup_steps: int = 20) -> float:
        """Return aspiration warmup for recently-advanced tier phases.

        Ramps from 0→1 over warmup_steps after a tier phase advance.
        Prevents gradient shock from full aspiration pressure when the model
        encounters harder quality requirements at the new phase.
        """
        if tier not in self.tier_steps_at_phase_start:
            return 1.0  # No record → full aspiration
        steps_since = self.step_count - self.tier_steps_at_phase_start[tier]
        if steps_since >= warmup_steps:
            return 1.0
        return max(0.0, steps_since / warmup_steps)

    def get_aspiration_warmup_factor(self, prompt_id: str) -> float:
        """Return aspiration warmup multiplier (0→1) for recently unlocked skills.

        Ramps aspiration beta linearly from 0 to 1 over aspiration_warmup_steps
        after a skill unlocks. Prevents gradient shock from full aspiration pressure
        on prompts the model has never seen.

        Returns 1.0 for root skills, mastered skills, and untracked prompts.
        """
        skill_key = self._prompt_to_skill.get(prompt_id)
        if skill_key is None:
            return 1.0
        node = self.skill_tree[skill_key]
        if node._unlock_step is None:
            return 1.0  # Root skill (never locked) or not yet tracked
        steps_since_unlock = self.step_count - node._unlock_step
        if steps_since_unlock >= node.aspiration_warmup_steps:
            return 1.0
        return max(0.0, steps_since_unlock / node.aspiration_warmup_steps)

    def get_tutorial_metrics(self) -> dict:
        """Return per-step tutorial metrics for logging."""
        if not self.tutorial_enabled:
            return {}
        metrics = {}
        active = sum(1 for n in self.skill_tree.values() if n.status == SkillStatus.ACTIVE)
        mastered = sum(1 for n in self.skill_tree.values() if n.status == SkillStatus.MASTERED)
        locked = sum(1 for n in self.skill_tree.values() if n.status == SkillStatus.LOCKED)
        metrics["tutorial/active_skills"] = active
        metrics["tutorial/mastered_skills"] = mastered
        metrics["tutorial/locked_skills"] = locked
        metrics["tutorial/total_skills"] = len(self.skill_tree)
        # Active prompt pool size (deterministic — rebuild cache if needed, but don't include random review sampling)
        if self._active_prompts_dirty:
            self.get_active_prompts()  # Rebuild cache (result discarded — we only want the deterministic pools)
        metrics["tutorial/active_prompt_pool_size"] = len(self._active_base_pool)
        metrics["tutorial/mastered_review_pool_size"] = sum(
            len(n.prompts) for n in self._mastered_pool
        )
        for key, node in self.skill_tree.items():
            metrics[f"tutorial/skill/{key}/mastery"] = node.mastery_score
            metrics[f"tutorial/skill/{key}/status"] = node.status.value
            metrics[f"tutorial/skill/{key}/completions"] = len(node.recent_scores)
            metrics[f"tutorial/skill/{key}/aspiration_target"] = node.mastery_threshold
            # Aspiration warmup factor (visible when ramping)
            if node._unlock_step is not None:
                steps_since = self.step_count - node._unlock_step
                if steps_since < node.aspiration_warmup_steps:
                    metrics[f"tutorial/skill/{key}/aspiration_warmup"] = (
                        steps_since / node.aspiration_warmup_steps
                    )
            # Transfer lift: initial mastery after unlock (logged once at 5 completions)
            if node.initial_mastery is not None and not node._initial_mastery_logged:
                metrics[f"tutorial/skill/{key}/initial_mastery"] = node.initial_mastery
                # Compute transfer_lift: how much did prerequisites help?
                transfer_lift = node.initial_mastery - (node._pre_unlock_baseline or 0.0)
                metrics[f"tutorial/skill/{key}/transfer_lift"] = transfer_lift
                node._initial_mastery_logged = True
                # Log prominently — this is the key metric for validating the tutorial system
                logger.info(
                    f"[TUTORIAL] *** TRANSFER LIFT: {key} ***\n"
                    f"  initial_mastery = {node.initial_mastery:.3f} (first 5 completions)\n"
                    f"  pre_unlock_baseline = {node._pre_unlock_baseline or 0:.3f} (prerequisite mastery at unlock)\n"
                    f"  transfer_lift = {transfer_lift:+.3f} "
                    f"({'POSITIVE — prerequisites helped' if transfer_lift > 0 else 'ZERO/NEGATIVE — check prerequisite structure'})",
                )
        return metrics

    def tutorial_state_dict(self) -> dict:
        """Serialize tutorial state for checkpointing."""
        if not self.tutorial_enabled:
            return {}
        return {
            "skill_tracker": {
                key: {
                    "scores": list(node.recent_scores),
                    "was_mastered": node._was_mastered,
                    "status": node._status.value,
                    "total_completions": node._total_completions,
                    "initial_scores": list(node._initial_scores),
                    "initial_mastery_logged": node._initial_mastery_logged,
                    "unlock_step": node._unlock_step,
                    "pre_unlock_baseline": node._pre_unlock_baseline,
                }
                for key, node in self.skill_tree.items()
            },
        }

    def load_tutorial_state_dict(self, state: dict):
        """Restore tutorial state from checkpoint."""
        if "skill_tracker" not in state:
            if self.tutorial_enabled:
                logger.warning("[TUTORIAL] No tutorial state in checkpoint — starting fresh")
            return
        for key, data in state["skill_tracker"].items():
            if key not in self.skill_tree:
                logger.warning(
                    f"[TUTORIAL] Checkpoint skill '{key}' not in current config — dropping"
                )
                continue
            if isinstance(data, list):
                scores, was_mastered, status = data, False, None
            else:
                scores = data["scores"]
                was_mastered = data.get("was_mastered", False)
                status = data.get("status", None)
            self.skill_tree[key].recent_scores = deque(
                scores,
                maxlen=self.skill_tree[key].mastery_window,
            )
            self.skill_tree[key]._was_mastered = was_mastered
            if status is not None:
                self.skill_tree[key]._status = (
                    SkillStatus(status) if isinstance(status, str) else status
                )
            if isinstance(data, dict):
                self.skill_tree[key]._total_completions = data.get("total_completions", 0)
                self.skill_tree[key]._initial_scores = data.get("initial_scores", [])
                self.skill_tree[key]._initial_mastery_logged = data.get(
                    "initial_mastery_logged", False
                )
                self.skill_tree[key]._unlock_step = data.get("unlock_step", None)
                self.skill_tree[key]._pre_unlock_baseline = data.get("pre_unlock_baseline", None)
        self._active_prompts_dirty = True

    # ── 1D backward compat properties ──

    @property
    def phase(self) -> int:
        """Global phase = min phase across active tiers. For 1D compat."""
        if not self.tier_phases:
            return 1
        active = [self.tier_phases[t] for t in self.active_tiers if t in self.tier_phases]
        return min(active) if active else 1

    @property
    def step_mastery(self) -> dict:
        """Legacy: return default tier's mastery windows."""
        return self.tier_mastery.get("default", {})

    # ── 2D mastery tracking ──

    def record_tier_step_score(self, tier: str, step_num: int, score: float):
        """Record a quality score for a tier+step cell."""
        if tier not in self.tier_mastery:
            self.tier_mastery[tier] = {}
        if step_num not in self.tier_mastery[tier]:
            self.tier_mastery[tier][step_num] = deque(maxlen=self.quality_window_size)
        self.tier_mastery[tier][step_num].append(score)

    def get_tier_step_mastery(self, tier: str, step_num: int) -> float:
        """Get the mean quality score for a tier+step cell."""
        windows = self.tier_mastery.get(tier, {})
        window = windows.get(step_num)
        if not window:
            return 0.0
        return sum(window) / len(window)

    min_observations_before_advance: int = 10

    def check_tier_phase_advance(self, tier: str, max_phase: int) -> bool:
        """Check if a tier's current quality phase is mastered. Advance if so."""
        current_phase = self.tier_phases.get(tier, 1)
        if current_phase >= max_phase:
            return False

        observations = self.tier_mastery.get(tier, {}).get(current_phase)
        not_enough_observations = (
            observations is not None and len(observations) < self.min_observations_before_advance
        )
        if not_enough_observations:
            return False

        mastery = self.get_tier_step_mastery(tier, current_phase)
        if mastery >= self.mastery_threshold:
            import logging

            _logger = logging.getLogger("qgre.types")
            old_phase = current_phase
            new_phase = current_phase + 1
            self.tier_phases[tier] = new_phase
            self.phase_history.append((self.step_count, tier, old_phase, new_phase))
            # TP2: Reset timestamp on phase change (advance or regression)
            self.tier_steps_at_phase_start[tier] = self.step_count
            _logger.warning(
                f"[PHASE ADVANCE] tier={tier}, {old_phase}→{new_phase}, step={self.step_count}, mastery={mastery:.3f}"
            )
            return True
        # TP2: Reset timestamp on phase regression if phase decreased
        if tier in self.tier_phases and self.tier_phases[tier] < current_phase:
            self.tier_steps_at_phase_start[tier] = self.step_count
        return False

    def check_tier_unlock(
        self,
        tier_order: list[str],
        unlock_phase: int,
        unlock_threshold: float,
    ) -> str | None:
        """Check if the next tier should be unlocked.

        Finds the first inactive tier in tier_order. Checks if ALL active tiers
        before it have reached unlock_phase with mastery >= unlock_threshold.
        Returns the newly unlocked tier name, or None.
        """
        active_set = set(self.active_tiers)

        # Find first inactive tier in tier_order
        next_tier = None
        next_idx = -1
        for i, t in enumerate(tier_order):
            if t not in active_set:
                next_tier = t
                next_idx = i
                break

        if next_tier is None:
            return None  # All tiers already unlocked

        # All active tiers before next_tier must have reached unlock_phase with threshold
        for t in tier_order[:next_idx]:
            if t not in active_set:
                continue  # Skip tiers not in active set (shouldn't happen in ordered unlock)
            phase = self.tier_phases.get(t, 1)
            if phase < unlock_phase:
                return None  # This tier hasn't reached the required quality phase
            mastery = self.get_tier_step_mastery(t, unlock_phase)
            if mastery < unlock_threshold:
                return None  # This tier hasn't mastered the required phase

        # All active tiers before next_tier are ready — unlock it
        if next_tier not in self.active_tiers:
            self.active_tiers.append(next_tier)
        self.tier_phases[next_tier] = 1
        self.tier_steps_at_phase_start[next_tier] = self.step_count
        return next_tier

    def check_tier_stagnation(self, tier: str) -> StagnationStatus:
        """Check if a specific tier is stagnating in its current phase."""
        start = self.tier_steps_at_phase_start.get(tier, 0)
        steps_in_phase = self.step_count - start
        if steps_in_phase >= self.stagnation_timeout:
            return StagnationStatus.STUCK

        current_phase = self.tier_phases.get(tier, 1)
        windows = self.tier_mastery.get(tier, {})
        window = windows.get(current_phase)
        if window and len(window) >= self.plateau_window:
            recent = list(window)[-self.plateau_window :]
            half = len(recent) // 2
            first_half_mean = sum(recent[:half]) / half
            second_half_mean = sum(recent[half:]) / (len(recent) - half)
            if abs(second_half_mean - first_half_mean) < self.plateau_threshold:
                return StagnationStatus.STAGNATING

        return StagnationStatus.NORMAL

    # ── 1D compat methods (delegate to "default" tier) ──

    def record_step_score(self, step_num: int, score: float):
        """Legacy 1D: record to default tier."""
        self.record_tier_step_score("default", step_num, score)

    def get_step_mastery(self, step_num: int) -> float:
        """Legacy 1D: read from default tier."""
        return self.get_tier_step_mastery("default", step_num)

    def check_phase_advance(self, max_phase: int) -> bool:
        """Legacy 1D: advance default tier's phase."""
        return self.check_tier_phase_advance("default", max_phase)

    def check_stagnation(self) -> StagnationStatus:
        """Legacy 1D: check default tier stagnation."""
        return self.check_tier_stagnation("default")
