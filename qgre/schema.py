"""Schema validation for checkpoint and state boundaries.

This module provides typed schema validation at serialization/deserialization points,
replacing scattered isinstance checks with a single declarative schema definition.

The leverage point: 28 bugs across 3 harden rounds traced to untyped dict boundaries.
One schema, one validation pass, zero scattered isinstance checks.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar


if TYPE_CHECKING:
    from collections.abc import Callable


T = TypeVar("T")


class Required(Enum):
    """Marker for required fields."""

    YES = "required"
    NO = "optional"


@dataclass
class FieldSpec:
    """Specification for a single field in a schema.

    Attributes:
        expected_type: The expected Python type (int, float, str, dict, list, etc.)
        required: Whether the field must be present
        default: Default value if field is missing and not required
        coerce: Whether to attempt type coercion (e.g., str "123" -> int 123)
        filter_nan: Whether to filter NaN/Inf values from numeric lists
        validate: Optional custom validation function
        nested_schema: Schema for nested dict validation
    """

    expected_type: type | tuple[type, ...]
    required: Required = Required.NO
    default: Any = None
    coerce: bool = True
    filter_nan: bool = False
    validate: Callable[[Any], bool] | None = None
    nested_schema: dict[str, FieldSpec] | None = None


def validate_field(
    value: Any,
    spec: FieldSpec,
    path: str,
) -> Any:
    """Validate and coerce a single field value.

    Args:
        value: The value to validate
        spec: Field specification
        path: Dot-separated path for error messages (e.g., "game_state.tier_phases")

    Returns:
        The validated (and possibly coerced) value

    Raises:
        TypeError: If type doesn't match and coercion fails
        ValueError: If custom validation fails
    """
    # Handle None
    if value is None:
        if spec.required == Required.YES:
            raise ValueError(f"SCHEMA: Required field '{path}' is None")
        return spec.default

    # Type checking with coercion
    expected = spec.expected_type
    if isinstance(expected, tuple):
        type_match = isinstance(value, expected)
    else:
        type_match = isinstance(value, expected)

    if not type_match and spec.coerce:
        value = _coerce_type(value, expected, path)
    elif not type_match:
        raise TypeError(
            f"SCHEMA: Field '{path}' expected {_type_name(expected)}, got {type(value).__name__}",
        )

    # Filter NaN/Inf from numeric lists
    if spec.filter_nan and isinstance(value, (list, tuple)):
        original_len = len(value)
        value = [v for v in value if _is_finite(v)]
        if len(value) < original_len:
            warnings.warn(
                f"SCHEMA: Filtered {original_len - len(value)} NaN/Inf values from '{path}'",
                stacklevel=2,
            )

    # Filter NaN/Inf from dict values
    if spec.filter_nan and isinstance(value, dict):
        filtered = {}
        nan_count = 0
        for k, v in value.items():
            if _is_finite(v):
                filtered[k] = v
            else:
                nan_count += 1
        if nan_count > 0:
            warnings.warn(
                f"SCHEMA: Filtered {nan_count} NaN/Inf values from '{path}'",
                stacklevel=2,
            )
        value = filtered

    # Nested schema validation
    if spec.nested_schema and isinstance(value, dict):
        value = validate_schema(value, spec.nested_schema, path)

    # Custom validation
    if spec.validate and not spec.validate(value):
        raise ValueError(f"SCHEMA: Field '{path}' failed custom validation")

    return value


def validate_schema(
    data: dict,
    schema: dict[str, FieldSpec],
    base_path: str = "",
) -> dict:
    """Validate a dict against a schema.

    Args:
        data: The dict to validate
        schema: Field name -> FieldSpec mapping
        base_path: Base path for nested error messages

    Returns:
        Dict with validated and coerced values

    Raises:
        TypeError: If data is not a dict
        ValueError: If required fields are missing
        TypeError: If field types don't match
    """
    if not isinstance(data, dict):
        raise TypeError(
            f"SCHEMA: Expected dict at '{base_path or 'root'}', got {type(data).__name__}",
        )

    result = {}

    for field_name, spec in schema.items():
        path = f"{base_path}.{field_name}" if base_path else field_name

        if field_name not in data:
            if spec.required == Required.YES:
                raise ValueError(f"SCHEMA: Required field '{path}' is missing")
            result[field_name] = spec.default
            continue

        result[field_name] = validate_field(data[field_name], spec, path)

    return result


def _coerce_type(value: Any, expected: type | tuple[type, ...], path: str) -> Any:
    """Attempt to coerce value to expected type."""
    # Handle tuple of types - try each in order
    if isinstance(expected, tuple):
        for t in expected:
            try:
                return _coerce_single_type(value, t, path)
            except (TypeError, ValueError):
                continue
        raise TypeError(
            f"SCHEMA: Cannot coerce '{path}' value {type(value).__name__} "
            f"to any of {_type_name(expected)}",
        )
    return _coerce_single_type(value, expected, path)


def _coerce_single_type(value: Any, expected: type, path: str) -> Any:
    """Coerce to a single type."""
    try:
        if expected == int:
            result = int(value)
            # Warn on precision loss from float
            if isinstance(value, float) and value != result:
                warnings.warn(
                    f"SCHEMA: Precision loss coercing '{path}' from float {value} to int {result}",
                    stacklevel=2,
                )
            return result
        if expected == float:
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("Coercion to float produced non-finite value")
            return result
        if expected == str:
            return str(value)
        if expected == bool:
            return bool(value)
        if expected == list:
            return list(value)
        if expected == dict:
            if not isinstance(value, dict):
                raise TypeError("Cannot coerce non-dict to dict")
            return dict(value)
        # Can't coerce to complex types
        raise TypeError(f"Cannot coerce to {expected.__name__}")
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"SCHEMA: Cannot coerce '{path}' value {value!r} ({type(value).__name__}) "
            f"to {expected.__name__}: {e}",
        ) from e


def _is_finite(value: Any) -> bool:
    """Check if a value is finite (not NaN/Inf)."""
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True  # Non-numeric values pass through


def _type_name(t: type | tuple[type, ...]) -> str:
    """Get readable name for type or tuple of types."""
    if isinstance(t, tuple):
        return " | ".join(x.__name__ for x in t)
    return t.__name__


# ============================================================================
# Pre-defined schemas for QGRE checkpoint boundaries
# ============================================================================


def non_negative(value: Any) -> bool:
    """Validate that numeric value is non-negative."""
    if isinstance(value, (int, float)):
        return value >= 0
    return True


def positive(value: Any) -> bool:
    """Validate that numeric value is positive."""
    if isinstance(value, (int, float)):
        return value > 0
    return True


def positive_finite(value: Any) -> bool:
    """Validate that numeric value is positive and finite."""
    if isinstance(value, (int, float)):
        return value > 0 and math.isfinite(value)
    return True


def non_negative_finite(value: Any) -> bool:
    """Validate that numeric value is non-negative and finite."""
    if isinstance(value, (int, float)):
        return value >= 0 and math.isfinite(value)
    return True


# GameState schema
GAME_STATE_SCHEMA: dict[str, FieldSpec] = {
    "step_count": FieldSpec(int, Required.NO, default=0, validate=non_negative),
    "mastery_threshold": FieldSpec(float, Required.NO, default=0.8, validate=positive_finite),
    "phase_history": FieldSpec(list, Required.NO, default=[]),
    "stagnation_timeout": FieldSpec(int, Required.NO, default=200, validate=positive),
    "plateau_window": FieldSpec(int, Required.NO, default=50, validate=positive),
    "plateau_threshold": FieldSpec(float, Required.NO, default=0.02, validate=non_negative_finite),
    "quality_window_size": FieldSpec(int, Required.NO, default=20, validate=positive),
    "tier_phases": FieldSpec(dict, Required.NO, default={}),
    "active_tiers": FieldSpec(list, Required.NO, default=["default"]),
    "tier_steps_at_phase_start": FieldSpec(dict, Required.NO, default={}),
    "tier_mastery": FieldSpec(dict, Required.NO, default={}),
    # Old format field - converted during migration
    "step_mastery": FieldSpec(dict, Required.NO, default=None),
    "phase": FieldSpec(int, Required.NO, default=1),  # Old format fallback
}

# TrainerState schema (complete — all fields from dataclass)
TRAINER_STATE_SCHEMA: dict[str, FieldSpec] = {
    "global_step": FieldSpec(int, Required.YES, validate=non_negative),
    "accumulated_loss": FieldSpec(float, Required.NO, default=0.0),
    "accumulation_count": FieldSpec(int, Required.NO, default=0, validate=non_negative),
    "accumulated_samples": FieldSpec(int, Required.NO, default=0, validate=non_negative),
    "resumed_mid_accumulation": FieldSpec(bool, Required.NO, default=False),
    "fused_validated": FieldSpec(bool, Required.NO, default=False),
    "needs_weight_sync": FieldSpec(bool, Required.NO, default=False),
    # RNG state is object type (torch tensor), validated separately
    "rng_state": FieldSpec((object, type(None)), Required.NO, default=None, coerce=False),
    "cuda_rng_state": FieldSpec((object, type(None)), Required.NO, default=None, coerce=False),
}

# DataLoaderState schema (matches DataLoaderState dataclass fields)
DATALOADER_STATE_SCHEMA: dict[str, FieldSpec] = {
    "epoch": FieldSpec(int, Required.NO, default=0, validate=non_negative),
    "step_in_epoch": FieldSpec(int, Required.NO, default=0, validate=non_negative),
    "total_steps": FieldSpec(int, Required.NO, default=0, validate=non_negative),
    "priority_weights": FieldSpec(
        (list, dict, type(None)), Required.NO, default=None, filter_nan=True
    ),
    "difficulty_gate": FieldSpec((tuple, dict, type(None)), Required.NO, default=None),
}


def convert_difficulty_gate(raw: dict | tuple | None) -> tuple[set[str], str] | None:
    """Convert difficulty_gate from dict format (checkpoint) to tuple[set[str], str].

    Handles legacy checkpoint format where difficulty_gate was stored as:
        {"allowed_difficulties": [...], "difficulty_column": "..."}
    Converts to runtime format:
        (set of allowed difficulties, column name)
    """
    if raw is None:
        return None
    if isinstance(raw, dict):
        allowed = raw.get("allowed_difficulties", [])
        col = raw.get("difficulty_column", "")
        return (set(allowed), col)
    if isinstance(raw, tuple):
        return raw
    return None


# AdvantageEstimatorState schema
ADVANTAGE_ESTIMATOR_STATE_SCHEMA: dict[str, FieldSpec] = {
    "state_dict": FieldSpec(dict, Required.NO, default=None),
}

# WeightLoaderState schema
WEIGHT_LOADER_STATE_SCHEMA: dict[str, FieldSpec] = {
    "load_lora_called": FieldSpec(bool, Required.NO, default=False),
    "initialized": FieldSpec(bool, Required.NO, default=False),
    "cleaned_up": FieldSpec(bool, Required.NO, default=False),
    "lora_request_id": FieldSpec(int, Required.NO, default=None),
    "lifecycle": FieldSpec(str, Required.NO, default="uninitialized"),
}


# Priority weights schema (for data.py)
def validate_priority_weights(weights: dict) -> bool:
    """Validate priority weights are non-negative and finite."""
    for v in weights.values():
        if not isinstance(v, (int, float)):
            return False
        if v < 0 or not math.isfinite(v):
            return False
    return True


PRIORITY_WEIGHTS_SCHEMA: dict[str, FieldSpec] = {
    "weights": FieldSpec(dict, Required.YES, validate=validate_priority_weights),
}


# HintEntry schema (for hints.py)
def validate_hint_tokens(tokens: Any) -> bool:
    """Validate hint_tokens is a non-empty list of ints."""
    if not isinstance(tokens, list):
        return False
    if len(tokens) == 0:
        return False
    return all(isinstance(t, int) for t in tokens)


HINT_ENTRY_SCHEMA: dict[str, FieldSpec] = {
    "prompt_id": FieldSpec(int, Required.YES),
    "span_id": FieldSpec(str, Required.YES),
    "hint_tokens": FieldSpec(list, Required.YES, validate=validate_hint_tokens),
    "mastery_at_flag": FieldSpec(float, Required.YES, validate=non_negative_finite),
    "flagged_step": FieldSpec(int, Required.YES, validate=non_negative),
    "success_count": FieldSpec(int, Required.NO, default=0, validate=non_negative),
    "total_uses": FieldSpec(int, Required.NO, default=0, validate=non_negative),
}


HINT_REGISTRY_SCHEMA: dict[str, FieldSpec] = {
    "mastery_threshold": FieldSpec(float, Required.NO, default=0.8, validate=positive_finite),
    "success_streak_to_clear": FieldSpec(int, Required.NO, default=2, validate=positive),
    "hints": FieldSpec(list, Required.NO, default=[]),
}
