"""Tests for schema validation at serialization boundaries."""

import math

import pytest

from qgre.schema import (
    GAME_STATE_SCHEMA,
    TRAINER_STATE_SCHEMA,
    FieldSpec,
    Required,
    validate_field,
    validate_schema,
)


class TestFieldValidation:
    """Test individual field validation."""

    def test_required_field_missing_raises(self):
        """Required field that is None should raise ValueError."""
        spec = FieldSpec(int, Required.YES)
        with pytest.raises(ValueError, match="Required field"):
            validate_field(None, spec, "test_field")

    def test_optional_field_missing_returns_default(self):
        """Optional field that is None should return default."""
        spec = FieldSpec(int, Required.NO, default=42)
        assert validate_field(None, spec, "test_field") == 42

    def test_type_coercion_string_to_int(self):
        """String '123' should coerce to int 123."""
        spec = FieldSpec(int, coerce=True)
        assert validate_field("123", spec, "test_field") == 123

    def test_type_coercion_float_to_int_warns(self):
        """Float to int should work but warn about precision loss."""
        spec = FieldSpec(int, coerce=True)
        with pytest.warns(UserWarning, match="Precision loss"):
            result = validate_field(3.7, spec, "test_field")
        assert result == 3

    def test_nan_filtering_in_list(self):
        """NaN values should be filtered from lists."""
        spec = FieldSpec(list, filter_nan=True)
        with pytest.warns(UserWarning, match="Filtered.*NaN"):
            result = validate_field([1.0, float("nan"), 2.0, float("inf")], spec, "test_field")
        assert result == [1.0, 2.0]

    def test_nan_filtering_in_dict(self):
        """NaN values should be filtered from dict values."""
        spec = FieldSpec(dict, filter_nan=True)
        with pytest.warns(UserWarning, match="Filtered.*NaN"):
            result = validate_field({"a": 1.0, "b": float("nan"), "c": 2.0}, spec, "test_field")
        assert result == {"a": 1.0, "c": 2.0}

    def test_type_mismatch_without_coercion_raises(self):
        """Type mismatch with coerce=False should raise TypeError."""
        spec = FieldSpec(int, coerce=False)
        with pytest.raises(TypeError, match="expected int"):
            validate_field("not_an_int", spec, "test_field")

    def test_custom_validation(self):
        """Custom validation function should be called."""
        spec = FieldSpec(int, validate=lambda x: x > 0)
        assert validate_field(5, spec, "test_field") == 5
        with pytest.raises(ValueError, match="failed custom validation"):
            validate_field(-1, spec, "test_field")


class TestSchemaValidation:
    """Test full schema validation."""

    def test_validates_simple_schema(self):
        """Basic schema validation should work."""
        schema = {
            "name": FieldSpec(str, Required.YES),
            "count": FieldSpec(int, Required.NO, default=0),
        }
        data = {"name": "test"}
        result = validate_schema(data, schema)
        assert result == {"name": "test", "count": 0}

    def test_rejects_non_dict(self):
        """Non-dict input should raise TypeError."""
        schema = {"field": FieldSpec(str)}
        with pytest.raises(TypeError, match="Expected dict"):
            validate_schema("not_a_dict", schema)

    def test_missing_required_field_raises(self):
        """Missing required field should raise ValueError."""
        schema = {
            "required_field": FieldSpec(str, Required.YES),
        }
        with pytest.raises(ValueError, match="Required field.*required_field.*missing"):
            validate_schema({}, schema)

    def test_path_in_error_messages(self):
        """Error messages should include the field path."""
        schema = {
            "nested": FieldSpec(
                dict,
                nested_schema={
                    "inner": FieldSpec(int, Required.YES),
                },
            ),
        }
        with pytest.raises(ValueError, match="nested.inner"):
            validate_schema({"nested": {}}, schema)


class TestGameStateSchema:
    """Test GameState schema validation."""

    def test_validates_minimal_game_state(self):
        """Minimal game state dict should validate."""
        data = {}  # All fields optional
        result = validate_schema(data, GAME_STATE_SCHEMA, "game_state")
        assert result["step_count"] == 0
        assert result["mastery_threshold"] == 0.8

    def test_coerces_float_step_count_to_int(self):
        """Float step_count should be coerced to int."""
        data = {"step_count": 100.0}
        result = validate_schema(data, GAME_STATE_SCHEMA, "game_state")
        assert result["step_count"] == 100
        assert isinstance(result["step_count"], int)

    def test_rejects_negative_step_count(self):
        """Negative step_count should fail validation."""
        data = {"step_count": -1}
        with pytest.raises(ValueError, match="failed custom validation"):
            validate_schema(data, GAME_STATE_SCHEMA, "game_state")


class TestTrainerStateSchema:
    """Test TrainerState schema validation."""

    def test_requires_global_step(self):
        """global_step is required."""
        with pytest.raises(ValueError, match="global_step.*missing"):
            validate_schema({}, TRAINER_STATE_SCHEMA)

    def test_validates_complete_trainer_state(self):
        """Complete trainer state should validate."""
        data = {
            "global_step": 100,
            "accumulated_loss": 0.5,
            "accumulated_samples": 10,
        }
        result = validate_schema(data, TRAINER_STATE_SCHEMA)
        assert result["global_step"] == 100
        assert result["accumulated_loss"] == 0.5
        assert result["accumulated_samples"] == 10


class TestBugClassPrevention:
    """Test that schema prevents the bug classes identified in harden."""

    def test_prevents_tuple_get_attributeerror(self):
        """Validates dict type before .items() call."""
        # R1-6: Tuple .get() AttributeError
        schema = {"data": FieldSpec(dict, Required.YES)}
        with pytest.raises(TypeError):
            validate_schema({"data": (1, 2, 3)}, schema)  # Tuple, not dict

    def test_prevents_nan_corruption(self):
        """NaN values are filtered before storage."""
        # R1-4, R2-3: NaN coercion silently accepted
        spec = FieldSpec(list, filter_nan=True)
        result = validate_field([0.5, float("nan"), 0.7], spec, "test")
        assert not any(math.isnan(v) for v in result if isinstance(v, float))

    def test_prevents_type_coercion_bugs(self):
        """Type coercion is explicit and warns on precision loss."""
        # R1-3: tier_steps_at_phase_start values not cast to int
        spec = FieldSpec(int, coerce=True)
        # String to int works
        assert validate_field("100", spec, "test") == 100
        # Float to int warns
        with pytest.warns(UserWarning):
            validate_field(100.5, spec, "test")

    def test_prevents_negative_weights(self):
        """Priority weights must be non-negative."""
        # R3-3: Priority weights accept negative/NaN
        from qgre.schema import validate_priority_weights

        assert validate_priority_weights({"a": 1.0, "b": 2.0}) is True
        assert validate_priority_weights({"a": -1.0}) is False
        assert validate_priority_weights({"a": float("nan")}) is False
