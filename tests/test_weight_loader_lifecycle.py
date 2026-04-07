"""Test WeightLoader lifecycle state machine."""

import pytest
from torch import nn

from qgre.types import WeightLoaderLifecycle
from qgre.weight_load import WeightLoader


class MockModel(nn.Module):
    """Minimal mock for WeightLoader init."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


class TestWeightLoaderLifecycle:
    """Test lifecycle state transitions."""

    def test_initial_state_is_uninitialized(self):
        """WeightLoader starts in UNINITIALIZED state."""
        model = MockModel()
        loader = WeightLoader(model)
        assert loader.lifecycle == WeightLoaderLifecycle.UNINITIALIZED

    def test_transition_to_loading(self):
        """Can transition from UNINITIALIZED to LOADING."""
        model = MockModel()
        loader = WeightLoader(model)

        loader._transition_to_loading()
        assert loader.lifecycle == WeightLoaderLifecycle.LOADING

    def test_transition_to_loading_from_error(self):
        """Can transition from ERROR to LOADING (retry path)."""
        model = MockModel()
        loader = WeightLoader(model)

        # Manually set error state
        loader._transition_to_error()
        assert loader.lifecycle == WeightLoaderLifecycle.ERROR

        # Can retry from error
        loader._transition_to_loading()
        assert loader.lifecycle == WeightLoaderLifecycle.LOADING

    def test_transition_to_ready(self):
        """Can transition from LOADING to READY."""
        model = MockModel()
        loader = WeightLoader(model)

        loader._transition_to_loading()
        loader._transition_to_ready()
        assert loader.lifecycle == WeightLoaderLifecycle.READY

    # ELI-001: Removed test_transition_to_dropout and test_transition_back_from_dropout_to_ready
    # DROPOUT_ACTIVE state was removed - dropout is tracked externally by lora_dropout module

    def test_invalid_transition_raises(self):
        """Invalid transitions raise RuntimeError."""
        model = MockModel()
        loader = WeightLoader(model)

        # Can't go from UNINITIALIZED directly to READY
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            loader._transition_to_ready()

    def test_reset_state_transitions_to_uninitialized(self):
        """reset_state() transitions back to UNINITIALIZED."""
        model = MockModel()
        loader = WeightLoader(model)

        # Go through full lifecycle
        loader._transition_to_loading()
        loader._transition_to_ready()

        # Reset
        loader.reset_state()
        assert loader.lifecycle == WeightLoaderLifecycle.UNINITIALIZED
        assert loader._lora_request is None

    def test_error_transition_clears_lora_request(self):
        """Transition to ERROR clears _lora_request."""
        model = MockModel()
        loader = WeightLoader(model)

        loader._lora_request = "mock_request"
        loader._transition_to_error()

        assert loader.lifecycle == WeightLoaderLifecycle.ERROR
        assert loader._lora_request is None


class TestLegacyCompatibility:
    """Test legacy property compatibility."""

    def test_direct_ready_false_when_uninitialized(self):
        """_direct_ready is False when UNINITIALIZED."""
        model = MockModel()
        loader = WeightLoader(model)
        assert loader._direct_ready is False

    def test_direct_ready_false_when_loading(self):
        """_direct_ready is False when LOADING."""
        model = MockModel()
        loader = WeightLoader(model)
        loader._transition_to_loading()
        assert loader._direct_ready is False

    def test_direct_ready_true_when_ready(self):
        """_direct_ready is True when READY."""
        model = MockModel()
        loader = WeightLoader(model)
        loader._transition_to_loading()
        loader._transition_to_ready()
        assert loader._direct_ready is True

    # ELI-001: Removed test_direct_ready_true_when_dropout_active
    # DROPOUT_ACTIVE state was removed from the state machine

    def test_load_lora_called_reflects_lifecycle(self):
        """_load_lora_called is True when past UNINITIALIZED."""
        model = MockModel()
        loader = WeightLoader(model)

        assert loader._load_lora_called is False

        loader._transition_to_loading()
        assert loader._load_lora_called is True


class TestThreadSafety:
    """Test thread safety of state machine."""

    def test_has_lock_attribute(self):
        """WeightLoader has a threading lock."""
        import threading

        model = MockModel()
        loader = WeightLoader(model)
        assert hasattr(loader, "_lock")
        assert isinstance(loader._lock, type(threading.Lock()))

    def test_cache_stale_flag_initialized(self):
        """WeightLoader has cache_potentially_stale flag."""
        model = MockModel()
        loader = WeightLoader(model)
        assert hasattr(loader, "_cache_potentially_stale")
        assert loader._cache_potentially_stale is False
