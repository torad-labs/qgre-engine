"""Test WeightLoader lifecycle state machine."""

import pytest
from torch import nn

from qgre.sync_state import SyncLifecycle, SyncState
from qgre.weight_load import WeightLoader


class MockModel(nn.Module):
    """Minimal mock for WeightLoader init."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)


def make_loader() -> WeightLoader:
    """Construct a WeightLoader with a fresh SyncState for testing."""
    return WeightLoader(MockModel(), SyncState())


class TestWeightLoaderLifecycle:
    """Test lifecycle state transitions."""

    def test_initial_state_is_uninitialized(self):
        """WeightLoader starts in UNINITIALIZED state."""
        loader = make_loader()
        assert loader.lifecycle == SyncLifecycle.UNINITIALIZED

    def test_transition_to_loading(self):
        """Can transition from UNINITIALIZED to LOADING."""
        loader = make_loader()

        loader._transition_to_loading()
        assert loader.lifecycle == SyncLifecycle.LOADING

    def test_transition_to_loading_from_error(self):
        """Can transition from ERROR to LOADING (retry path)."""
        loader = make_loader()

        # Manually set error state
        loader._transition_to_error()
        assert loader.lifecycle == SyncLifecycle.ERROR

        # Can retry from error
        loader._transition_to_loading()
        assert loader.lifecycle == SyncLifecycle.LOADING

    def test_transition_to_ready(self):
        """Can transition from LOADING to READY."""
        loader = make_loader()

        loader._transition_to_loading()
        loader._transition_to_ready()
        assert loader.lifecycle == SyncLifecycle.READY

    # ELI-001: Removed test_transition_to_dropout and test_transition_back_from_dropout_to_ready
    # DROPOUT_ACTIVE state was removed - dropout is tracked externally by lora_dropout module

    def test_invalid_transition_raises(self):
        """Invalid transitions raise RuntimeError."""
        loader = make_loader()

        # Can't go from UNINITIALIZED directly to READY
        with pytest.raises(RuntimeError, match="Invalid state transition"):
            loader._transition_to_ready()

    def test_reset_state_transitions_to_uninitialized(self):
        """reset_state() transitions back to UNINITIALIZED."""
        loader = make_loader()

        # Go through full lifecycle
        loader._transition_to_loading()
        loader._transition_to_ready()

        # Reset
        loader.reset_state()
        assert loader.lifecycle == SyncLifecycle.UNINITIALIZED
        assert loader._lora_request is None

    def test_error_transition_clears_lora_request(self):
        """Transition to ERROR clears _lora_request."""
        loader = make_loader()

        loader._lora_request = "mock_request"
        loader._transition_to_error()

        assert loader.lifecycle == SyncLifecycle.ERROR
        assert loader._lora_request is None


class TestLegacyCompatibility:
    """Test legacy property compatibility."""

    def test_direct_ready_false_when_uninitialized(self):
        """_direct_ready is False when UNINITIALIZED."""
        loader = make_loader()
        assert loader._direct_ready is False

    def test_direct_ready_false_when_loading(self):
        """_direct_ready is False when LOADING."""
        loader = make_loader()
        loader._transition_to_loading()
        assert loader._direct_ready is False

    def test_direct_ready_true_when_ready(self):
        """_direct_ready is True when READY."""
        loader = make_loader()
        loader._transition_to_loading()
        loader._transition_to_ready()
        assert loader._direct_ready is True

    # ELI-001: Removed test_direct_ready_true_when_dropout_active
    # DROPOUT_ACTIVE state was removed from the state machine

    def test_load_lora_called_reflects_lifecycle(self):
        """_load_lora_called is True when past UNINITIALIZED."""
        loader = make_loader()

        assert loader._load_lora_called is False

        loader._transition_to_loading()
        assert loader._load_lora_called is True


class TestThreadSafety:
    """Test thread safety of state machine."""

    def test_has_lock_attribute(self):
        """WeightLoader has a threading lock for its own critical sections."""
        import threading

        loader = make_loader()
        assert hasattr(loader, "_lock")
        assert isinstance(loader._lock, type(threading.Lock()))

    def test_cache_stale_flag_lives_in_sync_state(self):
        """Cache staleness is tracked via the injected SyncState, not on the loader."""
        loader = make_loader()
        # _cache_potentially_stale was removed; the flag lives on SyncState now
        assert not hasattr(loader, "_cache_potentially_stale")
        assert loader._state.cache_stale is False
