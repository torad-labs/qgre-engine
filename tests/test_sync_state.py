"""Tests for unified SyncState."""

import pytest


class TestSyncStateTransitions:
    """Test state transition logic."""

    def test_enter_dropout_sets_flag(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        assert not state.dropout_active
        state.enter_dropout()
        assert state.dropout_active

    def test_exit_dropout_clears_flag(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.exit_dropout(success=True)
        assert not state.dropout_active
        assert not state.restore_failed

    def test_exit_dropout_failure_sets_restore_failed(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.exit_dropout(success=False)
        assert not state.dropout_active
        assert state.restore_failed

    def test_enter_dropout_after_failure_raises(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.restore_failed = True
        with pytest.raises(RuntimeError, match="Weights are corrupted"):
            state.enter_dropout()

    def test_successful_restore_resets_restore_failed(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.restore_failed = True
        # Manually clear to simulate recovery
        state.restore_failed = False
        state.enter_dropout()
        state.exit_dropout(success=True)
        assert not state.restore_failed

    def test_double_dropout_warns(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        with pytest.warns(UserWarning, match="twice without exit_dropout"):
            state.enter_dropout()


class TestSyncStateCanSync:
    """Test sync precondition checks."""

    def test_can_sync_when_clean(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        assert state.can_sync()

    def test_cannot_sync_when_dropout_active(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        assert not state.can_sync()

    def test_cannot_sync_when_cache_stale(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.mark_cache_stale()
        assert not state.can_sync()

    def test_check_sync_allowed_raises_on_stale(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.mark_cache_stale()
        with pytest.raises(RuntimeError, match="KV cache is potentially stale"):
            state.check_sync_allowed()


class TestSyncStateLifecycle:
    """Test lifecycle transitions."""

    def test_begin_sync_transitions_to_loading(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.begin_sync()
        assert state.lifecycle == SyncLifecycle.LOADING

    def test_complete_sync_transitions_to_ready(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.begin_sync()
        state.complete_sync()
        assert state.lifecycle == SyncLifecycle.READY

    def test_complete_sync_first_call_sets_initialized(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        assert not state.initialized
        state.begin_sync()
        state.complete_sync(first_call=True)
        assert state.initialized

    def test_fail_sync_transitions_to_error(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.begin_sync()
        state.fail_sync()
        assert state.lifecycle == SyncLifecycle.ERROR

    def test_fail_sync_clears_dropout_flag(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.fail_sync()
        assert not state.dropout_active


class TestSyncStateReset:
    """Test engine recreation reset."""

    def test_reset_clears_transient_state(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.enter_dropout()
        state.mark_cache_stale()
        state.begin_sync()
        state.complete_sync(first_call=True)

        state.reset_for_engine_recreate()

        assert state.lifecycle == SyncLifecycle.UNINITIALIZED
        assert not state.initialized
        assert not state.cache_stale
        assert not state.dropout_active

    def test_reset_preserves_restore_failed(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.restore_failed = True
        state.reset_for_engine_recreate()
        # restore_failed survives reset - engine recreation doesn't fix corrupted weights
        assert state.restore_failed


class TestSyncStateSerialization:
    """Test checkpoint serialization."""

    def test_state_dict_captures_persistent_state(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.begin_sync()
        state.complete_sync(first_call=True)
        state.restore_failed = True

        sd = state.state_dict()

        assert sd["initialized"] is True
        assert sd["restore_failed"] is True
        assert sd["lifecycle"] == "READY"

    def test_state_dict_excludes_transient_state(self):
        from qgre.sync_state import SyncState

        state = SyncState()
        state.enter_dropout()
        state.mark_cache_stale()

        sd = state.state_dict()

        assert "dropout_active" not in sd
        assert "cache_stale" not in sd

    def test_load_state_dict_restores_persistent_state(self):
        from qgre.sync_state import SyncLifecycle, SyncState

        state = SyncState()
        state.load_state_dict(
            {
                "initialized": True,
                "restore_failed": True,
                "lifecycle": "READY",
            }
        )

        assert state.initialized
        assert state.restore_failed
        assert state.lifecycle == SyncLifecycle.READY
        assert not state.dropout_active
        assert not state.cache_stale

    def test_roundtrip_serialization(self):
        from qgre.sync_state import SyncState

        state1 = SyncState()
        state1.begin_sync()
        state1.complete_sync(first_call=True)

        sd = state1.state_dict()

        state2 = SyncState()
        state2.load_state_dict(sd)

        assert state2.initialized == state1.initialized
        assert state2.lifecycle == state1.lifecycle
