"""Unified sync state management for weight synchronization.

Implements the leverage point from hardening audit: replace scattered boolean flags
with a single SyncState object that owns all sync-related state and enforces valid
transitions atomically.

Before: _dropout_active, _restore_failed, _cache_potentially_stale, _initialized
        scattered across functions and modules with manual synchronization.

After: One SyncState object, explicit transition methods, impossible states
       unrepresentable.
"""

from __future__ import annotations

import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto


class SyncLifecycle(Enum):
    """Weight loader lifecycle states."""

    UNINITIALIZED = auto()  # Fresh state, needs first sync
    LOADING = auto()  # Currently syncing weights
    READY = auto()  # Weights synced, ready for generation
    ERROR = auto()  # Unrecoverable error occurred


@dataclass
class SyncState:
    """Unified state for weight synchronization.

    All sync-related state lives here. Transition methods enforce valid state
    changes and handle cleanup atomically. Thread-safe via internal lock.

    Usage:
        state = SyncState()

        # Enter dropout for generation
        state.enter_dropout()
        try:
            generate(...)
        finally:
            state.exit_dropout(success=True)

        # Sync weights
        state.begin_sync()
        try:
            sync_weights(...)
            state.complete_sync()
        except Exception:
            state.fail_sync()
            raise
    """

    # Core state flags
    dropout_active: bool = False
    restore_failed: bool = False
    cache_stale: bool = False
    initialized: bool = False
    lifecycle: SyncLifecycle = SyncLifecycle.UNINITIALIZED

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def enter_dropout(self) -> None:
        """Enter dropout state for generation.

        Raises:
            RuntimeError: If previous restore failed (weights corrupted)
        """
        with self._lock:
            if self.restore_failed:
                raise RuntimeError(
                    "Previous LoRA dropout restore failed. Weights are corrupted. "
                    "Cannot apply dropout again. Restart training from checkpoint."
                )
            if self.dropout_active:
                warnings.warn(
                    "LoRA dropout applied twice without exit_dropout() call between. "
                    "Previous dropout state may be stale.",
                    stacklevel=2,
                )
            self.dropout_active = True

    def exit_dropout(self, success: bool) -> None:
        """Exit dropout state after generation.

        Args:
            success: True if restore succeeded, False if it failed.
        """
        with self._lock:
            self.dropout_active = False
            if success:
                self.restore_failed = False  # Reset on successful restore
            else:
                self.restore_failed = True

    @contextmanager
    def dropout_context(self):
        """Context manager for dropout application.

        Wraps enter_dropout/exit_dropout to guarantee cleanup even if
        exceptions occur during the protected block — including
        BaseException subclasses like KeyboardInterrupt and SystemExit.
        This is the mechanism that makes "forgot to restore" structurally
        impossible.

        Uses try/finally with a success flag rather than try/except so
        that cleanup runs on ANY exit path (normal return, Exception,
        KeyboardInterrupt, SystemExit, GeneratorExit). The success flag
        is only set after the yield returns normally — any exception
        leaves it False and the restore is recorded as failed.

        Usage:
            with state.dropout_context():
                apply_dropout(model)
                generate(...)
        """
        self.enter_dropout()
        success = False
        try:
            yield
            success = True
        finally:
            self.exit_dropout(success=success)

    def can_sync(self) -> bool:
        """Check if sync is currently allowed.

        Returns:
            False if dropout is active or cache is stale, True otherwise.
        """
        with self._lock:
            return not self.dropout_active and not self.cache_stale

    def check_sync_allowed(self) -> None:
        """Check sync preconditions, raise if not met.

        Raises:
            RuntimeError: If cache is stale or dropout is active.
        """
        with self._lock:
            if self.cache_stale:
                raise RuntimeError(
                    "KV cache is potentially stale due to previous flush failure. "
                    "Generations may use corrupted cache. Call reset_for_engine_recreate()."
                )
            if self.dropout_active:
                raise RuntimeError(
                    "Cannot sync while LoRA dropout is active. "
                    "Call can_sync() first or disable dropout."
                )

    def begin_sync(self) -> None:
        """Begin sync operation. Transitions to LOADING state."""
        with self._lock:
            self.lifecycle = SyncLifecycle.LOADING

    def complete_sync(self, first_call: bool = False) -> None:
        """Complete sync operation successfully.

        Args:
            first_call: True if this was the first sync (sets initialized).
        """
        with self._lock:
            self.lifecycle = SyncLifecycle.READY
            if first_call:
                self.initialized = True

    def fail_sync(self) -> None:
        """Mark sync as failed. Clears dropout state to allow recovery."""
        with self._lock:
            self.lifecycle = SyncLifecycle.ERROR
            # Clear dropout flag to prevent blocking all future syncs
            self.dropout_active = False

    def mark_cache_stale(self) -> None:
        """Mark KV cache as potentially stale after flush failure."""
        with self._lock:
            self.cache_stale = True

    def reset_for_engine_recreate(self) -> None:
        """Reset state for engine recreation.

        Clears all transient state, transitions to UNINITIALIZED.
        Called when vLLM engine is recreated.
        """
        with self._lock:
            self.lifecycle = SyncLifecycle.UNINITIALIZED
            self.initialized = False
            self.cache_stale = False
            self.dropout_active = False
            # Note: restore_failed is NOT cleared - if weights are corrupted,
            # engine recreation doesn't fix that.

    def state_dict(self) -> dict:
        """Serialize state for checkpointing.

        Only serializes state that should persist across checkpoint/resume.
        Transient state (dropout_active, cache_stale) is not serialized.
        """
        with self._lock:
            return {
                "initialized": self.initialized,
                "restore_failed": self.restore_failed,
                "lifecycle": self.lifecycle.name,
            }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore state from checkpoint."""
        with self._lock:
            self.initialized = state_dict.get("initialized", False)
            self.restore_failed = state_dict.get("restore_failed", False)
            lifecycle_name = state_dict.get("lifecycle", "UNINITIALIZED")
            self.lifecycle = SyncLifecycle[lifecycle_name]
            # Transient state starts fresh
            self.dropout_active = False
            self.cache_stale = False
