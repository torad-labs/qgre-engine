"""Tests for LoRA verification harness (Step 0g)."""

import tempfile
from pathlib import Path

import pytest

from qgre.lora_verify import LoRAVerifier


@pytest.fixture
def lora_dir_with_weights():
    """Create a temp dir with fake safetensors files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)
        (p / "adapter_model.safetensors").write_bytes(b"fake_weights_v1" * 100)
        (p / "adapter_config.json").write_text('{"r": 16}')
        yield p


@pytest.fixture
def lora_dir_empty():
    """Create a temp dir with no weight files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_hash_lora_dir(lora_dir_with_weights):
    """Hash produces consistent, non-empty result."""
    verifier = LoRAVerifier()
    h1 = verifier.hash_lora_dir(lora_dir_with_weights)
    h2 = verifier.hash_lora_dir(lora_dir_with_weights)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_hash_lora_dir_no_weights(lora_dir_empty):
    """No weight files → raises FileNotFoundError."""
    verifier = LoRAVerifier()
    with pytest.raises(FileNotFoundError, match="No weight files"):
        verifier.hash_lora_dir(lora_dir_empty)


def test_hash_lora_dir_nonexistent():
    """Nonexistent path → raises FileNotFoundError."""
    verifier = LoRAVerifier()
    with pytest.raises(FileNotFoundError):
        verifier.hash_lora_dir("/tmp/nonexistent_lora_path_xyz")


def test_verify_sync_passes(lora_dir_with_weights):
    """verify_sync after on_save with same dir → True."""
    verifier = LoRAVerifier()
    verifier.on_save(lora_dir_with_weights)
    assert verifier.verify_sync(lora_dir_with_weights) is True


def test_verify_sync_detects_mismatch(lora_dir_with_weights):
    """verify_sync after weights change → raises ValueError."""
    verifier = LoRAVerifier()
    verifier.on_save(lora_dir_with_weights)

    # Modify weights
    (lora_dir_with_weights / "adapter_model.safetensors").write_bytes(b"different_weights" * 100)

    with pytest.raises(ValueError, match="LoRA weight mismatch"):
        verifier.verify_sync(lora_dir_with_weights)


def test_should_recreate_engine():
    """Returns True every N steps."""
    verifier = LoRAVerifier(recreate_interval=3)

    assert verifier.should_recreate_engine() is False  # step 1
    assert verifier.should_recreate_engine() is False  # step 2
    assert verifier.should_recreate_engine() is True  # step 3 — recreate
    assert verifier.should_recreate_engine() is False  # step 4 (reset)
    assert verifier.should_recreate_engine() is False  # step 5
    assert verifier.should_recreate_engine() is True  # step 6 — recreate


def test_reset_recreate_counter():
    """reset_recreate_counter resets the step counter."""
    verifier = LoRAVerifier(recreate_interval=5)
    for _ in range(4):
        verifier.should_recreate_engine()

    verifier.reset_recreate_counter()
    # Should need 5 more steps, not 1
    assert verifier.should_recreate_engine() is False
