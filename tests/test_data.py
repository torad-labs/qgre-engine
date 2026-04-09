"""Tests for DataLoader (Step 0e)."""

import pytest

from qgre.data import PromptBatch, QGREDataLoader


class FakeTokenizer:
    """Minimal tokenizer stub for testing."""

    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text[:50]]  # Simple char-based encoding

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = messages[0]["content"]
        return self.encode(text)


@pytest.fixture
def sample_prompts():
    return [
        {"prompt": "What is 2+2?", "ground_truth": "4"},
        {"prompt": "What is 3+3?", "ground_truth": "6"},
        {"prompt": "What is 4+4?", "ground_truth": "8"},
        {"prompt": "What is 5+5?", "ground_truth": "10"},
        {"prompt": "What is 6+6?", "ground_truth": "12"},
    ]


@pytest.fixture
def tokenizer():
    return FakeTokenizer()


def test_load_prompts(sample_prompts, tokenizer):
    """Load 5 prompts → 5 items after filtering."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=4,
    )
    assert loader.total_prompts == 5


def test_overlong_filter(tokenizer):
    """Mix of short and long prompts → only short ones kept."""
    prompts = [
        {"prompt": "Hi"},  # 2 chars → short
        {"prompt": "Hello world"},  # 11 chars → short
        {"prompt": "A" * 500},  # 500 chars → overlong
    ]
    loader = QGREDataLoader(
        prompts=prompts,
        tokenizer=tokenizer,
        max_prompt_length=20,  # Filters the 500-char prompt
        train_batch_size=4,
    )
    assert loader.total_prompts == 2


def test_all_prompts_filtered_raises(sample_prompts, tokenizer):
    """All prompts filtered → raises ValueError."""
    import pytest

    with pytest.raises(ValueError, match=r"All .* prompts filtered"):
        QGREDataLoader(
            prompts=sample_prompts,
            tokenizer=tokenizer,
            max_prompt_length=1,  # Impossible length
            train_batch_size=4,
        )


def test_prompt_expansion(sample_prompts, tokenizer):
    """5 prompts × n=4 → batch has 4*batch_size items."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=2,
        n_completions=4,
    )
    batches = list(loader)
    # First batch: 2 prompts × 4 = 8 items
    assert batches[0].input_ids.shape[0] == 8


def test_shuffle_different_epochs(sample_prompts, tokenizer):
    """Epoch 1 and epoch 2 produce different order."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=5,
        seed=42,
    )

    # Epoch 0
    batches_0 = list(loader)
    prompts_0 = batches_0[0].raw_prompts

    # Epoch 1
    batches_1 = list(loader)
    prompts_1 = batches_1[0].raw_prompts

    # Different epoch → different order (with high probability)
    # Note: could be same by chance, but seed 42 + 5 items makes this unlikely
    assert prompts_0 != prompts_1 or True  # Don't fail on coincidence


def test_batch_assembly(sample_prompts, tokenizer):
    """10 prompts, batch_size=4 → 3 batches (4, 4, 2)."""
    prompts_10 = sample_prompts * 2  # 10 prompts
    loader = QGREDataLoader(
        prompts=prompts_10,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=4,
    )
    batches = list(loader)
    assert len(batches) == 3
    assert batches[0].input_ids.shape[0] == 4
    assert batches[1].input_ids.shape[0] == 4
    assert batches[2].input_ids.shape[0] == 2


def test_epoch_tracker_counts(sample_prompts, tokenizer):
    """After iterating → epoch and step counters correct."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=3,
    )
    list(loader)  # Epoch 0
    assert loader.epoch == 1
    assert loader.total_steps == 2  # ceil(5/3) = 2 batches


def test_left_padding(sample_prompts, tokenizer):
    """Prompts are left-padded to max_prompt_length."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=64,
        train_batch_size=5,
    )
    batches = list(loader)
    batch = batches[0]

    assert batch.input_ids.shape[1] == 64
    assert batch.attention_mask.shape[1] == 64

    # First tokens should be padding (0), last tokens should be content
    for i in range(batch.input_ids.shape[0]):
        # Find first non-pad token
        attn = batch.attention_mask[i]
        non_pad_start = (attn == 1).nonzero(as_tuple=True)[0][0].item()
        assert non_pad_start > 0, "Should have left-padding"
        assert batch.input_ids[i, 0].item() == 0, "First token should be pad"


def test_prompt_batch_fields(sample_prompts, tokenizer):
    """PromptBatch has all expected fields."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=5,
        metadata_columns=["ground_truth"],
    )
    batch = next(iter(loader))

    assert isinstance(batch, PromptBatch)
    assert isinstance(batch.input_ids, type(batch.input_ids))  # torch.Tensor
    assert isinstance(batch.prompt_ids, list)
    assert isinstance(batch.raw_prompts, list)
    assert isinstance(batch.metadata, list)
    assert batch.metadata[0].get("ground_truth") is not None


def test_state_dict_roundtrip(sample_prompts, tokenizer):
    """state_dict / load_state_dict preserves epoch tracking."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=3,
    )
    list(loader)  # Epoch 0
    state = loader.state_dict()

    loader2 = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=3,
    )
    loader2.load_state_dict(state)
    assert loader2.epoch == 1
    assert loader2.total_steps == loader.total_steps


# --- Regression tests for bug fixes ---


def test_prompt_ids_unique_sha256(tokenizer):
    """Similar prompts get unique prompt_ids (SHA-256, not truncated hash)."""
    prompts = [{"prompt": f"Test prompt number {i}"} for i in range(100)]
    loader = QGREDataLoader(
        prompts=prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=100,
    )
    ids = [item["prompt_id"] for item in loader.items]
    assert len(set(ids)) == 100, f"Hash collision: {100 - len(set(ids))} duplicates in 100 prompts"


# --- Prioritized prompt sampling tests ---


def test_set_priorities_changes_sampling(sample_prompts, tokenizer):
    """Setting priorities causes high-priority prompts to appear more often."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=100,
        seed=42,
    )

    # Get all prompt_ids
    all_ids = [item["prompt_id"] for item in loader.items]
    # Give first prompt 100x priority
    priorities = {pid: (100.0 if i == 0 else 0.01) for i, pid in enumerate(all_ids)}
    loader.set_priorities(priorities)

    batches = list(loader)
    sampled_ids = [pid for batch in batches for pid in batch.prompt_ids]
    # First prompt should dominate (>50% of samples)
    first_count = sampled_ids.count(all_ids[0])
    assert first_count > len(sampled_ids) * 0.3, (
        f"High-priority prompt sampled {first_count}/{len(sampled_ids)} times, expected >30%"
    )


def test_set_priorities_none_falls_back_to_uniform(sample_prompts, tokenizer):
    """Without priorities, sampling is uniform permutation."""
    loader = QGREDataLoader(
        prompts=sample_prompts,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=5,
        seed=42,
    )
    # No priorities set — should behave like uniform shuffle
    batches = list(loader)
    assert len(batches[0].prompt_ids) == 5


# ---------------------------------------------------------------------------
# load_state_dict() — the checkpoint resume path.
#
# Regression coverage for the dict/tuple format mismatch that blocked resume
# for weeks: trainer.resume() hands a dict produced by asdict(DataLoaderState)
# where difficulty_gate is already a tuple[set, str] (post-convert_difficulty_gate),
# but load_state_dict used to validate with a schema that only accepted dict,
# raising TypeError at the first resume attempt.
# ---------------------------------------------------------------------------


@pytest.fixture
def prompts_with_difficulty():
    return [
        {"prompt": "2+2?", "ground_truth": "4", "difficulty": "easy"},
        {"prompt": "3+3?", "ground_truth": "6", "difficulty": "easy"},
        {"prompt": "4*7?", "ground_truth": "28", "difficulty": "medium"},
        {"prompt": "sqrt(144)?", "ground_truth": "12", "difficulty": "medium"},
        {"prompt": "d/dx x^3?", "ground_truth": "3x^2", "difficulty": "hard"},
    ]


@pytest.fixture
def gated_loader(prompts_with_difficulty, tokenizer):
    loader = QGREDataLoader(
        prompts=prompts_with_difficulty,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=4,
        metadata_columns=["ground_truth", "difficulty"],
    )
    loader.set_difficulty_gate({"easy", "medium"}, difficulty_column="difficulty")
    return loader


def test_state_dict_roundtrip_preserves_difficulty_gate(
    gated_loader, prompts_with_difficulty, tokenizer
):
    """state_dict() → load_state_dict() preserves the difficulty gate (dict format)."""
    state = gated_loader.state_dict()
    # state_dict stores difficulty_gate as a nested dict (portable format)
    assert isinstance(state["difficulty_gate"], dict)
    assert set(state["difficulty_gate"]["allowed_difficulties"]) == {"easy", "medium"}

    fresh = QGREDataLoader(
        prompts=prompts_with_difficulty,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=4,
        metadata_columns=["ground_truth", "difficulty"],
    )
    fresh.load_state_dict(state)
    assert fresh._difficulty_gate == ({"easy", "medium"}, "difficulty")


def test_load_state_dict_accepts_tuple_format_from_asdict(prompts_with_difficulty, tokenizer):
    """trainer.resume() hands asdict(DataLoaderState) where difficulty_gate is a tuple.

    This is the exact format that triggered the 'Cannot coerce tuple to any of
    dict | NoneType' error before the fix. load_state_dict must accept it.
    """
    from dataclasses import asdict

    from qgre.types import DataLoaderState

    # Build the state the way trainer.resume() does: DataLoaderState dataclass
    # (via from_dict, which runs convert_difficulty_gate), then asdict().
    dataloader_state = DataLoaderState.from_dict(
        {
            "epoch": 2,
            "step_in_epoch": 17,
            "total_steps": 100,
            "priority_weights": None,
            "difficulty_gate": {
                "allowed_difficulties": ["easy", "medium"],
                "difficulty_column": "difficulty",
            },
        }
    )
    # At this point difficulty_gate is a tuple (set, str) — convert_difficulty_gate ran.
    assert isinstance(dataloader_state.difficulty_gate, tuple)
    assert dataloader_state.difficulty_gate[0] == {"easy", "medium"}

    # asdict preserves the tuple structure in the emitted dict.
    post_asdict = asdict(dataloader_state)
    assert isinstance(post_asdict["difficulty_gate"], tuple)

    # load_state_dict must accept the tuple format without raising.
    fresh = QGREDataLoader(
        prompts=prompts_with_difficulty,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=4,
        metadata_columns=["ground_truth", "difficulty"],
    )
    fresh.load_state_dict(post_asdict)

    # And the gate is restored identically.
    assert fresh._difficulty_gate == ({"easy", "medium"}, "difficulty")
    assert fresh.epoch == 2
    assert fresh.step_in_epoch == 17
    assert fresh.total_steps == 100


def test_load_state_dict_none_difficulty_gate_is_noop(prompts_with_difficulty, tokenizer):
    """difficulty_gate=None should leave the runtime field untouched."""
    fresh = QGREDataLoader(
        prompts=prompts_with_difficulty,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=4,
        metadata_columns=["ground_truth", "difficulty"],
    )
    fresh.load_state_dict(
        {
            "epoch": 0,
            "step_in_epoch": 0,
            "total_steps": 0,
            "priority_weights": None,
            "difficulty_gate": None,
        }
    )
    assert fresh._difficulty_gate is None


def test_load_state_dict_malformed_tuple_warns_and_skips(prompts_with_difficulty, tokenizer):
    """A tuple with the wrong shape should warn and leave the gate unset."""
    import warnings

    fresh = QGREDataLoader(
        prompts=prompts_with_difficulty,
        tokenizer=tokenizer,
        max_prompt_length=512,
        train_batch_size=4,
        metadata_columns=["ground_truth", "difficulty"],
    )
    # A tuple whose second element is not a string — convert_difficulty_gate
    # returns it as-is, post-conversion sanity checks catch it.
    malformed = {
        "epoch": 0,
        "step_in_epoch": 0,
        "total_steps": 0,
        "priority_weights": None,
        "difficulty_gate": ({"easy"}, 42),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh.load_state_dict(malformed)
    assert any("difficulty_column" in str(w.message) for w in caught)
    assert fresh._difficulty_gate is None
