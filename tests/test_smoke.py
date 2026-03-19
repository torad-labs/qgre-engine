"""GPU smoke tests — end-to-end training verification on RTX 5080.

These tests verify the full pipeline works on real hardware:
generate → score → advantages → loss → backward → update.

Run with: pytest tests/test_smoke.py -v -m gpu
"""

import tempfile
from pathlib import Path

import pytest
import torch

gpu = pytest.mark.gpu


@gpu
@pytest.mark.slow
def test_three_steps_no_crash():
    """Load model, run 3 full training steps on GPU.

    Assert: no crash, no nan loss, no OOM.
    This is THE integration test that validates everything works together.
    """
    from qgre.config import QGREConfig, ModelConfig, GenerationConfig
    from qgre.generation import UnslothBackend
    from qgre.trainer import QGRETrainer
    from qgre.data import QGREDataLoader, PromptBatch
    from qgre.types import RewardResult
    from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES as STEP_QUALITIES

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QGREConfig()
        cfg.model.path = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
        cfg.model.load_in_4bit = True
        cfg.model.fast_inference = True
        cfg.model.gpu_memory_utilization = 0.6
        cfg.model.lora_rank = 8
        cfg.model.lora_alpha = 16
        cfg.generation.temperature = 1.0
        cfg.generation.max_tokens = 64
        cfg.generation.top_k = 20
        cfg.generation.top_p = 0.95
        cfg.algorithm.mode = "spo"
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "checkpoints")
        cfg.logging.completion_dir = str(Path(tmpdir) / "completions")

        # Load model
        backend = UnslothBackend(cfg.model, cfg.generation)
        model, tokenizer = backend.load()

        # Simple prompts
        prompts = [
            {"prompt": "Analyze the relationship between modules A and B."},
            {"prompt": "Describe how function X depends on function Y."},
        ]

        loader = QGREDataLoader(
            prompts=prompts,
            tokenizer=tokenizer,
            max_prompt_length=512,
            train_batch_size=2,
            n_completions=1,
        )

        # Stub reward function
        def stub_reward(prompt, completion, meta=None):
            all_q = []
            for qs in STEP_QUALITIES.values():
                all_q.extend(qs)
            scores = {q: 0.5 + 0.1 * (hash(q + completion[:20]) % 5) for q in all_q}
            return RewardResult(reward=sum(scores.values()) / len(scores), scores=scores, phase=1)

        trainer = QGRETrainer(
            model=model,
            tokenizer=tokenizer,
            reward_fn=stub_reward,
            config=cfg,
            generation_backend=backend,
        )
        trainer.setup_optimizer()

        losses = []
        for step_num in range(3):
            for batch in loader:
                # Generate (inference mode — fast Unsloth kernels)
                backend.set_inference_mode()
                output = backend.generate(batch.input_ids.cuda(), batch.attention_mask.cuda())

                # Score
                reward_results = [
                    stub_reward(batch.raw_prompts[i], output.texts[i])
                    for i in range(len(output.texts))
                ]

                # Train step (training mode — no inplace ops, ref: unsloth #895)
                backend.set_training_mode()
                metrics = trainer.step(batch, output.token_ids, reward_results)
                losses.append(metrics["loss"])

                assert torch.isfinite(torch.tensor(metrics["loss"])), f"NaN/Inf loss at step {step_num}"
                break  # One batch per epoch for smoke test

        assert len(losses) == 3
        assert all(torch.isfinite(torch.tensor(l)) for l in losses)


@gpu
def test_lora_sync_verification():
    """After save_lora + load_lora, LoRA weights are present.

    Catches the ms-swift #8233 failure mode where LoRA is silently dropped.
    """
    import tempfile
    from pathlib import Path
    from qgre.config import ModelConfig, GenerationConfig
    from qgre.generation import UnslothBackend
    from qgre.lora_verify import LoRAVerifier

    model_cfg = ModelConfig(
        path="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
        load_in_4bit=True,
        fast_inference=True,
        gpu_memory_utilization=0.6,
        lora_rank=8,
        lora_alpha=16,
    )
    gen_cfg = GenerationConfig(max_tokens=32)

    backend = UnslothBackend(model_cfg, gen_cfg)
    backend.load()
    verifier = LoRAVerifier()

    with tempfile.TemporaryDirectory() as tmpdir:
        lora_path = Path(tmpdir) / "lora_test"
        backend.save_weights(lora_path)
        verifier.on_save(lora_path)

        backend.load_weights(lora_path)
        verifier.verify_sync(lora_path)  # Should not raise


@gpu
def test_vram_does_not_grow():
    """Run 5 steps, verify VRAM doesn't grow more than 10%.

    Catches the unsloth #3864 memory leak.
    """
    import tempfile
    from pathlib import Path
    from qgre.config import QGREConfig
    from qgre.generation import UnslothBackend
    from qgre.trainer import QGRETrainer
    from qgre.data import QGREDataLoader
    from qgre.types import RewardResult
    from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES as STEP_QUALITIES

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = QGREConfig()
        cfg.model.path = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
        cfg.model.load_in_4bit = True
        cfg.model.fast_inference = True
        cfg.model.gpu_memory_utilization = 0.6
        cfg.model.lora_rank = 8
        cfg.model.lora_alpha = 16
        cfg.generation.max_tokens = 32
        cfg.algorithm.mode = "spo"
        cfg.logging.checkpoint_dir = str(Path(tmpdir) / "ckpt")
        cfg.logging.completion_dir = str(Path(tmpdir) / "comp")

        backend = UnslothBackend(cfg.model, cfg.generation)
        model, tokenizer = backend.load()

        prompts = [{"prompt": f"Test prompt {i}"} for i in range(5)]
        loader = QGREDataLoader(
            prompts=prompts, tokenizer=tokenizer,
            max_prompt_length=256, train_batch_size=1, n_completions=1,
        )

        def stub_reward(prompt, completion, meta=None):
            all_q = []
            for qs in STEP_QUALITIES.values():
                all_q.extend(qs)
            return RewardResult(reward=0.5, scores={q: 0.5 for q in all_q}, phase=1)

        trainer = QGRETrainer(
            model=model, tokenizer=tokenizer, reward_fn=stub_reward,
            config=cfg, generation_backend=backend,
        )
        trainer.setup_optimizer()

        torch.cuda.synchronize()
        mem_start = torch.cuda.memory_allocated()

        for step in range(5):
            for batch in loader:
                backend.set_inference_mode()
                output = backend.generate(batch.input_ids.cuda(), batch.attention_mask.cuda())
                rrs = [stub_reward(batch.raw_prompts[0], output.texts[0])]
                backend.set_training_mode()
                trainer.step(batch, output.token_ids, rrs)
                break

        torch.cuda.synchronize()
        mem_end = torch.cuda.memory_allocated()

        growth = (mem_end - mem_start) / max(mem_start, 1)
        assert growth < 0.1, f"VRAM grew {growth:.1%} over 5 steps (threshold: 10%)"
