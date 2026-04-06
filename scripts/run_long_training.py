#!/usr/bin/env python3
"""Long training run — 50 steps × 8 completions × max_tokens=4096.

Verifies:
- No OOM at 4096 tokens (vs 512 in quick e2e test)
- No VRAM growth over 50 steps (unsloth #3864)
- Reward curve movement (loss should change after warm-start period)
- Phase advancement if mastery threshold reached
"""

import sys
import warnings
from pathlib import Path

import torch


warnings.filterwarnings("ignore", message=".*does not have a padding token.*")
warnings.filterwarnings("ignore", message=".*PAD_TOKEN.*")

sys.path.insert(0, str(Path(__file__).parent.parent))

from qgre.config import QGREConfig
from qgre.data import PromptBatch
from qgre.generation import UnslothBackend
from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES
from qgre.trainer import QGRETrainer
from qgre.types import GameState, RewardResult


PROMPTS = [
    "Analyze the relationship between shared_mutable_state and thread_safety in concurrent programming.",
    "Analyze the relationship between database_normalization and query_performance in SQL systems.",
    "Analyze the relationship between gradient_descent and loss_landscape in neural network training.",
    "Analyze the relationship between cache_invalidation and consistency in distributed systems.",
    "Analyze the relationship between type_inference and compile_time in statically typed languages.",
    "Analyze the relationship between backpressure and throughput in stream processing systems.",
    "Analyze the relationship between consensus_algorithm and partition_tolerance in distributed databases.",
    "Analyze the relationship between attention_mechanism and context_length in transformer models.",
]


def score_completion(prompt: str, completion: str, meta: dict | None = None) -> RewardResult:
    scores = {}
    has_step1 = "<step1" in completion.lower() or "step 1" in completion.lower()
    has_step2 = "<step2" in completion.lower() or "step 2" in completion.lower()
    has_step3 = "<step3" in completion.lower() or "step 3" in completion.lower()
    has_step4 = "<step4" in completion.lower() or "step 4" in completion.lower()

    scores["q_format_tags"] = 1.0 if has_step1 else 0.0
    scores["q_tag_content"] = 0.8 if has_step1 and len(completion) > 100 else 0.2
    scores["q_node_in_prompt"] = (
        0.7 if any(w in completion.lower() for w in prompt.lower().split()[:3]) else 0.3
    )
    scores["q_node_format"] = 0.9 if has_step1 else 0.1
    scores["q_node_length"] = min(1.0, len(completion) / 500)
    scores["q_chain_s2_refs_s1"] = 0.6 if has_step2 and has_step1 else 0.1
    scores["q_chain_s3_refs_s2"] = 0.5 if has_step3 and has_step2 else 0.1
    scores["q_self_consistency"] = 0.7 if has_step3 else 0.2
    scores["q_step4_valid_json"] = 0.4 if has_step4 and "{" in completion else 0.0
    scores["q_step4_has_keys"] = 0.3 if has_step4 else 0.0
    scores["q_existence_correct"] = 0.5 if has_step4 else 0.0
    scores["q_archetype_correct"] = 0.4 if has_step4 else 0.0
    scores["q_node_f1"] = 0.3 if has_step4 else 0.0
    scores["q_eos_correct"] = 1.0

    step1_keys = [
        "q_format_tags",
        "q_tag_content",
        "q_node_in_prompt",
        "q_node_format",
        "q_node_length",
    ]
    reward = sum(scores[k] for k in step1_keys) / len(step1_keys)
    return RewardResult(reward=reward, scores=scores, phase=1)


def main():
    N_STEPS = 50
    MAX_TOKENS = 4096

    print(f"{'='*80}")
    print(f"  Long Training Run — {N_STEPS} steps × 8 prompts × {MAX_TOKENS} max_tokens")
    print(f"{'='*80}")

    config = QGREConfig.from_yaml("examples/hypergraph/config.yaml")
    config.training.total_steps = N_STEPS
    config.training.save_freq = 999
    config.generation.max_tokens = MAX_TOKENS
    config.algorithm.mode = "spo"

    print("\nLoading model...")
    backend = UnslothBackend(config.model, config.generation)
    model, tokenizer = backend.load()

    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after load: {vram_after_load:.2f} GB")

    trainer = QGRETrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=score_completion,
        config=config,
        generation_backend=backend,
        game_state=GameState(),
        step_qualities=HYPERGRAPH_V1_STEP_QUALITIES,
    )
    trainer.setup_optimizer()

    # Track metrics
    losses = []
    rewards = []
    vram_usage = []

    for step in range(N_STEPS):
        # Generate
        backend.set_inference_mode()
        # Apply chat template (same as QGREDataLoader) so the model sees proper
        # <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n structure.
        # Without this, raw text produces unconditioned garbage.
        chat_token_ids = []
        for p in PROMPTS:
            messages = [{"role": "user", "content": p}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            chat_token_ids.append(ids)

        max_len = min(max(len(ids) for ids in chat_token_ids), config.data.max_prompt_length)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(PROMPTS), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(PROMPTS), max_len, dtype=torch.long)
        for i, ids in enumerate(chat_token_ids):
            ids = ids[-max_len:]
            input_ids[i, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - len(ids) :] = 1
        input_ids = input_ids.to("cuda")
        attention_mask = attention_mask.to("cuda")
        gen_output = backend.generate(input_ids, attention_mask)

        # Score
        reward_results = [
            score_completion(PROMPTS[i], gen_output.texts[i]) for i in range(len(gen_output.texts))
        ]

        # Train
        backend.set_training_mode()
        batch = PromptBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_ids=list(range(len(PROMPTS))),
            raw_prompts=PROMPTS,
            metadata=[{} for _ in PROMPTS],
        )
        metrics = trainer.step(
            batch,
            gen_output.token_ids,
            reward_results,
            generation_logprobs=gen_output.logprobs,
        )

        # Track
        loss = metrics.get("loss", 0.0)
        reward_mean = metrics.get("reward/mean", 0.0)
        vram_now = torch.cuda.memory_allocated() / 1024**3
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3
        losses.append(loss)
        rewards.append(reward_mean)
        vram_usage.append(vram_now)

        # Compact output every 5 steps
        if (step + 1) % 5 == 0 or step == 0:
            mastery = trainer.game_state.get_step_mastery(1)
            phase = trainer.game_state.phase
            avg_tokens = sum(len(t) for t in gen_output.token_ids) / len(gen_output.token_ids)
            print(
                f"  Step {step+1:3d}/{N_STEPS} | loss={loss:+.6f} | reward={reward_mean:.3f} | "
                f"mastery_s1={mastery:.3f} | phase={phase} | "
                f"avg_tokens={avg_tokens:.0f} | VRAM={vram_now:.2f}/{vram_peak:.2f} GB"
            )

    # Summary
    print(f"\n{'='*80}")
    print(f"  TRAINING COMPLETE — {N_STEPS} steps")
    print(f"{'='*80}")
    print(f"  Final phase: {trainer.game_state.phase}")
    print(f"  Final mastery (step 1): {trainer.game_state.get_step_mastery(1):.3f}")
    print(f"  Loss range: [{min(losses):.6f}, {max(losses):.6f}]")
    print(f"  Reward range: [{min(rewards):.3f}, {max(rewards):.3f}]")
    print(f"  VRAM start: {vram_usage[0]:.2f} GB")
    print(f"  VRAM end: {vram_usage[-1]:.2f} GB")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    vram_growth = vram_usage[-1] - vram_usage[0]
    print(
        f"  VRAM growth: {vram_growth:+.3f} GB ({'OK' if abs(vram_growth) < 0.5 else 'WARNING: leak?'})"
    )
    print(f"  Skipped steps: {sum(1 for m in losses if m == 0.0)}/{N_STEPS}")


if __name__ == "__main__":
    main()
