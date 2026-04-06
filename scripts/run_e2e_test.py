#!/usr/bin/env python3
"""Real end-to-end QGRE engine test with actual model generation.

Loads Qwen3-1.7B, generates 8 completions per step (SPO mode, n=1),
scores with partial reward function, trains for 10 steps.
Outputs full untruncated prompt + completion + per-step scores.
"""

import sys
import warnings
from pathlib import Path

import torch


# Suppress the Unsloth pad token warning (cosmetic — PAD=151669 is correct)
warnings.filterwarnings("ignore", message=".*does not have a padding token.*")
warnings.filterwarnings("ignore", message=".*PAD_TOKEN.*")

sys.path.insert(0, str(Path(__file__).parent.parent))

from qgre.config import QGREConfig
from qgre.data import PromptBatch
from qgre.generation import UnslothBackend
from qgre.segments import HYPERGRAPH_V1_STEP_QUALITIES
from qgre.trainer import QGRETrainer
from qgre.types import GameState, RewardResult


# --- 8 diverse prompts that should trigger hypergraph-style output ---
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
    """Partial reward function — different per-step scores based on content."""
    scores = {}

    # Step 1: format quality (does it have step tags?)
    has_step1 = "<step1" in completion.lower() or "step 1" in completion.lower()
    has_step2 = "<step2" in completion.lower() or "step 2" in completion.lower()
    has_step3 = "<step3" in completion.lower() or "step 3" in completion.lower()
    has_step4 = "<step4" in completion.lower() or "step 4" in completion.lower()

    # Step 1 qualities — format
    scores["q_format_tags"] = 1.0 if has_step1 else 0.0
    scores["q_tag_content"] = 0.8 if has_step1 and len(completion) > 100 else 0.2
    scores["q_node_in_prompt"] = (
        0.7 if any(w in completion.lower() for w in prompt.lower().split()[:3]) else 0.3
    )
    scores["q_node_format"] = 0.9 if has_step1 else 0.1
    scores["q_node_length"] = min(1.0, len(completion) / 500)

    # Step 2 qualities — grounding
    scores["q_chain_s2_refs_s1"] = 0.6 if has_step2 and has_step1 else 0.1

    # Step 3 qualities — coherence
    scores["q_chain_s3_refs_s2"] = 0.5 if has_step3 and has_step2 else 0.1
    scores["q_self_consistency"] = 0.7 if has_step3 else 0.2

    # Step 4 qualities — accuracy
    scores["q_step4_valid_json"] = 0.4 if has_step4 and "{" in completion else 0.0
    scores["q_step4_has_keys"] = 0.3 if has_step4 else 0.0
    scores["q_existence_correct"] = 0.5 if has_step4 else 0.0
    scores["q_archetype_correct"] = 0.4 if has_step4 else 0.0
    scores["q_node_f1"] = 0.3 if has_step4 else 0.0

    # Global
    scores["q_eos_correct"] = 1.0

    # Overall: weighted mean of step 1 qualities (phase 1)
    step1_keys = [
        "q_format_tags",
        "q_tag_content",
        "q_node_in_prompt",
        "q_node_format",
        "q_node_length",
    ]
    reward = sum(scores[k] for k in step1_keys) / len(step1_keys)

    return RewardResult(reward=reward, scores=scores, phase=1)


def write_step_table_html(
    html_file,
    step_num: int,
    prompts: list[str],
    completions: list[str],
    reward_results: list[RewardResult],
    step_qualities: dict,
    token_counts: list[int] | None = None,
):
    """Write one step's results as an HTML table — full text, no truncation."""
    import html as html_mod

    html_file.write(f"<h2>Step {step_num}</h2>\n")
    html_file.write(
        '<table border="1" cellpadding="8" cellspacing="0" style="border-collapse:collapse; width:100%; table-layout:fixed; word-wrap:break-word;">\n'
    )

    # Header
    html_file.write('<tr style="background:#333; color:#fff;">')
    html_file.write('<th style="width:5%">#</th>')
    html_file.write('<th style="width:20%">Prompt</th>')
    html_file.write('<th style="width:40%">Completion</th>')
    for sn in sorted(step_qualities.keys()):
        html_file.write(f'<th style="width:7%">S{sn}</th>')
    html_file.write('<th style="width:7%">Overall</th>')
    html_file.write('<th style="width:5%">Tokens</th>')
    html_file.write("</tr>\n")

    # Rows
    for i, (prompt, completion, rr) in enumerate(
        zip(prompts, completions, reward_results, strict=False)
    ):
        bg = "#f9f9f9" if i % 2 == 0 else "#ffffff"
        html_file.write(f'<tr style="background:{bg}; vertical-align:top;">')
        html_file.write(f"<td>{i+1}</td>")
        html_file.write(f'<td style="white-space:pre-wrap;">{html_mod.escape(prompt)}</td>')
        html_file.write(
            f'<td style="white-space:pre-wrap; font-size:12px;">{html_mod.escape(completion)}</td>'
        )

        for sn in sorted(step_qualities.keys()):
            qualities = step_qualities[sn]
            active_scores = [rr.scores.get(q, 0.0) for q in qualities]
            step_mean = sum(active_scores) / max(len(active_scores), 1)
            # Color: green if > 0.5, yellow if > 0.2, red otherwise
            color = "#2d7" if step_mean > 0.5 else "#da2" if step_mean > 0.2 else "#d44"
            tooltip = "&#10;".join(f"{q}: {rr.scores.get(q, 0.0):.2f}" for q in qualities)
            html_file.write(
                f'<td style="text-align:center; color:{color}; font-weight:bold;" title="{tooltip}">{step_mean:.3f}</td>'
            )

        overall_color = "#2d7" if rr.reward > 0.5 else "#da2" if rr.reward > 0.2 else "#d44"
        html_file.write(
            f'<td style="text-align:center; color:{overall_color}; font-weight:bold;">{rr.reward:.3f}</td>'
        )
        n_tokens = token_counts[i] if token_counts else 0
        html_file.write(f'<td style="text-align:center;">{n_tokens}</td>')
        html_file.write("</tr>\n")

    html_file.write("</table>\n")


def main():
    print("=" * 80)
    print("  QGRE Engine — Real End-to-End Test")
    print("  Model: Qwen3-1.7B (4-bit quantized)")
    print("  Mode: SPO (n=1, 8 prompts per step)")
    print("  Steps: 10")
    print("=" * 80)

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. This test requires a GPU.")
        sys.exit(1)

    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {gpu_mem:.1f} GB")
    print(
        f"Free: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3:.1f} GB"
    )

    # Load config
    config = QGREConfig.from_yaml("examples/hypergraph/config.yaml")
    config.training.total_steps = 10
    config.training.save_freq = 100  # Don't save during test
    config.generation.max_tokens = 512  # Shorter for test speed
    config.algorithm.mode = "spo"

    # Load model via UnslothBackend
    print("\nLoading Qwen3-1.7B with Unsloth + vLLM...")
    backend = UnslothBackend(config.model, config.generation)
    model, tokenizer = backend.load()

    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"Stop tokens: {config.generation.stop_token_ids}")

    vram_after_load = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after model load: {vram_after_load:.2f} GB")

    # Create trainer
    game_state = GameState()
    trainer = QGRETrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=score_completion,
        config=config,
        generation_backend=backend,
        game_state=game_state,
        step_qualities=HYPERGRAPH_V1_STEP_QUALITIES,
    )
    trainer.setup_optimizer()

    step_qualities = HYPERGRAPH_V1_STEP_QUALITIES

    # HTML output file — proper table, no truncation
    output_path = Path("output/e2e_results.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_file = open(output_path, "w")
    html_file.write('<!DOCTYPE html>\n<html><head><meta charset="utf-8">\n')
    html_file.write("<title>QGRE E2E Test Results</title>\n")
    html_file.write(
        "<style>body{font-family:monospace;margin:20px;background:#1a1a1a;color:#eee;} "
    )
    html_file.write("table{border:1px solid #555;margin-bottom:30px;} ")
    html_file.write("th{background:#333;padding:8px;text-align:left;} ")
    html_file.write("td{padding:8px;border:1px solid #444;vertical-align:top;} ")
    html_file.write(
        "h1{color:#6cf;} h2{color:#9cf;border-bottom:1px solid #555;padding-bottom:5px;} "
    )
    html_file.write(
        ".metric{display:inline-block;margin:0 15px 5px 0;padding:4px 10px;background:#2a2a2a;border-radius:4px;} "
    )
    html_file.write("</style></head><body>\n")
    html_file.write("<h1>QGRE Engine — E2E Test Results</h1>\n")
    html_file.write(f"<p>Model: {config.model.path} | Mode: {config.algorithm.mode} | ")
    html_file.write(
        f"Max tokens: {config.generation.max_tokens} | Batch: {len(PROMPTS)} prompts</p>\n"
    )

    print(f"Writing HTML results to {output_path}")

    # Training loop
    for step in range(10):
        print(f"Step {step + 1}/10 — generating...")
        backend.set_inference_mode()

        # Apply chat template (same as QGREDataLoader) so the model sees proper
        # <|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n structure.
        # Without this, raw text produces unconditioned garbage.
        chat_token_ids = []
        for p in PROMPTS:
            messages = [{"role": "user", "content": p}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            chat_token_ids.append(ids)

        # Left-pad to max length
        max_len = min(max(len(ids) for ids in chat_token_ids), config.data.max_prompt_length)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(PROMPTS), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros(len(PROMPTS), max_len, dtype=torch.long)
        for i, ids in enumerate(chat_token_ids):
            ids = ids[-max_len:]  # Truncate from left if too long
            input_ids[i, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, max_len - len(ids) :] = 1
        input_ids = input_ids.to(model.device if hasattr(model, "device") else "cuda")
        attention_mask = attention_mask.to(input_ids.device)

        gen_output = backend.generate(input_ids, attention_mask)

        reward_results = []
        for i in range(len(gen_output.texts)):
            rr = score_completion(PROMPTS[i], gen_output.texts[i])
            reward_results.append(rr)

        # Write HTML table for this step
        token_counts = [len(t) for t in gen_output.token_ids]
        write_step_table_html(
            html_file,
            step + 1,
            PROMPTS,
            gen_output.texts,
            reward_results,
            step_qualities,
            token_counts,
        )

        # Train
        print(f"Step {step + 1}/10 — training...")
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

        vram_now = torch.cuda.memory_allocated() / 1024**3
        vram_peak = torch.cuda.max_memory_allocated() / 1024**3

        # Write step metrics to HTML
        html_file.write('<div style="margin:10px 0 20px 0;">')
        html_file.write(f'<span class="metric">Loss: {metrics.get("loss", 0.0):.6f}</span>')
        html_file.write(
            f'<span class="metric">Reward: {metrics.get("reward/mean", 0.0):.3f}</span>'
        )
        html_file.write(f'<span class="metric">Phase: {game_state.phase}</span>')
        html_file.writelines(f'<span class="metric">Mastery S{sn}: {game_state.get_step_mastery(sn):.3f}</span>' for sn in sorted(step_qualities.keys()))
        html_file.write(f'<span class="metric">VRAM: {vram_now:.2f}/{vram_peak:.2f} GB</span>')
        html_file.write("</div>\n")

        print(
            f"  loss={metrics.get('loss', 0.0):.6f} reward={metrics.get('reward/mean', 0.0):.3f} "
            f"phase={game_state.phase} VRAM={vram_now:.2f}/{vram_peak:.2f} GB"
        )

    # Summary
    html_file.write("<h2>Summary</h2>\n")
    html_file.write(f"<p>Final phase: {game_state.phase} | Steps: {trainer.global_step} | ")
    html_file.write(f"VRAM peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB</p>\n")
    html_file.write("</body></html>\n")
    html_file.close()

    print(f"\n{'='*80}")
    print("  TEST COMPLETE")
    print(f"  Final phase: {game_state.phase}")
    print(f"  Steps trained: {trainer.global_step}")
    print(f"  VRAM peak: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"  HTML results: {output_path.resolve()}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
