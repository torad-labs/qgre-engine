"""25-step gradient probe. No monkey-patching — just hooks on parameters.

Registers gradient hooks on lm_head and LoRA weights, runs normal training,
captures gradient magnitudes and weight deltas at every backward pass.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).parent.parent))

# Global storage — hooks write here during backward
GRAD_LOG = []
STEP_COUNTER = [0]
WEIGHT_SNAPSHOTS = {}


def install_hooks(model):
    """Register backward hooks on lm_head and representative LoRA params."""
    hooks = []
    tracked = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        track = False
        if "lm_head" in name:
            track = True
        elif any(f"layers.{i}." in name for i in [0, 14, 27]):
            if "lora_A" in name or "lora_B" in name:
                track = True

        if track:
            tracked[name] = param
            # Snapshot initial weights
            WEIGHT_SNAPSHOTS[name] = param.data.clone()

            def make_hook(n):
                def hook(grad):
                    GRAD_LOG.append(
                        {
                            "step": STEP_COUNTER[0],
                            "name": n,
                            "grad_norm": grad.norm().item(),
                            "grad_mean": grad.mean().item(),
                            "grad_max": grad.abs().max().item(),
                        }
                    )

                return hook

            hooks.append(param.register_hook(make_hook(name)))

    print(f"Gradient hooks installed on {len(tracked)} parameters")
    return hooks, tracked


def analyze_results(tracked_params, config_advantage_scale, lr):
    """Analyze captured gradients and weight changes."""
    print("\n" + "=" * 70)
    print("GRADIENT PROBE RESULTS")
    print("=" * 70)

    # Group grads by step
    by_step = defaultdict(list)
    for entry in GRAD_LOG:
        by_step[entry["step"]].append(entry)

    # Per-step summary
    for step in sorted(by_step.keys()):
        entries = by_step[step]
        lm_grads = [e for e in entries if "lm_head" in e["name"]]
        lora_grads = [e for e in entries if "lora" in e["name"]]

        lm_norm = lm_grads[0]["grad_norm"] if lm_grads else 0
        lm_max = lm_grads[0]["grad_max"] if lm_grads else 0
        lora_avg_norm = np.mean([e["grad_norm"] for e in lora_grads]) if lora_grads else 0
        lora_max_norm = max([e["grad_norm"] for e in lora_grads]) if lora_grads else 0

        print(f"\nStep {step}:")
        print(f"  lm_head grad: norm={lm_norm:.6f}  max={lm_max:.8f}")
        print(f"  LoRA grads:   avg_norm={lora_avg_norm:.6f}  max_norm={lora_max_norm:.6f}")

    # Weight deltas (initial → final)
    print("\n" + "=" * 70)
    print("WEIGHT CHANGES (initial → final after all steps)")
    print("=" * 70)

    physics_tokens = {"p": 79, "V": 53, "T": 51, "H": 39, "x": 87, "m": 76}
    hidden_norm = np.sqrt(2048)  # ~45.25

    for name, param in tracked_params.items():
        initial = WEIGHT_SNAPSHOTS[name]
        final = param.data
        delta = final - initial
        delta_norm = delta.norm().item()
        weight_norm = initial.norm().item()
        relative = delta_norm / max(weight_norm, 1e-10)

        print(f"\n  {name}:")
        print(f"    weight norm: {weight_norm:.4f}")
        print(f"    delta norm:  {delta_norm:.8f}")
        print(f"    relative:    {relative:.8f} ({relative*100:.6f}%)")

        if "lm_head" in name:
            # Per-row delta for physics tokens
            print("    Physics token lm_head row deltas:")
            for tok_name, tok_id in physics_tokens.items():
                row_delta = delta[tok_id].norm().item()
                logit_nudge = hidden_norm * row_delta
                print(
                    f"      '{tok_name}' (id={tok_id}): delta_norm={row_delta:.8f}, est_logit_nudge={logit_nudge:.6f}"
                )

            # Average per-row delta across all vocab
            per_row_delta = delta.norm(dim=1).mean().item()
            avg_logit_nudge = hidden_norm * per_row_delta
            print(f"    Avg per-row delta: {per_row_delta:.8f}")
            print(f"    Avg logit nudge:   {avg_logit_nudge:.6f}")

    # Compute amplification factor
    print("\n" + "=" * 70)
    print("AMPLIFICATION & OPTIMAL SCALE")
    print("=" * 70)

    n_steps = len(by_step)
    if n_steps == 0:
        print("No gradient data captured!")
        return

    # Average lm_head gradient norm per step
    lm_grad_norms = [e["grad_norm"] for e in GRAD_LOG if "lm_head" in e["name"]]
    avg_lm_grad = np.mean(lm_grad_norms) if lm_grad_norms else 0

    # lm_head weight change per step
    for name, param in tracked_params.items():
        if "lm_head" in name:
            initial = WEIGHT_SNAPSHOTS[name]
            total_delta = (param.data - initial).norm().item()
            per_step_delta = total_delta / max(n_steps, 1)

            # Per-row per-step
            per_row_total = (param.data - initial).norm(dim=1).mean().item()
            per_row_per_step = per_row_total / max(n_steps, 1)
            logit_nudge_per_step = hidden_norm * per_row_per_step

            logit_gap = 89.84  # measured earlier

            print(
                f"\n  lm_head analysis ({n_steps} steps, advantage_scale={config_advantage_scale}):"
            )
            print(f"    Avg gradient norm per step: {avg_lm_grad:.6f}")
            print(f"    Total weight delta norm: {total_delta:.8f}")
            print(f"    Per-step weight delta: {per_step_delta:.8f}")
            print(f"    Per-row per-step delta: {per_row_per_step:.10f}")
            print(f"    Logit nudge per step: {logit_nudge_per_step:.8f}")
            print(f"    Logit gap (physics tokens): {logit_gap:.2f}")
            print(f"    Nudge as % of gap per step: {logit_nudge_per_step/logit_gap*100:.6f}%")
            print(
                f"    Steps to close 10% of gap: {0.1*logit_gap/max(logit_nudge_per_step, 1e-15):.0f}"
            )

            # Optimal scale for different nudge targets
            if logit_nudge_per_step > 0:
                print("\n  Optimal advantage_scale for target nudge %:")
                for pct in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
                    target = pct / 100 * logit_gap
                    scale = config_advantage_scale * (target / logit_nudge_per_step)
                    print(f"    {pct:5.2f}% → advantage_scale = {scale:.6f}")

    # Save
    output_dir = Path("output/hamiltonian/study/gradient_probe")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "probe_results.json", "w") as f:
        json.dump(
            {
                "config": {"advantage_scale": config_advantage_scale, "lr": lr, "n_steps": n_steps},
                "grad_log": GRAD_LOG,
            },
            f,
            indent=2,
        )

    print(f"\nFull data saved to {output_dir / 'probe_results.json'}")


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "examples/hamiltonian/config.yaml"
    reward_module = (
        sys.argv[2] if len(sys.argv) > 2 else "examples.hamiltonian.reward_fn:hamiltonian_reward"
    )
    n_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    from qgre.__main__ import import_reward_fn, import_segmenter
    from qgre.config import QGREConfig

    config = QGREConfig.from_yaml(config_path)
    config.training.total_steps = n_steps
    config.training.save_freq = 9999

    reward_fn = import_reward_fn(reward_module)
    segmenter = import_segmenter(config.algorithm.segmenter)

    from qgre.generation import UnslothBackend

    backend = UnslothBackend(
        config.model, config.generation, max_prompt_length=config.data.max_prompt_length
    )
    model, tokenizer = backend.load()
    backend.restore_random_state(config.training.seed)

    from qgre.data import QGREDataLoader, load_prompts_from_parquet

    all_prompts = []
    for f in config.data.train_files:
        all_prompts.extend(load_prompts_from_parquet(f))

    n_completions = (
        config.algorithm.spo.n if config.algorithm.mode == "spo" else config.algorithm.grpo.n
    )
    dataloader = QGREDataLoader(
        prompts=all_prompts,
        tokenizer=tokenizer,
        max_prompt_length=config.data.max_prompt_length,
        train_batch_size=config.data.train_batch_size,
        n_completions=n_completions,
        prompt_column=config.data.prompt_column,
        metadata_columns=config.data.metadata_columns,
        system_prompt_column=config.data.system_prompt_column,
    )

    from qgre.trainer import QGRETrainer

    trainer = QGRETrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=config,
        generation_backend=backend,
        segmenter=segmenter,
    )

    # Install hooks AFTER trainer setup (model may be modified)
    hooks, tracked = install_hooks(trainer.model)

    print(f"\nRunning {n_steps}-step gradient probe...")
    print("=" * 70)

    try:
        trainer.train()
    except Exception as e:
        print(f"\nTraining stopped: {e}")

    for h in hooks:
        h.remove()

    analyze_results(tracked, config.algorithm.advantage_scale, config.training.lr)


if __name__ == "__main__":
    main()
