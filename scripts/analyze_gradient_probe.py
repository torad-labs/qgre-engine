"""Analyze gradient probe measurements to compute optimal advantage_scale.

Reads probe_results.json (captured from actual training run) and computes:
- Actual logit nudge per step from direct logit measurements
- Optimal advantage_scale for target logit nudge relative to discriminative gap

No weight deltas, no hidden state inference — just direct logit measurements.
"""

import json
import sys
from pathlib import Path
import numpy as np


def analyze_probe_results(probe_file: Path):
    """Analyze captured gradient probe measurements."""
    with open(probe_file) as f:
        data = json.load(f)

    config = data["config"]
    measurements = data["measurements"]

    if not measurements:
        print("ERROR: No measurements captured!")
        return

    print("=" * 70)
    print("GRADIENT PROBE ANALYSIS — Direct logit measurements")
    print("=" * 70)

    # Config
    adv_scale = config["advantage_scale"]
    lr = config["lr"]
    lora_rank = config.get("lora_rank", 32)
    lora_alpha = config.get("lora_alpha", 64)
    n_steps = config["n_steps"]

    print(f"\nConfiguration:")
    print(f"  advantage_scale: {adv_scale}")
    print(f"  learning rate: {lr}")
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}")
    print(f"  steps captured: {n_steps}")

    # Per-step statistics
    print(f"\n{'-'*70}")
    print(f"Per-step measurements:")
    print(f"{'-'*70}")

    for m in measurements[:5]:  # Show first 5 steps
        step = m["step"]
        loss = m["loss"]
        mean_delta = m["mean_logit_delta"]
        max_delta = m["max_logit_delta"]

        print(f"\nStep {step}:")
        print(f"  Loss: {loss:.6f}")
        print(f"  Mean logit delta (physics tokens): {mean_delta:.6f}")
        print(f"  Max logit delta (physics tokens): {max_delta:.6f}")

    if len(measurements) > 5:
        print(f"\n  ... ({len(measurements) - 5} more steps)")

    # Aggregate statistics
    mean_deltas = [m["mean_logit_delta"] for m in measurements]
    max_deltas = [m["max_logit_delta"] for m in measurements]
    losses = [m["loss"] for m in measurements]

    avg_mean_delta = np.mean(mean_deltas)
    avg_max_delta = np.mean(max_deltas)
    avg_loss = np.mean(losses)

    print(f"\n{'-'*70}")
    print(f"Aggregated across {n_steps} steps:")
    print(f"{'-'*70}")
    print(f"  Mean loss: {avg_loss:.6f}")
    print(f"  Average mean logit delta: {avg_mean_delta:.6f}")
    print(f"  Average max logit delta: {avg_max_delta:.6f}")

    # Compare to discriminative gap
    # From previous measurement: physics tokens have ~90 logit difference
    logit_gap = 89.84  # Measured earlier
    nudge_fraction = avg_mean_delta / logit_gap

    print(f"\n{'-'*70}")
    print(f"Percentage of discriminative gap:")
    print(f"{'-'*70}")
    print(f"  Logit gap (physics tokens): {logit_gap:.2f}")
    print(f"  Mean nudge per step: {avg_mean_delta:.6f}")
    print(f"  Nudge as % of gap: {nudge_fraction*100:.4f}%")

    if nudge_fraction > 0:
        steps_for_1pct = 0.01 * logit_gap / avg_mean_delta
        steps_for_10pct = 0.1 * logit_gap / avg_mean_delta
        print(f"  Steps to close 1% of gap: {steps_for_1pct:.1f}")
        print(f"  Steps to close 10% of gap: {steps_for_10pct:.1f}")

    # Optimal advantage_scale for different target nudges
    print(f"\n{'-'*70}")
    print(f"Optimal advantage_scale for target nudge:")
    print(f"{'-'*70}")

    if avg_mean_delta > 0:
        for target_pct in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            target_nudge = target_pct / 100 * logit_gap
            optimal_scale = adv_scale * (target_nudge / avg_mean_delta)
            print(f"  {target_pct:5.2f}% → advantage_scale = {optimal_scale:.4f}")

        # Recommendation
        print(f"\n{'-'*70}")
        print(f"RECOMMENDATION:")
        print(f"{'-'*70}")
        target_pct = 1.0  # 1% of gap per step
        target_nudge = target_pct / 100 * logit_gap
        optimal_scale = adv_scale * (target_nudge / avg_mean_delta)
        print(f"  For {target_pct}% logit nudge per step (gentle but meaningful):")
        print(f"  advantage_scale = {optimal_scale:.4f}")
        print(f"\n  Current scale ({adv_scale}) produces {nudge_fraction*100:.4f}% nudge per step")

        scale_ratio = optimal_scale / adv_scale
        if nudge_fraction < 0.005:
            print(f"  → Too weak, scale UP by {scale_ratio:.1f}x")
        elif nudge_fraction > 0.05:
            print(f"  → Too aggressive, scale DOWN by {1/scale_ratio:.1f}x")
        else:
            print(f"  → Current scale is reasonable")


def main():
    probe_file = Path("output/hamiltonian/study/gradient_probe/probe_results.json")

    if not probe_file.exists():
        print(f"ERROR: Probe results not found at {probe_file}")
        print("Run the gradient probe training first:")
        print("  python -m qgre train --config examples/hamiltonian/config_smoke.yaml --reward examples.hamiltonian.reward_fn:hamiltonian_reward")
        return 1

    analyze_probe_results(probe_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
