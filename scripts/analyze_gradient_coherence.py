"""Analyze gradient coherence to detect laminar→turbulent transitions.

Reads coherence_log.json and identifies:
- When layers start disagreeing (cosine similarity drops)
- When gradient norms become unstable (ratio spikes)
- Phase transitions where coherent flow becomes turbulent

This diagnoses the "eddies forming" phenomenon in Phase 7.
"""

import json
import sys
from pathlib import Path

import numpy as np


def detect_transitions(measurements: list) -> list:
    """Detect sudden changes in gradient coherence.

    Args:
        measurements: List of coherence stat dicts

    Returns:
        List of transition events
    """
    if len(measurements) < 2:
        return []

    transitions = []

    for i in range(1, len(measurements)):
        prev = measurements[i - 1]
        curr = measurements[i]

        # Detect sudden cosine drops (layers disagreeing)
        cosine_delta = curr["mean_cosine"] - prev["mean_cosine"]
        cosine_pct = abs(cosine_delta) / max(abs(prev["mean_cosine"]), 1e-6) * 100

        # Detect norm ratio spikes (instability)
        norm_delta = curr["norm_ratio"] - prev["norm_ratio"]
        norm_pct = abs(norm_delta) / max(prev["norm_ratio"], 1e-6) * 100

        # Flag significant transitions
        is_transition = (
            cosine_delta < -0.2  # Large cosine drop
            or cosine_pct > 50  # Large relative change
            or norm_pct > 100  # Norm ratio doubled
        )

        if is_transition:
            transitions.append(
                {
                    "step": curr["step"],
                    "prev_phase": prev.get("phase", "unknown"),
                    "curr_phase": curr.get("phase", "unknown"),
                    "cosine_delta": cosine_delta,
                    "cosine_pct_change": cosine_pct,
                    "norm_ratio_delta": norm_delta,
                    "norm_ratio_pct_change": norm_pct,
                    "prev_reward": prev.get("reward", 0.0),
                    "curr_reward": curr.get("reward", 0.0),
                    "prev_loss": prev.get("loss", 0.0),
                    "curr_loss": curr.get("loss", 0.0),
                }
            )

    return transitions


def summarize_by_phase(measurements: list) -> dict:
    """Aggregate coherence statistics by phase."""
    by_phase = {}

    for m in measurements:
        phase = m.get("phase", "unknown")
        if phase not in by_phase:
            by_phase[phase] = {
                "measurements": [],
                "cosines": [],
                "norm_ratios": [],
                "grad_norms": [],
                "rewards": [],
                "losses": [],
            }

        by_phase[phase]["measurements"].append(m)
        by_phase[phase]["cosines"].append(m["mean_cosine"])
        by_phase[phase]["norm_ratios"].append(m["norm_ratio"])
        by_phase[phase]["grad_norms"].append(m["mean_grad_norm"])
        by_phase[phase]["rewards"].append(m.get("reward", 0.0))
        by_phase[phase]["losses"].append(m.get("loss", 0.0))

    # Compute aggregate statistics
    summary = {}
    for phase, data in by_phase.items():
        summary[phase] = {
            "n_measurements": len(data["measurements"]),
            "cosine": {
                "mean": np.mean(data["cosines"]),
                "std": np.std(data["cosines"]),
                "min": np.min(data["cosines"]),
                "max": np.max(data["cosines"]),
            },
            "norm_ratio": {
                "mean": np.mean(data["norm_ratios"]),
                "std": np.std(data["norm_ratios"]),
                "min": np.min(data["norm_ratios"]),
                "max": np.max(data["norm_ratios"]),
            },
            "grad_norm": {
                "mean": np.mean(data["grad_norms"]),
                "std": np.std(data["grad_norms"]),
            },
            "reward": {
                "mean": np.mean(data["rewards"]),
                "std": np.std(data["rewards"]),
            },
            "loss": {
                "mean": np.mean(data["losses"]),
                "std": np.std(data["losses"]),
            },
        }

    return summary


def main():
    coherence_file = Path("output/hamiltonian/gradient_coherence/coherence_log.json")

    if not coherence_file.exists():
        print(f"ERROR: Gradient coherence log not found at {coherence_file}")
        print("Enable coherence logging with:")
        print("  training:")
        print("    log_attention_patterns: true  # Name kept for compatibility")
        print("    attention_log_freq: 10")
        return 1

    with open(coherence_file) as f:
        data = json.load(f)

    config = data["config"]
    measurements = data["measurements"]

    print("=" * 80)
    print("GRADIENT COHERENCE ANALYSIS — Laminar→Turbulent Detection")
    print("=" * 80)

    print("\nConfiguration:")
    print(f"  Log frequency: every {config['log_freq']} steps")
    print(f"  Total measurements: {config['n_measurements']}")

    # Detect transitions
    transitions = detect_transitions(measurements)

    print(f"\n{'-'*80}")
    print(f"DETECTED TRANSITIONS: {len(transitions)}")
    print(f"{'-'*80}")

    for t in transitions:
        print(f"\nStep {t['step']} | Phase {t['prev_phase']} → {t['curr_phase']}")
        print(f"  Reward: {t['prev_reward']:.3f} → {t['curr_reward']:.3f}")
        print(f"  Loss: {t['prev_loss']:.3f} → {t['curr_loss']:.3f}")
        print(f"  Cosine similarity: {t['cosine_pct_change']:+.1f}% change")
        print(f"  Norm ratio: {t['norm_ratio_pct_change']:+.1f}% change")

    # Summarize by phase
    phase_summary = summarize_by_phase(measurements)

    print(f"\n{'-'*80}")
    print("SUMMARY BY PHASE")
    print(f"{'-'*80}")

    for phase in sorted(phase_summary.keys()):
        stats = phase_summary[phase]
        print(f"\nPhase {phase} ({stats['n_measurements']} measurements):")
        print(f"  Reward: {stats['reward']['mean']:.3f} ± {stats['reward']['std']:.3f}")
        print(f"  Loss: {stats['loss']['mean']:.3f} ± {stats['loss']['std']:.3f}")
        print(
            f"  Gradient alignment (cosine): {stats['cosine']['mean']:.3f} ± {stats['cosine']['std']:.3f}"
        )
        print(
            f"  Gradient norm ratio: {stats['norm_ratio']['mean']:.1f} ± {stats['norm_ratio']['std']:.1f}"
        )
        print(
            f"  Mean gradient norm: {stats['grad_norm']['mean']:.6f} ± {stats['grad_norm']['std']:.6f}"
        )

    # Detect laminar→turbulent transition
    print(f"\n{'-'*80}")
    print("LAMINAR → TURBULENT DIAGNOSIS")
    print(f"{'-'*80}")

    phases_sorted = sorted(phase_summary.keys())
    if len(phases_sorted) >= 2:
        for i in range(len(phases_sorted) - 1):
            phase1 = phases_sorted[i]
            phase2 = phases_sorted[i + 1]
            stats1 = phase_summary[phase1]
            stats2 = phase_summary[phase2]

            # Check for turbulence indicators
            cosine_drop = stats2["cosine"]["mean"] - stats1["cosine"]["mean"]
            norm_ratio_increase = stats2["norm_ratio"]["mean"] - stats1["norm_ratio"]["mean"]
            reward_drop = stats2["reward"]["mean"] - stats1["reward"]["mean"]

            if (cosine_drop < -0.2 or norm_ratio_increase > 10) and reward_drop < -0.1:
                print(f"\n⚠️  TURBULENCE DETECTED: Phase {phase1} → Phase {phase2}")
                print(f"   Reward dropped: {reward_drop:+.3f}")
                print(f"   Gradient alignment dropped: {cosine_drop:+.3f}")
                print(f"   Norm ratio increased: {norm_ratio_increase:+.1f}")

                if cosine_drop < -0.3:
                    print("   → LAYERS DISAGREEING: gradients misaligned")
                    print('   → "Eddies forming" — conflicting gradient directions')
                if norm_ratio_increase > 20:
                    print("   → GRADIENT INSTABILITY: some layers exploding, others vanishing")
                    print('   → "Energy dissipating" — unstable optimization dynamics')

    print(f"\n{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
