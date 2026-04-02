"""Analyze attention patterns to detect laminar→turbulent transitions.

Reads attention_patterns.json and identifies:
- Sudden entropy changes (phase transitions)
- Attention collapse events (all heads converge)
- Attention fragmentation (heads become random)
- Recency bias spikes (model only looks at recent tokens)

This helps diagnose why Phase 7 causes sudden performance collapse.
"""

import json
import sys
from pathlib import Path
import numpy as np


def detect_transitions(measurements: list) -> list:
    """Detect sudden changes in attention patterns.

    Args:
        measurements: List of attention stat dicts

    Returns:
        List of transition events with step number and metrics
    """
    if len(measurements) < 2:
        return []

    transitions = []

    # Compute deltas between consecutive measurements
    for i in range(1, len(measurements)):
        prev = measurements[i-1]
        curr = measurements[i]

        # Detect sudden entropy changes (±50% or more)
        entropy_delta = curr["mean_entropy"] - prev["mean_entropy"]
        entropy_pct = abs(entropy_delta) / max(prev["mean_entropy"], 1e-6) * 100

        # Detect collapse/fragmentation spikes
        collapse_delta = curr["collapse_score"] - prev["collapse_score"]
        frag_delta = curr["fragmentation_score"] - prev["fragmentation_score"]

        # Detect recency bias spikes
        recency_delta = curr["recency_ratio"] - prev["recency_ratio"]

        # Detect loop strength changes
        loop_delta = curr["loop_strength"] - prev["loop_strength"]

        # Flag significant transitions
        is_transition = (
            entropy_pct > 50 or
            abs(collapse_delta) > 0.3 or
            abs(frag_delta) > 0.3 or
            abs(recency_delta) > 0.2 or
            abs(loop_delta) > 0.2
        )

        if is_transition:
            transitions.append({
                "step": curr["step"],
                "prev_phase": prev.get("phase", "unknown"),
                "curr_phase": curr.get("phase", "unknown"),
                "entropy_delta": entropy_delta,
                "entropy_pct_change": entropy_pct,
                "collapse_delta": collapse_delta,
                "fragmentation_delta": frag_delta,
                "recency_delta": recency_delta,
                "loop_delta": loop_delta,
                "prev_reward": prev.get("reward", 0.0),
                "curr_reward": curr.get("reward", 0.0),
                "prev_loss": prev.get("loss", 0.0),
                "curr_loss": curr.get("loss", 0.0),
            })

    return transitions


def summarize_by_phase(measurements: list) -> dict:
    """Aggregate attention statistics by phase.

    Args:
        measurements: List of attention stat dicts

    Returns:
        Dictionary mapping phase -> aggregate stats
    """
    by_phase = {}

    for m in measurements:
        phase = m.get("phase", "unknown")
        if phase not in by_phase:
            by_phase[phase] = {
                "measurements": [],
                "entropies": [],
                "collapse_scores": [],
                "fragmentation_scores": [],
                "recency_ratios": [],
                "loop_strengths": [],
                "rewards": [],
                "losses": [],
            }

        by_phase[phase]["measurements"].append(m)
        by_phase[phase]["entropies"].append(m["mean_entropy"])
        by_phase[phase]["collapse_scores"].append(m["collapse_score"])
        by_phase[phase]["fragmentation_scores"].append(m["fragmentation_score"])
        by_phase[phase]["recency_ratios"].append(m["recency_ratio"])
        by_phase[phase]["loop_strengths"].append(m["loop_strength"])
        by_phase[phase]["rewards"].append(m.get("reward", 0.0))
        by_phase[phase]["losses"].append(m.get("loss", 0.0))

    # Compute aggregate statistics
    summary = {}
    for phase, data in by_phase.items():
        summary[phase] = {
            "n_measurements": len(data["measurements"]),
            "entropy": {
                "mean": np.mean(data["entropies"]),
                "std": np.std(data["entropies"]),
                "min": np.min(data["entropies"]),
                "max": np.max(data["entropies"]),
            },
            "collapse": {
                "mean": np.mean(data["collapse_scores"]),
                "std": np.std(data["collapse_scores"]),
            },
            "fragmentation": {
                "mean": np.mean(data["fragmentation_scores"]),
                "std": np.std(data["fragmentation_scores"]),
            },
            "recency": {
                "mean": np.mean(data["recency_ratios"]),
                "std": np.std(data["recency_ratios"]),
            },
            "loop_strength": {
                "mean": np.mean(data["loop_strengths"]),
                "std": np.std(data["loop_strengths"]),
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
    pattern_file = Path("output/hamiltonian/attention_analysis/attention_patterns.json")

    if not pattern_file.exists():
        print(f"ERROR: Attention patterns not found at {pattern_file}")
        print("Enable attention logging with:")
        print("  training:")
        print("    log_attention_patterns: true")
        print("    attention_log_freq: 10")
        return 1

    with open(pattern_file) as f:
        data = json.load(f)

    config = data["config"]
    measurements = data["measurements"]

    print("=" * 80)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 80)

    print(f"\nConfiguration:")
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
        print(f"  Entropy: {t['entropy_pct_change']:+.1f}% change")
        print(f"  Collapse: {t['collapse_delta']:+.3f}")
        print(f"  Fragmentation: {t['fragmentation_delta']:+.3f}")
        print(f"  Recency bias: {t['recency_delta']:+.3f}")
        print(f"  Loop strength: {t['loop_delta']:+.3f}")

    # Summarize by phase
    phase_summary = summarize_by_phase(measurements)

    print(f"\n{'-'*80}")
    print(f"SUMMARY BY PHASE")
    print(f"{'-'*80}")

    for phase in sorted(phase_summary.keys()):
        stats = phase_summary[phase]
        print(f"\nPhase {phase} ({stats['n_measurements']} measurements):")
        print(f"  Reward: {stats['reward']['mean']:.3f} ± {stats['reward']['std']:.3f}")
        print(f"  Loss: {stats['loss']['mean']:.3f} ± {stats['loss']['std']:.3f}")
        print(f"  Entropy: {stats['entropy']['mean']:.3f} ± {stats['entropy']['std']:.3f}")
        print(f"  Collapse score: {stats['collapse']['mean']:.3f} ± {stats['collapse']['std']:.3f}")
        print(f"  Fragmentation score: {stats['fragmentation']['mean']:.3f} ± {stats['fragmentation']['std']:.3f}")
        print(f"  Recency bias: {stats['recency']['mean']:.3f} ± {stats['recency']['std']:.3f}")
        print(f"  Loop strength: {stats['loop_strength']['mean']:.3f} ± {stats['loop_strength']['std']:.3f}")

    # Detect laminar→turbulent transition
    print(f"\n{'-'*80}")
    print(f"LAMINAR → TURBULENT DIAGNOSIS")
    print(f"{'-'*80}")

    phases_sorted = sorted(phase_summary.keys())
    if len(phases_sorted) >= 2:
        for i in range(len(phases_sorted) - 1):
            phase1 = phases_sorted[i]
            phase2 = phases_sorted[i+1]
            stats1 = phase_summary[phase1]
            stats2 = phase_summary[phase2]

            # Check for turbulence indicators
            entropy_increase = stats2["entropy"]["mean"] - stats1["entropy"]["mean"]
            collapse_increase = stats2["collapse"]["mean"] - stats1["collapse"]["mean"]
            frag_increase = stats2["fragmentation"]["mean"] - stats1["fragmentation"]["mean"]
            reward_drop = stats2["reward"]["mean"] - stats1["reward"]["mean"]

            if (abs(entropy_increase) > 0.5 or
                abs(collapse_increase) > 0.2 or
                abs(frag_increase) > 0.2) and reward_drop < -0.1:

                print(f"\n⚠️  TURBULENCE DETECTED: Phase {phase1} → Phase {phase2}")
                print(f"   Reward dropped: {reward_drop:+.3f}")
                print(f"   Entropy changed: {entropy_increase:+.3f}")
                print(f"   Collapse changed: {collapse_increase:+.3f}")
                print(f"   Fragmentation changed: {frag_increase:+.3f}")

                if frag_increase > 0.2:
                    print(f"   → Attention FRAGMENTATION: heads attending randomly")
                    print(f"   → Likely cause: conflicting gradient directions")
                elif collapse_increase > 0.2:
                    print(f"   → Attention COLLAPSE: all heads attending to same tokens")
                    print(f"   → Likely cause: gradient saturation or mode collapse")
                elif abs(entropy_increase) > 0.5:
                    print(f"   → Attention PATTERN SHIFT: sudden change in attention strategy")
                    print(f"   → Likely cause: phase transition introducing new objectives")

    print(f"\n{'='*80}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
