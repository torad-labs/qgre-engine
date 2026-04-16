"""Curriculum management: tier-based mastery tracking and difficulty gating.

Standalone functions extracted from QGRETrainer so they can be tested and
reused independently of the full trainer.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from qgre.types import GameState, RewardResult


def get_prompt_tier(metadata: dict, difficulty_column: str | None) -> str:
    """Get the difficulty tier for a prompt from its metadata."""
    if difficulty_column:
        return metadata.get(difficulty_column, "default")
    return "default"


def record_mastery_and_advance(
    *,
    game_state: GameState,
    reward_results: list[RewardResult],
    active_qualities: list[list[str]],
    batch_contexts: list[Any],
    step_qualities: dict[int, list[str]],
    advantage_estimator: Any,
    dataloader: Any,
    difficulty_column: str | None,
    tier_order: list[str] | None,
    tier_advance_phase: int,
    tier_advance_threshold: float,
    global_step: int,
    metrics: dict,
) -> None:
    """Record per-tier mastery scores, check per-tier phase advancement, check tier unlock."""
    max_phase = max(step_qualities.keys())

    # Group reward results by tier
    # Note: reward_results and batch_contexts are FILTERED (post-SPO), batch.metadata is UNFILTERED
    # Use batch_contexts[i].tier which is already computed and aligned with filtered indices
    tier_groups = defaultdict(list)
    for i, rr in enumerate(reward_results):
        tier = batch_contexts[i].tier
        tier_groups[tier].append((rr, active_qualities[i]))

    # Record mastery per tier, check per-tier quality phase advance
    for tier, items in tier_groups.items():
        tier_active_qs = items[0][1]  # All items in same tier share active qualities
        for step_num, quality_keys in step_qualities.items():
            active_keys = [k for k in quality_keys if k in tier_active_qs]
            if active_keys:
                scores = [
                    float(np.mean([rr.scores.get(k, 0.0) for k in active_keys])) for rr, _ in items
                ]
                mean_score = float(np.mean(scores))
                game_state.record_tier_step_score(tier, step_num, mean_score)
                metrics[f"mastery/{tier}/step_{step_num}"] = mean_score

        # Gate quality phase advancement on tutorial skill mastery.
        # Without this, the quality phase races ahead of tutorial skills —
        # the model scores high on all qualities, SPO baseline catches up,
        # advantages zero out, loss=0 permanently.
        if game_state.tutorial_enabled:
            skills_in_tier = game_state._tier_to_skills.get(tier, [])
            unmastered = [
                sk
                for sk in skills_in_tier
                if sk in game_state.skill_tree and not game_state.skill_tree[sk].mastered
            ]
            if unmastered:
                continue  # Don't advance phase until tutorial skills are mastered

        if not game_state.check_tier_phase_advance(tier, max_phase):
            continue

        new_phase = game_state.tier_phases[tier]
        metrics[f"tier_phase_advanced/{tier}"] = new_phase
        # Reset SPO baselines for ALL prompts in this tier (not just current batch)
        col = difficulty_column
        if dataloader and col:
            tier_pids = [
                item["prompt_id"]
                for item in dataloader.items
                if item.get("metadata", {}).get(col, "default") == tier
            ]
            warnings.warn(
                f"[RESET TRIGGER] tier={tier}, phase→{new_phase}, dataloader path, found {len(tier_pids)} prompts",
                stacklevel=2,
            )
        else:
            tier_pids = [
                batch_contexts[i].prompt_id
                for i in range(len(reward_results))
                if batch_contexts[i].tier == tier
            ]
            warnings.warn(
                f"[RESET TRIGGER] tier={tier}, phase→{new_phase}, batch path, found {len(tier_pids)} prompts",
                stacklevel=2,
            )
        advantage_estimator.on_tier_advance(
            new_tier=new_phase,
            prompt_tier_map=dict.fromkeys(tier_pids, new_phase),
        )

    # Check tier unlock — tutorial gates tier advancement
    if tier_order:
        # Pre-check: which tier WOULD be next?
        active_set = set(game_state.active_tiers)
        candidate_tier = None
        for t in tier_order:
            if t not in active_set:
                candidate_tier = t
                break

        # Only attempt unlock if tutorial allows it
        if candidate_tier is None or game_state.can_tier_unlock(candidate_tier):
            new_tier = game_state.check_tier_unlock(
                tier_order,
                tier_advance_phase,
                tier_advance_threshold,
            )
            if new_tier:
                metrics["tier_unlocked"] = new_tier
                print(f"\n┌{'─' * 60}┐")
                print(f"│{'🔓 TIER UNLOCKED':^60}│")
                print(f"├{'─' * 60}┤")
                print(f"│  Step: {global_step:<51}│")
                print(f"│  Tier: {new_tier:<51}│")
                print(f"│  Active: {', '.join(game_state.active_tiers):<49}│")
                print(f"└{'─' * 60}┘")
                apply_difficulty_gate(game_state, dataloader, difficulty_column)
                # Reset baselines for ALL prompts in ALL active tiers on tier unlock
                # New tier means new prompt distribution — stale baselines must go
                col = difficulty_column
                if dataloader and col:
                    active_set = set(game_state.active_tiers)
                    all_active_pids = [
                        item["prompt_id"]
                        for item in dataloader.items
                        if item.get("metadata", {}).get(col, "default") in active_set
                    ]
                else:
                    all_active_pids = [
                        batch_contexts[i].prompt_id for i in range(len(reward_results))
                    ]
                    warnings.warn(
                        f"[RESET TRIGGER] tier_unlock={new_tier}, batch path, "
                        f"found {len(all_active_pids)} prompts — dataloader unavailable, "
                        f"most tier prompts will keep stale baselines",
                        stacklevel=2,
                    )
                advantage_estimator.on_tier_advance(
                    new_tier=0,  # Not used anymore — full reset for all affected pids
                    prompt_tier_map=dict.fromkeys(all_active_pids, 0),
                )

    # Tutorial skill tree: record per-skill mastery score
    if game_state.tutorial_enabled:
        cache_snapshot = game_state.snapshot_pool_version()
        for i, rr in enumerate(reward_results):
            ctx = batch_contexts[i]
            score = game_state.resolve_mastery_score(ctx.prompt_id_str, rr)
            game_state.record_completion(ctx.prompt_id_str, score)
        # Re-apply difficulty gate if tutorial state changed (skill mastered/unlocked/relocked)
        if game_state.did_prompt_pool_change(cache_snapshot):
            apply_difficulty_gate(game_state, dataloader, difficulty_column)
        metrics.update(game_state.get_tutorial_metrics())

    game_state.step_count = global_step
    metrics["phase"] = game_state.phase

    # Per-tier stagnation
    for tier in game_state.active_tiers:
        stag = game_state.check_tier_stagnation(tier)
        metrics[f"stagnation/{tier}"] = {"normal": 0, "stagnating": 1, "stuck": 2}[stag.value]


def apply_difficulty_gate(
    game_state: GameState,
    dataloader: Any,
    difficulty_column: str | None,
) -> None:
    """Apply difficulty gate using active_tiers from GameState.

    Sets dataloader to only sample prompts from active tiers, with equal
    weight per tier (prevents large tiers drowning small ones).

    When the tutorial system is enabled, tutorial-tracked prompts in active
    skills BYPASS the tier gate (tutorial is the authority for its prompts).
    Untracked prompts still respect the tier gate. Locked skill prompts are
    always zeroed out.
    """
    if dataloader is None or not hasattr(dataloader, "set_difficulty_gate"):
        return
    col = difficulty_column
    if not col:
        # Clear difficulty gate when tier_order is None
        if hasattr(dataloader, "_difficulty_gate"):
            dataloader._difficulty_gate = None
        return  # No difficulty column → no gating (default tier, all prompts)

    allowed = set(game_state.active_tiers)
    dataloader.set_difficulty_gate(allowed, col)

    # Tutorial: tracked prompts in active skills bypass tier gate
    tutorial_active_pids = None
    tutorial_tracked_pids = None
    if game_state.tutorial_enabled:
        tutorial_active_pids = set(game_state.get_active_prompts())
        tutorial_tracked_pids = set(game_state._prompt_to_skill.keys())

    tier_counts = Counter(item["metadata"].get(col, "default") for item in dataloader.items)
    tier_weights = {}
    for item in dataloader.items:
        tier = item["metadata"].get(col, "default")
        pid = item["prompt_id"]
        pid_str = str(pid)

        if tutorial_tracked_pids is not None and pid_str in tutorial_tracked_pids:
            # Tutorial-tracked prompt: skill gate is the authority, tier gate bypassed
            if pid_str in tutorial_active_pids:  # type: ignore[operator]
                tier_weights[pid] = 1.0 / max(tier_counts.get(tier, 1), 1)
            else:
                tier_weights[pid] = 0.0  # Locked skill
        elif tier in allowed:
            tier_weights[pid] = 1.0 / max(tier_counts[tier], 1)
            # else: difficulty gate already zeros it out

    if tier_weights:
        dataloader.set_priorities(tier_weights)

    active_count = sum(1 for w in tier_weights.values() if w > 0) if tier_weights else 0
    tut_count = len(tutorial_active_pids) if tutorial_active_pids else "N/A"
    print(f"\n┌{'─' * 60}┐")
    print(f"│{'⚙ DIFFICULTY GATE':^60}│")
    print(f"├{'─' * 60}┤")
    print(f"│  Tiers: {', '.join(sorted(allowed)):<50}│")
    print(f"│  Active prompts: {active_count:<41}│")
    print(f"│  Tutorial active: {tut_count!s:<40}│")
    if active_count == 0:
        print(f"│  ⚠ ZERO PROMPTS — falling back to uniform{' ' * 17}│")
    print(f"└{'─' * 60}┘")
