"""Hint Registry for EGRS Q4 (uncertain+wrong) directional guidance.

The hint system provides training wheels for tokens/spans where the model is
lost (uncertain + wrong). Instead of zero gradient, we inject hint tokens
at generation time to point the model in the right direction.

Key design decisions:
1. Hints are per (prompt_id, span_id) - tied to specific prompts and spans
2. Hints decay with mastery - as model learns, hints fade out
3. Success WITHOUT hint clears the flag - we want model to do it alone
4. Hints persist across checkpoints - recovery continues after restart

Hint Extraction:
The hint_extractor callable extracts ground-truth hint text from metadata.
Domain-specific extractors map span_id to the relevant metadata field.
Example: Hamiltonian maps STEP_5 → H_expr, STEP_3 → T_expr, etc.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class HintExtractor(Protocol):
    """Protocol for domain-specific hint extraction.

    Given a span_id and metadata dict, returns hint text (ground truth)
    that should be shown to the model when it's struggling with that span.

    Returns None if no hint is available for this span.
    """
    def __call__(self, span_id: str, metadata: dict[str, Any]) -> str | None:
        ...


@dataclass
class HintEntry:
    """A single hint entry in the registry."""
    prompt_id: int
    span_id: str  # e.g., "STEP_1", "STEP_2"
    hint_tokens: list[int]  # Token IDs to inject
    mastery_at_flag: float  # Mastery when hint was first flagged
    flagged_step: int  # Training step when flagged
    success_count: int = 0  # Consecutive successes without hint
    total_uses: int = 0  # How many times hint was injected


class HintRegistry:
    """Registry for managing hint injection in EGRS Q4.

    Tracks which (prompt, span) pairs need hints and manages their lifecycle:
    - Flag when uncertain+wrong
    - Inject at generation time (with probability decay)
    - Clear when model succeeds without the hint
    """

    def __init__(
        self,
        mastery_threshold: float = 0.8,
        success_streak_to_clear: int = 2,
        seed: int | None = None,
    ):
        """Initialize hint registry.

        Args:
            mastery_threshold: Mastery score at which hints stop completely.
            success_streak_to_clear: Consecutive successes without hint needed to clear.
            seed: Random seed for hint injection probability. None = system random (default).
        """
        self.mastery_threshold = mastery_threshold
        self.success_streak_to_clear = success_streak_to_clear
        # Key: (prompt_id, span_id) -> HintEntry
        self._hints: dict[tuple[int, str], HintEntry] = {}
        # R3-MIO-002: Add optional seed parameter, default to None for system random
        import random
        if seed is None:
            self._random = random.Random()
        else:
            self._random = random.Random(seed)

    def flag_for_hint(
        self,
        prompt_id: int,
        span_id: str,
        hint_tokens: list[int],
        current_mastery: float,
        current_step: int,
    ) -> None:
        """Flag a (prompt, span) pair for hint injection.

        If already flagged, updates hint tokens (in case ground truth changed).
        Resets success streak since we're re-flagging.

        MIO-004: Expected ordering — hints extracted BEFORE flagging in same batch.
        mastery_fn uses value BEFORE this batch's training updates it.
        This is intentional: we flag based on pre-training mastery.

        Args:
            prompt_id: Prompt identifier.
            span_id: Span identifier (e.g., "STEP_1").
            hint_tokens: Token IDs to inject as hint.
            current_mastery: Current mastery score for decay calculation.
            current_step: Training step for tracking.
        """
        key = (prompt_id, span_id)
        if key in self._hints:
            # Update existing entry
            entry = self._hints[key]
            entry.hint_tokens = hint_tokens
            entry.success_count = 0  # Reset streak on re-flag
        else:
            # Create new entry
            self._hints[key] = HintEntry(
                prompt_id=prompt_id,
                span_id=span_id,
                hint_tokens=hint_tokens,
                mastery_at_flag=current_mastery,
                flagged_step=current_step,
            )

    def get_hint(
        self,
        prompt_id: int,
        span_id: str,
        current_mastery: float,
    ) -> list[int] | None:
        """Get hint tokens for a (prompt, span) if available and should inject.

        Uses probabilistic decay based on mastery:
        hint_probability = max(0, 1 - mastery / threshold)

        Args:
            prompt_id: Prompt identifier.
            span_id: Span identifier.
            current_mastery: Current mastery score.

        Returns:
            List of token IDs to inject, or None if no hint or probability miss.
        """
        key = (prompt_id, span_id)
        entry = self._hints.get(key)
        if entry is None:
            return None

        # Compute hint probability based on mastery decay
        hint_prob = self._compute_hint_probability(current_mastery)
        if hint_prob <= 0:
            return None

        # Probabilistic injection (R2-MIO-003: use seeded random)
        if self._random.random() < hint_prob:
            entry.total_uses += 1
            return entry.hint_tokens
        return None

    def get_hints_for_prompt(
        self,
        prompt_id: int,
        mastery_fn: Callable[[str], float] | None = None,
    ) -> dict[str, list[int]]:
        """Get all hints for a prompt that should be injected.

        Args:
            prompt_id: Prompt identifier.
            mastery_fn: Function mapping span_id -> mastery score.
                If None, mastery defaults to 0.0 (100% hint probability).

        Returns:
            Dict mapping span_id -> hint_tokens for spans that should get hints.
        """
        result = {}
        for (pid, span_id), entry in self._hints.items():
            if pid != prompt_id:
                continue
            # Look up mastery for this specific span
            mastery = mastery_fn(span_id) if mastery_fn else 0.0
            hint = self.get_hint(prompt_id, span_id, mastery)
            if hint is not None:
                result[span_id] = hint
        return result

    def record_success(
        self,
        prompt_id: int,
        span_id: str,
        hint_was_used: bool,
    ) -> bool:
        """Record a success on a (prompt, span) pair.

        If success was WITHOUT hint, increment streak. If streak reaches
        threshold, clear the hint flag (model can do it alone).

        Args:
            prompt_id: Prompt identifier.
            span_id: Span identifier.
            hint_was_used: Whether hint was injected for this success.

        Returns:
            True if hint was cleared (model graduated), False otherwise.
        """
        key = (prompt_id, span_id)
        entry = self._hints.get(key)
        if entry is None:
            return False

        if hint_was_used:
            # Success with hint doesn't count toward clearing
            entry.success_count = 0
            return False

        # Success without hint - increment streak
        entry.success_count += 1
        if entry.success_count >= self.success_streak_to_clear:
            # Model can do it alone - clear the hint
            del self._hints[key]
            return True

        return False

    def record_failure(
        self,
        prompt_id: int,
        span_id: str,
    ) -> None:
        """Record a failure - resets success streak.

        Args:
            prompt_id: Prompt identifier.
            span_id: Span identifier.
        """
        key = (prompt_id, span_id)
        entry = self._hints.get(key)
        if entry is not None:
            entry.success_count = 0

    def _compute_hint_probability(self, mastery: float) -> float:
        """Compute hint injection probability based on mastery.

        Formula: max(0, 1 - mastery / threshold)
        - mastery=0 → 100% hint
        - mastery=threshold/2 → 50% hint
        - mastery>=threshold → 0% hint
        """
        if mastery >= self.mastery_threshold:
            return 0.0
        prob = 1.0 - mastery / self.mastery_threshold
        # R3-MIO-001: Clamp probability to [0.0, 1.0] and warn if mastery is negative
        if mastery < 0.0:
            import warnings
            warnings.warn(
                f"R3-MIO-001: Negative mastery {mastery} detected in hint probability calculation. "
                f"Clamping probability from {prob} to [0.0, 1.0]."
            )
        return max(0.0, min(1.0, prob))

    def clear_all(self) -> int:
        """Clear all hints (e.g., at epoch boundary).

        Returns:
            Number of hints cleared.
        """
        count = len(self._hints)
        self._hints.clear()
        return count

    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics for logging."""
        if not self._hints:
            return {"hint_count": 0}

        total_uses = sum(e.total_uses for e in self._hints.values())
        avg_success = (
            sum(e.success_count for e in self._hints.values()) / len(self._hints)
            if self._hints else 0
        )
        return {
            "hint_count": len(self._hints),
            "total_hint_uses": total_uses,
            "avg_success_streak": avg_success,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry for checkpointing."""
        return {
            "mastery_threshold": self.mastery_threshold,
            "success_streak_to_clear": self.success_streak_to_clear,
            "hints": [
                {
                    "prompt_id": e.prompt_id,
                    "span_id": e.span_id,
                    "hint_tokens": e.hint_tokens,
                    "mastery_at_flag": e.mastery_at_flag,
                    "flagged_step": e.flagged_step,
                    "success_count": e.success_count,
                    "total_uses": e.total_uses,
                }
                for e in self._hints.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HintRegistry:
        """Deserialize registry from checkpoint."""
        import warnings

        registry = cls(
            mastery_threshold=data.get("mastery_threshold", 0.8),
            success_streak_to_clear=data.get("success_streak_to_clear", 2),
        )
        # R2-CSM-004: Validate hints field is a list before iteration
        hints_data = data.get("hints", [])
        if not isinstance(hints_data, list):
            warnings.warn(
                f"R2-CSM-004: hints field is not a list (type: {type(hints_data).__name__}). "
                "Skipping hint registry restoration."
            )
            return registry
        skipped_entries = 0
        first_error_msg = ""  # Initialize before loop to avoid UnboundLocalError
        for hint_data in hints_data:
            try:
                # R2-MIO-002: Skip entries with empty hint_tokens during deserialization
                hint_tokens = hint_data["hint_tokens"]
                if not hint_tokens or (isinstance(hint_tokens, list) and len(hint_tokens) == 0):
                    skipped_entries += 1
                    if skipped_entries == 1:
                        warnings.warn(
                            "R2-MIO-002: Skipping hint entry with empty hint_tokens. "
                            "This creates zombie entries. Checkpoint may be corrupted."
                        )
                    continue
                entry = HintEntry(
                    prompt_id=hint_data["prompt_id"],
                    span_id=hint_data["span_id"],
                    hint_tokens=hint_tokens,
                    mastery_at_flag=hint_data["mastery_at_flag"],
                    flagged_step=hint_data["flagged_step"],
                    success_count=hint_data.get("success_count", 0),
                    total_uses=hint_data.get("total_uses", 0),
                )
                registry._hints[(entry.prompt_id, entry.span_id)] = entry
            except (KeyError, TypeError) as e:
                skipped_entries += 1
                if skipped_entries == 1:
                    first_error_msg = str(e)
                    warnings.warn(
                        f"HintRegistry.from_dict: skipping corrupted entry: {e}. "
                        "Checkpoint may be corrupted or from incompatible version."
                    )
        if skipped_entries > 0:
            total_entries = len(data.get("hints", []))
            all_corrupted = " ALL entries corrupted — hint registry is empty!" if skipped_entries == total_entries else ""
            first_err = f" First error: {first_error_msg}" if skipped_entries > 1 else ""
            warnings.warn(
                f"HintRegistry.from_dict: skipped {skipped_entries}/{total_entries} corrupted entries.{first_err}{all_corrupted}"
            )
            if skipped_entries == total_entries:
                raise ValueError(
                    f"HintRegistry.from_dict: ALL {total_entries} entries corrupted. "
                    "Checkpoint may be incompatible or corrupt. Cannot restore empty hint registry."
                )
        return registry

    def __len__(self) -> int:
        return len(self._hints)

    def __contains__(self, key: tuple[int, str]) -> bool:
        return key in self._hints


# ─── Hint Extractors ──────────────────────────────────────────────────────────


def make_hamiltonian_hint_extractor() -> HintExtractor:
    """Create hint extractor for Hamiltonian mechanics domain.

    Maps span_id to metadata fields:
    - STEP_1 (COORDINATES): No hint (coordinates are given in prompt)
    - STEP_2 (MOMENTUM): No hint (definition is standard)
    - STEP_3 (KINETIC): T_expr from metadata
    - STEP_4 (POTENTIAL): V_expr from metadata
    - STEP_5 (HAMILTONIAN): H_expr from metadata
    - STEP_6 (EQUATIONS): Extract from ground_truth

    Returns:
        HintExtractor function that extracts hint text from metadata.
    """
    def extractor(span_id: str, metadata: dict[str, Any]) -> str | None:
        if not metadata:
            return None

        if span_id == "STEP_3":
            # KINETIC: T = ...
            t_expr = metadata.get("T_expr")
            return f"T = {t_expr}" if t_expr else None

        elif span_id == "STEP_4":
            # POTENTIAL: V = ...
            v_expr = metadata.get("V_expr")
            return f"V = {v_expr}" if v_expr else None

        elif span_id == "STEP_5":
            # HAMILTONIAN: H = ...
            h_expr = metadata.get("H_expr")
            return f"H = {h_expr}" if h_expr else None

        elif span_id == "STEP_6":
            # EQUATIONS: extract dq/dt and dp/dt from ground_truth
            gt = metadata.get("ground_truth", "")
            if not gt:
                return None
            # ground_truth format: "H = ...; dq/dt = ...; dp/dt = ..."
            parts = [p.strip() for p in gt.split(";")]
            equations = [p for p in parts if "dq/dt" in p or "dp/dt" in p]
            return "; ".join(equations) if equations else None

        # STEP_1 (COORDINATES) and STEP_2 (MOMENTUM) don't need hints
        # They're either given in prompt or are standard definitions
        return None

    return extractor


def make_generic_hint_extractor(
    span_to_field: dict[str, str],
    field_format: str = "{value}",
) -> HintExtractor:
    """Create configurable hint extractor from span_id → metadata field mapping.

    Args:
        span_to_field: Maps span_id (e.g., "STEP_1") to metadata key (e.g., "answer_1")
        field_format: Format string for hint text. Use {value} for the field value.
            Example: "The answer is: {value}"

    Returns:
        HintExtractor function.

    Example:
        extractor = make_generic_hint_extractor({
            "STEP_1": "section_1_answer",
            "STEP_2": "section_2_answer",
        }, field_format="Hint: {value}")
    """
    def extractor(span_id: str, metadata: dict[str, Any]) -> str | None:
        if not metadata:
            return None
        field = span_to_field.get(span_id)
        if not field:
            return None
        value = metadata.get(field)
        if not value:
            return None
        return field_format.format(value=value)

    return extractor
