"""Reward function for Hamiltonian mechanics training (hard mode).

Model must derive H from physical descriptions — no hand-holding.
Checks: format, physics identification, correct derivation, correct equations.
"""

from __future__ import annotations

import re

from qgre.types import RewardResult


def _normalize(s: str) -> str:
    """Normalize math expression for fuzzy matching."""
    s = s.lower()
    s = s.replace("**", "^")
    s = s.replace("\\frac", "")
    s = s.replace("\\cdot", "*")
    s = s.replace("·", "*")
    s = s.replace("×", "*")
    s = s.replace(" ", "")
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r'(\d)\*([a-z])', r'\1\2', s)
    return s


def _has_any(text: str, patterns: list[str], case_sensitive: bool = False) -> bool:
    """Check if any pattern appears in text."""
    t = text if case_sensitive else text.lower()
    for p in patterns:
        pp = p if case_sensitive else p.lower()
        if pp in t:
            return True
    return False


def _count_matches(text: str, patterns: list[str]) -> int:
    """Count how many patterns appear in text."""
    t = text.lower()
    return sum(1 for p in patterns if p.lower() in t)


def hamiltonian_reward(
    prompt: str,
    completion: str,
    metadata: dict | None = None,
) -> RewardResult:
    """Score a Hamiltonian mechanics derivation.

    Step 1 qualities (format + identification):
      q_format:         Substantial response with math content
      q_identifies_T:   Identifies/writes kinetic energy
      q_identifies_V:   Identifies/writes potential energy

    Step 2 qualities (derivation correctness):
      q_correct_dqdt:   Hamilton's equation dq/dt matches ground truth
      q_correct_dpdt:   Hamilton's equation dp/dt matches ground truth
    """
    scores: dict[str, float] = {}
    meta = metadata or {}
    text = completion
    norm = _normalize(text)

    # ─── q_format: substantial physics response ───
    has_length = len(text.strip()) > 100
    has_math = _has_any(text, ["=", "hamiltonian", "H ", "H=", "kinetic", "potential", "equation"])
    has_structure = _has_any(text, ["dq/dt", "dp/dt", "∂H/∂p", "∂H/∂q", "∂H/∂x",
                                    "hamilton", "equation of motion", "ṗ", "q̇"])
    if has_length and has_math and has_structure:
        scores["q_format"] = 1.0
    elif has_length and has_math:
        scores["q_format"] = 0.7
    elif has_length:
        scores["q_format"] = 0.3
    else:
        scores["q_format"] = 0.0

    # ─── q_identifies_T: kinetic energy term ───
    t_patterns = [
        "kinetic energy", "T =", "T=",
        "p^2/", "p²/", "p_r^2", "p_theta^2", "p₁", "p₂",
        "p^2/(2m", "p²/(2m", "½mv²", "(1/2)mv",
        "p_s^2", "p_x^2", "p_y^2",
    ]
    scores["q_identifies_T"] = min(1.0, _count_matches(text, t_patterns) / 2)

    # ─── q_identifies_V: potential energy term ───
    v_patterns = [
        "potential energy", "V =", "V=", "V(",
        "mgh", "mgL", "mgl", "-mgL",
        "kx²", "kx^2", "(1/2)k",
        "cos(θ", "cos(theta", "-cos(",
        "/r^", "/r²", "-α/r", "-G",
        "sin(θ", "sin(theta",
    ]
    scores["q_identifies_V"] = min(1.0, _count_matches(text, v_patterns) / 2)

    # ─── q_correct_dqdt: Hamilton's first equation ───
    expected_dqdt = meta.get("dqdt", "")
    if expected_dqdt:
        norm_expected = _normalize(expected_dqdt)
        # Direct match
        if norm_expected in norm:
            scores["q_correct_dqdt"] = 1.0
        # Check for key structure: dq/dt = ∂H/∂p mentioned AND correct form
        elif _has_any(text, ["dq/dt", "∂H/∂p", "dx/dt", "dr/dt", "ds/dt", "dθ/dt", "dtheta/dt"]):
            # Extract numbers from expected and check presence
            nums = re.findall(r'\d+', expected_dqdt)
            if nums:
                found = sum(1 for n in nums if n in norm)
                scores["q_correct_dqdt"] = 0.3 + 0.5 * (found / len(nums))
            else:
                # Expected is simple like "p" or "p_r"
                if _has_any(text, ["= p", "=p", "= p_r", "= p_s"]):
                    scores["q_correct_dqdt"] = 0.8
                else:
                    scores["q_correct_dqdt"] = 0.3
        else:
            scores["q_correct_dqdt"] = 0.0
    else:
        scores["q_correct_dqdt"] = 0.0

    # ─── q_correct_dpdt: Hamilton's second equation ───
    expected_dpdt = meta.get("dpdt", "")
    if expected_dpdt and expected_dpdt != "complex" and expected_dpdt != "cyclotron terms":
        norm_expected = _normalize(expected_dpdt)
        if norm_expected in norm:
            scores["q_correct_dpdt"] = 1.0
        elif _has_any(text, ["dp/dt", "∂H/∂q", "∂H/∂x", "dp_r/dt", "dp_s/dt", "dp_theta/dt", "ṗ"]):
            nums = re.findall(r'\d+', expected_dpdt)
            vars_present = _has_any(text, ["sin(", "cos(", "/r", "x^3", "x³", "x^2", "x²"])
            if nums:
                found = sum(1 for n in nums if n in norm)
                scores["q_correct_dpdt"] = 0.3 + 0.4 * (found / len(nums)) + (0.2 if vars_present else 0)
            else:
                scores["q_correct_dpdt"] = 0.3
        else:
            scores["q_correct_dpdt"] = 0.0
    elif expected_dpdt in ("complex", "cyclotron terms"):
        # For complex systems, give partial credit for mentioning the equation
        if _has_any(text, ["dp/dt", "∂H/∂q", "∂H/∂x", "∂H/∂θ", "dp_theta/dt"]):
            scores["q_correct_dpdt"] = 0.6
        else:
            scores["q_correct_dpdt"] = 0.0
    else:
        scores["q_correct_dpdt"] = 0.0

    total = sum(scores.values()) / max(len(scores), 1)
    return RewardResult(reward=total, scores=scores)
