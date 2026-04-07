"""Generate Hamiltonian mechanics training data with sympy-verified ground truth.

120+ problems across 4 difficulty tiers + adversarial + edge cases.
Every ground truth is symbolically verified: dH/dp == dqdt, -dH/dq == dpdt.
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd
import sympy as sp

from examples.hamiltonian.system_prompts import ABBREVIATED, FULL, MINIMAL, NONE


# System prompt fade: earlier tiers get full coaching, later tiers get less
TIER_SYSTEM_PROMPTS = {
    "tutorial_gravity": FULL,
    "tutorial_spring": FULL,
    "combined": FULL,
    "tier1": FULL,
    "tier1b": FULL,
    "edge": FULL,
    "tier2": ABBREVIATED,
    "tier2b": ABBREVIATED,
    "tier3": MINIMAL,
    "tier3b": MINIMAL,
    "tier4": NONE,
    "adversarial": NONE,
}


# ─── Sympy verification ─────────────────────────────────────────────────────


def _verify(H, q_vars, p_vars, dqdt_exprs, dpdt_exprs):
    """Verify Hamilton's equations: dH/dp == dqdt, -dH/dq == dpdt."""
    for p_i, expected in zip(p_vars, dqdt_exprs, strict=False):
        actual = sp.diff(H, p_i)
        diff = sp.simplify(actual - expected)
        assert diff == 0, f"dH/d{p_i} = {actual}, expected {expected}, diff = {diff}"
    for q_i, expected in zip(q_vars, dpdt_exprs, strict=False):
        actual = -sp.diff(H, q_i)
        diff = sp.simplify(actual - expected)
        assert diff == 0, f"-dH/d{q_i} = {actual}, expected {expected}, diff = {diff}"


def _problem(
    prompt, H, T, V, q_vars, p_vars, dqdt_exprs, dpdt_exprs, difficulty, system, coordinates
):
    """Build a verified problem dict with separate system prompt."""
    _verify(H, q_vars, p_vars, dqdt_exprs, dpdt_exprs)

    # Format dqdt/dpdt as semicolon-separated for multi-DOF
    dqdt_strs = [f"d{q}/dt = {dq}" for q, dq in zip(q_vars, dqdt_exprs, strict=False)]
    dpdt_strs = [f"d{p}/dt = {dp}" for p, dp in zip(p_vars, dpdt_exprs, strict=False)]

    sys_prompt = TIER_SYSTEM_PROMPTS.get(difficulty, FULL)

    return {
        "prompt": prompt,
        "system_prompt": sys_prompt,
        "ground_truth": f"H = {H}; {'; '.join(dqdt_strs)}; {'; '.join(dpdt_strs)}",
        "H_expr": str(H),
        "T_expr": str(T),
        "V_expr": str(V),
        "dqdt": "; ".join(str(d) for d in dqdt_exprs),
        "dpdt": "; ".join(str(d) for d in dpdt_exprs),
        "coordinates": ", ".join(str(q) for q in q_vars),
        "difficulty": difficulty,
        "system": system,
    }


# ─── Tier 1: Simple 1D systems ──────────────────────────────────────────────


def _tier1_spring():
    x, p = sp.symbols("x p", real=True)
    problems = []
    for m in [1, 2, 3, 5]:
        for k in [1, 2, 4, 6]:
            T = p**2 / (2 * m)
            V = sp.Rational(k, 2) * x**2
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A block of mass {m} kg is attached to a spring with spring constant "
                        f"k = {k} N/m on a frictionless surface. Let x be the displacement from "
                        f"equilibrium.\n\nDerive the Hamiltonian H(x, p) from first principles "
                        f"and find Hamilton's equations of motion."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, x)],
                    difficulty="tier1",
                    system="spring",
                    coordinates="x",
                )
            )
    return problems


def _tier1_freefall():
    y, p = sp.symbols("y p", real=True)
    g = sp.Rational(98, 10)  # 9.8 as exact rational
    problems = []
    for m in [1, 2, 3, 5]:
        T = p**2 / (2 * m)
        V = m * g * y
        H = T + V
        problems.append(
            _problem(
                prompt=(
                    f"A particle of mass {m} kg falls vertically under gravity (g = 9.8 m/s²). "
                    f"Let y be the height above the ground.\n\n"
                    f"Derive the Hamiltonian H(y, p) and Hamilton's equations of motion."
                ),
                H=H,
                T=T,
                V=V,
                q_vars=[y],
                p_vars=[p],
                dqdt_exprs=[sp.diff(H, p)],
                dpdt_exprs=[-sp.diff(H, y)],
                difficulty="tier1",
                system="freefall",
                coordinates="y",
            )
        )
    return problems


def _tier1_incline():
    s, p_s = sp.symbols("s p_s", real=True)
    g = sp.Rational(98, 10)
    problems = []
    for angle_deg in [30, 45, 60]:
        angle_rad = sp.pi * angle_deg / 180
        sin_a = sp.sin(angle_rad)
        for m in [1, 2, 3]:
            T = p_s**2 / (2 * m)
            V = -m * g * sin_a * s
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A particle of mass m = {m} kg slides without friction down a plane "
                        f"inclined at {angle_deg}° to the horizontal. Let s be the distance "
                        f"measured along the plane from the top.\n\n"
                        f"Derive the Hamiltonian H(s, p_s) and Hamilton's equations of motion. "
                        f"Use g = 9.8 m/s²."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[s],
                    p_vars=[p_s],
                    dqdt_exprs=[sp.diff(H, p_s)],
                    dpdt_exprs=[-sp.diff(H, s)],
                    difficulty="tier1b",
                    system="inclined_plane",
                    coordinates="s",
                )
            )
    return problems


def _tier1_constant_force():
    """Particle under constant force F (not gravity)."""
    x, p = sp.symbols("x p", real=True)
    problems = []
    for m in [1, 2]:
        for F in [3, 5, 10]:
            T = p**2 / (2 * m)
            V = -F * x
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A particle of mass {m} kg moves along the x-axis under a constant "
                        f"force F = {F} N. Let x be the position.\n\n"
                        f"Derive the Hamiltonian H(x, p) and Hamilton's equations of motion."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, x)],
                    difficulty="tier1",
                    system="constant_force",
                    coordinates="x",
                )
            )
    return problems


def _tier1_gravity_spring():
    """Mass on vertical spring — combines gravity and spring potential."""
    x, p = sp.symbols("x p", real=True)
    g = sp.Rational(98, 10)
    problems = []
    for m in [1, 2, 3]:
        for k in [2, 4, 8]:
            T = p**2 / (2 * m)
            V = sp.Rational(k, 2) * x**2 + m * g * x
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A mass m = {m} kg hangs from a vertical spring with constant "
                        f"k = {k} N/m. Let x be the displacement from the natural length "
                        f"(positive downward). g = 9.8 m/s².\n\n"
                        f"Derive the Hamiltonian H(x, p) and Hamilton's equations of motion."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, x)],
                    difficulty="tier1b",
                    system="gravity_spring",
                    coordinates="x",
                )
            )
    return problems


# ─── Tutorial phases: isolated sub-skills with expanded parameter grids ──────

_GRAVITY_PROMPT_TEMPLATES = [
    "A particle of mass {m} kg falls vertically under gravity (g = 9.8 m/s²). Let y be the height above the ground.\n\nDerive the Hamiltonian H(y, p) and Hamilton's equations of motion.",
    "A {m} kg object is dropped from rest in a uniform gravitational field (g = 9.8 m/s²). Using y as the height coordinate:\n\nDerive the Hamiltonian H(y, p) and Hamilton's equations of motion.",
    "Consider a mass m = {m} kg falling freely under gravity (g = 9.8 m/s²). Let y denote height.\n\nDerive the Hamiltonian H(y, p) and find Hamilton's equations of motion.",
]

_SPRING_PROMPT_TEMPLATES = [
    "A block of mass {m} kg is attached to a spring with spring constant k = {k} N/m on a frictionless surface. Let x be the displacement from equilibrium.\n\nDerive the Hamiltonian H(x, p) from first principles and find Hamilton's equations of motion.",
    "A {m} kg mass oscillates on a spring (k = {k} N/m) without friction. Using displacement x from equilibrium:\n\nDerive the Hamiltonian H(x, p) and Hamilton's equations of motion.",
    "Consider a horizontal spring-mass system: mass m = {m} kg, spring constant k = {k} N/m, frictionless. Let x be displacement.\n\nDerive the Hamiltonian H(x, p) and find Hamilton's equations.",
]


def _tutorial_gravity():
    """Expanded freefall tutorial: 40+ variants to learn V = mgy as a pattern."""
    y, p = sp.symbols("y p", real=True)
    g = sp.Rational(98, 10)
    problems = []
    masses = [sp.Rational(1, 2), 1, sp.Rational(3, 2), 2, sp.Rational(5, 2), 3, 4, 5, 7, 10]
    for m_val in masses:
        m = sp.nsimplify(m_val)
        T = p**2 / (2 * m)
        V = m * g * y
        H = T + V
        template = random.choice(_GRAVITY_PROMPT_TEMPLATES)
        problems.append(
            _problem(
                prompt=template.format(m=float(m_val)),
                H=H,
                T=T,
                V=V,
                q_vars=[y],
                p_vars=[p],
                dqdt_exprs=[sp.diff(H, p)],
                dpdt_exprs=[-sp.diff(H, y)],
                difficulty="tutorial_gravity",
                system="freefall",
                coordinates="y",
            )
        )
    return problems


def _tutorial_spring():
    """Expanded spring tutorial: 50+ variants to learn V = kx²/2 as a pattern."""
    x, p = sp.symbols("x p", real=True)
    problems = []
    masses = [sp.Rational(1, 2), 1, sp.Rational(3, 2), 2, 3, 5, 7, 10]
    spring_constants = [sp.Rational(1, 2), 1, 2, 3, 4, 6, 8, 10]
    for m_val in masses:
        for k_val in spring_constants[:6]:  # 8 × 6 = 48 variants
            m = sp.nsimplify(m_val)
            k = sp.nsimplify(k_val)
            T = p**2 / (2 * m)
            V = k / 2 * x**2
            H = T + V
            template = random.choice(_SPRING_PROMPT_TEMPLATES)
            problems.append(
                _problem(
                    prompt=template.format(m=float(m_val), k=float(k_val)),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, x)],
                    difficulty="tutorial_spring",
                    system="spring",
                    coordinates="x",
                )
            )
    return problems


def _tutorial_combined():
    """gravity_spring duplicated as tutorial 'combined' tier with expanded grid."""
    x, p = sp.symbols("x p", real=True)
    g = sp.Rational(98, 10)
    problems = []
    for m in [1, 2, 3, 5]:
        for k in [1, 2, 4, 6, 8]:
            T = p**2 / (2 * m)
            V = sp.Rational(k, 2) * x**2 + m * g * x
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A mass m = {m} kg hangs from a vertical spring with constant "
                        f"k = {k} N/m. Let x be the displacement from the natural length "
                        f"(positive downward). g = 9.8 m/s².\n\n"
                        f"Derive the Hamiltonian H(x, p) and Hamilton's equations of motion."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, x)],
                    difficulty="combined",
                    system="gravity_spring",
                    coordinates="x",
                )
            )
    return problems


# ─── Tier 2: Angular / nonlinear systems ────────────────────────────────────


def _tier2_pendulum():
    theta, p_theta = sp.symbols("theta p_theta", real=True)
    g = sp.Rational(98, 10)
    problems = []
    for L in [1, 2, 3]:
        for m in [1, 2, 3]:
            I = m * L**2  # moment of inertia
            T = p_theta**2 / (2 * I)
            V = -m * g * L * sp.cos(theta)
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A simple pendulum consists of a mass m = {m} kg on a rigid rod of "
                        f"length L = {L} m, swinging in a uniform gravitational field "
                        f"(g = 9.8 m/s²). Use the angle θ from the vertical as the "
                        f"generalized coordinate.\n\n"
                        f"Derive the Hamiltonian H(θ, p_θ) and Hamilton's equations of motion. "
                        f"Express the kinetic energy in terms of the conjugate momentum p_θ."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[theta],
                    p_vars=[p_theta],
                    dqdt_exprs=[sp.diff(H, p_theta)],
                    dpdt_exprs=[-sp.diff(H, theta)],
                    difficulty="tier2b",
                    system="pendulum",
                    coordinates="theta",
                )
            )
    return problems


def _tier2_central_force():
    r, theta, p_r, p_theta = sp.symbols("r theta p_r p_theta", positive=True)
    problems = []
    for n in [1, 2]:
        for alpha in [1, 2, 5]:
            m = 1
            T = p_r**2 / (2 * m) + p_theta**2 / (2 * m * r**2)
            V = -alpha / r**n
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A particle of mass m = 1 kg moves in a central force field with "
                        f"potential V(r) = -{alpha}/r{'²' if n == 2 else ''}. "
                        f"Use polar coordinates (r, θ).\n\n"
                        f"Derive the Hamiltonian H(r, θ, p_r, p_θ) and Hamilton's equations. "
                        f"Show that p_θ (angular momentum) is conserved."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[r, theta],
                    p_vars=[p_r, p_theta],
                    dqdt_exprs=[sp.diff(H, p_r), sp.diff(H, p_theta)],
                    dpdt_exprs=[-sp.diff(H, r), -sp.diff(H, theta)],
                    difficulty="tier2b",
                    system="central_force",
                    coordinates="r, theta",
                )
            )
    return problems


def _tier2_kepler():
    r, theta, p_r, p_theta = sp.symbols("r theta p_r p_theta", positive=True)
    problems = []
    for M in [1, 5, 10]:
        # G=1 natural units, m=1
        T = p_r**2 / 2 + p_theta**2 / (2 * r**2)
        V = -M / r
        H = T + V
        problems.append(
            _problem(
                prompt=(
                    f"A satellite of mass m = 1 kg orbits a planet of mass M = {M} kg "
                    f"(gravitational constant G = 1 in natural units). "
                    f"Use polar coordinates (r, θ).\n\n"
                    f"Derive the Hamiltonian for the reduced one-body problem. "
                    f"Show that the angular momentum p_θ is a constant of motion."
                ),
                H=H,
                T=T,
                V=V,
                q_vars=[r, theta],
                p_vars=[p_r, p_theta],
                dqdt_exprs=[sp.diff(H, p_r), sp.diff(H, p_theta)],
                dpdt_exprs=[-sp.diff(H, r), -sp.diff(H, theta)],
                difficulty="tier2b",
                system="kepler",
                coordinates="r, theta",
            )
        )
    return problems


def _tier2_anharmonic():
    x, p = sp.symbols("x p", real=True)
    problems = []
    for a in [1, 2, 3]:
        for b in [1, 3, 5]:
            T = p**2 / 2
            V = a * x**2 + b * x**4
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A particle of mass m = 1 kg moves in a one-dimensional anharmonic "
                        f"potential V(x) = {a}x² + {b}x⁴ (Duffing oscillator).\n\n"
                        f"Derive the Hamiltonian and Hamilton's equations of motion."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, x)],
                    difficulty="tier2",
                    system="duffing",
                    coordinates="x",
                )
            )
    return problems


def _tier2_morse():
    """Morse potential: V(x) = D*(1 - exp(-a*x))^2."""
    x, p = sp.symbols("x p", real=True)
    problems = []
    for D in [1, 5]:
        for a in [1, 2]:
            T = p**2 / 2
            V = D * (1 - sp.exp(-a * x)) ** 2
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A particle of mass m = 1 kg moves in a Morse potential "
                        f"V(x) = {D}*(1 - exp(-{a}*x))². This models molecular bonding.\n\n"
                        f"Derive the Hamiltonian H(x, p) and Hamilton's equations of motion."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, x)],
                    difficulty="tier2",
                    system="morse",
                    coordinates="x",
                )
            )
    return problems


# ─── Tier 3: Multi-DOF and electromagnetic ──────────────────────────────────


def _tier3_coupled_oscillators():
    x1, x2, p1, p2 = sp.symbols("x1 x2 p1 p2", real=True)
    problems = []
    for k1 in [1, 2, 4]:
        for kc in [1, 3]:
            m = 1
            T = p1**2 / (2 * m) + p2**2 / (2 * m)
            V = (
                sp.Rational(k1, 2) * x1**2
                + sp.Rational(k1, 2) * x2**2
                + sp.Rational(kc, 2) * (x1 - x2) ** 2
            )
            H = sp.expand(T + V)
            problems.append(
                _problem(
                    prompt=(
                        f"Two identical masses m = 1 kg are connected in a line by three "
                        f"springs: the outer two have constant k₁ = {k1} N/m (attached to "
                        f"fixed walls), and the middle coupling spring has constant "
                        f"k_c = {kc} N/m. Let x₁ and x₂ be the displacements from "
                        f"equilibrium.\n\n"
                        f"Write the Hamiltonian H(x₁, x₂, p₁, p₂) and derive all four "
                        f"Hamilton's equations."
                    ),
                    H=H,
                    T=T,
                    V=sp.expand(V),
                    q_vars=[x1, x2],
                    p_vars=[p1, p2],
                    dqdt_exprs=[sp.diff(H, p1), sp.diff(H, p2)],
                    dpdt_exprs=[-sp.diff(H, x1), -sp.diff(H, x2)],
                    difficulty="tier3",
                    system="coupled_oscillators",
                    coordinates="x1, x2",
                )
            )
    return problems


def _tier3_magnetic_field():
    x, y, px, py = sp.symbols("x y p_x p_y", real=True)
    problems = []
    for B in [1, 2]:
        for q in [1, -1]:
            m = 1
            # Symmetric gauge: A = (-By/2, Bx/2, 0)
            # Canonical momentum: p = mv + qA
            # H = (p_x - qA_x)^2/(2m) + (p_y - qA_y)^2/(2m)
            # A_x = -B*y/2, A_y = B*x/2
            # But canonical: H = (p_x + q*B*y/2)^2/(2m) + (p_y - q*B*x/2)^2/(2m)
            # Wait: A_x = -By/2, so q*A_x = -q*B*y/2
            # p_x = m*vx + q*A_x = m*vx - q*B*y/2
            # vx = (p_x + q*B*y/2)/m
            # Similarly A_y = Bx/2, q*A_y = q*B*x/2
            # p_y = m*vy + q*A_y = m*vy + q*B*x/2
            # vy = (p_y - q*B*x/2)/m
            # H = m*vx^2/2 + m*vy^2/2 = (p_x + q*B*y/2)^2/(2m) + (p_y - q*B*x/2)^2/(2m)
            qB = q * B
            T_eff_x = (px + sp.Rational(qB, 2) * y) ** 2 / (2 * m)
            T_eff_y = (py - sp.Rational(qB, 2) * x) ** 2 / (2 * m)
            H = sp.expand(T_eff_x + T_eff_y)
            T = H  # All kinetic for free particle in B-field
            V = sp.Integer(0)

            charge_word = "positive" if q == 1 else "negative"
            problems.append(
                _problem(
                    prompt=(
                        f"A charged particle (charge q = {'+' if q > 0 else ''}{q} C, "
                        f"mass m = 1 kg) moves in the x-y plane under a uniform magnetic "
                        f"field B = {B} T along the z-axis. The vector potential in the "
                        f"symmetric gauge is A = (-By/2, Bx/2, 0).\n\n"
                        f"Derive the Hamiltonian H(x, y, p_x, p_y) using canonical momenta "
                        f"p_x = mẋ + qA_x, p_y = mẏ + qA_y. Then find Hamilton's equations."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x, y],
                    p_vars=[px, py],
                    dqdt_exprs=[sp.diff(H, px), sp.diff(H, py)],
                    dpdt_exprs=[-sp.diff(H, x), -sp.diff(H, y)],
                    difficulty="tier3b",
                    system="magnetic_field",
                    coordinates="x, y",
                )
            )
    return problems


def _tier3_rotating_hoop():
    theta, p_theta = sp.symbols("theta p_theta", real=True)
    g = sp.Rational(98, 10)
    problems = []
    for omega in [1, 2, 3]:
        for R in [1, 2]:
            m = 1
            # Bead on rotating hoop: T = (1/2)*m*R^2*theta_dot^2
            # V_eff = -m*g*R*cos(theta) - (1/2)*m*omega^2*R^2*sin^2(theta)
            # H = p_theta^2/(2*m*R^2) + V_eff
            I = m * R**2
            T = p_theta**2 / (2 * I)
            V = (
                -m * g * R * sp.cos(theta)
                - sp.Rational(1, 2) * m * omega**2 * R**2 * sp.sin(theta) ** 2
            )
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A bead of mass m = 1 kg slides without friction on a circular hoop "
                        f"of radius R = {R} m. The hoop rotates about its vertical diameter "
                        f"with constant angular velocity ω = {omega} rad/s. Use the angle θ "
                        f"from the bottom of the hoop as the generalized coordinate.\n\n"
                        f"Derive the Hamiltonian H(θ, p_θ). Account for gravitational and "
                        f"effective rotational potential. Then find Hamilton's equations."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[theta],
                    p_vars=[p_theta],
                    dqdt_exprs=[sp.diff(H, p_theta)],
                    dpdt_exprs=[-sp.diff(H, theta)],
                    difficulty="tier3b",
                    system="rotating_hoop",
                    coordinates="theta",
                )
            )
    return problems


def _tier3_2d_oscillator():
    """Anisotropic 2D harmonic oscillator."""
    x, y, px, py = sp.symbols("x y p_x p_y", real=True)
    problems = []
    for kx in [1, 2]:
        for ky in [3, 5]:
            m = 1
            T = px**2 / (2 * m) + py**2 / (2 * m)
            V = sp.Rational(kx, 2) * x**2 + sp.Rational(ky, 2) * y**2
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"A particle of mass m = 1 kg moves in an anisotropic 2D harmonic "
                        f"potential V(x,y) = ({kx}/2)x² + ({ky}/2)y².\n\n"
                        f"Derive the Hamiltonian H(x, y, p_x, p_y) and all four Hamilton's "
                        f"equations."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[x, y],
                    p_vars=[px, py],
                    dqdt_exprs=[sp.diff(H, px), sp.diff(H, py)],
                    dpdt_exprs=[-sp.diff(H, x), -sp.diff(H, y)],
                    difficulty="tier3",
                    system="2d_oscillator",
                    coordinates="x, y",
                )
            )
    return problems


# ─── Tier 4: Legendre transform required ────────────────────────────────────


def _tier4_double_pendulum():
    """Double pendulum with varying masses and lengths."""
    theta1, theta2 = sp.symbols("theta1 theta2", real=True)
    p1, p2 = sp.symbols("p1 p2", real=True)
    g = sp.Rational(98, 10)
    problems = []

    configs = [
        (1, 1, 1, 1),  # m1, m2, L1, L2
        (2, 1, 1, 1),
        (1, 1, 2, 1),
    ]
    for m1, m2, L1, L2 in configs:
        # The double pendulum Hamiltonian in terms of (theta1, theta2, p1, p2)
        # requires inverting the mass matrix. For m1=m2=1, L1=L2=1:
        #
        # T = (1/2)*(m1+m2)*L1^2*θ̇₁² + (1/2)*m2*L2^2*θ̇₂²
        #     + m2*L1*L2*θ̇₁*θ̇₂*cos(θ₁-θ₂)
        #
        # The mass matrix M is:
        # M = [[( m1+m2)*L1^2,  m2*L1*L2*cos(θ₁-θ₂)],
        #      [m2*L1*L2*cos(θ₁-θ₂),  m2*L2^2]]
        #
        # p = M * θ̇, so θ̇ = M^{-1} * p
        # H = p^T M^{-1} p / 2 + V  (but must be careful — H = p·θ̇ - L)
        #
        # For the Hamiltonian we use: H = (p1*θ̇1 + p2*θ̇2) - L
        # which equals T + V when expressed in momenta.
        #
        # Since the mass matrix inversion involves cos(θ₁-θ₂), the full
        # expression is complex. We'll provide it symbolically.

        delta = theta1 - theta2
        a11 = (m1 + m2) * L1**2
        a12 = m2 * L1 * L2 * sp.cos(delta)
        a22 = m2 * L2**2
        det = a11 * a22 - a12**2

        # θ̇ = M^{-1} p → θ̇₁ = (a22*p1 - a12*p2)/det, θ̇₂ = (a11*p2 - a12*p1)/det
        # T in terms of momenta: T = (a22*p1^2 - 2*a12*p1*p2 + a11*p2^2) / (2*det)
        T = (a22 * p1**2 - 2 * a12 * p1 * p2 + a11 * p2**2) / (2 * det)
        V = -(m1 + m2) * g * L1 * sp.cos(theta1) - m2 * g * L2 * sp.cos(theta2)
        H = T + V

        problems.append(
            _problem(
                prompt=(
                    f"A double pendulum: rod 1 has length L₁ = {L1} m with mass "
                    f"m₁ = {m1} kg at its end, rod 2 has length L₂ = {L2} m with mass "
                    f"m₂ = {m2} kg at its end. Rod 1 hangs from a fixed pivot, rod 2 "
                    f"from the end of rod 1. Angles θ₁, θ₂ measured from vertical. "
                    f"g = 9.8 m/s².\n\n"
                    f"Derive the Hamiltonian H(θ₁, θ₂, p₁, p₂) by:\n"
                    f"1. Writing T and V in terms of θ₁, θ₂, θ̇₁, θ̇₂\n"
                    f"2. Computing the conjugate momenta\n"
                    f"3. Performing the Legendre transform\n"
                    f"4. Writing Hamilton's equations"
                ),
                H=H,
                T=T,
                V=V,
                q_vars=[theta1, theta2],
                p_vars=[p1, p2],
                dqdt_exprs=[sp.diff(H, p1), sp.diff(H, p2)],
                dpdt_exprs=[-sp.diff(H, theta1), -sp.diff(H, theta2)],
                difficulty="tier4",
                system="double_pendulum",
                coordinates="theta1, theta2",
            )
        )
    return problems


def _tier4_lagrangian_to_hamiltonian():
    """Problems where a Lagrangian is given and you must perform the transform."""
    q, p = sp.symbols("q p", real=True)
    problems = []

    # L = (1/2)*a*q_dot^2 - V(q) → p = a*q_dot → q_dot = p/a → H = p^2/(2a) + V
    for a in [1, 3, 5]:
        for c in [2, 4]:
            T = p**2 / (2 * a)
            V = c * q**4
            H = T + V
            problems.append(
                _problem(
                    prompt=(
                        f"Given the Lagrangian L = ({a}/2)*q̇² - {c}*q⁴, perform the "
                        f"Legendre transform to find the Hamiltonian.\n\n"
                        f"1. Find the conjugate momentum p = ∂L/∂q̇\n"
                        f"2. Express q̇ in terms of p\n"
                        f"3. Compute H = p*q̇ - L\n"
                        f"4. Write Hamilton's equations"
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[q],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, q)],
                    difficulty="tier4",
                    system="legendre_transform",
                    coordinates="q",
                )
            )

    # Velocity-dependent Lagrangian: L = (1/2)*m*q_dot^2 + b*q*q_dot - V(q)
    # p = m*q_dot + b*q → q_dot = (p - b*q)/m
    # H = p*q_dot - L = p*(p-bq)/m - (1/2)*m*((p-bq)/m)^2 - b*q*(p-bq)/m + V
    # H = (p-bq)^2/(2m) + V(q)  (after simplification)
    m_val = 1
    for b_val in [1, 2]:
        for k_val in [2, 4]:
            b_sym = b_val
            # p = q_dot + b*q → q_dot = p - b*q
            # H = (p - b*q)^2/2 + k*q^2/2
            T = (p - b_sym * q) ** 2 / 2
            V = sp.Rational(k_val, 2) * q**2
            H = sp.expand(T + V)
            problems.append(
                _problem(
                    prompt=(
                        f"Given the Lagrangian L = (1/2)*q̇² + {b_val}*q*q̇ - ({k_val}/2)*q², "
                        f"perform the Legendre transform.\n\n"
                        f"Note: the conjugate momentum p = ∂L/∂q̇ = q̇ + {b_val}*q, "
                        f"so q̇ = p - {b_val}*q.\n\n"
                        f"Find H(q, p) and Hamilton's equations."
                    ),
                    H=H,
                    T=T,
                    V=V,
                    q_vars=[q],
                    p_vars=[p],
                    dqdt_exprs=[sp.diff(H, p)],
                    dpdt_exprs=[-sp.diff(H, q)],
                    difficulty="tier4",
                    system="velocity_dependent_L",
                    coordinates="q",
                )
            )
    return problems


# ─── Adversarial problems ───────────────────────────────────────────────────


def _adversarial():
    """Problems that test methodology resistance."""
    x, p = sp.symbols("x p", real=True)
    problems = []

    # Trick: system with friction — no valid conservative Hamiltonian
    # Model should note this is non-conservative
    # We give H=0, dqdt=0, dpdt=0 as "ground truth" signaling no valid H
    T = sp.Integer(0)
    V = sp.Integer(0)
    H = sp.Integer(0)
    for gamma in [1, 2]:
        problems.append(
            {
                "prompt": (
                    f"A block of mass m = 1 kg slides on a surface with friction coefficient "
                    f"γ = {gamma}. The friction force is -γ*v.\n\n"
                    f"Derive the Hamiltonian and Hamilton's equations. "
                    f"Is this system conservative? Can a standard Hamiltonian be written?"
                ),
                "system_prompt": TIER_SYSTEM_PROMPTS["adversarial"],
                "ground_truth": "Non-conservative system — no standard Hamiltonian exists",
                "H_expr": "none",
                "T_expr": "p**2/2",
                "V_expr": "none",
                "dqdt": "none",
                "dpdt": "none",
                "coordinates": "x",
                "difficulty": "adversarial",
                "system": "friction",
            }
        )

    # Trick: time-dependent force — Hamiltonian exists but is not conserved
    t = sp.Symbol("t")
    for F0 in [1, 5]:
        T = p**2 / 2
        V = -F0 * sp.cos(t) * x  # time-dependent — H exists but dH/dt ≠ 0
        H = T + V
        problems.append(
            {
                "prompt": (
                    f"A particle of mass m = 1 kg is driven by a time-dependent force "
                    f"F(t) = {F0}*cos(t).\n\n"
                    f"Can you write a Hamiltonian? Is it conserved? "
                    f"Derive Hamilton's equations if possible."
                ),
                "system_prompt": TIER_SYSTEM_PROMPTS["adversarial"],
                "ground_truth": f"H = p**2/2 - {F0}*cos(t)*x; H is NOT conserved (explicit time dependence)",
                "H_expr": str(H),
                "T_expr": str(T),
                "V_expr": str(V),
                "dqdt": str(sp.diff(H, p)),
                "dpdt": str(-sp.diff(H, x)),
                "coordinates": "x",
                "difficulty": "adversarial",
                "system": "time_dependent",
            }
        )

    # Trick: already in Hamiltonian form — just verify, don't re-derive
    for a in [1, 3]:
        for b in [2, 4]:
            H = a * p**2 + b * x**2
            T = a * p**2
            V = b * x**2
            problems.append(
                {
                    "prompt": (
                        f"Consider the Hamiltonian H(x, p) = {a}p² + {b}x².\n\n"
                        f"Verify this is a valid Hamiltonian by computing Hamilton's equations. "
                        f"What physical system does this describe?"
                    ),
                    "system_prompt": TIER_SYSTEM_PROMPTS["adversarial"],
                    "ground_truth": f"H = {a}*p**2 + {b}*x**2; dq/dt = {2 * a}*p; dp/dt = -{2 * b}*x",
                    "H_expr": str(H),
                    "T_expr": str(T),
                    "V_expr": str(V),
                    "dqdt": str(sp.diff(H, p)),
                    "dpdt": str(-sp.diff(H, x)),
                    "coordinates": "x",
                    "difficulty": "adversarial",
                    "system": "verify_H",
                }
            )

    return problems


# ─── Edge cases ──────────────────────────────────────────────────────────────


def _edge_cases():
    x, p = sp.symbols("x p", real=True)
    problems = []

    # Very large spring constant
    for k in [1000, 10000]:
        T = p**2 / 2
        V = sp.Rational(k, 2) * x**2
        H = T + V
        problems.append(
            _problem(
                prompt=(
                    f"A particle of mass m = 1 kg on a spring with extremely stiff "
                    f"spring constant k = {k} N/m.\n\n"
                    f"Derive the Hamiltonian and Hamilton's equations."
                ),
                H=H,
                T=T,
                V=V,
                q_vars=[x],
                p_vars=[p],
                dqdt_exprs=[sp.diff(H, p)],
                dpdt_exprs=[-sp.diff(H, x)],
                difficulty="edge",
                system="stiff_spring",
                coordinates="x",
            )
        )

    # Free particle (no potential)
    for m in [1, 5]:
        T = p**2 / (2 * m)
        V = sp.Integer(0)
        H = T + V
        problems.append(
            _problem(
                prompt=(
                    f"A free particle of mass m = {m} kg moves along the x-axis with "
                    f"no forces acting on it.\n\n"
                    f"Derive the Hamiltonian and Hamilton's equations. "
                    f"What are the conserved quantities?"
                ),
                H=H,
                T=T,
                V=V,
                q_vars=[x],
                p_vars=[p],
                dqdt_exprs=[sp.diff(H, p)],
                dpdt_exprs=[-sp.diff(H, x)],
                difficulty="edge",
                system="free_particle",
                coordinates="x",
            )
        )

    # Particle at equilibrium in quartic potential (unusual)
    T = p**2 / 2
    V = x**4
    H = T + V
    problems.append(
        _problem(
            prompt=(
                "A particle of mass m = 1 kg in a pure quartic potential V(x) = x⁴ "
                "(no quadratic term).\n\n"
                "Derive the Hamiltonian and Hamilton's equations. "
                "Note: this potential has no harmonic approximation near x=0."
            ),
            H=H,
            T=T,
            V=V,
            q_vars=[x],
            p_vars=[p],
            dqdt_exprs=[sp.diff(H, p)],
            dpdt_exprs=[-sp.diff(H, x)],
            difficulty="edge",
            system="quartic_only",
            coordinates="x",
        )
    )

    return problems


# ─── Main ────────────────────────────────────────────────────────────────────


def _make_problems():
    all_problems = []
    generators = [
        _tutorial_gravity,
        _tutorial_spring,
        _tutorial_combined,
        _tier1_spring,
        _tier1_freefall,
        _tier1_incline,
        _tier1_constant_force,
        _tier1_gravity_spring,
        _tier2_pendulum,
        _tier2_central_force,
        _tier2_kepler,
        _tier2_anharmonic,
        _tier2_morse,
        _tier3_coupled_oscillators,
        _tier3_magnetic_field,
        _tier3_rotating_hoop,
        _tier3_2d_oscillator,
        _tier4_double_pendulum,
        _tier4_lagrangian_to_hamiltonian,
        _edge_cases,
    ]
    for gen in generators:
        all_problems.extend(gen())

    # Adversarial problems bypass _verify (non-standard ground truth)
    all_problems.extend(_adversarial())

    return all_problems


def main():
    random.seed(42)  # BEFORE _make_problems — tutorial prompt templates must be deterministic
    problems = _make_problems()

    from collections import Counter

    diff_counts = Counter(p["difficulty"] for p in problems)
    sys_counts = Counter(p["system"] for p in problems)

    random.seed(42)
    random.shuffle(problems)

    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write combined file
    df = pd.DataFrame(problems)
    df.to_parquet(out_dir / "train.parquet", index=False)

    # Write per-phase files for curriculum staging
    # Phase 1: tier1 + edge (simple 1D — full system prompt)
    # Phase 2: + tier1b + tier2 (combined potentials + 1D nonlinear — abbreviated prompt)
    # Phase 3: + tier2b + tier3 (angular + multi-DOF — minimal prompt)
    # Phase 4: + tier3b + tier4 + adversarial (electromagnetic + Legendre — no prompt)
    phase_tiers = {
        1: {"tier1", "edge"},
        2: {"tier1", "tier1b", "edge", "tier2"},
        3: {"tier1", "tier1b", "edge", "tier2", "tier2b", "tier3"},
        4: {
            "tier1",
            "tier1b",
            "edge",
            "tier2",
            "tier2b",
            "tier3",
            "tier3b",
            "tier4",
            "adversarial",
        },
    }
    for phase, tiers in phase_tiers.items():
        phase_df = df[df["difficulty"].isin(tiers)]
        phase_path = out_dir / f"phase{phase}.parquet"
        phase_df.to_parquet(phase_path, index=False)
        print(f"Phase {phase}: {len(phase_df)} problems → {phase_path}")

    print(f"\nTotal: {len(df)} Hamiltonian problems")
    print(f"Difficulty: {dict(diff_counts)}")
    print(f"Systems:    {dict(sys_counts)}")

    verified = sum(1 for p in problems if p["difficulty"] not in ("adversarial",))
    print(f"Sympy-verified: {verified}/{len(problems)}")

    for tier in ["tier1", "tier2", "tier3", "tier4"]:
        samples = [p for p in problems if p["difficulty"] == tier]
        if samples:
            s = samples[0]
            prompt_part = s["prompt"].split("---\n\n", 1)[-1]
            print(f"\nSample ({tier}, {s['system']}): {prompt_part[:100]}...")


if __name__ == "__main__":
    main()
