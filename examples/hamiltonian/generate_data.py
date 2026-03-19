"""Generate Hamiltonian mechanics training data as parquet.

Hard mode: describe physical systems in natural language. Model must DERIVE
the Hamiltonian from scratch — identify coordinates, write T and V, construct H,
then derive Hamilton's equations. No hand-holding.
"""

import random
import pandas as pd


def _make_problems():
    """Generate Hamiltonian problems of increasing difficulty."""
    problems = []

    # ── TIER 1: 1D systems described physically (no H given) ──

    for m in [1, 2, 3, 5]:
        for k in [1, 2, 4, 6]:
            problems.append({
                "prompt": (
                    f"A block of mass {m} kg is attached to a spring with spring constant k = {k} N/m "
                    f"on a frictionless surface. Let x be the displacement from equilibrium.\n\n"
                    f"Derive the Hamiltonian H(x, p) from first principles and find Hamilton's equations of motion."
                ),
                "ground_truth": f"H = p^2/{2*m} + {k}x^2/2; dq/dt = p/{m}; dp/dt = -{k}x",
                "dqdt": f"p/{m}" if m > 1 else "p",
                "dpdt": f"-{k}x" if k > 1 else "-x",
                "H_key": f"p^2/{2*m} + {k}x^2/2",
                "difficulty": "tier1",
                "system": "spring",
            })

    for m in [1, 2, 3]:
        problems.append({
            "prompt": (
                f"A particle of mass {m} kg falls vertically under gravity (g = 9.8 m/s²). "
                f"Let y be the height above the ground.\n\n"
                f"Derive the Hamiltonian H(y, p) and Hamilton's equations of motion."
            ),
            "ground_truth": f"H = p^2/{2*m} + {m}*9.8*y; dq/dt = p/{m}; dp/dt = -{m}*9.8",
            "dqdt": f"p/{m}" if m > 1 else "p",
            "dpdt": f"-{float(m)*9.8}",
            "H_key": f"p^2/{2*m} + {m}gy",
            "difficulty": "tier1",
            "system": "gravity",
        })

    # ── TIER 2: Pendulum and rotational systems ──

    for L in [1, 2]:
        for m in [1, 2, 3]:
            problems.append({
                "prompt": (
                    f"A simple pendulum consists of a mass m = {m} kg on a rigid rod of length "
                    f"L = {L} m, swinging in a uniform gravitational field (g = 9.8 m/s²). "
                    f"Use the angle θ from the vertical as the generalized coordinate.\n\n"
                    f"Derive the Hamiltonian H(θ, p_θ) and Hamilton's equations of motion.\n"
                    f"Express the kinetic energy in terms of the conjugate momentum p_θ."
                ),
                "ground_truth": (
                    f"T = p_theta^2/(2*{m}*{L}^2); V = -{m}*g*{L}*cos(theta); "
                    f"H = p_theta^2/(2*{m}*{L**2}) - {m}*g*{L}*cos(theta); "
                    f"dtheta/dt = p_theta/({m}*{L**2}); dp_theta/dt = -{m}*g*{L}*sin(theta)"
                ),
                "dqdt": f"p_theta/({m*L**2})" if m*L**2 != 1 else "p_theta",
                "dpdt": f"-{float(m)*9.8*L}*sin(theta)",
                "H_key": "p_theta^2/(2mL^2) - mgLcos(theta)",
                "difficulty": "tier2",
                "system": "pendulum",
            })

    # Bead on rotating hoop
    for omega in [1, 2, 3]:
        for R in [1, 2]:
            problems.append({
                "prompt": (
                    f"A bead of mass m = 1 kg slides without friction on a circular hoop of radius "
                    f"R = {R} m. The hoop rotates about its vertical diameter with constant angular "
                    f"velocity ω = {omega} rad/s. Use the angle θ from the bottom of the hoop as the "
                    f"generalized coordinate.\n\n"
                    f"Derive the Hamiltonian H(θ, p_θ). Account for both the gravitational potential "
                    f"energy and the effective potential from the rotation. Then find Hamilton's equations."
                ),
                "ground_truth": (
                    f"T = (1/2)*R^2*theta_dot^2; "
                    f"V_eff = -gR*cos(theta) - (1/2)*omega^2*R^2*sin^2(theta); "
                    f"H = p_theta^2/(2R^2) + V_eff"
                ),
                "dqdt": f"p_theta/{R**2}",
                "dpdt": "complex",
                "H_key": "rotation + gravity",
                "difficulty": "tier2",
                "system": "rotating_hoop",
            })

    # Central force problem
    for n in [1, 2, 3]:
        for alpha in [1, 2, 5]:
            problems.append({
                "prompt": (
                    f"A particle of mass m = 1 kg moves in a central force field with potential "
                    f"V(r) = -{alpha}/r{'²' if n == 2 else ('^' + str(n) if n > 2 else '')}. "
                    f"Use polar coordinates (r, θ).\n\n"
                    f"Derive the Hamiltonian H(r, θ, p_r, p_θ) and Hamilton's equations of motion. "
                    f"Show that p_θ (angular momentum) is conserved."
                ),
                "ground_truth": (
                    f"H = p_r^2/2 + p_theta^2/(2r^2) - {alpha}/r^{n}; "
                    f"dr/dt = p_r; dtheta/dt = p_theta/r^2; "
                    f"dp_r/dt = p_theta^2/r^3 - {n}*{alpha}/r^{n+1}; dp_theta/dt = 0"
                ),
                "dqdt": "p_r",
                "dpdt": f"p_theta^2/r^3 + {n}*{alpha}/r^{n+1}",
                "H_key": f"p_r^2/2 + p_theta^2/(2r^2) - {alpha}/r^{n}",
                "difficulty": "tier2",
                "system": "central_force",
            })

    # ── TIER 3: Multi-DOF and electromagnetic ──

    # Coupled oscillators
    for k1 in [1, 2]:
        for k_c in [1, 3]:
            problems.append({
                "prompt": (
                    f"Two identical masses m = 1 kg are connected in a line by three springs: "
                    f"the outer two springs have constant k₁ = {k1} N/m (attached to fixed walls), "
                    f"and the middle coupling spring has constant k_c = {k_c} N/m.\n"
                    f"Let x₁ and x₂ be the displacements of the two masses from equilibrium.\n\n"
                    f"Write the Hamiltonian H(x₁, x₂, p₁, p₂) and derive all four Hamilton's equations."
                ),
                "ground_truth": (
                    f"H = p1^2/2 + p2^2/2 + {k1}*x1^2/2 + {k1}*x2^2/2 + {k_c}*(x1-x2)^2/2; "
                    f"dx1/dt = p1; dx2/dt = p2; "
                    f"dp1/dt = -{k1}*x1 - {k_c}*(x1-x2); dp2/dt = -{k1}*x2 + {k_c}*(x1-x2)"
                ),
                "dqdt": "p1",
                "dpdt": f"-{k1}*x1 - {k_c}*(x1-x2)",
                "H_key": f"p1^2/2 + p2^2/2 + {k1}(x1^2+x2^2)/2 + {k_c}(x1-x2)^2/2",
                "difficulty": "tier3",
                "system": "coupled_oscillators",
            })

    # Charged particle in magnetic field
    for B in [1, 2]:
        for q in [1, -1]:
            charge_word = "electron" if q == -1 else "proton"
            problems.append({
                "prompt": (
                    f"A charged particle (charge q = {'+' if q > 0 else ''}{q} C, mass m = 1 kg) "
                    f"moves in the x-y plane under a uniform magnetic field B = {B} T directed "
                    f"along the z-axis. The vector potential in the symmetric gauge is "
                    f"A = (-By/2, Bx/2, 0).\n\n"
                    f"Derive the Hamiltonian H(x, y, p_x, p_y) using the canonical momenta "
                    f"p_x = mẋ + qA_x and p_y = mẏ + qA_y. Then find Hamilton's equations."
                ),
                "ground_truth": (
                    f"H = (p_x + {q}*{B}*y/2)^2/2 + (p_y - {q}*{B}*x/2)^2/2"
                ),
                "dqdt": f"p_x + {q*B}*y/2",
                "dpdt": "cyclotron terms",
                "H_key": "canonical momentum with vector potential",
                "difficulty": "tier3",
                "system": "magnetic_field",
            })

    # Double pendulum (hardest)
    problems.append({
        "prompt": (
            "A double pendulum consists of two rigid rods of equal length L = 1 m, each with a "
            "point mass m = 1 kg at the end. The first rod is attached to a fixed pivot, and the "
            "second rod hangs from the end of the first. Use angles θ₁ and θ₂ (each measured from "
            "the vertical) as generalized coordinates.\n\n"
            "Derive the Hamiltonian H(θ₁, θ₂, p₁, p₂). You will need to:\n"
            "1. Express the positions of both masses in terms of θ₁ and θ₂\n"
            "2. Compute the kinetic energy T(θ₁, θ₂, θ̇₁, θ̇₂)\n"
            "3. Compute the potential energy V(θ₁, θ₂)\n"
            "4. Find the conjugate momenta and perform the Legendre transform\n"
            "5. Write Hamilton's equations"
        ),
        "ground_truth": (
            "T = (1/2)*(2*L^2*theta1_dot^2 + L^2*theta2_dot^2 + 2*L^2*theta1_dot*theta2_dot*cos(theta1-theta2)); "
            "V = -(2+1)*g*L*cos(theta1) - g*L*cos(theta2); complex H"
        ),
        "dqdt": "involves inverse mass matrix",
        "dpdt": "nonlinear in theta1, theta2",
        "H_key": "Legendre transform of double pendulum Lagrangian",
        "difficulty": "tier3",
        "system": "double_pendulum",
    })

    # Kepler problem with specific parameters
    for M in [1, 5, 10]:
        problems.append({
            "prompt": (
                f"A satellite of mass m = 1 kg orbits a planet of mass M = {M} kg "
                f"(gravitational constant G = 1 in natural units). "
                f"Use polar coordinates (r, θ).\n\n"
                f"Derive the Hamiltonian for the reduced one-body problem. "
                f"Show that the angular momentum p_θ is a constant of motion. "
                f"Write the effective one-dimensional Hamiltonian H_eff(r, p_r) "
                f"by treating p_θ = ℓ as a constant."
            ),
            "ground_truth": (
                f"H = p_r^2/2 + p_theta^2/(2r^2) - {M}/r; "
                f"H_eff = p_r^2/2 + ell^2/(2r^2) - {M}/r; "
                f"dp_theta/dt = -dH/dtheta = 0 → conserved"
            ),
            "dqdt": "p_r",
            "dpdt": f"ell^2/r^3 - {M}/r^2",
            "H_key": f"p_r^2/2 + L^2/(2r^2) - {M}/r",
            "difficulty": "tier2",
            "system": "kepler",
        })

    # Particle on inclined plane
    for angle in [30, 45, 60]:
        for m in [1, 2]:
            problems.append({
                "prompt": (
                    f"A particle of mass m = {m} kg slides without friction down a plane "
                    f"inclined at {angle}° to the horizontal. Let s be the distance measured "
                    f"along the plane from the top.\n\n"
                    f"Derive the Hamiltonian H(s, p_s) and Hamilton's equations of motion. "
                    f"Use g = 9.8 m/s²."
                ),
                "ground_truth": (
                    f"H = p_s^2/(2*{m}) - {m}*g*s*sin({angle}°); "
                    f"ds/dt = p_s/{m}; dp_s/dt = {m}*g*sin({angle}°)"
                ),
                "dqdt": f"p_s/{m}" if m > 1 else "p_s",
                "dpdt": f"{m}*9.8*sin({angle})",
                "H_key": f"p_s^2/{2*m} - {m}*g*s*sin({angle})",
                "difficulty": "tier1",
                "system": "inclined_plane",
            })

    # Particle in quartic potential (anharmonic oscillator)
    for a in [1, 2]:
        for b in [1, 3]:
            problems.append({
                "prompt": (
                    f"A particle of mass m = 1 kg moves in a one-dimensional anharmonic potential "
                    f"V(x) = {a}x² + {b}x⁴. This is the Duffing oscillator — a classic nonlinear system.\n\n"
                    f"Derive the Hamiltonian and Hamilton's equations of motion. "
                    f"Discuss whether the system is integrable."
                ),
                "ground_truth": (
                    f"H = p^2/2 + {a}*x^2 + {b}*x^4; "
                    f"dq/dt = p; dp/dt = -{2*a}*x - {4*b}*x^3"
                ),
                "dqdt": "p",
                "dpdt": f"-{2*a}x - {4*b}x^3",
                "H_key": f"p^2/2 + {a}x^2 + {b}x^4",
                "difficulty": "tier2",
                "system": "duffing",
            })

    return problems


def main():
    import os
    problems = _make_problems()
    random.seed(42)
    random.shuffle(problems)

    # Take up to 80
    problems = problems[:80]

    df = pd.DataFrame(problems)
    out_path = "examples/hamiltonian/data/train.parquet"
    os.makedirs("examples/hamiltonian/data", exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Generated {len(df)} Hamiltonian problems → {out_path}")
    print(f"Difficulty: {df['difficulty'].value_counts().to_dict()}")
    print(f"Systems: {df['system'].value_counts().to_dict()}")
    print(f"\nSample (tier2):")
    sample = df[df['difficulty'] == 'tier2'].iloc[0]
    print(f"  System: {sample['system']}")
    print(f"  Prompt: {sample['prompt'][:150]}...")
    print(f"  GT: {sample['ground_truth'][:100]}...")


if __name__ == "__main__":
    main()
