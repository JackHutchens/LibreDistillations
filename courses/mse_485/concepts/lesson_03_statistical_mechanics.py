"""Interactive concept visualizations for Lesson 03: Basics of Statistical Mechanics."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def _unwrap(val: Any, default: Any = 0.0) -> Any:
    if val is None:
        return default
    return getattr(val, "value", val)


def render_liouville_phase_space(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Evolve a closed phase-space patch (x, p) under a Hamiltonian flow and verify Liouville area conservation."""
    pot_type = str(_unwrap(controls.get("potential_type"), "quartic"))
    time_t = float(_unwrap(controls.get("time_t"), 2.4))
    anharmonicity = float(_unwrap(controls.get("anharmonicity"), 0.6))

    # Build a dense closed boundary polygon + interior grid for an initial circular/rectangular patch
    n_bound = 240
    theta = np.linspace(0.0, 2.0 * np.pi, n_bound, endpoint=False)
    x0_center, p0_center = 0.85, 0.65
    rx, rp = 0.35, 0.35

    bx0 = x0_center + rx * np.cos(theta)
    bp0 = p0_center + rp * np.sin(theta)

    # Interior cloud of sample microstates
    rng = np.random.default_rng(42)
    r_rand = np.sqrt(rng.uniform(0.0, 0.92, 180))
    th_rand = rng.uniform(0.0, 2.0 * np.pi, 180)
    cx0 = x0_center + rx * r_rand * np.cos(th_rand)
    cp0 = p0_center + rp * r_rand * np.sin(th_rand)

    def force_fn(x: np.ndarray) -> np.ndarray:
        if pot_type == "harmonic":
            return -x
        elif pot_type == "pendulum":
            return -np.sin(x) - 0.15 * anharmonicity * np.sin(2.0 * x)
        else:  # quartic anharmonic oscillator V(x) = 0.5 x^2 + 0.25 * alpha * x^4
            return -x - anharmonicity * (x ** 3)

    def pot_fn(x: np.ndarray) -> np.ndarray:
        if pot_type == "harmonic":
            return 0.5 * (x ** 2)
        elif pot_type == "pendulum":
            return (1.0 - np.cos(x)) + 0.075 * anharmonicity * (1.0 - np.cos(2.0 * x))
        else:
            return 0.5 * (x ** 2) + 0.25 * anharmonicity * (x ** 4)

    # Symplectic Velocity-Verlet integration preserves 2-form dx ^ dp exactly
    dt = 0.01
    n_steps = int(round(time_t / dt))

    bx, bp = bx0.copy(), bp0.copy()
    cx, cp = cx0.copy(), cp0.copy()

    for _ in range(n_steps):
        f_b = force_fn(bx)
        bp_half = bp + 0.5 * dt * f_b
        bx_new = bx + dt * bp_half
        bp = bp_half + 0.5 * dt * force_fn(bx_new)
        bx = bx_new

        f_c = force_fn(cx)
        cp_half = cp + 0.5 * dt * f_c
        cx_new = cx + dt * cp_half
        cp = cp_half + 0.5 * dt * force_fn(cx_new)
        cx = cx_new

    # Shoelace formula for area of closed boundary polygon
    def poly_area(x_arr: np.ndarray, p_arr: np.ndarray) -> float:
        return float(0.5 * np.abs(np.sum(x_arr * np.roll(p_arr, -1) - np.roll(x_arr, -1) * p_arr)))

    area_0 = poly_area(bx0, bp0)
    area_t = poly_area(bx, bp)
    area_ratio = area_t / max(area_0, 1e-12)

    # Bounding-box estimate (coarse-grained / apparent Gibbs-vs-coarse entropy illustration)
    bbox_0 = (np.max(bx0) - np.min(bx0)) * (np.max(bp0) - np.min(bp0))
    bbox_t = (np.max(bx) - np.min(bx)) * (np.max(bp) - np.min(bp))
    bbox_ratio = bbox_t / max(bbox_0, 1e-12)

    # Background energy contours H(x, p)
    xg = np.linspace(-2.2, 2.2, 120)
    pg = np.linspace(-2.2, 2.2, 120)
    XX, PP = np.meshgrid(xg, pg)
    HH = 0.5 * (PP ** 2) + pot_fn(XX)
    ax.contour(XX, PP, HH, levels=8, colors="#CBD5E1", linewidths=0.8, alpha=0.75)

    # Plot initial patch at t = 0
    bx0_closed = np.append(bx0, bx0[0])
    bp0_closed = np.append(bp0, bp0[0])
    ax.fill(bx0_closed, bp0_closed, color=COLORS["muted"], alpha=0.22, label=r"Initial patch $\Omega(0)$")
    ax.plot(bx0_closed, bp0_closed, color=COLORS["muted"], linestyle="--", linewidth=1.4)

    # Plot evolved patch at time t
    bx_closed = np.append(bx, bx[0])
    bp_closed = np.append(bp, bp[0])
    ax.fill(bx_closed, bp_closed, color=COLORS["primary"], alpha=0.30, label=rf"Evolved patch $\Omega(t={time_t:.1f})$")
    ax.plot(bx_closed, bp_closed, color=COLORS["primary"], linewidth=2.0)
    ax.scatter(cx, cp, s=10, color=COLORS["accent"], alpha=0.75, zorder=4, label="Microstate trajectories")

    ax.axhline(0.0, color="#94A3B8", linewidth=0.7, linestyle=":")
    ax.axvline(0.0, color="#94A3B8", linewidth=0.7, linestyle=":")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel(r"Coordinate $q$")
    ax.set_ylabel(r"Momentum $p$")
    ax.set_title(f"Liouville Phase-Space Flow ({pot_type.capitalize()} Hamiltonian)")
    ax.legend(loc="lower left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Liouville's Theorem in Action:** At $t = {time_t:.2f}$, the exact phase-space area is "
        f"$\\Omega(t)/\\Omega(0) = {area_ratio:.4f}$ ($\\nabla_q \\dot{{q}} + \\nabla_p \\dot{{p}} = 0$), "
        f"confirming that Hamiltonian flow is strictly incompressible. However, in an anharmonic potential "
        f"($\\alpha = {anharmonicity:.2f}$), points at higher energy orbit at different frequencies and shear the "
        f"patch into thin filaments whose axis-aligned bounding box expands by **{bbox_ratio:.2f}×**."
    )


def render_lyapunov_chaos(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize exponential Lyapunov divergence of two nearby trajectories vs. machine precision horizon."""
    log10_eps = float(_unwrap(controls.get("log10_eps"), -12.0))
    lyap_lambda = float(_unwrap(controls.get("lyapunov_lambda"), 1.5))
    roughness = float(_unwrap(controls.get("roughness"), 1.0))

    # Time in picoseconds (0 to 30 ps)
    t_ps = np.linspace(0.0, 30.0, 300)
    eps0 = 10.0 ** log10_eps
    eff_lambda = lyap_lambda * (0.25 + 0.75 * roughness)

    # Deterministic chaotic fluctuation superposed on exponential growth, saturating at box scale L_sat = 1.0
    osc = 1.0 + 0.18 * np.sin(2.3 * t_ps) * np.cos(0.9 * t_ps)
    raw_sep = eps0 * np.exp(eff_lambda * t_ps) * osc
    sep = np.minimum(1.0, raw_sep)
    asymptotic = np.minimum(1.0, eps0 * np.exp(eff_lambda * t_ps))

    # Horizon time where separation reaches macroscopic O(1) scale: eps0 * exp(lambda * t_h) = 1
    t_horizon = -np.log(eps0) / max(eff_lambda, 1e-6)

    ax.semilogy(
        t_ps,
        sep,
        color=COLORS["primary"],
        linewidth=2.2,
        label=r"Trajectory separation $|\Delta \mathbf{R}(t)| / \sigma$",
    )
    ax.semilogy(
        t_ps,
        asymptotic,
        color=COLORS["accent"],
        linestyle="--",
        linewidth=1.6,
        label=rf"Lyapunov law $\varepsilon_0 e^{{\lambda t}}$ ($\lambda={eff_lambda:.2f}\ \mathrm{{ps}}^{{-1}}$)",
    )
    ax.axhline(1.0, color=COLORS["danger"], linestyle=":", linewidth=1.4, label=r"Decorrelation ceiling ($|\Delta \mathbf{R}| \sim \sigma$)")

    if t_horizon <= 30.0:
        ax.axvline(
            t_horizon,
            color=COLORS["good"],
            linestyle="-.",
            linewidth=1.5,
            label=rf"Predictability horizon $t^* \approx {t_horizon:.1f}\ \mathrm{{ps}}$",
        )

    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(1e-16, 5.0)
    ax.set_xlabel(r"Simulation Time $t$ (ps)")
    ax.set_ylabel(r"Normalized Separation $|\Delta \mathbf{R}(t)| / \sigma$")
    ax.set_title("Lyapunov Instability & The Trajectory Predictability Horizon")
    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="lower right", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Butterfly Effect in MD:** With initial perturbation $\\varepsilon_0 = 10^{{{log10_eps:.0f}}}$ "
        f"and effective Lyapunov exponent $\\lambda = {eff_lambda:.2f}\\ \\mathrm{{ps}}^{{-1}}$, the coordinate deviation "
        f"grows as $|\\Delta \\mathbf{{R}}(t)| \\approx \\varepsilon_0 e^{{\\lambda t}}$ and reaches the interatomic spacing "
        f"at **$t^* \\approx {t_horizon:.1f}\\ \\mathrm{{ps}}$**. Even IEEE-754 double precision ($10^{{-16}}$) loses "
        f"individual trajectory accuracy within tens of picoseconds—yet ensemble averages remain valid by **ergodicity**."
    )


def render_fpu_mode_sharing(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare Fermi-Pasta-Ulam (1954) normal-mode energy sloshing at low energy vs. ergodic equipartition at high energy."""
    specific_energy = float(_unwrap(controls.get("specific_energy"), 0.08))
    n_modes_shown = int(_unwrap(controls.get("n_modes_shown"), 4))

    t_norm = np.linspace(0.0, 20.0, 350)
    # Transition parameter between non-ergodic recurrence (low E) and chaotic equipartition (high E)
    chi = float(np.clip((specific_energy - 0.05) / 0.95, 0.0, 1.0))
    equipartition_level = 1.0 / 32.0

    palette = [COLORS["primary"], COLORS["accent"], COLORS["good"], COLORS["quaternary"], COLORS["secondary"]]

    # Mode 1 starts with 100% of energy
    decay_env = np.exp(-0.28 * chi * t_norm)
    e1_quasi = 1.0 - 0.68 * (np.sin(0.55 * t_norm) ** 2) * (1.0 - 0.15 * np.sin(0.11 * t_norm) ** 2)
    e1 = (1.0 - chi) * e1_quasi + chi * (equipartition_level + (1.0 - equipartition_level) * decay_env * np.cos(0.4 * t_norm) ** 2)
    ax.plot(t_norm, e1, color=palette[0], linewidth=2.2, label="Mode $k=1$ (initial)")

    for k in range(2, n_modes_shown + 1):
        phase = 0.55 * t_norm - 0.45 * (k - 1)
        amp_k = 0.45 / (k ** 1.15)
        ek_quasi = amp_k * (np.sin(phase) ** 2) * (np.sin(0.55 * t_norm) ** 2)
        ek_erg = equipartition_level * (1.0 - np.exp(-0.22 * (chi + 0.05) * t_norm / k)) + 0.04 * (1.0 - chi) * ek_quasi
        ek = (1.0 - chi) * ek_quasi + chi * ek_erg
        ax.plot(t_norm, ek, color=palette[(k - 1) % len(palette)], linewidth=1.7, label=f"Mode $k={k}$")

    ax.axhline(
        equipartition_level,
        color=COLORS["danger"],
        linestyle="--",
        linewidth=1.4,
        label=r"Equipartition limit $E_{\mathrm{tot}}/32 \approx 0.031$",
    )

    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Normalized Time $t / T_1$ ($10^3$ MD steps)")
    ax.set_ylabel(r"Fractional Mode Energy $E_k(t) / E_{\mathrm{tot}}$")
    regime_str = "Quasi-Periodic FPU Recurrence (Non-Ergodic)" if chi < 0.35 else "Chaotic Mixing Toward Equipartition (Ergodic)"
    ax.set_title(f"FPU Chain Mode Sharing ($\\epsilon = {specific_energy:.2f}$): {regime_str}")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Fermi-Pasta-Ulam-Tsingou (1954) Regime ($\\epsilon = {specific_energy:.2f}$):** "
        + (
            "At low specific energy, weak cubic anharmonicity fails to thermalize the 32-particle chain: energy "
            "sloshes quasi-periodically among only the lowest few normal modes and **violates ergodicity**."
            if chi < 0.35
            else "Above the stochasticity threshold, nonlinear mode coupling breaks integrability and drives all "
            "normal modes toward the **equipartition** share $E_k / E_{\\mathrm{tot}} \\to 1/N = 1/32$."
        )
    )


def render_virial_pressure_and_fluctuations(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Explore the Clausius virial pressure decomposition and canonical energy fluctuation width vs. density and N."""
    temperature = float(_unwrap(controls.get("temperature"), 1.2))
    n_particles = int(_unwrap(controls.get("n_particles"), 256))
    highlight_rho = float(_unwrap(controls.get("density_rho"), 0.65))

    rho = np.linspace(0.05, 0.95, 200)
    # Ideal kinetic pressure: P_id = rho * k_B * T
    p_ideal = rho * temperature
    # Lennard-Jones-like virial contribution: negative (attractive) at moderate rho, strongly positive (repulsive core) at high rho
    a_att = 3.8
    b_rep = 5.4
    p_virial = -a_att * (rho ** 2) + b_rep * (rho ** 4) * (1.0 + 0.15 * np.sqrt(temperature))
    p_total = p_ideal + p_virial

    # Evaluate at selected density
    p_id_pt = highlight_rho * temperature
    p_vir_pt = -a_att * (highlight_rho ** 2) + b_rep * (highlight_rho ** 4) * (1.0 + 0.15 * np.sqrt(temperature))
    p_tot_pt = p_id_pt + p_vir_pt
    z_factor = p_tot_pt / max(p_id_pt, 1e-8)

    # Canonical energy fluctuation relative width sigma_E / <E> ~ sqrt(k_B T^2 C_V) / (N * e) ~ 1/sqrt(N)
    cv_per_atom = 1.5 + 1.1 * (highlight_rho ** 1.5) / max(temperature, 0.3)
    rel_fluct = np.sqrt(cv_per_atom / (1.5 ** 2)) / np.sqrt(n_particles)

    ax.plot(rho, p_ideal, color=COLORS["secondary"], linestyle="--", linewidth=1.8, label=r"Kinetic (Ideal) $P_{\mathrm{id}} = \rho k_B T$")
    ax.plot(rho, p_virial, color=COLORS["accent"], linestyle="-.", linewidth=1.8, label=r"Virial $P_{\mathrm{vir}} = \frac{1}{3V}\langle \sum \mathbf{r}_{ij}\cdot\mathbf{F}_{ij}\rangle$")
    ax.plot(rho, p_total, color=COLORS["primary"], linewidth=2.4, label=r"Total Pressure $P = P_{\mathrm{id}} + P_{\mathrm{vir}}$")

    ax.axhline(0.0, color="#64748B", linewidth=0.9, linestyle=":")
    ax.axvline(highlight_rho, color=COLORS["good"], linewidth=1.4, linestyle=":")
    ax.scatter([highlight_rho, highlight_rho, highlight_rho], [p_id_pt, p_vir_pt, p_tot_pt], color=[COLORS["secondary"], COLORS["accent"], COLORS["primary"]], s=45, zorder=5)

    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(-2.0, 5.5)
    ax.set_xlabel(r"Reduced Number Density $\rho^* = N\sigma^3 / V$")
    ax.set_ylabel(r"Reduced Pressure $P^* = P\sigma^3 / \varepsilon$")
    ax.set_title(
        rf"Clausius Virial Pressure ($T^*={temperature:.1f}$) & Canonical Fluctuation ($N={n_particles}$)"
    )
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    regime = "attraction-dominated ($P_{\\mathrm{vir}} < 0$, $Z < 1$)" if p_vir_pt < 0 else "repulsive-core-dominated ($P_{\\mathrm{vir}} > 0$, $Z > 1$)"
    return (
        f"**State Point ($\\rho^* = {highlight_rho:.2f}, T^* = {temperature:.2f}, N = {n_particles}$):** "
        f"$P_{{\\mathrm{{id}}}}^* = {p_id_pt:.2f}$, $P_{{\\mathrm{{vir}}}}^* = {p_vir_pt:+.2f}$, giving total pressure "
        f"**$P^* = {p_tot_pt:.2f}$** (compressibility factor $Z = P/(\\rho k_B T) = {z_factor:.2f}$, {regime}). "
        f"In the canonical ($NVT$) ensemble, relative energy fluctuations scale as $\\sigma_H / \\langle K \\rangle \\approx {100.0 * rel_fluct:.1f}\\% \\propto N^{{-1/2}}$."
    )
