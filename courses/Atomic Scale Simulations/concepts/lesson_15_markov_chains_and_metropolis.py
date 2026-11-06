"""Concept visualization functions for Lesson 15: Markov Chains, Detailed Balance & Metropolis Monte Carlo."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def _style_axes(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render_markov_eigenvalue_convergence(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot discrete-time Markov chain state probabilities f_n(s) and spectral decay |eps_1|^n."""
    chain_type = str(controls.get("chain_type", "reliability"))
    f0_s0 = float(controls.get("f0_s0", 1.0))
    n_steps = int(controls.get("n_steps", 20))

    if chain_type == "reliability":
        # Worked example from 12.randomwalk.pdf: Up/Down reliability chain
        P = np.array([[0.6, 0.4], [0.3, 0.7]], dtype=float)
        pi_exact = np.array([3.0 / 7.0, 4.0 / 7.0])
        eps_1 = 0.3
        label_title = "Computer Reliability Chain: P = [[0.6, 0.4], [0.3, 0.7]]"
        s0_label, s1_label = "State 0 ('Up')", "State 1 ('Down')"
    elif chain_type == "bottleneck":
        # Slow-mixing metastable trap with small barrier crossing rate
        P = np.array([[0.94, 0.06], [0.04, 0.96]], dtype=float)
        pi_exact = np.array([0.4, 0.6])
        eps_1 = 0.90
        label_title = "Slow-Mixing Metastable Chain: P = [[0.94, 0.06], [0.04, 0.96]]"
        s0_label, s1_label = "Well A", "Well B"
    elif chain_type == "bipartite":
        # Periodic bipartite trap: eps_1 = -1.0 (oscillates forever)
        P = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        pi_exact = np.array([0.5, 0.5])
        eps_1 = -1.0
        label_title = "Periodic Bipartite Chain: P = [[0, 1], [1, 0]] (Non-Ergodic!)"
        s0_label, s1_label = "Sublattice A", "Sublattice B"
    else:
        # Reducible disconnected chain: eps_0 = 1, eps_1 = 1 (degenerate stationary states)
        P = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=float)
        pi_exact = np.array([f0_s0, 1.0 - f0_s0])
        eps_1 = 1.0
        label_title = "Reducible Disconnected Chain: P = [[1, 0], [0, 1]] (Infinite Barrier!)"
        s0_label, s1_label = "Component 0", "Component 1"

    steps = np.arange(n_steps + 1)
    f_hist = np.zeros((n_steps + 1, 2), dtype=float)
    f_curr = np.array([f0_s0, 1.0 - f0_s0], dtype=float)
    f_hist[0] = f_curr
    for n in range(1, n_steps + 1):
        f_curr = f_curr @ P
        f_hist[n] = f_curr

    dev0 = abs(f0_s0 - pi_exact[0])
    env_upper = pi_exact[0] + dev0 * (abs(eps_1) ** steps)
    env_lower = pi_exact[0] - dev0 * (abs(eps_1) ** steps)

    ax.plot(
        steps,
        f_hist[:, 0],
        "-o",
        color=COLORS["primary"],
        linewidth=2.2,
        markersize=4.5,
        label=f"$f_n(0)$: {s0_label}",
    )
    ax.plot(
        steps,
        f_hist[:, 1],
        "-s",
        color=COLORS["accent"],
        linewidth=1.9,
        markersize=4.0,
        alpha=0.85,
        label=f"$f_n(1)$: {s1_label}",
    )
    ax.axhline(
        pi_exact[0],
        color=COLORS["primary"],
        linestyle="--",
        linewidth=1.3,
        label=f"Stationary $\\pi(0) = {pi_exact[0]:.3f}$",
    )

    if abs(eps_1) < 1.0 and dev0 > 1e-6:
        ax.plot(
            steps,
            env_upper,
            ":",
            color=COLORS["good"],
            linewidth=1.5,
            label=f"Spectral envelope $\\pi(0) \\pm |\\varepsilon_1|^n$ ($|\\varepsilon_1|={abs(eps_1):.2f}$)",
        )
        ax.plot(steps, env_lower, ":", color=COLORS["good"], linewidth=1.5)

    _style_axes(ax, "Markov Step $n$", "State Probability $f_n(s)$", label_title)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8.2)

    if abs(eps_1) < 1.0:
        tau_eq = -1.0 / np.log(abs(eps_1))
        status_str = (
            f"**Ergodic convergence:** $\\varepsilon_0 = 1$, $\\varepsilon_1 = {eps_1:.2f}$, "
            f"relaxation time $\\tau_{{\\mathrm{{eq}}}} = -1/\\ln|\\varepsilon_1| \\approx {tau_eq:.2f}$ steps. "
            f"At $n={n_steps}$, $f_n = ({f_hist[-1, 0]:.4f},\\, {f_hist[-1, 1]:.4f})$ vs. $\\pi = ({pi_exact[0]:.4f},\\, {pi_exact[1]:.4f})$."
        )
    elif chain_type == "bipartite":
        status_str = (
            f"**Ergodicity failure (Periodicity):** $\\varepsilon_1 = -1.0$ has unit magnitude $|\\varepsilon_1|=1$. "
            f"The distribution alternates between $({f0_s0:.2f}, {1-f0_s0:.2f})$ and $({1-f0_s0:.2f}, {f0_s0:.2f})$ forever without converging!"
        )
    else:
        status_str = (
            f"**Ergodicity failure (Reducibility / Infinite Barrier):** Eigenvalue $\\varepsilon = 1$ is doubly degenerate. "
            f"The chain remains trapped in its initial distribution $f_n = f_0 = ({f0_s0:.2f}, {1-f0_s0:.2f})$."
        )
    return status_str


def render_metropolis_vs_barker_acceptance(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare Metropolis A_M = min(1, e^{-beta dE}) and Barker A_B = 1/(1 + e^{beta dE}) acceptance rules."""
    beta_dE_probe = float(controls.get("beta_dE", 0.8))
    u_rand = float(controls.get("u_rand", 0.35))

    x = np.linspace(-3.0, 4.0, 350)
    q = np.exp(-x)
    A_metro = np.minimum(1.0, q)
    A_barker = q / (1.0 + q)

    ax.plot(x, A_metro, color=COLORS["primary"], linewidth=2.5, label="Metropolis $A_M = \\min(1, e^{-\\beta\\Delta E})$")
    ax.plot(
        x,
        A_barker,
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=2.2,
        label="Barker / Heat-Bath $A_B = \\frac{1}{1 + e^{\\beta\\Delta E}}$",
    )
    ax.fill_between(
        x,
        A_barker,
        A_metro,
        color=COLORS["primary"],
        alpha=0.14,
        label="Peskun advantage ($A_M - A_B \\ge 0$)",
    )

    ax.axvline(0.0, color=COLORS["muted"], linestyle=":", linewidth=1.2)
    ax.axvline(beta_dE_probe, color=COLORS["accent"], linestyle="-.", linewidth=1.4)

    am_val = float(min(1.0, np.exp(-beta_dE_probe)))
    ab_val = float(1.0 / (1.0 + np.exp(beta_dE_probe)))
    accepted_m = u_rand <= am_val

    ax.scatter(
        [beta_dE_probe],
        [u_rand],
        color=COLORS["good"] if accepted_m else COLORS["danger"],
        s=75,
        zorder=6,
        edgecolor="white",
        linewidth=1.5,
        label=f"Trial $(\\beta\\Delta E={beta_dE_probe:.2f},\\, u={u_rand:.2f})$: {'ACCEPT' if accepted_m else 'REJECT'}",
    )

    _style_axes(
        ax,
        "Dimensionless Energy Change $\\beta \\Delta E = (E(s') - E(s)) / k_B T$",
        "Acceptance Probability $A(s \\to s')$",
        "Metropolis vs. Barker Acceptance Rules & Peskun's Theorem",
    )
    ax.set_xlim(-3.0, 4.0)
    ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="lower left", fontsize=8.0)

    return (
        f"At $\\beta\\Delta E = {beta_dE_probe:.2f}$ ($q = e^{{-\\beta\\Delta E}} = {np.exp(-beta_dE_probe):.3f}$): "
        f"**Metropolis** $A_M = {am_val:.3f}$ vs. **Barker** $A_B = {ab_val:.3f}$ "
        f"(advantage $A_M - A_B = {am_val - ab_val:.3f}$). "
        f"With uniform random draw $u = {u_rand:.2f}$, Metropolis **{'ACCEPTS' if accepted_m else 'REJECTS'}** the move."
    )


def render_metropolis_step_size_optimizer(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot acceptance ratio A(Delta) and diffusive efficiency MSD(Delta) vs proposal cube width Delta."""
    delta_probe = float(controls.get("delta_over_sigma", 0.24))
    temp_scale = float(controls.get("temp_scale", 1.0))

    deltas = np.linspace(0.02, 0.60, 120)
    # Realistic liquid-Argon-like acceptance decay and MSD curve modeled after 12.randomwalk.pdf Fig 1 & Fig 3
    char_width = 0.19 * np.sqrt(temp_scale)
    acc_curve = np.exp(-((deltas / char_width) ** 1.35))
    # Mean-squared displacement per 100 steps peaks around Delta/sigma ~ 0.30-0.36 where A ~ 0.20-0.30
    msd_raw = (deltas ** 2) * np.exp(-((deltas / (1.15 * char_width)) ** 1.55))
    msd_norm = 5.6 * msd_raw / np.max(msd_raw)

    acc_probe = float(np.exp(-((delta_probe / char_width) ** 1.35)))
    msd_probe_raw = (delta_probe ** 2) * np.exp(-((delta_probe / (1.15 * char_width)) ** 1.55))
    msd_probe = float(5.6 * msd_probe_raw / np.max(msd_raw))
    opt_idx = int(np.argmax(msd_norm))
    delta_opt = float(deltas[opt_idx])
    acc_opt = float(acc_curve[opt_idx])

    ax.plot(
        deltas,
        acc_curve,
        color=COLORS["primary"],
        linewidth=2.3,
        label="Acceptance Ratio $A(\\Delta/\\sigma)$",
    )
    ax.plot(
        deltas,
        msd_norm / 5.6,
        color=COLORS["accent"],
        linewidth=2.3,
        linestyle="-",
        label="Normalized Diffusion $\\langle \\Delta r^2(100) \\rangle / \\mathrm{max}$",
    )

    ax.axhspan(0.10, 0.90, color=COLORS["good"], alpha=0.08, label="Safe Rule Band ($0.1 < A < 0.9$)")
    ax.axhline(0.50, color=COLORS["primary"], linestyle=":", linewidth=1.1, label="50% Rule of Thumb ($A=0.5$)")
    ax.axvline(delta_probe, color=COLORS["text"], linestyle="--", linewidth=1.4)
    ax.scatter([delta_probe], [acc_probe], color=COLORS["primary"], s=55, zorder=5)
    ax.scatter([delta_probe], [msd_probe / 5.6], color=COLORS["accent"], s=55, zorder=5)

    _style_axes(
        ax,
        "Maximum Trial Displacement Width $\\Delta / \\sigma$",
        "Acceptance $A$  /  Relative Diffusion Efficiency",
        "Step-Size Tradeoff: Acceptance Ratio vs. Phase-Space Diffusion",
    )
    ax.set_xlim(0.02, 0.60)
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="upper right", fontsize=7.8)

    return (
        f"At $\\Delta/\\sigma = {delta_probe:.2f}$: **Acceptance ratio** $A = {acc_probe:.2f}$ "
        f"and **100-step MSD** $\\langle \\Delta r^2(100)\\rangle = {msd_probe:.2f}$ a.u. "
        f"Peak diffusive efficiency occurs near $\\Delta^*/\\sigma \\approx {delta_opt:.2f}$ "
        f"where the acceptance ratio is $A(\\Delta^*) \\approx {acc_opt:.2f}$ ($\\sim 20\\%\\text{{--}}25\\%$), "
        f"outperforming the conservative $A = 0.50$ rule of thumb!"
    )


def render_rejected_state_bug_demo(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Demonstrate how omitting rejected states from MCMC averages biases P(E) away from Boltzmann."""
    delta_step = float(controls.get("delta_step", 1.8))
    beta = float(controls.get("beta", 1.0))

    # Discrete 1D harmonic energy levels E_k = 0.5 * x_k^2 on a fine grid for exact deterministic Markov matrix
    x_grid = np.linspace(-3.5, 3.5, 141)
    dx = x_grid[1] - x_grid[0]
    E_grid = 0.5 * (x_grid ** 2)
    pi_boltz = np.exp(-beta * E_grid)
    pi_boltz /= np.sum(pi_boltz)

    # Construct transition proposal T(i -> j): uniform within |x_j - x_i| <= delta_step
    n_pts = len(x_grid)
    half_span = max(1, int(round(delta_step / dx)))
    P_correct = np.zeros((n_pts, n_pts), dtype=float)
    P_buggy = np.zeros((n_pts, n_pts), dtype=float)

    for i in range(n_pts):
        j_min = max(0, i - half_span)
        j_max = min(n_pts - 1, i + half_span)
        n_cand = j_max - j_min + 1
        for j in range(j_min, j_max + 1):
            dE = E_grid[j] - E_grid[i]
            acc = 1.0 if dE <= 0.0 else np.exp(-beta * dE)
            if j != i:
                P_correct[i, j] = acc / n_cand
                P_buggy[i, j] = acc / n_cand
        # Diagonal of correct Markov chain includes all rejected moves (staying at state i)
        P_correct[i, i] = 1.0 - np.sum(P_correct[i, :])
        # Buggy chain only records accepted moves -> renormalizes off-diagonal acceptance probabilities!
        row_sum = np.sum(P_buggy[i, :])
        if row_sum > 0:
            P_buggy[i, :] /= row_sum
        else:
            P_buggy[i, i] = 1.0

    # Find stationary distribution of buggy chain via power iteration
    pi_bug = np.copy(pi_boltz)
    for _ in range(400):
        pi_bug = pi_bug @ P_buggy
    pi_bug /= np.sum(pi_bug)

    # Compute acceptance rate at equilibrium
    avg_acc_per_state = 1.0 - np.diag(P_correct)
    overall_acc = float(np.sum(pi_boltz * avg_acc_per_state))
    mean_E_exact = float(np.sum(pi_boltz * E_grid))
    mean_E_correct = float(np.sum((pi_boltz @ P_correct) * E_grid))
    mean_E_bug = float(np.sum(pi_bug * E_grid))

    ax.plot(
        x_grid,
        pi_boltz / dx,
        color=COLORS["good"],
        linewidth=2.6,
        label=f"Exact Boltzmann $\\pi(x)$ ($\\langle E \\rangle = {mean_E_exact:.3f}$)",
    )
    ax.plot(
        x_grid,
        (pi_boltz @ P_correct) / dx,
        "--",
        color=COLORS["primary"],
        linewidth=2.0,
        label=f"Correct MCMC (Re-counts rejected $s_{{n+1}}=s_n$, $\\langle E \\rangle = {mean_E_correct:.3f}$)",
    )
    ax.plot(
        x_grid,
        pi_bug / dx,
        "-.",
        color=COLORS["danger"],
        linewidth=2.2,
        label=f"BUG: Accepted-Only Average ($\\langle E \\rangle = {mean_E_bug:.3f}$)",
    )
    ax.fill_between(x_grid, pi_boltz / dx, pi_bug / dx, color=COLORS["danger"], alpha=0.14)

    _style_axes(
        ax,
        "Coordinate $x$ in Harmonic Potential $V(x) = \\frac{1}{2}x^2$",
        "Sampled Probability Density $P(x)$",
        "Classic MCMC Bug: Why Rejected States Must Count Equally",
    )
    ax.legend(loc="upper right", fontsize=7.8)

    return (
        f"At proposal width $\\Delta = {delta_step:.2f}$ and $\\beta = {beta:.2f}$ (equilibrium acceptance $A = {overall_acc:.2f}$): "
        f"low-energy states near $x=0$ reject uphill moves more frequently than high-energy states reject downhill moves. "
        f"Omitting rejections underweights the potential minimum and inflates the mean energy from "
        f"$\\langle E \\rangle_{{\\mathrm{{exact}}}} = {mean_E_exact:.3f}$ to $\\langle E \\rangle_{{\\mathrm{{bug}}}} = {mean_E_bug:.3f}$ "
        f"(+{100.0 * (mean_E_bug - mean_E_exact) / max(1e-8, mean_E_exact):.1f}% systematic bias)!"
    )
