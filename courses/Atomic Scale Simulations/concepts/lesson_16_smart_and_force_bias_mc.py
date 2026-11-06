"""Concept visualization functions for Lesson 16: Smarter Monte Carlo — Menus, Preferential, Heat-Bath, Force-Bias & Smart MC."""

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


def render_preferential_surface_sampling(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize preferential surface sampling weight h(z) and the asymmetric proposal ratio T(R'->R)/T(R->R')."""
    z0_decay = float(controls.get("z0_decay", 2.0))
    z_old = float(controls.get("z_old", 8.5))
    dz_move = float(controls.get("dz_move", -1.2))

    z_top = 10.0
    z_grid = np.linspace(0.0, z_top, 200)
    # Weight function h(z) = 1 / [1 + ((z_top - z)/z0)^2] peaking at the top surface z = z_top
    h_grid = 1.0 / (1.0 + ((z_top - z_grid) / z0_decay) ** 2)

    # 19 background particles + 1 active particle i at z_old
    z_bg = np.linspace(0.5, 9.5, 19)
    H_bg = float(np.sum(1.0 / (1.0 + ((z_top - z_bg) / z0_decay) ** 2)))

    z_new = float(np.clip(z_old + dz_move, 0.0, z_top))
    h_old = float(1.0 / (1.0 + ((z_top - z_old) / z0_decay) ** 2))
    h_new = float(1.0 / (1.0 + ((z_top - z_new) / z0_decay) ** 2))

    H_old = H_bg + h_old
    H_new = H_bg + h_new

    T_fwd = h_old / H_old
    T_rev = h_new / H_new
    prop_ratio = T_rev / max(1e-12, T_fwd)

    # Shade slab layers (bulk vs subsurface vs surface)
    ax.axvspan(0.0, 6.0, color=COLORS["secondary"], alpha=0.10, label="Bulk Region ($z < 6$)")
    ax.axvspan(6.0, 8.2, color=COLORS["highlight"], alpha=0.12, label="Subsurface Layer ($6 \\le z < 8.2$)")
    ax.axvspan(8.2, 10.0, color=COLORS["accent"], alpha=0.14, label="Active Top Surface ($z \\ge 8.2$)")

    ax.plot(
        z_grid,
        h_grid,
        color=COLORS["primary"],
        linewidth=2.5,
        label=f"Selection Weight $h(z) = [1 + ((z_{{\\mathrm{{top}}}}-z)/z_0)^2]^{{-1}}$ ($z_0={z0_decay:.1f}$)",
    )
    ax.scatter(
        [z_old],
        [h_old],
        color=COLORS["primary"],
        s=75,
        zorder=6,
        label=f"Old Position $z_i = {z_old:.2f}$ ($h = {h_old:.3f}$)",
    )
    ax.scatter(
        [z_new],
        [h_new],
        color=COLORS["accent"],
        s=75,
        marker="D",
        zorder=6,
        label=f"Trial Position $z_i' = {z_new:.2f}$ ($h' = {h_new:.3f}$)",
    )
    ax.annotate(
        "",
        xy=(z_new, h_new),
        xytext=(z_old, h_old),
        arrowprops={"arrowstyle": "->", "color": COLORS["text"], "lw": 1.8},
    )

    _style_axes(
        ax,
        "Height Coordinate $z$ (Top Surface at $z_{\\mathrm{top}} = 10$)",
        "Preferential Selection Weight $h(z)$",
        "Preferential Surface Sampling & Asymmetric Proposal Ratio",
    )
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(0.0, 1.08)
    ax.legend(loc="upper left", fontsize=7.6)

    return (
        f"Moving particle $i$ from $z_i = {z_old:.2f}$ to $z_i' = {z_new:.2f}$ changes its selection weight from "
        f"$h(z_i) = {h_old:.3f}$ to $h(z_i') = {h_new:.3f}$ and updates the normalization sum from "
        f"$H(\\mathbf{{R}}) = {H_old:.3f}$ to $H(\\mathbf{{R}}') = {H_new:.3f}$. "
        f"The forward/reverse selection probabilities are $T_{{\\mathrm{{fwd}}}} = {T_fwd:.4f}$ and $T_{{\\mathrm{{rev}}}} = {T_rev:.4f}$, "
        f"requiring the Jacobian correction factor $\\frac{{T(\\mathbf{{R}}'\\to\\mathbf{{R}})}}{{T(\\mathbf{{R}}\\to\\mathbf{{R}}')}} = "
        f"\\frac{{h(z_i')/H'}}{{h(z_i)/H}} = {prop_ratio:.3f}$ in the Metropolis test!"
    )


def render_heat_bath_vs_metropolis_local(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare single-spin Heat-Bath (Glauber) conditional sampling against Metropolis spin flip."""
    beta_h_probe = float(controls.get("beta_h_loc", 0.6))
    current_spin = int(controls.get("current_spin", 1))

    bh = np.linspace(-2.5, 2.5, 300)
    # Energy of spin sigma in {-1, +1} in local field h_loc is E(sigma) = -h_loc * sigma
    # Flipping from current_spin to -current_spin changes energy by dE = 2 * current_spin * h_loc
    dE_beta = 2.0 * current_spin * bh

    # Metropolis flip probability: min(1, exp(-beta * dE))
    P_flip_metro = np.minimum(1.0, np.exp(-dE_beta))
    # Heat-Bath draws new spin s' in {-1, +1} with probability pi(s') / (pi(+1) + pi(-1))
    # Probability of ending up in flipped state s' = -current_spin is 1 / (1 + exp(+beta * dE))
    P_flip_hb = 1.0 / (1.0 + np.exp(dE_beta))
    # Notice Heat-Bath proposal T(s -> s') = pi(s')/C(s) has acceptance A = min(1, C(s)/C(s')) = 1.0!
    A_accept_hb = np.ones_like(bh)

    ax.plot(
        bh,
        P_flip_metro,
        color=COLORS["primary"],
        linewidth=2.4,
        label=f"Metropolis Flip Prob $P_M(\\sigma={current_spin:+d} \\to {-current_spin:+d})$",
    )
    ax.plot(
        bh,
        P_flip_hb,
        "--",
        color=COLORS["accent"],
        linewidth=2.2,
        label=f"Heat-Bath Flip Prob $T_{{HB}}(\\sigma' = {-current_spin:+d}) = \\frac{{\\pi(-\\sigma)}}{{\\pi(+1)+\\pi(-1)}}$",
    )
    ax.plot(
        bh,
        A_accept_hb,
        ":",
        color=COLORS["good"],
        linewidth=2.0,
        label="Heat-Bath Acceptance Ratio $A_{HB} = \\min(1, C(s)/C(s')) = 1.0$",
    )

    ax.axvline(beta_h_probe, color=COLORS["text"], linestyle="-.", linewidth=1.3)
    dE_p = 2.0 * current_spin * beta_h_probe
    pm_p = float(min(1.0, np.exp(-dE_p)))
    phb_p = float(1.0 / (1.0 + np.exp(dE_p)))

    ax.scatter([beta_h_probe], [pm_p], color=COLORS["primary"], s=60, zorder=5)
    ax.scatter([beta_h_probe], [phb_p], color=COLORS["accent"], s=60, zorder=5)

    _style_axes(
        ax,
        "Dimensionless Local Field $\\beta h_{\\mathrm{loc}} = \\beta J \\sum_{j \\in nn} \\sigma_j$",
        "Transition / Acceptance Probability",
        f"Heat-Bath Local Equilibrium vs. Metropolis (Current Spin $\\sigma_i = {current_spin:+d}$)",
    )
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.04, 1.08)
    ax.legend(loc="best", fontsize=7.8)

    return (
        f"At $\\beta h_{{\\mathrm{{loc}}}} = {beta_h_probe:.2f}$ with $\\sigma_i = {current_spin:+d}$ ($\\beta\\Delta E_{{\\mathrm{{flip}}}} = {dE_p:.2f}$): "
        f"Heat-Bath samples the local neighborhood $\\mathcal{{N}}(s) = \\{{-1, +1\\}}$ directly from $T(s\\to s') = \\pi(s')/C(s)$. "
        f"Because $C(s) = \\pi(+1) + \\pi(-1) = C(s')$ is independent of $\\sigma_i$, **Heat-Bath has $100\\%$ acceptance ($A=1$)** "
        f"and sets $\\sigma_i' = {-current_spin:+d}$ with probability ${phb_p:.3f}$ (vs. Metropolis flip probability ${pm_p:.3f}$)."
    )


def render_smart_mc_proposal_asymmetry(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare forward T(x->x') and reverse T(x'->x) Gaussian proposals in Smart Monte Carlo."""
    tau = float(controls.get("tau", 0.12))
    dx_trial = float(controls.get("dx_trial", 0.45))
    curvature = float(controls.get("curvature", 0.8))

    beta = 1.0
    x_old = 0.0
    x_new = x_old + dx_trial

    # Potential V(x) = -F0 * x + 0.5 * curvature * x^2  =>  F(x) = -V'(x) = F0 - curvature * x
    F0 = 2.0  # Strong downhill force to the right (+x)
    F_old = F0 - curvature * x_old
    F_new = F0 - curvature * x_new

    V_old = -F0 * x_old + 0.5 * curvature * (x_old ** 2)
    V_new = -F0 * x_new + 0.5 * curvature * (x_new ** 2)
    dV = V_new - V_old

    # Forward Gaussian centered at x_old + tau * beta * F_old with variance 2*tau
    mean_fwd = x_old + tau * beta * F_old
    # Reverse Gaussian (for displacement x - x' relative to x') centered at x_old = x_new - dx_trial
    # Plotted as a function of target coordinate u: T(x -> u) vs T(x' -> x) evaluated along displacement
    grid = np.linspace(-1.2, 1.8, 350)
    T_fwd_curve = (1.0 / np.sqrt(4.0 * np.pi * tau)) * np.exp(-((grid - mean_fwd) ** 2) / (4.0 * tau))

    # For reverse move starting at x_new, its Gaussian over destination y is centered at x_new + tau * beta * F_new
    mean_rev_dest = x_new + tau * beta * F_new
    T_rev_curve = (1.0 / np.sqrt(4.0 * np.pi * tau)) * np.exp(-((grid - mean_rev_dest) ** 2) / (4.0 * tau))

    T_fwd_val = float((1.0 / np.sqrt(4.0 * np.pi * tau)) * np.exp(-((x_new - mean_fwd) ** 2) / (4.0 * tau)))
    T_rev_val = float((1.0 / np.sqrt(4.0 * np.pi * tau)) * np.exp(-((x_old - mean_rev_dest) ** 2) / (4.0 * tau)))

    # Smart MC acceptance exponent:
    # -beta * [ dV + 0.5*(F_old + F_new)*(x_new - x_old) + 0.25*tau*beta*(F_new^2 - F_old^2) ]
    smart_exponent = -beta * (
        dV + 0.5 * (F_old + F_new) * (x_new - x_old) + 0.25 * tau * beta * ((F_new ** 2) - (F_old ** 2))
    )
    A_smart = float(min(1.0, np.exp(smart_exponent)))
    A_metro = float(min(1.0, np.exp(-beta * dV)))

    ax.plot(
        grid,
        T_fwd_curve,
        color=COLORS["primary"],
        linewidth=2.4,
        label=f"Forward Proposal $T(x \\to u)$ (drift $+\\tau\\beta F(x) = {mean_fwd:.2f}$)",
    )
    ax.plot(
        grid,
        T_rev_curve,
        "--",
        color=COLORS["accent"],
        linewidth=2.1,
        label=f"Reverse Proposal $T(x' \\to u)$ (centered at $x'+\\tau\\beta F(x') = {mean_rev_dest:.2f}$)",
    )

    ax.axvline(x_old, color=COLORS["text"], linestyle=":", linewidth=1.2, label=f"Start $x = {x_old:.2f}$")
    ax.axvline(x_new, color=COLORS["good"], linestyle="-.", linewidth=1.4, label=f"Trial $x' = {x_new:.2f}$")
    ax.scatter([x_new], [T_fwd_val], color=COLORS["primary"], s=65, zorder=6)
    ax.scatter([x_old], [T_rev_val], color=COLORS["accent"], s=65, zorder=6)

    _style_axes(
        ax,
        "Coordinate $u$",
        "Proposal Density $T$",
        f"Smart MC Displaced Gaussian Proposals ($F_0={F0:.1f}$, curvature $V''={curvature:.2f}$)",
    )
    ax.legend(loc="upper right", fontsize=7.6)

    return (
        f"For step $x={x_old:.2f} \\to x'={x_new:.2f}$ ($\\Delta V = {dV:.3f}$, $F(x)={F_old:.2f}$, $F(x')={F_new:.2f}$): "
        f"Forward density $T(x\\to x') = {T_fwd_val:.3f}$, Reverse density $T(x'\\to x) = {T_rev_val:.3f}$ "
        f"(ratio $T(x'\\to x)/T(x\\to x') = {T_rev_val / max(1e-12, T_fwd_val):.3f}$). "
        f"**Smart MC acceptance:** $A_{{\\mathrm{{Smart}}}} = {A_smart:.4f}$ "
        f"(note: when curvature $V'' = 0$, $A_{{\\mathrm{{Smart}}}} = 1.0000$ identically for any $\\tau$ and $\\Delta x$!)."
    )


def render_metropolis_vs_smart_mc_efficiency(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare acceptance probability and 100-step MSD across Metropolis, Force-Bias, and Smart MC."""
    delta_probe = float(controls.get("delta_over_sigma", 0.24))
    metric = str(controls.get("metric", "msd"))

    deltas = np.linspace(0.02, 0.48, 120)
    # Parameterize curves after Pangali, Rao & Berne / Rossky, Doll & Friedman (Figs. 1 & 3)
    acc_metro = np.exp(-((deltas / 0.19) ** 1.35))
    acc_fb_half = 1.0 / (1.0 + (deltas / 0.255) ** 2.3)
    acc_smart = 1.0 / (1.0 + (deltas / 0.245) ** 2.35)
    acc_fb_one = np.exp(-((deltas / 0.225) ** 1.45))

    # 100-step Mean-Squared Displacement <Delta r^2(100)>
    # Notice in Fig. 3: Force-Bias (lambda=0.5) & Smart MC peak near Delta/sigma ~ 0.22-0.25 (~5.0),
    # whereas classic Metropolis peaks higher (~5.6) at larger step size Delta/sigma ~ 0.34!
    msd_metro = 5.6 * ((deltas / 0.34) ** 1.8) * np.exp(1.0 - (deltas / 0.34) ** 1.8)
    msd_fb_half = 5.1 * ((deltas / 0.24) ** 1.9) * np.exp(1.0 - (deltas / 0.24) ** 1.9)
    msd_smart = 4.9 * ((deltas / 0.22) ** 1.95) * np.exp(1.0 - (deltas / 0.22) ** 1.95)
    msd_fb_one = 3.5 * ((deltas / 0.21) ** 1.85) * np.exp(1.0 - (deltas / 0.21) ** 1.85)

    idx_p = int(np.argmin(np.abs(deltas - delta_probe)))

    if metric == "acceptance":
        ax.plot(deltas, acc_metro, "-o", markevery=15, color=COLORS["primary"], linewidth=2.2, label="Metropolis MC")
        ax.plot(
            deltas,
            acc_fb_half,
            "-s",
            markevery=15,
            color=COLORS["good"],
            linewidth=2.2,
            label="Force-Bias MC ($\\lambda = 0.5$)",
        )
        ax.plot(
            deltas,
            acc_smart,
            "-^",
            markevery=15,
            color=COLORS["accent"],
            linewidth=2.2,
            label="Smart MC (Gaussian Drift)",
        )
        ax.plot(
            deltas,
            acc_fb_one,
            "--d",
            markevery=15,
            color=COLORS["quaternary"],
            linewidth=1.8,
            label="Force-Bias MC ($\\lambda = 1.0$)",
        )
        ax.axvline(delta_probe, color=COLORS["text"], linestyle=":", linewidth=1.4)
        _style_axes(
            ax,
            "Step Size $\\Delta / \\sigma$",
            "Average Acceptance Probability $P$",
            "Fig. 1 Benchmark: Acceptance Probability vs. Step Size",
        )
        ax.set_ylim(0.0, 1.05)
        ax.legend(loc="upper right", fontsize=8.0)
        return (
            f"At $\\Delta/\\sigma = {delta_probe:.2f}$: **Force-Bias ($\\lambda=0.5$)** ($A={acc_fb_half[idx_p]:.2f}$) and "
            f"**Smart MC** ($A={acc_smart[idx_p]:.2f}$) achieve significantly higher acceptance ratios than "
            f"**Classic Metropolis** ($A={acc_metro[idx_p]:.2f}$), especially at small-to-moderate $\\Delta/\\sigma$. "
            f"Notice that $\\lambda = 0.5$ outperforms $\\lambda = 1.0$ ($A={acc_fb_one[idx_p]:.2f}$) because $\\lambda=0.5$ cancels the trapezoidal force error!"
        )
    else:
        ax.plot(deltas, msd_metro, "-o", markevery=15, color=COLORS["primary"], linewidth=2.2, label="Metropolis MC")
        ax.plot(
            deltas,
            msd_fb_half,
            "-s",
            markevery=15,
            color=COLORS["good"],
            linewidth=2.2,
            label="Force-Bias MC ($\\lambda = 0.5$)",
        )
        ax.plot(
            deltas,
            msd_smart,
            "-^",
            markevery=15,
            color=COLORS["accent"],
            linewidth=2.2,
            label="Smart MC (Gaussian Drift)",
        )
        ax.plot(
            deltas,
            msd_fb_one,
            "--d",
            markevery=15,
            color=COLORS["quaternary"],
            linewidth=1.8,
            label="Force-Bias MC ($\\lambda = 1.0$)",
        )
        ax.axvline(delta_probe, color=COLORS["text"], linestyle=":", linewidth=1.4)
        _style_axes(
            ax,
            "Step Size $\\Delta / \\sigma$",
            "100-Step Diffusion $\\langle (\\mathbf{r}(i) - \\mathbf{r}(i+100))^2 \\rangle$ (a.u.)",
            "Fig. 3 Benchmark: Phase-Space Diffusion vs. Step Size",
        )
        ax.set_ylim(0.0, 6.2)
        ax.legend(loc="lower right", fontsize=8.0)
        return (
            f"At $\\Delta/\\sigma = {delta_probe:.2f}$: 100-step MSD is **{msd_fb_half[idx_p]:.2f}** (Force-Bias $\\lambda=0.5$), "
            f"**{msd_smart[idx_p]:.2f}** (Smart MC), and **{msd_metro[idx_p]:.2f}** (Metropolis). "
            f"While Force-Bias/Smart MC reduces energy autocorrelation $C(50)$ at moderate $\\Delta/\\sigma \\approx 0.20$, "
            f"classic Metropolis at a larger step $\\Delta/\\sigma \\approx 0.34$ ($A \\approx 0.20$) reaches the highest raw 100-step MSD ($\\sim 5.6$) without needing force evaluations!"
        )
