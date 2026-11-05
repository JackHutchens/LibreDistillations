"""Interactive concept visualizations for Lesson 23: Phase Transitions, Renormalization Group & Finite-Size Scaling."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def render_swendsen_wang_slowing_down(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare local Metropolis vs. Swendsen-Wang cluster autocorrelation scaling and FK bond probability."""
    dim = int(controls.get("dimension", 2))
    L_val = int(controls.get("L", 32))
    view_mode = str(controls.get("view_mode", "tau_scaling"))

    if dim == 2:
        z_local = 2.125
        z_cluster = 0.35
        kT_c = 2.269185
        dim_label = "2D Square Lattice"
    else:
        z_local = 2.02
        z_cluster = 0.50
        kT_c = 4.5115
        dim_label = "3D Simple Cubic Lattice"

    L_grid = np.logspace(0.6, 2.3, 120)
    tau_local = 0.65 * (L_grid ** z_local)
    tau_cluster = 1.40 * (L_grid ** z_cluster)

    tau_loc_L = 0.65 * (L_val ** z_local)
    tau_clu_L = 1.40 * (L_val ** z_cluster)
    speedup_tau = tau_loc_L / tau_clu_L

    if view_mode == "cpu_cost":
        y_local = (L_grid ** dim) * tau_local
        y_cluster = (L_grid ** dim) * tau_cluster
        y_loc_L = (L_val ** dim) * tau_loc_L
        y_clu_L = (L_val ** dim) * tau_clu_L

        ax.loglog(
            L_grid,
            y_local,
            color=COLORS["danger"],
            linewidth=2.4,
            label=f"Local Metropolis: $\\propto L^{{d+z}} = L^{{{dim + z_local:.3f}}}$",
        )
        ax.loglog(
            L_grid,
            y_cluster,
            color=COLORS["primary"],
            linewidth=2.4,
            label=f"Swendsen–Wang Cluster: $\\propto L^{{d+z}} = L^{{{dim + z_cluster:.2f}}}$",
        )
        ax.scatter([L_val, L_val], [y_loc_L, y_clu_L], color=[COLORS["danger"], COLORS["primary"]], s=65, zorder=5)
        ax.axvline(L_val, color=COLORS["muted"], linestyle="--", linewidth=1.2)
        ax.set_xlabel("Linear System Size $L$")
        ax.set_ylabel("CPU Effort per Independent Sample ($\\propto L^d \\tau$)")
        ax.set_title(f"Total CPU Cost at $T_c$ ({dim_label})")
    else:
        ax.loglog(
            L_grid,
            tau_local,
            color=COLORS["danger"],
            linewidth=2.4,
            label=f"Local Metropolis ($z = {z_local:.3f}$)",
        )
        ax.loglog(
            L_grid,
            tau_cluster,
            color=COLORS["primary"],
            linewidth=2.4,
            label=f"Swendsen–Wang Cluster ($z = {z_cluster:.2f}$)",
        )
        ax.scatter(
            [L_val, L_val],
            [tau_loc_L, tau_clu_L],
            color=[COLORS["danger"], COLORS["primary"]],
            s=65,
            zorder=5,
        )
        ax.axvline(L_val, color=COLORS["muted"], linestyle="--", linewidth=1.2, label=f"Selected $L = {L_val}$")
        ax.set_xlabel("Linear System Size $L$ (Cutoff for $\\xi$ at $T_c$)")
        ax.set_ylabel("Autocorrelation Time $\\tau$ (MC sweeps)")
        ax.set_title(f"Critical Slowing Down at $T_c$ ({dim_label}: PRLs 1987/1989)")

    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    p_fk_tc = 1.0 - np.exp(-2.0 / kT_c)
    return (
        f"**At $L = {L_val}$ in {dim}D ($k_B T_c / J \\approx {kT_c:.3f}$):** "
        f"Local single-spin Metropolis has $\\tau_\\mathrm{{local}} \\approx {tau_loc_L:,.1f}$ sweeps ($z = {z_local:.3f}$), "
        f"whereas Swendsen–Wang cluster updates bond aligned neighbors with Fortuin–Kasteleyn probability "
        f"$p(T_c) = 1 - e^{{-2J/k_B T_c}} \\approx {p_fk_tc:.3f}$, reducing $\\tau_\\mathrm{{cluster}}$ to "
        f"$\\approx {tau_clu_L:,.1f}$ sweeps ($z = {z_cluster:.2f}$) — a **{speedup_tau:,.1f}$\\times$ reduction** in decorrelation time!"
    )


def render_kadanoff_rg_flow(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render the Kadanoff-Wilson RG recursion map K' = R(K) and cobweb flow trajectory."""
    K0 = float(controls.get("K0", 0.42))
    n_steps = int(controls.get("n_steps", 5))
    b_scale = int(controls.get("b_scale", 2))

    # Exact Migdal-Kadanoff / approximate 2D majority-rule style recursion with fixed point K*
    # For b=2 in 2D Ising, K* ~ 0.4407 and slope at K* is b^(1/nu) = 2^1 = 2.0 (exact 2D nu = 1.0)
    # Use a smooth, physically accurate RG recursion curve R(K) with R(0)=0, R(K*)=K*, R'(K*)=b^(1/nu), R(K)->b*K at large K.
    K_star = 0.4407
    nu_exact = 1.0 if b_scale == 2 else 1.02
    slope_target = float(b_scale) ** (1.0 / nu_exact)

    def rg_map(K: np.ndarray | float) -> np.ndarray | float:
        u = K / K_star
        # Rational-power RG map: R(K) = K* * u^p / (1 + (u^p - 1)*exp(-c*(u-1)))
        # Simple clean form with exact R(0)=0, R(K*)=K*, and R'(K*)=slope_target:
        p = slope_target
        return K_star * (u ** p) / (0.35 * (u ** p) + 0.65 * (u ** (p - 1.0) if np.all(u > 0) else 1.0))

    def rg_scalar(k_val: float) -> float:
        if k_val <= 1e-6:
            return 0.0
        u = k_val / K_star
        p = slope_target + 0.35 * (slope_target - 1.0)
        # Ensure R(K*) = K* and derivative at u=1 is slope_target:
        # Let f(u) = u^a / (1 + alpha*(u^(a-1) - 1)). At u=1, f(1)=1, f'(1) = a - alpha*(a-1).
        a = 2.45 if b_scale == 2 else 3.25
        alpha = (a - slope_target) / (a - 1.0)
        denom = 1.0 + alpha * (u ** (a - 1.0) - 1.0)
        return float(K_star * (u ** a) / max(1e-6, denom))

    K_vals = np.linspace(0.0, 1.15, 250)
    R_vals = np.array([rg_scalar(k) for k in K_vals])

    ax.plot(K_vals, R_vals, color=COLORS["primary"], linewidth=2.5, label="RG Recursion $K' = R(K)$")
    ax.plot(K_vals, K_vals, color=COLORS["muted"], linestyle="--", linewidth=1.6, label="Fixed-Point Line $K' = K$")

    # Mark fixed points
    ax.scatter([0.0, K_star], [0.0, K_star], color=[COLORS["good"], COLORS["danger"]], s=75, zorder=6)
    ax.annotate(
        f"Unstable Critical FP\n$K^* \\approx {K_star:.3f}$ ($T = T_c$)",
        xy=(K_star, K_star),
        xytext=(K_star + 0.12, K_star - 0.16),
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["danger"],
        arrowprops={"arrowstyle": "->", "color": COLORS["danger"], "lw": 1.2},
    )
    ax.annotate(
        "High-$T$ Stable FP\n$K=0$ ($T=\\infty$)",
        xy=(0.01, 0.01),
        xytext=(0.12, 0.04),
        fontsize=8.0,
        color=COLORS["good"],
    )

    # Cobweb trajectory from K0
    traj_K = [K0]
    curr_k = K0
    for step in range(n_steps):
        next_k = rg_scalar(curr_k)
        next_k_clipped = min(next_k, 1.35)
        # Vertical segment from (curr_k, curr_k) or (curr_k, 0) to (curr_k, next_k)
        y_start = 0.0 if step == 0 else curr_k
        ax.plot([curr_k, curr_k], [y_start, next_k_clipped], color=COLORS["accent"], linewidth=1.8, alpha=0.9)
        # Horizontal segment from (curr_k, next_k) to (next_k, next_k)
        ax.plot([curr_k, next_k_clipped], [next_k_clipped, next_k_clipped], color=COLORS["accent"], linewidth=1.8, alpha=0.9)
        curr_k = next_k_clipped
        traj_K.append(curr_k)
        if curr_k >= 1.30 or curr_k <= 1e-4:
            break

    ax.scatter([K0], [0.0], color=COLORS["accent"], s=60, zorder=7, label=f"Initial $K_0 = {K0:.3f}$")

    ax.set_xlim(0.0, 1.15)
    ax.set_ylim(0.0, 1.15)
    ax.set_xlabel("Dimensionless Coupling $K = J / (k_B T)$")
    ax.set_ylabel("Coarse-Grained Coupling $K' = R(K)$")
    ax.set_title(f"Kadanoff Block-Spin RG Flow ($b = {b_scale}$, $\\lambda_T = dR/dK|_{{K^*}} = {slope_target:.2f}$)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_T = np.log(slope_target) / np.log(b_scale)
    nu_est = 1.0 / y_T
    if abs(K0 - K_star) < 0.008:
        regime = "Right on the critical manifold ($K_0 \\approx K^*$, $\\xi = \\infty$): scale-invariant fluctuations persist!"
    elif K0 < K_star:
        regime = (
            f"Disordered phase ($K_0 < K^* \\iff T > T_c$): RG coarse-graining drives $K^{{({len(traj_K)-1})}} \\to {traj_K[-1]:.4f}$ "
            f"toward the $K=0$ ($T=\\infty$) trivial paramagnetic sink as $\\xi' = \\xi / {b_scale}$ shrinks."
        )
    else:
        regime = (
            f"Ordered phase ($K_0 > K^* \\iff T < T_c$): RG coarse-graining drives $K^{{({len(traj_K)-1})}} \\to {traj_K[-1]:.3f}$ "
            f"toward the $K=\\infty$ ($T=0$) ferromagnetic sink."
        )

    return (
        f"**RG Flow Diagnosis:** {regime} "
        f"Linearizing around $K^*$ gives slope $\\lambda_T = b^{{y_T}} = {slope_target:.2f}$, so "
        f"$y_T = \\frac{{\\ln \\lambda_T}}{{\\ln b}} = {y_T:.2f}$ and $\\nu = 1/y_T = {nu_est:.2f}$."
    )


def render_finite_size_scaling_collapse(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Toggle between raw finite-size susceptibility curves and rescaled FSS data collapse."""
    mode = str(controls.get("plot_mode", "collapsed"))
    nu_trial = float(controls.get("nu_trial", 1.00))
    gamma_trial = float(controls.get("gamma_trial", 1.75))

    # True 2D Ising exact exponents: Tc = 2.269, nu = 1.0, gamma = 1.75
    Tc_true = 2.269
    nu_true = 1.00
    gamma_true = 1.75

    L_list = [8, 16, 32, 64]
    palette = [COLORS["community_a"], COLORS["community_b"], COLORS["community_c"], COLORS["community_d"]]

    # Universal scaling function chi_tilde(x) where x = t * L^(1/nu_true)
    # Peak occurs at x_peak = -0.65 (representing the finite-size shift Tc(L) - Tc(inf) ~ x_peak * L^(-1/nu))
    def chi_scaling_fn(x: np.ndarray) -> np.ndarray:
        x_p = -0.60
        return 1.25 / (1.0 + 0.42 * ((x - x_p) ** 2) + 0.015 * ((x - x_p) ** 4)) ** (gamma_true / (2.0 * nu_true))

    T_grid = np.linspace(1.90, 2.65, 220)
    t_grid = (T_grid - Tc_true) / Tc_true

    spread_metric = 0.0
    ref_curve = None

    for idx, L in enumerate(L_list):
        x_true = t_grid * (L ** (1.0 / nu_true))
        chi_raw = (L ** (gamma_true / nu_true)) * chi_scaling_fn(x_true)

        if mode == "raw":
            ax.plot(T_grid, chi_raw, color=palette[idx], linewidth=2.1, label=f"$L = {L}$")
            # Mark peak
            pk_idx = int(np.argmax(chi_raw))
            ax.scatter([T_grid[pk_idx]], [chi_raw[pk_idx]], color=palette[idx], s=35, zorder=5)
        else:
            x_scaled = t_grid * (L ** (1.0 / nu_trial))
            y_scaled = chi_raw * (L ** (-gamma_trial / nu_trial))
            ax.plot(x_scaled, y_scaled, color=palette[idx], linewidth=2.1, label=f"$L = {L}$")

            # Evaluate on common x_eval grid in [-3, 3] to measure data collapse quality
            x_eval = np.linspace(-3.0, 3.0, 80)
            t_eval = x_eval / (L ** (1.0 / nu_trial))
            chi_eval = (L ** (gamma_true / nu_true)) * chi_scaling_fn(t_eval * (L ** (1.0 / nu_true)))
            y_eval = chi_eval * (L ** (-gamma_trial / nu_trial))
            if ref_curve is None:
                ref_curve = y_eval
            else:
                spread_metric += float(np.mean(np.abs(y_eval - ref_curve)))

    if mode == "raw":
        ax.axvline(Tc_true, color=COLORS["muted"], linestyle="--", linewidth=1.3, label=f"Bulk $T_c = {Tc_true:.3f}$")
        ax.set_xlabel("Temperature $T$ ($J/k_B$)")
        ax.set_ylabel("Finite-Size Susceptibility $\\chi(T, L)$")
        ax.set_title("Unscaled Finite-Size Susceptibility $\\chi(T, L)$: Peak Rounding & Shift")
        status_msg = (
            "In finite boxes $L$, the bulk divergence $\\chi \\propto |t|^{-\\gamma}$ is cut off when $\\xi \\sim L$, "
            "yielding rounded peaks of height $\\chi_{\\max}(L) \\propto L^{\\gamma/\\nu} = L^{1.75}$ shifted to $T_c(L) - T_c \\propto L^{-1/\\nu}$."
        )
    else:
        ax.axvline(0.0, color=COLORS["muted"], linestyle="--", linewidth=1.2)
        ax.set_xlim(-4.5, 4.5)
        ax.set_xlabel("Scaled Temperature Variable $x = t \\, L^{1/\\nu_{\\text{trial}}}$")
        ax.set_ylabel("Scaled Susceptibility $\\chi(T, L) \\, L^{-\\gamma_{\\text{trial}} / \\nu_{\\text{trial}}}$")
        ax.set_title(
            f"FSS Data Collapse ($\\nu_{{\\text{{trial}}}} = {nu_trial:.2f}$, $\\gamma_{{\\text{{trial}}}} = {gamma_trial:.2f}$)"
        )
        is_collapsed = abs(nu_trial - nu_true) < 0.04 and abs(gamma_trial - gamma_true) < 0.06
        quality = "EXCELLENT Universal Collapse!" if is_collapsed else f"Misaligned (curve spread = {spread_metric:.2f}) — tune $\\nu \\to 1.00, \\gamma \\to 1.75$!"
        status_msg = (
            f"**Collapse Status:** {quality} "
            f"True 2D Ising exponents are $\\nu = {nu_true:.2f}$ and $\\gamma = {gamma_true:.2f}$ ($\\gamma/\\nu = 7/4 = 1.75$)."
        )

    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return status_msg


def render_binder_cumulant_crossing(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot Binder's 4th-order cumulant U_4(T, L) across system sizes L and highlight the crossing at Tc."""
    T_probe = float(controls.get("T_probe", 2.27))
    show_limits = int(controls.get("show_limits", 1))

    Tc = 2.269185
    nu = 1.0
    U4_crit = 0.6106  # Universal 2D Ising periodic-box Binder cumulant at Tc

    L_list = [10, 20, 40, 80]
    palette = [COLORS["community_a"], COLORS["community_b"], COLORS["community_c"], COLORS["community_d"]]

    T_grid = np.linspace(2.12, 2.42, 240)
    t_grid = (T_grid - Tc) / Tc

    def u4_curve(t_arr: np.ndarray, L: int) -> np.ndarray:
        x = t_arr * (L ** (1.0 / nu))
        # Smooth monotonic transition from 2/3 (x -> -inf) through U4_crit at x=0 to 0 (x -> +inf)
        # Matching U4(0) = U4_crit = 0.6106:
        # Let U4(x) = (2/3) / (1 + c * exp(k * x)) where 1/(1+c) = U4_crit / (2/3) => c = (2/3)/0.6106 - 1 = 0.09182
        c = (2.0 / 3.0) / U4_crit - 1.0
        k = 1.55
        return (2.0 / 3.0) / (1.0 + c * np.exp(k * x))

    u4_probe_vals = []
    for idx, L in enumerate(L_list):
        u_vals = u4_curve(t_grid, L)
        ax.plot(T_grid, u_vals, color=palette[idx], linewidth=2.2, label=f"$L = {L}$")
        u_p = float(u4_curve(np.array([(T_probe - Tc) / Tc]), L)[0])
        u4_probe_vals.append(u_p)
        ax.scatter([T_probe], [u_p], color=palette[idx], s=38, zorder=5)

    ax.axvline(Tc, color=COLORS["danger"], linestyle="--", linewidth=1.4, label=f"$T_c \\approx {Tc:.3f}$ Crossing")
    ax.axvline(T_probe, color=COLORS["accent"], linestyle=":", linewidth=1.5, label=f"Probe $T = {T_probe:.3f}$")
    ax.scatter([Tc], [U4_crit], color=COLORS["danger"], s=75, zorder=6)

    if show_limits:
        ax.axhline(2.0 / 3.0, color=COLORS["good"], linestyle="-.", linewidth=1.2, label="Ordered Limit $2/3$ ($T < T_c$)")
        ax.axhline(0.0, color=COLORS["muted"], linestyle="-.", linewidth=1.2, label="Gaussian Limit $0$ ($T > T_c$)")
        ax.set_ylim(-0.05, 0.72)
    else:
        ax.set_ylim(0.25, 0.68)

    ax.set_xlim(2.12, 2.42)
    ax.set_xlabel("Temperature $T$ ($J/k_B$)")
    ax.set_ylabel("Binder Cumulant $U_4(T, L) = 1 - \\frac{\\langle M^4 \\rangle}{3\\langle M^2 \\rangle^2}$")
    ax.set_title("Binder 4th-Order Cumulant Crossing for 2D Ising Lattices")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="lower left", fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    spread = max(u4_probe_vals) - min(u4_probe_vals)
    if abs(T_probe - Tc) < 0.008:
        phase_desc = (
            f"At $T \\approx T_c = {Tc:.3f}$, scale invariance cancels all $L^{{-\\beta/\\nu}}$ prefactors so all box sizes intersect at "
            f"$U_4^* \\approx {U4_crit:.3f}$ (spread = {spread:.4f}) without needing prior knowledge of $\\beta$ or $\\nu$!"
        )
    elif T_probe < Tc:
        phase_desc = (
            f"In the ferromagnetic phase ($T = {T_probe:.3f} < T_c$), $P_L(M)$ is bimodal at $\\pm M_0$, driving "
            f"$U_4(T, L=80) = {u4_probe_vals[-1]:.3f} \\to 2/3 \\approx 0.667$ as $L \\to \\infty$."
        )
    else:
        phase_desc = (
            f"In the paramagnetic phase ($T = {T_probe:.3f} > T_c$), central-limit Gaussian fluctuations obey "
            f"$\\langle M^4 \\rangle \\to 3\\langle M^2 \\rangle^2$, driving $U_4(T, L=80) = {u4_probe_vals[-1]:.3f} \\to 0$."
        )

    return f"**Probe at $T = {T_probe:.3f}\\,J/k_B$:** {phase_desc}"
