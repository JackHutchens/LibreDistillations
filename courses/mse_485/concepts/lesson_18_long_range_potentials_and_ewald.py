"""Interactive concept visualizations for Lesson 18: Neighbor Lists, Long-Range Potentials, and Ewald Summation."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
try:
    from scipy.special import erfc, erf
except ImportError:
    import math

    def _v_erf(x: Any) -> Any:
        arr = np.asarray(x, dtype=float)
        return np.vectorize(math.erf)(arr)

    def _v_erfc(x: Any) -> Any:
        arr = np.asarray(x, dtype=float)
        return np.vectorize(math.erfc)(arr)

    erf = _v_erf
    erfc = _v_erfc
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def render_neighbor_skin_optimization(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot total CPU time per MD step vs. Verlet neighbor-list skin depth Delta."""
    n_particles = int(controls.get("n_particles", 4000))
    rc = float(controls.get("rc", 2.5))
    rebuild_mode = str(controls.get("rebuild_mode", "verlet_O_N2"))

    delta = np.linspace(0.08, 1.50, 250)
    rho = 0.80  # liquid number density

    # Average number of neighbors inside (rc + delta) sphere: (4/3)*pi*rho*(rc + delta)^3
    m_neighbors = (4.0 / 3.0) * np.pi * rho * (rc + delta) ** 3
    # Per-step force evaluation cost (only pairs inside list): proportional to 0.5 * N * m_neighbors
    cost_force_step = 1e-4 * 0.5 * n_particles * m_neighbors

    # Number of steps between rebuilds: proportional to delta / v_thermal (plus diffusive bend)
    steps_between_rebuilds = 18.0 * delta
    if rebuild_mode == "cell_O_N":
        # Linked-cell assisted Verlet list rebuild: O(27 * N * m_cell) ~ O(N)
        cost_rebuild_single = 1e-4 * 4.5 * n_particles * m_neighbors
        mode_label = "Cell-Assisted Rebuild O(N)"
    else:
        # Naive all-pairs Verlet list rebuild: O(N^2 / 2)
        cost_rebuild_single = 1e-4 * 0.18 * (n_particles**2)
        mode_label = "All-Pairs Rebuild O(N²)"

    cost_rebuild_amortized = cost_rebuild_single / steps_between_rebuilds
    cost_total = cost_force_step + cost_rebuild_amortized

    idx_opt = int(np.argmin(cost_total))
    delta_opt = float(delta[idx_opt])
    min_cost = float(cost_total[idx_opt])

    ax.plot(delta, cost_force_step, color=COLORS["primary"], linestyle="--", linewidth=1.9, label=r"Pair Evaluation $\propto (r_c + \Delta)^3$")
    ax.plot(delta, cost_rebuild_amortized, color=COLORS["secondary"], linestyle=":", linewidth=2.0, label=f"Amortized Rebuild ({mode_label})")
    ax.plot(delta, cost_total, color=COLORS["total"], linewidth=2.6, label=r"Total CPU Time per Step $t(\Delta)$")
    ax.scatter([delta_opt], [min_cost], color=COLORS["accent"], s=65, zorder=5, label=f"Optimal Skin $\\Delta^* = {delta_opt:.2f}\\sigma$")
    ax.axvline(delta_opt, color=COLORS["accent"], linestyle="-.", linewidth=1.3, alpha=0.8)

    ax.set_xlabel(r"Verlet Skin Depth $\Delta / \sigma$")
    ax.set_ylabel("Relative CPU Time per Timestep")
    ax.set_title(f"Verlet Neighbor List Skin Optimization ($N = {n_particles},\\; r_c = {rc:.1f}\\sigma$)")
    ax.set_xlim(0.08, 1.50)
    ax.set_ylim(0.0, float(np.percentile(cost_total, 92)) * 1.15)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    m_opt = float(m_neighbors[idx_opt])
    n_reb = float(steps_between_rebuilds[idx_opt])
    return (
        f"**Optimal Skin $\\Delta^* = {delta_opt:.2f}\\sigma$ ({mode_label}):** "
        f"Each particle tracks $m \\approx$ **{m_opt:.0f}** neighbors inside $r_c + \\Delta^*$ and rebuilds the table "
        f"every **{n_reb:.0f}** steps (triggered as soon as the two largest displacements satisfy $d_1 + d_2 \\ge \\Delta$)."
    )


def render_madelung_convergence(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare 1D alternating ionic chain partial sums against Ewald real+reciprocal+self convergence to 2 ln 2."""
    max_shells = int(controls.get("max_shells", 25))
    kappa_1d = float(controls.get("kappa_1d", 1.2))

    m_vals = np.arange(1, max_shells + 1)
    # Direct alternating sum: alpha_M = 2 * sum_{n=1}^M (-1)^{n-1} / n
    terms_direct = 2.0 * ((-1.0) ** (m_vals - 1)) / m_vals
    partial_direct = np.cumsum(terms_direct)

    # Ewald-screened 1D convergence: erfc(kappa * n) real-space + Gaussian k-space shells
    exact_madelung = 2.0 * np.log(2.0)
    # Accurate asymptotic envelope of Ewald truncation error after M shells:
    ewald_err = (2.0 * erfc(kappa_1d * m_vals) / m_vals - 2.0 * np.exp(-((np.pi * m_vals / (2.0 * kappa_1d)) ** 2)) / m_vals) * (
        (-1.0) ** (m_vals - 1)
    )
    partial_ewald = exact_madelung + ewald_err

    ax.axhline(exact_madelung, color=COLORS["good"], linestyle="-", linewidth=2.2, label=f"Exact $2\\ln 2 = {exact_madelung:.6f}$")
    ax.plot(
        m_vals,
        partial_direct,
        color=COLORS["danger"],
        marker="o",
        markersize=4.5,
        linewidth=1.8,
        linestyle="--",
        label=r"Direct Sum $2\sum_{n=1}^M \frac{(-1)^{n-1}}{n}$",
    )
    ax.plot(
        m_vals,
        partial_ewald,
        color=COLORS["primary"],
        marker="s",
        markersize=4.5,
        linewidth=2.3,
        label=f"Ewald Sum ($\\kappa R = {kappa_1d:.1f}$)",
    )

    ax.set_xlabel(r"Number of Neighbor Shells $M$")
    ax.set_ylabel(r"1D Madelung Constant Estimate $\alpha_M$")
    ax.set_title("1D Ionic Chain Madelung Convergence: Direct vs. Ewald")
    ax.set_xlim(1, max_shells)
    ax.set_ylim(1.15, 2.05)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=8.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    err_dir_m = abs(float(partial_direct[-1]) - exact_madelung)
    err_ew_m = abs(float(partial_ewald[-1]) - exact_madelung)
    return (
        f"**At $M = {max_shells}$ shells:** Direct conditionally convergent sum gives $\\alpha_{{{max_shells}}} = $ "
        f"**{float(partial_direct[-1]):.5f}** (error **{err_dir_m:.2e}**), whereas Ewald summation with $\\kappa R = {kappa_1d:.1f}$ "
        f"reaches **{float(partial_ewald[-1]):.6f}** (error **{err_ew_m:.2e}** vs. $2\\ln 2 = 1.386294$)."
    )


def render_ewald_splitting_tradeoff(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize Ewald parameter kappa splitting in real space, k-space, or CPU cost balance."""
    kappa = float(controls.get("kappa", 1.2))
    view_mode = str(controls.get("view_mode", "real_potentials"))

    if view_mode == "real_potentials":
        r = np.linspace(0.15, 4.0, 350)
        v_coulomb = 1.0 / r
        v_short = erfc(kappa * r) / r
        v_long = erf(kappa * r) / r

        ax.plot(r, v_coulomb, color=COLORS["total"], linestyle=":", linewidth=2.0, label=r"Full Coulomb $1/r$")
        ax.plot(r, v_short, color=COLORS["primary"], linewidth=2.5, label=r"$v_{\mathrm{short}}(r) = \frac{\mathrm{erfc}(\kappa r)}{r}$ ($r$-space)")
        ax.plot(r, v_long, color=COLORS["accent"], linestyle="--", linewidth=2.3, label=r"$v_{\mathrm{long}}(r) = \frac{\mathrm{erf}(\kappa r)}{r}$ ($k$-space)")
        ax.axvline(2.5, color=COLORS["muted"], linestyle="-.", linewidth=1.4, label=r"Half-Box $L/2 = 2.5$")
        ax.set_xlabel(r"Distance $r$")
        ax.set_ylabel(r"Pair Potential $v(r) / q_1 q_2$")
        ax.set_title(f"Ewald Real-Space Decomposition ($\\kappa = {kappa:.2f}$)")
        ax.set_xlim(0.15, 4.0)
        ax.set_ylim(0.0, 4.5)
        ax.legend(loc="upper right", fontsize=8.3)
    elif view_mode == "k_space":
        k = np.linspace(0.4, 10.0, 350)
        vk_bare = 4.0 * np.pi / (k**2)
        vk_ewald = vk_bare * np.exp(-(k**2) / (4.0 * kappa**2))

        ax.plot(k, vk_bare, color=COLORS["danger"], linestyle="--", linewidth=1.9, label=r"Bare Coulomb $\frac{4\pi}{k^2}$ (Slow $k\to\infty$ tail)")
        ax.plot(k, vk_ewald, color=COLORS["primary"], linewidth=2.6, label=r"Ewald $v_{\mathrm{long},k} = \frac{4\pi}{k^2}e^{-k^2/(4\kappa^2)}$")
        ax.fill_between(k, 0, vk_ewald, color=COLORS["primary"], alpha=0.14)
        ax.set_xlabel(r"Reciprocal Wavevector Magnitude $k$")
        ax.set_ylabel(r"Fourier Potential $\Omega \, v_k / q_1 q_2$")
        ax.set_title(f"Reciprocal-Space Gaussian Decay ($\\kappa = {kappa:.2f}$)")
        ax.set_xlim(0.4, 10.0)
        ax.set_ylim(0.0, 15.0)
        ax.legend(loc="upper right", fontsize=8.3)
    else:
        # CPU complexity tradeoff vs. kappa for fixed accuracy p:
        # Real-space cutoff rc ~ sqrt(p)/kappa => C_real ~ N * rho * rc^3 ~ N^2 / (L*kappa)^3
        # k-space cutoff kc ~ 2*kappa*sqrt(p) => C_recip ~ N * (L*kc)^3 ~ N * (L*kappa)^3
        kap_grid = np.linspace(0.4, 2.8, 300)
        c_real = 1.8 / (kap_grid**3)
        c_recip = 0.8 * (kap_grid**3)
        c_tot = c_real + c_recip

        kap_opt = float((1.8 / 0.8) ** (1.0 / 6.0))
        c_cur = 1.8 / (kappa**3) + 0.8 * (kappa**3)

        ax.plot(kap_grid, c_real, color=COLORS["primary"], linestyle="--", linewidth=2.0, label=r"Real-Space Work $\propto \kappa^{-3}$")
        ax.plot(kap_grid, c_recip, color=COLORS["accent"], linestyle=":", linewidth=2.1, label=r"Reciprocal-Space Work $\propto \kappa^{3}$")
        ax.plot(kap_grid, c_tot, color=COLORS["total"], linewidth=2.6, label=r"Total Ewald CPU Cost ($\mathcal{O}(N^{3/2})$ at opt)")
        ax.scatter([kappa], [c_cur], color=COLORS["danger"], s=65, zorder=5, label=f"Selected $\\kappa = {kappa:.2f}$")
        ax.axvline(kap_opt, color=COLORS["good"], linestyle="-.", linewidth=1.4, label=f"Optimal $\\kappa^* = {kap_opt:.2f}$")
        ax.set_xlabel(r"Ewald Screening Parameter $\kappa$")
        ax.set_ylabel("Relative CPU Operations")
        ax.set_title(r"Balancing Real-Space and $k$-Space Ewald Cost")
        ax.set_xlim(0.4, 2.8)
        ax.set_ylim(0.0, 12.0)
        ax.legend(loc="upper right", fontsize=8.1)

    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    erfc_halfbox = float(erfc(kappa * 2.5))
    k4_damp = float(np.exp(-16.0 / (4.0 * kappa**2)))
    return (
        f"**Ewald Screening $\\kappa = {kappa:.2f}$:** At the minimum-image boundary $r = L/2 = 2.5$, "
        f"$\\mathrm{{erfc}}(\\kappa L/2) = $ **{erfc_halfbox:.2e}**. In reciprocal space at $k = 4.0$, the Gaussian factor is "
        f"$e^{{-k^2/(4\\kappa^2)}} = $ **{k4_damp:.2e}**. Increasing $\\kappa$ shrinks the real-space cutoff $r_c$ but expands the required $k$-space sphere $k_c$."
    )


def render_electrostatic_algorithm_scaling(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot log-log CPU time vs. particle count N for Direct O(N^2), Ewald O(N^1.5), PME O(N ln N), and FMM O(N)."""
    log10_n = float(controls.get("log10_n", 4.0))
    fmm_accuracy_p = int(controls.get("fmm_accuracy_p", 6))

    n_grid = np.logspace(2, 7, 300)
    n_sel = 10.0**log10_n

    # Realistic relative timings (in milliseconds per step)
    t_direct = 1.2e-4 * (n_grid**2)
    t_ewald = 2.5e-3 * (n_grid**1.5)
    t_pme = 1.8e-2 * n_grid * np.log2(n_grid)
    # FMM prefactor scales with multipole expansion order p^2 or p^3
    fmm_pref = 0.08 * ((fmm_accuracy_p / 4.0) ** 2)
    t_fmm = fmm_pref * n_grid

    ax.loglog(n_grid, t_direct, color=COLORS["danger"], linestyle="--", linewidth=2.0, label=r"Direct / Unoptimized Ewald $\mathcal{O}(N^2)$")
    ax.loglog(n_grid, t_ewald, color=COLORS["accent"], linestyle="-.", linewidth=2.2, label=r"Optimized Ewald $\mathcal{O}(N^{3/2})$")
    ax.loglog(n_grid, t_pme, color=COLORS["primary"], linewidth=2.5, label=r"Particle-Mesh (PME / PPPM) $\mathcal{O}(N \ln N)$")
    ax.loglog(n_grid, t_fmm, color=COLORS["good"], linewidth=2.5, label=f"Fast Multipole (FMM, $p={fmm_accuracy_p}$) $\\mathcal{{O}}(N)$")
    ax.axvline(n_sel, color=COLORS["muted"], linestyle=":", linewidth=1.5, label=f"Selected $N = 10^{{{log10_n:.1f}}}$")

    ax.set_xlabel(r"Number of Charged Particles $N$")
    ax.set_ylabel("CPU Time per Timestep (ms, log scale)")
    ax.set_title("Asymptotic Complexity of Long-Range Electrostatic Solvers")
    ax.set_xlim(1e2, 1e7)
    ax.set_ylim(1e-1, 1e9)
    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    val_ew = 2.5e-3 * (n_sel**1.5)
    val_pme = 1.8e-2 * n_sel * np.log2(n_sel)
    val_fmm = fmm_pref * n_sel
    best_method = min([("Optimized Ewald", val_ew), ("PME/PPPM", val_pme), ("Fast Multipole (FMM)", val_fmm)], key=lambda x: x[1])[0]

    return (
        f"**At $N = 10^{{{log10_n:.1f}}} \\approx {int(round(n_sel)):,d}$ particles:** "
        f"Optimized Ewald = **{val_ew:.1f} ms**, PME/PPPM = **{val_pme:.1f} ms**, FMM ($p={fmm_accuracy_p}$) = **{val_fmm:.1f} ms**. "
        f"Fastest solver at this scale: **{best_method}**."
    )
