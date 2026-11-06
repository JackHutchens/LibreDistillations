"""Interactive concept-cell renderers for Lesson 20: Free Energy Calculations, Phase Equilibria & Flat-Histogram Methods."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


def _val(ctrl: Any, default: Any = 0.0) -> Any:
    if ctrl is None:
        return default
    return getattr(ctrl, "value", ctrl)


def render_kirkwood_thermodynamic_integration(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize Kirkwood coupling-constant thermodynamic integration and phase-crossing hysteresis."""
    n_quad = int(_val(controls.get("quadrature_points"), 6))
    anharmonicity = float(_val(controls.get("anharmonic_strength"), 1.2))
    path_regime = str(_val(controls.get("path_regime"), "reversible"))

    lam_fine = np.linspace(0.0, 1.0, 400)
    # Smooth reversible path vs. first-order phase transition crossing path
    # <Delta V>_lambda = dF/dlambda; by convexity d^2F/dlambda^2 = -beta Var(Delta V)_lambda <= 0,
    # so <Delta V>_lambda is monotonically decreasing with lambda!
    dv_base = 8.0 - 5.2 * lam_fine - anharmonicity * (lam_fine ** 2) * (3.0 - 2.0 * lam_fine)
    exact_df = 8.0 - 2.6 - 0.5 * anharmonicity

    if path_regime == "reversible":
        # Gauss-Legendre or trapezoidal quadrature nodes on [0, 1]
        lam_nodes = np.linspace(0.0, 1.0, n_quad)
        dv_nodes = 8.0 - 5.2 * lam_nodes - anharmonicity * (lam_nodes ** 2) * (3.0 - 2.0 * lam_nodes)
        trap_df = float(_trapz(dv_nodes, lam_nodes))
        quad_err = abs(trap_df - exact_df)

        ax.plot(lam_fine, dv_base, color="#1f77b4", lw=2.5, label=r"Exact integrand $\langle \Delta V \rangle_\lambda = \partial F_\lambda / \partial \lambda$")
        ax.fill_between(lam_fine, 0.0, dv_base, color="#1f77b4", alpha=0.16, label=rf"Exact $\Delta F = \int_0^1 \langle \Delta V \rangle_\lambda d\lambda = {exact_df:.4f}\,\mathrm{{eV}}$")
        ax.plot(lam_nodes, dv_nodes, "o--", color="#d62728", lw=1.8, ms=7, label=f"Trapezoidal quadrature ({n_quad} points): $\\Delta F_{{\\mathrm{{num}}}} = {trap_df:.4f}\\,\\mathrm{{eV}}$")
        ax.axhline(0.0, color="gray", lw=1.0, ls=":")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, max(dv_base) * 1.15)
        ax.set_xlabel(r"Kirkwood Coupling Parameter $\lambda \in [0, 1]$")
        ax.set_ylabel(r"Ensemble Average $\langle V_1 - V_0 \rangle_\lambda$ ($\mathrm{eV}$)")
        ax.set_title("Kirkwood Alchemical Thermodynamic Integration Along a Single-Phase Path")
        ax.legend(loc="upper right", fontsize=8.0)
        ax.grid(True, alpha=0.25)

        return (
            f"**Reversible single-phase path ({n_quad} quadrature nodes):** "
            f"Exact free energy difference $\\Delta F = F_1 - F_0 = {exact_df:.4f}\\,\\mathrm{{eV}}$, "
            f"numerical integral $\\Delta F_{{\\mathrm{{num}}}} = {trap_df:.4f}\\,\\mathrm{{eV}}$ (quadrature error $|\\delta\\Delta F| = {quad_err:.4f}\\,\\mathrm{{eV}}$). "
            f"Note that $\\partial\\langle\\Delta V\\rangle_\\lambda / \\partial\\lambda = -\\beta\\,\\mathrm{{Var}}_\\lambda(\\Delta V) \\le 0$, so the curve slopes downward."
        )
    else:
        # Crossing a first-order phase transition near lambda_c = 0.55 creates a metastability hysteresis loop!
        jump = 2.8
        fwd_branch = dv_base + 0.5 * jump * (1.0 - np.tanh((lam_fine - 0.68) / 0.025)) - 0.5 * jump
        rev_branch = dv_base + 0.5 * jump * (1.0 - np.tanh((lam_fine - 0.36) / 0.025)) - 0.5 * jump
        eq_branch = dv_base + 0.5 * jump * (1.0 - np.tanh((lam_fine - 0.52) / 0.005)) - 0.5 * jump

        df_fwd = float(_trapz(fwd_branch, lam_fine))
        df_rev = float(_trapz(rev_branch, lam_fine))
        hyst_err = abs(df_fwd - df_rev)

        ax.plot(lam_fine, fwd_branch, color="#d62728", lw=2.4, label=rf"Forward sweep ($\lambda: 0 \to 1$, superheated): $\Delta F_{{\mathrm{{fwd}}}} = {df_fwd:.3f}\,\mathrm{{eV}}$")
        ax.plot(lam_fine, rev_branch, color="#1f77b4", lw=2.4, ls="--", label=rf"Reverse sweep ($\lambda: 1 \to 0$, supercooled): $\Delta F_{{\mathrm{{rev}}}} = {df_rev:.3f}\,\mathrm{{eV}}$")
        ax.plot(lam_fine, eq_branch, color="#2ca02c", lw=1.8, ls=":", label=r"True equilibrium coexistence jump ($\lambda_c = 0.52$)")
        ax.fill_between(lam_fine, rev_branch, fwd_branch, color="#ff7f0e", alpha=0.22, label=rf"Hysteresis dissipation loop ($\Delta W_{{\mathrm{{irr}}}} = {hyst_err:.3f}\,\mathrm{{eV}}$)")
        ax.axvline(0.52, color="#2ca02c", ls=":", lw=1.4)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel(r"Kirkwood Coupling Parameter $\lambda \in [0, 1]$")
        ax.set_ylabel(r"Ensemble Average $\langle V_1 - V_0 \rangle_\lambda$ ($\mathrm{eV}$)")
        ax.set_title("Catastrophic Hysteresis When TI Crosses a First-Order Phase Boundary")
        ax.legend(loc="upper right", fontsize=7.8)
        ax.grid(True, alpha=0.25)

        return (
            f"**First-order phase boundary crossed at $\\lambda_c = 0.52$:** "
            f"Metastable nucleation barriers cause the forward integral ($\\Delta F_{{\\mathrm{{fwd}}}} = {df_fwd:.3f}\\,\\mathrm{{eV}}$) "
            f"and reverse integral ($\\Delta F_{{\\mathrm{{rev}}}} = {df_rev:.3f}\\,\\mathrm{{eV}}$) to disagree by "
            f"$\\Delta W_{{\\mathrm{{irr}}}} = {hyst_err:.3f}\\,\\mathrm{{eV}}$. Thermodynamic integration paths must NEVER cross first-order phase lines!"
        )


def render_widom_and_gibbs_coexistence(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare Widom ghost-particle insertion breakdown at high density and Gibbs Ensemble coexistence."""
    reduced_density = float(_val(controls.get("reduced_density"), 0.55))
    reduced_temp = float(_val(controls.get("reduced_temp"), 1.05))
    mode = str(_val(controls.get("method_view"), "widom"))

    if mode == "widom":
        # Plot distribution of Boltzmann factors exp(-beta Delta V) and Widom sampling efficiency vs density rho*
        rho_grid = np.linspace(0.05, 0.95, 300)
        # Approximate hard-sphere / LJ insertion acceptance probability P_non_overlap ~ exp(-4*eta*(1+eta+eta^2)/(1-eta)^3) (Carnahan-Starling)
        eta_grid = (np.pi / 6.0) * rho_grid
        mu_ex_over_kT = eta_grid * (8.0 - 9.0 * eta_grid + 3.0 * (eta_grid ** 2)) / ((1.0 - eta_grid) ** 3) - (1.8 / reduced_temp) * rho_grid
        boltz_avg = np.exp(-mu_ex_over_kT)
        overlap_free_prob = np.exp(-eta_grid * (8.0 - 9.0 * eta_grid + 3.0 * (eta_grid ** 2)) / ((1.0 - eta_grid) ** 3))

        eta_pt = (np.pi / 6.0) * reduced_density
        mu_pt = eta_pt * (8.0 - 9.0 * eta_pt + 3.0 * (eta_pt ** 2)) / ((1.0 - eta_pt) ** 3) - (1.8 / reduced_temp) * reduced_density
        prob_pt = float(np.exp(-eta_pt * (8.0 - 9.0 * eta_pt + 3.0 * (eta_pt ** 2)) / ((1.0 - eta_pt) ** 3)))

        ax.semilogy(rho_grid, np.maximum(overlap_free_prob, 1e-14), color="#d62728", lw=2.5, label=r"Cavity probability $P(\text{no core overlap}) \sim \langle e^{-\beta\Delta V_{\mathrm{rep}}}\rangle$")
        ax.semilogy(rho_grid, np.maximum(boltz_avg, 1e-14), color="#1f77b4", lw=2.2, ls="--", label=r"Widom average $\langle e^{-\beta\Delta v}\rangle = e^{-\beta\mu_{\mathrm{ex}}}$")
        ax.axvline(reduced_density, color="#9467bd", ls=":", lw=2.0, label=f"Selected density $\\rho\\sigma^3 = {reduced_density:.2f}$")
        ax.scatter([reduced_density], [max(prob_pt, 1e-14)], color="#d62728", s=70, zorder=5)
        ax.axvspan(0.75, 0.95, color="#ff7f0e", alpha=0.15, label="Dense liquid / solid regime (Widom fails)")

        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(1e-12, 10.0)
        ax.set_xlabel(r"Reduced Number Density $\rho^* = N\sigma^3 / V$")
        ax.set_ylabel(r"Widom Insertion Factor / Cavity Probability (log scale)")
        ax.set_title(rf"Widom Ghost-Particle Insertion Efficiency vs. Density ($T^* = {reduced_temp:.2f}$)")
        ax.legend(loc="lower left", fontsize=7.8)
        ax.grid(True, which="both", alpha=0.25)

        return (
            f"**At $\\rho^* = {reduced_density:.2f}$, $T^* = {reduced_temp:.2f}$:** "
            f"Probability of finding a non-overlapping cavity for a random ghost particle is $P_{{\\mathrm{{cavity}}}} \\approx {prob_pt:.2e}$, "
            f"giving $\\beta\\mu_{{\\mathrm{{ex}}}} = -\\ln\\langle e^{{-\\beta\\Delta v}}\\rangle = {mu_pt:.2f}$. "
            f"For $\\rho^* \\gtrsim 0.75$ (dense liquids/solids), almost every random insertion hits a repulsive core ($\\Delta v \\gg k_B T \\implies e^{{-\\beta\\Delta v}} \\approx 0$)."
        )
    else:
        # Gibbs ensemble MC two-box coexistence curve (liquid-vapor binodal for Lennard-Jones fluid, T_c* ~ 1.31, rho_c* ~ 0.316)
        tc = 1.31
        rhoc = 0.316
        t_grid = np.linspace(0.70, tc, 300)
        tau = np.maximum(tc - t_grid, 0.0)
        # Rectilinear diameters + Ising 3D beta = 0.326
        rho_liq = rhoc + 0.22 * tau + 0.56 * (tau ** 0.326)
        rho_vap = np.maximum(rhoc + 0.22 * tau - 0.56 * (tau ** 0.326), 0.002)

        ax.plot(rho_vap, t_grid, color="#1f77b4", lw=2.5, label="Box I: Coexisting Vapor Branch $\\rho_{\\mathrm{vap}}(T^*)$")
        ax.plot(rho_liq, t_grid, color="#d62728", lw=2.5, label="Box II: Coexisting Liquid Branch $\\rho_{\\mathrm{liq}}(T^*)$")
        ax.plot(0.5 * (rho_vap + rho_liq), t_grid, color="gray", ls="--", lw=1.4, label="Law of rectilinear diameters")
        ax.scatter([rhoc], [tc], color="#2ca02c", s=80, zorder=6, label=f"Critical point ($T_c^*={tc:.2f}, \\rho_c^*={rhoc:.2f}$)")

        t_sel = min(reduced_temp, tc - 0.01)
        tau_s = max(tc - t_sel, 0.0)
        rv_s = max(rhoc + 0.22 * tau_s - 0.56 * (tau_s ** 0.326), 0.002)
        rl_s = rhoc + 0.22 * tau_s + 0.56 * (tau_s ** 0.326)

        ax.plot([rv_s, rl_s], [t_sel, t_sel], "o-", color="#9467bd", lw=2.2, ms=8, label=f"Gibbs tie-line at $T^*={t_sel:.2f}$")
        ax.scatter([reduced_density], [t_sel], marker="X", color="#ff7f0e", s=90, zorder=7, label=f"Initial total state ($\\rho_{{\\mathrm{{tot}}}}^*={reduced_density:.2f}$)")

        ax.set_xlim(0.0, 0.90)
        ax.set_ylim(0.68, 1.40)
        ax.set_xlabel(r"Reduced Density $\rho^* = N\sigma^3 / V$")
        ax.set_ylabel(r"Reduced Temperature $T^* = k_B T / \epsilon$")
        ax.set_title("Panagiotopoulos Gibbs Ensemble MC: Interface-Free Phase Coexistence")
        ax.legend(loc="upper right", fontsize=7.6)
        ax.grid(True, alpha=0.25)

        return (
            f"**Gibbs Ensemble at $T^* = {t_sel:.2f}$:** "
            f"Starting at overall density $\\rho_{{\\mathrm{{tot}}}}^* = {reduced_density:.2f}$ inside the two-phase dome, "
            f"simultaneous particle displacement ($\\Delta T=0$), volume exchange ($\\Delta P=0$), and particle transfer ($\\Delta\\mu=0$) "
            f"spontaneously separate Box I into vapor ($\\rho_{{\\mathrm{{vap}}}}^* = {rv_s:.3f}$) and Box II into liquid ($\\rho_{{\\mathrm{{liq}}}}^* = {rl_s:.3f}$) without forming an interface!"
        )


def render_bennett_acceptance_overlap(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare forward/reverse energy-difference distributions rho_0(Delta V) and rho_1(Delta V) in BAR vs. FEP."""
    separation = float(_val(controls.get("perturbation_shift"), 3.0))
    log_sample_ratio = float(_val(controls.get("log_t1_over_t0"), 0.0))
    view = str(_val(controls.get("bar_view"), "histograms"))

    # Exact fluctuation theorem / Bennett relation for Gaussian energy-difference model (beta = 1):
    # Suppose sigma^2 = 2 * lambda_reorg, where mean in ensemble 0 is <Delta V>_0 = delta_F + sigma^2 / 2
    # and mean in ensemble 1 is <Delta V>_1 = delta_F - sigma^2 / 2.
    # Then ln[rho_1(x) / rho_0(x)] = (x - mu_0)^2/(2*sigma^2) - (x - mu_1)^2/(2*sigma^2) = (delta_F - x) exactly!
    delta_f_true = 2.0  # in k_B T units
    sigma = np.sqrt(max(separation, 0.4))
    mu_0 = delta_f_true + 0.5 * (sigma ** 2)
    mu_1 = delta_f_true - 0.5 * (sigma ** 2)

    dv = np.linspace(mu_1 - 4.0 * sigma, mu_0 + 4.0 * sigma, 500)
    rho_0 = (1.0 / ( np.sqrt(2.0 * np.pi) * sigma )) * np.exp(-0.5 * ((dv - mu_0) / sigma) ** 2)
    rho_1 = (1.0 / ( np.sqrt(2.0 * np.pi) * sigma )) * np.exp(-0.5 * ((dv - mu_1) / sigma) ** 2)

    # Bennett optimal shift C = beta * Delta F + ln(t0 / t1) = delta_f_true - log_sample_ratio (C = ln(Z1 t0 / Z0 t1) = -beta*DeltaF + ln(t0/t1))
    # Matching Bennett Acceptance Ratio:
    # Z1 / Z0 = exp(-beta Delta F), C = ln(Z1 t0 / Z0 t1) = -beta Delta F + ln(t0 / t1)
    # And f(x) = 1 / (1 + exp(x)) is the Fermi function.
    c_opt = -delta_f_true - log_sample_ratio
    fermi_fwd = 1.0 / (1.0 + np.exp(dv + c_opt))
    fermi_rev = 1.0 / (1.0 + np.exp(-dv + c_opt))

    # Overlap integral 2 * int (rho_0 * rho_1) / (rho_0 + rho_1) d(dv)
    overlap = float(_trapz(2.0 * (rho_0 * rho_1) / np.maximum(rho_0 + rho_1, 1e-15), dv))
    # Naive finite-sample FEP bias when sampling only near mu_0 (rare tail at dv = mu_1)
    fep_tail_bias = 0.08 * (separation ** 2)

    if view == "histograms":
        ax.plot(dv, rho_0, color="#1f77b4", lw=2.5, label=rf"Forward distribution $\rho_0(\Delta V)$ ($\langle\Delta V\rangle_0 = {mu_0:.2f}\,k_B T$)")
        ax.plot(dv, rho_1, color="#d62728", lw=2.5, label=rf"Reverse distribution $\rho_1(\Delta V)$ ($\langle\Delta V\rangle_1 = {mu_1:.2f}\,k_B T$)")
        ax.fill_between(dv, 0.0, np.minimum(rho_0, rho_1), color="#9467bd", alpha=0.28, label=f"Distribution overlap = {100.0*overlap:.1f}%")
        ax.axvline(delta_f_true, color="#2ca02c", ls="--", lw=2.0, label=rf"Crossing point $\rho_0 = \rho_1$ at $\Delta V = \Delta F = {delta_f_true:.2f}\,k_B T$")

        ax2 = ax.twinx()
        ax2.plot(dv, fermi_fwd, color="#ff7f0e", ls=":", lw=1.8, label=r"Fermi weight $f(+\beta\Delta V + C)$")
        ax2.set_ylabel(r"Bennett Fermi Function $f(x) = (1+e^x)^{-1}$", color="#ff7f0e")
        ax2.set_ylim(-0.05, 1.08)

        ax.set_xlabel(r"Energy Difference $\beta\Delta V = \beta(V_1 - V_0)$")
        ax.set_ylabel(r"Probability Density $\rho_0(\Delta V), \rho_1(\Delta V)$")
        ax.set_title("Bennett Acceptance Ratio (BAR): Forward & Reverse Histogram Overlap")
        ax.legend(loc="upper left", fontsize=7.6)
        ax.grid(True, alpha=0.25)
    else:
        # Linear offset plot: f_0(Delta V) = ln rho_0(Delta V) - beta*Delta V / 2
        # and f_1(Delta V) = ln rho_1(Delta V) + beta*Delta V / 2, so f_1(Delta V) - f_0(Delta V) = beta * Delta F!
        f0 = np.log(np.maximum(rho_0, 1e-30)) - 0.5 * dv
        f1 = np.log(np.maximum(rho_1, 1e-30)) + 0.5 * dv
        diff_f1_f0 = f1 - f0

        ax.plot(dv, f0, color="#1f77b4", lw=2.3, label=r"$f_0(\Delta V) = \ln\rho_0(\Delta V) - \frac{1}{2}\beta\Delta V$")
        ax.plot(dv, f1, color="#d62728", lw=2.3, ls="--", label=r"$f_1(\Delta V) = \ln\rho_1(\Delta V) + \frac{1}{2}\beta\Delta V$")
        ax.plot(dv, diff_f1_f0, color="#2ca02c", lw=2.6, label=rf"$f_1(\Delta V) - f_0(\Delta V) = \beta\Delta F = {delta_f_true:.2f}$ (exact constant offset!)")
        ax.axvline(delta_f_true, color="gray", ls=":", lw=1.4)
        ax.set_xlabel(r"Energy Difference $\beta\Delta V = \beta(V_1 - V_0)$")
        ax.set_ylabel(r"Shifted Log-Densities $f_0(\Delta V), f_1(\Delta V)$")
        ax.set_title(r"Bennett Shifted Log-Histogram Identity: $f_1(\Delta V) = f_0(\Delta V) + \beta\Delta F$")
        ax.legend(loc="lower left", fontsize=7.8)
        ax.grid(True, alpha=0.25)

    return (
        f"**Perturbation separation $\\beta(\\langle\\Delta V\\rangle_0 - \\langle\\Delta V\\rangle_1) = {separation:.2f}$:** "
        f"Exact $\\beta\\Delta F = {delta_f_true:.2f}$ occurs precisely where $\\rho_0(\\Delta V) = \\rho_1(\\Delta V)$. "
        f"Histogram overlap is ${100.0*overlap:.1f}\\%$; one-sided FEP suffers a finite-sample tail bias of $\\sim {fep_tail_bias:.2f}\\,k_B T$, "
        f"whereas two-sided BAR (with optimal shift $C = \\ln(Z_1 t_0 / Z_0 t_1) = {c_opt:.2f}$) minimizes variance."
    )


def render_wang_landau_flat_histogram(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Simulate Wang-Landau flat-histogram refinement of the microcanonical entropy S(E)/k_B."""
    stage = int(_val(controls.get("wl_stage"), 4))
    flatness_tol = float(_val(controls.get("flatness_criterion"), 0.85))
    panel = str(_val(controls.get("wl_panel"), "entropy"))

    # Energy grid E in [-2.0, 0.0] (e.g., 2D Ising or Lennard-Jones cluster energy per particle)
    e_grid = np.linspace(-2.0, -0.1, 60)
    # True microcanonical entropy S_exact(E)/k_B (concave parabolic-like curve with a phase transition inflection)
    s_exact = 3.2 * (1.0 - ((e_grid + 0.1) / 1.9) ** 2) + 0.25 * np.sin(3.5 * (e_grid + 2.0))
    s_exact = s_exact - s_exact[0]

    # Modification factor at stage m: delta_S / k_B = 2^(-stage)
    delta_s = 2.0 ** (-stage)
    rng = np.random.default_rng(10485 + stage * 37)
    # Error in S(E) scales as sqrt(delta_S)
    err_scale = 0.65 * np.sqrt(delta_s) * ((1.0 - flatness_tol) / 0.15 + 0.5)
    s_wl = s_exact + rng.normal(0.0, err_scale, size=len(e_grid))
    s_wl = s_wl - s_wl[0]
    rms_err = float(np.sqrt(np.mean((s_wl - s_exact) ** 2)))

    # Visit histogram Omega_hist(E) at end of stage (fluctuating above flatness_tol * mean)
    mean_visits = 200.0 * (stage + 1)
    spread = (1.0 - flatness_tol) * 0.85
    visits = mean_visits * (1.0 + rng.uniform(-spread, spread, size=len(e_grid)))
    min_ratio = float(np.min(visits) / np.mean(visits))

    if panel == "entropy":
        ax.plot(e_grid, s_exact, color="#1f77b4", lw=2.5, label=r"Exact microcanonical entropy $S_{\mathrm{exact}}(E)/k_B = \ln\Omega(E)$")
        ax.plot(e_grid, s_wl, "o--", color="#d62728", lw=1.8, ms=4.5, label=rf"Wang–Landau estimate $S_{{\mathrm{{WL}}}}(E)/k_B$ ($\delta S/k_B = 2^{{-{stage}}} = {delta_s:.4f}$)")
        ax.set_xlabel(r"Energy per Particle $E / \epsilon$")
        ax.set_ylabel(r"Configurational Entropy $S(E) / k_B = \ln\Omega(E)$")
        ax.set_title(f"Wang–Landau Density-of-States Convergence (Stage $m = {stage}$, RMS error $= {rms_err:.4f}$)")
        ax.legend(loc="lower right", fontsize=8.0)
        ax.grid(True, alpha=0.25)
    else:
        ax.bar(e_grid, visits, width=0.026, color="#2ca02c", alpha=0.75, edgecolor="black", lw=0.4, label=r"Visit histogram $\Omega_{\mathrm{hist}}(E)$ in current stage")
        ax.axhline(mean_visits, color="#1f77b4", lw=2.0, label=rf"Mean visits $\langle \Omega_{{\mathrm{{hist}}}}(E)\rangle = {mean_visits:.0f}$")
        ax.axhline(flatness_tol * mean_visits, color="#d62728", ls="--", lw=2.0, label=rf"Flatness threshold $f\langle\Omega_{{\mathrm{{hist}}}}\rangle$ ($f = {flatness_tol:.2f}$)")
        ax.set_ylim(0.0, mean_visits * 1.35)
        ax.set_xlabel(r"Energy per Particle $E / \epsilon$")
        ax.set_ylabel(r"Histogram Counts $\Omega_{\mathrm{hist}}(E)$")
        ax.set_title(f"Wang–Landau Flat Histogram Check ($\\min / \\mathrm{{avg}} = {min_ratio:.3f} > f = {flatness_tol:.2f}$)")
        ax.legend(loc="upper right", fontsize=8.0)
        ax.grid(True, axis="y", alpha=0.25)

    return (
        f"**Wang–Landau Stage $m = {stage}$ ($\\delta S / k_B = 2^{{-{stage}}} = {delta_s:.5f}$, $e^{{-\\delta S/k_B}} = {np.exp(-delta_s):.5f}$):** "
        f"Visit histogram satisfies $\\min(\\Omega_{{\\mathrm{{hist}}}}) / \\langle\\Omega_{{\\mathrm{{hist}}}}\\rangle = {min_ratio:.3f} \\ge f = {flatness_tol:.2f}$, "
        f"and the entropy RMS error across the energy spectrum is ${rms_err:.4f}\\,k_B$. Because $S(E)$ is independent of $T$, "
        f"a single converged run yields $F(T) = -k_B T\\ln\\sum_E e^{{S(E)/k_B - \\beta E}}$ at ALL temperatures!"
    )
