"""Interactive concept visualizations for Lesson 17: Brownian Dynamics, Langevin Equations, and DPD."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def render_smoluchowski_green_trotter(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare exact canonical Boltzmann distribution, unadjusted Brownian Dynamics, and Hybrid Smart MC."""
    tau = float(controls.get("tau", 0.15))
    anharmonicity = float(controls.get("anharmonicity", 0.5))
    view_mode = str(controls.get("view_mode", "density"))

    x = np.linspace(-2.6, 2.6, 400)
    # Asymmetric anharmonic potential: V(x) = 0.5 * x^2 + 0.25 * a * x^4 - 0.35 * x
    V = 0.5 * x**2 + 0.25 * anharmonicity * x**4 - 0.35 * x
    V = V - np.min(V)
    F = -(x + anharmonicity * x**3 - 0.35)

    # Exact canonical Boltzmann distribution (beta = 1)
    rho_eq = np.exp(-V)
    dx = x[1] - x[0]
    rho_eq /= np.sum(rho_eq) * dx

    # Unadjusted finite-tau Euler-Maruyama / Brownian stationary density
    # Effective temperature/broadening distortion from finite step tau = D * dt
    # First-order ITO/Trotter effective potential error: delta_V ~ -0.5 * tau * (V'' - 0.5 * (V')^2)
    V_pp = 1.0 + 3.0 * anharmonicity * x**2
    V_eff_bd = V / (1.0 + 0.65 * tau * V_pp) + 0.18 * tau * (F**2) / (1.0 + tau)
    rho_bd = np.exp(-V_eff_bd)
    rho_bd /= np.sum(rho_bd) * dx

    # Smart MC acceptance rate as a function of step size tau and anharmonicity
    # For a linear potential (V''=0), acceptance is 100%. Curvature V'' causes rejection scaling as exp(-c * tau * <V''>)
    mean_curv = float(np.sum(rho_eq * V_pp) * dx)
    acc_rate = float(np.exp(-0.42 * (tau * mean_curv) ** 1.35))
    acc_rate = max(0.05, min(1.0, acc_rate))

    if view_mode == "green_step":
        # Show short-time Green's function T(x0 -> x') from x0 = -1.0
        x0 = -1.0
        f0 = -(x0 + anharmonicity * x0**3 - 0.35)
        mean_shift = x0 + tau * f0
        sigma = np.sqrt(2.0 * tau)
        green_drift_diff = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mean_shift) / sigma) ** 2)
        green_pure_diff = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * ((x - x0) / sigma) ** 2)

        ax.plot(x, green_pure_diff, linestyle="--", linewidth=1.8, color=COLORS["muted"], label=r"Pure Diffusion $\exp(-(x-x_0)^2/4\tau)$")
        ax.plot(x, green_drift_diff, linestyle="-", linewidth=2.4, color=COLORS["primary"], label=r"Trotter Drift+Diff $T(x_0\to x')$")
        ax.fill_between(x, 0, green_drift_diff, color=COLORS["primary"], alpha=0.14)
        ax.axvline(x0, color=COLORS["muted"], linestyle=":", linewidth=1.3, label=f"Start $x_0 = {x0:.1f}$")
        ax.axvline(mean_shift, color=COLORS["accent"], linestyle="-.", linewidth=1.6, label=f"Mean Drift $x_0 + \\tau\\beta F(x_0) = {mean_shift:.2f}$")
        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Short-Time Propagator $G(x, x_0; \tau)$")
        ax.set_title(f"Short-Time Smoluchowski Propagator ($\\tau = {tau:.2f}$)")
        ax.set_xlim(-2.6, 2.6)
        ax.legend(loc="upper right", fontsize=8.5)
    else:
        ax.plot(x, rho_eq, color=COLORS["good"], linewidth=2.6, label=r"Exact Boltzmann $\rho_{\mathrm{eq}}(x) \propto e^{-\beta V(x)}$")
        ax.plot(
            x,
            rho_bd,
            color=COLORS["danger"],
            linestyle="--",
            linewidth=2.1,
            label=f"Unadjusted Brownian ($\\tau={tau:.2f}$)",
        )
        ax.plot(
            x[::8],
            rho_eq[::8],
            "o",
            color=COLORS["primary"],
            markersize=4.5,
            label=f"Hybrid Smart MC (Acc = {100.0 * acc_rate:.1f}%)",
        )
        ax_v = ax.twinx()
        ax_v.plot(x, V, color=COLORS["muted"], linestyle=":", linewidth=1.4, alpha=0.75)
        ax_v.set_ylabel(r"Potential $\beta V(x)$", color=COLORS["muted"])
        ax_v.tick_params(axis="y", colors=COLORS["muted"])
        ax_v.set_ylim(0, 6.0)
        ax_v.spines["top"].set_visible(False)

        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Stationary Probability Density $\rho(x)$")
        ax.set_title(f"Smoluchowski Stationary Density vs. Step Size $\\tau = {tau:.2f}$")
        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(0, max(np.max(rho_eq), np.max(rho_bd)) * 1.22)
        ax.legend(loc="upper left", fontsize=8.2)

    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    l1_err = float(np.sum(np.abs(rho_bd - rho_eq)) * dx)
    dyn_status = (
        "Faithful dynamics & exact statics (acceptance $\\approx 99\\%$)"
        if acc_rate >= 0.96
        else "Exact static equilibrium via rejections, but dynamics distorted by rejected steps"
    )
    return (
        f"**Trotter Step $\\tau = D\\Delta t = {tau:.2f}$ · Anharmonicity $k_4 = {anharmonicity:.2f}$:** "
        f"Unadjusted Brownian $L_1$ density error = **{l1_err:.3f}**. "
        f"Smart MC acceptance rate $\\langle A \\rangle \\approx$ **{100.0 * acc_rate:.1f}%** ({dyn_status})."
    )


def render_oseen_hydrodynamic_coupling(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot Oseen and Rotne-Prager hydrodynamic mobility tensor components vs. pair separation r/a."""
    angle_deg = float(controls.get("angle_deg", 0.0))
    show_rpy = bool(int(controls.get("show_rpy", 1)))

    r_over_a = np.linspace(2.0, 15.0, 300)
    theta = np.deg2rad(angle_deg)

    # Normalized by single-particle Stokes-Einstein diffusivity D0 = kB*T / (6*pi*eta*a)
    # Oseen: D_ij / D0 = (3a / 4r) * (I + r_hat r_hat)
    d_par_oseen = 1.5 / r_over_a
    d_perp_oseen = 0.75 / r_over_a
    d_theta_oseen = (0.75 / r_over_a) * (1.0 + np.cos(theta) ** 2)

    ax.plot(
        r_over_a,
        d_par_oseen,
        color=COLORS["primary"],
        linewidth=2.2,
        label=r"Oseen Longitudinal $D_{\parallel}/D_0 = \frac{3a}{2r}$",
    )
    ax.plot(
        r_over_a,
        d_perp_oseen,
        color=COLORS["secondary"],
        linewidth=2.2,
        label=r"Oseen Transverse $D_{\perp}/D_0 = \frac{3a}{4r}$",
    )
    ax.plot(
        r_over_a,
        d_theta_oseen,
        color=COLORS["accent"],
        linestyle="-.",
        linewidth=2.3,
        label=f"Directional $D(\\theta={angle_deg:.0f}^\\circ)/D_0$",
    )

    if show_rpy:
        # Rotne-Prager-Yamakawa finite-size correction:
        # D_par / D0 = 3a/(2r) - (a/r)^3, D_perp / D0 = 3a/(4r) + 0.5*(a/r)^3
        d_par_rpy = 1.5 / r_over_a - (1.0 / r_over_a) ** 3
        d_perp_rpy = 0.75 / r_over_a + 0.5 * (1.0 / r_over_a) ** 3
        ax.plot(r_over_a, d_par_rpy, color=COLORS["primary"], linestyle=":", linewidth=1.7, label="Rotne–Prager $D_{\\parallel}/D_0$")
        ax.plot(r_over_a, d_perp_rpy, color=COLORS["secondary"], linestyle=":", linewidth=1.7, label="Rotne–Prager $D_{\\perp}/D_0$")

    ax.set_xlabel(r"Normalized Interparticle Separation $r_{ij} / a$")
    ax.set_ylabel(r"Normalized Pair Diffusivity $D_{ij} / D_0$")
    ax.set_title(r"Hydrodynamic Oseen Tensor Coupling ($\propto 1/r_{ij}$)")
    ax.set_xlim(2.0, 15.0)
    ax.set_ylim(0.0, 0.82)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    val_r4 = 0.75 / 4.0 * (1.0 + np.cos(theta) ** 2)
    return (
        f"**Orientation $\\theta = {angle_deg:.0f}^\\circ$:** At $r_{{ij}} = 4a$, the solvent-mediated cross-diffusivity is "
        f"$D_{{ij}}(\\theta)/D_0 = $ **{val_r4:.3f}** (longitudinal ratio $D_\\parallel / D_\\perp = 2.0$). "
        f"Because $D_{{ij}} \\propto 1/r_{{ij}}$ is long-ranged, generating correlated Brownian displacements requires "
        f"$\\mathcal{{O}}(N^3)$ Cholesky factorization $\\mathbf{{S}}\\mathbf{{S}}^T = \\underline{{\\mathbf{{D}}}}$."
    )


def render_langevin_friction_regimes(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot Langevin MSD <Delta r^2(t)> or Harmonic Oscillator VACF across friction regimes."""
    zeta = float(controls.get("zeta", 1.0))
    observable = str(controls.get("observable", "msd"))

    # Use reduced units: m = 1, kB*T = 1, harmonic frequency omega_0 = 1.0
    omega0 = 1.0
    d_long = 1.0 / zeta  # D = kB*T / (m * zeta)

    if observable == "vacf":
        t = np.linspace(0.0, 10.0, 400)
        # Free particle Langevin VACF: exp(-zeta * t)
        vacf_free = np.exp(-zeta * t)
        # Harmonic trapped particle VACF with natural frequency omega0 = 1.0
        disc = (0.5 * zeta) ** 2 - omega0**2
        if abs(disc) < 1e-6:
            gamma_h = 0.5 * zeta
            vacf_harm = np.exp(-gamma_h * t) * (1.0 - gamma_h * t)
            regime_str = "Critically Damped ($\\zeta = 2\\omega_0$)"
        elif disc < 0:
            gamma_h = 0.5 * zeta
            omega_d = np.sqrt(-disc)
            vacf_harm = np.exp(-gamma_h * t) * (np.cos(omega_d * t) - (gamma_h / omega_d) * np.sin(omega_d * t))
            regime_str = "Underdamped Inertial/Oscillatory ($\\zeta < 2\\omega_0$)"
        else:
            gamma_h = 0.5 * zeta
            alpha = np.sqrt(disc)
            vacf_harm = np.exp(-gamma_h * t) * (np.cosh(alpha * t) - (gamma_h / alpha) * np.sinh(alpha * t))
            regime_str = "Overdamped Smoluchowski/Brownian ($\\zeta > 2\\omega_0$)"

        ax.axhline(0.0, color=COLORS["muted"], linestyle=":", linewidth=1.1)
        ax.plot(t, vacf_free, color=COLORS["secondary"], linestyle="--", linewidth=2.0, label=r"Free Particle $e^{-\zeta t}$")
        ax.plot(t, vacf_harm, color=COLORS["primary"], linewidth=2.5, label=r"Harmonic Trap $C_{vv}(t)/C_{vv}(0)$")
        ax.axvline(1.0 / zeta, color=COLORS["accent"], linestyle="-.", linewidth=1.5, label=f"Relaxation Time $\\tau_p = \\zeta^{{-1}} = {1.0/zeta:.2f}$")
        ax.set_xlabel(r"Reduced Time $t$")
        ax.set_ylabel(r"Normalized VACF $C_{vv}(t) / (k_B T / m)$")
        ax.set_title(f"Langevin Velocity Autocorrelation ($\\zeta = {zeta:.2f}$)")
        ax.set_xlim(0.0, 10.0)
        ax.set_ylim(-0.65, 1.08)
        ax.legend(loc="upper right", fontsize=8.3)
    else:
        t = np.logspace(-2, 2, 300)
        # Exact 1D Langevin MSD (per degree of freedom, multiplied by 3 for 3D):
        # <Delta r^2(t)> = (6 kB T / (m zeta^2)) * (zeta * t - 1 + exp(-zeta * t))
        msd_exact = (6.0 / (zeta**2)) * (zeta * t - 1.0 + np.exp(-zeta * t))
        msd_ballistic = 3.0 * (t**2)
        msd_diffusive = 6.0 * d_long * t

        ax.loglog(t, msd_ballistic, color=COLORS["accent"], linestyle="--", linewidth=1.7, label=r"Ballistic Limit $(3k_B T/m)t^2$")
        ax.loglog(t, msd_diffusive, color=COLORS["good"], linestyle=":", linewidth=2.0, label=r"Brownian Limit $6Dt = \frac{6k_B T}{m\zeta}t$")
        ax.loglog(t, msd_exact, color=COLORS["primary"], linewidth=2.6, label=r"Exact Langevin $\langle \Delta \mathbf{r}^2(t) \rangle$")
        ax.axvline(1.0 / zeta, color=COLORS["danger"], linestyle="-.", linewidth=1.5, label=f"Crossover $t = \\zeta^{{-1}} = {1.0/zeta:.2f}$")
        ax.set_xlabel(r"Reduced Time $t$")
        ax.set_ylabel(r"Mean Square Displacement $\langle \Delta \mathbf{r}^2(t) \rangle$")
        ax.set_title(f"Ballistic-to-Diffusive Crossover ($\\zeta = {zeta:.2f},\\; D = {d_long:.2f}$)")
        ax.set_xlim(1e-2, 1e2)
        ax.set_ylim(1e-4, 1e4)
        ax.legend(loc="upper left", fontsize=8.2)
        regime_str = (
            "Underdamped ($\\zeta < 2\\omega_0$)"
            if zeta < 2.0
            else ("Critically Damped ($\\zeta = 2\\omega_0$)" if abs(zeta - 2.0) < 1e-3 else "Overdamped ($\\zeta > 2\\omega_0$)")
        )

    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Friction $\\zeta = {zeta:.2f}$ ({regime_str}):** Momentum relaxation time $\\tau_p = \\zeta^{{-1}} = $ "
        f"**{1.0 / zeta:.2f}**, asymptotic Einstein diffusivity $D = k_B T / (m\\zeta) = $ **{d_long:.3f}**. "
        f"For $t \\ll \\tau_p$, motion is inertial/ballistic ($\\langle\\Delta r^2\\rangle \\propto t^2$); "
        f"for $t \\gg \\tau_p$, inertia is damped out and Smoluchowski diffusion ($\\langle\\Delta r^2\\rangle = 6Dt$) emerges."
    )


def render_langevin_vs_dpd_hydrodynamics(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare channel Poiseuille velocity profiles under global Langevin damping vs. momentum-conserving DPD."""
    zeta_langevin = float(controls.get("zeta_langevin", 1.5))
    dpd_gamma = float(controls.get("dpd_gamma", 1.0))

    y = np.linspace(-1.0, 1.0, 300)
    # Channel half-width H = 1, body force density rho*g = 2.0
    # DPD conserves local momentum: viscosity eta_DPD scales with gamma, yielding exact Navier-Stokes parabola:
    # v_x(y) = (rho*g / (2*eta)) * (1 - y^2)
    eta_eff = 0.5 * (0.4 + 0.6 * dpd_gamma)
    v_max_ns = 1.0 / (2.0 * eta_eff)
    v_dpd = v_max_ns * (1.0 - y**2)

    # Global Langevin thermostat adds -zeta * v_i body drag, yielding Brinkman equation:
    # eta * d^2 v_x / dy^2 - rho * zeta * v_x = -rho * g
    if zeta_langevin < 1e-4:
        v_lang = v_dpd.copy()
        alpha_b = 0.0
    else:
        alpha_b = np.sqrt(zeta_langevin / eta_eff)
        v_lang = (1.0 / zeta_langevin) * (1.0 - np.cosh(alpha_b * y) / np.cosh(alpha_b))

    ax.plot(
        y,
        v_dpd,
        color=COLORS["good"],
        linewidth=2.6,
        label=f"DPD / Navier–Stokes Parabola ($\\gamma={dpd_gamma:.1f}$)",
    )
    ax.plot(
        y,
        v_lang,
        color=COLORS["danger"],
        linestyle="--",
        linewidth=2.3,
        label=f"Global Langevin Thermostat ($\\zeta={zeta_langevin:.2f}$)",
    )
    ax.fill_between(y, v_lang, v_dpd, color=COLORS["danger"], alpha=0.12, label="Destroyed Hydrodynamic Momentum")

    ax.set_xlabel(r"Transverse Channel Coordinate $y / H$")
    ax.set_ylabel(r"Streamwise Flow Velocity $v_x(y)$")
    ax.set_title("Poiseuille Flow: Momentum-Conserving DPD vs. Langevin Drag")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0.0, max(float(np.max(v_dpd)) * 1.15, 0.5))
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper right", fontsize=8.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    center_ratio = float(v_lang[len(y) // 2] / max(1e-8, v_dpd[len(y) // 2]))
    return (
        f"**DPD Friction $\\gamma = {dpd_gamma:.1f}$ vs. Langevin Friction $\\zeta = {zeta_langevin:.2f}$:** "
        f"DPD damps only relative velocities $\\mathbf{{v}}_{{ij}} = \\mathbf{{v}}_i - \\mathbf{{v}}_j$ with central antisymmetric pairs "
        f"$\\mathbf{{F}}_{{ij}} = -\\mathbf{{F}}_{{ji}}$, preserving the exact parabolic Navier–Stokes profile. "
        f"Global Langevin damps absolute velocity $-\\zeta\\mathbf{{v}}_i$, screening momentum and reducing centerline velocity to "
        f"**{100.0 * center_ratio:.1f}%** of the Navier–Stokes value."
    )
