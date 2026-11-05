"""Concept rendering functions for Lesson 10: Dynamical Correlations, Linear Response, and Transport Coefficients."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def render_vacf_and_msd_duality(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Interactive explorer showing VACF C_vv(t), running Green-Kubo integral D(t), and MSD across regimes."""
    regime = str(controls.get("regime", "dense_liquid"))
    view_mode = str(controls.get("view_mode", "vacf_and_running_d"))
    temp_scale = float(controls.get("temperature", 1.0))

    t = np.linspace(0.0, 4.0, 500)
    dt = t[1] - t[0]

    # Normalized VACF psi(t) = <v(t).v(0)> / <v^2> where <v^2> = 3 * temp_scale
    v2_3d = 3.0 * temp_scale

    if regime == "dilute_gas":
        # Monotonic exponential Langevin/Enskog decay exp(-t / tau_c)
        tau_c = 0.65
        psi = np.exp(-t / tau_c)
        regime_name = "Dilute Gas (Monotonic Exponential Decay)"
    elif regime == "dense_liquid":
        # Classic LJ liquid near triple point: cage backscattering rebound (negative dip)
        tau_d = 0.42
        omega_e = 6.8
        psi = np.exp(-t / tau_d) * (np.cos(omega_e * t) - 0.28 * np.sin(omega_e * t))
        regime_name = "Dense Liquid (Cage Backscattering Dip)"
    elif regime == "supercooled_glass":
        # Strong cage rattling with almost complete cancellation of integral (D ~ 0)
        tau_g = 0.55
        omega_g = 9.2
        psi = np.exp(-t / tau_g) * (np.cos(omega_g * t) - (1.0 / (omega_g * tau_g)) * np.sin(omega_g * t))
        regime_name = "Glassy / Arrested State (Cage Trapping, D → 0)"
    else:
        # Harmonic / warm crystalline solid: damped phonon oscillations integrating to 0
        tau_s = 1.15
        omega_s = 10.5
        psi = np.exp(-t / tau_s) * (np.cos(omega_s * t) - (1.0 / (omega_s * tau_s)) * np.sin(omega_s * t))
        regime_name = "Crystalline Solid (Vibrational Phonons, D = 0)"

    c_vv = v2_3d * psi
    # Running Green-Kubo diffusivity D(t) = (1/3) int_0^t <v(t').v(0)> dt'
    d_running = np.cumsum(c_vv) * (dt / 3.0)
    d_inf = max(0.0, float(d_running[-1]))

    # Exact MSD from double integral of VACF: <|r(t)-r(0)|^2> = 6 * int_0^t D(t') dt'
    msd = 6.0 * np.cumsum(d_running) * dt

    if view_mode == "vacf_and_running_d":
        ax.axhline(0.0, color=COLORS["muted"], ls="--", lw=1.0)
        ax.plot(t, psi, color=COLORS["primary"], lw=2.3, label=r"Normalized VACF $C_{vv}(t)/C_{vv}(0)$")
        ax.fill_between(
            t,
            0.0,
            psi,
            where=(psi >= 0),
            color=COLORS["primary"],
            alpha=0.20,
            label="Forward persistence (+)",
        )
        ax.fill_between(
            t,
            0.0,
            psi,
            where=(psi < 0),
            color=COLORS["danger"],
            alpha=0.25,
            label="Cage backscattering rebound (−)",
        )

        ax2 = ax.twinx()
        ax2.plot(
            t,
            d_running,
            color=COLORS["accent"],
            lw=2.1,
            ls="-.",
            label=r"Running Green-Kubo $D(t)$",
        )
        ax2.set_ylabel(r"Running Diffusivity $D(t) = \frac{1}{3}\int_0^t C_{vv}(t')dt'$", color=COLORS["accent"])
        ax2.tick_params(axis="y", labelcolor=COLORS["accent"])
        ax2.set_ylim(-0.05, max(0.85, float(np.max(d_running)) * 1.18))
        ax2.spines["top"].set_visible(False)

        ax.set_xlabel(r"Lag Time $t / \tau_0$")
        ax.set_ylabel(r"Normalized VACF $\langle \mathbf{v}(t)\cdot\mathbf{v}(0)\rangle / \langle v^2\rangle$")
        ax.set_ylim(-0.48, 1.08)
        ax.set_title(f"{regime_name} · Green-Kubo $D = {d_inf:.3f}$")
        ax.legend(loc="upper right", fontsize=8.2)
    else:
        # Plot MSD <|r(t)-r(0)|^2> showing ballistic t^2 -> diffusive 6Dt crossover
        msd_ballistic = v2_3d * (t**2)
        ax.plot(t, msd, color=COLORS["primary"], lw=2.4, label=r"True MSD $\langle |\mathbf{r}(t)-\mathbf{r}(0)|^2\rangle$")
        ax.plot(
            t,
            msd_ballistic,
            color=COLORS["muted"],
            ls=":",
            lw=1.6,
            label=r"Short-time ballistic $\langle v^2\rangle t^2 = (3k_B T/m)t^2$",
        )
        ax.plot(
            t,
            6.0 * d_inf * t,
            color=COLORS["accent"],
            ls="--",
            lw=1.7,
            label=f"Long-time Einstein slope $6Dt$ ($D={d_inf:.3f}$)",
        )
        ax.set_xlabel(r"Time $t / \tau_0$")
        ax.set_ylabel(r"Mean-Squared Displacement $\langle |\Delta \mathbf{r}(t)|^2\rangle$")
        ax.set_ylim(0.0, max(2.5, float(msd[-1]) * 1.15))
        ax.set_title(f"Einstein MSD Route · {regime_name}")
        ax.legend(loc="upper left", fontsize=8.3)

    ax.set_xlim(0.0, 4.0)
    ax.grid(True, color=COLORS["grid"], ls="--", lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)

    min_psi = float(np.min(psi))
    return (
        f"**Regime:** {regime_name} ($T^* = {temp_scale:.2f}$). "
        f"Minimum VACF dip: $\\min C_{{vv}}(t)/C_{{vv}}(0) = {min_psi:.2f}$. "
        f"Asymptotic self-diffusion coefficient: **$D = {d_inf:.3f}$** "
        f"(Einstein long-time MSD slope $6D = {6.0*d_inf:.3f}$)."
    )


def render_pbc_unwrapping_msd(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare true unwrapped MSD (6Dt) against erroneously PBC-wrapped box coordinates."""
    diff_coeff = float(controls.get("diffusivity", 0.35))
    box_length = float(controls.get("box_length", 6.0))
    show_components = bool(int(controls.get("show_1d_limit", 1)))

    t = np.linspace(0.0, 25.0, 400)
    # True unwrapped 3D MSD: <|\Delta r_unwrap(t)|^2> = (3*v2*tau^2)*(t/tau - 1 + exp(-t/tau)) -> 6 D t
    tau_b = 0.25
    msd_unwrap = 6.0 * diff_coeff * (t - tau_b * (1.0 - np.exp(-t / tau_b)))

    # If a user mistakenly uses PBC-wrapped coordinates in [0, L)^3 without unwrapping box crossings,
    # the uniform displacement distribution in [-L/2, L/2]^3 or [0, L)^3 saturates!
    # For wrapped coordinates in [0, L)^3, two independent uniform positions have <|r1 - r2|^2> = 3 * (L^2 / 6) = L^2 / 2
    # For minimum-image displacement in [-L/2, L/2]^3, saturation is 3 * (L^2 / 12) = L^2 / 4.
    sat_min_image = 0.25 * (box_length**2)
    sat_box_coord = 0.50 * (box_length**2)

    msd_wrapped_box = sat_box_coord * (1.0 - np.exp(-msd_unwrap / max(sat_box_coord, 1e-6)))
    msd_min_img = sat_min_image * (1.0 - np.exp(-msd_unwrap / max(sat_min_image, 1e-6)))

    ax.plot(
        t,
        msd_unwrap,
        color=COLORS["good"],
        lw=2.5,
        label=f"Correct Unwrapped MSD $\\langle |\\mathbf{{r}}_{{\\mathrm{{unwrap}}}}(t)-\\mathbf{{r}}(0)|^2\\rangle \\to 6Dt$",
    )
    ax.plot(
        t,
        msd_wrapped_box,
        color=COLORS["danger"],
        lw=2.2,
        ls="--",
        label=f"BUG: PBC-Wrapped Box Coords $\\in [0, L)^3$ (saturates at $L^2/2 = {sat_box_coord:.1f}$)",
    )
    if show_components:
        ax.plot(
            t,
            msd_min_img,
            color=COLORS["warning"],
            lw=1.9,
            ls=":",
            label=f"BUG: Minimum-Image $\\Delta\\mathbf{{r}} \\in [-L/2, L/2]^3$ (saturates at $L^2/4 = {sat_min_image:.1f}$)",
        )

    ax.axhline(sat_box_coord, color=COLORS["muted"], ls="-.", lw=1.0)
    ax.set_xlabel(r"Simulation Time $t$ (ps)")
    ax.set_ylabel(r"Mean-Squared Displacement ($\text{Å}^2$)")
    ax.set_xlim(0.0, 25.0)
    ax.set_ylim(0.0, max(55.0, float(msd_unwrap[-1]) * 1.08))
    ax.set_title(f"Why Einstein MSD Requires 'Unwound' Coordinates ($D={diff_coeff:.2f}\\,\\mathrm{{Å^2/ps}}, L={box_length:.1f}\\,\\text{{Å}}$)")
    ax.grid(True, color=COLORS["grid"], ls="--", lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", fontsize=8.0)

    slope_true = 6.0 * diff_coeff
    return (
        f"**True Unwrapped Slope:** $\\frac{{d}}{{dt}}\\mathrm{{MSD}}_{{\\mathrm{{unwrap}}}} = 6D = {slope_true:.2f}\\,\\mathrm{{Å^2/ps}} \\implies D = {diff_coeff:.2f}\\,\\mathrm{{Å^2/ps}}$. "
        f"**Wrapped-Coordinate Bug:** When $\\sqrt{{\\mathrm{{MSD}}}} \\gtrsim L/2 = {box_length/2:.1f}\\,\\text{{Å}}$, PBC-wrapped displacements plateau at $\\sim L^2/2 = {sat_box_coord:.1f}\\,\\mathrm{{Å^2}}$, falsely giving an apparent long-time slope $D_{{\\mathrm{{app}}}} = 0$!"
    )


def render_alder_wainwright_tails(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Log-log plot comparing exponential kinetic theory against Alder-Wainwright hydrodynamic t^{-d/2} tails."""
    dim = int(controls.get("dimension", 2))
    tau_e = float(controls.get("collision_time", 1.0))
    panel_mode = str(controls.get("panel_mode", "loglog_vacf"))

    t = np.logspace(0.0, 3.0, 400)
    # Uncorrelated Enskog kinetic theory predicts pure exponential decay exp(-t / tau_e)
    c_exp = np.exp(-t / tau_e)

    # Hydrodynamic vortex recirculation produces algebraic tail ~ A * (t / tau_e)^(-d/2)
    amp_tail = 0.18
    c_hydro_2d = 0.82 * np.exp(-t / tau_e) + amp_tail * ((t / tau_e) ** (-1.0))
    c_hydro_3d = 0.82 * np.exp(-t / tau_e) + amp_tail * ((t / tau_e) ** (-1.5))

    if panel_mode == "loglog_vacf":
        ax.loglog(
            t,
            np.clip(c_exp, 1e-7, None),
            color=COLORS["muted"],
            ls="--",
            lw=1.9,
            label=r"Molecular Chaos / Enskog ($\propto e^{-t/\tau_E}$)",
        )
        ax.loglog(
            t,
            c_hydro_3d,
            color=COLORS["primary"],
            lw=2.3 if dim == 3 else 1.6,
            alpha=1.0 if dim == 3 else 0.65,
            label=r"3D Hydrodynamic Vortex Tail ($\propto t^{-3/2}$, integrable)",
        )
        ax.loglog(
            t,
            c_hydro_2d,
            color=COLORS["accent"],
            lw=2.3 if dim == 2 else 1.6,
            alpha=1.0 if dim == 2 else 0.65,
            label=r"2D Hydrodynamic Vortex Tail ($\propto t^{-1}$, diverges!)",
        )
        ax.set_xlabel(r"Dimensionless Time $t / \tau_E$ (log scale)")
        ax.set_ylabel(r"Velocity Autocorrelation $C_{vv}(t) / C_{vv}(0)$ (log scale)")
        ax.set_xlim(1.0, 1000.0)
        ax.set_ylim(1e-5, 1.2)
        ax.set_title(f"Alder–Wainwright Long-Time Tails $C_{{vv}}(t) \\propto t^{{-d/2}}$ (Focus: {dim}D)")
        ax.legend(loc="lower left", fontsize=8.2)
    else:
        # Running Green-Kubo integral D(t) = (1/d) int_1^t C_vv(t') dt'
        # Analytic integration on log grid using trapezoidal rule
        d_exp = np.zeros_like(t)
        d_3d = np.zeros_like(t)
        d_2d = np.zeros_like(t)
        for i in range(1, len(t)):
            dt_step = t[i] - t[i - 1]
            d_exp[i] = d_exp[i - 1] + 0.5 * (c_exp[i] + c_exp[i - 1]) * dt_step / 3.0
            d_3d[i] = d_3d[i - 1] + 0.5 * (c_hydro_3d[i] + c_hydro_3d[i - 1]) * dt_step / 3.0
            d_2d[i] = d_2d[i - 1] + 0.5 * (c_hydro_2d[i] + c_hydro_2d[i - 1]) * dt_step / 2.0

        ax.semilogx(t, d_exp, color=COLORS["muted"], ls="--", lw=1.9, label="Exponential Theory (converges rapidly)")
        ax.semilogx(t, d_3d, color=COLORS["primary"], lw=2.3, label=r"3D Green-Kubo $D_{3\mathrm{D}}(t)$ ($\int t^{-3/2}dt$ converges)")
        ax.semilogx(t, d_2d, color=COLORS["accent"], lw=2.3, label=r"2D Green-Kubo $D_{2\mathrm{D}}(t) \propto \ln(t)$ (DIVERGES!)")
        ax.set_xlabel(r"Upper Integration Time $t / \tau_E$ (log scale)")
        ax.set_ylabel(r"Running Green-Kubo Integral $D(t)$")
        ax.set_xlim(1.0, 1000.0)
        ax.set_title("Running Green-Kubo Integral: 3D Convergence vs. 2D Logarithmic Divergence")
        ax.legend(loc="upper left", fontsize=8.2)

    ax.grid(True, which="both", color=COLORS["grid"], ls="--", lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if dim == 2:
        tail_msg = "In **2D ($d=2$)**, the vortex tail decays as **$t^{-2/2} = t^{-1}$**, so $\\int_0^t (t')^{-1} dt' \\propto \\ln t$ diverges logarithmically—strictly speaking, the 2D Fickian diffusion constant does not exist in the thermodynamic limit!"
    else:
        tail_msg = "In **3D ($d=3$)**, the vortex tail decays as **$t^{-3/2}$**, whose integral $\\int^\\infty t^{-3/2} dt < \\infty$ converges to a finite enhancement of $D$ (approached as $t^{-1/2}$)."

    return f"**Selected Dimension $d = {dim}$ ($\\tau_E = {tau_e:.1f}$):** {tail_msg}"
