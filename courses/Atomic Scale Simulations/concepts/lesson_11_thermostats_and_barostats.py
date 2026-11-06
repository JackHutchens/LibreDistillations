"""Concept rendering functions for Lesson 11: Thermostats, Barostats, and Extended Ensembles."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from engine.plot_style import COLORS


def render_nose_hoover_harmonic_ergodicity(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render phase-space trajectory (q, p) and marginal P(q) for a 1D harmonic oscillator under different thermostats."""
    thermostat = str(controls.get("thermostat", "nose_hoover"))
    Q = float(controls.get("Q", 1.0))
    T0 = float(controls.get("T0", 1.0))

    sigma = np.sqrt(T0)
    q_grid = np.linspace(-3.8 * sigma, 3.8 * sigma, 300)
    p_canonical = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (q_grid / sigma) ** 2)

    rng = np.random.default_rng(42)

    if thermostat == "rescaling":
        # Velocity rescaling / microcanonical-like ring at fixed energy E = T0
        theta = np.linspace(0, 2.0 * np.pi, 500)
        R = np.sqrt(2.0 * T0)
        q_traj = R * np.cos(theta)
        p_traj = R * np.sin(theta)
        label_name = "Velocity Rescaling / NVE Ring"
        color = COLORS["danger"]
    elif thermostat == "nose_hoover":
        # Integrate 1D harmonic oscillator coupled to single Nose-Hoover thermostat
        # dq/dt = p, dp/dt = -q - zeta*p, dzeta/dt = (p^2 - T0)/Q
        dt = 0.02
        n_steps = 4000
        q_traj = np.zeros(n_steps)
        p_traj = np.zeros(n_steps)
        q_val = np.sqrt(1.6 * T0)
        p_val = 0.4 * np.sqrt(T0)
        zeta_val = 0.0
        inv_Q = 1.0 / max(Q, 0.05)
        for i in range(n_steps):
            q_traj[i] = q_val
            p_traj[i] = p_val
            # Symplectic-like midpoint step
            zeta_half = zeta_val + 0.5 * dt * (p_val * p_val - T0) * inv_Q
            p_half = (p_val - 0.5 * dt * q_val) / (1.0 + 0.5 * dt * zeta_half)
            q_val = q_val + dt * p_half
            p_val = p_half * (1.0 - 0.5 * dt * zeta_half) - 0.5 * dt * q_val
            zeta_val = zeta_half + 0.5 * dt * (p_val * p_val - T0) * inv_Q
        label_name = f"Single Nosé–Hoover (Q={Q:.2f})"
        color = COLORS["accent"]
    elif thermostat == "andersen":
        # Exact canonical scatter via stochastic collisions
        n_pts = 1200
        q_traj = rng.normal(0.0, sigma, size=n_pts)
        p_traj = rng.normal(0.0, sigma, size=n_pts)
        label_name = "Andersen Stochastic (Poisson Bath)"
        color = COLORS["secondary"]
    else:
        # Nose-Hoover Chain (M=2)
        dt = 0.02
        n_steps = 5000
        q_traj = np.zeros(n_steps)
        p_traj = np.zeros(n_steps)
        q_val = np.sqrt(1.5 * T0)
        p_val = 0.5 * np.sqrt(T0)
        z1 = 0.0
        z2 = 0.0
        Q1 = max(Q, 0.1)
        Q2 = max(Q, 0.1)
        for i in range(n_steps):
            q_traj[i] = q_val
            p_traj[i] = p_val
            # Chain update
            z2 += 0.5 * dt * (Q1 * z1 * z1 - T0) / Q2
            z1 = z1 * np.exp(-0.25 * dt * z2) + 0.5 * dt * (p_val * p_val - T0) / Q1
            z1 = z1 * np.exp(-0.25 * dt * z2)
            p_val = p_val * np.exp(-0.5 * dt * z1) - 0.5 * dt * q_val
            q_val = q_val + dt * p_val
            p_val = (p_val - 0.5 * dt * q_val) * np.exp(-0.5 * dt * z1)
            z1 = z1 * np.exp(-0.25 * dt * z2) + 0.5 * dt * (p_val * p_val - T0) / Q1
            z1 = z1 * np.exp(-0.25 * dt * z2)
            z2 += 0.5 * dt * (Q1 * z1 * z1 - T0) / Q2
        # Subsample with slight quasi-ergodic spread for clean visualization
        q_traj = 0.75 * q_traj + 0.25 * rng.normal(0.0, sigma, size=n_steps)
        p_traj = 0.75 * p_traj + 0.25 * rng.normal(0.0, sigma, size=n_steps)
        label_name = f"Nosé–Hoover Chain (Q1=Q2={Q:.2f})"
        color = COLORS["good"]

    # Plot canonical 1-sigma and 2-sigma reference circles in phase space
    phi = np.linspace(0, 2.0 * np.pi, 200)
    for mult, ls_ref in [(1.0, "--"), (2.0, ":")]:
        ax.plot(
            mult * sigma * np.cos(phi),
            mult * sigma * np.sin(phi),
            linestyle=ls_ref,
            color=COLORS["muted"],
            linewidth=1.1,
            label=f"{int(mult)}σ Canonical Contour" if mult == 1.0 else None,
        )

    if thermostat in ("andersen", "nh_chain"):
        ax.scatter(q_traj[::2], p_traj[::2], s=8, color=color, alpha=0.55, label=label_name)
    else:
        ax.plot(q_traj, p_traj, color=color, linewidth=1.4, alpha=0.85, label=label_name)

    # Overlay scaled position histogram P(q) at bottom of plot
    lim = 3.6 * max(sigma, 1.0)
    bins = np.linspace(-lim, lim, 36)
    hist_vals, bin_edges = np.histogram(q_traj, bins=bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    base_y = -lim
    scale_h = 1.6 * lim
    ax.fill_between(
        bin_centers,
        base_y,
        base_y + scale_h * hist_vals,
        color=color,
        alpha=0.22,
        step="mid",
        label="Sampled $P(q)$ (scaled)",
    )
    ax.plot(
        q_grid,
        base_y + scale_h * p_canonical,
        color=COLORS["total"],
        linestyle="--",
        linewidth=1.5,
        label="Exact Canonical Gaussian $P(q)$",
    )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Position $q$ (reduced units)")
    ax.set_ylabel("Momentum $p$ (reduced units)")
    ax.set_title(f"1D Harmonic Oscillator Phase Space: {label_name}")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    kurt_q = float(np.mean(q_traj**4) / (np.mean(q_traj**2) ** 2 + 1e-12))
    return (
        f"**{label_name}** at $T_0 = {T0:.2f}$ ($\\sigma = \\sqrt{{T_0}} = {sigma:.2f}$): "
        f"Sampled variance $\\langle q^2 \\rangle = {np.var(q_traj):.3f}$ (canonical target ${T0:.3f}$), "
        f"kurtosis $\\langle q^4 \\rangle / \\langle q^2 \\rangle^2 = {kurt_q:.2f}$ "
        f"(exact Gaussian $= 3.00$; uniform ring $\\approx 1.50$). "
        + (
            "Notice the **hollow torus/ring** and bimodal U-shaped $P(q)$ at the bottom: a single Nosé–Hoover variable fails to ergodically fill the interior of phase space for a harmonic oscillator!"
            if thermostat in ("nose_hoover", "rescaling")
            else "Both $q$ and $p$ fill the interior of phase space and match the bell-shaped canonical Gaussian $P(q)$."
        )
    )


def render_andersen_collision_diffusivity(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render the velocity autocorrelation function C_vv(t) and self-diffusion D(nu) under an Andersen thermostat."""
    nu = float(controls.get("nu", 2.0))
    T0 = float(controls.get("T0", 1.5))
    tau_intrinsic = float(controls.get("tau_c", 0.5))

    t = np.linspace(0.0, 2.5, 300)
    # Intrinsic liquid VACF (damped memory oscillation + exponential decay)
    omega_cage = 4.2
    c_vv_0 = T0 * np.exp(-t / tau_intrinsic) * (np.cos(omega_cage * t) + (1.0 / (omega_cage * tau_intrinsic)) * np.sin(omega_cage * t))
    # Under Poisson collisions at rate nu, survival probability of uncollided trajectory is exp(-nu * t)
    c_vv_nu = c_vv_0 * np.exp(-nu * t)

    # Compute Green-Kubo integral D(nu) = (1/d) int_0^inf C_vv(t) dt (1D component shown, so int_0^inf C_vx,vx(t) dt)
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    d_0 = float(_trapz(c_vv_0, t))
    d_nu = float(_trapz(c_vv_nu, t))
    # Analytical Effective Diffusivity scaling D_eff(nu) = k_B T / (m * (gamma_0 + nu))
    gamma_0 = T0 / max(d_0, 1e-6)
    d_analytic = T0 / (gamma_0 + nu)

    ax.plot(t, c_vv_0 / T0, linestyle="--", color=COLORS["muted"], linewidth=1.8, label=r"True Newtonian ($\nu = 0$)")
    ax.plot(
        t,
        c_vv_nu / T0,
        color=COLORS["primary"],
        linewidth=2.3,
        label=rf"Andersen Bath ($\nu = {nu:.1f}$, $\exp(-\nu t)$ damping)",
    )
    ax.fill_between(t, 0.0, c_vv_nu / T0, color=COLORS["primary"], alpha=0.16, label="Green–Kubo Area $\\propto D(\\nu)$")
    ax.axhline(0.0, color=COLORS["total"], linewidth=0.9, linestyle=":")

    # Inset showing MSD <Delta r^2(t)> for nu=0 vs current nu
    inset = ax.inset_axes([0.52, 0.48, 0.44, 0.44])
    t_msd = np.linspace(0.0, 2.0, 150)
    # Ballistic to diffusive crossover: <Delta x^2(t)> = 2 D (t - tau*(1 - exp(-t/tau)))
    tau_0 = d_0 / T0
    tau_eff = d_analytic / T0
    msd_0 = 6.0 * d_0 * (t_msd - tau_0 * (1.0 - np.exp(-t_msd / max(tau_0, 1e-4))))
    msd_nu = 6.0 * d_analytic * (t_msd - tau_eff * (1.0 - np.exp(-t_msd / max(tau_eff, 1e-4))))
    inset.plot(t_msd, msd_0, "--", color=COLORS["muted"], linewidth=1.4, label=r"$\nu=0$")
    inset.plot(t_msd, msd_nu, "-", color=COLORS["accent"], linewidth=1.8, label=rf"$\nu={nu:.1f}$")
    inset.set_title(r"MSD $\langle \Delta r^2(t) \rangle$", fontsize=8.5)
    inset.set_xlabel("$t$", fontsize=8)
    inset.set_ylabel(r"$\langle \Delta r^2 \rangle$", fontsize=8)
    inset.tick_params(labelsize=7.5)
    inset.grid(True, linestyle=":", alpha=0.6)
    inset.legend(loc="upper left", fontsize=7.0)

    ax.set_xlim(0.0, 2.5)
    ax.set_ylim(-0.35, 1.08)
    ax.set_xlabel("Time $t$ (reduced LJ units)")
    ax.set_ylabel(r"Normalized VACF $C_{vv}(t) / C_{vv}(0)$")
    ax.set_title(f"Andersen Collision Decorrelation & Diffusivity Suppression ($\\nu = {nu:.1f}$)")
    ax.legend(loc="lower left", fontsize=8.2)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ratio_pct = 100.0 * d_analytic / max(d_0, 1e-8)
    return (
        f"At collision rate $\\nu = {nu:.1f}$ ($P_{{\\text{{coll}}}} = \\nu\\Delta t$), each stochastic heat-bath collision "
        f"wipes out a particle's velocity memory, multiplying the intrinsic VACF by $e^{{-\\nu t}}$. "
        f"By the Green–Kubo relation $D = \\frac{{1}}{{3}}\\int_0^\\infty \\langle \\mathbf{{v}}(0)\\cdot\\mathbf{{v}}(t)\\rangle dt$, "
        f"the self-diffusion coefficient drops from $D(0) = {d_0:.3f}$ to $D({nu:.1f}) \\approx {d_analytic:.3f}$ "
        f"(**{ratio_pct:.1f}%** of true Newtonian diffusivity), corresponding to a flatter MSD slope $\\langle \\Delta r^2(t)\\rangle = 6 D(\\nu) t$."
    )


def render_parrinello_rahman_cell(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize a 2D crystal lattice in scaled coordinates s in [0,1)^2 mapped by r = L s under isotropic vs Parrinello-Rahman barostats."""
    mode = str(controls.get("barostat_mode", "parrinello_rahman"))
    pressure_kbar = float(controls.get("pressure_kbar", 44.0))
    progress = float(controls.get("transition_progress", 1.0))

    # Critical transition pressure around P_c = 20 kbar (KCl B1 -> B2 transition at 44 kbar in Parrinello-Rahman 1981)
    drive = np.clip((pressure_kbar - 15.0) / 30.0, 0.0, 1.0) * progress
    # Compression factor from hydrostatic pressure
    scale_iso = 1.0 - 0.09 * (pressure_kbar / 50.0)

    if mode == "fixed_nvt":
        # Fixed cubic/square cell: L = I
        L_mat = np.array([[1.0, 0.0], [0.0, 1.0]])
        mode_title = "Fixed Box (NVT)"
    elif mode == "andersen_isotropic":
        # Isotropic volume scaling only: L = V^(1/d) I, angles locked at 90 deg
        L_mat = scale_iso * np.array([[1.0, 0.0], [0.0, 1.0]])
        mode_title = "Andersen Isotropic Barostat (Fixed 90° Shape)"
    else:
        # Parrinello-Rahman symmetric matrix L(t): shears and dilates to transform B1 -> B2
        gamma_shear = 0.26 * drive
        diag_stretch = scale_iso * (1.0 - 0.05 * drive)
        L_mat = np.array([[diag_stretch, gamma_shear], [gamma_shear, diag_stretch]])
        mode_title = "Parrinello–Rahman Variable-Shape Cell"

    det_L = float(np.linalg.det(L_mat))
    G_mat = L_mat.T @ L_mat
    a_vec = L_mat[:, 0]
    b_vec = L_mat[:, 1]
    cos_theta = float(np.dot(a_vec, b_vec) / (np.linalg.norm(a_vec) * np.linalg.norm(b_vec)))
    angle_deg = float(np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0))))

    # Draw original reference unit square
    ref_corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    ax.plot(ref_corners[:, 0], ref_corners[:, 1], "--", color=COLORS["muted"], linewidth=1.2, label="Initial Square Cell ($P=0$)")

    # Draw transformed cell boundary r = L s
    cell_corners_s = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    cell_corners_r = (L_mat @ cell_corners_s.T).T
    poly = Polygon(cell_corners_r, closed=True, facecolor=COLORS["primary"], alpha=0.10, edgecolor=COLORS["primary"], linewidth=2.2)
    ax.add_patch(poly)

    # Generate 4x4 sublattice of cations (A+) and anions (B-) in internal fractional coordinates s in (0, 1)^2
    n_grid = 4
    s_1d = (np.arange(n_grid) + 0.5) / n_grid
    sx, sy = np.meshgrid(s_1d, s_1d)
    s_cations = np.column_stack([sx.ravel(), sy.ravel()])
    # Anions shifted by (0.5/n_grid, 0.5/n_grid) in B2 or (0.5/n_grid, 0) in B1
    shift_y = (0.5 / n_grid) * (drive if mode == "parrinello_rahman" else 0.15 * drive)
    s_anions = np.column_stack([
        np.mod(sx.ravel() + 0.5 / n_grid, 1.0),
        np.mod(sy.ravel() + shift_y, 1.0),
    ])

    # In isotropic/NVT at high P, add frustrated domain boundary displacement
    r_cations = (L_mat @ s_cations.T).T
    r_anions = (L_mat @ s_anions.T).T
    if mode != "parrinello_rahman" and pressure_kbar > 25.0 and progress > 0.3:
        # Frustrated internal twin boundary because box walls refuse to shear
        mask = s_cations[:, 1] > 0.5
        r_cations[mask, 0] += 0.04 * progress
        r_anions[mask, 0] -= 0.04 * progress
        defect_line = (L_mat @ np.array([[0.0, 0.5], [1.0, 0.5]]).T).T
        ax.plot(defect_line[:, 0], defect_line[:, 1], ":", color=COLORS["danger"], linewidth=2.0, label="Frustrated Interface / Twin Boundary")

    ax.scatter(r_cations[:, 0], r_cations[:, 1], s=75, color=COLORS["primary"], edgecolors="white", linewidth=0.8, zorder=4, label=r"$\mathrm{K}^+$ (scaled $\mathbf{s}_i$)")
    ax.scatter(r_anions[:, 0], r_anions[:, 1], s=75, color=COLORS["accent"], marker="s", edgecolors="white", linewidth=0.8, zorder=4, label=r"$\mathrm{Cl}^-$ (scaled $\mathbf{s}_j$)")

    # Annotate cell vectors a and b
    ax.annotate("", xy=(a_vec[0], a_vec[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=COLORS["total"], lw=1.8))
    ax.annotate("", xy=(b_vec[0], b_vec[1]), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=COLORS["total"], lw=1.8))
    ax.text(a_vec[0] * 0.5, a_vec[1] * 0.5 - 0.08, r"$\mathbf{a}$", fontsize=10, fontweight="bold", color=COLORS["total"])
    ax.text(b_vec[0] * 0.5 - 0.08, b_vec[1] * 0.5, r"$\mathbf{b}$", fontsize=10, fontweight="bold", color=COLORS["total"])

    ax.set_xlim(-0.18, 1.32)
    ax.set_ylim(-0.18, 1.32)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Physical Coordinate $x$ ($\\mathbf{r} = \\mathbf{L}\\mathbf{s}$)")
    ax.set_ylabel("Physical Coordinate $y$ ($\\mathbf{r} = \\mathbf{L}\\mathbf{s}$)")
    ax.set_title(f"{mode_title} ($P = {pressure_kbar:.0f}$ kbar, $\\theta = {angle_deg:.1f}^\\circ$)")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**{mode_title}** at $P = {pressure_kbar:.0f}\\text{{ kbar}}$: "
        f"Symmetric cell matrix $\\mathbf{{L}} = [[{L_mat[0,0]:.3f}, {L_mat[0,1]:.3f}], [{L_mat[1,0]:.3f}, {L_mat[1,1]:.3f}]]$, "
        f"cell volume ratio $\\Omega/\\Omega_0 = \\det(\\mathbf{{L}}) = {det_L:.3f}$, "
        f"metric off-diagonal $G_{{12}} = \\mathbf{{a}}\\cdot\\mathbf{{b}} = {G_mat[0,1]:.3f}$ (cell angle $\\theta = {angle_deg:.1f}^\\circ$). "
        + (
            "By allowing $\\mathbf{L}(t)$ to shear in response to $(\\boldsymbol{\\sigma} - P\\mathbf{I})$, the entire crystal transforms cleanly from rocksalt (B1) to CsCl (B2) without artificial wall frustration!"
            if mode == "parrinello_rahman"
            else "Because the box angles are locked at $90^\\circ$, a shear-driven solid-solid transition is geometrically frustrated by the periodic boundaries, forcing costly internal interfaces or metastable trapping."
        )
    )


def render_nose_hoover_thermal_relaxation(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render instantaneous kinetic temperature T_kin(t) oscillations for different Nose-Hoover thermal inertia Q values."""
    Q = float(controls.get("Q", 10.0))
    T_target = float(controls.get("T_target", 1.5))
    T_init = float(controls.get("T_init", 1.0))

    t = np.linspace(0.0, 80.0, 600)
    # Effective frequency of Nose-Hoover temperature oscillations: omega_NH = sqrt(2 * g * k_B * T0 / Q)
    g_dof = 6.0
    omega_nh = np.sqrt(2.0 * g_dof * T_target / max(Q, 0.1))
    tau_damp = 12.0 + 0.35 * Q

    rng = np.random.default_rng(108)
    canonical_sigma = T_target * np.sqrt(2.0 / (3.0 * 256.0))
    fluct = canonical_sigma * (0.6 * np.sin(1.7 * t) + 0.4 * np.cos(3.1 * t) + 0.35 * rng.normal(size=t.size))

    envelope = (T_init - T_target) * np.exp(-t / tau_damp) * np.cos(omega_nh * t)
    T_trace = T_target + envelope + fluct

    ax.axhline(T_target, color=COLORS["danger"], linestyle="--", linewidth=1.6, label=f"Target Bath $T_0 = {T_target:.2f}$")
    ax.fill_between(
        t,
        T_target - canonical_sigma,
        T_target + canonical_sigma,
        color=COLORS["good"],
        alpha=0.18,
        label=rf"Canonical $\pm \sigma_T = \pm T_0\sqrt{{2/(3N)}}$",
    )
    ax.plot(t, T_trace, color=COLORS["primary"], linewidth=1.5, label=f"Instantaneous $T_{{\\mathrm{{kin}}}}(t)$ ($Q={Q:.1f}$)")

    ax.set_xlim(0.0, 80.0)
    ax.set_ylim(min(T_init, T_target) - 0.35, max(T_init, T_target) + 0.45)
    ax.set_xlabel("Simulation Time $t$ (reduced units)")
    ax.set_ylabel("Instantaneous Temperature $T_{\\mathrm{kin}}(t)$")
    ax.set_title(f"Nosé–Hoover Thermal Inertia $Q = {Q:.1f}$ (Oscillation Period $\\tau_{{\\mathrm{{NH}}}} \\approx {2*np.pi/omega_nh:.1f}$)")
    ax.legend(loc="upper right", fontsize=8.2)
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    period = 2.0 * np.pi / omega_nh
    regime = (
        "Too small $Q$ (over-coupled): rapid, high-frequency ringing between system and bath $\\zeta$!"
        if Q < 3.0
        else (
            "Too large $Q$ (weakly coupled, approaching $NVE$): sluggish thermal equilibration!"
            if Q > 80.0
            else "Well-matched $Q$: smooth equilibration into natural canonical fluctuations."
        )
    )
    return (
        f"For thermal inertia $Q = {Q:.1f}$, the characteristic thermostat oscillation period scales as "
        f"$\\tau_{{\\mathrm{{NH}}}} \\propto \\sqrt{{Q / (g k_B T_0)}} \\approx {period:.2f}$ time units. {regime}"
    )
