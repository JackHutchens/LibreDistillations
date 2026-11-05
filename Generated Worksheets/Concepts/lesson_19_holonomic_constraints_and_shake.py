"""Interactive concept-cell renderers for Lesson 19: Holonomic Constraints, Rigid Bodies & The SHAKE Algorithm."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt


def _val(ctrl: Any, default: Any = 0.0) -> Any:
    if ctrl is None:
        return default
    return getattr(ctrl, "value", ctrl)


def render_timescale_and_quantum_freeze(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare vibrational quantum excitation and MD timestep limits with vs. without bond constraints."""
    temp_K = float(_val(controls.get("temperature_K"), 300.0))
    wavenumber_cm = float(_val(controls.get("wavenumber_cm"), 3000.0))
    freeze_mode = str(_val(controls.get("view_mode"), "quantum_ratio"))

    # Constants
    # hc / k_B = 1.438777 cm * K
    hc_over_kB = 1.438777
    theta_vib = hc_over_kB * wavenumber_cm  # Characteristic vibrational temperature hbar*omega / k_B in K
    x_val = theta_vib / max(temp_K, 1.0)
    # Speed of light in cm/fs = 2.99792458e-5 cm/fs
    c_cm_fs = 2.99792458e-5
    period_fs = 1.0 / (c_cm_fs * wavenumber_cm)
    dt_unconstrained = period_fs / 20.0
    dt_constrained = 4.0  # typical fs when fast stretches are frozen (limited by bends/torsions ~ 800 cm^-1)

    if freeze_mode == "quantum_ratio":
        wn_grid = np.linspace(100.0, 4000.0, 400)
        x_grid = hc_over_kB * wn_grid / max(temp_K, 1.0)
        # Quantum heat capacity contribution C_vib / k_B = x^2 e^x / (e^x - 1)^2
        x_safe = np.clip(x_grid, 1e-4, 100.0)
        c_quantum = (x_safe ** 2) * np.exp(-x_safe) / ((1.0 - np.exp(-x_safe)) ** 2)
        n_bose = 1.0 / (np.exp(x_safe) - 1.0)

        ax.plot(wn_grid, c_quantum, color="#1f77b4", lw=2.4, label=r"Quantum heat capacity $C_{\mathrm{vib}}/k_B$")
        ax.axhline(1.0, color="#d62728", ls="--", lw=1.6, label=r"Classical equipartition limit ($1\,k_B$)")
        ax.plot(wn_grid, n_bose, color="#2ca02c", lw=1.8, ls="-.", label=r"Mean phonon occupancy $\langle n\rangle = (e^{\hbar\omega/k_B T}-1)^{-1}$")

        c_pt = (x_val ** 2) * np.exp(-x_val) / ((1.0 - np.exp(-min(x_val, 80.0))) ** 2) if x_val < 80 else 0.0
        ax.axvline(wavenumber_cm, color="#9467bd", ls=":", lw=1.8)
        ax.scatter([wavenumber_cm], [c_pt], color="#9467bd", s=70, zorder=5, label=f"Selected mode ($\\tilde{{\\nu}}={wavenumber_cm:.0f}\\,\\mathrm{{cm}}^{{-1}}$)")

        ax.axvspan(2500, 4000, color="#ff7f0e", alpha=0.12, label="C–H / O–H stretch band (quantum-frozen)")
        ax.set_xlim(100, 4000)
        ax.set_ylim(-0.04, 1.18)
        ax.set_xlabel(r"Vibrational Wavenumber $\tilde{\nu} = \omega / (2\pi c)$ ($\mathrm{cm}^{-1}$)")
        ax.set_ylabel(r"Normalized Value ($C_{\mathrm{vib}}/k_B$ or $\langle n\rangle$)")
        ax.set_title(f"Quantum Freezing of Intramolecular Vibrations at $T = {temp_K:.0f}\\,\\mathrm{{K}}$")
        ax.legend(loc="upper right", fontsize=8.2)
        ax.grid(True, alpha=0.25)
    else:
        modes = ["C–H / O–H\nStretch\n(~3300 cm⁻¹)", "C=C / C=O\nStretch\n(~1650 cm⁻¹)", " Selected\nMode\n(Slider)", "Bond Angle\nBend\n(~600 cm⁻¹)", "Torsion /\nLibration\n(~180 cm⁻¹)"]
        wns = np.array([3300.0, 1650.0, wavenumber_cm, 600.0, 180.0])
        periods = 1.0 / (c_cm_fs * wns)
        max_dts = periods / 20.0
        colors = ["#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#2ca02c"]

        bars = ax.bar(modes, max_dts, color=colors, alpha=0.85, edgecolor="black", lw=0.8)
        for bar, dt_v, p_v in zip(bars, max_dts, periods):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.15,
                f"$\\Delta t \\approx {dt_v:.2f}$ fs\n($T={p_v:.1f}$ fs)",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )
        ax.set_ylim(0, max(max_dts) * 1.25)
        ax.set_ylabel(r"Stable Verlet Timestep $\Delta t \approx T_{\mathrm{vib}} / 20$ (fs)")
        ax.set_title("MD Timestep Bottleneck Across Intramolecular Degrees of Freedom")
        ax.grid(True, axis="y", alpha=0.25)

    n_occ = 1.0 / (np.exp(min(x_val, 80.0)) - 1.0) if x_val < 80 else 0.0
    speedup = max(1.0, dt_constrained / dt_unconstrained)
    return (
        f"**Selected mode ($\\tilde{{\\nu}} = {wavenumber_cm:.0f}\\,\\mathrm{{cm}}^{{-1}}$ at $T = {temp_K:.0f}\\,\\mathrm{{K}}$):** "
        f"$\\hbar\\omega / k_B T = {x_val:.2f}$ (characteristic temperature $\\Theta_{{\\mathrm{{vib}}}} = {theta_vib:.0f}\\,\\mathrm{{K}}$), "
        f"mean quantum excitation $\\langle n\\rangle = {n_occ:.2e}$. "
        f"Classical vibrational period $T_{{\\mathrm{{vib}}}} = {period_fs:.2f}\\,\\mathrm{{fs}}$ forces an unconstrained MD step "
        f"$\\Delta t \\approx {dt_unconstrained:.2f}\\,\\mathrm{{fs}}$ (potential $\\sim {speedup:.1f}\\times$ timestep increase when constrained)."
    )


def render_stiff_spring_vs_rigid_ensemble(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot the equilibrium bond-angle distribution P(theta) for stiff harmonic springs vs. rigid holonomic constraints."""
    mass_ratio = float(_val(controls.get("mass_ratio_m_over_mB"), 1.0))
    show_mode = str(_val(controls.get("plot_mode"), "distributions"))

    # For a symmetric triatomic A-B-C with outer masses m_A = m_C = m and central mass m_B,
    # define coupling parameter gamma = m / (m + m_B) = r / (1 + r) where r = m / m_B.
    # Notice 0 < gamma < 1; for identical atoms (r = 1), gamma = 1/2, so det M(theta) ~ 1 - (1/4)*cos^2(theta).
    gamma = mass_ratio / (1.0 + mass_ratio)
    theta_deg = np.linspace(1.0, 179.0, 500)
    theta_rad = np.deg2rad(theta_deg)

    # Stiff-spring limit (flexible bonds with k -> infinity, no angular bending potential V(theta)=0):
    # P_spring(theta) = 0.5 * sin(theta)
    p_spring = 0.5 * np.sin(theta_rad)

    # Rigid holonomic bond constraints (sigma_1 = r_12^2 - d^2 = 0, sigma_2 = r_23^2 - d^2 = 0):
    # Metric determinant det M(theta) propto 1 - gamma^2 cos^2(theta)
    metric_factor = np.sqrt(1.0 - (gamma ** 2) * (np.cos(theta_rad) ** 2))
    p_rigid_unnorm = np.sin(theta_rad) * metric_factor
    norm_rigid = np.trapezoid(p_rigid_unnorm, theta_rad)
    p_rigid = p_rigid_unnorm / norm_rigid

    # Fixman compensating potential in units of k_B T:
    # U_Fixman(theta) / (k_B T) = +0.5 * ln(det G) = -0.5 * ln(1 - gamma^2 cos^2(theta))
    # Adding exp(-beta U_Fixman) to the rigid simulation cancels metric_factor and restores P_spring(theta)!
    u_fixman_kT = -0.5 * np.log(1.0 - (gamma ** 2) * (np.cos(theta_rad) ** 2))

    if show_mode == "distributions":
        ax.plot(theta_deg, p_spring, color="#1f77b4", lw=2.3, ls="--", label=r"Stiff-spring limit ($k_{\mathrm{bond}}\to\infty$): $P_{\mathrm{spring}}(\theta) = \frac{1}{2}\sin\theta$")
        ax.plot(
            theta_deg,
            p_rigid,
            color="#d62728",
            lw=2.5,
            label=rf"Holonomic rigid constraint ($\sigma=0$): $P_{{\mathrm{{rigid}}}}(\theta) \propto \sin\theta\sqrt{{1 - \gamma^2\cos^2\theta}}$",
        )
        ax.fill_between(theta_deg, p_spring, p_rigid, color="#9467bd", alpha=0.18, label="Ensemble discrepancy (metric tensor bias)")
        ax.axvline(90.0, color="gray", ls=":", lw=1.2)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, max(np.max(p_rigid), np.max(p_spring)) * 1.18)
        ax.set_xlabel(r"Bond Angle $\theta$ (degrees)")
        ax.set_ylabel(r"Probability Density $P(\theta)$ ($\mathrm{rad}^{-1}$)")
        ax.set_title(rf"Stiff-Spring vs. Rigid-Constraint Bond Angle Ensemble ($m/m_B = {mass_ratio:.2f}, \gamma = {gamma:.3f}$)")
        ax.legend(loc="upper right", fontsize=8.0)
        ax.grid(True, alpha=0.25)
    else:
        ax.plot(theta_deg, metric_factor, color="#2ca02c", lw=2.4, label=r"Metric factor $|\det \mathbf{M}(\theta)|^{1/2} = \sqrt{1 - \gamma^2\cos^2\theta}$")
        ax.plot(theta_deg, p_rigid / np.maximum(p_spring, 1e-8), color="#d62728", lw=2.2, ls="-.", label=r"Density ratio $P_{\mathrm{rigid}}(\theta) / P_{\mathrm{spring}}(\theta)$")
        ax.plot(theta_deg, u_fixman_kT, color="#ff7f0e", lw=2.2, label=r"Fixman potential $U_{\mathrm{Fixman}}(\theta)/k_B T = -\frac{1}{2}\ln(1-\gamma^2\cos^2\theta)$")
        ax.axhline(1.0, color="gray", ls=":", lw=1.2)
        ax.set_xlim(0, 180)
        ax.set_xlabel(r"Bond Angle $\theta$ (degrees)")
        ax.set_ylabel("Dimensionless Factor / Potential ($k_B T$)")
        ax.set_title(rf"Jacobian Metric Factor & Fixman Potential ($m/m_B = {mass_ratio:.2f}, \gamma = {gamma:.3f}$)")
        ax.legend(loc="best", fontsize=8.0)
        ax.grid(True, alpha=0.25)

    ratio_90_to_0 = 1.0 / np.sqrt(1.0 - gamma ** 2)
    peak_ratio = p_rigid[len(theta_deg) // 2] / p_spring[len(theta_deg) // 2]
    return (
        f"**Mass ratio $m/m_B = {mass_ratio:.2f}$ ($\\gamma = \\frac{{m}}{{m+m_B}} = {gamma:.3f}$, $\\gamma^2 = {gamma**2:.3f}$):** "
        f"At $\\theta = 90^\\circ$, $P_{{\\mathrm{{rigid}}}}(90^\\circ)/P_{{\\mathrm{{spring}}}}(90^\\circ) = {peak_ratio:.4f}$. "
        f"Collinear configurations ($\\theta \\to 0^\\circ, 180^\\circ$) are suppressed by $\\sqrt{{1-\\gamma^2}} = {np.sqrt(1.0-gamma**2):.4f}$ "
        f"(Fixman barrier $\\Delta U_{{\\mathrm{{Fixman}}}} = {u_fixman_kT[0]:.3f}\\,k_B T$, relative enhancement at $90^\\circ$ of ${ratio_90_to_0:.3f}\\times$)."
    )


def render_euler_gimbal_vs_quaternion(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare Euler angle coordinate singularity (1/sin(theta) gimbal lock) against smooth unit-quaternion propagation."""
    theta_min_deg = float(_val(controls.get("min_tilt_deg"), 5.0))
    omega_perp = float(_val(controls.get("omega_perp"), 1.0))

    t = np.linspace(-2.0, 2.0, 500)
    theta_min_rad = np.deg2rad(max(theta_min_deg, 0.5))

    # A rigid body passes near the North Pole (z-axis) with minimum tilt angle theta_min at t = 0:
    # Set body trajectory on unit sphere: n_x(t) = sin(theta(t)) cos(phi(t)), etc.
    # For constant angular speed omega_perp passing at distance theta_min from pole:
    # cos(theta(t)) = cos(theta_min) * cos(omega_perp * t) -> near t=0, theta(t) ~ sqrt(theta_min^2 + (omega_perp*t)^2)
    theta_t = np.sqrt(theta_min_rad ** 2 + (omega_perp * t) ** 2)
    theta_t = np.clip(theta_t, 1e-4, np.pi - 1e-4)

    # Euler precession rate: d(phi)/dt = omega_perp * theta_min / sin^2(theta) or ~ omega_perp / sin(theta(t))
    dphi_dt = omega_perp / np.sin(theta_t)

    # Unit quaternion components q = (q0, q1, q2, q3) evolving smoothly under dQ/dt = 0.5 * W(omega) * Q
    # |dQ/dt| = 0.5 * |omega| = const!
    half_angle = 0.5 * omega_perp * t
    q0 = np.cos(half_angle) * np.cos(0.5 * theta_min_rad)
    q1 = np.sin(half_angle) * np.cos(0.5 * theta_min_rad)
    q2 = np.cos(half_angle) * np.sin(0.5 * theta_min_rad)
    q3 = np.sin(half_angle) * np.sin(0.5 * theta_min_rad)
    dq_norm = 0.5 * omega_perp * np.ones_like(t)

    ax.plot(t, dphi_dt, color="#d62728", lw=2.5, label=r"Euler precession rate $|\dot{\phi}(t)| \sim \omega_\perp / \sin\theta(t)$")
    ax.plot(t, dq_norm, color="#1f77b4", lw=2.3, label=r"Quaternion derivative norm $\|\dot{\mathbf{Q}}(t)\| = \frac{1}{2}\|\boldsymbol{\omega}\|$")
    ax.plot(t, q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2, color="#2ca02c", lw=1.8, ls="--", label=r"Quaternion norm $\sum_{m=0}^3 q_m^2 = 1$")
    ax.plot(t, np.rad2deg(theta_t) / 90.0, color="#9467bd", lw=1.6, ls=":", label=r"Normalized tilt angle $\theta(t) / 90^\circ$")

    peak_dphi = omega_perp / np.sin(theta_min_rad)
    ax.scatter([0.0], [peak_dphi], color="#d62728", s=65, zorder=5)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.2, min(max(peak_dphi * 1.15, 4.0), 125.0))
    ax.set_xlabel(r"Time $t$ around North-Pole Passage ($\mathrm{ps}$)")
    ax.set_ylabel(r"Rate of Change ($\mathrm{ps}^{-1}$) / Norm")
    ax.set_title(rf"Euler Gimbal-Lock Singularity vs. Unit Quaternion Smoothness ($\theta_{{\min}} = {theta_min_deg:.1f}^\circ$)")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, alpha=0.25)

    return (
        f"**At closest approach $\\theta_{{\\min}} = {theta_min_deg:.1f}^\\circ$ ($\\sin\\theta_{{\\min}} = {np.sin(theta_min_rad):.4f}$):** "
        f"Euler azimuthal rate spikes to $|\\dot{{\\phi}}_{{\\max}}| = \\omega_\\perp / \\sin\\theta_{{\\min}} = {peak_dphi:.2f}\\,\\mathrm{{ps}}^{{-1}}$ "
        f"(requiring $\\Delta t \\propto \\sin\\theta_{{\\min}}$ to avoid integrator blow-up), whereas the 4-component unit quaternion "
        f"rate remains strictly constant at $\\|\\dot{{\\mathbf{{Q}}}}\\| = \\frac{{1}}{{2}}\\omega_\\perp = {0.5*omega_perp:.2f}\\,\\mathrm{{ps}}^{{-1}}$ everywhere on $S^3$."
    )


def render_shake_iteration_tracer(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Trace iterative SHAKE constraint relaxation on a coupled triatomic molecule (1-2-3)."""
    n_iter = int(_val(controls.get("shake_iterations"), 4))
    perturbation = float(_val(controls.get("initial_distortion"), 0.25))
    view_type = str(_val(controls.get("panel_mode"), "residuals"))

    # Reference geometry at time t: A-B-C molecule with bond length d = 1.0 A and angle 104.5 deg
    d = 1.0
    half_ang = np.deg2rad(104.5 / 2.0)
    r1_old = np.array([-d * np.sin(half_ang), -d * np.cos(half_ang)])
    r2_old = np.array([0.0, 0.0])
    r3_old = np.array([+d * np.sin(half_ang), -d * np.cos(half_ang)])

    # Equal masses for clear coupling demonstration (or m1=m2=m3=1)
    inv_m1, inv_m2, inv_m3 = 1.0, 1.0, 1.0

    # Unconstrained Verlet step r^unc(t+h) stretches bond 1-2 and compresses/stretches bond 2-3 asymmetrically
    r1 = r1_old + np.array([-0.85 * perturbation, -0.55 * perturbation])
    r2 = r2_old + np.array([0.15 * perturbation, 0.70 * perturbation])
    r3 = r3_old + np.array([0.55 * perturbation, -0.80 * perturbation])

    r12_old = r1_old - r2_old
    r23_old = r2_old - r3_old

    hist_s1 = [abs(np.dot(r1 - r2, r1 - r2) - d ** 2)]
    hist_s2 = [abs(np.dot(r2 - r3, r2 - r3) - d ** 2)]
    snapshots = [(r1.copy(), r2.copy(), r3.copy())]

    max_sweeps = 15
    for _ in range(max_sweeps):
        # Constraint 1: bond (1, 2) updated along r12_old
        r12_cur = r1 - r2
        sig1 = np.dot(r12_cur, r12_cur) - d ** 2
        denom1 = 2.0 * (inv_m1 + inv_m2) * np.dot(r12_cur, r12_old)
        g1 = sig1 / denom1
        r1 = r1 - inv_m1 * g1 * r12_old
        r2 = r2 + inv_m2 * g1 * r12_old

        # Constraint 2: bond (2, 3) updated along r23_old (perturbs atom 2 and hence bond 1-2!)
        r23_cur = r2 - r3
        sig2 = np.dot(r23_cur, r23_cur) - d ** 2
        denom2 = 2.0 * (inv_m2 + inv_m3) * np.dot(r23_cur, r23_old)
        g2 = sig2 / denom2
        r2 = r2 - inv_m2 * g2 * r23_old
        r3 = r3 + inv_m3 * g2 * r23_old

        hist_s1.append(abs(np.dot(r1 - r2, r1 - r2) - d ** 2))
        hist_s2.append(abs(np.dot(r2 - r3, r2 - r3) - d ** 2))
        snapshots.append((r1.copy(), r2.copy(), r3.copy()))

    idx = int(np.clip(n_iter, 0, max_sweeps))
    r1_m, r2_m, r3_m = snapshots[idx]
    r1_0, r2_0, r3_0 = snapshots[0]

    if view_type == "residuals":
        sweeps = np.arange(0, max_sweeps + 1)
        ax.semilogy(sweeps, np.maximum(hist_s1, 1e-16), "o-", color="#1f77b4", lw=2.2, label=r"Bond 1–2 residual $|\sigma_{12}^{(m)}| = |r_{12}^2 - d^2|$")
        ax.semilogy(sweeps, np.maximum(hist_s2, 1e-16), "s--", color="#d62728", lw=2.0, label=r"Bond 2–3 residual $|\sigma_{23}^{(m)}| = |r_{23}^2 - d^2|$")
        ax.axvline(idx, color="#9467bd", ls=":", lw=2.0, label=f"Selected SHAKE sweep $m = {idx}$")
        ax.axhline(1e-8, color="#2ca02c", ls="-.", lw=1.5, label=r"Convergence tolerance $\varepsilon = 10^{-8}\,\text{Å}^2$")
        ax.set_xlim(0, max_sweeps)
        ax.set_ylim(1e-14, 2.0)
        ax.set_xlabel("SHAKE Gauss–Seidel Sweep Number $m$")
        ax.set_ylabel(r"Constraint Residual $|\sigma_k^{(m)}|$ ($\text{Å}^2$)")
        ax.set_title("Exponential Convergence of Coupled Bond Constraints in SHAKE")
        ax.legend(loc="upper right", fontsize=8.0)
        ax.grid(True, which="both", alpha=0.25)
    else:
        # Plot molecular geometry in 2D plane
        pts_old = np.array([r1_old, r2_old, r3_old])
        pts_unc = np.array([r1_0, r2_0, r3_0])
        pts_cur = np.array([r1_m, r2_m, r3_m])

        ax.plot(pts_old[:, 0], pts_old[:, 1], "o:", color="gray", lw=1.6, ms=7, label=r"Old configuration $\mathbf{R}(t)$ (reference directions)")
        ax.plot(pts_unc[:, 0], pts_unc[:, 1], "^--", color="#ff7f0e", lw=1.8, ms=8, label=r"Unconstrained Verlet step $\mathbf{R}^{\mathrm{unc}}(t+h)$ ($m=0$)")
        ax.plot(pts_cur[:, 0], pts_cur[:, 1], "o-", color="#1f77b4", lw=2.6, ms=9, label=rf"SHAKE iterate $\mathbf{{R}}^{{({idx})}}(t+h)$")

        for p_u, p_c in zip(pts_unc, pts_cur):
            if np.linalg.norm(p_c - p_u) > 1e-3:
                ax.annotate("", xy=p_c, xytext=p_u, arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel(r"$x$ coordinate ($\text{Å}$)")
        ax.set_ylabel(r"$y$ coordinate ($\text{Å}$)")
        ax.set_title(f"Triatomic Geometry Projection Along Old Bond Directions (Sweep $m = {idx}$)")
        ax.legend(loc="best", fontsize=7.8)
        ax.grid(True, alpha=0.25)

    d12_cur = np.linalg.norm(r1_m - r2_m)
    d23_cur = np.linalg.norm(r2_m - r3_m)
    return (
        f"**After $m = {idx}$ SHAKE sweeps:** "
        f"$r_{{12}}^{{({idx})}} = {d12_cur:.6f}\\,\\text{{Å}}$ ($|\\sigma_{{12}}| = {hist_s1[idx]:.2e}\\,\\text{{Å}}^2$) and "
        f"$r_{{23}}^{{({idx})}} = {d23_cur:.6f}\\,\\text{{Å}}$ ($|\\sigma_{{23}}| = {hist_s2[idx]:.2e}\\,\\text{{Å}}^2$). "
        f"Note that after each sweep, enforcing constraint 2–3 exacts a small kick on shared atom 2 that slightly re-perturbs bond 1–2, requiring cyclic iteration."
    )
