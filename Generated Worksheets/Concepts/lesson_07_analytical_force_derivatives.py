"""Interactive concept visualizations for Lesson 07: Analytical Force Derivatives."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from engine.plot_style import COLORS


def render_bond_stretch_forces(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render 2-body harmonic bond-stretching potential E(r12), scalar derivative, and 3D force vectors."""
    r12 = float(controls.get("r12", 1.80))
    r0 = float(controls.get("r0", 1.50))
    kr = float(controls.get("kr", 15.0))
    view_mode = str(controls.get("view_mode", "potential_and_force"))

    r_grid = np.linspace(0.8, 2.5, 300)
    e_grid = kr * (r_grid - r0) ** 2
    f_mag_signed = -2.0 * kr * (r_grid - r0)  # -dE/dr12

    e_val = kr * (r12 - r0) ** 2
    dE_dr = 2.0 * kr * (r12 - r0)
    # Vector r12 = r1 - r2; put atom 2 at (-r12/2, 0), atom 1 at (+r12/2, 0)
    # F1 = -(dE/dr12) * r12_hat, F2 = -F1
    f1_x = -dE_dr
    f2_x = +dE_dr

    if view_mode == "geometry_vectors":
        ax.axhline(0.0, color=COLORS["grid"], linestyle="--", linewidth=1.0, zorder=1)
        x2, x1 = -0.5 * r12, 0.5 * r12
        ax.plot([x2, x1], [0.0, 0.0], color=COLORS["muted"], linewidth=3.5, zorder=2, label="Bond $r_{12}$")
        ax.scatter([x2, x1], [0.0, 0.0], s=360, color=[COLORS["secondary"], COLORS["primary"]], edgecolors="white", linewidth=2.0, zorder=4)
        ax.text(x2, 0.25, "Atom 2\n$\\mathbf{r}_2$", ha="center", va="bottom", fontweight="bold", color=COLORS["secondary"])
        ax.text(x1, 0.25, "Atom 1\n$\\mathbf{r}_1$", ha="center", va="bottom", fontweight="bold", color=COLORS["primary"])

        # Draw unit vector r12_hat and r21_hat
        ax.annotate(
            "",
            xy=(x2 + 0.38, -0.25),
            xytext=(x2, -0.25),
            arrowprops=dict(arrowstyle="->", color=COLORS["text"], lw=1.5),
        )
        ax.text(x2 + 0.19, -0.40, "$\\hat{\\mathbf{r}}_{12} = (\\mathbf{r}_1-\\mathbf{r}_2)/r_{12}$", ha="center", fontsize=8.5)

        scale = 0.035
        if abs(f1_x) > 1e-3:
            ax.quiver(
                [x1, x2],
                [0.0, 0.0],
                [f1_x * scale, f2_x * scale],
                [0.0, 0.0],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=[COLORS["danger"], COLORS["good"]],
                width=0.014,
                zorder=5,
            )
            ax.text(x1 + 0.5 * f1_x * scale, -0.22, f"$\\mathbf{{F}}_1 = {f1_x:+.1f}\\,\\hat{{\\mathbf{{r}}}}_{{12}}$", ha="center", color=COLORS["danger"], fontweight="bold")
            ax.text(x2 + 0.5 * f2_x * scale, -0.22, f"$\\mathbf{{F}}_2 = {f2_x:+.1f}\\,\\hat{{\\mathbf{{r}}}}_{{12}}$", ha="center", color=COLORS["good"], fontweight="bold")

        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-0.75, 0.75)
        ax.set_xlabel("Position along bond axis $x$ (Å)")
        ax.set_yticks([])
        ax.set_title(f"2-Body Bond Stretch Force Geometry ($r_{{12}}={r12:.2f}$ Å, $r_0={r0:.2f}$ Å)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
    else:
        ax.plot(r_grid, e_grid, color=COLORS["primary"], linewidth=2.3, label="$E_{\\mathrm{bond}}(r_{12}) = K_r(r_{12}-r_0)^2$")
        ax.plot(r_grid, f_mag_signed, color=COLORS["accent"], linestyle="--", linewidth=2.0, label="$-dE/dr_{12} = -2K_r(r_{12}-r_0)$")
        ax.axhline(0.0, color=COLORS["muted"], linewidth=0.9, linestyle=":")
        ax.axvline(r0, color=COLORS["good"], linewidth=1.2, linestyle=":", label=f"Equilibrium $r_0 = {r0:.2f}$ Å")
        ax.scatter([r12], [e_val], color=COLORS["primary"], s=65, zorder=5)
        ax.scatter([r12], [-dE_dr], color=COLORS["accent"], s=65, zorder=5)
        ax.set_xlim(0.8, 2.5)
        y_lim = max(15.0, float(np.max(np.abs(f_mag_signed))) * 0.85)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_xlabel("Interatomic Distance $r_{12} = |\\mathbf{r}_1 - \\mathbf{r}_2|$ (Å)")
        ax.set_ylabel("Energy (eV) / Force Projection (eV/Å)")
        ax.set_title("Harmonic Bond Energy $E(r_{12})$ and Radial Force $-dE/dr_{12}$")
        ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    regime = "stretched ($r_{12} > r_0$, attractive restoring force)" if r12 > r0 + 1e-4 else (
        "compressed ($r_{12} < r_0$, repulsive restoring force)" if r12 < r0 - 1e-4 else "at mechanical equilibrium ($r_{12} = r_0$, zero force)"
    )
    return (
        f"**State:** $r_{{12}} = {r12:.2f}\\text{{ Å}}$ ({regime}), $E_{{\\text{{bond}}}} = {e_val:.2f}\\text{{ eV}}$, "
        f"$\\frac{{dE}}{{dr_{{12}}}} = {dE_dr:+.2f}\\text{{ eV/Å}}$. "
        f"By the chain rule, $\\mathbf{{F}}_1 = -\\frac{{dE}}{{dr_{{12}}}}\\hat{{\\mathbf{{r}}}}_{{12}} = ({f1_x:+.2f}\\text{{ eV/Å}})\\hat{{\\mathbf{{r}}}}_{{12}}$ "
        f"and $\\mathbf{{F}}_2 = -\\mathbf{{F}}_1$, guaranteeing $\\mathbf{{F}}_1 + \\mathbf{{F}}_2 = \\mathbf{{0}}$."
    )


def render_three_body_angle_forces(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render a 3-atom angle (1-2-3) showing exact analytical forces F1, F2, F3 and orthogonality F1 perp r12, F3 perp r32."""
    theta_deg = float(controls.get("theta_deg", 104.5))
    theta0_deg = float(controls.get("theta0_deg", 109.5))
    ratio_r12_r32 = float(controls.get("ratio_r12_r32", 1.0))
    k_theta = float(controls.get("k_theta", 4.0))

    theta = np.deg2rad(np.clip(theta_deg, 15.0, 175.0))
    theta0 = np.deg2rad(theta0_deg)
    r32 = 1.25
    r12 = r32 * ratio_r12_r32

    # Central vertex atom 2 at origin (0, 0)
    r2_vec = np.array([0.0, 0.0])
    # Place bond 2->3 at symmetric angle -theta/2 from +y axis, and 2->1 at +theta/2
    half_th = 0.5 * theta
    r12_hat = np.array([-np.sin(half_th), np.cos(half_th)])
    r32_hat = np.array([+np.sin(half_th), np.cos(half_th)])
    r1_vec = r2_vec + r12 * r12_hat
    r3_vec = r2_vec + r32 * r32_hat

    cos_th = float(np.dot(r12_hat, r32_hat))
    sin_th = max(1e-6, float(np.sin(theta)))

    # Analytical angle gradients:
    # grad_r1(theta) = -1/(r12 * sin(theta)) * (r32_hat - cos(theta)*r12_hat)
    # grad_r3(theta) = -1/(r32 * sin(theta)) * (r12_hat - cos(theta)*r32_hat)
    grad_r1_theta = -(1.0 / (r12 * sin_th)) * (r32_hat - cos_th * r12_hat)
    grad_r3_theta = -(1.0 / (r32 * sin_th)) * (r12_hat - cos_th * r32_hat)
    grad_r2_theta = -(grad_r1_theta + grad_r3_theta)

    dE_dtheta = 2.0 * k_theta * (theta - theta0)
    F1 = -dE_dtheta * grad_r1_theta
    F3 = -dE_dtheta * grad_r3_theta
    F2 = -(F1 + F3)

    # Plot bonds
    ax.plot([r2_vec[0], r1_vec[0]], [r2_vec[1], r1_vec[1]], color=COLORS["primary"], linewidth=3.2, zorder=2, label="Bond $\\mathbf{r}_{12} = \\mathbf{r}_1 - \\mathbf{r}_2$")
    ax.plot([r2_vec[0], r3_vec[0]], [r2_vec[1], r3_vec[1]], color=COLORS["secondary"], linewidth=3.2, zorder=2, label="Bond $\\mathbf{r}_{32} = \\mathbf{r}_3 - \\mathbf{r}_2$")

    # Draw dashed extension lines to show perpendicularity clearly
    ax.plot([0.0, 1.35 * r1_vec[0]], [0.0, 1.35 * r1_vec[1]], color=COLORS["primary"], linestyle=":", linewidth=1.0, alpha=0.6)
    ax.plot([0.0, 1.35 * r3_vec[0]], [0.0, 1.35 * r3_vec[1]], color=COLORS["secondary"], linestyle=":", linewidth=1.0, alpha=0.6)

    # Plot atoms
    ax.scatter(
        [r1_vec[0], r2_vec[0], r3_vec[0]],
        [r1_vec[1], r2_vec[1], r3_vec[1]],
        s=[280, 340, 280],
        color=[COLORS["primary"], COLORS["total"], COLORS["secondary"]],
        edgecolors="white",
        linewidth=2.0,
        zorder=4,
    )
    ax.text(r1_vec[0] - 0.18, r1_vec[1] + 0.14, "Atom 1", fontweight="bold", color=COLORS["primary"], ha="right")
    ax.text(r2_vec[0], r2_vec[1] - 0.24, "Atom 2 (vertex)", fontweight="bold", color=COLORS["total"], ha="center")
    ax.text(r3_vec[0] + 0.18, r3_vec[1] + 0.14, "Atom 3", fontweight="bold", color=COLORS["secondary"], ha="left")

    # Angle arc
    arc_deg1 = 90.0 - 0.5 * theta_deg
    arc_deg2 = 90.0 + 0.5 * theta_deg
    arc = Arc((0.0, 0.0), 0.75, 0.75, angle=0.0, theta1=arc_deg1, theta2=arc_deg2, color=COLORS["highlight"], linewidth=2.0)
    ax.add_patch(arc)
    ax.text(0.0, 0.46, f"$\\theta_{{123}} = {theta_deg:.1f}^\\circ$", ha="center", va="bottom", fontsize=9.0, fontweight="bold", color=COLORS["text"])

    # Plot force vectors
    vec_scale = 0.22
    for pos_vec, f_vec, col, lbl in [
        (r1_vec, F1, COLORS["danger"], "$\\mathbf{F}_1$"),
        (r3_vec, F3, COLORS["accent"], "$\\mathbf{F}_3$"),
        (r2_vec, F2, COLORS["good"], "$\\mathbf{F}_2 = -(\\mathbf{F}_1+\\mathbf{F}_3)$"),
    ]:
        if np.linalg.norm(f_vec) > 1e-4:
            ax.quiver(
                [pos_vec[0]],
                [pos_vec[1]],
                [f_vec[0] * vec_scale],
                [f_vec[1] * vec_scale],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=col,
                width=0.013,
                zorder=5,
                label=f"{lbl} ($|F|={np.linalg.norm(f_vec):.2f}$)",
            )

    dot1 = float(np.dot(F1, r12_hat))
    dot3 = float(np.dot(F3, r32_hat))
    f_sum = float(np.linalg.norm(F1 + F2 + F3))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-0.95, 2.15)
    ax.set_xlabel("$x$ coordinate (Å)")
    ax.set_ylabel("$y$ coordinate (Å)")
    ax.set_title(f"3-Body Angle Bending Forces ($\\theta_0={theta0_deg:.1f}^\\circ$, $r_{{12}}/r_{{32}}={ratio_r12_r32:.2f}$)")
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Invariants verified:** $\\mathbf{{F}}_1 \\cdot \\hat{{\\mathbf{{r}}}}_{{12}} = {dot1:.2e}$ (strictly $\\perp \\mathbf{{r}}_{{12}}$), "
        f"$\\mathbf{{F}}_3 \\cdot \\hat{{\\mathbf{{r}}}}_{{32}} = {dot3:.2e}$ (strictly $\\perp \\mathbf{{r}}_{{32}}$), "
        f"and $|\\mathbf{{F}}_1 + \\mathbf{{F}}_2 + \\mathbf{{F}}_3| = {f_sum:.2e}$. "
        f"Because $|\\nabla_{{\\mathbf{{r}}_1}}\\theta_{{123}}| = 1/r_{{12}}$, $|\\mathbf{{F}}_1|/|\\mathbf{{F}}_3| = r_{{32}}/r_{{12}} = {1.0/ratio_r12_r32:.2f}$."
    )


def render_opls_dihedral_torsion(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render the OPLS 4-term Fourier dihedral potential E(phi) and analytical torque -dE/dphi across [-180, 180] deg."""
    v1 = float(controls.get("v1", 1.5))
    v2 = float(controls.get("v2", -0.5))
    v3 = float(controls.get("v3", 3.0))
    phi_deg = float(controls.get("phi_deg", 60.0))

    phi_grid_deg = np.linspace(-180.0, 180.0, 400)
    phi_grid = np.deg2rad(phi_grid_deg)

    # OPLS convention: E(phi) = 0.5 * [V1*(1 + cos(phi)) + V2*(1 - cos(2*phi)) + V3*(1 + cos(3*phi))]
    e_grid = 0.5 * (
        v1 * (1.0 + np.cos(phi_grid))
        + v2 * (1.0 - np.cos(2.0 * phi_grid))
        + v3 * (1.0 + np.cos(3.0 * phi_grid))
    )
    # dE/dphi = 0.5 * [-V1*sin(phi) + 2*V2*sin(2*phi) - 3*V3*sin(3*phi)]
    dE_dphi_grid = 0.5 * (
        -v1 * np.sin(phi_grid)
        + 2.0 * v2 * np.sin(2.0 * phi_grid)
        - 3.0 * v3 * np.sin(3.0 * phi_grid)
    )

    phi_rad = np.deg2rad(phi_deg)
    e_curr = 0.5 * (
        v1 * (1.0 + np.cos(phi_rad))
        + v2 * (1.0 - np.cos(2.0 * phi_rad))
        + v3 * (1.0 + np.cos(3.0 * phi_rad))
    )
    dE_curr = 0.5 * (
        -v1 * np.sin(phi_rad)
        + 2.0 * v2 * np.sin(2.0 * phi_rad)
        - 3.0 * v3 * np.sin(3.0 * phi_rad)
    )

    # Blondel-Karplus normal force magnitude for unit bond lengths and 109.5 deg bond angles
    # |nabla_r1 phi| = r23 / |n123|^2 = 1 / (r12 * sin(theta123))
    sin_th_bond = np.sin(np.deg2rad(109.5))
    bk_grad_norm = 1.0 / sin_th_bond  # finite and constant across all phi!
    f1_mag = abs(dE_curr) * bk_grad_norm

    ax.plot(phi_grid_deg, e_grid, color=COLORS["primary"], linewidth=2.4, label="$E_{\\mathrm{torsion}}(\\phi)$ (kcal/mol)")
    ax.plot(phi_grid_deg, -dE_dphi_grid, color=COLORS["accent"], linestyle="--", linewidth=1.9, label="Torque $-dE/d\\phi$ (kcal/mol/rad)")
    ax.axhline(0.0, color=COLORS["muted"], linestyle=":", linewidth=0.9)
    ax.axvline(phi_deg, color=COLORS["good"], linestyle="-.", linewidth=1.3, label=f"Current $\\phi = {phi_deg:.0f}^\\circ$")
    ax.scatter([phi_deg], [e_curr], color=COLORS["primary"], s=70, zorder=5)
    ax.scatter([phi_deg], [-dE_curr], color=COLORS["accent"], s=70, zorder=5)

    # Highlight planar cis/trans configurations where sin(phi) = 0
    for p_sing in (-180.0, 0.0, 180.0):
        ax.axvline(p_sing, color=COLORS["quaternary"], linestyle=":", linewidth=1.0, alpha=0.55)

    ax.set_xlim(-180.0, 180.0)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_xlabel("Dihedral Torsion Angle $\\phi_{1234}$ (degrees)")
    ax.set_ylabel("Energy / Torque")
    ax.set_title(f"OPLS Fourier Torsion Potential ($V_1={v1:.1f}, V_2={v2:.1f}, V_3={v3:.1f}$)")
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**At $\\phi = {phi_deg:.1f}^\\circ$:** $E(\\phi) = {e_curr:.2f}\\text{{ kcal/mol}}$, "
        f"$\\frac{{dE}}{{d\\phi}} = {dE_curr:+.2f}\\text{{ kcal/(mol}}\\cdot\\text{{rad)}}$. "
        f"Via Blondel–Karplus, $\\nabla_{{\\mathbf{{r}}_1}}\\phi = \\frac{{r_{{23}}}}{{n_{{123}}^2}}\\mathbf{{n}}_{{123}}$ depends only on the $(1,2,3)$ plane normal "
        f"($|\\nabla_{{\\mathbf{{r}}_1}}\\phi| = {bk_grad_norm:.3f}\\text{{ Å}}^{{-1}}$), completely avoiding the $1/\\sin\\phi$ singularity at $\\phi = 0^\\circ, \\pm 180^\\circ$ ($|\\mathbf{{F}}_1| = {f1_mag:.2f}$)."
    )


def render_finite_difference_verification(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render log-log relative force error vs finite-difference step delta, showing O(delta) vs O(delta^2) and roundoff floor."""
    log10_delta = float(controls.get("log10_delta", -5.0))
    curv_scale = float(controls.get("curv_scale", 1.0))
    eps_mach = 2.220446049250313e-16

    deltas = np.logspace(-12.0, -1.0, 250)
    # Test function representing an anharmonic / angle-bending energy E(x) around x0 = 1.2
    # E(x) = curv_scale * exp(-1.5 * x) * (x - 1.0)^2 + 2.0 * x^3
    # Analytical derivatives at x0:
    e0 = 1.0 * curv_scale
    f_exact = 3.5 * curv_scale
    d2E = 6.0 * curv_scale
    d3E = 8.4 * curv_scale

    # Truncation + IEEE-754 floating-point cancellation error model (plus exact IEEE simulation)
    x0 = 1.15
    def _energy_fn(x: np.ndarray | float) -> np.ndarray | float:
        return curv_scale * (np.exp(0.8 * (x - x0)) + 0.5 * np.sin(2.5 * x))

    f_analytic = -curv_scale * (0.8 * np.exp(0.0) + 1.25 * np.cos(2.5 * x0))
    f_fwd_num = -(_energy_fn(x0 + deltas) - _energy_fn(x0)) / deltas
    f_cen_num = -(_energy_fn(x0 + deltas) - _energy_fn(x0 - deltas)) / (2.0 * deltas)

    err_fwd = np.maximum(1e-16, np.abs(f_fwd_num - f_analytic) / abs(f_analytic))
    err_cen = np.maximum(1e-16, np.abs(f_cen_num - f_analytic) / abs(f_analytic))

    delta_sel = 10.0 ** log10_delta
    f_fwd_sel = float(-(_energy_fn(x0 + delta_sel) - _energy_fn(x0)) / delta_sel)
    f_cen_sel = float(-(_energy_fn(x0 + delta_sel) - _energy_fn(x0 - delta_sel)) / (2.0 * delta_sel))
    err_fwd_sel = max(1e-16, abs(f_fwd_sel - f_analytic) / abs(f_analytic))
    err_cen_sel = max(1e-16, abs(f_cen_sel - f_analytic) / abs(f_analytic))

    ax.loglog(deltas, err_fwd, color=COLORS["accent"], linewidth=2.0, label="Forward diff: $\\mathcal{O}(\\delta) + \\mathcal{O}(\\epsilon_{\\mathrm{mach}}/\\delta)$")
    ax.loglog(deltas, err_cen, color=COLORS["primary"], linewidth=2.2, label="Central diff: $\\mathcal{O}(\\delta^2) + \\mathcal{O}(\\epsilon_{\\mathrm{mach}}/\\delta)$")
    ax.axvline(delta_sel, color=COLORS["good"], linestyle="--", linewidth=1.4, label=f"Selected $\\delta = 10^{{{log10_delta:.1f}}}$ Å")
    ax.scatter([delta_sel, delta_sel], [err_fwd_sel, err_cen_sel], color=[COLORS["accent"], COLORS["primary"]], s=65, zorder=5)

    # Optimal step markers: delta_opt_fwd ~ sqrt(eps) ~ 1.5e-8, delta_opt_cen ~ eps^(1/3) ~ 6e-6
    ax.axvline(6.0e-6, color=COLORS["primary"], linestyle=":", linewidth=1.1, alpha=0.7)
    ax.text(6.0e-6, 2e-13, "Optimal central\n$\\delta^* \\sim \\epsilon_{\\mathrm{mach}}^{1/3} \\approx 6\\times 10^{-6}$", fontsize=8.0, ha="center", color=COLORS["primary"])

    ax.set_xlim(1e-12, 1e-1)
    ax.set_ylim(1e-14, 1e0)
    ax.set_xlabel("Finite-Difference Step Size $\\delta$ (Å)")
    ax.set_ylabel("Relative Force Error $|F_{\\mathrm{FD}} - F_{\\mathrm{analytic}}| / |F_{\\mathrm{analytic}}|$")
    ax.set_title("Finite-Difference Gradient Verification V-Curve (IEEE-754 Double Precision)")
    ax.grid(True, which="both", color=COLORS["grid"], linestyle="--", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    regime = (
        "dominated by Taylor truncation error ($\\propto \\delta^2$ for central difference)"
        if log10_delta > -5.0
        else (
            "dominated by IEEE-754 floating-point cancellation error ($\\propto \\epsilon_{\\text{mach}}/\\delta$)"
            if log10_delta < -6.2
            else "near the sweet-spot minimum where truncation and roundoff balance"
        )
    )
    return (
        f"**At $\\delta = 10^{{{log10_delta:.1f}}}\\text{{ Å}}$:** Forward-difference relative error $= {err_fwd_sel:.2e}$, "
        f"Central-difference relative error $= {err_cen_sel:.2e}$ ({regime})."
    )
