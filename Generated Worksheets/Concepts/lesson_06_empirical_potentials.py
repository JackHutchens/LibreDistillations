"""Interactive concept visuals for Lesson 06: Interatomic & Empirical Potentials."""

from __future__ import annotations
from typing import Any, Mapping
import matplotlib.pyplot as plt
import numpy as np

from engine.plot_style import COLORS


def _val(ctrl: Any, default: Any = 0.0) -> Any:
    if hasattr(ctrl, "value"):
        return ctrl.value
    return ctrl if ctrl is not None else default


def render_lj_vs_morse(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare Lennard-Jones 12-6 and Morse potentials (or forces) matched at equilibrium r0 and well depth eps."""
    alpha_r0 = float(_val(controls.get("alpha_r0"), 6.0))
    show_force = bool(int(_val(controls.get("show_force"), 0)))

    # Reduced distance u = r / r0 where r0 = 2^(1/6) sigma for LJ
    u = np.linspace(0.72, 2.50, 400)

    # In terms of u = r / r_min, LJ 12-6 is V_LJ / eps = u^(-12) - 2 * u^(-6)
    v_lj = u ** (-12) - 2.0 * (u ** (-6))
    f_lj = 12.0 * (u ** (-13) - u ** (-7))  # F * r0 / eps = -d(V/eps)/du

    # Morse potential: V_M / eps = exp(-2 * alpha_r0 * (u - 1)) - 2 * exp(-alpha_r0 * (u - 1))
    exp1 = np.exp(-alpha_r0 * (u - 1.0))
    exp2 = np.exp(-2.0 * alpha_r0 * (u - 1.0))
    v_morse = exp2 - 2.0 * exp1
    f_morse = 2.0 * alpha_r0 * (exp2 - exp1)

    # Curvature k * r0^2 / eps at u = 1: for LJ it is 72; for Morse it is 2 * (alpha_r0)^2
    k_lj = 72.0
    k_morse = 2.0 * (alpha_r0**2)

    if not show_force:
        ax.plot(
            u,
            v_lj,
            color=COLORS["primary"],
            linewidth=2.4,
            label=r"Lennard-Jones 12-6 ($k_{\mathrm{LJ}} r_0^2/\varepsilon = 72$)",
        )
        ax.plot(
            u,
            v_morse,
            color=COLORS["accent"],
            linewidth=2.4,
            linestyle="--",
            label=f"Morse ($\\alpha r_0 = {alpha_r0:.1f}$, $k_M r_0^2/\\varepsilon = {k_morse:.1f}$)",
        )
        ax.axhline(0.0, color=COLORS["muted"], linestyle=":", linewidth=1.1)
        ax.axvline(1.0, color=COLORS["good"], linestyle=":", linewidth=1.3, label=r"Equilibrium $r / r_0 = 1.0$")
        ax.scatter([1.0], [-1.0], color=COLORS["good"], s=75, zorder=5)
        ax.set_ylim(-1.15, 2.0)
        ax.set_ylabel(r"Reduced Potential $V(r) / \varepsilon$")
        ax.set_title(f"Lennard-Jones 12-6 vs. Morse Potential ($\\alpha r_0 = {alpha_r0:.1f}$)")
    else:
        ax.plot(
            u,
            f_lj,
            color=COLORS["primary"],
            linewidth=2.4,
            label="LJ 12-6 Force $F(r) = -dV/dr$",
        )
        ax.plot(
            u,
            f_morse,
            color=COLORS["accent"],
            linewidth=2.4,
            linestyle="--",
            label=f"Morse Force ($\\alpha r_0 = {alpha_r0:.1f}$)",
        )
        ax.axhline(0.0, color=COLORS["muted"], linestyle=":", linewidth=1.1)
        ax.axvline(1.0, color=COLORS["good"], linestyle=":", linewidth=1.3, label=r"Zero-Force Equilibrium $r = r_0$")
        ax.scatter([1.0], [0.0], color=COLORS["good"], s=75, zorder=5)
        ax.set_ylim(-6.0, 10.0)
        ax.set_ylabel(r"Reduced Force $F(r) \cdot r_0 / \varepsilon$")
        ax.set_title(f"Interatomic Force Comparison ($F = -dV/dr$, $\\alpha r_0 = {alpha_r0:.1f}$)")

    ax.set_xlim(0.75, 2.40)
    ax.set_xlabel(r"Normalized Separation $r / r_0$ ($r_0 = 2^{1/6}\sigma \approx 1.122\sigma$)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Curvature (Bulk Modulus) Comparison at $r = r_0$:** LJ 12-6 has a fixed dimensionless curvature "
        f"$V''(r_0) r_0^2 / \\varepsilon = 72$, which corresponds to $\\alpha r_0 = 6.0$. "
        f"Your selected Morse width parameter $\\alpha r_0 = {alpha_r0:.1f}$ gives curvature **$2(\\alpha r_0)^2 = {k_morse:.1f}$** "
        f"({'softer, wider well' if k_morse < k_lj else ('identical curvature at minimum' if abs(k_morse - k_lj) < 1e-3 else 'stiffer, narrower well')} than LJ), "
        f"allowing Morse to fit lattice constant $r_0$, cohesive energy $\\varepsilon$, and bulk modulus $B$ independently!"
    )


def render_eam_vs_pair_metals(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot cohesive energy per atom or bond strength vs coordination number Z for Pair (-Z) vs EAM (-Z^p)."""
    exponent_p = float(_val(controls.get("exponent_p"), 0.5))
    plot_mode = str(_val(controls.get("plot_mode"), "cohesive"))

    z_vals = np.arange(1, 13, dtype=float)
    z_smooth = np.linspace(1.0, 12.0, 200)
    z_bulk = 12.0  # FCC bulk coordination

    # Normalized to E(Z=12) = -1.0 (in units of bulk cohesive energy E_coh)
    e_pair_smooth = -(z_smooth / z_bulk)
    e_eam_smooth = -((z_smooth / z_bulk) ** exponent_p)

    e_pair_pts = -(z_vals / z_bulk)
    e_eam_pts = -((z_vals / z_bulk) ** exponent_p)

    # Energy per bond: E(Z) / Z (normalized to 1 at Z=12)
    b_pair_smooth = np.ones_like(z_smooth)
    b_eam_smooth = (z_smooth / z_bulk) ** (exponent_p - 1.0)
    b_eam_pts = (z_vals / z_bulk) ** (exponent_p - 1.0)

    # Unrelaxed FCC vacancy formation energy ratio Ev / Ecoh:
    # Removing 1 bulk atom breaks 12 bonds; 12 neighbors each change from Z=12 to Z=11:
    # Ev / Ecoh = 12 * ((11/12)^p - 1) - 1  Wait: E_tot(N) = N * E(12).
    # With 1 vacancy (and removed atom placed back into bulk):
    # E_tot_final - E_tot_initial = 12 * E(11) - 12 * E(12) = 12 * E_coh * (1 - (11/12)^p)
    ev_over_ec = 12.0 * (1.0 - (11.0 / 12.0) ** exponent_p)
    surf_bond_strengthening = (8.0 / 12.0) ** (exponent_p - 1.0)

    if plot_mode == "cohesive":
        ax.plot(
            z_smooth,
            e_pair_smooth,
            color=COLORS["muted"],
            linestyle="--",
            linewidth=2.0,
            label=r"Pair Potential: $E(Z)/E_{\mathrm{coh}} = -Z / 12$",
        )
        ax.plot(
            z_smooth,
            e_eam_smooth,
            color=COLORS["primary"],
            linewidth=2.5,
            label=f"Many-Body EAM: $E(Z)/E_{{\\mathrm{{coh}}}} = -(Z / 12)^{{{exponent_p:.2f}}}$",
        )
        ax.scatter(z_vals, e_pair_pts, color=COLORS["muted"], s=35, zorder=4)
        ax.scatter(z_vals, e_eam_pts, color=COLORS["primary"], s=50, zorder=5)
        ax.scatter([8.0, 12.0], [-((8.0 / 12.0) ** exponent_p), -1.0], color=COLORS["accent"], s=85, zorder=6)
        ax.set_ylabel(r"Binding Energy per Atom $E(Z) / E_{\mathrm{coh}}$")
        ax.set_ylim(-1.10, 0.0)
        ax.set_title(f"Cohesive Energy vs. Coordination Number $Z$ ($p={exponent_p:.2f}$)")
    else:
        ax.plot(
            z_smooth,
            b_pair_smooth,
            color=COLORS["muted"],
            linestyle="--",
            linewidth=2.0,
            label="Pair Potential: Constant Bond Strength",
        )
        ax.plot(
            z_smooth,
            b_eam_smooth,
            color=COLORS["accent"],
            linewidth=2.5,
            label=f"EAM Bond Strength $\\propto Z^{{{exponent_p - 1.0:.2f}}}$",
        )
        ax.scatter(z_vals, b_eam_pts, color=COLORS["accent"], s=50, zorder=5)
        ax.axvline(8.0, color=COLORS["good"], linestyle=":", linewidth=1.3, label="Surface Atom ($Z=8$)")
        ax.axvline(12.0, color=COLORS["primary"], linestyle=":", linewidth=1.3, label="Bulk FCC Atom ($Z=12$)")
        ax.set_ylabel("Relative Strength per Bond (vs. Bulk $Z=12$)")
        ax.set_ylim(0.85, max(2.6, float(b_eam_pts[0]) * 1.08))
        ax.set_title(f"Bond Strengthening at Lower Coordination ($p={exponent_p:.2f}$)")

    ax.set_xlim(0.5, 12.5)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Coordination Number $Z$ (Number of Nearest Neighbors)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="best", fontsize=8.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**EAM Non-Linearity ($F(\\rho) \\propto -\\rho^{{{exponent_p:.2f}}}$):** "
        f"Predicted Vacancy Formation Energy ratio **$E_v / E_{{\\mathrm{{coh}}}} \\approx {ev_over_ec:.2f}$** "
        f"(vs. $1.00$ for pair potentials and $\\sim 0.25\\text{{–}}0.36$ in real FCC metals). "
        f"At an FCC(100) surface ($Z=8$), each remaining metallic bond is **{surf_bond_strengthening:.2f}$\\times$ stronger** "
        f"than a bulk bond, pulling the top atomic layer **inward (surface contraction)**!"
    )


def render_stillinger_weber_angle(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot the Stillinger-Weber 3-body angular penalty vs bond angle theta for silicon."""
    lam = float(_val(controls.get("lambda_sw"), 21.0))
    r_ratio = float(_val(controls.get("r_ratio"), 1.15))

    theta_deg = np.linspace(40.0, 180.0, 350)
    theta_rad = np.deg2rad(theta_deg)

    # SW parameters for Si: a = 1.80, gamma = 1.20
    a_cut = 1.80
    gamma = 1.20
    radial_factor = np.exp(2.0 * gamma / (r_ratio - a_cut)) if r_ratio < a_cut else 0.0

    h_theta = lam * radial_factor * ((np.cos(theta_rad) + 1.0 / 3.0) ** 2)

    theta_tet = 109.4712
    h_60 = lam * radial_factor * ((np.cos(np.deg2rad(60.0)) + 1.0 / 3.0) ** 2)
    h_90 = lam * radial_factor * ((np.cos(np.deg2rad(90.0)) + 1.0 / 3.0) ** 2)
    h_180 = lam * radial_factor * ((np.cos(np.deg2rad(180.0)) + 1.0 / 3.0) ** 2)

    ax.plot(
        theta_deg,
        h_theta,
        color=COLORS["primary"],
        linewidth=2.6,
        label=rf"$h(r, r, \theta) = \lambda e^{{2\gamma/(r-a)}}(\cos\theta + 1/3)^2$",
    )

    ax.axvline(
        theta_tet,
        color=COLORS["good"],
        linestyle="--",
        linewidth=1.6,
        label=r"Tetrahedral $\theta_{\mathrm{tet}} = 109.47^\circ$ ($\cos\theta = -1/3$)",
    )
    ax.scatter([theta_tet], [0.0], color=COLORS["good"], s=95, zorder=5)

    # Mark close-packed 60 deg and 90 deg angles
    ax.scatter([60.0, 90.0], [h_60, h_90], color=COLORS["danger"], s=75, zorder=5)
    ax.annotate(
        f"FCC/HCP $60^\\circ$\n(+{h_60:.2f}$\\,\\varepsilon$)",
        xy=(60.0, h_60),
        xytext=(64.0, h_60 + 0.12),
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["danger"],
    )
    ax.annotate(
        f"Octahedral $90^\\circ$\n(+{h_90:.2f}$\\,\\varepsilon$)",
        xy=(90.0, h_90),
        xytext=(82.0, h_90 + 0.22),
        fontsize=8.3,
        fontweight="bold",
        color=COLORS["danger"],
    )

    ax.set_xlim(40.0, 180.0)
    ax.set_ylim(-0.05, max(1.8, float(np.max(h_theta)) * 1.12))
    ax.set_xlabel(r"Triplet Bond Angle $\theta_{jik}$ (degrees)")
    ax.set_ylabel(r"3-Body Energy Penalty $h_{jik} / \varepsilon$")
    ax.set_title(f"Stillinger-Weber 3-Body Angular Penalty ($\\lambda={lam:.1f}$, $r/\\sigma={r_ratio:.2f}$)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Tetrahedral Minimum ($\\theta = 109.47^\\circ$, $\\cos\\theta = -1/3$):** $h = 0.00\\,\\varepsilon$.  \n"
        f"**Close-Packed Penalties at $r = {r_ratio:.2f}\\sigma$:** $\\theta = 60^\\circ$ costs **+{h_60:.3f}$\\,\\varepsilon$**, "
        f"$\\theta = 90^\\circ$ costs **+{h_90:.3f}$\\,\\varepsilon$**, and linear $\\theta = 180^\\circ$ costs **+{h_180:.3f}$\\,\\varepsilon$** per triplet, "
        f"stabilizing open $sp^3$ diamond-cubic Silicon ($Z=4$) against collapse into metallic FCC/BCC!"
    )


def render_organic_dihedral_explorer(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot 4-body dihedral torsion potential V(phi) = 0.5 * Vn * [1 - cos(n*phi)] or alkane trans/gauche curve."""
    multiplicity_n = int(_val(controls.get("multiplicity_n"), 3))
    barrier_vn = float(_val(controls.get("barrier_vn"), 3.0))

    phi_deg = np.linspace(-180.0, 180.0, 361)
    phi_rad = np.deg2rad(phi_deg)

    # Standard AMBER-style torsional term V(phi) = (Vn / 2) * [1 + cos(n * phi)] where phi=0 is cis/eclipsed
    v_torsion = 0.5 * barrier_vn * (1.0 + np.cos(multiplicity_n * phi_rad))

    ax.plot(
        phi_deg,
        v_torsion,
        color=COLORS["primary"],
        linewidth=2.5,
        label=rf"$V_{{\mathrm{{torsion}}}}(\phi) = \frac{{V_n}}{{2}}[1 + \cos({multiplicity_n}\phi)]$",
    )
    ax.fill_between(phi_deg, 0, v_torsion, color=COLORS["primary"], alpha=0.12)

    # Mark eclipsed maxima and staggered minima for n=3 (ethane)
    ax.scatter([0.0], [barrier_vn], color=COLORS["danger"], s=80, zorder=5, label=r"Eclipsed Maximum ($\phi = 0^\circ$)")
    min_angle = 180.0 / multiplicity_n
    ax.scatter(
        [-min_angle, min_angle],
        [0.0, 0.0],
        color=COLORS["good"],
        s=80,
        zorder=5,
        label=rf"Staggered Minima ($\phi = \pm {min_angle:.0f}^\circ$)",
    )

    ax.set_xlim(-180.0, 180.0)
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_ylim(-0.2, max(6.5, barrier_vn * 1.2))
    ax.set_xlabel(r"Dihedral Torsion Angle $\phi$ (degrees)")
    ax.set_ylabel(r"Torsional Potential Energy $V(\phi)$ (kcal/mol)")
    ax.set_title(f"4-Body Dihedral Torsion Potential ($n={multiplicity_n}$, Barrier $V_n={barrier_vn:.1f}$ kcal/mol)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**4-Body Dihedral Periodicity ($n={multiplicity_n}$):** Contains **{multiplicity_n} minima** "
        f"(staggered conformations at $\\phi = \\pm {min_angle:.0f}^\\circ$) and **{multiplicity_n} maxima** "
        f"(eclipsed conformations with barrier height **$V_n = {barrier_vn:.1f}$ kcal/mol**). "
        f"Defining a dihedral angle requires **4 bonded atoms** ($A\\text{{–}}B\\text{{–}}C\\text{{–}}D$) to specify two intersecting planes."
    )
