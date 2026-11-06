"""Interactive concept visualizations for Lesson 21: Simulation of Polymers."""

from __future__ import annotations
from typing import Any, Mapping
import matplotlib.pyplot as plt
import numpy as np
from engine.plot_style import COLORS


def render_fene_vs_harmonic_bonds(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare FENE bond potential + WCA repulsive potential against standard harmonic bonds."""
    k_spring = float(controls.get("k_spring", 30.0))
    r0_max = float(controls.get("r0_max", 1.5))
    show_wca = int(controls.get("show_wca", 1))

    r = np.linspace(0.85, 1.95, 500)
    r_fene = r[r < r0_max * 0.998]

    # FENE potential: -0.5 * k * R0^2 * ln(1 - (r/R0)^2)
    u_fene = -0.5 * k_spring * (r0_max**2) * np.log(1.0 - (r_fene / r0_max) ** 2)
    # Harmonic approximation with same curvature at r=0: 0.5 * k * r^2
    u_harm = 0.5 * k_spring * (r**2)

    # WCA repulsive core (shifted LJ truncated at 2^(1/6) sigma, with sigma=1, eps=1)
    r_cut = 2.0 ** (1.0 / 6.0)
    u_wca_full = np.zeros_like(r)
    mask_wca = r < r_cut
    sr6 = (1.0 / r[mask_wca]) ** 6
    u_wca_full[mask_wca] = 4.0 * (sr6**2 - sr6) + 1.0

    ax.plot(
        r,
        u_harm,
        linestyle="--",
        linewidth=2.0,
        color=COLORS["muted"],
        label=r"Harmonic $\frac{1}{2}k r^2$ (allows bond stretching)",
    )
    ax.plot(
        r_fene,
        u_fene,
        linestyle="-",
        linewidth=2.4,
        color=COLORS["primary"],
        label=rf"FENE $-\frac{{1}}{{2}}k R_0^2 \ln[1-(r/R_0)^2]$",
    )

    if show_wca:
        u_wca_fene = np.interp(r_fene, r, u_wca_full)
        u_total = u_fene + u_wca_fene
        ax.plot(
            r,
            u_wca_full,
            linestyle=":",
            linewidth=1.8,
            color=COLORS["secondary"],
            label=r"WCA Excluded Volume ($r < 2^{1/6}\sigma$)",
        )
        ax.plot(
            r_fene,
            u_total,
            linestyle="-",
            linewidth=2.8,
            color=COLORS["accent"],
            label=r"Total Kremer–Grest Bond ($U_{\mathrm{FENE}} + U_{\mathrm{WCA}}$)",
        )
        idx_min = int(np.argmin(u_total))
        r_eq = float(r_fene[idx_min])
        u_eq = float(u_total[idx_min])
        ax.scatter([r_eq], [u_eq], color=COLORS["accent"], s=55, zorder=5)
        ax.annotate(
            f"$r_{{\\mathrm{{eq}}}} \\approx {r_eq:.2f}\\sigma$",
            xy=(r_eq, u_eq),
            xytext=(r_eq - 0.22, u_eq - 5.5),
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["accent"],
            arrowprops={"arrowstyle": "->", "color": COLORS["accent"], "lw": 1.2},
        )
    else:
        r_eq = 0.0
        u_eq = 0.0

    ax.axvline(
        r0_max,
        color=COLORS["danger"],
        linestyle="-.",
        linewidth=1.8,
        label=f"Finite Extensibility Wall $R_0 = {r0_max:.2f}\\sigma$",
    )
    ax.axvspan(r0_max, 1.95, color=COLORS["danger"], alpha=0.08)

    u_harm_at_r0 = 0.5 * k_spring * (r0_max**2)
    ax.set_xlim(0.85, 1.95)
    ax.set_ylim(0.0, max(45.0, u_harm_at_r0 * 1.45))
    ax.set_xlabel(r"Inter-bead separation $r / \sigma$")
    ax.set_ylabel(r"Potential energy $U(r) / \epsilon$")
    ax.set_title("Kremer–Grest FENE vs. Harmonic Polymer Bond Potential")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    eq_note = (
        f"With WCA repulsion active, the equilibrium bond length settles near $r_{{\\mathrm{{eq}}}} \\approx {r_eq:.2f}\\sigma$."
        if show_wca
        else "Enable WCA excluded volume to see how hard-core repulsion shifts the bond minimum away from $r=0$."
    )
    return (
        f"**FENE Parameter Readout:** For $k = {k_spring:.0f}\\,\\epsilon/\\sigma^2$ and $R_0 = {r0_max:.2f}\\sigma$, "
        f"a harmonic spring at $r = R_0$ only costs $\\frac{{1}}{{2}}k R_0^2 = {u_harm_at_r0:.1f}\\,\\epsilon$, "
        f"which thermal fluctuations can occasionally surmount—allowing two polymer chains to pass through one another "
        f"(violating topological entanglement). By contrast, $U_{{\\mathrm{{FENE}}}}(r \\to R_0) \\to +\\infty$ creates "
        f"an impenetrable logarithmic wall at $R_0 = {r0_max:.2f}\\sigma$. {eq_note}"
    )


def render_flory_scaling_regimes(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Log-log scaling plot of R_g vs N and representative 2D coil conformation inset."""
    regime = str(controls.get("solvent_regime", "good"))
    n_beads = int(controls.get("chain_length", 120))
    dim = int(controls.get("dimension", 3))

    # Exponents nu across solvent regimes
    nu_good = 0.588 if dim == 3 else (0.75 if dim == 2 else 1.0)
    nu_flory = 3.0 / (dim + 2.0) if dim <= 4 else 0.5
    nu_theta = 0.5
    nu_poor = 1.0 / float(dim)
    nu_rod = 1.0

    regime_info = {
        "poor": ("Poor Solvent (Collapsed Globule)", nu_poor, COLORS["quaternary"]),
        "theta": (r"$\Theta$-Solvent (Ideal Random Walk)", nu_theta, COLORS["secondary"]),
        "good": (f"Good Solvent (Swollen SAW, d={dim})", nu_good, COLORS["primary"]),
        "rod": ("Rigid Rod / Stretched Limit", nu_rod, COLORS["accent"]),
    }
    reg_label, nu_active, col_active = regime_info.get(regime, regime_info["good"])

    n_arr = np.logspace(1.0, 3.0, 200)
    a0 = 1.0  # statistical segment length

    rg_poor = 0.45 * a0 * (n_arr**nu_poor)
    rg_theta = (a0 / np.sqrt(6.0)) * (n_arr**nu_theta)
    rg_good = 0.42 * a0 * (n_arr**nu_good)
    rg_rod = (a0 / np.sqrt(12.0)) * (n_arr**nu_rod)

    ax.loglog(
        n_arr,
        rg_rod,
        linestyle=":",
        linewidth=1.6,
        color=COLORS["accent"],
        alpha=0.85 if regime == "rod" else 0.55,
        label=f"Rigid Rod ($\\nu = 1.00$)",
    )
    ax.loglog(
        n_arr,
        rg_good,
        linestyle="-" if regime == "good" else "--",
        linewidth=2.6 if regime == "good" else 1.7,
        color=COLORS["primary"],
        label=f"Good Solvent SAW ($\\nu \\approx {nu_good:.3f}$, Flory $3/(d+2)={nu_flory:.2f}$)",
    )
    ax.loglog(
        n_arr,
        rg_theta,
        linestyle="-" if regime == "theta" else "--",
        linewidth=2.6 if regime == "theta" else 1.7,
        color=COLORS["secondary"],
        label=r"$\Theta$-Solvent Ideal Walk ($\nu = 0.500$)",
    )
    ax.loglog(
        n_arr,
        rg_poor,
        linestyle="-" if regime == "poor" else "--",
        linewidth=2.6 if regime == "poor" else 1.7,
        color=COLORS["quaternary"],
        label=f"Poor Solvent Globule ($\\nu = 1/d = {nu_poor:.3f}$)",
    )

    # Active point marker
    pref_map = {
        "poor": 0.45,
        "theta": 1.0 / np.sqrt(6.0),
        "good": 0.42,
        "rod": 1.0 / np.sqrt(12.0),
    }
    rg_val = pref_map.get(regime, 0.42) * a0 * (n_beads**nu_active)
    r_ee_val = a0 * (n_beads**nu_active)
    tau_relax = float(n_beads) ** (2.0 * nu_active + 1.0)

    ax.scatter([n_beads], [rg_val], color=col_active, s=75, zorder=6, edgecolors="white", linewidth=1.5)
    ax.axvline(n_beads, color=col_active, linestyle=":", linewidth=1.2, alpha=0.6)

    ax.set_xlim(10, 1000)
    ax.set_ylim(0.8, 350.0)
    ax.set_xlabel("Degree of Polymerization $N$ (monomers)")
    ax.set_ylabel(r"Root-Mean-Square Radius of Gyration $\langle R_g^2 \rangle^{1/2} / a$")
    ax.set_title(f"Flory Size Scaling $\\langle R_g^2 \\rangle^{{1/2}} \\propto N^\\nu$ ({reg_label})")
    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.6, alpha=0.8)
    ax.legend(loc="upper left", fontsize=7.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Inset showing representative 2D coil projection
    ax_in = ax.inset_axes([0.62, 0.08, 0.35, 0.38])
    rng = np.random.default_rng(42 + int(n_beads))
    n_draw = min(n_beads, 90)
    if regime == "rod":
        angles = np.linspace(0, 0.18, n_draw)
        xs = np.cumsum(np.cos(angles))
        ys = np.cumsum(np.sin(angles))
    elif regime == "good":
        # Swollen persistent self-avoiding style path
        angles = np.cumsum(rng.normal(0.0, 0.62, size=n_draw))
        xs = np.cumsum(np.cos(angles))
        ys = np.cumsum(np.sin(angles))
    elif regime == "theta":
        angles = rng.uniform(0.0, 2.0 * np.pi, size=n_draw)
        xs = np.cumsum(np.cos(angles))
        ys = np.cumsum(np.sin(angles))
    else:
        # Compact globule spiral with jitter
        t_glob = np.linspace(0.5, 4.8 * np.pi, n_draw)
        rad = np.sqrt(np.linspace(0.2, 1.0, n_draw))
        xs = rad * np.cos(t_glob) + rng.normal(0.0, 0.08, size=n_draw)
        ys = rad * np.sin(t_glob) + rng.normal(0.0, 0.08, size=n_draw)

    xs -= np.mean(xs)
    ys -= np.mean(ys)
    ax_in.plot(xs, ys, "-o", color=col_active, markersize=2.8, linewidth=1.3, alpha=0.9)
    ax_in.scatter([xs[0], xs[-1]], [ys[0], ys[-1]], color=COLORS["danger"], s=18, zorder=5)
    ax_in.set_title(f"Coil ($\\nu={nu_active:.3f}$)", fontsize=8.0, pad=2)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.set_aspect("equal", adjustable="datalim")

    return (
        f"**Active Regime ({reg_label}, $d={dim}$):** At $N = {n_beads}$, the scaling exponent is "
        f"$\\nu = {nu_active:.3f}$ (Flory mean-field estimate $\\nu_{{\\mathrm{{Flory}}}} = 3/(d+2) = {nu_flory:.3f}$), "
        f"yielding $\\langle R_g^2\\rangle^{{1/2}} \\approx {rg_val:.2f}\\,a$, end-to-end size "
        f"$\\langle R_{{ee}}^2\\rangle^{{1/2}} \\approx {r_ee_val:.1f}\\,a$, and Rouse relaxation time scaling "
        f"$\\tau_{{\\mathrm{{relax}}}} \\sim N^{{2\\nu + 1}} \\approx N^{{{2*nu_active+1:.2f}}} \\sim {tau_relax:.2e}$ steps."
    )


def render_reptation_and_pivot_moves(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize 2D square-lattice polymer Monte Carlo moves: Reptation, Pivot, Crankshaft, and Double Cul-de-Sac trap."""
    move_type = str(controls.get("move_type", "reptation"))
    pivot_idx = int(controls.get("pivot_index", 6))
    rot_code = int(controls.get("rotation_angle", 90))

    # Base 13-monomer self-avoiding walk on 2D square lattice (indices 0..12)
    base_chain = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [2, 1],
            [2, 2],
            [3, 2],
            [4, 2],
            [4, 3],
            [5, 3],
            [5, 4],
            [6, 4],
            [6, 5],
            [7, 5],
        ],
        dtype=float,
    )

    if move_type == "cul_de_sac":
        # Non-ergodic double cul-de-sac configuration where both ends (0 and N-1) are surrounded by occupied chain sites
        trap_chain = np.array(
            [
                [1, 1],  # 0: trapped left end! Neighbors (0,1),(2,1),(1,2),(1,0) all occupied!
                [1, 2],  # 1
                [0, 2],  # 2
                [0, 1],  # 3
                [0, 0],  # 4
                [1, 0],  # 5
                [2, 0],  # 6
                [2, 1],  # 7
                [2, 2],  # 8
                [3, 2],  # 9
                [4, 2],  # 10
                [5, 2],  # 11
                [6, 2],  # 12
                [6, 1],  # 13
                [6, 0],  # 14
                [5, 0],  # 15
                [4, 0],  # 16
                [4, 1],  # 17
                [5, 1],  # 18: trapped right end! Neighbors (4,1),(6,1),(5,2),(5,0) all occupied!
            ],
            dtype=float,
        )
        ax.plot(
            trap_chain[:, 0],
            trap_chain[:, 1],
            "-o",
            color=COLORS["primary"],
            linewidth=2.6,
            markersize=8,
            label="Trapped Polymer Backbone",
        )
        ax.scatter(
            [trap_chain[0, 0], trap_chain[-1, 0]],
            [trap_chain[0, 1], trap_chain[-1, 1]],
            color=COLORS["danger"],
            s=130,
            zorder=6,
            label="Trapped Chain Ends (0 Open Neighbors!)",
        )
        for end_pt, lbl in [(trap_chain[0], "End A (Blocked)"), (trap_chain[-1], "End B (Blocked)")]:
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ax.plot(
                    [end_pt[0], end_pt[0] + 0.45 * dx],
                    [end_pt[1], end_pt[1] + 0.45 * dy],
                    color=COLORS["danger"],
                    linestyle=":",
                    linewidth=1.8,
                )
            ax.annotate(
                lbl,
                xy=(end_pt[0], end_pt[1]),
                xytext=(end_pt[0] - 0.6, end_pt[1] + 1.35),
                fontsize=8.8,
                fontweight="bold",
                color=COLORS["danger"],
                arrowprops={"arrowstyle": "->", "color": COLORS["danger"], "lw": 1.2},
            )
        ax.set_xlim(-1.2, 7.5)
        ax.set_ylim(-1.0, 3.6)
        ax.set_aspect("equal")
        ax.grid(True, color="#CBD5E1", linestyle="-", linewidth=0.6)
        ax.set_title("Pure Reptation Non-Ergodicity: The Double Cul-de-Sac Trap")
        ax.legend(loc="lower center", fontsize=8.2)
        return (
            "**Non-Ergodicity of Pure Reptation:** Both chain ends $(1,1)$ and $(5,1)$ have all $q=4$ nearest-neighbor "
            "lattice sites already occupied by interior monomers ($q_{\\mathrm{open}} = 0$). Neither forward nor backward "
            "slithering-snake moves can proceed ($A = 0$), permanently freezing the chain and violating ergodicity unless "
            "supplemented with kink-jump, crankshaft, or pivot moves."
        )

    ax.plot(
        base_chain[:, 0],
        base_chain[:, 1],
        "--o",
        color=COLORS["muted"],
        linewidth=1.8,
        markersize=6,
        alpha=0.65,
        label="Old Conformation",
    )

    if move_type == "reptation":
        # Delete tail monomer 0 at (0,0), shift chain along contour, add new head at (7,6)
        new_head = np.array([[7.0, 6.0]])
        new_chain = np.vstack([base_chain[1:], new_head])
        ax.plot(
            new_chain[:, 0],
            new_chain[:, 1],
            "-o",
            color=COLORS["good"],
            linewidth=2.6,
            markersize=7.5,
            label="New Conformation (Slithered +1 step)",
        )
        ax.scatter(
            [base_chain[0, 0]],
            [base_chain[0, 1]],
            color=COLORS["danger"],
            s=110,
            marker="x",
            linewidths=2.5,
            zorder=6,
            label="Deleted Tail Monomer $r_1$",
        )
        ax.scatter(
            [new_head[0, 0]],
            [new_head[0, 1]],
            color=COLORS["accent"],
            s=115,
            zorder=6,
            label="Appended Head Monomer $r'_N$",
        )
        ax.annotate(
            "Slithering Snake Tube Motion $\\mathcal{O}(1)$",
            xy=(4.0, 3.0),
            xytext=(0.5, 5.2),
            fontsize=9.0,
            fontweight="bold",
            color=COLORS["primary"],
            arrowprops={"arrowstyle": "->", "color": COLORS["primary"], "lw": 1.3},
        )
        caption = (
            "**Reptation ('Slithering Snake') Move:** Deleting the tail monomer at $(0,0)$ and appending a new head monomer "
            "at $(7,6)$ advances the entire chain by one lattice constant along its contour tube in $\\mathcal{O}(1)$ CPU operations, "
            "reducing global decorrelation time from $\\mathcal{O}(N^3)$ to $\\tau \\sim \\mathcal{O}(N^2)$ even in dense melts."
        )
    elif move_type == "crankshaft":
        # Local corner-flip / crankshaft update on interior beads 3=(2,1) -> (3,1)
        new_chain = base_chain.copy()
        new_chain[3] = np.array([3.0, 1.0])
        new_chain[7] = np.array([5.0, 2.0])
        ax.plot(
            new_chain[:, 0],
            new_chain[:, 1],
            "-o",
            color=COLORS["secondary"],
            linewidth=2.5,
            markersize=7.5,
            label="New Conformation (Local Kink-Jump / Crankshaft)",
        )
        ax.scatter(
            [new_chain[3, 0], new_chain[7, 0]],
            [new_chain[3, 1], new_chain[7, 1]],
            color=COLORS["accent"],
            s=105,
            zorder=6,
            label="Flipped Corner Monomers",
        )
        caption = (
            "**Local Kink-Jump / Crankshaft Move:** Flipping 1–2 interior monomers across a lattice square updates short-wavelength "
            "bond orientations and restores ergodicity when combined with reptation or pivot algorithms, though pure local Rouse-like "
            "moves alone relax slowly as $\\tau \\sim N^{2\\nu+1}$."
        )
    else:
        # Pivot move around monomer k = pivot_idx
        k = int(np.clip(pivot_idx, 2, len(base_chain) - 3))
        pivot_pt = base_chain[k]
        theta = np.deg2rad(float(rot_code))
        c, s = np.cos(theta), np.sin(theta)
        rot_mat = np.array([[np.round(c), -np.round(s)], [np.round(s), np.round(c)]])

        new_chain = base_chain.copy()
        rel = base_chain[k + 1 :] - pivot_pt
        new_chain[k + 1 :] = pivot_pt + rel @ rot_mat.T

        # Check self-avoidance overlap between subchains
        head_set = {tuple(np.round(pt).astype(int)) for pt in new_chain[: k + 1]}
        tail_pts = [tuple(np.round(pt).astype(int)) for pt in new_chain[k + 1 :]]
        overlaps = [pt for pt in tail_pts if pt in head_set]
        accepted = len(overlaps) == 0

        col_piv = COLORS["good"] if accepted else COLORS["danger"]
        status_str = "ACCEPTED (Self-Avoiding)" if accepted else f"REJECTED ({len(overlaps)} Overlap)"
        ax.plot(
            new_chain[:, 0],
            new_chain[:, 1],
            "-o",
            color=col_piv,
            linewidth=2.6,
            markersize=7.5,
            label=f"Pivoted Subchain ({rot_code}° around k={k}): {status_str}",
        )
        ax.scatter(
            [pivot_pt[0]],
            [pivot_pt[1]],
            color=COLORS["highlight"],
            edgecolors=COLORS["text"],
            linewidth=1.8,
            s=140,
            zorder=7,
            label=f"Pivot Monomer $k={k}$",
        )
        if overlaps:
            ov_arr = np.array(overlaps, dtype=float)
            ax.scatter(
                ov_arr[:, 0],
                ov_arr[:, 1],
                color=COLORS["danger"],
                marker="X",
                s=160,
                zorder=8,
                label="Excluded-Volume Collision!",
            )
        caption = (
            f"**Pivot Algorithm ($k={k}$, angle ${rot_code}^\\circ$):** Rotating the entire subchain $(k+1 \\dots N)$ "
            f"around monomer $k={k}$ produces a massive global conformation change in a single step (**{status_str}**). "
            f"For a single dilute SAW, the pivot algorithm achieves an extraordinarily small autocorrelation exponent "
            f"$\\tau \\sim \\mathcal{{O}}(N^{{0.2}})$, though acceptance drops rapidly in concentrated melts."
        )

    ax.set_xlim(-2.5, 9.5)
    ax.set_ylim(-3.5, 8.5)
    ax.set_aspect("equal")
    ax.set_xlabel("Lattice coordinate $x / a$")
    ax.set_ylabel("Lattice coordinate $y / a$")
    ax.set_title(f"Polymer Monte Carlo Move Menu: {move_type.replace('_', ' ').title()}")
    ax.grid(True, color="#E2E8F0", linestyle="-", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8.0)
    return caption


def render_cbmc_rosenbluth_growth(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare blind SAW growth attrition against Rosenbluth look-ahead weights and CBMC acceptance."""
    n_regrow = int(controls.get("regrow_steps", 8))
    q_coord = int(controls.get("lattice_coord", 4))
    w_ratio = float(controls.get("weight_ratio", 1.25))

    steps = np.arange(1, n_regrow + 1)
    # Deterministic representative sequence of open choices q_i out of q_coord
    rng = np.random.default_rng(17 + q_coord)
    q_open_new = np.clip(
        np.round(q_coord * (0.72 + 0.15 * np.sin(steps * 1.1))).astype(int),
        1,
        q_coord - 1,
    )
    step_factors_new = q_open_new / float(q_coord)
    w_new_cum = np.cumprod(step_factors_new)

    # Old trajectory cumulative weight scaled so final W_new / W_old matches w_ratio
    w_old_final = w_new_cum[-1] / max(0.05, w_ratio)
    w_old_cum = np.geomspace(step_factors_new[0], w_old_final, num=n_regrow)

    # Blind random walk survival probability (attrition)
    blind_survival = ((q_coord - 1.0) / float(q_coord)) ** steps * np.exp(-0.18 * steps)

    ax.semilogy(
        steps,
        blind_survival,
        "--s",
        color=COLORS["danger"],
        linewidth=1.9,
        markersize=5.5,
        label=r"Blind Growth Survival $\propto e^{-\lambda n}$ (Attrition)",
    )
    ax.semilogy(
        steps,
        w_old_cum,
        "-o",
        color=COLORS["muted"],
        linewidth=2.2,
        markersize=6.0,
        label=r"Old Segment Rosenbluth Weight $W_{\mathrm{old}}(n)$",
    )
    ax.semilogy(
        steps,
        w_new_cum,
        "-^",
        color=COLORS["primary"],
        linewidth=2.5,
        markersize=6.8,
        label=r"New Regrown Segment $W_{\mathrm{new}}(n) = \prod_{i=1}^n (q_i / q)$",
    )

    for s_idx, q_i, w_val in zip(steps, q_open_new, w_new_cum):
        ax.annotate(
            f"$q_{{{s_idx}}}={q_i}/{q_coord}$",
            xy=(s_idx, w_val),
            xytext=(s_idx - 0.3, w_val * 1.38),
            fontsize=7.8,
            color=COLORS["primary"],
        )

    acc_prob = min(1.0, float(w_new_cum[-1] / w_old_cum[-1]))
    ax.set_xlim(0.7, n_regrow + 0.5)
    ax.set_xlabel("Regrown Segment Step $i$")
    ax.set_ylabel("Cumulative Rosenbluth Weight $W_i$ (log scale)")
    ax.set_title(
        f"Configurational Bias MC (CBMC): $W_{{\\mathrm{{new}}}}/W_{{\\mathrm{{old}}}} = {w_ratio:.2f} "
        f"\\Rightarrow A = \\min(1, W_{{\\mathrm{{new}}}}/W_{{\\mathrm{{old}}}}) = {acc_prob:.2f}$"
    )
    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="lower left", fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Rosenbluth & CBMC Readout ($q={q_coord}$, $n={n_regrow}$ steps):** At each growth step $i$, the algorithm "
        f"scans all $q={q_coord}$ neighbor directions, counts the $q_i$ unoccupied sites, and selects one open site "
        f"with probability $1/q_i$ while multiplying the Rosenbluth weight by $q_i/q$. After $n={n_regrow}$ steps, "
        f"$W_{{\\mathrm{{new}}}} = {w_new_cum[-1]:.4e}$ and $W_{{\\mathrm{{old}}}} = {w_old_cum[-1]:.4e}$. "
        f"Accepting the regrown segment with $A(\\mathrm{{old}}\\to\\mathrm{{new}}) = \\min(1, W_{{\\mathrm{{new}}}}/W_{{\\mathrm{{old}}}}) = {acc_prob:.3f}$ "
        f"exactly cancels the look-ahead bias and restores the canonical Boltzmann distribution."
    )
