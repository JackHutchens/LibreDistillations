"""Interactive concept visuals for Lesson 05: Periodic Boundary Conditions & Neighbor Lists."""

from __future__ import annotations
from typing import Any, Mapping
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from engine.plot_style import COLORS


def _val(ctrl: Any, default: Any = 0.0) -> Any:
    if hasattr(ctrl, "value"):
        return ctrl.value
    return ctrl if ctrl is not None else default


def render_surface_vs_bulk_scaling(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot fraction of surface atoms vs system size N for a 3D cubic cluster with surface thickness d."""
    n_side = int(_val(controls.get("n_side"), 10))
    d_surf = int(_val(controls.get("d_surf"), 1))

    sides = np.arange(max(2 * d_surf + 1, 3), 121)
    n_total = sides**3
    n_interior = np.maximum(0, sides - 2 * d_surf) ** 3
    f_surf = 100.0 * (n_total - n_interior) / n_total
    f_asymp = np.minimum(100.0, 100.0 * 6.0 * d_surf / sides)

    ax.plot(
        n_total,
        f_surf,
        color=COLORS["primary"],
        linewidth=2.4,
        label=f"Exact Cubic Cluster ($d={d_surf}$ layer{'s' if d_surf > 1 else ''})",
    )
    ax.plot(
        n_total,
        f_asymp,
        color=COLORS["muted"],
        linestyle="--",
        linewidth=1.6,
        label=r"Asymptotic $6d / N^{1/3}$",
    )

    N_sel = n_side**3
    n_int_sel = max(0, n_side - 2 * d_surf) ** 3
    n_surf_sel = N_sel - n_int_sel
    f_sel = 100.0 * n_surf_sel / N_sel

    ax.scatter([N_sel], [f_sel], color=COLORS["accent"], s=90, zorder=5)
    ax.axvline(N_sel, color=COLORS["accent"], linestyle=":", linewidth=1.3)
    ax.axhline(f_sel, color=COLORS["accent"], linestyle=":", linewidth=1.3)

    ax.set_xscale("log")
    ax.set_xlim(20, 2.0e6)
    ax.set_ylim(0, 105)
    ax.set_xlabel(r"Total Number of Atoms $N = n^3$ (log scale)")
    ax.set_ylabel("Surface Atom Percentage (%)")
    ax.set_title(f"3D Free Cluster Surface Fraction: $n={n_side}$ ($N={N_sel:,}$)")
    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Selected Cluster ($n={n_side}$, $N={N_sel:,}$ atoms, surface shell $d={d_surf}$):** "
        f"Interior bulk atoms = $(n - 2d)^3 = {n_int_sel:,}$, Surface atoms = **{n_surf_sel:,} ({f_sel:.1f}%)**. "
        f"Under Periodic Boundary Conditions (PBC), the surface fraction drops to **0%** regardless of $N$."
    )


def render_minimum_image_explorer(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize a 2D periodic box with 8 surrounding ghost tiles, MIC vector, and cutoff sphere."""
    xj = float(_val(controls.get("xj"), 0.85))
    yj = float(_val(controls.get("yj"), 0.80))
    rc = float(_val(controls.get("rc"), 0.42))

    # Reference particle i near lower-left of primary box [0, 1] x [0, 1]
    xi, yi = 0.20, 0.25
    L = 1.0

    # Raw box displacement and Minimum Image Convention (MIC) displacement
    dx_raw = xj - xi
    dy_raw = yj - yi
    r_raw = float(np.hypot(dx_raw, dy_raw))

    shift_x = np.round(dx_raw / L)
    shift_y = np.round(dy_raw / L)
    dx_mic = dx_raw - L * shift_x
    dy_mic = dy_raw - L * shift_y
    r_mic = float(np.hypot(dx_mic, dy_mic))

    xj_mic = xi + dx_mic
    yj_mic = yi + dy_mic

    # Draw 3x3 periodic grid tiles in [-1, 2] x [-1, 2]
    for nx_tile in (-1, 0, 1):
        for ny_tile in (-1, 0, 1):
            is_primary = nx_tile == 0 and ny_tile == 0
            rect = patches.Rectangle(
                (nx_tile * L, ny_tile * L),
                L,
                L,
                linewidth=2.0 if is_primary else 0.9,
                edgecolor=COLORS["primary"] if is_primary else "#CBD5E1",
                facecolor="#F1F5F9" if is_primary else "#FAFAFA",
                linestyle="-" if is_primary else "--",
                zorder=1,
            )
            ax.add_patch(rect)

            # Plot ghost images of j
            gx = xj + nx_tile * L
            gy = yj + ny_tile * L
            is_closest = (nx_tile == -int(shift_x)) and (ny_tile == -int(shift_y))
            ax.scatter(
                [gx],
                [gy],
                s=95 if is_closest else 45,
                color=COLORS["good"] if is_closest else (COLORS["secondary"] if is_primary else COLORS["muted"]),
                edgecolors="black" if is_closest else "none",
                linewidths=1.4 if is_closest else 0.0,
                alpha=1.0 if (is_closest or is_primary) else 0.55,
                zorder=4 if is_closest else 3,
            )

    # Plot reference particle i in primary cell
    ax.scatter(
        [xi],
        [yi],
        s=120,
        color=COLORS["accent"],
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label=r"Particle $i$ $(0.20, 0.25)$",
    )

    # Draw raw displacement vector inside primary box
    ax.annotate(
        "",
        xy=(xj, yj),
        xytext=(xi, yi),
        arrowprops=dict(arrowstyle="->", color=COLORS["muted"], lw=1.6, ls="--"),
        zorder=4,
    )

    # Draw minimum image vector
    ax.annotate(
        "",
        xy=(xj_mic, yj_mic),
        xytext=(xi, yi),
        arrowprops=dict(arrowstyle="->", color=COLORS["good"], lw=2.4),
        zorder=5,
    )

    # Draw cutoff circle around particle i and MIC square [-L/2, L/2]
    overlap_error = rc >= 0.5 * L
    circle_col = COLORS["danger"] if overlap_error else COLORS["accent"]
    cutoff_circle = patches.Circle(
        (xi, yi),
        rc,
        fill=True,
        facecolor=circle_col,
        alpha=0.14,
        edgecolor=circle_col,
        linewidth=1.8,
        linestyle="-" if not overlap_error else "-.",
        zorder=2,
    )
    ax.add_patch(cutoff_circle)

    mic_box = patches.Rectangle(
        (xi - 0.5 * L, yi - 0.5 * L),
        L,
        L,
        fill=False,
        edgecolor=COLORS["highlight"],
        linewidth=1.4,
        linestyle=":",
        zorder=3,
    )
    ax.add_patch(mic_box)

    # Count how many periodic images of j fall inside rc
    images_in_rc = 0
    for nx_tile in (-1, 0, 1):
        for ny_tile in (-1, 0, 1):
            dist = np.hypot((xj + nx_tile * L) - xi, (yj + ny_tile * L) - yi)
            if dist <= rc:
                images_in_rc += 1

    ax.set_xlim(-0.85, 1.85)
    ax.set_ylim(-0.85, 1.85)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x / L$")
    ax.set_ylabel("$y / L$")
    status_str = "VIOLATES $r_c < L/2$!" if overlap_error else "Valid ($r_c < L/2$)"
    ax.set_title(f"Minimum Image Convention ($r_{{\\mathrm{{mic}}}}={r_mic:.3f}L$, {status_str})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Raw primary-box displacement:** $\\Delta\\mathbf{{r}}_{{\\mathrm{{raw}}}} = ({dx_raw:+.2f}L, {dy_raw:+.2f}L)$, "
        f"$|\\Delta\\mathbf{{r}}_{{\\mathrm{{raw}}}}| = {r_raw:.3f}L$.  \n"
        f"**Minimum-image displacement:** $\\Delta\\mathbf{{r}}_{{\\mathrm{{mic}}}} = ({dx_mic:+.2f}L, {dy_mic:+.2f}L)$, "
        f"$|\\Delta\\mathbf{{r}}_{{\\mathrm{{mic}}}}| = {r_mic:.3f}L$ (image shift $\\mathbf{{n}} = ({-int(shift_x):+d}, {-int(shift_y):+d})$).  \n"
        f"**Images of $j$ inside $r_c = {rc:.2f}L$:** **{images_in_rc}** "
        + (
            "⚠️ **Cutoff exceeds $L/2$ — sphere extends outside the yellow Minimum Image Wigner-Seitz square and can double-count periodic images!**"
            if overlap_error
            else "✓ **Strictly inside the yellow $L \\times L$ minimum-image domain ($r_c < L/2$).**"
        )
    )


def render_verlet_skin_optimizer(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot per-step CPU cost vs skin depth Delta/rc showing force evaluation vs list rebuild trade-off."""
    n_atoms = int(_val(controls.get("n_atoms"), 2000))
    temp_star = float(_val(controls.get("temp_star"), 1.0))

    delta_ratio = np.linspace(0.03, 0.65, 200)  # Delta / rc
    rc = 2.5
    rho = 0.85
    dt = 0.005

    # Thermal maximum particle speed scales as sqrt(T*); steps between rebuilds n_reb ~ Delta / (2 * v_max * dt)
    v_max = 2.6 * np.sqrt(temp_star)
    delta = delta_ratio * rc
    n_rebuild = np.maximum(1.0, delta / (2.0 * v_max * dt))

    # Cost per step:
    # 1) Force loop over Verlet table: proportional to N * (4/3 pi rho (rc + Delta)^3 / 2)
    c_pair = 1.0e-3  # relative microseconds per neighbor pair check
    neighbors_verlet = 0.5 * (4.0 / 3.0) * np.pi * rho * ((rc + delta) ** 3)
    cost_force = c_pair * n_atoms * neighbors_verlet

    # 2) Amortized O(N^2) rebuild cost per step: proportional to (0.5 * N^2) / n_rebuild
    c_rebuild_pair = 0.35e-3
    cost_rebuild = (c_rebuild_pair * 0.5 * (n_atoms**2)) / n_rebuild

    cost_total = cost_force + cost_rebuild

    idx_opt = int(np.argmin(cost_total))
    opt_ratio = float(delta_ratio[idx_opt])
    opt_cost = float(cost_total[idx_opt])
    opt_steps = float(n_rebuild[idx_opt])

    ax.plot(
        delta_ratio,
        cost_force,
        color=COLORS["secondary"],
        linestyle="--",
        linewidth=1.8,
        label=r"Force Step Cost $\propto N(r_c + \Delta)^3$",
    )
    ax.plot(
        delta_ratio,
        cost_rebuild,
        color=COLORS["quaternary"],
        linestyle=":",
        linewidth=1.8,
        label=r"Amortized Rebuild Cost $\propto N^2 / n_{\mathrm{reb}}(\Delta)$",
    )
    ax.plot(
        delta_ratio,
        cost_total,
        color=COLORS["primary"],
        linewidth=2.6,
        label="Total CPU Time / Step",
    )

    ax.scatter([opt_ratio], [opt_cost], color=COLORS["accent"], s=95, zorder=5)
    ax.axvline(
        opt_ratio,
        color=COLORS["accent"],
        linestyle="-.",
        linewidth=1.4,
        label=f"Optimal $\\Delta^*/r_c = {opt_ratio:.2f}$",
    )

    ax.set_xlim(0.03, 0.65)
    ax.set_ylim(0, max(float(np.percentile(cost_total, 92)) * 1.15, opt_cost * 2.2))
    ax.set_xlabel(r"Normalized Skin Thickness $\Delta / r_c$")
    ax.set_ylabel("Normalized CPU Time per MD Step")
    ax.set_title(f"Verlet Neighbor List Skin Optimization ($N={n_atoms}$, $T^*={temp_star:.1f}$)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Optimal Skin Depth at $N={n_atoms}, T^*={temp_star:.1f}$:** $\\Delta^* / r_c = {opt_ratio:.2f}$ "
        f"($\\Delta^* = {opt_ratio * rc:.2f}\\sigma$), rebuilding every **$n_{{\\mathrm{{reb}}}} \\approx {opt_steps:.0f}$ steps** "
        f"when $d_1 + d_2 \\ge \\Delta$. Higher $N$ or higher $T^*$ shifts $\\Delta^*$ to larger values to amortize expensive rebuilds."
    )


def render_cell_linked_list(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize a 2D 3x3 Cell Linked-List grid, its 9-cell stencil, and the head[c] / list[i] pointer chain."""
    selected_cell = int(_val(controls.get("selected_cell"), 4))
    use_newton_half = bool(int(_val(controls.get("use_newton_half"), 0)))

    # 3x3 grid of cells c = 0..8 (row-major: cx = c % 3, cy = c // 3)
    # Fixed deterministic set of 12 particles (0-indexed: 0..11)
    particles = [
        (0.35, 0.45),  # p0 in cell 0
        (0.75, 0.25),  # p1 in cell 0
        (1.45, 0.60),  # p2 in cell 1
        (2.30, 0.50),  # p3 in cell 2
        (0.40, 1.55),  # p4 in cell 3
        (1.25, 1.35),  # p5 in cell 4
        (1.70, 1.65),  # p6 in cell 4
        (1.45, 1.80),  # p7 in cell 4
        (2.55, 1.40),  # p8 in cell 5
        (0.60, 2.40),  # p9 in cell 6
        (1.50, 2.55),  # p10 in cell 7
        (2.35, 2.30),  # p11 in cell 8
    ]

    head = [-1] * 9
    linked_list = [-1] * len(particles)
    cell_of_particle = [0] * len(particles)

    for pid, (px, py) in enumerate(particles):
        cx = min(2, max(0, int(px)))
        cy = min(2, max(0, int(py)))
        c_idx = cy * 3 + cx
        cell_of_particle[pid] = c_idx
        linked_list[pid] = head[c_idx]
        head[c_idx] = pid

    sel_cx = selected_cell % 3
    sel_cy = selected_cell // 3

    # Determine active neighbor cells (all 9 in 2D, or 5 half-stencil with Newton's 3rd law)
    half_offsets = [(0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    all_offsets = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
    active_offsets = half_offsets if use_newton_half else all_offsets

    neighbor_cells = set()
    for dx, dy in active_offsets:
        ncx = (sel_cx + dx) % 3
        ncy = (sel_cy + dy) % 3
        neighbor_cells.add(ncy * 3 + ncx)

    for c_idx in range(9):
        cx = c_idx % 3
        cy = c_idx // 3
        if c_idx == selected_cell:
            fc = "#DBEAFE"
            ec = COLORS["primary"]
            lw = 2.4
        elif c_idx in neighbor_cells:
            fc = "#ECFDF5"
            ec = COLORS["good"]
            lw = 1.6
        else:
            fc = "#F8FAFC"
            ec = "#CBD5E1"
            lw = 1.0

        rect = patches.Rectangle((cx, cy), 1.0, 1.0, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=1)
        ax.add_patch(rect)
        ax.text(
            cx + 0.06,
            cy + 0.86,
            f"Cell {c_idx}\nhead={head[c_idx]}",
            fontsize=8.0,
            fontweight="bold",
            color=COLORS["text"],
            va="top",
            zorder=3,
        )

    # Plot particles and linked-list arrows within each cell
    for c_idx in range(9):
        curr = head[c_idx]
        while curr != -1:
            px, py = particles[curr]
            nxt = linked_list[curr]
            is_sel = c_idx == selected_cell
            ax.scatter(
                [px],
                [py],
                s=170 if is_sel else 125,
                color=COLORS["accent"] if is_sel else COLORS["primary"],
                edgecolors="white",
                linewidths=1.3,
                zorder=4,
            )
            ax.text(
                px,
                py,
                str(curr),
                color="white",
                fontsize=8.0,
                fontweight="bold",
                ha="center",
                va="center",
                zorder=5,
            )
            if nxt != -1:
                nx_p, ny_p = particles[nxt]
                ax.annotate(
                    "",
                    xy=(nx_p, ny_p),
                    xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", color=COLORS["accent"] if is_sel else COLORS["secondary"], lw=1.8),
                    zorder=4,
                )
            curr = nxt

    # Build chain string for selected cell
    chain = []
    curr = head[selected_cell]
    while curr != -1:
        chain.append(str(curr))
        curr = linked_list[curr]
    chain_str = " → ".join(chain + ["-1 (END)"])

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(["$c_x=0$", "$c_x=1$", "$c_x=2$"])
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(["$c_y=0$", "$c_y=1$", "$c_y=2$"])
    stencil_label = "5-Cell Newton Half-Stencil" if use_newton_half else "Full 9-Cell Stencil (2D)"
    ax.set_title(f"Cell Linked-List Method: Cell {selected_cell} ({stencil_label})")

    return (
        f"**Selected Cell {selected_cell} Pointer Traversal:** `head[{selected_cell}] = {head[selected_cell]}` $\\rightarrow$ `{chain_str}`.  \n"
        f"**Full `head` array (size $M^2=9$):** `{head}`  \n"
        f"**Full `list` array (size $N=12$):** `{linked_list}` — requires strictly **$O(N + M^d)$ static integer memory**!"
    )
