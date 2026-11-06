"""Interactive concept visualizations for Lesson 24: Kinetic Monte Carlo (KMC) & Rare-Event Dynamics."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def render_bkl_event_selector_and_clock(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize the BKL (N-Fold Way) cumulative rate bar and Poisson waiting-time distribution."""
    E1 = float(controls.get("E1_eV", 0.45))
    E2 = float(controls.get("E2_eV", 0.52))
    E3 = float(controls.get("E3_eV", 0.60))
    T_K = float(controls.get("T_K", 500.0))
    u1 = float(controls.get("u1_draw", 0.62))
    view_mode = str(controls.get("view_mode", "catalog"))

    kB_eV = 8.617333262e-5  # eV / K
    nu0 = 1.0e13  # attempt frequency in s^-1
    kT = kB_eV * T_K

    barriers = np.array([E1, E2, E3], dtype=float)
    multiplicities = np.array([2, 4, 2], dtype=int)  # class multiplicities N_c
    W_single = nu0 * np.exp(-barriers / kT)
    R_class = multiplicities * W_single
    R_tot = float(np.sum(R_class))
    probs = R_class / R_tot
    cum_probs = np.cumsum(probs)

    # Determine selected class from u1
    selected_idx = int(np.searchsorted(cum_probs, u1, side="left"))
    selected_idx = min(max(selected_idx, 0), 2)

    mean_dt_ns = 1.0e9 / R_tot  # in nanoseconds
    u2_fixed = 0.37
    sampled_dt_ns = -np.log(u2_fixed) * mean_dt_ns

    class_colors = [COLORS["community_a"], COLORS["community_b"], COLORS["community_c"]]
    class_labels = [
        f"Class 1 ($N_1=2$, $\\Delta E_1={E1:.2f}$ eV)",
        f"Class 2 ($N_2=4$, $\\Delta E_2={E2:.2f}$ eV)",
        f"Class 3 ($N_3=2$, $\\Delta E_3={E3:.2f}$ eV)",
    ]

    if view_mode == "waiting_time":
        t_max = mean_dt_ns * 4.2
        t_grid = np.linspace(0.0, t_max, 220)
        # PDF in terms of u = t / mean_dt_ns: p(t) = (1/mean_dt_ns) * exp(-t / mean_dt_ns)
        pdf_vals = (1.0 / mean_dt_ns) * np.exp(-t_grid / mean_dt_ns)
        ax.plot(t_grid, pdf_vals, color=COLORS["primary"], linewidth=2.5, label="Poisson Waiting PDF $p(\\Delta t) = R_{\\text{tot}} e^{-R_{\\text{tot}} \\Delta t}$")
        ax.fill_between(t_grid, 0.0, pdf_vals, color=COLORS["primary"], alpha=0.14)
        ax.axvline(mean_dt_ns, color=COLORS["good"], linestyle="--", linewidth=1.6, label=f"Mean $\\langle \\Delta t \\rangle = 1/R_{{\\text{{tot}}}} = {mean_dt_ns:.2f}$ ns")
        ax.axvline(sampled_dt_ns, color=COLORS["accent"], linestyle="-", linewidth=2.0, label=f"Draw $\\Delta t = -\\ln(0.37)/R_{{\\text{{tot}}}} = {sampled_dt_ns:.2f}$ ns")
        ax.set_xlim(0.0, t_max)
        ax.set_ylim(0.0, max(pdf_vals) * 1.15)
        ax.set_xlabel("Stochastic Residence Time Increment $\\Delta t$ (ns)")
        ax.set_ylabel("Probability Density $p(\\Delta t)$ ($\\text{ns}^{-1}$)")
        ax.set_title(f"BKL Poisson Escape Clock at $T = {T_K:.0f}$ K ($R_{{\\text{{tot}}}} = {R_tot:.2e}\\,\\text{{s}}^{{-1}}$)")
        ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
        ax.legend(loc="upper right", fontsize=8.5)
    else:
        # Plot cumulative interval bar and class selection probabilities
        left = 0.0
        for idx in range(3):
            width = probs[idx]
            edge_lw = 2.6 if idx == selected_idx else 1.0
            edge_col = COLORS["danger"] if idx == selected_idx else "white"
            ax.barh(
                y=1.0,
                width=width,
                left=left,
                height=0.45,
                color=class_colors[idx],
                edgecolor=edge_col,
                linewidth=edge_lw,
                label=f"{class_labels[idx]}: {100.0*width:.1f}%",
            )
            if width > 0.08:
                ax.text(
                    left + 0.5 * width,
                    1.0,
                    f"$C_{idx+1}$\n({100.0*width:.1f}%)",
                    ha="center",
                    va="center",
                    color="white",
                    fontweight="bold",
                    fontsize=9.0,
                )
            left += width

        # Mark random draw u1
        ax.axvline(u1, color=COLORS["danger"], linestyle="--", linewidth=2.0, label=f"Uniform Draw $u_1 = {u1:.2f} \\to$ Class {selected_idx+1}")
        ax.scatter([u1], [1.0], color=COLORS["danger"], s=80, zorder=6)

        # Also plot individual class rates as horizontal bars below
        for idx in range(3):
            ax.barh(
                y=0.25 - 0.25 * idx,
                width=probs[idx],
                left=0.0,
                height=0.18,
                color=class_colors[idx],
                alpha=0.85,
            )
            ax.text(
                min(probs[idx] + 0.02, 0.88),
                0.25 - 0.25 * idx,
                f"$R_{idx+1} = N_{idx+1}W_{idx+1} = {R_class[idx]:.2e}\\,\\text{{s}}^{{-1}}$",
                va="center",
                fontsize=8.5,
                color=COLORS["text"],
            )

        ax.set_xlim(0.0, 1.05)
        ax.set_ylim(-0.55, 1.55)
        ax.set_yticks([-0.25, 0.0, 0.25, 1.0])
        ax.set_yticklabels(["Class 3", "Class 2", "Class 1", "Cumulative\n$S_k / R_{\\text{tot}}$"])
        ax.set_xlabel("Normalized Cumulative Rate Coordinate $S_k / R_{\\text{tot}} \\in [0, 1]$")
        ax.set_title(f"BKL ($N$-Fold Way) Event Selection at $T = {T_K:.0f}$ K (100% Rejection-Free)")
        ax.grid(True, axis="x", color="#E2E8F0", linestyle="--", linewidth=0.7)
        ax.legend(loc="upper right", fontsize=8.0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compare to naive vibration step acceptance probability exp(-E_min / kT)
    p_naive = float(np.exp(-min(barriers) / kT))
    return (
        f"**BKL Step Summary ($T = {T_K:.0f}$ K):** Total escape rate $R_{{\\text{{tot}}}} = \\sum_c N_c W_c = {R_tot:.2e}\\,\\text{{s}}^{{-1}}$, "
        f"mean residence time $\\langle \\Delta t \\rangle = 1/R_{{\\text{{tot}}}} = {mean_dt_ns:.2f}$ ns. "
        f"Uniform draw $u_1 = {u1:.2f}$ lands in **Class {selected_idx+1}** ($P_{selected_idx+1} = {100.0*probs[selected_idx]:.1f}\\%$). "
        f"A direct MD/Metropolis vibrational trial would accept only $\\sim e^{{-\\Delta E_{{\\min}}/k_B T}} \\approx {p_naive:.2e}$ of attempts, "
        f"whereas BKL executes a move on **100% of steps**!"
    )


def render_adsorption_desorption_kmc(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare live stochastic KMC surface coverage trajectory theta(t) against the exact Master Equation ODE."""
    r_A = float(controls.get("r_A", 1.5))
    r_D = float(controls.get("r_D", 0.5))
    N_sites = int(controls.get("N_sites", 50))
    theta_0 = float(controls.get("theta_0", 0.10))

    theta_eq = r_A / (r_A + r_D)
    rate_sum = r_A + r_D
    tau_relax = 1.0 / rate_sum

    t_end = 5.0 * tau_relax
    t_analytic = np.linspace(0.0, t_end, 200)
    theta_analytic = theta_eq + (theta_0 - theta_eq) * np.exp(-rate_sum * t_analytic)

    # Deterministic-seed stochastic BKL simulation on N_sites
    rng = np.random.default_rng(seed=10485 + int(N_sites * 13 + r_A * 100 + r_D * 37))
    n_occ = int( round(theta_0 * N_sites) )
    t_curr = 0.0

    t_traj = [0.0]
    theta_traj = [n_occ / N_sites]
    dt_weights = []
    theta_states = []

    max_steps = 1200
    for _ in range(max_steps):
        if t_curr >= t_end:
            break
        n_empty = N_sites - n_occ
        R_ads = n_empty * r_A
        R_des = n_occ * r_D
        R_tot = R_ads + R_des
        if R_tot <= 1e-12:
            break
        u1 = float(rng.uniform(1e-9, 1.0))
        u2 = float(rng.uniform(1e-9, 1.0))
        dt = -np.log(u2) / R_tot

        # Record residence-time weight before state change
        dt_clipped = min(dt, max(0.0, t_end - t_curr))
        if t_curr > 1.5 * tau_relax and dt_clipped > 0:
            dt_weights.append(dt_clipped)
            theta_states.append(n_occ / N_sites)

        t_curr += dt
        if u1 * R_tot <= R_ads:
            n_occ += 1
        else:
            n_occ -= 1
        t_traj.append(t_curr)
        theta_traj.append(n_occ / N_sites)

    ax.step(t_traj, theta_traj, where="post", color=COLORS["primary"], linewidth=1.8, alpha=0.88, label=f"Stochastic KMC ($N_s = {N_sites}$ sites)")
    ax.plot(t_analytic, theta_analytic, color=COLORS["accent"], linewidth=2.5, label="Exact Master Eq: $\\theta(t) = \\theta_{\\text{eq}} + (\\theta_0 - \\theta_{\\text{eq}})e^{-(r_A+r_D)t}$")
    ax.axhline(theta_eq, color=COLORS["good"], linestyle="--", linewidth=1.5, label=f"Steady State $\\theta(\\infty) = r_A/(r_A+r_D) = {theta_eq:.3f}$")

    ax.set_xlim(0.0, t_end)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xlabel("Physical Time $t$ ($\\text{s}$)")
    ax.set_ylabel("Surface Coverage Fraction $\\theta(t) = N_{\\text{occ}} / N_s$")
    ax.set_title(f"Surface Adsorption–Desorption KMC vs. Continuum Master Equation ($r_A={r_A:.1f}, r_D={r_D:.1f}$)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="best", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if dt_weights:
        time_weighted_avg = float(np.sum(np.array(theta_states) * np.array(dt_weights)) / np.sum(dt_weights))
    else:
        time_weighted_avg = theta_eq

    return (
        f"**Master Equation & Detailed Balance:** Analytic steady-state coverage is $\\theta(\\infty) = \\frac{{r_A}}{{r_A + r_D}} = {theta_eq:.3f}$ "
        f"(satisfying detailed balance $r_A(1-\\theta) = r_D\\theta$), with relaxation time $\\tau = \\frac{{1}}{{r_A + r_D}} = {tau_relax:.3f}$ s. "
        f"Post-equilibration residence-time weighted KMC average is $\\bar{{\\theta}} = \\frac{{\\sum \\theta_n \\Delta t_n}}{{\\sum \\Delta t_n}} = {time_weighted_avg:.3f}$."
    )


def render_superbasin_flickering_trap(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Illustrate the KMC superbasin flickering trap and how shallow barriers collapse the physical time step."""
    E_flicker = float(controls.get("E_flicker_eV", 0.20))
    E_escape = float(controls.get("E_escape_eV", 0.70))
    T_K = float(controls.get("T_K", 500.0))
    plot_view = str(controls.get("plot_view", "landscape"))

    kB_eV = 8.617333262e-5
    kT = kB_eV * T_K
    nu0 = 1.0e13

    W_flicker = nu0 * np.exp(-E_flicker / kT)
    W_escape = nu0 * np.exp(-E_escape / kT)
    R_tot = W_flicker + W_escape
    p_escape_per_step = W_escape / R_tot
    expected_flickers = W_flicker / W_escape
    dt_flicker_s = 1.0 / R_tot
    dt_coarse_s = 1.0 / (2.0 * W_escape)

    if plot_view == "clock_advance":
        steps = np.arange(1, 201)
        t_standard = steps * dt_flicker_s
        t_superbasin = steps * dt_coarse_s
        ax.semilogy(
            steps,
            t_standard,
            color=COLORS["danger"],
            linewidth=2.4,
            label=f"Standard KMC (Trapped Flickering): $\\Delta t \\approx {dt_flicker_s:.2e}$ s/step",
        )
        ax.semilogy(
            steps,
            t_superbasin,
            color=COLORS["good"],
            linewidth=2.4,
            label=f"Superbasin Coarse-Grained KMC: $\\Delta t_\\mathrm{{esc}} \\approx {dt_coarse_s:.2e}$ s/step",
        )
        ax.set_xlabel("Number of Executed KMC Steps $N_{\\text{steps}}$")
        ax.set_ylabel("Cumulative Simulated Physical Time $t$ (s)")
        ax.set_title(f"KMC Clock Stagnation in a Flickering Superbasin ($T = {T_K:.0f}$ K)")
        ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7)
        ax.legend(loc="center right", fontsize=8.2)
    else:
        # Plot double-well superbasin potential energy profile
        x = np.linspace(-2.2, 2.8, 300)
        # Construct smooth profile with two shallow wells at x = -0.5, +0.5 separated by E_flicker at x = 0,
        # and outer escape barriers of height E_escape at x = -1.5, +1.5
        V = np.zeros_like(x)
        for i, xi in enumerate(x):
            if abs(xi) <= 1.0:
                # Double well inside [-1, 1]: minima at +/-0.5 (V=0), barrier at 0 (V=E_flicker), rising toward +/-1
                u = xi / 0.5
                V[i] = E_flicker * ((u ** 2 - 1.0) ** 2)
            elif abs(xi) <= 2.0:
                # Outer barrier peaking at |x| = 1.5 with height E_escape
                s = (abs(xi) - 1.0) / 0.5
                # At |x|=1, V = 9*E_flicker; smoothly interpolate peak at |x|=1.5 to E_escape
                V_base = E_escape * np.sin(0.5 * np.pi * s) ** 2
                V_match = E_flicker * 9.0 * (1.0 - s) ** 2
                V[i] = max(V_base, V_match)
            else:
                s = (abs(xi) - 1.5) / 0.5
                V[i] = E_escape * np.exp(-2.2 * (s - 1.0) ** 2) if abs(xi) < 1.5 else E_escape * np.cos(0.5 * np.pi * min(1.0, (abs(xi) - 1.5) / 1.0)) ** 2

        # Clean analytical profile:
        # Minima at x = -0.55 and +0.55 at E = 0; inner saddle at x = 0 at E = E_flicker; outer saddles at x = +/-1.65 at E = E_escape
        x_pts = np.linspace(-2.4, 2.4, 350)
        # Use piecewise cosine/polynomial spline between critical points
        def smooth_step(u: np.ndarray) -> np.ndarray:
            u_c = np.clip(u, 0.0, 1.0)
            return 0.5 - 0.5 * np.cos(np.pi * u_c)

        V_clean = np.zeros_like(x_pts)
        ax_abs = np.abs(x_pts)
        m1 = ax_abs <= 0.55
        m2 = (ax_abs > 0.55) & (ax_abs <= 1.65)
        m3 = ax_abs > 1.65
        V_clean[m1] = E_flicker * (1.0 - smooth_step(ax_abs[m1] / 0.55))
        V_clean[m2] = E_flicker * 0.0 + E_escape * smooth_step((ax_abs[m2] - 0.55) / 1.10)
        V_clean[m3] = E_escape * (1.0 - 0.85 * smooth_step((ax_abs[m3] - 1.65) / 0.75))

        ax.plot(x_pts, V_clean, color=COLORS["primary"], linewidth=2.6, label="Potential Energy Surface $E(x)$")
        ax.fill_between(x_pts, 0.0, V_clean, where=(ax_abs <= 1.65), color=COLORS["highlight"], alpha=0.15, label="Superbasin Trap ($s_A \\leftrightarrow s_B$)")

        # Annotate barriers
        ax.annotate(
            f"Flicker Barrier\n$\\Delta E_{{\\text{{flicker}}}} = {E_flicker:.2f}$ eV",
            xy=(0.0, E_flicker),
            xytext=(0.0, E_flicker + 0.14),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["danger"],
            arrowprops={"arrowstyle": "->", "color": COLORS["danger"], "lw": 1.3},
        )
        ax.annotate(
            f"Escape Barrier\n$\\Delta E_{{\\text{{escape}}}} = {E_escape:.2f}$ eV",
            xy=(1.65, E_escape),
            xytext=(1.15, E_escape + 0.12),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=COLORS["good"],
            arrowprops={"arrowstyle": "->", "color": COLORS["good"], "lw": 1.3},
        )
        ax.text(-0.55, -0.07, "$s_A$", ha="center", fontweight="bold", fontsize=9.5, color=COLORS["text"])
        ax.text(0.55, -0.07, "$s_B$", ha="center", fontweight="bold", fontsize=9.5, color=COLORS["text"])

        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-0.12, max(E_escape + 0.28, 1.05))
        ax.set_xlabel("Reaction Coordinate $x$")
        ax.set_ylabel("Potential Energy $E(x)$ (eV)")
        ax.set_title(f"Superbasin Flickering Trap ($W_{{\\text{{flicker}}}}/W_{{\\text{{escape}}}} = {expected_flickers:,.1e}$ at $T={T_K:.0f}$ K)")
        ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
        ax.legend(loc="upper left", fontsize=8.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Superbasin Diagnosis ($T = {T_K:.0f}$ K):** $W_{{\\text{{flicker}}}} = {W_flicker:.2e}\\,\\text{{s}}^{{-1}}$ vs. "
        f"$W_{{\\text{{escape}}}} = {W_escape:.2e}\\,\\text{{s}}^{{-1}}$. "
        f"The system executes an average of **$W_{{\\text{{flicker}}}}/W_{{\\text{{escape}}}} \\approx {expected_flickers:,.0f}$ unproductive back-and-forth flickers** "
        f"(each advancing the clock by only $\\Delta t \\approx {dt_flicker_s:.2e}$ s) before a single productive escape over $\\Delta E_{{\\text{{escape}}}}$ occurs!"
    )
