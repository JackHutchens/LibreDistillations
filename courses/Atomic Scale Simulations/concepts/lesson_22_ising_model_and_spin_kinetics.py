"""Interactive concept visualizations for Lesson 22: The Ising Model & Spin Kinetics."""

from __future__ import annotations
from typing import Any, Mapping
import matplotlib.pyplot as plt
import numpy as np
from engine.plot_style import COLORS


def render_ising_magnetization_pdf(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Interactive finite-box Ising magnetization PDF P_L(m) and free energy barrier F(m)."""
    t_ratio = float(controls.get("temp_ratio", 0.85))
    box_l = int(controls.get("box_size", 16))
    h_field = float(controls.get("magnetic_field", 0.0))
    view_mode = str(controls.get("view_mode", "pdf"))

    m = np.linspace(-0.995, 0.995, 600)
    n_spins = float(box_l**2)

    # Effective finite-size Landau-Onsager free energy density f(m) in 2D
    # For T < Tc, spontaneous magnetization scales as (1 - T/Tc)^(1/8) in 2D
    if t_ratio < 1.0:
        m_spont = (1.0 - t_ratio**2) ** 0.125
        # Effective barrier height scales as 2 * sigma(T) * L^(d-1) = 2 * sigma * L in 2D
        sigma_int = 1.76 * (1.0 - t_ratio)
        barrier_total = 2.0 * sigma_int * float(box_l)
        # Construct smooth double-well free energy matching +/- m_spont and interfacial flat top
        f_well = barrier_total * ((m**2 - m_spont**2) / max(0.08, m_spont**2)) ** 2
    else:
        m_spont = 0.0
        barrier_total = 0.0
        chi_inv = max(0.06, 1.8 * (t_ratio - 1.0) + 1.5 / (float(box_l) ** 1.75))
        f_well = 0.5 * n_spins * chi_inv * (m**2) + 0.12 * n_spins * (m**4)

    # External magnetic field tilts the free energy by -N * h * m / (k_B T)
    beta_eff = 1.0 / max(0.25, t_ratio)
    beta_f_total = f_well - (float(box_l) * 3.5) * beta_eff * h_field * m
    beta_f_total -= np.min(beta_f_total)

    p_m = np.exp(-np.clip(beta_f_total, 0.0, 60.0))
    dm = float(m[1] - m[0])
    p_m /= max(1e-12, np.sum(p_m) * dm)

    mean_m = float(np.sum(m * p_m) * dm)
    mean_abs_m = float(np.sum(np.abs(m) * p_m) * dm)

    if view_mode == "free_energy":
        ax.plot(
            m,
            beta_f_total,
            color=COLORS["primary"],
            linewidth=2.6,
            label=r"Dimensionless Free Energy $\beta F_L(m) = -\ln P_L(m)$",
        )
        if t_ratio < 1.0 and abs(h_field) < 0.05:
            ax.axvline(-m_spont, color=COLORS["spin_down"], linestyle="--", linewidth=1.4, label=f"$-m_0 = {-m_spont:.2f}$")
            ax.axvline(m_spont, color=COLORS["spin_up"], linestyle="--", linewidth=1.4, label=f"$+m_0 = +{m_spont:.2f}$")
            ax.annotate(
                f"Interfacial Barrier\n$\\Delta(\\beta F) \\approx 2\\beta\\sigma L = {barrier_total:.1f}$",
                xy=(0.0, barrier_total),
                xytext=(-0.42, barrier_total * 1.22 + 1.0),
                fontsize=8.8,
                fontweight="bold",
                color=COLORS["danger"],
                arrowprops={"arrowstyle": "->", "color": COLORS["danger"], "lw": 1.2},
            )
        ax.set_ylabel(r"Free Energy Barrier $\beta \Delta F_L(m)$")
        ax.set_ylim(-0.5, max(12.0, barrier_total * 1.65 + 2.0))
    else:
        ax.plot(
            m,
            p_m,
            color=COLORS["primary"],
            linewidth=2.6,
            label=f"$P_L(m)$ ($T/T_c={t_ratio:.2f},\\; L={box_l},\\; h={h_field:+.2f}$)",
        )
        ax.fill_between(m, 0.0, p_m, where=(m >= 0), color=COLORS["spin_up"], alpha=0.18, label="Up-spin domain ($m > 0$)")
        ax.fill_between(m, 0.0, p_m, where=(m < 0), color=COLORS["spin_down"], alpha=0.18, label="Down-spin domain ($m < 0$)")
        ax.axvline(mean_m, color=COLORS["danger"], linestyle=":", linewidth=1.8, label=f"$\\langle m \\rangle = {mean_m:+.2f}$")
        ax.axvline(mean_abs_m, color=COLORS["good"], linestyle="--", linewidth=1.8, label=f"$\\langle |m| \\rangle = {mean_abs_m:.2f}$")
        ax.set_ylabel(r"Probability Density $P_L(m)$")
        ax.set_ylim(0.0, max(2.5, float(np.max(p_m)) * 1.18))

    ax.set_xlim(-1.0, 1.0)
    ax.set_xlabel(r"Magnetization per spin $m = \frac{1}{N}\sum_i s_i$")
    ax.set_title(f"Finite-Size Ising Order Parameter Distribution ($T/T_c = {t_ratio:.2f}, L = {box_l}$)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    regime_desc = (
        f"Below $T_c$ ($T/T_c = {t_ratio:.2f}$), $P_L(m)$ splits into two broken-symmetry peaks near $\\pm m_0 = \\pm {m_spont:.2f}$ "
        f"separated by an interfacial slab barrier $\\Delta(\\beta F) \\approx 2\\beta\\sigma L \\approx {barrier_total:.1f}$. "
        f"At $h = 0$, exact symmetry forces $\\langle m \\rangle = {mean_m:+.2f}$, so simulations measure $\\langle |m| \\rangle = {mean_abs_m:.2f}$."
        if t_ratio < 1.0
        else f"At or above $T_c$ ($T/T_c = {t_ratio:.2f}$), the system is paramagnetic with a single peak centered near $m = {mean_m:+.2f}$ "
        f"whose variance $\\langle m^2 \\rangle - \\langle m \\rangle^2 = k_B T \\chi / L^2$ narrows as $L^{{-1}}$."
    )
    return f"**Magnetization Distribution Analysis:** {regime_desc}"


def render_glauber_vs_kawasaki_coarsening(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare non-conserved Glauber spin-flip coarsening vs. conserved Kawasaki spin-exchange spinodal decomposition."""
    dyn_mode = str(controls.get("dynamics_mode", "kawasaki"))
    quench_time = int(controls.get("quench_time", 40))
    init_mag = float(controls.get("initial_magnetization", 0.15))

    grid_n = 48
    rng = np.random.default_rng(101)
    raw_noise = rng.normal(loc=init_mag, scale=1.0, size=(grid_n, grid_n))

    # Spectral coarsening filter in Fourier space to deterministically model Allen-Cahn (t^1/2) vs Cahn-Hilliard (t^1/3)
    kx = np.fft.fftfreq(grid_n) * 2.0 * np.pi
    ky = np.fft.fftfreq(grid_n) * 2.0 * np.pi
    kx_g, ky_g = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx_g**2 + ky_g**2

    f_hat = np.fft.fft2(raw_noise - np.mean(raw_noise))

    if dyn_mode == "glauber":
        # Non-conserved Model A (Allen-Cahn): L(t) ~ t^(1/2), low-k modes grow, total magnetization drifts toward sign(init_mag)
        ell_t = 1.0 + 0.55 * (float(quench_time) ** 0.5)
        envelope = np.exp(-0.5 * k2 * (ell_t**2))
        field = np.real(np.fft.ifft2(f_hat * envelope))
        field /= max(1e-8, np.std(field))
        drift_bias = init_mag * (1.0 + 0.045 * quench_time)
        spins = np.where(field + drift_bias >= 0.0, 1.0, -1.0)
        z_exp = "1/2"
        law_name = "Allen–Cahn Non-Conserved Domain Growth ($R(t) \\propto t^{1/2}$)"
    else:
        # Conserved Model B (Kawasaki / Cahn-Hilliard spinodal decomposition): L(t) ~ t^(1/3), k=0 mode strictly conserved!
        ell_t = 1.0 + 0.42 * (float(quench_time) ** (1.0 / 3.0))
        k_peak = 1.25 / ell_t
        # Bicontinuous ring spectrum characteristic of conserved spinodal decomposition
        envelope = (k2 / (k2 + k_peak**2 + 1e-6)) * np.exp(-0.5 * k2 * (ell_t**2))
        field = np.real(np.fft.ifft2(f_hat * envelope))
        # Strictly conserve initial magnetization fraction by thresholding at exact quantile!
        up_frac_target = np.clip(0.5 * (1.0 + init_mag), 0.05, 0.95)
        thresh = np.quantile(field, 1.0 - up_frac_target)
        spins = np.where(field >= thresh, 1.0, -1.0)
        z_exp = "1/3"
        law_name = "Lifshitz–Slyozov–Wagner Conserved Spinodal Decomposition ($R(t) \\propto t^{1/3}$)"

    actual_m = float(np.mean(spins))
    actual_n = 0.5 * (actual_m + 1.0)

    ax.imshow(
        spins,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        origin="lower",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{'Glauber Spin-Flip (Non-Conserved)' if dyn_mode == 'glauber' else 'Kawasaki Spin-Exchange (Conserved)'} "
        f"at $t = {quench_time}$ MCS ($m = {actual_m:+.2f}$)"
    )

    # Inset showing domain size R(t) scaling comparison t^(1/2) vs t^(1/3)
    ax_in = ax.inset_axes([0.56, 0.06, 0.40, 0.34])
    t_arr = np.linspace(1.0, 100.0, 120)
    r_glaub = 1.0 + 0.55 * (t_arr**0.5)
    r_kawa = 1.0 + 0.42 * (t_arr ** (1.0 / 3.0))
    ax_in.plot(t_arr, r_glaub, color=COLORS["primary"], linestyle="-" if dyn_mode == "glauber" else "--", linewidth=1.8, label="$t^{1/2}$ (Flip)")
    ax_in.plot(t_arr, r_kawa, color=COLORS["accent"], linestyle="-" if dyn_mode == "kawasaki" else "--", linewidth=1.8, label="$t^{1/3}$ (Exch)")
    ax_in.scatter([quench_time], [ell_t], color=COLORS["danger"], s=28, zorder=5)
    ax_in.set_xlabel("$t$ (MCS)", fontsize=7.5)
    ax_in.set_ylabel("$R_{\\mathrm{dom}}(t)$", fontsize=7.5)
    ax_in.tick_params(labelsize=7.0)
    ax_in.legend(loc="upper left", fontsize=6.8)

    return (
        f"**Coarsening Readout ({law_name}):** At quench time $t = {quench_time}$ MCS ($T = 0.6\\,T_c$), "
        f"the characteristic domain length scale is $R(t) \\approx {ell_t:.2f}\\,a \\propto t^{{{z_exp}}}$. "
        f"The current magnetization is $m = {actual_m:+.2f}$ (alloy A-atom concentration $\\langle n \\rangle = (m+1)/2 = {actual_n:.2f}$). "
        f"Notice how **Glauber spin flips** allow the majority phase to consume the minority phase ($m \\to \\pm 1$), whereas "
        f"**Kawasaki nearest-neighbor spin exchanges** strictly conserve $m = {init_mag:+.2f}$ and form an interconnected "
        f"bicontinuous **spinodal decomposition** microstructure!"
    )


def render_swendsen_wang_clusters(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize Swendsen-Wang / Wolff Fortuin-Kasteleyn bond probability p = 1 - exp(-2*beta*J) and dynamic exponent z."""
    t_over_tc = float(controls.get("t_over_tc", 1.00))
    box_l = int(controls.get("lattice_size", 32))

    # Exact 2D square-lattice critical coupling: K_c = J/(k_B T_c) = 0.5 * ln(1 + sqrt(2)) = 0.4406868
    kc_2d = 0.5 * np.log(1.0 + np.sqrt(2.0))
    beta_j = kc_2d / max(0.2, t_over_tc)

    # Fortuin-Kasteleyn bond activation probability between aligned neighbors: p = 1 - exp(-2 * beta * J)
    p_bond = 1.0 - np.exp(-2.0 * beta_j)
    p_crit = 1.0 - np.exp(-2.0 * kc_2d)  # = sqrt(2)/(1+sqrt(2)) = 2 - sqrt(2) = 0.585786

    t_grid = np.linspace(0.35, 2.2, 300)
    bj_grid = kc_2d / t_grid
    p_grid = 1.0 - np.exp(-2.0 * bj_grid)

    ax.plot(
        t_grid,
        p_grid,
        color=COLORS["primary"],
        linewidth=2.6,
        label=r"FK Bond Probability $p_{\mathrm{bond}} = 1 - e^{-2\beta J}$",
    )
    ax.axvline(
        1.0,
        color=COLORS["danger"],
        linestyle="--",
        linewidth=1.6,
        label=f"Critical Point $T = T_c$ ($p_c = 2 - \\sqrt{{2}} \\approx {p_crit:.3f}$)",
    )
    ax.axhline(p_crit, color=COLORS["muted"], linestyle=":", linewidth=1.3)
    ax.scatter([t_over_tc], [p_bond], color=COLORS["accent"], s=80, zorder=6)

    # Compare autocorrelation time tau_int ~ L^z at T_c on secondary y-axis
    z_metropolis = 2.17
    z_sw = 0.25
    tau_metro = (float(box_l) ** z_metropolis) * np.exp(-2.2 * (t_over_tc - 1.0) ** 2)
    tau_sw = (float(box_l) ** z_sw) * np.exp(-0.4 * (t_over_tc - 1.0) ** 2)
    speedup = tau_metro / max(1.0, tau_sw)

    ax.annotate(
        f"$T/T_c = {t_over_tc:.2f}$\n$\\beta J = {beta_j:.3f}$\n$p_{{\\mathrm{{bond}}}} = {p_bond:.3f}$\n"
        f"Cluster Speedup ($L={box_l}$): $\\mathbf{{{speedup:.0f}\\times}}$",
        xy=(t_over_tc, p_bond),
        xytext=(min(1.52, t_over_tc + 0.15), min(0.92, max(0.28, p_bond + 0.08))),
        fontsize=8.6,
        fontweight="bold",
        color=COLORS["text"],
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F8FAFC", "ec": COLORS["accent"], "lw": 1.2},
        arrowprops={"arrowstyle": "->", "color": COLORS["accent"], "lw": 1.3},
    )

    ax.set_xlim(0.35, 2.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel(r"Reduced Temperature $T / T_c$")
    ax.set_ylabel(r"Fortuin–Kasteleyn Bond Activation Probability $p_{\mathrm{bond}}$")
    ax.set_title("Swendsen–Wang & Wolff Cluster Bond Probability $p = 1 - e^{-2\\beta J}$")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="lower left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Swendsen–Wang / Wolff Cluster Readout ($T/T_c = {t_over_tc:.2f}$, $L = {box_l}$):** "
        f"At coupling $\\beta J = J/(k_B T) = {beta_j:.3f}$, aligned nearest-neighbor spins ($s_i = s_j$) are linked by "
        f"Fortuin–Kasteleyn bonds with exact probability $p_{{\\mathrm{{bond}}}} = 1 - e^{{-2\\beta J}} = {p_bond:.3f}$. "
        f"Right at $T = T_c$, $p_{{\\mathrm{{bond}}}}(T_c) = 1 - e^{{-2 K_c}} = 2 - \\sqrt{{2}} \\approx 0.586$, where bonded clusters "
        f"percolate with the exact Ising correlation length $\\xi$. Flipping entire clusters with acceptance probability $A = 1$ "
        f"reduces the critical autocorrelation time from $\\tau_{{\\mathrm{{Metro}}}} \\sim L^{{2.17}} \\approx {tau_metro:.0f}$ sweeps "
        f"to $\\tau_{{\\mathrm{{SW}}}} \\sim L^{{0.25}} \\approx {tau_sw:.1f}$ sweeps—a **${speedup:.0f}\\times$ speedup**!"
    )
