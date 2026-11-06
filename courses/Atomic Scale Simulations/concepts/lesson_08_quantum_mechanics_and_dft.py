"""Interactive concept visualizations for Lesson 08: Quantum Mechanics and Density Functional Theory."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def render_variational_hydrogen(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render Hydrogen trial wavefunction radial density and variational energy components T(alpha), V(alpha), E(alpha)."""
    alpha = float(controls.get("alpha", 1.0))
    view_mode = str(controls.get("view_mode", "energy_landscape"))

    alpha_grid = np.linspace(0.2, 2.2, 300)
    t_grid = 0.5 * alpha_grid ** 2
    v_grid = -alpha_grid
    e_grid = t_grid + v_grid

    t_val = 0.5 * alpha ** 2
    v_val = -alpha
    e_val = t_val + v_val
    e_ev = e_val * 27.211386

    if view_mode == "radial_density":
        r = np.linspace(0.0, 6.0, 350)
        # Normalized trial wavefunction: Psi_alpha(r) = sqrt(alpha^3 / pi) * exp(-alpha * r)
        # Radial probability density P(r) = 4 * pi * r^2 * |Psi_alpha(r)|^2 = 4 * alpha^3 * r^2 * exp(-2 * alpha * r)
        p_trial = 4.0 * (alpha ** 3) * (r ** 2) * np.exp(-2.0 * alpha * r)
        p_exact = 4.0 * (1.0 ** 3) * (r ** 2) * np.exp(-2.0 * 1.0 * r)

        ax.plot(r, p_exact, color=COLORS["muted"], linestyle="--", linewidth=2.0, label="Exact Ground State ($\\alpha^* = 1.0\\,a_0^{-1}$)")
        ax.plot(r, p_trial, color=COLORS["primary"], linewidth=2.4, label=f"Trial Density $4\\pi r^2|\\Psi_\\alpha(r)|^2$ ($\\alpha = {alpha:.2f}\\,a_0^{{-1}}$)")
        ax.fill_between(r, 0.0, p_trial, color=COLORS["primary"], alpha=0.15)
        ax.axvline(1.0 / alpha, color=COLORS["accent"], linestyle=":", linewidth=1.4, label=f"Peak radius $r_{{\\max}} = 1/\\alpha = {1.0/alpha:.2f}\\,a_0$")
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(0.0, max(1.25, float(np.max(p_trial)) * 1.12))
        ax.set_xlabel("Radius $r / a_0$ (Bohr)")
        ax.set_ylabel("Radial Probability Density $P(r) = 4\\pi r^2|\\Psi_\\alpha|^2$")
        ax.set_title(f"Hydrogen Trial Wavefunction Localization ($\\alpha={alpha:.2f}\\,a_0^{{-1}}$)")
        ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
        ax.legend(loc="upper right", fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.plot(alpha_grid, t_grid, color=COLORS["kinetic"], linestyle="--", linewidth=2.0, label="Kinetic $T(\\alpha) = \\frac{1}{2}\\alpha^2$")
        ax.plot(alpha_grid, v_grid, color=COLORS["potential"], linestyle="-.", linewidth=2.0, label="Potential $V(\\alpha) = -\\alpha$")
        ax.plot(alpha_grid, e_grid, color=COLORS["total"], linewidth=2.5, label="Total $E(\\alpha) = \\frac{1}{2}\\alpha^2 - \\alpha$")
        ax.axhline(-0.5, color=COLORS["good"], linestyle=":", linewidth=1.2, label="Exact Ground State $E_0 = -0.50\\text{ Ha}$")
        ax.axvline(alpha, color=COLORS["quaternary"], linestyle=":", linewidth=1.3)
        ax.scatter([alpha, alpha, alpha], [t_val, v_val, e_val], color=[COLORS["kinetic"], COLORS["potential"], COLORS["total"]], s=65, zorder=5)
        ax.scatter([1.0], [-0.5], color=COLORS["good"], s=90, marker="*", zorder=6)

        ax.set_xlim(0.2, 2.2)
        ax.set_ylim(-2.3, 2.5)
        ax.set_xlabel("Variational Inverse Decay Length $\\alpha$ ($a_0^{-1}$)")
        ax.set_ylabel("Energy Expectation Value (Hartree, Ha)")
        ax.set_title("Variational Principle for Hydrogen: Competition of $T(\\alpha)$ and $V(\\alpha)$")
        ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
        ax.legend(loc="upper left", fontsize=8.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    virial_ratio = -v_val / max(1e-8, t_val)
    return (
        f"**At $\\alpha = {alpha:.2f}\\,a_0^{{-1}}$:** $T(\\alpha) = {t_val:+.3f}\\text{{ Ha}}$, "
        f"$V(\\alpha) = {v_val:+.3f}\\text{{ Ha}}$, and total variational energy "
        f"$E(\\alpha) = {e_val:+.3f}\\text{{ Ha}} = {e_ev:+.2f}\\text{{ eV}} \\ge E_0 = -0.500\\text{{ Ha}}$ "
        f"(virial ratio $-V/T = {virial_ratio:.2f}$, equal to $2.00$ at the exact ground state $\\alpha^* = 1$)."
    )


def render_kohn_sham_scf_loop(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render Kohn-Sham DFT self-consistent field (SCF) convergence and functional comparison (LDA, GGA, Hybrid)."""
    mixing_beta = float(controls.get("mixing_beta", 0.35))
    functional = str(controls.get("functional", "PBE_GGA"))

    iters = np.arange(1, 26)
    # Linear density mixing convergence model: n_{k+1} = (1 - beta)*n_k + beta*F[n_k]
    # Eigenvalue of dielectric response ~ -1.6; effective contraction factor q = |1 - beta * 2.2|
    q = abs(1.0 - mixing_beta * 2.15)
    dn_norm = 0.85 * (q ** (iters - 1))
    dn_norm = np.clip(dn_norm, 1e-10, 10.0)

    # Energy error scales quadratically with density error |delta n|^2 by the 2nd Hohenberg-Kohn variational theorem!
    de_variational = 1.2 * (dn_norm ** 2)
    de_variational = np.clip(de_variational, 1e-16, 100.0)

    func_props = {
        "LDA": ("LDA (Ceperley-Alder HEG)", -1.8, +12.0, -45.0),
        "PBE_GGA": ("GGA-PBE (Density + $|\\nabla n|$)", +0.8, -3.0, -38.0),
        "HSE_Hybrid": ("Hybrid HSE06 (25% Exact HF Exchange)", +0.2, -1.0, -8.0),
    }
    f_label, da_pct, ecoh_pct, egap_pct = func_props.get(functional, func_props["PBE_GGA"])

    ax.semilogy(iters, dn_norm, color=COLORS["primary"], marker="o", markersize=4.5, linewidth=2.0, label="Density residual $\\|\\rho^{(k+1)} - \\rho^{(k)}\\|_2$")
    ax.semilogy(iters, de_variational, color=COLORS["good"], marker="s", markersize=4.5, linestyle="--", linewidth=2.0, label="Variational energy error $|E[\\rho^{(k)}] - E_0| \\propto \\|\\delta\\rho\\|^2$")
    ax.axhline(1e-6, color=COLORS["danger"], linestyle=":", linewidth=1.2, label="SCF convergence threshold ($10^{-6}$ Ha)")

    ax.set_xlim(1, 25)
    ax.set_ylim(1e-12, 1e2)
    ax.set_xlabel("Kohn–Sham SCF Iteration $k$")
    ax.set_ylabel("Residual Norm / Energy Error (log scale)")
    ax.set_title(f"Kohn–Sham Self-Consistent Field Convergence ($\\beta={mixing_beta:.2f}$, {f_label})")
    ax.grid(True, which="both", color=COLORS["grid"], linestyle="--", linewidth=0.6)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    status = "converges rapidly" if q < 0.65 else ("oscillates/diverges (charge sloshing!)" if q >= 1.0 else "converges slowly")
    return (
        f"**SCF Mixing $\\beta = {mixing_beta:.2f}$:** Contraction factor $q = {q:.2f}$ ({status}). "
        f"Because $E[\\rho]$ is stationary at $\\rho_0$ (2nd Hohenberg–Kohn theorem), the total energy error scales quadratically as $\\mathcal{{O}}(\\|\\delta\\rho\\|^2)$. "
        f"**Selected Functional ({f_label}):** typical solid lattice constant error $\\approx {da_pct:+.1f}\\%$, cohesive energy error $\\approx {ecoh_pct:+.1f}\\%$, band-gap error $\\approx {egap_pct:+.0f}\\%$."
    )


def render_pseudopotential_smoothing(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render all-electron nodal wavefunction & Coulomb potential vs smooth nodeless pseudopotential inside cutoff rc."""
    rc = float(controls.get("rc", 1.6))
    view_mode = str(controls.get("view_mode", "real_space"))

    r = np.linspace(0.04, 4.5, 450)
    # Model a 3s valence wavefunction (e.g. Al or Si) with 2 radial nodes inside r < rc
    psi_ae = np.cos(3.8 * np.pi * r / (rc * 1.65)) * np.exp(-0.75 * r) * (r / (r + 0.15))
    # Smooth nodeless pseudo-wavefunction matching psi_ae exactly for r >= rc
    idx_rc = int(np.searchsorted(r, rc))
    val_rc = float(psi_ae[idx_rc])
    deriv_rc = float((psi_ae[idx_rc + 1] - psi_ae[idx_rc - 1]) / (r[idx_rc + 1] - r[idx_rc - 1]))

    # Construct smooth polynomial r^2 * (a + b * r^2) inside r < rc matching value and derivative at rc
    # Let u(r) = (r/rc)^2 * [c0 + c1 * ((r/rc)^2 - 1)]
    # At r = rc: u(rc) = c0 = val_rc
    # u'(rc) = (2/rc)*c0 + (2/rc)*c1 = deriv_rc => c1 = 0.5 * rc * deriv_rc - val_rc
    c0 = val_rc
    c1 = 0.5 * rc * deriv_rc - val_rc
    s = (r / rc) ** 2
    psi_ps_inner = s * (c0 + c1 * (s - 1.0))
    psi_ps = np.where(r < rc, psi_ps_inner, psi_ae)

    # Potentials (scaled for plotting alongside wavefunctions)
    v_ae = -1.8 / np.maximum(0.25, r)
    v_rc = -1.8 / rc
    v_ps = np.where(r < rc, v_rc * (1.0 + 0.35 * (1.0 - (r / rc) ** 2)), v_ae)

    # Plane-wave cutoff scaling: E_cut ~ (G_max)^2 ~ (pi / delta_x)^2
    # All-electron requires resolving inner core node spacing ~ rc / 4; pseudopotential only requires resolving rc
    ecut_ae_ry = 180.0
    ecut_ps_ry = 180.0 / ((rc / 0.42) ** 2)

    if view_mode == "fourier_cutoff":
        g_grid = np.linspace(0.2, 12.0, 300)
        # Fourier transform decay |psi(G)|^2
        ft_ae = np.exp(-0.18 * (g_grid ** 2)) + 0.18 * np.exp(-0.035 * ((g_grid - 5.5) ** 2))
        ft_ps = np.exp(-0.32 * (rc ** 1.3) * (g_grid ** 2))
        ax.semilogy(g_grid, np.maximum(1e-8, ft_ae), color=COLORS["primary"], linestyle="--", linewidth=2.1, label="All-Electron $|\\tilde{\\psi}_{\\mathrm{AE}}(G)|^2$ (core wiggles $\\Rightarrow$ high $G$)")
        ax.semilogy(g_grid, np.maximum(1e-8, ft_ps), color=COLORS["danger"], linewidth=2.4, label=f"Pseudo-Wavefunction $|\\tilde{{\\psi}}_{{\\mathrm{{PS}}}}(G)|^2$ ($r_c={rc:.2f}\\,a_0$)")
        g_cut = np.sqrt(ecut_ps_ry / 2.0)
        ax.axvline(g_cut, color=COLORS["good"], linestyle=":", linewidth=1.5, label=f"Required $G_{{\\max}} \\approx {g_cut:.1f}\\,a_0^{{-1}}$ ($E_{{\\mathrm{{cut}}}} \\approx {ecut_ps_ry:.1f}$ Ry)")
        ax.set_xlim(0.2, 12.0)
        ax.set_ylim(1e-6, 2.0)
        ax.set_xlabel("Reciprocal Wavevector Magnitude $G = |\\mathbf{k}+\\mathbf{G}|$ ($a_0^{-1}$)")
        ax.set_ylabel("Fourier Power Spectrum $|c(\\mathbf{G})|^2$ (log scale)")
        ax.set_title("Plane-Wave Basis Convergence: All-Electron vs. Pseudopotential")
        ax.grid(True, which="both", color=COLORS["grid"], linestyle="--", linewidth=0.6)
        ax.legend(loc="upper right", fontsize=8.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.axhline(0.0, color=COLORS["muted"], linewidth=0.9, linestyle=":")
        ax.axvline(rc, color=COLORS["total"], linewidth=1.5, linestyle="--", label=f"Core cutoff radius $r_c = {rc:.2f}\\,a_0$")
        ax.plot(r, psi_ae, color=COLORS["primary"], linestyle="--", linewidth=2.0, label="All-Electron $\\Psi_{\\mathrm{AE}}(r)$ (rapid core oscillations)")
        ax.plot(r, psi_ps, color=COLORS["danger"], linewidth=2.4, label="Pseudized $\\Psi_{\\mathrm{pseudo}}(r)$ (nodeless inside $r_c$)")
        ax.plot(r, v_ae, color=COLORS["tertiary"], linestyle="--", linewidth=1.8, alpha=0.85, label="Coulomb $V_{\\mathrm{AE}}(r) \\sim -Z/r$")
        ax.plot(r, v_ps, color=COLORS["accent"], linewidth=2.1, label="Pseudopotential $V_{\\mathrm{pseudo}}(r)$")
        ax.axvspan(0.0, rc, color=COLORS["grid"], alpha=0.35, label="Core region ($r < r_c$)")

        ax.set_xlim(0.0, 4.5)
        ax.set_ylim(-3.2, 1.3)
        ax.set_xlabel("Radial Distance $r$ ($a_0$)")
        ax.set_ylabel("Wavefunction $\\Psi(r)$ / Potential $V(r)$ (arb. units)")
        ax.set_title(f"Pseudopotential & Pseudo-Wavefunction Smoothing ($r_c = {rc:.2f}\\,a_0$)")
        ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
        ax.legend(loc="lower right", fontsize=8.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    tradeoff = (
        "softer potential (lower plane-wave cutoff $E_{\\text{cut}}$) but reduced transferability across chemical environments"
        if rc >= 1.75
        else (
            "harder potential (higher plane-wave cutoff $E_{\\text{cut}}$) with higher transferability"
            if rc <= 1.25
            else "balanced norm-conserving cutoff trading off $E_{\\text{cut}}$ and chemical transferability"
        )
    )
    return (
        f"**Cutoff $r_c = {rc:.2f}\\,a_0$:** Required plane-wave cutoff drops from $\\sim {ecut_ae_ry:.0f}\\text{{ Ry}}$ (all-electron) "
        f"to $\\sim {ecut_ps_ry:.1f}\\text{{ Ry}}$ ({tradeoff}). For $r \\ge r_c$, $\\Psi_{{\\text{{pseudo}}}}(r) \\equiv \\Psi_{{\\text{{AE}}}}(r)$ and $V_{{\\text{{pseudo}}}}(r) \\equiv -Z_{{\\text{{ion}}}}/r$."
    )


def render_bloch_bandgap_and_dft(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render nearly-free electron band gap opening E_gap = 2U at k = +/- pi/a or DFT vs GW semiconductor band gaps."""
    u_ev = float(controls.get("u_ev", 0.6))
    view_mode = str(controls.get("view_mode", "brillouin_zone"))

    if view_mode == "dft_bandgap_comparison":
        materials = ["InAs", "Ge", "Si", "GaAs", "AlP", "GaN", "Diamond"]
        lda_gaps = [-0.42, 0.00, 0.52, 0.42, 1.45, 2.10, 4.15]
        gw_gaps = [0.40, 0.71, 1.15, 1.48, 2.48, 3.35, 5.55]
        exp_gaps = [0.41, 0.74, 1.17, 1.52, 2.45, 3.44, 5.48]

        x = np.arange(len(materials))
        w = 0.26
        ax.bar(x - w, lda_gaps, width=w, color=COLORS["danger"], label="Kohn–Sham LDA (underestimates by 30–100%)")
        ax.bar(x, gw_gaps, width=w, color=COLORS["primary"], label="$GW$ Quasiparticle Approximation")
        ax.bar(x + w, exp_gaps, width=w, color=COLORS["good"], label="Experimental Band Gap")
        ax.axhline(0.0, color=COLORS["total"], linewidth=1.0, linestyle="--")

        ax.set_xticks(x)
        ax.set_xticklabels(materials)
        ax.set_ylabel("Electronic Band Gap $E_{\\mathrm{gap}}$ (eV)")
        ax.set_title("The DFT Band-Gap Problem Across Semiconductors (LDA vs. $GW$ vs. Expt.)")
        ax.grid(True, axis="y", color=COLORS["grid"], linestyle="--", linewidth=0.7)
        ax.legend(loc="upper left", fontsize=8.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return (
            "**DFT Band-Gap Problem:** Standard Kohn–Sham LDA/GGA systematically underestimates semiconductor band gaps by 20–50% "
            "and wrongly predicts narrow-gap semiconductors like **InAs** ($E_{\\text{exp}}=0.41\\text{ eV}$, $E_{\\text{LDA}}=-0.42\\text{ eV}$) "
            "and **Ge** ($E_{\\text{exp}}=0.74\\text{ eV}$) to be **metallic**! Hybrid functionals (HSE/B3LYP) and the $GW$ method restore accurate gaps."
        )

    # 1D Brillouin zone k in [-1.5 * pi/a, 1.5 * pi/a] normalized by u = k / (pi/a) in [-1.5, 1.5]
    q = np.linspace(-1.45, 1.45, 500)
    e_zone = 3.8  # Free electron energy at zone boundary k = pi/a (in eV)
    e_free = e_zone * (q ** 2)

    # 2x2 degenerate perturbation theory coupling k and k - G (G = +/- 2*pi/a => q_G = +/- 2)
    q_fold = np.abs(q)
    e1 = e_zone * (q_fold ** 2)
    e2 = e_zone * ((q_fold - 2.0) ** 2)
    e_avg = 0.5 * (e1 + e2)
    split = np.sqrt((0.5 * (e1 - e2)) ** 2 + (u_ev ** 2))
    e_lower = e_avg - split
    e_upper = e_avg + split

    # Plot inside 1st BZ [-1, 1] and extended zone
    mask_bz1 = np.abs(q) <= 1.0
    ax.plot(q, e_free, color=COLORS["muted"], linestyle="--", linewidth=1.6, label="Free electron $E_{\\mathrm{free}}(k) = \\frac{\\hbar^2 k^2}{2m}$")
    ax.plot(q[mask_bz1], e_lower[mask_bz1], color=COLORS["primary"], linewidth=2.5, label="Band 1 (bonding standing wave $\\psi_+ \\propto \\cos(\\pi x/a)$ at $k=\\pm\\pi/a$)")
    ax.plot(q[mask_bz1], e_upper[mask_bz1], color=COLORS["accent"], linewidth=2.5, label="Band 2 (antibonding $\\psi_- \\propto \\sin(\\pi x/a)$ at $k=\\pm\\pi/a$)")
    ax.axvline(-1.0, color=COLORS["total"], linestyle=":", linewidth=1.3)
    ax.axvline(+1.0, color=COLORS["total"], linestyle=":", linewidth=1.3, label="1st Brillouin Zone boundary $k = \\pm \\pi/a$")

    if u_ev > 0.02:
        ax.annotate(
            "",
            xy=(1.0, e_zone + u_ev),
            xytext=(1.0, e_zone - u_ev),
            arrowprops=dict(arrowstyle="<->", color=COLORS["danger"], lw=1.8),
        )
        ax.text(1.04, e_zone, f"$E_{{\\mathrm{{gap}}}} = 2U$\n$= {2.0*u_ev:.2f}$ eV", va="center", ha="left", fontsize=8.8, fontweight="bold", color=COLORS["danger"])

    ax.set_xlim(-1.45, 1.45)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["$-\\pi/a$", "$-\\pi/2a$", "$0$ ($\\Gamma$)", "$+\\pi/2a$", "$+\\pi/a$"])
    ax.set_ylim(0.0, 7.5)
    ax.set_xlabel("Crystal Wavevector $k$ in 1st Brillouin Zone")
    ax.set_ylabel("Electron Energy $E(k)$ (eV)")
    ax.set_title(f"Bragg Diffraction & Band Gap Opening at $k = \\pm\\pi/a$ ($U = {u_ev:.2f}$ eV)")
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.legend(loc="upper center", fontsize=8.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Zone-Boundary Diffraction ($k = \\pm\\pi/a$):** Bragg-reflected waves form standing waves "
        f"$\\psi_+(x) = 2\\cos(\\pi x/a)$ (piling electron density **on** the positive ions, lowering energy to $E_{{\\text{{free}}}} - U = {e_zone - u_ev:.2f}\\text{{ eV}}$) "
        f"and $\\psi_-(x) = 2\\sin(\\pi x/a)$ (piling density **between** ions, raising energy to $E_{{\\text{{free}}}} + U = {e_zone + u_ev:.2f}\\text{{ eV}}$), "
        f"opening a direct energy gap $E_{{\\text{{gap}}}} = 2U = {2.0 * u_ev:.2f}\\text{{ eV}}$ with group velocity $v_g = \\frac{{1}}{{\\hbar}}\\frac{{dE}}{{dk}} = 0$."
    )
