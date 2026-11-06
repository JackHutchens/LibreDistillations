"""Concept rendering functions for Lesson 09: Static Correlations, Pair Distribution Functions, and Structure Factor."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def _generate_rdf_curve(phase: str, r: np.ndarray, rho: float) -> np.ndarray:
    """Generate realistic synthetic radial distribution function g(r) across phases."""
    g = np.zeros_like(r)
    mask = r > 0.72
    r_m = r[mask]

    if phase == "ideal_gas":
        return np.ones_like(r)

    if phase == "low_density_gas":
        # Low-density Mayer f-function limit: g(r) = exp(-beta V_LJ(r)) at T* = 1.6
        T_star = 1.6
        inv_r6 = (1.0 / np.clip(r_m, 0.82, None)) ** 6
        v_lj = 4.0 * (inv_r6**2 - inv_r6)
        g_m = np.exp(-np.clip(v_lj / T_star, -5.0, 20.0))
        g_m[r_m < 0.85] *= np.exp(-30.0 * (0.85 - r_m[r_m < 0.85]) ** 2)
        g[mask] = g_m
        return g

    if phase == "dense_liquid":
        # Classic Lennard-Jones liquid near triple point (rho* ~ 0.84, T* ~ 0.8)
        r0 = 1.09
        core = 0.5 * (1.0 + np.tanh((r_m - 0.96) / 0.035))
        osc = (
            1.85 * np.exp(-1.25 * (r_m - r0)) * np.cos(2.0 * np.pi * (r_m - r0) / 0.96)
            + 0.45 * np.exp(-28.0 * (r_m - r0) ** 2)
        )
        g_m = core * np.clip(1.0 + osc, 0.0, None)
        g_m[r_m < 0.88] = 0.0
        g[mask] = g_m
        return g

    # FCC coordination shells: r_n / r_1 = sqrt(n) for n = 1, 2, 3, 4, 5, 6, 7...
    # With coordination numbers Z_n = 12, 6, 24, 12, 24, 8, 48
    r1 = (np.sqrt(2.0) / max(rho, 0.2)) ** (1.0 / 3.0) if rho > 0.5 else 1.09
    shells = [
        (1.0, 12),
        (np.sqrt(2.0), 6),
        (np.sqrt(3.0), 24),
        (2.0, 12),
        (np.sqrt(5.0), 24),
        (np.sqrt(6.0), 8),
        (np.sqrt(7.0), 48),
    ]
    sigma_w = 0.085 if phase == "warm_crystal" else 0.038

    for ratio, z_n in shells:
        r_s = ratio * r1
        # Shell contribution so 4*pi*rho*r_s^2 * g(r) integrates to Z_n
        pref = z_n / (4.0 * np.pi * max(rho, 0.1) * (r_s**2) * np.sqrt(2.0 * np.pi) * sigma_w)
        g += pref * np.exp(-0.5 * ((r - r_s) / sigma_w) ** 2)

    return g


def render_rdf_across_phases(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Interactive g(r) and running coordination number N(r) explorer across phases."""
    phase = str(controls.get("phase", "dense_liquid"))
    rho = float(controls.get("rho", 0.84))
    r_max_int = float(controls.get("r_cutoff_shell", 1.55))

    r = np.linspace(0.01, 3.4, 500)
    g = _generate_rdf_curve(phase, r, rho)

    # Running coordination number N(r) = 4 * pi * rho * int_0^r g(r') r'^2 dr'
    dr = r[1] - r[0]
    integrand = 4.0 * np.pi * rho * g * (r**2)
    n_running = np.cumsum(integrand) * dr

    idx_cut = int(np.searchsorted(r, r_max_int))
    idx_cut = min(idx_cut, len(r) - 1)
    z_shell = float(n_running[idx_cut])

    ax.plot(r, g, color=COLORS["primary"], lw=2.3, label=r"Pair correlation $g(r)$")
    ax.axhline(1.0, color=COLORS["muted"], ls="--", lw=1.1, label=r"Ideal gas limit ($g=1$)")

    mask_fill = r <= r_max_int
    ax.fill_between(
        r[mask_fill],
        0.0,
        g[mask_fill],
        color=COLORS["secondary"],
        alpha=0.25,
        label=f"Integrated shell ($r \\le {r_max_int:.2f}\\sigma$)",
    )
    ax.axvline(r_max_int, color=COLORS["accent"], ls=":", lw=1.6)

    ax2 = ax.twinx()
    ax2.plot(
        r,
        n_running,
        color=COLORS["accent"],
        lw=1.9,
        ls="-.",
        label=r"Running coord. $N(r)$",
    )
    ax2.plot(r_max_int, z_shell, marker="o", markersize=7, color=COLORS["accent"])
    ax2.set_ylabel(r"Running Coordination $N(r)$", color=COLORS["accent"])
    ax2.tick_params(axis="y", labelcolor=COLORS["accent"])
    ax2.set_ylim(0, max(16.0, float(n_running[np.searchsorted(r, 2.2)]) * 1.1))
    ax2.spines["top"].set_visible(False)

    ax.set_xlabel(r"Pair Separation $r / \sigma$")
    ax.set_ylabel(r"Radial Distribution Function $g(r)$")
    ax.set_xlim(0.0, 3.4)
    ax.set_ylim(0.0, max(3.6, float(np.max(g)) * 1.08))
    ax.spines["top"].set_visible(False)
    ax.grid(True, color=COLORS["grid"], ls="--", lw=0.7, alpha=0.8)

    phase_labels = {
        "ideal_gas": "Ideal Gas ($V(r)=0$)",
        "low_density_gas": "Low-Density LJ Gas ($e^{-\\beta V(r)}$)",
        "dense_liquid": "Dense LJ Liquid (Short-Range Order)",
        "warm_crystal": "Warm FCC Crystal (Thermal Broadening)",
        "cold_crystal": "Cold FCC Crystal (Sharp Coordination Shells)",
    }
    p_title = phase_labels.get(phase, phase)
    ax.set_title(f"{p_title} · $N(r \\le {r_max_int:.2f}\\sigma) = {z_shell:.2f}$ neighbors")
    ax.legend(loc="upper left", fontsize=8.3)

    return (
        f"**Phase:** {p_title} at reduced number density $\\rho\\sigma^3 = {rho:.2f}$. "
        f"Integrating $4\\pi\\rho g(r)r^2$ from $0$ to $r = {r_max_int:.2f}\\sigma$ yields "
        f"**$N(r) = {z_shell:.2f}$** neighbors inside the selected sphere."
    )


def render_structure_factor_debye_waller(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot static structure factor S(k) with Debye-Waller thermal attenuation and system-size scaling."""
    u_ratio = float(controls.get("lindemann_u", 0.08))
    n_atoms = int(controls.get("n_atoms", 32))
    show_liquid = bool(int(controls.get("compare_liquid", 0)))

    k = np.linspace(0.5, 18.0, 600)
    # Reciprocal lattice shells G_n for FCC with nearest neighbor d = 1
    g_shells = [2.0 * np.pi * np.sqrt(n) for n in (1.0, 4.0 / 3.0, 8.0 / 3.0, 11.0 / 3.0)]

    # Debye-Waller envelope exp(-G^2 <u^2> / 3) where u_ratio = sqrt(<u^2>)/d and d = 1
    u2 = u_ratio**2
    dw_envelope = np.exp(-(k**2) * u2 / 3.0)
    diffuse_bg = 1.0 - dw_envelope

    # Construct crystal S(k): diffuse thermal background + finite-size Bragg peaks at G_n
    s_crystal = diffuse_bg.copy()
    width_k = 0.22
    for g_val in g_shells:
        dw_g = np.exp(-(g_val**2) * u2 / 3.0)
        peak_amp = (n_atoms - 1) * dw_g
        s_crystal += peak_amp * np.exp(-0.5 * ((k - g_val) / width_k) ** 2)

    # Liquid S(k) comparison (does not scale with N, only O(1) broad oscillations)
    g1 = g_shells[0]
    s_liquid = 1.0 + 1.85 * np.exp(-0.18 * (k - g1)) * np.cos(1.02 * (k - g1)) / (1.0 + 0.08 * (k - g1) ** 2)
    s_liquid[k < 4.5] = 0.06 + 0.94 * ((k[k < 4.5] / 4.5) ** 4)
    s_liquid = np.clip(s_liquid, 0.04, None)

    ax.plot(k, s_crystal, color=COLORS["solid"], lw=2.1, label=f"Crystal $S(k)$ ($N={n_atoms}$, $u/d={u_ratio:.2f}$)")
    ax.plot(
        k,
        1.0 + (n_atoms - 1) * dw_envelope,
        color=COLORS["accent"],
        ls="--",
        lw=1.5,
        label=r"Debye-Waller Bragg Ceiling $1+(N-1)e^{-k^2\langle u^2\rangle/3}$",
    )
    ax.plot(
        k,
        diffuse_bg,
        color=COLORS["muted"],
        ls=":",
        lw=1.5,
        label=r"Thermal Diffuse Scattering $1 - e^{-k^2\langle u^2\rangle/3}$",
    )

    if show_liquid:
        ax.plot(k, s_liquid, color=COLORS["liquid"], lw=2.0, alpha=0.85, label="Liquid $S(k)$ (Broad $O(1)$ rings)")

    g1_val = g_shells[0]
    s_g1 = 1.0 + (n_atoms - 1) * np.exp(-(g1_val**2) * u2 / 3.0)
    melted_flag = "MELTED (Exceeds Lindemann ~0.15)" if u_ratio >= 0.15 else "Crystalline Solid"

    ax.set_xlabel(r"Wavevector Magnitude $k d$")
    ax.set_ylabel(r"Static Structure Factor $S(k)$")
    ax.set_xlim(0.5, 18.0)
    ax.set_ylim(0.0, max(8.0, float(n_atoms) * 1.05))
    ax.set_title(f"Structure Factor & Debye-Waller Factor · {melted_flag}")
    ax.grid(True, color=COLORS["grid"], ls="--", lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8.0)

    return (
        f"**First Bragg Peak ($G_1 d = 2\\pi$):** $S(G_1) = 1 + ({n_atoms}-1)\\exp(-G_1^2 u^2/3) = {s_g1:.2f}$ "
        f"(attenuation factor $e^{{-G_1^2 u^2/3}} = {np.exp(-(g1_val**2)*u2/3.0):.3f}$). "
        f"At $u/d = {u_ratio:.2f}$, status is **{melted_flag}**."
    )


def render_compressibility_and_thermodynamics(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Show how g(r) kernels determine excess internal energy, virial pressure, and compressibility."""
    rho = float(controls.get("rho", 0.80))
    temp = float(controls.get("temperature", 1.15))
    kernel_mode = str(controls.get("kernel_mode", "potential_energy"))

    r = np.linspace(0.75, 3.2, 450)
    dr = r[1] - r[0]

    # Smooth realistic LJ g(r) modulated by density and temperature
    r0 = 1.09
    core = np.exp(-((0.97 / r) ** 18))
    amp = (0.4 + 1.65 * rho) / np.sqrt(max(temp, 0.5))
    g = core * np.clip(
        1.0 + amp * np.exp(-1.2 * (r - r0)) * np.cos(2.0 * np.pi * (r - r0) / 0.96),
        0.0,
        None,
    )

    # Lennard-Jones potential V(r) = 4[(1/r)^12 - (1/r)^6] and force derivative r * dV/dr
    inv_r6 = (1.0 / r) ** 6
    v_lj = 4.0 * (inv_r6**2 - inv_r6)
    r_dv_dr = 4.0 * (-12.0 * (inv_r6**2) + 6.0 * inv_r6)

    # Integrands
    u_integrand = 2.0 * np.pi * rho * v_lj * g * (r**2)
    p_integrand = -(2.0 * np.pi * (rho**2) / 3.0) * r_dv_dr * g * (r**2)
    s0_integrand = 4.0 * np.pi * rho * (g - 1.0) * (r**2)

    u_ex_per_n = float(np.sum(u_integrand) * dr)
    p_ex = float(np.sum(p_integrand) * dr)
    p_total = rho * temp + p_ex
    s_zero = max(0.05, 1.0 + float(np.sum(s0_integrand) * dr) * 0.25)
    chi_t = s_zero / (rho * temp)

    ax.axhline(0.0, color=COLORS["muted"], ls="--", lw=1.0)

    if kernel_mode == "potential_energy":
        ax.plot(r, u_integrand, color=COLORS["potential"], lw=2.3, label=r"$2\pi\rho\, V(r)g(r)r^2$")
        ax.fill_between(r, 0.0, u_integrand, where=(u_integrand < 0), color=COLORS["primary"], alpha=0.25, label="Attractive well ($V<0$)")
        ax.fill_between(r, 0.0, u_integrand, where=(u_integrand > 0), color=COLORS["danger"], alpha=0.25, label="Repulsive core ($V>0$)")
        ax.set_ylabel(r"Energy Integrand $2\pi\rho V(r)g(r)r^2$")
        ax.set_title(f"Excess Potential Energy per Particle $U/N = {u_ex_per_n:.2f}\\varepsilon$")
    elif kernel_mode == "virial_pressure":
        ax.plot(r, p_integrand, color=COLORS["accent"], lw=2.3, label=r"$-\frac{2\pi\rho^2}{3} r \frac{dV}{dr} g(r)r^2$")
        ax.fill_between(r, 0.0, p_integrand, where=(p_integrand > 0), color=COLORS["accent"], alpha=0.25, label="Repulsive wall ($dV/dr < 0 \\Rightarrow +P$)")
        ax.fill_between(r, 0.0, p_integrand, where=(p_integrand < 0), color=COLORS["secondary"], alpha=0.25, label="Attractive tail ($dV/dr > 0 \\Rightarrow -P$)")
        ax.set_ylabel(r"Virial Pressure Integrand")
        ax.set_title(f"Total Pressure $P = \\rho k_B T + P_{{\\mathrm{{ex}}}} = {p_total:.2f}\\varepsilon/\\sigma^3$")
    else:
        ax.plot(r, s0_integrand, color=COLORS["secondary"], lw=2.3, label=r"$4\pi\rho [g(r)-1]r^2$")
        ax.fill_between(r, 0.0, s0_integrand, color=COLORS["secondary"], alpha=0.22, label=r"Excess pair correlation $[g(r)-1]$")
        ax.set_ylabel(r"Compressibility Integrand $4\pi\rho[g(r)-1]r^2$")
        ax.set_title(f"Long-Wavelength Limit $S(0) = {s_zero:.2f} \\Rightarrow \\chi_T = {chi_t:.3f}$")

    ax.set_xlabel(r"Pair Separation $r / \sigma$")
    ax.set_xlim(0.8, 3.2)
    ax.grid(True, color=COLORS["grid"], ls="--", lw=0.7, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", fontsize=8.3)

    return (
        f"**Thermodynamic Summary ($\\rho^*={rho:.2f}, T^*={temp:.2f}$):** "
        f"Total internal energy $E/N = \\frac{{3}}{{2}}T^* + U/N = {1.5*temp + u_ex_per_n:.2f}\\varepsilon$ "
        f"(with $U/N = {u_ex_per_n:.2f}\\varepsilon$), virial pressure $P^* = {rho*temp:.2f} + ({p_ex:.2f}) = {p_total:.2f}$, "
        f"and isothermal compressibility $\\chi_T^* = S(0)/(\\rho^* T^*) = {chi_t:.3f}$."
    )
