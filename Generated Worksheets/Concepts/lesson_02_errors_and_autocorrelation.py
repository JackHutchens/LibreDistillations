"""Interactive concept visualizations for Lesson 02: Statistical vs. Systematic Errors, Autocorrelation, Data Blocking & Resampling."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def _generate_ar1_trace(n_steps: int, tau: float, sigma: float = 1.0, seed: int = 48502) -> np.ndarray:
    """Generate a stationary zero-mean AR(1) time series with exponential autocorrelation exp(-t/tau)."""
    rng = np.random.default_rng(seed)
    if tau <= 1e-6:
        return rng.normal(0.0, sigma, size=n_steps)
    phi = float(np.exp(-1.0 / tau))
    noise_scale = sigma * np.sqrt(max(1.0 - phi**2, 1e-12))
    eps = rng.normal(0.0, 1.0, size=n_steps)
    x = np.empty(n_steps, dtype=float)
    x[0] = sigma * eps[0]
    for i in range(1, n_steps):
        x[i] = phi * x[i - 1] + noise_scale * eps[i]
    return x


def render_autocorrelation_lab(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render normalized autocorrelation C(t) or AR(1) trace as correlation time tau varies."""
    tau = float(controls.get("tau", 8.0))
    n_steps = int(controls.get("n_steps", 600))
    view_mode = str(controls.get("view_mode", "autocorrelation"))

    sigma_a = 1.0
    mu_true = -7.25  # Like liquid argon / MD potential energy per atom in eV
    trace = mu_true + _generate_ar1_trace(n_steps=n_steps, tau=tau, sigma=sigma_a, seed=48521)

    phi = np.exp(-1.0 / max(tau, 1e-6)) if tau > 0 else 0.0
    # Exact geometric series sum for AR(1): kappa = 1 + 2 * phi / (1 - phi) ≈ 1 + 2*tau
    kappa_exact = (1.0 + phi) / max(1.0 - phi, 1e-6) if tau > 0 else 1.0
    kappa_approx = 1.0 + 2.0 * tau
    n_eff = n_steps / kappa_approx

    sample_mean = float(np.mean(trace))
    sample_var = float(np.var(trace, ddof=1))
    se_naive = np.sqrt(sample_var / n_steps)
    se_true = se_naive * np.sqrt(kappa_approx)

    if view_mode == "trace":
        t_arr = np.arange(n_steps)
        ax.plot(t_arr, trace, color=COLORS["primary"], linewidth=1.1, alpha=0.85, label="MD/MC Observable $A_t$")
        ax.axhline(sample_mean, color=COLORS["accent"], linewidth=2.0, label=f"Sample Mean $\\bar{{A}} = {sample_mean:.3f}$")
        ax.fill_between(
            t_arr,
            sample_mean - se_true,
            sample_mean + se_true,
            color=COLORS["accent"],
            alpha=0.25,
            label=f"True $\\pm 1\\sigma_{{\\bar{{A}}}}$ ($\\pm {se_true:.4f}$)",
        )
        ax.set_xlim(0, n_steps)
        ax.set_xlabel("Iteration / Time Step $t$")
        ax.set_ylabel("Observable $A_t$ (e.g., Potential Energy)")
        ax.set_title(f"Correlated Time Series ($N={n_steps},\\ \\tau={tau:.1f},\\ \\kappa\\approx {kappa_approx:.1f}$)")
    else:
        max_lag = min(80, n_steps // 3)
        lags = np.arange(max_lag + 1)
        da = trace - sample_mean
        denom = float(np.dot(da, da))
        c_emp = np.array([1.0 if lag == 0 else float(np.dot(da[:-lag], da[lag:])) / denom for lag in lags])
        c_theo = np.exp(-lags / max(tau, 1e-6)) if tau > 0 else np.where(lags == 0, 1.0, 0.0)

        ax.plot(lags, c_emp, color=COLORS["primary"], linewidth=1.8, marker="o", markersize=3.5, alpha=0.85, label="Empirical $c_A(t)$")
        ax.plot(lags, c_theo, color=COLORS["accent"], linewidth=2.2, linestyle="--", label=f"Theoretical $e^{{-t/\\tau}}$ ($\\tau={tau:.1f}$)")
        ax.axhline(0.0, color=COLORS["text"], linewidth=0.9, alpha=0.6)
        ax.axvline(tau, color=COLORS["secondary"], linestyle=":", linewidth=1.6, label=f"Decay time $\\tau = {tau:.1f}$ ($C(\\tau)=e^{{-1}}$)")
        ax.fill_between(lags, 0.0, np.maximum(c_theo, 0.0), color=COLORS["accent"], alpha=0.14)
        ax.set_xlim(0, max_lag)
        ax.set_ylim(-0.25, 1.08)
        ax.set_xlabel("Lag Time / Step Separation $\\Delta t = |t - t'|$")
        ax.set_ylabel(r"Autocorrelation $C(\Delta t) = \langle \delta A_t \delta A_{t+\Delta t}\rangle / \sigma_A^2$")
        ax.set_title(f"Normalized Autocorrelation Function ($\\kappa \\approx 1 + 2\\tau = {kappa_approx:.1f}$)")

    ax.legend(loc="upper right", fontsize=8.2)
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    underest_factor = np.sqrt(kappa_approx)
    return (
        f"**Autocorrelation Diagnostics ($N = {n_steps},\\ \\tau = {tau:.1f}\\text{{ steps}}$):**\n\n"
        f"- **Statistical Inefficiency (Correlation Time):** $\\kappa = 1 + 2\\sum_{{t=1}}^\\infty C(t) \\approx 1 + 2\\tau = **{kappa_approx:.1f}**$ "
        f"(exact geometric sum $= {kappa_exact:.2f}$).\n"
        f"- **Effective Independent Samples:** $N_{{\\text{{eff}}}} = N / \\kappa = **{n_eff:.1f}**$ "
        f"({'✓ Sufficient ($N_{\\text{eff}} \\ge 20$)' if n_eff >= 20 else '⚠️ Too few independent samples ($N_{\\text{eff}} < 20$)!'}).\n"
        f"- **Naive vs. True Standard Error:** $\\sigma_{{\\text{{naive}}}} = s_A/\\sqrt{{N}} = {se_naive:.4f}$, whereas "
        f"$\\sigma_{{\\text{{true}}}} = \\sigma_{{\\text{{naive}}}}\\sqrt{{\\kappa}} = **{se_true:.4f}**$ "
        f"(naive formula underestimates error by a factor of $\\sqrt{{\\kappa}} = **{underest_factor:.2f}\\times**$)."
    )


def render_data_blocking_plateau(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render Flyvbjerg-Petersen data blocking curve: standard error vs block size b."""
    tau = float(controls.get("tau", 10.0))
    n_total = int(controls.get("n_total", 4096))
    selected_b = int(controls.get("selected_block_size", 32))

    sigma_a = 1.0
    trace = _generate_ar1_trace(n_steps=n_total, tau=tau, sigma=sigma_a, seed=48522)
    kappa_true = 1.0 + 2.0 * tau

    block_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    block_sizes = [b for b in block_sizes if n_total // b >= 4]

    se_estimates = []
    se_uncertainties = []
    for b in block_sizes:
        n_b = n_total // b
        trimmed = trace[: n_b * b].reshape(n_b, b)
        block_means = np.mean(trimmed, axis=1)
        var_b = float(np.var(block_means, ddof=1))
        se_b = np.sqrt(var_b / n_b)
        # Error of the error estimate ~ se_b / sqrt(2 * (n_b - 1))
        se_err = se_b / np.sqrt(2.0 * max(n_b - 1, 1))
        se_estimates.append(se_b)
        se_uncertainties.append(se_err)

    b_arr = np.array(block_sizes, dtype=float)
    se_arr = np.array(se_estimates, dtype=float)
    err_arr = np.array(se_uncertainties, dtype=float)

    se_naive = se_arr[0]
    se_theo_plateau = np.sqrt(sigma_a**2 * kappa_true / n_total)

    ax.errorbar(
        b_arr,
        se_arr,
        yerr=err_arr,
        fmt="-o",
        color=COLORS["primary"],
        ecolor=COLORS["muted"],
        elinewidth=1.4,
        capsize=3.5,
        linewidth=2.2,
        markersize=5.5,
        label="Blocking Estimate $\\hat{\\sigma}_{\\bar{A}}(b)$",
    )
    ax.axhline(
        se_theo_plateau,
        color=COLORS["good"],
        linestyle="--",
        linewidth=2.0,
        label=f"True Plateau $\\sigma_A\\sqrt{{\\kappa/N}} = {se_theo_plateau:.4f}$",
    )
    ax.axhline(
        se_naive,
        color=COLORS["danger"],
        linestyle=":",
        linewidth=1.5,
        label=f"Naive $b=1$ Error $= {se_naive:.4f}$",
    )

    # Highlight user-selected block size
    if selected_b in block_sizes:
        idx_sel = block_sizes.index(selected_b)
    else:
        idx_sel = int(np.argmin(np.abs(b_arr - selected_b)))
        selected_b = block_sizes[idx_sel]

    sel_se = se_arr[idx_sel]
    sel_nb = n_total // selected_b
    ax.scatter([selected_b], [sel_se], color=COLORS["accent"], s=95, zorder=6, label=f"Selected $b={selected_b}$ ($N_b={sel_nb}$)")

    # Shade unreliable region where N_b < 20
    b_limit = n_total / 20.0
    ax.axvspan(b_limit, max(b_arr) * 1.2, color=COLORS["warning"], alpha=0.10, label=r"Noisy regime ($N_b = N/b < 20$)")

    ax.set_xscale("log", base=2)
    ax.set_xlim(0.8, max(b_arr) * 1.2)
    ax.set_xlabel("Block Length $b$ (steps per block, $\\log_2$ scale)")
    ax.set_ylabel(r"Estimated Standard Error $\hat{\sigma}_{\bar{A}}(b)$")
    ax.set_title(f"Flyvbjerg–Petersen Data Blocking Curve ($N={n_total},\\ \\tau={tau:.1f}$)")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    est_kappa_at_b = (sel_se / se_naive) ** 2
    rel_err_of_err = 1.0 / np.sqrt(2.0 * max(sel_nb - 1, 1))
    return (
        f"**Data Blocking at Block Size $b = {selected_b}$ ($N_b = N/b = {sel_nb}$ blocks):**\n\n"
        f"- **Estimated Standard Error:** $\\hat{{\\sigma}}_{{\\bar{{A}}}}(b={selected_b}) = **{sel_se:.4f}** \\pm {err_arr[idx_sel]:.4f}$ "
        f"(relative error-of-error $\\approx 1/\\sqrt{{2(N_b-1)}} = {100.0*rel_err_of_err:.1f}\\%$). \n"
        f"- **Extracted Statistical Inefficiency:** $\\hat{{\\kappa}}(b) = [\\hat{{\\sigma}}_{{\\bar{{A}}}}(b) / \\sigma_{{\\text{{naive}}}}]^2 = **{est_kappa_at_b:.2f}**$ "
        f"(theoretical plateau $\\kappa = 1 + 2\\tau = {kappa_true:.1f}$ reached once $b \\gtrsim 3\\tau = {3.0*tau:.0f}$)."
    )


def render_warmup_and_jackknife(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render equilibration transient trace and show trade-off of warmup cutoff k1 (I_eq)."""
    k1_cut = int(controls.get("warmup_cutoff", 120))
    tau_eq = float(controls.get("equilibration_tau", 50.0))
    init_spike = float(controls.get("initial_offset", 2.5))

    n_total = 600
    t = np.arange(n_total)
    eq_mean = -2.25
    sigma_noise = 0.18
    corr_noise = _generate_ar1_trace(n_steps=n_total, tau=4.0, sigma=sigma_noise, seed=48523)
    transient = init_spike * np.exp(-t / max(tau_eq, 1.0))
    trace = eq_mean + transient + corr_noise

    k1_cut = min(max(k1_cut, 0), n_total - 20)
    prod_trace = trace[k1_cut:]
    n_eq = len(prod_trace)

    est_mean = float(np.mean(prod_trace))
    # Systematic warmup bias remaining in [k1_cut, n_total): average of transient[k1_cut:]
    rem_bias = float(np.mean(transient[k1_cut:]))
    # Statistical error of production window (with kappa = 1 + 2*4 = 9)
    kappa = 9.0
    stat_err = float(np.std(prod_trace, ddof=1) * np.sqrt(kappa / n_eq))
    total_rmse = float(np.sqrt(rem_bias**2 + stat_err**2))

    # Jackknife over M=10 blocks of production window for nonlinear observable g(A) = A^2
    n_jk_blocks = 10
    b_len = n_eq // n_jk_blocks
    jk_data = prod_trace[: n_jk_blocks * b_len].reshape(n_jk_blocks, b_len)
    block_sums = np.sum(jk_data, axis=1)
    total_sum = float(np.sum(block_sums))
    total_cnt = n_jk_blocks * b_len
    g_full = (total_sum / total_cnt) ** 2
    g_jk = np.array([((total_sum - block_sums[m]) / (total_cnt - b_len)) ** 2 for m in range(n_jk_blocks)])
    jk_mean_g = float(np.mean(g_jk))
    jk_se_g = float(np.sqrt((n_jk_blocks - 1.0) / n_jk_blocks * np.sum((g_jk - jk_mean_g) ** 2)))
    jk_bias_g = float((n_jk_blocks - 1.0) * (jk_mean_g - g_full))

    ax.plot(t[:k1_cut], trace[:k1_cut], color=COLORS["muted"], linewidth=1.2, alpha=0.75, label=f"Discarded Warmup ($k < {k1_cut}$)")
    ax.plot(t[k1_cut:], trace[k1_cut:], color=COLORS["primary"], linewidth=1.3, label=f"Equilibrated Production ($N_{{\\text{{eq}}}}={n_eq}$)")
    ax.plot(t, eq_mean + transient, color=COLORS["danger"], linestyle="--", linewidth=1.8, label="Underlying Drift $\\langle A_k \\rangle$")
    ax.axhline(eq_mean, color=COLORS["good"], linestyle=":", linewidth=1.8, label=f"True Equilibrium $\\langle A \\rangle_{{F^*}} = {eq_mean:.2f}$")
    ax.axvline(k1_cut, color=COLORS["accent"], linewidth=2.2, label=f"Cutoff $k_1 = {k1_cut}$ ($t_1 \\approx {k1_cut / max(tau_eq, 1.0):.1f}\\,\\tau_{{\\text{{eq}}}}$)")
    ax.axvspan(0, k1_cut, color=COLORS["muted"], alpha=0.15)

    ax.set_xlim(0, n_total)
    ax.set_xlabel("Simulation Step $k$")
    ax.set_ylabel("Potential Energy $E_k$ (eV/atom)")
    ax.set_title(f"Warmup Truncation ($k_1={k1_cut}$) & Bias–Variance Trade-Off")
    ax.legend(loc="upper right", fontsize=7.8)
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Equilibration & Jackknife Summary ($k_1 = {k1_cut},\\ N_{{\\text{{eq}}}} = {n_eq},\\ \\tau_{{\\text{{eq}}}} = {tau_eq:.0f}$):**\n\n"
        f"- **Estimated Mean:** $\\bar{{A}} = {est_mean:.4f}\\text{{ eV}}$ (true equilibrium $= {eq_mean:.4f}\\text{{ eV}}$).\n"
        f"- **Remaining Warmup Bias (Systematic):** $+{rem_bias:.4f}\\text{{ eV}}$ | "
        f"**Statistical Error:** $\\pm {stat_err:.4f}\\text{{ eV}}$ | **Total RMSE:** $**{total_rmse:.4f}\\text{{ eV}}**$.\n"
        f"- **10-Block Jackknife on Nonlinear $F(\\bar{{A}}) = \\bar{{A}}^2$:** "
        f"$F(\\bar{{A}}) = {g_full:.4f}$, Jackknife SE $= \\pm {jk_se_g:.4f}$, estimated nonlinear bias $= {jk_bias_g:+.5f}$."
    )


def render_nonlinear_bias_lab(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize O(1/N) systematic bias vs O(1/sqrt(N)) statistical error for nonlinear F(Z)."""
    curvature_c = float(controls.get("curvature_c", 2.0))
    sigma_z = float(controls.get("sigma_z", 0.8))

    # F(Z) = Z + 0.5 * curvature_c * (Z - Z_bar)^2 around Z_bar = 1.0 -> F'(Z_bar)=1, F''(Z_bar)=curvature_c
    n_grid = np.logspace(1.0, 4.0, 120)
    stat_error = np.abs(1.0) * (sigma_z / np.sqrt(n_grid))
    sys_bias = 0.5 * np.abs(curvature_c) * (sigma_z**2 / n_grid)
    total_err = np.sqrt(stat_error**2 + sys_bias**2)

    ax.loglog(n_grid, stat_error, color=COLORS["primary"], linewidth=2.3, label=r"Statistical Error $|F'(\bar{Z})|\sigma_Z/\sqrt{N} \propto N^{-1/2}$")
    ax.loglog(n_grid, sys_bias, color=COLORS["danger"], linewidth=2.3, linestyle="--", label=r"Nonlinear Bias $\frac{1}{2}|F''(\bar{Z})|\sigma_Z^2/N \propto N^{-1}$")
    ax.loglog(n_grid, total_err, color=COLORS["text"], linewidth=1.6, linestyle=":", label="Total RMSE $\\sqrt{\\mathrm{error}^2 + \\mathrm{bias}^2}$")

    ax.set_xlim(10, 10000)
    ax.set_xlabel("Effective Independent Sample Count $N_{\\mathrm{eff}}$ (log scale)")
    ax.set_ylabel("Magnitude of Error / Bias in $F(\\bar{Z})$ (log scale)")
    ax.set_title(f"Nonlinear Estimator Scaling: Bias $\\mathcal{{O}}(N^{{-1}})$ vs. Error $\\mathcal{{O}}(N^{{-1/2}})$")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(True, which="both", color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_20_bias = 0.5 * curvature_c * (sigma_z**2 / 20.0)
    n_20_err = sigma_z / np.sqrt(20.0)
    return (
        f"**Taylor Expansion of $F(Z)$ around $\\bar{{Z}}$ ($F'(\\bar{{Z}})=1.0,\\ F''(\\bar{{Z}})={curvature_c:.2f},\\ \\sigma_Z={sigma_z:.2f}$):**\n\n"
        f"- **At $N_{{\\text{{eff}}}} = 20$:** Statistical error $= {n_20_err:.4f}$ ($\\propto N^{{-1/2}}$), "
        f"while nonlinear systematic bias $= {n_20_bias:+.4f}$ ($\\propto N^{{-1}}$), so $|\\mathrm{{bias}}|/\\mathrm{{error}} = {abs(n_20_bias)/max(n_20_err, 1e-9):.2f}$.\n"
        f"- **Key Insight:** Because bias decays as $N^{{-1}}$ while statistical error decays slower as $N^{{-1/2}}$, "
        f"statistical noise dominates at large $N$, but at small $N_{{\\text{{eff}}}}$ or high curvature $F''(\\bar{{Z}})$ you must apply Jackknife/Bootstrap bias correction!"
    )
