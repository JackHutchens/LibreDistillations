"""Concept rendering functions for Lesson 12: Monte Carlo Foundations & Non-Uniform Sampling."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def render_dimensionality_crossover(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render log-log scaling of integration error vs N and CPU time vs dimension D comparing quadrature rules against Monte Carlo."""
    rule = str(controls.get("quadrature_rule", "trapezoidal"))
    dim_D = int(controls.get("dimension_D", 6))
    view_mode = str(controls.get("view_mode", "error_vs_N"))

    alpha_map = {
        "trapezoidal": (2, "Trapezoidal Rule ($\\alpha = 2$)"),
        "simpson": (4, "Simpson's Rule ($\\alpha = 4$)"),
        "gauss6": (6, "3-Point Gauss–Legendre ($\\alpha = 6$)"),
    }
    alpha, rule_label = alpha_map.get(rule, (2, "Trapezoidal Rule ($\\alpha = 2$)"))
    d_cross = 2 * alpha

    if view_mode == "error_vs_N":
        N_vals = np.logspace(1, 7, 250)
        err_mc = N_vals ** (-0.5)
        err_quad_current = N_vals ** (-float(alpha) / float(dim_D))
        err_quad_2d = N_vals ** (-float(alpha) / 2.0)
        err_quad_cross = N_vals ** (-float(alpha) / float(d_cross))

        ax.loglog(N_vals, err_mc, color=COLORS["accent"], linewidth=2.5, label=r"Monte Carlo: $\varepsilon \propto M^{-1/2}$ (any $D$)")
        ax.loglog(
            N_vals,
            err_quad_current,
            color=COLORS["primary"],
            linewidth=2.3,
            label=rf"{rule_label} in $D={dim_D}$: $\varepsilon \propto N^{{-{alpha}/{dim_D}}}$",
        )
        ax.loglog(
            N_vals,
            err_quad_2d,
            linestyle=":",
            color=COLORS["muted"],
            linewidth=1.5,
            label=rf"Quadrature in $D=2$: $\varepsilon \propto N^{{-{alpha}/2}}$",
        )
        ax.loglog(
            N_vals,
            err_quad_cross,
            linestyle="--",
            color=COLORS["good"],
            linewidth=1.5,
            label=rf"Crossover $D^* = 2\alpha = {d_cross}$: $\varepsilon \propto N^{{-1/2}}$",
        )

        ax.set_xlim(1e1, 1e7)
        ax.set_ylim(1e-6, 1.0)
        ax.set_xlabel("Total Function Evaluations $N$ (or Samples $M$)")
        ax.set_ylabel("Integration Error Scaling $\\varepsilon(N)$")
        ax.set_title(f"Error Scaling in $D = {dim_D}$ Dimensions vs. Crossover $D^* = 2\\alpha = {d_cross}$")
        ax.legend(loc="lower left", fontsize=8.0)
    else:
        # Plot CPU time exponent (D / alpha vs 2) across dimensions D = 1..20 for target error eps = 1e-3
        D_arr = np.arange(1, 17)
        target_eps = 1e-3
        log10_N_quad = (D_arr / float(alpha)) * (-np.log10(target_eps))
        log10_M_mc = np.full_like(D_arr, 2.0 * (-np.log10(target_eps)), dtype=float)

        ax.plot(D_arr, log10_N_quad, "-o", color=COLORS["primary"], linewidth=2.2, markersize=5, label=rf"{rule_label}: $\log_{{10}} T_{{\mathrm{{CPU}}}} \propto (D/{alpha})\log_{{10}}(1/\varepsilon)$")
        ax.axhline(log10_M_mc[0], color=COLORS["accent"], linewidth=2.4, label=r"Monte Carlo: $\log_{10} T_{\mathrm{CPU}} \propto 2\log_{10}(1/\varepsilon)$")
        ax.axvline(d_cross, color=COLORS["good"], linestyle="--", linewidth=1.8, label=rf"Crossover $D^* = 2\alpha = {d_cross}$")
        ax.scatter([dim_D], [(dim_D / float(alpha)) * (-np.log10(target_eps))], s=90, color=COLORS["danger"], zorder=5, label=f"Selected $D = {dim_D}$")

        ax.set_xlim(1, 16)
        ax.set_xlabel("Integration Dimension $D$")
        ax.set_ylabel(r"$\log_{10}(N)$ Evaluations for $\varepsilon = 10^{-3}$")
        ax.set_title(f"Curse of Dimensionality: CPU Cost Crossover at $D^* = 2\\alpha = {d_cross}$")
        ax.legend(loc="upper left", fontsize=8.0)

    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    winner = (
        f"Because $D = {dim_D} > 2\\alpha = {d_cross}$, **Monte Carlo wins** asymptotically ($M^{{-1/2}}$ decays faster than $N^{{-{alpha}/{dim_D}}}$)!"
        if dim_D > d_cross
        else (
            f"At the exact crossover $D = 2\\alpha = {d_cross}$, deterministic quadrature ($N^{{-{alpha}/{dim_D}}} = N^{{-1/2}}$) and Monte Carlo ($M^{{-1/2}}$) have the **same** asymptotic exponent!"
            if dim_D == d_cross
            else f"Because $D = {dim_D} < 2\\alpha = {d_cross}$, **deterministic quadrature wins** ($N^{{-{alpha}/{dim_D}}}$ decays faster than $M^{{-1/2}}$)."
        )
    )
    return (
        f"**{rule_label}** in $D = {dim_D}$ dimensions: Grid spacing is $h = N^{{-1/D}}$, so truncation error scales as "
        f"$\\varepsilon_{{\\text{{quad}}}} \\propto h^{{\\alpha}} = N^{{-{alpha}/{dim_D}}} = N^{{-{alpha/dim_D:.3f}}}$ "
        f"(requiring $T_{{\\text{{CPU}}}} \\propto \\varepsilon^{{-D/\\alpha}} = \\varepsilon^{{-{dim_D/alpha:.2f}}}$), "
        f"whereas Monte Carlo error scales as $\\varepsilon_{{\\text{{MC}}}} \\propto M^{{-1/2}}$ ($T_{{\\text{{CPU}}}} \\propto \\varepsilon^{{-2}}$). {winner}"
    )


def render_hit_or_miss_pi(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render interactive quarter-circle dartboard and running convergence of hat(pi)(M) with 95% confidence band."""
    M = int(controls.get("M_samples", 400))
    seed = int(controls.get("seed", 42))
    panel_mode = str(controls.get("panel_mode", "dartboard"))

    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=M)
    y = rng.uniform(0.0, 1.0, size=M)
    r2 = x * x + y * y
    inside = r2 <= 1.0

    hits = int(np.sum(inside))
    pi_hat = 4.0 * hits / float(M)
    sigma_pi = np.sqrt(np.pi * (4.0 - np.pi) / float(M))
    abs_err = abs(pi_hat - np.pi)

    if panel_mode == "dartboard":
        theta = np.linspace(0.0, 0.5 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color=COLORS["total"], linewidth=2.2, label=r"Quarter Circle $x^2 + y^2 = 1$")
        ax.fill_between(np.cos(theta), 0.0, np.sin(theta), color=COLORS["primary"], alpha=0.10)

        ax.scatter(x[inside], y[inside], s=14, color=COLORS["primary"], alpha=0.75, label=f"Hits Inside ($N_{{\\mathrm{{hit}}}} = {hits}$)")
        ax.scatter(x[~inside], y[~inside], s=14, color=COLORS["accent"], marker="x", alpha=0.75, label=f"Misses Outside ($N_{{\\mathrm{{miss}}}} = {M - hits}$)")

        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, 1.02)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Uniform Coordinate $x \\sim U(0, 1)$")
        ax.set_ylabel("Uniform Coordinate $y \\sim U(0, 1)$")
        ax.set_title(f"Hit-or-Miss Dartboard ($M = {M}$): $\\hat{{\\pi}} = 4({hits}/{M}) = {pi_hat:.4f}$")
        ax.legend(loc="upper right", fontsize=8.2)
    else:
        m_idx = np.arange(1, M + 1)
        cum_pi = 4.0 * np.cumsum(inside) / m_idx
        ci_95 = 1.96 * np.sqrt(np.pi * (4.0 - np.pi) / m_idx)

        ax.axhline(np.pi, color=COLORS["total"], linestyle="--", linewidth=1.6, label=r"Exact $\pi = 3.14159...$")
        ax.fill_between(
            m_idx,
            np.pi - ci_95,
            np.pi + ci_95,
            color=COLORS["good"],
            alpha=0.18,
            label=r"Theoretical $\pm 1.96\sqrt{\pi(4-\pi)/M}$ Band",
        )
        ax.plot(m_idx, cum_pi, color=COLORS["primary"], linewidth=1.7, label=r"Running Estimate $\hat{\pi}(M)$")

        ax.set_xlim(1, M)
        ax.set_ylim(2.4, 3.85)
        ax.set_xlabel("Number of Dart Throws $M$")
        ax.set_ylabel(r"Running Estimate $\hat{\pi}(M)$")
        ax.set_title(f"Running Convergence $\\hat{{\\pi}}(M) = {pi_hat:.4f}$ ($|\\hat{{\\pi}}-\\pi| = {abs_err:.4f}$)")
        ax.legend(loc="upper right", fontsize=8.0)

    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"With $M = {M}$ random darts (seed {seed}), $N_{{\\text{{hit}}}} = {hits}$ landed inside $x^2 + y^2 \\le 1$, giving "
        f"$\\hat{{p}} = {hits/M:.4f}$ (exact $\\pi/4 \\approx 0.7854$) and $\\hat{{\\pi}} = 4\\hat{{p}} = {pi_hat:.4f}$. "
        f"Theoretical standard error for $\\hat{{\\pi}}$ is $\\sigma_{{\\hat{{\\pi}}}} = 4\\sqrt{{p(1-p)/M}} = \\sqrt{{\\pi(4-\\pi)/M}} = {sigma_pi:.4f}$ "
        f"(95% CI $\\pm {1.96*sigma_pi:.4f}$; observed error $|\\hat{{\\pi}} - \\pi| = {abs_err:.4f}$)."
    )


def render_inverse_and_box_muller(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Visualize inverse CDF mapping u -> F^{-1}(u) for exponential/Cauchy distributions or the 2D Box-Muller polar mapping."""
    dist_mode = str(controls.get("transform_mode", "exponential"))
    param_val = float(controls.get("rate_or_sigma", 1.5))
    n_samples = int(controls.get("n_samples", 600))

    rng = np.random.default_rng(123)

    if dist_mode == "exponential":
        lam = max(param_val, 0.2)
        u_pts = rng.uniform(0.0, 1.0, size=n_samples)
        y_pts = -np.log(1.0 - u_pts) / lam
        y_grid = np.linspace(0.0, 5.0 / lam, 300)
        cdf_grid = 1.0 - np.exp(-lam * y_grid)
        pdf_grid = lam * np.exp(-lam * y_grid)

        ax.plot(y_grid, cdf_grid, color=COLORS["primary"], linewidth=2.4, label=rf"CDF $C(y) = 1 - e^{{-\lambda y}}$ ($\lambda={lam:.2f}$)")
        ax.plot(y_grid, pdf_grid / max(lam, 1.0), "--", color=COLORS["secondary"], linewidth=1.6, label=r"Scaled PDF $p(y)/\lambda = e^{-\lambda y}$")

        # Draw quantile inversion rays for 5 representative u values
        for u_ray in [0.15, 0.35, 0.55, 0.75, 0.90]:
            y_ray = -np.log(1.0 - u_ray) / lam
            ax.plot([0.0, y_ray], [u_ray, u_ray], ":", color=COLORS["accent"], linewidth=1.2)
            ax.plot([y_ray, y_ray], [0.0, u_ray], ":", color=COLORS["accent"], linewidth=1.2)
            ax.scatter([y_ray], [u_ray], s=32, color=COLORS["accent"], zorder=5)

        ax.set_xlim(0.0, 5.0 / lam)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r"Transformed Deviate $y = -\frac{1}{\lambda}\ln(1 - u)$")
        ax.set_ylabel(r"Uniform Input $u = C(y) \in (0, 1)$")
        ax.set_title(f"Inverse Transform Sampling: Exponential Decay ($\\lambda = {lam:.2f}$)")
        ax.legend(loc="lower right", fontsize=8.2)
        caption = (
            f"**Exponential Inverse Transform** ($\\lambda = {lam:.2f}$): Horizontal lines at uniform quantiles $u \\in (0,1)$ "
            f"intersect $C(y) = 1 - e^{{-\\lambda y}}$ at $y = -\\ln(1-u)/\\lambda$. Sample mean from $N={n_samples}$ draws is "
            f"$\\bar{{y}} = {np.mean(y_pts):.3f}$ (exact $1/\\lambda = {1.0/lam:.3f}$)."
        )

    elif dist_mode == "cauchy":
        gamma = max(param_val, 0.2)
        u_pts = rng.uniform(0.02, 0.98, size=n_samples)
        y_pts = gamma * np.tan(np.pi * (u_pts - 0.5))
        y_grid = np.linspace(-4.0 * gamma, 4.0 * gamma, 300)
        cdf_grid = 0.5 + (1.0 / np.pi) * np.arctan(y_grid / gamma)
        pdf_grid = (1.0 / np.pi) * (gamma / (y_grid**2 + gamma**2))

        ax.plot(y_grid, cdf_grid, color=COLORS["primary"], linewidth=2.4, label=rf"Cauchy CDF $C(y) = \frac{{1}}{{2}} + \frac{{1}}{{\pi}}\arctan(y/\gamma)$")
        ax.plot(y_grid, pdf_grid * np.pi * gamma, "--", color=COLORS["secondary"], linewidth=1.6, label=r"Normalized Lorentzian PDF $\pi\gamma\, p(y)$")

        for u_ray in [0.15, 0.30, 0.50, 0.70, 0.85]:
            y_ray = gamma * np.tan(np.pi * (u_ray - 0.5))
            ax.plot([-4.0 * gamma, y_ray], [u_ray, u_ray], ":", color=COLORS["accent"], linewidth=1.2)
            ax.plot([y_ray, y_ray], [0.0, u_ray], ":", color=COLORS["accent"], linewidth=1.2)
            ax.scatter([y_ray], [u_ray], s=32, color=COLORS["accent"], zorder=5)

        ax.set_xlim(-4.0 * gamma, 4.0 * gamma)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(r"Transformed Deviate $y = \gamma \tan(\pi(u - 1/2))$")
        ax.set_ylabel(r"Uniform Input $u = C(y) \in (0, 1)$")
        ax.set_title(f"Inverse Transform Sampling: Lorentzian / Cauchy ($\\gamma = {gamma:.2f}$)")
        ax.legend(loc="lower right", fontsize=8.0)
        caption = (
            f"**Cauchy / Lorentzian Inverse Transform** ($\\gamma = {gamma:.2f}$): Inverting $u = \\frac{{1}}{{2}} + \\frac{{1}}{{\\pi}}\\arctan(y/\\gamma)$ "
            f"yields $y = \\gamma\\tan(\\pi(u - 1/2))$ directly in closed form, with heavy power-law tails $p(y) \\propto y^{{-2}}$."
        )

    else:
        # Box-Muller 2D polar coordinate warp: rings of constant x1 map to circles of radius r = sigma * sqrt(-2 ln x1)
        sigma = max(param_val, 0.2)
        u1 = rng.uniform(1e-4, 1.0, size=n_samples)
        u2 = rng.uniform(0.0, 1.0, size=n_samples)
        r = sigma * np.sqrt(-2.0 * np.log(u1))
        theta = 2.0 * np.pi * u2
        z1 = r * np.cos(theta)
        z2 = r * np.sin(theta)

        # Draw constant-u1 concentric rings and constant-u2 radial spokes
        phi = np.linspace(0.0, 2.0 * np.pi, 200)
        for u1_lvl, col_lvl in [(0.6065, COLORS["good"]), (0.1353, COLORS["muted"])]:
            r_lvl = sigma * np.sqrt(-2.0 * np.log(u1_lvl))
            ax.plot(
                r_lvl * np.cos(phi),
                r_lvl * np.sin(phi),
                linestyle="--",
                color=col_lvl,
                linewidth=1.5,
                label=rf"$x_1 = {u1_lvl:.3f} \Rightarrow r = {r_lvl/sigma:.1f}\sigma$",
            )

        ax.scatter(z1, z2, c=u1, cmap="viridis", s=16, alpha=0.75, label="Box–Muller Pairs $(y_1, y_2)$")
        lim = 3.4 * sigma
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$y_1 = \sigma\sqrt{-2\ln x_1}\cos(2\pi x_2)$")
        ax.set_ylabel(r"$y_2 = \sigma\sqrt{-2\ln x_1}\sin(2\pi x_2)$")
        ax.set_title(f"2D Box–Muller Transform ($\\sigma = {sigma:.2f}$, $N = {n_samples}$ pairs)")
        ax.legend(loc="upper right", fontsize=7.8)
        caption = (
            f"**2D Box–Muller Gaussian Transform** ($\\sigma = {sigma:.2f}$): Because the 1D Gaussian CDF $\\Phi(y)$ has no closed-form elementary inverse, "
            f"we pair two uniforms $(x_1, x_2) \\in (0,1)^2$ in polar coordinates: $x_1 = e^{{-(y_1^2+y_2^2)/(2\\sigma^2)}}$ sets the radius "
            f"$r = \\sigma\\sqrt{{-2\\ln x_1}}$ (note $x_1 = e^{{-1/2}} \\approx 0.607$ maps to $r = 1\\sigma$), and $x_2$ sweeps the angle $\\theta = 2\\pi x_2$."
        )

    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return caption
