"""Interactive concept-cell renderers for Lesson 14: Non-Uniform Sampling & Variance Reduction."""

from __future__ import annotations
from typing import Any
import numpy as np
import matplotlib.pyplot as plt


def _val(ctrl: Any, default: Any = None) -> Any:
    if ctrl is None:
        return default
    return getattr(ctrl, "value", ctrl)


def render_rejection_sampling_lab(controls: dict[str, Any], ax: plt.Axes) -> str:
    """Render von Neumann Rejection Sampling with target p(x), proposal envelope a*q(x), and accepted/rejected trials."""
    target_kind = str(_val(controls.get("target"), "semicircle"))
    slack = float(_val(controls.get("envelope_slack"), 1.05))
    n_trials = int(_val(controls.get("n_trials"), 400))
    seed = int(_val(controls.get("seed"), 11))

    rng = np.random.default_rng(seed)
    xs = np.linspace(0.0, 1.0, 400)

    if target_kind == "semicircle":
        # Wigner semicircle on [0, 1]: p(x) = (8/pi) * sqrt(x*(1-x)), peak at x=0.5 is 4/pi = 1.27324
        p_fn = lambda x: (8.0 / np.pi) * np.sqrt(np.clip(x * (1.0 - x), 0.0, None))
        a_min = 4.0 / np.pi
        title_name = r"Wigner Semicircle $p(x) = \frac{8}{\pi}\sqrt{x(1-x)}$ ($a_{\min} = 4/\pi \approx 1.273$)"
    elif target_kind == "bimodal":
        # Normalized bimodal density on [0, 1]
        raw_fn = lambda x: 0.55 * np.exp(-((x - 0.25) / 0.11) ** 2) + 0.85 * np.exp(-((x - 0.75) / 0.12) ** 2)
        norm_c = float(np.trapezoid(raw_fn(xs), xs))
        p_fn = lambda x: raw_fn(x) / norm_c
        a_min = float(np.max(p_fn(xs)))
        title_name = f"Bimodal Double-Well Target $p(x)$ ($a_{{\\min}} \\approx {a_min:.3f}$)"
    else:
        # Sharp peaked beta-like target p(x) = 30 x^4 (1-x) -> peak at x=0.8 is 30*(0.4096)*(0.2) = 2.4576
        p_fn = lambda x: 30.0 * (x ** 4) * (1.0 - x)
        a_min = 2.4576
        title_name = r"Skewed Beta Target $p(x) = 30\,x^4(1-x)$ ($a_{\min} = 2.458$)"

    # Uniform proposal q(x) = 1 on [0, 1]
    a_val = a_min * slack
    p_curve = p_fn(xs)
    env_curve = np.full_like(xs, a_val)

    # Draw n_trials from q(x) = U(0, 1) and u ~ U(0, 1); accept if u <= c(x) = p(x) / (a * q(x))
    x_try = rng.random(n_trials)
    u_try = rng.random(n_trials)
    y_try = u_try * a_val
    p_at_x = p_fn(x_try)
    accepted = y_try <= p_at_x

    n_acc = int(np.sum(accepted))
    emp_eff = n_acc / float(n_trials)
    theo_eff = 1.0 / a_val

    ax.set_facecolor("#F8FAFC")
    ax.fill_between(xs, 0.0, p_curve, color="#10B981", alpha=0.14)
    ax.fill_between(xs, p_curve, env_curve, color="#EF4444", alpha=0.08)
    ax.plot(xs, p_curve, color="#047857", lw=2.3, label="Target PDF $p(x)$")
    ax.plot(xs, env_curve, color="#B91C1C", lw=1.8, ls="--", label=f"Proposal Envelope $a\\,q(x) = {a_val:.3f}$")

    ax.scatter(
        x_try[accepted],
        y_try[accepted],
        color="#10B981",
        s=18,
        alpha=0.85,
        edgecolors="#064E3B",
        linewidths=0.4,
        label=f"Accepted ($u \\le c(x)$): {n_acc}",
    )
    ax.scatter(
        x_try[~accepted],
        y_try[~accepted],
        color="#EF4444",
        s=18,
        marker="x",
        alpha=0.75,
        linewidths=0.9,
        label=f"Rejected ($u > c(x)$): {n_trials - n_acc}",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, a_val * 1.16)
    ax.set_xlabel("Candidate Value $x \\sim q(x)$", fontsize=10)
    ax.set_ylabel("Density / Envelope Height $a\\,u$", fontsize=10)
    ax.set_title(
        f"{title_name}\nTheoretical Efficiency $1/a = {100.0*theo_eff:.1f}\\%$ · Empirical Acceptance $= {100.0*emp_eff:.1f}\\%$ ({n_acc}/{n_trials})",
        fontsize=9.8,
        fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.3)

    dim_10_eff = (theo_eff) ** 10
    return (
        f"**von Neumann Rejection Diagnostics ($N = {n_trials}$ proposals, $q(x)=1$):**\n"
        f"- **Envelope Multiplier:** $a = {a_val:.3f} \\ge \\max_x [p(x)/q(x)] = {a_min:.3f}$ so acceptance ratio $c(x) = p(x)/[a\\,q(x)] \\le 1$ everywhere.\n"
        f"- **Acceptance Efficiency:** Theoretical $= 1/a = {100.0*theo_eff:.2f}\\%$, Empirical $= {n_acc}/{n_trials} = {100.0*emp_eff:.2f}\\%$ "
        f"(requiring on average $a = {a_val:.2f}$ proposals per accepted point).\n"
        f"- **High-Dimensional Curse:** In $D = 10$ dimensions, this same envelope factor would collapse efficiency to $(1/a)^{{10}} = {dim_10_eff:.2e}$!"
    )


def render_cholesky_multivariate_gaussian(controls: dict[str, Any], ax: plt.Axes) -> str:
    """Render 2D correlated Gaussian generated via Cholesky decomposition S S^T = T."""
    sigma1 = float(_val(controls.get("sigma1"), 1.5))
    sigma2 = float(_val(controls.get("sigma2"), 1.0))
    rho = float(_val(controls.get("rho"), 0.75))
    n_samples = int(_val(controls.get("n_samples"), 400))

    rho = float(np.clip(rho, -0.98, 0.98))
    # Covariance matrix T = [[sigma1^2, rho*sigma1*sigma2], [rho*sigma1*sigma2, sigma2^2]]
    t11 = sigma1 ** 2
    t12 = rho * sigma1 * sigma2
    t22 = sigma2 ** 2

    # Analytic lower-triangular Cholesky factor S such that S @ S.T = T:
    # S = [[sigma1, 0], [rho*sigma2, sigma2*sqrt(1 - rho^2)]]
    s11 = sigma1
    s21 = rho * sigma2
    s22 = sigma2 * np.sqrt(max(1e-9, 1.0 - rho ** 2))
    S = np.array([[s11, 0.0], [s21, s22]])

    rng = np.random.default_rng(24)
    y_iso = rng.standard_normal((2, n_samples))
    x_corr = S @ y_iso

    emp_cov = np.cov(x_corr)
    emp_rho = emp_cov[0, 1] / np.sqrt(max(1e-12, emp_cov[0, 0] * emp_cov[1, 1]))

    ax.set_facecolor("#F8FAFC")
    ax.scatter(
        y_iso[0],
        y_iso[1],
        color="#94A3B8",
        s=14,
        alpha=0.4,
        label=r"Isotropic Seed $\mathbf{y} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$",
    )
    ax.scatter(
        x_corr[0],
        x_corr[1],
        color="#2563EB",
        s=18,
        alpha=0.75,
        edgecolors="#1E3A8A",
        linewidths=0.35,
        label=r"Correlated $\mathbf{x} = \mathbf{S}\mathbf{y}$",
    )

    # Plot 1-sigma and 2-sigma covariance ellipses by transforming unit circle
    theta = np.linspace(0, 2.0 * np.pi, 200)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])
    for k_sig, ls_style, lbl in [(1.0, "--", "$1\\sigma$ Ellipse"), (2.0, "-", "$2\\sigma$ Covariance Contour")]:
        ell = S @ (k_sig * unit_circle)
        ax.plot(ell[0], ell[1], color="#DC2626", lw=1.7 if k_sig == 2.0 else 1.2, ls=ls_style, label=lbl)

    lim = max(3.5, 2.6 * max(sigma1, sigma2))
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$x_1$", fontsize=10)
    ax.set_ylabel("$x_2$", fontsize=10)
    ax.set_title(
        f"Cholesky Transformation $\\mathbf{{x}} = \\mathbf{{S}}\\mathbf{{y}}$ ($\\mathbf{{S}}\\mathbf{{S}}^T = \\mathbf{{T}}$)\n"
        f"Target $\\rho = {rho:.2f}$ (Empirical $\\hat{{\\rho}} = {emp_rho:.2f}$)",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(loc="upper left", fontsize=7.8, framealpha=0.92)
    ax.grid(True, alpha=0.3)

    return (
        f"**Cholesky Factorization $\\mathbf{{S}}\\mathbf{{S}}^T = \\mathbf{{T}}$ ($N = {n_samples}$):**\n"
        f"- **Target Covariance Matrix:** $T_{{11}} = {t11:.3f},\\; T_{{12}} = T_{{21}} = {t12:.3f},\\; T_{{22}} = {t22:.3f}$.\n"
        f"- **Lower-Triangular Cholesky Factor:** $S_{{11}} = {s11:.3f},\\; S_{{12}} = 0,\\; S_{{21}} = {s21:.3f},\\; S_{{22}} = {s22:.3f}$.\n"
        f"- **Verification:** $\\langle x_1^2 \\rangle = {emp_cov[0,0]:.3f}$ (target ${t11:.3f}$), "
        f"$\\langle x_1 x_2 \\rangle = {emp_cov[0,1]:.3f}$ (target ${t12:.3f}$), "
        f"$\\langle x_2^2 \\rangle = {emp_cov[1,1]:.3f}$ (target ${t22:.3f}$)."
    )


def render_importance_sampling_variance_trap(controls: dict[str, Any], ax: plt.Axes) -> str:
    """Render the lecture's Gaussian Importance Sampling example f(x) = exp(-x^2/2)/(1+x^2) with p_a(x) = exp(-x^2/(2a))/sqrt(2*pi*a)."""
    a_val = float(_val(controls.get("a_width"), 0.60))
    view_mode = str(_val(controls.get("view_mode"), "estimator_trace"))
    n_samples = int(_val(controls.get("n_samples"), 500))
    seed = int(_val(controls.get("seed"), 19))

    # Exact integral I = \int_{-\infty}^{\infty} exp(-x^2/2)/(1+x^2) dx
    grid_x = np.linspace(-12.0, 12.0, 20001)
    f_grid = np.exp(-0.5 * grid_x ** 2) / (1.0 + grid_x ** 2)
    I_exact = float(np.trapezoid(f_grid, grid_x))  # ~ 1.64359

    # Draw N samples from p_a(x) = N(0, a) -> x = sqrt(a) * z
    rng = np.random.default_rng(seed)
    x_samples = np.sqrt(a_val) * rng.standard_normal(n_samples)
    f_vals = np.exp(-0.5 * x_samples ** 2) / (1.0 + x_samples ** 2)
    p_vals = np.exp(-0.5 * (x_samples ** 2) / a_val) / np.sqrt(2.0 * np.pi * a_val)
    weights = f_vals / p_vals  # estimator values g(x_i) = f(x_i)/p_a(x_i)

    sample_mean = float(np.mean(weights))
    sample_err = float(np.std(weights, ddof=1) / np.sqrt(n_samples))
    max_spike = float(np.max(weights))

    # Theoretical variance v(a) for a > 0.5: \int \sqrt{2\pi a} \frac{\exp(-x^2(1 - 1/(2a)))}{(1+x^2)^2} dx - I^2
    if a_val > 0.505:
        var_integrand = np.sqrt(2.0 * np.pi * a_val) * np.exp(-(grid_x ** 2) * (1.0 - 0.5 / a_val)) / ((1.0 + grid_x ** 2) ** 2)
        theo_var = max(0.0, float(np.trapezoid(var_integrand, grid_x)) - I_exact ** 2)
        theo_status = f"Finite Theoretical Variance $\\sigma^2(a) = {theo_var:.4f}$"
    else:
        theo_var = float("inf")
        theo_status = "INFINITE Theoretical Variance ($a \\le 0.5$: tail exponent $1 - 1/(2a) \\le 0$!)"

    ax.set_facecolor("#F8FAFC")
    if view_mode == "variance_curve":
        a_grid = np.linspace(0.515, 1.60, 120)
        v_curve = []
        for av in a_grid:
            vi = np.sqrt(2.0 * np.pi * av) * np.exp(-(grid_x ** 2) * (1.0 - 0.5 / av)) / ((1.0 + grid_x ** 2) ** 2)
            v_curve.append(max(1e-5, float(np.trapezoid(vi, grid_x)) - I_exact ** 2))
        v_curve_arr = np.array(v_curve)
        a_opt_idx = int(np.argmin(v_curve_arr))
        a_opt = float(a_grid[a_opt_idx])
        v_opt = float(v_curve_arr[a_opt_idx])

        ax.plot(a_grid, v_curve_arr, color="#1D4ED8", lw=2.3, label=r"Theoretical Variance $\sigma^2(a)$")
        ax.axvspan(0.0, 0.5, color="#EF4444", alpha=0.18, label=r"Infinite Variance Regime ($a \le 0.5$)")
        ax.axvline(0.5, color="#DC2626", lw=1.8, ls="--", label="Divergence Boundary $a = 0.5$")
        ax.scatter([a_opt], [v_opt], color="#10B981", s=70, zorder=5, label=f"Optimal Width $a^* \\approx {a_opt:.2f}$ ($\\sigma^2={v_opt:.4f}$)")
        if a_val > 0.515:
            ax.scatter([a_val], [theo_var], color="#F59E0B", s=85, marker="D", zorder=6, label=f"Selected $a = {a_val:.2f}$")
        ax.set_xlim(0.2, 1.6)
        ax.set_ylim(0.0, min(1.5, float(np.max(v_curve_arr))))
        ax.set_xlabel("Trial Gaussian Variance Parameter $a$ in $p_a(x) = \\exp(-x^2/2a)/\\sqrt{2\\pi a}$", fontsize=9.5)
        ax.set_ylabel("Estimator Variance $\\sigma^2(a)$ ( Proportional to CPU Time )", fontsize=9.5)
        ax.set_title("Lecture Slide 13–15: Relative CPU Time / Variance vs. Width $a$", fontsize=10.2, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        ax.grid(True, alpha=0.3)
    elif view_mode == "integrand_tails":
        xs_plot = np.linspace(-4.0, 4.0, 400)
        f_plot = np.exp(-0.5 * xs_plot ** 2) / (1.0 + xs_plot ** 2)
        p_opt = np.abs(f_plot) / I_exact
        p_sel = np.exp(-0.5 * (xs_plot ** 2) / a_val) / np.sqrt(2.0 * np.pi * a_val)
        ax.plot(xs_plot, p_opt, color="#0F172A", lw=2.2, label=r"Optimal $p^*(x) = |f(x)| / I$")
        ax.plot(xs_plot, p_sel, color="#2563EB", lw=2.0, ls="--", label=f"Trial Gaussian $p_a(x)$ ($a = {a_val:.2f}$)")
        ax.set_xlabel("$x$", fontsize=10)
        ax.set_ylabel("Probability Density", fontsize=10)
        ax.set_title(f"Importance Function Matching: Optimal $p^*(x)$ vs. Gaussian $p_a(x)$ ($a={a_val:.2f}$)", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right", fontsize=8.5)
        ax.grid(True, alpha=0.3)
    else:
        # Estimator trace reproducing Slide 16 ("What does infinite variance look like? Spikes!")
        steps = np.arange(1, n_samples + 1)
        trace_color = "#DC2626" if a_val <= 0.5 else ("#10B981" if 0.55 <= a_val <= 0.75 else "#2563EB")
        ax.plot(steps, weights, color=trace_color, lw=0.9, alpha=0.9, label="Sample Estimator $f(x_i)/p_a(x_i)$")
        ax.axhline(I_exact, color="#0F172A", lw=1.6, ls="--", label=f"Exact Integral $I = {I_exact:.4f}$")
        ax.set_xlabel("Monte Carlo Step $i$", fontsize=10)
        ax.set_ylabel("Estimator Value $f(x_i) / p_a(x_i)$", fontsize=10)
        status_tag = "INFINITE VARIANCE SPIKES!" if a_val <= 0.5 else ("Near-Optimal Fluctuations" if 0.55 <= a_val <= 0.75 else "Finite Variance (Over-Sampled Tails)")
        ax.set_title(
            f"Lecture Slide 16 Trace ($a = {a_val:.2f}$) — {status_tag}\n"
            f"Estimate $\\hat{{I}} = {sample_mean:.4f} \\pm {sample_err:.4f}$ (Max Spike $= {max_spike:.2f}$)",
            fontsize=9.8,
            fontweight="bold",
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
        ax.grid(True, alpha=0.3)

    return (
        f"**Gaussian Importance Sampling Analysis ($a = {a_val:.2f},\\; N = {n_samples}$):**\n"
        f"- **Variance Exponent Check:** The variance integral contains $\\exp[-x^2(1 - 1/(2a))]$. {theo_status}.\n"
        f"- **Empirical Estimate:** $\\hat{{I}} = {sample_mean:.4f} \\pm {sample_err:.4f}$ vs. exact $I = {I_exact:.4f}$ (largest single-sample weight $= {max_spike:.2f}$).\n"
        f"- **Golden Rule of Importance Sampling:** *Never under-sample the tails!* Choosing $a \\le 0.5$ makes $p_a(x)$ decay faster than $\\sqrt{{|f(x)|}}$, producing catastrophic Pareto spikes."
    )
