"""Interactive concept visualizations for Lesson 01: Introduction to Atomic-Scale Simulations & Probability Statistics."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from engine.plot_style import COLORS


def render_multiscale_map(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render length-vs-time multiscale map and compute particle-step budget."""
    method = str(controls.get("highlight_regime", "md"))
    flop_exp = int(controls.get("flop_rate_log10", 15))  # 10^15 FLOP/s = 1 PFLOP/s
    wall_hours = float(controls.get("wall_hours", 24.0))

    regimes = {
        "qm": {
            "label": "QM / Ab Initio DFT",
            "x_range": (-0.5, 1.5),   # log10(Angstrom): 0.3 A to 30 A
            "y_range": (-15.5, -11.0), # log10(s): sub-fs to 10 ps
            "color": COLORS["quaternary"],
            "flops_per_atom_step": 1e9,
            "desc": "Solves electronic Schrödinger / Kohn-Sham equations directly; captures bond breaking/formation but scales as $\\mathcal{O}(N_e^3)$.",
        },
        "md": {
            "label": "Classical MD",
            "x_range": (1.0, 4.0),    # 1 nm to 1 um (10 to 10^4 A)
            "y_range": (-12.5, -6.0), # ps to us
            "color": COLORS["primary"],
            "flops_per_atom_step": 1e3,
            "desc": "Integrates Newton's $F=ma$ with empirical/ML interatomic potentials ($\\Delta t \\sim 1\\text{ fs}$); scales linearly $\\mathcal{O}(N)$.",
        },
        "kmc": {
            "label": "Accelerated MD / KMC",
            "x_range": (1.5, 4.5),    # nm to um
            "y_range": (-8.0, 1.0),   # 10 ns to 10 s
            "color": COLORS["secondary"],
            "flops_per_atom_step": 1e4,
            "desc": "Coarse-grains over vibrational periods via rare-event state-to-state Markov jumps ($S_n \\to S_{n+1}$) to reach diffusive timescales.",
        },
        "meso": {
            "label": "Mesoscale / Phase-Field",
            "x_range": (3.0, 6.5),    # 100 nm to 0.3 mm
            "y_range": (-6.0, 3.0),   # us to hours
            "color": COLORS["accent"],
            "flops_per_atom_step": 1e2,
            "desc": "Replaces discrete atoms with coarse-grained beads, dislocation lines, or continuous order parameters.",
        },
        "continuum": {
            "label": "Continuum / FEM",
            "x_range": (5.0, 9.5),    # 10 um to meters
            "y_range": (-3.0, 6.0),   # ms to semanas
            "color": COLORS["good"],
            "flops_per_atom_step": 1e1,
            "desc": "Solves macroscopic PDEs (elasticity, Navier-Stokes, heat conduction) using constitutive laws parameterized from lower scales.",
        },
    }

    for key, info in regimes.items():
        x0, x1 = info["x_range"]
        y0, y1 = info["y_range"]
        is_sel = key == method
        alpha_fill = 0.38 if is_sel else 0.14
        lw = 2.4 if is_sel else 1.1
        rect = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor=info["color"],
            edgecolor=info["color"],
            alpha=alpha_fill,
            linewidth=lw,
            zorder=3 if is_sel else 2,
        )
        ax.add_patch(rect)
        ax.plot(
            [x0, x1, x1, x0, x0],
            [y0, y0, y1, y1, y0],
            color=info["color"],
            linewidth=lw,
            linestyle="-" if is_sel else "--",
            zorder=4 if is_sel else 2,
        )
        ax.text(
            0.5 * (x0 + x1),
            0.5 * (y0 + y1),
            info["label"],
            ha="center",
            va="center",
            fontsize=8.8 if is_sel else 8.0,
            fontweight="bold" if is_sel else "normal",
            color=COLORS["text"] if is_sel else info["color"],
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.82 if is_sel else 0.65),
            zorder=5,
        )

    ax.set_xlim(-1.0, 10.0)
    ax.set_ylim(-16.5, 7.0)
    ax.set_xlabel(r"Length Scale $\log_{10}(L / \text{Å})$  [$0=\text{Å},\ 1=\mathrm{nm},\ 4=\mu\mathrm{m},\ 7=\mathrm{mm}$]")
    ax.set_ylabel(r"Time Scale $\log_{10}(t / \mathrm{s})$  [$-15=\mathrm{fs},\ -12=\mathrm{ps},\ -9=\mathrm{ns},\ 0=\mathrm{s}$]")
    ax.set_title("Hierarchy of Materials Simulation Scales (Length vs. Time)")
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sel = regimes.get(method, regimes["md"])
    total_flops = (10.0 ** flop_exp) * (wall_hours * 3600.0)
    budget_atom_steps = total_flops / sel["flops_per_atom_step"]
    # For a 1-million-step run (e.g., 1 ns at 1 fs step), how many particles/DOF can we afford?
    n_particles_at_1m_steps = budget_atom_steps / 1e6

    return (
        f"**Selected Regime — {sel['label']}:** {sel['desc']}\n\n"
        f"- **Compute Hardware:** $10^{{{flop_exp}}}\\text{{ FLOP/s}}$ running for ${wall_hours:.1f}\\text{{ hours}}$ "
        f"delivers $N_{{\\text{{FLOP}}}} = {total_flops:.2e}\\text{{ FLOPs}}$.\n"
        f"- **Particle-Step Budget ($N_{{\\text{{particles}}}} \\times N_{{\\text{{steps}}}}$):** "
        f"At $\\sim 10^{{{int(np.log10(sel['flops_per_atom_step']))}}}\\text{{ FLOPs}}$ per degree-of-freedom update, "
        f"total budget is **${budget_atom_steps:.2e}$ particle-steps** "
        f"(e.g., $N_{{\\text{{particles}}}} \\approx {n_particles_at_1m_steps:.2e}$ for $10^6$ time steps)."
    )


def render_phase_space_oscillator(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render 1D harmonic oscillator constant-energy phase space ellipses (x, p)."""
    energy = float(controls.get("energy", 2.0))
    spring_k = float(controls.get("spring_k", 1.5))
    mass = float(controls.get("mass", 1.0))

    theta = np.linspace(0.0, 2.0 * np.pi, 400)
    # Reference background energy contours
    for e_ref in (0.5, 1.0, 2.0, 3.5, 5.0):
        x_r = np.sqrt(2.0 * e_ref / spring_k) * np.cos(theta)
        p_r = np.sqrt(2.0 * mass * e_ref) * np.sin(theta)
        ax.plot(x_r, p_r, color=COLORS["muted"], linestyle=":", linewidth=1.0, alpha=0.6)

    x_max = np.sqrt(2.0 * energy / spring_k)
    p_max = np.sqrt(2.0 * mass * energy)
    omega = np.sqrt(spring_k / mass)

    x_traj = x_max * np.cos(theta)
    p_traj = p_max * np.sin(theta)
    ax.plot(x_traj, p_traj, color=COLORS["primary"], linewidth=2.6, label=f"Microstate Orbit ($E={energy:.2f}$)")
    ax.fill(x_traj, p_traj, color=COLORS["primary"], alpha=0.10)

    # Mark turning points and maximum momentum points
    ax.scatter([x_max, -x_max], [0.0, 0.0], color=COLORS["accent"], s=55, zorder=5, label=r"Turning points ($K=0, U=E$)")
    ax.scatter([0.0, 0.0], [p_max, -p_max], color=COLORS["secondary"], s=55, zorder=5, label=r"Max momentum ($U=0, K=E$)")

    ax.axhline(0.0, color=COLORS["text"], linewidth=0.8, alpha=0.5)
    ax.axvline(0.0, color=COLORS["text"], linewidth=0.8, alpha=0.5)

    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.set_xlabel(r"Position Coordinate $x - x_0$ ($\sqrt{2E/k}$ intercept)")
    ax.set_ylabel(r"Momentum $p = mv$ ($\sqrt{2mE}$ intercept)")
    ax.set_title(r"Harmonic Oscillator Phase Space: $\mathcal{H}(x,p) = \frac{p^2}{2m} + \frac{1}{2}k(x-x_0)^2 = E$")
    ax.legend(loc="upper right", fontsize=8.2)
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    area = np.pi * x_max * p_max
    return (
        f"**Phase Space Ellipse Parameters ($E = {energy:.2f},\\ k = {spring_k:.2f},\\ m = {mass:.2f}$):**\n\n"
        f"- **Position semi-axis:** $x_{{\\max}} = \\sqrt{{2E/k}} = {x_max:.3f}$ | "
        f"**Momentum semi-axis:** $p_{{\\max}} = \\sqrt{{2mE}} = {p_max:.3f}$\n"
        f"- **Angular frequency:** $\\omega = \\sqrt{{k/m}} = {omega:.3f}\\text{{ rad/s}}$ | "
        f"**Enclosed phase-space area:** $\\oint p\\,dx = \\pi x_{{\\max}} p_{{\\max}} = \\frac{{2\\pi E}}{{\\omega}} = {area:.3f}$"
    )


def render_moments_and_covariance(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render 2D correlated Gaussian scatter, covariance ellipse, and principal axes."""
    rho = float(controls.get("rho", 0.6))
    sigma_x = float(controls.get("sigma_x", 1.2))
    sigma_y = float(controls.get("sigma_y", 1.0))

    cov_xy = rho * sigma_x * sigma_y
    var_sum = sigma_x**2 + sigma_y**2 + 2.0 * cov_xy
    var_diff = sigma_x**2 + sigma_y**2 - 2.0 * cov_xy

    cov_mat = np.array([[sigma_x**2, cov_xy], [cov_xy, sigma_y**2]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-8)
    eigvecs = eigvecs[:, order]

    rng = np.random.default_rng(48501)
    z = rng.standard_normal((2, 350))
    # Cholesky / eigen transform for exact deterministic visualization
    transform = eigvecs @ np.diag(np.sqrt(eigvals))
    pts = transform @ z

    ax.scatter(pts[0], pts[1], s=14, color=COLORS["primary"], alpha=0.35, edgecolors="none", label="Sampled microstates $(X_i, Y_i)$")

    angle_deg = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    for n_std, style, alpha_e in [(1.0, "-", 0.25), (2.0, "--", 0.12)]:
        ell = Ellipse(
            xy=(0.0, 0.0),
            width=2.0 * n_std * np.sqrt(eigvals[0]),
            height=2.0 * n_std * np.sqrt(eigvals[1]),
            angle=angle_deg,
            facecolor=COLORS["secondary"],
            edgecolor=COLORS["secondary"],
            linestyle=style,
            linewidth=2.0 if n_std == 1.0 else 1.4,
            alpha=alpha_e,
        )
        ax.add_patch(ell)
        ell_border = Ellipse(
            xy=(0.0, 0.0),
            width=2.0 * n_std * np.sqrt(eigvals[0]),
            height=2.0 * n_std * np.sqrt(eigvals[1]),
            angle=angle_deg,
            facecolor="none",
            edgecolor=COLORS["secondary"],
            linestyle=style,
            linewidth=2.0 if n_std == 1.0 else 1.3,
            label=f"${int(n_std)}\\sigma$ Covariance Contour" if n_std == 1.0 else f"$2\\sigma$ Contour",
        )
        ax.add_patch(ell_border)

    # Draw principal axes
    for idx, col, lbl in [(0, COLORS["accent"], "Major Principal Axis ($\\lambda_1$)"), (1, COLORS["quaternary"], "Minor Principal Axis ($\\lambda_2$)")]:
        scale = 2.0 * np.sqrt(eigvals[idx])
        dx = scale * eigvecs[0, idx]
        dy = scale * eigvecs[1, idx]
        ax.plot([-dx, dx], [-dy, dy], color=col, linewidth=2.2, label=lbl)

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.set_xlabel(r"Fluctuation $\delta X = X - \langle X \rangle$")
    ax.set_ylabel(r"Fluctuation $\delta Y = Y - \langle Y \rangle$")
    ax.set_title(f"2D Covariance Matrix $V$ & Principal Axes ($\\rho = {rho:+.2f}$)")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    det_v = float(np.linalg.det(cov_mat))
    return (
        f"**Covariance & Linear Combinations ($\\sigma_X={sigma_x:.2f},\\ \\sigma_Y={sigma_y:.2f},\\ \\rho={rho:+.2f}$):**\n\n"
        f"- **Covariance:** $\\mathrm{{Cov}}(X,Y) = V_{{12}} = \\rho\\sigma_X\\sigma_Y = {cov_xy:+.3f}$ | "
        f"$\\det(V) = \\sigma_X^2\\sigma_Y^2(1-\\rho^2) = {det_v:.3f}$\n"
        f"- **Principal Variances (Eigenvalues of $V$):** $\\lambda_1 = {eigvals[0]:.3f},\\ \\lambda_2 = {eigvals[1]:.3f}$\n"
        f"- **Sum & Difference Variances:** $\\mathrm{{Var}}(X+Y) = \\sigma_X^2 + \\sigma_Y^2 + 2\\mathrm{{Cov}}(X,Y) = **{var_sum:.3f}**$, "
        f"while $\\mathrm{{Var}}(X-Y) = **{var_diff:.3f}**$."
    )


def render_clt_vs_lorentzian(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare N-sample averages of finite-variance distributions vs. Cauchy/Lorentzian."""
    dist_name = str(controls.get("distribution", "uniform"))
    n_avg = int(controls.get("sample_size_n", 12))

    rng = np.random.default_rng(48502)
    n_trials = 6000

    if dist_name == "uniform":
        # Uniform on [-sqrt(3), sqrt(3)] -> mean 0, variance 1
        a = np.sqrt(3.0)
        samples = rng.uniform(-a, a, size=(n_trials, n_avg))
        mu_true, sigma_true = 0.0, 1.0
        title_lbl = r"Uniform $[-\sqrt{3}, \sqrt{3}]$ ($\mu=0, \sigma^2=1$)"
    elif dist_name == "exponential":
        # Shifted exponential Exp(1) - 1 -> mean 0, variance 1
        samples = rng.exponential(scale=1.0, size=(n_trials, n_avg)) - 1.0
        mu_true, sigma_true = 0.0, 1.0
        title_lbl = r"Shifted Exponential ($\mu=0, \sigma^2=1, \kappa_3=2$)"
    elif dist_name == "bimodal":
        # Symmetric two-peak mixture at +/- 0.9 with std 0.43589 -> mean 0, variance 1
        signs = rng.choice([-0.9, 0.9], size=(n_trials, n_avg))
        noise = rng.normal(0.0, np.sqrt(1.0 - 0.81), size=(n_trials, n_avg))
        samples = signs + noise
        mu_true, sigma_true = 0.0, 1.0
        title_lbl = r"Symmetric Bimodal Mixture ($\mu=0, \sigma^2=1$)"
    else:
        # Cauchy / Lorentzian p(x) = 1 / [pi (1 + x^2)] -> infinite variance!
        samples = rng.standard_cauchy(size=(n_trials, n_avg))
        mu_true, sigma_true = 0.0, np.inf
        title_lbl = r"Cauchy / Lorentzian $p(x)=\frac{1}{\pi(1+x^2)}$ ($\sigma^2 = \infty$)"

    means = np.mean(samples, axis=1)
    x_grid = np.linspace(-3.5, 3.5, 500)

    clipped_means = means[(means >= -3.5) & (means <= 3.5)]
    ax.hist(
        clipped_means,
        bins=55,
        range=(-3.5, 3.5),
        density=True,
        color=COLORS["primary"] if np.isfinite(sigma_true) else COLORS["danger"],
        alpha=0.55,
        edgecolor="white",
        label=f"Empirical PDF of $\\bar{{X}}_{{{n_avg}}}$ ($N={n_avg}$)",
    )

    if np.isfinite(sigma_true):
        se = sigma_true / np.sqrt(n_avg)
        gauss_pdf = (1.0 / (np.sqrt(2.0 * np.pi) * se)) * np.exp(-0.5 * ((x_grid - mu_true) / se) ** 2)
        ax.plot(x_grid, gauss_pdf, color=COLORS["accent"], linewidth=2.5, label=f"CLT Gaussian ($\\sigma/\\sqrt{{N}} = {se:.3f}$)")
        single_gauss = (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x_grid**2)
        ax.plot(x_grid, single_gauss, color=COLORS["muted"], linestyle="--", linewidth=1.4, label="Single-sample ref ($N=1, \\sigma=1$)")
        ax.set_ylim(0.0, max(1.6, float(np.max(gauss_pdf)) * 1.15))
    else:
        cauchy_pdf = 1.0 / (np.pi * (1.0 + x_grid**2))
        ax.plot(x_grid, cauchy_pdf, color=COLORS["Text"] if "Text" in COLORS else "#1E293B", linewidth=2.5, linestyle="-", label=r"Single-Draw Lorentzian $\frac{1}{\pi(1+x^2)}$")
        ax.set_ylim(0.0, 0.45)

    ax.set_xlim(-3.5, 3.5)
    ax.set_xlabel(r"Sample Mean $y = \frac{1}{N}\sum_{i=1}^N X_i$")
    ax.set_ylabel("Probability Density $P(y)$")
    ax.set_title(f"Central Limit Theorem Convergence: {title_lbl}")
    ax.legend(loc="upper right", fontsize=8.2)
    ax.grid(True, color=COLORS["grid"], linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if np.isfinite(sigma_true):
        se = sigma_true / np.sqrt(n_avg)
        q25, q75 = np.percentile(means, [25, 75])
        return (
            f"**CLT Convergence ($N = {n_avg}$ independent draws from `{dist_name}`):**\n\n"
            f"- **Second Cumulant (Variance of Mean):** $\\kappa_2(y) = \\sigma^2 / N = {1.0/n_avg:.4f}$ "
            f"$\\implies$ **Standard Error:** $\\sigma_{{\\bar{{X}}}} = \\sigma/\\sqrt{{N}} = **{se:.3f}**$.\n"
            f"- **Higher Cumulants Suppressed:** Skewness cumulant scales as $\\kappa_3(y) = \\kappa_3 / N^2$ "
            f"(reduced by factor $N^2 = {n_avg**2}$) and kurtosis as $\\kappa_4(y) = \\kappa_4 / N^3$ (factor ${n_avg**3}$).\n"
            f"- **Empirical IQR:** ${q75 - q25:.3f}$ (theoretical Gaussian IQR $= 1.349\\,\\sigma/\\sqrt{{N}} = {1.349*se:.3f}$)."
        )
    else:
        q25, q75 = np.percentile(means, [25, 75])
        max_spike = float(np.max(np.abs(means)))
        return (
            f"**CLT Failure for Cauchy/Lorentzian ($N = {n_avg}$ independent draws):**\n\n"
            f"- **Why CLT Fails:** The tails decay only as $F^*(x) \\sim x^{{-2}}$, violating $\\lim_{{x\\to\\pm\\infty}} x^3 F^*(x) = 0$. "
            f"Thus the raw second moment $I_2 = \\int_{{-\\infty}}^\\infty x^2 F^*(x)\\,dx = \\infty$ diverges!\n"
            f"- **No Narrowing with $N$:** By characteristic functions, $c_x(k) = e^{{-|k|}}$, so $c_y(k) = [c_x(k/N)]^N = (e^{{-|k|/N}})^N = e^{{-|k|}}$. "
            f"The average of $N={n_avg}$ Cauchy draws has the **exact same Lorentzian distribution** as $N=1$ (empirical IQR $\\approx {q75-q25:.2f}$ vs. theoretical $2.00$, max outlier spike $|y|_{{\\max}} = {max_spike:.1f}$)."
        )
