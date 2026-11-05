"""Interactive concept-cell renderers for Lesson 13: PRNGs & Statistical Tests."""

from __future__ import annotations
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaincc


def _val(ctrl: Any, default: Any = None) -> Any:
    if ctrl is None:
        return default
    return getattr(ctrl, "value", ctrl)


def _generate_lcg(a: int, c: int, m: int, seed: int, count: int) -> tuple[np.ndarray, int]:
    """Generate `count` integers from LCG(a, c, m, seed) and compute exact cycle period from seed."""
    seq = np.empty(count, dtype=np.int64)
    state = int(seed) % int(m)
    seen: dict[int, int] = {}
    period = 0
    for idx in range(count):
        if period == 0:
            if state in seen:
                period = idx - seen[state]
            else:
                seen[state] = idx
        state = (int(a) * state + int(c)) % int(m)
        seq[idx] = state
    if period == 0:
        # Continue up to m+2 steps to find exact period if m is small
        max_steps = min(int(m) + 2, 10000)
        for idx in range(count, max_steps):
            if state in seen:
                period = idx - seen[state]
                break
            seen[state] = idx
            state = (int(a) * state + int(c)) % int(m)
        if period == 0:
            period = len(seen)
    return seq, period


def render_lcg_hyperplanes(controls: dict[str, Any], ax: plt.Axes) -> str:
    """Render 2D delay-space plot (u_n, u_{n+1}) and Marsaglia hyperplane structure for LCGs."""
    preset = str(_val(controls.get("preset"), "full_16"))
    seed = int(_val(controls.get("seed"), 0))
    n_points = int(_val(controls.get("n_points"), 256))

    presets = {
        "full_16": (5, 1, 16, "LCG(5, 1, 16) — Full Period (Hull-Dobell satisfied)"),
        "short_16": (4, 1, 16, "LCG(4, 1, 16) — Degraded Multiplier (Period 4)"),
        "even_c_16": (5, 2, 16, "LCG(5, 2, 16) — Even Increment c=2 (Half Period 8)"),
        "marsaglia_256": (13, 1, 256, "LCG(13, 1, 256) — 2D Hyperplane Lattice"),
        "bad_256": (65, 1, 256, "LCG(65, 1, 256) — Severe Marsaglia Planes (4 lines)"),
        "randu_like": (19, 0, 256, "LCG(19, 0, 256) — Multiplicative Odd/Even Split"),
    }
    a, c, m, title_label = presets.get(preset, presets["full_16"])
    if preset == "randu_like" and seed % 2 == 0:
        eff_seed = seed + 1
    else:
        eff_seed = seed % m

    seq, period = _generate_lcg(a, c, m, eff_seed, max(n_points + 2, m + 2))
    u = seq.astype(float) / float(m)

    # All possible grid points or delay pairs
    u_x = u[:n_points]
    u_y = u[1 : n_points + 1]

    ax.set_facecolor("#F8FAFC")
    # Plot faint theoretical line segments of (x, (a*x + c/m) mod 1)
    xs = np.linspace(0, 1, 1000)
    ys = (a * xs + (c / float(m))) % 1.0
    # Break wraparound jumps
    jumps = np.abs(np.diff(ys)) > 0.5
    xs_plot = xs.copy()
    ys_plot = ys.copy()
    xs_plot[:-1][jumps] = np.nan
    ys_plot[:-1][jumps] = np.nan
    ax.plot(xs_plot, ys_plot, color="#94A3B8", lw=1.1, ls="--", alpha=0.7, label="LCG Modular Branches")

    # Connect first 16 trajectory steps with faint arrows
    trace_len = min(16, len(u_x) - 1)
    ax.plot(u_x[:trace_len], u_y[:trace_len], color="#CBD5E1", lw=1.0, zorder=2)

    scatter = ax.scatter(
        u_x,
        u_y,
        c=np.arange(len(u_x)),
        cmap="viridis",
        s=42 if m <= 16 else 24,
        edgecolors="#0F172A",
        linewidths=0.5,
        zorder=3,
        label=f"Generated Pairs $(u_n, u_{{n+1}})$",
    )
    ax.scatter([u_x[0]], [u_y[0]], color="#EF4444", s=85, marker="*", zorder=4, label=f"Start ($I_0={eff_seed}$)")

    marsaglia_bound_2d = (2.0 * m) ** 0.5
    marsaglia_bound_3d = (6.0 * m) ** (1.0 / 3.0)

    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("$u_n = I_n / m$", fontsize=10)
    ax.set_ylabel("$u_{n+1} = I_{n+1} / m$", fontsize=10)
    ax.set_title(f"{title_label}\nCycle Period $L = {period}$ (Max Possible $m = {m}$)", fontsize=10.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    first_eight = ", ".join(str(int(x)) for x in seq[: min(16, period + 1)])
    LSB_seq = ", ".join(str(int(x) % 2) for x in seq[:8])
    return (
        f"**LCG Diagnostics (`a={a}, c={c}, m={m}, I_0={eff_seed}`):**\n"
        f"- **Observed Cycle Length:** $L = {period}$ out of $m = {m}$ states ({100.0 * period / m:.1f}% state coverage).\n"
        f"- **Integer Trajectory:** `{first_eight}, ...` (LSBs mod 2: `{LSB_seq}...`).\n"
        f"- **Marsaglia Hyperplane Bound:** In $d=2$, points lie on at most $(2!\\,m)^{{1/2}} \\approx {marsaglia_bound_2d:.1f}$ parallel lines; "
        f"in $d=3$, at most $(3!\\,m)^{{1/3}} \\approx {marsaglia_bound_3d:.1f}$ planes."
    )


def _van_der_corput(n: int, base: int) -> np.ndarray:
    """Generate first `n` points of the 1D van der Corput radical-inverse sequence in `base`."""
    out = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        val = 0.0
        f = 1.0 / float(base)
        num = i
        while num > 0:
            val += (num % base) * f
            num //= base
            f /= float(base)
        out[i - 1] = val
    return out


def render_prng_vs_qrng_discrepancy(controls: dict[str, Any], ax: plt.Axes) -> str:
    """Compare 2D Pseudo-Random (PRNG) clumping vs Quasi-Random (Halton QRNG) low-discrepancy space filling."""
    mode = str(_val(controls.get("mode"), "side_by_side"))
    n_samples = int(_val(controls.get("n_samples"), 256))
    seed = int(_val(controls.get("seed"), 42))

    rng = np.random.default_rng(seed)
    prng_pts = rng.random((n_samples, 2))
    qrng_x = _van_der_corput(n_samples, 2)
    qrng_y = _van_der_corput(n_samples, 3)
    qrng_pts = np.column_stack([qrng_x, qrng_y])

    # Test integral: f(x, y) = (x^2 + y^2 <= 1) quarter-circle area pi/4, or smooth f(x,y) = 4*x*y (exact = 1.0)
    # Also compute grid occupancy uniformity on an 8x8 mesh
    grid_k = 8
    prng_counts, _, _ = np.histogram2d(prng_pts[:, 0], prng_pts[:, 1], bins=grid_k, range=[[0, 1], [0, 1]])
    qrng_counts, _, _ = np.histogram2d(qrng_pts[:, 0], qrng_pts[:, 1], bins=grid_k, range=[[0, 1], [0, 1]])
    expected_per_cell = n_samples / float(grid_k * grid_k)
    prng_empty = int(np.sum(prng_counts == 0))
    qrng_empty = int(np.sum(qrng_counts == 0))

    # Test smooth 2D integral I = \int_0^1 \int_0^1 (\pi/2) \sin(\pi x)\sin(\pi y) dx dy ... let's use exact=1.0: (\pi/2)^2 \sin(\pi x)\sin(\pi y)
    f_prng = (0.5 * np.pi) ** 2 * np.sin(np.pi * prng_pts[:, 0]) * np.sin(np.pi * prng_pts[:, 1])
    f_qrng = (0.5 * np.pi) ** 2 * np.sin(np.pi * qrng_pts[:, 0]) * np.sin(np.pi * qrng_pts[:, 1])
    err_prng = abs(float(np.mean(f_prng)) - 1.0)
    err_qrng = abs(float(np.mean(f_qrng)) - 1.0)

    ax.set_facecolor("#F8FAFC")
    if mode == "error_scaling":
        ns = np.unique(np.geomspace(16, max(n_samples, 64), 25).astype(int))
        errs_p = [abs(float(np.mean(f_prng[:k])) - 1.0) + 1e-6 for k in ns]
        errs_q = [abs(float(np.mean(f_qrng[:k])) - 1.0) + 1e-6 for k in ns]
        ax.loglog(ns, errs_p, "o-", color="#EF4444", lw=1.6, ms=4, label="PRNG Error (Uniform Random)")
        ax.loglog(ns, errs_q, "s-", color="#0284C7", lw=1.8, ms=4, label="QRNG Error (Halton Base 2,3)")
        ax.loglog(ns, 0.6 * ns ** (-0.5), "--", color="#991B1B", lw=1.2, label=r"MC Scaling $\propto N^{-1/2}$")
        ax.loglog(ns, 1.2 * (np.log(ns) ** 2) / ns, ":", color="#0369A1", lw=1.4, label=r"QRNG Scaling $\propto (\ln N)^2 / N$")
        ax.set_xlabel("Number of Samples $N$", fontsize=10)
        ax.set_ylabel("Absolute Quadrature Error $|\\hat{I}_N - 1|$", fontsize=10)
        ax.set_title("2D Integration Error Convergence: PRNG vs. Low-Discrepancy QRNG", fontsize=10.5, fontweight="bold")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
    elif mode == "prng_only":
        ax.scatter(prng_pts[:, 0], prng_pts[:, 1], color="#EF4444", s=20, alpha=0.8, edgecolors="#7F1D1D", linewidths=0.4)
        ax.set_xticks(np.linspace(0, 1, grid_k + 1))
        ax.set_yticks(np.linspace(0, 1, grid_k + 1))
        ax.grid(True, color="#94A3B8", ls="-", lw=0.7, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Pseudo-Random (PRNG, N={n_samples}): Poisson Clumps & {prng_empty} Empty 8x8 Subcells", fontsize=10, fontweight="bold")
    elif mode == "qrng_only":
        ax.scatter(qrng_pts[:, 0], qrng_pts[:, 1], color="#0284C7", s=20, alpha=0.85, edgecolors="#0C4A6E", linewidths=0.4)
        ax.set_xticks(np.linspace(0, 1, grid_k + 1))
        ax.set_yticks(np.linspace(0, 1, grid_k + 1))
        ax.grid(True, color="#94A3B8", ls="-", lw=0.7, alpha=0.6)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Quasi-Random (Halton QRNG, N={n_samples}): Space-Filling & {qrng_empty} Empty 8x8 Subcells", fontsize=10, fontweight="bold")
    else:
        # Side-by-side split in single axes: x in [0, 1] for PRNG, x in [1.15, 2.15] for QRNG
        ax.scatter(prng_pts[:, 0], prng_pts[:, 1], color="#EF4444", s=16, alpha=0.75, label=f"PRNG (err={err_prng:.4f})")
        ax.scatter(qrng_pts[:, 0] + 1.15, qrng_pts[:, 1], color="#0284C7", s=16, alpha=0.8, label=f"QRNG Halton (err={err_qrng:.4f})")
        ax.axvline(1.075, color="#334155", lw=1.5, ls="--")
        ax.text(0.5, 1.03, f"PRNG ({prng_empty} voids)", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#B91C1C")
        ax.text(1.65, 1.03, f"QRNG ({qrng_empty} voids)", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0369A1")
        ax.set_xlim(-0.02, 2.17)
        ax.set_ylim(0.0, 1.12)
        ax.set_xticks([0.0, 0.5, 1.0, 1.15, 1.65, 2.15])
        ax.set_xticklabels(["0.0", "0.5", "1.0", "0.0", "0.5", "1.0"])
        ax.set_ylabel("$y \\in [0, 1)$", fontsize=10)
        ax.set_title(f"2D Discrepancy Comparison ($N = {n_samples}$ points)", fontsize=10.5, fontweight="bold")
        ax.legend(loc="lower center", ncol=2, fontsize=8, framealpha=0.95)
        ax.grid(True, alpha=0.25)

    return (
        f"**PRNG vs. QRNG Summary ($N = {n_samples}$, $8\\times 8$ Grid Expected $= {expected_per_cell:.2f}$/cell):**\n"
        f"- **Pseudo-Random (PRNG):** Uncorrelated draws exhibit Poisson fluctuations ($\\sigma_n = \\sqrt{{n_i}}$), leaving **{prng_empty}** empty cells and integration error $|\\hat{{I}} - 1| = {err_prng:.5f}$ ($\\mathcal{{O}}(N^{{-1/2}})$).\n"
        f"- **Quasi-Random (Halton QRNG):** Sequentially fills largest voids by sacrificing serial independence, leaving **{qrng_empty}** empty cells and integration error $|\\hat{{I}} - 1| = {err_qrng:.5f}$ ($\\mathcal{{O}}((\\ln N)^2/N)$)."
    )


def render_chi_squared_diagnostic(controls: dict[str, Any], ax: plt.Axes) -> str:
    """Render binned histogram of generator draws and compute Pearson's chi-squared statistic and Q(chi^2|nu)."""
    gen_type = str(_val(controls.get("generator"), "good_prng"))
    n_bins = int(_val(controls.get("n_bins"), 20))
    n_samples = int(_val(controls.get("n_samples"), 1000))
    seed = int(_val(controls.get("seed"), 7))

    rng = np.random.default_rng(seed)
    expected = n_samples / float(n_bins)
    sigma_bin = np.sqrt(expected)

    if gen_type == "good_prng":
        samples = rng.random(n_samples)
        counts, _ = np.histogram(samples, bins=n_bins, range=(0.0, 1.0))
        label_name = "Genuine Uniform PRNG (Expected Fluctuations)"
    elif gen_type == "biased_lcg":
        # Short cycle or non-uniform power transform
        u = rng.random(n_samples)
        samples = np.mod(u ** 1.28, 1.0)
        counts, _ = np.histogram(samples, bins=n_bins, range=(0.0, 1.0))
        label_name = "Biased / Correlated Generator (Too Lumpy)"
    elif gen_type == "short_cycle_lcg":
        seq, _ = _generate_lcg(5, 2, 32, 1, n_samples)
        samples = seq.astype(float) / 32.0
        counts, _ = np.histogram(samples, bins=n_bins, range=(0.0, 1.0))
        label_name = "Short-Cycle LCG(5, 2, 32) (Missing States)"
    else:
        # Quasi-random / over-uniform fake generator
        samples = _van_der_corput(n_samples, 2)
        counts, _ = np.histogram(samples, bins=n_bins, range=(0.0, 1.0))
        label_name = "Quasi-Random / Over-Uniform Sequence (Suspiciously Flat!)"

    chi2_val = float(np.sum((counts - expected) ** 2 / expected))
    nu = n_bins - 1
    # Q(chi^2 | nu) = gammaincc(nu / 2, chi^2 / 2)
    q_prob = float(gammaincc(0.5 * nu, 0.5 * chi2_val))

    if q_prob < 0.01 or q_prob > 0.99:
        verdict = "REJECT (<1% or >99%)"
        bar_color = "#EF4444"
    elif q_prob < 0.05 or q_prob > 0.95:
        verdict = "SUSPECT (1–5% or 95–99%)"
        bar_color = "#F59E0B"
    elif q_prob < 0.10 or q_prob > 0.90:
        verdict = "ALMOST SUSPECT (5–10% or 90–95%)"
        bar_color = "#EAB308"
    else:
        verdict = "PASS (10%–90%)"
        bar_color = "#10B981"

    ax.set_facecolor("#F8FAFC")
    bin_Idx = np.arange(1, n_bins + 1)
    ax.bar(bin_Idx, counts, color=bar_color, alpha=0.82, edgecolor="#0F172A", linewidth=0.7, label=f"Observed Counts $N_i$")
    ax.axhline(expected, color="#0F172A", lw=1.8, ls="--", label=f"Expected $n_i = P/N = {expected:.1f}$")
    ax.axhspan(
        max(0.0, expected - sigma_bin),
        expected + sigma_bin,
        color="#64748B",
        alpha=0.18,
        label=f"$\\pm 1\\sigma$ Band ($\\pm\\sqrt{{n_i}} = \\pm {sigma_bin:.1f}$)",
    )

    ax.set_xlabel(f"Bin Index $i \\in \\{{1, \\dots, {n_bins}\\}}$ ($\\nu = N - 1 = {nu}$ d.o.f.)", fontsize=10)
    ax.set_ylabel("Bin Count $N_i$", fontsize=10)
    ax.set_title(
        f"{label_name}\n$\\chi^2 = {chi2_val:.2f}$ (Target $\\nu \\pm \\sqrt{{2\\nu}} = {nu} \\pm {np.sqrt(2*nu):.1f}$) · "
        f"$Q(\\chi^2|\\nu) = {100.0*q_prob:.2f}\\%$ → {verdict}",
        fontsize=9.8,
        fontweight="bold",
    )
    ax.set_ylim(0, max(float(np.max(counts)) * 1.22, expected + 3.0 * sigma_bin))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(True, axis="y", alpha=0.3)

    return (
        f"**Pearson $\\chi^2$ Hypothesis Test ($P = {n_samples}$ draws into $N = {n_bins}$ bins, $\\nu = {nu}$):**\n"
        f"- **Statistic:** $\\chi^2 = \\sum_{{i=1}}^{{{n_bins}}} (N_i - {expected:.1f})^2 / {expected:.1f} = {chi2_val:.2f}$ "
        f"(Theoretical $\\chi^2_\\nu$ mean $= {nu}$, standard deviation $\\sqrt{{2\\nu}} = {np.sqrt(2*nu):.2f}$).\n"
        f"- **Tail Probability:** $Q(\\chi^2 | \\nu) = {q_prob:.4f}$ ({100.0*q_prob:.2f}%) → **{verdict}**. "
        f"Note that $Q > 99\\%$ ($\\chi^2 \\ll \\nu$) is rejected just like $Q < 1\\%$ ($\\chi^2 \\gg \\nu$) because true random numbers *must* fluctuate by $\\sim \\sqrt{{n_i}}$!"
    )
