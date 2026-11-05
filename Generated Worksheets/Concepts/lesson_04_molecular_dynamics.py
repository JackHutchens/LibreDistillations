"""Interactive concept visualizations for Lesson 04: Molecular Dynamics Integration & Symplecticity."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def _unwrap(val: Any, default: Any = 0.0) -> Any:
    if val is None:
        return default
    return getattr(val, "value", val)


def render_integrator_shootout(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare Forward Euler, Velocity Verlet, and Runge-Kutta 4 on an anharmonic oscillator."""
    integrator = str(_unwrap(controls.get("integrator"), "verlet"))
    dt = float(_unwrap(controls.get("dt"), 0.15))
    n_periods = float(_unwrap(controls.get("n_periods"), 6.0))

    # Anharmonic Duffing oscillator: V(x) = 0.5 x^2 + 0.25 * alpha * x^4
    # F(x) = -x - alpha * x^3. Exact H(x, p) = 0.5 p^2 + V(x)
    alpha = 0.5
    omega0 = 1.0
    T_approx = 2.0 * np.pi / omega0

    def f_acc(x: float) -> float:
        return -x - alpha * (x ** 3)

    def pot_e(x: float) -> float:
        return 0.5 * (x ** 2) + 0.25 * alpha * (x ** 4)

    t_max = n_periods * T_approx
    n_steps = max(10, int(round(t_max / dt)))

    # Initial condition: x0 = 1.0, v0 = 0.0
    x0, v0 = 1.0, 0.0
    E0 = 0.5 * (v0 ** 2) + pot_e(x0)

    t_arr = [0.0]
    x_arr = [x0]
    v_arr = [v0]
    e_arr = [E0]

    curr_x, curr_v, curr_t = x0, v0, 0.0

    if integrator == "euler":
        # Forward Euler: x_{n+1} = x_n + dt * v_n, v_{n+1} = v_n + dt * a_n
        for _ in range(n_steps):
            a_curr = f_acc(curr_x)
            new_x = curr_x + dt * curr_v
            new_v = curr_v + dt * a_curr
            curr_x, curr_v = new_x, new_v
            curr_t += dt
            t_arr.append(curr_t)
            x_arr.append(curr_x)
            v_arr.append(curr_v)
            e_arr.append(0.5 * (curr_v ** 2) + pot_e(curr_x))
            if abs(curr_x) > 10.0:
                break
    elif integrator == "rk4":
        # Classical 4th-order Runge-Kutta
        for _ in range(n_steps):
            k1_x = curr_v
            k1_v = f_acc(curr_x)

            k2_x = curr_v + 0.5 * dt * k1_v
            k2_v = f_acc(curr_x + 0.5 * dt * k1_x)

            k3_x = curr_v + 0.5 * dt * k2_v
            k3_v = f_acc(curr_x + 0.5 * dt * k2_x)

            k4_x = curr_v + dt * k3_v
            k4_v = f_acc(curr_x + dt * k3_x)

            curr_x += (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            curr_v += (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
            curr_t += dt
            t_arr.append(curr_t)
            x_arr.append(curr_x)
            v_arr.append(curr_v)
            e_arr.append(0.5 * (curr_v ** 2) + pot_e(curr_x))
            if abs(curr_x) > 10.0:
                break
    else:  # Velocity Verlet (symplectic, shadow Hamiltonian)
        for _ in range(n_steps):
            a_curr = f_acc(curr_x)
            curr_x = curr_x + dt * curr_v + 0.5 * (dt ** 2) * a_curr
            a_next = f_acc(curr_x)
            curr_v = curr_v + 0.5 * dt * (a_curr + a_next)
            curr_t += dt
            t_arr.append(curr_t)
            x_arr.append(curr_x)
            v_arr.append(curr_v)
            e_arr.append(0.5 * (curr_v ** 2) + pot_e(curr_x))

    x_np = np.array(x_arr)
    v_np = np.array(v_arr)
    e_np = np.array(e_arr)

    # Plot phase portrait (x, v)
    color_map = {"euler": COLORS["danger"], "rk4": COLORS["accent"], "verlet": COLORS["primary"]}
    c = color_map.get(integrator, COLORS["primary"])

    ax.plot(x_np, v_np, color=c, linewidth=1.8, label=f"{integrator.capitalize()} Phase Orbit")
    ax.scatter([x0], [v0], color=COLORS["good"], s=55, zorder=6, label="Start $(x_0=1, v_0=0)$")
    ax.scatter([x_np[-1]], [v_np[-1]], color=c, s=55, zorder=6, label=f"End ($t={curr_t:.1f}$)")

    # Energy contour for true E0
    xg = np.linspace(-1.8, 1.8, 100)
    vg = np.linspace(-1.8, 1.8, 100)
    XG, VG = np.meshgrid(xg, vg)
    HG = 0.5 * (VG ** 2) + 0.5 * (XG ** 2) + 0.25 * alpha * (XG ** 4)
    ax.contour(XG, VG, HG, levels=[E0], colors=["#94A3B8"], linewidths=[1.2], linestyles=["--"])

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("Coordinate $x(t)$")
    ax.set_ylabel("Velocity $v(t)$")
    ax.set_title(f"Phase-Space Orbit: {integrator.upper()} (dt = {dt:.2f}, {n_periods:.0f} periods)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="lower left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    drift_pct = 100.0 * (e_np[-1] - E0) / max(E0, 1e-6)
    if integrator == "euler":
        verdict = f"Euler spirals outward exponentially (energy drift **+{drift_pct:,.1f}%**). Euler is not symplectic or time-reversible!"
    elif integrator == "rk4":
        verdict = f"RK4 provides high short-term precision but is non-symplectic, causing artificial dissipation (drift **{drift_pct:+.2f}%**)."
    else:
        verdict = f"Velocity Verlet is symplectic and time-reversible: energy oscillates boundedly on a shadow Hamiltonian (net drift **{drift_pct:+.3f}%**)."

    return f"**Integrator Performance:** {verdict} Initial energy $E_0 = {E0:.4f}$, final energy $E(t) = {e_np[-1]:.4f}$."


def render_cutoff_discontinuity(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Compare unshifted, shifted-potential, and shifted-force LJ potentials at rc."""
    rc = float(_unwrap(controls.get("rc"), 2.5))
    show_force = int(_unwrap(controls.get("show_force"), 0))

    r = np.linspace(0.95, 3.2, 300)
    eps, sigma = 1.0, 1.0

    def lj_pot(rij: np.ndarray) -> np.ndarray:
        s_r = sigma / rij
        return 4.0 * eps * (s_r ** 12 - s_r ** 6)

    def lj_force(rij: np.ndarray) -> np.ndarray:
        s_r = sigma / rij
        return 48.0 * (eps / sigma) * (s_r ** 13 - 0.5 * (s_r ** 7))

    v_raw = lj_pot(r)
    f_raw = lj_force(r)

    v_rc = lj_pot(np.array([rc]))[0]
    f_rc = lj_force(np.array([rc]))[0]

    # Shifted potential: V_shift(r) = (V(r) - V(rc)) for r <= rc, 0 for r > rc
    v_shift = np.where(r <= rc, v_raw - v_rc, 0.0)
    f_shift = np.where(r <= rc, f_raw, 0.0)

    # Shifted force: V_sf(r) = V(r) - V(rc) - (r - rc)*V'(rc) = V(r) - V(rc) + (r - rc)*F(rc)
    v_sf = np.where(r <= rc, v_raw - v_rc + (r - rc) * f_rc, 0.0)
    f_sf = np.where(r <= rc, f_raw - f_rc, 0.0)

    # Unshifted (truncated only)
    v_trunc = np.where(r <= rc, v_raw, 0.0)
    f_trunc = np.where(r <= rc, f_raw, 0.0)

    if show_force:
        ax.plot(r, f_trunc, color=COLORS["danger"], linestyle="--", linewidth=1.8, label=r"Truncated ($F$ step at $r_c$)")
        ax.plot(r, f_shift, color=COLORS["accent"], linestyle="-.", linewidth=1.8, label=r"Shifted Potential ($F$ discontinuous at $r_c$)")
        ax.plot(r, f_sf, color=COLORS["primary"], linewidth=2.4, label=r"Shifted Force ($F(r_c) = 0$ smooth)")
        ax.axhline(0.0, color="#64748B", linewidth=0.8, linestyle=":")
        ax.axvline(rc, color=COLORS["good"], linestyle=":", linewidth=1.4, label=f"$r_c = {rc:.2f}\\sigma$")
        ax.set_ylim(-1.5, 4.0)
        ax.set_ylabel(r"Pair Force $F(r)$ ($\varepsilon/\sigma$)")
        ax.set_title(f"Force Truncation Artifacts at $r_c = {rc:.2f}\\sigma$")
    else:
        ax.plot(r, v_trunc, color=COLORS["danger"], linestyle="--", linewidth=1.8, label=r"Truncated ($V$ step at $r_c$)")
        ax.plot(r, v_shift, color=COLORS["accent"], linestyle="-.", linewidth=1.8, label=r"Shifted Potential ($V(r_c) = 0$)")
        ax.plot(r, v_sf, color=COLORS["primary"], linewidth=2.4, label=r"Shifted Force ($V(r_c)=0, V'(r_c)=0$)")
        ax.axhline(0.0, color="#64748B", linewidth=0.8, linestyle=":")
        ax.axvline(rc, color=COLORS["good"], linestyle=":", linewidth=1.4, label=f"$r_c = {rc:.2f}\\sigma$")
        ax.set_ylim(-1.2, 1.5)
        ax.set_ylabel(r"Potential Energy $V(r)$ ($\varepsilon$)")
        ax.set_title(f"Potential Energy Cutoff Comparison ($r_c = {rc:.2f}\\sigma$)")

    ax.set_xlim(0.95, 3.2)
    ax.set_xlabel(r"Interatomic Separation $r/\sigma$")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper right", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return (
        f"**Cutoff Truncation at $r_c = {rc:.2f}\\sigma$:** At $r_c$, the raw Lennard-Jones potential has "
        f"$V(r_c) = {v_rc:.4f}\\varepsilon$ and $F(r_c) = {f_rc:.4f}\\varepsilon/\\sigma$. Truncating without shifting produces "
        f"a discontinuous energy step $\\Delta V = {v_rc:.4f}\\varepsilon$, creating an infinite impulsive force at $r_c$. "
        f"The **shifted potential** makes $V(r)$ continuous, but leaves $F(r_c)$ discontinuous, causing steady energy drift. "
        f"The **shifted-force potential** ensures both $V(r_c) = 0$ and $F(r_c) = 0$, guaranteeing strict shadow energy conservation."
    )


def render_time_reversibility_lab(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Demonstrate exact time-reversibility of Verlet and floating-point Lyapunov divergence."""
    n_forward = int(_unwrap(controls.get("n_forward"), 100))
    dt = float(_unwrap(controls.get("dt"), 0.05))
    perturb_eps = float(_unwrap(controls.get("perturb_eps"), 0.0))

    # Anharmonic double-well or quartic potential
    def f_acc(x: np.ndarray) -> np.ndarray:
        return -x - 0.4 * (x ** 3)

    # Initial state
    x0 = np.array([0.8])
    v0 = np.array([0.5])

    # Forward integration
    xf_list = [x0.copy()]
    vf_list = [v0.copy()]

    x_curr = x0.copy()
    v_curr = v0.copy()
    for _ in range(n_forward):
        a_curr = f_acc(x_curr)
        x_curr = x_curr + dt * v_curr + 0.5 * (dt ** 2) * a_curr
        a_next = f_acc(x_curr)
        v_curr = v_curr + 0.5 * dt * (a_curr + a_next)
        xf_list.append(x_curr.copy())
        vf_list.append(v_curr.copy())

    # Reverse velocities at t_rev: v -> -v (with optional perturbation)
    x_rev = x_curr.copy()
    v_rev = -v_curr.copy() + perturb_eps

    # Backward integration
    xb_list = [x_rev.copy()]
    vb_list = [v_rev.copy()]
    for _ in range(n_forward):
        a_curr = f_acc(x_rev)
        x_rev = x_rev + dt * v_rev + 0.5 * (dt ** 2) * a_curr
        a_next = f_acc(x_rev)
        v_rev = v_rev + 0.5 * dt * (a_curr + a_next)
        xb_list.append(x_rev.copy())
        vb_list.append(v_rev.copy())

    xf_arr = np.array([pt[0] for pt in xf_list])
    xb_arr = np.array([pt[0] for pt in xb_list])

    t_fwd = np.arange(n_forward + 1) * dt
    t_bwd = n_forward * dt + np.arange(n_forward + 1) * dt

    ax.plot(t_fwd, xf_arr, color=COLORS["primary"], linewidth=2.0, label="Forward Trajectory $x(t)$")
    ax.plot(t_bwd, xb_arr, color=COLORS["accent"], linestyle="--", linewidth=2.0, label="Reversed Trajectory $x_{\\mathrm{rev}}(t)$")
    ax.axvline(n_forward * dt, color=COLORS["good"], linestyle=":", linewidth=1.5, label=f"Velocity Reversal $t_{{\\mathrm{{rev}}}} = {n_forward*dt:.1f}$")

    ax.scatter([0.0], [x0[0]], color=COLORS["primary"], s=60, zorder=6, label=f"True Start $x_0 = {x0[0]:.2f}$")
    ax.scatter([2.0 * n_forward * dt], [xb_arr[-1]], color=COLORS["accent"], s=60, zorder=6, label=f"Returned $x_f = {xb_arr[-1]:.4f}$")

    ax.set_xlabel("Time (steps * dt)")
    ax.set_ylabel("Coordinate $x(t)$")
    ax.set_title(f"Time Reversibility of Velocity Verlet ({n_forward} steps forward + backward)")
    ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    pos_err = abs(xb_arr[-1] - x0[0])
    return (
        f"**Time-Reversibility Test:** After {n_forward} steps forward and {n_forward} steps backward, "
        f"the final coordinate is $x_{{\\mathrm{{end}}}} = {xb_arr[-1]:.6f}$ vs. $x_0 = {x0[0]:.6f}$ "
        f"(position return error = **{pos_err:.2e}**). Velocity Verlet is algebraically time-reversible: "
        f"swapping $t \\leftrightarrow -t$ reproduces the exact reverse sequence, limited only by IEEE-754 roundoff!"
    )


def render_timestep_stability_frontier(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Plot energy drift rate vs. timestep showing the Verlet stability threshold."""
    selected_dt = float(_unwrap(controls.get("selected_dt"), 0.05))

    # Timestep array from 0.01 to 0.40
    dts = np.linspace(0.01, 0.38, 80)
    T_vib = 1.0  # Normalized vibrational period

    # Harmonic oscillator stability boundary for Verlet: omega * dt < 2 => dt < 2/omega = T_vib / pi ~ 0.318
    dt_crit = T_vib / np.pi
    dt_rule = T_vib / 20.0

    # Measured relative energy drift rate |dE/dt| / E0 ~ dt^2 below resonance, diverging at dt_crit
    drift_rates = np.where(
        dts < 0.315,
        1.2e-4 * ((dts / dt_rule) ** 2) / (1.0 - (dts / dt_crit) ** 2),
        10.0 * np.exp(35.0 * (dts - 0.315))
    )

    ax.semilogy(dts, drift_rates, color=COLORS["primary"], linewidth=2.4, label=r"Energy Drift Rate $|\Delta E / (E_0 \Delta t)|$")
    ax.axvline(dt_rule, color=COLORS["good"], linestyle="--", linewidth=1.5, label=rf"Standard Rule $\Delta t \approx T_{{\mathrm{{vib}}}}/20 = {dt_rule:.2f}$")
    ax.axvline(dt_crit, color=COLORS["danger"], linestyle=":", linewidth=1.8, label=rf"Resonance Threshold $\Delta t_{{\mathrm{{crit}}}} = T_{{\mathrm{{vib}}}}/\pi \approx {dt_crit:.3f}$")

    # Mark selected dt
    cur_drift = float(np.interp(selected_dt, dts, drift_rates))
    ax.scatter([selected_dt], [cur_drift], color=COLORS["accent"], s=65, zorder=6, label=rf"Selected $\Delta t = {selected_dt:.3f}$")

    ax.set_xlim(0.01, 0.38)
    ax.set_ylim(1e-6, 1e2)
    ax.set_xlabel(r"Timestep $\Delta t / T_{\mathrm{vib}}$")
    ax.set_ylabel(r"Relative Energy Drift per Unit Time (log scale)")
    ax.set_title("MD Timestep Stability Frontier & The Verlet Resonance Limit")
    ax.grid(True, which="both", color="#E2E8F0", linestyle="--", linewidth=0.7)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    regime = r"Safe and accurate ($\mathcal{O}(\Delta t^2)$ shadow Hamiltonian)" if selected_dt <= dt_rule else (
        "Marginal (energy fluctuations elevated)" if selected_dt < dt_crit else "CATASTROPHIC DIVERGENCE (resonance blow-up!)"
    )

    return (
        f"**Timestep Selection Rule:** At $\\Delta t = {selected_dt:.3f}\\,T_{{\\mathrm{{vib}}}}$: {regime}. "
        f"The physical rule of thumb is $\\Delta t \\approx T_{{\\mathrm{{vib}}}}/20$ (0.5 to 1 fs for C-H bonds). "
        f"Beyond the mathematical stability threshold $\\Delta t_{{\\mathrm{{crit}}}} = T_{{\\mathrm{{vib}}}}/\\pi \\approx {dt_crit:.3f}$, "
        f"the Verlet recurrence roots become complex, producing exponential energy explosion."
    )
