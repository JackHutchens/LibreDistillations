"""Interactive concept visualizations for Lesson 01: Core Foundations and Governing Equations."""

from __future__ import annotations
from typing import Any, Mapping
import numpy as np
import matplotlib.pyplot as plt
from engine.plot_style import COLORS


def _val(ctrl: Any, default: Any = 0.0) -> Any:
    if ctrl is None:
        return default
    return getattr(ctrl, "value", ctrl)


def render_sample_visualizer(controls: Mapping[str, Any], ax: plt.Axes) -> str:
    """Render interactive simulation or response curve across slider controls."""
    alpha = float(_val(controls.get("parameter_alpha"), 1.0))

    x = np.linspace(0.0, 10.0, 300)
    y = np.exp(-alpha * x) * np.cos(2.0 * np.pi * x)

    ax.plot(x, y, color=COLORS["primary"], lw=2.2, label=rf"Response $f(x; \alpha={alpha:.2f})$")
    ax.axhline(0.0, color=COLORS["muted"], ls="--", lw=0.9)
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Independent Variable $x$")
    ax.set_ylabel("Response $f(x)$")
    ax.set_title(rf"Dynamic System Relaxation ($\alpha = {alpha:.2f}$)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    return (
        f"**System State:** Decay rate $\\alpha = {alpha:.2f}$. "
        f"Half-life $t_{{1/2}} = {np.log(2)/max(alpha, 1e-6):.2f}$ units."
    )
