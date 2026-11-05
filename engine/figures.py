"""Static and shared figure rendering for worksheet problems and sections."""

from __future__ import annotations
import copy
from typing import Any, Callable, Mapping
import matplotlib.pyplot as plt
import numpy as np
from .plot_style import COLORS, draw_graph


def resolve_figure_spec(
    fig_spec: Mapping[str, Any],
    shared_figures: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge a figure reference (`ref` + `overrides`) with its shared_figures base."""
    if "ref" in fig_spec:
        ref_id = fig_spec["ref"]
        if not shared_figures or ref_id not in shared_figures:
            raise KeyError(f"Unknown shared figure reference: {ref_id!r}")
        merged = copy.deepcopy(dict(shared_figures[ref_id]))
        overrides = fig_spec.get("overrides") or {}
        for k, v in overrides.items():
            merged[k] = copy.deepcopy(v)
        for k in ("caption", "title", "figsize", "highlight_nodes", "highlight_edges", "edge_values", "node_groups"):
            if k in fig_spec and k not in overrides:
                merged[k] = copy.deepcopy(fig_spec[k])
        return merged
    return copy.deepcopy(dict(fig_spec))


def render_static_figure(
    fig_spec: Mapping[str, Any],
    shared_figures: Mapping[str, Any] | None = None,
    concept_module: Any = None,
) -> tuple[plt.Figure, str | None]:
    """Render a static problem or section figure and return (Figure, caption)."""
    spec = resolve_figure_spec(fig_spec, shared_figures)
    kind = spec.get("kind", "graph_static")
    figsize = tuple(spec.get("figsize", [5.8, 3.6]))
    caption = spec.get("caption")

    fig, ax = plt.subplots(figsize=figsize)

    if kind in ("graph_static", "graph"):
        import networkx as nx

        G = nx.Graph()
        pos: dict[Any, Any] = dict(spec.get("positions") or {})
        for u in spec.get("nodes", []):
            if isinstance(u, Mapping):
                nid = u["id"]
                G.add_node(nid)
                if "pos" in u:
                    pos[nid] = tuple(u["pos"])
            else:
                G.add_node(u)
        for edge in spec.get("edges", []):
            G.add_edge(edge[0], edge[1])

        edge_labels_raw = spec.get("edge_labels") or spec.get("edge_values")
        edge_labels = None
        if isinstance(edge_labels_raw, Mapping):
            edge_labels = {}
            for k, v in edge_labels_raw.items():
                if isinstance(k, str) and "-" in k:
                    parts = k.split("-", 1)
                    edge_labels[(parts[0].strip(), parts[1].strip())] = v
                elif isinstance(k, str) and "–" in k:
                    parts = k.split("–", 1)
                    edge_labels[(parts[0].strip(), parts[1].strip())] = v
                elif isinstance(k, (tuple, list)) and len(k) == 2:
                    edge_labels[(k[0], k[1])] = v
        elif isinstance(edge_labels_raw, list):
            edge_labels = {(item[0], item[1]): item[2] for item in edge_labels_raw}

        draw_graph(
            G,
            pos=pos if pos else None,
            ax=ax,
            node_size=spec.get("node_size"),
            node_labels=spec.get("node_labels", True),
            edge_labels=edge_labels,
            highlight_nodes=spec.get("highlight_nodes"),
            highlight_edges=spec.get("highlight_edges"),
            communities=spec.get("communities"),
            node_groups=spec.get("node_groups"),
            group_colors=spec.get("group_colors"),
            frame_nodes=spec.get("frame_nodes", True),
            title=spec.get("title"),
        )

    elif kind == "concept_static":
        func_name = spec["render_function"]
        params = spec.get("params", {})
        if concept_module is None or not hasattr(concept_module, func_name):
            raise AttributeError(f"Concept module does not define {func_name!r}")
        fn: Callable = getattr(concept_module, func_name)
        from .worksheet import wrap_controls_dict
        ret_caption = fn(wrap_controls_dict(dict(params)), ax)
        if not caption and isinstance(ret_caption, str):
            caption = ret_caption
        if spec.get("title"):
            ax.set_title(spec["title"])

    elif kind == "plot_2d":
        for idx, curve in enumerate(spec.get("curves", [])):
            x = np.asarray(curve["x"], dtype=float)
            y = np.asarray(curve["y"], dtype=float)
            color = curve.get("color", list(COLORS.values())[idx % 6])
            style = curve.get("linestyle", "-")
            lw = curve.get("linewidth", 2.0)
            marker = curve.get("marker", None)
            label = curve.get("label", None)
            ax.plot(x, y, linestyle=style, linewidth=lw, color=color, marker=marker, label=label)
            if curve.get("fill_under"):
                ax.fill_between(x, 0, y, color=color, alpha=0.15)
        for vline in spec.get("vlines", []):
            ax.axvline(
                vline["x"],
                color=vline.get("color", COLORS["muted"]),
                linestyle=vline.get("linestyle", "--"),
                linewidth=1.3,
                label=vline.get("label"),
            )
        for hline in spec.get("hlines", []):
            ax.axhline(
                hline["y"],
                color=hline.get("color", COLORS["muted"]),
                linestyle=hline.get("linestyle", "--"),
                linewidth=1.3,
                label=hline.get("label"),
            )
        if spec.get("xlabel"):
            ax.set_xlabel(spec["xlabel"])
        if spec.get("ylabel"):
            ax.set_ylabel(spec["ylabel"])
        if spec.get("xlim"):
            ax.set_xlim(spec["xlim"])
        if spec.get("ylim"):
            ax.set_ylim(spec["ylim"])
        if spec.get("xscale"):
            ax.set_xscale(spec["xscale"])
        if spec.get("yscale"):
            ax.set_yscale(spec["yscale"])
        if spec.get("title"):
            ax.set_title(spec["title"])
        if spec.get("grid", True):
            ax.grid(True, color="#E2E8F0", linestyle="--", linewidth=0.7, alpha=0.8)
        if any(c.get("label") for c in spec.get("curves", [])) or any(
            v.get("label") for v in spec.get("vlines", [])
        ):
            ax.legend(loc=spec.get("legend_loc", "best"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    elif kind == "bar_chart":
        categories = spec.get("categories", [])
        values = spec.get("values", [])
        colors = spec.get("colors", COLORS["primary"])
        ax.bar(categories, values, color=colors, edgecolor="white", width=0.6)
        if spec.get("xlabel"):
            ax.set_xlabel(spec["xlabel"])
        if spec.get("ylabel"):
            ax.set_ylabel(spec["ylabel"])
        if spec.get("title"):
            ax.set_title(spec["title"])
        ax.grid(True, axis="y", color="#E2E8F0", linestyle="--", linewidth=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    elif kind == "lattice_2d":
        grid_data = np.asarray(spec["grid"], dtype=float)
        cmap = spec.get("cmap", "coolwarm")
        im = ax.imshow(
            grid_data,
            cmap=cmap,
            vmin=spec.get("vmin", -1.0),
            vmax=spec.get("vmax", 1.0),
            origin="lower",
        )
        if spec.get("colorbar", False):
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if spec.get("title"):
            ax.set_title(spec["title"])
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig, caption
