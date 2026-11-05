"""Unified visual styling and plotting utilities for interactive worksheets."""

from __future__ import annotations
from typing import Any, Mapping, Sequence
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

COLORS: dict[str, str] = {
    "primary": "#2B5C8F",
    "secondary": "#388E7F",
    "tertiary": "#4A7BB7",
    "quaternary": "#8E6BB0",
    "good": "#3B9A6E",
    "accent": "#D95F02",
    "highlight": "#E6AB02",
    "muted": "#8C96A6",
    "minor_tick": "#D8DEE9",
    "text": "#1E293B",
    "success": "#2E7D32",
    "warning": "#E65100",
    "danger": "#C62828",
    "node_default": "#4A7BB7",
    "edge_default": "#7F8C9A",
    "community_a": "#3B75AF",
    "community_b": "#E07A35",
    "community_c": "#469B76",
    "community_d": "#9B59B6",
    # Physics / Materials Science semantic aliases
    "solid": "#2B5C8F",
    "liquid": "#388E7F",
    "gas": "#E07A35",
    "potential": "#2B5C8F",
    "kinetic": "#D95F02",
    "total": "#1B1F24",
    "spin_up": "#2B5C8F",
    "spin_down": "#E07A35",
    "grid": "#E5E9F0",
    "bg_panel": "#F8FAFC",
}

COMMUNITY_PALETTE: list[str] = [
    COLORS["community_a"],
    COLORS["community_b"],
    COLORS["community_c"],
    COLORS["community_d"],
    "#D45079",
    "#5C928F",
]

GRAPH_STYLE: dict[str, Any] = {
    "node_size": 520,
    "edge_width": 1.8,
    "font_size": 10,
    "font_weight": "bold",
    "group_fill_alpha": 0.55,
    "node_edge_color": "#2B5C8F",
    "node_edge_width": 1.8,
    "group_node_edge_width": 2.0,
    "group_legend_marker_size": 10,
    "group_legend_anchor": (0.5, 1.12),
    "group_legend_size": 9.0,
    "annotation_font_size": 9.5,
    "edge_value_color": "#1E293B",
    "edge_value_gap": 4.0,
    "value_annotation_gap": 6.0,
    "value_annotation_size": 9.5,
    "label_font_size": 10.5,
}


def apply_default_style() -> None:
    """Apply clean, readable Matplotlib defaults suitable for Jupyter notebooks."""
    mpl.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "axes.titlesize": 11.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 10.0,
        "axes.labelcolor": "#1E293B",
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "xtick.color": "#334155",
        "ytick.color": "#334155",
        "legend.fontsize": 9.0,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#E2E8F0",
        "figure.dpi": 110,
        "savefig.dpi": 120,
        "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI", "sans-serif"],
        "mathtext.fontset": "dejavusans",
    })


def draw_graph(
    G: Any,
    ax_or_pos: Any = None,
    pos: Mapping[Any, Sequence[float]] | None = None,
    ax: plt.Axes | None = None,
    *,
    node_color: Any = None,
    node_size: Any = None,
    node_labels: Mapping[Any, str] | bool | None = True,
    edge_color: Any = None,
    edge_width: Any = None,
    edge_style: str = "solid",
    edge_styles: Mapping[tuple[Any, Any], str] | None = None,
    edge_labels: Mapping[tuple[Any, Any], Any] | None = None,
    edge_values: Mapping[tuple[Any, Any], Any] | None = None,
    edge_values_upright: bool = True,
    edge_values_stacked_fractions: bool = False,
    highlight_nodes: Sequence[Any] | None = None,
    highlight_edges: Sequence[tuple[Any, Any]] | None = None,
    communities: Sequence[Sequence[Any]] | None = None,
    node_groups: Mapping[Any, Any] | None = None,
    group_colors: Mapping[Any, str] | None = None,
    group_legend: bool = False,
    frame_nodes: bool = False,
    title: str | None = None,
    **_extra_kwargs: Any,
) -> plt.Axes:
    """Render a NetworkX graph with consistent pedagogical styling."""
    import networkx as nx

    # Handle both `draw_graph(G, ax, pos=...)` and `draw_graph(G, pos, ax=...)`
    if isinstance(ax_or_pos, plt.Axes):
        ax = ax_or_pos
    elif isinstance(ax_or_pos, Mapping) and pos is None:
        pos = ax_or_pos

    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 4.2))
    if pos is None:
        pos = nx.spring_layout(G, seed=42)

    nodes = list(G.nodes())
    edges = list(G.edges())

    # Resolve node colors
    if node_color is None:
        if node_groups and group_colors:
            resolved_node_color = [
                group_colors.get(node_groups.get(u), "white") for u in nodes
            ]
        elif communities:
            cmap_dict: dict[Any, str] = {}
            for idx, comm in enumerate(communities):
                col = COMMUNITY_PALETTE[idx % len(COMMUNITY_PALETTE)]
                for u in comm:
                    cmap_dict[u] = col
            resolved_node_color = [cmap_dict.get(u, "white") for u in nodes]
        else:
            resolved_node_color = ["white" for _ in nodes]
    elif isinstance(node_color, Mapping):
        resolved_node_color = [node_color.get(u, "white") for u in nodes]
    elif isinstance(node_color, str):
        resolved_node_color = [node_color for _ in nodes]
    else:
        resolved_node_color = list(node_color)

    if highlight_nodes:
        hset = set(highlight_nodes)
        resolved_node_color = [
            COLORS["accent"] if u in hset else c
            for u, c in zip(nodes, resolved_node_color)
        ]

    # Resolve node sizes
    if node_size is None:
        resolved_node_size = [GRAPH_STYLE["node_size"] for _ in nodes]
    elif isinstance(node_size, Mapping):
        resolved_node_size = [node_size.get(u, GRAPH_STYLE["node_size"]) for u in nodes]
    elif isinstance(node_size, (int, float)):
        resolved_node_size = [float(node_size) for _ in nodes]
    else:
        resolved_node_size = list(node_size)

    # Resolve edge colors and widths
    h_edges_norm = set()
    if highlight_edges:
        for u, v in highlight_edges:
            h_edges_norm.add((u, v))
            h_edges_norm.add((v, u))

    if edge_color is None:
        resolved_edge_color = [
            COLORS["accent"] if (u, v) in h_edges_norm else COLORS["primary"]
            for u, v in edges
        ]
    elif isinstance(edge_color, Mapping):
        resolved_edge_color = [
            edge_color.get((u, v), edge_color.get((v, u), COLORS["primary"]))
            for u, v in edges
        ]
    elif isinstance(edge_color, str):
        resolved_edge_color = [edge_color for _ in edges]
    else:
        resolved_edge_color = list(edge_color)

    if edge_width is None:
        resolved_edge_width = [
            2.8 if (u, v) in h_edges_norm else GRAPH_STYLE["edge_width"]
            for u, v in edges
        ]
    elif isinstance(edge_width, Mapping):
        resolved_edge_width = [
            edge_width.get((u, v), edge_width.get((v, u), GRAPH_STYLE["edge_width"]))
            for u, v in edges
        ]
    elif isinstance(edge_width, (int, float)):
        resolved_edge_width = [float(edge_width) for _ in edges]
    else:
        resolved_edge_width = list(edge_width)

    # Draw edges (supporting per-edge `edge_styles` dict if provided)
    if edges:
        if edge_styles:
            style_list = [
                edge_styles.get((u, v), edge_styles.get((v, u), edge_style))
                for u, v in edges
            ]
        else:
            style_list = edge_style
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=edges,
            ax=ax,
            edge_color=resolved_edge_color,
            width=resolved_edge_width,
            style=style_list,
            alpha=0.9,
        )

    # Draw nodes
    if nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodes,
            ax=ax,
            node_color=resolved_node_color,
            node_size=resolved_node_size,
            edgecolors=GRAPH_STYLE["node_edge_color"],
            linewidths=GRAPH_STYLE["node_edge_width"],
        )

    # Draw node labels
    if node_labels:
        labels_map = (
            {u: str(u) for u in nodes}
            if node_labels is True
            else {u: str(node_labels.get(u, u)) for u in nodes}
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels_map,
            ax=ax,
            font_size=GRAPH_STYLE["font_size"],
            font_weight=GRAPH_STYLE["font_weight"],
            font_color=COLORS["text"],
        )

    # Draw edge labels / edge_values
    merged_elabels = {}
    if edge_labels:
        merged_elabels.update(edge_labels)
    if edge_values:
        merged_elabels.update(edge_values)
    if merged_elabels:
        normalized_elabels = {}
        edge_set = set(edges)
        for (u, v), lbl in merged_elabels.items():
            if (u, v) in edge_set:
                normalized_elabels[(u, v)] = str(lbl)
            elif (v, u) in edge_set:
                normalized_elabels[(v, u)] = str(lbl)
            else:
                normalized_elabels[(u, v)] = str(lbl)
        nx.draw_networkx_edge_labels(
            G,
            pos,
            edge_labels=normalized_elabels,
            ax=ax,
            font_size=8.5,
            font_color=GRAPH_STYLE["edge_value_color"],
            rotate=not edge_values_upright,
            bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.85},
        )

    if frame_nodes and pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        if xs and ys:
            dx = max(0.4, (max(xs) - min(xs)) * 0.18)
            dy = max(0.4, (max(ys) - min(ys)) * 0.18)
            ax.set_xlim(min(xs) - dx, max(xs) + dx)
            ax.set_ylim(min(ys) - dy, max(ys) + dy)

    if title:
        ax.set_title(title, fontsize=11, pad=8)
    ax.set_axis_off()
    return ax
