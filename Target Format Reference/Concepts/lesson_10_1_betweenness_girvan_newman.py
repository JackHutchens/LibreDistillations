"""Concept-cell render functions for Worksheet 10.1 — betweenness and Girvan-Newman.

Each function takes ``(controls, ax)``, draws into the axes and returns a Markdown
caption. The engine clears ``ax`` and re-runs on every control change, so these
are pure redraws.

DISCIPLINES THIS MODULE KEEPS

1. EXACT, DERIVED, NEVER TYPED. Every betweenness, flow, shortest-path count and
   modularity printed or drawn is computed HERE, at render time, from the graph
   the figure draws — by the same functions ``scripts/derive_l10_keys.py``
   cross-checks against three independent implementations. Nothing is a literal.
   Values are exact ``Fraction``s, so "9/2" is a fraction and not a rounded 4.5.

2. THE BOOK'S OWN NUMBERS ARE THE RED CASE. ``_ek_graph`` reproduces E&K
   Figure 3.18(a) (READ FROM THE PDF AS AN IMAGE — the c3.txt OCR carries this
   figure as loose label tokens and no edges at all). If the walkthrough cells
   ever stop printing Figure 3.19's shortest-path counts (1,1,1,1,2,1,2,3,3,6)
   and Figure 3.20's flows (2,2,4,2 out of A), the graph has drifted.

3. EVERY CAPTION HOLDS AT EVERY CONTROL VALUE (AUTHORING rule 23e). Each cell
   below was read at every setting of its control, not just the default — the
   root selector runs over all seven roots, the cascade over all four steps.

4. RULE 23: a computed quantity is shown WITH ITS ARITHMETIC AND ITS INPUTS
   NAMED. The flow at a node is printed as "1 for itself + what came up from
   below", not as a bare number, because the split IS the lesson.

NOTATION, stated once and reused verbatim (rule 22b):
  * ``sigma(v)`` — the number of shortest paths from the current root to v.
  * ``flow`` on an edge — the amount of the root's traffic that edge carries,
    where the root sends ONE unit to every other node.
  * ``betweenness`` of an edge — the flows summed over every root, then halved.
  * ``Q`` — modularity: (fraction of edges inside communities) minus (the
    fraction a degree-matched random graph would put there).
"""
import itertools
import math
from collections import defaultdict, deque
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction

import networkx as nx

from engine.plot_style import COLORS, GRAPH_STYLE, draw_graph

# ---------------------------------------------------------------------------
# The two graphs
# ---------------------------------------------------------------------------
#
# ⚠ BOTH ARE MIRRORED IN THE YAML's `shared_figures` (F8 — two copies WILL
# drift). `scripts/check_10_figures.py` re-derives the mirror and BLOCKS on a
# mismatch, so the copies cannot silently diverge.

# Hari's `pics/girvan-new.pdf`, read edge-by-edge off the rendered PDF. Two
# natural communities, {A,B,C} and {D,E,F,G}, joined by the single bridge B-D.
_COMM_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"),
               ("D", "E"), ("D", "F"), ("D", "G"), ("E", "F"), ("F", "G")]

# Layout follows the SOURCE figure's geometry — triangle on the left, a square
# with one diagonal on the right — so a student who has seen the lecture notes
# recognizes the same picture. Positions are stretched from the source's tight
# spacing to open up the D-F diagonal, which is the only place two edge labels
# come close; `check_10_figures.py` measures the clearance of every one.
_COMM_POS = {
    "A": (0.00, 0.00), "B": (0.95, 1.25), "C": (1.90, 0.00),
    "D": (4.30, 1.25), "E": (6.30, 1.25), "G": (4.30, -0.35), "F": (6.30, -0.35),
}
_COMM_NODE_SIZE = 620
_COMM_SIDES = {"A": "left", "B": "left", "C": "left",
               "D": "right", "E": "right", "F": "right", "G": "right"}

# E&K Figure 3.18(a). ⚠ 16 EDGES — there is NO C-D edge; C sits directly above D
# and the C-A DIAGONAL reads as a C-D vertical at low zoom. Settled at 420 dpi
# and corroborated by the book's own Figure 3.18(b), which draws B-C as a
# same-layer non-tree edge and draws no C-D.
_EK_EDGES = [("F", "B"), ("F", "C"), ("F", "I"), ("B", "C"), ("B", "A"),
             ("C", "A"), ("A", "D"), ("A", "E"), ("D", "G"), ("D", "H"),
             ("I", "G"), ("I", "K"), ("G", "J"), ("J", "K"), ("E", "H"),
             ("H", "J")]
# Drawn as the BOOK draws it in Figure 3.18(b): layers stacked downward from A,
# so "one layer at a time" is literally what the eye follows.
_EK_POS = {
    "A": (3.00, 4.0),
    "B": (0.00, 3.0), "C": (2.00, 3.0), "D": (4.00, 3.0), "E": (6.00, 3.0),
    "F": (1.00, 2.0), "G": (3.00, 2.0), "H": (5.00, 2.0),
    "I": (1.80, 1.0), "J": (4.20, 1.0),
    "K": (3.00, 0.0),
}
_EK_NODE_SIZE = 520


def _comm_graph():
    G = nx.Graph()
    G.add_nodes_from(sorted(_COMM_POS))
    G.add_edges_from(_COMM_EDGES)
    return G


def _ek_graph():
    G = nx.Graph()
    G.add_nodes_from(sorted(_EK_POS))
    G.add_edges_from(_EK_EDGES)
    return G


# ---------------------------------------------------------------------------
# The lecture notes' eight steps, as code
# ---------------------------------------------------------------------------

def _bfs(G, root):
    """Steps 1-2: layers, then the shortest-path count sigma built downward.

    ``sigma(v)`` is the sum of ``sigma`` over the nodes one layer above v that
    have an edge to v — the book's Figure 3.19 rule, and the reason K reads 6.
    """
    dist = {root: 0}
    sigma = {root: 1}
    parents = defaultdict(list)
    order = [root]
    q = deque([root])
    while q:
        u = q.popleft()
        for w in sorted(G[u]):
            if w not in dist:
                dist[w] = dist[u] + 1
                sigma[w] = 0
                order.append(w)
                q.append(w)
            if dist[w] == dist[u] + 1:
                sigma[w] += sigma[u]
                parents[w].append(u)
    return dist, sigma, parents, order


def _flows(G, root):
    """Steps 3-6: one unit of flow to every other node, pushed UP the layers.

    A node's total flow is 1 (for itself) plus everything arriving from below;
    it is split over the nodes above it IN PROPORTION TO THEIR sigma.
    """
    dist, sigma, parents, order = _bfs(G, root)
    edge_flow = {}
    node_flow = {v: Fraction(1) for v in order if v != root}
    for v in reversed(order):
        if v == root:
            continue
        denom = sum(sigma[p] for p in parents[v])
        for p in parents[v]:
            share = node_flow[v] * Fraction(sigma[p], denom)
            edge_flow[frozenset((p, v))] = share
            if p != root:
                node_flow[p] += share
    return dist, sigma, parents, order, edge_flow, node_flow


def _betweenness(G, roots=None):
    """Steps 7-8: sum the flows over the chosen roots, then divide by 2."""
    total = defaultdict(Fraction)
    for r in (sorted(G.nodes()) if roots is None else roots):
        _, _, _, _, ef, _ = _flows(G, r)
        for e, f in ef.items():
            total[e] += f
    return {e: f / 2 for e, f in total.items()}


def _modularity(G, parts):
    """Q under the worksheet's own normalization (ws1 Prob 8 / ws2 Eq. 1).

    The double sum runs over ORDERED pairs and INCLUDES i == j, so each internal
    edge is counted twice and the diagonal null term is present. That is the
    form the student reads, so it is the form computed.
    """
    m = G.number_of_edges()
    k = dict(G.degree())
    where = {v: i for i, part in enumerate(parts) for v in part}
    total = Fraction(0)
    for i in G.nodes():
        for j in G.nodes():
            if where[i] != where[j]:
                continue
            a = Fraction(1) if G.has_edge(i, j) else Fraction(0)
            total += a - Fraction(k[i] * k[j], 2 * m)
    return total / (2 * m)


def _frac(x):
    """Plain-text exact value — '9/2', never '4.5' and never mathtext."""
    f = Fraction(x)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def _pairs(e):
    return tuple(sorted(e))


# 2026-10-28 (teaching voice): a count written in words inside a caption, so that a caption
# can say "two shortest paths" without adding a numeral the 2026-10-22 caption did not print.
_NUM_WORD = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


# ⚠ MEASURED: `GROUP_COLOR_CYCLE[0]` IS `COLORS["accent"]`, the same orange the
# highlight uses. A cell that colors by group AND highlights something therefore
# paints its first group in the "look here" color — and with a SINGLE group it
# paints the entire graph in it, which reads as "everything is selected". So
# these cells pin group colors past index 0, and pass no groups at all when
# there is only one group (or when every group is a singleton, where a color per
# node carries no information).
_GROUP_PIN = [COLORS["tertiary"], COLORS["good"], COLORS["quaternary"], COLORS["highlight"]]


def _group_kwargs(parts):
    """`node_groups` + `group_colors` — or nothing, when coloring says nothing."""
    if not 2 <= len(parts) <= len(_GROUP_PIN) or all(len(p) == 1 for p in parts):
        return {}
    groups = {v: f"C{i + 1}" for i, p in enumerate(parts) for v in p}
    colors = {f"C{i + 1}": _GROUP_PIN[i] for i in range(len(parts))}
    return {"node_groups": groups, "group_colors": colors, "group_legend": False}


# ---------------------------------------------------------------------------
# Community painting (2026-10-22 fix pass: RT101-8, figures F7 and F8)
# ---------------------------------------------------------------------------
#
# ⚠ WHY THE CELLS PAINT THEIR OWN COMMUNITY FILLS. Three measured defects:
#   * At the pin (v0.13.0) a HIGHLIGHTED node is drawn over its group color
#     (`draw_graph`: "highlight still wins on top"), so the knob cell, which
#     highlighted every covered node, showed no community at all (RT101-8).
#   * The engine's group fill is translucent (alpha 0.55), so every edge shows
#     through to the node's centre and strikes its letter (figures F7: D read
#     as "Đ"). The fill below is the SAME tint drawn OPAQUE over the white node,
#     so the edges stop at the ring.
#   * A cover is not a partition, and the engine's groups cannot show a node in
#     two communities. A node in two is drawn with a SPLIT fill (left half one
#     community's tint, right half the other's), a node in none in light gray,
#     and the cover cells carry a legend (RT101-8; figures F8: caramel had
#     meant "in both", a singleton and an ordinary group on one sheet).
# `draw_graph` is called WITHOUT node_groups (plain white nodes, navy rings,
# its own labels), then these markers go on top at zorder 2.5: above the
# engine's nodes (2) and edges (1), under the letters (Text, zorder 3).

_OUT_FILL = COLORS["minor_tick"]          # light gray: in no community


def _tint(color):
    """The opaque color a translucent group fill shows over white."""
    from matplotlib.colors import to_rgb
    a = GRAPH_STYLE["group_fill_alpha"]
    return tuple(a * c + (1.0 - a) for c in to_rgb(color))


def _marker_kwargs(cols, size_points):
    """Line2D marker keywords for a node in `cols` communities (0, 1 or 2)."""
    kw = dict(marker="o", markersize=size_points, linestyle="none")
    if not cols:
        kw.update(markerfacecolor=_OUT_FILL, markeredgecolor=GRAPH_STYLE["node_edge_color"],
                  markeredgewidth=GRAPH_STYLE["node_edge_width"])
    elif len(cols) == 1:
        kw.update(markerfacecolor=_tint(cols[0]), markeredgecolor=cols[0],
                  markeredgewidth=GRAPH_STYLE["group_node_edge_width"])
    else:
        kw.update(fillstyle="left", markerfacecolor=_tint(cols[0]),
                  markerfacecoloralt=_tint(cols[1]),
                  markeredgecolor=GRAPH_STYLE["node_edge_color"],
                  markeredgewidth=GRAPH_STYLE["group_node_edge_width"])
    return kw


def _paint_nodes(ax, pos, node_size, paint):
    """Paint each node in `paint` ({node: [colors]}) over the plain node draw_graph made.

    `node_size` is a marker AREA in points squared (networkx's convention), so the
    diameter in points is its square root: the painted disk covers the engine's.
    """
    d = math.sqrt(node_size)
    for n, cols in paint.items():
        x, y = pos[n]
        ax.plot([x], [y], zorder=2.5, **_marker_kwargs(cols, d))


def _community_legend(ax, entries):
    """A legend for painted communities: [(label, cols), ...] in the order given."""
    import matplotlib.lines as mlines
    handles = [mlines.Line2D([], [], label=label,
                             **_marker_kwargs(cols, GRAPH_STYLE["group_legend_marker_size"]))
               for label, cols in entries]
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=GRAPH_STYLE["group_legend_anchor"],
                    ncol=min(len(handles), 4), frameon=False, handletextpad=0.3,
                    columnspacing=1.1, fontsize=GRAPH_STYLE["group_legend_size"])
    for txt in leg.get_texts():
        txt.set_color(COLORS["text"])


def _paint_partition(ax, pos, node_size, parts, colors=None):
    """Paint a partition, one tint per part (none for singletons-only or one part).

    `colors` pins a color to each part, so a group keeps its color from one control
    state to the next (figures F8: E, F and G changed color while their group did not).
    """
    if not 2 <= len(parts) <= len(_GROUP_PIN) or all(len(p) == 1 for p in parts):
        return
    colors = colors or _GROUP_PIN
    _paint_nodes(ax, pos, node_size,
                 {v: [colors[i]] for i, p in enumerate(parts) for v in p})


def _name_set(c):
    return "{" + ", ".join(c) + "}"


def _d3(x):
    """Three decimals, a tie rounded half up from the exact value: 1/16 -> '0.063'.
    Python's float format rounds an exact binary tie to even ('0.062'), which a
    student checking by hand reads as a slip."""
    f = Fraction(x)
    exact = Decimal(f.numerator) / Decimal(f.denominator)
    return str(exact.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))


def _num(x):
    """One math span: '$8/9 \\approx 0.889$', or '$1$' for a whole number (F12)."""
    if Fraction(x).denominator == 1:
        return f"${_frac(x)}$"
    return f"${_frac(x)} \\approx {_d3(x)}$"


def _fmt_q(q):
    """'$Q = 59/162 \\approx 0.364$', or '$Q = 0$' when the value is a whole number
    (figures F12: no '≈ 0.000' after an exact integer)."""
    if Fraction(q).denominator == 1:
        return f"$Q = {_frac(q)}$"
    return f"$Q = {_frac(q)} \\approx {_d3(q)}$"


# ---------------------------------------------------------------------------
# CELL 1 — why we expect communities at all
# ---------------------------------------------------------------------------

_EXTRA_BRIDGES = [("C", "G"), ("A", "E"), ("C", "E")]
# Where C–E's label sits, as a fraction of the way from C to E: between C and the
# point (t = 0.545) where C–E crosses the D–G line, and clear of the A–E line.
_CE_LABEL_T = 0.27


def _why_graph(extra):
    G = _comm_graph()
    G.add_edges_from(_EXTRA_BRIDGES[:extra])
    return G


def _edge_label_at(ax, pos, u, v, t, value, side):
    """Write `value` beside edge u–v at fraction `t` of the way from u to v.

    The engine's `draw_edge_value_labels` writes at the midpoint only. This keeps
    its look (font size, color, plain-text value, upright) and its push: the
    label's own measured box clears the shaft by `edge_value_gap` POINTS along the
    display-space normal, `side` (+1 or -1) choosing which side. In points, so the
    clearance does not move with dpi (F7).
    """
    scale = ax.figure.dpi / 72.0
    p0 = [c / scale for c in ax.transData.transform(pos[u])]
    p1 = [c / scale for c in ax.transData.transform(pos[v])]
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    nx_, ny_ = -dy / length * side, dx / length * side
    at = (pos[u][0] + t * (pos[v][0] - pos[u][0]), pos[u][1] + t * (pos[v][1] - pos[u][1]))
    txt = ax.annotate(_frac(value), xy=at, xytext=(0.0, 0.0), textcoords="offset points",
                      ha="center", va="center", rotation=0,
                      fontsize=GRAPH_STYLE["annotation_font_size"],
                      color=GRAPH_STYLE["edge_value_color"], annotation_clip=False)
    try:                                   # same fallback as the engine's _text_size_points
        renderer = ax.figure.canvas.get_renderer()
    except AttributeError:
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
    box = txt.get_window_extent(renderer)
    w, h = box.width / scale, box.height / scale
    push = GRAPH_STYLE["edge_value_gap"] + (w / 2.0) * abs(nx_) + (h / 2.0) * abs(ny_)
    txt.xyann = (push * nx_, push * ny_)
    return txt


def render_why_communities(controls, ax):
    extra = int(controls["bridges"].value) - 1
    G = _why_graph(extra)
    added = _EXTRA_BRIDGES[:extra]
    bridges = [("B", "D")] + list(added)

    bet = _betweenness(G)
    labels = {e: bet[frozenset(e)] for e in bridges}
    # ⚠ stacked_fractions is OFF on this cell ALONE — a LEGIBILITY choice Hari made
    # on the picture, NOT a clean-vs-dirty fix. Do not "restore" it for consistency.
    #
    # THE C–E LABEL CANNOT BE MADE CLEAN BY ANY FLAG. The engine pushes a label clear
    # of its OWN shaft only, and C–E's midpoint has TWO foreign edges crossing it
    # (A–E diagonal, D–G vertical). MEASURED in-container at bridges=4, ink against
    # ANY non-background pixel:
    #     upright + inline  (SHIPPED) : 40 of 408 ink px on the navy D–G line (10%)
    #     upright + stacked           : 31 of 153 ink px on the orange A–E line (20%)
    #     rotated                     : 10 of 466 ink px on orange A–E (2.1%)
    # Inline ships because the D–G line coincides with the SLASH stroke, leaving both
    # numerals readable, whereas stacked put the A–E line through the numeral "5".
    # Fewest pixels (rotated) is not the same as least damage.
    #
    # ⚠ TWO INSTRUMENTS WERE WRONG HERE, both docketed — read them before re-measuring:
    #   * draw_edge_value_labels reported clearance_points = +4.21 pt for a label a
    #     foreign edge cuts 5.85 pt through. Do NOT trust that number on the upright
    #     path. (The ordering story — frame_nodes moving the limits after labelling —
    #     was TESTED AND REFUTED: the limits do not move.)
    #   * the first ink check keyed on ORANGE rgb and was blind to every dark-navy
    #     edge, which is how "inline is clean" was briefly believed. Identify ink by
    #     "not background", never by the colour you happened to look at (F3).
    #
    # ⚠ 2026-10-22 (RT101-7, figures F4): C–E's label is NO LONGER drawn at the
    # edge's midpoint. The midpoint has the D–G and D–F lines crossing it, and the
    # label sat on D–G, about 48 px from C–E, where a student read it as D–G's
    # value. The engine places at the midpoint only, so this cell writes C–E's
    # value itself, at a point along C–E between C and the D–G line
    # (`_CE_LABEL_T`), pushed to the side of C–E away from A–E, in the engine's
    # font, color and points-based push. `text_fit_audit.py` checks it.
    engine_labels = {e: v for e, v in labels.items() if e != ("C", "E")}
    draw_graph(G, ax, pos=_COMM_POS, node_size=_COMM_NODE_SIZE,
               highlight_edges=bridges, frame_nodes=True, edge_values=engine_labels,
               edge_values_upright=True, edge_values_stacked_fractions=False)
    _paint_partition(ax, _COMM_POS, _COMM_NODE_SIZE,
                     [["A", "B", "C"], ["D", "E", "F", "G"]])
    if ("C", "E") in labels:
        _edge_label_at(ax, _COMM_POS, "C", "E", _CE_LABEL_T, labels[("C", "E")], side=-1)

    q = _modularity(G, [["A", "B", "C"], ["D", "E", "F", "G"]])
    top = max(bet.values())
    inside = sum(1 for u, v in G.edges() if _COMM_SIDES[u] == _COMM_SIDES[v])
    across = G.number_of_edges() - inside

    named = ", ".join(f"{u}–{v} carries {_frac(labels[(u, v)])}"
                      for u, v in bridges)
    head = (f"With {extra + 1} {'route' if extra == 0 else 'routes'} between the groups, "
            f"the network has {G.number_of_edges()} edges: {inside} inside a group and "
            f"{across} across. ")
    # 2026-10-28 teaching voice (W101-1, point 3): each state's caption is written for that
    # state; only `head`, the line that turns the state into numbers, is shared. Every number
    # is still computed (`named`, `moves`, `top`, the two Q values); the only literals are the
    # ones the 2026-10-22 captions carried (3, 4 and 12 at one route; 12 at the others).
    if extra == 0:
        return (
            head + f"We mark the crossing in orange and label it with its *betweenness*, the "
            f"shortest-path traffic it carries when each pair of nodes exchanges one "
            f"unit: {named}. Where does that number come from? Each of the "
            f"(3 nodes on the left) × (4 on the right) = 12 trips between the two sides "
            f"has to use B–D, since it is the only way across.\n\n"
            f"As for modularity, the two-group split scores {_fmt_q(q)}, a score for a "
            f"division that we define in the second half of this sheet (higher means "
            f"better-separated groups). What do you expect to happen to B–D's 12 when you "
            f"add a second route?")
    prev_bet = _betweenness(_why_graph(extra - 1))
    prev_q = _modularity(_why_graph(extra - 1), [["A", "B", "C"], ["D", "E", "F", "G"]])
    olds = [e for e in bridges[:-1]]
    moves = ", ".join(f"{u}–{v} from {_frac(prev_bet[frozenset((u, v))])} to "
                      f"{_frac(labels[(u, v)])}" for u, v in olds)
    new = "–".join(bridges[-1])
    if extra == 1:
        return (
            head + f"Now a second crossing, {new}, joins the two sides, and we label both "
            f"crossings with their betweenness: {named}. On this network the second route lowers the number "
            f"on the first ({moves}), and the largest number on any edge is now "
            f"{_frac(top)}, down from 12 with one route.\n\n"
            f"Modularity has fallen too; the split scored ${_frac(prev_q)}$ with one route, "
            f"and it now scores {_fmt_q(q)}, although neither group has lost an edge. "
            f"Thus a community boundary is about how few ways there are across it, as "
            f"well as how dense the groups are. Add another route and see if the pattern "
            f"holds.")
    if extra < len(_EXTRA_BRIDGES):
        return (
            head + f"Now {new} joins the two sides as well, and we label each crossing: "
            f"{named}. Again the older crossings drop ({moves}), and the "
            f"largest number on any edge is now {_frac(top)}, down from 12 with one "
            f"route.\n\n"
            f"Then the split's score drops again, from ${_frac(prev_q)}$ to {_fmt_q(q)}. "
            f"Notice that the two groups are as dense as they were; only the waist has "
            f"widened. Add the last route and check the pattern once more.")
    return (
        head + f"With {new} added, the trips between the two sides spread over all the "
        f"crossings: {named}. On this network each added route has lowered the numbers "
        f"on the crossings already there ({moves}), and the largest number on any edge "
        f"is now {_frac(top)}, down from 12 with one route.\n\n"
        f"As for the split, it now scores {_fmt_q(q)}, down from ${_frac(prev_q)}$ "
        f"before this route. Drag back to one route and compare both numbers with the ones you "
        f"started from.")


# ---------------------------------------------------------------------------
# CELL 2 — Step 1 and Step 2: layers, then the shortest-path counts
# ---------------------------------------------------------------------------

_LAYER_STEPS = ["layer 1 only", "through layer 2", "through layer 3",
                "the whole search"]
# Points from a node's centre to its sigma label — the node radius plus the
# engine's own clearance token. In POINTS, so it is INK and does not move with
# the canvas (F7).
_SIGMA_GAP = math.sqrt(_EK_NODE_SIZE / math.pi) + GRAPH_STYLE["value_annotation_gap"]


def _sigma_labels(ax, sigma, G):
    """Write sigma beside each node, LEFT or RIGHT, whichever has more room.

    A fixed "always to the right" offset was the first version, and it put B's
    count squarely on the dashed B-C edge — B is the leftmost node and its only
    same-layer neighbour lies exactly to the right. Rather than hand-tune per
    node (which rots the moment a position moves), each label MEASURES its
    options in POINTS against every incident edge and every other node. Same
    discipline as the engine's edge-value layer, and the same reason: a rule that
    measures survives a layout edit; a rule that was tuned does not.

    ⚠ THE CANDIDATE SET IS TWO DIRECTIONS, NOT EIGHT — AND THAT IS A LEGIBILITY
    DECISION, NOT A SIMPLIFICATION. Scoring all 8 compass points picks the
    roomiest spot for each label independently, which is locally optimal and
    reads as ARBITRARY: on the layer-1 row alone it put C's count lower-right,
    D's lower-left and E's upper-right, so a reader looking for "where does the
    count live" finds no answer and re-locates every one. Restricted to
    left/right the placement is a RULE the eye learns once — and the collision
    avoidance survives intact, because left-vs-right is the axis the collisions
    are actually on (this layout stacks nodes in horizontal layers, so a label
    is never squeezed vertically and often is horizontally).
    """
    scale = ax.figure.dpi / 72.0
    disp = {n: tuple(c / scale for c in ax.transData.transform(p))
            for n, p in _EK_POS.items()}
    node_r = math.sqrt(_EK_NODE_SIZE / math.pi)
    segs = [(disp[u], disp[v]) for u, v in G.edges()]
    mid_x = sum(p[0] for p in disp.values()) / len(disp)
    for v, s in sorted(sigma.items()):
        cand = []
        for dx in (-1.0, 1.0):
            anchor = (disp[v][0] + _SIGMA_GAP * dx, disp[v][1])
            room = min(
                [math.hypot(anchor[0] - q[0], anchor[1] - q[1]) - node_r
                 for n, q in disp.items() if n != v]
                + [_segment_distance(anchor, a, b) for a, b in segs
                   if disp[v] not in (a, b)]
                + [_segment_distance(anchor, a, b) for a, b in segs
                   if disp[v] in (a, b)])
            cand.append((room, dx))
        # Roomier side wins. On a TIE the label goes OUTWARD, away from the
        # layout's centre — so the symmetric nodes (K at the apex, and any pair
        # mirrored across the middle) resolve the same way every time instead of
        # on float noise or dict order (invariant 8). 0.5 pt is the tie band:
        # below that the two sides are indistinguishable to the eye anyway.
        outward = 1.0 if disp[v][0] >= mid_x else -1.0
        best_room = max(r for r, _ in cand)
        dx = next((d for r, d in cand if r >= best_room - 0.5 and d == outward),
                  max(cand, key=lambda c: c[0])[1])
        dy = 0.0
        ax.annotate(str(s), xy=_EK_POS[v],
                    xytext=(_SIGMA_GAP * dx, _SIGMA_GAP * dy),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=GRAPH_STYLE["value_annotation_size"],
                    color=COLORS["accent"], annotation_clip=False)


def _segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / denom))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def render_bfs_counts(controls, ax):
    upto = _LAYER_STEPS.index(str(controls["reveal"].value)) + 1
    G = _ek_graph()
    dist, sigma, parents, _ = _bfs(G, "A")

    # Same-layer edges are not part of the search tree; draw them dashed so the
    # "count only what is one layer above you" rule is visible in the picture.
    styles = {(u, v): "dashed" for u, v in G.edges() if dist[u] == dist[v]}
    frontier = [v for v in G.nodes() if dist[v] == upto]
    draw_graph(G, ax, pos=_EK_POS, node_size=_EK_NODE_SIZE,
               edge_styles=styles, highlight_nodes=["A"] + frontier,
               frame_nodes=True)
    # sigma is written by hand rather than through `node_values`, for two
    # measured reasons: (1) `draw_graph` hard-codes the row caption "above:
    # value", which is meaningless here and cannot be overridden from its
    # signature; (2) the layout is a hand-placed LAYERED tree with wide
    # horizontal gaps, so a fixed offset to the RIGHT of each circle collides
    # with nothing — the engine's free-direction placement is solving a problem
    # this figure does not have.
    _sigma_labels(ax, {v: sigma[v] for v in G.nodes() if 0 < dist[v] <= upto}, G)

    layer = sorted(v for v in G.nodes() if dist[v] == upto)
    detail = ", ".join(
        f"{v} = {' + '.join(str(sigma[p]) for p in sorted(parents[v]))} = {sigma[v]}"
        if len(parents[v]) > 1 else f"{v} = {sigma[v]}"
        for v in layer)
    # 2026-10-22 (RT101-20): the dashed edge's "contribute nothing" is scoped to
    # root A. From other roots B–C carries traffic, at least the unit between B
    # and C themselves, which is what 10.1 q_9's incorrect statement denies.
    # 2026-10-28 teaching voice (W101-1, point 3): the two sentences that defined Steps 1 and
    # 2 were repeated at all four states. State 1 now defines them; each later state says what
    # its own layer adds, and only "Layer d: ..." (the line that turns the state into numbers)
    # is shared. Each state keeps the 2026-10-22 literals and math spans (voice_pass_check).
    twos = [v for v in layer if len(parents[v]) > 1]
    ones = [v for v in layer if len(parents[v]) == 1]
    if upto == 1:
        return (
            f"Step 1 is a breadth-first search from the root, A, which sorts the nodes into "
            f"layers running downward; a node in layer $d$ is $d$ steps from A. Step 2 "
            f"counts $\\sigma(v)$, the number of different shortest paths from A to $v$, "
            f"with one rule: $\\sigma(v)$ is the sum of $\\sigma$ over the nodes one layer "
            f"above $v$ that are joined to it.\n\n"
            f"Layer {upto}: {detail}. Each node next to the root is reached by one shortest "
            f"path, the single edge from A. That is, layer 1 is all 1s, and everything "
            f"below is built from them. Reveal the next layer to see the first sum.")
    if upto == 4:
        return (
            f"Now the search is complete; Step 1 has put each node in its layer (a node "
            f"in layer $d$ is $d$ steps from A), and Step 2 has given each node $v$ its "
            f"count $\\sigma(v)$, the sum of $\\sigma$ over the joined nodes one layer "
            f"above.\n\n"
            f"Layer {upto}: {detail}. K reads 6, and we never listed six paths: "
            f"$\\sigma(K) = 3 + 3$, and those 3s were themselves 2 + 1 and 1 + 2. Counting "
            f"layer by layer replaces listing paths. Step back through the layers and "
            f"check each sum against the rule.")
    if ones:
        what = (f"Notice that {' and '.join(twos)} each have two nodes above them and add "
                f"their two counts, while {' and '.join(ones)} "
                f"{'has' if len(ones) == 1 else 'have'} one above and copies that count.")
        tail = ("Now look at the dashed edge B–C, which joins two nodes in the same layer "
                "and lies on no shortest path from A; thus B–C adds nothing to A's counts. (From other "
                "roots B–C does carry traffic.) Reveal the next layer and predict its counts "
                "before you look.")
    else:
        what = (f"In this layer {' and '.join(twos)} each add up two counts that differ, and "
                f"the two totals still come out the same.")
        tail = ("Can you predict the count at K, the one node left, before you reveal it?")
    return (
        f"Steps 1 and 2 now run together, and the search reaches the next layer down (a "
        f"node in layer $d$ is $d$ steps from A); each new node $v$ gets $\\sigma(v)$ by "
        f"adding up $\\sigma$ over the nodes one layer above it that are joined to it.\n\n"
        f"Layer {upto}: {detail}. {what} {tail}")


# ---------------------------------------------------------------------------
# CELL 3 — Step 3: the flows, assigned from the bottom up
# ---------------------------------------------------------------------------

_FLOW_STEPS = ["start at the bottom", "one layer up", "two layers up",
               "all the way to A"]


def render_flow_up(controls, ax):
    stage = _FLOW_STEPS.index(str(controls["climb"].value))
    G = _ek_graph()
    dist, sigma, parents, order, edge_flow, node_flow = _flows(G, "A")
    deepest = max(dist.values())
    # Reveal edges from the bottom layer upward: at stage s, show every edge whose
    # LOWER endpoint sits at depth >= deepest - s.
    cutoff = deepest - stage
    shown = {}
    for e, f in edge_flow.items():
        u, v = tuple(e)
        low = max(dist[u], dist[v])
        if low >= cutoff:
            shown[(u, v)] = f
    styles = {(u, v): "dashed" for u, v in G.edges() if dist[u] == dist[v]}
    # ⚠ sigma is NOT drawn here, deliberately. The first render of this cell put
    # sigma beside every node AND the flow on every edge, and on this tree the
    # two label sets landed on top of each other — B's "1" over the B-F flow,
    # I's "3" over the F-I flow, unreadable at four states out of four. sigma has
    # its own cell immediately above; here the caption names the sigma values it
    # actually reasons from, which is what rule 22 asks for anyway.
    draw_graph(G, ax, pos=_EK_POS, node_size=_EK_NODE_SIZE,
               edge_styles=styles, highlight_nodes=["A"] if stage == 3 else [],
               frame_nodes=True, edge_values=shown,
               edge_values_upright=True, edge_values_stacked_fractions=True)

    # 2026-10-22 (RT101-4): the caption used to send the student to σ "shown beside
    # each node", which no state draws (see the ⚠ above for why). It now points at
    # Step 2's figure and LISTS the σ values, computed by `_bfs`, never typed.
    by_sigma = defaultdict(list)
    for v in sorted(sigma):
        if v != "A":
            by_sigma[sigma[v]].append(v)
    sig_list = "; ".join(f"{s} at {', '.join(vs[:-1]) + ' and ' + vs[-1] if len(vs) > 1 else vs[0]}"
                         for s, vs in sorted(by_sigma.items()))
    # 2026-10-28 teaching voice (W101-1, point 3): the rule's two sentences were repeated at
    # all four states. State 1 states the rule; each state opens on its own layer (computed
    # from `dist`); the sigma list, the line that turns the state into numbers, is the one
    # sentence every state shares. Literals and math spans per state are as on 2026-10-22.
    def _layer(d):
        names = sorted(v for v in G.nodes() if dist[v] == d)
        return ", ".join(names[:-1]) + " and " + names[-1] if len(names) > 1 else names[0]
    counts = (f"We take the counts from Step 2 and its figure in the cell above ($\\sigma$ = "
              f"{sig_list}).")
    if stage == 0:
        return (
            f"Steps 3 to 6 climb from the deepest layer back up to A. Each node's total is "
            f"one unit for itself plus everything arriving from below, and the node divides "
            f"that total over the edges leading upward in proportion to the counts $\\sigma$ "
            f"of the nodes above it. {counts}\n\n"
            f"Let's start at the bottom, with {_layer(deepest)}. One unit of flow is "
            f"destined for K, and both nodes above it, I and J, are reached by the same "
            f"number of shortest paths from A ($\\sigma(I) = \\sigma(J) = 3$). Thus K's "
            f"single unit splits evenly, 1/2 on each of the two edges. Climb one layer to "
            f"see an uneven split.")
    if stage == 1:
        return (
            f"Now we climb one layer, and Steps 3 to 6 carry the flow from "
            f"{_layer(deepest - 1)} up to {_layer(deepest - 2)}. {counts}\n\n"
            f"Node I's total flow is 1 for itself plus the 1/2 passing through on its way "
            f"to K, which makes {_frac(node_flow['I'])}. Node I has two nodes above it, F "
            f"and G, and their path counts differ: $\\sigma(F) = 2$ against "
            f"$\\sigma(G) = 1$. Thus the {_frac(node_flow['I'])} splits in the ratio 2:1, "
            f"with 1 onto F and 1/2 onto G. Since twice as many shortest paths came "
            f"through F, twice as much flow goes back that way. Climb again and watch a "
            f"node that splits evenly.")
    if stage == 2:
        return (
            f"We climb one more layer, and Steps 3 to 6 now carry the flow from {_layer(deepest - 2)} "
            f"up to {_layer(deepest - 3)}. {counts}\n\n"
            f"Look at node H, which splits evenly. H's total is 1 for itself plus the 1 "
            f"arriving from J, which makes {_frac(node_flow['H'])}; since its two upper "
            f"neighbors D and E have $\\sigma(D) = \\sigma(E) = 1$, H sends 1 to each. "
            f"Whatever the layer, the proportions come from the $\\sigma$ counts of "
            f"Step 2. Climb to A and check the total that leaves the root.")
    out = sorted((v, edge_flow[frozenset(("A", v))]) for v in G["A"])
    arith = " + ".join(_frac(f) for _, f in out)
    return (
        f"Steps 3 to 6 end at the root. Everything reaches A, and now we can check our "
        f"work. {counts}\n\n"
        f"Notice that the four edges leaving A carry {arith} = "
        f"{_frac(sum(f for _, f in out))}, which "
        f"equals $n - 1 = {G.number_of_nodes() - 1}$: one unit for each node A can "
        f"reach. Since this network is connected, that is each of the other nodes, and "
        f"this total is the arithmetic check on the whole pass.")


# ---------------------------------------------------------------------------
# CELL 4 — Steps 7-8: sum over roots, then halve
# ---------------------------------------------------------------------------

def render_sum_over_roots(controls, ax):
    upto = str(controls["roots"].value)
    all_roots = sorted(_comm_graph().nodes())
    G = _comm_graph()
    if upto == "all seven, then halve":
        roots = all_roots
        bet = _betweenness(G, roots)
        labels = {_pairs(e): v for e, v in bet.items()}
        halved = True
    else:
        roots = all_roots[:all_roots.index(upto) + 1]
        running = defaultdict(Fraction)
        for r in roots:
            _, _, _, _, ef, _ = _flows(G, r)
            for e, f in ef.items():
                running[e] += f
        labels = {_pairs(e): v for e, v in running.items()}
        halved = False

    top_edge = max(sorted(labels), key=lambda e: labels[e])
    draw_graph(G, ax, pos=_COMM_POS, node_size=_COMM_NODE_SIZE,
               highlight_nodes=roots if not halved else [],
               highlight_edges=[top_edge], frame_nodes=True, edge_values=labels,
               edge_values_upright=True, edge_values_stacked_fractions=True)

    # 2026-10-28 teaching voice (W101-1, point 3): three sentences were repeated at six or
    # seven of the eight states. Each running state now says what ITS root does, computed:
    # the root's tied routes if it has any, else the edge carrying the largest share of its
    # traffic and the destinations that edge serves; the leader is named when it changes.
    # The list of shares (the line that turns the state into numbers) stays in every
    # running state. Literals per state as on 2026-10-22: 7, 8, 0 and 1 (running states);
    # 7, 8, 2, 3, 4 and 12 (the halving).
    def _names(vs):
        vs = sorted(vs)
        return ", ".join(vs[:-1]) + " and " + vs[-1] if len(vs) > 1 else vs[0]

    if halved:
        raw = {e: v * 2 for e, v in labels.items()}
        # 2026-10-22 (RT101-19): X and Y are now introduced as a PAIR OF NODES. The
        # old subject ("Every edge's flow was counted twice — once with X as the
        # root and once with Y") bound them to an edge's two endpoints, which is
        # 10.1 q_14's distractor.
        return (
            f"Steps 7 and 8 end here. Every node has had its turn as the root "
            f"({', '.join(roots)}), and now we halve.\n\n"
            f"The traffic between each pair of nodes X and Y was counted twice, once when "
            f"X was the root and once when Y was. Since the graph is undirected, the "
            f"shortest path from X to Y is the same object as the one from Y to X, and an "
            f"edge's importance can't depend on which end we started from. Thus we divide "
            f"each total by 2. For example, the bridge B–D had a raw total of "
            f"{_frac(raw[('B', 'D')])} and has betweenness {_frac(labels[('B', 'D')])}, "
            f"which is (3 nodes on the left) × (4 on the right): each of those 12 pairs "
            f"has no other way across. Step back through the roots and watch B–D's total "
            f"grow toward {_frac(raw[('B', 'D')])}.")
    last = roots[-1]
    i = all_roots.index(last)
    dist_r, sig_r, _, _, ef, _ = _flows(G, last)
    # 2026-10-22 (A-m2, F10): this list was headed "the fractional ones" and is
    # mostly whole numbers (+5, +4, +3). It lists each edge where this root added a
    # share other than 0 or 1 (an edge the root's traffic never crosses adds 0).
    contrib = ", ".join(f"{'–'.join(_pairs(e))} +{_frac(f)}"
                        for e, f in sorted(ef.items(), key=lambda kv: _pairs(kv[0]))
                        if f != 1)
    shares = f"(The edges where {last} added a share other than 0 or 1: {contrib}.)"
    ties = sorted(v for v in sig_r if sig_r[v] > 1)
    if ties:
        w = ties[0]
        routes = " and ".join("–".join(p) for p in sorted(nx.all_shortest_paths(G, last, w)))
        how = (f"From {last}, {_NUM_WORD[sig_r[w]]} shortest paths lead to {w} ({routes}), "
               f"and the unit for {w} splits evenly between them.")
    else:
        big = max(sorted((e for e in ef if last in e), key=_pairs), key=lambda e: ef[e])
        far = next(x for x in big if x != last)
        d_r = nx.single_source_shortest_path_length(G, last)
        d_f = nx.single_source_shortest_path_length(G, far)
        dests = [x for x in G if d_f[x] + 1 == d_r[x]]
        how = (f"From {last}, the largest share of the traffic leaves along "
               f"{'–'.join(_pairs(big))}, which carries {last}'s trips to {_names(dests)}.")
    if i == 0:
        opener = (f"Steps 7 and 8 start with {last} as the root, and the running totals on "
                  f"the edges are just {last}'s flows.")
    elif i == len(all_roots) - 1:
        opener = (f"Steps 7 and 8 have now used each node as the root ({', '.join(roots)}), "
                  f"and {last}'s flows are the last to go into the running totals.")
    else:
        opener = (f"Steps 7 and 8 go on with {last} as the root, and we add {last}'s flows "
                  f"to the running totals (roots used: {', '.join(roots)}).")
    lead = "–".join(top_edge)
    if i == 0:
        prev_lead = None
    else:
        prev = defaultdict(Fraction)
        for r in roots[:-1]:
            for e, f in _flows(G, r)[4].items():
                prev[e] += f
        prev_lead = "–".join(max(sorted(_pairs(e) for e in prev),
                                 key=lambda e: prev[frozenset(e)]))
    final_lead = "–".join(max(sorted(_pairs(e) for e in _betweenness(G)),
                              key=lambda e: _betweenness(G)[frozenset(e)]))
    nxt = all_roots[i + 1] if i + 1 < len(all_roots) else None
    if i == 0:
        rest = (f"Notice that the numbers on the edges are running sums; they become "
                f"betweenness only after each node has had a turn as root and we halve. For "
                f"now the orange edge, {lead}, leads. Add {nxt} as the next root and watch "
                f"which totals grow.")
    elif lead != prev_lead:
        rest = (f"{lead} takes the lead from {prev_lead}. Before you add {nxt}, guess which "
                f"edge {nxt}'s traffic will load most heavily.")
    elif nxt is None:
        rest = (f"{lead} still leads"
                + (", and it will still lead after we halve." if lead == final_lead else ".")
                + " Take the last setting to halve.")
    elif max(_flows(G, nxt)[1].values()) > 1:
        rest = (f"Add {nxt} next, and look for {_NUM_WORD[max(_flows(G, nxt)[1].values())]} "
                f"routes from {nxt} that tie.")
    elif _COMM_SIDES[nxt] != _COMM_SIDES[last]:
        rest = f"Now add {nxt}, the first root on the {_COMM_SIDES[nxt]}-hand side."
    else:
        rest = f"Add {nxt} and see which edge its traffic loads most heavily."
    return f"{opener} {shares}\n\n{how} {rest}"


# ---------------------------------------------------------------------------
# CELL 5 — the Girvan-Newman cascade
# ---------------------------------------------------------------------------

_CASCADE = ["the whole graph", "cut the busiest edge",
            "recompute, then cut again", "keep going"]


def render_gn_cascade(controls, ax):
    stage = _CASCADE.index(str(controls["stage"].value))
    G = _comm_graph()
    H = G.copy()
    removed = []
    tie_note = ""

    if stage >= 1:
        H.remove_edge("B", "D")
        removed.append(("B", "D"))
    if stage >= 2:
        bet = {_pairs(e): v for e, v in _betweenness(H).items()}
        top = max(bet.values())
        tied = sorted(e for e, v in bet.items() if v == top)
        H.remove_edges_from(tied)
        removed.extend(tied)
        # 2026-10-28 teaching voice: no count opener ("Four edges tie"); the tied edges
        # are still computed, and the person, not "an answer", does the choosing.
        tie_note = (
            f"After the first cut, {', '.join('–'.join(e) for e in tied[:-1])} and "
            f"{'–'.join(tied[-1])} tie at {_frac(top)}, and Easley and Kleinberg's method removes \"the edge "
            f"or edges of highest betweenness\" (p. 68). Thus all four go together. No "
            f"rule picks one of them, and if we named a single next edge, we would be "
            f"choosing arbitrarily.")
    if stage >= 3 and H.number_of_edges():
        bet = {_pairs(e): v for e, v in _betweenness(H).items()}
        top = max(bet.values())
        tied = sorted(e for e, v in bet.items() if v == top)
        H.remove_edges_from(tied)
        removed.extend(tied)

    comps = sorted((sorted(c) for c in nx.connected_components(H)), key=lambda c: c[0])
    labels = {_pairs(e): v for e, v in _betweenness(H).items()} if H.number_of_edges() else {}
    # 2026-10-22 (figures F9): removed edges were dashed, the mark the BFS cells
    # use for a same-layer edge. They are now drawn as light-gray context under
    # the surviving network, in the engine's edge width, and the caption says so.
    if removed:
        nx.draw_networkx_edges(G, _COMM_POS, edgelist=removed, ax=ax,
                               edge_color=COLORS["minor_tick"],
                               width=GRAPH_STYLE["edge_width"], node_size=_COMM_NODE_SIZE)
    draw_graph(H.copy(), ax, pos=_COMM_POS, node_size=_COMM_NODE_SIZE,
               frame_nodes=True, edge_values=labels,
               edge_values_upright=True, edge_values_stacked_fractions=True)
    _paint_partition(ax, _COMM_POS, _COMM_NODE_SIZE, comps)

    q = _modularity(G, comps)
    shape = " | ".join("{" + ", ".join(c) + "}" for c in comps)
    if stage == 0:
        body = ("Nothing has been removed yet, and the whole network is one piece. Notice "
                "that the bridge B–D carries 12, more than any other edge, because it is the "
                "only way between the two halves. Which edge do you expect to lead once B–D "
                "is gone? Cut B–D and check your guess.")
    elif stage == 1:
        # 2026-10-22 (RT101-3): "the numbers on the remaining edges have all
        # CHANGED" was false: A–C, E–F and F–G keep 1, 3/2 and 3/2. The changed
        # and kept edges are COMPUTED against stage 0, never typed.
        before = {_pairs(e): v for e, v in _betweenness(G).items()}
        moved = [e for e in sorted(labels) if labels[e] != before[e]]
        kept = [e for e in sorted(labels) if labels[e] == before[e]]
        ex = [e for e in moved if e in (("A", "B"), ("D", "E"))]
        body = (
            f"With one cut the network falls into the two groups anyone would have "
            f"drawn by eye, and across the four settings $Q$ is highest here. Now most "
            f"of the numbers on the remaining edges have changed ("
            + ", ".join(f"{'–'.join(e)} fell from {_frac(before[e])} to {_frac(labels[e])}"
                        for e in ex)
            + f"), while {', '.join('–'.join(e) for e in kept[:-1])} and "
            f"{'–'.join(kept[-1])} kept theirs, and only recomputing tells us which. "
            f"Since betweenness is a property of the whole network, we recompute it "
            f"after each removal instead of reusing the old values. From now on, a "
            f"root sends traffic only to the nodes in its own piece. Step on and see "
            f"which edges the new numbers single out.")
    elif stage == 2:
        body = (tie_note + " Modularity has fallen, a sign that we have cut past the "
                "real structure and are now breaking up a group that belonged "
                "together.")
    else:
        # ⚠ THIS BRANCH DELIBERATELY STOPS SHORT OF NAMING THE STOPPING RULE.
        # An earlier draft ended "record Q at every stage and keep the split
        # where Q was largest" — which is q_20's key, near-verbatim, printed one
        # cell above it (the 9.2 q_32 give-away shape). The cell now shows that Q
        # rises and then falls and that the method itself never halts; turning
        # that into a rule is the student's step, and it is q_20.
        qs = [_modularity(G, c) for c in _cascade_divisions(G)]
        body = ("Keep going and the network dissolves into smaller and smaller pieces, "
                "down to each node on its own. Then the method goes on to destroy the "
                "structure it just found, since it never stops by itself. Look back at $Q$ across "
                f"the four settings: ${', '.join(_frac(x) for x in qs)}$. That is, $Q$ "
                "rose and then fell. Each stage of the cascade is a division, and the "
                "stages are not equally good. What would you need to choose one?")
    return (
        f"Removed so far (light gray): "
        f"{', '.join('–'.join(e) for e in removed) if removed else 'nothing'}. "
        f"Each connected piece that remains is a community: {shape}. "
        f"This division's modularity is {_fmt_q(q)}.\n\n"
        f"{body}")


def _cascade_divisions(G):
    """The connected pieces after each of the cascade's four settings (stage 0 to 3)."""
    out, H = [], G.copy()
    out.append(sorted(sorted(c) for c in nx.connected_components(H)))
    for _ in range(3):
        if not H.number_of_edges():
            break
        bet = {_pairs(e): v for e, v in _betweenness(H).items()}
        top = max(bet.values())
        H.remove_edges_from([e for e, v in bet.items() if v == top])
        out.append(sorted(sorted(c) for c in nx.connected_components(H)))
    return out


# ---------------------------------------------------------------------------
# CELL 6 — the null model: how many edges SHOULD fall between i and j?
# ---------------------------------------------------------------------------

def render_expected_edges(controls, ax):
    ki = int(controls["k_i"].value)
    kj = int(controls["k_j"].value)
    m = int(controls["m"].value)
    two_m = 2 * m
    # ⚠ 2026-10-22 (RT101-5, RT101-6; Hari's ruling 6). Two corrections:
    #   * the m slider now starts at 8 (YAML), so 2m >= 16 >= k_i + k_j at every
    #     state; at m = 4 to 7, 70 of 1,728 states described a network in which
    #     nodes i and j held more stubs than the whole network;
    #   * the stub probability is k_j/(2m - 1) EXACTLY (a stub cannot pair with
    #     itself), and k_j/2m is the approximation modularity uses. Both are
    #     printed, with the factor 2m/(2m - 1) between them.
    exact_p = Fraction(kj, two_m - 1)
    approx_p = Fraction(kj, two_m)
    expected = Fraction(ki * kj, two_m)
    exact_e = Fraction(ki * kj, two_m - 1)

    # A native stub picture: two nodes, each with its degree drawn as loose
    # half-edges, and (2026-10-22, figures F13) the network's other
    # 2m - k_i - k_j stubs drawn as a pool below them, so the share of the
    # 2m - 1 other ends that belong to j can be seen and moves with m.
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.95, 0.80)
    ax.set_aspect("equal")
    for x, k, name, deg in ((-0.62, ki, "i", ki), (0.62, kj, "j", kj)):
        ax.add_patch(__import__("matplotlib").patches.Circle(
            (x, 0.12), 0.13, facecolor="white",
            edgecolor=COLORS["primary"], linewidth=2, zorder=3))
        ax.text(x, 0.12, f"${name}$", ha="center", va="center", zorder=4,
                fontsize=GRAPH_STYLE["label_font_size"], color=COLORS["primary"])
        # Raised from 0.30 (2026-10-22, F13): at k = 7 and 8 the top stub dot
        # touched the subscript of the old label.
        ax.text(x, 0.50, f"$k_{name} = {deg}$", ha="center", va="bottom",
                fontsize=GRAPH_STYLE["annotation_font_size"], color=COLORS["text"])
        inward = 1 if x < 0 else -1
        for s in range(k):
            theta = math.pi * ((s + 0.5) / max(k, 1) - 0.5)
            dx = inward * 0.30 * math.cos(theta)
            dy = 0.30 * math.sin(theta)
            ax.plot([x, x + dx], [0.12, 0.12 + dy], color=COLORS["primary"],
                    linewidth=1.4, zorder=2)
            ax.plot([x + dx], [0.12 + dy], marker="o", markersize=3.4,
                    color=COLORS["accent"], zorder=3)
    rest = two_m - ki - kj
    per_row = 20
    for s in range(rest):
        r, c = divmod(s, per_row)
        ax.plot([-0.95 + c * (1.90 / (per_row - 1))], [-0.36 - 0.09 * r], marker="o",
                markersize=3.4, color=COLORS["primary"], zorder=3)
    ax.text(0.0, -0.82,
            f"$2m = {two_m}$ stubs: {ki} on $i$, {kj} on $j$, {rest} elsewhere",
            ha="center", va="center", fontsize=GRAPH_STYLE["annotation_font_size"],
            color=COLORS["text"])
    ax.set_axis_off()

    stubs = "stub" if ki == 1 else "stubs"
    # 2026-10-28 teaching voice (W101-1, point 3). At all 1,472 states the caption repeated
    # four sentences word for word ("Now pair the stubs blindly", "Take one of i's stubs",
    # the null-model sentence, "Raise m ..."). The first two are folded into the sentences
    # that carry the state's numbers; the null-model sentence moved to the description; the
    # closing instruction depends on the state (at m = 30 the slider cannot be raised).
    # SCOPE FIX, same date: "blind pairing can join i and j more than once" is false when i or
    # j holds a single stub (one stub, at most one i-j edge: 345 of the 1,472 states); those
    # states now say so. Literals per state as on 2026-10-22 (one "1"); math spans as before.
    num_e = _num(exact_e)
    if min(ki, kj) == 1:          # then exact_e <= 8/15 < 1 at every state
        lone = "$i$" if ki == 1 else "$j$"
        repeat = (f"Here the exact count, {num_e}, is not above 1, and since {lone} holds a "
                  f"single stub, $i$ and $j$ can meet at most once. With more stubs on both, "
                  f"blind pairing can join two nodes more than once (and can join a node to "
                  f"itself).")
    elif exact_e > 1:
        repeat = (f"Notice that the exact count, {num_e}, is above 1, a count no simple "
                  f"network allows; blind pairing can join $i$ and $j$ more than once (and "
                  f"can join a node to itself).")
    else:
        repeat = (f"Here the exact count, {num_e}, is not above 1, though at higher degrees "
                  f"it does pass that mark; blind pairing can join $i$ and $j$ more than once "
                  f"(and can join a node to itself).")
    # The m slider's maximum is 30 (YAML); at the top the instruction points down.
    if m < 30:
        move = "Raise $m$ and watch the two probabilities draw together."
    else:
        move = (f"At $m = {m}$ the slider stops; lower $m$ and watch the two probabilities "
                f"move apart.")
    return (
        f"With each edge cut in half, node $i$ holds $k_i = {ki}$ {stubs}, node $j$ holds "
        f"$k_j = {kj}$, and the whole network has $2m = {two_m}$ stubs, two for each of its "
        f"$m = {m}$ edges; now we pair them up blindly.\n\n"
        f"Any one of $i$'s stubs pairs with one of the other $2m - 1 = {two_m - 1}$ stubs "
        f"(a stub can't pair with itself), of which $j$ owns $k_j = {kj}$, and the "
        f"probability that it lands on $j$ is\n\n"
        f"$$\\frac{{k_j}}{{2m - 1}} = \\frac{{{kj}}}{{{two_m - 1}}} \\approx "
        f"{_d3(exact_p)}.$$\n\n"
        f"Modularity uses the simpler $\\frac{{k_j}}{{2m}} = \\frac{{{kj}}}{{{two_m}}} "
        f"\\approx {_d3(approx_p)}$, which is very close once the network is large; here "
        f"the two differ by the factor $\\frac{{2m}}{{2m - 1}} = "
        f"\\frac{{{two_m}}}{{{two_m - 1}}} \\approx {_d3(Fraction(two_m, two_m - 1))}$. "
        f"Since node $i$ has $k_i = {ki}$ {stubs} to place, the baseline's expected "
        f"number of $i$–$j$ edges is\n\n"
        f"$$k_i \\cdot \\frac{{k_j}}{{2m}} = {ki} \\cdot \\frac{{{kj}}}{{{two_m}}} = "
        f"{_frac(expected)}"
        + (f" \\approx {_d3(expected)}" if expected.denominator != 1 else "")
        + f",$$\n\nagainst the exact {num_e}. "
        f"{repeat}\n\n{move}")


# ---------------------------------------------------------------------------
# CELL 7 — modularity as observed-minus-expected
# ---------------------------------------------------------------------------

# Kept OUT of the f-string below on purpose: LaTeX braces and f-string braces
# fight, and doubling every one of them is where a formula quietly loses a pair.
_Q_FORMULA = (
    r"$$Q = \underbrace{\frac{\text{edges inside groups}}{m}}_{\text{observed}}"
    r" \;-\; \underbrace{\frac{1}{2m}\sum_{s}\sum_{i,j \in s}"
    r"\frac{k_i k_j}{2m}}_{\text{expected under the baseline}}$$")
# 2026-10-22 (definitions D-m1): every symbol of the displayed formula is named in
# the sentence after it (GLOSSARY, the m ruling), including the group index s and
# the i = j terms of the double sum, which 10.1 q_23 needs.
# 2026-10-28 teaching voice: one sentence (it was two), the one sentence this cell repeats
# at every state, since every state reads its numbers off this formula.
_Q_GLOSS = (
    "Here $m$ is the number of edges, $k_i$ is node $i$'s degree, $s$ runs over the "
    "groups of the division, and the double sum runs over each ordered pair $(i, j)$ "
    "of nodes in group $s$, a node paired with itself included, which makes each "
    "group's inner sum $\\frac{1}{2m}\\cdot\\frac{(\\text{the group's degree total})^2}{2m}$.")

_PARTITIONS = {
    "the two obvious groups": [["A", "B", "C"], ["D", "E", "F", "G"]],
    "move D across": [["A", "B", "C", "D"], ["E", "F", "G"]],
    "the two bridge ends together": [["B", "D"], ["A", "C", "E", "F", "G"]],
    "everyone in one group": [["A", "B", "C", "D", "E", "F", "G"]],
    "everyone alone": [["A"], ["B"], ["C"], ["D"], ["E"], ["F"], ["G"]],
}


def render_modularity_score(controls, ax):
    choice = str(controls["partition"].value)
    parts = _PARTITIONS[choice]
    G = _comm_graph()
    m = G.number_of_edges()
    k = dict(G.degree())
    where = {v: i for i, p in enumerate(parts) for v in p}
    inside = [(u, v) for u, v in G.edges() if where[u] == where[v]]
    between = [(u, v) for u, v in G.edges() if where[u] != where[v]]
    # 2026-10-22 (figures F8): orange now marks the edges BETWEEN groups, as it does
    # in `concept_why_communities` on the same network and layout. It used to mark
    # the inside edges here, the opposite meaning one color had two cells earlier.
    draw_graph(G, ax, pos=_COMM_POS, node_size=_COMM_NODE_SIZE,
               highlight_edges=between, frame_nodes=True)
    _paint_partition(ax, _COMM_POS, _COMM_NODE_SIZE, parts)

    observed = Fraction(len(inside), m)
    q = _modularity(G, parts)
    expected = observed - q
    per = "; ".join(
        f"{_name_set(p)} holds {n_in} {'edge' if n_in == 1 else 'edges'} and has degree "
        f"total {sum(k[v] for v in p)}"
        for i, p in enumerate(parts)
        for n_in in [sum(1 for u, v in inside if where[u] == i)])
    deg = sorted(k.items(), key=lambda kv: (-kv[1], kv[0]))

    if choice == "everyone in one group":
        # 2026-10-28 (Hari's walk, W101-2): "$Q$ can't be raised just by lumping
        # everyone together" was false: lumping everyone together raises any division
        # that scores below 0 to exactly 0 (10.1 q_28's division scores -1/2). The true
        # statement, as in q_27's explanation (the lead's fix of 2026-10-22): one group
        # holding every node scores exactly 0 on every network, so lumping everyone
        # together never makes a network look structured.
        note = ("Why does the baseline put all of them inside, an expected fraction of 1? A "
                "random rewiring also has to put each edge inside the single group, and the "
                "two fractions cancel. Thus one group holding every node scores exactly $0$, "
                "whatever the network, and lumping everyone together never makes a network "
                "look structured. Now try the opposite extreme, everyone alone.")
    elif choice == "everyone alone":
        note = ("With no edge inside a group, only the baseline's share is left, with "
                "a minus sign. That is, splitting everything apart scores lower than not "
                "splitting at all.")
    elif choice == "the two bridge ends together":
        # 2026-10-22 (A-m4, B-m5): B is not one of "the two highest-degree nodes"
        # (B ties F at 3), and the baseline expects 2/3 of a B–D edge, not a sure one.
        kb, kd = k["B"], k["D"]
        note = (f"B and D are the two ends of the bridge, and their group does hold an "
                f"edge. Then why does it earn so little? D has the highest degree ({kd}) "
                f"and B is tied for second ({kb}, with F), and thus the baseline already "
                f"expects ${kb} \\cdot {kd}/{2 * m} = {_frac(Fraction(kb * kd, 2 * m))}$ "
                f"of a B–D edge. A group earns credit only for the edges it holds beyond "
                f"what the baseline expects, and one edge is close to "
                f"{_frac(Fraction(kb * kd, 2 * m))}.")
    elif choice == "move D across":
        q0 = _modularity(G, _PARTITIONS["the two obvious groups"])
        note = (f"D has {k['D']} edges and only one of them reaches the left group, and "
                f"thus dragging D across trades three inside edges for one. Notice how much "
                f"a single node costs: $Q$ falls from ${_frac(q0)}$ to ${_frac(q)}$, by more "
                f"than half.")
    else:
        note = ("Are the two obvious groups the best we can do? We checked all 63 ways of "
                "cutting seven nodes into two groups, not just the ones that look "
                "sensible, and none scores higher; no division into more groups does "
                "better either. Now move D across and watch what one node does to the "
                "score.")
    n_b = len(between)
    across = ("none runs between groups" if n_b == 0 else
              f"1 runs between the groups (orange)" if n_b == 1 else
              f"{n_b} run between groups (orange)")
    # 2026-10-28 teaching voice: the state's numbers in one sentence (it was four short ones,
    # "The baseline puts, on average, ..." word for word at three states, and "The difference
    # is" at all five).
    return (
        _Q_FORMULA + "\n\n" + _Q_GLOSS + "\n\n"
        f"For the division *{choice}*, {per}. Of the $m = {m}$ edges, {len(inside)} lie "
        f"inside a group and {across}, which makes the observed fraction "
        f"{_num(observed)}; the baseline puts "
        + ("all of them" if expected == 1 else f"{_num(expected)} of them")
        + f" inside on average, and the difference is\n\n"
        f"$$Q = {_frac(observed)} - {_frac(expected)} = {_frac(q)}"
        + (f" \\approx {_d3(q)}" if Fraction(q).denominator != 1 else "")
        + f".$$\n\n{note}")


# ---------------------------------------------------------------------------
# CELLS 8-10 — CLIQUE PERCOLATION (Palla, Derenyi, Farkas & Vicsek, Nature 2005)
# ---------------------------------------------------------------------------
#
# WHY THESE THREE EXIST. Every method on this sheet so far returns a PARTITION:
# each node gets exactly one label. The lecture notes' own closing section calls
# that out as a simplification and names the paper that repairs it. These cells
# introduce the repair, and they are placed AFTER the modularity block on
# purpose -- the contrast only lands once the student has a score that is
# defined on partitions and nothing else.
#
# ⚠ EVERY COMMUNITY BELOW IS COMPUTED, NEVER TYPED, by the same three-route
# harness `scripts/derive_l10_papers_keys.py` cross-checks: networkx, the
# clique-clique overlap matrix (the recipe the paper itself describes), and a
# brute-force enumeration of k-subsets that calls no clique finder at all.
# The local implementation here is the brute-force route, restated so the cell
# does not import a script; the harness asserts the three agree.
#
# ⚠ AND THE GRAPHS ARE CHOSEN SO NO ITEM'S KEY IS PRINTED HERE (rule 24). The
# overlap cell runs on the lesson network WITH ONE EDGE ADDED, so every Q it
# prints belongs to a graph that appears nowhere else on the sheet.

def _k_cliques(G, k):
    """Every set of k nodes in which all k(k-1)/2 pairs are joined."""
    return [frozenset(s) for s in itertools.combinations(sorted(G.nodes()), k)
            if all(G.has_edge(u, v) for u, v in itertools.combinations(s, 2))]


def _percolate(G, k):
    """Palla's k-clique communities: roll from k-clique to adjacent k-clique,
    where ADJACENT means sharing exactly k-1 nodes. Returns a list of node sets,
    sorted, which may OVERLAP and need not cover every node."""
    cliques = _k_cliques(G, k)
    parent = {c: c for c in cliques}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in itertools.combinations(cliques, 2):
        if len(a & b) == k - 1:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for c in cliques:
        groups.setdefault(find(c), set()).update(c)
    # Sorted by their own node lists, so the printed order follows the drawn
    # layout left-to-right rather than by size — a reader matches the text to
    # the picture without re-sorting it in their head.
    return sorted(sorted(g) for g in groups.values())


def _membership(comms, G):
    """Palla's membership number for every node — including the zeros."""
    m = {n: 0 for n in sorted(G.nodes())}
    for c in comms:
        for n in c:
            m[n] += 1
    return m


def _paint_cover(ax, pos, node_size, comms, G, legend=True):
    """Paint a cover: one tint per community, a SPLIT fill for a node in two, light
    gray for a node in none, and (by default) a legend naming each community.

    2026-10-22 (RT101-8, figures F8). The old `_cover_kwargs` gave every node in
    more than one community the same caramel, which on this sheet also meant a
    singleton and an ordinary group; and at the pin a highlight hid it anyway. A
    community's color is its index in `comms` (sorted by node list), so the same
    community keeps its color from one control state to the next.
    """
    if not comms:
        _paint_nodes(ax, pos, node_size, {n: [] for n in G.nodes()})
        if legend:
            _community_legend(ax, [("in no community", [])])
        return
    color = {i: _GROUP_PIN[i % len(_GROUP_PIN)] for i in range(len(comms))}
    paint = {}
    for n in G.nodes():
        cols = [color[i] for i, c in enumerate(comms) if n in c]
        paint[n] = cols[:2]
    _paint_nodes(ax, pos, node_size, paint)
    if legend:
        entries = [(_name_set(c), [color[i]]) for i, c in enumerate(comms)]
        if any(len(v) > 1 for v in paint.values()):
            entries.append(("in two at once", [color[0], color[1]]))
        if any(not v for v in paint.values()):
            entries.append(("in no community", []))
        _community_legend(ax, entries)


def _name_sets(comms):
    return ", ".join("{" + ", ".join(c) + "}" for c in comms) or "none at all"


# --- CELL 8 — what a k-clique community IS ---------------------------------

_CLIQUE_NETWORKS = ["the lesson network",
                    "two triangles joined by an edge",
                    "two triangles joined at a node"]

_TRI_EDGE = [("P", "Q"), ("P", "R"), ("Q", "R"), ("R", "S"),
             ("S", "T"), ("S", "U"), ("T", "U")]
_TRI_EDGE_POS = {"P": (0.0, 1.05), "Q": (0.0, -0.35), "R": (1.25, 0.35),
                 "S": (3.05, 0.35), "T": (4.3, 1.05), "U": (4.3, -0.35)}

_TRI_NODE = [("V", "W"), ("V", "X"), ("W", "X"), ("X", "Y"), ("Y", "Z"), ("X", "Z")]
_TRI_NODE_POS = {"V": (0.0, 1.0), "W": (0.0, -0.5), "X": (1.45, 0.25),
                 "Y": (2.9, 1.0), "Z": (2.9, -0.5)}


def render_clique_communities(controls, ax):
    choice = str(controls["network"].value)
    if choice == "two triangles joined by an edge":
        G, pos, size = nx.Graph(_TRI_EDGE), _TRI_EDGE_POS, 620
    elif choice == "two triangles joined at a node":
        G, pos, size = nx.Graph(_TRI_NODE), _TRI_NODE_POS, 620
    else:
        G, pos, size = _comm_graph(), _COMM_POS, _COMM_NODE_SIZE

    k = 3
    comms = _percolate(G, k)
    mem = _membership(comms, G)
    shared = [n for n, c in mem.items() if c > 1]
    loose = [n for n, c in mem.items() if c == 0]
    tris = _k_cliques(G, k)

    draw_graph(G, ax, pos=pos, node_size=size, frame_nodes=True)
    _paint_cover(ax, pos, size, comms, G)

    # The adjacency check, run on this graph and reported rather than asserted.
    # 2026-10-22 (RT101-1): the rule is stated for general k in the cell's
    # description, as "k - 1 nodes"; the caption applies it at k = 3 and says so.
    # (The old caption printed "share $2$ nodes — one fewer than a whole clique"
    # as the GENERAL rule, which is 10.1 q_32's distractor at k = 4.)
    adjacent = [(a, b) for a, b in itertools.combinations(tris, 2)
                if len(a & b) == k - 1]
    if adjacent:
        adj_text = " ".join(
            f"{_name_set(sorted(a))} and {_name_set(sorted(b))} share "
            f"{' and '.join(sorted(a & b))}, which is $k - 1 = {k - 1}$ nodes, and thus "
            f"they roll together." for a, b in adjacent)
    else:
        adj_text = (f"No two of them share $k - 1 = {k - 1}$ nodes, and thus none of "
                    f"them roll together.")

    body = (f"At $k = {k}$ on {choice}, the triangles are "
            f"{_name_sets([sorted(t) for t in tris])}. {adj_text} Rolling gives the "
            f"communities {_name_sets(comms)}.")

    # 2026-10-28 teaching voice (W101-1): the lesson network's and the edge-joined pair's
    # captions shared three sentences word for word, and "the method does not promise one"
    # made the method the speaker. Each state's tail is now written for that state (state 3
    # follows the caption the lead drafted with the approved description, with its sets in
    # plain text, since the 2026-10-22 caption had no math there). Literals per state as
    # before: 1 (states 1 and 2); 2, 2 and 1 (state 3).
    bridge = [f"{u}–{v}" for u, v in G.edges()
              if not any(u in c and v in c for c in comms)]
    if shared:
        x = shared[0]
        tail = (f"This is an answer no partition can give. The two triangles meet only at "
                f"{x}, one node short of the two they would need to roll together, and thus {x} "
                f"sits in {mem[x]} communities at once. {x}'s membership number is "
                f"{mem[x]} (the split color in the figure), and each other node's is 1. "
                f"Notice that Girvan–Newman could never give this answer, since once a "
                f"network is cut into pieces, each node lies in exactly one of them. Now go "
                f"back to the lesson network. Does any node there belong to two "
                f"communities?")
    elif loose:
        tail = (f"Nodes {', '.join(loose)} are in no community at all, since they "
                f"belong to no triangle. A method built on cliques is allowed to leave "
                f"a node out, and that is a finding about the node: it sits in no "
                f"densely connected group.")
    elif choice == "the lesson network":
        tail = (f"On this network the answer happens to be a partition, since each node's "
                f"membership number is 1, and it is the same pair of groups the edge-cutting "
                f"cascade found. We had no guarantee of that, though; under the rolling rule "
                f"a node can land in two communities. Notice that {' and '.join(bridge)} "
                f"lies inside neither community, since no triangle contains it. Try *two "
                f"triangles joined at a node* to see a node land in two.")
    else:
        tail = (f"On this network too, each node's membership number is 1, and the answer "
                f"happens to be a partition. The edge {' and '.join(bridge)} lies inside neither "
                f"community, since no triangle contains it. What do you expect when the two "
                f"triangles meet at a node, with no edge between them? Try *two triangles "
                f"joined at a node*.")
    return f"{body}\n\n{tail}"


# --- CELL 9 — k as a resolution knob ---------------------------------------

_LADDER_BLOCKS = (["a", "b", "c", "d", "e"], ["a", "b", "f", "g"], ["h", "i", "j"])
_LADDER_TETHER = ("e", "h")
# ⚠ THE FIVE-CLIQUE MUST BE IN CONVEX POSITION, AND THAT IS NOT A STYLE CHOICE.
# {a,b,c,d,e} is complete, so all ten of its chords are drawn. Any node inside
# the hull of the others has a chord passing through it, and `edge_audit` reds
# on exactly that ("a reader sees a line entering the node and leaving it").
# MEASURED: a column layout put c-e through d (4 findings); bowing d outward
# merely moved the problem (a-e and b-c then passed within 12.1 and 10.5 pt of
# it, against a 14.0 pt node radius). A convex polygon has no interior vertex,
# so no diagonal can reach one. These five sit on a pentagon; a and b take the
# two right-hand corners so the second block hangs off that side without any of
# its chords re-entering the pentagon. Do not "tidy" these into rows.
_LADDER_POS = {"c": (0.00, 2.05), "d": (-1.28, 1.12), "e": (-0.79, -0.39),
               "b": (0.79, -0.39), "a": (1.28, 1.12),
               "f": (2.75, 1.55), "g": (2.45, -0.30),
               "h": (-1.75, -1.65), "i": (-0.45, -2.15), "j": (-2.25, -2.70)}
_LADDER_NODE_SIZE = 470


def _ladder_graph():
    G = nx.Graph()
    G.add_nodes_from(sorted(_LADDER_POS))
    for block in _LADDER_BLOCKS:
        G.add_edges_from(itertools.combinations(block, 2))
    G.add_edge(*_LADDER_TETHER)
    return G


def render_resolution_knob(controls, ax):
    k = int(controls["k"].value)
    G = _ladder_graph()
    comms = _percolate(G, k)
    mem = _membership(comms, G)
    shared = [n for n, c in mem.items() if c > 1]
    loose = [n for n, c in mem.items() if c == 0]
    sizes = [len(c) for c in comms]

    # ⚠ 2026-10-22 (RT101-8). The figure used to ring every COVERED node in
    # orange and nothing else: at k = 3 all ten nodes looked like one group
    # against a caption naming two, and at k = 4 neither the two communities nor
    # the overlap at a and b could be seen (at the pin a highlight hides a group
    # color). Now each community has its own tint, a node in two has a split
    # fill, a node in none is gray, and a legend decodes them. The states still
    # differ from one another (rule 25): the colors and the legend change with k.
    draw_graph(G, ax, pos=_LADDER_POS, node_size=_LADDER_NODE_SIZE, frame_nodes=True)
    _paint_cover(ax, _LADDER_POS, _LADDER_NODE_SIZE, comms, G)

    if not comms:
        found = (f"no community at all, since this network holds no clique of {k} nodes "
                 f"and nothing can roll.")
    else:
        word = "community" if len(comms) == 1 else "communities"
        sized = "size" if len(comms) == 1 else "sizes"
        found = (f"{len(comms)} {word}, of {sized} "
                 f"{', '.join(str(s) for s in sizes[:-1]) + ' and ' + str(sizes[-1]) if len(sizes) > 1 else sizes[0]}: "
                 f"{_name_sets(comms)}.")
    n = G.number_of_nodes()
    cover = (f"All {n} nodes are left out." if len(loose) == n else
             f"Left out: {', '.join(loose)} ({len(loose)} of the {n} nodes)." if loose
             else "Each node lands in at least one community.")
    # 2026-10-28 teaching voice: with every node left out (k = 6), "No node is in two" said
    # nothing and made a third repeat of the sentence; it is dropped there.
    over = (f"In two communities at once: {', '.join(shared)} (split color)." if shared
            else "" if not comms else "No node is in two communities.")

    # Palla et al.'s rule for an unweighted network, EVALUATED here with this
    # sheet's test for a giant community (more than twice the next largest).
    by_size = sorted(sizes, reverse=True)
    if len(by_size) >= 2:
        ratio = Fraction(by_size[0], by_size[1])
        if ratio > 2:
            verdict = (f"At this setting the largest community is {_frac(ratio)} times the "
                       f"second largest, more than twice, and thus a giant community has "
                       f"formed. This $k$ is too low; the structure below it has merged into "
                       f"one piece. Raise $k$ by one and watch the giant split.")
        else:
            first_ok = next(kk for kk in range(3, 7) if _passes(G, kk))
            verdict = (f"Now the largest community is {_frac(ratio)} times the second "
                       f"largest, not more than twice. Thus no giant community has "
                       f"formed, and this $k$ passes the test."
                       + (f" Since $k = {k}$ is the smallest setting that passes, the "
                          f"rule stops here. Raise $k$ once more and count who drops out."
                          if k == first_ok else
                          f" A smaller setting already passed, so the rule would have "
                          f"stopped there."))
    elif len(by_size) == 1:
        verdict = (f"Only one community is left, holding {by_size[0]} of the {n} nodes, "
                   f"and there is no second community to compare it with. How much changes "
                   f"from one $k$ to the next depends on how large the network's complete "
                   f"groups are. Raise $k$ once more and see what is left.")
    else:
        verdict = ("Past the largest clique in the network nothing can roll, and the "
                   "rolling rule returns an empty answer. Walk the control back down to "
                   "$k = 3$ and watch the communities come back.")

    # 2026-10-28 teaching voice (W101-1, point 3): the nesting fact is the one sentence every
    # state repeats (each state's numbers obey it); Palla's rule is stated once, at k = 3,
    # where the test first bites, and "how much changes" sits at k = 5, where a group drops.
    rule = ("For an unweighted network like this one, Palla and his coauthors take the "
            "smallest $k$ at which no giant community appears, low enough to find as much "
            "structure as possible and high enough that no one community has absorbed it. "
            if k == 3 else "")
    return (
        f"Raising $k$ can shrink a community, split it or drop nodes from it, and it "
        f"never adds to one; each community at $k$ sits inside a community at "
        f"$k - 1$, since adjacent $k$-cliques share $k - 1$ nodes, which already form "
        f"a $(k-1)$-clique in both.\n\n"
        f"At $k = {k}$ the rolling finds {found} {cover}" + (f" {over}" if over else "")
        + f"\n\n{rule}{verdict}")


def _passes(G, k):
    """This sheet's giant test at k: at least two communities, the largest not
    more than twice the second largest."""
    sizes = sorted((len(c) for c in _percolate(G, k)), reverse=True)
    return len(sizes) >= 2 and Fraction(sizes[0], sizes[1]) <= 2


# --- CELL 10 — a partition cannot hold an overlap --------------------------

_OVERLAP_EDGES = _COMM_EDGES + [("A", "D")]
_OVERLAP_CHOICES = ["clique percolation: D is in both",
                    "force D into the left group",
                    "force D into the right group",
                    "give D a group of its own"]
_OVERLAP_PARTS = {
    "force D into the left group":  [["A", "B", "C", "D"], ["E", "F", "G"]],
    "force D into the right group": [["A", "B", "C"], ["D", "E", "F", "G"]],
    "give D a group of its own":    [["A", "B", "C"], ["D"], ["E", "F", "G"]],
}
# 2026-10-22 (figures F8): each group keeps ONE color across the four states. The
# left side is always blue and the right side always green, whether or not D is
# with it, and D alone takes the third color. (E, F and G used to turn caramel at
# the last state although their group had not changed.)
_OVERLAP_COLORS = {
    "force D into the left group":  [_GROUP_PIN[0], _GROUP_PIN[1]],
    "force D into the right group": [_GROUP_PIN[0], _GROUP_PIN[1]],
    "give D a group of its own":    [_GROUP_PIN[0], _GROUP_PIN[2], _GROUP_PIN[1]],
}


def _overlap_graph():
    G = nx.Graph()
    G.add_nodes_from(sorted(_COMM_POS))
    G.add_edges_from(_OVERLAP_EDGES)
    return G


def _group_sum(G, groups):
    """The group sum of the modularity formula, evaluated on ANY list of groups.

    On a partition it is Q. On a cover (a node in two groups) it still returns a
    number, and that number is not a modularity (RT101-11): the shared node's
    degree is counted in both groups. Exact, by Fraction.
    """
    m = G.number_of_edges()
    k = dict(G.degree())
    inside = sum(1 for u, v in G.edges() if any(u in s and v in s for s in groups))
    expect = sum(Fraction(sum(k[i] for i in s) ** 2, (2 * m) ** 2) for s in groups)
    return Fraction(inside, m) - expect


def render_partition_vs_cover(controls, ax):
    choice = str(controls["answer"].value)
    G = _overlap_graph()
    m = G.number_of_edges()
    merged = _modularity(G, [sorted(G.nodes())])
    # 2026-10-28 teaching voice (W101-1, point 3): the opening two sentences and the closing
    # three were repeated at all four states, and "Here is what the forcing costs" at three.
    # The added edge and the four moves are now set out once, in the first state; each
    # forced state names the edges of D that its own division cuts (computed) and what
    # merging would score. Math spans per state as on 2026-10-22 ($Q = 0$ in each).

    if choice == _OVERLAP_CHOICES[0]:
        comms = _percolate(G, 3)
        draw_graph(G, ax, pos=_COMM_POS, node_size=_COMM_NODE_SIZE, frame_nodes=True)
        _paint_cover(ax, _COMM_POS, _COMM_NODE_SIZE, comms, G)
        mem = _membership(comms, G)
        dual = [n for n, c in mem.items() if c > 1]
        d = dual[0]
        # 2026-10-22 (RT101-11): "the formula has nothing to compute" was false
        # for the group sum the sheet shows; it returns 87/400 on this cover. The
        # reason now given is the definition (a division) and the double count.
        pushed = _group_sum(G, [set(c) for c in comms])
        return (
            "We start from the lesson network with one edge added, between A and D. That "
            "edge completes the triangle A–B–D, and D now sits in triangles on both sides "
            "(A–B–D on the left, D–E–F and D–F–G on the right).\n\n"
            f"At $k = 3$ the triangles roll into {_name_sets(comms)}: {{A, B, D}} rolls "
            f"with {{A, B, C}} (they share A and B), and {{D, E, F}} rolls with "
            f"{{D, F, G}}, but the two sides share only {d}. Thus {d} is in both (the "
            f"split color).\n\n"
            f"$Q$ has no value for this answer, because modularity is defined for a "
            f"division, in which each node is in exactly one group. You can push this "
            f"answer through the group sum anyway, but {d}'s degree is then counted in "
            f"both groups (the sum gives ${_frac(pushed)}$ here), and that number is not a "
            f"modularity; we can't rank it against the $Q$ of a partition.\n\n"
            f"A partition method has three moves for {d}: put it in one group, put it in "
            f"the other, or cut it out on its own. Merging the two groups is a fourth, and "
            f"it gives up the division altogether (it scores $Q = {_frac(merged)}$). None "
            f"of the four lets us record that {d} belongs to both. Now force {d} into one "
            f"group and see what the score charges for it.")

    parts = _OVERLAP_PARTS[choice]
    q = _modularity(G, parts)
    draw_graph(G, ax, pos=_COMM_POS, node_size=_COMM_NODE_SIZE, frame_nodes=True)
    _paint_partition(ax, _COMM_POS, _COMM_NODE_SIZE, parts, _OVERLAP_COLORS[choice])
    _community_legend(ax, [(_name_set(p), [c]) for p, c in
                           zip(parts, _OVERLAP_COLORS[choice])])
    alts = {c: _modularity(G, p) for c, p in _OVERLAP_PARTS.items()}
    best = max(alts.values())
    ties = sorted(c for c, v in alts.items() if v == q and c != choice)
    tie_note = (f" This division scores the same as the setting *{ties[0]}* "
                f"(${_frac(q)}$ either way), which means the two are equally poor "
                f"descriptions, not that they are the same description."
                if ties else "")
    where = {v: i for i, p in enumerate(parts) for v in p}
    cut = sorted(f"D–{u}" if u > "D" else f"{u}–D" for u in G["D"] if where[u] != where["D"])
    cut_text = ", ".join(cut[:-1]) + " and " + cut[-1] if len(cut) > 1 else cut[0]
    if choice == "force D into the left group":
        cost = (f"What does forcing D into one group cost? With D on the left, {cut_text} run between the "
                f"groups and count against the division. The score takes a penalty for a fact "
                f"about the network, not for a fault in the answer. Merging the two groups "
                f"instead would score $Q = {_frac(merged)}$ and give up the division.")
    elif choice == "force D into the right group":
        # No "only ... and this forcing scores best": fewer cut edges is not why a division
        # scores best (10.1 q_39's best division cuts MORE of X's edges), so the caption
        # does not pair the two.
        cost = (f"With D on the right, {cut_text} run between the groups and count against "
                f"the division, and D sits in one place only. Merging the groups instead would "
                f"score $Q = {_frac(merged)}$.")
    else:
        cost = (f"With D on its own, all of D's edges ({cut_text}) run between groups, and "
                f"each of them counts against the division. Merging everything into one group would score "
                f"$Q = {_frac(merged)}$, the score of no division at all.")
    return (
        f"{choice[0].upper() + choice[1:]}. Since each node now has one group, $Q$ "
        f"is defined again, and with $m = {m}$ edges this division scores "
        f"{_fmt_q(q)}."
        + (" This division is the best of the three forced answers." if q == best else
           f" Of the three forced answers, the best scores ${_frac(best)}$.")
        + f"{tie_note}\n\n{cost}")
