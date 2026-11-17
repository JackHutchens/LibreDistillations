"""
Tree-of-Knowledge Pushdown Remediation Automaton & Learner KV Evolution Engine.

100% pure Python standard library implementation compatible with both desktop
JupyterLab and browser Pyodide/JupyterLite WASM runtimes.

Key capabilities:
1. LearnerKVEvolutionEngine:
   - Tracks per-learner mastery vector theta_t in [0, 1]^K, epistemic uncertainty
     sigma_t in (0, 1]^K, and memory stability S_k (in hours) with spacing decay
     R_k(dt) = exp(-dt / S_k) (Cepeda2008).
   - Evaluates Certainty of Response Index (CRI in [0, 5], HasanBagayokoKelley1999)
     and flags high-confidence errors for hypercorrection (ButterfieldMetcalfe2001).
   - Executes Graph-Laplacian Relational Propagation (theta_prior_propagation)
     across prerequisite edges of G_course with weak-bridge attenuation
     plus online EWMA temporal co-activation Psi_evol.
   - Maintains an append-only SHA-256 hash-chained telemetry trajectory and emits
     HMAC-SHA-256 sealed session receipts.
     Explicitly tags receipts with
     receipt_security_tier="CLIENT_SIDE_INTEGRITY_SEAL_NOT_ADVERSARIAL_PROOF":
     in an open-source client-side runtime (JupyterLite Pyodide WASM / local
     JupyterLab), client-side HMAC detects accidental corruption and casual
     plaintext edits, whereas verified summative credit requires server/CI
     replay.
2. MisconceptionClassifier:
   - Classifies single_select / multi_select distractors, numeric formulaic traps
     (e.g., missing 1/2 factor, unit mismatch, Forward Euler vs. Velocity Verlet),
     and code_exercise conceptual_gap_tags into tractable remediation edges or
     untractable slips.
3. TreeOfKnowledgeAutomaton:
   - Pushdown remediation automaton (S0_ROOT_ANCHOR -> S1_DIAGNOSTIC_PROBE ->
     S2_REMEDIAL_CLUSTER -> S0_REFRESHED_ANCHOR_RETURN -> MASTERED_COMPLETED,
     with bounded terminal escalation to S3_DIAGNOSTIC_ESCALATION_DEFERRED when
     anchor_return_failures >= max_anchor_return_attempts at D_max)
     with bounded stack depth D_max in {1, 2}, single zero-navigation slip nudge,
     POE-gated visualizer state (dt >= 1.5 s across >= 2 regimes),
     progressive 3-tier hinting, and deterministic isomorphic question refresh.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import hmac
import json
import math
import operator
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# 1. Safe AST Arithmetic Expression Evaluator (for Parametric & Trap Rules)
# ---------------------------------------------------------------------------

_ALLOWED_BIN_OPS: Dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARY_OPS: Dict[type, Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_ALLOWED_MATH_FUNCS: Dict[str, Callable[..., float]] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
}

_DEFAULT_CONSTANTS: Dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "ev_to_j": 1.602176634e-19,
    "j_to_ev": 1.0 / 1.602176634e-19,
    "kb_ev": 8.617333262e-5,
}


def safe_eval_arithmetic_expr(
    expr: str,
    variables: Optional[Mapping[str, float]] = None,
) -> float:
    """
    Deterministically evaluate a pure arithmetic formula string over numeric
    variables without using Python eval(). Raises ValueError on unsafe nodes.
    """
    env: Dict[str, float] = dict(_DEFAULT_CONSTANTS)
    if variables:
        for k, v in variables.items():
            env[str(k)] = float(v)

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid arithmetic expression syntax: {expr!r}") from exc

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in env:
                return float(env[node.id])
            raise ValueError(f"Unknown variable {node.id!r} in expression {expr!r}")
        if isinstance(node, ast.BinOp):
            op_fn = _ALLOWED_BIN_OPS.get(type(node.op))
            if op_fn is None:
                raise ValueError(f"Disallowed binary operator {type(node.op).__name__}")
            return float(op_fn(_eval_node(node.left), _eval_node(node.right)))
        if isinstance(node, ast.UnaryOp):
            u_fn = _ALLOWED_UNARY_OPS.get(type(node.op))
            if u_fn is None:
                raise ValueError(f"Disallowed unary operator {type(node.op).__name__}")
            return float(u_fn(_eval_node(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = _ALLOWED_MATH_FUNCS.get(node.func.id)
            if fn is None:
                raise ValueError(f"Disallowed function call {node.func.id!r}")
            args = [_eval_node(arg) for arg in node.args]
            return float(fn(*args))
        raise ValueError(f"Disallowed AST node {type(node).__name__} in {expr!r}")

    return _eval_node(tree)


# ---------------------------------------------------------------------------
# 2. Certainty of Response Index (CRI, HasanBagayokoKelley1999)
# ---------------------------------------------------------------------------

class CRIQuadrant(str, Enum):
    """Four epistemic quadrants of the Certainty of Response Index (0..5 scale)."""

    KNOWLEDGE_MASTERY = "KNOWLEDGE_MASTERY"              # Correct + High CRI (>= 2.5)
    LUCKY_GUESS_OR_LACUNA = "LUCKY_GUESS_OR_LACUNA"      # Correct + Low CRI (< 2.5)
    LACK_OF_KNOWLEDGE = "LACK_OF_KNOWLEDGE"              # Incorrect + Low CRI (< 2.5)
    ENTRENCHED_MISCONCEPTION = "ENTRENCHED_MISCONCEPTION"  # Incorrect + High CRI (>= 2.5)


@dataclass(frozen=True)
class CRIEvaluation:
    raw_cri: float
    normalized_confidence: float
    is_correct: bool
    quadrant: CRIQuadrant
    overconfidence_bias: float
    hypercorrection_eligible: bool


def evaluate_cri_response(is_correct: bool, cri_score: float = 3.0) -> CRIEvaluation:
    """
    Classify a learner response and Certainty of Response Index (CRI in [0, 5])
    into the canonical Hasan-Bagayoko-Kelley (1999) matrix and compute
    Overconfidence Bias (OCB = confidence - accuracy) and Butterfield-Metcalfe
    (2001) hypercorrection eligibility.
    """
    clamped_cri = max(0.0, min(5.0, float(cri_score)))
    norm_conf = round(clamped_cri / 5.0, 6)
    high_conf = norm_conf >= 0.50

    if is_correct and high_conf:
        quadrant = CRIQuadrant.KNOWLEDGE_MASTERY
    elif is_correct and not high_conf:
        quadrant = CRIQuadrant.LUCKY_GUESS_OR_LACUNA
    elif (not is_correct) and (not high_conf):
        quadrant = CRIQuadrant.LACK_OF_KNOWLEDGE
    else:
        quadrant = CRIQuadrant.ENTRENCHED_MISCONCEPTION

    y_num = 1.0 if is_correct else 0.0
    ocb = round(norm_conf - y_num, 6)
    return CRIEvaluation(
        raw_cri=clamped_cri,
        normalized_confidence=norm_conf,
        is_correct=bool(is_correct),
        quadrant=quadrant,
        overconfidence_bias=ocb,
        hypercorrection_eligible=(quadrant == CRIQuadrant.ENTRENCHED_MISCONCEPTION),
    )


# ---------------------------------------------------------------------------
# 3. Immutable Telemetry Trajectory & SHA-256 Receipt Seal (Gap 6 & Gap 7 Fix)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImmutableTelemetryEvent:
    seq: int
    timestamp_unix: float
    event_type: str
    item_id: str
    automaton_state: str
    latency_sec: float
    cri: float
    hints_used: Tuple[int, ...]
    theta_snapshot: Dict[str, float]
    payload: Dict[str, Any]
    prev_event_sha256: str
    event_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.seq,
            "timestamp_unix": round(self.timestamp_unix, 6),
            "event_type": self.event_type,
            "item_id": self.item_id,
            "automaton_state": self.automaton_state,
            "latency_sec": round(self.latency_sec, 4),
            "cri": round(self.cri, 3),
            "hints_used": list(self.hints_used),
            "theta_snapshot": {k: round(v, 6) for k, v in sorted(self.theta_snapshot.items())},
            "payload": self.payload,
            "prev_event_sha256": self.prev_event_sha256,
            "event_sha256": self.event_sha256,
        }


def compute_canonical_json_sha256(data: Mapping[str, Any]) -> str:
    canonical_bytes = json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical_bytes).hexdigest()


# ---------------------------------------------------------------------------
# 4. Learner KV State & Evolution Engine (Cepeda2008 + Graph-Laplacian + EWMA)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PrerequisiteEdgeSpec:
    source_concept_id: str
    target_concept_id: str
    weight: float = 1.0
    is_weak_bridge: bool = False


@dataclass
class LearnerKVState:
    learner_id: str
    session_id: str
    theta: Dict[str, float] = field(default_factory=dict)
    sigma: Dict[str, float] = field(default_factory=dict)
    stability_hours: Dict[str, float] = field(default_factory=dict)
    last_updated_unix: Dict[str, float] = field(default_factory=dict)
    psi_evol: Dict[str, float] = field(default_factory=dict)

    def ensure_concept(
        self,
        concept_id: str,
        initial_theta: float = 0.50,
        initial_sigma: float = 0.35,
        initial_stability_hours: float = 24.0,
        timestamp_unix: float = 0.0,
    ) -> None:
        if concept_id not in self.theta:
            self.theta[concept_id] = max(0.0, min(1.0, float(initial_theta)))
        if concept_id not in self.sigma:
            self.sigma[concept_id] = max(0.02, min(0.50, float(initial_sigma)))
        if concept_id not in self.stability_hours:
            self.stability_hours[concept_id] = max(1.0, float(initial_stability_hours))
        if concept_id not in self.last_updated_unix:
            self.last_updated_unix[concept_id] = float(timestamp_unix)
        if concept_id not in self.psi_evol:
            self.psi_evol[concept_id] = max(0.0, min(1.0, float(initial_theta)))


PARAMETER_RATIONALE_TABLE: Dict[str, Dict[str, Any]] = {
    "beta_prior_net_evol": {
        "values": {"beta_prior": 0.45, "beta_net": 0.35, "beta_evol": 0.20},
        "sum": 1.00,
        "basis": (
            "Convex combination weights calibrated for hierarchical concept mastery propagation: "
            "direct concept evidence (0.45) dominates 1-hop relational smoothing (0.35) "
            "and online EWMA co-activation (0.20)."
        ),
    },
    "mu_laplacian": {
        "value": 0.25,
        "basis": (
            "Damped random-walk Graph-Laplacian diffusion step mu in (0, 0.50): "
            "retains (1 - mu) = 0.75 self-weight per step, guaranteeing aperiodicity "
            "and the discrete maximum principle min_j theta_j <= theta_i' <= max_j theta_j."
        ),
    },
    "w_weak_threshold": {
        "value": 0.40,
        "basis": (
            "Cross-concept prerequisite edge weight cutoff below which an edge is "
            "treated as a weak bridge subject to over-smoothing attenuation."
        ),
    },
    "w_eff": {
        "value": 0.15,
        "basis": (
            "Weak-bridge multiplicative attenuation factor (w_eff = w * 0.15 when "
            "is_weak_bridge=True or w < 0.40), bounding single-step cross-bridge "
            "leakage below 3.75% of mu_laplacian."
        ),
    },
    "denom": {
        "value": "max(1.0, w_sum)",
        "basis": (
            "Confidence-regularized sub-stochastic Laplacian denominator with "
            "self-retention (denom - w_sum): prevents degree-1 weak bridges from "
            "normalizing w_eff / w_eff = 1.0 and erasing bridge attenuation."
        ),
    },
    "stability_multiplier": {
        "formula": "1.0 + 0.65 * g_space * cri_mult",
        "g_space_spaced": "1.0 + 1.40 * (1.0 - R_k) for dt >= 0.25 h",
        "g_space_massed": "0.18 for dt < 0.25 h (15 min massed threshold)",
        "min_positive_multiplier": 1.06435,
        "basis": (
            "Cepeda et al. (2008) spacing effect: g_space is the additive spacing "
            "gain inside stability_multiplier = 1.0 + 0.65 * g_space * cri_mult >= 1.06435 > 1.0 "
            "(never >= 1.0585 lower bound). Every correct answer strictly increases "
            "memory stability S_k' = S_k * stability_multiplier > S_k, while massed "
            "practice (dt < 0.25 h, g_space = 0.18) yields a smaller positive gain "
            "(+6.4% to +11.7%) than spaced retrieval (+35.8% to +156.0%)."
        ),
    },
}


class LearnerKVEvolutionEngine:
    """
    Tracks and evolves per-learner mastery vector theta_t in [0,1]^K, epistemic
    uncertainty sigma_t, memory stability S_k with Cepeda (2008) spacing decay
    R_k(dt) = exp(-dt / S_k), CRI epistemic weighting, and Graph-Laplacian
    relational propagation across G_course + online EWMA co-activation Psi_evol.
    """

    PARAMETER_RATIONALE_TABLE: Dict[str, Dict[str, Any]] = PARAMETER_RATIONALE_TABLE
    BETA_PRIOR: float = 0.45
    BETA_NET: float = 0.35
    BETA_EVOL: float = 0.20
    LAPLACIAN_MU: float = 0.25
    WEAK_BRIDGE_ATTENUATION: float = 0.15
    WEAK_BRIDGE_WEIGHT_CUTOFF: float = 0.40
    EWMA_ALPHA: float = 0.25
    MIN_STABILITY_HOURS: float = 1.0
    MAX_STABILITY_HOURS: float = 2400.0
    THETA_FLOOR: float = 0.05

    def __init__(
        self,
        concept_ids: Sequence[str],
        prerequisite_edges: Optional[Sequence[PrerequisiteEdgeSpec]] = None,
        learner_id: str = "anonymous_learner",
        session_id: str = "session_default",
        initial_theta: float = 0.50,
        initial_sigma: float = 0.35,
        initial_stability_hours: float = 24.0,
        initial_timestamp_unix: float = 1700000000.0,
        hmac_secret: str = "libredistillations_v2_integrity_key",
    ) -> None:
        self.state = LearnerKVState(learner_id=learner_id, session_id=session_id)
        for cid in concept_ids:
            self.state.ensure_concept(
                cid,
                initial_theta=initial_theta,
                initial_sigma=initial_sigma,
                initial_stability_hours=initial_stability_hours,
                timestamp_unix=initial_timestamp_unix,
            )
        self.edges: List[PrerequisiteEdgeSpec] = list(prerequisite_edges or [])
        for edge in self.edges:
            self.state.ensure_concept(edge.source_concept_id, timestamp_unix=initial_timestamp_unix)
            self.state.ensure_concept(edge.target_concept_id, timestamp_unix=initial_timestamp_unix)
        self.hmac_secret = hmac_secret
        self.event_log: List[ImmutableTelemetryEvent] = []
        self.cri_evaluations: List[CRIEvaluation] = []

    def compute_retrievability(self, concept_id: str, dt_hours: float) -> float:
        """
        Compute Cepeda et al. (2008) exponential spacing retrievability:
            R_k(dt) = exp(-dt / S_k) in (0, 1].
        """
        self.state.ensure_concept(concept_id)
        dt = max(0.0, float(dt_hours))
        s_k = max(self.MIN_STABILITY_HOURS, float(self.state.stability_hours[concept_id]))
        return math.exp(-dt / s_k)

    def apply_spacing_decay(
        self,
        current_timestamp_unix: float,
        mutate_state: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Apply spacing decay R_k(dt) = exp(-dt / S_k) across all tracked concepts
        for elapsed time dt since each concept's last interaction.
        """
        report: Dict[str, Dict[str, float]] = {}
        for cid, theta_val in list(self.state.theta.items()):
            last_t = self.state.last_updated_unix.get(cid, current_timestamp_unix)
            dt_hours = max(0.0, (float(current_timestamp_unix) - float(last_t)) / 3600.0)
            r_k = self.compute_retrievability(cid, dt_hours)
            decayed_theta = self.THETA_FLOOR + (theta_val - self.THETA_FLOOR) * r_k if theta_val > self.THETA_FLOOR else theta_val * r_k
            decayed_theta = max(0.0, min(1.0, decayed_theta))
            cur_sigma = self.state.sigma.get(cid, 0.35)
            decayed_sigma = min(0.50, cur_sigma + 0.20 * (1.0 - r_k))
            if mutate_state and dt_hours > 0.0:
                self.state.theta[cid] = round(decayed_theta, 6)
                self.state.sigma[cid] = round(decayed_sigma, 6)
                self.state.last_updated_unix[cid] = float(current_timestamp_unix)
            report[cid] = {
                "dt_hours": round(dt_hours, 6),
                "retrievability": round(r_k, 6),
                "theta_before": round(theta_val, 6),
                "theta_after_decay": round(decayed_theta, 6),
                "sigma_after_decay": round(decayed_sigma, 6),
            }
        return report

    def select_spaced_warmup_concepts(
        self,
        current_timestamp_unix: float,
        max_items: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Rank concepts by retention urgency (1 - R_k(dt) * theta_k) to populate
        the 4-6 item Spaced Warmup queue on Worksheet.load.
        """
        decay_snapshot = self.apply_spacing_decay(current_timestamp_unix, mutate_state=False)
        scored: List[Tuple[str, float]] = []
        for cid, metrics in decay_snapshot.items():
            retention_product = metrics["retrievability"] * metrics["theta_before"]
            urgency = round(1.0 - retention_product, 6)
            scored.append((cid, urgency))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[: max(1, int(max_items))]

    def record_ewma_coactivation(
        self,
        activated_concepts: Sequence[str],
        positive_signal: bool = True,
    ) -> Dict[str, float]:
        """
        Update Online EWMA Temporal Co-Activation vector Psi_evol(k, t).
        """
        active_set: Set[str] = set(activated_concepts)
        for cid in active_set:
            self.state.ensure_concept(cid)
        alpha = self.EWMA_ALPHA
        target_val = 1.0 if positive_signal else 0.15
        for cid in list(self.state.psi_evol.keys()):
            cur = self.state.psi_evol[cid]
            if cid in active_set:
                self.state.psi_evol[cid] = round((1.0 - alpha) * cur + alpha * target_val, 6)
            else:
                # Gentle relaxation toward current mastery prior
                self.state.psi_evol[cid] = round((1.0 - 0.05 * alpha) * cur + (0.05 * alpha) * self.state.theta[cid], 6)
        return dict(self.state.psi_evol)

    def theta_prior_propagation(
        self,
        steps: int = 2,
        include_ewma: bool = True,
        mutate_state: bool = True,
    ) -> Dict[str, float]:
        """
        Execute Graph-Laplacian Relational Propagation across prerequisite edges
        of G_course:
            theta_lap = (I - mu * L_rw)^steps theta_t
        with weak-bridge attenuation and prerequisite readiness
        coupling, blended with online EWMA co-activation Psi_evol.
        """
        nodes = sorted(self.state.theta.keys())
        if not nodes:
            return {}

        # Build effective symmetric/directed neighbor weights with weak-bridge attenuation
        adj: Dict[str, List[Tuple[str, float, str]]] = {u: [] for u in nodes}
        for edge in self.edges:
            u = edge.source_concept_id
            v = edge.target_concept_id
            if u not in adj or v not in adj:
                continue
            w = max(0.0, min(1.0, float(edge.weight)))
            if edge.is_weak_bridge or w < self.WEAK_BRIDGE_WEIGHT_CUTOFF:
                w_eff = w * self.WEAK_BRIDGE_ATTENUATION
            else:
                w_eff = w
            if w_eff > 0.0:
                # Forward edge: prerequisite u -> downstream v
                adj[v].append((u, w_eff, "prereq_in"))
                # Backward diagnostic evidence edge: v -> u (attenuated by 0.65)
                adj[u].append((v, w_eff * 0.65, "downstream_out"))

        phi: Dict[str, float] = {u: float(self.state.theta[u]) for u in nodes}
        mu = self.LAPLACIAN_MU

        for _ in range(max(1, int(steps))):
            nxt: Dict[str, float] = {}
            for u in nodes:
                nbrs = adj.get(u, [])
                if not nbrs:
                    nxt[u] = phi[u]
                    continue
                w_sum = sum(w for _, w, _ in nbrs)
                if w_sum <= 1e-12:
                    nxt[u] = phi[u]
                    continue
                # Confidence-regularized Laplacian denominator max(1.0, w_sum)
                # preserves weak-bridge attenuation even on degree-1 leaf nodes 
                denom = max(1.0, w_sum)
                self_retention = denom - w_sum
                nbr_avg = (self_retention * phi[u] + sum(w * phi[nb] for nb, w, _ in nbrs)) / denom

                # Prerequisite ceiling readiness effect: if a direct strong prerequisite
                # has significantly lower mastery than downstream node u, pull u's prior down
                prereq_vals = [
                    phi[nb]
                    for nb, w, direction in nbrs
                    if direction == "prereq_in" and w >= self.WEAK_BRIDGE_WEIGHT_CUTOFF
                ]
                if prereq_vals:
                    min_prereq = min(prereq_vals)
                    if min_prereq < phi[u]:
                        nbr_avg = 0.65 * nbr_avg + 0.35 * min_prereq

                # Damped Laplacian step: (1 - mu)*phi[u] + mu*nbr_avg
                nxt[u] = (1.0 - mu) * phi[u] + mu * nbr_avg
            phi = nxt

        propagated: Dict[str, float] = {}
        for u in nodes:
            p_self = self.state.theta[u]
            p_net = phi[u]
            p_evol = self.state.psi_evol.get(u, p_self)
            if include_ewma:
                blended = (
                    self.BETA_PRIOR * p_self
                    + self.BETA_NET * p_net
                    + self.BETA_EVOL * p_evol
                )
            else:
                blended = (self.BETA_PRIOR + self.BETA_EVOL) * p_self + self.BETA_NET * p_net
            propagated[u] = round(max(0.0, min(1.0, blended)), 6)

        if mutate_state:
            for u, val in propagated.items():
                self.state.theta[u] = val
        return propagated

    @staticmethod
    def compute_stability_multiplier(
        dt_hours: float,
        r_k: float,
        is_correct: bool,
        cri_mult: float = 1.0,
    ) -> Tuple[float, float]:
        """
        Return (g_space, stability_multiplier) where S_k' = S_k * stability_multiplier.
        On any correct answer (is_correct=True):
          - g_space = 1.0 + 1.40 * (1.0 - R_k) for spaced retrieval (dt >= 0.25 h)
          - g_space = 0.18 for massed practice (dt < 0.25 h)
          - stability_multiplier = 1.0 + 0.65 * g_space * cri_mult >= 1.06435 >= 1.0585 > 1.0
        Thus massed practice yields a smaller POSITIVE stability expansion multiplier
        (> 1.0), never shrinking stability S_k.
        """
        if not is_correct:
            return 0.0, 0.55
        g_space = 1.0 + 1.40 * (1.0 - float(r_k)) if float(dt_hours) >= 0.25 else 0.18
        stability_multiplier = 1.0 + 0.65 * g_space * float(cri_mult)
        return round(g_space, 6), round(stability_multiplier, 6)

    def update_on_observation(
        self,
        concept_id: str,
        is_correct: bool,
        cri_score: float = 3.0,
        timestamp_unix: Optional[float] = None,
        coactivated_concepts: Optional[Sequence[str]] = None,
        propagate_laplacian: bool = True,
    ) -> Dict[str, Any]:
        """
        Update mastery theta_t(k), epistemic uncertainty sigma_t(k), and memory
        stability S_k on a graded observation, then optionally propagate across
        G_course via Graph-Laplacian diffusion.
        """
        self.state.ensure_concept(concept_id)
        now = float(timestamp_unix) if timestamp_unix is not None else (
            self.state.last_updated_unix.get(concept_id, 1700000000.0) + 60.0
        )
        last_t = self.state.last_updated_unix.get(concept_id, now)
        dt_hours = max(0.0, (now - last_t) / 3600.0)
        r_k = self.compute_retrievability(concept_id, dt_hours)

        cri_eval = evaluate_cri_response(is_correct=is_correct, cri_score=cri_score)
        self.cri_evaluations.append(cri_eval)

        theta_prev = self.state.theta[concept_id]
        sigma_prev = self.state.sigma[concept_id]
        s_prev = self.state.stability_hours[concept_id]

        if is_correct:
            cri_mult = 1.0 if cri_eval.quadrant == CRIQuadrant.KNOWLEDGE_MASTERY else 0.55
            learning_rate = (0.28 + 0.45 * sigma_prev) * cri_mult
            theta_post = theta_prev + learning_rate * (1.0 - theta_prev)
            if theta_prev < 1.0:
                theta_post = max(theta_prev + 1e-4, theta_post)
            sigma_post = max(0.02, sigma_prev * (0.78 if cri_mult == 1.0 else 0.90))

            # Cepeda (2008) spacing gain: g_space = 0.18 for dt < 0.25h vs 1.0 + 1.40*(1 - R_k)
            # stability_multiplier = 1.0 + 0.65 * g_space * cri_mult >= 1.06435 > 1.0
            g_space, stability_multiplier = self.compute_stability_multiplier(
                dt_hours=dt_hours,
                r_k=r_k,
                is_correct=True,
                cri_mult=cri_mult,
            )
            s_post = min(
                self.MAX_STABILITY_HOURS,
                s_prev * stability_multiplier,
            )
        else:
            # Entrenched misconception (high CRI error) applies stronger prior revision
            penalty_mult = 1.25 if cri_eval.quadrant == CRIQuadrant.ENTRENCHED_MISCONCEPTION else 0.90
            drop_rate = min(0.65, (0.24 + 0.35 * sigma_prev) * penalty_mult)
            theta_post = theta_prev * (1.0 - drop_rate)
            if theta_prev > 0.0:
                theta_post = min(theta_prev - 1e-4, theta_post)
            sigma_post = min(0.50, sigma_prev * 1.15 + 0.03)
            g_space, stability_multiplier = self.compute_stability_multiplier(
                dt_hours=dt_hours,
                r_k=r_k,
                is_correct=False,
                cri_mult=0.0,
            )
            s_post = max(self.MIN_STABILITY_HOURS, s_prev * stability_multiplier)

        self.state.theta[concept_id] = round(max(0.0, min(1.0, theta_post)), 6)
        self.state.sigma[concept_id] = round(max(0.02, min(0.50, sigma_post)), 6)
        self.state.stability_hours[concept_id] = round(s_post, 6)
        self.state.last_updated_unix[concept_id] = now

        active_list = [concept_id]
        if coactivated_concepts:
            for c_extra in coactivated_concepts:
                if c_extra not in active_list:
                    active_list.append(c_extra)
        self.record_ewma_coactivation(active_list, positive_signal=is_correct)

        direct_theta = self.state.theta[concept_id]
        if propagate_laplacian and self.edges:
            self.theta_prior_propagation(steps=1, include_ewma=True, mutate_state=True)
            # Preserve strict monotonicity on the directly observed concept
            if is_correct and theta_prev < 1.0:
                self.state.theta[concept_id] = round(max(direct_theta, self.state.theta[concept_id]), 6)
            elif (not is_correct) and theta_prev > 0.0:
                self.state.theta[concept_id] = round(min(direct_theta, self.state.theta[concept_id]), 6)

        return {
            "concept_id": concept_id,
            "is_correct": is_correct,
            "theta_before": round(theta_prev, 6),
            "theta_after": self.state.theta[concept_id],
            "sigma_before": round(sigma_prev, 6),
            "sigma_after": self.state.sigma[concept_id],
            "stability_before": round(s_prev, 6),
            "stability_after": self.state.stability_hours[concept_id],
            "g_space": g_space,
            "stability_multiplier": stability_multiplier,
            "retrievability_at_obs": round(r_k, 6),
            "cri_quadrant": cri_eval.quadrant.value,
            "overconfidence_bias": cri_eval.overconfidence_bias,
            "hypercorrection_eligible": cri_eval.hypercorrection_eligible,
        }

    def append_telemetry_event(
        self,
        event_type: str,
        item_id: str,
        automaton_state: str,
        latency_sec: float,
        cri: float,
        hints_used: Sequence[int],
        payload: Mapping[str, Any],
        timestamp_unix: Optional[float] = None,
    ) -> ImmutableTelemetryEvent:
        """
        Append an immutable, SHA-256 hash-chained telemetry event.
        """
        seq = len(self.event_log)
        prev_hash = self.event_log[-1].event_sha256 if self.event_log else ("0" * 64)
        ts = float(timestamp_unix) if timestamp_unix is not None else (1700000000.0 + seq * 15.0)
        theta_snap = {k: round(v, 6) for k, v in sorted(self.state.theta.items())}
        hints_tuple = tuple(int(h) for h in hints_used)
        clean_payload = dict(payload)

        unsigned_body = {
            "seq": seq,
            "timestamp_unix": round(ts, 6),
            "event_type": str(event_type),
            "item_id": str(item_id),
            "automaton_state": str(automaton_state),
            "latency_sec": round(float(latency_sec), 4),
            "cri": round(float(cri), 3),
            "hints_used": list(hints_tuple),
            "theta_snapshot": theta_snap,
            "payload": clean_payload,
            "prev_event_sha256": prev_hash,
        }
        event_hash = compute_canonical_json_sha256(unsigned_body)
        ev = ImmutableTelemetryEvent(
            seq=seq,
            timestamp_unix=round(ts, 6),
            event_type=str(event_type),
            item_id=str(item_id),
            automaton_state=str(automaton_state),
            latency_sec=round(float(latency_sec), 4),
            cri=round(float(cri), 3),
            hints_used=hints_tuple,
            theta_snapshot=theta_snap,
            payload=clean_payload,
            prev_event_sha256=prev_hash,
            event_sha256=event_hash,
        )
        self.event_log.append(ev)
        return ev

    def finalize_session_receipt(
        self,
        worksheet_id: str,
        completed_problems: int,
        total_problems: int,
        recorded_credit: float,
    ) -> Dict[str, Any]:
        """
        Generate an HMAC-SHA-256 integrity-sealed session receipt.

        Open-Source Client-Side Threat Model:
        Because this runtime executes inside client-side JupyterLite (Pyodide
        WASM) or local JupyterLab, the session HMAC secret resides in client
        memory. Accordingly, the receipt explicitly declares
        receipt_security_tier="CLIENT_SIDE_INTEGRITY_SEAL_NOT_ADVERSARIAL_PROOF":
        the SHA-256 event hash chain and HMAC-SHA-256 seal detect accidental file
        corruption and deter casual plaintext JSON edits during self-study, while
        verified summative grading requires server/CI replay of the event log.
        """
        chain_head = self.event_log[-1].event_sha256 if self.event_log else ("0" * 64)
        mean_ocb = (
            round(sum(e.overconfidence_bias for e in self.cri_evaluations) / len(self.cri_evaluations), 6)
            if self.cri_evaluations
            else 0.0
        )
        core_payload = {
            "schema_version": "V2-FINALIZED-RECEIPT-2.0",
            "receipt_security_tier": "CLIENT_SIDE_INTEGRITY_SEAL_NOT_ADVERSARIAL_PROOF",
            "worksheet_id": str(worksheet_id),
            "learner_id": self.state.learner_id,
            "session_id": self.state.session_id,
            "completed_problems": int(completed_problems),
            "total_problems": int(total_problems),
            "recorded_credit": round(float(recorded_credit), 4),
            "event_count": len(self.event_log),
            "event_chain_head_sha256": chain_head,
            "final_theta_vector": {k: round(v, 6) for k, v in sorted(self.state.theta.items())},
            "final_stability_hours": {k: round(v, 6) for k, v in sorted(self.state.stability_hours.items())},
            "mean_overconfidence_bias": mean_ocb,
        }
        payload_sha256 = compute_canonical_json_sha256(core_payload)
        key_bytes = f"{self.hmac_secret}:{self.state.session_id}".encode("utf-8")
        hmac_seal = hmac.new(key_bytes, payload_sha256.encode("utf-8"), hashlib.sha256).hexdigest()

        return {
            **core_payload,
            "events": [ev.to_dict() for ev in self.event_log],
            "canonical_payload_sha256": payload_sha256,
            "hmac_sha256_seal": hmac_seal,
        }

    def verify_receipt_integrity(self, receipt: Mapping[str, Any]) -> bool:
        """
        Verify both the event hash chain and the HMAC-SHA-256 integrity seal of
        a finalized session receipt. Returns False if any field is tampered.
        """
        try:
            if (
                str(receipt.get("receipt_security_tier", ""))
                != "CLIENT_SIDE_INTEGRITY_SEAL_NOT_ADVERSARIAL_PROOF"
            ):
                return False
            events_raw = receipt.get("events", [])
            prev_hash = "0" * 64
            for idx, ev in enumerate(events_raw):
                if int(ev.get("seq", -1)) != idx:
                    return False
                if str(ev.get("prev_event_sha256", "")) != prev_hash:
                    return False
                unsigned_body = {
                    "seq": int(ev["seq"]),
                    "timestamp_unix": round(float(ev["timestamp_unix"]), 6),
                    "event_type": str(ev["event_type"]),
                    "item_id": str(ev["item_id"]),
                    "automaton_state": str(ev["automaton_state"]),
                    "latency_sec": round(float(ev["latency_sec"]), 4),
                    "cri": round(float(ev["cri"]), 3),
                    "hints_used": [int(h) for h in ev["hints_used"]],
                    "theta_snapshot": {k: round(float(v), 6) for k, v in sorted(ev["theta_snapshot"].items())},
                    "payload": dict(ev["payload"]),
                    "prev_event_sha256": str(ev["prev_event_sha256"]),
                }
                expected_ev_hash = compute_canonical_json_sha256(unsigned_body)
                if expected_ev_hash != str(ev.get("event_sha256", "")):
                    return False
                prev_hash = expected_ev_hash

            if str(receipt.get("event_chain_head_sha256", "")) != prev_hash:
                return False
            if int(receipt.get("event_count", -1)) != len(events_raw):
                return False

            core_payload = {
                "schema_version": str(receipt["schema_version"]),
                "receipt_security_tier": str(receipt["receipt_security_tier"]),
                "worksheet_id": str(receipt["worksheet_id"]),
                "learner_id": str(receipt["learner_id"]),
                "session_id": str(receipt["session_id"]),
                "completed_problems": int(receipt["completed_problems"]),
                "total_problems": int(receipt["total_problems"]),
                "recorded_credit": round(float(receipt["recorded_credit"]), 4),
                "event_count": int(receipt["event_count"]),
                "event_chain_head_sha256": str(receipt["event_chain_head_sha256"]),
                "final_theta_vector": {
                    k: round(float(v), 6) for k, v in sorted(receipt["final_theta_vector"].items())
                },
                "final_stability_hours": {
                    k: round(float(v), 6) for k, v in sorted(receipt["final_stability_hours"].items())
                },
                "mean_overconfidence_bias": round(float(receipt["mean_overconfidence_bias"]), 6),
            }
            expected_payload_sha = compute_canonical_json_sha256(core_payload)
            if expected_payload_sha != str(receipt.get("canonical_payload_sha256", "")):
                return False

            key_bytes = f"{self.hmac_secret}:{core_payload['session_id']}".encode("utf-8")
            expected_hmac = hmac.new(
                key_bytes,
                expected_payload_sha.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return hmac.compare_digest(expected_hmac, str(receipt.get("hmac_sha256_seal", "")))
        except (KeyError, TypeError, ValueError):
            return False


# ---------------------------------------------------------------------------
# 5. Tractable Misconception & Error-Trap Classifier
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassificationResult:
    is_correct: bool
    tractable: bool
    target_concept_id: Optional[str]
    misconception_id: Optional[str]
    diagnostic_rationale: str
    error_category: str  # "CORRECT", "DISTRACTOR_MISCONCEPTION", "NUMERIC_TRAP", "CODE_CONCEPT_GAP", "UNTRACTABLE_SLIP"


class MisconceptionClassifier:
    """
    Deterministic classifier mapping learner errors across single_select,
    multi_select, numeric, and code_exercise items into tractable conceptual
    misconception edges or untractable slips.
    """

    @staticmethod
    def classify_response(
        item_spec: Mapping[str, Any],
        learner_response: Any,
        active_parameters: Optional[Mapping[str, float]] = None,
    ) -> ClassificationResult:
        qtype = str(item_spec.get("question_type", "single_select"))

        if qtype in ("single_select", "multi_select"):
            return MisconceptionClassifier._classify_choice(item_spec, learner_response)
        if qtype == "numeric":
            return MisconceptionClassifier._classify_numeric(
                item_spec,
                learner_response,
                active_parameters=active_parameters,
            )
        if qtype == "code_exercise":
            return MisconceptionClassifier._classify_code_exercise(item_spec, learner_response)
        raise ValueError(f"Unsupported question_type: {qtype!r}")

    @staticmethod
    def _classify_choice(
        item_spec: Mapping[str, Any],
        learner_response: Any,
    ) -> ClassificationResult:
        correct_raw = item_spec.get("correct")
        if isinstance(correct_raw, (list, tuple, set)):
            correct_set = {str(x).strip() for x in correct_raw}
        else:
            correct_set = {str(correct_raw).strip()}

        if isinstance(learner_response, (list, tuple, set)):
            selected_set = {str(x).strip() for x in learner_response}
        else:
            selected_set = {str(learner_response).strip()}

        if selected_set == correct_set:
            return ClassificationResult(
                is_correct=True,
                tractable=False,
                target_concept_id=item_spec.get("primary_concept_id") or item_spec.get("taught_by"),
                misconception_id=None,
                diagnostic_rationale=str(item_spec.get("rationale_markdown", "Correct.")),
                error_category="CORRECT",
            )

        diag_map: Mapping[str, Any] = item_spec.get("distractor_diagnostics") or {}
        combo_key = ",".join(sorted(selected_set))
        edge = diag_map.get(combo_key)

        if edge is None:
            # Check individual wrongly selected distractors in deterministic order
            wrong_selected = sorted(selected_set - correct_set)
            for opt_id in wrong_selected:
                if opt_id in diag_map:
                    edge = diag_map[opt_id]
                    break

        if edge is None:
            # Check missed required options in multi_select
            missed_correct = sorted(correct_set - selected_set)
            for opt_id in missed_correct:
                miss_key = f"missed_{opt_id}"
                if miss_key in diag_map:
                    edge = diag_map[miss_key]
                    break

        fallback_concept = str(
            item_spec.get("primary_concept_id")
            or item_spec.get("taught_by")
            or "concept_general"
        )
        if edge is not None:
            return ClassificationResult(
                is_correct=False,
                tractable=bool(edge.get("tractable", True)),
                target_concept_id=str(edge.get("target_concept_id", fallback_concept)),
                misconception_id=str(edge.get("misconception_id", f"MISC_{combo_key}")),
                diagnostic_rationale=str(
                    edge.get("diagnostic_rationale", "Diagnostic misconception identified.")
                ),
                error_category="DISTRACTOR_MISCONCEPTION",
            )

        return ClassificationResult(
            is_correct=False,
            tractable=True,
            target_concept_id=fallback_concept,
            misconception_id=f"MISC_UNMAPPED_CHOICE_{combo_key}",
            diagnostic_rationale=str(item_spec.get("hint_markdown", "Review the governing physical principle.")),
            error_category="DISTRACTOR_MISCONCEPTION",
        )

    @staticmethod
    def _classify_numeric(
        item_spec: Mapping[str, Any],
        learner_response: Any,
        active_parameters: Optional[Mapping[str, float]] = None,
    ) -> ClassificationResult:
        try:
            val = float(learner_response)
        except (TypeError, ValueError):
            return ClassificationResult(
                is_correct=False,
                tractable=False,
                target_concept_id=None,
                misconception_id="UNTRACTABLE_SLIP",
                diagnostic_rationale="Check units/order-of-magnitude: non-numeric input.",
                error_category="UNTRACTABLE_SLIP",
            )

        param_spec = item_spec.get("parametric_spec")
        params: Dict[str, float] = {}
        if isinstance(param_spec, Mapping):
            base_p = param_spec.get("base_parameters") or {}
            for k, v in base_p.items():
                params[str(k)] = float(v)
        if active_parameters:
            for k, v in active_parameters.items():
                params[str(k)] = float(v)

        corr_raw = item_spec.get("correct")
        if isinstance(param_spec, Mapping) and param_spec.get("target_expr") and params:
            target = safe_eval_arithmetic_expr(str(param_spec["target_expr"]), params)
            tol = float(param_spec.get("tolerance", 1e-3))
        elif isinstance(corr_raw, Mapping):
            target = float(corr_raw.get("value", corr_raw.get("answer", 0.0)))
            tol = float(corr_raw.get("tolerance", item_spec.get("tolerance", 1e-2)))
        else:
            target = float(corr_raw)
            tol = float(item_spec.get("tolerance", 1e-2))

        if abs(val - target) <= tol:
            return ClassificationResult(
                is_correct=True,
                tractable=False,
                target_concept_id=item_spec.get("primary_concept_id") or item_spec.get("taught_by"),
                misconception_id=None,
                diagnostic_rationale=str(item_spec.get("rationale_markdown", "Correct.")),
                error_category="CORRECT",
            )

        # Evaluate deterministic numeric_trap_rules
        trap_rules: Sequence[Mapping[str, Any]] = item_spec.get("numeric_trap_rules") or []
        eval_vars = dict(params)
        eval_vars["target"] = target

        for rule in trap_rules:
            trap_tol = float(rule.get("tolerance", tol))
            trap_val: Optional[float] = None
            if "formula_expr" in rule and rule["formula_expr"]:
                trap_val = safe_eval_arithmetic_expr(str(rule["formula_expr"]), eval_vars)
            elif "trap_value" in rule and rule["trap_value"] is not None:
                trap_val = float(rule["trap_value"])
            elif "scale_factor" in rule and rule["scale_factor"] is not None:
                trap_val = target * float(rule["scale_factor"])
            elif "additive_offset" in rule and rule["additive_offset"] is not None:
                trap_val = target + float(rule["additive_offset"])

            if trap_val is not None and abs(val - trap_val) <= trap_tol:
                return ClassificationResult(
                    is_correct=False,
                    tractable=bool(rule.get("tractable", True)),
                    target_concept_id=str(
                        rule.get("target_concept_id")
                        or item_spec.get("primary_concept_id")
                        or item_spec.get("taught_by")
                    ),
                    misconception_id=str(rule.get("misconception_id", rule.get("trap_id", "NUMERIC_TRAP"))),
                    diagnostic_rationale=str(
                        rule.get("diagnostic_rationale", "Formulaic physics trap detected.")
                    ),
                    error_category="NUMERIC_TRAP",
                )

        # Out-of-band numeric error -> Untractable Slip
        return ClassificationResult(
            is_correct=False,
            tractable=False,
            target_concept_id=None,
            misconception_id="UNTRACTABLE_SLIP",
            diagnostic_rationale=(
                "Check units/order-of-magnitude or rounding and try again "
                f"(tolerance +/- {tol:g})."
            ),
            error_category="UNTRACTABLE_SLIP",
        )

    @staticmethod
    def _classify_code_exercise(
        item_spec: Mapping[str, Any],
        learner_response: Any,
    ) -> ClassificationResult:
        """
        Accepts a SuiteExecutionReport dict (or conceptual_gap_tag string) emitted
        by the V2 Module 3 hidden unit-test runner and routes coding misconceptions
        into the Tree of Knowledge.
        """
        code_spec: Mapping[str, Any] = item_spec.get("code_exercise_spec") or {}
        routing_map: Mapping[str, Any] = code_spec.get("code_gap_routing") or {}
        fallback_concept = str(
            item_spec.get("primary_concept_id")
            or item_spec.get("taught_by")
            or "concept_code_default"
        )

        if isinstance(learner_response, Mapping):
            all_passed = bool(learner_response.get("passed", False))
            if all_passed:
                return ClassificationResult(
                    is_correct=True,
                    tractable=False,
                    target_concept_id=fallback_concept,
                    misconception_id=None,
                    diagnostic_rationale="All hidden physical invariant unit tests passed.",
                    error_category="CORRECT",
                )
            gap_tag = learner_response.get("conceptual_gap_tag")
            fail_diag = str(
                learner_response.get("failure_diagnostic")
                or "Hidden invariant test failed."
            )
        else:
            gap_tag = str(learner_response) if learner_response else None
            fail_diag = "Code exercise conceptual gap detected."

        if not gap_tag or str(gap_tag) in ("SYNTAX_ERROR", "UNTRACTABLE_CODE_SLIP"):
            return ClassificationResult(
                is_correct=False,
                tractable=False,
                target_concept_id=None,
                misconception_id="UNTRACTABLE_SLIP",
                diagnostic_rationale=f"Check syntax/variable names and try again: {fail_diag}",
                error_category="UNTRACTABLE_SLIP",
            )

        gap_key = str(gap_tag)
        if gap_key in routing_map:
            edge = routing_map[gap_key]
            return ClassificationResult(
                is_correct=False,
                tractable=bool(edge.get("tractable", True)),
                target_concept_id=str(edge.get("target_concept_id", fallback_concept)),
                misconception_id=str(edge.get("misconception_id", gap_key)),
                diagnostic_rationale=str(edge.get("diagnostic_rationale", fail_diag)),
                error_category="CODE_CONCEPT_GAP",
            )

        return ClassificationResult(
            is_correct=False,
            tractable=True,
            target_concept_id=gap_key if gap_key.startswith("c") else fallback_concept,
            misconception_id=gap_key,
            diagnostic_rationale=fail_diag,
            error_category="CODE_CONCEPT_GAP",
        )


# ---------------------------------------------------------------------------
# 6. Tree-of-Knowledge Pushdown Remediation Automaton (S0 -> S1 -> S2 -> S0')
# ---------------------------------------------------------------------------

class AutomatonState(str, Enum):
    S0_ROOT_ANCHOR = "S0_ROOT_ANCHOR"
    S1_DIAGNOSTIC_PROBE = "S1_DIAGNOSTIC_PROBE"
    S2_REMEDIAL_CLUSTER = "S2_REMEDIAL_CLUSTER"
    S0_REFRESHED_ANCHOR_RETURN = "S0_REFRESHED_ANCHOR_RETURN"
    S3_DIAGNOSTIC_ESCALATION_DEFERRED = "S3_DIAGNOSTIC_ESCALATION_DEFERRED"
    MASTERED_COMPLETED = "MASTERED_COMPLETED"


@dataclass(frozen=True)
class ReturnFrame:
    frame_id: str
    origin_state: AutomatonState
    root_item_id: str
    missed_concept_id: str
    misconception_id: str
    seed_bump: int
    depth: int
    diagnostic_probe_id: str
    remedial_cluster_id: str


@dataclass
class POEVisualizerGateState:
    visualizer_id: str
    min_dwell_seconds: float = 1.5
    min_regimes_required: int = 2
    poe_prediction_submitted: bool = False
    poe_selected_option_id: Optional[str] = None
    sliders_unlocked: bool = False
    regime_dwell_log: Dict[str, float] = field(default_factory=dict)
    valid_explored_regimes: Set[str] = field(default_factory=set)
    takeaway_visible: bool = False

    @property
    def is_satisfied(self) -> bool:
        return (
            self.poe_prediction_submitted
            and self.sliders_unlocked
            and len(self.valid_explored_regimes) >= self.min_regimes_required
            and self.takeaway_visible
        )


@dataclass
class ActiveClusterSession:
    cluster_id: str
    target_concept_id: str
    poe_gate: POEVisualizerGateState
    graduated_item_ids: List[str]
    completed_graduated_items: Set[str] = field(default_factory=set)
    item_hint_tiers_unlocked: Dict[str, int] = field(default_factory=dict)
    item_attempts: Dict[str, int] = field(default_factory=dict)
    mastery_threshold: float = 1.0

    @property
    def graduated_progress(self) -> float:
        if not self.graduated_item_ids:
            return 1.0
        return len(self.completed_graduated_items) / float(len(self.graduated_item_ids))

    @property
    def is_mastered(self) -> bool:
        return (
            self.poe_gate.is_satisfied
            and self.graduated_progress >= self.mastery_threshold - 1e-9
        )


@dataclass(frozen=True)
class AutomatonStepOutcome:
    previous_state: AutomatonState
    current_state: AutomatonState
    active_item_id: str
    classification: ClassificationResult
    stack_depth: int
    seed_bump: int
    slip_nudge_emitted: bool
    unlocked_hint_tier: int
    unlocked_hint_text: Optional[str]
    cluster_mastered: bool
    message: str
    agency_criteria_triggered: bool = False
    return_to_worksheet_visible: bool = False
    rabbit_hole_available: bool = False
    pending_rabbit_hole_concept_id: Optional[str] = None


class TreeOfKnowledgeAutomaton:
    """
    Pushdown Remediation Automaton governing adaptive transitions across:
      - S0_ROOT_ANCHOR (active root question q_0)
      - S1_DIAGNOSTIC_PROBE (targeted concept probe q_1(c_m))
      - S2_REMEDIAL_CLUSTER (refutation + faded worked step + POE visualizer +
        graduated mastery items C(c_m))
      - S0_REFRESHED_ANCHOR_RETURN (isomorphically re-seeded root question q_0')
      - S3_DIAGNOSTIC_ESCALATION_DEFERRED (bounded terminal escalation on repeated S0' failure)
      - MASTERED_COMPLETED (root anchor solved)
    Includes learner-agency Rabbit Hole continuation (d = 1 -> 2 -> 3 ... <= D_safety)
    and trigger-gated 'Return to Worksheet' (strictly hidden and refused before agency criteria trigger).
    """

    def __init__(
        self,
        curriculum_spec: Mapping[str, Any],
        kv_engine: LearnerKVEvolutionEngine,
        max_recursion_depth: int = 1,
        untractable_slip_free_nudges: int = 1,
        poe_min_dwell_seconds: float = 1.5,
        poe_min_regimes_required: int = 2,
        max_anchor_return_attempts: int = 2,
        max_in_cluster_item_attempts: int = 3,
        max_rabbit_hole_depth: int = 8,
    ) -> None:
        if max_recursion_depth not in (1, 2):
            raise ValueError("max_recursion_depth D_max must be 1 or 2")
        self.worksheet = curriculum_spec.get("worksheet", curriculum_spec)
        self.kv_engine = kv_engine
        self.max_recursion_depth = int(max_recursion_depth)
        self.max_rabbit_hole_depth = max(self.max_recursion_depth, int(max_rabbit_hole_depth))
        self.untractable_slip_free_nudges = int(untractable_slip_free_nudges)
        self.poe_min_dwell_seconds = float(poe_min_dwell_seconds)
        self.poe_min_regimes_required = int(poe_min_regimes_required)
        self.max_anchor_return_attempts = max(1, int(max_anchor_return_attempts))
        self.max_in_cluster_item_attempts = max(1, int(max_in_cluster_item_attempts))

        self.root_problems: Dict[str, Dict[str, Any]] = {}
        for sec in self.worksheet.get("sections", []):
            if sec.get("type") == "problem":
                self.root_problems[str(sec["id"])] = copy.deepcopy(dict(sec))

        self.probe_pools: Dict[str, List[Dict[str, Any]]] = {}
        for cid, probes in (self.worksheet.get("diagnostic_probe_pools") or {}).items():
            self.probe_pools[str(cid)] = [copy.deepcopy(dict(p)) for p in probes]

        self.remedial_clusters: Dict[str, Dict[str, Any]] = {}
        for cid, cspec in (self.worksheet.get("remedial_clusters") or {}).items():
            self.remedial_clusters[str(cid)] = copy.deepcopy(dict(cspec))

        self.state: AutomatonState = AutomatonState.S0_ROOT_ANCHOR
        self.call_stack: List[ReturnFrame] = []
        self.active_root_id: str = next(iter(self.root_problems.keys()), "")
        self.active_item_spec: Dict[str, Any] = (
            copy.deepcopy(self.root_problems[self.active_root_id])
            if self.active_root_id
            else {}
        )
        self.active_parameters: Dict[str, float] = self._extract_parameters(self.active_item_spec, seed_bump=0)
        self.root_seed_bumps: Dict[str, int] = {pid: 0 for pid in self.root_problems}
        self.anchor_return_failures: Dict[str, int] = {pid: 0 for pid in self.root_problems}
        self.deferred_escalation_queue: List[Dict[str, Any]] = []
        self.untractable_slip_counts: Dict[str, int] = {}
        self.root_hint_tiers: Dict[str, int] = {}
        self.active_cluster_session: Optional[ActiveClusterSession] = None
        self.suspended_cluster_stack: List[Tuple[ActiveClusterSession, Dict[str, Any]]] = []
        self.agency_criteria_triggered: bool = False
        self.pending_rabbit_hole_concept_id: Optional[str] = None
        self.pending_rabbit_hole_misconception_id: Optional[str] = None
        self.visited_concepts_in_chain: Set[str] = set()
        self.mastered_clusters: Set[str] = set()
        self.completed_roots: Dict[str, float] = {}

    @property
    def return_to_worksheet_visible(self) -> bool:
        """
        Strictly False BEFORE learner-agency criteria trigger. Flips to True once agency
        criteria trigger while inside an active remediation stack (len(call_stack) > 0).
        """
        return bool(self.agency_criteria_triggered and len(self.call_stack) > 0)

    @property
    def rabbit_hole_available(self) -> bool:
        """
        True iff learner-agency criteria have triggered, a valid unvisited deeper prerequisite
        cluster is pending, and stack depth is below max_rabbit_hole_depth (D_safety).
        """
        return bool(
            self.agency_criteria_triggered
            and self.pending_rabbit_hole_concept_id is not None
            and len(self.call_stack) < self.max_rabbit_hole_depth
        )

    def _make_step_outcome(
        self,
        previous_state: AutomatonState,
        current_state: AutomatonState,
        active_item_id: str,
        classification: ClassificationResult,
        stack_depth: int,
        seed_bump: int,
        slip_nudge_emitted: bool,
        unlocked_hint_tier: int,
        unlocked_hint_text: Optional[str],
        cluster_mastered: bool,
        message: str,
    ) -> AutomatonStepOutcome:
        return AutomatonStepOutcome(
            previous_state=previous_state,
            current_state=current_state,
            active_item_id=active_item_id,
            classification=classification,
            stack_depth=stack_depth,
            seed_bump=seed_bump,
            slip_nudge_emitted=slip_nudge_emitted,
            unlocked_hint_tier=unlocked_hint_tier,
            unlocked_hint_text=unlocked_hint_text,
            cluster_mastered=cluster_mastered,
            message=message,
            agency_criteria_triggered=self.agency_criteria_triggered,
            return_to_worksheet_visible=self.return_to_worksheet_visible,
            rabbit_hole_available=self.rabbit_hole_available,
            pending_rabbit_hole_concept_id=self.pending_rabbit_hole_concept_id,
        )

    def select_root_question(self, root_item_id: str) -> Dict[str, Any]:
        if root_item_id not in self.root_problems:
            raise KeyError(f"Unknown root problem id: {root_item_id!r}")
        self.call_stack.clear()
        self.suspended_cluster_stack.clear()
        self.active_cluster_session = None
        self.agency_criteria_triggered = False
        self.pending_rabbit_hole_concept_id = None
        self.pending_rabbit_hole_misconception_id = None
        self.visited_concepts_in_chain.clear()
        self.active_root_id = root_item_id
        bump = self.root_seed_bumps.get(root_item_id, 0)
        self.state = (
            AutomatonState.S0_REFRESHED_ANCHOR_RETURN
            if bump > 0
            else AutomatonState.S0_ROOT_ANCHOR
        )
        self.active_item_spec = self.instantiate_refreshed_item(
            self.root_problems[root_item_id],
            seed_bump=bump,
        )
        self.active_parameters = self._extract_parameters(self.active_item_spec, seed_bump=bump)
        return self.active_item_spec

    @staticmethod
    def _extract_parameters(item_spec: Mapping[str, Any], seed_bump: int = 0) -> Dict[str, float]:
        param_spec = item_spec.get("parametric_spec")
        if not isinstance(param_spec, Mapping):
            return {}
        if seed_bump <= 0:
            base = param_spec.get("base_parameters") or {}
            return {str(k): float(v) for k, v in base.items()}
        variants = param_spec.get("reseed_variants") or []
        if variants:
            chosen = variants[(seed_bump - 1) % len(variants)]
            return {str(k): float(v) for k, v in chosen.items()}
        # Fallback deterministic multiplicative perturbation if reseed_variants is empty
        base = param_spec.get("base_parameters") or {}
        factor = 1.0 + 0.25 * float(seed_bump)
        return {str(k): round(float(v) * factor, 6) for k, v in base.items()}

    def instantiate_refreshed_item(
        self,
        base_item: Mapping[str, Any],
        seed_bump: int,
    ) -> Dict[str, Any]:
        """
        Produce an isomorphically refreshed problem specification q_0' for
        S0_REFRESHED_ANCHOR_RETURN:
          - For choice items: deterministically permutes options into a guaranteed
            derangement (different presentation order from seed_bump=0 when len >= 2).
          - For parametric numeric items: selects re-seeded parameters and
            recomputes target value and formatted prompt text.
        """
        refreshed = copy.deepcopy(dict(base_item))
        refreshed["seed_bump"] = int(seed_bump)
        if seed_bump <= 0:
            return refreshed

        # 1. Option permutation with guaranteed derangement for >= 2 options
        options = refreshed.get("options")
        if isinstance(options, list) and len(options) >= 2:
            orig_ids = [str(o.get("id")) for o in options]
            seed_str = f"{self.kv_engine.state.session_id}:{refreshed.get('id')}:bump_{seed_bump}"
            seed_int = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
            rng = random.Random(seed_int)
            permuted = list(options)
            rng.shuffle(permuted)
            if [str(o.get("id")) for o in permuted] == orig_ids:
                # Deterministic cyclic shift guarantees non-identical ordering
                permuted = permuted[1:] + permuted[:1]
            refreshed["options"] = permuted

        # 2. Parametric numeric re-seeding
        param_spec = refreshed.get("parametric_spec")
        if isinstance(param_spec, Mapping):
            new_params = self._extract_parameters(refreshed, seed_bump=seed_bump)
            refreshed["active_parameters"] = new_params
            if param_spec.get("target_expr"):
                new_target = safe_eval_arithmetic_expr(str(param_spec["target_expr"]), new_params)
                tol = float(param_spec.get("tolerance", 1e-3))
                refreshed["correct"] = {"value": round(new_target, 6), "tolerance": tol}
            if param_spec.get("prompt_template"):
                try:
                    refreshed["prompt_markdown"] = str(param_spec["prompt_template"]).format(**new_params)
                except (KeyError, ValueError):
                    pass

        return refreshed

    @staticmethod
    def _get_hint_for_tier(item_spec: Mapping[str, Any], tier: int) -> str:
        tiered = item_spec.get("tiered_hints")
        if isinstance(tiered, Mapping):
            if tier == 1:
                return str(tiered.get("tier_1_conceptual", ""))
            if tier == 2:
                return str(tiered.get("tier_2_structural", ""))
            return str(tiered.get("tier_3_worked_step", ""))
        return str(item_spec.get("hint_markdown", "Review the governing equation."))

    def submit_poe_prediction(
        self,
        selected_option_id: str,
        timestamp_unix: Optional[float] = None,
    ) -> POEVisualizerGateState:
        """
        Record the learner's Predict-Observe-Explain (POE) commitment in an active
        S2_REMEDIAL_CLUSTER, unlocking the interactive visualizer sliders.
        """
        if self.state != AutomatonState.S2_REMEDIAL_CLUSTER or self.active_cluster_session is None:
            raise RuntimeError("POE prediction is only valid inside S2_REMEDIAL_CLUSTER")

        gate = self.active_cluster_session.poe_gate
        gate.poe_prediction_submitted = True
        gate.poe_selected_option_id = str(selected_option_id)
        gate.sliders_unlocked = True

        self.kv_engine.append_telemetry_event(
            event_type="POE_COMMIT",
            item_id=gate.visualizer_id,
            automaton_state=self.state.value,
            latency_sec=1.0,
            cri=3.0,
            hints_used=(),
            payload={
                "cluster_id": self.active_cluster_session.cluster_id,
                "selected_option_id": gate.poe_selected_option_id,
                "sliders_unlocked": gate.sliders_unlocked,
            },
            timestamp_unix=timestamp_unix,
        )
        return gate

    def record_regime_dwell(
        self,
        regime_id: str,
        dwell_seconds: float,
        timestamp_unix: Optional[float] = None,
    ) -> POEVisualizerGateState:
        """
        Record exploration dwell time in a visualizer parameter regime.
        Enforces: sliders must be unlocked via POE prediction first, and
        only regimes with dwell_seconds >= min_dwell_seconds (1.5 s) count toward
        unlocking the takeaway and cluster completion gate.
        """
        if self.state != AutomatonState.S2_REMEDIAL_CLUSTER or self.active_cluster_session is None:
            raise RuntimeError("Regime dwell logging is only valid inside S2_REMEDIAL_CLUSTER")

        gate = self.active_cluster_session.poe_gate
        if not gate.sliders_unlocked:
            raise PermissionError("Cannot explore visualizer regimes before submitting POE prediction")

        dwell = max(0.0, float(dwell_seconds))
        prev_dwell = gate.regime_dwell_log.get(str(regime_id), 0.0)
        new_dwell = round(prev_dwell + dwell, 4)
        gate.regime_dwell_log[str(regime_id)] = new_dwell

        if new_dwell >= gate.min_dwell_seconds:
            gate.valid_explored_regimes.add(str(regime_id))

        if len(gate.valid_explored_regimes) >= gate.min_regimes_required:
            gate.takeaway_visible = True

        self.kv_engine.append_telemetry_event(
            event_type="REGIME_DWELL",
            item_id=gate.visualizer_id,
            automaton_state=self.state.value,
            latency_sec=dwell,
            cri=3.0,
            hints_used=(),
            payload={
                "cluster_id": self.active_cluster_session.cluster_id,
                "regime_id": str(regime_id),
                "dwell_seconds": dwell,
                "cumulative_regime_dwell": new_dwell,
                "valid_regimes_count": len(gate.valid_explored_regimes),
                "takeaway_visible": gate.takeaway_visible,
            },
            timestamp_unix=timestamp_unix,
        )
        return gate

    def _spawn_remedial_cluster_session(self, concept_id: str) -> ActiveClusterSession:
        cluster_spec = self.remedial_clusters.get(concept_id)
        if cluster_spec is None:
            raise KeyError(f"No remedial cluster configured for concept_id={concept_id!r}")
        vis_spec = cluster_spec.get("poe_visualizer_spec") or {}
        poe_spec = vis_spec.get("poe_gate") or {}
        min_dwell = float(poe_spec.get("min_dwell_seconds", self.poe_min_dwell_seconds))
        min_regimes = int(poe_spec.get("min_regimes_required", self.poe_min_regimes_required))

        grad_items = cluster_spec.get("graduated_mastery_items") or []
        grad_ids = [str(item["id"]) for item in grad_items]
        session = ActiveClusterSession(
            cluster_id=str(cluster_spec["cluster_id"]),
            target_concept_id=concept_id,
            poe_gate=POEVisualizerGateState(
                visualizer_id=str(vis_spec.get("id", f"vis_{concept_id}")),
                min_dwell_seconds=min_dwell,
                min_regimes_required=min_regimes,
            ),
            graduated_item_ids=grad_ids,
            mastery_threshold=float(cluster_spec.get("mastery_threshold", 1.0)),
        )
        self.active_cluster_session = session
        if grad_items:
            self.active_item_spec = copy.deepcopy(dict(grad_items[0]))
            self.active_parameters = self._extract_parameters(self.active_item_spec, seed_bump=0)
        return session

    def submit_response(
        self,
        learner_response: Any,
        cri_score: float = 3.0,
        latency_sec: float = 12.0,
        timestamp_unix: Optional[float] = None,
    ) -> AutomatonStepOutcome:
        """
        Process a learner submission in the current automaton state and execute
        the corresponding pushdown transition and KV state update.
        """
        prev_state = self.state
        cur_item_id = str(self.active_item_spec.get("id", "unknown_item"))

        if prev_state in (AutomatonState.S0_ROOT_ANCHOR, AutomatonState.S0_REFRESHED_ANCHOR_RETURN):
            return self._handle_root_submission(
                learner_response=learner_response,
                cri_score=cri_score,
                latency_sec=latency_sec,
                timestamp_unix=timestamp_unix,
            )

        if prev_state == AutomatonState.S1_DIAGNOSTIC_PROBE:
            return self._handle_probe_submission(
                learner_response=learner_response,
                cri_score=cri_score,
                latency_sec=latency_sec,
                timestamp_unix=timestamp_unix,
            )

        if prev_state == AutomatonState.S2_REMEDIAL_CLUSTER:
            return self._handle_cluster_item_submission(
                learner_response=learner_response,
                cri_score=cri_score,
                latency_sec=latency_sec,
                timestamp_unix=timestamp_unix,
            )

        raise RuntimeError(f"Cannot submit response in terminal state {prev_state.value} ({cur_item_id})")

    def continue_rabbit_hole(
        self,
        timestamp_unix: Optional[float] = None,
    ) -> AutomatonStepOutcome:
        """
        Learner-Agency Action 1: Continue Down the Rabbit Hole.
        Pushes a deeper prerequisite remedial cluster (d -> d + 1, up to D_safety)
        when a sub-prerequisite gap was diagnosed at or beyond max_recursion_depth.
        """
        if (
            not self.rabbit_hole_available
            or self.pending_rabbit_hole_concept_id is None
            or self.active_cluster_session is None
            or not self.call_stack
        ):
            raise PermissionError(
                "No deeper prerequisite rabbit-hole cluster is currently available."
            )

        prev_state = self.state
        sub_cid = self.pending_rabbit_hole_concept_id
        misc_id = str(self.pending_rabbit_hole_misconception_id or "NESTED_RABBIT_HOLE_GAP")

        self.suspended_cluster_stack.append(
            (self.active_cluster_session, copy.deepcopy(self.active_item_spec))
        )
        parent_frame = self.call_stack[-1]
        sub_cluster_id = str(self.remedial_clusters[sub_cid]["cluster_id"])
        sub_frame = ReturnFrame(
            frame_id=f"frame_rabbithole_{sub_cid}_{len(self.call_stack) + 1}",
            origin_state=AutomatonState.S2_REMEDIAL_CLUSTER,
            root_item_id=parent_frame.root_item_id,
            missed_concept_id=sub_cid,
            misconception_id=misc_id,
            seed_bump=parent_frame.seed_bump,
            depth=len(self.call_stack) + 1,
            diagnostic_probe_id="NONE",
            remedial_cluster_id=sub_cluster_id,
        )
        self.call_stack.append(sub_frame)
        self.visited_concepts_in_chain.add(sub_cid)
        self.pending_rabbit_hole_concept_id = None
        self.pending_rabbit_hole_misconception_id = None
        self.state = AutomatonState.S2_REMEDIAL_CLUSTER
        sub_session = self._spawn_remedial_cluster_session(sub_cid)

        self.kv_engine.append_telemetry_event(
            event_type="AGENCY_CONTINUE_RABBIT_HOLE",
            item_id=sub_session.cluster_id,
            automaton_state=self.state.value,
            latency_sec=1.0,
            cri=3.0,
            hints_used=(),
            payload={
                "sub_concept_id": sub_cid,
                "sub_cluster_id": sub_session.cluster_id,
                "stack_depth": len(self.call_stack),
                "visited_chain": sorted(self.visited_concepts_in_chain),
            },
            timestamp_unix=timestamp_unix,
        )
        synthetic_classification = ClassificationResult(
            is_correct=False,
            tractable=True,
            target_concept_id=sub_cid,
            misconception_id=misc_id,
            diagnostic_rationale=f"Learner elected to continue down the rabbit hole to {sub_cid}.",
            error_category="RABBIT_HOLE_DESCENT",
        )
        return self._make_step_outcome(
            previous_state=prev_state,
            current_state=self.state,
            active_item_id=str(self.active_item_spec["id"]),
            classification=synthetic_classification,
            stack_depth=len(self.call_stack),
            seed_bump=sub_frame.seed_bump,
            slip_nudge_emitted=False,
            unlocked_hint_tier=0,
            unlocked_hint_text=None,
            cluster_mastered=False,
            message=f"Entered depth-{len(self.call_stack)} rabbit-hole cluster {sub_session.cluster_id} for {sub_cid}.",
        )

    def return_to_worksheet(
        self,
        timestamp_unix: Optional[float] = None,
    ) -> AutomatonStepOutcome:
        """
        Learner-Agency Action 2: Trigger-Gated Return to Worksheet.
        Strictly hidden and refused BEFORE learner-agency criteria trigger. Once agency criteria
        trigger, unwinds the active remediation stack from any depth d >= 1 back to
        the isomorphically refreshed root question q_0' (S0_REFRESHED_ANCHOR_RETURN)
        and logs unmastered prerequisite concepts to the Spaced Review Queue & OLM.
        """
        if not self.return_to_worksheet_visible:
            raise PermissionError(
                "Return to Worksheet option does not appear until learner-agency criteria trigger."
            )

        prev_state = self.state
        root_id = self.active_root_id
        bump = self.root_seed_bumps.get(root_id, 1)

        deferred_concepts: List[str] = [f.missed_concept_id for f in self.call_stack]
        if (
            self.pending_rabbit_hole_concept_id is not None
            and self.pending_rabbit_hole_concept_id not in deferred_concepts
        ):
            deferred_concepts.append(self.pending_rabbit_hole_concept_id)

        self.call_stack.clear()
        self.suspended_cluster_stack.clear()
        self.active_cluster_session = None
        self.pending_rabbit_hole_concept_id = None
        self.pending_rabbit_hole_misconception_id = None
        self.visited_concepts_in_chain.clear()
        self.agency_criteria_triggered = False

        deferred_entry = {
            "root_item_id": root_id,
            "unresolved_concept_id": deferred_concepts[0] if deferred_concepts else "unknown",
            "deferred_concepts": deferred_concepts,
            "anchor_return_failures": self.anchor_return_failures.get(root_id, 0),
            "seed_bump": bump,
            "recommended_action": "SPACED_REVIEW_QUEUE_AND_OLM_RABBIT_HOLE_DEFERRED",
        }
        self.deferred_escalation_queue.append(deferred_entry)

        self.state = AutomatonState.S0_REFRESHED_ANCHOR_RETURN
        self.active_item_spec = self.instantiate_refreshed_item(
            self.root_problems[root_id],
            seed_bump=bump,
        )
        self.active_parameters = self._extract_parameters(
            self.active_item_spec,
            seed_bump=bump,
        )

        self.kv_engine.append_telemetry_event(
            event_type="AGENCY_RETURN_TO_WORKSHEET",
            item_id=root_id,
            automaton_state=self.state.value,
            latency_sec=1.0,
            cri=3.0,
            hints_used=(),
            payload=deferred_entry,
            timestamp_unix=timestamp_unix,
        )
        synthetic_classification = ClassificationResult(
            is_correct=False,
            tractable=True,
            target_concept_id=deferred_entry["unresolved_concept_id"],
            misconception_id="AGENCY_USER_RETURN_TO_WORKSHEET",
            diagnostic_rationale="Learner elected Return to Worksheet; deferred concepts logged to OLM.",
            error_category="USER_RETURN_TO_WORKSHEET",
        )
        return self._make_step_outcome(
            previous_state=prev_state,
            current_state=self.state,
            active_item_id=root_id,
            classification=synthetic_classification,
            stack_depth=0,
            seed_bump=bump,
            slip_nudge_emitted=False,
            unlocked_hint_tier=0,
            unlocked_hint_text=None,
            cluster_mastered=False,
            message=f"Unwound rabbit-hole stack to refreshed root question {root_id}; logged {deferred_concepts} to OLM.",
        )

    def _handle_root_submission(
        self,
        learner_response: Any,
        cri_score: float,
        latency_sec: float,
        timestamp_unix: Optional[float],
    ) -> AutomatonStepOutcome:
        prev_state = self.state
        root_id = self.active_root_id
        classification = MisconceptionClassifier.classify_response(
            self.active_item_spec,
            learner_response,
            active_parameters=self.active_parameters,
        )
        primary_cid = str(
            self.active_item_spec.get("primary_concept_id")
            or self.active_item_spec.get("taught_by")
            or "concept_general"
        )
        coactivated = list((self.active_item_spec.get("ckv_loadings") or {}).keys())

        if classification.is_correct:
            self.kv_engine.update_on_observation(
                concept_id=primary_cid,
                is_correct=True,
                cri_score=cri_score,
                timestamp_unix=timestamp_unix,
                coactivated_concepts=coactivated,
                propagate_laplacian=True,
            )
            hint_tier = self.root_hint_tiers.get(root_id, 0)
            credit_penalty = 0.0 if hint_tier <= 1 else (0.10 if hint_tier == 2 else 0.20)
            self.completed_roots[root_id] = round(max(0.25, 1.0 - credit_penalty), 2)
            self.state = AutomatonState.MASTERED_COMPLETED
            self.agency_criteria_triggered = False
            self.pending_rabbit_hole_concept_id = None
            self.pending_rabbit_hole_misconception_id = None
            self.visited_concepts_in_chain.clear()

            self.kv_engine.append_telemetry_event(
                event_type="ROOT_CORRECT_MASTERED",
                item_id=root_id,
                automaton_state=self.state.value,
                latency_sec=latency_sec,
                cri=cri_score,
                hints_used=tuple(range(1, hint_tier + 1)),
                payload={
                    "seed_bump": self.root_seed_bumps.get(root_id, 0),
                    "recorded_credit": self.completed_roots[root_id],
                },
                timestamp_unix=timestamp_unix,
            )
            return self._make_step_outcome(
                previous_state=prev_state,
                current_state=self.state,
                active_item_id=root_id,
                classification=classification,
                stack_depth=len(self.call_stack),
                seed_bump=self.root_seed_bumps.get(root_id, 0),
                slip_nudge_emitted=False,
                unlocked_hint_tier=hint_tier,
                unlocked_hint_text=None,
                cluster_mastered=False,
                message=classification.diagnostic_rationale,
            )

        # Handle untractable slip vs tractable misconception
        if not classification.tractable:
            slip_count = self.untractable_slip_counts.get(root_id, 0) + 1
            self.untractable_slip_counts[root_id] = slip_count
            if slip_count <= self.untractable_slip_free_nudges:
                self.kv_engine.append_telemetry_event(
                    event_type="UNTRACTABLE_SLIP_NUDGE",
                    item_id=root_id,
                    automaton_state=self.state.value,
                    latency_sec=latency_sec,
                    cri=cri_score,
                    hints_used=(),
                    payload={"slip_count": slip_count, "message": classification.diagnostic_rationale},
                    timestamp_unix=timestamp_unix,
                )
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=root_id,
                    classification=classification,
                    stack_depth=len(self.call_stack),
                    seed_bump=self.root_seed_bumps.get(root_id, 0),
                    slip_nudge_emitted=True,
                    unlocked_hint_tier=self.root_hint_tiers.get(root_id, 0),
                    unlocked_hint_text=None,
                    cluster_mastered=False,
                    message=classification.diagnostic_rationale,
                )
            # Repeated untractable failure -> escalate to primary_concept_id
            target_cid = primary_cid
            misc_id = "ESCALATED_REPEATED_UNTRACTABLE_SLIP"
        else:
            target_cid = str(classification.target_concept_id or primary_cid)
            misc_id = str(classification.misconception_id or "TRACTABLE_MISCONCEPTION")

        self.kv_engine.update_on_observation(
            concept_id=target_cid,
            is_correct=False,
            cri_score=cri_score,
            timestamp_unix=timestamp_unix,
            coactivated_concepts=coactivated,
            propagate_laplacian=True,
        )

        # Finite-step termination guarantee: if learner repeatedly fails
        # S0_REFRESHED_ANCHOR_RETURN (anchor_return_failures >= max_anchor_return_attempts),
        # transition deterministically to terminal state S3_DIAGNOSTIC_ESCALATION_DEFERRED
        if prev_state == AutomatonState.S0_REFRESHED_ANCHOR_RETURN:
            ret_fails = self.anchor_return_failures.get(root_id, 0) + 1
            self.anchor_return_failures[root_id] = ret_fails
            if ret_fails >= self.max_anchor_return_attempts:
                self.call_stack.clear()
                self.suspended_cluster_stack.clear()
                self.active_cluster_session = None
                self.pending_rabbit_hole_concept_id = None
                self.pending_rabbit_hole_misconception_id = None
                self.visited_concepts_in_chain.clear()
                self.agency_criteria_triggered = False
                self.state = AutomatonState.S3_DIAGNOSTIC_ESCALATION_DEFERRED
                self.root_hint_tiers[root_id] = 3
                self.completed_roots[root_id] = 0.25
                worked_solution = self._get_hint_for_tier(self.active_item_spec, 3)
                deferred_entry = {
                    "root_item_id": root_id,
                    "unresolved_concept_id": target_cid,
                    "misconception_id": misc_id,
                    "anchor_return_failures": ret_fails,
                    "seed_bump": self.root_seed_bumps.get(root_id, 0),
                    "recommended_action": "SPACED_REVIEW_QUEUE_AND_OLM_FLAG",
                }
                self.deferred_escalation_queue.append(deferred_entry)
                self.kv_engine.append_telemetry_event(
                    event_type="S3_DIAGNOSTIC_ESCALATION_DEFERRED",
                    item_id=root_id,
                    automaton_state=self.state.value,
                    latency_sec=latency_sec,
                    cri=cri_score,
                    hints_used=(1, 2, 3),
                    payload=deferred_entry,
                    timestamp_unix=timestamp_unix,
                )
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=root_id,
                    classification=classification,
                    stack_depth=0,
                    seed_bump=self.root_seed_bumps.get(root_id, 0),
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=3,
                    unlocked_hint_text=worked_solution,
                    cluster_mastered=False,
                    message=(
                        f"Reached max_anchor_return_attempts={self.max_anchor_return_attempts} on "
                        f"{root_id}; logged {target_cid} to Spaced Review Queue & OLM and unlocked "
                        f"full worked solution ({worked_solution})."
                    ),
                )

        # Push ReturnFrame onto bounded call stack and transition to S1_DIAGNOSTIC_PROBE
        new_bump = self.root_seed_bumps.get(root_id, 0) + 1
        self.root_seed_bumps[root_id] = new_bump
        self.visited_concepts_in_chain = {target_cid}
        self.pending_rabbit_hole_concept_id = None
        self.pending_rabbit_hole_misconception_id = None
        # Condition B: if learner already failed S0_REFRESHED_ANCHOR_RETURN >= 1 time,
        # Learner-agency criteria trigger immediately upon re-entering remediation
        self.agency_criteria_triggered = bool(self.anchor_return_failures.get(root_id, 0) >= 1)

        probes = self.probe_pools.get(target_cid) or []
        if not probes:
            # Fallback if concept has no dedicated S1 probe: go straight to S2 if cluster exists
            if target_cid not in self.remedial_clusters:
                next_tier = min(3, self.root_hint_tiers.get(root_id, 0) + 1)
                self.root_hint_tiers[root_id] = next_tier
                hint_txt = self._get_hint_for_tier(self.active_item_spec, next_tier)
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=prev_state,
                    active_item_id=root_id,
                    classification=classification,
                    stack_depth=0,
                    seed_bump=self.root_seed_bumps.get(root_id, 0),
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=next_tier,
                    unlocked_hint_text=hint_txt,
                    cluster_mastered=False,
                    message=classification.diagnostic_rationale,
                )
            if target_cid in self.remedial_clusters:
                cluster_id = str(self.remedial_clusters[target_cid]["cluster_id"])
                frame = ReturnFrame(
                    frame_id=f"frame_{root_id}_{new_bump}",
                    origin_state=prev_state,
                    root_item_id=root_id,
                    missed_concept_id=target_cid,
                    misconception_id=misc_id,
                    seed_bump=new_bump,
                    depth=len(self.call_stack) + 1,
                    diagnostic_probe_id="NONE",
                    remedial_cluster_id=cluster_id,
                )
                self.call_stack.append(frame)
                self.state = AutomatonState.S2_REMEDIAL_CLUSTER
                self._spawn_remedial_cluster_session(target_cid)
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=str(self.active_item_spec["id"]),
                    classification=classification,
                    stack_depth=len(self.call_stack),
                    seed_bump=new_bump,
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=0,
                    unlocked_hint_text=None,
                    cluster_mastered=False,
                    message=classification.diagnostic_rationale,
                )

        probe_spec = probes[(new_bump - 1) % len(probes)]
        cluster_id = (
            str(self.remedial_clusters[target_cid]["cluster_id"])
            if target_cid in self.remedial_clusters
            else f"cluster_{target_cid}"
        )
        frame = ReturnFrame(
            frame_id=f"frame_{root_id}_{new_bump}",
            origin_state=prev_state,
            root_item_id=root_id,
            missed_concept_id=target_cid,
            misconception_id=misc_id,
            seed_bump=new_bump,
            depth=len(self.call_stack) + 1,
            diagnostic_probe_id=str(probe_spec["id"]),
            remedial_cluster_id=cluster_id,
        )
        self.call_stack.append(frame)
        self.state = AutomatonState.S1_DIAGNOSTIC_PROBE
        self.active_item_spec = copy.deepcopy(probe_spec)
        self.active_parameters = self._extract_parameters(self.active_item_spec, seed_bump=0)

        self.kv_engine.append_telemetry_event(
            event_type="PUSH_S1_DIAGNOSTIC_PROBE",
            item_id=str(probe_spec["id"]),
            automaton_state=self.state.value,
            latency_sec=latency_sec,
            cri=cri_score,
            hints_used=(),
            payload={
                "root_item_id": root_id,
                "missed_concept_id": target_cid,
                "misconception_id": misc_id,
                "seed_bump": new_bump,
                "stack_depth": len(self.call_stack),
            },
            timestamp_unix=timestamp_unix,
        )
        return self._make_step_outcome(
            previous_state=prev_state,
            current_state=self.state,
            active_item_id=str(probe_spec["id"]),
            classification=classification,
            stack_depth=len(self.call_stack),
            seed_bump=new_bump,
            slip_nudge_emitted=False,
            unlocked_hint_tier=0,
            unlocked_hint_text=None,
            cluster_mastered=False,
            message=classification.diagnostic_rationale,
        )

    def _handle_probe_submission(
        self,
        learner_response: Any,
        cri_score: float,
        latency_sec: float,
        timestamp_unix: Optional[float],
    ) -> AutomatonStepOutcome:
        prev_state = self.state
        frame = self.call_stack[-1]
        probe_id = str(self.active_item_spec.get("id", frame.diagnostic_probe_id))
        classification = MisconceptionClassifier.classify_response(
            self.active_item_spec,
            learner_response,
            active_parameters=self.active_parameters,
        )

        if classification.is_correct:
            # Shallow slip repaired! Update theta positively, pop stack, return to refreshed q_0'
            self.kv_engine.update_on_observation(
                concept_id=frame.missed_concept_id,
                is_correct=True,
                cri_score=cri_score,
                timestamp_unix=timestamp_unix,
                propagate_laplacian=True,
            )
            popped = self.call_stack.pop()
            self.visited_concepts_in_chain.clear()
            self.pending_rabbit_hole_concept_id = None
            self.pending_rabbit_hole_misconception_id = None
            self.agency_criteria_triggered = False
            self.state = AutomatonState.S0_REFRESHED_ANCHOR_RETURN
            self.active_root_id = popped.root_item_id
            self.active_item_spec = self.instantiate_refreshed_item(
                self.root_problems[popped.root_item_id],
                seed_bump=popped.seed_bump,
            )
            self.active_parameters = self._extract_parameters(
                self.active_item_spec,
                seed_bump=popped.seed_bump,
            )

            self.kv_engine.append_telemetry_event(
                event_type="S1_PROBE_PASSED_POP_TO_REFRESHED_S0",
                item_id=popped.root_item_id,
                automaton_state=self.state.value,
                latency_sec=latency_sec,
                cri=cri_score,
                hints_used=(),
                payload={
                    "probe_id": probe_id,
                    "repaired_concept_id": popped.missed_concept_id,
                    "seed_bump": popped.seed_bump,
                },
                timestamp_unix=timestamp_unix,
            )
            return self._make_step_outcome(
                previous_state=prev_state,
                current_state=self.state,
                active_item_id=popped.root_item_id,
                classification=classification,
                stack_depth=len(self.call_stack),
                seed_bump=popped.seed_bump,
                slip_nudge_emitted=False,
                unlocked_hint_tier=0,
                unlocked_hint_text=None,
                cluster_mastered=False,
                message="Diagnostic probe passed! Returning to refreshed root question.",
            )

        # Probe failed -> confirmed conceptual gap! Transition S1 -> S2_REMEDIAL_CLUSTER
        self.kv_engine.update_on_observation(
            concept_id=frame.missed_concept_id,
            is_correct=False,
            cri_score=cri_score,
            timestamp_unix=timestamp_unix,
            propagate_laplacian=True,
        )
        self.visited_concepts_in_chain.add(frame.missed_concept_id)
        self.state = AutomatonState.S2_REMEDIAL_CLUSTER
        cluster_session = self._spawn_remedial_cluster_session(frame.missed_concept_id)

        self.kv_engine.append_telemetry_event(
            event_type="S1_PROBE_FAILED_ENTER_S2_CLUSTER",
            item_id=cluster_session.cluster_id,
            automaton_state=self.state.value,
            latency_sec=latency_sec,
            cri=cri_score,
            hints_used=(),
            payload={
                "probe_id": probe_id,
                "missed_concept_id": frame.missed_concept_id,
                "cluster_id": cluster_session.cluster_id,
            },
            timestamp_unix=timestamp_unix,
        )
        return self._make_step_outcome(
            previous_state=prev_state,
            current_state=self.state,
            active_item_id=str(self.active_item_spec["id"]),
            classification=classification,
            stack_depth=len(self.call_stack),
            seed_bump=frame.seed_bump,
            slip_nudge_emitted=False,
            unlocked_hint_tier=0,
            unlocked_hint_text=None,
            cluster_mastered=False,
            message=f"Confirmed conceptual gap on {frame.missed_concept_id}; spawning remedial cluster {cluster_session.cluster_id}.",
        )

    def _handle_cluster_item_submission(
        self,
        learner_response: Any,
        cri_score: float,
        latency_sec: float,
        timestamp_unix: Optional[float],
    ) -> AutomatonStepOutcome:
        if self.active_cluster_session is None:
            raise RuntimeError("No active S2 remedial cluster session")

        prev_state = self.state
        session = self.active_cluster_session
        cur_item_id = str(self.active_item_spec["id"])
        session.item_attempts[cur_item_id] = session.item_attempts.get(cur_item_id, 0) + 1

        classification = MisconceptionClassifier.classify_response(
            self.active_item_spec,
            learner_response,
            active_parameters=self.active_parameters,
        )
        cluster_spec = self.remedial_clusters[session.target_concept_id]
        grad_items = cluster_spec.get("graduated_mastery_items") or []

        if classification.is_correct:
            self.pending_rabbit_hole_concept_id = None
            self.pending_rabbit_hole_misconception_id = None
            session.completed_graduated_items.add(cur_item_id)
            self.kv_engine.update_on_observation(
                concept_id=session.target_concept_id,
                is_correct=True,
                cri_score=cri_score,
                timestamp_unix=timestamp_unix,
                propagate_laplacian=True,
            )

            # Check if more graduated items remain
            remaining = [
                item for item in grad_items if str(item["id"]) not in session.completed_graduated_items
            ]
            if remaining:
                next_item = copy.deepcopy(dict(remaining[0]))
                self.active_item_spec = next_item
                self.active_parameters = self._extract_parameters(next_item, seed_bump=0)
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=str(next_item["id"]),
                    classification=classification,
                    stack_depth=len(self.call_stack),
                    seed_bump=self.call_stack[-1].seed_bump if self.call_stack else 0,
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=0,
                    unlocked_hint_text=None,
                    cluster_mastered=False,
                    message=f"Graduated item {cur_item_id} solved; advancing to {next_item['id']}.",
                )

            # All graduated items solved; verify POE visualizer gate satisfaction
            if not session.poe_gate.is_satisfied:
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=cur_item_id,
                    classification=classification,
                    stack_depth=len(self.call_stack),
                    seed_bump=self.call_stack[-1].seed_bump if self.call_stack else 0,
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=0,
                    unlocked_hint_text=None,
                    cluster_mastered=False,
                    message=(
                        "Graduated items complete, but POE visualizer gate requires prediction "
                        f"plus >= {session.poe_gate.min_regimes_required} regimes with dwell >= "
                        f"{session.poe_gate.min_dwell_seconds:.1f} s."
                    ),
                )

            # Cluster C(c_m) is 100% mastered! Pop stack and return to parent cluster or refreshed q_0'
            self.mastered_clusters.add(session.cluster_id)
            popped = self.call_stack.pop()
            if popped.origin_state == AutomatonState.S2_REMEDIAL_CLUSTER and self.suspended_cluster_stack:
                parent_session, parent_item = self.suspended_cluster_stack.pop()
                self.active_cluster_session = parent_session
                self.state = AutomatonState.S2_REMEDIAL_CLUSTER
                self.active_item_spec = parent_item
                self.active_parameters = self._extract_parameters(parent_item, seed_bump=0)
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=str(parent_item["id"]),
                    classification=classification,
                    stack_depth=len(self.call_stack),
                    seed_bump=popped.seed_bump,
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=0,
                    unlocked_hint_text=None,
                    cluster_mastered=True,
                    message=f"Nested sub-cluster {session.cluster_id} mastered! Returning to parent cluster {parent_session.cluster_id}.",
                )

            self.active_cluster_session = None
            self.visited_concepts_in_chain.clear()
            self.pending_rabbit_hole_concept_id = None
            self.pending_rabbit_hole_misconception_id = None
            self.agency_criteria_triggered = False
            self.state = AutomatonState.S0_REFRESHED_ANCHOR_RETURN
            self.active_root_id = popped.root_item_id
            self.active_item_spec = self.instantiate_refreshed_item(
                self.root_problems[popped.root_item_id],
                seed_bump=popped.seed_bump,
            )
            self.active_parameters = self._extract_parameters(
                self.active_item_spec,
                seed_bump=popped.seed_bump,
            )

            self.kv_engine.append_telemetry_event(
                event_type="S2_CLUSTER_MASTERED_POP_TO_REFRESHED_S0",
                item_id=popped.root_item_id,
                automaton_state=self.state.value,
                latency_sec=latency_sec,
                cri=cri_score,
                hints_used=(),
                payload={
                    "mastered_cluster_id": session.cluster_id,
                    "mastered_concept_id": session.target_concept_id,
                    "seed_bump": popped.seed_bump,
                },
                timestamp_unix=timestamp_unix,
            )
            return self._make_step_outcome(
                previous_state=prev_state,
                current_state=self.state,
                active_item_id=popped.root_item_id,
                classification=classification,
                stack_depth=len(self.call_stack),
                seed_bump=popped.seed_bump,
                slip_nudge_emitted=False,
                unlocked_hint_tier=0,
                unlocked_hint_text=None,
                cluster_mastered=True,
                message=f"Remedial cluster {session.cluster_id} mastered! Returning to refreshed root question {popped.root_item_id}.",
            )

        # Incorrect answer inside S2_REMEDIAL_CLUSTER:
        # Automatic recursion when stack depth < max_recursion_depth (e.g. D_trigger = 2 on bridge concepts)
        sub_cid = classification.target_concept_id
        if (
            len(self.call_stack) < self.max_recursion_depth
            and classification.tractable
            and sub_cid is not None
            and sub_cid != session.target_concept_id
            and sub_cid in self.remedial_clusters
            and sub_cid not in self.visited_concepts_in_chain
        ):
            self.suspended_cluster_stack.append((session, copy.deepcopy(self.active_item_spec)))
            sub_cluster_id = str(self.remedial_clusters[sub_cid]["cluster_id"])
            parent_frame = self.call_stack[-1]
            sub_frame = ReturnFrame(
                frame_id=f"frame_nested_{sub_cid}_{len(self.call_stack) + 1}",
                origin_state=AutomatonState.S2_REMEDIAL_CLUSTER,
                root_item_id=parent_frame.root_item_id,
                missed_concept_id=sub_cid,
                misconception_id=str(classification.misconception_id or "NESTED_SUB_GAP"),
                seed_bump=parent_frame.seed_bump,
                depth=len(self.call_stack) + 1,
                diagnostic_probe_id="NONE",
                remedial_cluster_id=sub_cluster_id,
            )
            self.call_stack.append(sub_frame)
            self.visited_concepts_in_chain.add(sub_cid)
            sub_session = self._spawn_remedial_cluster_session(sub_cid)
            return self._make_step_outcome(
                previous_state=prev_state,
                current_state=self.state,
                active_item_id=str(self.active_item_spec["id"]),
                classification=classification,
                stack_depth=len(self.call_stack),
                seed_bump=sub_frame.seed_bump,
                slip_nudge_emitted=False,
                unlocked_hint_tier=0,
                unlocked_hint_text=None,
                cluster_mastered=False,
                message=f"Entered depth-{len(self.call_stack)} sub-cluster {sub_session.cluster_id} for {sub_cid}.",
            )

        # Condition A (Deep Prerequisite Boundary at stack_depth >= max_recursion_depth):
        # Trigger learner-agency options ('Return to Worksheet' and optional 'Continue Down the Rabbit Hole')
        if len(self.call_stack) >= self.max_recursion_depth:
            self.agency_criteria_triggered = True
            self.pending_rabbit_hole_concept_id = None
            self.pending_rabbit_hole_misconception_id = None
            if (
                classification.tractable
                and sub_cid is not None
                and sub_cid != session.target_concept_id
                and sub_cid in self.remedial_clusters
                and sub_cid not in self.visited_concepts_in_chain
                and len(self.call_stack) < self.max_rabbit_hole_depth
            ):
                self.pending_rabbit_hole_concept_id = sub_cid
                self.pending_rabbit_hole_misconception_id = str(
                    classification.misconception_id or "NESTED_RABBIT_HOLE_GAP"
                )

        # Unlock progressive 3-tier hints in-place as fallback
        cur_tier = session.item_hint_tiers_unlocked.get(cur_item_id, 0)
        next_tier = min(3, cur_tier + 1)
        session.item_hint_tiers_unlocked[cur_item_id] = next_tier
        hint_text = self._get_hint_for_tier(self.active_item_spec, next_tier)

        # Bounded in-cluster bottom-out advance (E4): if learner fails a graduated item
        # beyond max_in_cluster_item_attempts (after Tier 3 worked step was already shown),
        # advance via bottom-out completion so the cluster cannot stall indefinitely
        if session.item_attempts.get(cur_item_id, 0) > self.max_in_cluster_item_attempts:
            self.pending_rabbit_hole_concept_id = None
            self.pending_rabbit_hole_misconception_id = None
            session.completed_graduated_items.add(cur_item_id)
            remaining = [
                item for item in grad_items if str(item["id"]) not in session.completed_graduated_items
            ]
            if remaining:
                next_item = copy.deepcopy(dict(remaining[0]))
                self.active_item_spec = next_item
                self.active_parameters = self._extract_parameters(next_item, seed_bump=0)
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=str(next_item["id"]),
                    classification=classification,
                    stack_depth=len(self.call_stack),
                    seed_bump=self.call_stack[-1].seed_bump if self.call_stack else 0,
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=3,
                    unlocked_hint_text=hint_text,
                    cluster_mastered=False,
                    message=f"Bottom-out worked step applied on {cur_item_id}; advancing to {next_item['id']}.",
                )

            # All graduated items exhausted via bottom-out scaffolding -> pop stack
            popped = self.call_stack.pop()
            if popped.origin_state == AutomatonState.S2_REMEDIAL_CLUSTER and self.suspended_cluster_stack:
                parent_session, parent_item = self.suspended_cluster_stack.pop()
                self.active_cluster_session = parent_session
                self.state = AutomatonState.S2_REMEDIAL_CLUSTER
                self.active_item_spec = parent_item
                self.active_parameters = self._extract_parameters(parent_item, seed_bump=0)
                return self._make_step_outcome(
                    previous_state=prev_state,
                    current_state=self.state,
                    active_item_id=str(parent_item["id"]),
                    classification=classification,
                    stack_depth=len(self.call_stack),
                    seed_bump=popped.seed_bump,
                    slip_nudge_emitted=False,
                    unlocked_hint_tier=3,
                    unlocked_hint_text=hint_text,
                    cluster_mastered=False,
                    message=f"Sub-cluster {session.cluster_id} bottomed out; returning to parent cluster {parent_session.cluster_id}.",
                )

            self.active_cluster_session = None
            self.visited_concepts_in_chain.clear()
            self.pending_rabbit_hole_concept_id = None
            self.pending_rabbit_hole_misconception_id = None
            self.agency_criteria_triggered = False
            self.state = AutomatonState.S0_REFRESHED_ANCHOR_RETURN
            self.active_root_id = popped.root_item_id
            self.active_item_spec = self.instantiate_refreshed_item(
                self.root_problems[popped.root_item_id],
                seed_bump=popped.seed_bump,
            )
            self.active_parameters = self._extract_parameters(
                self.active_item_spec,
                seed_bump=popped.seed_bump,
            )
            return self._make_step_outcome(
                previous_state=prev_state,
                current_state=self.state,
                active_item_id=popped.root_item_id,
                classification=classification,
                stack_depth=len(self.call_stack),
                seed_bump=popped.seed_bump,
                slip_nudge_emitted=False,
                unlocked_hint_tier=3,
                unlocked_hint_text=hint_text,
                cluster_mastered=False,
                message=f"Cluster {session.cluster_id} bottom-out scaffolding complete; returning to refreshed {popped.root_item_id}.",
            )

        self.kv_engine.append_telemetry_event(
            event_type="S2_IN_CLUSTER_HINT_ESC",
            item_id=cur_item_id,
            automaton_state=self.state.value,
            latency_sec=latency_sec,
            cri=cri_score,
            hints_used=tuple(range(1, next_tier + 1)),
            payload={
                "cluster_id": session.cluster_id,
                "unlocked_tier": next_tier,
                "recursion_capped_at_d_max": len(self.call_stack) >= self.max_recursion_depth,
                "agency_criteria_triggered": self.agency_criteria_triggered,
                "return_to_worksheet_visible": self.return_to_worksheet_visible,
                "rabbit_hole_available": self.rabbit_hole_available,
                "pending_rabbit_hole_concept_id": self.pending_rabbit_hole_concept_id,
            },
            timestamp_unix=timestamp_unix,
        )
        return self._make_step_outcome(
            previous_state=prev_state,
            current_state=self.state,
            active_item_id=cur_item_id,
            classification=classification,
            stack_depth=len(self.call_stack),
            seed_bump=self.call_stack[-1].seed_bump if self.call_stack else 0,
            slip_nudge_emitted=False,
            unlocked_hint_tier=next_tier,
            unlocked_hint_text=hint_text,
            cluster_mastered=False,
            message=f"Tier-{next_tier} scaffold unlocked inside {session.cluster_id}: {hint_text}",
        )


def load_lesson_04_worked_example(
    schema_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Load the concrete V2 worked cluster example for Lesson 04 (Velocity Verlet
    Symplectic Drift vs. Euler Energy Blowup) from v2_curriculum_and_cluster_schema.json.
    """
    if schema_path is None:
        schema_path = (
            Path(__file__).resolve().parent.parent
            / "Schemas"
            / "v2_curriculum_and_cluster_schema.json"
        )
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    return dict(data["v2_worked_example_lesson_04"])



# ---------------------------------------------------------------------------
# 7. Interactive Jupyter Remediation Widget & Learner-Agency Controls
# ---------------------------------------------------------------------------

try:
    import ipywidgets as widgets
    import matplotlib.pyplot as plt
    from IPython.display import display, Markdown
except ImportError:  # pragma: no cover
    widgets = None  # type: ignore
    plt = None  # type: ignore


def _default_html_escape(text: str) -> str:
    if not text:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


class ToKRemediationWidget:
    """
    Interactive ipywidgets view over `TreeOfKnowledgeAutomaton` supporting:
      - S1_DIAGNOSTIC_PROBE (targeted prerequisite probe)
      - S2_REMEDIAL_CLUSTER (refutation + faded worked step + POE prediction gate
        unlocking concept visualizer sliders & regime dwell logging + graduated
        mastery items with progressive Tier 1/2/3 hints)
      - S0_REFRESHED_ANCHOR_RETURN & S3_DIAGNOSTIC_ESCALATION_DEFERRED
      - Prominent Learner-Agency Controls (`return_to_worksheet_btn`,
        `continue_rabbit_hole_btn`, and `agency_banner_box`): strictly hidden and
        disabled before learner-agency criteria trigger, immediately surfaced upon trigger!
    """

    def __init__(
        self,
        automaton: TreeOfKnowledgeAutomaton,
        concept_module: Any = None,
        format_html_fn: Optional[Callable[[str], str]] = None,
        wrap_controls_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        on_root_resolved: Optional[Callable[[AutomatonStepOutcome], None]] = None,
        initial_outcome: Optional[AutomatonStepOutcome] = None,
    ) -> None:
        if widgets is None:
            raise ImportError("ipywidgets is required to instantiate ToKRemediationWidget")

        self.automaton = automaton
        self.concept_module = concept_module
        self._fmt = format_html_fn or _default_html_escape
        self._wrap_controls = wrap_controls_fn or (lambda d: d)
        self.on_root_resolved = on_root_resolved
        self.last_outcome: Optional[AutomatonStepOutcome] = initial_outcome

        # Prominent Learner-Agency Button Controls
        self.return_to_worksheet_btn = widgets.Button(
            description="Return to Worksheet",
            icon="arrow-left",
            button_style="info",
            tooltip="Unwind deep prerequisite exploration and return to the active worksheet question",
            layout=widgets.Layout(width="220px", height="38px", display="none"),
            disabled=True,
        )
        self.continue_rabbit_hole_btn = widgets.Button(
            description="Continue Down the Rabbit Hole",
            icon="sitemap",
            button_style="primary",
            layout=widgets.Layout(width="250px", height="38px", display="none"),
            disabled=True,
        )
        self.agency_banner_box = widgets.HBox(
            [],
            layout=widgets.Layout(
                flex_flow="row wrap",
                align_items="center",
                margin="6px 0",
                padding="6px 8px",
            ),
        )

        self.return_to_worksheet_btn.on_click(self._on_click_return_to_worksheet)
        self.continue_rabbit_hole_btn.on_click(self._on_click_continue_rabbit_hole)

        # Interactive sub-controls exposed for live interaction & deterministic unit tests
        self.cri_slider = widgets.FloatSlider(
            value=3.0,
            min=0.0,
            max=5.0,
            step=0.5,
            description="CRI (0-5):",
            continuous_update=False,
            layout=widgets.Layout(width="280px"),
        )
        self.active_choice_checkboxes: Dict[str, widgets.Checkbox] = {}
        self.active_numeric_input = widgets.FloatText(
            value=0.0,
            description="Answer:",
            layout=widgets.Layout(width="220px"),
        )
        self.item_submit_btn = widgets.Button(
            description="Submit Step Answer",
            button_style="primary",
            layout=widgets.Layout(width="180px", margin="6px 0"),
        )
        self.item_submit_btn.on_click(self._on_click_submit_item)

        self.poe_radio = widgets.RadioButtons(
            options=[],
            layout=widgets.Layout(width="98%"),
        )
        self.poe_commit_btn = widgets.Button(
            description="Commit Prediction & Unlock Sliders",
            button_style="primary",
            layout=widgets.Layout(width="260px", margin="4px 0"),
        )
        self.poe_commit_btn.on_click(self._on_click_poe_commit)

        self.regime_buttons: Dict[str, widgets.Button] = {}
        self.visualizer_sliders: Dict[str, widgets.Widget] = {}
        self.status_feedback_html = widgets.HTMLMath(value="")
        self.body_box = widgets.VBox([], layout=widgets.Layout(width="100%"))
        self.container = widgets.VBox(
            [self.agency_banner_box, self.status_feedback_html, self.body_box],
            layout=widgets.Layout(
                width="100%",
                border="1px solid #CBD5E1",
                padding="10px 12px",
                margin="8px 0",
            ),
        )
        self.refresh()

    def _sync_agency_controls(self) -> None:
        """
        Enforce strict learner-agency button visibility invariants:
        - Before trigger (`return_to_worksheet_visible == False`): both buttons
          have `display == "none"`, `disabled == True`, and `agency_banner_box.children == ()`.
        - Once triggered (`return_to_worksheet_visible == True`): `return_to_worksheet_btn`
          flips to `display = "inline-flex"`, `disabled = False` inside `agency_banner_box`,
          alongside `continue_rabbit_hole_btn` whenever `rabbit_hole_available` is `True`.
        """
        if not self.automaton.return_to_worksheet_visible:
            self.return_to_worksheet_btn.layout.display = "none"
            self.return_to_worksheet_btn.disabled = True
            self.continue_rabbit_hole_btn.layout.display = "none"
            self.continue_rabbit_hole_btn.disabled = True
            self.agency_banner_box.children = ()
            return

        self.return_to_worksheet_btn.layout.display = "inline-flex"
        self.return_to_worksheet_btn.disabled = False

        depth = len(self.automaton.call_stack)
        if self.automaton.rabbit_hole_available:
            self.continue_rabbit_hole_btn.layout.display = "inline-flex"
            self.continue_rabbit_hole_btn.disabled = False
            pending_cid = self.automaton.pending_rabbit_hole_concept_id or "prerequisite"
            banner_msg = widgets.HTMLMath(
                value=(
                    f"<div style='background:#FEF3C7; border-left:4px solid #D97706; padding:6px 10px; "
                    f"width:100%; margin-bottom:4px;'><b>Deep Prerequisite Choice (Depth {depth}):</b> "
                    f"A foundational gap in <code>{pending_cid}</code> was detected. You may explore the "
                    f"deeper prerequisite cluster or unwind back to your worksheet problem.</div>"
                )
            )
            self.agency_banner_box.children = (
                banner_msg,
                self.continue_rabbit_hole_btn,
                self.return_to_worksheet_btn,
            )
        else:
            self.continue_rabbit_hole_btn.layout.display = "none"
            self.continue_rabbit_hole_btn.disabled = True
            banner_msg = widgets.HTMLMath(
                value=(
                    f"<div style='background:#FEF3C7; border-left:4px solid #D97706; padding:6px 10px; "
                    f"width:100%; margin-bottom:4px;'><b>Prerequisite Exploration Active (Depth {depth}):</b> "
                    f"Complete this cluster to step back up, or click <b>Return to Worksheet</b> at any time.</div>"
                )
            )
            self.agency_banner_box.children = (
                banner_msg,
                self.return_to_worksheet_btn,
            )

    def _on_click_continue_rabbit_hole(self, _btn: Any = None) -> None:
        outcome = self.automaton.continue_rabbit_hole()
        self.last_outcome = outcome
        self.refresh()

    def _on_click_return_to_worksheet(self, _btn: Any = None) -> None:
        outcome = self.automaton.return_to_worksheet()
        self.last_outcome = outcome
        self.refresh()
        if self.on_root_resolved is not None:
            self.on_root_resolved(outcome)

    def _on_click_poe_commit(self, _btn: Any = None) -> None:
        if self.automaton.state != AutomatonState.S2_REMEDIAL_CLUSTER:
            return
        selected = str(self.poe_radio.value or "opt_0")
        self.automaton.submit_poe_prediction(selected)
        self.refresh()

    def _build_item_input_widget(self, item_spec: Mapping[str, Any]) -> widgets.Widget:
        qtype = str(item_spec.get("question_type", "single_select"))
        self.active_choice_checkboxes = {}
        if qtype in ("single_select", "multi_select"):
            options = item_spec.get("options") or []
            rows: List[widgets.Widget] = []
            for opt in options:
                oid = str(opt["id"])
                otxt = str(opt.get("text") or opt.get("text_markdown") or opt.get("label") or "")
                cb = widgets.Checkbox(
                    value=False,
                    indent=False,
                    layout=widgets.Layout(width="24px", margin="2px 6px 2px 0"),
                )
                self.active_choice_checkboxes[oid] = cb
                lbl = widgets.HTMLMath(
                    value=f"<div><b>({oid.upper()})</b> {self._fmt(otxt)}</div>"
                )
                rows.append(widgets.HBox([cb, lbl], layout=widgets.Layout(align_items="flex-start")))

            if qtype == "single_select":
                def _make_excl(active_oid: str) -> Callable[[Dict[str, Any]], None]:
                    def _handler(change: Dict[str, Any]) -> None:
                        if change.get("new"):
                            for k, other_cb in self.active_choice_checkboxes.items():
                                if k != active_oid and other_cb.value:
                                    other_cb.value = False
                    return _handler

                for oid, cb in self.active_choice_checkboxes.items():
                    cb.observe(_make_excl(oid), names="value")

            return widgets.VBox([*rows, self.cri_slider, self.item_submit_btn])

        if qtype == "numeric":
            self.active_numeric_input.value = 0.0
            return widgets.VBox([self.active_numeric_input, self.cri_slider, self.item_submit_btn])

        return widgets.HTMLMath(value="<i>Submit code cell above to re-evaluate coding exercise.</i>")

    def _extract_current_widget_response(self) -> Any:
        item_spec = self.automaton.active_item_spec
        qtype = str(item_spec.get("question_type", "single_select"))
        if qtype == "single_select":
            chosen = [oid for oid, cb in self.active_choice_checkboxes.items() if cb.value]
            return chosen[0] if chosen else ""
        if qtype == "multi_select":
            return sorted(oid for oid, cb in self.active_choice_checkboxes.items() if cb.value)
        if qtype == "numeric":
            return float(self.active_numeric_input.value)
        return ""

    def _on_click_submit_item(self, _btn: Any = None) -> None:
        resp = self._extract_current_widget_response()
        if resp == "" or resp == []:
            self.status_feedback_html.value = (
                "<div style='color:#B45309; padding:4px 0;'><i>Please select or enter an answer before submitting.</i></div>"
            )
            return
        self.submit_active_answer(resp, cri_score=float(self.cri_slider.value))

    def submit_active_answer(
        self,
        learner_response: Any,
        cri_score: Optional[float] = None,
    ) -> AutomatonStepOutcome:
        cri_val = float(self.cri_slider.value if cri_score is None else cri_score)
        outcome = self.automaton.submit_response(
            learner_response=learner_response,
            cri_score=cri_val,
        )
        self.last_outcome = outcome
        self.refresh()
        if (
            outcome.current_state
            in (AutomatonState.MASTERED_COMPLETED, AutomatonState.S3_DIAGNOSTIC_ESCALATION_DEFERRED)
            and self.on_root_resolved is not None
        ):
            self.on_root_resolved(outcome)
        return outcome

    def _build_s2_cluster_view(self) -> widgets.Widget:
        session = self.automaton.active_cluster_session
        assert session is not None
        cspec = self.automaton.remedial_clusters[session.target_concept_id]
        depth = len(self.automaton.call_stack)

        refut_md = str(cspec.get("refutation_markdown", ""))
        faded_md = str(cspec.get("faded_worked_step_markdown", ""))
        scaffold_card = widgets.HTMLMath(
            value=(
                f"<div style='background:#EFF6FF; border-left:4px solid #2563EB; padding:8px 10px; margin:4px 0;'>"
                f"<b>Remedial Micro-Cluster ({session.cluster_id} &middot; Depth {depth}):</b><br>"
                f"{self._fmt(refut_md)}<hr style='margin:6px 0; border:0; border-top:1px solid #BFDBFE;'>"
                f"<b>Faded Worked Step:</b><br>{self._fmt(faded_md)}</div>"
            )
        )

        # POE Visualizer Gate Block
        vis_spec = cspec.get("poe_visualizer_spec") or {}
        poe_spec = vis_spec.get("poe_gate") or {}
        poe_prompt = str(poe_spec.get("prompt_markdown") or poe_spec.get("poe_prompt") or "Predict the physical response:")
        raw_opts = poe_spec.get("prediction_options") or poe_spec.get("poe_options") or [
            {"id": "opt_a", "label": "Increases monotonically"},
            {"id": "opt_b", "label": "Remains bounded / invariant"},
        ]
        radio_opts = []
        for o in raw_opts:
            oid = str(o.get("id", "opt"))
            olbl = str(o.get("label") or o.get("text") or oid)
            radio_opts.append((olbl, oid))
        self.poe_radio.options = radio_opts
        if radio_opts:
            self.poe_radio.value = session.poe_gate.poe_selected_option_id or radio_opts[0][1]
        self.poe_radio.disabled = session.poe_gate.poe_prediction_submitted
        self.poe_commit_btn.disabled = session.poe_gate.poe_prediction_submitted

        gate = session.poe_gate
        self.regime_buttons = {}
        regime_list = (
            poe_spec.get("regime_labels")
            or [r.get("regime_id", f"regime_{i+1}") for i, r in enumerate(poe_spec.get("target_regimes") or [])]
            or ["regime_1_baseline", "regime_2_perturbed"]
        )
        reg_btn_widgets: List[widgets.Widget] = []
        for reg_item in regime_list:
            reg_id = str(reg_item["regime_id"] if isinstance(reg_item, Mapping) else reg_item)
            explored = reg_id in gate.valid_explored_regimes
            r_btn = widgets.Button(
                description=f"{'[Explored] ' if explored else 'Explore '}{reg_id}",
                button_style="success" if explored else "",
                disabled=not gate.sliders_unlocked,
                layout=widgets.Layout(width="auto", margin="2px 4px 2px 0"),
            )

            def _make_reg_handler(rid: str) -> Callable[[Any], None]:
                def _on_reg(_b: Any) -> None:
                    self.automaton.record_regime_dwell(rid, max(1.6, gate.min_dwell_seconds + 0.1))
                    self.refresh()
                return _on_reg

            r_btn.on_click(_make_reg_handler(reg_id))
            self.regime_buttons[reg_id] = r_btn
            reg_btn_widgets.append(r_btn)

        fig_out = widgets.Output()
        render_fn_name = vis_spec.get("render_function")
        if gate.sliders_unlocked and render_fn_name and self.concept_module is not None and hasattr(self.concept_module, render_fn_name):
            fn = getattr(self.concept_module, render_fn_name)
            ctrl_defaults = vis_spec.get("default_controls") or {}
            with fig_out:
                try:
                    fig, ax = plt.subplots(figsize=(5.6, 3.6))
                    fn(self._wrap_controls(dict(ctrl_defaults)), ax)
                    plt.close(fig)
                except Exception:
                    plt.close("all")

        poe_status = (
            f"POE Gate: {'SATISFIED' if gate.is_satisfied else 'LOCKED'} "
            f"(Prediction={'Yes' if gate.poe_prediction_submitted else 'No'}, "
            f"Explored Regimes={len(gate.valid_explored_regimes)}/{gate.min_regimes_required})"
        )
        poe_box = widgets.VBox(
            [
                widgets.HTMLMath(value=f"<b>1. Predict-Observe-Explain (POE) Visualizer Gate:</b><br>{self._fmt(poe_prompt)}"),
                self.poe_radio,
                self.poe_commit_btn,
                widgets.HTMLMath(value=f"<div style='font-size:0.9em; color:#334155;'>{poe_status}</div>"),
                widgets.HBox(reg_btn_widgets, layout=widgets.Layout(flex_flow="row wrap")),
                fig_out,
            ],
            layout=widgets.Layout(padding="6px 8px", border="1px dashed #94A3B8", margin="4px 0"),
        )

        # Graduated Mastery Item + Progressive Hints
        item_spec = self.automaton.active_item_spec
        item_id = str(item_spec.get("id", ""))
        done_n = len(session.completed_graduated_items)
        tot_n = len(session.graduated_item_ids)
        unlocked_tier = session.item_hint_tiers_unlocked.get(item_id, 0)
        hint_blocks: List[str] = []
        for t_idx in range(1, unlocked_tier + 1):
            htxt = self.automaton._get_hint_for_tier(item_spec, t_idx)
            if htxt:
                hint_blocks.append(
                    f"<div style='background:#FEF9C3; padding:4px 8px; margin:3px 0; border-left:3px solid #CA8A04;'>"
                    f"<b>Hint Tier {t_idx}/3:</b> {self._fmt(htxt)}</div>"
                )

        grad_header = widgets.HTMLMath(
            value=(
                f"<div style='margin-top:6px;'><b>2. Graduated Mastery Check ({done_n}/{tot_n} completed) "
                f"&mdash; <code>{item_id}</code>:</b><br>{self._fmt(str(item_spec.get('prompt_markdown', '')))}</div>"
                + "".join(hint_blocks)
            )
        )
        input_box = self._build_item_input_widget(item_spec)
        return widgets.VBox([scaffold_card, poe_box, grad_header, input_box])

    def refresh(self) -> None:
        """Synchronize learner-agency buttons and rebuild the active state view."""
        self._sync_agency_controls()
        st = self.automaton.state

        msg_html = ""
        if self.last_outcome is not None and self.last_outcome.message:
            msg_html = (
                f"<div style='background:#F8FAFC; border-left:3px solid #64748B; padding:5px 8px; "
                f"margin-bottom:6px; font-size:0.93em;'><b>Diagnostic:</b> "
                f"{self._fmt(self.last_outcome.message)}</div>"
            )
        self.status_feedback_html.value = msg_html

        if st == AutomatonState.S1_DIAGNOSTIC_PROBE:
            probe = self.automaton.active_item_spec
            hdr = widgets.HTMLMath(
                value=(
                    f"<div style='background:#F0FDF4; border-left:4px solid #16A34A; padding:6px 10px;'>"
                    f"<b>Diagnostic Concept Probe (S1 &middot; <code>{probe.get('id', '')}</code>):</b><br>"
                    f"{self._fmt(str(probe.get('prompt_markdown', '')))}</div>"
                )
            )
            self.item_submit_btn.description = "Submit Probe Answer"
            self.body_box.children = (hdr, self._build_item_input_widget(probe))
            return

        if st == AutomatonState.S2_REMEDIAL_CLUSTER and self.automaton.active_cluster_session is not None:
            self.item_submit_btn.description = "Submit Cluster Step"
            self.body_box.children = (self._build_s2_cluster_view(),)
            return

        if st in (AutomatonState.S0_REFRESHED_ANCHOR_RETURN, AutomatonState.S0_ROOT_ANCHOR):
            item = self.automaton.active_item_spec
            bump = int(item.get("seed_bump", 0))
            badge = f"Refreshed Root Question (Seed +{bump})" if bump > 0 else "Active Root Question"
            hdr = widgets.HTMLMath(
                value=(
                    f"<div style='background:#EEF2FF; border-left:4px solid #4F46E5; padding:6px 10px;'>"
                    f"<b>{badge} &mdash; <code>{item.get('id', '')}</code>:</b><br>"
                    f"{self._fmt(str(item.get('prompt_markdown', '')))}</div>"
                )
            )
            self.item_submit_btn.description = "Submit Refreshed Answer"
            self.body_box.children = (hdr, self._build_item_input_widget(item))
            return

        if st == AutomatonState.MASTERED_COMPLETED:
            self.body_box.children = (
                widgets.HTMLMath(
                    value=(
                        "<div style='background:#DCFCE7; border-left:4px solid #15803D; padding:8px 10px;'>"
                        "<b>Mastery Verified:</b> Concept transfer confirmed and KV state updated.</div>"
                    )
                ),
            )
            return

        if st == AutomatonState.S3_DIAGNOSTIC_ESCALATION_DEFERRED:
            worked = ""
            if self.last_outcome and self.last_outcome.unlocked_hint_text:
                worked = f"<br><b>Worked Solution:</b> {self._fmt(self.last_outcome.unlocked_hint_text)}"
            self.body_box.children = (
                widgets.HTMLMath(
                    value=(
                        "<div style='background:#FEF2F2; border-left:4px solid #DC2626; padding:8px 10px;'>"
                        "<b>Queued for Spaced Review (S3):</b> Concept logged to Open Learner Model."
                        f"{worked}</div>"
                    )
                ),
            )
