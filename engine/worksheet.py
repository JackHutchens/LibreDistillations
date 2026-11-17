"""Core Worksheet runtime, interactive widget builders, and grading engine."""

from __future__ import annotations
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Callable, Mapping, Optional

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

try:
    import yaml
except ImportError:
    yaml = None

import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
import ipywidgets as widgets

from .figures import render_static_figure
from .tok_runtime import (
    AutomatonState,
    AutomatonStepOutcome,
    LearnerKVEvolutionEngine,
    PrerequisiteEdgeSpec,
    ToKRemediationWidget,
    TreeOfKnowledgeAutomaton,
)
from .code_runner import (
    SuiteExecutionReport,
    execute_student_submission,
)


class _SmartInt(int):
    """Int subclass that also exposes `.value` for compatibility with widget-style access."""

    def __new__(cls, val: int) -> "_SmartInt":
        obj = super().__new__(cls, int(val))
        obj.value = int(val)
        return obj


class _SmartFloat(float):
    """Float subclass that also exposes `.value` for compatibility with widget-style access."""

    def __new__(cls, val: float) -> "_SmartFloat":
        obj = super().__new__(cls, float(val))
        obj.value = float(val)
        return obj


class _SmartStr(str):
    """Str subclass that also exposes `.value` for compatibility with widget-style access."""

    def __new__(cls, val: str) -> "_SmartStr":
        obj = super().__new__(cls, str(val))
        obj.value = str(val)
        return obj


def wrap_control_value(val: Any) -> Any:
    """Wrap a scalar control value so both `controls[k]` and `controls[k].value` work."""
    if isinstance(val, bool):
        return _SmartInt(int(val))
    if isinstance(val, int):
        return _SmartInt(val)
    if isinstance(val, float):
        return _SmartFloat(val)
    if isinstance(val, str):
        return _SmartStr(val)
    return val


def wrap_controls_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Wrap all values in a controls dictionary with `.value`-compatible scalar types."""
    return {k: wrap_control_value(v) for k, v in d.items()}


def normalize_controls_spec(controls_raw: Any) -> dict[str, dict[str, Any]]:
    """Normalize `controls` from either a list of `{name, kind, ...}` or a dict `{name: {...}}`."""
    if not controls_raw:
        return {}
    if isinstance(controls_raw, Mapping):
        return {str(k): dict(v) for k, v in controls_raw.items()}
    if isinstance(controls_raw, list):
        out: dict[str, dict[str, Any]] = {}
        for item in controls_raw:
            d = dict(item)
            name = str(d.get("name") or d.get("id"))
            out[name] = d
        return out
    return {}


def normalize_angstrom(text: str) -> str:
    """Normalize all Angstrom symbols across both math and text contexts."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Specific units and corruptions
    specifics = [
        ("eV/Ų", "eV/Å^2"),
        ("eV/A.", "eV/Å."),
        ("eV/A ", "eV/Å "),
        ("eV/A,", "eV/Å,"),
        ("eV/A)", "eV/Å)"),
        ("eV/A\"", "eV/Å\""),
        ("eV/A'", "eV/Å'"),
        ("rad/A", "rad/Å"),
        (r"\mathrm{\AA^2/ps}", r"\text{Å}^2/\text{ps}"),
        (r"\\mathrm{\\AA^2/ps}", r"\\text{Å}^2/\\text{ps}"),
        (r"\mathrm{eV/\AA^2}", r"\text{eV/Å}^2"),
        (r"\\mathrm{eV/\\AA^2}", r"\\text{eV/Å}^2"),
        (r"\mathrm{\AA}^2", r"\text{Å}^2"),
        (r"\\mathrm{\\AA}^2", r"\\text{Å}^2"),
        (r"(\mathrm{\AA}^2)", "(Å²)"),
        (r"(\mathrm{\AA})", "(Å)"),
        (r"(\mathrm{Å})", "(Å)"),
        (r"(\mathrm{Å}^2)", "(Å²)"),
        (r"$\mathrm{\AA}$", r"$\text{Å}$"),
    ]
    for old, new in specifics:
        text = text.replace(old, new)

    # 2. Quad-backslash LaTeX commands
    text = re.sub(r'\\\\text\{(\s*)\\\\AA\s*\}', r'\\\\text{\1Å}', text)
    text = re.sub(r'\\\\mathrm\{(\s*)\\\\AA\s*\}', r'\\\\text{\1Å}', text)
    text = re.sub(r'\\\\text\{/AA\}', r'\\\\text{Å}', text)
    text = re.sub(r'\\\\mathrm\{\{(\s*)\\\\AA\s*\}\}', r'\\\\text{\1Å}', text)
    text = re.sub(r'\\\\text\{\s*\\\\mathrm\{\s*\\\\AA\s*\}\s*\}', r'\\\\text{Å}', text)
    text = re.sub(r'(\\\\text\{[^\}]*?)\\\\AA', r'\1Å', text)
    text = re.sub(r'(\\\\mathrm\{[^\}]*?)\\\\AA', r'\1Å', text)

    # 3. Double-backslash LaTeX commands
    text = re.sub(r'\\text\{(\s*)\\AA\s*\}', r'\\text{\1Å}', text)
    text = re.sub(r'\\mathrm\{(\s*)\\AA\s*\}', r'\\text{\1Å}', text)
    text = re.sub(r'\\text\{/AA\}', r'\\text{Å}', text)
    text = re.sub(r'\\mathrm\{\{(\s*)\\AA\s*\}\}', r'\\text{\1Å}', text)
    text = re.sub(r'\\text\{\s*\\mathrm\{\s*\\AA\s*\}\s*\}', r'\\text{Å}', text)
    text = re.sub(r'(\\text\{[^\}]*?)\\AA', r'\1Å', text)
    text = re.sub(r'(\\mathrm\{[^\}]*?)\\AA', r'\1Å', text)

    # 4. Bare \AA or \\AA inside math mode ($...$ or $$...$$)
    def replace_math(match: re.Match) -> str:
        m = match.group(0)
        m = re.sub(r'(?<![A-Za-z0-9_\\])\\\\AA\b', r'\\\\text{Å}', m)
        m = re.sub(r'(?<![A-Za-z0-9_\\])\\AA\b', r'\\text{Å}', m)
        m = re.sub(r'/AA\b', r'\\text{Å}', m)
        return m

    text = re.sub(r'(\$\$.*?\$\$|\$[^$\n]+?\$)', replace_math, text, flags=re.DOTALL)

    # 5. Outside math: /AA -> Å
    text = re.sub(r'(?<![A-Za-z0-9])/AA\b', 'Å', text)

    # 6. Outside math: bare \AA -> Å
    text = re.sub(r'(?<![A-Za-z0-9\\])\\\\AA\b', 'Å', text)
    text = re.sub(r'(?<![A-Za-z0-9\\])\\AA\b', 'Å', text)

    # 7. Clean nested text tags
    text = text.replace(r'\text{\text{Å}}', r'\text{Å}')
    text = text.replace(r'\\text{\\text{Å}}', r'\\text{Å}')
    text = text.replace(r'\mathrm{\text{Å}}', r'\text{Å}')
    text = text.replace(r'\\mathrm{\\text{Å}}', r'\\text{Å}')
    text = text.replace(r'\text{ \text{Å}}', r'\text{Å}')
    text = text.replace(r'\\text{ \\text{Å}}', r'\\text{Å}')

    # 8. Normalize double-escaped LaTeX commands (\\cmd -> \cmd)
    text = normalize_latex(text)

    return text


def normalize_latex(text: str) -> str:
    """Normalize double-escaped LaTeX command backslashes (\\\\cmd -> \\cmd)."""
    if not text:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return re.sub(r"\\{2,}([a-zA-Z]+)", r"\\\1", text)


def _format_latex_and_math(text: str) -> str:
    """Format markdown and preserve LaTeX math blocks ($...$ and $$...$$) for HTMLMath widgets."""
    if not text:
        return ""
    text = normalize_angstrom(str(text))
    text = normalize_latex(text)

    math_blocks: list[str] = []
    def save_math(m: re.Match) -> str:
        math_blocks.append(m.group(0))
        return f"MATHBLOCKXYZ{len(math_blocks)-1}ZYX"

    # Match $$...$$, $...$, \(...\), \[...\] (allowing inline math across wrapped newlines)
    pattern = r"(\$\$[\s\S]*?\$\$|\$[^\$]+?\$|\\\([\s\S]*?\\\)|\s*\\\[[\s\S]*?\\\])"
    protected = re.sub(pattern, save_math, text)

    # Convert markdown formatting to HTML without requiring external 'markdown' library
    try:
        import markdown
        html = markdown.markdown(protected)
        if html.startswith("<p>") and html.endswith("</p>") and html.count("<p>") == 1:
            html = html[3:-4]
    except ImportError:
        # Zero-dependency markdown formatting fallback
        html = re.sub(r"`([^`]+)`", r"<code>\1</code>", protected)
        html = re.sub(r"(\*\*|__)(.*?)\1", r"<b>\2</b>", html)
        html = re.sub(r"(\*|_)(.*?)\1", r"<i>\2</i>", html)
        html = html.replace("\n", "<br>")

    # Restore math blocks with HTML-escaped special characters (<, >, &).
    # In browser HTML, raw '<' followed by letters is parsed by the browser DOM parser
    # as an unclosed HTML tag (e.g., '$2 < \alpha$' becomes '<alpha>'), destroying the
    # math delimiters. Using &lt;, &gt;, &amp; ensures the DOM text node contains literal
    # '<', '>', '&', which MathJax then typesets without interference.
    for i, mb in enumerate(math_blocks):
        clean_mb = (
            mb.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        html = html.replace(f"MATHBLOCKXYZ{i}ZYX", clean_mb)

    return html


class Worksheet:
    """Interactive Jupyter worksheet loaded from a YAML specification and Concept module."""

    def __init__(
        self,
        spec: dict[str, Any],
        yaml_path: Path,
        concept_module: Any = None,
        course_v2_meta: Optional[dict[str, Any]] = None,
        v2_bank: Optional[dict[str, Any]] = None,
    ) -> None:
        # Unwrap top-level `worksheet:` key if present
        root = spec.get("worksheet", spec) if isinstance(spec, Mapping) else spec
        self.spec = dict(root)
        self.yaml_path = yaml_path
        self.base_dir = yaml_path.parent
        self.id: str = str(self.spec.get("id", yaml_path.stem))
        self.title: str = str(self.spec.get("title", self.id))
        self.estimated_minutes: int = int(
            self.spec.get("estimated_minutes") or self.spec.get("duration_minutes") or 120
        )
        self.learning_objectives: list[str] = list(self.spec.get("learning_objectives", []))
        self.shared_figures: dict[str, Any] = dict(self.spec.get("shared_figures") or {})
        self.sections: list[dict[str, Any]] = list(self.spec.get("sections", []))
        self.concept_module = concept_module
        self.shuffle_options: bool = bool(self.spec.get("shuffle_options", False))

        # Index sections and assign sequential numbers to problems
        self._section_map: dict[str, dict[str, Any]] = {}
        self._problem_number: dict[str, int] = {}
        p_idx = 0
        for sec in self.sections:
            sid = sec["id"]
            self._section_map[sid] = sec
            if sec.get("type") == "problem":
                p_idx += 1
                self._problem_number[sid] = p_idx

        # Session & persistence state
        session_hash = hashlib.sha1(f"{self.id}:{self.yaml_path}".encode("utf-8")).hexdigest()[:8]
        self.session_id = session_hash
        self._records_dir = self.base_dir / ".worksheet_records"
        self._record_file = self._records_dir / f"{self.id}_record.json"
        self.code_records: dict[str, dict[str, Any]] = {}
        self.records: dict[str, dict[str, Any]] = self._load_records()

        # V2 companion bank & Learner KV / Tree-of-Knowledge initialization
        self.course_v2_meta: dict[str, Any] = (
            dict(course_v2_meta)
            if course_v2_meta is not None
            else self._discover_json_companion("course_fragility_and_kvs.json")
        )
        self.v2_bank: dict[str, Any] = (
            dict(v2_bank)
            if v2_bank is not None
            else self._discover_json_companion(f"{yaml_path.stem}.v2.json")
        )
        self.code_exercise_list: list[dict[str, Any]] = list(
            self.v2_bank.get("code_exercises") or []
        )
        self.code_exercises: dict[str, dict[str, Any]] = {
            str(ex["exercise_id"]): dict(ex)
            for ex in self.code_exercise_list
            if isinstance(ex, Mapping) and "exercise_id" in ex
        }
        self._active_remediation_widgets: dict[str, ToKRemediationWidget] = {}
        self._item_automatons: dict[str, TreeOfKnowledgeAutomaton] = {}
        self.kv_engine: LearnerKVEvolutionEngine = self._init_v2_kv_engine()
        self._v2_curriculum_spec: dict[str, Any] = self._build_merged_v2_curriculum_spec()

    def _discover_json_companion(self, filename: str) -> dict[str, Any]:
        candidates = [
            self.yaml_path.parent.parent / "v2_banks" / filename,
            self.yaml_path.parent / "v2_banks" / filename,
        ]
        for cpath in candidates:
            if cpath.is_file():
                try:
                    with open(cpath, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                        if isinstance(data, dict):
                            return data
                except Exception:
                    return {}
        return {}

    def _default_primary_concept_for_section(self, sec: Mapping[str, Any]) -> str:
        tb = sec.get("taught_by")
        if isinstance(tb, list) and tb:
            first_tb = str(tb[0])
        elif isinstance(tb, str) and tb:
            first_tb = tb
        else:
            first_tb = "general"
        if "::" in first_tb:
            return first_tb
        return f"{self.id}::{first_tb}"

    def _init_v2_kv_engine(self) -> LearnerKVEvolutionEngine:
        ckv_ids = [f"CKV-{k}" for k in range(8)]
        concept_ids: list[str] = list(ckv_ids)

        for cid in self.v2_bank.get("ckv_vector_space") or []:
            s_cid = str(cid)
            if s_cid not in concept_ids:
                concept_ids.append(s_cid)

        lesson_summaries = self.course_v2_meta.get("lesson_summaries") or {}
        l_sum = lesson_summaries.get(self.id) or {}
        for cid in l_sum.get("micro_concept_ids") or []:
            if str(cid) not in concept_ids:
                concept_ids.append(str(cid))

        for sec in self.sections:
            if sec.get("type") in ("markdown_section", "concept_cell", "markdown", "concept", "intro"):
                full_cid = f"{self.id}::{sec['id']}"
                if full_cid not in concept_ids:
                    concept_ids.append(full_cid)

        edges: list[PrerequisiteEdgeSpec] = []
        seen_edges: set[tuple[str, str]] = set()
        raw_edge_lists = [
            self.course_v2_meta.get("ckv_macro_prerequisite_edges") or [],
            self.course_v2_meta.get("prerequisite_edges") or [],
            self.v2_bank.get("concept_prerequisite_edges") or [],
        ]
        for edge_list in raw_edge_lists:
            for e in edge_list:
                if not isinstance(e, Mapping):
                    continue
                src = str(e.get("source_concept_id", ""))
                dst = str(e.get("target_concept_id", ""))
                if not src or not dst or (src, dst) in seen_edges:
                    continue
                seen_edges.add((src, dst))
                edges.append(
                    PrerequisiteEdgeSpec(
                        source_concept_id=src,
                        target_concept_id=dst,
                        weight=float(e.get("weight", 1.0)),
                        is_weak_bridge=bool(e.get("is_weak_bridge", False)),
                    )
                )

        return LearnerKVEvolutionEngine(
            concept_ids=concept_ids,
            prerequisite_edges=edges,
            session_id=self.session_id,
        )

    def _expand_concept_aliases(self, mapping_obj: Mapping[str, Any]) -> dict[str, Any]:
        """Support `<lesson_stem>::<sec>`, `LXX::<sec>`, and bare `<sec>` keys seamlessly."""
        out: dict[str, Any] = {}
        m_num = re.match(r"^lesson_(\d+)", self.id)
        short_prefix = f"L{int(m_num.group(1)):02d}" if m_num else ""
        for raw_key, val in mapping_obj.items():
            k = str(raw_key)
            out[k] = copy.deepcopy(val)
            if "::" in k:
                prefix, suffix = k.split("::", 1)
                full_k = f"{self.id}::{suffix}"
                out.setdefault(full_k, copy.deepcopy(val))
                if short_prefix:
                    out.setdefault(f"{short_prefix}::{suffix}", copy.deepcopy(val))
            else:
                out.setdefault(f"{self.id}::{k}", copy.deepcopy(val))
                if short_prefix:
                    out.setdefault(f"{short_prefix}::{k}", copy.deepcopy(val))
        return out

    def _build_merged_v2_curriculum_spec(self) -> dict[str, Any]:
        prob_diags = self.v2_bank.get("problem_diagnostics") or {}
        merged_sections: list[dict[str, Any]] = []
        for sec in self.sections:
            sec_copy = copy.deepcopy(dict(sec))
            if sec_copy.get("type") == "problem":
                pid = str(sec_copy["id"])
                sec_copy.setdefault(
                    "primary_concept_id", self._default_primary_concept_for_section(sec_copy)
                )
                diag = prob_diags.get(pid)
                if isinstance(diag, Mapping):
                    for dk, dv in diag.items():
                        sec_copy[str(dk)] = copy.deepcopy(dv)
            merged_sections.append(sec_copy)

        for ex_id, ex in self.code_exercises.items():
            routing = (
                ex.get("code_gap_routing")
                or (ex.get("v2_code_exercise_spec") or {}).get("code_gap_routing")
                or {}
            )
            merged_sections.append(
                {
                    "id": ex_id,
                    "type": "problem",
                    "question_type": "code_exercise",
                    "difficulty": str(ex.get("difficulty", "medium")),
                    "taught_by": str(ex.get("primary_concept_id", f"{self.id}::code")),
                    "primary_concept_id": str(ex.get("primary_concept_id", f"{self.id}::code")),
                    "prompt_markdown": str(ex.get("prompt_markdown", ex.get("title", ex_id))),
                    "ckv_loadings": dict(ex.get("ckv_loadings") or {}),
                    "code_exercise_spec": {
                        "function_name": str(ex.get("function_name", "")),
                        "signature": str(ex.get("signature", "")),
                        "starter_code": str(ex.get("starter_code", "")),
                        "hidden_suite_ref": str(
                            (ex.get("sealed_test_bundle") or {}).get("suite_id", ex_id)
                        ),
                        "code_gap_routing": copy.deepcopy(dict(routing)),
                    },
                }
            )

        probes = self._expand_concept_aliases(self.v2_bank.get("diagnostic_probe_pools") or {})
        clusters = self._expand_concept_aliases(self.v2_bank.get("remedial_clusters") or {})
        for cid, cspec in clusters.items():
            if isinstance(cspec, dict):
                cspec["target_concept_id"] = cid

        return {
            "worksheet": {
                "id": self.id,
                "title": self.title,
                "estimated_minutes": self.estimated_minutes,
                "learning_objectives": list(self.learning_objectives),
                "sections": merged_sections,
                "diagnostic_probe_pools": probes,
                "remedial_clusters": clusters,
            }
        }

    def get_item_automaton(self, item_id: str) -> Optional[TreeOfKnowledgeAutomaton]:
        """Return or initialize a dedicated pushdown automaton for `item_id` sharing `self.kv_engine`."""
        if not self.v2_bank and not (
            self._v2_curriculum_spec["worksheet"]["diagnostic_probe_pools"]
            or self._v2_curriculum_spec["worksheet"]["remedial_clusters"]
        ):
            return None
        if item_id not in self._item_automatons:
            lesson_summaries = self.course_v2_meta.get("lesson_summaries") or {}
            l_sum = lesson_summaries.get(self.id) or {}
            d_trig = 2 if int(l_sum.get("max_d_trigger", 1)) >= 2 else 1
            auto = TreeOfKnowledgeAutomaton(
                curriculum_spec=self._v2_curriculum_spec,
                kv_engine=self.kv_engine,
                max_recursion_depth=d_trig,
            )
            if item_id in auto.root_problems:
                auto.select_root_question(item_id)
            self._item_automatons[item_id] = auto
        return self._item_automatons[item_id]

    @classmethod
    def load(cls, yaml_path: str | Path) -> "Worksheet":
        """Load a worksheet from a YAML/JSON file path and dynamically import its Concepts module."""
        repo_root = Path(__file__).resolve().parent.parent
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        p = Path(yaml_path)
        if not p.is_file():
            filename = Path(yaml_path).name
            alt_names = [filename]
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                alt_names.append(Path(filename).with_suffix(".json").name)
            elif filename.endswith(".json"):
                alt_names.append(Path(filename).with_suffix(".yaml").name)
                alt_names.append(Path(filename).with_suffix(".yml").name)

            candidates = []
            for fname in alt_names:
                candidates.extend([
                    Path(fname),
                    Path.cwd() / fname,
                    Path.cwd().parent / "worksheets" / fname,
                    Path.cwd() / "worksheets" / fname,
                ])
                candidates.extend(repo_root.glob(f"courses/*/worksheets/{fname}"))
                candidates.extend([
                    repo_root / fname,
                    repo_root / "worksheets" / fname,
                    Path.cwd() / "Target Format Reference" / fname,
                    Path.cwd() / "Generated Worksheets" / fname,
                ])
                cur = Path.cwd().resolve()
                while cur != cur.parent:
                    candidates.append(cur / fname)
                    candidates.append(cur / "worksheets" / fname)
                    candidates.extend(cur.glob(f"courses/*/worksheets/{fname}"))
                    cur = cur.parent

            for c in candidates:
                if c.is_file():
                    p = c
                    break
            else:
                raise FileNotFoundError(f"Could not locate worksheet file: {yaml_path}")

        p = p.resolve()
        json_path = p.with_suffix(".json")

        if p.suffix == ".json":
            with open(p, "r", encoding="utf-8") as fh:
                spec = json.load(fh)
        elif json_path.is_file() and yaml is None:
            with open(json_path, "r", encoding="utf-8") as fh:
                spec = json.load(fh)
        elif yaml is not None:
            with open(p, "r", encoding="utf-8") as fh:
                spec = yaml.safe_load(fh)
        elif json_path.is_file():
            with open(json_path, "r", encoding="utf-8") as fh:
                spec = json.load(fh)
        else:
            raise ImportError(
                f"Cannot load '{p.name}': PyYAML is not installed and no pre-compiled '{json_path.name}' "
                "was found. Please install pyyaml or compile worksheets using 'cram build'."
            )

        concept_candidates = [
            p.parent.parent / "concepts" / f"{p.stem}.py",
            p.parent.parent / "Concepts" / f"{p.stem}.py",
            p.parent / "concepts" / f"{p.stem}.py",
            p.parent / "Concepts" / f"{p.stem}.py",
            p.parent / f"{p.stem}.py",
        ]
        concept_path = None
        for cc in concept_candidates:
            if cc.is_file():
                concept_path = cc
                break

        concept_mod = None
        if concept_path and concept_path.is_file():
            mod_name = f"concepts_{p.stem}"
            spec_obj = importlib.util.spec_from_file_location(mod_name, concept_path)
            if spec_obj and spec_obj.loader:
                concept_mod = importlib.util.module_from_spec(spec_obj)
                spec_obj.loader.exec_module(concept_mod)

        return cls(spec=spec, yaml_path=p, concept_module=concept_mod)

    def _load_records(self) -> dict[str, dict[str, Any]]:
        if self._record_file.is_file():
            try:
                with open(self._record_file, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data.get("code_exercises"), Mapping):
                        self.code_records = dict(data["code_exercises"])
                    return dict(data.get("answers", {}))
            except Exception:
                return {}
        return {}

    def _save_records(self) -> None:
        try:
            self._records_dir.mkdir(parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "worksheet_id": self.id,
                "title": self.title,
                "session_id": self.session_id,
                "answers": self.records,
            }
            if getattr(self, "code_records", None):
                payload["code_exercises"] = self.code_records
            with open(self._record_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception:
            pass

    def start(self) -> None:
        """Display the worksheet header, learning objectives, and session status."""
        lines = [
            f"# {self.title}",
            "",
            f"_Session {self.session_id} · Estimated time: ~{self.estimated_minutes} minutes · {len(self._problem_number)} problems_",
            "",
            "**Recording your answers.** Submit each problem as you go. Your answers are saved to the workspace as you submit, *and* the running record at the bottom of the notebook keeps your progress. When you finish, run the last two cells — **`show_record()`** and **`finalize()`** — to record your participation. **If you reopened this workspace** (or the kernel restarted), use **Kernel → Restart Kernel and Run All Cells** before finalizing, so every answer is reloaded.",
        ]
        if self.learning_objectives:
            lines.append("")
            lines.append("### Learning Objectives")
            for obj in self.learning_objectives:
                lines.append(f"- {normalize_angstrom(obj)}")

        if self.records:
            lines.append("")
            lines.append(
                f"_Continuing your earlier session — **{len(self.records)}** answer(s) restored from this workspace._"
            )
        display(Markdown("$\\def\\AA{\\text{Å}}$\n\n" + "\n".join(lines)))

    def render_section(self, section_id: str) -> None:
        """Render a markdown/intro section, concept cell, or problem by its ID."""
        if section_id not in self._section_map:
            raise KeyError(f"Section {section_id!r} not found in worksheet {self.id!r}")
        sec = self._section_map[section_id]
        stype = sec.get("type")

        if stype in ("markdown_section", "markdown", "intro"):
            self._render_markdown_section(sec)
        elif stype in ("concept_cell", "concept"):
            self._render_concept_cell(sec)
        elif stype == "problem":
            self._render_problem(sec)
        else:
            raise ValueError(f"Unknown section type {stype!r} for section {section_id!r}")

    def _render_markdown_section(self, sec: dict[str, Any]) -> None:
        body = (sec.get("body_markdown") or sec.get("markdown") or "").strip()
        title = (sec.get("title") or "").strip()
        if title and not body.startswith("#"):
            text = f"## {title}\n\n{body}"
        else:
            text = body
        display(Markdown(normalize_angstrom(text)))
        if sec.get("figure"):
            self._display_figure_widget(sec["figure"])

    def _display_figure_widget(self, fig_spec: dict[str, Any]) -> None:
        out = widgets.Output()
        with out:
            fig, caption = render_static_figure(
                fig_spec,
                shared_figures=self.shared_figures,
                concept_module=self.concept_module,
            )
            plt.show()
            plt.close(fig)
            if caption:
                display(Markdown(f"_{normalize_angstrom(caption)}_"))
        display(out)

    def _render_concept_cell(self, sec: dict[str, Any]) -> None:
        title = sec.get("title", "")
        if title:
            display(Markdown(f"### {title}"))
        desc = (sec.get("description_markdown") or "").strip()
        if desc:
            display(Markdown(normalize_angstrom(desc)))

        controls_spec = normalize_controls_spec(sec.get("controls"))
        render_fn_name = sec.get("render_function")
        layout_mode = sec.get("layout", "figure_left")

        control_widgets: dict[str, widgets.Widget] = {}
        ui_controls: list[widgets.Widget] = []

        for cname, cspec in controls_spec.items():
            kind = cspec.get("kind", "slider")
            label = cspec.get("label", cname)
            default = cspec.get("default")

            lbl_widget = widgets.HTMLMath(
                f"<div style='font-weight:600; font-size:0.92em; margin-top:4px;'>{_format_latex_and_math(label)}</div>"
            )
            if kind == "dropdown":
                options = cspec.get("options", [])
                if options and isinstance(options[0], dict):
                    opts = [(o.get("label", o["value"]), o["value"]) for o in options]
                    default_val = default if default is not None else opts[0][1]
                else:
                    opts = list(options)
                    default_val = default if default is not None else (opts[0] if opts else None)
                w = widgets.Dropdown(
                    options=opts,
                    value=default_val,
                    layout=widgets.Layout(width="96%"),
                )
            elif kind == "slider":
                cmin = cspec.get("min", 0)
                cmax = cspec.get("max", 10)
                cstep = cspec.get("step", 1)
                cdef = default if default is not None else cmin
                is_float = any(isinstance(x, float) for x in (cmin, cmax, cstep, cdef))
                if is_float:
                    w = widgets.FloatSlider(
                        value=float(cdef),
                        min=float(cmin),
                        max=float(cmax),
                        step=float(cstep),
                        continuous_update=False,
                        readout_format=cspec.get("format", ".2f"),
                        layout=widgets.Layout(width="96%"),
                    )
                else:
                    w = widgets.IntSlider(
                        value=int(cdef),
                        min=int(cmin),
                        max=int(cmax),
                        step=int(cstep),
                        continuous_update=False,
                        layout=widgets.Layout(width="96%"),
                    )
            else:
                w = widgets.Text(value=str(default or ""))

            control_widgets[cname] = w
            ui_controls.extend([lbl_widget, w])

        fig_out = widgets.Output()
        caption_out = widgets.Output()

        def _update_plot(*_args: Any) -> None:
            current_vals = wrap_controls_dict({k: cw.value for k, cw in control_widgets.items()})
            fig_out.clear_output(wait=True)
            caption_out.clear_output(wait=True)

            if self.concept_module is None or not hasattr(self.concept_module, render_fn_name):
                with caption_out:
                    display(Markdown(f"_Missing render function `{render_fn_name}`._"))
                return

            fn = getattr(self.concept_module, render_fn_name)
            figsize = (6.2, 4.2) if layout_mode == "figure_left" else (7.4, 4.2)
            with fig_out:
                fig, ax = plt.subplots(figsize=figsize)
                caption_md = fn(current_vals, ax)
                fig.tight_layout()
                plt.show()
                plt.close(fig)
            if caption_md:
                with caption_out:
                    display(Markdown(normalize_angstrom(str(caption_md))))

        for cw in control_widgets.values():
            cw.observe(_update_plot, names="value")

        _update_plot()

        if layout_mode == "figure_top":
            ctrl_box = widgets.VBox(ui_controls, layout=widgets.Layout(width="100%", padding="0 0 0.5em 0"))
            container = widgets.VBox([ctrl_box, fig_out, caption_out])
        else:
            left_box = widgets.VBox(
                [fig_out],
                layout=widgets.Layout(padding="0 1em 0 0", width="60%"),
            )
            right_box = widgets.VBox(
                [*ui_controls, caption_out],
                layout=widgets.Layout(width="40%"),
            )
            container = widgets.HBox([left_box, right_box])

        display(container)

        takeaway = (sec.get("takeaway_markdown") or "").strip()
        if takeaway:
            display(Markdown(normalize_angstrom(takeaway)))

    def _render_problem(self, sec: dict[str, Any]) -> None:
        pid = sec["id"]
        qnum = self._problem_number.get(pid, 0)
        diff = sec.get("difficulty", "medium")
        hint_md = (sec.get("hint_markdown") or "").strip()
        prompt_md = (sec.get("prompt_markdown") or "").strip()
        rationale_md = (sec.get("rationale_markdown") or "").strip()
        qtype = sec.get("question_type", "single_select")

        header = f"**Q{qnum} · {diff}**" + (" · 💡" if hint_md else "")
        display(Markdown(header))
        display(Markdown(normalize_angstrom(prompt_md)))

        if hint_md:
            norm_hint = normalize_angstrom(hint_md)
            hint_html = (
                '<details style="margin: 0.5em 0;">\n'
                '<summary style="cursor: pointer; user-select: none; display: inline-block; '
                'padding: 0.3em 0.7em; margin: 0.2em 0; background: #EEF0F7; border-radius: 4px; '
                'font-weight: 500;">💡 Show hint</summary>\n\n'
                f'<div style="padding: 0.5em 1em;">\n\n{norm_hint}\n\n</div>\n</details>'
            )
            display(Markdown(hint_html))

        if sec.get("figure"):
            self._display_figure_widget(sec["figure"])

        existing = self.records.get(pid)
        if existing and existing.get("completed"):
            display(
                Markdown(
                    "_Restored from your earlier session — already answered. Your recorded credit is shown below; you don't need to redo this problem._"
                )
            )

        feedback_out = widgets.Output()
        remediation_out = widgets.Output()
        primary_cid = self._default_primary_concept_for_section(sec)

        def _handle_v2_remediation_step(
            raw_submission: Any,
            is_correct_v1: bool,
            on_mastered_ui_lock: Callable[[], None],
        ) -> None:
            if is_correct_v1:
                remediation_out.clear_output(wait=True)
                try:
                    self.kv_engine.update_on_observation(
                        concept_id=primary_cid,
                        is_correct=True,
                        cri_score=3.5,
                        propagate_laplacian=True,
                    )
                    self.kv_engine.append_telemetry_event(
                        event_type="V1_PROBLEM_MASTERED",
                        item_id=pid,
                        automaton_state=AutomatonState.MASTERED_COMPLETED.value,
                        latency_sec=10.0,
                        cri=3.5,
                        hints_used=(),
                        payload={"tries": self.records.get(pid, {}).get("tries", 1)},
                    )
                except Exception:
                    pass
                return

            auto = self.get_item_automaton(pid)
            if auto is None:
                return

            try:
                if auto.state not in (
                    AutomatonState.S0_ROOT_ANCHOR,
                    AutomatonState.S0_REFRESHED_ANCHOR_RETURN,
                ):
                    auto.select_root_question(pid)
                sub_val = (
                    next(iter(raw_submission))
                    if (qtype == "single_select" and isinstance(raw_submission, set) and len(raw_submission) == 1)
                    else (sorted(raw_submission) if isinstance(raw_submission, set) else raw_submission)
                )
                outcome = auto.submit_response(sub_val, cri_score=3.0)
            except Exception:
                return

            def _on_tok_resolved(res_outcome: AutomatonStepOutcome) -> None:
                if res_outcome.current_state == AutomatonState.MASTERED_COMPLETED:
                    rec_now = self.records.get(pid, {"tries": 1, "completed": False, "credit": 0.0})
                    rec_now["completed"] = True
                    rec_now["credit"] = max(float(rec_now.get("credit", 0.0)), float(auto.completed_roots.get(pid, 0.75)))
                    self.records[pid] = rec_now
                    self._save_records()
                    on_mastered_ui_lock()

            with remediation_out:
                remediation_out.clear_output(wait=True)
                if outcome.slip_nudge_emitted:
                    display(Markdown(f"> **Slip Check:** {normalize_angstrom(outcome.message)}"))
                elif auto.state in (
                    AutomatonState.S1_DIAGNOSTIC_PROBE,
                    AutomatonState.S2_REMEDIAL_CLUSTER,
                    AutomatonState.S0_REFRESHED_ANCHOR_RETURN,
                    AutomatonState.S3_DIAGNOSTIC_ESCALATION_DEFERRED,
                ):
                    tok_w = ToKRemediationWidget(
                        automaton=auto,
                        concept_module=self.concept_module,
                        format_html_fn=_format_latex_and_math,
                        wrap_controls_fn=wrap_controls_dict,
                        on_root_resolved=_on_tok_resolved,
                        initial_outcome=outcome,
                    )
                    self._active_remediation_widgets[pid] = tok_w
                    display(tok_w.container)

        if qtype in ("single_select", "multi_select"):
            options = sec.get("options", [])
            correct_raw = sec.get("correct")
            if isinstance(correct_raw, list):
                correct_set = {str(x) for x in correct_raw}
            else:
                correct_set = {str(correct_raw)}

            checkboxes: dict[str, widgets.Checkbox] = {}
            rows: list[widgets.Widget] = []
            saved_sel = set(existing.get("selected", [])) if existing else set()

            display_options = list(options)
            if self.shuffle_options and len(display_options) > 1:
                seed_str = f"{self.session_id}:{pid}:options"
                seed_val = int(hashlib.sha256(seed_str.encode("utf-8")).hexdigest()[:8], 16)
                rng = random.Random(seed_val)
                rng.shuffle(display_options)

            for opt in display_options:
                oid = str(opt["id"])
                otext = str(opt.get("text") or opt.get("text_markdown") or "")
                cb = widgets.Checkbox(
                    value=(oid in saved_sel),
                    disabled=bool(existing and existing.get("completed")),
                    indent=False,
                    layout=widgets.Layout(margin="2px 8px 2px 0", width="24px"),
                )
                checkboxes[oid] = cb
                formatted_opt = _format_latex_and_math(otext)
                lbl = widgets.HTMLMath(
                    f"<div style='line-height:1.4;'><b>({oid.upper()})</b> {formatted_opt}</div>",
                    layout=widgets.Layout(width="auto"),
                )
                rows.append(widgets.HBox([cb, lbl], layout=widgets.Layout(align_items="flex-start", margin="4px 0")))

            if qtype == "single_select":
                def _make_exclusivity_handler(active_id: str):
                    def _on_toggle(change: dict[str, Any]) -> None:
                        if change["new"]:
                            for other_id, other_cb in checkboxes.items():
                                if other_id != active_id and other_cb.value:
                                    other_cb.value = False
                    return _on_toggle

                for oid, cb in checkboxes.items():
                    cb.observe(_make_exclusivity_handler(oid), names="value")

            submit_btn = widgets.Button(
                description="Submit Answer",
                button_style="primary",
                disabled=bool(existing and existing.get("completed")),
                layout=widgets.Layout(margin="8px 0 4px 0", width="150px"),
            )

            def _lock_choice_ui() -> None:
                for cb in checkboxes.values():
                    cb.disabled = True
                submit_btn.disabled = True

            def _on_submit_choice(_btn: Any) -> None:
                chosen = {oid for oid, cb in checkboxes.items() if cb.value}
                if not chosen:
                    with feedback_out:
                        feedback_out.clear_output(wait=True)
                        display(Markdown("_Please select an option before submitting._"))
                    return

                rec = self.records.get(pid, {"tries": 0, "completed": False, "credit": 0.0})
                rec["tries"] = int(rec.get("tries", 0)) + 1
                rec["selected"] = sorted(chosen)
                is_correct = chosen == correct_set

                if is_correct:
                    tries = rec["tries"]
                    credit = 1.0 if tries == 1 else max(0.5, 1.0 - 0.25 * (tries - 1))
                    rec["completed"] = True
                    rec["credit"] = round(credit, 2)
                    _lock_choice_ui()
                self.records[pid] = rec
                self._save_records()

                with feedback_out:
                    feedback_out.clear_output(wait=True)
                    if is_correct:
                        t_word = "try" if rec["tries"] == 1 else "tries"
                        display(
                            Markdown(
                                f"**✓ Correct ({rec['tries']} {t_word})** — recorded credit {rec['credit']:.2f}\n\n**Rationale:** {normalize_angstrom(rationale_md)}"
                            )
                        )
                    else:
                        display(
                            Markdown(
                                f"**✗ Not quite (attempt {rec['tries']})** — review the hint or concept section above and try again."
                            )
                        )
                _handle_v2_remediation_step(chosen, is_correct, _lock_choice_ui)

            submit_btn.on_click(_on_submit_choice)
            display(widgets.VBox([*rows, submit_btn]))

        elif qtype == "numeric":
            corr_val = sec["correct"]
            if isinstance(corr_val, Mapping):
                target = float(corr_val.get("value", corr_val.get("answer", 0.0)))
                tol = float(corr_val.get("tolerance", sec.get("tolerance", 1e-2)))
            else:
                target = float(corr_val)
                tol = float(sec.get("tolerance", 1e-2))
            saved_val = existing.get("value", 0.0) if existing else 0.0

            num_input = widgets.FloatText(
                value=float(saved_val),
                description="Answer:",
                disabled=bool(existing and existing.get("completed")),
                layout=widgets.Layout(width="240px", margin="4px 8px 4px 0"),
            )

            unit_str = sec.get("unit") or sec.get("units")
            unit_widgets: list[widgets.Widget] = []
            if unit_str:
                formatted_unit = _format_latex_and_math(unit_str)
                unit_lbl = widgets.HTMLMath(
                    f"<span style='line-height:2.2; margin-right:12px; color:#444;'><b>{formatted_unit}</b></span>"
                )
                unit_widgets.append(unit_lbl)

            submit_btn = widgets.Button(
                description="Submit Answer",
                button_style="primary",
                disabled=bool(existing and existing.get("completed")),
                layout=widgets.Layout(width="150px", margin="4px 0"),
            )

            def _lock_numeric_ui() -> None:
                num_input.disabled = True
                submit_btn.disabled = True

            def _on_submit_numeric(_btn: Any) -> None:
                val = float(num_input.value)
                rec = self.records.get(pid, {"tries": 0, "completed": False, "credit": 0.0})
                rec["tries"] = int(rec.get("tries", 0)) + 1
                rec["value"] = val
                is_correct = abs(val - target) <= tol

                if is_correct:
                    tries = rec["tries"]
                    credit = 1.0 if tries == 1 else max(0.5, 1.0 - 0.25 * (tries - 1))
                    rec["completed"] = True
                    rec["credit"] = round(credit, 2)
                    _lock_numeric_ui()
                self.records[pid] = rec
                self._save_records()

                with feedback_out:
                    feedback_out.clear_output(wait=True)
                    if is_correct:
                        t_word = "try" if rec["tries"] == 1 else "tries"
                        display(
                            Markdown(
                                f"**✓ Correct ({rec['tries']} {t_word})** — recorded credit {rec['credit']:.2f}\n\n**Rationale:** {normalize_angstrom(rationale_md)}"
                            )
                        )
                    else:
                        display(
                            Markdown(
                                f"**✗ Not quite (attempt {rec['tries']})** — check your units, rounding, or formula and try again (tolerance $\\pm {tol:g}$)."
                            )
                        )
                _handle_v2_remediation_step(val, is_correct, _lock_numeric_ui)

            submit_btn.on_click(_on_submit_numeric)
            display(widgets.HBox([num_input, *unit_widgets, submit_btn], layout=widgets.Layout(align_items="center")))

        if existing and existing.get("completed"):
            t_word = "try" if existing.get("tries", 1) == 1 else "tries"
            display(
                Markdown(
                    f"**✓ Correct ({existing.get('tries', 1)} {t_word})** — recorded credit {existing.get('credit', 1.0):.2f}\n\n**Rationale:** {normalize_angstrom(rationale_md)}"
                )
            )
        else:
            display(feedback_out)
            display(remediation_out)

    def render_code_exercise(self, exercise_id: str) -> None:
        """Render the prompt, signature, and public test suite manifest for a V2 coding exercise."""
        if exercise_id not in self.code_exercises:
            raise KeyError(f"Coding exercise {exercise_id!r} not found in worksheet {self.id!r}")
        ex = self.code_exercises[exercise_id]
        title = str(ex.get("title") or exercise_id)
        diff = str(ex.get("difficulty", "medium"))
        sig = str(ex.get("signature") or f"def {ex.get('function_name', 'solution')}(...):")
        prompt = normalize_angstrom(str(ex.get("prompt_markdown") or ex.get("pedagogical_objective") or ""))
        pub_tests = ex.get("public_test_names") or [
            m.get("public_test_name", m.get("test_id", ""))
            for m in ((ex.get("sealed_test_bundle") or {}).get("public_manifest") or [])
        ]

        md_lines = [
            f"### 💻 Coding Exercise `{exercise_id}` — {title} (`{diff}`)",
            "",
            prompt,
            "",
            f"**Required Function Signature:** `{sig}`",
        ]
        if pub_tests:
            md_lines.append("")
            md_lines.append("**Verification Suite (3-Tier Physical Invariants):**")
            for tname in pub_tests:
                md_lines.append(f"- {normalize_angstrom(str(tname))}")

        display(Markdown("\n".join(md_lines)))

    def check_code(
        self,
        exercise_id: str,
        student_fn_or_source: Any,
        cri_score: float = 3.0,
    ) -> SuiteExecutionReport:
        """
        Execute `student_fn_or_source` (live callable or Python source string) against
        the exercise's sealed 3-tier test bundle, update Learner KV state & telemetry,
        and mount Tree-of-Knowledge remediation on conceptual physics gaps.
        """
        if exercise_id not in self.code_exercises:
            raise KeyError(f"Coding exercise {exercise_id!r} not found in worksheet {self.id!r}")
        ex = self.code_exercises[exercise_id]
        bundle = ex["sealed_test_bundle"]

        crec = self.code_records.get(
            exercise_id, {"tries": 0, "completed": False, "credit": 0.0}
        )
        attempt_idx = int(crec.get("tries", 0))
        report = execute_student_submission(
            bundle=bundle,
            student_fn_or_source=student_fn_or_source,
            session_id=self.session_id,
            attempt_index=attempt_idx,
        )

        crec["tries"] = attempt_idx + 1
        crec["passed_count"] = report.passed_count
        crec["total_count"] = report.total_count
        crec["t2_rng_seed_hex"] = report.t2_rng_seed_hex
        if report.all_passed:
            tries = crec["tries"]
            credit = 1.0 if tries == 1 else max(0.5, 1.0 - 0.15 * (tries - 1))
            crec["completed"] = True
            crec["credit"] = round(credit, 2)
        self.code_records[exercise_id] = crec
        self._save_records()

        primary_cid = str(ex.get("primary_concept_id") or f"{self.id}::code")
        coactivated = list((ex.get("ckv_loadings") or {}).keys())
        self.kv_engine.update_on_observation(
            concept_id=primary_cid,
            is_correct=report.all_passed,
            cri_score=float(cri_score),
            coactivated_concepts=coactivated,
            propagate_laplacian=True,
        )
        self.kv_engine.append_telemetry_event(
            event_type="CODE_EXERCISE_SUBMISSION",
            item_id=exercise_id,
            automaton_state=(
                AutomatonState.MASTERED_COMPLETED.value
                if report.all_passed
                else AutomatonState.S0_ROOT_ANCHOR.value
            ),
            latency_sec=15.0,
            cri=float(cri_score),
            hints_used=(),
            payload={
                "passed_count": report.passed_count,
                "total_count": report.total_count,
                "all_passed": report.all_passed,
                "t2_rng_seed_hex": report.t2_rng_seed_hex,
                "report_sha256": report.report_sha256,
                "triggered_conceptual_gap_tags": list(report.triggered_conceptual_gap_tags),
            },
        )

        display(Markdown(f"```text\n{report.format_student_card()}\n```"))

        if not report.all_passed and report.triggered_conceptual_gap_tags:
            first_gap = report.triggered_conceptual_gap_tags[0]
            first_diag = next(
                (tr.failure_diagnostic for tr in report.test_reports if not tr.passed),
                "Hidden invariant check failed.",
            )
            auto = self.get_item_automaton(exercise_id)
            if auto is not None and exercise_id in auto.root_problems:
                try:
                    auto.select_root_question(exercise_id)
                    outcome = auto.submit_response(
                        {
                            "passed": False,
                            "conceptual_gap_tag": first_gap,
                            "failure_diagnostic": first_diag,
                        },
                        cri_score=float(cri_score),
                    )
                    if auto.state in (
                        AutomatonState.S1_DIAGNOSTIC_PROBE,
                        AutomatonState.S2_REMEDIAL_CLUSTER,
                    ):
                        tok_w = ToKRemediationWidget(
                            automaton=auto,
                            concept_module=self.concept_module,
                            format_html_fn=_format_latex_and_math,
                            wrap_controls_fn=wrap_controls_dict,
                            initial_outcome=outcome,
                        )
                        self._active_remediation_widgets[exercise_id] = tok_w
                        display(tok_w.container)
                except Exception:
                    pass

        return report

    def show_record(self) -> None:
        """Render a summary table of problem completion, coding exercises, and CKV mastery."""
        total_q = len(self._problem_number)
        completed_q = sum(1 for r in self.records.values() if r.get("completed"))
        total_credit = sum(float(r.get("credit", 0.0)) for r in self.records.values() if r.get("completed"))

        by_diff: dict[str, list[int]] = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}
        for sec in self.sections:
            if sec.get("type") == "problem":
                d = sec.get("difficulty", "medium")
                by_diff.setdefault(d, [0, 0])[1] += 1
                if self.records.get(sec["id"], {}).get("completed"):
                    by_diff[d][0] += 1

        lines = [
            f"### Worksheet Progress Record — `{self.id}`",
            "",
            f"- **Completed Problems:** {completed_q} / {total_q} ({(100.0 * completed_q / max(1, total_q)):.1f}%)",
            f"- **Recorded Credit:** {total_credit:.2f} / {total_q:.2f}",
        ]

        if self.code_exercises:
            tot_c = len(self.code_exercises)
            done_c = sum(1 for r in self.code_records.values() if r.get("completed"))
            cred_c = sum(float(r.get("credit", 0.0)) for r in self.code_records.values() if r.get("completed"))
            lines.append(
                f"- **Completed Coding Exercises:** {done_c} / {tot_c} (credit {cred_c:.2f} / {tot_c:.2f})"
            )

        lines.extend(
            [
                "",
                "| Difficulty Tier | Completed | Total | Completion Rate |",
                "| :--- | :---: | :---: | :---: |",
            ]
        )
        for tier in ("easy", "medium", "hard"):
            done, tot = by_diff.get(tier, [0, 0])
            pct = 100.0 * done / max(1, tot)
            lines.append(f"| **{tier.capitalize()}** | {done} | {tot} | {pct:.0f}% |")

        ckv_keys = [f"CKV-{k}" for k in range(8) if f"CKV-{k}" in self.kv_engine.state.theta]
        if ckv_keys:
            lines.extend(
                [
                    "",
                    "#### Course Knowledge Vector (`CKV-0..7`) Mastery Snapshot",
                    "",
                    "| CKV | Mastery $\\theta_k$ | Uncertainty $\\sigma_k$ | Stability $S_k$ (h) |",
                    "| :--- | :---: | :---: | :---: |",
                ]
            )
            for ck in ckv_keys:
                th = self.kv_engine.state.theta[ck]
                sg = self.kv_engine.state.sigma[ck]
                st_h = self.kv_engine.state.stability_hours[ck]
                lines.append(f"| `{ck}` | {th:.3f} | {sg:.3f} | {st_h:.1f} |")

        display(Markdown("\n".join(lines)))

    def finalize(self) -> None:
        """Finalize and save the student's session receipt + V2 SHA-256 telemetry sidecar."""
        self._save_records()
        total_q = len(self._problem_number)
        completed_q = sum(1 for r in self.records.values() if r.get("completed"))
        total_credit = sum(float(r.get("credit", 0.0)) for r in self.records.values() if r.get("completed"))

        ckv_snapshot = {
            f"CKV-{k}": round(self.kv_engine.state.theta.get(f"CKV-{k}", 0.5), 4)
            for k in range(8)
        }
        v2_receipt = self.kv_engine.finalize_session_receipt(
            worksheet_id=self.id,
            completed_problems=completed_q,
            total_problems=total_q,
            recorded_credit=round(total_credit, 2),
        )

        receipt_path = self._records_dir / f"{self.id}_finalized.json"
        v2_sidecar_path = self._records_dir / f"{self.id}_v2_telemetry_receipt.json"
        try:
            with open(receipt_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "worksheet_id": self.id,
                        "title": self.title,
                        "session_id": self.session_id,
                        "completed_problems": completed_q,
                        "total_problems": total_q,
                        "recorded_credit": round(total_credit, 2),
                        "v2_code_exercises_completed": sum(
                            1 for r in self.code_records.values() if r.get("completed")
                        ),
                        "v2_code_exercises_total": len(self.code_exercises),
                        "v2_ckv_mastery": ckv_snapshot,
                        "v2_telemetry_head_sha256": v2_receipt.get("event_chain_head_sha256", ""),
                    },
                    fh,
                    indent=2,
                )
            with open(v2_sidecar_path, "w", encoding="utf-8") as vfh:
                json.dump(v2_receipt, vfh, indent=2)
        except Exception:
            pass

        display(
            Markdown(
                f"**Session Finalized (`{self.session_id}`)** — Saved **{completed_q}/{total_q}** completed problems "
                f"(credit **{total_credit:.2f}/{total_q:.2f}**) to `{receipt_path.name}`."
            )
        )
