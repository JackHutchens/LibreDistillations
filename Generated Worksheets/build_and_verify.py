"""Build and verify interactive Jupyter Notebook worksheets from YAML + Concept modules.

Usage:
    python "Generated Worksheets/build_and_verify.py"                  # Build & verify all worksheets
    python "Generated Worksheets/build_and_verify.py" lesson_01_*.yaml # Build & verify specific YAML(s)
"""

from __future__ import annotations
import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
import yaml

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DIR = Path(__file__).resolve().parent
if str(GENERATED_DIR) not in sys.path:
    sys.path.insert(0, str(GENERATED_DIR))

from cs470_engine import Worksheet, apply_default_style
from cs470_engine.figures import render_static_figure
from cs470_engine.worksheet import normalize_controls_spec, wrap_controls_dict, normalize_angstrom, _format_latex_and_math


def _cell_id(seed_str: str) -> str:
    return hashlib.sha1(seed_str.encode("utf-8")).hexdigest()[:8]


def _md_output(text: str) -> nbformat.NotebookNode:
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]
    return new_output(
        output_type="display_data",
        data={
            "text/markdown": lines,
            "text/plain": ["<IPython.core.display.Markdown object>"],
        },
        metadata={},
    )


def _html_output(html_str: str) -> nbformat.NotebookNode:
    return new_output(
        output_type="display_data",
        data={
            "text/html": [html_str],
            "text/plain": ["<IPython.core.display.HTML object>"],
        },
        metadata={},
    )


def _extract_taught_by_ids(taught_by_raw: Any) -> list[str]:
    if not taught_by_raw:
        return []
    if isinstance(taught_by_raw, str):
        return [taught_by_raw]
    if isinstance(taught_by_raw, list):
        out = []
        for item in taught_by_raw:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, Mapping) and "cell" in item:
                out.append(str(item["cell"]))
        return out
    return []


def validate_and_compile_worksheet(
    yaml_path: Path,
    write_ipynb: bool = True,
    strict_count: bool = True,
) -> dict[str, Any]:
    """Validate a worksheet YAML + Concept module and compile its .ipynb notebook."""
    apply_default_style()
    ws = Worksheet.load(yaml_path)
    errors: list[str] = []
    warnings: list[str] = []

    sections = ws.sections
    seen_ids: set[str] = set()
    teaching_ids: set[str] = set()
    problems: list[dict[str, Any]] = []
    concept_cells: list[dict[str, Any]] = []
    markdown_sections: list[dict[str, Any]] = []

    for sec in sections:
        sid = sec.get("id")
        stype = sec.get("type")
        if not sid:
            errors.append("Section missing 'id'")
            continue
        if sid in seen_ids:
            errors.append(f"Duplicate section id: {sid!r}")
        seen_ids.add(sid)

        if stype in ("markdown_section", "markdown", "intro"):
            markdown_sections.append(sec)
            teaching_ids.add(sid)
            body = (sec.get("body_markdown") or sec.get("markdown") or "").strip()
            if not body:
                errors.append(f"markdown_section {sid!r} has empty body_markdown/markdown")
            if sec.get("figure"):
                try:
                    fig, _ = render_static_figure(
                        sec["figure"],
                        shared_figures=ws.shared_figures,
                        concept_module=ws.concept_module,
                    )
                    plt.close(fig)
                except Exception as exc:
                    errors.append(f"markdown_section {sid!r} figure failed: {exc}")

        elif stype in ("concept_cell", "concept"):
            concept_cells.append(sec)
            teaching_ids.add(sid)
            fn_name = sec.get("render_function")
            if not fn_name:
                errors.append(f"concept_cell {sid!r} missing render_function")
            elif ws.concept_module is None or not hasattr(ws.concept_module, fn_name):
                errors.append(f"concept_cell {sid!r} render_function {fn_name!r} not found in Concept module")
            else:
                fn = getattr(ws.concept_module, fn_name)
                controls = normalize_controls_spec(sec.get("controls"))
                default_vals = {}
                min_vals = {}
                max_vals = {}
                for cname, cspec in controls.items():
                    kind = cspec.get("kind", "slider")
                    if kind == "dropdown":
                        opts = cspec.get("options", [])
                        first_val = (
                            opts[0]["value"]
                            if (opts and isinstance(opts[0], dict))
                            else (opts[0] if opts else None)
                        )
                        last_val = (
                            opts[-1]["value"]
                            if (opts and isinstance(opts[-1], dict))
                            else (opts[-1] if opts else None)
                        )
                        default_vals[cname] = cspec.get("default", first_val)
                        min_vals[cname] = first_val
                        max_vals[cname] = last_val
                    else:
                        default_vals[cname] = cspec.get("default", cspec.get("min", 0))
                        min_vals[cname] = cspec.get("min", default_vals[cname])
                        max_vals[cname] = cspec.get("max", default_vals[cname])

                for label_try, ctrl_dict in [("default", default_vals), ("min", min_vals), ("max", max_vals)]:
                    try:
                        fig, ax = plt.subplots(figsize=(6.0, 4.0))
                        cap = fn(wrap_controls_dict(ctrl_dict), ax)
                        plt.close(fig)
                        if not isinstance(cap, str) or not cap.strip():
                            warnings.append(f"concept_cell {sid!r} ({label_try}) returned empty caption")
                    except Exception as exc:
                        plt.close("all")
                        errors.append(f"concept_cell {sid!r} ({label_try}) failed in {fn_name}: {exc}")

        elif stype == "problem":
            problems.append(sec)
            qtype = sec.get("question_type")
            if qtype not in ("single_select", "multi_select", "numeric"):
                errors.append(f"problem {sid!r} has invalid question_type {qtype!r}")
            if sec.get("difficulty") not in ("easy", "medium", "hard"):
                errors.append(f"problem {sid!r} has invalid difficulty {sec.get('difficulty')!r}")
            tb_ids = _extract_taught_by_ids(sec.get("taught_by"))
            if not tb_ids or any(t not in teaching_ids for t in tb_ids):
                errors.append(
                    f"problem {sid!r} taught_by={sec.get('taught_by')!r} does not reference an earlier teaching section"
                )
            if not (sec.get("prompt_markdown") or "").strip():
                errors.append(f"problem {sid!r} missing prompt_markdown")
            if not (sec.get("rationale_markdown") or "").strip():
                errors.append(f"problem {sid!r} missing rationale_markdown")

            if qtype in ("single_select", "multi_select"):
                opts = sec.get("options", [])
                if len(opts) < 2:
                    errors.append(f"problem {sid!r} has fewer than 2 options")
                for opt in opts:
                    if not (opt.get("text") or opt.get("text_markdown")):
                        errors.append(f"problem {sid!r} option {opt.get('id')!r} missing 'text'")
                opt_ids = {str(o.get("id")) for o in opts}
                corr = sec.get("correct")
                if isinstance(corr, list):
                    corr_set = {str(x) for x in corr}
                else:
                    corr_set = {str(corr)}
                if not corr_set or not corr_set.issubset(opt_ids):
                    errors.append(f"problem {sid!r} correct={corr!r} invalid for {sorted(opt_ids)}")
            elif qtype == "numeric":
                try:
                    cval = sec["correct"]
                    if isinstance(cval, Mapping):
                        float(cval.get("value", cval.get("answer")))
                    else:
                        float(cval)
                except Exception:
                    errors.append(f"problem {sid!r} numeric 'correct' is not a float: {sec.get('correct')!r}")

            if sec.get("figure"):
                try:
                    fig, _ = render_static_figure(
                        sec["figure"],
                        shared_figures=ws.shared_figures,
                        concept_module=ws.concept_module,
                    )
                    plt.close(fig)
                except Exception as exc:
                    errors.append(f"problem {sid!r} figure failed: {exc}")
        else:
            errors.append(f"Section {sid!r} has unknown type {stype!r}")

    q_count = len(problems)
    if strict_count and not (35 <= q_count <= 50):
        errors.append(f"Question count {q_count} is outside required [35, 50] range")

    if write_ipynb and not errors:
        nb = new_notebook()
        nb.metadata = {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0",
            },
        }

        cells = []
        start_source = (
            "# Auto-generated by build_and_verify.py — do not hand-edit.\n"
            f"# Rebuild with: python build_and_verify.py {yaml_path.name}\n\n"
            "from cs470_engine import Worksheet\n"
            "from cs470_engine.plot_style import apply_default_style\n\n"
            "apply_default_style()\n\n"
            f"ws = Worksheet.load('{yaml_path.name}')\n"
            "ws.start()\n"
        )
        header_lines = [
            "$\\def\\AA{\\text{Å}}$\n\n",
            f"# {ws.title}\n\n",
            f"_Session {ws.session_id} · Estimated time: ~{ws.estimated_minutes} minutes · {q_count} problems_\n\n",
            "**Recording your answers.** Submit each problem as you go. Your answers are saved to the workspace as you submit, *and* the running record at the bottom of the notebook keeps your progress. When you finish, run the last two cells — **`show_record()`** and **`finalize()`** — to record your participation. **If you reopened this workspace** (or the kernel restarted), use **Kernel → Restart Kernel and Run All Cells** before finalizing, so every answer is reloaded.",
        ]
        if ws.learning_objectives:
            header_lines.append("\n\n### Learning Objectives\n")
            for obj in ws.learning_objectives:
                header_lines.append(f"- {normalize_angstrom(obj)}\n")

        c0 = new_code_cell(
            source=start_source,
            execution_count=1,
            outputs=[_md_output("".join(header_lines))],
        )
        c0["id"] = _cell_id(f"{ws.id}:start")
        cells.append(c0)

        exec_counter = 2
        for sec in sections:
            sid = sec["id"]
            stype = sec["type"]
            outs = []
            if stype in ("markdown_section", "markdown", "intro"):
                body = (sec.get("body_markdown") or sec.get("markdown") or "").strip()
                title = (sec.get("title") or "").strip()
                text = f"## {title}\n\n{body}\n" if (title and not body.startswith("#")) else f"{body}\n"
                outs.append(_md_output(normalize_angstrom(text)))
            elif stype in ("concept_cell", "concept"):
                if sec.get("title"):
                    outs.append(_md_output(f"### {normalize_angstrom(sec['title'])}"))
                if sec.get("description_markdown"):
                    outs.append(_md_output(normalize_angstrom(sec["description_markdown"])))
                if sec.get("takeaway_markdown"):
                    outs.append(_md_output(normalize_angstrom(sec["takeaway_markdown"])))
            elif stype == "problem":
                qnum = ws._problem_number[sid]
                diff = sec.get("difficulty", "medium")
                hint_md = (sec.get("hint_markdown") or "").strip()
                header = f"**Q{qnum} · {diff}**" + (" · 💡" if hint_md else "")
                outs.append(_md_output(header))
                outs.append(_md_output(normalize_angstrom(sec.get("prompt_markdown", ""))))
                if hint_md:
                    norm_hint = normalize_angstrom(hint_md)
                    hint_html = (
                        '<details style="margin: 0.5em 0;">\n'
                        '<summary style="cursor: pointer; user-select: none; display: inline-block; '
                        'padding: 0.3em 0.7em; margin: 0.2em 0; background: #EEF0F7; border-radius: 4px; '
                        'font-weight: 500;">💡 Show hint</summary>\n\n'
                        f'<div style="padding: 0.5em 1em;">\n\n{norm_hint}\n\n</div>\n</details>'
                    )
                    outs.append(_md_output(hint_html))

                if sec.get("question_type") in ("single_select", "multi_select"):
                    options = sec.get("options", [])
                    if options:
                        opt_lines = []
                        for opt in options:
                            oid = str(opt["id"]).upper()
                            otxt = normalize_angstrom(str(opt.get("text") or opt.get("text_markdown") or ""))
                            opt_lines.append(f"- [ ] **({oid})** {otxt}")
                        outs.append(_md_output("\n".join(opt_lines)))
                elif sec.get("question_type") == "numeric":
                    unit_str = sec.get("unit") or sec.get("units")
                    unit_disp = f" (in {normalize_angstrom(unit_str)})" if unit_str else ""
                    outs.append(_md_output(f"_Numeric answer required{unit_disp}._"))

            cell = new_code_cell(
                source=f"ws.render_section('{sid}')",
                execution_count=exec_counter,
                outputs=outs,
            )
            cell["id"] = _cell_id(f"{ws.id}:{sid}")
            cells.append(cell)
            exec_counter += 1

        c_rec = new_code_cell(source="ws.show_record()", execution_count=None, outputs=[])
        c_rec["id"] = _cell_id(f"{ws.id}:show_record")
        cells.append(c_rec)

        c_fin = new_code_cell(source="ws.finalize()", execution_count=None, outputs=[])
        c_fin["id"] = _cell_id(f"{ws.id}:finalize")
        cells.append(c_fin)

        nb.cells = cells
        ipynb_path = yaml_path.with_suffix(".ipynb")
        with open(ipynb_path, "w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)

    return {
        "id": ws.id,
        "yaml": yaml_path.name,
        "questions": q_count,
        "concept_cells": len(concept_cells),
        "markdown_sections": len(markdown_sections),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    args = sys.argv[1:]
    if args:
        yaml_files = []
        for pat in args:
            p = Path(pat)
            if p.is_file():
                yaml_files.append(p.resolve())
            else:
                yaml_files.extend(sorted(GENERATED_DIR.glob(pat)))
    else:
        yaml_files = sorted(GENERATED_DIR.glob("lesson_*.yaml"))

    if not yaml_files:
        print("No worksheet YAML files matched.")
        return 0

    any_err = False
    print(f"{'Worksheet ID':<46} | {'Qs':>3} | {'Concepts':>8} | {'MD Secs':>7} | Status")
    print("-" * 86)
    for yp in yaml_files:
        strict = yp.parent == GENERATED_DIR
        res = validate_and_compile_worksheet(yp, write_ipynb=strict, strict_count=strict)
        status = "OK" if not res["errors"] else f"FAIL ({len(res['errors'])} errs)"
        print(
            f"{res['id']:<46} | {res['questions']:>3} | {res['concept_cells']:>8} | {res['markdown_sections']:>7} | {status}"
        )
        for err in res["errors"]:
            any_err = True
            print(f"  [ERROR] {err}")
        for warn in res["warnings"]:
            print(f"  [WARN]  {warn}")

    return 1 if any_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
