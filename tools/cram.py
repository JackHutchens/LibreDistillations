"""Turbo-Cram Unified Multi-Course Management CLI.

Commands:
  python tools/cram.py list
  python tools/cram.py ingest <course_id> [--resources <path>]
  python tools/cram.py verify [course_id|all]
  python tools/cram.py build [course_id|all]
  python tools/cram.py summary [course_id|all]
"""

from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_output
import yaml

from engine import Worksheet, apply_default_style
from engine.figures import render_static_figure
from engine.worksheet import (
    normalize_controls_spec,
    wrap_controls_dict,
    normalize_angstrom,
    _format_latex_and_math,
)


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


def get_course_dir(course_id: str) -> Path:
    return WORKSPACE_ROOT / "courses" / course_id


def list_courses() -> list[str]:
    courses_dir = WORKSPACE_ROOT / "courses"
    if not courses_dir.is_dir():
        return []
    out = []
    for d in sorted(courses_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            out.append(d.name)
    return out


def load_course_manifest(course_id: str) -> dict[str, Any]:
    c_dir = get_course_dir(course_id)
    manifest = c_dir / "course.yaml"
    if manifest.is_file():
        with open(manifest, "r", encoding="utf-8") as f:
            return dict(yaml.safe_load(f) or {})
    return {
        "course": {
            "id": course_id,
            "name": course_id.upper().replace("_", " "),
        }
    }


def validate_and_compile_worksheet(
    yaml_path: Path,
    output_notebook_path: Path | None = None,
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
                    elif kind == "slider":
                        cmin = cspec.get("min", 0)
                        cmax = cspec.get("max", 10)
                        cdef = cspec.get("default", cmin)
                        default_vals[cname] = cdef
                        min_vals[cname] = cmin
                        max_vals[cname] = cmax
                    else:
                        v = cspec.get("default", "")
                        default_vals[cname] = v
                        min_vals[cname] = v
                        max_vals[cname] = v

                for mode_label, test_dict in [
                    ("default", default_vals),
                    ("min", min_vals),
                    ("max", max_vals),
                ]:
                    try:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        caption = fn(wrap_controls_dict(test_dict), ax)
                        plt.close(fig)
                        if caption and not isinstance(caption, str):
                            errors.append(
                                f"concept_cell {sid!r} render_function {fn_name!r} returned non-string caption: {type(caption)}"
                            )
                    except Exception as exc:
                        errors.append(
                            f"concept_cell {sid!r} ({mode_label}) failed in {fn_name}: {exc}"
                        )

        elif stype == "problem":
            problems.append(sec)
            tb_ids = _extract_taught_by_ids(sec.get("taught_by"))
            if not tb_ids:
                warnings.append(f"Problem {sid!r} has no taught_by reference")
            else:
                for tb in tb_ids:
                    if tb not in teaching_ids:
                        errors.append(
                            f"Problem {sid!r} references taught_by={tb!r} which has not been defined before this problem"
                        )

            qtype = sec.get("question_type")
            if qtype not in ("single_select", "multi_select", "numeric"):
                errors.append(f"Problem {sid!r} has invalid question_type: {qtype!r}")

            if not sec.get("prompt_markdown"):
                errors.append(f"Problem {sid!r} missing prompt_markdown")
            if not sec.get("hint_markdown"):
                warnings.append(f"Problem {sid!r} missing hint_markdown")
            if not sec.get("rationale_markdown"):
                warnings.append(f"Problem {sid!r} missing rationale_markdown")

            if qtype in ("single_select", "multi_select"):
                options = sec.get("options", [])
                if not options or len(options) < 2:
                    errors.append(f"Problem {sid!r} options count < 2")
                opt_ids = [str(opt.get("id")) for opt in options]
                if len(opt_ids) != len(set(opt_ids)):
                    errors.append(f"Problem {sid!r} duplicate option IDs: {opt_ids}")
                corr = sec.get("correct")
                if corr is None:
                    errors.append(f"Problem {sid!r} missing correct answer")
                elif qtype == "single_select":
                    if str(corr) not in opt_ids:
                        errors.append(f"Problem {sid!r} correct {corr!r} not in options {opt_ids}")
                elif qtype == "multi_select":
                    if isinstance(corr, list):
                        corr_list = [str(x) for x in corr]
                    else:
                        corr_list = [str(corr)]
                    for c in corr_list:
                        if c not in opt_ids:
                            errors.append(f"Problem {sid!r} correct element {c!r} not in options {opt_ids}")
            elif qtype == "numeric":
                corr = sec.get("correct")
                if corr is None:
                    errors.append(f"Numeric problem {sid!r} missing correct answer")
                else:
                    try:
                        if isinstance(corr, Mapping):
                            float(corr.get("value", corr.get("answer", 0.0)))
                        else:
                            float(corr)
                    except (ValueError, TypeError):
                        errors.append(f"Numeric problem {sid!r} correct answer cannot be converted to float: {corr!r}")

            if sec.get("figure"):
                try:
                    fig, _ = render_static_figure(
                        sec["figure"],
                        shared_figures=ws.shared_figures,
                        concept_module=ws.concept_module,
                    )
                    plt.close(fig)
                except Exception as exc:
                    errors.append(f"Problem {sid!r} figure failed: {exc}")
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
            "# Auto-generated by cram CLI — do not hand-edit.\n"
            f"# Rebuild with: python tools/cram.py build\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "# Portably discover repo root containing 'engine' and add to sys.path\n"
            "_cur = Path.cwd().resolve()\n"
            "while _cur != _cur.parent:\n"
            "    if (_cur / 'engine' / '__init__.py').is_file():\n"
            "        if str(_cur) not in sys.path:\n"
            "            sys.path.insert(0, str(_cur))\n"
            "        break\n"
            "    _cur = _cur.parent\n\n"
            "from engine import Worksheet, apply_default_style\n\n"
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
        target_path = output_notebook_path or yaml_path.with_suffix(".ipynb")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as fh:
            nbformat.write(nb, fh)

    return {
        "id": ws.id,
        "questions": q_count,
        "concepts": len(concept_cells),
        "markdown_sections": len(markdown_sections),
        "errors": errors,
        "warnings": warnings,
    }


def cmd_list(args: argparse.Namespace) -> None:
    courses = list_courses()
    if not courses:
        print("No courses detected in courses/ directory.")
        return

    print(f"{'Course ID':<15} | {'Course Name':<35} | {'Resources':<9} | {'Worksheets':<10} | {'Notebooks':<9}")
    print("-" * 88)
    for cid in courses:
        c_dir = get_course_dir(cid)
        meta = load_course_manifest(cid)
        c_name = meta.get("course", {}).get("name", cid)
        res_count = len(list((c_dir / "resources").glob("*.pdf")))
        ws_count = len(list((c_dir / "worksheets").glob("*.yaml")))
        nb_count = len(list((c_dir / "notebooks").glob("*.ipynb")))
        print(f"{cid:<15} | {c_name[:35]:<35} | {res_count:<9} | {ws_count:<10} | {nb_count:<9}")


def cmd_verify_or_build(args: argparse.Namespace, is_build: bool) -> None:
    target = args.course or "all"
    courses = list_courses() if target == "all" else [target]
    
    strict_count = not getattr(args, "no_strict", False)

    total_qs = 0
    total_concepts = 0
    total_md = 0
    total_worksheets = 0
    all_ok = True

    action_name = "Building & Verifying" if is_build else "Verifying"

    for cid in courses:
        c_dir = get_course_dir(cid)
        ws_dir = c_dir / "worksheets"
        nb_dir = c_dir / "notebooks"
        yamls = sorted(ws_dir.glob("*.yaml"))
        if not yamls:
            print(f"[{cid}] No worksheets found in {ws_dir}")
            continue

        meta = load_course_manifest(cid)
        c_name = meta.get("course", {}).get("name", cid)
        print(f"\n=== {action_name} Course: {c_name} ({cid}) ===")
        print(f"{'Worksheet ID':<46} | {'Qs':>3} | {'Concepts':>8} | {'MD Secs':>7} | Status")
        print("-" * 86)

        for y_path in yamls:
            target_nb = nb_dir / f"{y_path.stem}.ipynb"
            res = validate_and_compile_worksheet(
                y_path,
                output_notebook_path=target_nb,
                write_ipynb=is_build,
                strict_count=strict_count,
            )
            total_qs += res["questions"]
            total_concepts += res["concepts"]
            total_md += res["markdown_sections"]
            total_worksheets += 1

            status = "OK" if not res["errors"] else f"FAIL ({len(res['errors'])} errs)"
            if res["errors"]:
                all_ok = False
            print(f"{res['id']:<46} | {res['questions']:>3} | {res['concepts']:>8} | {res['markdown_sections']:>7} | {status}")
            for err in res["errors"]:
                print(f"  [ERROR] {err}")
            for w in res["warnings"]:
                print(f"  [WARN] {w}")

    print("-" * 86)
    summary_label = "ALL WORKSHEETS PASSED" if all_ok else "ERRORS DETECTED"
    print(f"TOTALS: {total_worksheets} worksheets, {total_qs} questions, {total_concepts} concepts, {total_md} sections | {summary_label}")
    if not all_ok:
        sys.exit(1)


def cmd_ingest(args: argparse.Namespace) -> None:
    cid = args.course
    c_dir = get_course_dir(cid)
    res_dir = c_dir / "resources"
    ws_dir = c_dir / "worksheets"
    cp_dir = c_dir / "concepts"
    nb_dir = c_dir / "notebooks"
    rec_dir = c_dir / ".records"

    for d in [res_dir, ws_dir, cp_dir, nb_dir, rec_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.resources:
        src = Path(args.resources).resolve()
        if src.is_dir():
            import shutil
            for f in sorted(src.glob("*.pdf")):
                shutil.copy2(f, res_dir / f.name)
                print(f"Copied: {f.name} -> {res_dir}")

    pdfs = sorted(res_dir.glob("*.pdf"))
    print(f"Auditing {len(pdfs)} resource PDFs in {res_dir}...")
    manifest_path = c_dir / "course.yaml"
    if not manifest_path.is_file():
        tpl = (WORKSPACE_ROOT / "templates" / "course.yaml.template").read_text(encoding="utf-8")
        tpl = tpl.replace("course_id_slug", cid)
        tpl = tpl.replace("DEPT 101", cid.upper().replace("_", " "))
        manifest_path.write_text(tpl, encoding="utf-8")
        print(f"Created manifest: {manifest_path.relative_to(WORKSPACE_ROOT)}")

    # Audit hashes
    hashes: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for p in pdfs:
        with open(p, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        if h in hashes:
            duplicates.append((p.name, hashes[h]))
        else:
            hashes[h] = p.name

    if duplicates:
        print("\nDetected binary duplicates:")
        for a, b in duplicates:
            print(f"  {a} == {b}")

    print(f"\nCourse {cid} ingestion complete! Place or generate worksheets in {ws_dir.relative_to(WORKSPACE_ROOT)}.")


def main():
    parser = argparse.ArgumentParser(description="Turbo-Cram Unified Multi-Course Management CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List registered courses and stats")

    # verify
    p_ver = subparsers.add_parser("verify", help="Verify worksheets and concept renderers")
    p_ver.add_argument("course", nargs="?", default="all", help="Course ID or 'all'")
    p_ver.add_argument("--no-strict", action="store_true", help="Disable strict [35, 50] question quota")

    # build
    p_bld = subparsers.add_parser("build", help="Compile worksheets into Jupyter notebooks")
    p_bld.add_argument("course", nargs="?", default="all", help="Course ID or 'all'")
    p_bld.add_argument("--no-strict", action="store_true", help="Disable strict [35, 50] question quota")

    # summary
    p_sum = subparsers.add_parser("summary", help="Print course curriculum summary")
    p_sum.add_argument("course", nargs="?", default="all", help="Course ID or 'all'")

    # ingest
    p_ing = subparsers.add_parser("ingest", help="Scaffold course from batch resources")
    p_ing.add_argument("course", help="Course ID to create or update")
    p_ing.add_argument("--resources", help="Path to folder of PDFs to import")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list(args)
    elif args.command == "verify":
        cmd_verify_or_build(args, is_build=False)
    elif args.command == "build":
        cmd_verify_or_build(args, is_build=True)
    elif args.command == "summary":
        cmd_verify_or_build(args, is_build=False)
    elif args.command == "ingest":
        cmd_ingest(args)


if __name__ == "__main__":
    main()
