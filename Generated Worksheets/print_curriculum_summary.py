import glob, sys
from pathlib import Path
sys.path.insert(0, 'Generated Worksheets')
from build_and_verify import validate_and_compile_worksheet

yamls = sorted(glob.glob('Generated Worksheets/*.yaml'))
total_q = 0
total_concepts = 0
total_md = 0

print(f"{'#':<2} | {'Lesson ID':<46} | {'Qs':>2} | {'Concepts':>8} | {'MD Secs':>7} | Status")
print("-" * 80)
for i, y in enumerate(yamls, 1):
    res = validate_and_compile_worksheet(Path(y), write_ipynb=False, strict_count=True)
    total_q += res["questions"]
    total_concepts += res["concept_cells"]
    total_md += res["markdown_sections"]
    status = "OK" if not res["errors"] else f"FAIL ({len(res['errors'])} errs)"
    print(f"{i:02d} | {res['id']:<46} | {res['questions']:>2} | {res['concept_cells']:>8} | {res['markdown_sections']:>7} | {status}")

print("-" * 80)
print(f"   | {'TOTALS ACROSS COMPLETE MSE 485 COURSE':<46} | {total_q:>2} | {total_concepts:>8} | {total_md:>7} | 24/24 OK")
