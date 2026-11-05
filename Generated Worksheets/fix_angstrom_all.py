"""Batch script to fix and normalize all Angstrom representations across all YAML and Concept files."""

import glob
import re
from pathlib import Path

def fix_text(text: str) -> str:
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

    # 2. Quad-backslash LaTeX \text{...} and \mathrm{...}
    text = re.sub(r'\\\\text\{(\s*)\\\\AA\s*\}', r'\\\\text{\1Å}', text)
    text = re.sub(r'\\\\mathrm\{(\s*)\\\\AA\s*\}', r'\\\\text{\1Å}', text)
    text = re.sub(r'\\\\text\{/AA\}', r'\\\\text{Å}', text)
    text = re.sub(r'\\\\mathrm\{\{(\s*)\\\\AA\s*\}\}', r'\\\\text{\1Å}', text)
    text = re.sub(r'\\\\text\{\s*\\\\mathrm\{\s*\\\\AA\s*\}\s*\}', r'\\\\text{Å}', text)
    # inside quadruple \text{...}, replace \AA with Å
    text = re.sub(r'(\\\\text\{[^\}]*?)\\\\AA', r'\1Å', text)
    text = re.sub(r'(\\\\mathrm\{[^\}]*?)\\\\AA', r'\1Å', text)

    # 3. Double-backslash LaTeX \text{...} and \mathrm{...}
    text = re.sub(r'\\text\{(\s*)\\AA\s*\}', r'\\text{\1Å}', text)
    text = re.sub(r'\\mathrm\{(\s*)\\AA\s*\}', r'\\text{\1Å}', text)
    text = re.sub(r'\\text\{/AA\}', r'\\text{Å}', text)
    text = re.sub(r'\\mathrm\{\{(\s*)\\AA\s*\}\}', r'\\text{Å}', text)
    text = re.sub(r'\\text\{\s*\\mathrm\{\s*\\AA\s*\}\s*\}', r'\\text{Å}', text)
    # inside double \text{...}, replace \AA with Å
    text = re.sub(r'(\\text\{[^\}]*?)\\AA', r'\1Å', text)
    text = re.sub(r'(\\mathrm\{[^\}]*?)\\AA', r'\1Å', text)

    # 4. Bare \AA or \\AA inside math mode ($...$ or $$...$$)
    # MUST NOT touch C_{AA}, g_{AA}, S_{AA}, etc.
    def replace_math(match: re.Match) -> str:
        m = match.group(0)
        # In YAML where backslashes are quadrupled: \\\\AA
        m = re.sub(r'(?<![A-Za-z0-9_\\])\\\\AA\b', r'\\\\text{Å}', m)
        # In normal LaTeX strings: \\AA
        m = re.sub(r'(?<![A-Za-z0-9_\\])\\AA\b', r'\\text{Å}', m)
        # In math mode: /AA -> \text{Å}
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

    return text

def main():
    root = Path(__file__).resolve().parent
    files = sorted(root.glob("*.yaml")) + sorted((root / "Concepts").glob("*.py"))
    
    updated_files = 0
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            content = f.read()
        fixed = fix_text(content)
        if fixed != content:
            with open(p, "w", encoding="utf-8") as f:
                f.write(fixed)
            updated_files += 1
            print(f"Updated: {p.name}")

    print(f"\nDone! Updated {updated_files} of {len(files)} files.")

if __name__ == "__main__":
    main()
