import re
import sys
from pathlib import Path
import yaml

def fix_yaml(file_path):
    p = Path(file_path)
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        m = re.match(r'^(\s*(?:rationale_markdown|prompt_markdown|hint_markdown|body_markdown|description_markdown|takeaway_markdown|text|title):\s*)"(.*)"\s*$', line)
        if m:
            indent_key = m.group(1)
            val = m.group(2)
            if '\\' in val:
                key_name = indent_key.strip().rstrip(':')
                indent = re.match(r'^(\s*)', indent_key).group(1)
                val_clean = val.replace('\\"', '"')
                new_lines.append(f"{indent}{key_name}: >-\n")
                new_lines.append(f"{indent}  {val_clean}\n")
                continue
        new_lines.append(line)

    content = "".join(new_lines)
    try:
        yaml.safe_load(content)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Fixed {p.name} successfully and validated YAML!")
    except Exception as e:
        print(f"Still error in {p.name}: {e}")

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        fix_yaml(arg)
