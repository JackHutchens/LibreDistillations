import glob, re

files = glob.glob('Generated Worksheets/*.yaml') + glob.glob('Generated Worksheets/Concepts/*.py')

for p in sorted(files):
    with open(p, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find all contexts with AA or \AA or Å
    found = set()
    for m in re.finditer(r'(\\text\{\\AA\}|\\mathrm\{\\AA\}|\\AA|/AA|Å|\u00c5|\u212b)', text):
        # get surrounding context of 30 chars
        start = max(0, m.start() - 15)
        end = min(len(text), m.end() + 15)
        snippet = text[start:end].replace('\n', ' ')
        found.add(snippet)
    if found:
        print(f"=== {p} ({len(found)} contexts) ===")
        for snip in list(sorted(found))[:5]:
            print("  ", repr(snip))
