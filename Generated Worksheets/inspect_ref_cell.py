import json

with open('Target Format Reference/lesson_10_1_betweenness_girvan_newman.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell['source'])
    if "ws.render_section('q_2')" in src:
        print('=== q_2 outputs in Target Format Reference ===')
        for i, out in enumerate(cell.get('outputs', [])):
            d = out.get('data', {})
            print('Output', i, out.get('output_type'), list(d.keys()))
            for k in ('text/html', 'text/markdown', 'text/plain'):
                if k in d:
                    val = ''.join(d[k]) if isinstance(d[k], list) else str(d[k])
                    print('  [' + k + ']:', repr(val[:150]))
