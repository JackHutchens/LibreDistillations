import json

with open('Target Format Reference/lesson_10_1_betweenness_girvan_newman.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = ''.join(cell['source'])
    if "ws.render_section('q_6')" in src:
        print('Cell q_6 outputs count:', len(cell.get('outputs', [])))
        for out in cell.get('outputs', []):
            d = out.get('data', {})
            if 'application/vnd.jupyter.widget-view+json' in d:
                print('--- Widget text/plain ---')
                val = d.get('text/plain', '')
                if isinstance(val, list): val = ''.join(val)
                print(val.encode('ascii', errors='replace').decode())
                print('--- Widget view JSON ---')
                print(d['application/vnd.jupyter.widget-view+json'])
