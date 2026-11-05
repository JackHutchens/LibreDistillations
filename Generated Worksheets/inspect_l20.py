import re

with open('Generated Worksheets/Concepts/lesson_20_free_energy_and_phase_equilibria.py', 'r', encoding='utf-8') as f:
    text = f.read()

funcs = ['render_kirkwood_thermodynamic_integration', 'render_widom_and_gibbs_coexistence', 'render_bennett_acceptance_overlap', 'render_wang_landau_flat_histogram']
for fn in funcs:
    idx = text.find('def ' + fn)
    end = text.find('def ', idx + 10)
    chunk = text[idx:end] if end != -1 else text[idx:]
    print('=== ' + fn + ' ===')
    for m in re.finditer(r'_val\(controls\.get\([\"\'](\w+)[\"\'],\s*([^)]+)\)', chunk):
        print(' ', m.group(1), 'default:', m.group(2))
