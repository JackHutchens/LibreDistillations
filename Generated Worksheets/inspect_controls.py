import glob, os, re

files = [
    'lesson_04_molecular_dynamics.py',
    'lesson_20_free_energy_and_phase_equilibria.py',
    'lesson_22_ising_model_and_spin_kinetics.py',
    'lesson_24_kinetic_monte_carlo.py'
]

for fname in files:
    path = os.path.join('Generated Worksheets', 'Concepts', fname)
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    print('*** ' + fname + ' ***')
    # Find all def render_...
    sections = re.split(r'\ndef (render_\w+)', text)
    for i in range(1, len(sections), 2):
        fn_name = sections[i]
        fn_body = sections[i+1]
        matches = re.findall(r'controls\.get\([\"\'](\w+)[\"\'],\s*([^)]+)\)', fn_body)
        print('  ' + fn_name + ':')
        for m in matches:
            print('    ' + m[0] + ' (default: ' + m[1] + ')')
