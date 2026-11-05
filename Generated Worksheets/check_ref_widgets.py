import json

with open('Target Format Reference/lesson_10_1_betweenness_girvan_newman.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

wstate = nb.get('metadata', {}).get('widgets', {}).get('state', {})
print('Num widgets in state:', len(wstate))
types = set()
for wid, w in wstate.items():
    types.add(w.get('model_name'))
print('Widget model types in reference notebook:', types)

# Look at one of the label widgets in problem options
for wid, w in wstate.items():
    mname = w.get('model_name')
    if mname in ('HTMLModel', 'HTMLMathModel', 'LabelModel'):
        state = w.get('state', {})
        val = state.get('value', '')
        if '(A)' in val or '(B)' in val:
            print('Model:', mname)
            print('Value:', val[:200])
            break
