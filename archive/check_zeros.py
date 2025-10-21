import json

with open('logs/factor_monitoring_20251017_041520.json') as f:
    data = json.load(f)

zeros = [f for f in data['factors'] if f['success_rate'] == 0.0]
print(f'\nFound {len(zeros)} factors at 0%\n')

# Group by category
groups = {}
for f in zeros:
    group = f['group']
    if group not in groups:
        groups[group] = []
    groups[group].append(f['name'])

# Print by group
for g in sorted(groups.keys()):
    print(f'\n{g.upper()}:')
    for name in sorted(groups[g]):
        print(f'  - {name}')

print('\n')
