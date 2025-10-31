"""
Fix Grafana Dashboard Time Ranges
==================================

Changes all time ranges from -1h to -30d so data is visible
"""

import json

dashboard_path = r'C:\Users\milan\Desktop\Git-Projects\ml_trading_system\production\grafana\dashboards\stock_trading_dashboard.json'

print("="*80)
print("FIXING GRAFANA DASHBOARD TIME RANGES")
print("="*80)

# Load dashboard
with open(dashboard_path, 'r') as f:
    dashboard = json.load(f)

# Counter for changes
changes = 0

# Fix global time range
if 'time' in dashboard:
    if dashboard['time'].get('from') == 'now-1h':
        dashboard['time']['from'] = 'now-30d'
        changes += 1
        print("[OK] Fixed global time range from 'now-1h' to 'now-30d'")

# Process each panel
for panel in dashboard.get('panels', []):
    panel_title = panel.get('title', 'Untitled')

    # Process targets (queries)
    for target in panel.get('targets', []):
        if 'query' in target and isinstance(target['query'], str):
            original_query = target['query']

            # Replace -1h with -30d
            if 'range(start: -1h)' in original_query:
                target['query'] = original_query.replace('range(start: -1h)', 'range(start: -30d)')
                changes += 1
                print(f"[OK] Fixed query in panel: {panel_title}")

            # Note: Keep -1m for "Data Points (Last Minute)" panel as that's intentional
            if 'range(start: -1m)' in original_query and panel_title != 'Data Points (Last Minute)':
                target['query'] = original_query.replace('range(start: -1m)', 'range(start: -30d)')
                changes += 1
                print(f"[OK] Fixed -1m query in panel: {panel_title}")

print(f"\n{'='*80}")
print(f"Total changes made: {changes}")
print(f"{'='*80}")

# Save fixed dashboard
with open(dashboard_path, 'w') as f:
    json.dump(dashboard, f, indent=2)

print("\n[OK] Dashboard saved!")
print(f"Path: {dashboard_path}")
print("\nYou need to restart Grafana for changes to take effect:")
print("  docker-compose restart grafana")
