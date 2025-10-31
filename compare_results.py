"""
Quick script to compare old vs new walk-forward results
"""

import pandas as pd
import json
from pathlib import Path

print("=" * 80)
print("COMPARISON: OLD vs NEW WALK-FORWARD RESULTS")
print("=" * 80)

# Load old results (Strict regime only)
old_strict_file = Path('results/walk_forward/aggregate_metrics.json')
if old_strict_file.exists():
    with open(old_strict_file, 'r') as f:
        old_strict = json.load(f)
    print("\n[OLD] Strict Regime (from aggregate_metrics.json):")
    print(f"  Mean Sharpe: {old_strict['mean_sharpe']:.2f}")
    print(f"  Mean Return: {old_strict['mean_return']*100:.2f}%")
    print(f"  Worst Drawdown: {old_strict['worst_drawdown']:.2f}%")
    print(f"  % Positive: {old_strict['positive_sharpe_pct']:.1f}%")

# Load old per-window results to find worst return
old_windows_file = Path('results/walk_forward/window_results.csv')
if old_windows_file.exists():
    old_windows = pd.read_csv(old_windows_file)
    worst_return = old_windows['annual_return'].min()
    worst_window = old_windows.loc[old_windows['annual_return'].idxmin()]
    print(f"  Worst Return (single window): {worst_return*100:.2f}% (Window {worst_window['window_num']}: {worst_window['test_start']})")

# Load new results (all 3 strategies)
new_file = Path('results/walk_forward_hybrid/aggregate_comparison.json')
if new_file.exists():
    with open(new_file, 'r') as f:
        new_results = json.load(f)

    print("\n[NEW] Baseline (no regime):")
    baseline = new_results['baseline']
    print(f"  Mean Sharpe: {baseline['mean_sharpe']:.2f}")
    print(f"  Mean Return: {baseline['mean_return']*100:.2f}%")
    print(f"  Worst DD: {baseline['worst_dd']*100:.2f}%")
    print(f"  % Positive: {baseline['pct_positive']:.1f}%")

    print("\n[NEW] Strict Regime:")
    strict = new_results['strict']
    print(f"  Mean Sharpe: {strict['mean_sharpe']:.2f}")
    print(f"  Mean Return: {strict['mean_return']*100:.2f}%")
    print(f"  Worst DD: {strict['worst_dd']*100:.2f}%")
    print(f"  % Positive: {strict['pct_positive']:.1f}%")

    print("\n[NEW] Hybrid:")
    hybrid = new_results['hybrid']
    print(f"  Mean Sharpe: {hybrid['mean_sharpe']:.2f}")
    print(f"  Mean Return: {hybrid['mean_return']*100:.2f}%")
    print(f"  Worst DD: {hybrid['worst_dd']*100:.2f}%")
    print(f"  % Positive: {hybrid['pct_positive']:.1f}%")

print("\n" + "=" * 80)
print("KEY DIFFERENCES:")
print("=" * 80)
print("1. OLD results show 'Worst Return' (per-window return)")
print("2. NEW results show 'Worst DD' (max drawdown)")
print("3. These are DIFFERENT metrics!")
print()
print("OLD Strict: Sharpe 1.13, Return 7.1%, Worst DD -4.62%")
print("NEW Strict: Sharpe 0.69, Return 4.8%, Worst DD -4.62%")
print()
print("→ Sharpe DROPPED by 39%!")
print("→ Return DROPPED by 32%!")
print("→ Worst DD SAME!")
print()
print("CONCLUSION: New results are WORSE, but worst DD is same.")
print("Need to investigate why Sharpe/Return dropped!")
print("=" * 80)
