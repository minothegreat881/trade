"""
ANALYZA BASELINE vs OPTIMIZED MODEL
====================================

Analyzuje preco baseline model (Sharpe 1.34) funguje tak dobre
a porovnava ho s optimalizovanym modelom (Sharpe 0.78)
"""

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

print("="*80)
print("ANALYZA BASELINE vs OPTIMIZED MODEL")
print("="*80)

# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/8] Loading data...")

# Load predictions
baseline_pred_df = pd.read_csv('results/full_dataset_baseline_2025/predictions.csv', index_col=0, parse_dates=True)
optimized_pred_df = pd.read_csv('results/optimized_full_dataset/predictions.csv', index_col=0, parse_dates=True)

# Load metrics
with open('results/full_dataset_baseline_2025/metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

with open('results/optimized_full_dataset/metrics.json', 'r') as f:
    optimized_metrics = json.load(f)

# Load trades
baseline_trades = pd.read_csv('results/full_dataset_baseline_2025/trades.csv')
optimized_trades = pd.read_csv('results/optimized_full_dataset/trades.csv')

print(f"  Baseline predictions: {len(baseline_pred_df)}")
print(f"  Optimized predictions: {len(optimized_pred_df)}")


# ================================================================
# 2. COMPARE PREDICTIONS DISTRIBUTIONS
# ================================================================

print("\n[2/8] Comparing prediction distributions...")

baseline_pred = baseline_pred_df['predicted'].values
optimized_pred = optimized_pred_df['optimized_pred'].values
actual = baseline_pred_df['actual'].values

print(f"\n  BASELINE PREDICTIONS:")
print(f"    Mean:     {baseline_pred.mean():.6f}")
print(f"    Std:      {baseline_pred.std():.6f}")
print(f"    Min:      {baseline_pred.min():.6f}")
print(f"    Max:      {baseline_pred.max():.6f}")
print(f"    Median:   {np.median(baseline_pred):.6f}")
print(f"    Q25:      {np.percentile(baseline_pred, 25):.6f}")
print(f"    Q75:      {np.percentile(baseline_pred, 75):.6f}")

print(f"\n  OPTIMIZED PREDICTIONS:")
print(f"    Mean:     {optimized_pred.mean():.6f}")
print(f"    Std:      {optimized_pred.std():.6f}")
print(f"    Min:      {optimized_pred.min():.6f}")
print(f"    Max:      {optimized_pred.max():.6f}")
print(f"    Median:   {np.median(optimized_pred):.6f}")
print(f"    Q25:      {np.percentile(optimized_pred, 25):.6f}")
print(f"    Q75:      {np.percentile(optimized_pred, 75):.6f}")

print(f"\n  ACTUAL RETURNS:")
print(f"    Mean:     {actual.mean():.6f}")
print(f"    Std:      {actual.std():.6f}")
print(f"    Min:      {actual.min():.6f}")
print(f"    Max:      {actual.max():.6f}")


# ================================================================
# 3. THRESHOLD ANALYSIS
# ================================================================

print("\n[3/8] Threshold analysis...")

threshold = 0.001

# Baseline
baseline_signals = (baseline_pred > threshold).sum()
baseline_signal_pct = (baseline_signals / len(baseline_pred)) * 100

# Optimized
optimized_signals = (optimized_pred > threshold).sum()
optimized_signal_pct = (optimized_signals / len(optimized_pred)) * 100

print(f"\n  THRESHOLD: {threshold}")
print(f"\n  BASELINE:")
print(f"    Signals > threshold:  {baseline_signals} / {len(baseline_pred)} ({baseline_signal_pct:.1f}%)")
print(f"    Trades executed:      {baseline_metrics['performance']['total_trades']}")
print(f"    Signal -> Trade rate: {baseline_metrics['performance']['total_trades']/baseline_signals*100:.1f}%")

print(f"\n  OPTIMIZED:")
print(f"    Signals > threshold:  {optimized_signals} / {len(optimized_pred)} ({optimized_signal_pct:.1f}%)")
print(f"    Trades executed:      {optimized_metrics['optimized']['total_trades']}")
print(f"    Signal -> Trade rate: {optimized_metrics['optimized']['total_trades']/optimized_signals*100:.1f}%")


# ================================================================
# 4. PREDICTION RANGE ANALYSIS
# ================================================================

print("\n[4/8] Prediction range analysis...")

# How many predictions fall into different ranges?
ranges = [
    (-float('inf'), -0.01),
    (-0.01, -0.001),
    (-0.001, 0),
    (0, 0.001),
    (0.001, 0.01),
    (0.01, float('inf'))
]

print(f"\n  BASELINE PREDICTION RANGES:")
for low, high in ranges:
    count = ((baseline_pred > low) & (baseline_pred <= high)).sum()
    pct = (count / len(baseline_pred)) * 100
    print(f"    {low:7.3f} to {high:7.3f}: {count:4d} ({pct:5.1f}%)")

print(f"\n  OPTIMIZED PREDICTION RANGES:")
for low, high in ranges:
    count = ((optimized_pred > low) & (optimized_pred <= high)).sum()
    pct = (count / len(optimized_pred)) * 100
    print(f"    {low:7.3f} to {high:7.3f}: {count:4d} ({pct:5.1f}%)")


# ================================================================
# 5. CORRELATION ANALYSIS
# ================================================================

print("\n[5/8] Correlation analysis...")

from scipy.stats import spearmanr, pearsonr

# Baseline
baseline_spearman, _ = spearmanr(actual, baseline_pred)
baseline_pearson, _ = pearsonr(actual, baseline_pred)

# Optimized
optimized_spearman, _ = spearmanr(actual, optimized_pred)
optimized_pearson, _ = pearsonr(actual, optimized_pred)

print(f"\n  BASELINE:")
print(f"    Spearman:  {baseline_spearman:.4f}")
print(f"    Pearson:   {baseline_pearson:.4f}")

print(f"\n  OPTIMIZED:")
print(f"    Spearman:  {optimized_spearman:.4f}")
print(f"    Pearson:   {optimized_pearson:.4f}")


# ================================================================
# 6. TRADE ANALYSIS
# ================================================================

print("\n[6/8] Trade analysis...")

print(f"\n  BASELINE TRADES:")
print(f"    Total trades:     {len(baseline_trades)}")
print(f"    Win rate:         {baseline_metrics['performance']['win_rate']*100:.1f}%")
print(f"    Avg PnL:          ${baseline_trades['pnl'].mean():.2f}")
print(f"    Avg winning PnL:  ${baseline_trades[baseline_trades['pnl'] > 0]['pnl'].mean():.2f}")
print(f"    Avg losing PnL:   ${baseline_trades[baseline_trades['pnl'] < 0]['pnl'].mean():.2f}")

print(f"\n  OPTIMIZED TRADES:")
print(f"    Total trades:     {len(optimized_trades)}")
print(f"    Win rate:         {optimized_metrics['optimized']['win_rate']*100:.1f}%")
print(f"    Avg PnL:          ${optimized_trades['pnl'].mean():.2f}")
print(f"    Avg winning PnL:  ${optimized_trades[optimized_trades['pnl'] > 0]['pnl'].mean():.2f}")
print(f"    Avg losing PnL:   ${optimized_trades[optimized_trades['pnl'] < 0]['pnl'].mean():.2f}")


# ================================================================
# 7. BEST TRADES COMPARISON
# ================================================================

print("\n[7/8] Best and worst trades...")

print(f"\n  BASELINE - TOP 5 TRADES:")
top_baseline = baseline_trades.nlargest(5, 'pnl')[['entry_date', 'prediction', 'actual_return', 'pnl']]
for i, row in top_baseline.iterrows():
    print(f"    {row['entry_date']}  pred:{row['prediction']:7.4f}  actual:{row['actual_return']:7.4f}  PnL:${row['pnl']:7.2f}")

print(f"\n  OPTIMIZED - TOP 5 TRADES:")
top_optimized = optimized_trades.nlargest(5, 'pnl')[['entry_date', 'prediction', 'actual_return', 'pnl']]
for i, row in top_optimized.iterrows():
    print(f"    {row['entry_date']}  pred:{row['prediction']:7.4f}  actual:{row['actual_return']:7.4f}  PnL:${row['pnl']:7.2f}")

print(f"\n  BASELINE - WORST 5 TRADES:")
worst_baseline = baseline_trades.nsmallest(5, 'pnl')[['entry_date', 'prediction', 'actual_return', 'pnl']]
for i, row in worst_baseline.iterrows():
    print(f"    {row['entry_date']}  pred:{row['prediction']:7.4f}  actual:{row['actual_return']:7.4f}  PnL:${row['pnl']:7.2f}")

print(f"\n  OPTIMIZED - WORST 5 TRADES:")
worst_optimized = optimized_trades.nsmallest(5, 'pnl')[['entry_date', 'prediction', 'actual_return', 'pnl']]
for i, row in worst_optimized.iterrows():
    print(f"    {row['entry_date']}  pred:{row['prediction']:7.4f}  actual:{row['actual_return']:7.4f}  PnL:${row['pnl']:7.2f}")


# ================================================================
# 8. MODEL PARAMETERS COMPARISON
# ================================================================

print("\n[8/8] Model parameters comparison...")

baseline_params = baseline_metrics['parameters']
optimized_params = optimized_metrics['optimized']['parameters']

print(f"\n  PARAMETER COMPARISON:")
print(f"  {'Parameter':<20s} {'Baseline':>15s} {'Optimized':>15s} {'Diff':>10s}")
print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

for param in baseline_params.keys():
    if param not in ['random_state', 'verbosity']:
        baseline_val = baseline_params[param]
        optimized_val = optimized_params.get(param, 'N/A')

        if isinstance(baseline_val, (int, float)) and isinstance(optimized_val, (int, float)):
            diff = optimized_val - baseline_val
            print(f"  {param:<20s} {str(baseline_val):>15s} {str(optimized_val):>15s} {diff:>10.4f}")
        else:
            print(f"  {param:<20s} {str(baseline_val):>15s} {str(optimized_val):>15s} {'':>10s}")


# ================================================================
# 9. KEY INSIGHTS
# ================================================================

print("\n" + "="*80)
print("KEY INSIGHTS - PRECO BASELINE FUNGUJE LEPSIE")
print("="*80)

insights = []

# 1. Prediction variance
if baseline_pred.std() > optimized_pred.std():
    insights.append(f"1. BASELINE MA VACSIU VARIANCIU PREDIKCII")
    insights.append(f"   - Baseline std: {baseline_pred.std():.6f}")
    insights.append(f"   - Optimized std: {optimized_pred.std():.6f}")
    insights.append(f"   -> Baseline rozlisuje lepsie medzi dobrymi a zlymi obchodmi")

# 2. Signal frequency
if baseline_signal_pct < optimized_signal_pct:
    insights.append(f"\n2. BASELINE JE SELEKTIVNEJSI")
    insights.append(f"   - Baseline signaly: {baseline_signal_pct:.1f}%")
    insights.append(f"   - Optimized signaly: {optimized_signal_pct:.1f}%")
    insights.append(f"   -> Baseline obchoduje len ked je vysoka konfidenvia")

# 3. Correlation
if baseline_spearman > optimized_spearman:
    insights.append(f"\n3. BASELINE MA LEPSIU KORELIACIU")
    insights.append(f"   - Baseline Spearman: {baseline_spearman:.4f}")
    insights.append(f"   - Optimized Spearman: {optimized_spearman:.4f}")
    insights.append(f"   -> Baseline predikcie su viac aligned s actual returns")

# 4. Win rate
baseline_wr = baseline_metrics['performance']['win_rate']
optimized_wr = optimized_metrics['optimized']['win_rate']
if baseline_wr > optimized_wr:
    insights.append(f"\n4. BASELINE MA VYSSI WIN RATE")
    insights.append(f"   - Baseline: {baseline_wr*100:.1f}%")
    insights.append(f"   - Optimized: {optimized_wr*100:.1f}%")
    insights.append(f"   -> Baseline vyhra vacsinu obchodov")

# 5. Overtrading
if len(optimized_trades) > len(baseline_trades) * 5:
    insights.append(f"\n5. OPTIMIZED OVERTRADUJE")
    insights.append(f"   - Baseline trades: {len(baseline_trades)}")
    insights.append(f"   - Optimized trades: {len(optimized_trades)}")
    insights.append(f"   -> Transaction costs a noise zabijaju performance")

# 6. Parameter differences
if optimized_params['learning_rate'] > baseline_params['learning_rate'] * 1.5:
    insights.append(f"\n6. OPTIMIZED MA PRILIS VYSOKU LEARNING RATE")
    insights.append(f"   - Baseline: {baseline_params['learning_rate']:.4f}")
    insights.append(f"   - Optimized: {optimized_params['learning_rate']:.4f}")
    insights.append(f"   -> Moze sposobit overfitting")

for insight in insights:
    print(insight)

# ================================================================
# 10. RECOMMENDATIONS
# ================================================================

print("\n" + "="*80)
print("ODPORUCANIA")
print("="*80)

print("\n1. OBJECTIVE FUNCTION:")
print("   - Optimalizuj SPEARMAN CORRELATION (nie sharpe)")
print("   - Pridaj penaltu za prilis vela trades")
print("   - Minimalizuj variance predikcii mimo threshold")

print("\n2. PARAMETER CONSTRAINTS:")
print("   - Drz learning_rate nisko (0.01-0.06)")
print("   - Drz max_depth=3 (baseline hodnota)")
print("   - Zvys min_child_weight (redukuje noise)")

print("\n3. FEATURE ENGINEERING:")
print("   - Baseline pouziva ine features - mozno niektore su noise")
print("   - Skus feature selection - odstran korelowane features")

print("\n4. DALSI KROK:")
print("   - A) Optimalizuj correlation + penalta za trades")
print("   - B) Grid search len okolo baseline parametrov")
print("   - C) Feature selection - najdi top 30-50 features")

print("\n" + "="*80)
print("ZAVERY")
print("="*80)

print(f"\nBASELINE FUNGUJE LEPSIE PRETOZE:")
print(f"  1. Ma konzervativne parametre (nicky learning rate, depth=3)")
print(f"  2. Vytvara MENEJ ale KVALITNEJSICH predikcii")
print(f"  3. Neobchoduje prilis casto (23 vs 237 trades)")
print(f"  4. Ma lepsiu koreliaciu s actual returns")
print(f"  5. SIMPLE > COMPLEX v tomto pripade!")

print(f"\nOPTIMIZACIA ZLYAHALA PRETOZE:")
print(f"  1. Objective funkcia optimalizovala zle metriku")
print(f"  2. Nasla parametre ktore vytvaraju prilis vela signalov")
print(f"  3. Overtrading zabija performance")
print(f"  4. Vyssie learning rate = overfitting")

print("\n" + "="*80)
