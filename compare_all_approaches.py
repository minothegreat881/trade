"""
COMPREHENSIVE COMPARISON OF ALL 3 APPROACHES
=============================================

Compares:
1. ORIGINAL (124 features, uniform params)
2. MULTI-SCALE (185 features, uniform params)
3. ADAPTIVE (custom features & params by volatility)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("COMPREHENSIVE APPROACH COMPARISON")
print("="*80)
print()


# ================================================================
# LOAD ALL RESULTS
# ================================================================

print("[1/5] Loading results from all approaches...")

original = pd.read_csv('results/sp500/training_summary.csv')
multiscale = pd.read_csv('results/sp500_multiscale/training_summary.csv')
adaptive = pd.read_csv('results/sp500_adaptive/training_summary.csv')

print(f"  Original:    {len(original)} models")
print(f"  Multi-Scale: {len(multiscale)} models")
print(f"  Adaptive:    {len(adaptive)} models")


# ================================================================
# OVERALL STATISTICS
# ================================================================

print("\n[2/5] Overall Performance Comparison")
print("="*80)

stats = {
    'Approach': ['ORIGINAL', 'MULTI-SCALE', 'ADAPTIVE'],
    'Avg Sharpe': [
        original['sharpe'].mean(),
        multiscale['sharpe'].mean(),
        adaptive['sharpe'].mean()
    ],
    'Median Sharpe': [
        original['sharpe'].median(),
        multiscale['sharpe'].median(),
        adaptive['sharpe'].median()
    ],
    'Std Sharpe': [
        original['sharpe'].std(),
        multiscale['sharpe'].std(),
        adaptive['sharpe'].std()
    ],
    'Min Sharpe': [
        original['sharpe'].min(),
        multiscale['sharpe'].min(),
        adaptive['sharpe'].min()
    ],
    'Max Sharpe': [
        original['sharpe'].max(),
        multiscale['sharpe'].max(),
        adaptive['sharpe'].max()
    ],
    'Avg Win Rate': [
        original['win_rate'].mean(),
        multiscale['win_rate'].mean(),
        adaptive['win_rate'].mean()
    ],
    'Avg Correlation': [
        original['test_corr'].mean(),
        multiscale['test_corr'].mean(),
        adaptive['test_corr'].mean()
    ]
}

stats_df = pd.DataFrame(stats)
print("\n" + stats_df.to_string(index=False))

# Highlight winner
best_approach = stats_df.loc[stats_df['Avg Sharpe'].idxmax(), 'Approach']
print(f"\nWINNER: {best_approach} (Highest Avg Sharpe)")


# ================================================================
# STOCK-BY-STOCK COMPARISON
# ================================================================

print("\n[3/5] Stock-by-Stock Comparison")
print("="*80)

# Merge all results
comparison = original[['ticker', 'sharpe', 'win_rate', 'test_corr']].copy()
comparison.columns = ['ticker', 'sharpe_orig', 'win_rate_orig', 'corr_orig']

multiscale_sub = multiscale[['ticker', 'sharpe', 'win_rate', 'test_corr']].copy()
multiscale_sub.columns = ['ticker', 'sharpe_ms', 'win_rate_ms', 'corr_ms']

adaptive_sub = adaptive[['ticker', 'sharpe', 'win_rate', 'test_corr', 'group']].copy()
adaptive_sub.columns = ['ticker', 'sharpe_adap', 'win_rate_adap', 'corr_adap', 'group']

comparison = comparison.merge(multiscale_sub, on='ticker', how='outer')
comparison = comparison.merge(adaptive_sub, on='ticker', how='outer')

# Determine best approach for each stock
comparison['best_sharpe'] = comparison[['sharpe_orig', 'sharpe_ms', 'sharpe_adap']].max(axis=1)
comparison['best_approach'] = comparison[['sharpe_orig', 'sharpe_ms', 'sharpe_adap']].idxmax(axis=1)
comparison['best_approach'] = comparison['best_approach'].map({
    'sharpe_orig': 'ORIGINAL',
    'sharpe_ms': 'MULTI-SCALE',
    'sharpe_adap': 'ADAPTIVE'
})

# Calculate deltas
comparison['delta_orig_to_ms'] = comparison['sharpe_ms'] - comparison['sharpe_orig']
comparison['delta_orig_to_adap'] = comparison['sharpe_adap'] - comparison['sharpe_orig']
comparison['delta_ms_to_adap'] = comparison['sharpe_adap'] - comparison['sharpe_ms']

# Sort by best sharpe
comparison = comparison.sort_values('best_sharpe', ascending=False)

# Save full comparison
comparison.to_csv('results/comparison_all_approaches.csv', index=False)
print(f"\nSaved detailed comparison to: results/comparison_all_approaches.csv")


# ================================================================
# BEST APPROACH SUMMARY
# ================================================================

print("\n[4/5] Best Approach Per Stock")
print("="*80)

best_counts = comparison['best_approach'].value_counts()
print(f"\nBest approach count:")
for approach, count in best_counts.items():
    print(f"  {approach:12s}: {count:2d} stocks ({count/len(comparison)*100:.1f}%)")

# Show top 20 stocks
print(f"\nTOP 20 STOCKS (with best approach):")
print("\nTicker | Best Approach | Sharpe | Orig  | Multi | Adap  | Group")
print("-" * 80)
for _, row in comparison.head(20).iterrows():
    print(f"{row['ticker']:6s} | {row['best_approach']:13s} | "
          f"{row['best_sharpe']:6.2f} | "
          f"{row['sharpe_orig']:5.2f} | "
          f"{row['sharpe_ms']:5.2f} | "
          f"{row['sharpe_adap']:5.2f} | "
          f"{row['group']:6s}")


# ================================================================
# IMPROVEMENT ANALYSIS
# ================================================================

print("\n[5/5] Improvement Analysis")
print("="*80)

print(f"\nORIGINAL -> MULTI-SCALE:")
print(f"  Improved:  {(comparison['delta_orig_to_ms'] > 0).sum()}/{len(comparison)} stocks")
print(f"  Degraded:  {(comparison['delta_orig_to_ms'] < 0).sum()}/{len(comparison)} stocks")
print(f"  Avg Delta: {comparison['delta_orig_to_ms'].mean():+.3f}")
print(f"  Max Gain:  {comparison['delta_orig_to_ms'].max():+.2f} ({comparison.loc[comparison['delta_orig_to_ms'].idxmax(), 'ticker']})")
print(f"  Max Loss:  {comparison['delta_orig_to_ms'].min():+.2f} ({comparison.loc[comparison['delta_orig_to_ms'].idxmin(), 'ticker']})")

print(f"\nORIGINAL -> ADAPTIVE:")
print(f"  Improved:  {(comparison['delta_orig_to_adap'] > 0).sum()}/{len(comparison)} stocks")
print(f"  Degraded:  {(comparison['delta_orig_to_adap'] < 0).sum()}/{len(comparison)} stocks")
print(f"  Avg Delta: {comparison['delta_orig_to_adap'].mean():+.3f}")
print(f"  Max Gain:  {comparison['delta_orig_to_adap'].max():+.2f} ({comparison.loc[comparison['delta_orig_to_adap'].idxmax(), 'ticker']})")
print(f"  Max Loss:  {comparison['delta_orig_to_adap'].min():+.2f} ({comparison.loc[comparison['delta_orig_to_adap'].idxmin(), 'ticker']})")

print(f"\nMULTI-SCALE -> ADAPTIVE:")
print(f"  Improved:  {(comparison['delta_ms_to_adap'] > 0).sum()}/{len(comparison)} stocks")
print(f"  Degraded:  {(comparison['delta_ms_to_adap'] < 0).sum()}/{len(comparison)} stocks")
print(f"  Avg Delta: {comparison['delta_ms_to_adap'].mean():+.3f}")
print(f"  Max Gain:  {comparison['delta_ms_to_adap'].max():+.2f} ({comparison.loc[comparison['delta_ms_to_adap'].idxmax(), 'ticker']})")
print(f"  Max Loss:  {comparison['delta_ms_to_adap'].min():+.2f} ({comparison.loc[comparison['delta_ms_to_adap'].idxmin(), 'ticker']})")


# ================================================================
# ADAPTIVE GROUP ANALYSIS
# ================================================================

print("\n" + "="*80)
print("ADAPTIVE APPROACH - GROUP PERFORMANCE")
print("="*80)

for group in ['HIGH', 'MEDIUM', 'LOW']:
    group_data = comparison[comparison['group'] == group]
    if len(group_data) > 0:
        print(f"\n{group} VOLATILITY ({len(group_data)} stocks):")
        print(f"  Avg Sharpe:      {group_data['sharpe_adap'].mean():6.2f}")
        print(f"  vs Original:     {group_data['delta_orig_to_adap'].mean():+6.2f}")
        print(f"  vs Multi-Scale:  {group_data['delta_ms_to_adap'].mean():+6.2f}")
        print(f"  Best in {(group_data['best_approach'] == 'ADAPTIVE').sum()}/{len(group_data)} stocks")


# ================================================================
# RECOMMENDATIONS
# ================================================================

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print(f"\nOVERALL WINNER: {best_approach}")
print(f"  Avg Sharpe: {stats_df.loc[stats_df['Approach'] == best_approach, 'Avg Sharpe'].values[0]:.3f}")

print(f"\nPORTFOLIO STRATEGY:")

# Count best approach per group
for group in ['HIGH', 'MEDIUM', 'LOW']:
    group_data = comparison[comparison['group'] == group]
    if len(group_data) > 0:
        best_in_group = group_data['best_approach'].value_counts()
        best_approach_for_group = best_in_group.idxmax()
        print(f"\n  {group} VOL stocks ({len(group_data)} stocks):")
        print(f"    Recommended: {best_approach_for_group}")
        for approach, count in best_in_group.items():
            print(f"      {approach:12s}: {count}/{len(group_data)} stocks")

print(f"\nHYBRID PORTFOLIO STRATEGY:")
print(f"  Use BEST model for each individual stock")
print(f"  - ORIGINAL:    {best_counts.get('ORIGINAL', 0)} stocks")
print(f"  - MULTI-SCALE: {best_counts.get('MULTI-SCALE', 0)} stocks")
print(f"  - ADAPTIVE:    {best_counts.get('ADAPTIVE', 0)} stocks")
print(f"  Expected Avg Sharpe: {comparison['best_sharpe'].mean():.3f}")

# Save best model selection
best_models = comparison[['ticker', 'best_approach', 'best_sharpe', 'group']].copy()
best_models.to_csv('results/best_model_per_stock.csv', index=False)
print(f"\nSaved best model selection to: results/best_model_per_stock.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"\nFiles created:")
print(f"  - results/comparison_all_approaches.csv (full comparison)")
print(f"  - results/best_model_per_stock.csv (best model selection)")

print(f"\nNext steps:")
print(f"  1. Create portfolio using best models per stock")
print(f"  2. Backtest hybrid portfolio strategy")
print(f"  3. Analyze feature importance by approach")

print("\n" + "="*80)
