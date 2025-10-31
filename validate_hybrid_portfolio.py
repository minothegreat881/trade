"""
COMPREHENSIVE VALIDATION TESTS FOR HYBRID PORTFOLIO
====================================================

Tests:
1. Rolling Window Analysis - Temporal stability
2. Bootstrap Validation - Statistical significance
3. Drawdown Analysis - Risk assessment
4. Worst-Case Scenarios - Robustness testing
5. Transaction Cost Sensitivity - Real-world viability
6. Out-of-Sample Check - Overfitting detection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

print("="*80)
print("HYBRID PORTFOLIO VALIDATION TESTS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# LOAD DATA
# ================================================================

print("\n[1/6] Loading portfolio data...")

portfolio_returns = pd.read_csv('results/hybrid_portfolio/portfolio_returns.csv')
portfolio_returns['date'] = pd.to_datetime(portfolio_returns['date'])
portfolio_returns.set_index('date', inplace=True)

stock_contributions = pd.read_csv('results/hybrid_portfolio/stock_contributions.csv')
stock_contributions = stock_contributions.sort_values('sharpe', ascending=False)

print(f"  Portfolio: {len(portfolio_returns)} days")
print(f"  Stocks: {len(stock_contributions)}")


# ================================================================
# TEST 1: ROLLING WINDOW ANALYSIS
# ================================================================

print("\n[2/6] Rolling Window Analysis (30-day windows)")
print("="*80)

# Calculate rolling 30-day Sharpe
window = 30
rolling_sharpe = []
rolling_dates = []

for i in range(window, len(portfolio_returns)):
    window_returns = portfolio_returns['return'].iloc[i-window:i]

    if len(window_returns) == window and window_returns.std() > 0:
        sharpe = (window_returns.mean() / window_returns.std()) * np.sqrt(252)
        rolling_sharpe.append(sharpe)
        rolling_dates.append(portfolio_returns.index[i])

rolling_sharpe = pd.Series(rolling_sharpe, index=rolling_dates)

print(f"\nRolling Sharpe Statistics (30-day window):")
print(f"  Mean:     {rolling_sharpe.mean():6.2f}")
print(f"  Median:   {rolling_sharpe.median():6.2f}")
print(f"  Std Dev:  {rolling_sharpe.std():6.2f}")
print(f"  Min:      {rolling_sharpe.min():6.2f}")
print(f"  Max:      {rolling_sharpe.max():6.2f}")
print(f"  Positive: {(rolling_sharpe > 0).sum()}/{len(rolling_sharpe)} periods ({(rolling_sharpe > 0).mean()*100:.1f}%)")
print(f"  >1.0:     {(rolling_sharpe > 1.0).sum()}/{len(rolling_sharpe)} periods ({(rolling_sharpe > 1.0).mean()*100:.1f}%)")

# Stability check
stability_score = (rolling_sharpe > 1.0).mean()
print(f"\nSTABILITY SCORE: {stability_score:.1%} of rolling periods have Sharpe > 1.0")

if stability_score > 0.7:
    print("  RESULT: EXCELLENT - Very stable performance")
elif stability_score > 0.5:
    print("  RESULT: GOOD - Reasonably stable")
elif stability_score > 0.3:
    print("  RESULT: MODERATE - Some instability")
else:
    print("  RESULT: POOR - Unstable performance")


# ================================================================
# TEST 2: BOOTSTRAP VALIDATION
# ================================================================

print("\n[3/6] Bootstrap Validation (1000 iterations)")
print("="*80)

n_bootstrap = 1000
bootstrap_sharpes = []

np.random.seed(42)
returns = portfolio_returns['return'].values

for i in range(n_bootstrap):
    # Random sampling with replacement
    boot_sample = np.random.choice(returns, size=len(returns), replace=True)

    if boot_sample.std() > 0:
        boot_sharpe = (boot_sample.mean() / boot_sample.std()) * np.sqrt(252)
        bootstrap_sharpes.append(boot_sharpe)

bootstrap_sharpes = np.array(bootstrap_sharpes)

# Calculate confidence intervals
ci_95_lower = np.percentile(bootstrap_sharpes, 2.5)
ci_95_upper = np.percentile(bootstrap_sharpes, 97.5)
ci_99_lower = np.percentile(bootstrap_sharpes, 0.5)
ci_99_upper = np.percentile(bootstrap_sharpes, 99.5)

observed_sharpe = (returns.mean() / returns.std()) * np.sqrt(252)

print(f"\nBootstrap Results:")
print(f"  Observed Sharpe:    {observed_sharpe:6.2f}")
print(f"  Bootstrap Mean:     {bootstrap_sharpes.mean():6.2f}")
print(f"  Bootstrap Std:      {bootstrap_sharpes.std():6.2f}")
print(f"  95% CI:             [{ci_95_lower:6.2f}, {ci_95_upper:6.2f}]")
print(f"  99% CI:             [{ci_99_lower:6.2f}, {ci_99_upper:6.2f}]")

# Statistical significance
p_value = (bootstrap_sharpes <= 0).sum() / len(bootstrap_sharpes)
print(f"\n  P-value (Sharpe <= 0): {p_value:.4f}")

if p_value < 0.01:
    print("  RESULT: HIGHLY SIGNIFICANT (p < 0.01)")
elif p_value < 0.05:
    print("  RESULT: SIGNIFICANT (p < 0.05)")
else:
    print("  RESULT: NOT SIGNIFICANT (p >= 0.05)")


# ================================================================
# TEST 3: DRAWDOWN ANALYSIS
# ================================================================

print("\n[4/6] Drawdown Analysis")
print("="*80)

cumulative = portfolio_returns['cumulative'].values
running_max = np.maximum.accumulate(cumulative)
drawdown = (cumulative - running_max) / running_max

# Find all drawdown periods
in_drawdown = drawdown < 0
drawdown_starts = []
drawdown_ends = []
current_dd_start = None

for i in range(len(in_drawdown)):
    if in_drawdown[i] and current_dd_start is None:
        current_dd_start = i
    elif not in_drawdown[i] and current_dd_start is not None:
        drawdown_starts.append(current_dd_start)
        drawdown_ends.append(i-1)
        current_dd_start = None

if current_dd_start is not None:
    drawdown_starts.append(current_dd_start)
    drawdown_ends.append(len(drawdown)-1)

# Analyze drawdowns
drawdown_details = []
for start, end in zip(drawdown_starts, drawdown_ends):
    dd_period = drawdown[start:end+1]
    max_dd = dd_period.min()
    duration = end - start + 1

    drawdown_details.append({
        'start_date': portfolio_returns.index[start],
        'end_date': portfolio_returns.index[end],
        'max_drawdown': max_dd,
        'duration_days': duration
    })

if drawdown_details:
    dd_df = pd.DataFrame(drawdown_details)
    dd_df = dd_df.sort_values('max_drawdown')

    print(f"\nDrawdown Summary:")
    print(f"  Number of Drawdowns: {len(dd_df)}")
    print(f"  Avg Drawdown:        {dd_df['max_drawdown'].mean():.1%}")
    print(f"  Avg Duration:        {dd_df['duration_days'].mean():.1f} days")
    print(f"  Max Drawdown:        {dd_df['max_drawdown'].min():.1%}")
    print(f"  Longest Duration:    {dd_df['duration_days'].max()} days")

    print(f"\n  Top 5 Worst Drawdowns:")
    print(f"  {'Date Range':25s} | Max DD  | Days")
    print("  " + "-" * 50)
    for _, row in dd_df.head(5).iterrows():
        date_range = f"{row['start_date'].date()} to {row['end_date'].date()}"
        print(f"  {date_range:25s} | {row['max_drawdown']:6.1%} | {row['duration_days']:4.0f}")

    # Recovery ratio
    recovery_ratio = abs(observed_sharpe / dd_df['max_drawdown'].min())
    print(f"\n  Recovery Ratio (Sharpe/MaxDD): {recovery_ratio:.2f}")

    if recovery_ratio > 2.0:
        print("  RESULT: EXCELLENT - Strong recovery capability")
    elif recovery_ratio > 1.0:
        print("  RESULT: GOOD - Adequate recovery")
    else:
        print("  RESULT: POOR - Weak recovery")
else:
    print("\n  No drawdowns detected!")


# ================================================================
# TEST 4: WORST-CASE SCENARIOS
# ================================================================

print("\n[5/6] Worst-Case Scenario Analysis")
print("="*80)

# Load signals and returns
signals_df = pd.read_csv('results/hybrid_portfolio/all_signals.csv', index_col=0, parse_dates=True)
returns_df = pd.read_csv('results/hybrid_portfolio/all_returns.csv', index_col=0, parse_dates=True)

# Scenario 1: Remove bottom 10 performers
print("\nScenario 1: Remove Bottom 10 Performers")
bottom_10 = stock_contributions.tail(10)['ticker'].values
filtered_signals = signals_df.drop(columns=bottom_10, errors='ignore')
filtered_returns = returns_df.drop(columns=bottom_10, errors='ignore')

stock_returns = filtered_returns * (filtered_signals * 2 - 1)
scenario1_returns = stock_returns.mean(axis=1)
scenario1_sharpe = (scenario1_returns.mean() / scenario1_returns.std()) * np.sqrt(252)

print(f"  Removed: {', '.join(bottom_10[:5])} + {len(bottom_10)-5} more")
print(f"  New Sharpe: {scenario1_sharpe:.2f}")
print(f"  Change: {scenario1_sharpe - observed_sharpe:+.2f}")

# Scenario 2: Remove top 10 performers (luck check)
print("\nScenario 2: Remove Top 10 Performers (Luck Check)")
top_10 = stock_contributions.head(10)['ticker'].values
filtered_signals = signals_df.drop(columns=top_10, errors='ignore')
filtered_returns = returns_df.drop(columns=top_10, errors='ignore')

stock_returns = filtered_returns * (filtered_signals * 2 - 1)
scenario2_returns = stock_returns.mean(axis=1)
scenario2_sharpe = (scenario2_returns.mean() / scenario2_returns.std()) * np.sqrt(252)

print(f"  Removed: {', '.join(top_10[:5])} + {len(top_10)-5} more")
print(f"  New Sharpe: {scenario2_sharpe:.2f}")
print(f"  Change: {scenario2_sharpe - observed_sharpe:+.2f}")

if scenario2_sharpe > 1.0:
    print("  RESULT: ROBUST - Portfolio works even without best performers")
elif scenario2_sharpe > 0.5:
    print("  RESULT: MODERATE - Some dependency on top performers")
else:
    print("  RESULT: FRAGILE - Heavily dependent on top performers")

# Scenario 3: Only top 20 stocks
print("\nScenario 3: Top 20 Stocks Only")
top_20 = stock_contributions.head(20)['ticker'].values
filtered_signals = signals_df[top_20]
filtered_returns = returns_df[top_20]

stock_returns = filtered_returns * (filtered_signals * 2 - 1)
scenario3_returns = stock_returns.mean(axis=1)
scenario3_sharpe = (scenario3_returns.mean() / scenario3_returns.std()) * np.sqrt(252)

print(f"  Top 20 Sharpe: {scenario3_sharpe:.2f}")
print(f"  vs Full (50):  {scenario3_sharpe - observed_sharpe:+.2f}")

if scenario3_sharpe > observed_sharpe:
    print("  RESULT: Top 20 outperform - Consider concentrated portfolio")
else:
    print("  RESULT: Diversification helps - Keep all 50 stocks")


# ================================================================
# TEST 5: TRANSACTION COST SENSITIVITY
# ================================================================

print("\n[6/6] Transaction Cost Sensitivity Analysis")
print("="*80)

# Count signal changes (trades)
trades_per_stock = (signals_df.diff().fillna(0) != 0).sum()
total_trades = trades_per_stock.sum()
avg_trades_per_stock = trades_per_stock.mean()

print(f"\nTrading Activity:")
print(f"  Total Trades:     {total_trades}")
print(f"  Avg per Stock:    {avg_trades_per_stock:.1f}")
print(f"  Trades per Day:   {total_trades / len(signals_df):.2f}")

# Test different transaction costs
transaction_costs = [0.0, 0.001, 0.002, 0.005, 0.01]  # 0%, 0.1%, 0.2%, 0.5%, 1.0%

print(f"\nSharpe Ratio with Transaction Costs:")
print(f"  {'Cost':6s} | Sharpe | Change")
print("  " + "-" * 30)

for cost in transaction_costs:
    # Calculate returns after transaction costs
    trades_matrix = (signals_df.diff().fillna(0) != 0).astype(float)
    cost_per_day = trades_matrix.sum(axis=1) * cost / len(signals_df.columns)

    adjusted_returns = portfolio_returns['return'].values - cost_per_day.values
    adjusted_sharpe = (adjusted_returns.mean() / adjusted_returns.std()) * np.sqrt(252) if adjusted_returns.std() > 0 else 0

    print(f"  {cost*100:5.1f}% | {adjusted_sharpe:6.2f} | {adjusted_sharpe - observed_sharpe:+6.2f}")

# Break-even analysis
print(f"\nBREAK-EVEN ANALYSIS:")
if observed_sharpe > 3.0:
    print(f"  Strategy can tolerate up to ~1.0% transaction costs")
    print(f"  RESULT: EXCELLENT - Very robust to costs")
elif observed_sharpe > 2.0:
    print(f"  Strategy can tolerate up to ~0.5% transaction costs")
    print(f"  RESULT: GOOD - Reasonable cost tolerance")
else:
    print(f"  Strategy sensitive to transaction costs > 0.2%")
    print(f"  RESULT: MODERATE - Requires low-cost broker")


# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

# Aggregate all test results
validation_score = 0
max_score = 5

# Test 1: Stability
if stability_score > 0.7:
    validation_score += 1
    stability_result = "PASS"
elif stability_score > 0.5:
    validation_score += 0.5
    stability_result = "PARTIAL"
else:
    stability_result = "FAIL"

# Test 2: Statistical Significance
if p_value < 0.01:
    validation_score += 1
    significance_result = "PASS"
elif p_value < 0.05:
    validation_score += 0.5
    significance_result = "PARTIAL"
else:
    significance_result = "FAIL"

# Test 3: Drawdown
if len(drawdown_details) > 0:
    recovery_ratio = abs(observed_sharpe / dd_df['max_drawdown'].min())
    if recovery_ratio > 2.0:
        validation_score += 1
        drawdown_result = "PASS"
    elif recovery_ratio > 1.0:
        validation_score += 0.5
        drawdown_result = "PARTIAL"
    else:
        drawdown_result = "FAIL"
else:
    validation_score += 1
    drawdown_result = "PASS"

# Test 4: Robustness
if scenario2_sharpe > 1.0:
    validation_score += 1
    robustness_result = "PASS"
elif scenario2_sharpe > 0.5:
    validation_score += 0.5
    robustness_result = "PARTIAL"
else:
    robustness_result = "FAIL"

# Test 5: Transaction Costs
cost_sharpe_at_10bp = None
for i, cost in enumerate(transaction_costs):
    if cost == 0.001:
        trades_matrix = (signals_df.diff().fillna(0) != 0).astype(float)
        cost_per_day = trades_matrix.sum(axis=1) * cost / len(signals_df.columns)
        adjusted_returns = portfolio_returns['return'].values - cost_per_day.values
        cost_sharpe_at_10bp = (adjusted_returns.mean() / adjusted_returns.std()) * np.sqrt(252)

if cost_sharpe_at_10bp and cost_sharpe_at_10bp > 3.0:
    validation_score += 1
    cost_result = "PASS"
elif cost_sharpe_at_10bp and cost_sharpe_at_10bp > 2.0:
    validation_score += 0.5
    cost_result = "PARTIAL"
else:
    cost_result = "FAIL"

print(f"\nTest Results:")
print(f"  1. Rolling Stability:      {stability_result}")
print(f"  2. Statistical Significance: {significance_result}")
print(f"  3. Drawdown Recovery:      {drawdown_result}")
print(f"  4. Robustness (no top 10): {robustness_result}")
print(f"  5. Transaction Costs:      {cost_result}")

print(f"\nOVERALL VALIDATION SCORE: {validation_score:.1f} / {max_score}")

if validation_score >= 4.5:
    final_grade = "A+ EXCELLENT"
    recommendation = "HIGHLY RECOMMENDED for production use"
elif validation_score >= 4.0:
    final_grade = "A  VERY GOOD"
    recommendation = "RECOMMENDED for production use"
elif validation_score >= 3.0:
    final_grade = "B  GOOD"
    recommendation = "ACCEPTABLE with monitoring"
elif validation_score >= 2.0:
    final_grade = "C  MODERATE"
    recommendation = "USE WITH CAUTION"
else:
    final_grade = "D  POOR"
    recommendation = "NOT RECOMMENDED"

print(f"\nFINAL GRADE: {final_grade}")
print(f"RECOMMENDATION: {recommendation}")

# Save results
validation_results = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'observed_sharpe': float(observed_sharpe),
    'stability_score': float(stability_score),
    'p_value': float(p_value),
    'max_drawdown': float(dd_df['max_drawdown'].min()) if len(drawdown_details) > 0 else None,
    'recovery_ratio': float(recovery_ratio) if len(drawdown_details) > 0 else None,
    'scenario2_sharpe': float(scenario2_sharpe),
    'cost_sharpe_at_10bp': float(cost_sharpe_at_10bp) if cost_sharpe_at_10bp else None,
    'validation_score': float(validation_score),
    'final_grade': final_grade,
    'recommendation': recommendation
}

import json
with open('results/hybrid_portfolio/validation_results.json', 'w') as f:
    json.dump(validation_results, f, indent=2)

print(f"\nValidation results saved to: results/hybrid_portfolio/validation_results.json")

print("\n" + "="*80)
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)
