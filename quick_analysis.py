"""
Quick Analysis of Top 50 S&P 500 Stocks
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TOP 50 S&P 500 STOCKS - QUICK ANALYSIS")
print("="*80)

# ================================================================
# 1. LOAD DATA
# ================================================================

print("\n[1/6] Loading data...")

summary = pd.read_csv('data/sp500_top50_summary.csv')

print(f"\nDataset Overview:")
print(f"   Total stocks: {len(summary)}")
print(f"   Average features: {summary['Features'].mean():.0f}")
print(f"   Average data points: {summary['DataPoints'].mean():.0f}")
print(f"   Date range: {summary['StartDate'].iloc[0]} -> {summary['EndDate'].iloc[0]}")

# Load all stocks
stocks = {}
for ticker in summary['Ticker']:
    file_path = f'data/sp500_top50/{ticker}_features.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    stocks[ticker] = df

print(f"\n[OK] Loaded {len(stocks)} stocks")

# ================================================================
# 2. SUMMARY STATISTICS
# ================================================================

print("\n[2/6] Calculating summary statistics...")

summary_stats = []

for ticker in summary.head(20)['Ticker']:
    df = stocks[ticker]

    stats = {
        'Ticker': ticker,
        'Rows': len(df),
        'Features': len(df.columns),
        'Avg Return (%)': df['return_1d'].mean() * 100,
        'Volatility (%)': df['return_1d'].std() * 100,
        'Sharpe Est': (df['return_1d'].mean() / df['return_1d'].std()) * np.sqrt(252) if df['return_1d'].std() > 0 else 0,
        'Min Price': df['Close'].min(),
        'Max Price': df['Close'].max(),
        'Current Price': df['Close'].iloc[-1],
        'Total Return (%)': (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100,
    }
    summary_stats.append(stats)

summary_df = pd.DataFrame(summary_stats)

print("\nCOMPREHENSIVE SUMMARY (Top 20)")
print("="*100)
print(summary_df.to_string(index=False))

# ================================================================
# 3. TOP PERFORMERS
# ================================================================

print("\n[3/6] Identifying top performers...")

print("\n\nTOP 5 BY TOTAL RETURN")
print("="*80)
top_return = summary_df.nlargest(5, 'Total Return (%)')
print(top_return[['Ticker', 'Total Return (%)', 'Avg Return (%)', 'Volatility (%)']].to_string(index=False))

print("\n\nTOP 5 BY SHARPE RATIO")
print("="*80)
top_sharpe = summary_df.nlargest(5, 'Sharpe Est')
print(top_sharpe[['Ticker', 'Sharpe Est', 'Avg Return (%)', 'Volatility (%)']].to_string(index=False))

print("\n\nTOP 5 HIGHEST VOLATILITY")
print("="*80)
top_vol = summary_df.nlargest(5, 'Volatility (%)')
print(top_vol[['Ticker', 'Volatility (%)', 'Avg Return (%)', 'Total Return (%)']].to_string(index=False))

# ================================================================
# 4. CORRELATION ANALYSIS
# ================================================================

print("\n[4/6] Analyzing correlations...")

returns_matrix = pd.DataFrame()
for ticker in summary.head(15)['Ticker']:
    returns_matrix[ticker] = stocks[ticker]['return_1d']

corr = returns_matrix.corr()
avg_corr = corr.mean().sort_values(ascending=False)

print("\nAVERAGE CORRELATIONS (Top 15)")
print("="*80)
for ticker, corr_val in avg_corr.items():
    print(f"  {ticker:6s}  {corr_val:.4f}")

# Find most/least correlated pairs
corr_pairs = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        ticker1 = corr.columns[i]
        ticker2 = corr.columns[j]
        corr_val = corr.iloc[i, j]
        corr_pairs.append((ticker1, ticker2, corr_val))

corr_pairs_df = pd.DataFrame(corr_pairs, columns=['Ticker1', 'Ticker2', 'Correlation'])
corr_pairs_df = corr_pairs_df.sort_values('Correlation', ascending=False)

print("\n\nTOP 5 MOST CORRELATED PAIRS")
print("="*80)
for _, row in corr_pairs_df.head(5).iterrows():
    print(f"  {row['Ticker1']:6s} <-> {row['Ticker2']:6s}  Correlation: {row['Correlation']:.4f}")

print("\n\nTOP 5 LEAST CORRELATED PAIRS")
print("="*80)
for _, row in corr_pairs_df.tail(5).iterrows():
    print(f"  {row['Ticker1']:6s} <-> {row['Ticker2']:6s}  Correlation: {row['Correlation']:.4f}")

# ================================================================
# 5. FEATURE ANALYSIS
# ================================================================

print("\n[5/6] Analyzing features...")

sample = stocks['AAPL']

feature_groups = {
    'Price': ['Open', 'High', 'Low', 'Close', 'Adj Close'],
    'Returns': [col for col in sample.columns if 'return' in col.lower()],
    'Volatility': [col for col in sample.columns if 'volatility' in col.lower() or 'vol' in col.lower() or 'atr' in col.lower()],
    'Volume': [col for col in sample.columns if 'volume' in col.lower() or 'obv' in col.lower() or 'mfi' in col.lower()],
    'Moving Averages': [col for col in sample.columns if 'sma' in col.lower() or 'ema' in col.lower()],
    'Oscillators': [col for col in sample.columns if 'rsi' in col.lower() or 'stoch' in col.lower() or 'willr' in col.lower()],
    'MACD': [col for col in sample.columns if 'macd' in col.lower()],
    'Bollinger': [col for col in sample.columns if 'bb_' in col.lower() or 'bollinger' in col.lower()],
    'Trend': [col for col in sample.columns if 'trend' in col.lower() or 'aroon' in col.lower() or 'adx' in col.lower()],
    'Sentiment': [col for col in sample.columns if 'fear' in col.lower() or 'vix' in col.lower() or 'btc' in col.lower() or 'sentiment' in col.lower()],
    'Patterns': [col for col in sample.columns if 'gap' in col.lower() or 'doji' in col.lower() or 'engulf' in col.lower()],
}

print("\nFEATURE GROUPS")
print("="*80)
for group, features in feature_groups.items():
    print(f"  {group:20s} {len(features):3d} features")

total_categorized = sum(len(f) for f in feature_groups.values())
print(f"\n  Total categorized: {total_categorized}/{len(sample.columns)}")

# ================================================================
# 6. DATA QUALITY
# ================================================================

print("\n[6/6] Checking data quality...")

print("\nDATA QUALITY CHECK")
print("="*80)

for ticker in summary.head(10)['Ticker']:
    df = stocks[ticker]
    total_nulls = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    null_pct = (total_nulls / total_cells) * 100

    print(f"  {ticker:6s}  Shape: {df.shape}  Nulls: {total_nulls:5d} ({null_pct:.2f}%)")

# ================================================================
# FINAL SUMMARY
# ================================================================

print("\n\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)

print(f"\n[OK] Analyzed {len(stocks)} stocks")
print(f"[OK] Each stock has {summary_df['Features'].mean():.0f} features")
print(f"[OK] Average {summary_df['Rows'].mean():.0f} data points per stock")

print(f"\nKey Insights:")
print(f"   • Best performer: {summary_df.nlargest(1, 'Total Return (%)').iloc[0]['Ticker']} ({summary_df.nlargest(1, 'Total Return (%)').iloc[0]['Total Return (%)']:.1f}% return)")
print(f"   • Best Sharpe: {summary_df.nlargest(1, 'Sharpe Est').iloc[0]['Ticker']} (Sharpe {summary_df.nlargest(1, 'Sharpe Est').iloc[0]['Sharpe Est']:.2f})")
print(f"   • Most volatile: {summary_df.nlargest(1, 'Volatility (%)').iloc[0]['Ticker']} ({summary_df.nlargest(1, 'Volatility (%)').iloc[0]['Volatility (%)']:.2f}% daily vol)")
print(f"   • Average correlation: {avg_corr.mean():.4f}")

print("\n" + "="*80)
print("To explore interactively, open: explore_sp500_top50.ipynb")
print("="*80)
