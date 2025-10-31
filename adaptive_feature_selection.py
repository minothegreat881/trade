"""
ADAPTIVE FEATURE SELECTION FOR TOP 50 S&P 500 STOCKS
=====================================================

Classifies stocks by volatility and selects optimal features:
- HIGH VOL (>3%): 50-80 simple SHORT-TERM features
- LOW VOL (<1.5%): ALL 185 multi-scale features
- MEDIUM VOL (1.5-3%): 100 hybrid features (short+medium)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ADAPTIVE FEATURE SELECTION")
print("="*80)


# ================================================================
# STEP 1: CALCULATE VOLATILITY FOR EACH STOCK
# ================================================================

print("\n[1/4] Calculating volatility for all stocks...")

summary = pd.read_csv('data/sp500_top50_summary.csv')
stock_volatility = []

for ticker in summary['Ticker']:
    # Load original data to get clean volatility
    df = pd.read_csv(f'data/sp500_top50/{ticker}_features.csv',
                     index_col=0, parse_dates=True)

    # Calculate average daily volatility
    volatility = df['return_1d'].std() * 100  # Convert to percentage

    stock_volatility.append({
        'ticker': ticker,
        'volatility_pct': volatility
    })

vol_df = pd.DataFrame(stock_volatility)
vol_df = vol_df.sort_values('volatility_pct', ascending=False)

print(f"  Calculated volatility for {len(vol_df)} stocks")
print(f"\n  Range: {vol_df['volatility_pct'].min():.2f}% - {vol_df['volatility_pct'].max():.2f}%")


# ================================================================
# STEP 2: CLASSIFY INTO GROUPS
# ================================================================

print("\n[2/4] Classifying stocks into volatility groups...")

# Define thresholds
HIGH_VOL_THRESHOLD = 3.0
LOW_VOL_THRESHOLD = 1.5

# Classify
vol_df['group'] = 'MEDIUM'
vol_df.loc[vol_df['volatility_pct'] > HIGH_VOL_THRESHOLD, 'group'] = 'HIGH'
vol_df.loc[vol_df['volatility_pct'] < LOW_VOL_THRESHOLD, 'group'] = 'LOW'

# Count by group
group_counts = vol_df['group'].value_counts()

print(f"\n  Classification complete:")
print(f"    HIGH VOL (>{HIGH_VOL_THRESHOLD}%):   {group_counts.get('HIGH', 0)} stocks")
print(f"    MEDIUM VOL ({LOW_VOL_THRESHOLD}-{HIGH_VOL_THRESHOLD}%): {group_counts.get('MEDIUM', 0)} stocks")
print(f"    LOW VOL (<{LOW_VOL_THRESHOLD}%):  {group_counts.get('LOW', 0)} stocks")

# Display groups
print("\n  GROUP A - HIGH VOLATILITY (>3%):")
high_vol = vol_df[vol_df['group'] == 'HIGH']
for _, row in high_vol.iterrows():
    print(f"    {row['ticker']:6s}  {row['volatility_pct']:5.2f}%")

print("\n  GROUP B - LOW VOLATILITY (<1.5%):")
low_vol = vol_df[vol_df['group'] == 'LOW']
for _, row in low_vol.iterrows():
    print(f"    {row['ticker']:6s}  {row['volatility_pct']:5.2f}%")

print("\n  GROUP C - MEDIUM VOLATILITY (1.5-3%):")
med_vol = vol_df[vol_df['group'] == 'MEDIUM']
for _, row in med_vol.head(10).iterrows():  # Show first 10
    print(f"    {row['ticker']:6s}  {row['volatility_pct']:5.2f}%")
if len(med_vol) > 10:
    print(f"    ... and {len(med_vol) - 10} more")


# ================================================================
# STEP 3: DEFINE FEATURE SELECTION STRATEGY
# ================================================================

print("\n[3/4] Defining feature selection strategies...")

# Load a sample to get all available features
sample_original = pd.read_csv('data/sp500_top50/AAPL_features.csv',
                              index_col=0, parse_dates=True)
sample_multiscale = pd.read_csv('data/sp500_multiscale/AAPL_multiscale.csv',
                                index_col=0, parse_dates=True)

# Exclude columns
exclude_cols = [
    'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
    'Dividends', 'Stock Splits', 'fear_greed_classification',
    'target', 'target_5d_return', 'target_profit_3pct',
    'target_profit_any', 'target_max_drawdown_5d', 'target_max_profit_5d'
]

# Get all available features
original_features = [col for col in sample_original.columns if col not in exclude_cols]
multiscale_features = [col for col in sample_multiscale.columns if col not in exclude_cols]

# Categorize multi-scale features
short_term_features = [col for col in multiscale_features if col.startswith('st_')]
medium_term_features = [col for col in multiscale_features if col.startswith('mt_')]
long_term_features = [col for col in multiscale_features if col.startswith('lt_')]

print(f"\n  Available feature pools:")
print(f"    Original features:  {len(original_features)}")
print(f"    Multi-scale total:  {len(multiscale_features)}")
print(f"      - SHORT-TERM:     {len(short_term_features)}")
print(f"      - MEDIUM-TERM:    {len(medium_term_features)}")
print(f"      - LONG-TERM:      {len(long_term_features)}")

# Feature selection for each group
feature_strategy = {
    'HIGH': {
        'description': 'Simple SHORT-TERM features only',
        'features': original_features[:80],  # Top 80 original features
        'count': min(80, len(original_features)),
        'xgb_params': {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.5,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    },
    'LOW': {
        'description': 'ALL multi-scale features',
        'features': multiscale_features,
        'count': len(multiscale_features),
        'xgb_params': {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 150,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.01,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    },
    'MEDIUM': {
        'description': 'Hybrid: SHORT + MEDIUM term features',
        'features': (short_term_features + medium_term_features + original_features[:50])[:100],
        'count': 100,
        'xgb_params': {
            'max_depth': 4,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'min_child_weight': 1,
            'subsample': 0.8,
            'colsample_bytree': 0.6,
            'gamma': 0.05,
            'reg_alpha': 0.05,
            'reg_lambda': 0.05,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
    }
}

print(f"\n  Feature strategy per group:")
print(f"    HIGH:   {feature_strategy['HIGH']['count']} features - {feature_strategy['HIGH']['description']}")
print(f"    LOW:    {feature_strategy['LOW']['count']} features - {feature_strategy['LOW']['description']}")
print(f"    MEDIUM: {feature_strategy['MEDIUM']['count']} features - {feature_strategy['MEDIUM']['description']}")


# ================================================================
# STEP 4: CREATE STOCK-TO-STRATEGY MAPPING
# ================================================================

print("\n[4/4] Creating stock-to-strategy mapping...")

# Add strategy info to each stock
vol_df['feature_count'] = vol_df['group'].map(lambda g: feature_strategy[g]['count'])
vol_df['max_depth'] = vol_df['group'].map(lambda g: feature_strategy[g]['xgb_params']['max_depth'])
vol_df['data_source'] = vol_df['group'].map(lambda g: 'multiscale' if g in ['LOW', 'MEDIUM'] else 'original')

# Save classification
output_file = 'data/stock_volatility_classification.csv'
vol_df.to_csv(output_file, index=False)
print(f"  Saved classification to: {output_file}")

# Save feature strategy
strategy_file = 'data/adaptive_feature_strategy.json'
strategy_export = {
    group: {
        'description': info['description'],
        'feature_count': info['count'],
        'features': info['features'],
        'xgb_params': info['xgb_params']
    }
    for group, info in feature_strategy.items()
}

with open(strategy_file, 'w') as f:
    json.dump(strategy_export, f, indent=2)
print(f"  Saved feature strategy to: {strategy_file}")


# ================================================================
# SUMMARY
# ================================================================

print("\n" + "="*80)
print("ADAPTIVE FEATURE SELECTION COMPLETE!")
print("="*80)

print(f"\nStock Classification:")
print(f"  HIGH VOL:   {len(high_vol):2d} stocks -> {feature_strategy['HIGH']['count']:3d} features (original data)")
print(f"  MEDIUM VOL: {len(med_vol):2d} stocks -> {feature_strategy['MEDIUM']['count']:3d} features (hybrid)")
print(f"  LOW VOL:    {len(low_vol):2d} stocks -> {feature_strategy['LOW']['count']:3d} features (multi-scale)")

print(f"\nFiles created:")
print(f"  - data/stock_volatility_classification.csv")
print(f"  - data/adaptive_feature_strategy.json")

print(f"\nNext step:")
print(f"  Run: python train_adaptive_models.py")

print("\n" + "="*80)
