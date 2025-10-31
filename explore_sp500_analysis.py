[NbConvertApp] Converting notebook explore_sp500_top50.ipynb to python
#!/usr/bin/env python
# coding: utf-8

# # 📊 TOP 50 S&P 500 STOCKS - EXPLORATORY ANALYSIS
# 
# **Dataset:** Top 50 S&P 500 stocks by market capitalization  
# **Features:** 133 technical indicators per stock  
# **Period:** 2020-2025 (~1,011 trading days)  
# 
# ---

# In[ ]:


# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Plot settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
get_ipython().run_line_magic('matplotlib', 'inline')

print("✓ Libraries imported")


# ---
# ## 1. LOAD DATA
# ---

# In[ ]:


# Load summary
summary = pd.read_csv('data/sp500_top50_summary.csv')

print(f"📁 Top 50 S&P 500 Stocks")
print(f"   Total stocks: {len(summary)}")
print(f"   Average features: {summary['Features'].mean():.0f}")
print(f"   Average data points: {summary['DataPoints'].mean():.0f}")
print(f"   Date range: {summary['StartDate'].iloc[0]} → {summary['EndDate'].iloc[0]}")

summary.head(15)


# In[ ]:


# Load all stocks into dictionary
stocks = {}

for ticker in summary['Ticker']:
    file_path = f'data/sp500_top50/{ticker}_features.csv'
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    stocks[ticker] = df
    
print(f"✓ Loaded {len(stocks)} stocks")
print(f"\nSample (AAPL):")
print(f"  Shape: {stocks['AAPL'].shape}")
print(f"  Columns: {len(stocks['AAPL'].columns)}")
print(f"  Date range: {stocks['AAPL'].index.min()} → {stocks['AAPL'].index.max()}")


# ---
# ## 2. DATA OVERVIEW
# ---

# In[ ]:


# Feature groups
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

print("📋 FEATURE GROUPS")
print("="*80)
for group, features in feature_groups.items():
    print(f"  {group:20s} {len(features):3d} features")
    
total_categorized = sum(len(f) for f in feature_groups.values())
print(f"\n  Total categorized: {total_categorized}/{len(sample.columns)}")


# In[ ]:


# Top 15 stocks by market cap
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Market Cap
top15 = summary.head(15).copy()
axes[0,0].barh(top15['Ticker'], top15['MarketCap_B'], color='steelblue')
axes[0,0].set_xlabel('Market Cap (Billions $)')
axes[0,0].set_title('Top 15 Stocks by Market Cap')
axes[0,0].invert_yaxis()

# 2. Average Returns
axes[0,1].barh(top15['Ticker'], top15['AvgReturn']*100, color='green')
axes[0,1].set_xlabel('Average Daily Return (%)')
axes[0,1].set_title('Average Returns')
axes[0,1].invert_yaxis()
axes[0,1].axvline(0, color='black', linestyle='--', alpha=0.3)

# 3. Volatility
axes[1,0].barh(top15['Ticker'], top15['Volatility']*100, color='orange')
axes[1,0].set_xlabel('Volatility (%)')
axes[1,0].set_title('Daily Volatility')
axes[1,0].invert_yaxis()

# 4. Sharpe Estimate (Return/Volatility)
top15['sharpe_est'] = top15['AvgReturn'] / top15['Volatility']
axes[1,1].barh(top15['Ticker'], top15['sharpe_est'], color='purple')
axes[1,1].set_xlabel('Return/Volatility Ratio')
axes[1,1].set_title('Risk-Adjusted Return (Estimate)')
axes[1,1].invert_yaxis()

plt.tight_layout()
plt.show()


# ---
# ## 3. PRICE PERFORMANCE
# ---

# In[ ]:


# Normalize prices to 100 at start
fig, ax = plt.subplots(figsize=(16, 8))

top10_tickers = summary.head(10)['Ticker'].tolist()

for ticker in top10_tickers:
    df = stocks[ticker]
    normalized = (df['Close'] / df['Close'].iloc[0]) * 100
    ax.plot(normalized.index, normalized, label=ticker, linewidth=2)

ax.set_xlabel('Date')
ax.set_ylabel('Normalized Price (Start = 100)')
ax.set_title('Top 10 Stocks - Price Performance (2020-2025)', fontsize=14, fontweight='bold')
ax.legend(loc='best', ncol=2)
ax.grid(True, alpha=0.3)
ax.axhline(100, color='black', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate total returns
returns_data = []
for ticker in top10_tickers:
    df = stocks[ticker]
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    returns_data.append({'Ticker': ticker, 'Total Return (%)': total_return})
    
returns_df = pd.DataFrame(returns_data).sort_values('Total Return (%)', ascending=False)
print("\n📈 TOTAL RETURNS (2020-2025)")
print("="*80)
print(returns_df.to_string(index=False))


# ---
# ## 4. CORRELATION ANALYSIS
# ---

# In[ ]:


# Create returns dataframe for all stocks
returns_matrix = pd.DataFrame()

for ticker in summary.head(20)['Ticker']:
    returns_matrix[ticker] = stocks[ticker]['return_1d']

# Correlation matrix
corr = returns_matrix.corr()

fig, ax = plt.subplots(figsize=(14, 12))
sns.heatmap(corr, annot=False, cmap='RdYlGn', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            vmin=-0.5, vmax=1.0, ax=ax)
ax.set_title('Returns Correlation Matrix (Top 20 Stocks)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# Average correlations
avg_corr = corr.mean().sort_values(ascending=False)
print("\n🔗 AVERAGE CORRELATIONS")
print("="*80)
print(avg_corr.to_string())


# ---
# ## 5. FEATURE DISTRIBUTIONS
# ---

# In[ ]:


# Sample stock for feature analysis
ticker_sample = 'AAPL'
df_sample = stocks[ticker_sample]

# Key features to visualize
key_features = ['return_1d', 'volatility_20d', 'rsi', 'volume_ratio', 
                'macd', 'bb_position', 'trend', 'vix_regime']

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
axes = axes.flatten()

for i, feature in enumerate(key_features):
    if feature in df_sample.columns:
        axes[i].hist(df_sample[feature].dropna(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Distribution: {feature}')
        axes[i].grid(True, alpha=0.3)
        
        # Add stats
        mean_val = df_sample[feature].mean()
        std_val = df_sample[feature].std()
        axes[i].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        axes[i].legend()

plt.suptitle(f'Feature Distributions ({ticker_sample})', fontsize=16, fontweight='bold', y=1.001)
plt.tight_layout()
plt.show()


# ---
# ## 6. VOLATILITY ANALYSIS
# ---

# In[ ]:


# Volatility over time for top stocks
fig, ax = plt.subplots(figsize=(16, 8))

for ticker in ['NVDA', 'TSLA', 'AMD', 'AAPL', 'MSFT']:
    df = stocks[ticker]
    ax.plot(df.index, df['volatility_20d']*100, label=ticker, linewidth=2)

ax.set_xlabel('Date')
ax.set_ylabel('20-Day Volatility (%)')
ax.set_title('Volatility Over Time (Selected Stocks)', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Volatility ranking
vol_data = []
for ticker in summary.head(20)['Ticker']:
    df = stocks[ticker]
    avg_vol = df['volatility_20d'].mean() * 100
    max_vol = df['volatility_20d'].max() * 100
    vol_data.append({'Ticker': ticker, 'Avg Volatility (%)': avg_vol, 'Max Volatility (%)': max_vol})

vol_df = pd.DataFrame(vol_data).sort_values('Avg Volatility (%)', ascending=False)
print("\n📊 VOLATILITY RANKING (Top 20)")
print("="*80)
print(vol_df.to_string(index=False))


# ---
# ## 7. TARGET VARIABLE ANALYSIS
# ---

# In[ ]:


# Target distribution across stocks
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Target distribution for AAPL
df_aapl = stocks['AAPL']
axes[0,0].hist(df_aapl['target'].dropna()*100, bins=50, alpha=0.7, color='green', edgecolor='black')
axes[0,0].set_xlabel('5-Day Forward Return (%)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].set_title('Target Distribution (AAPL)')
axes[0,0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0,0].grid(True, alpha=0.3)

# 2. Target vs Return_1d
axes[0,1].scatter(df_aapl['return_1d']*100, df_aapl['target']*100, alpha=0.3, s=10)
axes[0,1].set_xlabel('1-Day Return (%)')
axes[0,1].set_ylabel('5-Day Forward Return (%)')
axes[0,1].set_title('Target vs 1-Day Return')
axes[0,1].axhline(0, color='red', linestyle='--', alpha=0.3)
axes[0,1].axvline(0, color='red', linestyle='--', alpha=0.3)
axes[0,1].grid(True, alpha=0.3)

# 3. Target vs RSI
axes[1,0].scatter(df_aapl['rsi'], df_aapl['target']*100, alpha=0.3, s=10, c='purple')
axes[1,0].set_xlabel('RSI')
axes[1,0].set_ylabel('5-Day Forward Return (%)')
axes[1,0].set_title('Target vs RSI')
axes[1,0].axhline(0, color='red', linestyle='--', alpha=0.3)
axes[1,0].grid(True, alpha=0.3)

# 4. Average target by stock
target_avgs = []
for ticker in summary.head(15)['Ticker']:
    avg_target = stocks[ticker]['target'].mean() * 100
    target_avgs.append(avg_target)

axes[1,1].barh(summary.head(15)['Ticker'], target_avgs, color='teal')
axes[1,1].set_xlabel('Average 5-Day Forward Return (%)')
axes[1,1].set_title('Average Target by Stock (Top 15)')
axes[1,1].invert_yaxis()
axes[1,1].axvline(0, color='red', linestyle='--', alpha=0.3)
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ---
# ## 8. SENTIMENT ANALYSIS
# ---

# In[ ]:


# Sentiment features over time
df_aapl = stocks['AAPL']

fig, axes = plt.subplots(3, 1, figsize=(16, 12))

# 1. Fear & Greed Index
if 'fear_greed_value' in df_aapl.columns:
    axes[0].plot(df_aapl.index, df_aapl['fear_greed_value'], linewidth=2, color='orange')
    axes[0].fill_between(df_aapl.index, 0, df_aapl['fear_greed_value'], alpha=0.3, color='orange')
    axes[0].axhline(25, color='red', linestyle='--', label='Extreme Fear', alpha=0.5)
    axes[0].axhline(75, color='green', linestyle='--', label='Extreme Greed', alpha=0.5)
    axes[0].set_ylabel('Fear & Greed Index')
    axes[0].set_title('Fear & Greed Index Over Time', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# 2. VIX
if 'VIX' in df_aapl.columns:
    axes[1].plot(df_aapl.index, df_aapl['VIX'], linewidth=2, color='red')
    axes[1].fill_between(df_aapl.index, 0, df_aapl['VIX'], alpha=0.3, color='red')
    axes[1].axhline(20, color='orange', linestyle='--', label='Elevated VIX', alpha=0.5)
    axes[1].axhline(30, color='darkred', linestyle='--', label='High VIX', alpha=0.5)
    axes[1].set_ylabel('VIX')
    axes[1].set_title('VIX (Volatility Index) Over Time', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

# 3. Bitcoin Price
if 'BTC_Close' in df_aapl.columns:
    axes[2].plot(df_aapl.index, df_aapl['BTC_Close'], linewidth=2, color='gold')
    axes[2].fill_between(df_aapl.index, df_aapl['BTC_Close'].min(), df_aapl['BTC_Close'], alpha=0.3, color='gold')
    axes[2].set_ylabel('Bitcoin Price ($)')
    axes[2].set_xlabel('Date')
    axes[2].set_title('Bitcoin Price Over Time', fontweight='bold')
    axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ---
# ## 9. DATA QUALITY CHECK
# ---

# In[ ]:


# Check for missing values
print("📋 DATA QUALITY CHECK")
print("="*80)

for ticker in summary.head(10)['Ticker']:
    df = stocks[ticker]
    total_nulls = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    null_pct = (total_nulls / total_cells) * 100
    
    print(f"  {ticker:6s}  Shape: {df.shape}  Nulls: {total_nulls:5d} ({null_pct:.2f}%)")

# Feature completeness
df_sample = stocks['AAPL']
feature_completeness = (1 - df_sample.isnull().sum() / len(df_sample)) * 100
incomplete_features = feature_completeness[feature_completeness < 100].sort_values()

if len(incomplete_features) > 0:
    print(f"\n⚠️  INCOMPLETE FEATURES (< 100% data):")
    print(incomplete_features)
else:
    print(f"\n✓ All features have 100% complete data!")


# ---
# ## 10. QUICK STATS SUMMARY
# ---

# In[ ]:


# Create comprehensive summary
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

print("\n📊 COMPREHENSIVE SUMMARY (Top 20)")
print("="*100)
print(summary_df.to_string(index=False))

# Best performers
print("\n\n🏆 TOP PERFORMERS")
print("="*80)
print("\nBy Total Return:")
print(summary_df.nlargest(5, 'Total Return (%)')[['Ticker', 'Total Return (%)']].to_string(index=False))

print("\nBy Sharpe Ratio:")
print(summary_df.nlargest(5, 'Sharpe Est')[['Ticker', 'Sharpe Est', 'Avg Return (%)', 'Volatility (%)']].to_string(index=False))

print("\n\n📉 HIGHEST VOLATILITY")
print("="*80)
print(summary_df.nlargest(5, 'Volatility (%)')[['Ticker', 'Volatility (%)', 'Avg Return (%)']].to_string(index=False))


# ---
# ## 11. EXPORT HELPER FUNCTIONS
# ---

# In[ ]:


# Helper function to get stock data
def get_stock(ticker):
    """Get stock data by ticker"""
    if ticker in stocks:
        return stocks[ticker]
    else:
        print(f"Ticker {ticker} not found")
        return None

# Helper function to compare stocks
def compare_stocks(tickers, feature='Close'):
    """Compare multiple stocks on a feature"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for ticker in tickers:
        if ticker in stocks and feature in stocks[ticker].columns:
            df = stocks[ticker]
            # Normalize
            normalized = (df[feature] / df[feature].iloc[0]) * 100
            ax.plot(normalized.index, normalized, label=ticker, linewidth=2)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(f'Normalized {feature} (Start = 100)')
    ax.set_title(f'Comparison: {feature}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(100, color='black', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

print("✓ Helper functions defined:")
print("  - get_stock(ticker)")
print("  - compare_stocks(tickers, feature='Close')")
print("\nExample usage:")
print("  df = get_stock('AAPL')")
print("  compare_stocks(['AAPL', 'MSFT', 'GOOGL'], 'Close')")


# ---
# ## 💡 NEXT STEPS
# 
# Now that you've explored the data, you can:
# 
# 1. **Train models on individual stocks**
#    - See which stocks are most predictable
#    
# 2. **Portfolio optimization**
#    - Select best stocks by Sharpe ratio
#    - Diversification analysis
#    
# 3. **Feature selection**
#    - Identify most important features
#    - Reduce dimensionality
#    
# 4. **Cross-stock predictions**
#    - Use correlated stocks as features
#    - Sector momentum
# 
# ---
