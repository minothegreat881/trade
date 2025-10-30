"""
Quick script to inspect processed data
Visual inspection of train/test data quality
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Check if data exists
if not os.path.exists('data/train_test/train_data.csv'):
    print("❌ Error: train_data.csv not found!")
    print("   Run pipeline.py first")
    exit(1)

# Load data
print("📂 Loading data...")
train = pd.read_csv('data/train_test/train_data.csv',
                    index_col=0, parse_dates=True)
test = pd.read_csv('data/train_test/test_data.csv',
                   index_col=0, parse_dates=True)

print("=" * 60)
print("TRAIN DATA")
print("=" * 60)
print(f"Rows: {len(train)}")
print(f"Date range: {train.index.min()} to {train.index.max()}")
print(f"\nColumns: {list(train.columns)}")
print(f"\nFirst 5 rows:")
print(train.head())
print(f"\nTarget statistics:")
print(train['target'].describe())

print("\n" + "=" * 60)
print("TEST DATA")
print("=" * 60)
print(f"Rows: {len(test)}")
print(f"Date range: {test.index.min()} to {test.index.max()}")
print(f"\nTarget statistics:")
print(test['target'].describe())

# Create plots
print("\n📊 Generating plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Target distribution
axes[0, 0].hist(train['target'], bins=50, alpha=0.7, label='Train', color='blue')
axes[0, 0].hist(test['target'], bins=50, alpha=0.7, label='Test', color='orange')
axes[0, 0].set_title('Target Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('5-Day Return')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Returns over time
axes[0, 1].plot(train.index, train['target'],
                alpha=0.5, label='Train', color='blue')
axes[0, 1].plot(test.index, test['target'],
                alpha=0.5, label='Test', color='orange')
axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_title('Target Over Time', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('5-Day Return')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Volatility
if 'volatility_20d' in train.columns:
    axes[1, 0].plot(train.index, train['volatility_20d'],
                    label='Train', color='blue')
    axes[1, 0].plot(test.index, test['volatility_20d'],
                    label='Test', color='orange')
    axes[1, 0].set_title('Volatility (20d)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Volatility')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Trend
if 'trend' in train.columns:
    train_trend_ma = train['trend'].rolling(20).mean()
    test_trend_ma = test['trend'].rolling(20).mean()
    axes[1, 1].plot(train.index, train_trend_ma,
                    label='Train', color='blue')
    axes[1, 1].plot(test.index, test_trend_ma, label='Test', color='orange')
    axes[1, 1].set_title('Trend (20d MA)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Uptrend %')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()

# Save plot
output_path = 'data/data_inspection.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to {output_path}")

# Show plot
plt.show()

print("\n" + "=" * 60)
print("✅ DATA INSPECTION COMPLETE")
print("=" * 60)
print("\n📋 Quick checks:")
print(f"  ✓ Target centered around 0: {abs(train['target'].mean()) < 0.01}")
print(
    f"  ✓ Train/test similar std: {abs(train['target'].std() - test['target'].std()) < 0.01}")
print(
    f"  ✓ No extreme outliers: {(train['target'].abs() < 0.5).sum() / len(train) > 0.95}")

print("\n💡 Review data_inspection.png for visual confirmation!")
