"""
Porovnanie sentiment datasetu (Sharpe 1.28) vs advanced datasetu (Sharpe 0.47)
"""
import pandas as pd

print("="*80)
print("POROVNANIE DATASETOV")
print("="*80)

# SENTIMENT DATASET (dobry model - Sharpe 1.28)
print("\n[1] SENTIMENT DATASET (Sharpe 1.28)")
print("-"*80)

train_sent = pd.read_csv('data/train_test_sentiment/train_X.csv', index_col=0, parse_dates=True)
test_sent = pd.read_csv('data/train_test_sentiment/test_X.csv', index_col=0, parse_dates=True)

print(f"\nTRAIN:")
print(f"  Vzorky:   {len(train_sent)}")
print(f"  Features: {len(train_sent.columns)}")
print(f"  Obdobie:  {train_sent.index.min().strftime('%Y-%m-%d')} -> {train_sent.index.max().strftime('%Y-%m-%d')}")
print(f"  Dni:      {(train_sent.index.max() - train_sent.index.min()).days}")

print(f"\nTEST:")
print(f"  Vzorky:   {len(test_sent)}")
print(f"  Features: {len(test_sent.columns)}")
print(f"  Obdobie:  {test_sent.index.min().strftime('%Y-%m-%d')} -> {test_sent.index.max().strftime('%Y-%m-%d')}")
print(f"  Dni:      {(test_sent.index.max() - test_sent.index.min()).days}")

print(f"\nFEATURES (prvych 30):")
for i, col in enumerate(train_sent.columns[:30], 1):
    print(f"  {i:2d}. {col}")
if len(train_sent.columns) > 30:
    print(f"  ... a {len(train_sent.columns)-30} dalsich")

# Sentiment features
sent_features = [col for col in train_sent.columns if 'fear' in col.lower() or 'greed' in col.lower() or 'sentiment' in col.lower()]
print(f"\nSENTIMENT FEATURES ({len(sent_features)}):")
for feat in sent_features:
    print(f"  - {feat}")


# ADVANCED DATASET (optimalizovany model - Sharpe 0.47)
print("\n\n[2] ADVANCED DATASET (Sharpe 0.47)")
print("-"*80)

train_adv = pd.read_csv('data/train_advanced.csv', index_col=0, parse_dates=True)
test_adv = pd.read_csv('data/test_advanced.csv', index_col=0, parse_dates=True)

print(f"\nTRAIN:")
print(f"  Vzorky:   {len(train_adv)}")
print(f"  Features: {len(train_adv.columns)}")
print(f"  Obdobie:  {train_adv.index.min().strftime('%Y-%m-%d')} -> {train_adv.index.max().strftime('%Y-%m-%d')}")
print(f"  Dni:      {(train_adv.index.max() - train_adv.index.min()).days}")

print(f"\nTEST:")
print(f"  Vzorky:   {len(test_adv)}")
print(f"  Features: {len(test_adv.columns)}")
print(f"  Obdobie:  {test_adv.index.min().strftime('%Y-%m-%d')} -> {test_adv.index.max().strftime('%Y-%m-%d')}")
print(f"  Dni:      {(test_adv.index.max() - test_adv.index.min()).days}")

# Exclude target and price columns
exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_classification']
feature_cols_adv = [col for col in train_adv.columns if col not in exclude_cols]

print(f"\nFEATURES (prvych 30 z {len(feature_cols_adv)} features):")
for i, col in enumerate(feature_cols_adv[:30], 1):
    print(f"  {i:2d}. {col}")
if len(feature_cols_adv) > 30:
    print(f"  ... a {len(feature_cols_adv)-30} dalsich")


# ROZDIEL
print("\n\n[3] POROVNANIE")
print("-"*80)
print(f"\n                    SENTIMENT      ADVANCED      ROZDIEL")
print(f"Train vzorky:       {len(train_sent):6d}         {len(train_adv):6d}        {len(train_sent)-len(train_adv):+6d}")
print(f"Test vzorky:        {len(test_sent):6d}         {len(test_adv):6d}        {len(test_sent)-len(test_adv):+6d}")
print(f"Features:           {len(train_sent.columns):6d}         {len(feature_cols_adv):6d}        {len(train_sent.columns)-len(feature_cols_adv):+6d}")
print(f"Sharpe:              1.28           0.47          -0.81")

# Spolocne features
common = set(train_sent.columns) & set(train_adv.columns)
only_sent = set(train_sent.columns) - set(train_adv.columns)
only_adv = set(train_adv.columns) - set(train_sent.columns)

print(f"\nSpolocne features:     {len(common)}")
print(f"Len v sentiment:       {len(only_sent)}")
print(f"Len v advanced:        {len(only_adv)}")

print(f"\nVYNIMAJUCE SENTIMENT FEATURES:")
sent_specific = [f for f in only_sent if 'fear' in f.lower() or 'greed' in f.lower() or 'sentiment' in f.lower()]
for feat in sent_specific[:10]:
    print(f"  - {feat}")
if len(sent_specific) > 10:
    print(f"  ... a {len(sent_specific)-10} dalsich")

print("\n" + "="*80)
