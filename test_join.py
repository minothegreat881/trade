import pandas as pd

# Load data
train_ohlcv = pd.read_csv('data/train_test/train_data.csv', index_col=0, parse_dates=True)
train_X = pd.read_csv('data/train_test_sentiment/train_X.csv', index_col=0, parse_dates=True)
train_y = pd.read_csv('data/train_test_sentiment/train_y.csv', index_col=0, parse_dates=True)

print("BEFORE timezone fix:")
print(f"OHLCV index type: {train_ohlcv.index}")
print(f"train_X index type: {train_X.index}")
print(f"OHLCV first date: {train_ohlcv.index[0]}")
print(f"train_X first date: {train_X.index[0]}")

# Remove timezone - proper way
train_ohlcv.index = pd.to_datetime(train_ohlcv.index).tz_localize(None)

print("\nAFTER timezone fix:")
print(f"OHLCV first date: {train_ohlcv.index[0]}")
print(f"train_X first date: {train_X.index[0]}")
print(f"Are they equal? {train_ohlcv.index[0] == train_X.index[0]}")

# Try join
merged = train_ohlcv[['Open', 'High', 'Low', 'Close', 'Volume']].join(train_X, how='inner')
merged['target'] = train_y['target']

print(f"\nJoined data shape: {merged.shape}")
print(f"Expected shape: ({len(train_ohlcv)}, {5 + len(train_X.columns) + 1})")
print(f"\nFirst 3 rows:")
print(merged.head(3))
print(f"\nTarget stats:")
print(merged['target'].describe())
