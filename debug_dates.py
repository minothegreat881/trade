import pandas as pd

# Load technical data
train_X = pd.read_csv('data/train_test/train_X.csv', index_col=0)
train_X.index = pd.to_datetime(train_X.index, utc=True).tz_localize(None)

# Load sentiment data
sentiment = pd.read_csv('data/sentiment_cache/fear_greed_2020-02-29_2024-12-20.csv',
                       index_col=0, parse_dates=True)

print("=" * 70)
print("TECHNICAL DATA")
print("=" * 70)
print(f"Index type: {type(train_X.index)}")
print(f"First 5 dates:")
for date in train_X.index[:5]:
    print(f"  {date} (type: {type(date)})")
print(f"\nLast 5 dates:")
for date in train_X.index[-5:]:
    print(f"  {date} (type: {type(date)})")

print("\n" + "=" * 70)
print("SENTIMENT DATA")
print("=" * 70)
print(f"Index type: {type(sentiment.index)}")
print(f"First 5 dates:")
for date in sentiment.index[:5]:
    print(f"  {date} (type: {type(date)})")
print(f"\nLast 5 dates:")
for date in sentiment.index[-5:]:
    print(f"  {date} (type: {type(date)})")

print("\n" + "=" * 70)
print("OVERLAP CHECK")
print("=" * 70)

# Find common dates
common = train_X.index.intersection(sentiment.index)
print(f"Common dates: {len(common)}")

if len(common) > 0:
    print(f"First common: {common[0]}")
    print(f"Last common: {common[-1]}")
else:
    print("\nNo common dates! Checking date formats...")
    print(f"\nTechnical sample: {train_X.index[0]}")
    print(f"Sentiment sample: {sentiment.index[0]}")

    # Check if dates are close
    tech_dates_set = set(train_X.index.date)
    sent_dates_set = set(sentiment.index.date)
    overlap_dates = tech_dates_set.intersection(sent_dates_set)
    print(f"\nDates that match (ignoring time): {len(overlap_dates)}")

    if len(overlap_dates) > 0:
        print("Problem: Dates match but time component differs!")
