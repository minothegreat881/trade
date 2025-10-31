import pandas as pd

# Check CSV features
train = pd.read_csv('data/train_test_sentiment/train_X.csv', index_col=0, parse_dates=True)

print('='*70)
print('FEATURES IN train_X.csv')
print('='*70)
print(f'Total columns: {len(train.columns)}\n')

print('All columns:')
for i, col in enumerate(train.columns, 1):
    print(f'{i:2d}. {col}')

# Sentiment columns
sent_cols = [col for col in train.columns
             if 'fear' in col.lower() or 'vix' in col.lower()
             or 'sentiment' in col.lower() or 'btc' in col.lower()]

print(f'\n\nSentiment-related columns ({len(sent_cols)}):')
for col in sent_cols:
    print(f'  - {col}')

# Technical columns
tech_cols = [col for col in train.columns if col not in sent_cols]
print(f'\n\nTechnical columns ({len(tech_cols)}):')
for col in tech_cols:
    print(f'  - {col}')
