import pandas as pd

df = pd.read_csv('data/sp500_top50/AAPL_features.csv', index_col=0)

print('='*80)
print('SAMPLE: AAPL FEATURES')
print('='*80)
print(f'\nRows: {len(df)}')
print(f'Columns: {len(df.columns)}')
print(f'\nAll Features ({len(df.columns)}):')
for i, col in enumerate(df.columns, 1):
    print(f'  {i:2d}. {col}')

print(f'\nFirst 3 rows:')
print(df.head(3))
