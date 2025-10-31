import pandas as pd

train = pd.read_csv('data/train_test_sentiment/train_X.csv', index_col=0)

print('='*80)
print('26 FEATURES - USPESNY MODEL (Sharpe 1.28)')
print('='*80)
print()
print('CELKOVO FEATURES:', len(train.columns))
print()
print('ZOZNAM VSETKYCH FEATURES:')
print()
for i, col in enumerate(train.columns, 1):
    print(f'{i:2d}. {col}')

print()
print('='*80)
print('KATEGORIE:')
print('='*80)

tech = ['return_1d', 'return_5d', 'return_10d', 'return_20d',
        'volatility_20d', 'volatility_60d', 'volume_ratio',
        'price_position', 'sma_20', 'sma_50', 'trend']
sent_fg = [c for c in train.columns if 'fear_greed' in c]
sent_vix = [c for c in train.columns if 'vix' in c]
sent_btc = [c for c in train.columns if 'btc' in c]
sent_comp = [c for c in train.columns if 'composite' in c]

print()
print(f'TECHNICAL ({len(tech)}):')
for f in tech:
    if f in train.columns:
        print(f'  - {f}')

print()
print(f'SENTIMENT - Fear and Greed ({len(sent_fg)}):')
for f in sent_fg:
    print(f'  - {f}')

print()
print(f'SENTIMENT - VIX ({len(sent_vix)}):')
for f in sent_vix:
    print(f'  - {f}')

print()
print(f'SENTIMENT - Bitcoin ({len(sent_btc)}):')
for f in sent_btc:
    print(f'  - {f}')

print()
print(f'SENTIMENT - Composite ({len(sent_comp)}):')
for f in sent_comp:
    print(f'  - {f}')

print()
print('='*80)
print('SUMMARY:')
print('='*80)
print(f'Technical:         {len(tech)} features')
print(f'Sentiment Total:   {len(sent_fg) + len(sent_vix) + len(sent_btc) + len(sent_comp)} features')
print(f'  - Fear & Greed:  {len(sent_fg)}')
print(f'  - VIX:           {len(sent_vix)}')
print(f'  - Bitcoin:       {len(sent_btc)}')
print(f'  - Composite:     {len(sent_comp)}')
print(f'TOTAL:             {len(train.columns)} features')
print('='*80)
