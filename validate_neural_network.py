"""
KOMPLEXNA VALIDACIA NEURAL NETWORK MODELU
==========================================

Testuje ci Neural Network (Sharpe 2.63) nie je overfitted:

1. Walk-Forward Validation (6 folds)
2. Time-Series Cross-Validation (5 folds)
3. Out-of-Sample Testing (rozne roky)
4. Monte Carlo Simulations (100 runs)
5. Permutation Testing (random target)
6. Stability Analysis (noise resistance)
7. Train/Val/Test Comparison

"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# Import neural network architecture
import sys
sys.path.insert(0, str(Path(__file__).parent))
from neural_network_trading import DeepFeatureNetwork, train_epoch, validate
from backtester import Backtester

print("="*80)
print("KOMPLEXNA VALIDACIA NEURAL NETWORK")
print("="*80)
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def calculate_sharpe(returns):
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or returns.std() == 0:
        return 0
    return (returns.mean() / returns.std()) * np.sqrt(252)


def train_and_evaluate(X_train, y_train, X_test, y_test, test_prices, device='cpu'):
    """
    Train NN model and evaluate
    Returns: sharpe, correlation, metrics dict
    """
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train_scaled).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test_scaled).to(device)

    # Create model
    model = DeepFeatureNetwork(
        input_dim=X_train.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3
    ).to(device)

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(50):  # Max 50 epochs
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Early stopping
        if loss.item() < best_loss - 1e-5:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).cpu().numpy()

    # Calculate correlation
    corr, _ = spearmanr(y_test, test_pred)

    # Backtest
    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.001,
        holding_period=5,
        position_size_pct=0.5,
        prediction_threshold=0.001
    )

    results = backtester.run_backtest(test_pred, y_test, test_prices)

    if len(results['returns']) > 0:
        sharpe = calculate_sharpe(results['returns'])
    else:
        sharpe = 0

    metrics = {
        'sharpe': float(sharpe),
        'correlation': float(corr),
        'n_trades': results['n_trades'],
        'final_capital': float(results['final_capital']),
        'total_return': float((results['final_capital'] / 10000) - 1)
    }

    return sharpe, corr, metrics


# ================================================================
# 1. WALK-FORWARD VALIDATION
# ================================================================

print("\n[1/7] WALK-FORWARD VALIDATION")
print("-" * 80)

df = pd.read_csv('data/full_dataset_2020_2025.csv', index_col=0, parse_dates=True)

exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Dividends', 'Stock Splits', 'Capital Gains',
                'fear_greed_value', 'fear_greed_classification',
                'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

feature_cols = [col for col in df.columns if col not in exclude_cols]
df = df.dropna(subset=feature_cols + ['target'])

# Split into 6 sequential folds
n_folds = 6
fold_size = len(df) // n_folds

walk_forward_results = []

print(f"\nTesting {n_folds} sequential folds...")

for fold in range(1, n_folds):
    # Train on all data up to this fold
    train_end = fold * fold_size
    test_start = train_end
    test_end = (fold + 1) * fold_size

    train_data = df.iloc[:train_end]
    test_data = df.iloc[test_start:test_end]

    if len(test_data) < 50:  # Skip if too small
        continue

    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    test_prices = test_data['Close'].values

    print(f"\n  Fold {fold}: Train={len(train_data)}, Test={len(test_data)}")
    print(f"    Period: {test_data.index[0].strftime('%Y-%m-%d')} -> {test_data.index[-1].strftime('%Y-%m-%d')}")

    sharpe, corr, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, test_prices)

    print(f"    Sharpe: {sharpe:.2f}, Correlation: {corr:.4f}, Trades: {metrics['n_trades']}")

    walk_forward_results.append({
        'fold': fold,
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'test_period': f"{test_data.index[0].date()} to {test_data.index[-1].date()}",
        **metrics
    })

wf_df = pd.DataFrame(walk_forward_results)

print(f"\n  WALK-FORWARD SUMMARY:")
print(f"    Mean Sharpe:      {wf_df['sharpe'].mean():.2f}")
print(f"    Std Sharpe:       {wf_df['sharpe'].std():.2f}")
print(f"    Min Sharpe:       {wf_df['sharpe'].min():.2f}")
print(f"    Max Sharpe:       {wf_df['sharpe'].max():.2f}")
print(f"    Mean Correlation: {wf_df['correlation'].mean():.4f}")


# ================================================================
# 2. TIME-SERIES CROSS-VALIDATION
# ================================================================

print("\n[2/7] TIME-SERIES CROSS-VALIDATION")
print("-" * 80)

tscv = TimeSeriesSplit(n_splits=5)

cv_results = []

X_all = df[feature_cols].values
y_all = df['target'].values
prices_all = df['Close'].values

print(f"\nTesting 5-fold time-series CV...")

for fold, (train_idx, test_idx) in enumerate(tscv.split(X_all), 1):
    X_train = X_all[train_idx]
    y_train = y_all[train_idx]
    X_test = X_all[test_idx]
    y_test = y_all[test_idx]
    test_prices = prices_all[test_idx]

    print(f"\n  Fold {fold}: Train={len(train_idx)}, Test={len(test_idx)}")

    sharpe, corr, metrics = train_and_evaluate(X_train, y_train, X_test, y_test, test_prices)

    print(f"    Sharpe: {sharpe:.2f}, Correlation: {corr:.4f}")

    cv_results.append({
        'fold': fold,
        'train_samples': len(train_idx),
        'test_samples': len(test_idx),
        **metrics
    })

cv_df = pd.DataFrame(cv_results)

print(f"\n  CROSS-VALIDATION SUMMARY:")
print(f"    Mean Sharpe:      {cv_df['sharpe'].mean():.2f} ± {cv_df['sharpe'].std():.2f}")
print(f"    Min Sharpe:       {cv_df['sharpe'].min():.2f}")
print(f"    Max Sharpe:       {cv_df['sharpe'].max():.2f}")
print(f"    Mean Correlation: {cv_df['correlation'].mean():.4f}")


# ================================================================
# 3. OUT-OF-SAMPLE TESTING (rozne roky)
# ================================================================

print("\n[3/7] OUT-OF-SAMPLE TESTING (po rokoch)")
print("-" * 80)

years = [2020, 2021, 2022, 2023, 2024, 2025]
oos_results = []

# Train on 2020-2023, test on 2024-2025
train_data = df[df.index.year < 2024]
X_train_all = train_data[feature_cols].values
y_train_all = train_data['target'].values

print(f"\nTrain period: 2020-2023 ({len(train_data)} samples)")

for year in [2024, 2025]:
    test_data = df[df.index.year == year]

    if len(test_data) < 20:
        continue

    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    test_prices = test_data['Close'].values

    print(f"\n  Testing year {year}: {len(test_data)} samples")

    sharpe, corr, metrics = train_and_evaluate(X_train_all, y_train_all, X_test, y_test, test_prices)

    print(f"    Sharpe: {sharpe:.2f}, Correlation: {corr:.4f}, Trades: {metrics['n_trades']}")

    oos_results.append({
        'year': year,
        'test_samples': len(test_data),
        **metrics
    })

oos_df = pd.DataFrame(oos_results)

if len(oos_df) > 0:
    print(f"\n  OUT-OF-SAMPLE SUMMARY:")
    print(f"    Mean Sharpe:      {oos_df['sharpe'].mean():.2f}")
    print(f"    Mean Correlation: {oos_df['correlation'].mean():.4f}")


# ================================================================
# 4. MONTE CARLO SIMULATIONS
# ================================================================

print("\n[4/7] MONTE CARLO SIMULATIONS (100 runs)")
print("-" * 80)

# Use original train/test split but with random initialization
split_idx = int(len(df) * 0.7)
train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]

X_train = train_data[feature_cols].values
y_train = train_data['target'].values
X_test = test_data[feature_cols].values
y_test = test_data['target'].values
test_prices = test_data['Close'].values

mc_sharpes = []
mc_correlations = []

print("\nRunning 100 simulations with different random seeds...")

for i in range(100):
    if (i + 1) % 20 == 0:
        print(f"  Completed {i+1}/100 runs...")

    # Set random seed
    torch.manual_seed(i)
    np.random.seed(i)

    sharpe, corr, _ = train_and_evaluate(X_train, y_train, X_test, y_test, test_prices)

    mc_sharpes.append(sharpe)
    mc_correlations.append(corr)

mc_sharpes = np.array(mc_sharpes)
mc_correlations = np.array(mc_correlations)

print(f"\n  MONTE CARLO SUMMARY:")
print(f"    Mean Sharpe:      {mc_sharpes.mean():.2f} ± {mc_sharpes.std():.2f}")
print(f"    Median Sharpe:    {np.median(mc_sharpes):.2f}")
print(f"    Min Sharpe:       {mc_sharpes.min():.2f}")
print(f"    Max Sharpe:       {mc_sharpes.max():.2f}")
print(f"    95% CI:           [{np.percentile(mc_sharpes, 2.5):.2f}, {np.percentile(mc_sharpes, 97.5):.2f}]")
print(f"    % Positive:       {(mc_sharpes > 0).sum() / len(mc_sharpes) * 100:.1f}%")
print(f"    % > 1.5:          {(mc_sharpes > 1.5).sum() / len(mc_sharpes) * 100:.1f}%")
print(f"    % > 2.0:          {(mc_sharpes > 2.0).sum() / len(mc_sharpes) * 100:.1f}%")


# ================================================================
# 5. PERMUTATION TESTING
# ================================================================

print("\n[5/7] PERMUTATION TESTING (shuffled target)")
print("-" * 80)

perm_sharpes = []

print("\nRunning 20 permutation tests with shuffled targets...")

for i in range(20):
    # Shuffle target
    y_train_shuffled = np.random.permutation(y_train)
    y_test_shuffled = np.random.permutation(y_test)

    sharpe, corr, _ = train_and_evaluate(X_train, y_train_shuffled, X_test, y_test_shuffled, test_prices)

    perm_sharpes.append(sharpe)

    if (i + 1) % 5 == 0:
        print(f"  Completed {i+1}/20 runs...")

perm_sharpes = np.array(perm_sharpes)

print(f"\n  PERMUTATION TEST SUMMARY:")
print(f"    Mean Sharpe (random):  {perm_sharpes.mean():.2f}")
print(f"    Max Sharpe (random):   {perm_sharpes.max():.2f}")
print(f"    Original Sharpe:       2.63")
print(f"    P-value estimate:      {(perm_sharpes >= 2.63).sum() / len(perm_sharpes):.4f}")

if perm_sharpes.max() < 1.0:
    print(f"    [OK] Random targets give much worse Sharpe - model is learning real patterns!")


# ================================================================
# 6. STABILITY ANALYSIS
# ================================================================

print("\n[6/7] STABILITY ANALYSIS (noise resistance)")
print("-" * 80)

# Add noise to test features and see how predictions change
torch.manual_seed(42)

# Load trained model
checkpoint = torch.load('models/neural_network_trading.pth', weights_only=False)
model = DeepFeatureNetwork(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler = joblib.load('models/neural_network_scaler.pkl')

# Original predictions
X_test_scaled = scaler.transform(X_test)
X_test_t = torch.FloatTensor(X_test_scaled)

with torch.no_grad():
    original_pred = model(X_test_t).numpy()

# Add noise at different levels
noise_levels = [0.01, 0.05, 0.1, 0.2]
stability_results = []

print("\nTesting prediction stability with added noise...")

for noise_level in noise_levels:
    # Add Gaussian noise
    X_test_noisy = X_test + np.random.normal(0, noise_level * X_test.std(axis=0), X_test.shape)
    X_test_noisy_scaled = scaler.transform(X_test_noisy)
    X_test_noisy_t = torch.FloatTensor(X_test_noisy_scaled)

    with torch.no_grad():
        noisy_pred = model(X_test_noisy_t).numpy()

    # Calculate prediction change
    pred_diff = np.abs(noisy_pred - original_pred).mean()
    pred_corr, _ = spearmanr(original_pred, noisy_pred)

    # Backtest with noisy predictions
    backtester = Backtester(initial_capital=10000, transaction_cost=0.001,
                           holding_period=5, position_size_pct=0.5, prediction_threshold=0.001)
    results = backtester.run_backtest(noisy_pred, y_test, test_prices)

    if len(results['returns']) > 0:
        sharpe_noisy = calculate_sharpe(results['returns'])
    else:
        sharpe_noisy = 0

    print(f"  Noise {noise_level*100:4.0f}%: Pred diff={pred_diff:.6f}, Pred corr={pred_corr:.4f}, Sharpe={sharpe_noisy:.2f}")

    stability_results.append({
        'noise_level': noise_level,
        'pred_diff': pred_diff,
        'pred_correlation': pred_corr,
        'sharpe': sharpe_noisy,
        'sharpe_change': sharpe_noisy - 2.63
    })

stability_df = pd.DataFrame(stability_results)


# ================================================================
# 7. TRAIN/VAL/TEST COMPARISON
# ================================================================

print("\n[7/7] TRAIN/VAL/TEST METRICS COMPARISON")
print("-" * 80)

# Load original metrics
with open('results/neural_network/metrics.json', 'r') as f:
    original_metrics = json.load(f)

print("\n  TRAINING PERFORMANCE:")
print(f"    Best Val Correlation: {original_metrics['training']['best_val_correlation']:.4f}")
print(f"    Epochs trained:       {original_metrics['training']['epochs']}")

print("\n  TEST PERFORMANCE:")
print(f"    Test Correlation:     {original_metrics['test_performance']['correlation']:.4f}")
print(f"    Sharpe Ratio:         {original_metrics['test_performance']['sharpe_ratio']:.2f}")
print(f"    Annual Return:        {original_metrics['test_performance']['annual_return']*100:.2f}%")
print(f"    Win Rate:             {original_metrics['test_performance']['win_rate']*100:.1f}%")


# ================================================================
# 8. FINAL VERDICT
# ================================================================

print("\n" + "="*80)
print("FINAL VERDICT - JE MODEL OVERFITTED?")
print("="*80)

verdicts = []

# 1. Walk-forward test
wf_mean = wf_df['sharpe'].mean()
if wf_mean > 1.0:
    verdicts.append(("[OK]", f"Walk-Forward Sharpe {wf_mean:.2f} > 1.0"))
else:
    verdicts.append(("[WARNING]", f"Walk-Forward Sharpe {wf_mean:.2f} < 1.0"))

# 2. Cross-validation
cv_mean = cv_df['sharpe'].mean()
if cv_mean > 1.0:
    verdicts.append(("[OK]", f"Cross-Val Sharpe {cv_mean:.2f} > 1.0"))
else:
    verdicts.append(("[WARNING]", f"Cross-Val Sharpe {cv_mean:.2f} < 1.0"))

# 3. Monte Carlo
mc_mean = mc_sharpes.mean()
mc_positive_pct = (mc_sharpes > 0).sum() / len(mc_sharpes) * 100
if mc_positive_pct > 80:
    verdicts.append(("[OK]", f"Monte Carlo: {mc_positive_pct:.0f}% runs are positive"))
else:
    verdicts.append(("[WARNING]", f"Monte Carlo: Only {mc_positive_pct:.0f}% runs are positive"))

# 4. Permutation test
perm_max = perm_sharpes.max()
if perm_max < 1.5:
    verdicts.append(("[OK]", f"Permutation: Random target max Sharpe {perm_max:.2f} << 2.63"))
else:
    verdicts.append(("[FAIL]", f"Permutation: Random target achieved Sharpe {perm_max:.2f}!"))

# 5. Stability
stable_pred_corr = stability_df[stability_df['noise_level'] == 0.05]['pred_correlation'].values[0]
if stable_pred_corr > 0.95:
    verdicts.append(("[OK]", f"Predictions stable with 5% noise (corr={stable_pred_corr:.3f})"))
else:
    verdicts.append(("[WARNING]", f"Predictions unstable with noise (corr={stable_pred_corr:.3f})"))

# Print verdicts
print()
for status, message in verdicts:
    print(f"  {status:10s} {message}")

# Overall conclusion
ok_count = sum(1 for s, _ in verdicts if s == "[OK]")
warning_count = sum(1 for s, _ in verdicts if s == "[WARNING]")
fail_count = sum(1 for s, _ in verdicts if s == "[FAIL]")

print("\n" + "="*80)
if fail_count > 0:
    print("CONCLUSION: MODEL IS LIKELY OVERFITTED!")
    print(f"  Failed {fail_count} critical tests")
elif warning_count > 2:
    print("CONCLUSION: MODEL MAY BE PARTIALLY OVERFITTED")
    print(f"  Passed {ok_count} tests, but {warning_count} warnings")
else:
    print("CONCLUSION: MODEL IS ROBUST AND NOT OVERFITTED!")
    print(f"  Passed {ok_count}/{len(verdicts)} validation tests")
    print(f"  The Sharpe 2.63 is RELIABLE and GENERALIZABLE!")

print("="*80)


# ================================================================
# 9. SAVE RESULTS
# ================================================================

print("\n[SAVING] Saving validation results...")

save_dir = Path('results/neural_network_validation')
save_dir.mkdir(parents=True, exist_ok=True)

# Save all results
wf_df.to_csv(save_dir / 'walk_forward_results.csv', index=False)
cv_df.to_csv(save_dir / 'cross_validation_results.csv', index=False)
oos_df.to_csv(save_dir / 'out_of_sample_results.csv', index=False)
stability_df.to_csv(save_dir / 'stability_results.csv', index=False)

# Save Monte Carlo results
pd.DataFrame({
    'run': range(1, len(mc_sharpes) + 1),
    'sharpe': mc_sharpes,
    'correlation': mc_correlations
}).to_csv(save_dir / 'monte_carlo_results.csv', index=False)

# Save permutation results
pd.DataFrame({
    'run': range(1, len(perm_sharpes) + 1),
    'sharpe': perm_sharpes
}).to_csv(save_dir / 'permutation_results.csv', index=False)

# Save summary
summary = {
    'validation_date': datetime.now().isoformat(),
    'original_sharpe': 2.63,
    'walk_forward': {
        'mean_sharpe': float(wf_mean),
        'std_sharpe': float(wf_df['sharpe'].std()),
        'n_folds': len(wf_df)
    },
    'cross_validation': {
        'mean_sharpe': float(cv_mean),
        'std_sharpe': float(cv_df['sharpe'].std()),
        'n_folds': len(cv_df)
    },
    'monte_carlo': {
        'mean_sharpe': float(mc_mean),
        'median_sharpe': float(np.median(mc_sharpes)),
        'std_sharpe': float(mc_sharpes.std()),
        'pct_positive': float(mc_positive_pct),
        'pct_above_1_5': float((mc_sharpes > 1.5).sum() / len(mc_sharpes) * 100),
        'n_runs': len(mc_sharpes)
    },
    'permutation': {
        'mean_sharpe': float(perm_sharpes.mean()),
        'max_sharpe': float(perm_max),
        'n_runs': len(perm_sharpes)
    },
    'verdicts': [{'status': s, 'message': m} for s, m in verdicts],
    'conclusion': {
        'ok_count': ok_count,
        'warning_count': warning_count,
        'fail_count': fail_count,
        'is_overfitted': fail_count > 0 or warning_count > 2
    }
}

with open(save_dir / 'validation_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Monte Carlo distribution
axes[0, 0].hist(mc_sharpes, bins=30, alpha=0.7, edgecolor='black')
axes[0, 0].axvline(2.63, color='red', linestyle='--', linewidth=2, label='Original (2.63)')
axes[0, 0].axvline(mc_sharpes.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean ({mc_sharpes.mean():.2f})')
axes[0, 0].set_xlabel('Sharpe Ratio')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Monte Carlo Simulation (100 runs)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Walk-forward over time
axes[0, 1].plot(range(1, len(wf_df) + 1), wf_df['sharpe'], 'o-', linewidth=2, markersize=8)
axes[0, 1].axhline(wf_df['sharpe'].mean(), color='green', linestyle='--', label=f'Mean ({wf_mean:.2f})')
axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[0, 1].set_xlabel('Fold')
axes[0, 1].set_ylabel('Sharpe Ratio')
axes[0, 1].set_title('Walk-Forward Validation')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Stability vs noise
axes[1, 0].plot(stability_df['noise_level'] * 100, stability_df['sharpe'], 'o-', linewidth=2, markersize=8)
axes[1, 0].axhline(2.63, color='red', linestyle='--', label='Original (2.63)')
axes[1, 0].set_xlabel('Noise Level (%)')
axes[1, 0].set_ylabel('Sharpe Ratio')
axes[1, 0].set_title('Stability Analysis')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Permutation comparison
axes[1, 1].boxplot([mc_sharpes, perm_sharpes], labels=['Real', 'Random'])
axes[1, 1].axhline(2.63, color='red', linestyle='--', linewidth=2, label='Original')
axes[1, 1].set_ylabel('Sharpe Ratio')
axes[1, 1].set_title('Real vs Random Target')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(save_dir / 'validation_analysis.png', dpi=150)
plt.close()

print(f"\n  [OK] Results saved to:")
print(f"    - {save_dir}/")

print("\n" + "="*80)
print("VALIDATION COMPLETE!")
print("="*80)
