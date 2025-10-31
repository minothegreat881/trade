"""
NEURAL NETWORK TRADING MODEL
============================

Deep Feature Network implementation pre trading
- Input: 117 features (same as XGBoost)
- Architecture: Deep network with attention mechanism
- Custom Sharpe loss function
- Target: Beat baseline Sharpe 1.34

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================================================
# 1. CUSTOM LOSS FUNCTIONS
# ================================================================

class SharpeLoss(nn.Module):
    """
    Custom loss function optimizing Sharpe ratio directly
    """
    def __init__(self, epsilon=1e-6):
        super(SharpeLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predictions, targets):
        """
        Calculate negative Sharpe ratio as loss
        """
        # Calculate strategy returns
        # Assume we go long when prediction > 0.001
        positions = (predictions > 0.001).float()

        # Calculate returns when we have positions
        strategy_returns = positions * targets

        # Calculate Sharpe
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std() + self.epsilon

        # Negative Sharpe (we minimize loss)
        sharpe = -(mean_return / std_return) * torch.sqrt(torch.tensor(252.0))

        # Add MSE component for stability
        mse = nn.MSELoss()(predictions, targets)

        # Combined loss
        return sharpe + 0.1 * mse


class RankCorrelationLoss(nn.Module):
    """
    Loss based on Spearman rank correlation
    """
    def __init__(self):
        super(RankCorrelationLoss, self).__init__()

    def forward(self, predictions, targets):
        """
        Negative Spearman correlation as loss
        """
        # Get ranks
        pred_ranks = predictions.argsort().argsort().float()
        target_ranks = targets.argsort().argsort().float()

        # Calculate correlation
        n = predictions.size(0)
        mean_pred = pred_ranks.mean()
        mean_target = target_ranks.mean()

        cov = ((pred_ranks - mean_pred) * (target_ranks - mean_target)).mean()
        std_pred = pred_ranks.std()
        std_target = target_ranks.std()

        correlation = cov / (std_pred * std_target + 1e-6)

        # Return negative correlation (we minimize loss)
        return -correlation


# ================================================================
# 2. NEURAL NETWORK ARCHITECTURE
# ================================================================

class AttentionBlock(nn.Module):
    """
    Self-attention mechanism for feature interactions
    """
    def __init__(self, dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Reshape for attention if needed
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        attn_out, _ = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))

        # Remove sequence dimension if added
        if x.shape[1] == 1:
            x = x.squeeze(1)

        return x


class DeepFeatureNetwork(nn.Module):
    """
    Deep neural network for trading predictions

    Architecture:
    - Input layer: 117 features
    - Feature extraction: 3 dense layers with BatchNorm and Dropout
    - Attention mechanism for feature interactions
    - Output layer: Single prediction
    """
    def __init__(self, input_dim=117, hidden_dims=[256, 128, 64], dropout_rate=0.3):
        super(DeepFeatureNetwork, self).__init__()

        self.input_dim = input_dim

        # Feature extraction layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Attention block
        self.attention = AttentionBlock(hidden_dims[-1])

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier initialization for better convergence"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)

        # Apply attention
        features = self.attention(features)

        # Output
        output = self.output_layers(features)

        return output.squeeze()


# ================================================================
# 3. TRAINING UTILITIES
# ================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=1e-5, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            total_loss += loss.item()
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())

    # Calculate correlation
    from scipy.stats import spearmanr
    correlation, _ = spearmanr(all_predictions, all_targets)

    return total_loss / len(val_loader), correlation


# ================================================================
# 4. MAIN TRAINING FUNCTION
# ================================================================

def train_neural_network():
    """
    Main function to train neural network on trading data
    """
    print("="*80)
    print("NEURAL NETWORK TRADING MODEL")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ----------------------------------------------------------------
    # Load data
    # ----------------------------------------------------------------
    print("\n[1/7] Loading data...")

    df = pd.read_csv('data/full_dataset_2020_2025.csv', index_col=0, parse_dates=True)

    # Exclude columns
    exclude_cols = ['target', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'Dividends', 'Stock Splits', 'Capital Gains',
                    'fear_greed_value', 'fear_greed_classification',
                    'BTC_Close', 'BTC_Volume', 'BTC_return_1d', 'VIX']

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(feature_cols)}")

    # Clean data
    df = df.dropna(subset=feature_cols + ['target'])
    print(f"  After cleaning: {len(df)} samples")

    # ----------------------------------------------------------------
    # Split data
    # ----------------------------------------------------------------
    print("\n[2/7] Splitting data (70/20/10)...")

    train_size = int(len(df) * 0.7)
    val_size = int(len(df) * 0.2)

    train_data = df.iloc[:train_size]
    val_data = df.iloc[train_size:train_size+val_size]
    test_data = df.iloc[train_size+val_size:]

    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples")

    # ----------------------------------------------------------------
    # Prepare features
    # ----------------------------------------------------------------
    print("\n[3/7] Preparing features...")

    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values

    X_val = val_data[feature_cols].values
    y_val = val_data['target'].values

    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    test_close = test_data['Close'].values

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, 'models/neural_network_scaler.pkl')

    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)

    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)

    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # ----------------------------------------------------------------
    # Initialize model
    # ----------------------------------------------------------------
    print("\n[4/7] Initializing model...")

    model = DeepFeatureNetwork(
        input_dim=len(feature_cols),
        hidden_dims=[256, 128, 64],
        dropout_rate=0.3
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = SharpeLoss()  # Custom Sharpe loss
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    # Early stopping
    early_stopping = EarlyStopping(patience=15, verbose=True)

    # ----------------------------------------------------------------
    # Training loop
    # ----------------------------------------------------------------
    print("\n[5/7] Training model...")

    epochs = 100
    train_losses = []
    val_losses = []
    correlations = []

    best_correlation = -1
    best_model_state = None

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, correlation = validate(model, val_loader, criterion, device)

        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        correlations.append(correlation)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if correlation > best_correlation:
            best_correlation = correlation
            best_model_state = model.state_dict().copy()
            print(f"  [NEW BEST] Epoch {epoch+1}: Correlation = {correlation:.4f}")

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
            print(f"    Train Loss: {train_loss:.6f}")
            print(f"    Val Loss:   {val_loss:.6f}")
            print(f"    Correlation: {correlation:.4f}")
            print(f"    LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping
        if early_stopping(val_loss):
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(best_model_state)

    # ----------------------------------------------------------------
    # Test evaluation
    # ----------------------------------------------------------------
    print("\n[6/7] Evaluating on test set...")

    model.eval()
    test_predictions = []

    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            predictions = model(batch_X)
            test_predictions.extend(predictions.cpu().numpy())

    test_predictions = np.array(test_predictions)

    # Calculate test correlation
    from scipy.stats import spearmanr
    test_correlation, _ = spearmanr(y_test.numpy(), test_predictions)

    print(f"  Test Correlation: {test_correlation:.4f}")
    print(f"  Prediction stats:")
    print(f"    Mean: {test_predictions.mean():.6f}")
    print(f"    Std:  {test_predictions.std():.6f}")
    print(f"    Min:  {test_predictions.min():.6f}")
    print(f"    Max:  {test_predictions.max():.6f}")

    # ----------------------------------------------------------------
    # Backtesting
    # ----------------------------------------------------------------
    print("\n[7/7] Running backtest...")

    from backtester import Backtester

    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.001,
        holding_period=5,
        position_size_pct=0.5,
        prediction_threshold=0.001
    )

    # Run backtest
    results = backtester.run_backtest(test_predictions, y_test.numpy(), test_close)

    # Calculate metrics
    returns = results['returns']
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 0 and returns.std() > 0 else 0

    # Max drawdown
    equity = results['equity_curve']
    rolling_max = equity.expanding().max()
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = drawdowns.min()

    # Win rate
    trades_df = results['trades']
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df) if len(trades_df) > 0 else 0

    # Annual return
    total_return = (results['final_capital'] / results['initial_capital']) - 1
    days_in_test = (test_data.index[-1] - test_data.index[0]).days
    annual_return = (1 + total_return) ** (365 / days_in_test) - 1 if days_in_test > 0 else 0

    print(f"\n  NEURAL NETWORK RESULTS:")
    print(f"    Sharpe Ratio:     {sharpe:.2f}")
    print(f"    Annual Return:    {annual_return*100:.2f}%")
    print(f"    Total Return:     {total_return*100:.2f}%")
    print(f"    Max Drawdown:     {max_drawdown*100:.2f}%")
    print(f"    Win Rate:         {win_rate*100:.1f}%")
    print(f"    Total Trades:     {results['n_trades']}")
    print(f"    Final Capital:    ${results['final_capital']:.2f}")

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    print("\n[SAVING] Saving model and results...")

    save_dir = Path('results/neural_network')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': len(feature_cols),
            'hidden_dims': [256, 128, 64],
            'dropout_rate': 0.3
        },
        'feature_cols': feature_cols,
        'best_correlation': best_correlation,
        'epochs_trained': epoch + 1
    }, 'models/neural_network_trading.pth')

    # Save metrics
    metrics = {
        'created_at': datetime.now().isoformat(),
        'model_type': 'DeepFeatureNetwork',
        'training': {
            'epochs': epoch + 1,
            'best_val_correlation': float(best_correlation),
            'final_train_loss': float(train_losses[-1]),
            'final_val_loss': float(val_losses[-1])
        },
        'test_performance': {
            'correlation': float(test_correlation),
            'sharpe_ratio': float(sharpe),
            'annual_return': float(annual_return),
            'total_return': float(total_return),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'total_trades': int(results['n_trades']),
            'final_capital': float(results['final_capital'])
        },
        'data': {
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'features': len(feature_cols)
        }
    }

    with open(save_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        'date': test_data.index,
        'actual': y_test.numpy(),
        'predicted': test_predictions
    })
    pred_df.to_csv(save_dir / 'predictions.csv', index=False)

    # Save trades
    trades_df.to_csv(save_dir / 'trades.csv', index=False)

    # Save training history
    history_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_correlation': correlations
    })
    history_df.to_csv(save_dir / 'training_history.csv', index=False)

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot
    axes[0].plot(history_df['epoch'], history_df['train_loss'], label='Train Loss')
    axes[0].plot(history_df['epoch'], history_df['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training History - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Correlation plot
    axes[1].plot(history_df['epoch'], history_df['val_correlation'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Correlation')
    axes[1].set_title('Validation Correlation')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=100)
    plt.close()

    print(f"\n  [OK] Model saved to:")
    print(f"    - models/neural_network_trading.pth")
    print(f"    - models/neural_network_scaler.pkl")
    print(f"    - results/neural_network/")

    # ----------------------------------------------------------------
    # Compare with XGBoost baseline
    # ----------------------------------------------------------------
    print("\n" + "="*80)
    print("POROVNANIE S XGBOOST BASELINE")
    print("="*80)

    baseline_sharpe = 1.34  # From previous results

    print(f"\n                    XGBOOST    NEURAL NET    ROZDIEL")
    print(f"Sharpe Ratio:        {baseline_sharpe:7.2f}      {sharpe:7.2f}      {sharpe-baseline_sharpe:+6.2f}")
    print(f"Correlation:         0.0800       {test_correlation:7.4f}      {test_correlation-0.08:+7.4f}")

    improvement = ((sharpe / baseline_sharpe) - 1) * 100

    if sharpe > baseline_sharpe:
        print(f"\n  [SUCCESS] Neural Network prekonala XGBoost o {improvement:.1f}%!")
    elif sharpe > baseline_sharpe * 0.9:
        print(f"\n  [OK] Podobny vykon ako XGBoost ({improvement:+.1f}%)")
    else:
        print(f"\n  [INFO] XGBoost stale lepsi. NN dosiahla {sharpe/baseline_sharpe*100:.1f}% vykonu XGBoost")

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)

    return model, metrics


# ================================================================
# 5. PREDICTION FUNCTION
# ================================================================

def predict_with_neural_network(model_path='models/neural_network_trading.pth',
                                scaler_path='models/neural_network_scaler.pkl',
                                data_path='data/full_dataset_2020_2025.csv'):
    """
    Make predictions using trained neural network
    """
    # Load model
    checkpoint = torch.load(model_path)
    model_config = checkpoint['model_config']
    feature_cols = checkpoint['feature_cols']

    model = DeepFeatureNetwork(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load scaler
    scaler = joblib.load(scaler_path)

    # Load data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Get latest data
    latest_data = df[feature_cols].iloc[-1:].values
    latest_data_scaled = scaler.transform(latest_data)

    # Predict
    with torch.no_grad():
        X = torch.FloatTensor(latest_data_scaled)
        prediction = model(X).item()

    print(f"Latest prediction: {prediction:.6f}")
    print(f"Signal: {'BUY' if prediction > 0.001 else 'HOLD'}")

    return prediction


# ================================================================
# 6. MAIN EXECUTION
# ================================================================

if __name__ == "__main__":
    # Train model
    model, metrics = train_neural_network()

    # Print final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Model: DeepFeatureNetwork")
    print(f"Features: 117")
    print(f"Test Sharpe: {metrics['test_performance']['sharpe_ratio']:.2f}")
    print(f"Test Correlation: {metrics['test_performance']['correlation']:.4f}")
    print(f"Annual Return: {metrics['test_performance']['annual_return']*100:.2f}%")
    print(f"Win Rate: {metrics['test_performance']['win_rate']*100:.1f}%")
    print("="*80)