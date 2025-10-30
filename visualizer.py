"""
Visualization Module - Trading Performance Charts

Generates comprehensive visualizations:
- Equity curves (strategy vs benchmark)
- Drawdown charts
- Returns distribution
- Monthly returns heatmap
- Feature importance
- Complete tearsheet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import logging
import sys
import codecs

# Fix Windows encoding
if sys.platform == 'win32' and hasattr(sys.stdout, 'buffer'):
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'


def plot_equity_curve(backtest_results, benchmark=None, save_path=None):
    """
    Plot equity curve with optional benchmark comparison.

    Args:
        backtest_results: Dict from Backtester.run_backtest()
        benchmark: Optional benchmark equity curve (pd.Series)
        save_path: Path to save plot (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_equity_curve(results, benchmark)
        >>> plt.show()
    """
    logger.info("Generating equity curve plot...")

    equity = backtest_results['equity_curve']

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot strategy
    ax.plot(equity.index, equity.values,
            label='Strategy', color='#2E86AB', linewidth=2)

    # Plot benchmark if provided
    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values,
                label='Buy & Hold', color='#A23B72', linewidth=2, alpha=0.7)

    # Add horizontal line at initial capital
    initial = backtest_results['initial_capital']
    ax.axhline(y=initial, color='gray', linestyle='--',
               alpha=0.5, label='Initial Capital')

    # Formatting
    ax.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add performance text
    final = backtest_results['final_capital']
    total_return = (final / initial - 1) * 100

    textstr = f'Total Return: {total_return:+.2f}%\nFinal Value: ${final:,.0f}'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")

    return fig


def plot_drawdown(backtest_results, save_path=None):
    """
    Plot underwater (drawdown) chart.

    Shows how far portfolio is from its peak at each point in time.

    Args:
        backtest_results: Dict from Backtester.run_backtest()
        save_path: Path to save plot (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_drawdown(results)
    """
    logger.info("Generating drawdown plot...")

    equity = backtest_results['equity_curve']

    # Calculate drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot drawdown
    ax.fill_between(drawdown.index, drawdown.values, 0,
                     color='#E63946', alpha=0.7, label='Drawdown')
    ax.plot(drawdown.index, drawdown.values, color='#E63946', linewidth=1)

    # Formatting
    ax.set_title('Underwater Plot (Drawdown)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)

    # Add max drawdown text
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()

    textstr = f'Max Drawdown: {max_dd:.2%}\nDate: {max_dd_date.strftime("%Y-%m-%d")}'
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes,
            fontsize=11, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")

    return fig


def plot_returns_distribution(backtest_results, save_path=None):
    """
    Plot returns distribution histogram.

    Args:
        backtest_results: Dict from Backtester.run_backtest()
        save_path: Path to save plot (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_returns_distribution(results)
    """
    logger.info("Generating returns distribution plot...")

    returns = backtest_results['returns']

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(returns, bins=50, color='#06A77D', alpha=0.7, edgecolor='black')

    # Add mean line
    mean_ret = returns.mean()
    ax.axvline(mean_ret, color='red', linestyle='--',
               linewidth=2, label=f'Mean: {mean_ret:.4f}')

    # Add zero line
    ax.axvline(0, color='gray', linestyle='-', alpha=0.5)

    # Formatting
    ax.set_title('Daily Returns Distribution', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    std_ret = returns.std()
    skew = returns.skew()
    kurt = returns.kurtosis()

    textstr = f'Mean: {mean_ret:.4f}\nStd: {std_ret:.4f}\nSkew: {skew:.2f}\nKurt: {kurt:.2f}'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")

    return fig


def plot_monthly_returns(backtest_results, save_path=None):
    """
    Plot monthly returns heatmap.

    Args:
        backtest_results: Dict from Backtester.run_backtest()
        save_path: Path to save plot (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_monthly_returns(results)
    """
    logger.info("Generating monthly returns heatmap...")

    equity = backtest_results['equity_curve']

    # Ensure index is DatetimeIndex and remove timezone
    if not isinstance(equity.index, pd.DatetimeIndex):
        equity.index = pd.to_datetime(equity.index)
    if hasattr(equity.index, 'tz') and equity.index.tz is not None:
        equity.index = equity.index.tz_localize(None)

    # Calculate monthly returns
    monthly_equity = equity.resample('ME').last()
    monthly_returns = monthly_equity.pct_change().dropna()

    # Create pivot table (Year x Month)
    monthly_returns_df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })

    pivot = monthly_returns_df.pivot(index='Year', columns='Month', values='Return')

    # Month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[int(m)-1] for m in pivot.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Heatmap
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0,
                cbar_kws={'label': 'Monthly Return'},
                linewidths=0.5, ax=ax)

    # Formatting
    ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Year', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")

    return fig


def plot_feature_importance(model, top_n=10, save_path=None):
    """
    Plot feature importance from trained model.

    Args:
        model: Trained BaselineModel instance
        top_n: Number of top features to show
        save_path: Path to save plot (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_feature_importance(model, top_n=10)
    """
    logger.info("Generating feature importance plot...")

    # Get feature importance
    importance = model.get_feature_importance(top_n=top_n)
    coefficients = model.get_coefficients()

    # Get colors based on sign
    colors = ['#06A77D' if coefficients[feat] > 0 else '#E63946'
              for feat in importance.index]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Horizontal bar chart
    y_pos = np.arange(len(importance))
    ax.barh(y_pos, importance.values, color=colors, alpha=0.8, edgecolor='black')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance.index)
    ax.invert_yaxis()
    ax.set_xlabel('Absolute Coefficient (Importance)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#06A77D', label='Positive'),
                       Patch(facecolor='#E63946', label='Negative')]
    ax.legend(handles=legend_elements, loc='best', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved to {save_path}")

    return fig


def generate_tearsheet(backtest_results, metrics, model, benchmark=None, save_path=None):
    """
    Generate comprehensive performance tearsheet.

    Creates multi-panel figure with:
    - Equity curve
    - Drawdown
    - Returns distribution
    - Monthly returns
    - Feature importance
    - Key metrics table

    Args:
        backtest_results: Dict from Backtester.run_backtest()
        metrics: Dict from generate_metrics_report()
        model: Trained BaselineModel instance
        benchmark: Optional benchmark equity curve
        save_path: Path to save tearsheet (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = generate_tearsheet(results, metrics, model, benchmark)
        >>> plt.show()
    """
    logger.info("Generating performance tearsheet...")

    # Create figure with GridSpec
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # === 1. Equity Curve ===
    ax1 = fig.add_subplot(gs[0, :])
    equity = backtest_results['equity_curve']
    initial = backtest_results['initial_capital']

    ax1.plot(equity.index, equity.values,
             label='Strategy', color='#2E86AB', linewidth=2)

    if benchmark is not None:
        ax1.plot(benchmark.index, benchmark.values,
                 label='Buy & Hold', color='#A23B72', linewidth=2, alpha=0.7)

    ax1.axhline(y=initial, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Portfolio Equity Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # === 2. Drawdown ===
    ax2 = fig.add_subplot(gs[1, 0])
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max

    ax2.fill_between(drawdown.index, drawdown.values, 0,
                      color='#E63946', alpha=0.7)
    ax2.plot(drawdown.index, drawdown.values, color='#E63946', linewidth=1)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, alpha=0.3)

    # === 3. Returns Distribution ===
    ax3 = fig.add_subplot(gs[1, 1])
    returns = backtest_results['returns']

    ax3.hist(returns, bins=40, color='#06A77D', alpha=0.7, edgecolor='black')
    ax3.axvline(returns.mean(), color='red', linestyle='--', linewidth=2)
    ax3.axvline(0, color='gray', linestyle='-', alpha=0.5)
    ax3.set_title('Returns Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Daily Return')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3, axis='y')

    # === 4. Feature Importance ===
    ax4 = fig.add_subplot(gs[2, 0])

    # Get feature importance (works for both Ridge and XGBoost)
    if hasattr(model, 'get_feature_importance'):
        importance_df = model.get_feature_importance(top_n=8)
        if isinstance(importance_df, pd.DataFrame):
            importance = importance_df['importance']
            features = importance_df['feature']
        else:
            importance = importance_df
            features = importance.index
    else:
        # Fallback for other models
        importance = pd.Series([0.1] * 8, index=['feature_' + str(i) for i in range(8)])
        features = importance.index

    # Use single color for all features
    y_pos = np.arange(len(importance))
    ax4.barh(y_pos, importance.values, color='#2E86AB', alpha=0.8, edgecolor='black')
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(features, fontsize=9)
    ax4.invert_yaxis()
    ax4.set_xlabel('Importance')
    ax4.set_title('Feature Importance (Top 8)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')

    # === 5. Metrics Table ===
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Create metrics text
    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*40}

    Total Return:        {metrics['total_return']*100:>8.2f}%
    Annual Return:       {metrics['annual_return']*100:>8.2f}%

    Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}
    Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}

    Max Drawdown:        {metrics['max_drawdown_pct']:>8.2f}%
    Return/DD Ratio:     {metrics['return_dd_ratio']:>8.2f}

    Total Trades:        {metrics['total_trades']:>8d}
    Win Rate:            {metrics['win_rate']*100:>8.1f}%
    Profit Factor:       {metrics['profit_factor']:>8.2f}

    Mean Daily Return:   {metrics['mean_return']*100:>8.4f}%
    Daily Volatility:    {metrics['std_return']*100:>8.4f}%
    """

    ax5.text(0.1, 0.95, metrics_text, transform=ax5.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Overall title
    fig.suptitle('Trading Strategy Performance Tearsheet',
                 fontsize=18, fontweight='bold', y=0.98)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Tearsheet saved to {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Testing visualizer module...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Simulate equity curve
    returns = np.random.randn(252) * 0.01 + 0.0005
    equity = pd.Series(10000 * (1 + returns).cumprod(), index=dates)

    # Create sample results
    backtest_results = {
        'equity_curve': equity,
        'returns': pd.Series(returns, index=dates[1:]),
        'initial_capital': 10000,
        'final_capital': equity.iloc[-1],
        'trades': pd.DataFrame({'pnl': np.random.randn(20) * 100}),
        'n_trades': 20,
        'total_costs': 50.0
    }

    # Sample metrics
    from metrics import generate_metrics_report
    metrics = generate_metrics_report(backtest_results)

    # Sample model
    from baseline_model import BaselineModel
    model = BaselineModel()
    X_dummy = pd.DataFrame({
        'return_1d': np.random.randn(100),
        'return_5d': np.random.randn(100),
        'volatility_20d': np.random.randn(100)
    })
    y_dummy = np.random.randn(100)
    model.train(X_dummy, y_dummy)

    # Test individual plots
    print("\n1. Testing equity curve...")
    fig1 = plot_equity_curve(backtest_results)
    plt.close()

    print("2. Testing drawdown...")
    fig2 = plot_drawdown(backtest_results)
    plt.close()

    print("3. Testing returns distribution...")
    fig3 = plot_returns_distribution(backtest_results)
    plt.close()

    print("4. Testing monthly returns...")
    fig4 = plot_monthly_returns(backtest_results)
    plt.close()

    print("5. Testing feature importance...")
    fig5 = plot_feature_importance(model, top_n=3)
    plt.close()

    print("6. Testing complete tearsheet...")
    fig6 = generate_tearsheet(backtest_results, metrics, model)
    plt.close()

    print("\nvisualization tests complete!")
