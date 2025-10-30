"""
Model Comparison Module

Compare performance of multiple trading models:
- Ridge Regression (baseline)
- XGBoost (advanced)
- Future models...

Generates:
- Comparison tables
- Side-by-side equity curves
- Metric comparison charts
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class ModelComparator:
    """
    Compare performance of multiple trading models.

    Attributes:
        models (dict): Dictionary of model objects
        results (dict): Dictionary of backtest results and metrics

    Example:
        >>> comparator = ModelComparator()
        >>> comparator.add_model('Ridge', ridge_model, ridge_results, ridge_metrics)
        >>> comparator.add_model('XGBoost', xgb_model, xgb_results, xgb_metrics)
        >>> comparison_table = comparator.create_comparison_table()
        >>> print(comparison_table)
    """

    def __init__(self):
        """Initialize model comparator."""
        self.models = {}
        self.results = {}

        logger.info("Initialized ModelComparator")

    def add_model(self, name, model, backtest_results, metrics):
        """
        Add model results for comparison.

        Args:
            name: Model name (e.g., "Ridge", "XGBoost")
            model: Trained model object
            backtest_results: Results from backtester (dict)
            metrics: Performance metrics dict

        Example:
            >>> comparator.add_model('Ridge', ridge_model,
            ...                      ridge_backtest_results, ridge_metrics)
        """
        self.models[name] = model
        self.results[name] = {
            'backtest': backtest_results,
            'metrics': metrics
        }

        logger.info(f"Added model: {name}")

    def create_comparison_table(self):
        """
        Create comparison table of all models.

        Returns:
            DataFrame with key metrics for each model

        Columns:
            - Model: Model name
            - Sharpe: Sharpe Ratio
            - Return: Total return
            - Ann. Return: Annual return (CAGR)
            - Max DD: Maximum drawdown
            - Win Rate: Win rate
            - Profit Factor: Profit factor
            - Trades: Number of trades

        Example:
            >>> table = comparator.create_comparison_table()
            >>> print(table.to_string(index=False))
        """
        if not self.results:
            raise ValueError("No models added yet. Use add_model() first!")

        comparison = []

        for name, result in self.results.items():
            metrics = result['metrics']
            comparison.append({
                'Model': name,
                'Sharpe': metrics['sharpe_ratio'],
                'Return': metrics['total_return'],
                'Ann. Return': metrics['annual_return'],
                'Max DD': metrics['max_drawdown'],
                'Win Rate': metrics['win_rate'],
                'Profit Factor': metrics['profit_factor'],
                'Trades': metrics['total_trades']
            })

        df = pd.DataFrame(comparison)

        # Sort by Sharpe ratio (best first)
        df = df.sort_values('Sharpe', ascending=False)

        return df

    def plot_equity_curves(self, save_path=None):
        """
        Plot equity curves of all models on same chart.

        Args:
            save_path: Path to save plot (optional)

        Returns:
            matplotlib Figure

        Example:
            >>> fig = comparator.plot_equity_curves()
            >>> plt.show()
        """
        logger.info("Generating equity curves comparison...")

        fig, ax = plt.subplots(figsize=(14, 7))

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

        for idx, (name, result) in enumerate(self.results.items()):
            equity = result['backtest']['equity_curve']
            color = colors[idx % len(colors)]

            ax.plot(equity.index, equity.values, label=name,
                    linewidth=2.5, color=color, alpha=0.9)

        # Add initial capital line
        initial_capital = list(self.results.values())[0]['backtest']['initial_capital']
        ax.axhline(y=initial_capital, color='gray', linestyle='--',
                   alpha=0.5, linewidth=1, label='Initial Capital')

        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.set_title('Model Comparison - Equity Curves',
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11, framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Add performance annotations
        textstr = ""
        for name, result in self.results.items():
            metrics = result['metrics']
            final_return = metrics['total_return'] * 100
            textstr += f"{name}: {final_return:+.2f}%\n"

        ax.text(0.02, 0.98, textstr.strip(), transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  Saved to {save_path}")

        return fig

    def plot_metrics_comparison(self, save_path=None):
        """
        Bar chart comparing key metrics across models.

        Args:
            save_path: Path to save plot (optional)

        Returns:
            matplotlib Figure

        Example:
            >>> fig = comparator.plot_metrics_comparison()
            >>> plt.show()
        """
        logger.info("Generating metrics comparison...")

        df = self.create_comparison_table()

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        metrics_to_plot = [
            ('Sharpe', 'Sharpe Ratio', False),
            ('Ann. Return', 'Annual Return (%)', True),
            ('Max DD', 'Max Drawdown (%)', True),
            ('Win Rate', 'Win Rate (%)', True),
            ('Profit Factor', 'Profit Factor', False),
            ('Trades', 'Number of Trades', False)
        ]

        colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D']

        for idx, (col, title, as_pct) in enumerate(metrics_to_plot):
            ax = axes[idx]

            values = df[col].values
            if as_pct:
                values = values * 100  # Convert to percentage

            # Color bars by model
            bar_colors = [colors[i % len(colors)] for i in range(len(df))]

            ax.bar(df['Model'], values, color=bar_colors, alpha=0.8, edgecolor='black')

            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel(title, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')

            # Rotate x labels if needed
            if len(df) > 2:
                ax.tick_params(axis='x', rotation=45)

            # Add value labels on bars
            for i, v in enumerate(values):
                ax.text(i, v, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

        # Overall title
        fig.suptitle('Model Comparison - Key Performance Metrics',
                     fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"  Saved to {save_path}")

        return fig

    def print_comparison_summary(self):
        """
        Print detailed comparison summary to console.

        Example:
            >>> comparator.print_comparison_summary()
        """
        print("\n" + "=" * 70)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 70)

        df = self.create_comparison_table()
        print("\n" + df.to_string(index=False))

        # Calculate improvements (from worst to best)
        if len(self.results) >= 2:
            print("\n" + "-" * 70)
            print("IMPROVEMENTS (from baseline)")
            print("-" * 70)

            # Assume first model is baseline (worst Sharpe)
            baseline_name = df.iloc[-1]['Model']
            baseline_metrics = self.results[baseline_name]['metrics']

            for idx, row in df.iterrows():
                if row['Model'] == baseline_name:
                    continue

                model_metrics = self.results[row['Model']]['metrics']

                sharpe_improvement = (model_metrics['sharpe_ratio'] -
                                      baseline_metrics['sharpe_ratio'])
                return_improvement = (model_metrics['annual_return'] -
                                      baseline_metrics['annual_return'])

                print(f"\n{row['Model']} vs {baseline_name}:")
                print(f"  Sharpe: {baseline_metrics['sharpe_ratio']:.2f} → "
                      f"{model_metrics['sharpe_ratio']:.2f} "
                      f"({sharpe_improvement:+.2f})")
                print(f"  Ann. Return: {baseline_metrics['annual_return']*100:.2f}% → "
                      f"{model_metrics['annual_return']*100:.2f}% "
                      f"({return_improvement*100:+.2f}%)")
                print(f"  Max DD: {baseline_metrics['max_drawdown_pct']:.2f}% → "
                      f"{model_metrics['max_drawdown_pct']:.2f}%")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    # Example usage
    print("Testing ModelComparator...")

    # Create sample results
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=252, freq='D')

    # Model 1: Ridge (baseline)
    ridge_equity = pd.Series(10000 * (1 + np.random.randn(252).cumsum() * 0.005),
                             index=dates)
    ridge_results = {
        'equity_curve': ridge_equity,
        'initial_capital': 10000,
        'final_capital': ridge_equity.iloc[-1]
    }
    ridge_metrics = {
        'sharpe_ratio': 0.09,
        'total_return': 0.0027,
        'annual_return': 0.0019,
        'max_drawdown': -0.042,
        'max_drawdown_pct': -4.2,
        'win_rate': 0.50,
        'profit_factor': 1.09,
        'total_trades': 12
    }

    # Model 2: XGBoost (better)
    xgb_equity = pd.Series(10000 * (1 + np.random.randn(252).cumsum() * 0.008),
                           index=dates)
    xgb_results = {
        'equity_curve': xgb_equity,
        'initial_capital': 10000,
        'final_capital': xgb_equity.iloc[-1]
    }
    xgb_metrics = {
        'sharpe_ratio': 0.52,
        'total_return': 0.0687,
        'annual_return': 0.0512,
        'max_drawdown': -0.178,
        'max_drawdown_pct': -17.8,
        'win_rate': 0.54,
        'profit_factor': 1.68,
        'total_trades': 28
    }

    # Create comparator
    comparator = ModelComparator()
    comparator.add_model('Ridge (Baseline)', None, ridge_results, ridge_metrics)
    comparator.add_model('XGBoost', None, xgb_results, xgb_metrics)

    # Test comparison table
    print("\nComparison Table:")
    table = comparator.create_comparison_table()
    print(table.to_string(index=False))

    # Test summary
    comparator.print_comparison_summary()

    # Test plots
    try:
        fig1 = comparator.plot_equity_curves()
        plt.close()
        print("\n✓ Equity curves plot OK")

        fig2 = comparator.plot_metrics_comparison()
        plt.close()
        print("✓ Metrics comparison plot OK")
    except Exception as e:
        print(f"\n(Skipped plotting: {e})")

    print("\n✓ ModelComparator test complete!")
