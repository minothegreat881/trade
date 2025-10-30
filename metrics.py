"""
Performance Metrics Module

Calculates trading performance metrics:
- Sharpe Ratio
- Maximum Drawdown
- Annual Return (CAGR)
- Win Rate
- Profit Factor
- Return/Drawdown Ratio
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate annualized Sharpe Ratio.

    Sharpe = (mean_return - risk_free_rate) / std_return * sqrt(periods_per_year)

    Args:
        returns: Series or array of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Trading days per year (default: 252)

    Returns:
        float: Annualized Sharpe Ratio

    Example:
        >>> sharpe = calculate_sharpe_ratio(daily_returns)
        >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    """
    if len(returns) == 0:
        return 0.0

    # Remove NaN
    returns = pd.Series(returns).dropna()

    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    # Calculate annualized metrics
    mean_return = returns.mean() * periods_per_year
    std_return = returns.std() * np.sqrt(periods_per_year)

    sharpe = (mean_return - risk_free_rate) / std_return

    return sharpe


def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown and related statistics.

    Drawdown = (Trough - Peak) / Peak

    Args:
        equity_curve: Series of portfolio values over time

    Returns:
        dict with:
            - max_drawdown: Maximum drawdown (as negative fraction)
            - max_drawdown_pct: Maximum drawdown as percentage
            - peak_date: Date of peak before max drawdown
            - trough_date: Date of trough (max drawdown)
            - recovery_date: Date of recovery (or None if not recovered)
            - drawdown_duration: Days in drawdown

    Example:
        >>> dd = calculate_max_drawdown(equity_curve)
        >>> print(f"Max Drawdown: {dd['max_drawdown_pct']:.2f}%")
    """
    if len(equity_curve) == 0:
        return {
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'peak_date': None,
            'trough_date': None,
            'recovery_date': None,
            'drawdown_duration': 0
        }

    equity = pd.Series(equity_curve)

    # Calculate running maximum
    running_max = equity.expanding().max()

    # Calculate drawdown
    drawdown = (equity - running_max) / running_max

    # Find maximum drawdown
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Find peak before max drawdown
    peak_idx = running_max[:max_dd_idx].idxmax()

    # Find recovery date (if any)
    recovery_idx = None
    after_trough = equity[max_dd_idx:]
    peak_value = running_max[max_dd_idx]
    recovered = after_trough >= peak_value

    if recovered.any():
        recovery_idx = recovered.idxmax()

    # Calculate duration
    if isinstance(max_dd_idx, pd.Timestamp) and isinstance(peak_idx, pd.Timestamp):
        duration = (max_dd_idx - peak_idx).days
    else:
        duration = max_dd_idx - peak_idx

    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'peak_date': peak_idx,
        'trough_date': max_dd_idx,
        'recovery_date': recovery_idx,
        'drawdown_duration': duration
    }


def calculate_annual_return(equity_curve, periods_per_year=252):
    """
    Calculate Compound Annual Growth Rate (CAGR).

    CAGR = (final_value / initial_value)^(periods_per_year / total_periods) - 1

    Args:
        equity_curve: Series of portfolio values
        periods_per_year: Trading days per year (default: 252)

    Returns:
        float: Annualized return

    Example:
        >>> annual_ret = calculate_annual_return(equity_curve)
        >>> print(f"Annual Return: {annual_ret*100:.2f}%")
    """
    if len(equity_curve) < 2:
        return 0.0

    equity = pd.Series(equity_curve)
    initial_value = equity.iloc[0]
    final_value = equity.iloc[-1]
    total_periods = len(equity)

    if initial_value == 0:
        return 0.0

    cagr = (final_value / initial_value) ** (periods_per_year / total_periods) - 1

    return cagr


def calculate_win_rate(trades):
    """
    Calculate win rate from list of trades.

    Win Rate = Number of profitable trades / Total trades

    Args:
        trades: DataFrame or list of dicts with 'pnl' column

    Returns:
        float: Win rate (0 to 1)

    Example:
        >>> win_rate = calculate_win_rate(trades_df)
        >>> print(f"Win Rate: {win_rate*100:.1f}%")
    """
    if len(trades) == 0:
        return 0.0

    if isinstance(trades, pd.DataFrame):
        winning_trades = (trades['pnl'] > 0).sum()
    else:
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)

    win_rate = winning_trades / len(trades)

    return win_rate


def calculate_profit_factor(trades):
    """
    Calculate profit factor.

    Profit Factor = Sum of winning trades / Abs(Sum of losing trades)

    Args:
        trades: DataFrame or list of dicts with 'pnl' column

    Returns:
        float: Profit factor (higher is better, > 1 is profitable)

    Example:
        >>> pf = calculate_profit_factor(trades_df)
        >>> print(f"Profit Factor: {pf:.2f}")
    """
    if len(trades) == 0:
        return 0.0

    if isinstance(trades, pd.DataFrame):
        gross_profit = trades[trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades[trades['pnl'] < 0]['pnl'].sum())
    else:
        gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))

    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0

    profit_factor = gross_profit / gross_loss

    return profit_factor


def calculate_return_drawdown_ratio(annual_return, max_drawdown):
    """
    Calculate Return/Drawdown ratio (Calmar-like ratio).

    Return/DD = Annual Return / Abs(Max Drawdown)

    From Yan (2025): XGBoost achieved 0.31

    Args:
        annual_return: Annualized return (fraction)
        max_drawdown: Maximum drawdown (negative fraction)

    Returns:
        float: Return/Drawdown ratio

    Example:
        >>> ratio = calculate_return_drawdown_ratio(0.15, -0.25)
        >>> print(f"Return/DD Ratio: {ratio:.2f}")
    """
    if max_drawdown == 0:
        return float('inf') if annual_return > 0 else 0.0

    ratio = annual_return / abs(max_drawdown)

    return ratio


def calculate_daily_returns(equity_curve):
    """
    Calculate daily returns from equity curve.

    Args:
        equity_curve: Series of portfolio values

    Returns:
        Series: Daily returns

    Example:
        >>> returns = calculate_daily_returns(equity_curve)
        >>> print(f"Mean daily return: {returns.mean():.4f}")
    """
    equity = pd.Series(equity_curve)
    returns = equity.pct_change().dropna()

    return returns


def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate Sortino Ratio (like Sharpe but only penalizes downside volatility).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading days per year

    Returns:
        float: Sortino ratio

    Example:
        >>> sortino = calculate_sortino_ratio(daily_returns)
        >>> print(f"Sortino Ratio: {sortino:.2f}")
    """
    if len(returns) == 0:
        return 0.0

    returns = pd.Series(returns).dropna()

    if len(returns) == 0:
        return 0.0

    # Calculate downside deviation
    downside_returns = returns[returns < 0]

    if len(downside_returns) == 0:
        downside_std = 0.0
    else:
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)

    if downside_std == 0:
        return 0.0

    mean_return = returns.mean() * periods_per_year

    sortino = (mean_return - risk_free_rate) / downside_std

    return sortino


def generate_metrics_report(backtest_results):
    """
    Generate comprehensive metrics report from backtest results.

    Args:
        backtest_results: Dict from Backtester.run_backtest()
            Expected keys: equity_curve, trades, returns

    Returns:
        dict: Complete metrics report

    Example:
        >>> metrics = generate_metrics_report(backtest_results)
        >>> print(f"Sharpe: {metrics['sharpe_ratio']:.2f}")
    """
    logger.info("Generating performance metrics...")

    equity_curve = backtest_results['equity_curve']
    trades = backtest_results.get('trades', [])

    # Calculate returns
    if 'returns' in backtest_results:
        returns = backtest_results['returns']
    else:
        returns = calculate_daily_returns(equity_curve)

    # Calculate all metrics
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    annual_return = calculate_annual_return(equity_curve)
    dd_info = calculate_max_drawdown(equity_curve)
    win_rate = calculate_win_rate(trades) if len(trades) > 0 else 0.0
    profit_factor = calculate_profit_factor(trades) if len(trades) > 0 else 0.0
    return_dd_ratio = calculate_return_drawdown_ratio(annual_return, dd_info['max_drawdown'])

    # Calculate additional stats
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) if len(equity_curve) > 0 else 0.0

    metrics = {
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'annual_return': annual_return,
        'total_return': total_return,
        'max_drawdown': dd_info['max_drawdown'],
        'max_drawdown_pct': dd_info['max_drawdown_pct'],
        'return_dd_ratio': return_dd_ratio,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_trades': len(trades),
        'mean_return': returns.mean() if len(returns) > 0 else 0.0,
        'std_return': returns.std() if len(returns) > 0 else 0.0,
        'drawdown_info': dd_info
    }

    logger.info("✓ Metrics calculated")

    return metrics


def print_metrics_summary(metrics):
    """
    Print formatted metrics summary.

    Args:
        metrics: Dict from generate_metrics_report()

    Example:
        >>> print_metrics_summary(metrics)
    """
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS SUMMARY")
    print("=" * 60)

    print(f"\n📊 Returns:")
    print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
    print(f"  Annual Return (CAGR): {metrics['annual_return']*100:>8.2f}%")
    print(f"  Mean Daily Return:   {metrics['mean_return']*100:>8.4f}%")

    print(f"\n📉 Risk:")
    print(f"  Maximum Drawdown:    {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Daily Volatility:    {metrics['std_return']*100:>8.4f}%")

    print(f"\n📈 Risk-Adjusted:")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}")
    print(f"  Return/DD Ratio:     {metrics['return_dd_ratio']:>8.2f}")

    print(f"\n🎯 Trading:")
    print(f"  Total Trades:        {metrics['total_trades']:>8d}")
    print(f"  Win Rate:            {metrics['win_rate']*100:>8.1f}%")
    print(f"  Profit Factor:       {metrics['profit_factor']:>8.2f}")

    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Testing metrics module...")

    # Create sample equity curve
    dates = pd.date_range('2020-01-01', periods=252, freq='D')
    equity = pd.Series(
        10000 * (1 + np.random.randn(252).cumsum() * 0.01),
        index=dates
    )

    # Create sample trades
    trades = pd.DataFrame({
        'pnl': np.random.randn(50) * 100
    })

    # Test individual metrics
    print("\nTesting individual metrics:")
    returns = calculate_daily_returns(equity)
    print(f"Sharpe Ratio: {calculate_sharpe_ratio(returns):.2f}")
    print(f"Annual Return: {calculate_annual_return(equity)*100:.2f}%")

    dd = calculate_max_drawdown(equity)
    print(f"Max Drawdown: {dd['max_drawdown_pct']:.2f}%")

    print(f"Win Rate: {calculate_win_rate(trades)*100:.1f}%")
    print(f"Profit Factor: {calculate_profit_factor(trades):.2f}")

    # Test complete report
    backtest_results = {
        'equity_curve': equity,
        'trades': trades,
        'returns': returns
    }

    metrics = generate_metrics_report(backtest_results)
    print_metrics_summary(metrics)

    print("\n✓ Metrics module test complete!")
