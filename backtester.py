"""
Backtesting Module - Trading Simulation

Simulates realistic trading with:
- Transaction costs
- Position sizing
- Entry/exit logic
- Portfolio tracking
"""

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class Backtester:
    """
    Backtesting framework for trading strategies.

    Simulates realistic trading with transaction costs,
    position sizing, and portfolio management.

    Attributes:
        initial_capital (float): Starting capital
        transaction_cost (float): Cost per trade (fraction)
        holding_period (int): Days to hold position
        position_size_pct (float): Fraction of capital per trade
    """

    def __init__(self,
                 initial_capital=10000,
                 transaction_cost=0.001,
                 holding_period=5,
                 position_size_pct=0.5,
                 prediction_threshold=0.001):
        """
        Initialize backtester.

        Args:
            initial_capital: Starting capital in dollars
            transaction_cost: Cost per trade (0.001 = 0.1%)
            holding_period: Days to hold each position
            position_size_pct: Fraction of capital to use per trade (0.5 = 50%)
            prediction_threshold: Minimum prediction to enter trade

        Example:
            >>> backtester = Backtester(initial_capital=10000,
            ...                         transaction_cost=0.001,
            ...                         holding_period=5)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.holding_period = holding_period
        self.position_size_pct = position_size_pct
        self.prediction_threshold = prediction_threshold

        logger.info(f"Initialized Backtester:")
        logger.info(f"  Initial capital: ${initial_capital:,.0f}")
        logger.info(f"  Transaction cost: {transaction_cost*100:.2f}%")
        logger.info(f"  Holding period: {holding_period} days")
        logger.info(f"  Position size: {position_size_pct*100:.0f}%")

    def run_backtest(self, predictions, actuals, prices):
        """
        Run backtest simulation.

        Args:
            predictions: Model predictions (pd.Series with DatetimeIndex)
            actuals: Actual returns (pd.Series)
            prices: Price data (pd.Series)

        Returns:
            dict with:
                - equity_curve: Portfolio value over time
                - trades: List of executed trades
                - returns: Daily returns
                - metrics: Basic statistics

        Example:
            >>> results = backtester.run_backtest(predictions, actuals, prices)
            >>> print(f"Final value: ${results['equity_curve'].iloc[-1]:,.0f}")
        """
        logger.info("Running backtest simulation...")

        # Align data
        df = pd.DataFrame({
            'prediction': predictions,
            'actual': actuals,
            'price': prices
        }).dropna()

        if len(df) == 0:
            logger.error("No valid data after alignment!")
            return self._empty_results()

        # Initialize tracking
        capital = self.initial_capital
        equity_curve = []
        trades = []
        positions = []  # Track open positions
        dates = []

        logger.info(f"Simulating {len(df)} trading days...")

        # Simulate trading
        for i in range(len(df)):
            current_date = df.index[i]
            prediction = df['prediction'].iloc[i]
            price = df['price'].iloc[i]

            # Check for position exits
            positions_to_remove = []
            for pos in positions:
                if (current_date - pos['entry_date']).days >= self.holding_period:
                    # Exit position
                    exit_price = price
                    pnl = self._calculate_pnl(pos, exit_price)

                    # Update capital
                    capital += pos['position_value'] + pnl

                    # Record trade
                    trades.append({
                        'entry_date': pos['entry_date'],
                        'exit_date': current_date,
                        'entry_price': pos['entry_price'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'position_value': pos['position_value'],
                        'pnl': pnl,
                        'pnl_pct': pnl / pos['position_value'],
                        'prediction': pos['prediction']
                    })

                    positions_to_remove.append(pos)

            # Remove closed positions
            for pos in positions_to_remove:
                positions.remove(pos)

            # Check for new entry (LONG ONLY - only buy when prediction is positive)
            if prediction > self.prediction_threshold:
                # Only enter if we have capital and not too many open positions
                available_capital = capital * self.position_size_pct

                if available_capital > 0 and len(positions) < 3:  # Max 3 concurrent positions
                    # Enter LONG position
                    entry_price = price

                    # Calculate position size (including transaction costs)
                    cost_adjusted = available_capital * (1 - self.transaction_cost)
                    shares = cost_adjusted / entry_price  # Always positive for long
                    position_value = shares * entry_price

                    # Deduct from capital
                    capital -= position_value

                    # Create position
                    pos = {
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'shares': shares,
                        'position_value': position_value,
                        'prediction': prediction
                    }
                    positions.append(pos)

            # Calculate total equity (capital + open positions value)
            open_positions_value = sum(pos['shares'] * price for pos in positions)
            total_equity = capital + open_positions_value

            equity_curve.append(total_equity)
            dates.append(current_date)

        # Close any remaining positions at the end
        if len(positions) > 0:
            logger.info(f"Closing {len(positions)} remaining positions...")
            final_price = df['price'].iloc[-1]
            for pos in positions:
                pnl = self._calculate_pnl(pos, final_price)
                capital += pos['position_value'] + pnl

                trades.append({
                    'entry_date': pos['entry_date'],
                    'exit_date': df.index[-1],
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'shares': pos['shares'],
                    'position_value': pos['position_value'],
                    'pnl': pnl,
                    'pnl_pct': pnl / pos['position_value'],
                    'prediction': pos['prediction']
                })

        # Final equity
        final_equity = capital
        equity_curve[-1] = final_equity

        # Convert to Series with DatetimeIndex (convert timezone-aware to naive)
        equity_curve = pd.Series(equity_curve, index=dates)
        if hasattr(equity_curve.index, 'tz') and equity_curve.index.tz is not None:
            equity_curve.index = equity_curve.index.tz_localize(None)

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        # Calculate statistics
        total_trades = len(trades)
        total_costs = sum(abs(t['position_value']) * self.transaction_cost * 2 for t in trades)

        logger.info(f"✓ Backtest complete:")
        logger.info(f"  Total trades: {total_trades}")
        logger.info(f"  Transaction costs: ${total_costs:.2f}")
        logger.info(f"  Final equity: ${final_equity:,.2f}")
        logger.info(f"  Total return: {(final_equity/self.initial_capital - 1)*100:.2f}%")

        # Return results
        results = {
            'equity_curve': equity_curve,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'returns': returns,
            'initial_capital': self.initial_capital,
            'final_capital': final_equity,
            'total_costs': total_costs,
            'n_trades': total_trades
        }

        return results

    def _calculate_pnl(self, position, exit_price):
        """
        Calculate P&L for a position.

        Args:
            position: Dict with position info
            exit_price: Exit price

        Returns:
            float: Profit/Loss in dollars
        """
        # For long: pnl = shares * (exit - entry)
        # For short: pnl = -shares * (exit - entry) = shares * (entry - exit)
        pnl = position['shares'] * (exit_price - position['entry_price'])

        # Apply transaction cost on exit
        exit_cost = abs(position['shares'] * exit_price) * self.transaction_cost
        pnl -= exit_cost

        return pnl

    def _empty_results(self):
        """Return empty results dict."""
        return {
            'equity_curve': pd.Series([self.initial_capital]),
            'trades': pd.DataFrame(),
            'returns': pd.Series(),
            'initial_capital': self.initial_capital,
            'final_capital': self.initial_capital,
            'total_costs': 0.0,
            'n_trades': 0
        }

    def calculate_benchmark(self, prices):
        """
        Calculate buy & hold benchmark.

        Args:
            prices: Price series

        Returns:
            pd.Series: Benchmark equity curve

        Example:
            >>> benchmark = backtester.calculate_benchmark(prices)
        """
        # Buy at first price, hold until end
        initial_price = prices.iloc[0]
        shares = self.initial_capital / initial_price

        # Calculate equity over time
        benchmark_equity = shares * prices

        return benchmark_equity


if __name__ == "__main__":
    # Example usage
    print("Testing Backtester...")

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')

    predictions = pd.Series(np.random.randn(100) * 0.01, index=dates)
    actuals = pd.Series(np.random.randn(100) * 0.01, index=dates)
    prices = pd.Series(100 * (1 + actuals).cumprod(), index=dates)

    # Run backtest
    backtester = Backtester(
        initial_capital=10000,
        transaction_cost=0.001,
        holding_period=5
    )

    results = backtester.run_backtest(predictions, actuals, prices)

    print(f"\nBacktest Results:")
    print(f"  Initial: ${results['initial_capital']:,.0f}")
    print(f"  Final: ${results['final_capital']:,.0f}")
    print(f"  Return: {(results['final_capital']/results['initial_capital']-1)*100:.2f}%")
    print(f"  Trades: {results['n_trades']}")
    print(f"  Costs: ${results['total_costs']:.2f}")

    print("\n✓ Backtester test complete!")
