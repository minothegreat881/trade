"""
Portfolio Simulator
Tracks portfolio state, executes trades, calculates P&L
"""

import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PortfolioSimulator:
    """
    Simulates portfolio with cash and positions
    Tracks trades, commissions, and P&L
    """

    def __init__(self, initial_capital: float = 100000.0, commission_rate: float = 0.001):
        """
        Args:
            initial_capital: Starting cash
            commission_rate: Commission per trade (0.001 = 0.1%)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.commission_rate = commission_rate
        self.trade_history = []

        logger.info(f"Portfolio initialized: ${initial_capital:,.2f} cash, {commission_rate*100:.2f}% commission")

    def buy(self, symbol: str, quantity: int, price: float) -> Optional[Dict]:
        """
        Execute buy order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share

        Returns:
            Trade result dict or None if failed
        """
        if quantity <= 0:
            logger.error(f"Invalid quantity: {quantity}")
            return None

        # Calculate cost
        cost = quantity * price
        commission = cost * self.commission_rate
        total_cost = cost + commission

        # Check if enough cash
        if total_cost > self.cash:
            logger.warning(f"Insufficient cash: ${self.cash:.2f} < ${total_cost:.2f}")
            return None

        # Execute trade
        self.cash -= total_cost
        self.positions[symbol] = self.positions.get(symbol, 0) + quantity

        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': 'BUY',
            'quantity': quantity,
            'price': price,
            'cost': cost,
            'commission': commission,
            'total_cost': total_cost,
            'cash_after': self.cash
        }

        self.trade_history.append(trade)

        logger.info(f"BUY: {quantity} {symbol} @ ${price:.2f} (cost: ${total_cost:.2f}, cash left: ${self.cash:.2f})")

        return trade

    def sell(self, symbol: str, quantity: int, price: float) -> Optional[Dict]:
        """
        Execute sell order

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Price per share

        Returns:
            Trade result dict or None if failed
        """
        if quantity <= 0:
            logger.error(f"Invalid quantity: {quantity}")
            return None

        # Check if enough shares
        current_position = self.positions.get(symbol, 0)
        if quantity > current_position:
            logger.warning(f"Insufficient shares: {current_position} < {quantity}")
            return None

        # Calculate proceeds
        proceeds = quantity * price
        commission = proceeds * self.commission_rate
        net_proceeds = proceeds - commission

        # Execute trade
        self.cash += net_proceeds
        self.positions[symbol] -= quantity

        # Remove position if fully closed
        if self.positions[symbol] == 0:
            del self.positions[symbol]

        trade = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'side': 'SELL',
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds,
            'commission': commission,
            'net_proceeds': net_proceeds,
            'cash_after': self.cash
        }

        self.trade_history.append(trade)

        logger.info(f"SELL: {quantity} {symbol} @ ${price:.2f} (proceeds: ${net_proceeds:.2f}, cash now: ${self.cash:.2f})")

        return trade

    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """
        Calculate total portfolio value

        Args:
            current_prices: Dict of symbol -> current_price
                           If single float, assumes it's for SPY

        Returns:
            Total portfolio value (cash + positions)
        """
        # Handle single price for SPY
        if isinstance(current_prices, (int, float)):
            current_prices = {'SPY': float(current_prices)}

        if current_prices is None:
            current_prices = {}

        position_value = 0.0

        for symbol, quantity in self.positions.items():
            price = current_prices.get(symbol, 0.0)
            position_value += quantity * price

        total_value = self.cash + position_value

        return total_value

    def get_return(self, current_prices: Dict[str, float] = None) -> float:
        """
        Calculate total return %

        Args:
            current_prices: Dict of symbol -> current_price

        Returns:
            Return percentage
        """
        portfolio_value = self.get_portfolio_value(current_prices)
        return ((portfolio_value / self.initial_capital) - 1) * 100

    def get_position_summary(self, current_prices: Dict[str, float] = None) -> Dict:
        """
        Get summary of current positions

        Args:
            current_prices: Dict of symbol -> current_price

        Returns:
            Dict with position details
        """
        if current_prices is None:
            current_prices = {}

        positions_summary = []

        for symbol, quantity in self.positions.items():
            price = current_prices.get(symbol, 0.0)
            value = quantity * price

            positions_summary.append({
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'value': value
            })

        return {
            'cash': self.cash,
            'positions': positions_summary,
            'total_position_value': sum(p['value'] for p in positions_summary),
            'total_portfolio_value': self.get_portfolio_value(current_prices),
            'return_pct': self.get_return(current_prices)
        }

    def reset(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_history = []
        logger.info("Portfolio reset to initial state")


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Portfolio Simulator...")
    print("=" * 60)

    # Initialize with $100k
    portfolio = PortfolioSimulator(initial_capital=100000.0)

    print(f"\n[INIT] Cash: ${portfolio.cash:,.2f}")
    print(f"[INIT] Positions: {portfolio.positions}")

    # Test BUY
    print("\n[TEST] Buying 100 SPY @ $580...")
    trade = portfolio.buy('SPY', 100, 580.0)

    if trade:
        print(f"[OK] Trade executed")
        print(f"[OK] Cost: ${trade['total_cost']:,.2f}")
        print(f"[OK] Cash left: ${portfolio.cash:,.2f}")
        print(f"[OK] Position: {portfolio.positions}")

    # Test portfolio value
    print("\n[TEST] Calculating portfolio value...")
    portfolio_value = portfolio.get_portfolio_value({'SPY': 590.0})
    print(f"[OK] Portfolio value: ${portfolio_value:,.2f}")

    return_pct = portfolio.get_return({'SPY': 590.0})
    print(f"[OK] Return: {return_pct:.2f}%")

    # Test SELL
    print("\n[TEST] Selling 50 SPY @ $590...")
    trade = portfolio.sell('SPY', 50, 590.0)

    if trade:
        print(f"[OK] Trade executed")
        print(f"[OK] Proceeds: ${trade['net_proceeds']:,.2f}")
        print(f"[OK] Cash now: ${portfolio.cash:,.2f}")
        print(f"[OK] Position: {portfolio.positions}")

    # Test summary
    print("\n[TEST] Getting position summary...")
    summary = portfolio.get_position_summary({'SPY': 590.0})
    print(f"[OK] Cash: ${summary['cash']:,.2f}")
    print(f"[OK] Position Value: ${summary['total_position_value']:,.2f}")
    print(f"[OK] Total Value: ${summary['total_portfolio_value']:,.2f}")
    print(f"[OK] Return: {summary['return_pct']:.2f}%")

    print("\n[OK] Portfolio simulator working!")
