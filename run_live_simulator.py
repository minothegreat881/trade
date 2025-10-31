"""
Live Trading Simulator - Main Executor
Runs daily paper trading simulation with real-time data
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time

from data_fetcher import DataFetcher
from trading_engine import TradingEngine
from portfolio_simulator import PortfolioSimulator
from database import DatabaseManager
import config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_simulator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LiveSimulator:
    """
    Main coordinator for live paper trading simulation
    """

    def __init__(self):
        """Initialize all components"""
        logger.info("=" * 60)
        logger.info("LIVE SIMULATOR STARTING")
        logger.info("=" * 60)

        # Initialize components
        self.data_fetcher = DataFetcher()
        self.trading_engine = TradingEngine()
        self.portfolio = PortfolioSimulator(initial_capital=config.INITIAL_CAPITAL)
        self.db = DatabaseManager()

        # Load historical data for features (need 100+ days)
        self._load_historical_data()

        logger.info("All components initialized successfully")

    def _load_historical_data(self):
        """Load historical data for feature engineering"""
        logger.info("Loading historical data...")

        # Load last 200 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=200)

        self.historical_data = self.data_fetcher.get_historical_data(
            symbol='SPY',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        logger.info(f"Loaded {len(self.historical_data)} days of historical data")

    def run_daily_cycle(self):
        """
        Execute one daily trading cycle:
        1. Fetch current market snapshot
        2. Generate trading signal
        3. Execute trade (if any)
        4. Update portfolio
        5. Save to database
        """
        try:
            logger.info("=" * 60)
            logger.info(f"DAILY CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)

            # Step 1: Fetch current market snapshot
            logger.info("Step 1: Fetching market snapshot...")
            snapshot = self.data_fetcher.get_current_snapshot()

            if not snapshot:
                logger.error("Failed to fetch market snapshot")
                return False

            logger.info(f"  SPY: ${snapshot['price']:.2f}")
            logger.info(f"  VIX: {snapshot.get('VIX', 'N/A')}")
            logger.info(f"  Fear & Greed: {snapshot.get('fear_greed_value', 'N/A')} ({snapshot.get('fear_greed_text', 'N/A')})")
            logger.info(f"  Market Open: {snapshot.get('is_market_open', False)}")

            # Save snapshot to database
            self.db.save_market_snapshot(snapshot)

            # Step 2: Generate trading signal
            logger.info("\nStep 2: Generating trading signal...")
            signal = self.trading_engine.generate_signal(
                historical_data=self.historical_data,
                current_snapshot=snapshot
            )

            logger.info(f"  Action: {signal['action']}")
            logger.info(f"  Reason: {signal['reason']}")
            logger.info(f"  Prediction: {signal['prediction']:.4f}")
            logger.info(f"  Regime: {signal['extreme_condition']}")
            logger.info(f"  Position Size: {signal['position_size']*100:.1f}%")

            # Save signal to database
            self.db.save_signal(signal)

            # Step 3: Evaluate and execute trade (if needed)
            logger.info("\nStep 3: Evaluating trade execution...")

            current_position = self.portfolio.positions.get('SPY', 0)
            trade_instruction = self.trading_engine.evaluate_signal(
                signal=signal,
                current_position=current_position,
                portfolio_value=self.portfolio.get_portfolio_value(snapshot['price']),
                cash=self.portfolio.cash
            )

            if trade_instruction:
                logger.info(f"  Executing: {trade_instruction['action']} {trade_instruction['quantity']} SPY @ ${trade_instruction['price']:.2f}")

                # Execute trade in portfolio
                if trade_instruction['action'] == 'BUY':
                    result = self.portfolio.buy(
                        symbol='SPY',
                        quantity=trade_instruction['quantity'],
                        price=trade_instruction['price']
                    )
                else:  # SELL
                    result = self.portfolio.sell(
                        symbol='SPY',
                        quantity=trade_instruction['quantity'],
                        price=trade_instruction['price']
                    )

                if result:
                    # Add metadata and save to database
                    result['timestamp'] = snapshot['timestamp']
                    result['reason'] = trade_instruction['reason']
                    self.db.save_trade(result)
                    logger.info(f"  Trade executed successfully")
                else:
                    logger.warning("  Trade execution failed")
            else:
                logger.info("  No trade needed")

            # Step 4: Update portfolio snapshot
            logger.info("\nStep 4: Updating portfolio state...")

            portfolio_value = self.portfolio.get_portfolio_value(snapshot['price'])
            position_value = self.portfolio.positions.get('SPY', 0) * snapshot['price']

            portfolio_snapshot = {
                'timestamp': snapshot['timestamp'],
                'cash': self.portfolio.cash,
                'position_value': position_value,
                'portfolio_value': portfolio_value,
                'return': (portfolio_value / self.portfolio.initial_capital - 1) * 100,
                'positions': self.portfolio.positions.copy()
            }

            self.db.save_portfolio_snapshot(portfolio_snapshot)

            logger.info(f"  Cash: ${self.portfolio.cash:,.2f}")
            logger.info(f"  Position: {self.portfolio.positions.get('SPY', 0)} shares (${position_value:,.2f})")
            logger.info(f"  Total: ${portfolio_value:,.2f}")
            logger.info(f"  Return: {portfolio_snapshot['return']:.2f}%")

            # Step 5: Update historical data for next cycle
            logger.info("\nStep 5: Updating historical data...")

            # Add today's snapshot to historical data
            new_row = pd.DataFrame([{
                'Open': snapshot.get('open', snapshot['price']),
                'High': snapshot.get('high', snapshot['price']),
                'Low': snapshot.get('low', snapshot['price']),
                'Close': snapshot['price'],
                'Volume': snapshot.get('volume', 0),
                'fear_greed_value': snapshot.get('fear_greed_value', 50),
                'VIX': snapshot.get('VIX', 15)
            }], index=[pd.Timestamp(snapshot['timestamp'])])

            self.historical_data = pd.concat([self.historical_data, new_row])
            self.historical_data = self.historical_data.tail(200)  # Keep last 200 days

            logger.info("=" * 60)
            logger.info("DAILY CYCLE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Error in daily cycle: {e}", exc_info=True)
            return False


def main():
    """Main entry point"""
    try:
        # Create logs directory
        Path('logs').mkdir(exist_ok=True)

        # Initialize simulator
        simulator = LiveSimulator()

        # Run one daily cycle
        success = simulator.run_daily_cycle()

        if success:
            logger.info("\nSimulator run completed successfully!")
            return 0
        else:
            logger.error("\nSimulator run failed!")
            return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
