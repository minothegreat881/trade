"""
Scheduler for Live Trading Simulator
Runs daily at market close (4:00 PM ET)
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
import pytz
from pathlib import Path

from run_live_simulator import LiveSimulator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Timezone
ET = pytz.timezone('America/New_York')


def is_market_day():
    """
    Check if today is a trading day
    Returns False on weekends
    """
    today = datetime.now(ET)
    # 0 = Monday, 6 = Sunday
    if today.weekday() >= 5:  # Saturday or Sunday
        return False
    return True


def run_daily_simulation():
    """
    Execute daily simulation
    Only runs on market days
    """
    try:
        logger.info("=" * 60)
        logger.info(f"SCHEDULED RUN - {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        logger.info("=" * 60)

        # Check if market day
        if not is_market_day():
            logger.info("Today is not a market day (weekend). Skipping.")
            return

        # Initialize and run simulator
        simulator = LiveSimulator()
        success = simulator.run_daily_cycle()

        if success:
            logger.info("Scheduled simulation completed successfully!")
        else:
            logger.error("Scheduled simulation failed!")

    except Exception as e:
        logger.error(f"Error in scheduled run: {e}", exc_info=True)


def main():
    """
    Main scheduler loop
    Runs daily at 4:30 PM ET (30 min after market close)
    """
    logger.info("=" * 60)
    logger.info("LIVE TRADING SCHEDULER STARTED")
    logger.info("=" * 60)
    logger.info("Schedule: Daily at 4:30 PM ET (30 min after market close)")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Schedule daily run at 4:30 PM ET
    schedule.every().day.at("16:30").do(run_daily_simulation)

    # Optional: Run on startup (for testing)
    # run_daily_simulation()

    # Main loop
    try:
        while True:
            # Check pending scheduled tasks
            schedule.run_pending()

            # Show next run time
            next_run = schedule.next_run()
            if next_run:
                time_until = next_run - datetime.now()
                logger.info(f"Next run in: {time_until} (at {next_run.strftime('%Y-%m-%d %H:%M:%S')})")

            # Sleep for 1 hour between checks
            time.sleep(3600)

    except KeyboardInterrupt:
        logger.info("\nScheduler stopped by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
