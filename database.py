"""
Database Manager
SQLite database for storing all simulator data
"""

import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manage SQLite database for simulator
    """

    def __init__(self, db_path: str = 'data/live_trading.db'):
        """
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Create directory if needed
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        logger.info(f"Database initialized: {db_path}")

    def _init_database(self):
        """Create tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table 1: Market snapshots (price + sentiment)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                volume INTEGER,
                vix REAL,
                fear_greed_value INTEGER,
                fear_greed_text TEXT,
                is_market_open BOOLEAN,
                UNIQUE(timestamp, symbol)
            )
        ''')

        # Table 2: Signals (model predictions + regime)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                action TEXT NOT NULL,
                reason TEXT,
                prediction REAL,
                extreme_condition TEXT,
                threshold REAL,
                position_size REAL,
                current_price REAL,
                vix REAL,
                fear_greed INTEGER,
                UNIQUE(timestamp)
            )
        ''')

        # Table 3: Trades (actual buys/sells)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                cost REAL,
                commission REAL,
                total_cost REAL,
                reason TEXT
            )
        ''')

        # Table 4: Portfolio snapshots (daily state)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                cash REAL NOT NULL,
                position_value REAL NOT NULL,
                portfolio_value REAL NOT NULL,
                return_pct REAL,
                spy_quantity INTEGER,
                UNIQUE(timestamp)
            )
        ''')

        conn.commit()
        conn.close()

        logger.info("Database tables created/verified")

    def save_market_snapshot(self, snapshot: Dict):
        """Save market snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO market_snapshots
                (timestamp, symbol, price, open, high, low, volume,
                 vix, fear_greed_value, fear_greed_text, is_market_open)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot['timestamp'],
                snapshot.get('symbol', 'SPY'),
                snapshot['price'],
                snapshot.get('open'),
                snapshot.get('high'),
                snapshot.get('low'),
                snapshot.get('volume'),
                snapshot.get('VIX'),
                snapshot.get('fear_greed_value'),
                snapshot.get('fear_greed_text'),
                snapshot.get('is_market_open', False)
            ))

            conn.commit()
            logger.debug("Market snapshot saved")

        except Exception as e:
            logger.error(f"Error saving market snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()

    def save_signal(self, signal: Dict):
        """Save trading signal"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO signals
                (timestamp, action, reason, prediction, extreme_condition,
                 threshold, position_size, current_price, vix, fear_greed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'],
                signal['action'],
                signal.get('reason'),
                signal.get('prediction'),
                signal.get('extreme_condition'),
                signal.get('threshold'),
                signal.get('position_size'),
                signal.get('current_price'),
                signal.get('vix'),
                signal.get('fear_greed')
            ))

            conn.commit()
            logger.debug("Signal saved")

        except Exception as e:
            logger.error(f"Error saving signal: {e}")
            conn.rollback()
        finally:
            conn.close()

    def save_trade(self, trade: Dict):
        """Save executed trade"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT INTO trades
                (timestamp, symbol, side, quantity, price,
                 cost, commission, total_cost, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade.get('timestamp', datetime.now()),
                trade.get('symbol', 'SPY'),
                trade['side'],
                trade['quantity'],
                trade['price'],
                trade.get('cost'),
                trade.get('commission'),
                trade.get('total_cost', trade.get('net_proceeds')),
                trade.get('reason')
            ))

            conn.commit()
            logger.debug(f"Trade saved: {trade['side']} {trade['quantity']} @ {trade['price']:.2f}")

        except Exception as e:
            logger.error(f"Error saving trade: {e}")
            conn.rollback()
        finally:
            conn.close()

    def save_portfolio_snapshot(self, snapshot: Dict):
        """Save portfolio state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            cursor.execute('''
                INSERT OR REPLACE INTO portfolio_snapshots
                (timestamp, cash, position_value, portfolio_value,
                 return_pct, spy_quantity)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                snapshot['timestamp'],
                snapshot['cash'],
                snapshot['position_value'],
                snapshot['portfolio_value'],
                snapshot.get('return'),
                snapshot.get('positions', {}).get('SPY', 0)
            ))

            conn.commit()
            logger.debug("Portfolio snapshot saved")

        except Exception as e:
            logger.error(f"Error saving portfolio snapshot: {e}")
            conn.rollback()
        finally:
            conn.close()

    def get_portfolio_history(self, days: int = 30) -> pd.DataFrame:
        """Get portfolio history"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT * FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT ?
        '''

        df = pd.read_sql_query(query, conn, params=(days,))
        conn.close()

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')

        return df

    def get_recent_trades(self, limit: int = 20) -> pd.DataFrame:
        """Get recent trades"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT * FROM trades
            ORDER BY timestamp DESC
            LIMIT ?
        '''

        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_recent_signals(self, limit: int = 20) -> pd.DataFrame:
        """Get recent signals"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT * FROM signals
            ORDER BY timestamp DESC
            LIMIT ?
        '''

        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        if len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        return df

    def get_current_position(self) -> Dict:
        """Get latest portfolio snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM portfolio_snapshots
            ORDER BY timestamp DESC
            LIMIT 1
        ''')

        row = cursor.fetchone()
        conn.close()

        if row:
            columns = ['id', 'timestamp', 'cash', 'position_value',
                      'portfolio_value', 'return_pct', 'spy_quantity']
            return dict(zip(columns, row))
        else:
            return None


# Test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing Database Manager...")
    print("="*60)

    db = DatabaseManager('data/test_trading.db')

    # Test saving data
    snapshot = {
        'timestamp': datetime.now(),
        'symbol': 'SPY',
        'price': 589.23,
        'VIX': 15.5,
        'fear_greed_value': 65,
        'fear_greed_text': 'Greed',
        'is_market_open': True
    }

    db.save_market_snapshot(snapshot)
    print("[OK] Market snapshot saved")

    signal = {
        'timestamp': datetime.now(),
        'action': 'BUY',
        'reason': 'Test signal',
        'prediction': 0.0082,
        'extreme_condition': 'NORMAL',
        'position_size': 0.5
    }

    db.save_signal(signal)
    print("[OK] Signal saved")

    # Test reading
    history = db.get_portfolio_history(days=7)
    print(f"\n[OK] Portfolio history: {len(history)} records")

    print("\n[OK] Database manager working!")
