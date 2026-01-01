"""REST API bridge for MQL5 integration - live data and backtesting."""
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("MQL5Bridge")

# Database for storing backtest results and live data
DB_PATH = Path("output_central/mql5_data.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class LiveTick:
    """Live price tick data from MT5."""
    symbol: str
    bid: float
    ask: float
    timestamp: float
    volume: int


@dataclass
class BacktestResult:
    """MQL5 backtest result record."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_return: float
    sharpe_ratio: float
    win_rate: float
    max_drawdown: float
    trades_count: int
    result_json: str
    import_timestamp: str


class MQL5Bridge:
    """REST API bridge for MQL5 integration."""

    def __init__(self):
        self.init_database()
        self.live_data_buffer = {}
        self.active_connections = {}

    def init_database(self):
        """Initialize SQLite database for MQL5 data."""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Backtest results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                total_return REAL,
                sharpe_ratio REAL,
                win_rate REAL,
                max_drawdown REAL,
                trades_count INTEGER,
                result_json TEXT,
                import_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Live tick data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_ticks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                bid REAL,
                ask REAL,
                volume INTEGER,
                timestamp REAL,
                received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Trading signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal_type TEXT,
                confidence REAL,
                entry_price REAL,
                stop_loss REAL,
                take_profit REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sent_to_mt5 INTEGER DEFAULT 0
            )
        ''')

        conn.commit()
        conn.close()
        logger.info(f"MQL5 database initialized at {DB_PATH}")

    def import_backtest_result(self, data: dict[str, Any]) -> bool:
        """Import backtest result from MQL5."""
        try:
            result = BacktestResult(
                strategy_name=data.get("strategy_name", "Unknown"),
                symbol=data.get("symbol", ""),
                timeframe=data.get("timeframe", ""),
                start_date=data.get("start_date", ""),
                end_date=data.get("end_date", ""),
                total_return=float(data.get("total_return", 0)),
                sharpe_ratio=float(data.get("sharpe_ratio", 0)),
                win_rate=float(data.get("win_rate", 0)),
                max_drawdown=float(data.get("max_drawdown", 0)),
                trades_count=int(data.get("trades_count", 0)),
                result_json=json.dumps(data.get("full_results", {})),
                import_timestamp=datetime.now().isoformat()
            )

            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO backtest_results 
                (strategy_name, symbol, timeframe, start_date, end_date, 
                 total_return, sharpe_ratio, win_rate, max_drawdown, 
                 trades_count, result_json, import_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.strategy_name, result.symbol, result.timeframe,
                result.start_date, result.end_date, result.total_return,
                result.sharpe_ratio, result.win_rate, result.max_drawdown,
                result.trades_count, result.result_json, result.import_timestamp
            ))

            conn.commit()
            conn.close()

            logger.info(f"Imported backtest: {result.strategy_name} {result.symbol}")
            return True

        except Exception as e:
            logger.error(f"Failed to import backtest: {e}")
            return False

    def store_live_tick(self, tick: dict[str, Any]) -> bool:
        """Store live price tick from MT5."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO live_ticks (symbol, bid, ask, volume, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                tick.get("symbol"),
                float(tick.get("bid", 0)),
                float(tick.get("ask", 0)),
                int(tick.get("volume", 0)),
                float(tick.get("timestamp", 0))
            ))

            conn.commit()
            conn.close()

            # Store in buffer for real-time access
            symbol = tick.get("symbol")
            if symbol not in self.live_data_buffer:
                self.live_data_buffer[symbol] = []
            self.live_data_buffer[symbol].append(tick)

            # Keep buffer limited to last 1000 ticks
            if len(self.live_data_buffer[symbol]) > 1000:
                self.live_data_buffer[symbol] = self.live_data_buffer[symbol][-1000:]

            return True

        except Exception as e:
            logger.error(f"Failed to store tick: {e}")
            return False

    def get_live_data(self, symbol: str, limit: int = 100) -> list[dict]:
        """Retrieve recent live data for symbol."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT symbol, bid, ask, volume, timestamp, received_at
                FROM live_ticks
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, limit))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "symbol": row[0],
                    "bid": row[1],
                    "ask": row[2],
                    "volume": row[3],
                    "timestamp": row[4],
                    "received_at": row[5]
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get live data: {e}")
            return []

    def create_signal(self, signal_data: dict[str, Any]) -> bool:
        """Create trading signal to send to MT5."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO trading_signals 
                (symbol, signal_type, confidence, entry_price, stop_loss, take_profit)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                signal_data.get("symbol"),
                signal_data.get("signal_type"),  # BUY, SELL, CLOSE
                float(signal_data.get("confidence", 0.5)),
                float(signal_data.get("entry_price", 0)),
                float(signal_data.get("stop_loss", 0)),
                float(signal_data.get("take_profit", 0))
            ))

            conn.commit()
            conn.close()

            logger.info(f"Signal created: {signal_data.get('symbol')} {signal_data.get('signal_type')}")
            return True

        except Exception as e:
            logger.error(f"Failed to create signal: {e}")
            return False

    def get_pending_signals(self, sent: bool = False) -> list[dict]:
        """Get pending signals for MT5."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, symbol, signal_type, confidence, entry_price, 
                       stop_loss, take_profit, created_at
                FROM trading_signals
                WHERE sent_to_mt5 = ?
                ORDER BY created_at DESC
                LIMIT 100
            ''', (1 if sent else 0,))

            rows = cursor.fetchall()
            conn.close()

            return [
                {
                    "id": row[0],
                    "symbol": row[1],
                    "signal_type": row[2],
                    "confidence": row[3],
                    "entry_price": row[4],
                    "stop_loss": row[5],
                    "take_profit": row[6],
                    "created_at": row[7]
                }
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to get signals: {e}")
            return []

    def mark_signal_sent(self, signal_id: int) -> bool:
        """Mark signal as sent to MT5."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE trading_signals
                SET sent_to_mt5 = 1
                WHERE id = ?
            ''', (signal_id,))

            conn.commit()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"Failed to mark signal sent: {e}")
            return False

    def get_backtest_comparison(self, symbol: str = None, limit: int = 10) -> pd.DataFrame:
        """Get comparison of backtest results."""
        try:
            conn = sqlite3.connect(DB_PATH)

            query = '''
                SELECT strategy_name, symbol, timeframe, start_date, end_date,
                       total_return, sharpe_ratio, win_rate, max_drawdown, 
                       trades_count, import_timestamp
                FROM backtest_results
            '''
            params = []

            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)

            query += " ORDER BY import_timestamp DESC LIMIT ?"
            params.append(limit)

            df = pd.read_sql_query(query, conn, params=params)
            conn.close()

            return df

        except Exception as e:
            logger.error(f"Failed to get comparison: {e}")
            return pd.DataFrame()

    def export_backtest_csv(self, output_path: str, symbol: str = None) -> bool:
        """Export backtest results to CSV for MT5 analysis."""
        try:
            df = self.get_backtest_comparison(symbol)
            if df.empty:
                return False

            df.to_csv(output_path, index=False)
            logger.info(f"Exported backtest results to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False

    def health_check(self) -> dict[str, Any]:
        """Check bridge health and statistics."""
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            # Get statistics
            cursor.execute("SELECT COUNT(*) FROM backtest_results")
            backtest_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM live_ticks")
            tick_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM trading_signals WHERE sent_to_mt5 = 0")
            pending_signals = cursor.fetchone()[0]

            cursor.execute("SELECT DISTINCT symbol FROM live_ticks ORDER BY symbol")
            symbols = [row[0] for row in cursor.fetchall()]

            conn.close()

            return {
                "status": "OPERATIONAL",
                "database_path": str(DB_PATH),
                "backtest_results_count": backtest_count,
                "live_ticks_count": tick_count,
                "pending_signals": pending_signals,
                "active_symbols": symbols,
                "buffer_size": {symbol: len(ticks) for symbol, ticks in self.live_data_buffer.items()},
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "ERROR",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global bridge instance
bridge = MQL5Bridge()

