"""
QTrader Backtest Engine.

Pure event-driven backtesting orchestrator that coordinates all services
via EventBus without direct service calls or state manipulation.
"""

from qtrader.backtest.config import BacktestConfig, load_backtest_config
from qtrader.backtest.engine import BacktestEngine, BacktestResult

__all__ = [
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResult",
    "load_backtest_config",
]
