"""
QTrader - Quantitative Trading Environment

Public API for building and running deterministic backtests.
"""

__version__ = "0.1.0"

from qtrader.api.backtest import Backtest, load_config, run_backtest
from qtrader.api.context import Context

# Public API exports (will be implemented in later stages)
from qtrader.api.strategy import Strategy

__all__ = [
    "Strategy",
    "Context",
    "Backtest",
    "load_config",
    "run_backtest",
    "__version__",
]
