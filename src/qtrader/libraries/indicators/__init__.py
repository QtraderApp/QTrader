"""
Indicator Library.

All indicators must inherit from BaseIndicator and implement:
- calculate(): Stateless batch calculation
- update(): Stateful incremental update
- reset(): Clear internal state
- value: Current indicator value
- is_ready: Whether indicator is ready

Available Indicators:

Moving Averages:
- SMA: Simple Moving Average
- EMA: Exponential Moving Average
- WMA: Weighted Moving Average
- DEMA: Double Exponential Moving Average
- TEMA: Triple Exponential Moving Average
- HMA: Hull Moving Average
- SMMA: Smoothed Moving Average (RMA)
"""

from qtrader.libraries.indicators.base import BaseIndicator
from qtrader.libraries.indicators.buildin.moving_averages import DEMA, EMA, HMA, SMA, SMMA, TEMA, WMA

__all__ = [
    # Base
    "BaseIndicator",
    # Moving Averages
    "SMA",
    "EMA",
    "WMA",
    "DEMA",
    "TEMA",
    "HMA",
    "SMMA",
]
