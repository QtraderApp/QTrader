"""
Technical indicators module for QTrader.

Contains indicator business logic organized by category:

Infrastructure (root level):
- Base Indicator class
- IndicatorManager for caching
- Helper functions for pattern detection

Built-in Indicators (by category):
- Trend: SMA, EMA
- Volatility: ATR, Bollinger Bands
- Momentum: RSI, MACD
"""

from qtrader.indicators.base import Indicator
from qtrader.indicators.helpers import (
    above_threshold,
    below_threshold,
    between_thresholds,
    crossed_above,
    crossed_above_threshold,
    crossed_below,
    crossed_below_threshold,
    divergence_bearish,
    divergence_bullish,
    histogram_flipped_negative,
    histogram_flipped_positive,
    is_decreasing,
    is_increasing,
)
from qtrader.indicators.manager import IndicatorManager
from qtrader.indicators.momentum import MACD, RSI, MACDIndicator, MACDResult
from qtrader.indicators.trend import EMA, SMA
from qtrader.indicators.volatility import ATR, BollingerBands, BollingerBandsIndicator

__all__ = [
    # Infrastructure
    "Indicator",
    "IndicatorManager",
    # Trend indicators
    "SMA",
    "EMA",
    # Volatility indicators
    "BollingerBandsIndicator",
    "BollingerBands",
    "ATR",
    # Momentum indicators
    "RSI",
    "MACDIndicator",
    "MACD",
    "MACDResult",
    # Helper functions
    "crossed_above",
    "crossed_below",
    "crossed_above_threshold",
    "crossed_below_threshold",
    "above_threshold",
    "below_threshold",
    "between_thresholds",
    "divergence_bullish",
    "divergence_bearish",
    "histogram_flipped_positive",
    "histogram_flipped_negative",
    "is_increasing",
    "is_decreasing",
]
