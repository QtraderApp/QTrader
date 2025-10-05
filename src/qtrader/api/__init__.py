"""Public API modules for QTrader."""

from qtrader.api.backtest import Backtest, load_config, run_backtest
from qtrader.api.context import Context
from qtrader.api.strategy import Strategy

# Expose indicators through API (business logic is in qtrader.indicators)
from qtrader.indicators import (
    ATR,
    EMA,
    MACD,
    RSI,
    SMA,
    BollingerBands,
    BollingerBandsIndicator,
    Indicator,
    IndicatorManager,
    MACDIndicator,
    MACDResult,
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

__all__ = [
    # Core API
    "Strategy",
    "Context",
    "Backtest",
    "load_config",
    "run_backtest",
    # Indicators (exposed from indicators module)
    "Indicator",
    "IndicatorManager",
    "SMA",
    "EMA",
    "BollingerBandsIndicator",
    "BollingerBands",
    "ATR",
    "RSI",
    "MACDIndicator",
    "MACD",
    "MACDResult",
    # Indicator helpers
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
