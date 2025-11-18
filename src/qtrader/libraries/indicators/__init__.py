"""
Indicators Library.

Technical indicators for quantitative analysis:
- Moving Averages: SMA, EMA, WMA, DEMA, TEMA, HMA, SMMA
- Momentum: RSI, MACD, Stochastic, CCI, ROC, Williams %R
- Volatility: ATR, Bollinger Bands, Standard Deviation
- Volume: VWAP, OBV, A/D, CMF
- Trend: ADX, Aroon
"""

from qtrader.libraries.indicators.base import BaseIndicator
from qtrader.libraries.indicators.buildin.momentum import CCI, MACD, ROC, RSI, Stochastic, WilliamsR
from qtrader.libraries.indicators.buildin.moving_averages import DEMA, EMA, HMA, SMA, SMMA, TEMA, WMA
from qtrader.libraries.indicators.buildin.trend import ADX, Aroon
from qtrader.libraries.indicators.buildin.volatility import ATR, BollingerBands, StdDev
from qtrader.libraries.indicators.buildin.volume import AD, CMF, OBV, VWAP

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
    # Momentum
    "RSI",
    "MACD",
    "Stochastic",
    "CCI",
    "ROC",
    "WilliamsR",
    # Volatility
    "ATR",
    "BollingerBands",
    "StdDev",
    # Volume
    "VWAP",
    "OBV",
    "AD",
    "CMF",
    # Trend
    "ADX",
    "Aroon",
]
