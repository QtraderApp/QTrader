"""
Custom Indicator Library.

User-defined indicators that extend the base indicator framework.
All indicators inherit from BaseIndicator and can be used alongside built-in indicators.

Available Custom Indicators:
- BollingerBands: Volatility bands using SMA and standard deviation
"""

from my_library.indicators.bollinger_bands import BollingerBands

__all__ = [
    "BollingerBands",
]
