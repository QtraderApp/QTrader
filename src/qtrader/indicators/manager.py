"""
Indicator manager for caching and convenience methods.

Provides high-level API for indicators with automatic caching per (symbol, indicator_type, params).
"""

from typing import TYPE_CHECKING, Optional

from qtrader.indicators.base import Indicator
from qtrader.indicators.momentum import RSI, MACDIndicator, MACDResult
from qtrader.indicators.trend import EMA, SMA
from qtrader.indicators.volatility import ATR, BollingerBands, BollingerBandsIndicator

if TYPE_CHECKING:
    from qtrader.api.context import Context


class IndicatorManager:
    """
    Manages indicator instances with automatic caching.

    Provides convenience methods for built-in indicators and
    supports custom indicator registration.

    Instances are cached per (indicator_type, params) to avoid
    redundant computation across symbols.
    """

    def __init__(self, context: "Context") -> None:
        """
        Initialize indicator manager.

        Args:
            context: Context object for accessing bar history
        """
        self.context = context
        self._indicators: dict[str, Indicator] = {}
        self._custom_indicators: dict[str, Indicator] = {}

    def sma(self, symbol: str, period: int, field: str = "close") -> Optional[float]:
        """
        Simple Moving Average.

        Args:
            symbol: Symbol to compute for
            period: Number of bars to average
            field: Bar field ('open', 'high', 'low', 'close', 'volume')

        Returns:
            SMA value or None if insufficient data
        """
        key = f"sma_{period}_{field}"
        if key not in self._indicators:
            self._indicators[key] = SMA(period, field)
        return self._indicators[key].compute(symbol, self.context)

    def ema(self, symbol: str, period: int, field: str = "close") -> Optional[float]:
        """
        Exponential Moving Average.

        Args:
            symbol: Symbol to compute for
            period: Number of bars for EMA
            field: Bar field ('open', 'high', 'low', 'close', 'volume')

        Returns:
            EMA value or None if insufficient data
        """
        key = f"ema_{period}_{field}"
        if key not in self._indicators:
            self._indicators[key] = EMA(period, field)
        return self._indicators[key].compute(symbol, self.context)

    def bollinger_bands(
        self, symbol: str, period: int, num_std: float = 2.0, field: str = "close"
    ) -> Optional[BollingerBands]:
        """
        Bollinger Bands.

        Args:
            symbol: Symbol to compute for
            period: Number of bars for SMA and std dev
            num_std: Number of standard deviations (default 2.0)
            field: Bar field (default 'close')

        Returns:
            BollingerBands(upper, middle, lower) or None if insufficient data
        """
        key = f"bb_{period}_{num_std}_{field}"
        if key not in self._indicators:
            self._indicators[key] = BollingerBandsIndicator(period, num_std, field)
        return self._indicators[key].compute(symbol, self.context)

    def atr(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Average True Range (volatility).

        Args:
            symbol: Symbol to compute for
            period: Number of bars for ATR (default 14)

        Returns:
            ATR value or None if insufficient data
        """
        key = f"atr_{period}"
        if key not in self._indicators:
            self._indicators[key] = ATR(period)
        return self._indicators[key].compute(symbol, self.context)

    def rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """
        Relative Strength Index (momentum).

        Args:
            symbol: Symbol to compute for
            period: Number of bars for RSI (default 14)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        key = f"rsi_{period}"
        if key not in self._indicators:
            self._indicators[key] = RSI(period)
        return self._indicators[key].compute(symbol, self.context)

    def macd(
        self,
        symbol: str,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        field: str = "close",
    ) -> Optional[MACDResult]:
        """
        Moving Average Convergence Divergence.

        Args:
            symbol: Symbol to compute for
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)
            field: Bar field (default 'close')

        Returns:
            MACDResult(macd_line, signal_line, histogram) or None if insufficient data
        """
        key = f"macd_{fast}_{slow}_{signal}_{field}"
        if key not in self._indicators:
            self._indicators[key] = MACDIndicator(fast, slow, signal, field)
        return self._indicators[key].compute(symbol, self.context)

    def register(self, name: str, indicator: Indicator) -> None:
        """
        Register custom indicator.

        Args:
            name: Name to register indicator under
            indicator: Indicator instance

        Example:
            manager.register("momentum", CustomMomentum(period=20))
        """
        self._custom_indicators[name] = indicator

    def get(self, name: str, symbol: str) -> Optional[float]:
        """
        Get custom indicator value.

        Args:
            name: Registered indicator name
            symbol: Symbol to compute for

        Returns:
            Indicator value or None if insufficient data

        Raises:
            KeyError: If indicator not registered
        """
        if name not in self._custom_indicators:
            raise KeyError(f"Indicator '{name}' not registered")
        return self._custom_indicators[name].compute(symbol, self.context)

    def reset(self, symbol: str) -> None:
        """
        Reset all indicator caches for symbol.

        Args:
            symbol: Symbol to reset
        """
        for indicator in self._indicators.values():
            indicator.reset(symbol)
        for indicator in self._custom_indicators.values():
            indicator.reset(symbol)

    def get_max_lookback(self) -> int:
        """
        Get maximum lookback period across all registered indicators.

        Used by warmup system to determine how many bars to process
        before starting strategy execution.

        Returns:
            Maximum lookback period (0 if no indicators)
        """
        max_lookback = 0

        # Check built-in indicators
        for key in self._indicators:
            parts = key.split("_")
            if parts[0] in ("sma", "ema", "bb", "atr", "rsi"):
                # Extract period from key (e.g., "sma_20_close" -> 20)
                try:
                    period = int(parts[1])
                    max_lookback = max(max_lookback, period)
                except (IndexError, ValueError):
                    pass
            elif parts[0] == "macd":
                # MACD needs slow period (e.g., "macd_12_26_9_close" -> 26)
                try:
                    slow = int(parts[2])
                    signal = int(parts[3])
                    # MACD needs slow + signal periods for full computation
                    max_lookback = max(max_lookback, slow + signal)
                except (IndexError, ValueError):
                    pass

        # Check custom indicators (if they have period attribute)
        for indicator in self._custom_indicators.values():
            if hasattr(indicator, "period"):
                period = getattr(indicator, "period")
                if isinstance(period, int):
                    max_lookback = max(max_lookback, period)

        return max_lookback
