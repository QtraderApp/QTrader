"""
Bollinger Bands Indicator - Custom Implementation.

A volatility indicator that creates bands around a moving average.
The bands expand and contract based on market volatility (standard deviation).

Developed by John Bollinger in the 1980s, Bollinger Bands are one of the
most popular technical indicators for identifying overbought/oversold conditions.

Formula:
    Middle Band = SMA(close, period)
    Upper Band = Middle Band + (num_std * Standard Deviation)
    Lower Band = Middle Band - (num_std * Standard Deviation)

Typical settings: period=20, num_std=2.0

Use cases:
- Price touching upper band → potentially overbought
- Price touching lower band → potentially oversold
- Band squeeze (narrow bands) → low volatility, potential breakout
- Band expansion → high volatility
"""

from collections import deque
from typing import Any

from qtrader.contracts.data import Bar
from qtrader.libraries.indicators import SMA, BaseIndicator


class BollingerBands(BaseIndicator):
    """
    Bollinger Bands - Volatility indicator with three bands.

    Uses a Simple Moving Average (SMA) for the middle band and adds/subtracts
    standard deviation multiples for upper and lower bands.

    Parameters:
        period: Number of bars for SMA and standard deviation (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
        price_field: Which price to use (default: "close")

    Returns:
        Dictionary with three values:
        - upper: Upper band value
        - middle: Middle band (SMA) value
        - lower: Lower band value

    Example:
        >>> bb = BollingerBands(period=20, num_std=2.0)
        >>> for bar in bars:
        ...     bands = bb.update(bar)
        ...     if bands is not None:
        ...         print(f"Upper: {bands['upper']:.2f}")
        ...         print(f"Middle: {bands['middle']:.2f}")
        ...         print(f"Lower: {bands['lower']:.2f}")
        ...
        ...         # Trading logic
        ...         if bar.close > bands['upper']:
        ...             print("Price above upper band (overbought)")
        ...         elif bar.close < bands['lower']:
        ...             print("Price below lower band (oversold)")

    Notes:
        - Returns None during warmup period (first 'period' bars)
        - Standard settings: period=20, num_std=2.0
        - ~95% of prices should fall within 2 standard deviations
        - ~99.7% of prices should fall within 3 standard deviations
    """

    def __init__(self, period: int = 20, num_std: float = 2.0, price_field: str = "close", **params: Any):
        """
        Initialize Bollinger Bands indicator.

        Args:
            period: Number of bars for moving average and std dev
            num_std: Number of standard deviations for bands
            price_field: Which price field to use
            **params: Additional parameters (ignored)

        Raises:
            ValueError: If period < 2 or num_std <= 0
        """
        if period < 2:
            raise ValueError(f"Period must be >= 2, got {period}")
        if num_std <= 0:
            raise ValueError(f"num_std must be > 0, got {num_std}")

        self.period = period
        self.num_std = num_std
        self.price_field = price_field

        # Use built-in SMA for middle band
        self._sma = SMA(period=period, price_field=price_field)

        # Track prices for standard deviation calculation
        self._prices: deque[float] = deque(maxlen=period)

    def calculate(self, bars: list[Bar]) -> list[dict[str, float] | None]:
        """
        Calculate Bollinger Bands for all bars (stateless).

        Args:
            bars: List of price bars

        Returns:
            List of band dictionaries (None during warmup)
        """
        if not bars:
            return []

        prices = [getattr(bar, self.price_field) for bar in bars]
        result: list[dict[str, float] | None] = []

        for i in range(len(prices)):
            if i < self.period - 1:
                result.append(None)
            else:
                # Get window of prices
                window = prices[i - self.period + 1 : i + 1]

                # Calculate middle band (SMA)
                middle = sum(window) / self.period

                # Calculate standard deviation
                variance = sum((p - middle) ** 2 for p in window) / self.period
                std_dev = variance**0.5

                # Calculate bands
                upper = middle + (self.num_std * std_dev)
                lower = middle - (self.num_std * std_dev)

                result.append(
                    {
                        "upper": upper,
                        "middle": middle,
                        "lower": lower,
                    }
                )

        return result

    def update(self, bar: Bar) -> dict[str, float] | None:
        """
        Update Bollinger Bands with new bar (stateful).

        Args:
            bar: New price bar

        Returns:
            Dictionary with upper, middle, lower bands or None if not ready
        """
        price = getattr(bar, self.price_field)

        # Update SMA and prices
        self._sma.update(bar)
        self._prices.append(price)

        # Check if ready
        if len(self._prices) < self.period:
            return None

        # Get middle band from SMA
        middle = self._sma.value
        if middle is None:
            return None

        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in self._prices) / self.period
        std_dev = variance**0.5

        # Calculate bands
        upper = middle + (self.num_std * std_dev)
        lower = middle - (self.num_std * std_dev)

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        }

    def reset(self) -> None:
        """Reset indicator state."""
        self._sma.reset()
        self._prices.clear()

    @property
    def value(self) -> dict[str, float] | None:
        """Get current Bollinger Bands without updating."""
        if len(self._prices) < self.period:
            return None

        middle = self._sma.value
        if middle is None:
            return None

        # Calculate standard deviation
        variance = sum((p - middle) ** 2 for p in self._prices) / self.period
        std_dev = variance**0.5

        # Calculate bands
        upper = middle + (self.num_std * std_dev)
        lower = middle - (self.num_std * std_dev)

        return {
            "upper": upper,
            "middle": middle,
            "lower": lower,
        }

    @property
    def is_ready(self) -> bool:
        """Check if indicator has enough data."""
        return len(self._prices) >= self.period

    @property
    def bandwidth(self) -> float | None:
        """
        Calculate Bollinger Band Width.

        Bandwidth = (Upper Band - Lower Band) / Middle Band

        Useful for identifying:
        - Band squeeze: Low bandwidth → potential breakout
        - High volatility: High bandwidth

        Returns:
            Bandwidth as percentage or None if not ready
        """
        bands = self.value
        if bands is None:
            return None

        return (bands["upper"] - bands["lower"]) / bands["middle"]

    @property
    def percent_b(self) -> float | None:
        """
        Calculate %B indicator.

        %B = (Close - Lower Band) / (Upper Band - Lower Band)

        Interpretation:
        - %B > 1.0: Price above upper band
        - %B = 0.5: Price at middle band
        - %B < 0.0: Price below lower band

        Returns:
            %B value or None if not ready

        Note:
            Requires at least one update() call to have current price
        """
        if len(self._prices) == 0:
            return None

        bands = self.value
        if bands is None:
            return None

        current_price = self._prices[-1]
        band_range = bands["upper"] - bands["lower"]

        if band_range == 0:
            return 0.5  # Avoid division by zero

        return (current_price - bands["lower"]) / band_range
