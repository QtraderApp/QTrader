"""Bollinger Bands indicator."""

from collections import defaultdict, deque
from typing import TYPE_CHECKING, NamedTuple, Optional

from qtrader.indicators.base import Indicator

if TYPE_CHECKING:
    from qtrader.api.context import Context


class BollingerBands(NamedTuple):
    """Bollinger Bands output (upper, middle, lower)."""

    upper: float
    middle: float
    lower: float


class BollingerBandsIndicator(Indicator[BollingerBands]):
    """
    Bollinger Bands.

    Computes upper and lower bands as:
    - Upper = SMA + (num_std * std_dev)
    - Middle = SMA
    - Lower = SMA - (num_std * std_dev)

    Args:
        period: Number of bars for SMA and standard deviation
        num_std: Number of standard deviations for bands (default 2.0)
        field: Bar field to use (default 'close')
    """

    def __init__(self, period: int, num_std: float = 2.0, field: str = "close") -> None:
        super().__init__()
        self.period = period
        self.num_std = num_std
        self.field = field
        self._window: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=period))

    def compute(self, symbol: str, ctx: "Context") -> Optional[BollingerBands]:
        """Compute Bollinger Bands for current bar."""
        bars = ctx.get_bar_history(symbol, 1)
        if not bars:
            return None

        bar = bars[-1]
        value = float(getattr(bar, self.field))

        window = self._window[symbol]
        window.append(value)

        if len(window) < self.period:
            return None

        # Calculate SMA and standard deviation
        values = list(window)
        sma = sum(values) / self.period
        variance = sum((x - sma) ** 2 for x in values) / self.period
        std_dev = variance**0.5

        upper = sma + (self.num_std * std_dev)
        lower = sma - (self.num_std * std_dev)

        return BollingerBands(upper=upper, middle=sma, lower=lower)

    def reset(self, symbol: str) -> None:
        """Clear cached state."""
        super().reset(symbol)
        self._window.pop(symbol, None)
