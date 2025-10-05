"""Simple Moving Average indicator."""

from collections import defaultdict, deque
from typing import TYPE_CHECKING, Optional

from qtrader.indicators.base import Indicator

if TYPE_CHECKING:
    from qtrader.api.context import Context


class SMA(Indicator[float]):
    """
    Simple Moving Average.

    Computes average of last N values of specified field.
    Returns None if insufficient data.

    Args:
        period: Number of bars to average
        field: Bar field to use ('open', 'high', 'low', 'close', 'volume')
    """

    def __init__(self, period: int, field: str = "close") -> None:
        super().__init__()
        self.period = period
        self.field = field
        # Rolling sum for O(1) updates
        self._rolling_sum: dict[str, float] = defaultdict(float)
        self._window: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=period))

    def compute(self, symbol: str, ctx: "Context") -> Optional[float]:
        """Compute SMA for current bar."""
        bars = ctx.get_bar_history(symbol, 1)
        if not bars:
            return None

        bar = bars[-1]
        value = float(getattr(bar, self.field))

        window = self._window[symbol]
        rolling_sum = self._rolling_sum[symbol]

        # Remove oldest value if window full
        if len(window) == self.period:
            rolling_sum -= window[0]

        # Add new value
        window.append(value)
        rolling_sum += value
        self._rolling_sum[symbol] = rolling_sum

        # Return average if sufficient data
        if len(window) < self.period:
            return None

        return rolling_sum / self.period

    def reset(self, symbol: str) -> None:
        """Clear cached state."""
        super().reset(symbol)
        self._rolling_sum.pop(symbol, None)
        self._window.pop(symbol, None)
