"""Exponential Moving Average indicator."""

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from qtrader.indicators.base import Indicator

if TYPE_CHECKING:
    from qtrader.api.context import Context


class EMA(Indicator[float]):
    """
    Exponential Moving Average.

    Uses exponential weighting with smoothing factor alpha = 2 / (period + 1).
    First EMA value is initialized with SMA.

    Args:
        period: Number of bars for EMA calculation
        field: Bar field to use ('open', 'high', 'low', 'close', 'volume')
    """

    def __init__(self, period: int, field: str = "close") -> None:
        super().__init__()
        self.period = period
        self.field = field
        self.alpha = 2.0 / (period + 1)
        self._ema_value: dict[str, float] = {}
        self._initialization: dict[str, list[float]] = defaultdict(list)

    def compute(self, symbol: str, ctx: "Context") -> Optional[float]:
        """Compute EMA for current bar."""
        bars = ctx.get_bar_history(symbol, 1)
        if not bars:
            return None

        bar = bars[-1]
        value = float(getattr(bar, self.field))

        # Initialize EMA with SMA of first period bars
        if symbol not in self._ema_value:
            init_values = self._initialization[symbol]
            init_values.append(value)

            if len(init_values) < self.period:
                return None

            # First EMA = SMA
            self._ema_value[symbol] = sum(init_values) / self.period
            return self._ema_value[symbol]

        # EMA = alpha * current + (1 - alpha) * previous_ema
        prev_ema = self._ema_value[symbol]
        ema = self.alpha * value + (1 - self.alpha) * prev_ema
        self._ema_value[symbol] = ema

        return ema

    def reset(self, symbol: str) -> None:
        """Clear cached state."""
        super().reset(symbol)
        self._ema_value.pop(symbol, None)
        self._initialization.pop(symbol, None)
