"""Average True Range indicator."""

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from qtrader.indicators.base import Indicator

if TYPE_CHECKING:
    from qtrader.api.context import Context


class ATR(Indicator[float]):
    """
    Average True Range (volatility measure).

    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
    ATR = EMA of True Range values

    Args:
        period: Number of bars for ATR calculation (default 14)
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period
        self.alpha = 1.0 / period
        self._atr_value: dict[str, float] = {}
        self._prev_close: dict[str, float] = {}
        self._initialization: dict[str, list[float]] = defaultdict(list)

    def compute(self, symbol: str, ctx: "Context") -> Optional[float]:
        """Compute ATR for current bar."""
        bars = ctx.get_bar_history(symbol, 1)
        if not bars:
            return None

        bar = bars[-1]
        high = float(bar.high)
        low = float(bar.low)
        close = float(bar.close)

        # First bar: TR = high - low
        if symbol not in self._prev_close:
            tr = high - low
            self._prev_close[symbol] = close
        else:
            # True Range = max(H-L, |H-PC|, |L-PC|)
            prev_close = self._prev_close[symbol]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            self._prev_close[symbol] = close

        # Initialize ATR with SMA of first period TRs
        if symbol not in self._atr_value:
            init_values = self._initialization[symbol]
            init_values.append(tr)

            if len(init_values) < self.period:
                return None

            # First ATR = average of TRs
            self._atr_value[symbol] = sum(init_values) / self.period
            return self._atr_value[symbol]

        # ATR = (prior_ATR * (period - 1) + TR) / period
        # Equivalent to EMA with alpha = 1/period
        prev_atr = self._atr_value[symbol]
        atr = prev_atr + self.alpha * (tr - prev_atr)
        self._atr_value[symbol] = atr

        return atr

    def reset(self, symbol: str) -> None:
        """Clear cached state."""
        super().reset(symbol)
        self._atr_value.pop(symbol, None)
        self._prev_close.pop(symbol, None)
        self._initialization.pop(symbol, None)
