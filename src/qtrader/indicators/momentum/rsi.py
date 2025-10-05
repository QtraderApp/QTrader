"""Relative Strength Index indicator."""

from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from qtrader.indicators.base import Indicator

if TYPE_CHECKING:
    from qtrader.api.context import Context


class RSI(Indicator[float]):
    """
    Relative Strength Index (momentum oscillator).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss

    Returns value between 0 and 100.
    Values > 70 typically indicate overbought, < 30 indicate oversold.

    Args:
        period: Number of bars for RSI calculation (default 14)
    """

    def __init__(self, period: int = 14) -> None:
        super().__init__()
        self.period = period
        self.alpha = 1.0 / period
        self._prev_close: dict[str, float] = {}
        self._avg_gain: dict[str, float] = {}
        self._avg_loss: dict[str, float] = {}
        self._initialization: dict[str, list[float]] = defaultdict(list)

    def compute(self, symbol: str, ctx: "Context") -> Optional[float]:
        """Compute RSI for current bar."""
        bars = ctx.get_bar_history(symbol, 1)
        if not bars:
            return None

        bar = bars[-1]
        close = float(bar.close)

        # First bar: no change to compute
        if symbol not in self._prev_close:
            self._prev_close[symbol] = close
            return None

        # Calculate gain/loss
        prev_close = self._prev_close[symbol]
        change = close - prev_close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        self._prev_close[symbol] = close

        # Initialize with SMA of gains/losses
        if symbol not in self._avg_gain:
            init_changes = self._initialization[symbol]
            init_changes.append(change)

            if len(init_changes) < self.period:
                return None

            # First averages = SMA
            gains = [max(c, 0.0) for c in init_changes]
            losses = [max(-c, 0.0) for c in init_changes]
            self._avg_gain[symbol] = sum(gains) / self.period
            self._avg_loss[symbol] = sum(losses) / self.period
        else:
            # Smoothed moving average (Wilder's method)
            prev_avg_gain = self._avg_gain[symbol]
            prev_avg_loss = self._avg_loss[symbol]
            self._avg_gain[symbol] = prev_avg_gain + self.alpha * (gain - prev_avg_gain)
            self._avg_loss[symbol] = prev_avg_loss + self.alpha * (loss - prev_avg_loss)

        avg_gain = self._avg_gain[symbol]
        avg_loss = self._avg_loss[symbol]

        # Calculate RSI
        if avg_loss == 0.0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        return rsi

    def reset(self, symbol: str) -> None:
        """Clear cached state."""
        super().reset(symbol)
        self._prev_close.pop(symbol, None)
        self._avg_gain.pop(symbol, None)
        self._avg_loss.pop(symbol, None)
        self._initialization.pop(symbol, None)
