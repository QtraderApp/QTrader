"""Moving Average Convergence Divergence indicator."""

from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple, Optional

from qtrader.indicators.base import Indicator
from qtrader.indicators.trend.ema import EMA

if TYPE_CHECKING:
    from qtrader.api.context import Context


class MACDResult(NamedTuple):
    """MACD output (macd_line, signal_line, histogram)."""

    macd_line: float
    signal_line: float
    histogram: float


# Convenience alias
MACD = MACDResult


class MACDIndicator(Indicator[MACDResult]):
    """
    Moving Average Convergence Divergence.

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line

    Args:
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line EMA period (default 9)
        field: Bar field to use (default 'close')
    """

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, field: str = "close") -> None:
        super().__init__()
        self.fast = fast
        self.slow = slow
        self.signal_period = signal
        self.field = field

        # EMAs for MACD line
        self.fast_ema = EMA(fast, field)
        self.slow_ema = EMA(slow, field)

        # EMA for signal line (applied to MACD line)
        self.signal_alpha = 2.0 / (signal + 1)
        self._signal_ema: dict[str, float] = {}
        self._macd_initialization: dict[str, list[float]] = defaultdict(list)

    def compute(self, symbol: str, ctx: "Context") -> Optional[MACDResult]:
        """Compute MACD for current bar."""
        # Compute fast and slow EMAs
        fast_value = self.fast_ema.compute(symbol, ctx)
        slow_value = self.slow_ema.compute(symbol, ctx)

        if fast_value is None or slow_value is None:
            return None

        # MACD line = fast EMA - slow EMA
        macd_line = fast_value - slow_value

        # Initialize signal line with SMA of MACD values
        if symbol not in self._signal_ema:
            init_values = self._macd_initialization[symbol]
            init_values.append(macd_line)

            if len(init_values) < self.signal_period:
                return None

            # First signal = SMA of MACD values
            self._signal_ema[symbol] = sum(init_values) / self.signal_period
            signal_line = self._signal_ema[symbol]
        else:
            # Signal line = EMA of MACD line
            prev_signal = self._signal_ema[symbol]
            signal_line = self.signal_alpha * macd_line + (1 - self.signal_alpha) * prev_signal
            self._signal_ema[symbol] = signal_line

        histogram = macd_line - signal_line

        return MACDResult(macd_line=macd_line, signal_line=signal_line, histogram=histogram)

    def reset(self, symbol: str) -> None:
        """Clear cached state."""
        super().reset(symbol)
        self.fast_ema.reset(symbol)
        self.slow_ema.reset(symbol)
        self._signal_ema.pop(symbol, None)
        self._macd_initialization.pop(symbol, None)
