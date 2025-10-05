"""Base indicator class for all technical indicators."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import TYPE_CHECKING, Generic, Optional, TypeVar

if TYPE_CHECKING:
    from qtrader.api.context import Context

T = TypeVar("T")


class Indicator(ABC, Generic[T]):
    """
    Base class for all indicators (built-in and custom).

    Subclasses must implement compute() method.
    Indicators automatically cache computed values per (symbol, timestamp).

    Example:
        class MyIndicator(Indicator[float]):
            def __init__(self, period: int):
                super().__init__()
                self.period = period

            def compute(self, symbol: str, ctx: Context) -> float | None:
                bars = ctx.get_bar_history(symbol, self.period)
                if len(bars) < self.period:
                    return None
                return sum(float(b.close) for b in bars) / self.period
    """

    def __init__(self) -> None:
        """Initialize indicator with empty cache."""
        self._cache: dict[tuple[str, int], T] = {}  # (symbol, bar_index) -> value
        self._bar_counter: dict[str, int] = defaultdict(int)

    @abstractmethod
    def compute(self, symbol: str, ctx: "Context") -> Optional[T]:
        """
        Compute indicator value for current bar.

        Args:
            symbol: Symbol to compute indicator for
            ctx: Context object providing bar history access

        Returns:
            Indicator value or None if insufficient data
        """
        ...

    def warmup(self, symbol: str, ctx: "Context") -> None:
        """
        Optional warmup phase before trading starts.

        Called during warmup period to pre-compute values.
        Default implementation does nothing (compute handles it).

        Args:
            symbol: Symbol to warmup for
            ctx: Context object
        """
        pass

    def reset(self, symbol: str) -> None:
        """
        Clear cached state for symbol.

        Called when restarting backtest or switching datasets.

        Args:
            symbol: Symbol to reset cache for
        """
        keys_to_remove = [k for k in self._cache.keys() if k[0] == symbol]
        for key in keys_to_remove:
            del self._cache[key]
        self._bar_counter.pop(symbol, None)

    def _get_cache_key(self, symbol: str, ctx: "Context") -> tuple[str, int]:
        """Get cache key for current bar."""
        bar_index = self._bar_counter[symbol]
        return (symbol, bar_index)

    def _increment_bar(self, symbol: str) -> None:
        """Increment bar counter for symbol."""
        self._bar_counter[symbol] += 1
