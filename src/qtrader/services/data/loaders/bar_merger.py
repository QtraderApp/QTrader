"""
Bar merger for coordinating multi-symbol data streams.

Merges multiple price series iterators by timestamp, ensuring
chronological order across all symbols.
"""

from typing import Dict, Optional

from qtrader.config import LoggerFactory
from qtrader.contracts.data import MultiBar
from qtrader.services.data.loaders.iterator import PriceSeriesIterator

logger = LoggerFactory.get_logger()


class BarMerger:
    """
    Merge multiple symbol iterators by timestamp.

    Coordinates iteration across multiple symbols, yielding bars
    in chronological order regardless of source symbol.

    Examples:
        >>> iterators = {
        ...     "AAPL": aapl_iterator,
        ...     "MSFT": msft_iterator
        ... }
        >>> merger = BarMerger(iterators)
        >>> while merger.has_next():
        ...     symbol, bar = merger.get_next_bar()
        ...     print(f"{symbol}: {bar.trade_datetime}")

    Notes:
        - Bars are yielded in chronological order across all symbols
        - If multiple symbols have bars at same timestamp, order is deterministic
        - Handles symbols with different date ranges gracefully
        - Memory efficient (buffers only one bar per symbol)
    """

    def __init__(self, iterators: Dict[str, PriceSeriesIterator]):
        """
        Initialize bar merger with symbol iterators.

        Args:
            iterators: Dict mapping symbol -> PriceSeriesIterator

        Raises:
            ValueError: If iterators dict is empty
        """
        if not iterators:
            raise ValueError("BarMerger requires at least one iterator")

        self.iterators = iterators
        self.current_bars: Dict[str, MultiBar] = {}
        self._total_bars_yielded = 0

        # Prime all iterators (read first bar from each)
        self._prime_iterators()

        logger.info(
            "bar_merger.initialized",
            symbol_count=len(iterators),
            active_symbols=len(self.current_bars),
        )

    def _prime_iterators(self) -> None:
        """
        Read first bar from each iterator.

        Populates current_bars with initial bars from all symbols.
        Symbols that are empty (StopIteration) are excluded.
        """
        for symbol, iterator in self.iterators.items():
            try:
                self.current_bars[symbol] = next(iterator)
                logger.debug("bar_merger.symbol_primed", symbol=symbol)
            except StopIteration:
                logger.warning(
                    "bar_merger.symbol_empty",
                    symbol=symbol,
                    msg="Iterator has no bars",
                )

    def get_next_bar(self) -> tuple[str, MultiBar]:
        """
        Get next bar across all symbols (earliest timestamp).

        Returns:
            Tuple of (symbol, MultiBar) for earliest timestamp

        Raises:
            StopIteration: If all iterators exhausted

        Notes:
            - Advances the iterator for the symbol whose bar was returned
            - If multiple symbols have same timestamp, returns deterministically
              (sorted by symbol name for consistency)
        """
        if not self.current_bars:
            raise StopIteration("All symbols exhausted")

        # Find earliest timestamp
        # If tie, use symbol name for deterministic ordering
        earliest_symbol = min(self.current_bars.keys(), key=lambda s: (self.current_bars[s].trade_datetime, s))

        bar = self.current_bars[earliest_symbol]

        logger.debug(
            "bar_merger.yielding_bar",
            symbol=earliest_symbol,
            trade_datetime=bar.trade_datetime,
            active_symbols=len(self.current_bars),
        )

        # Advance that iterator
        try:
            self.current_bars[earliest_symbol] = next(self.iterators[earliest_symbol])
        except StopIteration:
            # Symbol exhausted, remove from active bars
            del self.current_bars[earliest_symbol]
            logger.debug(
                "bar_merger.symbol_exhausted",
                symbol=earliest_symbol,
                remaining_symbols=len(self.current_bars),
            )

        self._total_bars_yielded += 1
        return earliest_symbol, bar

    def has_next(self) -> bool:
        """
        Check if more bars available.

        Returns:
            True if at least one symbol has bars remaining
        """
        return len(self.current_bars) > 0

    def peek_next(self) -> Optional[tuple[str, MultiBar]]:
        """
        Peek at next bar without consuming it.

        Returns:
            Tuple of (symbol, MultiBar) for next bar, or None if exhausted

        Notes:
            - Does not advance any iterators
            - Multiple calls return same result until get_next_bar() is called
        """
        if not self.current_bars:
            return None

        # Find earliest timestamp (same logic as get_next_bar)
        earliest_symbol = min(self.current_bars.keys(), key=lambda s: (self.current_bars[s].trade_datetime, s))

        return earliest_symbol, self.current_bars[earliest_symbol]

    def get_stats(self) -> Dict:
        """
        Get merger statistics.

        Returns:
            Dict with merger statistics

        Example:
            >>> stats = merger.get_stats()
            >>> print(f"Yielded {stats['total_bars_yielded']} bars")
        """
        return {
            "total_symbols": len(self.iterators),
            "active_symbols": len(self.current_bars),
            "exhausted_symbols": len(self.iterators) - len(self.current_bars),
            "total_bars_yielded": self._total_bars_yielded,
        }
