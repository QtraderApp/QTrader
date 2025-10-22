"""
Price series iterator for streaming multi-mode bars.

This module provides the PriceSeriesIterator class which wraps canonical price
series dictionaries and yields MultiBar instances for streaming data access.
The iterator supports peek-ahead functionality for strategy warmup.
"""

from typing import Dict, Iterator, Optional

from qtrader.contracts.data import MultiBar, PriceSeries


class PriceSeriesIterator:
    """
    Iterator wrapper for multi-mode price series.

    Provides streaming access to bars with all adjustment modes available
    simultaneously. Supports peek (look at next bar without consuming) for
    strategy warmup and lookahead functionality.

    The iterator yields MultiBar instances, each containing the same bar
    data in all three adjustment modes (unadjusted, adjusted, total_return).

    Attributes:
        series_dict: Dictionary of canonical series by mode
        symbol: Ticker symbol for all series
        _index: Current iteration position
        _peeked: Cached peek result

    Examples:
        >>> # Create iterator from canonical series
        >>> canonical = algoseek_series.to_canonical_series()
        >>> iterator = PriceSeriesIterator(canonical)
        >>>
        >>> # Stream bars
        >>> for multi_bar in iterator:
        ...     strategy_bar = multi_bar.adjusted
        ...     exec_bar = multi_bar.unadjusted
        ...
        >>> # Peek ahead without consuming
        >>> next_bar = iterator.peek()
        >>> if next_bar and next_bar.adjusted.close > threshold:
        ...     current_bar = next(iterator)
    """

    def __init__(self, series_dict: Dict[str, PriceSeries]) -> None:
        """
        Initialize iterator.

        Args:
            series_dict: Dict with keys 'unadjusted', 'adjusted', 'total_return',
                        each mapping to a PriceSeries

        Raises:
            ValueError: If required keys are missing or series lengths don't match

        Notes:
            - All three series must have the same length
            - All three series must have matching trade_datetime values
        """
        # Validate required keys
        required_keys = {"unadjusted", "adjusted", "total_return"}
        if not required_keys.issubset(series_dict.keys()):
            missing = required_keys - set(series_dict.keys())
            raise ValueError(f"Missing required series: {missing}")

        # Validate all series have same length
        lengths = {mode: len(series.bars) for mode, series in series_dict.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"Series length mismatch: {lengths}")

        # Validate all series have same symbol
        symbols = {mode: series.symbol for mode, series in series_dict.items()}
        if len(set(symbols.values())) > 1:
            raise ValueError(f"Symbol mismatch: {symbols}")

        self.series_dict = series_dict
        self.symbol = series_dict["unadjusted"].symbol
        self._index = 0
        self._peeked: Optional[MultiBar] = None

    def __iter__(self) -> Iterator[MultiBar]:
        """Return iterator."""
        return self

    def __next__(self) -> MultiBar:
        """
        Get next multi-mode bar.

        Returns:
            MultiBar containing all three adjustment modes

        Raises:
            StopIteration: When no more bars available
        """
        # If we peeked, return peeked bar and advance index
        if self._peeked is not None:
            bar = self._peeked
            self._peeked = None
            self._index += 1  # FIX: Must increment index when returning peeked bar
            return bar

        # Otherwise get next from series
        unadj_bars = self.series_dict["unadjusted"].bars
        if self._index >= len(unadj_bars):
            raise StopIteration

        # Build MultiBar from all three series
        multi_bar = MultiBar(
            symbol=self.symbol,
            trade_datetime=unadj_bars[self._index].trade_datetime.isoformat(),
            unadjusted=self.series_dict["unadjusted"].bars[self._index],
            adjusted=self.series_dict["adjusted"].bars[self._index],
            total_return=self.series_dict["total_return"].bars[self._index],
        )

        self._index += 1
        return multi_bar

    def peek(self) -> Optional[MultiBar]:
        """
        Peek at next bar without consuming.

        Useful for lookahead in strategies (e.g., warmup periods, conditional entry).
        The peeked bar will be returned by the next call to __next__().

        Returns:
            Next MultiBar or None if at end

        Examples:
            >>> # Check next bar before consuming
            >>> next_bar = iterator.peek()
            >>> if next_bar and next_bar.adjusted.close > sma:
            ...     current = next(iterator)  # Returns the peeked bar
        """
        if self._peeked is not None:
            return self._peeked

        unadj_bars = self.series_dict["unadjusted"].bars
        if self._index >= len(unadj_bars):
            return None

        self._peeked = MultiBar(
            symbol=self.symbol,
            trade_datetime=unadj_bars[self._index].trade_datetime.isoformat(),
            unadjusted=self.series_dict["unadjusted"].bars[self._index],
            adjusted=self.series_dict["adjusted"].bars[self._index],
            total_return=self.series_dict["total_return"].bars[self._index],
        )
        return self._peeked

    def has_next(self) -> bool:
        """
        Check if more bars available.

        Returns:
            True if more bars available, False otherwise

        Examples:
            >>> while iterator.has_next():
            ...     bar = next(iterator)
            ...     process(bar)
        """
        unadj_bars = self.series_dict["unadjusted"].bars
        return self._peeked is not None or self._index < len(unadj_bars)

    def reset(self) -> None:
        """
        Reset iterator to beginning.

        Clears any peeked value and resets index to 0.

        Examples:
            >>> # First pass
            >>> for bar in iterator:
            ...     analyze(bar)
            >>>
            >>> # Second pass (reset required)
            >>> iterator.reset()
            >>> for bar in iterator:
            ...     backtest(bar)
        """
        self._index = 0
        self._peeked = None

    def __len__(self) -> int:
        """
        Get total number of bars.

        Returns:
            Total number of bars in series
        """
        return len(self.series_dict["unadjusted"].bars)
