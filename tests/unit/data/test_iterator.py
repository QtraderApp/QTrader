"""Tests for PriceSeriesIterator."""

import pytest

from qtrader.data.iterator import PriceSeriesIterator
from qtrader.models.canonical_bar import CanonicalBar, CanonicalPriceSeries
from qtrader.models.multi_mode_bar import MultiModeBar


@pytest.fixture
def sample_canonical_series_dict():
    """Create sample canonical series for all modes."""
    # Create 3 bars for each mode
    unadj_bars = [
        CanonicalBar(
            trade_datetime="2020-01-01T00:00:00",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000000,
        ),
        CanonicalBar(
            trade_datetime="2020-01-02T00:00:00",
            open=104.0,
            high=108.0,
            low=103.0,
            close=107.0,
            volume=1100000,
        ),
        CanonicalBar(
            trade_datetime="2020-01-03T00:00:00",
            open=107.0,
            high=110.0,
            low=106.0,
            close=109.0,
            volume=1200000,
        ),
    ]

    adj_bars = [
        CanonicalBar(
            trade_datetime="2020-01-01T00:00:00",
            open=25.0,
            high=26.25,
            low=24.75,
            close=26.0,
            volume=1000000,
        ),
        CanonicalBar(
            trade_datetime="2020-01-02T00:00:00",
            open=26.0,
            high=27.0,
            low=25.75,
            close=26.75,
            volume=1100000,
        ),
        CanonicalBar(
            trade_datetime="2020-01-03T00:00:00",
            open=26.75,
            high=27.5,
            low=26.5,
            close=27.25,
            volume=1200000,
        ),
    ]

    tr_bars = [
        CanonicalBar(
            trade_datetime="2020-01-01T00:00:00",
            open=25.5,
            high=26.75,
            low=25.25,
            close=26.5,
            volume=1000000,
        ),
        CanonicalBar(
            trade_datetime="2020-01-02T00:00:00",
            open=26.5,
            high=27.5,
            low=26.25,
            close=27.25,
            volume=1100000,
        ),
        CanonicalBar(
            trade_datetime="2020-01-03T00:00:00",
            open=27.25,
            high=28.0,
            low=27.0,
            close=27.75,
            volume=1200000,
        ),
    ]

    return {
        "unadjusted": CanonicalPriceSeries(symbol="AAPL", mode="unadjusted", bars=unadj_bars),
        "adjusted": CanonicalPriceSeries(symbol="AAPL", mode="adjusted", bars=adj_bars),
        "total_return": CanonicalPriceSeries(symbol="AAPL", mode="total_return", bars=tr_bars),
    }


class TestPriceSeriesIteratorCreation:
    """Test iterator creation and validation."""

    def test_create_iterator(self, sample_canonical_series_dict):
        """Test creating a valid iterator."""
        # Act
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Assert
        assert iterator.symbol == "AAPL"
        assert iterator._index == 0
        assert iterator._peeked is None

    def test_create_iterator_missing_mode(self):
        """Test creating iterator with missing mode raises error."""
        # Arrange: Only two modes
        incomplete_dict = {
            "unadjusted": CanonicalPriceSeries(symbol="AAPL", mode="unadjusted", bars=[]),
            "adjusted": CanonicalPriceSeries(symbol="AAPL", mode="adjusted", bars=[]),
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Missing required series"):
            PriceSeriesIterator(incomplete_dict)

    def test_create_iterator_length_mismatch(self):
        """Test creating iterator with mismatched lengths raises error."""
        # Arrange: Different lengths
        bar1 = CanonicalBar(
            trade_datetime="2020-01-01T00:00:00",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000,
        )
        bar2 = CanonicalBar(
            trade_datetime="2020-01-02T00:00:00",
            open=104.0,
            high=108.0,
            low=103.0,
            close=107.0,
            volume=1100,
        )

        mismatched_dict = {
            "unadjusted": CanonicalPriceSeries(symbol="AAPL", mode="unadjusted", bars=[bar1, bar2]),
            "adjusted": CanonicalPriceSeries(symbol="AAPL", mode="adjusted", bars=[bar1]),  # Only 1 bar
            "total_return": CanonicalPriceSeries(symbol="AAPL", mode="total_return", bars=[bar1, bar2]),
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Series length mismatch"):
            PriceSeriesIterator(mismatched_dict)

    def test_create_iterator_symbol_mismatch(self):
        """Test creating iterator with mismatched symbols raises error."""
        # Arrange: Different symbols
        bar = CanonicalBar(
            trade_datetime="2020-01-01T00:00:00",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000,
        )

        mismatched_dict = {
            "unadjusted": CanonicalPriceSeries(symbol="AAPL", mode="unadjusted", bars=[bar]),
            "adjusted": CanonicalPriceSeries(symbol="MSFT", mode="adjusted", bars=[bar]),  # Wrong symbol
            "total_return": CanonicalPriceSeries(symbol="AAPL", mode="total_return", bars=[bar]),
        }

        # Act & Assert
        with pytest.raises(ValueError, match="Symbol mismatch"):
            PriceSeriesIterator(mismatched_dict)


class TestPriceSeriesIteratorIteration:
    """Test basic iteration functionality."""

    def test_iterate_all_bars(self, sample_canonical_series_dict):
        """Test iterating through all bars."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Act: Collect all bars
        bars = list(iterator)

        # Assert
        assert len(bars) == 3
        assert all(isinstance(bar, MultiModeBar) for bar in bars)
        assert bars[0].trade_datetime == "2020-01-01T00:00:00"
        assert bars[1].trade_datetime == "2020-01-02T00:00:00"
        assert bars[2].trade_datetime == "2020-01-03T00:00:00"

    def test_iterate_yields_multi_mode_bars(self, sample_canonical_series_dict):
        """Test that iteration yields MultiModeBar instances."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Act
        first_bar = next(iterator)

        # Assert: MultiModeBar with all three modes
        assert isinstance(first_bar, MultiModeBar)
        assert first_bar.symbol == "AAPL"
        assert first_bar.unadjusted.close == 104.0
        assert first_bar.adjusted.close == 26.0
        assert first_bar.total_return.close == 26.5

    def test_iterate_stop_iteration(self, sample_canonical_series_dict):
        """Test StopIteration raised when exhausted."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        list(iterator)  # Exhaust iterator

        # Act & Assert
        with pytest.raises(StopIteration):
            next(iterator)

    def test_iterate_empty_series(self):
        """Test iterating empty series."""
        # Arrange: Empty series
        empty_dict = {
            "unadjusted": CanonicalPriceSeries(symbol="AAPL", mode="unadjusted", bars=[]),
            "adjusted": CanonicalPriceSeries(symbol="AAPL", mode="adjusted", bars=[]),
            "total_return": CanonicalPriceSeries(symbol="AAPL", mode="total_return", bars=[]),
        }
        iterator = PriceSeriesIterator(empty_dict)

        # Act
        bars = list(iterator)

        # Assert
        assert len(bars) == 0


class TestPriceSeriesIteratorPeek:
    """Test peek functionality."""

    def test_peek_returns_next_bar(self, sample_canonical_series_dict):
        """Test peek returns next bar without consuming."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Act
        peeked_bar = iterator.peek()

        # Assert
        assert peeked_bar is not None
        assert peeked_bar.trade_datetime == "2020-01-01T00:00:00"
        assert iterator._index == 0  # Index not advanced yet (peek caches it internally)

    def test_peek_then_next_returns_same_bar(self, sample_canonical_series_dict):
        """Test that next() after peek() returns the peeked bar."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Act
        peeked = iterator.peek()
        next_bar = next(iterator)

        # Assert: Same bar
        assert peeked is not None
        assert peeked is next_bar
        assert peeked.trade_datetime == "2020-01-01T00:00:00"

    def test_peek_multiple_times_returns_same_bar(self, sample_canonical_series_dict):
        """Test multiple peeks return same bar."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Act
        peek1 = iterator.peek()
        peek2 = iterator.peek()
        peek3 = iterator.peek()

        # Assert: All same
        assert peek1 is not None
        assert peek1 is peek2 is peek3
        assert peek1.trade_datetime == "2020-01-01T00:00:00"

    def test_peek_at_end_returns_none(self, sample_canonical_series_dict):
        """Test peek at end of series returns None."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        list(iterator)  # Exhaust

        # Act
        peeked = iterator.peek()

        # Assert
        assert peeked is None

    def test_peek_after_next(self, sample_canonical_series_dict):
        """Test peek after consuming a bar."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Act
        first = next(iterator)
        peeked = iterator.peek()

        # Assert: Peek shows second bar
        assert first.trade_datetime == "2020-01-01T00:00:00"
        assert peeked is not None
        assert peeked.trade_datetime == "2020-01-02T00:00:00"


class TestPriceSeriesIteratorHelpers:
    """Test helper methods."""

    def test_has_next_true_initially(self, sample_canonical_series_dict):
        """Test has_next returns True initially."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Assert
        assert iterator.has_next() is True

    def test_has_next_false_when_exhausted(self, sample_canonical_series_dict):
        """Test has_next returns False when exhausted."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        list(iterator)  # Exhaust

        # Assert
        assert iterator.has_next() is False

    def test_has_next_with_peek(self, sample_canonical_series_dict):
        """Test has_next with peeked bar."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        iterator.peek()

        # Assert: Still has next (the peeked bar)
        assert iterator.has_next() is True

    def test_reset(self, sample_canonical_series_dict):
        """Test reset returns iterator to beginning."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        first_pass = list(iterator)

        # Act: Reset and iterate again
        iterator.reset()
        second_pass = list(iterator)

        # Assert: Same bars
        assert len(first_pass) == len(second_pass) == 3
        assert first_pass[0].trade_datetime == second_pass[0].trade_datetime

    def test_reset_clears_peek(self, sample_canonical_series_dict):
        """Test reset clears peeked value."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        iterator.peek()
        assert iterator._peeked is not None

        # Act
        iterator.reset()

        # Assert
        assert iterator._peeked is None
        assert iterator._index == 0

    def test_len(self, sample_canonical_series_dict):
        """Test __len__ returns total bars."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Assert
        assert len(iterator) == 3


class TestPriceSeriesIteratorUseCases:
    """Test real-world usage patterns."""

    def test_strategy_warmup_pattern(self, sample_canonical_series_dict):
        """Test strategy warmup pattern using peek."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        warmup_period = 2
        bars_seen = 0

        # Act: Peek ahead for warmup
        while bars_seen < warmup_period and iterator.has_next():
            peeked = iterator.peek()
            if peeked:
                bars_seen += 1
                next(iterator)  # Consume after peek

        # Now process remaining bars
        remaining = list(iterator)

        # Assert
        assert bars_seen == 2
        assert len(remaining) == 1  # One bar left

    def test_conditional_entry_pattern(self, sample_canonical_series_dict):
        """Test conditional entry pattern using peek."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)
        threshold = 27.0

        # Act: Only process bars where NEXT bar is above threshold
        processed = []
        for bar in iterator:
            # Peek at next before processing
            next_bar = iterator.peek()
            if next_bar and next_bar.adjusted.close > threshold:
                processed.append(bar)

        # Assert: Only bar 2 processed (next bar 3 has close=27.25 > 27.0)
        # Bar 1: next is 26.75 (< 27.0) → skip
        # Bar 2: next is 27.25 (> 27.0) → process
        # Bar 3: no next → skip
        assert len(processed) == 1
        assert processed[0].trade_datetime == "2020-01-02T00:00:00"

    def test_multi_component_mode_selection(self, sample_canonical_series_dict):
        """Test different components selecting different modes."""
        # Arrange
        iterator = PriceSeriesIterator(sample_canonical_series_dict)

        # Simulate component configs
        signal_mode = "adjusted"
        exec_mode = "unadjusted"
        perf_mode = "total_return"

        # Act: Process bars with different modes
        results = []
        for multi_bar in iterator:
            # Each component selects its mode
            signal_bar = multi_bar.get_bar(signal_mode)
            exec_bar = multi_bar.get_bar(exec_mode)
            perf_bar = multi_bar.get_bar(perf_mode)

            results.append(
                {
                    "signal_close": signal_bar.close,
                    "exec_close": exec_bar.close,
                    "perf_close": perf_bar.close,
                }
            )

        # Assert: Each component got different prices
        assert len(results) == 3
        assert results[0]["signal_close"] == 26.0  # adjusted
        assert results[0]["exec_close"] == 104.0  # unadjusted
        assert results[0]["perf_close"] == 26.5  # total_return
