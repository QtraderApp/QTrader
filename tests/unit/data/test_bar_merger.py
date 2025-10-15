"""Tests for BarMerger multi-symbol coordinator."""

from datetime import datetime

import pytest

from qtrader.data.bar_merger import BarMerger
from qtrader.data.iterator import PriceSeriesIterator
from qtrader.models.bar import Bar, PriceSeries
from qtrader.models.multi_bar import MultiBar


@pytest.fixture
def aapl_bars():
    """Create AAPL test bars."""
    return [
        Bar(
            trade_datetime=datetime(2020, 1, 2),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        ),
        Bar(
            trade_datetime=datetime(2020, 1, 3),
            open=103.0,
            high=107.0,
            low=102.0,
            close=106.0,
            volume=1100000,
        ),
        Bar(
            trade_datetime=datetime(2020, 1, 6),
            open=106.0,
            high=108.0,
            low=105.0,
            close=107.0,
            volume=1200000,
        ),
    ]


@pytest.fixture
def msft_bars():
    """Create MSFT test bars."""
    return [
        Bar(
            trade_datetime=datetime(2020, 1, 2),
            open=200.0,
            high=205.0,
            low=198.0,
            close=203.0,
            volume=2000000,
        ),
        Bar(
            trade_datetime=datetime(2020, 1, 3),
            open=203.0,
            high=207.0,
            low=201.0,
            close=205.0,
            volume=2100000,
        ),
        Bar(
            trade_datetime=datetime(2020, 1, 7),
            open=205.0,
            high=210.0,
            low=204.0,
            close=208.0,
            volume=2200000,
        ),
    ]


@pytest.fixture
def googl_bars():
    """Create GOOGL test bars (starts later)."""
    return [
        Bar(
            trade_datetime=datetime(2020, 1, 3),
            open=1300.0,
            high=1320.0,
            low=1295.0,
            close=1315.0,
            volume=500000,
        ),
        Bar(
            trade_datetime=datetime(2020, 1, 6),
            open=1315.0,
            high=1330.0,
            low=1310.0,
            close=1325.0,
            volume=550000,
        ),
    ]


def create_iterator(bars, symbol="AAPL"):
    """Helper to create price series iterator."""
    # Create 3 separate series for each mode
    unadj_series = PriceSeries(symbol=symbol, mode="unadjusted", bars=bars)
    adj_series = PriceSeries(symbol=symbol, mode="adjusted", bars=bars)
    total_series = PriceSeries(symbol=symbol, mode="total_return", bars=bars)

    # Create series dict for all 3 modes
    series_dict = {
        "unadjusted": unadj_series,
        "adjusted": adj_series,
        "total_return": total_series,
    }
    return PriceSeriesIterator(series_dict)


class TestBarMergerInitialization:
    """Test BarMerger initialization."""

    def test_create_merger_single_symbol(self, aapl_bars):
        """Test creating merger with single symbol."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        assert merger.has_next()
        assert len(merger.current_bars) == 1
        assert "AAPL" in merger.current_bars

    def test_create_merger_multiple_symbols(self, aapl_bars, msft_bars):
        """Test creating merger with multiple symbols."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        msft_iter = create_iterator(msft_bars, "MSFT")

        merger = BarMerger({"AAPL": aapl_iter, "MSFT": msft_iter})

        assert merger.has_next()
        assert len(merger.current_bars) == 2
        assert "AAPL" in merger.current_bars
        assert "MSFT" in merger.current_bars

    def test_create_merger_empty_raises(self):
        """Test creating merger with no iterators raises error."""
        with pytest.raises(ValueError, match="requires at least one iterator"):
            BarMerger({})

    def test_create_merger_with_empty_iterator(self, aapl_bars):
        """Test creating merger where one iterator is empty."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        empty_iter = create_iterator([], "EMPTY")

        merger = BarMerger({"AAPL": aapl_iter, "EMPTY": empty_iter})

        # Only AAPL should be active (EMPTY has no bars)
        assert merger.has_next()
        assert len(merger.current_bars) == 1
        assert "AAPL" in merger.current_bars
        assert "EMPTY" not in merger.current_bars


class TestBarMergerChronologicalOrder:
    """Test BarMerger yields bars in chronological order."""

    def test_get_next_bar_chronological(self, aapl_bars, msft_bars):
        """Test bars are yielded in chronological order."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        msft_iter = create_iterator(msft_bars, "MSFT")

        merger = BarMerger({"AAPL": aapl_iter, "MSFT": msft_iter})

        # Both have 2020-01-02, should return AAPL first (alphabetical)
        symbol1, bar1 = merger.get_next_bar()
        assert symbol1 == "AAPL"
        assert bar1.trade_datetime == datetime(2020, 1, 2, 0, 0, 0)

        # Next should be MSFT 2020-01-02
        symbol2, bar2 = merger.get_next_bar()
        assert symbol2 == "MSFT"
        assert bar2.trade_datetime == datetime(2020, 1, 2, 0, 0, 0)

        # Both have 2020-01-03, should return AAPL first
        symbol3, bar3 = merger.get_next_bar()
        assert symbol3 == "AAPL"
        assert bar3.trade_datetime == datetime(2020, 1, 3, 0, 0, 0)

        # Next should be MSFT 2020-01-03
        symbol4, bar4 = merger.get_next_bar()
        assert symbol4 == "MSFT"
        assert bar4.trade_datetime == datetime(2020, 1, 3, 0, 0, 0)

        # AAPL has 2020-01-06, MSFT has 2020-01-07
        symbol5, bar5 = merger.get_next_bar()
        assert symbol5 == "AAPL"
        assert bar5.trade_datetime == datetime(2020, 1, 6, 0, 0, 0)

        # Finally MSFT 2020-01-07
        symbol6, bar6 = merger.get_next_bar()
        assert symbol6 == "MSFT"
        assert bar6.trade_datetime == datetime(2020, 1, 7, 0, 0, 0)

        # All exhausted
        assert not merger.has_next()

    def test_get_next_bar_with_gaps(self, aapl_bars, googl_bars):
        """Test chronological order with different date ranges."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        googl_iter = create_iterator(googl_bars, "GOOGL")

        merger = BarMerger({"AAPL": aapl_iter, "GOOGL": googl_iter})

        # AAPL 2020-01-02 (GOOGL starts later)
        symbol1, bar1 = merger.get_next_bar()
        assert symbol1 == "AAPL"
        assert bar1.trade_datetime == datetime(2020, 1, 2, 0, 0, 0)

        # Both have 2020-01-03, AAPL first
        symbol2, bar2 = merger.get_next_bar()
        assert symbol2 == "AAPL"
        assert bar2.trade_datetime == datetime(2020, 1, 3, 0, 0, 0)

        # GOOGL 2020-01-03
        symbol3, bar3 = merger.get_next_bar()
        assert symbol3 == "GOOGL"
        assert bar3.trade_datetime == datetime(2020, 1, 3, 0, 0, 0)

        # Both have 2020-01-06, AAPL first
        symbol4, bar4 = merger.get_next_bar()
        assert symbol4 == "AAPL"
        assert bar4.trade_datetime == datetime(2020, 1, 6, 0, 0, 0)

        # GOOGL 2020-01-06
        symbol5, bar5 = merger.get_next_bar()
        assert symbol5 == "GOOGL"
        assert bar5.trade_datetime == datetime(2020, 1, 6, 0, 0, 0)

        # All exhausted
        assert not merger.has_next()


class TestBarMergerIteration:
    """Test BarMerger iteration behavior."""

    def test_iterate_until_exhausted(self, aapl_bars, msft_bars):
        """Test iterating until all symbols exhausted."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        msft_iter = create_iterator(msft_bars, "MSFT")

        merger = BarMerger({"AAPL": aapl_iter, "MSFT": msft_iter})

        count = 0
        while merger.has_next():
            symbol, bar = merger.get_next_bar()
            count += 1
            assert isinstance(symbol, str)
            assert isinstance(bar, MultiBar)

        # Should have yielded 6 bars total (3 AAPL + 3 MSFT)
        assert count == 6
        assert not merger.has_next()

    def test_get_next_bar_after_exhausted_raises(self, aapl_bars):
        """Test calling get_next_bar() after exhaustion raises StopIteration."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        # Exhaust iterator
        while merger.has_next():
            merger.get_next_bar()

        # Should raise StopIteration
        with pytest.raises(StopIteration, match="All symbols exhausted"):
            merger.get_next_bar()

    def test_single_symbol_yields_all_bars(self, aapl_bars):
        """Test single symbol merger yields all bars in order."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        symbols_yielded = []
        dates_yielded = []

        while merger.has_next():
            symbol, bar = merger.get_next_bar()
            symbols_yielded.append(symbol)
            dates_yielded.append(bar.trade_datetime)

        assert symbols_yielded == ["AAPL", "AAPL", "AAPL"]
        assert dates_yielded == [
            datetime(2020, 1, 2, 0, 0, 0),
            datetime(2020, 1, 3, 0, 0, 0),
            datetime(2020, 1, 6, 0, 0, 0),
        ]


class TestBarMergerPeek:
    """Test BarMerger peek functionality."""

    def test_peek_returns_next_bar(self, aapl_bars):
        """Test peek returns next bar without consuming."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        # Peek at next bar
        peeked = merger.peek_next()
        assert peeked is not None
        symbol, bar = peeked
        assert symbol == "AAPL"
        assert bar.trade_datetime == datetime(2020, 1, 2, 0, 0, 0)

        # Should still have bars
        assert merger.has_next()

        # Get next should return same bar
        actual_symbol, actual_bar = merger.get_next_bar()
        assert actual_symbol == symbol
        assert actual_bar.trade_datetime == bar.trade_datetime

    def test_peek_multiple_times_returns_same(self, aapl_bars):
        """Test peeking multiple times returns same bar."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        peek1 = merger.peek_next()
        peek2 = merger.peek_next()
        peek3 = merger.peek_next()

        assert peek1 is not None
        assert peek2 is not None
        assert peek3 is not None

        # All peeks should return same bar
        _, bar1 = peek1
        _, bar2 = peek2
        _, bar3 = peek3

        assert bar1.trade_datetime == bar2.trade_datetime == bar3.trade_datetime

    def test_peek_when_exhausted_returns_none(self, aapl_bars):
        """Test peek returns None when exhausted."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        # Exhaust iterator
        while merger.has_next():
            merger.get_next_bar()

        # Peek should return None
        assert merger.peek_next() is None


class TestBarMergerStatistics:
    """Test BarMerger statistics tracking."""

    def test_get_stats_initial(self, aapl_bars, msft_bars):
        """Test stats at initialization."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        msft_iter = create_iterator(msft_bars, "MSFT")

        merger = BarMerger({"AAPL": aapl_iter, "MSFT": msft_iter})

        stats = merger.get_stats()
        assert stats["total_symbols"] == 2
        assert stats["active_symbols"] == 2
        assert stats["exhausted_symbols"] == 0
        assert stats["total_bars_yielded"] == 0

    def test_get_stats_during_iteration(self, aapl_bars, msft_bars):
        """Test stats during iteration."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        msft_iter = create_iterator(msft_bars, "MSFT")

        merger = BarMerger({"AAPL": aapl_iter, "MSFT": msft_iter})

        # Yield 3 bars
        merger.get_next_bar()
        merger.get_next_bar()
        merger.get_next_bar()

        stats = merger.get_stats()
        assert stats["total_symbols"] == 2
        assert stats["active_symbols"] == 2
        assert stats["exhausted_symbols"] == 0
        assert stats["total_bars_yielded"] == 3

    def test_get_stats_after_exhaustion(self, aapl_bars, msft_bars):
        """Test stats after all symbols exhausted."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        msft_iter = create_iterator(msft_bars, "MSFT")

        merger = BarMerger({"AAPL": aapl_iter, "MSFT": msft_iter})

        # Exhaust all
        while merger.has_next():
            merger.get_next_bar()

        stats = merger.get_stats()
        assert stats["total_symbols"] == 2
        assert stats["active_symbols"] == 0
        assert stats["exhausted_symbols"] == 2
        assert stats["total_bars_yielded"] == 6  # 3 AAPL + 3 MSFT


class TestBarMergerMultiModeIntegration:
    """Test BarMerger works correctly with MultiBar."""

    def test_yields_multi_mode_bars(self, aapl_bars):
        """Test merger yields MultiBar with all modes."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        symbol, bar = merger.get_next_bar()

        # Should be MultiBar
        assert isinstance(bar, MultiBar)

        # Should have all 3 modes
        assert bar.unadjusted is not None
        assert bar.adjusted is not None
        assert bar.total_return is not None

        # All modes should have same timestamp
        assert bar.unadjusted.trade_datetime == bar.trade_datetime
        assert bar.adjusted.trade_datetime == bar.trade_datetime
        assert bar.total_return.trade_datetime == bar.trade_datetime

    def test_can_select_mode(self, aapl_bars):
        """Test can select different modes from merged bars."""
        aapl_iter = create_iterator(aapl_bars, "AAPL")
        merger = BarMerger({"AAPL": aapl_iter})

        symbol, bar = merger.get_next_bar()

        # Can access via direct attributes
        unadj_bar = bar.unadjusted
        assert unadj_bar.close == 103.0

        # Can access via get_bar method
        adj_bar = bar.get_bar("adjusted")
        assert adj_bar.close == 103.0

        total_bar = bar.get_bar("total_return")
        assert total_bar.close == 103.0
