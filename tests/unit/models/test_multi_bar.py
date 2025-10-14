"""Tests for MultiBar model."""

from decimal import Decimal

import pytest

from qtrader.models.bar import Bar
from qtrader.models.multi_bar import MultiBar


class TestMultiBarCreation:
    """Test MultiBar creation and validation."""

    def test_create_multi_mode_bar(self):
        """Test creating a valid MultiBar."""
        # Arrange: Create bars for each mode
        unadj_bar = Bar(
            trade_datetime="2020-08-31T00:00:00",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000000,
        )
        adj_bar = Bar(
            trade_datetime="2020-08-31T00:00:00",
            open=25.0,
            high=26.25,
            low=24.75,
            close=26.0,
            volume=1000000,
        )
        tr_bar = Bar(
            trade_datetime="2020-08-31T00:00:00",
            open=25.5,
            high=26.75,
            low=25.25,
            close=26.5,
            volume=1000000,
            dividend=Decimal("0.205"),
        )

        # Act: Create MultiBar
        multi_bar = MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-31T00:00:00",
            unadjusted=unadj_bar,
            adjusted=adj_bar,
            total_return=tr_bar,
        )

        # Assert: All fields populated correctly
        assert multi_bar.symbol == "AAPL"
        assert multi_bar.trade_datetime == "2020-08-31T00:00:00"
        assert multi_bar.unadjusted == unadj_bar
        assert multi_bar.adjusted == adj_bar
        assert multi_bar.total_return == tr_bar

    def test_multi_mode_bar_immutable(self):
        """Test that MultiBar is immutable (frozen)."""
        # Arrange: Create MultiBar
        bar = Bar(
            trade_datetime="2020-01-01T00:00:00",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000,
        )
        multi_bar = MultiBar(
            symbol="AAPL",
            trade_datetime="2020-01-01T00:00:00",
            unadjusted=bar,
            adjusted=bar,
            total_return=bar,
        )

        # Act & Assert: Cannot modify
        with pytest.raises(Exception):  # Pydantic raises validation error
            multi_bar.symbol = "MSFT"  # type: ignore


class TestMultiBarModeAccess:
    """Test accessing different adjustment modes."""

    @pytest.fixture
    def sample_multi_bar(self) -> MultiBar:
        """Create sample MultiBar for testing."""
        unadj = Bar(
            trade_datetime="2020-08-31T00:00:00",
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000000,
        )
        adj = Bar(
            trade_datetime="2020-08-31T00:00:00",
            open=25.0,
            high=26.25,
            low=24.75,
            close=26.0,
            volume=1000000,
        )
        tr = Bar(
            trade_datetime="2020-08-31T00:00:00",
            open=25.5,
            high=26.75,
            low=25.25,
            close=26.5,
            volume=1000000,
            dividend=Decimal("0.205"),
        )
        return MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-31T00:00:00",
            unadjusted=unadj,
            adjusted=adj,
            total_return=tr,
        )

    def test_direct_access_unadjusted(self, sample_multi_bar):
        """Test direct access to unadjusted mode."""
        bar = sample_multi_bar.unadjusted
        assert bar.close == 104.0
        assert bar.open == 100.0

    def test_direct_access_adjusted(self, sample_multi_bar):
        """Test direct access to adjusted mode."""
        bar = sample_multi_bar.adjusted
        assert bar.close == 26.0
        assert bar.open == 25.0

    def test_direct_access_total_return(self, sample_multi_bar):
        """Test direct access to total_return mode."""
        bar = sample_multi_bar.total_return
        assert bar.close == 26.5
        assert bar.dividend == Decimal("0.205")

    def test_get_bar_unadjusted(self, sample_multi_bar):
        """Test get_bar() with unadjusted mode."""
        bar = sample_multi_bar.get_bar("unadjusted")
        assert bar.close == 104.0
        assert bar is sample_multi_bar.unadjusted

    def test_get_bar_adjusted(self, sample_multi_bar):
        """Test get_bar() with adjusted mode."""
        bar = sample_multi_bar.get_bar("adjusted")
        assert bar.close == 26.0
        assert bar is sample_multi_bar.adjusted

    def test_get_bar_total_return(self, sample_multi_bar):
        """Test get_bar() with total_return mode."""
        bar = sample_multi_bar.get_bar("total_return")
        assert bar.close == 26.5
        assert bar is sample_multi_bar.total_return

    def test_get_bar_invalid_mode(self, sample_multi_bar):
        """Test get_bar() with invalid mode raises error."""
        with pytest.raises(ValueError, match="Invalid mode"):
            sample_multi_bar.get_bar("invalid")  # type: ignore

    def test_get_bar_dynamic_selection(self, sample_multi_bar):
        """Test dynamic mode selection based on config."""
        # Simulate different components selecting different modes
        signal_mode = "adjusted"
        exec_mode = "unadjusted"
        perf_mode = "total_return"

        signal_bar = sample_multi_bar.get_bar(signal_mode)
        exec_bar = sample_multi_bar.get_bar(exec_mode)
        perf_bar = sample_multi_bar.get_bar(perf_mode)

        # Each component gets its optimal mode
        assert signal_bar.close == 26.0  # adjusted
        assert exec_bar.close == 104.0  # unadjusted
        assert perf_bar.close == 26.5  # total_return


class TestMultiBarUseCases:
    """Test real-world usage patterns."""

    def test_strategy_uses_adjusted(self):
        """Test strategy component uses adjusted mode for indicators."""
        # Arrange: AAPL after 4:1 split
        multi_bar = MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-31T00:00:00",
            unadjusted=Bar(
                trade_datetime="2020-08-31T00:00:00",
                open=100.0,
                high=105.0,
                low=99.0,
                close=104.0,
                volume=1000000,
            ),
            adjusted=Bar(
                trade_datetime="2020-08-31T00:00:00",
                open=25.0,
                high=26.25,
                low=24.75,
                close=26.0,
                volume=1000000,
            ),
            total_return=Bar(
                trade_datetime="2020-08-31T00:00:00",
                open=25.0,
                high=26.25,
                low=24.75,
                close=26.0,
                volume=1000000,
            ),
        )

        # Act: Strategy selects adjusted for indicators
        strategy_bar = multi_bar.adjusted

        # Assert: Uses split-adjusted prices for consistent indicators
        assert strategy_bar.close == 26.0  # Post-split adjusted
        # This ensures SMA, RSI work correctly across the split

    def test_execution_uses_unadjusted(self):
        """Test execution component uses unadjusted for realistic fills."""
        # Arrange: Same AAPL bar
        multi_bar = MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-31T00:00:00",
            unadjusted=Bar(
                trade_datetime="2020-08-31T00:00:00",
                open=100.0,
                high=105.0,
                low=99.0,
                close=104.0,
                volume=1000000,
            ),
            adjusted=Bar(
                trade_datetime="2020-08-31T00:00:00",
                open=25.0,
                high=26.25,
                low=24.75,
                close=26.0,
                volume=1000000,
            ),
            total_return=Bar(
                trade_datetime="2020-08-31T00:00:00",
                open=25.0,
                high=26.25,
                low=24.75,
                close=26.0,
                volume=1000000,
            ),
        )

        # Act: Execution selects unadjusted for fills
        exec_bar = multi_bar.unadjusted
        shares = 100
        commission_rate = 0.001

        # Assert: Commission calculated on actual traded price
        fill_price = exec_bar.high  # 105.0
        commission = fill_price * shares * commission_rate  # 10.50
        assert commission == 10.50

        # If we mistakenly used adjusted:
        wrong_commission = multi_bar.adjusted.high * shares * commission_rate
        assert wrong_commission == 2.625  # WRONG! Too low

    def test_performance_uses_total_return(self):
        """Test performance component uses total_return for accurate returns."""
        # Arrange: Bar with dividend
        multi_bar = MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-07T00:00:00",
            unadjusted=Bar(
                trade_datetime="2020-08-07T00:00:00",
                open=100.0,
                high=105.0,
                low=99.0,
                close=104.0,
                volume=1000000,
                dividend=Decimal("0.82"),
            ),
            adjusted=Bar(
                trade_datetime="2020-08-07T00:00:00",
                open=100.0,
                high=105.0,
                low=99.0,
                close=104.0,
                volume=1000000,
                dividend=Decimal("0.82"),
            ),
            total_return=Bar(
                trade_datetime="2020-08-07T00:00:00",
                open=100.82,
                high=105.87,
                low=99.82,
                close=104.86,
                volume=1000000,
                dividend=Decimal("0.82"),
            ),
        )

        # Act: Performance selects total_return
        perf_bar = multi_bar.total_return

        # Assert: Price includes dividend reinvestment
        assert perf_bar.close == 104.86  # Includes $0.82 dividend
        assert multi_bar.adjusted.close == 104.0  # Does not include dividend

        # Total return calculation is accurate
        entry_price = 100.0
        return_with_div = (perf_bar.close - entry_price) / entry_price
        return_without_div = (multi_bar.adjusted.close - entry_price) / entry_price

        assert return_with_div > return_without_div  # Total return higher
