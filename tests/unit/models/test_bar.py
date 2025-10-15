"""Unit tests for Bar and PriceSeries models."""

from datetime import datetime
from decimal import Decimal

import pytest

from qtrader.models.bar import Bar, PriceSeries


class TestBar:
    """Test Bar model validation and behavior."""

    def test_create_valid_bar(self) -> None:
        """Should create valid bar with all required fields."""
        bar = Bar(
            trade_datetime=datetime(2023, 1, 15),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
            dividend=None,
        )

        assert bar.trade_datetime == datetime(2023, 1, 15)
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 103.0
        assert bar.volume == 1000000
        assert bar.dividend is None

    def test_bar_with_dividend(self) -> None:
        """Should accept valid dividend amount."""
        bar = Bar(
            trade_datetime=datetime(2023, 6, 15),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
            dividend=Decimal("0.50"),
        )

        assert bar.dividend == Decimal("0.50")

    def test_bar_is_immutable(self) -> None:
        """Bar should be frozen (immutable)."""
        bar = Bar(
            trade_datetime=datetime(2023, 1, 15),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )

        with pytest.raises(Exception):  # Pydantic raises validation error
            bar.close = 110.0

    def test_high_must_be_greater_than_or_equal_to_low(self) -> None:
        """Should enforce High >= Low."""
        # Valid: High == Low
        bar = Bar(
            trade_datetime=datetime(2023, 1, 15),
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000000,
        )
        assert bar.high == bar.low

        # Invalid: High < Low
        with pytest.raises(ValueError, match="High.*< Low"):
            Bar(
                trade_datetime=datetime(2023, 1, 15),
                open=100.0,
                high=99.0,  # Less than low!
                low=100.0,
                close=100.0,
                volume=1000000,
            )

    def test_prices_must_be_positive(self) -> None:
        """Should reject negative or zero prices."""
        with pytest.raises(ValueError):
            Bar(
                trade_datetime=datetime(2023, 1, 15),
                open=0.0,  # Zero not allowed
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )

        with pytest.raises(ValueError):
            Bar(
                trade_datetime=datetime(2023, 1, 15),
                open=100.0,
                high=105.0,
                low=-1.0,  # Negative not allowed
                close=103.0,
                volume=1000000,
            )

    def test_volume_must_be_non_negative(self) -> None:
        """Should accept zero volume but reject negative."""
        # Zero volume allowed
        bar = Bar(
            trade_datetime=datetime(2023, 1, 15),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=0,
        )
        assert bar.volume == 0

        # Negative not allowed
        with pytest.raises(ValueError):
            Bar(
                trade_datetime=datetime(2023, 1, 15),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=-1000,
            )

    def test_dividend_must_be_non_negative(self) -> None:
        """Should reject negative dividend amounts."""
        with pytest.raises(ValueError):
            Bar(
                trade_datetime=datetime(2023, 6, 15),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                dividend=Decimal("-0.50"),  # Negative not allowed
            )


class TestPriceSeries:
    """Test PriceSeries model validation and behavior."""

    def test_create_valid_series(self) -> None:
        """Should create valid price series with bars."""
        bars = [
            Bar(
                trade_datetime=datetime(2023, 1, 15),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            ),
            Bar(
                trade_datetime=datetime(2023, 1, 16),
                open=103.0,
                high=108.0,
                low=102.0,
                close=107.0,
                volume=1200000,
            ),
        ]

        series = PriceSeries(mode="unadjusted", symbol="AAPL", bars=bars)

        assert series.mode == "unadjusted"
        assert series.symbol == "AAPL"
        assert len(series.bars) == 2
        assert series.bars[0].close == 103.0

    def test_series_is_immutable(self) -> None:
        """Price series should be frozen (immutable)."""
        bars = [
            Bar(
                trade_datetime=datetime(2023, 1, 15),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]
        series = PriceSeries(mode="unadjusted", symbol="AAPL", bars=bars)

        with pytest.raises(Exception):  # Pydantic raises validation error
            series.symbol = "MSFT"

    def test_valid_modes(self) -> None:
        """Should accept only valid adjustment modes."""
        bars = [
            Bar(
                trade_datetime=datetime(2023, 1, 15),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]

        # Valid modes
        for mode in ["unadjusted", "adjusted", "total_return"]:
            series = PriceSeries(mode=mode, symbol="AAPL", bars=bars)
            assert series.mode == mode

        # Invalid mode
        with pytest.raises(ValueError, match="Invalid mode"):
            PriceSeries(mode="invalid_mode", symbol="AAPL", bars=bars)

    def test_empty_series(self) -> None:
        """Should allow empty bar list."""
        series = PriceSeries(mode="unadjusted", symbol="AAPL", bars=[])

        assert series.mode == "unadjusted"
        assert series.symbol == "AAPL"
        assert len(series.bars) == 0

    def test_series_with_dividend_bar(self) -> None:
        """Should handle bars with dividend information."""
        bars = [
            Bar(
                trade_datetime=datetime(2023, 6, 14),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                dividend=None,
            ),
            Bar(
                trade_datetime=datetime(2023, 6, 15),  # Ex-dividend date
                open=98.0,
                high=102.0,
                low=97.0,
                close=100.0,
                volume=1500000,
                dividend=Decimal("0.50"),
            ),
            Bar(
                trade_datetime=datetime(2023, 6, 16),
                open=100.0,
                high=104.0,
                low=99.0,
                close=102.0,
                volume=1100000,
                dividend=None,
            ),
        ]

        series = PriceSeries(mode="unadjusted", symbol="AAPL", bars=bars)

        assert series.bars[0].dividend is None
        assert series.bars[1].dividend == Decimal("0.50")
        assert series.bars[2].dividend is None
