"""Unit tests for CanonicalBar and CanonicalPriceSeries models."""

from decimal import Decimal

import pytest

from qtrader.models.canonical_bar import CanonicalBar, CanonicalPriceSeries


class TestCanonicalBar:
    """Test CanonicalBar model validation and behavior."""

    def test_create_valid_bar(self) -> None:
        """Should create valid bar with all required fields."""
        bar = CanonicalBar(
            trade_datetime="2023-01-15",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
            dividend=None,
        )

        assert bar.trade_datetime == "2023-01-15"
        assert bar.open == 100.0
        assert bar.high == 105.0
        assert bar.low == 99.0
        assert bar.close == 103.0
        assert bar.volume == 1000000
        assert bar.dividend is None

    def test_bar_with_dividend(self) -> None:
        """Should accept valid dividend amount."""
        bar = CanonicalBar(
            trade_datetime="2023-06-15",
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
        bar = CanonicalBar(
            trade_datetime="2023-01-15",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )

        with pytest.raises(Exception):  # Pydantic raises validation error
            bar.close = 110.0  # type: ignore

    def test_high_must_be_greater_than_or_equal_to_low(self) -> None:
        """Should enforce High >= Low."""
        # Valid: High == Low
        bar = CanonicalBar(
            trade_datetime="2023-01-15",
            open=100.0,
            high=100.0,
            low=100.0,
            close=100.0,
            volume=1000000,
        )
        assert bar.high == bar.low

        # Invalid: High < Low
        with pytest.raises(ValueError, match="High.*< Low"):
            CanonicalBar(
                trade_datetime="2023-01-15",
                open=100.0,
                high=99.0,  # Less than low!
                low=100.0,
                close=100.0,
                volume=1000000,
            )

    def test_prices_must_be_positive(self) -> None:
        """Should reject negative or zero prices."""
        with pytest.raises(ValueError):
            CanonicalBar(
                trade_datetime="2023-01-15",
                open=0.0,  # Zero not allowed
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )

        with pytest.raises(ValueError):
            CanonicalBar(
                trade_datetime="2023-01-15",
                open=100.0,
                high=105.0,
                low=-1.0,  # Negative not allowed
                close=103.0,
                volume=1000000,
            )

    def test_volume_must_be_non_negative(self) -> None:
        """Should accept zero volume but reject negative."""
        # Zero volume allowed
        bar = CanonicalBar(
            trade_datetime="2023-01-15",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=0,
        )
        assert bar.volume == 0

        # Negative not allowed
        with pytest.raises(ValueError):
            CanonicalBar(
                trade_datetime="2023-01-15",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=-1000,
            )

    def test_dividend_must_be_non_negative(self) -> None:
        """Should reject negative dividend amounts."""
        with pytest.raises(ValueError):
            CanonicalBar(
                trade_datetime="2023-06-15",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                dividend=Decimal("-0.50"),  # Negative not allowed
            )


class TestCanonicalPriceSeries:
    """Test CanonicalPriceSeries model validation and behavior."""

    def test_create_valid_series(self) -> None:
        """Should create valid price series with bars."""
        bars = [
            CanonicalBar(
                trade_datetime="2023-01-15",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            ),
            CanonicalBar(
                trade_datetime="2023-01-16",
                open=103.0,
                high=108.0,
                low=102.0,
                close=107.0,
                volume=1200000,
            ),
        ]

        series = CanonicalPriceSeries(mode="unadjusted", symbol="AAPL", bars=bars)

        assert series.mode == "unadjusted"
        assert series.symbol == "AAPL"
        assert len(series.bars) == 2
        assert series.bars[0].close == 103.0

    def test_series_is_immutable(self) -> None:
        """Price series should be frozen (immutable)."""
        bars = [
            CanonicalBar(
                trade_datetime="2023-01-15",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]
        series = CanonicalPriceSeries(mode="unadjusted", symbol="AAPL", bars=bars)

        with pytest.raises(Exception):  # Pydantic raises validation error
            series.symbol = "MSFT"  # type: ignore

    def test_valid_modes(self) -> None:
        """Should accept only valid adjustment modes."""
        bars = [
            CanonicalBar(
                trade_datetime="2023-01-15",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]

        # Valid modes
        for mode in ["unadjusted", "adjusted", "total_return"]:
            series = CanonicalPriceSeries(mode=mode, symbol="AAPL", bars=bars)
            assert series.mode == mode

        # Invalid mode
        with pytest.raises(ValueError, match="Invalid mode"):
            CanonicalPriceSeries(mode="invalid_mode", symbol="AAPL", bars=bars)

    def test_empty_series(self) -> None:
        """Should allow empty bar list."""
        series = CanonicalPriceSeries(mode="unadjusted", symbol="AAPL", bars=[])

        assert series.mode == "unadjusted"
        assert series.symbol == "AAPL"
        assert len(series.bars) == 0

    def test_series_with_dividend_bar(self) -> None:
        """Should handle bars with dividend information."""
        bars = [
            CanonicalBar(
                trade_datetime="2023-06-14",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
                dividend=None,
            ),
            CanonicalBar(
                trade_datetime="2023-06-15",  # Ex-dividend date
                open=98.0,
                high=102.0,
                low=97.0,
                close=100.0,
                volume=1500000,
                dividend=Decimal("0.50"),
            ),
            CanonicalBar(
                trade_datetime="2023-06-16",
                open=100.0,
                high=104.0,
                low=99.0,
                close=102.0,
                volume=1100000,
                dividend=None,
            ),
        ]

        series = CanonicalPriceSeries(mode="unadjusted", symbol="AAPL", bars=bars)

        assert series.bars[0].dividend is None
        assert series.bars[1].dividend == Decimal("0.50")
        assert series.bars[2].dividend is None
