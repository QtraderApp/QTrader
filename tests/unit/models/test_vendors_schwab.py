"""Unit tests for Schwab vendor-specific bar and price series models."""

import datetime
from typing import Any

import pytest

from qtrader.models.vendors.schwab import SchwabBar, SchwabPriceSeries


class TestSchwabBar:
    """Test SchwabBar model and data parsing."""

    def test_create_valid_bar(self) -> None:
        """Should create valid Schwab bar."""
        bar = SchwabBar(
            timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=10000000,
        )

        assert bar.timestamp == datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc)
        assert bar.open == 150.0
        assert bar.high == 155.0
        assert bar.low == 149.0
        assert bar.close == 153.0
        assert bar.volume == 10000000

    def test_parse_unix_timestamp_milliseconds(self) -> None:
        """Should parse Unix timestamp in milliseconds from Schwab API."""
        # 2023-01-15 00:00:00 UTC = 1673740800000 milliseconds
        bar = SchwabBar(
            timestamp=1673740800000,
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=10000000,
        )

        assert bar.timestamp == datetime.datetime(2023, 1, 15, 0, 0, 0, tzinfo=datetime.timezone.utc)

    def test_parse_iso_string_datetime(self) -> None:
        """Should parse ISO format datetime string."""
        bar = SchwabBar(
            timestamp="2023-01-15T09:30:00+00:00",
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=10000000,
        )

        assert bar.timestamp == datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc)

    def test_datetime_already_datetime(self) -> None:
        """Should handle datetime objects directly."""
        dt = datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc)
        bar = SchwabBar(
            timestamp=dt,
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=10000000,
        )

        assert bar.timestamp == dt

    def test_invalid_datetime_raises_error(self) -> None:
        """Should raise error for invalid datetime format."""
        with pytest.raises(ValueError, match="Cannot parse datetime"):
            SchwabBar(
                timestamp={"invalid": "object"},
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=10000000,
            )

    def test_default_volume_zero(self) -> None:
        """Should default volume to 0 if not provided."""
        bar = SchwabBar(
            timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
        )

        assert bar.volume == 0

    def test_ohlc_validation_valid(self) -> None:
        """Should pass OHLC validation for valid data."""
        bar = SchwabBar(
            timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
            open=150.0,
            high=155.0,
            low=149.0,
            close=153.0,
            volume=10000000,
        )

        # Should not raise
        assert bar.high >= bar.low
        assert bar.high >= bar.open
        assert bar.high >= bar.close
        assert bar.low <= bar.open
        assert bar.low <= bar.close

    def test_ohlc_validation_severe_violation(self) -> None:
        """Should raise error when High < Low (severe violation)."""
        with pytest.raises(ValueError, match="SEVERE: High .* < Low"):
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=150.0,
                high=148.0,  # High < Low (severe violation)
                low=149.0,
                close=148.5,
                volume=10000000,
            )

    def test_ohlc_validation_minor_violation_warning(self, capsys) -> None:
        """Should warn but allow minor OHLC violations within tolerance."""
        # High significantly less than Open (exceeds 5% tolerance)
        bar = SchwabBar(
            timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
            open=150.0,
            high=140.0,  # High < Open by more than 5% (should warn)
            low=139.0,
            close=139.5,
            volume=10000000,
        )

        # Should create bar and print warning
        assert bar.high == 140.0
        captured = capsys.readouterr()
        assert "OHLC warnings" in captured.out


class TestSchwabPriceSeries:
    """Test SchwabPriceSeries model and canonical conversion."""

    def test_create_valid_price_series(self) -> None:
        """Should create valid Schwab price series."""
        bars = [
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=10000000,
            ),
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 16, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=153.0,
                high=157.0,
                low=152.0,
                close=156.0,
                volume=12000000,
            ),
        ]

        series = SchwabPriceSeries(symbol="AAPL", bars=bars)

        assert series.symbol == "AAPL"
        assert len(series.bars) == 2
        assert series.bars[0].close == 153.0
        assert series.bars[1].close == 156.0

    def test_to_canonical_series_returns_partial_multibar(self) -> None:
        """Should return partial MultiBar with only adjusted data."""
        bars = [
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=10000000,
            ),
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 16, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=153.0,
                high=157.0,
                low=152.0,
                close=156.0,
                volume=12000000,
            ),
        ]

        series = SchwabPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        # Should have 3 keys
        assert set(canonical.keys()) == {"unadjusted", "adjusted", "total_return"}

        # All modes should be populated (Schwab returns adjusted data only)
        assert canonical["unadjusted"] is not None
        assert canonical["adjusted"] is not None
        assert canonical["total_return"] is not None

        # All modes should have the same data (Schwab limitation)
        for mode in ["unadjusted", "adjusted", "total_return"]:
            series_data = canonical[mode]
            assert series_data.symbol == "AAPL"
            assert series_data.mode == "adjusted"  # All modes use adjusted
            assert len(series_data.bars) == 2
            assert series_data.bars[0].close == 153.0
            assert series_data.bars[1].close == 156.0

    def test_to_canonical_series_converts_to_canonical_bars(self) -> None:
        """Should convert Schwab bars to canonical Bar objects."""
        bars = [
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=10000000,
            ),
        ]

        series = SchwabPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        adjusted = canonical["adjusted"]
        assert adjusted is not None
        bar = adjusted.bars[0]

        # Check field mapping
        assert bar.trade_datetime == datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc)
        assert bar.open == 150.0
        assert bar.high == 155.0
        assert bar.low == 149.0
        assert bar.close == 153.0
        assert bar.volume == 10000000
        assert bar.dividend is None  # Schwab doesn't provide dividend data

    def test_to_canonical_series_empty_bars(self) -> None:
        """Should handle empty bar list gracefully."""
        series = SchwabPriceSeries(symbol="AAPL", bars=[])
        canonical = series.to_canonical_series()

        # All modes should be populated with empty series
        for mode in ["unadjusted", "adjusted", "total_return"]:
            assert canonical[mode] is not None
            series_data = canonical[mode]
            assert series_data.symbol == "AAPL"
            assert series_data.mode == "adjusted"  # All modes use adjusted
            assert len(series_data.bars) == 0

    def test_to_canonical_series_preserves_chronological_order(self) -> None:
        """Should preserve chronological order of bars."""
        bars = [
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=10000000,
            ),
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 16, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=153.0,
                high=157.0,
                low=152.0,
                close=156.0,
                volume=12000000,
            ),
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 17, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=156.0,
                high=160.0,
                low=155.0,
                close=159.0,
                volume=11000000,
            ),
        ]

        series = SchwabPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        adjusted = canonical["adjusted"]
        assert adjusted is not None
        assert len(adjusted.bars) == 3
        assert adjusted.bars[0].trade_datetime < adjusted.bars[1].trade_datetime
        assert adjusted.bars[1].trade_datetime < adjusted.bars[2].trade_datetime

    def test_to_canonical_series_handles_zero_volume(self) -> None:
        """Should handle bars with zero volume."""
        bars = [
            SchwabBar(
                timestamp=datetime.datetime(2023, 1, 15, 9, 30, 0, tzinfo=datetime.timezone.utc),
                open=150.0,
                high=155.0,
                low=149.0,
                close=153.0,
                volume=0,  # Zero volume
            ),
        ]

        series = SchwabPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        adjusted = canonical["adjusted"]
        assert adjusted is not None
        assert len(adjusted.bars) == 1
        assert adjusted.bars[0].volume == 0

    def test_schwab_api_response_format(self) -> None:
        """Should parse typical Schwab API response format."""
        # Simulate parsing Schwab API response
        api_candles: list[dict[str, Any]] = [
            {
                "timestamp": 1673740800000,  # Unix milliseconds
                "open": 132.43,
                "high": 133.61,
                "low": 131.72,
                "close": 132.05,
                "volume": 143301900,
            },
            {
                "timestamp": 1673827200000,
                "open": 132.05,
                "high": 134.20,
                "low": 131.90,
                "close": 133.50,
                "volume": 156789200,
            },
        ]

        # Parse into SchwabBar objects
        bars = [SchwabBar(**candle) for candle in api_candles]

        series = SchwabPriceSeries(symbol="AAPL", bars=bars)
        canonical = series.to_canonical_series()

        adjusted = canonical["adjusted"]
        assert adjusted is not None
        assert len(adjusted.bars) == 2
        assert adjusted.bars[0].close == 132.05
        assert adjusted.bars[1].close == 133.50
        assert adjusted.bars[0].volume == 143301900
        assert adjusted.bars[1].volume == 156789200
