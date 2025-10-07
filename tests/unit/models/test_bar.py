"""Tests for Bar model and enums."""

from datetime import datetime
from decimal import Decimal

import pytz

from qtrader.models.bar import AdjustmentEvent, Bar, BarFrequency, DataMode, OHLCPolicy, PriceSeries


def create_test_bar(
    symbol: str = "AAPL",
    ts: datetime | None = None,
    open_price: Decimal = Decimal("150.25"),
    high_price: Decimal = Decimal("151.50"),
    low_price: Decimal = Decimal("149.75"),
    close_price: Decimal = Decimal("151.00"),
    volume: int = 1000000,
) -> Bar:
    """Helper to create test bar with all three price series."""
    if ts is None:
        ts = datetime(2023, 1, 1, tzinfo=pytz.UTC)

    # Create same prices for all series (for simple tests)
    price_series = PriceSeries(
        open=open_price,
        high=high_price,
        low=low_price,
        close=close_price,
        volume=volume,
    )

    return Bar(
        ts=ts,
        symbol=symbol,
        unadjusted=price_series,
        capital_adjusted=price_series,
        total_return=price_series,
        dividend=None,
        split=None,
    )


def test_bar_creation_with_decimal_prices():
    """Bar should store prices as Decimal in all three series."""
    bar = create_test_bar()

    # All three series should have Decimal prices
    assert isinstance(bar.unadjusted.open, Decimal)
    assert bar.unadjusted.open == Decimal("150.25")
    assert isinstance(bar.capital_adjusted.close, Decimal)
    assert isinstance(bar.total_return.close, Decimal)
    assert bar.unadjusted.volume == 1000000


def test_bar_is_vendor_agnostic():
    """Bar should be vendor-agnostic (no vendor-specific fields)."""
    bar = create_test_bar()

    # Bar should have required fields
    assert hasattr(bar, "ts")
    assert hasattr(bar, "symbol")
    assert hasattr(bar, "unadjusted")
    assert hasattr(bar, "capital_adjusted")
    assert hasattr(bar, "total_return")
    assert hasattr(bar, "dividend")
    assert hasattr(bar, "split")

    # Each series should have OHLCV
    assert hasattr(bar.unadjusted, "open")
    assert hasattr(bar.unadjusted, "high")
    assert hasattr(bar.unadjusted, "low")
    assert hasattr(bar.unadjusted, "close")
    assert hasattr(bar.unadjusted, "volume")


def test_adjustment_event_creation():
    """AdjustmentEvent should store adjustment metadata separately."""
    event = AdjustmentEvent(
        ts=datetime(2023, 2, 8, tzinfo=pytz.UTC),
        symbol="AAPL",
        event_type="CashDiv",
        px_factor=Decimal("7.9599520"),
        vol_factor=Decimal("7.0"),
        metadata={"amount": 0.23, "currency": "USD"},
    )
    assert event.event_type == "CashDiv"
    assert event.px_factor == Decimal("7.9599520")
    assert event.metadata["amount"] == 0.23


def test_bar_frequency_enum():
    """BarFrequency enum should have expected values."""
    assert BarFrequency.DAY_1.value == "1d"
    assert BarFrequency.MIN_5.value == "5m"
    assert BarFrequency.HOUR_1.value == "1h"


def test_data_mode_enum():
    """DataMode enum should have expected values."""
    assert DataMode.ADJUSTED.value == "adjusted"
    assert DataMode.UNADJUSTED.value == "unadjusted"
    assert DataMode.SPLIT_ADJUSTED.value == "split_adjusted"


def test_ohlc_policy_enum():
    """OHLCPolicy enum should have expected values."""
    assert OHLCPolicy.STRICT_RAISE.value == "strict_raise"
    assert OHLCPolicy.WARN_SKIP_BAR.value == "warn_skip_bar"
    assert OHLCPolicy.WARN_USE_CLOSE_ONLY.value == "warn_use_close_only"
