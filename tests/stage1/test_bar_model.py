"""Tests for Bar model and enums."""

import pytz
from datetime import datetime
from decimal import Decimal

from qtrader.models.bar import Bar, BarFrequency, DataMode, OHLCPolicy, AdjustmentEvent


def test_bar_creation_with_decimal_prices():
    """Bar should store prices as Decimal (OHLCV only)."""
    bar = Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("150.25"),
        high=Decimal("151.50"),
        low=Decimal("149.75"),
        close=Decimal("151.00"),
        volume=1000000,
    )
    assert isinstance(bar.open, Decimal)
    assert bar.open == Decimal("150.25")
    assert isinstance(bar.close, Decimal)
    assert bar.volume == 1000000


def test_bar_is_vendor_agnostic():
    """Bar should be vendor-agnostic (no vendor-specific fields)."""
    bar = Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("150.25"),
        high=Decimal("151.50"),
        low=Decimal("149.75"),
        close=Decimal("151.00"),
        volume=1000000,
    )
    # Bar should only have OHLCV fields
    assert hasattr(bar, "ts")
    assert hasattr(bar, "symbol")
    assert hasattr(bar, "open")
    assert hasattr(bar, "high")
    assert hasattr(bar, "low")
    assert hasattr(bar, "close")
    assert hasattr(bar, "volume")
    # No vendor-specific fields
    assert not hasattr(bar, "adj_reason")
    assert not hasattr(bar, "px_factor")


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
