"""Tests for Algoseek Parquet adapter."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from qtrader.adapters.algoseek_parquet import AlgoseekParquetAdapter
from qtrader.config.data_config import AdjustmentSchemaConfig, BarSchemaConfig, DataConfig
from qtrader.models.bar import AdjustmentEvent, DataMode


@pytest.fixture
def fixture_path():
    """Path to sample parquet data."""
    return Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample")


@pytest.fixture
def bar_schema():
    """Bar schema config for Algoseek data."""
    return BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="MarketHoursVolume",
    )


@pytest.fixture
def adjustment_schema():
    """Adjustment schema config for Algoseek data."""
    return AdjustmentSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        event_type="AdjustmentReason",
        px_factor="CumulativePriceFactor",
        vol_factor="CumulativeVolumeFactor",
    )


@pytest.fixture
def data_config(bar_schema, adjustment_schema):
    """Complete data configuration."""
    return DataConfig(
        timezone="America/New_York",
        frequency="1d",
        bar_schema=bar_schema,
        adjustment_schema=adjustment_schema,
    )


def test_adapter_can_read_fixture(fixture_path):
    """Adapter should detect parquet files."""
    adapter = AlgoseekParquetAdapter()
    assert adapter.can_read(fixture_path) is True


def test_adapter_cannot_read_missing_path():
    """Adapter should return False for missing path."""
    adapter = AlgoseekParquetAdapter()
    assert adapter.can_read(Path("nonexistent/path")) is False


def test_adapter_schema_version():
    """Adapter should report schema version."""
    adapter = AlgoseekParquetAdapter()
    assert adapter.schema_version() == "algoseek-parquet-v1.0"


def test_adapter_data_mode():
    """Adapter should declare data mode."""
    adapter = AlgoseekParquetAdapter()
    assert adapter.get_data_mode() == DataMode.ADJUSTED


def test_adapter_reads_fixture_bars(fixture_path, data_config):
    """Adapter should load all bars from fixture (OHLCV only)."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_bars(fixture_path, data_config))

    # Should have 3 symbols × 1258 days = 3774 bars
    assert len(bars) == 3774

    # Check symbols are present
    symbols = {bar.symbol for bar in bars}
    assert symbols == {"AAPL", "MSFT", "AMZN"}


def test_adapter_first_bar_format(fixture_path, data_config):
    """First bar should have correct format and types (OHLCV only)."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_bars(fixture_path, data_config))

    # Find first AAPL bar
    aapl_bars = [b for b in bars if b.symbol == "AAPL"]
    assert len(aapl_bars) == 1258

    first_bar = aapl_bars[0]
    assert first_bar.ts.date() == date(2019, 1, 2)
    assert first_bar.symbol == "AAPL"
    assert isinstance(first_bar.open, Decimal)
    assert isinstance(first_bar.close, Decimal)
    assert first_bar.close == Decimal("157.9200")  # Adjusted price
    assert first_bar.volume == 30606605  # Actual volume from fixture data
    # Bar should NOT have adjustment fields
    assert not hasattr(first_bar, "adj_reason")
    assert not hasattr(first_bar, "px_factor")


def test_adapter_reads_adjustments_separately(fixture_path, data_config):
    """Adapter should read adjustment metadata separately from bars."""
    adapter = AlgoseekParquetAdapter()
    adjustments = list(adapter.read_adjustments(fixture_path, data_config))

    # AAPL + MSFT have dividend events in 2019-2023
    assert len(adjustments) >= 30
    assert all(isinstance(adj, AdjustmentEvent) for adj in adjustments)

    # Fixture contains CashDiv, BonusSame, Subdiv
    event_types = {adj.event_type for adj in adjustments}
    assert "CashDiv" in event_types
    assert len(event_types) >= 1  # At least CashDiv present

    # Check factors are preserved
    assert all(adj.px_factor is not None for adj in adjustments)
    assert all(isinstance(adj.px_factor, Decimal) for adj in adjustments)


def test_adapter_bars_sorted_by_symbol_and_time(fixture_path, data_config):
    """Bars should be sorted by symbol, then timestamp."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_bars(fixture_path, data_config))

    # Check ordering
    for i in range(1, len(bars)):
        prev, curr = bars[i - 1], bars[i]
        # Either symbol increases, or same symbol with increasing time
        if prev.symbol == curr.symbol:
            assert curr.ts > prev.ts, f"Timestamps not sorted for {curr.symbol}"
        else:
            assert curr.symbol > prev.symbol, "Symbols not sorted"


def test_adapter_bar_prices_are_decimal(fixture_path, data_config):
    """All bar prices should be Decimal type."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_bars(fixture_path, data_config))

    # Check first 100 bars
    for bar in bars[:100]:
        assert isinstance(bar.open, Decimal), f"open not Decimal for {bar.symbol}"
        assert isinstance(bar.high, Decimal), f"high not Decimal for {bar.symbol}"
        assert isinstance(bar.low, Decimal), f"low not Decimal for {bar.symbol}"
        assert isinstance(bar.close, Decimal), f"close not Decimal for {bar.symbol}"


def test_adapter_bar_timestamps_are_timezone_aware(fixture_path, data_config):
    """All bar timestamps should be timezone-aware."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_bars(fixture_path, data_config))

    # Check first 100 bars
    for bar in bars[:100]:
        assert bar.ts.tzinfo is not None, f"Timestamp not tz-aware for {bar.symbol}"
        assert str(bar.ts.tzinfo) in [
            "America/New_York",
            "EST",
            "EDT",
        ], f"Wrong timezone for {bar.symbol}"


def test_adapter_no_adjustments_without_schema(fixture_path, bar_schema):
    """Adapter should skip adjustments if adjustment_schema is None."""
    config = DataConfig(
        timezone="America/New_York",
        frequency="1d",
        bar_schema=bar_schema,
        adjustment_schema=None,  # No adjustment schema
    )
    adapter = AlgoseekParquetAdapter()
    adjustments = list(adapter.read_adjustments(fixture_path, config))

    # Should be empty when no adjustment schema provided
    assert len(adjustments) == 0
