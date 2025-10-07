"""Tests for Algoseek OHLC adapter."""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from qtrader.adapters.algoseek import AlgoseekOHLCAdapter
from qtrader.config.data_config import AdjustmentSchemaConfig, BarSchemaConfig, DataConfig
from qtrader.models.bar import AdjustmentEvent, DataMode
from qtrader.models.instrument import DataSource, Instrument, InstrumentType


@pytest.fixture
def fixture_path():
    """Path to sample parquet data."""
    return Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample")


@pytest.fixture
def adapter_config(fixture_path):
    """Configuration for Algoseek adapter."""
    return {
        "root_path": str(fixture_path),
        "mode": "standard_adjusted",
        "path_template": "{root_path}/SecId={secid}/*.parquet",
        "symbol_map": "data/equity_security_master_sample.csv",
    }


@pytest.fixture
def instrument_aapl():
    """AAPL instrument for testing."""
    return Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)


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


def test_adapter_can_read_fixture(adapter_config, instrument_aapl):
    """Adapter should detect parquet files."""
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument_aapl)
    assert adapter.can_read() is True


def test_adapter_cannot_read_missing_path(instrument_aapl):
    """Adapter should return False for missing path."""
    bad_config = {
        "root_path": "nonexistent/path",
        "mode": "standard_adjusted",
        "path_template": "{root_path}/SecId={secid}/*.parquet",
        "symbol_map": "data/equity_security_master_sample.csv",
    }
    adapter = AlgoseekOHLCAdapter(bad_config, instrument_aapl)
    assert adapter.can_read() is False


def test_adapter_schema_version(adapter_config, instrument_aapl):
    """Adapter should report schema version."""
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument_aapl)
    assert adapter.schema_version() == "algoseek-ohlc-v1.0"


def test_adapter_data_mode(adapter_config, instrument_aapl):
    """Adapter should declare data mode."""
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument_aapl)
    assert adapter.get_data_mode() == DataMode.ADJUSTED


def test_adapter_reads_fixture_bars(adapter_config, data_config):
    """Adapter should load all bars from fixture (OHLCV only)."""
    # Test with AAPL
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    aapl_bars = list(adapter.read_bars(data_config))

    # AAPL should have 1258 bars
    assert len(aapl_bars) == 1258
    assert all(bar.symbol == "AAPL" for bar in aapl_bars)

    # Test with MSFT
    instrument = Instrument("MSFT", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    msft_bars = list(adapter.read_bars(data_config))
    assert len(msft_bars) == 1258
    assert all(bar.symbol == "MSFT" for bar in msft_bars)


def test_adapter_first_bar_format(adapter_config, data_config):
    """First bar should have correct format and types with 3 price series."""
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    bars = list(adapter.read_bars(data_config))

    # Should have 1258 bars for AAPL
    assert len(bars) == 1258

    first_bar = bars[0]
    assert first_bar.ts.date() == date(2019, 1, 2)
    assert first_bar.symbol == "AAPL"

    # Bar should have 3 price series
    assert hasattr(first_bar, "unadjusted")
    assert hasattr(first_bar, "capital_adjusted")
    assert hasattr(first_bar, "total_return")

    # Check total_return series (the adjusted data from vendor)
    assert isinstance(first_bar.total_return.open, Decimal)
    assert isinstance(first_bar.total_return.close, Decimal)
    assert first_bar.total_return.close == Decimal("157.9200")  # Adjusted price
    assert first_bar.total_return.volume == 30606605  # Actual volume from fixture data

    # Bar should have optional dividend/split fields
    assert hasattr(first_bar, "dividend")
    assert hasattr(first_bar, "split")


def test_adapter_reads_adjustments_separately(adapter_config, data_config):
    """Adapter should read adjustment metadata separately from bars."""
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    adjustments = list(adapter.read_adjustments(data_config))

    # AAPL has dividend events in 2019-2023
    assert len(adjustments) >= 10
    assert all(isinstance(adj, AdjustmentEvent) for adj in adjustments)

    # Fixture contains CashDiv, BonusSame, Subdiv
    event_types = {adj.event_type for adj in adjustments}
    assert "CashDiv" in event_types
    assert len(event_types) >= 1  # At least CashDiv present

    # Check factors are preserved
    assert all(adj.px_factor is not None for adj in adjustments)
    assert all(isinstance(adj.px_factor, Decimal) for adj in adjustments)


def test_adapter_bars_sorted_by_symbol_and_time(adapter_config, data_config):
    """Bars should be sorted by timestamp (single instrument)."""
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    bars = list(adapter.read_bars(data_config))

    # Check ordering - all bars should be for AAPL, sorted by time
    for i in range(1, len(bars)):
        prev, curr = bars[i - 1], bars[i]
        assert curr.symbol == "AAPL"
        assert curr.ts > prev.ts, f"Timestamps not sorted for {curr.symbol}"


def test_adapter_bar_prices_are_decimal(adapter_config, data_config):
    """All bar prices in all series should be Decimal type."""
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    bars = list(adapter.read_bars(data_config))

    # Check first 100 bars - verify all 3 price series have Decimal prices
    for bar in bars[:100]:
        for series_name in ["unadjusted", "capital_adjusted", "total_return"]:
            series = getattr(bar, series_name)
            assert isinstance(series.open, Decimal), f"{series_name}.open not Decimal for {bar.symbol}"
            assert isinstance(series.high, Decimal), f"{series_name}.high not Decimal for {bar.symbol}"
            assert isinstance(series.low, Decimal), f"{series_name}.low not Decimal for {bar.symbol}"
            assert isinstance(series.close, Decimal), f"{series_name}.close not Decimal for {bar.symbol}"


def test_adapter_bar_timestamps_are_timezone_aware(adapter_config, data_config):
    """All bar timestamps should be timezone-aware."""
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    bars = list(adapter.read_bars(data_config))

    # Check first 100 bars
    for bar in bars[:100]:
        assert bar.ts.tzinfo is not None, f"Timestamp not tz-aware for {bar.symbol}"
        assert str(bar.ts.tzinfo) in [
            "America/New_York",
            "EST",
            "EDT",
        ], f"Wrong timezone for {bar.symbol}"


def test_adapter_no_adjustments_without_schema(adapter_config, bar_schema):
    """Adapter should skip adjustments if adjustment_schema is None."""
    config = DataConfig(
        timezone="America/New_York",
        frequency="1d",
        bar_schema=bar_schema,
        adjustment_schema=None,  # No adjustment schema
    )
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCAdapter(adapter_config, instrument)
    adjustments = list(adapter.read_adjustments(config))

    # Should be empty when no adjustment schema provided
    assert len(adjustments) == 0
