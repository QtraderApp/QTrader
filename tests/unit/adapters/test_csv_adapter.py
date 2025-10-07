"""Tests for CSV adapter."""

from decimal import Decimal
from pathlib import Path

import pytest

from qtrader.adapters.csv_adapter import CSVAdapter
from qtrader.config.data_config import BarSchemaConfig, DataConfig
from qtrader.models.bar import DataMode
from qtrader.models.instrument import DataSource, Instrument, InstrumentType


@pytest.fixture
def csv_path():
    """Path to CSV sample data."""
    return Path("data/csv")


@pytest.fixture
def adapter_config(csv_path):
    """Configuration for CSV adapter."""
    return {
        "root_path": str(csv_path),
    }


@pytest.fixture
def instrument_aapl():
    """AAPL instrument for testing."""
    return Instrument("AAPL", InstrumentType.EQUITY, DataSource.CSV_FILE)


@pytest.fixture
def bar_schema():
    """Bar schema config for CSV data."""
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
def data_config(bar_schema):
    """Data configuration for CSV adapter."""
    return DataConfig(
        timezone="America/New_York",
        frequency="1d",
        bar_schema=bar_schema,
    )


def test_csv_adapter_can_read_directory(adapter_config, instrument_aapl, csv_path):
    """CSV adapter should detect CSV directory."""
    if not csv_path.exists():
        pytest.skip("CSV sample data not available")

    adapter = CSVAdapter(adapter_config, instrument_aapl)
    assert adapter.can_read() is True


def test_csv_adapter_cannot_read_missing_path(instrument_aapl):
    """CSV adapter should return False for missing path."""
    bad_config = {"root_path": "nonexistent/path"}
    adapter = CSVAdapter(bad_config, instrument_aapl)
    assert adapter.can_read() is False


def test_csv_adapter_schema_version(adapter_config, instrument_aapl):
    """CSV adapter should report schema version."""
    adapter = CSVAdapter(adapter_config, instrument_aapl)
    assert adapter.schema_version() == "csv-v1.0"


def test_csv_adapter_data_mode(adapter_config, instrument_aapl):
    """CSV adapter should declare data mode."""
    adapter = CSVAdapter(adapter_config, instrument_aapl)
    assert adapter.get_data_mode() == DataMode.ADJUSTED


def test_csv_adapter_reads_bars(adapter_config, data_config, csv_path):
    """CSV adapter should load bars with 3 price series."""
    if not csv_path.exists():
        pytest.skip("CSV sample data not available")

    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.CSV_FILE)
    adapter = CSVAdapter(adapter_config, instrument)
    bars = list(adapter.read_bars(data_config))

    # Should have 1258 bars for AAPL
    assert len(bars) == 1258

    # Check Bar structure with 3 price series
    first_bar = bars[0]
    assert hasattr(first_bar, "unadjusted")
    assert hasattr(first_bar, "capital_adjusted")
    assert hasattr(first_bar, "total_return")

    # Check types in total_return series (should be the adjusted data)
    assert isinstance(first_bar.total_return.open, Decimal)
    assert isinstance(first_bar.total_return.close, Decimal)
    assert isinstance(first_bar.total_return.volume, int)


def test_csv_adapter_matches_parquet_data(adapter_config, data_config, csv_path):
    """CSV data should match parquet data (sanity check)."""
    if not csv_path.exists():
        pytest.skip("CSV sample data not available")

    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.CSV_FILE)
    adapter = CSVAdapter(adapter_config, instrument)
    bars = list(adapter.read_bars(data_config))

    # Should have 1258 bars for AAPL
    assert len(bars) == 1258

    first_bar = bars[0]
    # Should match parquet data - check total_return series (adjusted)
    assert first_bar.total_return.close == Decimal("157.9200")


def test_csv_adapter_can_read_single_file(tmp_path, bar_schema):
    """CSV adapter should read a single CSV file."""
    # Create a temporary CSV file
    csv_file = tmp_path / "TEST.csv"
    csv_file.write_text(
        "TradeDate,Ticker,Open,High,Low,Close,MarketHoursVolume\n2023-01-01,TEST,100.0,105.0,99.0,102.0,1000000\n"
    )

    config = DataConfig(
        timezone="America/New_York",
        frequency="1d",
        bar_schema=bar_schema,
    )

    adapter_config = {"root_path": str(tmp_path)}
    instrument = Instrument("TEST", InstrumentType.EQUITY, DataSource.CSV_FILE)
    adapter = CSVAdapter(adapter_config, instrument)
    assert adapter.can_read() is True

    bars = list(adapter.read_bars(config))
    assert len(bars) == 1
    assert bars[0].symbol == "TEST"
    assert bars[0].total_return.close == Decimal("102.0000")
