"""Tests for CSV adapter."""

from decimal import Decimal
from pathlib import Path

import pytest

from qtrader.adapters.csv_adapter import CSVAdapter
from qtrader.config.data_config import BarSchemaConfig, DataConfig
from qtrader.models.bar import DataMode


@pytest.fixture
def csv_path():
    """Path to CSV sample data."""
    return Path("data/csv")


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


def test_csv_adapter_can_read_directory(csv_path):
    """CSV adapter should detect CSV directory."""
    if not csv_path.exists():
        pytest.skip("CSV sample data not available")

    adapter = CSVAdapter()
    assert adapter.can_read(csv_path) is True


def test_csv_adapter_cannot_read_missing_path():
    """CSV adapter should return False for missing path."""
    adapter = CSVAdapter()
    assert adapter.can_read(Path("nonexistent/path")) is False


def test_csv_adapter_schema_version():
    """CSV adapter should report schema version."""
    adapter = CSVAdapter()
    assert adapter.schema_version() == "csv-v1.0"


def test_csv_adapter_data_mode():
    """CSV adapter should declare data mode."""
    adapter = CSVAdapter()
    assert adapter.get_data_mode() == DataMode.ADJUSTED


def test_csv_adapter_reads_bars(csv_path, data_config):
    """CSV adapter should load bars from CSV files (OHLCV only)."""
    if not csv_path.exists():
        pytest.skip("CSV sample data not available")

    adapter = CSVAdapter()
    bars = list(adapter.read_bars(csv_path, data_config))

    # Should have 3 files × 1258 lines = 3774 bars
    assert len(bars) == 3774

    # Check types
    first_bar = bars[0]
    assert isinstance(first_bar.open, Decimal)
    assert isinstance(first_bar.close, Decimal)
    assert isinstance(first_bar.volume, int)
    # Bar should NOT have adjustment fields
    assert not hasattr(first_bar, "adj_reason")


def test_csv_adapter_matches_parquet_data(csv_path, data_config):
    """CSV data should match parquet data (sanity check)."""
    if not csv_path.exists():
        pytest.skip("CSV sample data not available")

    adapter = CSVAdapter()
    bars = list(adapter.read_bars(csv_path, data_config))

    # Find AAPL first bar
    aapl_bars = [b for b in bars if b.symbol == "AAPL"]
    assert len(aapl_bars) == 1258

    first_bar = aapl_bars[0]
    # Should match parquet data (OHLCV only)
    assert first_bar.close == Decimal("157.9200")


def test_csv_adapter_can_read_single_file(tmp_path, bar_schema):
    """CSV adapter should read a single CSV file."""
    # Create a temporary CSV file
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(
        "TradeDate,Ticker,Open,High,Low,Close,MarketHoursVolume\n"
        "2023-01-01,TEST,100.0,105.0,99.0,102.0,1000000\n"
    )

    config = DataConfig(
        timezone="America/New_York",
        frequency="1d",
        bar_schema=bar_schema,
    )

    adapter = CSVAdapter()
    assert adapter.can_read(csv_file) is True

    bars = list(adapter.read_bars(csv_file, config))
    assert len(bars) == 1
    assert bars[0].symbol == "TEST"
    assert bars[0].close == Decimal("102.0000")
