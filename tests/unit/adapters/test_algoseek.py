"""
Tests for AlgoseekOHLCVendorAdapter.

These tests verify the vendor adapter's ability to load raw Algoseek data
from parquet files and parse it into AlgoseekBar objects without performing
any adjustments or transformations.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from qtrader.models.instrument import DataSource, Instrument, InstrumentType
from qtrader.models.vendors.algoseek import AlgoseekBar
from qtrader.services.data.adapters.algoseek import AlgoseekOHLCVendorAdapter


class TestAlgoseekOHLCVendorAdapterInitialization:
    """Test adapter initialization and configuration."""

    def test_create_adapter_with_valid_config(self, tmp_path):
        """Test creating adapter with valid configuration."""
        # Create symbol map
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        # Create data directory
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "root_path": str(data_dir),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)

        adapter = AlgoseekOHLCVendorAdapter(config, instrument)

        assert adapter.instrument == instrument
        assert adapter.secid == 33449
        assert adapter.root_path == data_dir

    def test_create_adapter_missing_config_keys(self, tmp_path):
        """Test creating adapter with missing configuration keys."""
        config = {"root_path": str(tmp_path)}  # Missing path_template and symbol_map

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)

        with pytest.raises(ValueError, match="Missing required config keys"):
            AlgoseekOHLCVendorAdapter(config, instrument)

    def test_create_adapter_symbol_map_not_found(self, tmp_path):
        """Test creating adapter when symbol map file doesn't exist."""
        config = {
            "root_path": str(tmp_path),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(tmp_path / "nonexistent.csv"),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)

        with pytest.raises(FileNotFoundError, match="Symbol map not found"):
            AlgoseekOHLCVendorAdapter(config, instrument)

    def test_create_adapter_symbol_not_in_map(self, tmp_path):
        """Test creating adapter for symbol not in symbol map."""
        # Create symbol map without MSFT
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        config = {
            "root_path": str(data_dir),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("MSFT", InstrumentType.EQUITY, DataSource.ALGOSEEK)

        with pytest.raises(ValueError, match="Symbol not found in symbol map"):
            AlgoseekOHLCVendorAdapter(config, instrument)


class TestAlgoseekOHLCVendorAdapterReadBars:
    """Test reading bars from parquet files."""

    @pytest.fixture
    def adapter_with_data(self, tmp_path):
        """Create adapter with sample parquet data."""
        # Create symbol map
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        # Create data directory with Hive partitioning
        data_dir = tmp_path / "data" / "SecId=33449"
        data_dir.mkdir(parents=True)

        # Create sample parquet data
        data = pd.DataFrame(
            {
                "TradeDate": pd.to_datetime(
                    [
                        "2020-08-10",
                        "2020-08-11",
                        "2020-08-12",
                        "2020-08-31",  # Split date
                        "2020-09-01",
                        "2020-09-02",
                    ]
                ),
                "Ticker": ["AAPL"] * 6,
                "Open": [115.0, 116.0, 117.0, 127.0, 126.0, 125.0],
                "High": [116.0, 117.0, 118.0, 128.0, 127.0, 126.0],
                "Low": [114.0, 115.0, 116.0, 126.0, 125.0, 124.0],
                "Close": [115.5, 116.5, 117.5, 127.5, 126.5, 125.5],
                "MarketHoursVolume": [1000000, 1100000, 1200000, 1300000, 1400000, 1500000],
                "CumulativePriceFactor": [1.0, 1.0, 1.0, 0.25, 0.25, 0.25],  # 4:1 split
                "CumulativeVolumeFactor": [1.0, 1.0, 1.0, 4.0, 4.0, 4.0],
                "AdjustmentFactor": [None, None, None, 0.25, None, None],
                "AdjustmentReason": [None, None, None, "Subdiv", None, None],
            }
        )

        # Write to parquet
        parquet_path = data_dir / "data.parquet"
        table = pa.Table.from_pandas(data)
        pq.write_table(table, parquet_path)

        # Create config
        config = {
            "root_path": str(tmp_path / "data"),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        adapter = AlgoseekOHLCVendorAdapter(config, instrument)

        return adapter

    def test_read_bars_basic(self, adapter_with_data):
        """Test reading bars from parquet file."""
        bars = list(adapter_with_data.read_bars("2020-08-01", "2020-09-30"))

        assert len(bars) == 6
        assert all(isinstance(bar, AlgoseekBar) for bar in bars)

        # Check first bar
        first_bar = bars[0]
        assert first_bar.Ticker == "AAPL"
        assert first_bar.Close == 115.5
        assert first_bar.MarketHoursVolume == 1000000

    def test_read_bars_date_filtering(self, adapter_with_data):
        """Test date range filtering."""
        # Only August data
        bars = list(adapter_with_data.read_bars("2020-08-01", "2020-08-31"))
        assert len(bars) == 4

        # Only September data
        bars = list(adapter_with_data.read_bars("2020-09-01", "2020-09-30"))
        assert len(bars) == 2

        # Narrow range
        bars = list(adapter_with_data.read_bars("2020-08-11", "2020-08-12"))
        assert len(bars) == 2

    def test_read_bars_chronological_order(self, adapter_with_data):
        """Test bars returned in chronological order."""
        bars = list(adapter_with_data.read_bars("2020-08-01", "2020-09-30"))

        dates = [bar.TradeDate.date() for bar in bars]
        assert dates == sorted(dates)

    def test_read_bars_split_adjustment_fields(self, adapter_with_data):
        """Test that split adjustment fields are preserved."""
        bars = list(adapter_with_data.read_bars("2020-08-01", "2020-09-30"))

        # Bar on split date (2020-08-31)
        split_bar = bars[3]
        assert split_bar.AdjustmentFactor == 0.25
        assert split_bar.AdjustmentReason == "Subdiv"
        assert split_bar.CumulativePriceFactor == 0.25
        assert split_bar.CumulativeVolumeFactor == 4.0

        # Bar before split
        pre_split_bar = bars[2]
        assert pre_split_bar.CumulativeVolumeFactor == 1.0

        # Bar after split
        post_split_bar = bars[4]
        assert post_split_bar.AdjustmentFactor is None
        assert post_split_bar.CumulativeVolumeFactor == 4.0

    def test_read_bars_no_data_in_range(self, adapter_with_data):
        """Test reading bars when no data in date range."""
        bars = list(adapter_with_data.read_bars("2019-01-01", "2019-12-31"))
        assert len(bars) == 0

    def test_read_bars_data_path_not_found(self, tmp_path):
        """Test reading bars when data path doesn't exist."""
        # Create symbol map
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        # Don't create data directory
        config = {
            "root_path": str(tmp_path / "nonexistent"),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        adapter = AlgoseekOHLCVendorAdapter(config, instrument)

        with pytest.raises(FileNotFoundError, match="Data source not found"):
            list(adapter.read_bars("2020-01-01", "2020-12-31"))

    def test_read_bars_no_parquet_files(self, tmp_path):
        """Test reading bars when directory exists but has no parquet files."""
        # Create symbol map
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        # Create empty data directory
        data_dir = tmp_path / "data" / "SecId=33449"
        data_dir.mkdir(parents=True)

        config = {
            "root_path": str(tmp_path / "data"),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        adapter = AlgoseekOHLCVendorAdapter(config, instrument)

        with pytest.raises(FileNotFoundError, match="No parquet files found"):
            list(adapter.read_bars("2020-01-01", "2020-12-31"))


class TestAlgoseekOHLCVendorAdapterDateRange:
    """Test getting available date range."""

    def test_get_available_date_range_with_data(self, tmp_path):
        """Test getting date range when data exists."""
        # Create symbol map
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        # Create data directory
        data_dir = tmp_path / "data" / "SecId=33449"
        data_dir.mkdir(parents=True)

        # Create sample data
        data = pd.DataFrame(
            {
                "TradeDate": pd.to_datetime(["2020-01-01", "2020-06-15", "2020-12-31"]),
                "Ticker": ["AAPL"] * 3,
                "Open": [100.0] * 3,
                "High": [101.0] * 3,
                "Low": [99.0] * 3,
                "Close": [100.5] * 3,
                "MarketHoursVolume": [1000000] * 3,
                "CumulativePriceFactor": [1.0] * 3,
                "CumulativeVolumeFactor": [1.0] * 3,
                "AdjustmentFactor": [None] * 3,
                "AdjustmentReason": [None] * 3,
            }
        )

        parquet_path = data_dir / "data.parquet"
        table = pa.Table.from_pandas(data)
        pq.write_table(table, parquet_path)

        # Create adapter
        config = {
            "root_path": str(tmp_path / "data"),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        adapter = AlgoseekOHLCVendorAdapter(config, instrument)

        min_date, max_date = adapter.get_available_date_range()

        assert min_date == "2020-01-01"
        assert max_date == "2020-12-31"

    def test_get_available_date_range_no_data(self, tmp_path):
        """Test getting date range when no data exists."""
        # Create symbol map
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        # Don't create data directory
        config = {
            "root_path": str(tmp_path / "nonexistent"),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        adapter = AlgoseekOHLCVendorAdapter(config, instrument)

        min_date, max_date = adapter.get_available_date_range()

        assert min_date is None
        assert max_date is None


class TestAlgoseekOHLCVendorAdapterIntegration:
    """Integration tests with real-world scenarios."""

    def test_adapter_with_multiple_parquet_files(self, tmp_path):
        """Test adapter with multiple parquet files in directory."""
        # Create symbol map
        symbol_map_path = tmp_path / "symbol_map.csv"
        pd.DataFrame({"Symbol": ["AAPL"], "SecId": [33449]}).to_csv(symbol_map_path, index=False)

        # Create data directory
        data_dir = tmp_path / "data" / "SecId=33449"
        data_dir.mkdir(parents=True)

        # Create multiple parquet files
        for i, month in enumerate(["01", "02", "03"], start=1):
            data = pd.DataFrame(
                {
                    "TradeDate": pd.to_datetime([f"2020-{month}-01", f"2020-{month}-15"]),
                    "Ticker": ["AAPL"] * 2,
                    "Open": [100.0 + i] * 2,
                    "High": [101.0 + i] * 2,
                    "Low": [99.0 + i] * 2,
                    "Close": [100.5 + i] * 2,
                    "MarketHoursVolume": [1000000] * 2,
                    "CumulativePriceFactor": [1.0] * 2,
                    "CumulativeVolumeFactor": [1.0] * 2,
                    "AdjustmentFactor": [None] * 2,
                    "AdjustmentReason": [None] * 2,
                }
            )

            parquet_path = data_dir / f"data_{month}.parquet"
            table = pa.Table.from_pandas(data)
            pq.write_table(table, parquet_path)

        # Create adapter
        config = {
            "root_path": str(tmp_path / "data"),
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": str(symbol_map_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        adapter = AlgoseekOHLCVendorAdapter(config, instrument)

        # Read all bars
        bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

        # Should have 6 bars total (2 per file × 3 files)
        assert len(bars) == 6

        # Check chronological order across files
        dates = [bar.TradeDate.date() for bar in bars]
        assert dates == sorted(dates)
