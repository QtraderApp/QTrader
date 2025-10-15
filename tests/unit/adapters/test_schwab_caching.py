"""
Tests for Schwab adapter caching functionality.

These tests verify:
- MetadataManager read/write operations
- Cache hit/miss scenarios
- Cache writing with Parquet files
- Atomic file operations
- Date range handling
"""

import json
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from qtrader.adapters.schwab import MetadataManager, SchwabOHLCAdapter
from qtrader.models.instrument import DataSource, Instrument, InstrumentType
from qtrader.models.vendors.schwab import SchwabBar


class TestMetadataManager:
    """Test metadata manager functionality."""

    def test_create_metadata_manager(self, tmp_path):
        """Test creating metadata manager."""
        manager = MetadataManager(tmp_path, "AAPL")

        assert manager.cache_root == tmp_path
        assert manager.symbol == "AAPL"
        assert manager.symbol_dir == tmp_path / "AAPL"
        assert manager.metadata_file == tmp_path / "AAPL" / ".metadata.json"
        assert manager.data_file == tmp_path / "AAPL" / "data.parquet"

    def test_read_metadata_no_file(self, tmp_path):
        """Test reading metadata when file doesn't exist."""
        manager = MetadataManager(tmp_path, "AAPL")

        metadata = manager.read_metadata()

        assert metadata is None

    def test_read_metadata_valid_file(self, tmp_path):
        """Test reading valid metadata file."""
        manager = MetadataManager(tmp_path, "AAPL")

        # Create symbol directory and metadata file
        manager.symbol_dir.mkdir(parents=True)

        metadata_content = {
            "symbol": "AAPL",
            "last_update": "2025-10-15T10:30:00Z",
            "date_range": {"start": "2019-01-01", "end": "2025-10-15"},
            "row_count": 1658,
            "frequency_type": "daily",
            "frequency": 1,
            "source": "schwab",
        }

        with open(manager.metadata_file, "w") as f:
            json.dump(metadata_content, f)

        metadata = manager.read_metadata()

        assert metadata is not None
        assert metadata["symbol"] == "AAPL"
        assert metadata["row_count"] == 1658
        assert metadata["date_range"]["start"] == "2019-01-01"

    def test_read_metadata_invalid_json(self, tmp_path):
        """Test reading metadata with invalid JSON."""
        manager = MetadataManager(tmp_path, "AAPL")

        # Create directory and invalid file
        manager.symbol_dir.mkdir(parents=True)
        manager.metadata_file.write_text("not valid json")

        metadata = manager.read_metadata()

        assert metadata is None

    def test_write_metadata(self, tmp_path):
        """Test writing metadata to file."""
        manager = MetadataManager(tmp_path, "AAPL")

        manager.write_metadata(
            start_date="2019-01-01",
            end_date="2025-10-15",
            row_count=1658,
            frequency_type="daily",
            frequency=1,
        )

        # Verify file exists
        assert manager.metadata_file.exists()

        # Verify content
        with open(manager.metadata_file) as f:
            metadata = json.load(f)

        assert metadata["symbol"] == "AAPL"
        assert metadata["date_range"]["start"] == "2019-01-01"
        assert metadata["date_range"]["end"] == "2025-10-15"
        assert metadata["row_count"] == 1658
        assert metadata["frequency_type"] == "daily"
        assert metadata["frequency"] == 1
        assert metadata["source"] == "schwab"
        assert "last_update" in metadata

    def test_write_metadata_creates_directory(self, tmp_path):
        """Test that write_metadata creates parent directory."""
        manager = MetadataManager(tmp_path, "AAPL")

        # Directory doesn't exist yet
        assert not manager.symbol_dir.exists()

        manager.write_metadata(start_date="2019-01-01", end_date="2025-10-15", row_count=100)

        # Directory should now exist
        assert manager.symbol_dir.exists()

    def test_write_metadata_atomic(self, tmp_path):
        """Test that metadata write is atomic (uses temp file)."""
        manager = MetadataManager(tmp_path, "AAPL")

        manager.write_metadata(start_date="2019-01-01", end_date="2025-10-15", row_count=100)

        # Temp file should not exist after successful write
        temp_file = manager.metadata_file.with_suffix(".tmp")
        assert not temp_file.exists()

        # Actual file should exist
        assert manager.metadata_file.exists()

    def test_cache_exists(self, tmp_path):
        """Test checking if cache exists."""
        manager = MetadataManager(tmp_path, "AAPL")

        # No cache initially
        assert not manager.cache_exists()

        # Create files
        manager.symbol_dir.mkdir(parents=True)
        manager.metadata_file.touch()
        manager.data_file.touch()

        # Now cache exists
        assert manager.cache_exists()

    def test_get_cached_date_range_no_cache(self, tmp_path):
        """Test getting date range when no cache exists."""
        manager = MetadataManager(tmp_path, "AAPL")

        date_range = manager.get_cached_date_range()

        assert date_range is None

    def test_get_cached_date_range_valid_cache(self, tmp_path):
        """Test getting date range from valid cache."""
        manager = MetadataManager(tmp_path, "AAPL")

        # Write metadata
        manager.write_metadata(start_date="2019-01-01", end_date="2025-10-15", row_count=1658)

        date_range = manager.get_cached_date_range()

        assert date_range is not None
        assert date_range == ("2019-01-01", "2025-10-15")


class TestSchwabAdapterCaching:
    """Test Schwab adapter caching functionality."""

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_adapter_with_cache_enabled(self, mock_oauth_manager, tmp_path):
        """Test adapter initialization with caching enabled."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        assert adapter.cache_root == tmp_path
        assert adapter.metadata_manager is not None
        assert adapter.metadata_manager.symbol == "AAPL"

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_adapter_without_cache(self, mock_oauth_manager):
        """Test adapter initialization without caching."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        assert adapter.cache_root is None
        assert adapter.metadata_manager is None

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_read_from_cache_no_cache_file(self, mock_oauth_manager, tmp_path):
        """Test reading from cache when no cache exists."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = adapter._read_from_cache("2020-01-01", "2020-12-31")

        assert bars is None

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_write_to_cache(self, mock_oauth_manager, tmp_path):
        """Test writing bars to cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        # Create sample bars
        bars = [
            SchwabBar(
                timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            ),
            SchwabBar(
                timestamp=datetime(2020, 1, 3, tzinfo=timezone.utc),
                open=103.0,
                high=107.0,
                low=102.0,
                close=106.0,
                volume=1200000,
            ),
        ]

        adapter._write_to_cache(bars, "daily", 1)

        # Verify data file exists
        assert adapter.metadata_manager is not None
        assert adapter.metadata_manager.data_file.exists()

        # Verify metadata file exists
        assert adapter.metadata_manager.metadata_file.exists()

        # Verify metadata content
        metadata = adapter.metadata_manager.read_metadata()
        assert metadata is not None
        assert metadata["symbol"] == "AAPL"
        assert metadata["row_count"] == 2
        assert metadata["date_range"]["start"] == "2020-01-02"
        assert metadata["date_range"]["end"] == "2020-01-03"

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_read_from_cache_success(self, mock_oauth_manager, tmp_path):
        """Test successfully reading bars from cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        # Write bars to cache
        original_bars = [
            SchwabBar(
                timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            ),
            SchwabBar(
                timestamp=datetime(2020, 1, 3, tzinfo=timezone.utc),
                open=103.0,
                high=107.0,
                low=102.0,
                close=106.0,
                volume=1200000,
            ),
        ]

        adapter._write_to_cache(original_bars, "daily", 1)

        # Read bars from cache
        cached_bars = adapter._read_from_cache("2020-01-01", "2020-12-31")

        assert cached_bars is not None
        assert len(cached_bars) == 2

        # Verify first bar
        assert cached_bars[0].open == 100.0
        assert cached_bars[0].close == 103.0
        assert cached_bars[0].volume == 1000000

        # Verify second bar
        assert cached_bars[1].open == 103.0
        assert cached_bars[1].close == 106.0
        assert cached_bars[1].volume == 1200000

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_cache_miss_uses_api(self, mock_get, mock_oauth_manager, tmp_path):
        """Test that read_bars uses API when cache misses."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candles": [
                {
                    "datetime": 1577923200000,  # 2020-01-02
                    "open": 100.0,
                    "high": 105.0,
                    "low": 99.0,
                    "close": 103.0,
                    "volume": 1000000,
                }
            ],
            "symbol": "AAPL",
            "empty": False,
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        # Read bars (should use API)
        bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

        assert len(bars) == 1
        assert bars[0].open == 100.0
        assert bars[0].close == 103.0

        # Verify API was called
        assert mock_get.called

        # Verify cache was written
        assert adapter.metadata_manager is not None
        assert adapter.metadata_manager.cache_exists()

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_read_bars_cache_hit_skips_api(self, mock_oauth_manager, tmp_path):
        """Test that read_bars uses cache when available."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        # Pre-populate cache
        cached_bars = [
            SchwabBar(
                timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            ),
        ]

        adapter._write_to_cache(cached_bars, "daily", 1)

        # Read bars (should use cache)
        with patch("requests.Session.get") as mock_get:
            bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

            # Should not call API
            assert not mock_get.called

        assert len(bars) == 1
        assert bars[0].open == 100.0

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_write_to_cache_empty_bars(self, mock_oauth_manager, tmp_path):
        """Test writing empty bars list to cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        # Write empty list
        adapter._write_to_cache([], "daily", 1)

        # Cache should not be created
        assert adapter.metadata_manager is not None
        assert not adapter.metadata_manager.cache_exists()
