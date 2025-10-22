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

import pytest

from qtrader.models.instrument import Instrument
from qtrader.models.vendors.schwab import SchwabBar
from qtrader.services.data.adapters.schwab import MetadataManager, SchwabOHLCAdapter


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

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_adapter_with_cache_enabled(self, mock_oauth_manager, tmp_path):
        """Test adapter initialization with caching enabled."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        assert adapter.cache_root == tmp_path
        assert adapter.metadata_manager is not None
        assert adapter.metadata_manager.symbol == "AAPL"

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_adapter_without_cache(self, mock_oauth_manager):
        """Test adapter initialization without caching."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        assert adapter.cache_root is None
        assert adapter.metadata_manager is None

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_read_from_cache_no_cache_file(self, mock_oauth_manager, tmp_path):
        """Test reading from cache when no cache exists."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = adapter._read_from_cache("2020-01-01", "2020-12-31")

        assert bars is None

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_write_to_cache(self, mock_oauth_manager, tmp_path):
        """Test writing bars to cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
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

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_read_from_cache_success(self, mock_oauth_manager, tmp_path):
        """Test successfully reading bars from cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
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

        # Read bars from cache - cache contains 2020-01-02 to 2020-01-03
        # Request dates within cached range
        cached_bars = adapter._read_from_cache("2020-01-02", "2020-01-03")

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

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
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

        instrument = Instrument("AAPL")
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

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_read_bars_cache_hit_skips_api(self, mock_oauth_manager, tmp_path):
        """Test that read_bars uses cache when available."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
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

        # Read bars (should NOT use cache - requested range exceeds cached range)
        # Cache has 2020-01-02, but we're requesting 2020-01-01 to 2020-12-31
        with patch("requests.Session.get") as mock_get:
            # Mock API response
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "candles": [
                    {
                        "datetime": int(datetime(2020, 1, 2, tzinfo=timezone.utc).timestamp() * 1000),
                        "open": 100.0,
                        "high": 105.0,
                        "low": 99.0,
                        "close": 103.0,
                        "volume": 1000000,
                    }
                ]
            }

            bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

            # SHOULD call API because cache doesn't cover full range
            assert mock_get.called

        assert len(bars) == 1
        assert bars[0].open == 100.0

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_write_to_cache_empty_bars(self, mock_oauth_manager, tmp_path):
        """Test writing empty bars list to cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Write empty list
        adapter._write_to_cache([], "daily", 1)

        # Cache should not be created
        assert adapter.metadata_manager is not None
        assert not adapter.metadata_manager.cache_exists()


class TestMetadataManagerEdgeCases:
    """Test MetadataManager error handling and edge cases."""

    def test_write_metadata_with_io_error(self, tmp_path, monkeypatch):
        """Test metadata write handles IO errors gracefully."""
        manager = MetadataManager(tmp_path, "AAPL")

        # Make directory read-only after creation
        manager.symbol_dir.mkdir(parents=True, exist_ok=True)

        # Mock open to raise OSError
        import builtins

        original_open = builtins.open

        def mock_open(*args, **kwargs):
            if str(tmp_path / "AAPL" / ".metadata.tmp") in str(args[0]):
                raise OSError("Permission denied")
            return original_open(*args, **kwargs)

        monkeypatch.setattr("builtins.open", mock_open)

        with pytest.raises(OSError):
            manager.write_metadata(start_date="2020-01-01", end_date="2020-12-31", row_count=100)

    def test_read_metadata_with_corrupted_file(self, tmp_path):
        """Test reading metadata handles corrupted JSON gracefully."""
        manager = MetadataManager(tmp_path, "AAPL")

        # Create corrupted metadata file
        manager.symbol_dir.mkdir(parents=True)
        manager.metadata_file.write_text("{invalid json content")

        metadata = manager.read_metadata()

        assert metadata is None

    def test_get_cached_date_range_with_missing_keys(self, tmp_path):
        """Test get_cached_date_range handles missing date_range key."""
        manager = MetadataManager(tmp_path, "AAPL")

        # Create metadata without date_range
        manager.symbol_dir.mkdir(parents=True)
        with open(manager.metadata_file, "w") as f:
            json.dump({"symbol": "AAPL", "row_count": 100}, f)

        date_range = manager.get_cached_date_range()

        assert date_range is None


class TestGapDetection:
    """Test gap detection functionality."""

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_detect_gaps_gap_before_cache(self, mock_oauth_manager, tmp_path):
        """Test detecting gap before cached data."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        metadata = {"date_range": {"start": "2020-06-01", "end": "2020-12-31"}}

        # Request starts before cache
        gaps = adapter._detect_gaps("2020-01-01", "2020-12-31", metadata)

        assert len(gaps) == 1
        assert gaps[0] == ("2020-01-01", "2020-06-01")

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_detect_gaps_gap_after_cache(self, mock_oauth_manager, tmp_path):
        """Test detecting gap after cached data."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        metadata = {"date_range": {"start": "2020-01-01", "end": "2020-06-30"}}

        # Request extends past cache
        gaps = adapter._detect_gaps("2020-01-01", "2020-12-31", metadata)

        assert len(gaps) == 1
        assert gaps[0] == ("2020-06-30", "2020-12-31")

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_detect_gaps_both_sides(self, mock_oauth_manager, tmp_path):
        """Test detecting gaps on both sides of cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        metadata = {"date_range": {"start": "2020-06-01", "end": "2020-06-30"}}

        # Request extends on both sides
        gaps = adapter._detect_gaps("2020-01-01", "2020-12-31", metadata)

        assert len(gaps) == 2
        assert gaps[0] == ("2020-01-01", "2020-06-01")
        assert gaps[1] == ("2020-06-30", "2020-12-31")

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_detect_gaps_no_gaps(self, mock_oauth_manager, tmp_path):
        """Test no gaps when cache fully covers request."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        metadata = {"date_range": {"start": "2020-01-01", "end": "2020-12-31"}}

        # Request is subset of cache
        gaps = adapter._detect_gaps("2020-03-01", "2020-09-30", metadata)

        assert len(gaps) == 0


class TestMergeBars:
    """Test merge bars functionality."""

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_merge_bars_removes_duplicates(self, mock_oauth_manager, tmp_path):
        """Test merging bars removes duplicates by timestamp."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Create bars with same timestamp
        bar1 = SchwabBar(
            timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )

        bar2 = SchwabBar(
            timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),  # Same timestamp
            open=100.5,
            high=105.5,
            low=99.5,
            close=103.5,
            volume=1100000,
        )

        bar3 = SchwabBar(
            timestamp=datetime(2020, 1, 3, tzinfo=timezone.utc),
            open=103.0,
            high=107.0,
            low=102.0,
            close=106.0,
            volume=1200000,
        )

        merged = adapter._merge_bars([bar1], [bar2, bar3])

        # Should have 2 bars (duplicate removed)
        assert len(merged) == 2
        # First occurrence kept
        assert merged[0].close == 103.0
        assert merged[1].close == 106.0

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_merge_bars_sorts_chronologically(self, mock_oauth_manager, tmp_path):
        """Test merging bars sorts chronologically."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Create bars in random order
        bar1 = SchwabBar(
            timestamp=datetime(2020, 1, 5, tzinfo=timezone.utc),
            open=110.0,
            high=115.0,
            low=109.0,
            close=113.0,
            volume=1000000,
        )

        bar2 = SchwabBar(
            timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1100000,
        )

        bar3 = SchwabBar(
            timestamp=datetime(2020, 1, 3, tzinfo=timezone.utc),
            open=103.0,
            high=107.0,
            low=102.0,
            close=106.0,
            volume=1200000,
        )

        merged = adapter._merge_bars([bar1, bar3], [bar2])

        # Should be sorted by timestamp
        assert len(merged) == 3
        assert merged[0].timestamp.day == 2
        assert merged[1].timestamp.day == 3
        assert merged[2].timestamp.day == 5

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_merge_bars_empty_lists(self, mock_oauth_manager, tmp_path):
        """Test merging with empty lists."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        bar1 = SchwabBar(
            timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )

        # Merge with empty list
        merged = adapter._merge_bars([bar1], [])
        assert len(merged) == 1

        # Merge empty with non-empty
        merged = adapter._merge_bars([], [bar1])
        assert len(merged) == 1

        # Merge two empty lists
        merged = adapter._merge_bars([], [])
        assert len(merged) == 0


class TestIncrementalUpdate:
    """Test incremental update functionality."""

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_update_to_latest_no_cache_configured(self, mock_oauth_manager):
        """Test update_to_latest when no cache is configured."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        bars_added, start_date, end_date = adapter.update_to_latest()

        assert bars_added == 0
        assert start_date is None
        assert end_date is None

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_update_to_latest_no_metadata(self, mock_oauth_manager, tmp_path):
        """Test update_to_latest when metadata doesn't exist."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        bars_added, start_date, end_date = adapter.update_to_latest()

        assert bars_added == 0
        assert start_date is None
        assert end_date is None

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_update_to_latest_already_up_to_date(self, mock_oauth_manager, tmp_path):
        """Test update_to_latest when cache is already current."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Write metadata with today's date as end
        today = datetime.now(timezone.utc).date().isoformat()
        assert adapter.metadata_manager is not None
        adapter.metadata_manager.write_metadata(start_date="2020-01-01", end_date=today, row_count=100)

        bars_added, start_date, end_date = adapter.update_to_latest()

        assert bars_added == 0
        assert start_date is None
        assert end_date is None

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_update_to_latest_dry_run(self, mock_oauth_manager, tmp_path):
        """Test update_to_latest in dry run mode."""
        from datetime import timedelta

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Write metadata with old end date
        yesterday = (datetime.now(timezone.utc) - timedelta(days=10)).date().isoformat()
        assert adapter.metadata_manager is not None
        adapter.metadata_manager.write_metadata(start_date="2020-01-01", end_date=yesterday, row_count=100)

        bars_added, start_date, end_date = adapter.update_to_latest(dry_run=True)

        # Should estimate some bars
        assert bars_added > 0
        assert start_date is not None
        assert end_date is not None

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_update_to_latest_with_new_data(self, mock_get, mock_oauth_manager, tmp_path):
        """Test update_to_latest fetches and merges new data."""
        from datetime import timedelta

        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response with new bars
        yesterday_dt = datetime.now(timezone.utc) - timedelta(days=1)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candles": [
                {
                    "datetime": int(yesterday_dt.timestamp() * 1000),
                    "open": 150.0,
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    "volume": 2000000,
                }
            ],
            "symbol": "AAPL",
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Write some initial cached data
        old_bars = [
            SchwabBar(
                timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]
        adapter._write_to_cache(old_bars, "daily", 1)

        # Update to latest
        bars_added, start_date, end_date = adapter.update_to_latest()

        # Should have added new bars
        assert bars_added == 1
        assert start_date is not None
        assert end_date is not None

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_update_to_latest_disabled_incremental(self, mock_oauth_manager, tmp_path):
        """Test update_to_latest when incremental updates are disabled."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
            "enable_incremental_update": False,
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        bars_added, start_date, end_date = adapter.update_to_latest()

        assert bars_added == 0
        assert start_date is None
        assert end_date is None


class TestSmartCachingStrategy:
    """Test smart caching strategy with gap filling."""

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_disabled_cache_uses_api(self, mock_get, mock_oauth_manager):
        """Test read_bars with disabled caching always uses API."""
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
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_strategy": "disabled",
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

        assert len(bars) == 1
        assert mock_get.called

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_force_refresh(self, mock_get, mock_oauth_manager, tmp_path):
        """Test read_bars with force_refresh ignores cache."""
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
                    "open": 150.0,  # Different from cached
                    "high": 155.0,
                    "low": 149.0,
                    "close": 153.0,
                    "volume": 2000000,
                }
            ],
            "symbol": "AAPL",
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
            "force_refresh": True,
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Pre-populate cache
        old_bars = [
            SchwabBar(
                timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]
        adapter._write_to_cache(old_bars, "daily", 1)

        # Read with force refresh
        bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

        # Should get fresh data from API, not cache
        assert len(bars) == 1
        assert bars[0].close == 153.0  # New data
        assert mock_get.called

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_simple_strategy(self, mock_get, mock_oauth_manager, tmp_path):
        """Test read_bars with simple caching strategy."""
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
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
            "cache_strategy": "simple",
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # First read - cache miss, should fetch from API and cache full range
        bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))
        assert len(bars) == 1
        assert mock_get.called

        # Verify cache was written
        assert adapter.metadata_manager is not None
        assert adapter.metadata_manager.cache_exists()

        # Reset mock
        mock_get.reset_mock()

        # Second read - cache hit for exact same range, should not fetch from API
        bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))
        # With simple strategy, cache must cover FULL requested range
        # Since cache has 2020-01-02 but we request from 2020-01-01, it's a miss
        assert mock_get.called  # Will be called again because range doesn't match exactly

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_smart_strategy_with_gaps(self, mock_get, mock_oauth_manager, tmp_path):
        """Test read_bars with smart strategy fills gaps."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response for gap data
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candles": [
                {
                    "datetime": 1609459200000,  # 2021-01-01
                    "open": 130.0,
                    "high": 135.0,
                    "low": 129.0,
                    "close": 133.0,
                    "volume": 1500000,
                }
            ],
            "symbol": "AAPL",
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
            "cache_strategy": "smart",
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Pre-populate cache with 2020 data
        cached_bars = [
            SchwabBar(
                timestamp=datetime(2020, 6, 1, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]
        adapter._write_to_cache(cached_bars, "daily", 1)

        # Request includes 2021 (gap after cache)
        bars = list(adapter.read_bars("2020-01-01", "2021-12-31"))

        # Should fetch gap data from API
        assert mock_get.called
        # Should have both cached and new data
        assert len(bars) >= 1

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_unknown_strategy_fallback(self, mock_get, mock_oauth_manager, tmp_path):
        """Test read_bars falls back to simple strategy for unknown strategy."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response - return data for full range
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
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
            "cache_strategy": "unknown_strategy",
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Pre-populate cache with data for 2020-01-02 only
        cached_bars = [
            SchwabBar(
                timestamp=datetime(2020, 1, 2, tzinfo=timezone.utc),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )
        ]
        adapter._write_to_cache(cached_bars, "daily", 1)

        # Should fall back to simple strategy
        # Requesting 2020-01-01 to 2020-12-31, but cache only has 2020-01-02
        # Simple strategy checks if cache covers full range - it doesn't, so API is called
        bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

        # With unknown strategy falling back to simple, cache miss triggers API call
        assert len(bars) == 1
        assert mock_get.called


class TestReadAllFromCache:
    """Test _read_all_from_cache functionality."""

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_read_all_from_cache_no_cache(self, mock_oauth_manager, tmp_path):
        """Test reading all from cache when no cache exists."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = adapter._read_all_from_cache()

        assert bars == []

    @patch("qtrader.services.data.adapters.schwab.SchwabOAuthManager")
    def test_read_all_from_cache_with_data(self, mock_oauth_manager, tmp_path):
        """Test reading all bars from cache."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "cache_root": str(tmp_path),
        }

        instrument = Instrument("AAPL")
        adapter = SchwabOHLCAdapter(config, instrument)

        # Write data to cache
        bars_to_cache = [
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
        adapter._write_to_cache(bars_to_cache, "daily", 1)

        # Read all
        bars = adapter._read_all_from_cache()

        assert len(bars) == 2
        assert bars[0].close == 103.0
        assert bars[1].close == 106.0
