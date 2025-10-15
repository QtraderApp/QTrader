"""
Tests for SchwabOHLCAdapter.

These tests verify the adapter's ability to:
- Authenticate with Schwab OAuth
- Fetch price history from Schwab API
- Parse JSON responses into SchwabBar objects
- Handle rate limiting and retries
- Handle errors gracefully
"""

import time
from unittest.mock import Mock, patch

import pytest
import requests

from qtrader.adapters.schwab import RateLimiter, SchwabOHLCAdapter
from qtrader.models.instrument import DataSource, Instrument, InstrumentType
from qtrader.models.vendors.schwab import SchwabBar


class TestRateLimiter:
    """Test token bucket rate limiter."""

    def test_create_rate_limiter(self):
        """Test creating rate limiter with default rate."""
        limiter = RateLimiter()
        assert limiter.max_tokens == 10.0
        assert limiter.refill_rate == 10.0
        assert limiter.tokens == 10.0

    def test_create_rate_limiter_custom_rate(self):
        """Test creating rate limiter with custom rate."""
        limiter = RateLimiter(requests_per_second=5.0)
        assert limiter.max_tokens == 5.0
        assert limiter.refill_rate == 5.0
        assert limiter.tokens == 5.0

    def test_acquire_token_with_available_tokens(self):
        """Test acquiring token when tokens are available."""
        limiter = RateLimiter(requests_per_second=10.0)

        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start

        # Should return immediately
        assert elapsed < 0.01
        assert limiter.tokens < 10.0  # One token consumed

    def test_acquire_token_blocks_when_depleted(self):
        """Test that acquire blocks when tokens are depleted."""
        limiter = RateLimiter(requests_per_second=10.0)

        # Deplete tokens
        for _ in range(10):
            limiter.acquire()

        # This should block briefly
        start = time.monotonic()
        limiter.acquire()
        elapsed = time.monotonic() - start

        # Should wait approximately 0.1 seconds (1/10 second)
        assert 0.05 < elapsed < 0.15  # Allow some tolerance

    def test_tokens_refill_over_time(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(requests_per_second=10.0)

        # Deplete some tokens
        for _ in range(5):
            limiter.acquire()

        tokens_after_use = limiter.tokens

        # Wait for refill
        time.sleep(0.2)  # 0.2 seconds = 2 tokens at 10/sec

        limiter._refill()

        # Should have more tokens now
        assert limiter.tokens > tokens_after_use


class TestSchwabOHLCAdapterInitialization:
    """Test adapter initialization and configuration."""

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_create_adapter_with_valid_config(self, mock_oauth_manager):
        """Test creating adapter with valid configuration."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)

        adapter = SchwabOHLCAdapter(config, instrument)

        assert adapter.instrument == instrument
        assert adapter.config == config
        assert mock_oauth_manager.called

    def test_create_adapter_missing_client_id(self):
        """Test creating adapter without client_id."""
        config = {
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)

        with pytest.raises(ValueError, match="Missing required config keys"):
            SchwabOHLCAdapter(config, instrument)

    def test_create_adapter_missing_client_secret(self):
        """Test creating adapter without client_secret."""
        config = {
            "client_id": "test_client_id",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)

        with pytest.raises(ValueError, match="Missing required config keys"):
            SchwabOHLCAdapter(config, instrument)

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_create_adapter_with_optional_config(self, mock_oauth_manager):
        """Test creating adapter with optional configuration."""
        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
            "redirect_uri": "https://example.com/callback",
            "manual_mode": True,
            "requests_per_second": 5.0,
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)

        adapter = SchwabOHLCAdapter(config, instrument)

        assert adapter.rate_limiter.max_tokens == 5.0
        assert mock_oauth_manager.called


class TestSchwabOHLCAdapterAPICall:
    """Test API call methods."""

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    def test_get_auth_headers(self, mock_oauth_manager):
        """Test getting authorization headers."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token_12345"
        mock_oauth_manager.return_value = mock_oauth_instance

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        headers = adapter._get_auth_headers()

        assert headers["Authorization"] == "Bearer test_token_12345"
        assert headers["Accept"] == "application/json"

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_call_api_success(self, mock_get, mock_oauth_manager):
        """Test successful API call."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candles": [], "symbol": "AAPL"}
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        result = adapter._call_api("/test", {"symbol": "AAPL"})

        assert result == {"candles": [], "symbol": "AAPL"}
        assert mock_get.called

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_call_api_retries_on_server_error(self, mock_get, mock_oauth_manager):
        """Test API call retries on server error."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API responses: fail twice, then succeed
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_fail.raise_for_status.side_effect = requests.HTTPError(response=mock_response_fail)

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"candles": []}

        mock_get.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success,
        ]

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        result = adapter._call_api("/test", {"symbol": "AAPL"}, max_retries=3)

        assert result == {"candles": []}
        assert mock_get.call_count == 3

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_call_api_no_retry_on_client_error(self, mock_get, mock_oauth_manager):
        """Test API call doesn't retry on client error (4xx)."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock 400 response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        with pytest.raises(requests.HTTPError):
            adapter._call_api("/test", {"symbol": "AAPL"}, max_retries=3)

        # Should only try once (no retries on 4xx)
        assert mock_get.call_count == 1


class TestSchwabOHLCAdapterReadBars:
    """Test reading bars from Schwab API."""

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_success(self, mock_get, mock_oauth_manager):
        """Test successfully reading bars from API."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response with candles
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candles": [
                {
                    "datetime": 1609459200000,  # 2021-01-01 00:00:00 UTC
                    "open": 132.43,
                    "high": 133.61,
                    "low": 131.72,
                    "close": 132.05,
                    "volume": 143301900,
                },
                {
                    "datetime": 1609545600000,  # 2021-01-02 00:00:00 UTC
                    "open": 132.05,
                    "high": 134.20,
                    "low": 131.90,
                    "close": 133.50,
                    "volume": 156789200,
                },
            ],
            "symbol": "AAPL",
            "empty": False,
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = list(adapter.read_bars("2021-01-01", "2021-01-02"))

        assert len(bars) == 2
        assert isinstance(bars[0], SchwabBar)
        assert bars[0].close == 132.05
        assert bars[1].close == 133.50

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_empty_response(self, mock_get, mock_oauth_manager):
        """Test reading bars when API returns empty candles."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response with no candles
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candles": [],
            "symbol": "AAPL",
            "empty": True,
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = list(adapter.read_bars("2021-01-01", "2021-01-02"))

        assert len(bars) == 0

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_skips_invalid_candles(self, mock_get, mock_oauth_manager):
        """Test reading bars skips invalid candles but continues."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response with one invalid candle
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "candles": [
                {
                    "datetime": 1609459200000,
                    "open": 132.43,
                    "high": 133.61,
                    "low": 131.72,
                    "close": 132.05,
                    "volume": 143301900,
                },
                {
                    # Invalid: missing required fields
                    "datetime": 1609545600000,
                    "open": 132.05,
                },
                {
                    "datetime": 1609632000000,
                    "open": 133.50,
                    "high": 135.00,
                    "low": 133.00,
                    "close": 134.50,
                    "volume": 120000000,
                },
            ],
            "symbol": "AAPL",
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = list(adapter.read_bars("2021-01-01", "2021-01-03"))

        # Should get 2 valid bars (skipped the invalid one)
        assert len(bars) == 2
        assert bars[0].close == 132.05
        assert bars[1].close == 134.50

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_read_bars_with_minute_frequency(self, mock_get, mock_oauth_manager):
        """Test reading bars with minute frequency."""
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
                    "datetime": 1609459200000,
                    "open": 132.43,
                    "high": 132.50,
                    "low": 132.40,
                    "close": 132.45,
                    "volume": 100000,
                },
            ],
            "symbol": "AAPL",
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        bars = list(adapter.read_bars("2021-01-01", "2021-01-01", frequency_type="minute", frequency=5))

        assert len(bars) == 1
        assert bars[0].close == 132.45


class TestSchwabOHLCAdapterDateRange:
    """Test getting available date range."""

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_get_available_date_range_success(self, mock_get, mock_oauth_manager):
        """Test getting available date range."""
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
                    "datetime": 1577836800000,  # 2020-01-01
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1000000,
                },
                {
                    "datetime": 1609459200000,  # 2021-01-01
                    "open": 130.0,
                    "high": 131.0,
                    "low": 129.0,
                    "close": 130.5,
                    "volume": 1500000,
                },
            ],
            "symbol": "AAPL",
        }
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        min_date, max_date = adapter.get_available_date_range()

        assert min_date == "2020-01-01"
        assert max_date == "2021-01-01"

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_get_available_date_range_empty(self, mock_get, mock_oauth_manager):
        """Test getting date range when no data available."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API response with no candles
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"candles": [], "symbol": "AAPL"}
        mock_get.return_value = mock_response

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        min_date, max_date = adapter.get_available_date_range()

        assert min_date is None
        assert max_date is None

    @patch("qtrader.adapters.schwab.SchwabOAuthManager")
    @patch("requests.Session.get")
    def test_get_available_date_range_api_error(self, mock_get, mock_oauth_manager):
        """Test getting date range when API fails."""
        # Mock OAuth manager
        mock_oauth_instance = Mock()
        mock_oauth_instance.get_access_token.return_value = "test_token"
        mock_oauth_manager.return_value = mock_oauth_instance

        # Mock API error
        mock_get.side_effect = requests.RequestException("API error")

        config = {
            "client_id": "test_client_id",
            "client_secret": "test_client_secret",
        }

        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        adapter = SchwabOHLCAdapter(config, instrument)

        min_date, max_date = adapter.get_available_date_range()

        assert min_date is None
        assert max_date is None
