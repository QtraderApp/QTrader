"""
Schwab adapter for price history data.

This module contains the adapter for Schwab's Price History API:
- SchwabOHLCAdapter: Fetches split-adjusted daily/intraday OHLC bars

The adapter is responsible for:
- OAuth authentication via SchwabOAuthManager
- Calling Schwab's Price History API endpoint
- Rate limiting (10 requests/second)
- Parsing API responses into SchwabBar objects
- Error handling and retries

Separation of concerns:
- Adapter: API communication and data retrieval
- Vendor models: Validate and structure data
- Data layer: Transform to canonical format
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import requests

from qtrader.auth.schwab_oauth import SchwabOAuthManager
from qtrader.config.logging_config import LoggerFactory
from qtrader.models.instrument import Instrument
from qtrader.models.vendors.schwab import SchwabBar

logger = LoggerFactory.get_logger()


class MetadataManager:
    """
    Manage cache metadata for Schwab data.

    Tracks cache state per symbol including date ranges, row counts,
    and last update timestamps. Metadata is stored as JSON files
    alongside the cached Parquet data.

    Metadata format:
        {
            "symbol": "AAPL",
            "last_update": "2025-10-15T10:30:00Z",
            "date_range": {
                "start": "2019-01-01",
                "end": "2025-10-15"
            },
            "row_count": 1658,
            "frequency_type": "daily",
            "frequency": 1,
            "source": "schwab"
        }
    """

    def __init__(self, cache_root: Path, symbol: str):
        """
        Initialize metadata manager.

        Args:
            cache_root: Root directory for cached data
            symbol: Stock symbol
        """
        self.cache_root = cache_root
        self.symbol = symbol
        self.symbol_dir = cache_root / symbol
        self.metadata_file = self.symbol_dir / ".metadata.json"
        self.data_file = self.symbol_dir / "data.parquet"

    def read_metadata(self) -> Optional[dict]:
        """
        Read metadata from file.

        Returns:
            Metadata dict or None if file doesn't exist
        """
        if not self.metadata_file.exists():
            return None

        try:
            with open(self.metadata_file) as f:
                data: dict = json.load(f)
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(
                "schwab_metadata.read_error",
                symbol=self.symbol,
                path=str(self.metadata_file),
                error=str(e),
            )
            return None

    def write_metadata(
        self,
        start_date: str,
        end_date: str,
        row_count: int,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> None:
        """
        Write metadata to file.

        Args:
            start_date: Start of date range (ISO format)
            end_date: End of date range (ISO format)
            row_count: Number of rows in cache
            frequency_type: Bar frequency type
            frequency: Bar frequency value
        """
        # Ensure directory exists
        self.symbol_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "symbol": self.symbol,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "date_range": {"start": start_date, "end": end_date},
            "row_count": row_count,
            "frequency_type": frequency_type,
            "frequency": frequency,
            "source": "schwab",
        }

        # Write atomically (temp file + rename)
        temp_file = self.metadata_file.with_suffix(".tmp")
        try:
            with open(temp_file, "w") as f:
                json.dump(metadata, f, indent=2)

            # Set secure permissions
            temp_file.chmod(0o644)

            # Atomic rename
            temp_file.replace(self.metadata_file)

            logger.debug(
                "schwab_metadata.written",
                symbol=self.symbol,
                start_date=start_date,
                end_date=end_date,
                row_count=row_count,
            )

        except OSError as e:
            logger.error(
                "schwab_metadata.write_error",
                symbol=self.symbol,
                error=str(e),
            )
            if temp_file.exists():
                temp_file.unlink()
            raise

    def cache_exists(self) -> bool:
        """Check if cache files exist."""
        return self.data_file.exists() and self.metadata_file.exists()

    def get_cached_date_range(self) -> Optional[tuple[str, str]]:
        """
        Get date range from cached data.

        Returns:
            (start_date, end_date) tuple or None if no cache
        """
        metadata = self.read_metadata()
        if metadata and "date_range" in metadata:
            date_range = metadata["date_range"]
            return (date_range["start"], date_range["end"])
        return None


class RateLimiter:
    """
    Token bucket rate limiter.

    Implements token bucket algorithm to limit requests per second.
    Ensures we stay within Schwab's 10 requests/second limit.

    Attributes:
        max_tokens: Maximum tokens in bucket (capacity)
        refill_rate: Tokens added per second
        tokens: Current token count
        last_refill: Last time tokens were added
    """

    def __init__(self, requests_per_second: float = 10.0):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests allowed per second
        """
        self.max_tokens = requests_per_second
        self.refill_rate = requests_per_second
        self.tokens = requests_per_second
        self.last_refill = time.monotonic()

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.max_tokens, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def acquire(self) -> None:
        """
        Acquire a token (blocks if necessary).

        If no tokens available, sleeps until one is available.
        """
        self._refill()

        if self.tokens < 1:
            sleep_time = (1 - self.tokens) / self.refill_rate
            logger.debug("rate_limiter.sleeping", sleep_seconds=sleep_time)
            time.sleep(sleep_time)
            self._refill()

        self.tokens -= 1


class SchwabOHLCAdapter:
    """
    Schwab OHLC adapter - fetches price history from Schwab API.

    This adapter is responsible ONLY for:
    - OAuth authentication via SchwabOAuthManager
    - Calling Schwab Price History API
    - Rate limiting (10 requests/second)
    - Parsing JSON responses to SchwabBar objects
    - Returning Iterator[SchwabBar] in chronological order

    Does NOT:
    - Perform additional price adjustments (data is already split-adjusted)
    - Transform to canonical format (done in DataLoader)
    - Apply business logic (done in backtest engine)

    Configuration:
        client_id: Schwab API key
        client_secret: Schwab API secret
        redirect_uri: OAuth redirect URI (default: https://127.0.0.1:8182)
        token_cache_path: Path to token cache (default: ~/.qtrader/schwab_tokens.json)
        manual_mode: Enable manual OAuth code entry (default: False)
        requests_per_second: Rate limit (default: 10.0)

    API Endpoint:
        GET /marketdata/v1/pricehistory
        Query params: symbol, periodType, period, frequencyType, frequency

    Examples:
        >>> config = {
        ...     "client_id": os.getenv("SCHWAB_API_KEY"),
        ...     "client_secret": os.getenv("SCHWAB_API_SECRET"),
        ... }
        >>> instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
        >>> adapter = SchwabOHLCAdapter(config, instrument)
        >>> bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))
        >>> print(f"Loaded {len(bars)} bars")

    Notes:
        - Uses OAuth 2.0 for authentication
        - Tokens cached and auto-refreshed
        - Rate limited to 10 req/sec
        - Returns split-adjusted data only
    """

    # API configuration
    BASE_URL = "https://api.schwabapi.com"
    PRICE_HISTORY_ENDPOINT = "/marketdata/v1/pricehistory"

    def __init__(self, config: dict, instrument: Instrument):
        """
        Initialize Schwab OHLC adapter.

        Args:
            config: Adapter configuration with OAuth credentials and cache settings
            instrument: Instrument to load data for

        Raises:
            ValueError: If required config keys missing
        """
        self.config = config
        self.instrument = instrument

        # Validate configuration
        required_keys = ["client_id", "client_secret"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        # Initialize cache (optional)
        self.cache_root: Optional[Path] = None
        self.metadata_manager: Optional[MetadataManager] = None

        if "cache_root" in config:
            self.cache_root = Path(config["cache_root"])
            self.metadata_manager = MetadataManager(self.cache_root, instrument.symbol)
            logger.info(
                "schwab_ohlc_adapter.cache_enabled",
                symbol=instrument.symbol,
                cache_root=str(self.cache_root),
            )

        # Initialize OAuth manager
        oauth_config = {
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
        }

        # Optional OAuth config
        if "redirect_uri" in config:
            oauth_config["redirect_uri"] = config["redirect_uri"]
        if "token_cache_path" in config:
            oauth_config["token_cache_path"] = Path(config["token_cache_path"])
        if "manual_mode" in config:
            oauth_config["manual_mode"] = config["manual_mode"]

        self.oauth_manager = SchwabOAuthManager(**oauth_config)

        # Initialize rate limiter
        requests_per_second = config.get("requests_per_second", 10.0)
        self.rate_limiter = RateLimiter(requests_per_second)

        # Initialize session for connection pooling
        self.session = requests.Session()

        logger.info(
            "schwab_ohlc_adapter.initialized",
            symbol=instrument.symbol,
            rate_limit=requests_per_second,
        )

    def _get_auth_headers(self) -> dict[str, str]:
        """
        Get authorization headers for API requests.

        Returns:
            Headers dict with Bearer token
        """
        token = self.oauth_manager.get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _call_api(self, endpoint: str, params: dict, max_retries: int = 3) -> dict:
        """
        Call Schwab API with rate limiting and retries.

        Args:
            endpoint: API endpoint path
            params: Query parameters
            max_retries: Maximum retry attempts

        Returns:
            JSON response as dict

        Raises:
            requests.HTTPError: On persistent API errors
            ValueError: On invalid response format
        """
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(max_retries):
            try:
                # Rate limit
                self.rate_limiter.acquire()

                # Get fresh auth headers
                headers = self._get_auth_headers()

                logger.debug(
                    "schwab_ohlc_adapter.api_call",
                    url=url,
                    params=params,
                    attempt=attempt + 1,
                )

                # Make request
                response = self.session.get(url, headers=headers, params=params, timeout=30)

                # Check for errors
                response.raise_for_status()

                # Parse JSON
                data: dict = response.json()

                logger.debug(
                    "schwab_ohlc_adapter.api_success",
                    status_code=response.status_code,
                    symbol=params.get("symbol"),
                )

                return data

            except requests.HTTPError as e:
                status_code = e.response.status_code if e.response else None

                logger.warning(
                    "schwab_ohlc_adapter.api_error",
                    status_code=status_code,
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )

                # Don't retry on client errors (4xx)
                if status_code and 400 <= status_code < 500:
                    raise

                # Retry on server errors (5xx) or network issues
                if attempt < max_retries - 1:
                    sleep_time = 2**attempt  # Exponential backoff
                    logger.debug("schwab_ohlc_adapter.retry_backoff", sleep_seconds=sleep_time)
                    time.sleep(sleep_time)
                    continue

                # Max retries exceeded
                raise

            except (requests.RequestException, ValueError) as e:
                logger.warning(
                    "schwab_ohlc_adapter.request_error",
                    attempt=attempt + 1,
                    max_retries=max_retries,
                    error=str(e),
                )

                if attempt < max_retries - 1:
                    sleep_time = 2**attempt
                    time.sleep(sleep_time)
                    continue

                raise

        raise RuntimeError(f"API call failed after {max_retries} attempts")

    def _read_from_cache(self, start_date: str, end_date: str) -> Optional[list[SchwabBar]]:
        """
        Read bars from cache.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of SchwabBar objects or None if cache miss
        """
        if not self.metadata_manager or not self.metadata_manager.cache_exists():
            return None

        try:
            # Read Parquet file
            df = pd.read_parquet(self.metadata_manager.data_file)

            # Filter by date range
            df_filtered = df[(df["trade_datetime"] >= start_date) & (df["trade_datetime"] <= end_date)]

            if df_filtered.empty:
                return None

            # Convert to SchwabBar objects
            bars = []
            for _, row in df_filtered.iterrows():
                # Convert timestamp milliseconds to datetime
                timestamp_dt = datetime.fromtimestamp(int(row["timestamp"]) / 1000, tz=timezone.utc)

                bar = SchwabBar(
                    timestamp=timestamp_dt,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row["volume"]),
                )
                bars.append(bar)

            logger.info(
                "schwab_ohlc_adapter.cache_hit",
                symbol=self.instrument.symbol,
                count=len(bars),
                start_date=start_date,
                end_date=end_date,
            )

            return bars

        except Exception as e:
            logger.warning(
                "schwab_ohlc_adapter.cache_read_error",
                symbol=self.instrument.symbol,
                error=str(e),
            )
            return None

    def _write_to_cache(
        self,
        bars: list[SchwabBar],
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> None:
        """
        Write bars to cache.

        Args:
            bars: List of SchwabBar objects
            frequency_type: Bar frequency type
            frequency: Bar frequency value
        """
        if not self.metadata_manager or not bars:
            return

        try:
            # Ensure cache directory exists
            self.metadata_manager.symbol_dir.mkdir(parents=True, exist_ok=True)

            # Convert bars to DataFrame
            data = []
            for bar in bars:
                # bar.timestamp is already a datetime object
                trade_dt = bar.timestamp
                # Convert back to Unix milliseconds for storage
                timestamp_ms = int(trade_dt.timestamp() * 1000)

                data.append(
                    {
                        "trade_datetime": trade_dt.strftime("%Y-%m-%d"),
                        "timestamp": timestamp_ms,
                        "open": bar.open,
                        "high": bar.high,
                        "low": bar.low,
                        "close": bar.close,
                        "volume": bar.volume,
                    }
                )

            df = pd.DataFrame(data)

            # Sort by datetime
            df = df.sort_values("trade_datetime")

            # Write atomically (temp file + rename)
            temp_file = self.metadata_manager.data_file.with_suffix(".tmp")
            df.to_parquet(temp_file, index=False)

            # Set secure permissions
            temp_file.chmod(0o644)

            # Atomic rename
            temp_file.replace(self.metadata_manager.data_file)

            # Update metadata
            start_date = df["trade_datetime"].min()
            end_date = df["trade_datetime"].max()
            row_count = len(df)

            self.metadata_manager.write_metadata(
                start_date=start_date,
                end_date=end_date,
                row_count=row_count,
                frequency_type=frequency_type,
                frequency=frequency,
            )

            logger.info(
                "schwab_ohlc_adapter.cache_written",
                symbol=self.instrument.symbol,
                count=row_count,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            logger.error(
                "schwab_ohlc_adapter.cache_write_error",
                symbol=self.instrument.symbol,
                error=str(e),
            )
            # Clean up temp file if it exists
            temp_file = self.metadata_manager.data_file.with_suffix(".tmp")
            if temp_file.exists():
                temp_file.unlink()

    def read_bars(
        self,
        start_date: str,
        end_date: str,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> Iterator[SchwabBar]:
        """
        Read bars from cache or Schwab API (cache-first strategy).

        This method implements intelligent caching:
        1. Check cache for requested date range
        2. If cache covers range → return cached data (fast path)
        3. If no cache → fetch from API and cache
        4. Return bars in chronological order

        Args:
            start_date: Start date (ISO format, e.g., '2020-01-01')
            end_date: End date (ISO format, e.g., '2020-12-31')
            frequency_type: Bar frequency type ('daily', 'minute')
            frequency: Bar frequency (1 for daily, 1/5/15/30 for minute)

        Yields:
            SchwabBar objects in chronological order

        Raises:
            ValueError: On invalid parameters
            requests.HTTPError: On API errors

        Examples:
            >>> # Daily bars (uses cache if available)
            >>> bars = adapter.read_bars("2020-01-01", "2020-12-31")
            >>>
            >>> # 5-minute bars
            >>> bars = adapter.read_bars("2024-10-01", "2024-10-15", "minute", 5)

        Notes:
            - Caching enabled if cache_root configured
            - Returns split-adjusted prices only
            - Rate limited to 10 requests/second
            - Automatic token refresh
        """
        # Step 1: Try cache first
        if self.metadata_manager:
            cached_bars = self._read_from_cache(start_date, end_date)
            if cached_bars:
                # Cache hit - yield cached bars
                for bar in cached_bars:
                    yield bar
                return

        # Step 2: Cache miss - fetch from API
        bars_from_api = list(self._fetch_from_api(start_date, end_date, frequency_type, frequency))

        # Step 3: Write to cache (if configured)
        if self.metadata_manager and bars_from_api:
            self._write_to_cache(bars_from_api, frequency_type, frequency)

        # Step 4: Yield bars
        for bar in bars_from_api:
            yield bar

    def _fetch_from_api(
        self,
        start_date: str,
        end_date: str,
        frequency_type: str = "daily",
        frequency: int = 1,
    ) -> Iterator[SchwabBar]:
        """
        Fetch bars from Schwab API.

        Internal method that handles the actual API call.

        Args:
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            frequency_type: Bar frequency type
            frequency: Bar frequency value

        Yields:
            SchwabBar objects from API

        Raises:
            requests.HTTPError: On API errors
        """
        # Convert dates to Unix timestamps (milliseconds)
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)

        start_ms = int(start_dt.timestamp() * 1000)
        end_ms = int(end_dt.timestamp() * 1000)

        # Build API parameters
        params = {
            "symbol": self.instrument.symbol,
            "periodType": "month",  # Not used with startDate/endDate
            "frequencyType": frequency_type,
            "frequency": frequency,
            "startDate": start_ms,
            "endDate": end_ms,
            "needExtendedHoursData": False,
            "needPreviousClose": False,
        }

        logger.info(
            "schwab_ohlc_adapter.fetching_bars",
            symbol=self.instrument.symbol,
            start_date=start_date,
            end_date=end_date,
            frequency_type=frequency_type,
            frequency=frequency,
        )

        # Call API
        try:
            response = self._call_api(self.PRICE_HISTORY_ENDPOINT, params)

            # Check if response has candles
            if "candles" not in response:
                logger.warning(
                    "schwab_ohlc_adapter.no_candles",
                    symbol=self.instrument.symbol,
                    response_keys=list(response.keys()),
                )
                return

            candles = response["candles"]

            if not candles:
                logger.info(
                    "schwab_ohlc_adapter.empty_response",
                    symbol=self.instrument.symbol,
                    start_date=start_date,
                    end_date=end_date,
                )
                return

            # Parse and yield bars
            bar_count = 0
            for candle in candles:
                try:
                    # Map API field names to SchwabBar fields
                    bar_data = {
                        "timestamp": candle["datetime"],
                        "open": candle["open"],
                        "high": candle["high"],
                        "low": candle["low"],
                        "close": candle["close"],
                        "volume": candle.get("volume", 0),
                    }

                    bar = SchwabBar(**bar_data)
                    yield bar
                    bar_count += 1

                except Exception as e:
                    logger.warning(
                        "schwab_ohlc_adapter.bar_parse_error",
                        symbol=self.instrument.symbol,
                        candle_data=candle,
                        error=str(e),
                    )
                    # Skip invalid bars but continue processing
                    continue

            logger.info(
                "schwab_ohlc_adapter.bars_loaded",
                symbol=self.instrument.symbol,
                count=bar_count,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            logger.error(
                "schwab_ohlc_adapter.fetch_error",
                symbol=self.instrument.symbol,
                start_date=start_date,
                end_date=end_date,
                error=str(e),
            )
            raise

    def get_available_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get available date range for this instrument.

        Note: Schwab API doesn't provide a metadata endpoint for date ranges.
        This is a best-effort approach using a wide date range query.

        Returns:
            Tuple of (min_date, max_date) in ISO format, or (None, None) if no data
        """
        try:
            # Query last 20 years (max historical data)
            params = {
                "symbol": self.instrument.symbol,
                "periodType": "year",
                "period": 20,
                "frequencyType": "daily",
                "frequency": 1,
            }

            response = self._call_api(self.PRICE_HISTORY_ENDPOINT, params)

            if "candles" not in response or not response["candles"]:
                return None, None

            candles = response["candles"]

            # Get first and last candle timestamps
            first_candle = candles[0]
            last_candle = candles[-1]

            # Convert Unix milliseconds to ISO date
            first_dt = datetime.fromtimestamp(first_candle["datetime"] / 1000, tz=timezone.utc)
            last_dt = datetime.fromtimestamp(last_candle["datetime"] / 1000, tz=timezone.utc)

            min_date = first_dt.date().isoformat()
            max_date = last_dt.date().isoformat()

            return min_date, max_date

        except Exception as e:
            logger.warning(
                "schwab_ohlc_adapter.date_range_error",
                symbol=self.instrument.symbol,
                error=str(e),
            )
            return None, None

    def __del__(self):
        """Close session on cleanup."""
        if hasattr(self, "session"):
            self.session.close()
