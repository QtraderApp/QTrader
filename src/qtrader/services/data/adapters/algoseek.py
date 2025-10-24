"""
Algoseek adapters for various dataset types.

This module contains adapters for different Algoseek datasets:
- AlgoseekOHLCVendorAdapter: OHLC daily bars dataset

Each adapter is responsible for loading raw Algoseek data from parquet files
and parsing it into vendor-specific AlgoseekBar objects. Adapters do NOT perform
any price adjustments or transformations - those are handled by the data layer.

Separation of concerns:
- Adapter: Load and parse raw data
- Vendor models: Validate and structure data
- Data layer: Transform to canonical format
"""

from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Optional
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from qtrader.services.data.adapters.models.algoseek import AlgoseekBar
from qtrader.system import LoggerFactory

if TYPE_CHECKING:
    from qtrader.events.events import CorporateActionEvent, PriceBarEvent
    from qtrader.services.data.models import Instrument

logger = LoggerFactory.get_logger()


class AlgoseekOHLCVendorAdapter:
    """
    Algoseek OHLC vendor adapter - parses raw OHLC data to AlgoseekBar.

    This adapter is responsible ONLY for:
    - Reading parquet/CSV files using DuckDB
    - Parsing timestamps and data types
    - Validating data structure
    - Returning Iterator[AlgoseekBar] in chronological order

    Does NOT:
    - Perform price adjustments (done in AlgoseekPriceSeries)
    - Transform to canonical format (done in DataLoader)
    - Apply business logic (done in backtest engine)

    Configuration:
        root_path: Base directory for parquet files
        path_template: Path template with {root_path} and {secid} placeholders
        symbol_map: Path to security master CSV (symbol → SecId mapping)

    Examples:
        >>> config = {
        ...     "root_path": "data/algoseek",
        ...     "path_template": "{root_path}/SecId={secid}/*.parquet",
        ...     "symbol_map": "data/equity_security_master_sample.csv"
        ... }
        >>> instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        >>> adapter = AlgoseekOHLCVendorAdapter(config, instrument)
        >>> bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))
        >>> print(f"Loaded {len(bars)} bars")

    Notes:
        - Uses DuckDB for efficient parquet reading
        - Supports Hive partitioning (SecId=*/data.parquet)
        - Returns raw unadjusted data (adjustments in transformation layer)
    """

    def __init__(self, config: dict, instrument: "Instrument", dataset_name: Optional[str] = None):
        """
        Initialize Algoseek OHLC vendor adapter.

        Args:
            config: Adapter configuration with root_path, path_template, symbol_map
                   Optional cache_root for incremental update support
            instrument: Instrument to load data for
            dataset_name: Dataset name from data_sources.yaml (e.g., "algoseek-us-equity-1d-unadjusted")

        Raises:
            ValueError: If required config keys missing
            FileNotFoundError: If symbol_map file not found
        """
        self.config = config
        self.instrument = instrument
        self.dataset_name = dataset_name or "algoseek-us-equity-1d-unadjusted"  # Default for backward compat

        # Validate configuration
        required_keys = ["root_path", "path_template", "symbol_map"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        self.root_path = Path(config["root_path"])
        self.path_template = config["path_template"]
        self.symbol_map_path = Path(config["symbol_map"])

        # Optional cache support for incremental updates
        self.cache_root = Path(config["cache_root"]) if "cache_root" in config else None
        self.cache_enabled = self.cache_root is not None

        # Load symbol → SecId mapping
        if not self.symbol_map_path.exists():
            raise FileNotFoundError(f"Symbol map not found: {self.symbol_map_path}")

        self.symbol_map = self._load_symbol_map()

        # Get SecId for this instrument
        self.secid = self._get_secid(instrument.symbol)

        # Build data path
        self.data_path = self._build_data_path()

        # DEBUG: Adapter initialization details
        logger.debug(
            "algoseek_ohlc_vendor_adapter.initialized",
            symbol=instrument.symbol,
            secid=self.secid,
            data_path=str(self.data_path),
            cache_enabled=self.cache_enabled,
        )

    def _load_symbol_map(self) -> pd.DataFrame:
        """
        Load symbol → SecId mapping from CSV.

        Supports both formats:
        - Test format: Symbol, SecId
        - Algoseek format: Tickers, SecId

        Returns:
            DataFrame with Symbol and SecId columns (normalized)

        Raises:
            ValueError: If CSV missing required columns
        """
        logger.debug("algoseek_ohlc_vendor_adapter.loading_symbol_map", path=str(self.symbol_map_path))

        df = pd.read_csv(self.symbol_map_path)

        # Normalize column names (support both test and production formats)
        if "Tickers" in df.columns and "Symbol" not in df.columns:
            # Algoseek format: rename Tickers → Symbol
            df = df.rename(columns={"Tickers": "Symbol"})
            logger.debug("algoseek_ohlc_vendor_adapter.normalized_columns", from_col="Tickers", to_col="Symbol")

        # Validate required columns (after normalization)
        required_cols = ["Symbol", "SecId"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Symbol map missing columns: {missing_cols}")

        # DEBUG: Symbol map loading
        logger.debug("algoseek_ohlc_vendor_adapter.symbol_map_loaded", count=len(df))
        return df

    def _get_secid(self, symbol: str) -> int:
        """
        Map symbol to SecId using symbol map.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')

        Returns:
            SecId for the symbol

        Raises:
            ValueError: If symbol not found in map
        """
        matches = self.symbol_map[self.symbol_map["Symbol"] == symbol]

        if matches.empty:
            raise ValueError(f"Symbol not found in symbol map: {symbol}")

        secid = int(matches.iloc[0]["SecId"])

        logger.debug("algoseek_ohlc_vendor_adapter.secid_mapped", symbol=symbol, secid=secid)
        return secid

    def _build_data_path(self) -> Path:
        """
        Build data path from template.

        Returns:
            Path to data directory/file

        Examples:
            >>> # Template: "{root_path}/SecId={secid}/*.parquet"
            >>> # Result: "data/algoseek/SecId=33449/*.parquet"
        """
        path_str = self.path_template.format(root_path=str(self.root_path), secid=self.secid)
        return Path(path_str)

    def prime_cache(self, start_date: str, end_date: str) -> int:
        """
        Efficiently write source data to cache using DuckDB COPY.

        Uses DuckDB's native parquet→parquet pipeline to transfer data
        without Python materialization. This is the recommended way to
        initialize cache and has ZERO Python memory overhead.

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD), inclusive

        Returns:
            Number of bars written to cache

        Raises:
            ValueError: If cache_root not configured
            FileNotFoundError: If source data not found

        Examples:
            >>> # Prime cache for 5 years of data (zero Python memory)
            >>> bars_written = adapter.prime_cache("2020-01-01", "2024-12-31")
            >>> print(f"Cached {bars_written} bars")
            >>>
            >>> # Now read_bars() will use cache (fast)
            >>> for bar in adapter.read_bars("2024-01-01", "2024-12-31"):
            ...     print(bar.close)

        Notes:
            - Zero Python memory overhead (DuckDB direct copy)
            - 10-100x faster than Python iteration
            - Creates cache directory structure automatically
            - Overwrites existing cache file
            - For incremental updates after priming, use update_to_latest()
        """
        if not self.cache_enabled:
            raise ValueError(
                "Cannot prime cache when cache_root not configured. "
                f"Add 'cache_root' to adapter config in data_sources.yaml"
            )

        assert self.cache_root is not None, "cache_enabled should only be True when cache_root is set"

        # Validate source exists
        data_dir = self.data_path.parent if "*" in str(self.data_path) else self.data_path
        if not data_dir.exists():
            raise FileNotFoundError(f"Data source not found: {data_dir} for {self.instrument.symbol}")

        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Create cache directory
        symbol_cache_dir = self.cache_root / self.instrument.symbol
        symbol_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = symbol_cache_dir / "data.parquet"

        # Use DuckDB COPY for efficient bulk transfer
        conn = duckdb.connect(":memory:")

        try:
            query = f"""
            COPY (
                SELECT *
                FROM read_parquet('{self.data_path}', hive_partitioning=true)
                WHERE TradeDate >= '{start_date}'
                  AND TradeDate <= '{end_date}'
                ORDER BY TradeDate ASC
            ) TO '{cache_file}' (FORMAT PARQUET)
            """

            logger.debug(
                "algoseek_ohlc_vendor_adapter.priming_cache",
                symbol=self.instrument.symbol,
                start_date=start_date,
                end_date=end_date,
                cache_file=str(cache_file),
            )

            # Execute COPY (doesn't return a result set)
            conn.execute(query)

            # Count rows in the written parquet file
            count_query = f"SELECT COUNT(*) FROM read_parquet('{cache_file}')"
            count_result = conn.execute(count_query)
            count_row = count_result.fetchone()

            if count_row is None:
                raise ValueError("Failed to count rows in written cache file")

            rows_written = count_row[0]

            logger.info(
                "algoseek_ohlc_vendor_adapter.cache_primed",
                symbol=self.instrument.symbol,
                bars_written=rows_written,
                cache_file=str(cache_file),
                start_date=start_date,
                end_date=end_date,
            )

            return int(rows_written)

        except Exception as e:
            logger.error(
                "algoseek_ohlc_vendor_adapter.cache_prime_error",
                symbol=self.instrument.symbol,
                error=str(e),
            )
            raise

        finally:
            conn.close()

    def write_cache(self, bars: list[AlgoseekBar]) -> None:
        """
        Write bars to cache file.

        Creates cache directory if it doesn't exist and writes bars as parquet.

        This is the fallback caching method when prime_cache() is not suitable.
        Note that this materializes the full bar list in memory.

        Args:
            bars: List of AlgoseekBar to cache

        Raises:
            ValueError: If cache not enabled or no bars provided
        """
        if not self.cache_enabled:
            raise ValueError("Cannot write cache when cache_root not configured")

        if not bars:
            logger.warning(
                "algoseek_ohlc_vendor_adapter.no_bars_to_cache",
                symbol=self.instrument.symbol,
            )
            return

        assert self.cache_root is not None, "cache_enabled should only be True when cache_root is set"

        # Create cache directory structure
        symbol_cache_dir = self.cache_root / self.instrument.symbol
        symbol_cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = symbol_cache_dir / "data.parquet"

        # Convert bars to DataFrame
        df = pd.DataFrame([bar.model_dump() for bar in bars])

        # Sort by TradeDate
        df = df.sort_values("TradeDate").reset_index(drop=True)

        # Write to parquet
        df.to_parquet(cache_file, index=False)

        logger.info(
            "algoseek_ohlc_vendor_adapter.cache_written",
            symbol=self.instrument.symbol,
            bars_count=len(bars),
            cache_file=str(cache_file),
        )

    def read_bars(self, start_date: str, end_date: str) -> Iterator[AlgoseekBar]:
        """
        Read OHLC bars from Algoseek parquet dataset.

        Streams bars one at a time (does NOT load all into memory).
        If cache_root is configured and cache exists, reads from cache (fast).
        Otherwise reads from source parquet files.

        NOTE: Does NOT auto-create cache. Use prime_cache() to create cache first.

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD), inclusive

        Yields:
            AlgoseekBar objects in chronological order

        Raises:
            FileNotFoundError: If data source or parquet files not found
            ValueError: If query execution fails

        Examples:
            >>> # Prime cache first (zero Python memory)
            >>> adapter.prime_cache("2020-01-01", "2024-12-31")
            >>>
            >>> # Now read from cache (fast)
            >>> for bar in adapter.read_bars("2024-01-01", "2024-12-31"):
            ...     print(f"Date: {bar.TradeDate}, Close: {bar.Close}")
            >>>
            >>> # Or read directly from source (no caching)
            >>> for bar in adapter.read_bars("2020-01-01", "2020-12-31"):
            ...     print(f"Date: {bar.TradeDate}, Close: {bar.Close}")

        Notes:
            - Uses DuckDB for efficient parquet scanning
            - Supports Hive partitioning automatically
            - Memory efficient (yields one bar at a time)
            - Returns raw unadjusted data as stored
            - No hidden side effects (truly streaming)
            - For cache creation, use prime_cache() first
            - For incremental updates, use update_to_latest()
        """
        # Check if we should use cache
        if self.cache_enabled:
            assert self.cache_root is not None, "cache_enabled should only be True when cache_root is set"
            symbol_cache_dir = self.cache_root / self.instrument.symbol
            cache_file = symbol_cache_dir / "data.parquet"

            if cache_file.exists():
                # Use cached data
                logger.debug(
                    "algoseek_ohlc_vendor_adapter.reading_from_cache",
                    symbol=self.instrument.symbol,
                    start_date=start_date,
                    end_date=end_date,
                )

                try:
                    df = pd.read_parquet(cache_file)
                    # Filter by date range
                    df = df[(df["TradeDate"] >= start_date) & (df["TradeDate"] <= end_date)]
                    # Sort by date
                    df = df.sort_values("TradeDate")

                    # Yield bars
                    for _, row in df.iterrows():
                        try:
                            bar = AlgoseekBar(**row.to_dict())
                            yield bar
                        except Exception as e:
                            logger.warning(
                                "algoseek_ohlc_vendor_adapter.cache_bar_parse_error",
                                symbol=self.instrument.symbol,
                                error=str(e),
                            )
                            continue
                    return
                except Exception as e:
                    logger.warning(
                        "algoseek_ohlc_vendor_adapter.cache_read_error",
                        symbol=self.instrument.symbol,
                        error=str(e),
                        fallback="reading_from_source",
                    )
                    # Fall through to read from source

        # Read from source (no auto-caching)
        # Validate source exists
        data_dir = self.data_path.parent if "*" in str(self.data_path) else self.data_path

        if not data_dir.exists():
            logger.error(
                "algoseek_ohlc_vendor_adapter.source_not_found",
                symbol=self.instrument.symbol,
                secid=self.secid,
                source=str(data_dir),
            )
            raise FileNotFoundError(f"Data source not found: {data_dir} for {self.instrument.symbol}")

        # Check if any parquet files exist
        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            logger.error(
                "algoseek_ohlc_vendor_adapter.no_parquet_files",
                symbol=self.instrument.symbol,
                secid=self.secid,
                source=str(data_dir),
            )
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # DEBUG: Reading bars details
        logger.debug(
            "algoseek_ohlc_vendor_adapter.reading_bars",
            symbol=self.instrument.symbol,
            start_date=start_date,
            end_date=end_date,
            parquet_count=len(parquet_files),
        )
        # Validate source exists
        data_dir = self.data_path.parent if "*" in str(self.data_path) else self.data_path

        if not data_dir.exists():
            logger.error(
                "algoseek_ohlc_vendor_adapter.source_not_found",
                symbol=self.instrument.symbol,
                secid=self.secid,
                source=str(data_dir),
            )
            raise FileNotFoundError(f"Data source not found: {data_dir} for {self.instrument.symbol}")

        # Check if any parquet files exist
        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            logger.error(
                "algoseek_ohlc_vendor_adapter.no_parquet_files",
                symbol=self.instrument.symbol,
                secid=self.secid,
                source=str(data_dir),
            )
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # DEBUG: Reading bars details
        logger.debug(
            "algoseek_ohlc_vendor_adapter.reading_bars",
            symbol=self.instrument.symbol,
            start_date=start_date,
            end_date=end_date,
            parquet_count=len(parquet_files),
        )

        # Connect to DuckDB (in-memory)
        conn = duckdb.connect(":memory:")

        # Build DuckDB query
        query = f"""
        SELECT *
        FROM read_parquet('{self.data_path}', hive_partitioning=true)
        WHERE TradeDate >= '{start_date}'
          AND TradeDate <= '{end_date}'
        ORDER BY TradeDate ASC
        """

        try:
            logger.debug("algoseek_ohlc_vendor_adapter.executing_query", query=query[:200])

            # Execute query
            result = conn.execute(query)

            # Get column names from description
            if result.description is None:
                raise ValueError("Query returned no description")
            columns = [desc[0] for desc in result.description]

            # Stream bars one at a time using fetchone() (truly streaming, no materialization)
            bar_count = 0
            while True:
                row = result.fetchone()  # type: ignore[assignment]
                if row is None:
                    break  # No more rows

                row_dict = dict(zip(columns, row))

                # Parse to AlgoseekBar (validation happens in Pydantic model)
                try:
                    bar = AlgoseekBar(**row_dict)
                    yield bar
                    bar_count += 1

                except Exception as e:
                    logger.warning(
                        "algoseek_ohlc_vendor_adapter.bar_parse_error",
                        symbol=self.instrument.symbol,
                        trade_date=row_dict.get("TradeDate"),
                        error=str(e),
                    )
                    # Skip invalid bars but continue processing
                    continue

            # DEBUG: Bars loaded details
            logger.debug(
                "algoseek_ohlc_vendor_adapter.bars_loaded",
                symbol=self.instrument.symbol,
                count=bar_count,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            logger.error(
                "algoseek_ohlc_vendor_adapter.query_error",
                symbol=self.instrument.symbol,
                error=str(e),
                query=query[:200],
            )
            raise

        finally:
            conn.close()

    def get_corporate_actions(self, start_date: str, end_date: str):
        """
        Detect corporate actions (splits, dividends) in date range.

        Returns list of CorporateActionEvent in chronological order.
        """

        actions = []
        previous_close = None
        for bar in self.read_bars(start_date, end_date):
            # Dividend event
            if bar.is_dividend():
                dividend_amount: Decimal | None = bar.get_dividend_amount(previous_close) if previous_close else None
                actions.append(
                    CorporateActionEvent(
                        symbol=bar.Ticker,
                        asset_class="equity",
                        action_type="dividend",
                        announcement_date=bar.TradeDate.date().isoformat(),
                        ex_date=bar.TradeDate.date().isoformat(),
                        effective_date=bar.TradeDate.date().isoformat(),
                        source=self.dataset_name,  # Full dataset name
                        dividend_amount=dividend_amount,
                        dividend_type="cash",
                    )
                )
            # Split event
            if bar.is_split():
                split_ratio = bar.get_split_ratio()
                actions.append(
                    CorporateActionEvent(
                        symbol=bar.Ticker,
                        asset_class="equity",
                        action_type="split",
                        announcement_date=bar.TradeDate.date().isoformat(),
                        ex_date=bar.TradeDate.date().isoformat(),
                        effective_date=bar.TradeDate.date().isoformat(),
                        source=self.dataset_name,  # Full dataset name
                        split_ratio=split_ratio,
                    )
                )
            previous_close = bar.Close
        return actions

    def get_available_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get available date range for this instrument.

        Returns:
            Tuple of (min_date, max_date) in ISO format, or (None, None) if no data

        Examples:
            >>> min_date, max_date = adapter.get_available_date_range()
            >>> print(f"Data available from {min_date} to {max_date}")
        """
        data_dir = self.data_path.parent if "*" in str(self.data_path) else self.data_path

        if not data_dir.exists():
            return None, None

        parquet_files = list(data_dir.rglob("*.parquet"))
        if not parquet_files:
            return None, None

        conn = duckdb.connect(":memory:")

        try:
            query = f"""
            SELECT
                MIN(TradeDate) as min_date,
                MAX(TradeDate) as max_date
            FROM read_parquet('{self.data_path}', hive_partitioning=true)
            """

            result = conn.execute(query).fetchone()

            if result and result[0] and result[1]:
                # Convert datetime to ISO string
                min_date = result[0].date().isoformat()
                max_date = result[1].date().isoformat()
                return min_date, max_date

            return None, None

        except Exception as e:
            logger.warning(
                "algoseek_ohlc_vendor_adapter.date_range_query_error",
                symbol=self.instrument.symbol,
                error=str(e),
            )
            return None, None

        finally:
            conn.close()

    def to_price_bar_event(self, bar: AlgoseekBar) -> "PriceBarEvent":
        """
        Convert AlgoseekBar directly to PriceBarEvent (unadjusted).

        This method provides direct mapping from raw Algoseek data to events,
        bypassing the complex adjustment logic in AlgoseekPriceSeries.

        Args:
            bar: Raw Algoseek bar

        Returns:
            PriceBarEvent with unadjusted OHLCV data

        Examples:
            >>> for bar in adapter.read_bars("2020-01-01", "2020-12-31"):
            ...     event = adapter.to_price_bar_event(bar)
            ...     event_bus.publish(event)
        """

        # Get dataset metadata from config
        tz_name = self.config.get("timezone", "America/New_York")
        asset_class = self.config.get("asset_class", "equity")
        price_currency = self.config.get("price_currency", "USD")
        price_scale = self.config.get("price_scale", 2)
        adjusted = self.config.get("adjusted", False)

        # Market close time for US equities: 16:00 ET
        trade_date = bar.TradeDate.date()
        market_close_time = time(16, 0, 0)

        # Create naive datetime for market close
        market_close_naive = datetime.combine(trade_date, market_close_time)

        # Import zoneinfo to handle timezone properly (including DST)
        try:
            tz = ZoneInfo(tz_name)
        except ImportError:
            # Fallback for Python < 3.9
            import pytz

            tz = pytz.timezone(tz_name)  # type: ignore[assignment]

        # Localize to ET timezone (automatically handles EST/EDT based on date)
        market_close_local = market_close_naive.replace(tzinfo=tz)

        # Convert to UTC for timestamp field
        market_close_utc = market_close_local.astimezone(timezone.utc)

        # Format timestamp_local with proper offset (will be -04:00 for EDT or -05:00 for EST)
        timestamp_local = market_close_local.isoformat()

        # Get split ratio for volume adjustment factor
        split_ratio = bar.get_split_ratio() if bar.is_split() else None

        return PriceBarEvent(
            symbol=bar.Ticker,
            asset_class=asset_class,
            interval="1d",
            timestamp=market_close_utc.isoformat(),  # UTC timestamp
            timestamp_local=timestamp_local,  # Local market time with proper DST offset
            timezone=tz_name,
            open=Decimal(str(bar.Open)),
            high=Decimal(str(bar.High)),
            low=Decimal(str(bar.Low)),
            close=Decimal(str(bar.Close)),
            volume=bar.MarketHoursVolume,
            adjusted=adjusted,
            cumulative_price_factor=Decimal(str(bar.CumulativePriceFactor)),
            cumulative_volume_factor=Decimal(str(bar.CumulativeVolumeFactor)),
            price_adjustment_factor=Decimal(str(bar.AdjustmentFactor)) if bar.AdjustmentFactor else None,
            volume_adjustment_factor=Decimal(str(split_ratio)) if split_ratio else None,
            adjustment_reason=bar.AdjustmentReason,
            price_currency=price_currency,
            price_scale=price_scale,
            source=self.dataset_name,  # Full dataset name (e.g., "algoseek-us-equity-1d-unadjusted")
        )

    def to_corporate_action_event(
        self, bar: AlgoseekBar, prev_bar: Optional[AlgoseekBar] = None
    ) -> Optional["CorporateActionEvent"]:
        """
        Extract corporate action event from AlgoseekBar (if present).

        Corporate actions are detected from AdjustmentReason and AdjustmentFactor fields.
        Returns None if no corporate action on this bar.

        Args:
            bar: Current Algoseek bar
            prev_bar: Previous bar (required for dividend calculation)

        Returns:
            CorporateActionEvent or None

        Examples:
            >>> prev = None
            >>> for bar in adapter.read_bars("2020-01-01", "2020-12-31"):
            ...     event = adapter.to_corporate_action_event(bar, prev)
            ...     if event:
            ...         event_bus.publish(event)
            ...     prev = bar

        Notes:
            - Dividends require prev_bar for amount calculation
            - Splits use AdjustmentFactor directly
            - Both event types use TradeDate as effective_date
        """

        # Check for dividend
        if bar.is_dividend():
            dividend_amount = None
            if prev_bar is not None:
                dividend_amount = bar.get_dividend_amount(prev_bar.Close)

            return CorporateActionEvent(
                symbol=bar.Ticker,
                asset_class="equity",
                action_type="dividend",
                announcement_date=bar.TradeDate.date().isoformat(),  # Use ex-date as announcement
                ex_date=bar.TradeDate.date().isoformat(),
                effective_date=bar.TradeDate.date().isoformat(),
                source=self.dataset_name,  # Full dataset name
                dividend_amount=dividend_amount,
                dividend_currency="USD",
                dividend_type="cash",
                price_adjustment_factor=Decimal(str(bar.AdjustmentFactor)) if bar.AdjustmentFactor else None,
            )

        # Check for split
        if bar.is_split():
            split_ratio = bar.get_split_ratio()

            return CorporateActionEvent(
                symbol=bar.Ticker,
                asset_class="equity",
                action_type="split",
                announcement_date=bar.TradeDate.date().isoformat(),  # Use ex-date as announcement
                ex_date=bar.TradeDate.date().isoformat(),
                effective_date=bar.TradeDate.date().isoformat(),
                source=self.dataset_name,  # Full dataset name
                split_ratio=split_ratio,
                volume_adjustment_factor=Decimal(str(bar.AdjustmentFactor)) if bar.AdjustmentFactor else None,
            )

        # No corporate action on this bar
        return None

    def get_timestamp(self, bar: AlgoseekBar) -> datetime:
        """
        Extract timestamp from AlgoseekBar for heap-merge synchronization.

        AlgoseekBar uses .TradeDate field (not .timestamp).

        Args:
            bar: AlgoseekBar instance

        Returns:
            Bar's TradeDate as datetime

        Examples:
            >>> bar = next(adapter.read_bars("2024-01-01", "2024-01-01"))
            >>> ts = adapter.get_timestamp(bar)
            >>> print(ts)  # datetime(2024, 1, 1, 0, 0, 0)
        """
        return bar.TradeDate

    def update_to_latest(self, dry_run: bool = False) -> tuple[int, Optional[date], Optional[date]]:
        """
        Incrementally update cached data to latest available.

        Reads the last cached date and fetches only new bars from source.
        Requires cache_root to be configured in adapter config.

        If no cache exists, raises FileNotFoundError - use prime_cache() to create
        initial cache first.

        Args:
            dry_run: If True, only check what would be updated without writing

        Returns:
            Tuple of (bars_added, start_date, end_date)

        Raises:
            ValueError: If cache_root not configured
            FileNotFoundError: If no cache exists (need full backfill via prime_cache() first)

        Examples:
            >>> # First time: create cache with full backfill (zero Python memory)
            >>> adapter = resolver.resolve_by_dataset("algoseek-us-equity-1d-unadjusted", Instrument("AAPL"))
            >>> adapter.prime_cache("2020-01-01", "2024-12-31")
            >>>
            >>> # Subsequent calls: incremental updates only
            >>> bars_added, start, end = adapter.update_to_latest()
            >>> print(f"Added {bars_added} new bars from {start} to {end}")

        Notes:
            - Only fetches bars after last cached date (incremental)
            - Materializes only new bars in memory (not entire dataset)
            - Merges new bars with existing cache
            - Safe for frequent updates (daily/weekly)
        """
        if not self.cache_enabled:
            raise ValueError(
                "Incremental updates require cache_root in adapter configuration. "
                f"Current config keys: {list(self.config.keys())}"
            )

        assert self.cache_root is not None, "cache_enabled should only be True when cache_root is set"

        # Get symbol cache directory
        symbol_cache_dir = self.cache_root / self.instrument.symbol
        cache_file = symbol_cache_dir / "data.parquet"

        # Check if cache exists
        if not cache_file.exists():
            # No cache: need full backfill first
            raise FileNotFoundError(
                f"No cache found for {self.instrument.symbol}. "
                f"Run full backfill first by calling prime_cache() or use DatasetUpdater."
            )

        # Read last date from cache
        try:
            cached_df = pd.read_parquet(cache_file)
            if len(cached_df) == 0:
                raise ValueError(f"Cache file is empty for {self.instrument.symbol}")

            # Get last date (cache should be sorted by TradeDate)
            last_cached_date = pd.to_datetime(cached_df["TradeDate"]).max().date()

            logger.debug(
                "algoseek_ohlc_vendor_adapter.last_cached_date",
                symbol=self.instrument.symbol,
                last_date=last_cached_date.isoformat(),
            )
        except Exception as e:
            raise ValueError(f"Failed to read cache for {self.instrument.symbol}: {e}")

        # Get available date range from source
        min_available, max_available = self.get_available_date_range()

        if not min_available or not max_available:
            logger.warning(
                "algoseek_ohlc_vendor_adapter.no_data_available",
                symbol=self.instrument.symbol,
            )
            return 0, None, None

        # Calculate update range: from day after last cached to latest available
        from datetime import timedelta

        update_start = last_cached_date + timedelta(days=1)
        update_end = date.fromisoformat(max_available)

        if update_start > update_end:
            # Already up to date
            logger.info(
                "algoseek_ohlc_vendor_adapter.already_current",
                symbol=self.instrument.symbol,
                last_cached=last_cached_date.isoformat(),
                latest_available=max_available,
            )
            return 0, None, None

        if dry_run:
            # Estimate bars to add (assume ~252 trading days per year)
            days_diff = (update_end - update_start).days
            estimated_bars = int(days_diff * 252 / 365)
            logger.info(
                "algoseek_ohlc_vendor_adapter.dry_run",
                symbol=self.instrument.symbol,
                update_start=update_start.isoformat(),
                update_end=update_end.isoformat(),
                estimated_bars=estimated_bars,
            )
            return estimated_bars, update_start, update_end

        # Fetch new bars (disable caching since we'll merge with existing cache)
        # Temporarily disable caching to avoid double-write
        cache_was_enabled = self.cache_enabled
        self.cache_enabled = False

        try:
            new_bars = list(self.read_bars(update_start.isoformat(), update_end.isoformat()))
        finally:
            self.cache_enabled = cache_was_enabled

        if not new_bars:
            logger.info(
                "algoseek_ohlc_vendor_adapter.no_new_data",
                symbol=self.instrument.symbol,
                update_start=update_start.isoformat(),
                update_end=update_end.isoformat(),
            )
            return 0, None, None

        # Convert to DataFrame
        new_df = pd.DataFrame([bar.model_dump() for bar in new_bars])

        # Append to cache
        combined_df = pd.concat([cached_df, new_df], ignore_index=True)

        # Ensure sorted by TradeDate
        combined_df = combined_df.sort_values("TradeDate").reset_index(drop=True)

        # Remove duplicates (in case of overlapping dates)
        combined_df = combined_df.drop_duplicates(subset=["TradeDate"], keep="last")

        # Ensure cache directory exists before writing
        symbol_cache_dir.mkdir(parents=True, exist_ok=True)

        # Write back to cache
        combined_df.to_parquet(cache_file, index=False)

        bars_added = len(new_bars)
        first_new_date = new_bars[0].TradeDate.date()
        last_new_date = new_bars[-1].TradeDate.date()

        logger.info(
            "algoseek_ohlc_vendor_adapter.update_complete",
            symbol=self.instrument.symbol,
            bars_added=bars_added,
            start_date=first_new_date.isoformat(),
            end_date=last_new_date.isoformat(),
        )

        return bars_added, first_new_date, last_new_date
