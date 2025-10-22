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

from pathlib import Path
from typing import Iterator, Optional

import duckdb
import pandas as pd

from qtrader.contracts.data import Instrument
from qtrader.services.data.adapters.models.algoseek import AlgoseekBar
from qtrader.system import LoggerFactory

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

    def __init__(self, config: dict, instrument: Instrument):
        """
        Initialize Algoseek OHLC vendor adapter.

        Args:
            config: Adapter configuration with root_path, path_template, symbol_map
            instrument: Instrument to load data for

        Raises:
            ValueError: If required config keys missing
            FileNotFoundError: If symbol_map file not found
        """
        self.config = config
        self.instrument = instrument

        # Validate configuration
        required_keys = ["root_path", "path_template", "symbol_map"]
        missing_keys = [key for key in required_keys if key not in config]
        if missing_keys:
            raise ValueError(f"Missing required config keys: {missing_keys}")

        self.root_path = Path(config["root_path"])
        self.path_template = config["path_template"]
        self.symbol_map_path = Path(config["symbol_map"])

        # Load symbol → SecId mapping
        if not self.symbol_map_path.exists():
            raise FileNotFoundError(f"Symbol map not found: {self.symbol_map_path}")

        self.symbol_map = self._load_symbol_map()

        # Get SecId for this instrument
        self.secid = self._get_secid(instrument.symbol)

        # Build data path
        self.data_path = self._build_data_path()

        logger.info(
            "algoseek_ohlc_vendor_adapter.initialized",
            symbol=instrument.symbol,
            secid=self.secid,
            data_path=str(self.data_path),
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

        logger.info("algoseek_ohlc_vendor_adapter.symbol_map_loaded", count=len(df))
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

    def read_bars(self, start_date: str, end_date: str) -> Iterator[AlgoseekBar]:
        """
        Read raw bars from data source.

        This method loads raw Algoseek data and yields AlgoseekBar objects.
        No adjustments or transformations are performed - just data loading.

        Args:
            start_date: Start date (ISO format, e.g., '2020-01-01')
            end_date: End date (ISO format, e.g., '2020-12-31')

        Yields:
            AlgoseekBar objects in chronological order (TradeDate ASC)

        Raises:
            FileNotFoundError: If data path doesn't exist or has no parquet files
            Exception: On DuckDB query errors

        Examples:
            >>> bars = adapter.read_bars("2020-01-01", "2020-03-31")
            >>> first_bar = next(bars)
            >>> print(f"Date: {first_bar.TradeDate}, Close: {first_bar.Close}")

        Notes:
            - Uses DuckDB for efficient parquet scanning
            - Supports Hive partitioning automatically
            - Memory efficient (yields one bar at a time)
            - Returns raw unadjusted data as stored
        """
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

        logger.info(
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

            # Execute query and fetch results
            result = conn.execute(query)

            # Get column names from description
            if result.description is None:
                raise ValueError("Query returned no description")
            columns = [desc[0] for desc in result.description]

            # Yield bars one at a time
            bar_count = 0
            for row in result.fetchall():
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

            logger.info(
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
        from decimal import Decimal

        from qtrader.events.events import CorporateActionEvent

        actions = []
        previous_close = None
        for bar in self.read_bars(start_date, end_date):
            # Dividend event
            if bar.is_dividend():
                dividend_amount: Decimal | None = bar.get_dividend_amount(previous_close) if previous_close else None
                actions.append(
                    CorporateActionEvent(
                        symbol=bar.Ticker,
                        action_type="dividend",
                        effective_date=bar.TradeDate,
                        ex_date=bar.TradeDate,
                        dividend_amount=dividend_amount,
                        dividend_type="cash",
                        split_ratio=None,
                    )
                )
            # Split event
            if bar.is_split():
                split_ratio = bar.get_split_ratio()
                actions.append(
                    CorporateActionEvent(
                        symbol=bar.Ticker,
                        action_type="split",
                        effective_date=bar.TradeDate,
                        ex_date=None,
                        dividend_amount=None,
                        dividend_type="cash",
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
