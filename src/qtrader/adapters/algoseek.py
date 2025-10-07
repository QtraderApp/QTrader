"""Adapter for Algoseek parquet data with Hive partitioning."""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, Iterator

import duckdb
import pandas as pd
import pytz

from qtrader.config.data_config import DataConfig
from qtrader.config.logging_config import LoggerFactory
from qtrader.models.bar import AdjustmentEvent, Bar, DataMode
from qtrader.models.instrument import Instrument

logger = LoggerFactory.get_logger()


class AlgoseekParquetAdapter:
    """
    Adapter for Algoseek parquet data with Hive partitioning.

    Reads from partitioned parquet files (SecId=*/data_0.parquet) using DuckDB.
    Normalizes Algoseek schema to canonical Bar (OHLCV) + AdjustmentEvent metadata.
    Uses Instrument to determine symbol and constructs file paths using SecId lookup.

    Data Mode: ADJUSTED (prices already total-return adjusted)

    Configuration (from data_sources.yaml):
        root_path: Base directory for parquet files
        path_template: Path template with {root_path} and {secid} placeholders
        symbol_map: Path to security master CSV (symbol → SecId mapping)
        mode: Data mode (standard_adjusted)

    Usage:
        config = {
            "root_path": "data/algoseek",
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": "data/equity_security_master_sample.csv"
        }
        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        adapter = AlgoseekParquetAdapter(config, instrument)
        bars = adapter.read_bars(data_config)
    """

    SCHEMA_VERSION = "algoseek-parquet-v1.0"

    def __init__(self, config: Dict[str, Any], instrument: Instrument):
        """
        Initialize adapter for specific instrument.

        Args:
            config: Configuration dict from data_sources.yaml
            instrument: Logical instrument specification

        Raises:
            ValueError: If required config fields missing
            FileNotFoundError: If symbol_map file not found
            KeyError: If symbol not found in security master
        """
        self.config = config
        self.instrument = instrument

        # Validate required config fields
        required = ["root_path", "path_template", "symbol_map"]
        missing = [f for f in required if f not in config]
        if missing:
            raise ValueError(f"Missing required config fields: {missing}")

        # Load symbol → SecId mapping
        self.symbol_map_path = Path(config["symbol_map"])
        if not self.symbol_map_path.exists():
            raise FileNotFoundError(f"Symbol map not found: {self.symbol_map_path}")

        self.symbol_to_secid = self._load_symbol_map()

        # Resolve SecId for this instrument
        if instrument.symbol not in self.symbol_to_secid:
            raise KeyError(
                f"Symbol '{instrument.symbol}' not found in security master. "
                f"Available symbols: {list(self.symbol_to_secid.keys())}"
            )

        self.secid = self.symbol_to_secid[instrument.symbol]

        # Build data directory from template (without wildcard)
        path_template = config["path_template"]
        # Remove wildcard pattern if present
        path_str = path_template.format(root_path=config["root_path"], secid=self.secid)
        # Strip the wildcard pattern - data_path should be a directory
        if "*" in path_str:
            path_str = path_str.rsplit("/", 1)[0]  # Remove the "/*.parquet" part

        self.data_path = Path(path_str)

        logger.info(
            "algoseek_adapter.initialized",
            symbol=instrument.symbol,
            secid=self.secid,
            data_path=str(self.data_path),
        )

    def _load_symbol_map(self) -> Dict[str, int]:
        """
        Load symbol → SecId mapping from security master CSV.

        Returns:
            Dict mapping symbol to SecId

        Raises:
            ValueError: If CSV format invalid
        """
        try:
            df = pd.read_csv(self.symbol_map_path)
        except Exception as e:
            raise ValueError(f"Failed to read symbol map {self.symbol_map_path}: {e}")

        if "Tickers" not in df.columns or "SecId" not in df.columns:
            raise ValueError(f"Symbol map must have 'Tickers' and 'SecId' columns. Found: {df.columns.tolist()}")

        # Map ticker to SecId
        symbol_map = {}
        for _, row in df.iterrows():
            ticker = row["Tickers"]
            secid = int(row["SecId"])
            symbol_map[ticker] = secid

        logger.debug(
            "algoseek_adapter.symbol_map_loaded",
            symbol_count=len(symbol_map),
            symbols=list(symbol_map.keys()),
        )

        return symbol_map

    def can_read(self) -> bool:
        """Check if data path exists and contains parquet files."""
        if not self.data_path.exists():
            return False
        # Check for .parquet files or Hive-style partitions
        return any(self.data_path.rglob("*.parquet"))

    def schema_version(self) -> str:
        """Return adapter schema version."""
        return self.SCHEMA_VERSION

    def get_data_mode(self) -> DataMode:
        """Algoseek data is total-return adjusted."""
        return DataMode.ADJUSTED

    def read_bars(self, config: DataConfig) -> Iterator[Bar]:
        """
        Read parquet files for this instrument and yield canonical Bar objects.

        Args:
            config: DataConfig with schema mappings and formatting rules

        Yields:
            Bar objects in timestamp order

        Raises:
            FileNotFoundError: If data path doesn't exist or has no parquet files
            Exception: On query or conversion errors

        Steps:
        1. Connect to DuckDB in-memory
        2. Read parquet with hive_partitioning=true
        3. Apply bar_schema mapping (vendor columns → Bar fields)
        4. Convert types (float → Decimal, timestamp → tz-aware)
        5. Yield Bar objects in timestamp order
        """
        # Validate source exists
        if not self.data_path.exists():
            logger.error(
                "algoseek_adapter.source_not_found",
                symbol=self.instrument.symbol,
                secid=self.secid,
                source=str(self.data_path),
            )
            raise FileNotFoundError(f"Data source not found: {self.data_path} for {self.instrument.symbol}")

        # Build parquet glob pattern
        if self.data_path.is_dir():
            parquet_pattern = str(self.data_path / "**" / "*.parquet")
        else:
            # data_path is a single file
            parquet_pattern = str(self.data_path)

        # Check if any parquet files exist
        parquet_files = list(self.data_path.rglob("*.parquet") if self.data_path.is_dir() else [self.data_path])
        if not parquet_files:
            logger.error(
                "algoseek_adapter.no_parquet_files",
                symbol=self.instrument.symbol,
                secid=self.secid,
                source=str(self.data_path),
                pattern=parquet_pattern,
            )
            raise FileNotFoundError(f"No parquet files found in {self.data_path} for {self.instrument.symbol}")

        logger.debug(
            "algoseek_adapter.parquet_files_found",
            symbol=self.instrument.symbol,
            secid=self.secid,
            source=str(self.data_path),
            file_count=len(parquet_files),
            files=[f.name for f in parquet_files[:10]],  # Log first 10 files
        )

        # Configure timezone
        try:
            tz = pytz.timezone(config.timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            logger.error(
                "algoseek_adapter.invalid_timezone",
                timezone=config.timezone,
                error=str(e),
            )
            raise

        # Decimal quantization context
        price_decimals = config.decimals.get("price", 4)
        quantizer = Decimal(10) ** -price_decimals

        # Bar schema mapping
        bar_schema = config.bar_schema

        logger.info(
            "algoseek_adapter.reading_bars",
            symbol=self.instrument.symbol,
            secid=self.secid,
            source=str(self.data_path),
            timezone=config.timezone,
            price_decimals=price_decimals,
            data_mode="adjusted",
            file_count=len(parquet_files),
        )

        # Connect to DuckDB and read
        con = duckdb.connect(":memory:")
        try:
            # Build SELECT with bar schema mapping
            select_cols = [
                f"{bar_schema.ts} as ts",
                f"{bar_schema.symbol} as symbol",
                f"{bar_schema.open} as open",
                f"{bar_schema.high} as high",
                f"{bar_schema.low} as low",
                f"{bar_schema.close} as close",
                f"{bar_schema.volume} as volume",
            ]

            query = f"""
            SELECT {", ".join(select_cols)}
            FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            ORDER BY symbol, ts
            """

            try:
                result = con.execute(query)
            except Exception as e:
                logger.error(
                    "algoseek_adapter.query_failed",
                    query=query[:200],  # Log first 200 chars
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

            bar_count = 0
            last_symbol = None
            symbol_bar_counts: dict[str, int] = {}

            # Yield bars one at a time
            for row in result.fetchall():
                try:
                    ts_naive, symbol, open_f, high_f, low_f, close_f, volume = row

                    # Track symbols for logging
                    if symbol != last_symbol:
                        symbol_bar_counts[symbol] = symbol_bar_counts.get(symbol, 0) + 1
                        if last_symbol is not None and symbol_bar_counts[last_symbol] > 0:
                            logger.debug(
                                "algoseek_adapter.symbol_completed",
                                symbol=last_symbol,
                                bar_count=symbol_bar_counts[last_symbol],
                            )
                        last_symbol = symbol

                    # Localize timestamp
                    ts = tz.localize(ts_naive)

                    # Convert prices to Decimal
                    open_d = Decimal(str(open_f)).quantize(quantizer, rounding=ROUND_HALF_UP)
                    high_d = Decimal(str(high_f)).quantize(quantizer, rounding=ROUND_HALF_UP)
                    low_d = Decimal(str(low_f)).quantize(quantizer, rounding=ROUND_HALF_UP)
                    close_d = Decimal(str(close_f)).quantize(quantizer, rounding=ROUND_HALF_UP)

                    bar_count += 1
                    # Use instrument symbol (not vendor symbol which might differ)
                    yield Bar(
                        ts=ts,
                        symbol=self.instrument.symbol,
                        open=open_d,
                        high=high_d,
                        low=low_d,
                        close=close_d,
                        volume=int(volume),
                    )
                except Exception as e:
                    logger.error(
                        "algoseek_adapter.bar_conversion_failed",
                        row=str(row)[:200],  # Truncate long rows
                        error=str(e),
                        error_type=type(e).__name__,
                        bar_number=bar_count + 1,
                    )
                    raise

            logger.info(
                "algoseek_adapter.bars_completed",
                bar_count=bar_count,
                symbol_count=len(symbol_bar_counts),
                symbols=list(symbol_bar_counts.keys()),
            )

        finally:
            con.close()
            logger.debug("algoseek_adapter.bars_connection_closed")

    def read_adjustments(self, config: DataConfig) -> Iterator[AdjustmentEvent]:
        """
        Read adjustment metadata from parquet files for this instrument.

        Args:
            config: DataConfig with adjustment schema

        Yields:
            AdjustmentEvent objects

        Extracts rows with AdjustmentReason != NULL as AdjustmentEvent objects.
        """
        if config.adjustment_schema is None:
            logger.info(
                "algoseek_adapter.no_adjustment_schema",
                symbol=self.instrument.symbol,
                msg="Skipping adjustment metadata",
            )
            return

        parquet_pattern = str(self.data_path)

        try:
            tz = pytz.timezone(config.timezone)
        except pytz.exceptions.UnknownTimeZoneError as e:
            logger.error(
                "algoseek_adapter.invalid_timezone_adjustments",
                timezone=config.timezone,
                error=str(e),
            )
            raise

        # Adjustment schema mapping
        adj_schema = config.adjustment_schema

        logger.info(
            "algoseek_adapter.reading_adjustments",
            symbol=self.instrument.symbol,
            secid=self.secid,
            source=str(self.data_path),
        )

        con = duckdb.connect(":memory:")
        try:
            # Select only rows with adjustments
            select_cols = [
                f"{adj_schema.ts} as ts",
                f"{adj_schema.symbol} as symbol",
                f"{adj_schema.event_type} as event_type",
                f"{adj_schema.px_factor} as px_factor",
                f"{adj_schema.vol_factor} as vol_factor",
            ]

            # Add metadata fields if specified
            metadata_cols = ", ".join(adj_schema.metadata_fields) if adj_schema.metadata_fields else ""

            query = f"""
            SELECT {", ".join(select_cols)}{", " + metadata_cols if metadata_cols else ""}
            FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
            WHERE {adj_schema.event_type} IS NOT NULL
            ORDER BY symbol, ts
            """

            try:
                result = con.execute(query)
            except Exception as e:
                logger.error(
                    "algoseek_adapter.adjustments_query_failed",
                    query=query[:200],
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise

            adj_count = 0
            symbol_adj_counts: dict[str, int] = {}

            for row in result.fetchall():
                try:
                    ts_naive, symbol, event_type, px_f, vol_f = row[:5]
                    metadata_vals = row[5:] if len(row) > 5 else []

                    # Track adjustment counts by symbol
                    symbol_adj_counts[symbol] = symbol_adj_counts.get(symbol, 0) + 1

                    # Localize timestamp
                    ts = tz.localize(ts_naive)

                    # Convert factors to Decimal
                    px_factor = Decimal(str(px_f)).quantize(Decimal("0.0000001"))
                    vol_factor = Decimal(str(vol_f)).quantize(Decimal("0.0000001"))

                    # Build metadata dict
                    metadata = {}
                    if adj_schema.metadata_fields and metadata_vals:
                        metadata = dict(zip(adj_schema.metadata_fields, metadata_vals))

                    adj_count += 1
                    # Use instrument symbol (not vendor symbol)
                    yield AdjustmentEvent(
                        ts=ts,
                        symbol=self.instrument.symbol,
                        event_type=event_type,
                        px_factor=px_factor,
                        vol_factor=vol_factor,
                        metadata=metadata,
                    )

                    # Log individual adjustments for debugging
                    logger.debug(
                        "algoseek_adapter.adjustment_event",
                        symbol=symbol,
                        ts=ts.isoformat(),
                        event_type=event_type,
                        px_factor=float(px_factor),
                        vol_factor=float(vol_factor),
                    )
                except Exception as e:
                    logger.error(
                        "algoseek_adapter.adjustment_conversion_failed",
                        row=str(row)[:200],
                        error=str(e),
                        error_type=type(e).__name__,
                        adjustment_number=adj_count + 1,
                    )
                    raise

            logger.info(
                "algoseek_adapter.adjustments_completed",
                adj_count=adj_count,
                symbol_count=len(symbol_adj_counts),
                adjustments_by_symbol=symbol_adj_counts,
            )

        finally:
            con.close()
            logger.debug("algoseek_adapter.adjustments_connection_closed")
