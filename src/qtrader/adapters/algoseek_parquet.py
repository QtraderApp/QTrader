"""Adapter for Algoseek parquet data with Hive partitioning."""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterator

import duckdb
import pytz

from qtrader.config.data_config import DataConfig
from qtrader.config.logging_config import LoggerFactory
from qtrader.models.bar import AdjustmentEvent, Bar, DataMode

logger = LoggerFactory.get_logger()


class AlgoseekParquetAdapter:
    """
    Adapter for Algoseek parquet data with Hive partitioning.

    Reads from partitioned parquet files (SecId=*/data_0.parquet) using DuckDB.
    Normalizes Algoseek schema to canonical Bar (OHLCV) + AdjustmentEvent metadata.

    Data Mode: ADJUSTED (prices already total-return adjusted)
    """

    SCHEMA_VERSION = "algoseek-parquet-v1.0"

    def can_read(self, source: Path) -> bool:
        """Check if source contains parquet files."""
        if not source.exists():
            return False
        # Check for .parquet files or Hive-style partitions
        return any(source.rglob("*.parquet"))

    def schema_version(self) -> str:
        """Return adapter schema version."""
        return self.SCHEMA_VERSION

    def get_data_mode(self) -> DataMode:
        """Algoseek data is total-return adjusted."""
        return DataMode.ADJUSTED

    def read_bars(self, source: Path, config: DataConfig) -> Iterator[Bar]:
        """
        Read parquet files and yield canonical Bar objects (OHLCV only).

        Steps:
        1. Connect to DuckDB in-memory
        2. Read parquet with hive_partitioning=true
        3. Apply bar_schema mapping (vendor columns → Bar fields)
        4. Convert types (float → Decimal, timestamp → tz-aware)
        5. Yield Bar objects in symbol, timestamp order
        """
        # Validate source exists
        if not source.exists():
            logger.error(
                "algoseek_adapter.source_not_found",
                source=str(source),
            )
            raise FileNotFoundError(f"Data source not found: {source}")

        # Build parquet glob pattern
        parquet_pattern = str(source / "**" / "*.parquet")

        # Check if any parquet files exist
        parquet_files = list(source.rglob("*.parquet"))
        if not parquet_files:
            logger.error(
                "algoseek_adapter.no_parquet_files",
                source=str(source),
                pattern=parquet_pattern,
            )
            raise FileNotFoundError(f"No parquet files found in {source}")

        logger.debug(
            "algoseek_adapter.parquet_files_found",
            source=str(source),
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
            source=str(source),
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
                    yield Bar(
                        ts=ts,
                        symbol=symbol,
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

    def read_adjustments(self, source: Path, config: DataConfig) -> Iterator[AdjustmentEvent]:
        """
        Read adjustment metadata from parquet files.

        Extracts rows with AdjustmentReason != NULL as AdjustmentEvent objects.
        """
        if config.adjustment_schema is None:
            logger.info("algoseek_adapter.no_adjustment_schema", msg="Skipping adjustment metadata")
            return

        parquet_pattern = str(source / "**" / "*.parquet")

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

        logger.info("algoseek_adapter.reading_adjustments", source=str(source))

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
                    yield AdjustmentEvent(
                        ts=ts,
                        symbol=symbol,
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
