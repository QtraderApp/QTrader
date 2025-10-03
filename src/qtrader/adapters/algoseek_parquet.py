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
        # Build parquet glob pattern
        parquet_pattern = str(source / "**" / "*.parquet")

        # Configure timezone
        tz = pytz.timezone(config.timezone)

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

            result = con.execute(query)

            bar_count = 0
            # Yield bars one at a time
            for row in result.fetchall():
                ts_naive, symbol, open_f, high_f, low_f, close_f, volume = row

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

            logger.info("algoseek_adapter.bars_completed", bar_count=bar_count)

        finally:
            con.close()

    def read_adjustments(self, source: Path, config: DataConfig) -> Iterator[AdjustmentEvent]:
        """
        Read adjustment metadata from parquet files.

        Extracts rows with AdjustmentReason != NULL as AdjustmentEvent objects.
        """
        if config.adjustment_schema is None:
            logger.info("algoseek_adapter.no_adjustment_schema", msg="Skipping adjustment metadata")
            return

        parquet_pattern = str(source / "**" / "*.parquet")
        tz = pytz.timezone(config.timezone)

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

            result = con.execute(query)

            adj_count = 0
            for row in result.fetchall():
                ts_naive, symbol, event_type, px_f, vol_f = row[:5]
                metadata_vals = row[5:] if len(row) > 5 else []

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

            logger.info("algoseek_adapter.adjustments_completed", adj_count=adj_count)

        finally:
            con.close()
