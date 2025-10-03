"""Adapter for CSV files (e.g., security master or exported CSVs)."""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytz
import structlog

from qtrader.config.data_config import DataConfig
from qtrader.models.bar import AdjustmentEvent, Bar, DataMode

logger = structlog.get_logger()


class CSVAdapter:
    """
    Adapter for CSV files.

    Can read single CSV file or directory with multiple CSVs.
    Useful for security master data or testing with exported CSV samples.

    Data Mode: Configurable (depends on CSV source)
    """

    SCHEMA_VERSION = "csv-v1.0"

    def can_read(self, source: Path) -> bool:
        """Check if source is a CSV file or directory with CSVs."""
        if source.is_file() and source.suffix == ".csv":
            return True
        if source.is_dir():
            return any(source.glob("*.csv"))
        return False

    def schema_version(self) -> str:
        """Return adapter schema version."""
        return self.SCHEMA_VERSION

    def get_data_mode(self) -> DataMode:
        """Data mode determined by config (default: adjusted)."""
        # CSV could be adjusted or unadjusted depending on source
        # Default to adjusted; can be overridden in config
        return DataMode.ADJUSTED

    def read_bars(self, source: Path, config: DataConfig) -> Iterator[Bar]:
        """Read CSV files and yield Bar objects (OHLCV only)."""
        # Bar schema mapping
        bar_schema = config.bar_schema

        tz = pytz.timezone(config.timezone)
        price_decimals = config.decimals.get("price", 4)
        quantizer = Decimal(10) ** -price_decimals

        # Determine CSV files to read
        csv_files = []
        if source.is_file():
            csv_files = [source]
        else:
            csv_files = sorted(source.glob("*.csv"))

        logger.info("csv_adapter.reading_bars", file_count=len(csv_files))

        bar_count = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=[bar_schema.ts])

            for _, row in df.iterrows():
                ts_naive = pd.Timestamp(row[bar_schema.ts])
                ts = tz.localize(ts_naive.to_pydatetime())

                open_d = Decimal(str(row[bar_schema.open])).quantize(quantizer, ROUND_HALF_UP)
                high_d = Decimal(str(row[bar_schema.high])).quantize(quantizer, ROUND_HALF_UP)
                low_d = Decimal(str(row[bar_schema.low])).quantize(quantizer, ROUND_HALF_UP)
                close_d = Decimal(str(row[bar_schema.close])).quantize(quantizer, ROUND_HALF_UP)

                bar_count += 1
                yield Bar(
                    ts=ts,
                    symbol=row[bar_schema.symbol],
                    open=open_d,
                    high=high_d,
                    low=low_d,
                    close=close_d,
                    volume=int(row[bar_schema.volume]),
                )

        logger.info("csv_adapter.bars_completed", bar_count=bar_count)

    def read_adjustments(self, source: Path, config: DataConfig) -> Iterator[AdjustmentEvent]:
        """
        Read adjustment metadata from CSV (if available).

        Returns empty if adjustment_schema not configured.
        """
        if config.adjustment_schema is None:
            logger.info("csv_adapter.no_adjustment_schema", msg="Skipping adjustment metadata")
            return

        adj_schema = config.adjustment_schema
        tz = pytz.timezone(config.timezone)

        csv_files = []
        if source.is_file():
            csv_files = [source]
        else:
            csv_files = sorted(source.glob("*.csv"))

        logger.info("csv_adapter.reading_adjustments", file_count=len(csv_files))

        adj_count = 0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, parse_dates=[adj_schema.ts])

            # Filter to rows with adjustments
            if adj_schema.event_type in df.columns:
                df = df[df[adj_schema.event_type].notna()]

            for _, row in df.iterrows():
                ts_naive = pd.Timestamp(row[adj_schema.ts])
                ts = tz.localize(ts_naive.to_pydatetime())

                px_f = row.get(adj_schema.px_factor)
                vol_f = row.get(adj_schema.vol_factor)

                px_factor = Decimal(str(px_f)).quantize(Decimal("0.0000001")) if pd.notna(px_f) else Decimal("1.0")
                vol_factor = Decimal(str(vol_f)).quantize(Decimal("0.0000001")) if pd.notna(vol_f) else Decimal("1.0")

                # Build metadata dict
                metadata = {}
                if adj_schema.metadata_fields:
                    for field in adj_schema.metadata_fields:
                        if field in row.index:
                            metadata[field] = row[field]

                adj_count += 1
                yield AdjustmentEvent(
                    ts=ts,
                    symbol=row[adj_schema.symbol],
                    event_type=row[adj_schema.event_type],
                    px_factor=px_factor,
                    vol_factor=vol_factor,
                    metadata=metadata,
                )

        logger.info("csv_adapter.adjustments_completed", adj_count=adj_count)
