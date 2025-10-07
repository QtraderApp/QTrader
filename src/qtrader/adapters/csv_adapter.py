"""Adapter for CSV files (e.g., security master or exported CSVs)."""

from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Dict, Iterator

import pandas as pd
import pytz

from qtrader.config.data_config import DataConfig
from qtrader.config.logging_config import LoggerFactory
from qtrader.models.bar import AdjustmentEvent, Bar, DataMode, PriceSeries
from qtrader.models.instrument import Instrument

logger = LoggerFactory.get_logger()


class CSVAdapter:
    """
    Adapter for CSV files.

    Can read single CSV file or directory with multiple CSVs.
    Useful for security master data or testing with exported CSV samples.
    Uses Instrument to determine symbol and constructs file paths.

    Data Mode: Configurable (depends on CSV source)

    Configuration (from data_sources.yaml):
        root_path: Base directory for CSV files

    Usage:
        config = {"root_path": "data/csv"}
        instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.CSV_FILE)
        adapter = CSVAdapter(config, instrument)
        bars = adapter.read_bars(data_config)
    """

    SCHEMA_VERSION = "csv-v1.0"

    def __init__(self, config: Dict[str, Any], instrument: Instrument):
        """
        Initialize adapter for specific instrument.

        Args:
            config: Configuration dict from data_sources.yaml
            instrument: Logical instrument specification

        Raises:
            ValueError: If required config fields missing
        """
        self.config = config
        self.instrument = instrument

        # Validate required config fields
        if "root_path" not in config:
            raise ValueError("Missing required config field: root_path")

        # Build data path (root_path + symbol.csv)
        root = Path(config["root_path"])
        self.data_path = root / f"{instrument.symbol}.csv"

        logger.info(
            "csv_adapter.initialized",
            symbol=instrument.symbol,
            data_path=str(self.data_path),
        )

    def can_read(self) -> bool:
        """Check if data path is a CSV file or directory with CSVs."""
        if self.data_path.is_file() and self.data_path.suffix == ".csv":
            return True
        if self.data_path.is_dir():
            return any(self.data_path.glob("*.csv"))
        return False

    def schema_version(self) -> str:
        """Return adapter schema version."""
        return self.SCHEMA_VERSION

    def get_data_mode(self) -> DataMode:
        """Data mode determined by config (default: adjusted)."""
        # CSV could be adjusted or unadjusted depending on source
        # Default to adjusted; can be overridden in config
        return DataMode.ADJUSTED

    def read_bars(self, config: DataConfig) -> Iterator[Bar]:
        """
        Read CSV files for this instrument and yield Bar objects.

        Args:
            config: DataConfig with schema mappings

        Yields:
            Bar objects in timestamp order
        """
        # Bar schema mapping
        bar_schema = config.bar_schema

        tz = pytz.timezone(config.timezone)
        price_decimals = config.decimals.get("price", 4)
        quantizer = Decimal(10) ** -price_decimals

        # Determine CSV files to read
        csv_files = []
        if self.data_path.is_file():
            csv_files = [self.data_path]
        else:
            csv_files = sorted(self.data_path.glob("*.csv"))

        logger.info("csv_adapter.reading_bars", symbol=self.instrument.symbol, file_count=len(csv_files))

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

                # TODO: CSV data is typically adjusted. Same limitation as Algoseek:
                # All 3 series get the same values until we have unadjusted source data
                price_series = PriceSeries(
                    open=open_d,
                    high=high_d,
                    low=low_d,
                    close=close_d,
                    volume=int(row[bar_schema.volume]),
                )

                bar_count += 1
                # Use instrument symbol (not CSV column which might differ)
                yield Bar(
                    ts=ts,
                    symbol=self.instrument.symbol,
                    unadjusted=price_series,  # TODO: Should be true unadjusted
                    capital_adjusted=price_series,  # TODO: Should be split-adjusted only
                    total_return=price_series,  # Assuming CSV data is adjusted
                    dividend=None,  # TODO: Extract if CSV has dividend columns
                    split=None,  # TODO: Extract if CSV has split columns
                )

        logger.info("csv_adapter.bars_completed", symbol=self.instrument.symbol, bar_count=bar_count)

    def read_adjustments(self, config: DataConfig) -> Iterator[AdjustmentEvent]:
        """
        Read adjustment metadata from CSV for this instrument (if available).

        Args:
            config: DataConfig with adjustment schema

        Yields:
            AdjustmentEvent objects

        Returns empty if adjustment_schema not configured.
        """
        if config.adjustment_schema is None:
            logger.info(
                "csv_adapter.no_adjustment_schema", symbol=self.instrument.symbol, msg="Skipping adjustment metadata"
            )
            return

        adj_schema = config.adjustment_schema
        tz = pytz.timezone(config.timezone)

        csv_files = []
        if self.data_path.is_file():
            csv_files = [self.data_path]
        else:
            csv_files = sorted(self.data_path.glob("*.csv"))

        logger.info("csv_adapter.reading_adjustments", symbol=self.instrument.symbol, file_count=len(csv_files))

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
                # Use instrument symbol (not CSV column)
                yield AdjustmentEvent(
                    ts=ts,
                    symbol=self.instrument.symbol,
                    event_type=row[adj_schema.event_type],
                    px_factor=px_factor,
                    vol_factor=vol_factor,
                    metadata=metadata,
                )

        logger.info("csv_adapter.adjustments_completed", symbol=self.instrument.symbol, adj_count=adj_count)
