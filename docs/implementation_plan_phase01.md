# QTrader Phase 1 — Implementation Plan

**Version:** 2.0 **Date:** October 3, 2025 **Status:** In Progress - Stage 2 **Reference:** `docs/specs/phase01.md` v1.0

**Architecture:**

- ✅ Vendor-agnostic Bar model (pure OHLCV)
- ✅ Multi-dataset support (primary + auxiliary)
- ✅ Self-contained strategies
- ✅ No backward compatibility constraints

**Stage 1 Status:** ✅ COMPLETE (36/36 tests passing)

______________________________________________________________________

## 📊 Data Schema Analysis

### Sample Dataset Characteristics

**Location:** `data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample/`

**Format:** Parquet partitioned by `SecId` (Hive-style: `SecId=33127/data_0.parquet`)

**Universe:**

- AAPL (SecId=33449)
- AMZN (SecId=33127)
- MSFT (SecId=39827)

**Date Range:** 2019-01-02 to 2023-12-29 (1,258 trading days)

**Schema:**

```python
TradeDate                 datetime64[ns]  # Trading date (no timezone in source)
Ticker                    str             # Symbol (e.g., "AAPL")
Open                      float64         # Opening price
High                      float64         # High price
Low                       float64         # Low price
Close                     float64         # Closing price
MarketHoursVolume         int64           # Volume during market hours
CumulativePriceFactor     float64         # Cumulative adjustment factor for prices
CumulativeVolumeFactor    float64         # Cumulative adjustment factor for volume
AdjustmentFactor          float64         # Period adjustment (usually NaN)
AdjustmentReason          str|None        # e.g., "CashDiv", None for no adjustment
SecId                     int64           # Algoseek security identifier
```

**Adjustment Events:**

- AAPL: 20 dividend events (CashDiv)
- MSFT: 20 dividend events (CashDiv)
- AMZN: 0 dividend events
- No splits in sample period

**Data Mode:** `ADJUSTED` (total-return adjusted - dividends and splits embedded in OHLCV)

**Adapter Responsibility:** Convert Algoseek schema → Canonical Bar (OHLCV only) + Adjustment metadata

**Schema Mapping:**

```yaml
# Canonical Bar (OHLCV) - Universal Contract
bar_schema:
  ts: TradeDate
  symbol: Ticker
  open: Open
  high: High
  low: Low
  close: Close
  volume: MarketHoursVolume

# Adjustment Metadata (Optional) - Vendor-Specific
adjustment_schema:
  ts: TradeDate
  symbol: Ticker
  event_type: AdjustmentReason      # "CashDiv" → event_type
  px_factor: CumulativePriceFactor
  vol_factor: CumulativeVolumeFactor
```

______________________________________________________________________

## 🎯 Implementation Strategy

### Core Principles

1. **Vendor-Agnostic Bar Model:**

   - Canonical Bar = Universal OHLCV contract (works with ANY vendor)
   - Adjustment metadata = Separate, optional (vendor-specific)
   - DataMode declares if prices are adjusted or unadjusted
   - Adapter normalizes vendor schema → canonical Bar

2. **Decimal Precision:**

   - Bar prices (open/high/low/close): `Decimal` from adapter onward
   - Ledger (cash, PnL, costs): `Decimal`
   - Strategy indicators: `float64` for performance
   - Convert at adapter boundary and before indicators

3. **Public API First:**

   - Design as installable package (`pip install qtrader`)
   - Public API: `Strategy`, `Context`, `Backtest`, `load_config`, `run_backtest`
   - Internal engine components are private (not part of public API)
   - CLI entrypoint: `qtrader backtest`

4. **Data Adapters:**

   - Primary: Parquet adapter using DuckDB (matches fixture format)
   - Secondary: CSV adapter for security master linkage
   - Adapters emit canonical `Bar` objects (OHLCV only)
   - Adjustment metadata stored separately (optional)

5. **Testing Approach:**

   - TDD: Write tests first for each component
   - Focus on functional paths (not line coverage)
   - Use fixture data for all tests
   - Generate golden baselines in final stage

6. **Golden Baseline Generation:**

   - Create standalone scripts in `scripts/goldens/`
   - Run Buy-and-Hold and SMA Cross on fixture data
   - Manually validate results together
   - Commit golden files to `tests/goldens/fixtures/`
   - Automate validation in CI

______________________________________________________________________

## 🚀 Implementation Stages

### **Stage 1: Core Data Models & Adapters (Foundation)**

**Timeline:** Days 1-3 **Branch:** `stage-1-data-foundation`

#### Deliverables

##### 1.1 Project Structure & Dependencies

**Update `pyproject.toml`:**

```toml
[project]
name = "qtrader"
version = "0.1.0"
description = "Quantitative Trading Environment - Deterministic Backtesting Engine"
requires-python = ">=3.13"

dependencies = [
    "duckdb>=1.4.0",
    "pandas>=2.3.2",
    "pyarrow>=21.0.0",
    "click>=8.0.0",
    "pydantic>=2.11.9",
    "pyyaml>=6.0",
    "pytz>=2024.1",
    "structlog>=24.4.0",
]

[project.scripts]
qtrader = "qtrader.cli:main"

[tool.structlog]
# Structured logging configuration
```

**Directory Structure:**

```txt
src/
├── qtrader/                    # Public package (was just "src/")
│   ├── __init__.py            # Public API exports
│   ├── cli.py                 # CLI entrypoint
│   ├── api/                   # Public API
│   │   ├── __init__.py
│   │   ├── strategy.py        # Strategy base class
│   │   ├── context.py         # Context for strategy
│   │   └── backtest.py        # Backtest runner
│   ├── models/                # Core data models
│   │   ├── __init__.py
│   │   └── bar.py            # Bar, BarFrequency, DataMode, OHLCPolicy
│   ├── config/                # Configuration
│   │   ├── __init__.py
│   │   ├── data_config.py     # DataConfig, ValidationConfig
│   │   └── engine_config.py   # EngineConfig (fills, trading, costs)
│   ├── adapters/              # Data adapters (private)
│   │   ├── __init__.py
│   │   ├── base.py            # DataAdapter protocol
│   │   ├── algoseek_parquet.py
│   │   └── csv_adapter.py
│   ├── validation/            # Bar validation (private)
│   │   ├── __init__.py
│   │   └── bar_validator.py
│   └── engine/                # Execution engine (private, future stages)
```

##### 1.2 Core Bar Model (`qtrader/models/bar.py`)

```python
"""Core data model for OHLCV bars and adjustment metadata."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, NamedTuple, Optional


class Bar(NamedTuple):
    """
    Canonical OHLCV bar - the ONLY contract consumed by execution engine.

    This is vendor-agnostic, asset-agnostic, and frequency-agnostic.
    Works with equities, futures, crypto, forex at any timeframe.

    All vendor data is normalized to this format at the adapter boundary.
    Vendor-specific fields (adjustments, bid/ask, etc.) are stored separately.

    Attributes:
        ts: Timezone-aware timestamp of the bar
        symbol: Ticker/contract identifier (e.g., "AAPL", "ESH24")
        open: Opening price (Decimal for precision)
        high: High price (Decimal for precision)
        low: Low price (Decimal for precision)
        close: Closing price (Decimal for precision)
        volume: Volume in shares/contracts (integer)
    """

    ts: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class AdjustmentEvent(NamedTuple):
    """
    Corporate action metadata for analysis and validation.

    NOT used by execution engine. Stored for:
    - Audit trail (data provenance)
    - Performance attribution (dividend-adjusted returns)
    - Data validation (detect missing adjustments)

    Attributes:
        ts: Event timestamp (ex-date for dividends)
        symbol: Ticker symbol
        event_type: Type of corporate action (CashDiv, Split, StockDiv, SpinOff)
        px_factor: Cumulative price adjustment factor
        vol_factor: Cumulative volume adjustment factor
        metadata: Vendor-specific details (amount, ratio, etc.)
    """

    ts: datetime
    symbol: str
    event_type: str
    px_factor: Decimal
    vol_factor: Decimal
    metadata: dict[str, Any]


class BarFrequency(Enum):
    """Supported bar frequencies for backtesting."""

    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"


class DataMode(Enum):
    """
    Data adjustment mode - declares how prices in Bar are adjusted.

    ADJUSTED: Total-return adjusted (dividends + splits embedded in OHLCV)
    UNADJUSTED: Raw trade prices (no adjustments applied)
    SPLIT_ADJUSTED: Only splits adjusted, dividends not embedded
    """

    ADJUSTED = "adjusted"
    UNADJUSTED = "unadjusted"
    SPLIT_ADJUSTED = "split_adjusted"


class OHLCPolicy(Enum):
    """
    Policies for handling malformed OHLC bars.

    STRICT_RAISE: Raise error on first malformed bar (fail fast)
    WARN_SKIP_BAR: Log warning and skip the bar entirely (no fills, orders remain pending)
    WARN_USE_CLOSE_ONLY: Log warning, allow bar but disable limit/stop evaluation (close-only mode)
    """

    STRICT_RAISE = "strict_raise"
    WARN_SKIP_BAR = "warn_skip_bar"
    WARN_USE_CLOSE_ONLY = "warn_use_close_only"
```

##### 1.3 Configuration Schema (`qtrader/config/data_config.py`)

```python
"""Configuration for data loading and validation."""

from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field


class ValidationConfig(BaseModel):
    """OHLC validation configuration."""

    epsilon: float = Field(default=0.0, description="Tolerance for OHLC checks")
    ohlc_policy: str = Field(default="strict_raise", description="Policy for malformed bars")
    close_only_fields: list[str] = Field(default=["close"], description="Fields to trust in close-only mode")


class BarSchemaConfig(BaseModel):
    """Mapping from vendor columns to canonical Bar fields."""

    ts: str = Field(description="Vendor column for timestamp")
    symbol: str = Field(description="Vendor column for symbol")
    open: str = Field(description="Vendor column for open price")
    high: str = Field(description="Vendor column for high price")
    low: str = Field(description="Vendor column for low price")
    close: str = Field(description="Vendor column for close price")
    volume: str = Field(description="Vendor column for volume")


class AdjustmentSchemaConfig(BaseModel):
    """Mapping from vendor columns to AdjustmentEvent fields (optional)."""

    ts: str = Field(description="Vendor column for event timestamp")
    symbol: str = Field(description="Vendor column for symbol")
    event_type: str = Field(description="Vendor column for adjustment type")
    px_factor: str = Field(description="Vendor column for price factor")
    vol_factor: str = Field(description="Vendor column for volume factor")
    metadata_fields: list[str] = Field(default_factory=list, description="Additional fields to capture")


class DataConfig(BaseModel):
    """Data loading and processing configuration."""

    mode: str = Field(default="adjusted", description="Data adjustment mode (adjusted|unadjusted|split_adjusted)")
    frequency: str = Field(default="1d", description="Bar frequency (1m|5m|15m|1h|1d)")
    timezone: str = Field(default="America/New_York", description="Timezone for timestamps")
    strict_frequency: bool = Field(default=True, description="Raise on frequency mismatch")
    decimals: Dict[str, int] = Field(default={"price": 4, "cash": 4}, description="Decimal precision")
    source_tag: str = Field(default="algoseek-adjusted", description="Data source identifier")
    validation: ValidationConfig = Field(default_factory=ValidationConfig, description="Validation rules")
    bar_schema: BarSchemaConfig = Field(description="Vendor schema → Bar mapping")
    adjustment_schema: Optional[AdjustmentSchemaConfig] = Field(default=None, description="Vendor schema → AdjustmentEvent mapping (optional)")

    @classmethod
    def from_yaml(cls, path: Path) -> "DataConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data.get("data", {}))
```

##### 1.4 Data Adapter Protocol (`qtrader/adapters/base.py`)

```python
"""Protocol for data adapters that normalize vendor data to canonical Bar."""

from pathlib import Path
from typing import Iterator, Protocol

from qtrader.config.data_config import DataConfig
from qtrader.models.bar import Bar, AdjustmentEvent, DataMode


class DataAdapter(Protocol):
    """
    Protocol for data adapters that normalize vendor data to canonical Bar.

    Responsibilities:
    1. Convert vendor schema → canonical Bar (OHLCV only)
    2. Optionally extract adjustment metadata → AdjustmentEvent
    3. Declare DataMode (adjusted, unadjusted, split_adjusted)

    New vendor = new DataAdapter + schema config + unit tests; engine unchanged.
    """

    def can_read(self, source: Path) -> bool:
        """Check if this adapter can read from the given source."""
        ...

    def schema_version(self) -> str:
        """
        Return the adapter schema version for reproducibility.

        This version is persisted in run.json for audit trail.
        Example: "algoseek-parquet-v1.0", "iqfeed-tick-v2.1"
        """
        ...

    def get_data_mode(self) -> DataMode:
        """
        Declare if OHLCV prices are adjusted or unadjusted.

        Critical for execution engine to interpret prices correctly.
        """
        ...

    def read_bars(self, source: Path, config: DataConfig) -> Iterator[Bar]:
        """
        Read and normalize data to canonical Bar objects.

        Pipeline: Read RawRecord → Map columns → Convert types → Validate → Emit Bar

        Yields:
            Bar objects with:
            - Decimal prices (quantized to config.decimals.price)
            - Timezone-aware timestamps
            - Validated OHLC relationships (high >= max(o,c), low <= min(o,c))
        """
        ...

    def read_adjustments(self, source: Path, config: DataConfig) -> Iterator[AdjustmentEvent]:
        """
        Read adjustment metadata (optional).

        Returns empty iterator if:
        - Data is unadjusted with no adjustment table
        - Vendor doesn't provide adjustment metadata

        Yields:
            AdjustmentEvent objects with corporate action details
        """
        ...
```

##### 1.5 Algoseek Parquet Adapter (`qtrader/adapters/algoseek_parquet.py`)

```python
"""Adapter for Algoseek parquet data with Hive partitioning."""

import structlog
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterator

import duckdb
import pytz

from qtrader.adapters.base import DataAdapter
from qtrader.config.data_config import DataConfig
from qtrader.models.bar import Bar, AdjustmentEvent, DataMode

logger = structlog.get_logger()


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
            SELECT {', '.join(select_cols)}
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
            SELECT {', '.join(select_cols)}{', ' + metadata_cols if metadata_cols else ''}
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
```

##### 1.6 CSV Adapter (`qtrader/adapters/csv_adapter.py`)

```python
"""Adapter for CSV files (e.g., security master or exported CSVs)."""

import structlog
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Iterator

import pandas as pd
import pytz

from qtrader.adapters.base import DataAdapter
from qtrader.config.data_config import DataConfig
from qtrader.models.bar import Bar, AdjustmentEvent, DataMode

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
```

##### 1.7 Bar Validator (`qtrader/validation/bar_validator.py`)

```python
"""Validates Bar integrity and applies OHLC policies."""

import structlog
from datetime import timedelta
from decimal import Decimal
from typing import Optional

from qtrader.config.data_config import DataConfig
from qtrader.models.bar import Bar, OHLCPolicy

logger = structlog.get_logger()


class BarValidator:
    """
    Validates Bar integrity and applies OHLC policies.

    Tracks statistics on malformed bars for inclusion in run.json.
    """

    def __init__(self, config: DataConfig):
        self.config = config
        self.epsilon = Decimal(str(config.validation.epsilon))
        self.policy = OHLCPolicy(config.validation.ohlc_policy)
        self.malformed_count = 0
        self.skipped_count = 0
        self.close_only_count = 0
        self.malformed_samples: list[tuple[str, str]] = []  # (date, symbol) for first/last 10

    def validate_ohlc(self, bar: Bar) -> tuple[bool, Optional[str]]:
        """
        Validate OHLC relationships.

        Returns:
            (is_valid, reason) where reason is None if valid
        """
        # Check high >= max(open, close)
        if bar.high < max(bar.open, bar.close) - self.epsilon:
            return False, f"high ({bar.high}) < max(open={bar.open}, close={bar.close})"

        # Check low <= min(open, close)
        if bar.low > min(bar.open, bar.close) + self.epsilon:
            return False, f"low ({bar.low}) > min(open={bar.open}, close={bar.close})"

        # Check low <= high
        if bar.low > bar.high + self.epsilon:
            return False, f"low ({bar.low}) > high ({bar.high})"

        # Check volume >= 0
        if bar.volume < 0:
            return False, f"volume ({bar.volume}) < 0"

        return True, None

    def process_bar(self, bar: Bar) -> tuple[Optional[Bar], bool]:
        """
        Process bar according to OHLC policy.

        Returns:
            (bar_or_none, is_close_only)
            - bar_or_none: None if skipped, otherwise the bar
            - is_close_only: True if bar should only use close (no limit/stop)
        """
        is_valid, reason = self.validate_ohlc(bar)

        if is_valid:
            return bar, False

        # Malformed bar - apply policy
        self.malformed_count += 1

        # Store sample for reporting (first/last 10)
        if len(self.malformed_samples) < 10 or self.malformed_count > (self.malformed_count - 10):
            self.malformed_samples.append((bar.ts.strftime("%Y-%m-%d"), bar.symbol))
            if len(self.malformed_samples) > 20:  # Keep first 10 and last 10
                self.malformed_samples = self.malformed_samples[:10] + self.malformed_samples[-10:]

        if self.policy == OHLCPolicy.STRICT_RAISE:
            logger.error("bar_validator.malformed_strict", symbol=bar.symbol, ts=bar.ts, reason=reason)
            raise ValueError(f"Malformed OHLC bar at {bar.ts} for {bar.symbol}: {reason}")

        elif self.policy == OHLCPolicy.WARN_SKIP_BAR:
            logger.warning("bar_validator.malformed_skip", symbol=bar.symbol, ts=bar.ts, reason=reason)
            self.skipped_count += 1
            return None, False

        elif self.policy == OHLCPolicy.WARN_USE_CLOSE_ONLY:
            logger.warning("bar_validator.malformed_close_only", symbol=bar.symbol, ts=bar.ts, reason=reason)
            self.close_only_count += 1
            return bar, True

        return bar, False

    def validate_frequency(self, bars: list[Bar]) -> bool:
        """
        Validate that bars match expected frequency.

        Returns True if valid, raises ValueError if strict_frequency=true and invalid.
        """
        if not self.config.strict_frequency or len(bars) < 3:
            return True

        # Calculate median delta per symbol
        deltas = []
        for i in range(1, len(bars)):
            if bars[i].symbol == bars[i - 1].symbol:
                delta = bars[i].ts - bars[i - 1].ts
                deltas.append(delta)

        if not deltas:
            return True

        deltas.sort()
        median_delta = deltas[len(deltas) // 2]

        # Expected delta for frequency
        freq = self.config.frequency
        expected_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "1d": timedelta(days=1),
        }

        expected = expected_deltas.get(freq)
        if expected and abs((median_delta - expected).total_seconds()) > 3600:  # 1 hour tolerance
            msg = f"Frequency mismatch: expected {freq}, median delta {median_delta}"
            if self.config.strict_frequency:
                logger.error("bar_validator.frequency_mismatch", expected=freq, median=str(median_delta))
                raise ValueError(msg)
            logger.warning("bar_validator.frequency_mismatch", expected=freq, median=str(median_delta))
            return False

        logger.info("bar_validator.frequency_validated", expected=freq, median=str(median_delta))
        return True

    def get_stats(self) -> dict:
        """Return validation statistics for run.json."""
        return {
            "malformed_bars": self.malformed_count,
            "skipped": self.skipped_count,
            "close_only": self.close_only_count,
            "malformed_samples": self.malformed_samples,
        }
```

##### 1.8 Public API Stubs (`qtrader/__init__.py` and `qtrader/api/`)

For Stage 1, we'll create stubs for the public API to establish the package structure. Full implementation comes in later stages.

```python
# qtrader/__init__.py
"""
QTrader - Quantitative Trading Environment

Public API for building and running deterministic backtests.
"""

__version__ = "0.1.0"

# Public API exports (will be implemented in later stages)
from qtrader.api.strategy import Strategy
from qtrader.api.context import Context
from qtrader.api.backtest import Backtest, load_config, run_backtest

__all__ = [
    "Strategy",
    "Context",
    "Backtest",
    "load_config",
    "run_backtest",
    "__version__",
]
```

```python
# qtrader/api/strategy.py
"""Base Strategy class for user strategies."""

from typing import Protocol


class Strategy(Protocol):
    """
    Base strategy protocol for user-defined trading strategies.

    Users subclass this and implement on_bar() at minimum.
    """

    def on_start(self, ctx) -> None:
        """Called once before first bar. Optional."""
        pass

    def on_bar(self, bar, ctx) -> None:
        """
        Called for each bar in the dataset. Required.

        Args:
            bar: Current Bar object
            ctx: Context for accessing indicators and submitting orders
        """
        ...

    def on_fill(self, fill, ctx) -> None:
        """Called after each fill. Optional."""
        pass

    def on_end(self, ctx) -> None:
        """Called once after last bar. Optional."""
        pass
```

```python
# qtrader/api/context.py
"""Context object passed to strategy methods."""


class Context:
    """
    Context object providing strategy interface to engine.

    Stub for Stage 1. Full implementation in later stages.
    """

    def __init__(self):
        # TODO: Implement in later stages
        pass

    def buy_market(self, qty: int) -> str:
        """Submit market buy order."""
        raise NotImplementedError("Stage 1: Context stub only")

    def sell_market(self, qty: int) -> str:
        """Submit market sell order."""
        raise NotImplementedError("Stage 1: Context stub only")
```

```python
# qtrader/api/backtest.py
"""Backtest runner and config loader."""

from pathlib import Path


def load_config(path: Path):
    """Load configuration from YAML file."""
    raise NotImplementedError("Stage 1: Stub only")


class Backtest:
    """
    Backtest runner.

    Stub for Stage 1. Full implementation in later stages.
    """

    def __init__(self, config, strategy):
        self.config = config
        self.strategy = strategy

    def run(self, out_dir: Path):
        """Run backtest and write outputs."""
        raise NotImplementedError("Stage 1: Stub only")


def run_backtest(config_path: Path, strategy_class, out_dir: Path):
    """Convenience function to run backtest."""
    raise NotImplementedError("Stage 1: Stub only")
```

```python
# qtrader/cli.py
"""Command-line interface."""

import click


@click.group()
@click.version_option()
def main():
    """QTrader - Quantitative Trading Environment"""
    pass


@main.command()
@click.option("--strategy", type=click.Path(exists=True), required=True,
              help="Path to self-contained strategy Python file")
@click.option("--data", type=click.Path(exists=True), required=False,
              help="Path to data configuration YAML (optional, uses defaults if omitted)")
@click.option("--out", type=click.Path(), required=True,
              help="Output directory for results")
@click.option("--set", "overrides", multiple=True,
              help="Override strategy config: --set param=value")
def backtest(strategy, data, out, overrides):
    """
    Run a backtest with a self-contained strategy file.

    Strategy file must contain a Strategy class and optionally a config.
    Data config YAML contains system settings (data source, adapter, validation).

    Examples:
        # Basic usage
        qtrader backtest --strategy my_strategy.py --out results/

        # With data config
        qtrader backtest --strategy my_strategy.py --data algoseek.yaml --out results/

        # With parameter overrides
        qtrader backtest --strategy my_strategy.py --data algoseek.yaml --out results/ \\
          --set fast_period=10 --set position_size=200
    """
    click.echo("Stage 1: CLI stub only")
    raise NotImplementedError("Full CLI implementation in later stages")


@main.command()
@click.option("--data", type=click.Path(exists=True), required=True,
              help="Path to data configuration YAML")
def validate_data(data):
    """
    Validate dataset without running backtest.

    Loads data according to config and validates:
    - All bars load successfully
    - OHLC relationships are valid
    - Frequency matches expected
    - No missing data gaps

    Example:
        qtrader validate-data --data algoseek.yaml
    """
    click.echo("Stage 1: CLI stub only")
    raise NotImplementedError("Full CLI implementation in later stages")


if __name__ == "__main__":
    main()
```

#### Tests (`tests/stage1/`)

##### Test Bar Model

```python
# tests/stage1/test_bar_model.py
"""Tests for Bar model and enums."""

import pytest
import pytz
from datetime import datetime
from decimal import Decimal

from qtrader.models.bar import Bar, BarFrequency, DataMode, OHLCPolicy


def test_bar_creation_with_decimal_prices():
    """Bar should store prices as Decimal (OHLCV only)."""
    bar = Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("150.25"),
        high=Decimal("151.50"),
        low=Decimal("149.75"),
        close=Decimal("151.00"),
        volume=1000000,
    )
    assert isinstance(bar.open, Decimal)
    assert bar.open == Decimal("150.25")
    assert isinstance(bar.close, Decimal)
    assert bar.volume == 1000000


def test_bar_is_vendor_agnostic():
    """Bar should be vendor-agnostic (no vendor-specific fields)."""
    bar = Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="AAPL",
        open=Decimal("150.25"),
        high=Decimal("151.50"),
        low=Decimal("149.75"),
        close=Decimal("151.00"),
        volume=1000000,
    )
    # Bar should only have OHLCV fields
    assert hasattr(bar, "ts")
    assert hasattr(bar, "symbol")
    assert hasattr(bar, "open")
    assert hasattr(bar, "high")
    assert hasattr(bar, "low")
    assert hasattr(bar, "close")
    assert hasattr(bar, "volume")
    # No vendor-specific fields
    assert not hasattr(bar, "adj_reason")
    assert not hasattr(bar, "px_factor")


def test_adjustment_event_creation():
    """AdjustmentEvent should store adjustment metadata separately."""
    from qtrader.models.bar import AdjustmentEvent

    event = AdjustmentEvent(
        ts=datetime(2023, 2, 8, tzinfo=pytz.UTC),
        symbol="AAPL",
        event_type="CashDiv",
        px_factor=Decimal("7.9599520"),
        vol_factor=Decimal("7.0"),
        metadata={"amount": 0.23, "currency": "USD"},
    )
    assert event.event_type == "CashDiv"
    assert event.px_factor == Decimal("7.9599520")
    assert event.metadata["amount"] == 0.23


def test_bar_frequency_enum():
    """BarFrequency enum should have expected values."""
    assert BarFrequency.DAY_1.value == "1d"
    assert BarFrequency.MIN_5.value == "5m"
    assert BarFrequency.HOUR_1.value == "1h"


def test_ohlc_policy_enum():
    """OHLCPolicy enum should have expected values."""
    assert OHLCPolicy.STRICT_RAISE.value == "strict_raise"
    assert OHLCPolicy.WARN_SKIP_BAR.value == "warn_skip_bar"
    assert OHLCPolicy.WARN_USE_CLOSE_ONLY.value == "warn_use_close_only"
```

##### Test Algoseek Adapter

```python
# tests/stage1/test_algoseek_adapter.py
"""Tests for Algoseek Parquet adapter."""

import pytest
from pathlib import Path
from datetime import date
from decimal import Decimal

from qtrader.adapters.algoseek_parquet import AlgoseekParquetAdapter
from qtrader.config.data_config import DataConfig


@pytest.fixture
def fixture_path():
    """Path to sample parquet data."""
    return Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample")


@pytest.fixture
def data_config():
    """Default data configuration."""
    return DataConfig(timezone="America/New_York", frequency="1d")


def test_adapter_can_read_fixture(fixture_path):
    """Adapter should detect parquet files."""
    adapter = AlgoseekParquetAdapter()
    assert adapter.can_read(fixture_path) is True


def test_adapter_schema_version():
    """Adapter should report schema version."""
    adapter = AlgoseekParquetAdapter()
    assert adapter.schema_version() == "algoseek-parquet-v1.0"


def test_adapter_data_mode():
    """Adapter should declare data mode."""
    from qtrader.models.bar import DataMode

    adapter = AlgoseekParquetAdapter()
    assert adapter.get_data_mode() == DataMode.ADJUSTED


def test_adapter_reads_fixture_bars(fixture_path, data_config):
    """Adapter should load all bars from fixture (OHLCV only)."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_bars(fixture_path, data_config))

    # Should have 3 symbols × 1258 days = 3774 bars
    assert len(bars) == 3774

    # Check symbols are present
    symbols = {bar.symbol for bar in bars}
    assert symbols == {"AAPL", "MSFT", "AMZN"}


def test_adapter_first_bar_format(fixture_path, data_config):
    """First bar should have correct format and types (OHLCV only)."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_bars(fixture_path, data_config))

    # Find first AAPL bar
    aapl_bars = [b for b in bars if b.symbol == "AAPL"]
    assert len(aapl_bars) == 1258

    first_bar = aapl_bars[0]
    assert first_bar.ts.date() == date(2019, 1, 2)
    assert first_bar.symbol == "AAPL"
    assert isinstance(first_bar.open, Decimal)
    assert isinstance(first_bar.close, Decimal)
    assert first_bar.close == Decimal("157.92")  # From data inspection
    assert first_bar.volume == 30606605
    # Bar should NOT have adjustment fields
    assert not hasattr(first_bar, "adj_reason")
    assert not hasattr(first_bar, "px_factor")


def test_adapter_reads_adjustments_separately(fixture_path, data_config):
    """Adapter should read adjustment metadata separately from bars."""
    from qtrader.models.bar import AdjustmentEvent

    adapter = AlgoseekParquetAdapter()
    adjustments = list(adapter.read_adjustments(fixture_path, data_config))

    # AAPL + MSFT have ~40 dividend events in 2019-2023
    assert len(adjustments) >= 30
    assert all(isinstance(adj, AdjustmentEvent) for adj in adjustments)
    assert all(adj.event_type == "CashDiv" for adj in adjustments)

    # Check factors are preserved
    assert all(adj.px_factor is not None for adj in adjustments)
    assert all(isinstance(adj.px_factor, Decimal) for adj in adjustments)


def test_adapter_bars_sorted_by_symbol_and_time(fixture_path, data_config):
    """Bars should be sorted by symbol, then timestamp."""
    adapter = AlgoseekParquetAdapter()
    bars = list(adapter.read_iter(fixture_path, data_config))

    # Check ordering
    for i in range(1, len(bars)):
        prev, curr = bars[i - 1], bars[i]
        # Either symbol increases, or same symbol with increasing time
        if prev.symbol == curr.symbol:
            assert curr.ts > prev.ts, f"Timestamps not sorted for {curr.symbol}"
        else:
            assert curr.symbol > prev.symbol, "Symbols not sorted"
```

##### Test CSV Adapter

```python
# tests/stage1/test_csv_adapter.py
"""Tests for CSV adapter."""

import pytest
from pathlib import Path
from decimal import Decimal

from qtrader.adapters.csv_adapter import CSVAdapter
from qtrader.config.data_config import DataConfig


@pytest.fixture
def csv_path():
    """Path to CSV sample data."""
    return Path("data/csv")


@pytest.fixture
def data_config():
    """Default data configuration."""
    return DataConfig(timezone="America/New_York")


def test_csv_adapter_can_read_directory(csv_path):
    """CSV adapter should detect CSV directory."""
    adapter = CSVAdapter()
    assert adapter.can_read(csv_path) is True


def test_csv_adapter_reads_bars(csv_path, data_config):
    """CSV adapter should load bars from CSV files (OHLCV only)."""
    adapter = CSVAdapter()
    bars = list(adapter.read_bars(csv_path, data_config))

    # Should have 3 files × 1258 lines = 3774 bars
    assert len(bars) == 3774

    # Check types
    first_bar = bars[0]
    assert isinstance(first_bar.open, Decimal)
    assert isinstance(first_bar.close, Decimal)
    assert isinstance(first_bar.volume, int)
    # Bar should NOT have adjustment fields
    assert not hasattr(first_bar, "adj_reason")


def test_csv_adapter_matches_parquet_data(csv_path, data_config):
    """CSV data should match parquet data (sanity check)."""
    adapter = CSVAdapter()
    bars = list(adapter.read_bars(csv_path, data_config))

    # Find AAPL first bar
    aapl_bars = [b for b in bars if b.symbol == "AAPL"]
    assert len(aapl_bars) == 1258

    first_bar = aapl_bars[0]
    # Should match parquet data (OHLCV only)
    assert first_bar.close == Decimal("157.92")
```

##### Test Bar Validator

```python
# tests/stage1/test_bar_validator.py
"""Tests for Bar validator."""

import pytest
import pytz
from datetime import datetime
from decimal import Decimal

from qtrader.config.data_config import DataConfig, ValidationConfig
from qtrader.validation.bar_validator import BarValidator
from qtrader.models.bar import Bar


@pytest.fixture
def good_bar():
    """Create a valid bar."""
    return Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("99"),
        close=Decimal("102"),
        volume=1000000,
    )


@pytest.fixture
def bad_bar_high_low():
    """Create bar with high < open."""
    return Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("99"),  # Invalid: high < open
        low=Decimal("98"),
        close=Decimal("100"),
        volume=1000,
    )


@pytest.fixture
def bad_bar_low_high():
    """Create bar with low > close."""
    return Bar(
        ts=datetime(2023, 1, 1, tzinfo=pytz.UTC),
        symbol="TEST",
        open=Decimal("100"),
        high=Decimal("105"),
        low=Decimal("103"),  # Invalid: low > close
        close=Decimal("102"),
        volume=1000,
    )


def test_validator_accepts_good_bar(good_bar):
    """Validator should accept valid bars."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="strict_raise"))
    validator = BarValidator(config)

    is_valid, reason = validator.validate_ohlc(good_bar)
    assert is_valid is True
    assert reason is None


def test_validator_detects_bad_ohlc(bad_bar_high_low):
    """Validator should detect malformed OHLC."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="strict_raise"))
    validator = BarValidator(config)

    is_valid, reason = validator.validate_ohlc(bad_bar_high_low)
    assert is_valid is False
    assert "high" in reason


def test_validator_strict_raise_policy(bad_bar_high_low):
    """Validator should raise on malformed bar with strict policy."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="strict_raise"))
    validator = BarValidator(config)

    with pytest.raises(ValueError, match="Malformed OHLC"):
        validator.process_bar(bad_bar_high_low)


def test_validator_warn_skip_bar_policy(bad_bar_high_low):
    """Validator should skip bar with warn_skip_bar policy."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="warn_skip_bar"))
    validator = BarValidator(config)

    result, is_close_only = validator.process_bar(bad_bar_high_low)
    assert result is None
    assert is_close_only is False
    assert validator.skipped_count == 1


def test_validator_warn_use_close_only_policy(bad_bar_high_low):
    """Validator should allow bar but flag as close-only."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="warn_use_close_only"))
    validator = BarValidator(config)

    result, is_close_only = validator.process_bar(bad_bar_high_low)
    assert result is not None
    assert is_close_only is True
    assert validator.close_only_count == 1


def test_validator_tracks_statistics(bad_bar_high_low, bad_bar_low_high):
    """Validator should track statistics for run.json."""
    config = DataConfig(validation=ValidationConfig(ohlc_policy="warn_skip_bar"))
    validator = BarValidator(config)

    validator.process_bar(bad_bar_high_low)
    validator.process_bar(bad_bar_low_high)

    stats = validator.get_stats()
    assert stats["malformed_bars"] == 2
    assert stats["skipped"] == 2
    assert len(stats["malformed_samples"]) == 2
```

#### Acceptance Criteria

- ✅ Load 3,774 bars from parquet fixture (3 symbols × 1,258 days)
- ✅ Bar model is vendor-agnostic (OHLCV only, no vendor-specific fields)
- ✅ All prices stored as Decimal with 4 decimal places
- ✅ Timestamps timezone-aware (America/New_York)
- ✅ Adjustment events stored separately (40 CashDiv events as AdjustmentEvent objects)
- ✅ Adapter declares DataMode (ADJUSTED for Algoseek)
- ✅ All OHLC validation policies work correctly
- ✅ CSV adapter reads data matching parquet (OHLCV only)
- ✅ Public API package structure established
- ✅ CLI entrypoint created (stub)
- ✅ Config schema supports bar_schema and adjustment_schema mappings
- ✅ All Stage 1 tests pass (`make test`)
- ✅ Code quality passes (`make qa`)

______________________________________________________________________

### **Stage 2: Order Models & Ledger Foundation**

**Timeline:** Days 4-6 **Branch:** `stage-2-orders-ledger`

#### Summary

Implement order types, position tracking, and cash ledger. This stage builds the foundation for execution without actually filling orders yet.

**Key Components:**

- Order models (Market, MOC, Limit, Stop)
- Position tracker with average cost method
- Cash ledger with Decimal precision
- Order manager with validation (short selling checks)

**Tests Focus:**

- Order creation and state transitions
- Position updates on simulated fills
- Cash debit/credit operations
- Short selling validation

______________________________________________________________________

### **Stage 3: Execution Engine — Market & MOC**

**Timeline:** Days 7-10 **Branch:** `stage-3-market-moc`

#### Summary

Implement execution engine event loop with Market and MOC order fills. This is the first stage where orders actually execute.

**Key Components:**

- Execution engine with event loop
- Fill policy (conservative rules)
- Commission calculation (per-share + ticket minimum)
- Market orders fill at next bar open
- MOC orders fill at current bar close with slippage

**Tests Focus:**

- Market orders fill at correct price
- MOC orders with slippage calculation
- Commission enforcement
- Ledger updates after fills

______________________________________________________________________

### **Stage 4: Execution Engine — Limit & Stop**

**Timeline:** Days 11-13 **Branch:** `stage-4-limit-stop`

#### Summary

Add limit and stop order execution with conservative touch rules. Implement close-only bar handling.

**Key Components:**

- Limit/Stop evaluation logic
- Conservative vs optimistic fill modes
- Close-only bar detection (skip limit/stop evaluation)
- DAY TIF expiration

**Tests Focus:**

- All 4 limit/stop conservative cases
- Close-only bars skip limit/stop
- DAY orders expire correctly

______________________________________________________________________

### **Stage 5: Volume Participation & Partials**

**Timeline:** Days 14-16 **Branch:** `stage-5-participation`

#### Summary

Implement volume participation caps with partial fills and residual queuing.

**Key Components:**

- Participation cap calculation
- Partial fill tracking
- Residual queue management
- High participation guardrail

**Tests Focus:**

- Large orders split into partials
- Residuals carried forward
- Queue expiration after N bars
- Guardrail warns and clamps

______________________________________________________________________

### **Stage 6: Shorting, Accruals & Outputs**

**Timeline:** Days 17-20 **Branch:** `stage-6-accruals-outputs`

#### Summary

Complete ledger with short dividends, borrow costs, and output file generation.

**Key Components:**

- Short dividend handler (ex-date detection)
- Borrow cost accrual (EOD)
- Output writers (CSV, JSON)
- Run orchestrator

**Tests Focus:**

- Short dividends debited correctly
- Borrow costs accrue daily
- All output files generated
- Run.json contains complete metadata

______________________________________________________________________

### **Stage 7: Public API & CLI**

**Timeline:** Days 21-25 **Branch:** `stage-7-api-cli`

#### Summary

Implement full public API (Strategy, Context, Backtest) and working CLI with comprehensive debugging support.

**Key Components:**

- Strategy base class and protocol
- Context with order submission methods
- **Context debug API** (debug_state, debug_indicators, debug_orders)
- Indicator framework (SMA, EMA, etc.)
- Backtest runner with step-by-step mode
- CLI command implementation
- Config file loading
- **Interactive debugging support** (breakpoint-friendly execution)
- **Debug logging** (structured logging with levels)
- **Debug output files** (bars.csv, indicators.csv, portfolio_snapshots.csv)

**Tests Focus:**

- Strategy lifecycle (on_start, on_bar, on_fill, on_end)
- Context order submission
- Context debug methods return correct state
- Indicators calculate correctly
- CLI commands work end-to-end
- Backtest.next_bar() allows manual stepping
- Debug output files generated correctly

**Debugging Features (per spec §20):**

1. **Standard Python Debugging:**

   - Strategies work seamlessly with pdb, ipdb, VS Code, PyCharm debuggers
   - No special "debug mode" required
   - All state visible at breakpoints

2. **Context Debug API:**

   ```python
   ctx.debug_state()           # Complete snapshot
   ctx.debug_indicators()      # All indicator values
   ctx.debug_orders()          # Filtered order list
   ctx.debug_fills()           # Recent fills
   ctx.debug_bar_history()     # Historical bars
   ```

3. **Interactive Backtesting:**

   ```python
   bt = Backtest(strategy, config)
   bt.setup()
   while bar := bt.next_bar():
       # Inspect bt.ctx at each step
       print(f"{bar.ts}: {ctx.get_cash()}")
   ```

4. **Debug Logging:**

   ```bash
   qtrader backtest --strategy s.py --log-level DEBUG --log-output both
   ```

5. **Conditional Breakpoints:**

   ```python
   if bar.symbol == "AAPL" and bar.ts.date() == date(2023, 1, 15):
       breakpoint()
   ```

6. **Date Range Filtering:**

   ```bash
   qtrader backtest --strategy s.py --start-date 2023-01-10 --end-date 2023-01-20 --symbols AAPL
   ```

7. **Debug Output Files:**

   ```bash
   qtrader backtest --strategy s.py --debug-output
   # Creates: bars.csv, indicators.csv, portfolio_snapshots.csv, execution_log.jsonl
   ```

**Implementation Notes:**

- `Context` must maintain complete state visibility for debug methods
- `Backtest` must support both `run()` (all at once) and `next_bar()` (manual stepping)
- Logger must be accessible from strategy (`self.logger`)
- All debug methods must be read-only (no side effects)
- Debug output files optional (performance overhead)

______________________________________________________________________

### **Stage 8: Golden Baselines & Validation**

**Timeline:** Days 26-30 **Branch:** `stage-8-goldens`

#### Summary

Create golden baseline strategies, generate reference results, and automate validation. **Use debugging tools to validate strategy behavior before committing goldens.**

**Key Components:**

- Buy-and-Hold strategy (AAPL, MSFT, AMZN)
- SMA Cross strategy (MSFT)
- Golden file generator scripts
- Fixture hash calculation
- Golden validation tests
- **Debug validation workflow** (step through strategies before golden commit)

**Process:**

1. Implement reference strategies
2. **Debug strategies bar-by-bar** using `Backtest.next_bar()` and breakpoints
3. **Verify indicator calculations** with `ctx.debug_indicators()`
4. **Check fills** with `ctx.debug_fills()` and `ctx.debug_orders()`
5. Run with `--debug-output` to generate comprehensive debug files
6. Review results together (NAV curves, fills.csv, debug outputs)
7. Commit golden files once validated
8. Create automated validation tests
9. Add CI checks for determinism

**Debug-Assisted Golden Generation:**

```python
# scripts/goldens/generate_buy_hold.py
from qtrader import Backtest
from strategies.buy_hold import BuyHold

bt = Backtest(
    strategy=BuyHold(symbols=["AAPL"]),
    data_config="configs/algoseek_daily.yaml",
    output_dir="goldens/buy_hold_aapl/"
)

# Manual stepping for first 10 bars to verify
bt.setup()
for i in range(10):
    bar = bt.next_bar()
    print(f"Bar {i}: {bar.symbol} @ {bar.ts} close={bar.close}")
    state = bt.ctx.debug_state()
    print(f"  Cash: {state['portfolio']['cash']}")
    print(f"  Positions: {state['portfolio']['positions']}")

    # Breakpoint here to inspect if needed
    if i == 0:  # Check initial purchase
        assert len(state['orders']['filled']) == 1

# Run remaining bars
bt.run()
bt.finalize()

# Validate outputs exist
assert Path("goldens/buy_hold_aapl/performance.json").exists()
assert Path("goldens/buy_hold_aapl/fills.csv").exists()
```

**Validation Workflow:**

1. **First run:** Use `--debug-output` and `--log-level DEBUG`
2. **Inspect:** Review all debug files (bars.csv, indicators.csv, portfolio_snapshots.csv)
3. **Spot-check:** Use `Backtest.next_bar()` to step through suspicious dates
4. **Breakpoints:** Add conditional breakpoints for specific dates/symbols
5. **Verify math:** Check indicator calculations match expected formulas
6. **Confirm fills:** Verify limit/stop touches using bar high/low
7. **Finalize:** Once satisfied, commit golden files to version control

______________________________________________________________________

## 📦 Project Structure (Complete)

```txt
qtrader/                        # Main package (was src/)
├── __init__.py                # Public API exports
├── cli.py                     # CLI entrypoint
├── api/                       # Public API
│   ├── __init__.py
│   ├── strategy.py           # Strategy base class
│   ├── context.py            # Context for strategies
│   ├── backtest.py           # Backtest runner
│   └── indicators.py         # Indicator framework
├── models/                    # Data models
│   ├── __init__.py
│   ├── bar.py               # Bar (OHLCV), AdjustmentEvent, enums
│   ├── order.py             # Order types
│   └── fill.py              # Fill model
├── config/                    # Configuration
│   ├── __init__.py
│   ├── data_config.py        # Data configuration (BarSchemaConfig, AdjustmentSchemaConfig)
│   └── engine_config.py      # Engine configuration
├── adapters/                  # Data adapters (private)
│   ├── __init__.py
│   ├── base.py              # Protocol (DataAdapter with read_bars, read_adjustments)
│   ├── algoseek_parquet.py
│   └── csv_adapter.py
├── validation/                # Validation (private)
│   ├── __init__.py
│   └── bar_validator.py
├── engine/                    # Execution engine (private)
│   ├── __init__.py
│   ├── execution_engine.py   # Main engine
│   ├── order_manager.py     # Order management
│   ├── fill_policy.py       # Fill rules
│   └── participation.py     # Volume caps
├── ledger/                    # Accounting (private)
│   ├── __init__.py
│   ├── positions.py         # Position tracking
│   ├── cash.py              # Cash ledger
│   └── accruals.py          # Borrow, dividends
└── outputs/                   # Output generation (private)
    ├── __init__.py
    ├── writers.py           # File writers
    └── schemas.py           # Output schemas

tests/
├── stage1/                    # Data & adapters
├── stage2/                    # Orders & ledger
├── stage3/                    # Market & MOC
├── stage4/                    # Limit & Stop
├── stage5/                    # Participation
├── stage6/                    # Accruals & outputs
├── stage7/                    # API & CLI
├── stage8/                    # Golden baselines
└── goldens/
    ├── fixtures/             # Golden reference results
    │   ├── buy_and_hold_aapl_v1.0.json
    │   ├── buy_and_hold_msft_v1.0.json
    │   ├── buy_and_hold_amzn_v1.0.json
    │   └── sma_cross_msft_v1.0.json
    └── scripts/              # Golden generators
        ├── buy_and_hold.py
        └── sma_cross.py

scripts/
└── goldens/                   # Standalone golden generators
```

______________________________________________________________________

## 🔧 Development Workflow

### Per Stage

1. **Create feature branch:** `git checkout -b stage-N-name`
2. **Implement deliverables** following TDD (test → code → refactor)
3. **Run tests:** `make test` (or `pytest tests/stageN/`)
4. **Run QA:** `make qa` (format, lint, type-check, test)
5. **Commit with pre-commit:** Hooks auto-format and validate
6. **PR review:** Ensure all stage tests pass
7. **Merge to master:** Stage complete ✅

### Testing Commands

```bash
# Run all tests
make test

# Run specific stage
pytest tests/stage1/ -v

# Run with coverage
pytest --cov=qtrader --cov-report=html

# Run fast (no coverage)
make test-fast
```

### Code Quality

```bash
# Format code
make format

# Lint only
make lint

# Type check
make type-check

# All quality checks
make qa
```

______________________________________________________________________

## 📝 Dependencies Summary

```toml
[project]
name = "qtrader"
version = "0.1.0"
requires-python = ">=3.13"

dependencies = [
    "duckdb>=1.4.0",        # Parquet reading
    "pandas>=2.3.2",        # Data manipulation
    "pyarrow>=21.0.0",      # Parquet support
    "click>=8.0.0",         # CLI framework
    "pydantic>=2.11.9",     # Configuration validation
    "pyyaml>=6.0",          # YAML config loading
    "pytz>=2024.1",         # Timezone handling
    "structlog>=24.4.0",    # Structured logging
]

[project.scripts]
qtrader = "qtrader.cli:main"
```

______________________________________________________________________

## ✅ Success Metrics

### Per Stage

- All tests pass (`make test`)
- Code quality passes (`make qa`)
- Pre-commit hooks pass
- Documentation complete (docstrings)

### Overall Phase 1

- Load 3,774 bars from fixture ✅
- Execute all order types ✅
- Handle volume participation ✅
- Track positions/cash with Decimal ✅
- Generate golden baselines ✅
- CLI works end-to-end ✅
- Package installable via pip ✅
- All 8 stages merged to master ✅

______________________________________________________________________

## 🎯 Ready to Begin

**Stage 1 is approved and ready for implementation.**

Key Points:

01. **Vendor-Agnostic Architecture:** Bar = Universal OHLCV contract, works with ANY data source
02. **Separation of Concerns:** Adjustment metadata stored separately from Bar (AdjustmentEvent)
03. **DataMode Declaration:** Adapters declare if prices are adjusted/unadjusted/split_adjusted
04. **Schema Mapping:** Config-driven bar_schema and adjustment_schema for flexibility
05. **Package Structure:** `qtrader` as installable package (not just `src/`)
06. **Public API:** Strategy, Context, Backtest (stubs in Stage 1, full in Stage 7)
07. **CLI:** `qtrader backtest` command (stub in Stage 1, full in Stage 7)
08. **Data Foundation:** Bar model, adapters, validation (complete in Stage 1)
09. **Logging:** structlog with INFO default, DEBUG when needed
10. **Testing:** TDD approach, comprehensive tests per stage

**Next Step:** Begin Stage 1 implementation with vendor-agnostic data foundation! 🚀
