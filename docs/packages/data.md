# Data Package Documentation

**Package**: `qtrader.services.data`\
**Purpose**: Vendor-agnostic data adapter system for market data ingestion\
**Status**: Production Ready

______________________________________________________________________

## Overview

The data package provides a flexible, protocol-based architecture for loading and streaming market data from multiple vendors (Algoseek, IEX, Bloomberg, etc.) through a unified interface.

**Key Features**:

- **Protocol-based adapters**: Vendor-specific implementations behind common interface
- **Dataset-centric configuration**: YAML-defined data sources as single source of truth
- **Optional caching**: Disk-based, database-backed, or streaming (no cache)
- **Incremental updates**: Efficient cache updates without full re-downloads
- **Event-driven streaming**: Direct PriceBarEvent and CorporateActionEvent emission
- **Force-reprime support**: Automatic cache cleanup for legacy adapters
- **CLI integration**: Browse, update, and inspect data via command line

______________________________________________________________________

## Architecture Philosophy

### Protocol-Based Adapters

```python
# All adapters implement IDataAdapter protocol
class IDataAdapter(Protocol):
    def read_bars(self, start_date: str, end_date: str) -> Iterator[Any]: ...
    def to_price_bar_event(self, bar: Any) -> PriceBarEvent: ...
    def prime_cache(self, start_date: str, end_date: str) -> int: ...  # OPTIONAL
    def write_cache(self, bars: list[Any]) -> None: ...                # OPTIONAL
    def update_to_latest(self, dry_run: bool) -> tuple[int, date, date]: ...  # For incremental
```

**Benefits**:

- New vendors require only adapter implementation
- Core logic vendor-agnostic
- Easy to test with mock adapters

### Dataset-Centric Configuration

```yaml
# config/data_sources.yaml
algoseek-us-equity-1d-unadjusted:
  provider: algoseek
  adapter: parquet
  asset_class: equity
  frequency: 1d
  adjustment_mode: unadjusted
  cache_enabled: true
  params:
    base_path: data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample
```

**User Specifies**: `"Give me AAPL from algoseek-us-equity-1d-unadjusted"`\
**Not**: `"Give me AAPL (equity, from Algoseek, parquet format, 1d, unadjusted)"`

Configuration is the single source of truth - no need to pass metadata repeatedly.

### Caching Strategy

**Two-Tier Caching**:

1. **Prime Cache** (efficient): Bulk load with vendor-optimized methods
1. **Write Cache** (fallback): Serialize incremental bars to disk

**Three Adapter Types**:

- **Disk-cached** (Algoseek parquet): Fast prime + update_to_latest
- **Database-backed** (future): SQL query + incremental inserts
- **Streaming** (future IEX WebSocket): No cache, real-time only

______________________________________________________________________

## Package Structure

```
src/qtrader/services/data/
├── __init__.py                  # Public API exports
├── service.py                   # DataService - main interface
├── config.py                    # Configuration models
├── dataset_updater.py           # Incremental update logic
├── interface.py                 # Protocol definitions
├── models.py                    # Instrument and data models
├── source_selector.py           # Source selection logic
├── source_validator.py          # Dataset configuration validator
├── update_service.py            # Update orchestration
└── adapters/
    ├── __init__.py             # Adapter exports
    ├── protocol.py             # IDataAdapter protocol
    ├── resolver.py             # Dataset → Adapter resolution
    └── algoseek.py             # Algoseek parquet adapter
```

______________________________________________________________________

## Module: service.py

Main interface for data loading and streaming.

### Classes

#### DataService

Event-driven data service for market data streaming.

**Purpose**: Load and stream historical bars via adapters and EventBus.

```python
from qtrader.services.data.service import DataService
from qtrader.services.data.config import DataConfig
from qtrader.events.event_bus import EventBus

# Initialize with dataset name
bus = EventBus()
service = DataService(
    config=data_config,
    dataset="algoseek-us-equity-1d-unadjusted",
    event_bus=bus
)

# Stream bars (publishes PriceBarEvent to EventBus)
service.stream_bars(
    symbol="AAPL",
    start_date=date(2020, 1, 1),
    end_date=date(2020, 12, 31)
)
```

**Constructor Parameters**:

- `config` (DataConfig): Data configuration
- `dataset` (str): Dataset name from `data_sources.yaml`
- `resolver` (DataSourceResolver, optional): Custom resolver
- `event_bus` (IEventBus, optional): EventBus for publishing events

**Key Methods**:

- `stream_bars(symbol, start_date, end_date, is_warmup=False)`: Stream single symbol
- `stream_universe(symbols, start_date, end_date)`: Stream multiple symbols with timestamp synchronization
- `get_instrument(symbol)`: Get instrument metadata
- `get_corporate_actions(symbol, start_date, end_date)`: Get corporate actions

**Requirements**:

- Dataset parameter is **REQUIRED**
- EventBus required for `stream_bars()` (raises ValueError if None)
- Resolver auto-created if not provided

______________________________________________________________________

## Module: config.py

Configuration models for data service.

### Classes

#### BarSchemaConfig

Maps vendor-specific column names to canonical bar fields.

```python
from qtrader.services.data.config import BarSchemaConfig

# Algoseek parquet schema
schema = BarSchemaConfig(
    ts="trade_datetime",
    symbol="symbol",
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
)
```

**Fields**:

- `ts` (str): Timestamp column name
- `symbol` (str): Symbol column name
- `open` (str): Open price column
- `high` (str): High price column
- `low` (str): Low price column
- `close` (str): Close price column
- `volume` (str): Volume column

**Purpose**: Adapters use this to extract canonical fields from vendor data.

______________________________________________________________________

#### DataConfig

Data service configuration.

```python
from qtrader.services.data.config import DataConfig
from qtrader.services.data.source_selector import DataSourceSelector, AssetClass

selector = DataSourceSelector(
    provider="algoseek",
    asset_class=AssetClass.EQUITY
)

config = DataConfig(
    mode="adjusted",
    frequency="1d",
    timezone="America/New_York",
    source_selector=selector,
    bar_schema=bar_schema
)
```

**Fields**:

- `mode` (str): "adjusted" or "unadjusted"
- `frequency` (str): "1d", "1h", "1m", etc.
- `timezone` (str): IANA timezone (e.g., "America/New_York")
- `source_selector` (DataSourceSelector): Source selection criteria
- `bar_schema` (BarSchemaConfig): Column mapping

**Note**: With dataset-centric approach, most metadata comes from YAML config, not DataConfig.

______________________________________________________________________

## Module: models.py

Data models for instruments and bars.

### Classes

#### Instrument

Tradable instrument specification.

```python
from qtrader.services.data.models import Instrument

# Basic instrument
instrument = Instrument(symbol="AAPL")

# With frequency override
instrument = Instrument(symbol="BTCUSD", frequency="1m")

# With metadata
instrument = Instrument(
    symbol="ES_Z24",
    metadata={"contract_month": "2024-12", "exchange": "CME"}
)
```

**Fields**:

- `symbol` (str): Ticker symbol (e.g., "AAPL", "BTCUSD", "ES_Z24")
- `frequency` (str, optional): Override dataset default frequency
- `metadata` (dict, optional): Custom attributes

**Design Philosophy**:

- Minimal representation (symbol-centric)
- Dataset specified separately when resolving to adapter
- Metadata for custom attributes without schema pollution

______________________________________________________________________

## Module: adapters/protocol.py

Protocol definition for data adapters.

### Protocols

#### IDataAdapter

Protocol all data adapters must implement.

```python
from typing import Protocol, Iterator, Any, Optional
from datetime import date

class IDataAdapter(Protocol):
    """Protocol for data adapters."""

    def read_bars(self, start_date: str, end_date: str) -> Iterator[Any]:
        """
        Stream bars for date range.

        Returns:
            Iterator of vendor-specific bar objects
        """
        ...

    def to_price_bar_event(self, bar: Any) -> PriceBarEvent:
        """
        Convert vendor bar to canonical PriceBarEvent.

        Args:
            bar: Vendor-specific bar object

        Returns:
            PriceBarEvent with canonical fields
        """
        ...

    def to_corporate_action_event(
        self, bar: Any, prev_bar: Optional[Any] = None
    ) -> Optional[CorporateActionEvent]:
        """
        Detect corporate actions (splits, dividends) from bar.

        Args:
            bar: Current bar
            prev_bar: Previous bar (for change detection)

        Returns:
            CorporateActionEvent if action detected, else None
        """
        ...

    def get_timestamp(self, bar: Any) -> datetime:
        """Extract timestamp from vendor bar."""
        ...

    def get_available_date_range(self) -> tuple[Optional[str], Optional[str]]:
        """
        Get available date range in cache/source.

        Returns:
            (earliest_date, latest_date) or (None, None) if unavailable
        """
        ...

    # OPTIONAL METHODS (for caching)

    def prime_cache(self, start_date: str, end_date: str) -> int:
        """
        Prime cache with bulk load (vendor-optimized).

        OPTIONAL: Only implement if adapter supports caching.

        Returns:
            Number of bars cached
        """
        raise NotImplementedError("Adapter does not support caching")

    def write_cache(self, bars: list[Any]) -> None:
        """
        Write bars to cache (fallback method).

        OPTIONAL: Only implement if adapter supports caching.
        """
        raise NotImplementedError("Adapter does not support caching")

    def update_to_latest(self, dry_run: bool = False) -> tuple[int, date, date]:
        """
        Update cache to latest available data.

        REQUIRED for incremental updates. If adapter doesn't support
        incremental updates, raise NotImplementedError and use --force-reprime.

        Args:
            dry_run: If True, report what would be updated without changes

        Returns:
            (bars_added, start_date, end_date)
        """
        raise NotImplementedError(
            "Adapter does not support incremental updates. Use --force-reprime."
        )
```

**Key Points**:

- `read_bars()`, `to_price_bar_event()`, and `get_timestamp()` are **REQUIRED**
- `prime_cache()` and `write_cache()` are **OPTIONAL** (streaming adapters don't need them)
- `update_to_latest()` is **REQUIRED** for incremental updates, but can raise NotImplementedError

**Caching Tiers**:

1. **Prime Cache** (efficient): Bulk load optimized for vendor format
1. **Write Cache** (fallback): Serialize bars to generic format
1. **No Cache**: Streaming adapters (WebSocket, live feeds)

______________________________________________________________________

## Module: adapters/algoseek.py

Algoseek parquet adapter implementation.

### Classes

#### AlgoseekParquetAdapter

Adapter for Algoseek parquet files with DuckDB queries.

```python
from qtrader.services.data.adapters.algoseek import AlgoseekParquetAdapter

adapter = AlgoseekParquetAdapter(
    symbol="AAPL",
    dataset="algoseek-us-equity-1d-unadjusted",
    config={
        "base_path": "data/algoseek-parquet",
        "cache_enabled": True,
    }
)

# Read bars
bars = adapter.read_bars("2020-01-01", "2020-12-31")
for bar in bars:
    event = adapter.to_price_bar_event(bar)
    print(event.symbol, event.close)
```

**Features**:

- DuckDB for fast parquet queries
- Disk-based caching in `.qtrader_cache/`
- Incremental updates via `update_to_latest()`
- Corporate action detection (splits, dividends)

**Cache Structure**:

```
.qtrader_cache/
└── algoseek-us-equity-1d-unadjusted/
    ├── AAPL.parquet
    ├── MSFT.parquet
    └── GOOGL.parquet
```

**Update Strategy**:

1. Find latest cached date
1. Query source for new bars after that date
1. Append new bars to cache
1. Return count and date range

______________________________________________________________________

## Module: adapters/resolver.py

Dataset name resolution to adapter instances.

### Classes

#### DataSourceResolver

Resolve dataset names to adapter instances.

```python
from qtrader.services.data.adapters.resolver import DataSourceResolver
from qtrader.services.data.models import Instrument

resolver = DataSourceResolver()  # Auto-loads data_sources.yaml

# Resolve by dataset name (recommended)
instrument = Instrument(symbol="AAPL")
adapter = resolver.resolve_by_dataset(
    "algoseek-us-equity-1d-unadjusted",
    instrument
)

# Get all configured datasets
datasets = resolver.list_datasets()
for name, config in datasets.items():
    print(f"{name}: {config['provider']} - {config['asset_class']}")
```

**Key Methods**:

- `resolve_by_dataset(dataset, instrument)`: Resolve to adapter instance
- `list_datasets()`: Get all configured datasets
- `get_source_config(dataset)`: Get raw YAML config for dataset

**Adapter Registry**:

```python
# In resolver.py
ADAPTER_REGISTRY: ClassVar[dict[str, type[IDataAdapter]]] = {
    "parquet": AlgoseekParquetAdapter,
    # TODO: Add more as they're implemented
    # "rest": IEXRestAdapter,
    # "websocket": IEXWebSocketAdapter,
    # "sql": PostgresAdapter,
}
```

**Adding New Adapters**:

1. Implement `IDataAdapter` protocol
1. Add to `ADAPTER_REGISTRY`
1. Add dataset config to `data_sources.yaml`

______________________________________________________________________

## Module: dataset_updater.py

Incremental update logic for cached datasets.

### Classes

#### DatasetUpdater

Updates cached data to latest available.

```python
from qtrader.services.data.dataset_updater import DatasetUpdater

updater = DatasetUpdater(
    dataset="algoseek-us-equity-1d-unadjusted",
    symbol="AAPL"
)

# Update to latest
bars_added, start_date, end_date = updater.update_to_latest()
print(f"Added {bars_added} bars from {start_date} to {end_date}")

# Force re-prime (delete cache and re-download)
bars_added, start_date, end_date = updater.update_to_latest(force_reprime=True)
```

**Update Flow**:

1. Check if adapter supports incremental updates
1. Try `adapter.update_to_latest()` (efficient)
1. If not implemented, use fallback:
   - Read bars from source for incremental range
   - Try `adapter.prime_cache()` (bulk load)
   - Fallback to `adapter.write_cache()` (serialize)
1. If both fail and `force_reprime=True`, delete cache and re-prime from scratch

**Force Re-Prime**:

```python
# Automatic cache cleanup for legacy adapters
updater.update_to_latest(force_reprime=True)
```

**Use Cases**:

- Adapter without incremental support
- Corrupted cache files
- Schema version changes
- Testing from scratch

______________________________________________________________________

## Module: update_service.py

High-level update orchestration.

### Classes

#### UpdateService

Orchestrates multi-symbol updates.

```python
from qtrader.services.data.update_service import UpdateService

service = UpdateService("algoseek-us-equity-1d-unadjusted")

# Get symbols to update (from universe.csv or explicit list)
symbols, source = service.get_symbols_to_update(["AAPL", "MSFT"])

# Update with progress tracking
for result in service.update_symbols(symbols, force_reprime=False):
    print(f"{result.symbol}: {result.bars_added} bars, {result.success}")
```

**Symbol Priority**:

1. Explicit `--symbols` argument (highest priority)
1. `universe.csv` in cache directory
1. All cached symbols (lowest priority)

**Result Model**:

```python
@dataclass
class UpdateResult:
    symbol: str
    success: bool
    bars_added: int
    start_date: Optional[date]
    end_date: Optional[date]
    error: Optional[str]
```

______________________________________________________________________

## CLI Commands

Located in: `src/qtrader/cli/commands/data.py`

### Command: `qtrader data list`

List configured data sources.

```bash
# List all sources
qtrader data list

# Verbose output (show caching status)
qtrader data list --verbose
```

**Output**:

```
Available Data Sources
┌────────────────────────┬──────────┬─────────┬─────────────┬────────┐
│ Dataset                │ Provider │ Adapter │ Asset Class │ Cache  │
├────────────────────────┼──────────┼─────────┼─────────────┼────────┤
│ algoseek-us-equity-1d  │ algoseek │ parquet │ equity      │   ✓    │
└────────────────────────┴──────────┴─────────┴─────────────┴────────┘
```

______________________________________________________________________

### Command: `qtrader data raw`

Browse raw unadjusted data interactively.

```bash
# View AAPL raw data for January 2020
qtrader data raw \
  --dataset algoseek-us-equity-1d-unadjusted \
  --symbol AAPL \
  --start-date 2020-01-01 \
  --end-date 2020-01-31
```

**Features**:

- Interactive bar-by-bar browsing
- Press ENTER for next bar
- CTRL+C to exit
- Shows OHLCV + timestamp

**Use Cases**:

- Data quality inspection
- Verify cache contents
- Debug adapter implementations

______________________________________________________________________

### Command: `qtrader data update`

Update cached data to latest available.

```bash
# Update all symbols in universe.csv
qtrader data update --dataset algoseek-us-equity-1d-unadjusted

# Update specific symbols
qtrader data update \
  --dataset algoseek-us-equity-1d-unadjusted \
  --symbols AAPL,MSFT,GOOGL

# Dry run (show what would be updated)
qtrader data update \
  --dataset algoseek-us-equity-1d-unadjusted \
  --dry-run

# Force re-prime (delete cache and re-download)
qtrader data update \
  --dataset algoseek-us-equity-1d-unadjusted \
  --force-reprime

# Verbose output
qtrader data update \
  --dataset algoseek-us-equity-1d-unadjusted \
  --verbose
```

**Update Results**:

```
UPDATING Dataset: algoseek-us-equity-1d-unadjusted

✓ AAPL   [████████████████████] 100%
✓ MSFT   [████████████████████] 100%
✓ GOOGL  [████████████████████] 100%

Summary
┌────────┬─────────┬──────────────┬──────────────┬──────────┐
│ Symbol │ Success │ Bars Added   │ Date Range   │ Duration │
├────────┼─────────┼──────────────┼──────────────┼──────────┤
│ AAPL   │   ✓     │ 252          │ 2024-01-01 … │ 1.2s     │
│ MSFT   │   ✓     │ 252          │ 2024-01-01 … │ 0.9s     │
│ GOOGL  │   ✓     │ 252          │ 2024-01-01 … │ 1.1s     │
└────────┴─────────┴──────────────┴──────────────┴──────────┘
```

______________________________________________________________________

## Configuration Files

### Data Sources Configuration

**File**: `config/data_sources.yaml`

```yaml
# Algoseek parquet - unadjusted
algoseek-us-equity-1d-unadjusted:
  provider: algoseek
  adapter: parquet
  asset_class: equity
  frequency: 1d
  adjustment_mode: unadjusted
  cache_enabled: true
  params:
    base_path: data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample

# Algoseek parquet - adjusted
algoseek-us-equity-1d-adjusted:
  provider: algoseek
  adapter: parquet
  asset_class: equity
  frequency: 1d
  adjustment_mode: adjusted
  cache_enabled: true
  params:
    base_path: data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample

# Future: IEX REST API (no cache)
# iex-us-equity-1d:
#   provider: iex
#   adapter: rest
#   asset_class: equity
#   frequency: 1d
#   cache_enabled: false
#   params:
#     api_key: ${IEX_API_KEY}
```

**Field Descriptions**:

- `provider`: Vendor name (algoseek, iex, bloomberg, etc.)
- `adapter`: Adapter type (parquet, rest, websocket, sql)
- `asset_class`: equity, crypto, futures, forex, options
- `frequency`: 1d (daily), 1h (hourly), 1m (minute), etc.
- `adjustment_mode`: unadjusted, adjusted
- `cache_enabled`: true/false
- `params`: Adapter-specific configuration

______________________________________________________________________

### Universe Configuration

**File**: `.qtrader_cache/{dataset}/universe.csv`

```csv
symbol
AAPL
MSFT
GOOGL
TSLA
AMZN
```

**Purpose**: Default symbol list for `qtrader data update`

**Priority**:

1. `--symbols` CLI argument (highest)
1. `universe.csv` in cache
1. All cached symbols (lowest)

______________________________________________________________________

## Usage Examples

### Basic Data Loading

```python
from qtrader.services.data.service import DataService
from qtrader.services.data.config import DataConfig
from qtrader.events.event_bus import EventBus
from datetime import date

# Setup
bus = EventBus()
config = DataConfig(...)  # Simplified for brevity

service = DataService(
    config=config,
    dataset="algoseek-us-equity-1d-unadjusted",
    event_bus=bus
)

# Stream bars (publishes to EventBus)
service.stream_bars(
    symbol="AAPL",
    start_date=date(2020, 1, 1),
    end_date=date(2020, 12, 31)
)
```

______________________________________________________________________

### Direct Adapter Usage

```python
from qtrader.services.data.adapters.resolver import DataSourceResolver
from qtrader.services.data.models import Instrument

# Resolve adapter
resolver = DataSourceResolver()
instrument = Instrument(symbol="AAPL")
adapter = resolver.resolve_by_dataset(
    "algoseek-us-equity-1d-unadjusted",
    instrument
)

# Read bars
bars = adapter.read_bars("2020-01-01", "2020-12-31")
for bar in bars:
    event = adapter.to_price_bar_event(bar)
    print(f"{event.timestamp}: {event.close}")
```

______________________________________________________________________

### Incremental Updates

```python
from qtrader.services.data.dataset_updater import DatasetUpdater

updater = DatasetUpdater(
    dataset="algoseek-us-equity-1d-unadjusted",
    symbol="AAPL"
)

# Update to latest
bars_added, start, end = updater.update_to_latest()
print(f"Added {bars_added} bars: {start} to {end}")
```

______________________________________________________________________

### Multi-Symbol Streaming

```python
from qtrader.services.data.service import DataService

service = DataService(
    config=config,
    dataset="algoseek-us-equity-1d-unadjusted",
    event_bus=bus
)

# Stream multiple symbols with timestamp synchronization
service.stream_universe(
    symbols=["AAPL", "MSFT", "GOOGL"],
    start_date=date(2020, 1, 1),
    end_date=date(2020, 12, 31)
)
```

**Timestamp Synchronization**:

```
For each timestamp T across all symbols:
    1. Publish PriceBarEvent(symbol="AAPL", timestamp=T)
    2. Publish PriceBarEvent(symbol="MSFT", timestamp=T)
    3. Publish PriceBarEvent(symbol="GOOGL", timestamp=T)
    4. Advance to T+1 (next timestamp)
```

______________________________________________________________________

## Adding New Adapters

### Step 1: Implement Protocol

```python
from qtrader.services.data.adapters.protocol import IDataAdapter
from qtrader.events.events import PriceBarEvent

class MyVendorAdapter:
    """Adapter for MyVendor data source."""

    def __init__(self, symbol: str, dataset: str, config: dict):
        self.symbol = symbol
        self.dataset = dataset
        self.config = config
        self.api_key = config.get("api_key")

    def read_bars(self, start_date: str, end_date: str) -> Iterator[Any]:
        """Fetch bars from MyVendor API."""
        # Implementation here
        pass

    def to_price_bar_event(self, bar: Any) -> PriceBarEvent:
        """Convert MyVendor bar to PriceBarEvent."""
        return PriceBarEvent(
            symbol=bar.symbol,
            timestamp=bar.ts.isoformat(),
            open=Decimal(str(bar.open)),
            high=Decimal(str(bar.high)),
            low=Decimal(str(bar.low)),
            close=Decimal(str(bar.close)),
            volume=bar.volume,
            # ... other fields
        )

    def get_timestamp(self, bar: Any) -> datetime:
        return bar.ts

    # Optional: caching methods
    def update_to_latest(self, dry_run: bool = False):
        raise NotImplementedError("Streaming adapter, no cache")
```

### Step 2: Register Adapter

```python
# In adapters/resolver.py
ADAPTER_REGISTRY: ClassVar[dict[str, type[IDataAdapter]]] = {
    "parquet": AlgoseekParquetAdapter,
    "myvendor_rest": MyVendorAdapter,  # Add here
}
```

### Step 3: Add Configuration

```yaml
# In config/data_sources.yaml
myvendor-us-equity-1d:
  provider: myvendor
  adapter: myvendor_rest
  asset_class: equity
  frequency: 1d
  cache_enabled: false
  params:
    api_key: ${MYVENDOR_API_KEY}
```

### Step 4: Use It

```bash
qtrader data update --dataset myvendor-us-equity-1d --symbols AAPL,MSFT
```

______________________________________________________________________

## Design Patterns

### Protocol Pattern

Adapters implement `IDataAdapter` protocol, not inherit from base class:

- Structural subtyping (duck typing with type hints)
- No inheritance hierarchy
- Easy to mock for testing

### Factory Pattern

`DataSourceResolver` implements factory pattern:

- Centralizes adapter instantiation
- Maps dataset names to adapter classes
- Injects configuration from YAML

### Strategy Pattern

Different caching strategies:

- **Prime + Update**: Disk-based cache with incremental updates
- **Write Cache**: Fallback serialization
- **No Cache**: Streaming adapters

______________________________________________________________________

## Error Handling

### Adapter Not Found

```python
from qtrader.services.data.adapters.resolver import DataSourceResolver

resolver = DataSourceResolver()
try:
    adapter = resolver.resolve_by_dataset("nonexistent", instrument)
except KeyError as e:
    print(f"Dataset not configured: {e}")
```

### Update Failures

```python
from qtrader.services.data.dataset_updater import DatasetUpdater

updater = DatasetUpdater("algoseek-us-equity-1d-unadjusted", "AAPL")

try:
    bars, start, end = updater.update_to_latest()
except FileNotFoundError as e:
    print(f"Data source not found: {e}")
except NotImplementedError as e:
    print(f"Adapter doesn't support incremental: {e}")
    # Use force_reprime instead
    bars, start, end = updater.update_to_latest(force_reprime=True)
```

### Cache Corruption

```bash
# Delete and re-prime from scratch
qtrader data update --dataset algoseek-us-equity-1d-unadjusted --force-reprime
```

______________________________________________________________________

## Performance Considerations

### DuckDB Query Optimization

Algoseek adapter uses DuckDB for fast parquet queries:

- **Predicate pushdown**: Date filters applied at parquet level
- **Column pruning**: Only needed columns loaded
- **Zero-copy**: Memory-mapped files

**Benchmark** (100 symbols × 252 days):

- Cold load: ~500ms
- Warm load: ~150ms (OS cache)

### Cache File Size

**Estimation**: `~200 bytes/bar` (parquet compressed)

- 1 symbol × 252 days = ~50 KB
- 100 symbols × 252 days = ~5 MB
- 1000 symbols × 2520 days = ~500 MB

### Memory Usage

**Current Implementation**: Buffers all bars before publishing

**Estimated**: `~500 bytes/bar` (PriceBarEvent in memory)

- 100 symbols × 252 days = ~12.6 MB (manageable)
- 1000 symbols × 2520 days = ~1.26 GB (high)

**TODO**: Implement heap-merge streaming for incremental publishing.

______________________________________________________________________

## Testing

### Test Coverage

**Unit Tests**: `tests/unit/services/data/`

- Adapter protocol validation
- Configuration loading
- Dataset resolution
- Update logic

**Integration Tests**: `tests/integration/data/`

- End-to-end data loading
- Cache priming and updates
- Multi-symbol streaming

### Running Tests

```bash
# All data tests
pytest tests/unit/services/data/ -v

# Specific adapter tests
pytest tests/unit/services/data/test_algoseek.py -v

# Integration tests
pytest tests/integration/data/ -v
```

______________________________________________________________________

## Troubleshooting

### Problem: Dataset not found

**Error**: `KeyError: Dataset 'xxx' not configured`

**Solution**: Check `config/data_sources.yaml` for dataset name.

```bash
qtrader data list  # See available datasets
```

______________________________________________________________________

### Problem: Adapter doesn't support incremental updates

**Error**: `NotImplementedError: Adapter does not support incremental updates`

**Solution**: Use `--force-reprime` flag.

```bash
qtrader data update --dataset myvendor-us-equity-1d --force-reprime
```

______________________________________________________________________

### Problem: Cache directory permissions

**Error**: `PermissionError: [Errno 13] Permission denied: '.qtrader_cache'`

**Solution**: Fix directory permissions.

```bash
chmod 755 .qtrader_cache
```

______________________________________________________________________

### Problem: Parquet file corruption

**Symptoms**: Read errors, schema mismatches

**Solution**: Delete cache and re-prime.

```bash
rm -rf .qtrader_cache/algoseek-us-equity-1d-unadjusted/AAPL.parquet
qtrader data update --dataset algoseek-us-equity-1d-unadjusted --symbols AAPL
```

______________________________________________________________________

## Best Practices

### 1. Use Dataset Names, Not Direct Adapter Creation

```python
# ✅ Good - Uses configuration
resolver = DataSourceResolver()
adapter = resolver.resolve_by_dataset("algoseek-us-equity-1d-unadjusted", instrument)

# ❌ Bad - Hardcodes details
adapter = AlgoseekParquetAdapter(
    symbol="AAPL",
    dataset="algoseek-us-equity-1d-unadjusted",
    config={"base_path": "..."}
)
```

### 2. Prefer Event-Driven Streaming

```python
# ✅ Good - Event-driven via EventBus
service = DataService(config, dataset, event_bus=bus)
service.stream_bars("AAPL", start, end)

# ❌ Bad - Direct adapter usage for production
adapter = resolver.resolve_by_dataset(dataset, instrument)
for bar in adapter.read_bars(start, end):
    # Manual event handling
```

### 3. Use --dry-run Before Updates

```bash
# Check what would be updated
qtrader data update --dataset algoseek-us-equity-1d-unadjusted --dry-run

# Then run actual update
qtrader data update --dataset algoseek-us-equity-1d-unadjusted
```

### 4. Maintain universe.csv

```bash
# Create universe file
cat > .qtrader_cache/algoseek-us-equity-1d-unadjusted/universe.csv << EOF
symbol
AAPL
MSFT
GOOGL
EOF

# Update all symbols in universe
qtrader data update --dataset algoseek-us-equity-1d-unadjusted
```

### 5. Handle NotImplementedError Gracefully

```python
try:
    bars, start, end = updater.update_to_latest()
except NotImplementedError:
    # Adapter doesn't support incremental updates
    if user_confirms_force_reprime():
        bars, start, end = updater.update_to_latest(force_reprime=True)
```

______________________________________________________________________

## Future Enhancements

### Phase 2: Additional Adapters

- [ ] IEX REST API adapter
- [ ] IEX WebSocket adapter (real-time)
- [ ] Bloomberg Terminal adapter
- [ ] Polygon.io adapter
- [ ] PostgreSQL/TimescaleDB adapter

### Phase 3: Advanced Features

- [ ] Heap-merge streaming (low memory)
- [ ] Multi-source priority (primary + fallback)
- [ ] Data quality checks (gap detection, outliers)
- [ ] Automatic symbol discovery
- [ ] Parallel downloads (multi-symbol)
- [ ] Compression options (zstd, lz4)
- [ ] S3/Cloud storage backends

### Phase 4: Live Trading

- [ ] Real-time WebSocket adapters
- [ ] Order book reconstruction
- [ ] Tick-level data support
- [ ] Market depth (L2/L3 data)

______________________________________________________________________

## Related Documentation

- **Engine Package**: `docs/packages/engine.md` - Backtesting engine
- **Events Package**: `docs/packages/events.md` - Event types and validation
- **Architecture**: `docs/ARCHITECTURE_ALIGNMENT.md` - System architecture
- **Data Update Guide**: `docs/DATA_UPDATE_GUIDE.md` - Update workflows

______________________________________________________________________

## API Reference Summary

### Public API (`qtrader.services.data`)

**Service**:

- `DataService` - Main data loading interface

**Configuration**:

- `DataConfig` - Service configuration
- `BarSchemaConfig` - Column mapping

**Models**:

- `Instrument` - Tradable instrument specification

**Adapters**:

- `IDataAdapter` - Protocol all adapters implement
- `DataSourceResolver` - Dataset → Adapter resolution
- `AlgoseekParquetAdapter` - Algoseek implementation

**Update Management**:

- `DatasetUpdater` - Incremental update logic
- `UpdateService` - Multi-symbol orchestration

**Example Import**:

```python
from qtrader.services.data import (
    DataService,
    DataConfig,
    Instrument,
    DataSourceResolver,
)
```

______________________________________________________________________

## Migration Guide

### From Old Architecture

**Before** (Hardcoded adapter instantiation):

```python
# Old way - Manual adapter creation
adapter = AlgoseekParquetAdapter(
    symbol="AAPL",
    dataset="algoseek-us-equity-1d-unadjusted",
    config={"base_path": "data/algoseek"}
)
```

**After** (Configuration-driven):

```python
# New way - Resolver + YAML config
resolver = DataSourceResolver()  # Auto-loads data_sources.yaml
instrument = Instrument(symbol="AAPL")
adapter = resolver.resolve_by_dataset(
    "algoseek-us-equity-1d-unadjusted",
    instrument
)
```

______________________________________________________________________

**Last Updated**: 2024-10-24\
**Version**: 1.0\
**Status**: Production Ready
