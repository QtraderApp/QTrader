# Stage 6C: Instrument Abstraction & Data Source Resolver

**Status:** Phase 1 Complete (Model + Resolver + Config) **Date:** October 6, 2025 **Objective:** Replace file path-based configuration with Instrument abstraction

______________________________________________________________________

## Overview

Stage 6C introduces a logical abstraction layer for instruments and data sources, decoupling strategy code from physical data storage. This enables:

- Mix multiple asset types (equity, crypto, signals) in one backtest
- Swap data sources without changing strategy code
- Environment-specific configuration (dev/prod/test)
- Clean separation between logical intent and physical implementation

______________________________________________________________________

## Architecture

### Instrument Model

**Core Types:**

```python
class InstrumentType(Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    FUTURE = "future"
    FOREX = "forex"
    SIGNAL = "signal"  # Alternative data

class DataSource(Enum):
    ALGOSEEK = "algoseek"
    DATABASE = "database"
    IQFEED = "iqfeed"
    BINANCE = "binance"
    CSV_FILE = "csv_file"
    API = "api"

class Instrument(NamedTuple):
    symbol: str
    instrument_type: InstrumentType
    data_source: DataSource
    frequency: Optional[str] = None  # None = use global default
    metadata: Dict[str, Any] = {}    # Custom attributes
```

**Examples:**

```python
# Simple equity
apple = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)

# Crypto with custom frequency
btc = Instrument("BTCUSD", InstrumentType.CRYPTO, DataSource.BINANCE, frequency="1m")

# Alternative data signal
sentiment = Instrument(
    "NEWS_SENTIMENT",
    InstrumentType.SIGNAL,
    DataSource.DATABASE,
    metadata={"provider": "RavenPack", "lag_days": 1}
)
```

### DataSourceResolver

**Purpose:** Map logical `Instrument` to physical `DataAdapter` using external configuration.

**Configuration File (`data_sources.yaml`):**

```yaml
data_sources:
  algoseek:
    adapter: algoseek_parquet
    root_path: "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample"
    mode: standard_adjusted
    path_template: "{root_path}/SecId={secid}/*.parquet"
    symbol_map: "data/equity_security_master_sample.csv"

  database:
    adapter: postgres_adapter
    connection_string: "${DB_CONNECTION_STRING}"  # Environment variable
    schema: "market_data"
```

**Usage:**

```python
resolver = DataSourceResolver()  # Auto-finds config
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
adapter = resolver.resolve(instrument)
bars = adapter.read_bars(instrument, start_date="2020-01-01")
```

**Features:**

- Environment variable substitution (`${VAR_NAME}`)
- Config file search path:
  1. Explicit path provided to constructor
  1. `./config/data_sources.yaml` (project-relative)
  1. `~/.qtrader/data_sources.yaml` (user home)
- Adapter caching (one instance per adapter type)
- Validation on load

______________________________________________________________________

## Configuration Changes

### Before (Old Pattern)

```python
# Strategy file: examples/sma_crossover_strategy.py
backtest_config = {
    "data_paths": [
        "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample/SecId=33127/data_0.parquet"
    ],
    "symbols": ["AAPL"],
    # ...
}
```

**Problems:**

- ❌ Exposes file system structure
- ❌ Hard to mix data sources
- ❌ Can't parameterize by environment
- ❌ No instrument metadata
- ❌ Frequency not explicit

### After (New Pattern)

```python
# Strategy file: examples/sma_crossover_strategy.py
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

backtest_config = {
    "instruments": [
        Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK),
        Instrument("MSFT", InstrumentType.EQUITY, DataSource.ALGOSEEK),
        Instrument("BTCUSD", InstrumentType.CRYPTO, DataSource.BINANCE, frequency="1h"),
    ],
    "frequency": "1d",  # Global default
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    # ...
}
```

**Benefits:**

- ✅ Clean logical specification
- ✅ Mix asset types and sources
- ✅ Environment-independent strategy code
- ✅ Explicit frequency per instrument
- ✅ Rich metadata support

______________________________________________________________________

## Implementation Status

### Phase 0: Documentation ✅ COMPLETE

- Updated `docs/specs/phase01.md` (§2.4 Instrument Abstraction)
- Updated `docs/specs/phase01.md` (§3.2 Data Config)
- Updated `docs/implementation_plan_phase01.md` (Stage 6C section)
- Created `docs/stage6c_instrument_abstraction.md` (this file)

### Phase 1: Core Models ✅ COMPLETE

**Files Created:**

1. `src/qtrader/models/instrument.py` (78 lines)

   - `InstrumentType` enum (5 values)
   - `DataSource` enum (7 values)
   - `Instrument` NamedTuple with rich examples
   - Custom `__repr__` for debugging

1. `src/qtrader/adapters/resolver.py` (250 lines)

   - `DataSourceResolver` class
   - Config file search logic (3 locations)
   - Environment variable substitution
   - Adapter class loading (dynamic import)
   - `resolve()` method (Instrument → Adapter)
   - Helper methods (list_sources, get_source_config)

1. `config/data_sources.yaml` (46 lines)

   - Algoseek configuration (active)
   - CSV adapter configuration (active)
   - Database example (commented)
   - IQFeed example (commented)
   - Binance example (commented)

**Test Status:** Unit tests not yet created (Phase 2)

### Phase 2: Adapter Refactoring ⏳ NEXT

**Objective:** Update existing adapters to accept `Instrument` instead of file paths.

**Files to Modify:**

1. `src/qtrader/adapters/algoseek_parquet.py`

   - Change `__init__` to accept `(config: Dict, instrument: Instrument)`
   - Use `instrument.symbol` for symbol lookup
   - Use `config["symbol_map"]` for SecId lookup
   - Use `config["path_template"]` for file path construction

1. `src/qtrader/adapters/csv_adapter.py`

   - Change `__init__` to accept `(config: Dict, instrument: Instrument)`
   - Use `instrument.symbol` for file discovery
   - Use `config["root_path"]` for base directory

**Breaking Changes:**

- Remove `data_paths` parameter from adapters
- Remove hardcoded path logic
- All path construction via config templates

### Phase 3: Backtest Runner Integration ⏳ PENDING

**Files to Modify:**

1. `src/qtrader/api/backtest.py`
   - Change `__init__` to accept `instruments: List[Instrument]`
   - Instantiate `DataSourceResolver`
   - Call `resolver.resolve(instrument)` for each instrument
   - Remove `data_paths` parameter

### Phase 4: CLI Updates ⏳ PENDING

**Files to Modify:**

1. `src/qtrader/cli.py`
   - Update `_extract_backtest_config()` to handle `instruments` list
   - Remove `data_paths` and `symbols` extraction
   - Pass `instruments` to backtest runner
   - Update `_load_data_files()` helper

### Phase 5: Examples & Tests ⏳ PENDING

**Files to Modify:**

1. `examples/sma_crossover_strategy.py`

   - Replace `data_paths` + `symbols` with `instruments` list
   - Add import for `Instrument`, `InstrumentType`, `DataSource`

1. `tests/integration/test_backtest_full_execution.py`

   - Update 1-2 tests to use `Instrument` pattern
   - Verify multi-source scenarios work

### Phase 6: Validation & Documentation ⏳ PENDING

- Run full test suite (expect 455 → 460+ tests)
- Update CLI usage guide
- Create migration examples
- Document adapter development guide

______________________________________________________________________

## Design Decisions

### 1. Why NamedTuple for Instrument?

- Immutable (safer in multi-threaded future)
- Hashable (can use as dict key)
- Type-safe (strict field types)
- Lightweight (no overhead vs tuple)
- Easy to serialize (JSON/YAML)

### 2. Why Separate Config File (data_sources.yaml)?

- Environment-specific config (dev vs prod)
- Credentials isolation (API keys not in strategy code)
- Multi-strategy sharing (one config for all strategies)
- Version control friendly (config separate from code)

### 3. Why No Backward Compatibility?

User confirmed: "no need to backward compatibility this project is not yet in production won't affect anyone"

- Clean break from old pattern
- Simpler implementation
- No maintenance burden for deprecated code
- All examples/tests updated atomically

### 4. Why DataSourceResolver vs Direct Adapter Imports?

- Loose coupling (strategies don't import adapters)
- Dynamic adapter loading (plugin architecture)
- Config-driven adapter selection (no code changes)
- Testability (mock resolver vs mock adapters)

______________________________________________________________________

## Migration Guide

### Step 1: Update Strategy Configuration

**Before:**

```python
backtest_config = {
    "data_paths": ["data/.../data_0.parquet"],
    "symbols": ["AAPL"],
}
```

**After:**

```python
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

backtest_config = {
    "instruments": [
        Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK),
    ],
    "frequency": "1d",
}
```

### Step 2: Create data_sources.yaml (if not exists)

Copy `config/data_sources.yaml` to project root or `~/.qtrader/`.

### Step 3: Update Adapter Initialization (for custom adapters)

**Before:**

```python
adapter = AlgoseekParquetAdapter(data_path, config)
```

**After:**

```python
resolver = DataSourceResolver()
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
adapter = resolver.resolve(instrument)
```

______________________________________________________________________

## Next Steps

1. **Phase 2:** Refactor adapters to accept Instrument (2 days)
1. **Phase 3:** Update backtest runner (1 day)
1. **Phase 4:** Update CLI (1 day)
1. **Phase 5:** Update examples and tests (2 days)
1. **Phase 6:** Validation and documentation (1 day)

**Estimated Total:** 7 days (1.5 weeks)

______________________________________________________________________

## Files Created/Modified

### Created (Phase 1)

- `src/qtrader/models/instrument.py` (NEW)
- `src/qtrader/adapters/resolver.py` (NEW)
- `config/data_sources.yaml` (NEW)
- `docs/stage6c_instrument_abstraction.md` (NEW)

### Modified (Phase 0)

- `docs/specs/phase01.md` (§2.4, §3.2 updated)
- `docs/implementation_plan_phase01.md` (Stage 6C added, Stage 7 updated)

### To Be Modified (Phase 2-5)

- `src/qtrader/adapters/algoseek_parquet.py`
- `src/qtrader/adapters/csv_adapter.py`
- `src/qtrader/api/backtest.py`
- `src/qtrader/cli.py`
- `examples/sma_crossover_strategy.py`
- `tests/integration/test_backtest_full_execution.py`

______________________________________________________________________

**Version:** 1.0 **Status:** Phase 1 Complete, Phase 2 Ready to Start
