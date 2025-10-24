# Engine Package Documentation

**Package**: `qtrader.engine`\
**Purpose**: Pure event-driven backtesting orchestrator\
**Status**: Phase 1 - DataService Foundation (Minimal but Complete)

______________________________________________________________________

## Overview

The engine package provides the core backtesting orchestration infrastructure for QTrader. It coordinates all services via EventBus in a event-driven architecture

**Current Phase**: Phase 1 focuses on establishing the data layer foundation with full event persistence, providing a minimal but complete implementation that can be incrementally extended.

______________________________________________________________________

## Architecture Philosophy

### Clean Separation of Concerns

```yaml
# system.yaml - HOW services operate
execution:
  commission: 0.0005
  slippage: 5bp

# backtest.yaml - WHAT to test
start_date: 2020-01-01
universe: [AAPL, MSFT]
strategies: [momentum_20]
```

**Two Configuration Layers**:

1. **SystemConfig** (`system.yaml`): Service configurations (execution, risk, portfolio, data)
1. **BacktestConfig** (backtest YAML): Run parameters (dates, universe, capital, strategies)

This separation ensures:

- Fair comparison across backtests (same execution rules)
- Easy experimentation (change dates/universe without touching service configs)
- Professional architecture (mimics real trading systems)

______________________________________________________________________

## Package Structure

```
src/qtrader/engine/
├── __init__.py          # Public API exports
├── config.py            # Configuration models (BacktestConfig)
└── engine.py            # BacktestEngine orchestrator
```

______________________________________________________________________

## Module: config.py

Configuration models for backtest run parameters.

### Classes

#### DataSourceConfig

Configuration for a single data source with its universe.

**Purpose**: Allows specifying different symbols for different data sources.

```python
from qtrader.engine.config import DataSourceConfig

# Example: Load AAPL prices from algoseek
source = DataSourceConfig(
    name="algoseek-us-equity-1d-unadjusted",
    universe=["AAPL", "MSFT", "GOOGL"]
)
```

**Fields**:

- `name` (str): Data source name from `data_sources.yaml`
- `universe` (list[str]): Symbols to load from this data source

**Validation**:

- Both fields are required
- Empty universe list is allowed

______________________________________________________________________

#### DataSelectionConfig

Data selection configuration for backtest run.

**Purpose**: Specifies WHAT data to load for this specific backtest.

```python
from qtrader.engine.config import DataSelectionConfig, DataSourceConfig

# Example: Single data source
data_config = DataSelectionConfig(
    sources=[
        DataSourceConfig(
            name="algoseek-us-equity-1d-unadjusted",
            universe=["AAPL", "MSFT", "GOOGL"]
        )
    ]
)
```

**Fields**:

- `sources` (list[DataSourceConfig]): Data sources with their universes (min 1 required)

**Validation**:

- At least one source required (`min_length=1`)
- **Current Limitation**: Multiple sources raise ValidationError (multi-source streaming pending)

______________________________________________________________________

#### StrategyConfigItem

Configuration for a single strategy.

**Purpose**: References strategies by registry name, not file path.

```python
from qtrader.engine.config import StrategyConfigItem

# Example: Momentum strategy configuration
strategy = StrategyConfigItem(
    strategy_id="momentum_20",         # Registry name
    universe=["AAPL", "MSFT"],         # Symbols strategy trades
    data_sources=["algoseek-us-equity-1d-unadjusted"],
    config={
        "lookback": 20,
        "warmup_bars": 21
    }
)
```

**Fields**:

- `strategy_id` (str): Strategy name from registry (builtin or custom)
- `universe` (list[str]): Symbols this strategy trades (must be subset of backtest universe)
- `data_sources` (list[str]): Data sources this strategy uses
- `config` (dict[str, Any]): Strategy-specific config overrides (default: `{}`)

**Validation**:

- `strategy_id`, `universe`, and `data_sources` are required
- Strategy universe must be subset of loaded data symbols
- Raises `ValidationError` if strategy uses symbols not in data sources

______________________________________________________________________

#### RiskPolicyConfig

Risk policy configuration at Portfolio Manager level.

**Purpose**: References risk policies by registry name.

```python
from qtrader.engine.config import RiskPolicyConfig

# Example: Naive risk policy
risk = RiskPolicyConfig(
    name="naive",
    config={"max_pct_position_size": 0.30}
)
```

**Fields**:

- `name` (str): Risk policy name from registry
- `config` (dict[str, Any]): Policy-specific config overrides (default: `{}`)

**Validation**:

- `name` is required
- `config` defaults to empty dict

______________________________________________________________________

#### BacktestConfig

Complete backtest run configuration.

**Purpose**: Contains ONLY per-run parameters (dates, equity, data, strategies).

```python
from qtrader.engine.config import BacktestConfig, load_backtest_config

# Load from YAML
config = load_backtest_config("config/portfolio.yaml")

# Or construct programmatically
config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 12, 31),
    initial_equity=Decimal("100000"),
    replay_speed=0.0,
    data=DataSelectionConfig(...),
    strategies=[StrategyConfigItem(...)],
    risk_policy=RiskPolicyConfig(...)
)
```

**Fields**:

- `start_date` (datetime): Backtest start date
- `end_date` (datetime): Backtest end date
- `initial_equity` (Decimal): Starting equity
- `replay_speed` (float): Replay speed in seconds per bar (0.0 = full speed, 1.0 = 1 sec/bar)
- `data` (DataSelectionConfig): Data sources with per-source universes
- `strategies` (list[StrategyConfigItem]): Strategy configurations
- `risk_policy` (RiskPolicyConfig): Risk policy (Portfolio Manager level)

**Properties**:

- `all_symbols` (set[str]): Get all symbols across all data sources

**Validation**:

1. **Date Validation**: `end_date` must be after `start_date`
1. **Single Source**: Only one data source allowed (multi-source pending)
1. **Strategy Universe**: Strategy symbols must be subset of loaded data
1. **Replay Speed**: Must be >= 0.0

**Example YAML**:

```yaml
start_date: 2020-01-01
end_date: 2023-12-31
initial_equity: 100000
replay_speed: 0.0

data:
  sources:
    - name: algoseek-us-equity-1d-unadjusted
      universe: [AAPL, MSFT, GOOGL]

strategies:
  - strategy_id: momentum_20
    universe: [AAPL, MSFT]
    data_sources: [algoseek-us-equity-1d-unadjusted]
    config:
      lookback: 20
      warmup_bars: 21

risk_policy:
  name: naive
  config:
    max_pct_position_size: 0.30
```

______________________________________________________________________

### Functions

#### load_backtest_config()

Load and validate backtest configuration from YAML file.

```python
from qtrader.engine.config import load_backtest_config

# Load configuration
config = load_backtest_config("config/portfolio.yaml")

print(f"Running backtest from {config.start_date} to {config.end_date}")
print(f"Universe: {config.all_symbols}")
print(f"Strategies: {[s.strategy_id for s in config.strategies]}")
```

**Parameters**:

- `config_path` (str | Path): Path to YAML configuration file

**Returns**:

- `BacktestConfig`: Validated configuration object

**Raises**:

- `ConfigLoadError`: If file not found, invalid YAML, or validation fails

______________________________________________________________________

### Exceptions

#### ConfigLoadError

Raised when config loading fails.

```python
from qtrader.engine.config import ConfigLoadError, load_backtest_config

try:
    config = load_backtest_config("nonexistent.yaml")
except ConfigLoadError as e:
    print(f"Config error: {e}")
```

______________________________________________________________________

## Module: engine.py

Pure event-driven backtesting orchestrator.

### Classes

#### BacktestResult

Results from a backtest run.

```python
from qtrader.engine.engine import BacktestResult
from datetime import date, timedelta

result = BacktestResult(
    start_date=date(2020, 1, 1),
    end_date=date(2020, 12, 31),
    bars_processed=757,
    duration=timedelta(seconds=7)
)

print(f"Processed {result.bars_processed} bars in {result.duration}")
```

**Fields**:

- `start_date` (date): Backtest start date
- `end_date` (date): Backtest end date
- `bars_processed` (int): Number of price bars processed
- `duration` (timedelta): Execution duration

______________________________________________________________________

#### BacktestEngine

Event-driven backtesting orchestrator.

**Architecture**: Phase 1 - DataService Foundation

**Current Capabilities**:

- Load and validate backtest configuration
- Create and manage EventBus with EventStore persistence
- Initialize DataService with proper dataset configuration
- Stream historical data with timestamp synchronization
- Track execution metrics (bars processed, duration)

**Intentional Limitations** (Phase 1):

- No portfolio tracking (PortfolioService suspended)
- No order execution simulation (ExecutionService suspended)
- No risk management (RiskService suspended)
- No strategy signals (StrategyService suspended)

These services will be incrementally reintegrated following the lego architecture pattern.

______________________________________________________________________

### Event Flow

#### Current Phase (DataService Only)

```
For each timestamp T across all symbols:
    1. DataService publishes PriceBarEvent(symbol=A, timestamp=T)
    2. DataService publishes PriceBarEvent(symbol=B, timestamp=T)
    3. ...all symbols at T before advancing to T+1
    4. EventStore persists all events
```

#### Future Phase (After Service Integration)

```
For each timestamp T:
    Phase 1: MarketData
        - DataService publishes PriceBarEvent for ALL symbols at T
        - Services update internal state (prices, positions)

    Phase 2: Valuation (barrier)
        - Engine publishes ValuationTriggerEvent(ts=T)
        - PortfolioService calculates equity, positions, valuations

    Phase 3: RiskEvaluation (barrier)
        - Engine publishes RiskEvaluationTriggerEvent(ts=T)
        - RiskService processes signals from strategies
        - RiskService creates sized orders within risk limits

    Phase 4: Execution (next cycle)
        - ExecutionService fills orders at T+1 prices
        - FillEvent updates portfolio positions
```

______________________________________________________________________

### Constructor

```python
from qtrader.engine.engine import BacktestEngine
from qtrader.events.event_bus import EventBus
from qtrader.services.data.service import DataService

engine = BacktestEngine(
    config=backtest_config,
    event_bus=EventBus(),
    data_service=DataService(...),
    event_store=None,           # Optional
    results_dir=None            # Optional
)
```

**Parameters**:

- `config` (BacktestConfig): Backtest configuration
- `event_bus` (EventBus): Event bus for publishing events
- `data_service` (DataService): Data service for loading historical bars
- `event_store` (EventStore | None): Optional persistence backend
- `results_dir` (Path | None): Optional directory for run artifacts

**Note**: Prefer using `BacktestEngine.from_config()` factory method instead of direct construction.

______________________________________________________________________

### Class Methods

#### from_config()

Factory method to create engine from configuration.

**Recommended Usage**: This is the standard way to create a BacktestEngine.

```python
from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine

# Load configuration
config = load_backtest_config("config/portfolio.yaml")

# Create engine (auto-initializes all services)
engine = BacktestEngine.from_config(config)

# Run backtest
result = engine.run()

# Cleanup
engine.shutdown()
```

**What it does**:

1. Loads SystemConfig for service configurations
1. Initializes logging from system config
1. Creates EventBus
1. Initializes EventStore (SQLite or in-memory based on config)
1. Creates results directory with timestamp
1. Initializes DataService with proper dataset
1. Returns configured BacktestEngine

**Parameters**:

- `config` (BacktestConfig): Backtest configuration loaded from YAML

**Returns**:

- `BacktestEngine`: Configured engine instance

**Raises**:

- `ValueError`: If configuration is invalid or services fail to initialize

______________________________________________________________________

### Instance Methods

#### run()

Run the backtest - stream data and publish events.

```python
from qtrader.engine.engine import BacktestEngine

engine = BacktestEngine.from_config(config)

try:
    result = engine.run()

    print(f"Bars processed: {result.bars_processed}")
    print(f"Duration: {result.duration}")
finally:
    engine.shutdown()
```

**Current Implementation** (DataService only):

1. Stream historical data for all symbols in date range
1. DataService publishes PriceBarEvent for each bar
1. EventStore persists all events
1. Return basic metrics (bars processed, duration)

**Future** (After Service Refactoring):

- Add warmup phase for strategies
- Publish ValuationTriggerEvent after all bars for timestamp
- Publish RiskEvaluationTriggerEvent for signal processing
- Collect portfolio metrics and trade statistics
- Generate comprehensive results

**Returns**:

- `BacktestResult`: Basic metrics (bars, duration)

**Raises**:

- `RuntimeError`: If backtest execution fails

**Memory Warning**: Current implementation loads all bars into memory before publishing. For large universes or long date ranges, this can consume significant RAM.

Estimated memory: `~500 bytes/bar * symbols * trading_days`

- 100 symbols × 252 days × 500 bytes = ~12.6 MB (manageable)
- 1000 symbols × 2520 days × 500 bytes = ~1.26 GB (high)

TODO: Refactor to use heap-merge streaming for incremental publishing.

______________________________________________________________________

#### shutdown()

Clean up resources (close EventStore).

```python
from qtrader.engine.engine import BacktestEngine

engine = BacktestEngine.from_config(config)

try:
    result = engine.run()
finally:
    engine.shutdown()  # Always cleanup
```

**Purpose**: Properly close SQLite connections and release file handles.

**Important**: Call this method for long-lived daemons or repeated runs.

______________________________________________________________________

### Context Manager Support

The engine supports context manager protocol for automatic cleanup:

```python
from qtrader.engine.engine import BacktestEngine

# Recommended pattern
with BacktestEngine.from_config(config) as engine:
    result = engine.run()
    # shutdown() called automatically
```

______________________________________________________________________

## Usage Examples

### Basic Backtest

```python
from pathlib import Path
from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine

# Load configuration
config = load_backtest_config("config/portfolio.yaml")

# Create and run engine
with BacktestEngine.from_config(config) as engine:
    result = engine.run()

    print(f"Backtest completed!")
    print(f"  Date Range: {result.start_date} to {result.end_date}")
    print(f"  Bars Processed: {result.bars_processed:,}")
    print(f"  Duration: {result.duration}")
```

### Access Results Directory

```python
from qtrader.engine.engine import BacktestEngine

engine = BacktestEngine.from_config(config)

try:
    result = engine.run()

    # Access results directory
    if engine._results_dir:
        print(f"Results saved to: {engine._results_dir}")

        # Access event database
        event_db = engine._results_dir / "events.sqlite"
        if event_db.exists():
            size_mb = event_db.stat().st_size / (1024 * 1024)
            print(f"Event database: {size_mb:.2f} MB")
finally:
    engine.shutdown()
```

### Programmatic Configuration

```python
from datetime import datetime
from decimal import Decimal
from qtrader.engine.config import (
    BacktestConfig,
    DataSelectionConfig,
    DataSourceConfig,
    StrategyConfigItem,
    RiskPolicyConfig
)
from qtrader.engine.engine import BacktestEngine

# Build configuration programmatically
config = BacktestConfig(
    start_date=datetime(2020, 1, 1),
    end_date=datetime(2020, 12, 31),
    initial_equity=Decimal("100000"),
    replay_speed=0.0,
    data=DataSelectionConfig(
        sources=[
            DataSourceConfig(
                name="algoseek-us-equity-1d-unadjusted",
                universe=["AAPL", "MSFT", "GOOGL"]
            )
        ]
    ),
    strategies=[
        StrategyConfigItem(
            strategy_id="momentum_20",
            universe=["AAPL", "MSFT"],
            data_sources=["algoseek-us-equity-1d-unadjusted"],
            config={"lookback": 20}
        )
    ],
    risk_policy=RiskPolicyConfig(
        name="naive",
        config={"max_pct_position_size": 0.30}
    )
)

# Run backtest
with BacktestEngine.from_config(config) as engine:
    result = engine.run()
```

______________________________________________________________________

## Configuration Files

### Backtest Configuration (portfolio.yaml)

```yaml
# config/portfolio.yaml
start_date: 2020-01-01
end_date: 2023-12-31
initial_equity: 100000
replay_speed: 0.0  # Full speed

data:
  sources:
    - name: algoseek-us-equity-1d-unadjusted
      universe: [AAPL, MSFT, GOOGL]

strategies:
  - strategy_id: momentum_20
    universe: [AAPL, MSFT]
    data_sources: [algoseek-us-equity-1d-unadjusted]
    config:
      lookback: 20
      warmup_bars: 21

risk_policy:
  name: naive
  config:
    max_pct_position_size: 0.30
```

### System Configuration (system.yaml)

```yaml
# config/system.yaml
output:
  default_results_dir: "output/backtests"
  event_store:
    backend: "sqlite"      # or "memory"
    filename: "events.sqlite"
  organize_by_date: true   # Create YYYY/MM/DD subdirectories
  use_timestamps: true
  timestamp_format: "%Y%m%d_%H%M%S"

logging:
  level: "INFO"
  format: "console"
  enable_file: true
  file_level: "WARNING"

data:
  sources_config: "config/data_sources.yaml"
  default_mode: "adjusted"
  default_timezone: "America/New_York"
```

______________________________________________________________________

## Results Directory Structure

```
output/backtests/
└── 2024/                                    # organize_by_date=true
    └── 10/
        └── 24/
            └── 20241024_142153/             # use_timestamps=true
                ├── events.sqlite            # Event persistence
                └── (future: metrics, trades, etc.)
```

**Without organize_by_date**:

```
output/backtests/
└── 20241024_142153/
    └── events.sqlite
```

**Without timestamps** (uses source + dates):

```
output/backtests/
└── algoseek-us-equity-1d-unadjusted_20200101_20201231/
    └── events.sqlite
```

______________________________________________________________________

## Event Store Backends

### SQLite Backend (Default)

**Pros**:

- Persistent storage across runs
- SQL queryable for analysis
- Efficient for large backtests

**Cons**:

- File I/O overhead
- Requires cleanup (call `shutdown()`)

**Configuration**:

```yaml
output:
  event_store:
    backend: "sqlite"
    filename: "events.sqlite"
```

### InMemory Backend

**Pros**:

- Fast (no I/O)
- No cleanup required
- Good for testing/debugging

**Cons**:

- Lost after run completes
- High memory usage for long backtests

**Configuration**:

```yaml
output:
  event_store:
    backend: "memory"
    filename: "events.sqlite"  # Ignored for memory backend
```

______________________________________________________________________

## Design Patterns

### Factory Pattern

`BacktestEngine.from_config()` implements factory pattern:

- Centralizes complex initialization
- Hides service wiring details
- Provides single creation point

### Context Manager Pattern

Ensures proper resource cleanup:

```python
with BacktestEngine.from_config(config) as engine:
    result = engine.run()
    # shutdown() called automatically on exit
```

### Event-Driven Architecture

Engine orchestrates via events, not direct calls:

- Services subscribe to events
- Engine publishes coordination events
- Loose coupling between components

______________________________________________________________________

## Current Limitations (Phase 1)

1. **Single Data Source**: Only `sources[0]` is used. Multi-source streaming pending.
1. **Memory Usage**: Buffers all bars before publishing (heap-merge refactor pending).
1. **No Portfolio Tracking**: PortfolioService suspended until refactored.
1. **No Order Execution**: ExecutionService suspended until refactored.
1. **No Risk Management**: RiskService suspended until refactored.
1. **No Strategy Signals**: StrategyService suspended until refactored.
1. **Basic Metrics Only**: BacktestResult only tracks bars and duration.

______________________________________________________________________

## Future Enhancements

### Phase 2: Service Integration

1. **PortfolioService**: Track positions, equity, valuations
1. **ExecutionService**: Simulate order fills
1. **RiskService**: Apply risk limits, size orders
1. **StrategyService**: Generate trading signals

### Phase 3: Advanced Features

1. **Warmup Phase**: Load historical data for indicator initialization
1. **Multi-Source Streaming**: Support multiple concurrent data feeds
1. **Barrier Events**: ValuationTriggerEvent, RiskEvaluationTriggerEvent
1. **Performance Metrics**: Sharpe, drawdown, win rate, etc.
1. **Trade Analysis**: Entry/exit details, hold periods, P&L
1. **HTML Reports**: Interactive dashboards
1. **Live Trading**: Connect to broker APIs

______________________________________________________________________

## Testing

### Unit Tests

Located in: `tests/unit/engine/`

**test_config.py**:

- Configuration model validation
- YAML loading and error handling
- Date validation, universe validation
- Multi-source restriction enforcement

**test_engine.py**:

- Engine initialization
- Factory method (`from_config`)
- Run execution and error handling
- Context manager support
- Results directory creation
- Event store initialization

### Integration Tests

Located in: `tests/integration/engine/`

Test complete backtest runs with real data.

______________________________________________________________________

## API Reference Summary

### Public API (`qtrader.engine`)

**Configuration**:

- `BacktestConfig` - Complete backtest configuration
- `load_backtest_config(path)` - Load config from YAML

**Engine**:

- `BacktestEngine` - Orchestrator class
- `BacktestEngine.from_config(config)` - Factory method
- `BacktestResult` - Result dataclass

**Example Import**:

```python
from qtrader.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    load_backtest_config
)
```

______________________________________________________________________

## Migration Guide

### From Old Architecture

**Before** (Direct service instantiation):

```python
# Old way - manual wiring
data_service = DataService(...)
portfolio_service = PortfolioService(...)
execution_service = ExecutionService(...)

# Manual event wiring
event_bus.subscribe(...)
```

**After** (Factory pattern):

```python
# New way - automatic wiring
config = load_backtest_config("config/portfolio.yaml")
engine = BacktestEngine.from_config(config)
result = engine.run()
```

______________________________________________________________________

## Best Practices

1. **Always use `from_config()`**: Don't construct BacktestEngine directly
1. **Use context manager**: Ensures proper cleanup
1. **Separate configs**: Keep system.yaml (services) and backtest.yaml (runs) separate
1. **Version control configs**: Track backtest configurations in git
1. **Monitor memory**: Be aware of universe size × date range for large backtests
1. **Call shutdown()**: If not using context manager, always call `shutdown()`

______________________________________________________________________

## Troubleshooting

### Problem: Multiple data sources error

**Error**: `ValidationError: Multiple data sources not yet supported`

**Solution**: Use single data source until multi-source streaming is implemented:

```yaml
data:
  sources:
    - name: algoseek-us-equity-1d-unadjusted
      universe: [AAPL, MSFT, GOOGL]  # All symbols in one source
```

### Problem: Strategy universe validation error

**Error**: `Strategy 'momentum_20' universe contains symbols not in data sources: {'TSLA'}`

**Solution**: Ensure strategy universe is subset of data universe:

```yaml
data:
  sources:
    - name: algoseek-us-equity-1d-unadjusted
      universe: [AAPL, MSFT, TSLA]  # Include TSLA

strategies:
  - strategy_id: momentum_20
    universe: [AAPL, TSLA]  # Now valid
```

### Problem: High memory usage

**Symptoms**: Large universe + long date range = high RAM usage

**Solution**: Reduce universe size or date range until heap-merge refactor:

```yaml
# Instead of
universe: [100 symbols]
start_date: 2010-01-01
end_date: 2024-12-31

# Try
universe: [20 symbols]  # Smaller universe
start_date: 2020-01-01  # Shorter range
```

### Problem: SQLite database locked

**Error**: `database is locked`

**Solution**: Ensure `shutdown()` is called to close connections:

```python
engine = BacktestEngine.from_config(config)
try:
    result = engine.run()
finally:
    engine.shutdown()  # Critical!
```

______________________________________________________________________

## Related Documentation

- **Events Package**: `docs/packages/events.md` - Event types and EventBus
- **Data Service**: `docs/packages/data.md` - Data loading and streaming
- **System Config**: `docs/system_config.md` - Service configurations

______________________________________________________________________

**Last Updated**: 2024-10-24\
**Version**: Phase 1 - DataService Foundation\
**Status**: Stable (Production-ready for data-only backtests)
