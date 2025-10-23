# QTrader Configuration Architecture

This directory contains **system-level configuration** for the QTrader framework. These settings define **HOW the trading system operates**, not **WHAT to backtest**.

______________________________________________________________________

## Configuration Philosophy

QTrader has a clean separation between **framework configuration** and **backtest run configuration**:

### 1. System Configuration (This directory)

**What it is**: Framework-level settings that define how QTrader operates

**File**: `config/system.yaml`

**Stability**: Stable across backtests (the "trading system")

**Contains**:

- **ALL service configurations**: Execution, Risk, Portfolio, Data
- Commission models and slippage simulation
- Risk limits and position sizing defaults
- Portfolio accounting methods (lot tracking: FIFO/LIFO)
- Data infrastructure (sources, caching, timezone)
- Output and logging configuration
- Development/debug settings

**Philosophy**: *"In real life, the system is one"*

You don't change your broker's commission structure between backtests. The system config represents the infrastructure that all backtests run on.

### 2. Backtest Run Configuration (External YAML)

**What it is**: Per-run parameters for individual backtests

**File**: External YAML file (e.g., `my_backtest.yaml`)

**Location**: Outside the codebase (passed to backtest engine)

**Contains ONLY**:

- Start/end dates
- Universe (symbols to trade)
- Initial capital
- Warmup bars
- Dataset selection
- Strategy configurations (which strategies, their parameters)

**Philosophy**: *"The experiment"*

Each backtest varies these parameters while running on the same system infrastructure.

### 3. Data Sources Configuration

**What it is**: Data adapter infrastructure mapping

**File**: `config/data_sources.yaml`

**Purpose**: Maps logical data sources to physical adapters

**Why separate**: Security (API keys), reusability (shared across projects)

______________________________________________________________________

## Configuration Files

### `system.yaml` - System Configuration

The **authoritative** configuration for how QTrader operates.

**When to edit**:

- ✅ You changed brokers (different commission structure)
- ✅ You want more aggressive slippage assumptions
- ✅ You need debug-level logging
- ❌ You want to test a different date range (use backtest config)
- ❌ You want to try different strategies (use backtest config)

**Access in code**:

```python
from qtrader.system import get_system_config

config = get_system_config()
print(f"Commission: ${config.execution.commission.per_share} per share")
print(f"Max position: {config.risk.concentration.max_position_pct * 100}%")
```

**For logging**:

```python
from qtrader.system import LoggerFactory, LoggingConfig

# Configure at startup
LoggerFactory.configure(LoggingConfig(level="DEBUG"))

# Use throughout code
logger = LoggerFactory.get_logger()
logger.info("trading.order_placed", symbol="AAPL", quantity=100)
```

**Services load from system.yaml**:

```python
# BacktestEngine.from_config() loads system config
system_config = get_system_config()

# Services use system config
execution_service = ExecutionService.from_config(system_config.execution)
risk_service = RiskService.from_config(system_config.risk)
portfolio_service = PortfolioService.from_config(system_config.portfolio)
```

### `data_sources.yaml` - Data Adapter Configuration

Maps logical data sources (used in strategies) to physical data adapters.

**Why separate from `system.yaml`?**

1. **Security**: Contains API keys and credentials
1. **Reusability**: Can be shared via `~/.qtrader/data_sources.yaml`
1. **Modularity**: Data config changes independently of system config

______________________________________________________________________

## Configuration Precedence

Settings are loaded with the following priority (highest to lowest):

1. **CLI flags** - Direct command-line arguments
1. **Environment variables** - `${VAR_NAME}` syntax in YAML
1. **Project config** - `./config/system.yaml` (this directory)
1. **User config** - `~/.qtrader/system.yaml` (your home directory)
1. **Built-in defaults** - Hardcoded in `src/qtrader/system/config.py`

______________________________________________________________________

## Design Principles

### System Config = "The Trading System"

- **One system**: All backtests run on the same infrastructure
- **All services**: Execution, Risk, Portfolio, Data all configured here
- **Fair comparison**: Same commission/slippage across tests
- **Realistic simulation**: Matches your actual trading setup
- **Stable**: Changes rarely (only when system changes)

### Backtest Config = "The Experiment"

- **Run parameters only**: Dates, universe, capital, strategies
- **No service configs**: Services pull from system.yaml
- **Flexible**: Different parameters per test
- **Versioned**: Track config alongside research notebooks
- **Composable**: Mix and match strategies, datasets

### Data Sources Config = "Infrastructure Mapping"

- **Reusable**: Same config across projects
- **Secure**: Keep credentials separate
- **Flexible**: Swap data providers without changing code

______________________________________________________________________

## Important Notes

### What Goes Where?

| Setting                  | System Config (`system.yaml`) | Backtest Config (external YAML) |
| ------------------------ | ----------------------------- | ------------------------------- |
| Commission model         | ✅                            | ❌                              |
| Slippage model           | ✅                            | ❌                              |
| Risk limits              | ✅                            | ❌                              |
| Position sizing defaults | ✅                            | ❌                              |
| Lot tracking method      | ✅                            | ❌                              |
| Output directory         | ✅                            | ❌                              |
| Start/end dates          | ❌                            | ✅                              |
| Universe (symbols)       | ❌                            | ✅                              |
| Initial capital          | ❌                            | ✅                              |
| Warmup bars              | ❌                            | ✅                              |
| Replay speed             | ❌                            | ✅                              |
| Dataset selection        | ❌                            | ✅                              |
| Strategy selection       | ❌                            | ✅                              |
| Strategy parameters      | ❌                            | ✅                              |

**Rule of Thumb**:

- **System config** (`system.yaml`): ALL service configurations - how the framework operates
- **Backtest config** (external): ONLY run parameters - what to test

### Replay Speed (Historical Backtests Only)

The `replay_speed` parameter in backtest config controls how fast historical data is fed through the system. This is useful for:

- **Visualization/Debugging**: Watch the backtest unfold slowly (1.0 = 1 second per bar)
- **Educational purposes**: Demonstrate how the system processes data
- **Integration testing**: Test with external systems that can't handle full speed

**Values**:

- `0.0`: Full speed (default) - no delay between bars
- `0.1`: 100ms per bar - fast but visible
- `1.0`: 1 second per bar - easy to watch
- `5.0`: 5 seconds per bar - presentation mode

**Important**: This only affects historical backtests. Live trading ignores this parameter (data arrives in real-time).

**Example**:

```yaml
# backtest.yaml
start_date: 2020-01-01
end_date: 2023-12-31
initial_capital: 100000
universe: [AAPL, MSFT]
replay_speed: 1.0  # 1 second per bar for debugging

data:
  dataset: algoseek-us-equity-1d-unadjusted

strategies:
  - path: strategies/momentum.py
    strategy_id: momentum_v1
    config:
      lookback: 20
```
