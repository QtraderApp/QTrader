# Backtest Configuration Files

This directory contains YAML configuration files for QTrader backtests. The new configuration format supports both single-strategy and multi-strategy backtests with type-safe validation using Pydantic.

## File Structure

```
config/backtests/
├── README.md                          # This file
├── single_strategy_sma.yaml           # Simple SMA crossover example
└── multi_strategy_tech_vs_etf.yaml    # Multi-strategy with reallocation
```

## Configuration Schema

### Top-Level Fields

| Field                 | Type   | Required | Description                                           |
| --------------------- | ------ | -------- | ----------------------------------------------------- |
| `strategies`          | list   | Yes      | List of strategy configurations                       |
| `start_date`          | date   | Yes      | Backtest start date (YYYY-MM-DD)                      |
| `end_date`            | date   | Yes      | Backtest end date (YYYY-MM-DD)                        |
| `initial_capital`     | float  | Yes      | Starting capital (USD)                                |
| `execution`           | object | No       | Execution settings (commission, slippage, fill delay) |
| `risk`                | object | No       | Risk management settings                              |
| `reallocation_policy` | object | No       | Dynamic capital reallocation between strategies       |

### Strategy Configuration

Each strategy in the `strategies` list requires:

| Field                | Type   | Required | Description                                                         |
| -------------------- | ------ | -------- | ------------------------------------------------------------------- |
| `name`               | string | Yes      | Unique strategy identifier                                          |
| `module_path`        | string | Yes      | Python module path (e.g., `examples.sma_crossover_strategy`)        |
| `class_name`         | string | Yes      | Strategy class name                                                 |
| `capital_allocation` | float  | Yes      | Fraction of capital (0.0-1.0), must sum to 1.0                      |
| `instruments`        | list   | Yes      | List of symbols to trade                                            |
| `config`             | object | No       | Strategy-specific parameters (validated by StrategyConfig subclass) |

### Execution Settings

```yaml
execution:
  commission_pct: 0.001    # Commission as percentage (0.001 = 0.1%)
  slippage_bps: 5.0        # Slippage in basis points
  fill_delay: 1            # Bars to wait for fill (1 = next bar)
```

### Risk Management

```yaml
risk:
  max_position_size: 0.20   # Maximum position size as % of portfolio
  max_leverage: 1.0         # Maximum leverage (1.0 = no leverage)
  stop_loss_pct: 0.10       # Stop loss percentage (optional)
```

### Reallocation Policies

#### Fixed Allocation

```yaml
reallocation_policy:
  type: "fixed"
  allocations:
    Strategy1: 0.60
    Strategy2: 0.40
```

#### Performance-Based Reallocation

```yaml
reallocation_policy:
  type: "performance_based"
  frequency: "monthly"              # monthly, quarterly, annual
  lookback_days: 90                 # Performance window
  min_allocation: 0.20              # Min allocation per strategy
  max_allocation: 0.80              # Max allocation per strategy
  performance_metric: "sharpe_ratio"  # sharpe_ratio, total_return, sortino_ratio
  rebalance_threshold: 0.05         # Only rebalance if drift > 5%
```

## Usage

### CLI Usage

```bash
# Run with YAML config
qtrader backtest --config config/backtests/single_strategy_sma.yaml

# Run multi-strategy backtest
qtrader backtest --config config/backtests/multi_strategy_tech_vs_etf.yaml
```

### Python API

```python
from qtrader.config.backtest_config import BacktestConfig

# Load and validate configuration
config = BacktestConfig.from_yaml("config/backtests/single_strategy_sma.yaml")

# Check if multi-strategy
if config.is_multi_strategy:
    print(f"Running {len(config.strategies)} strategies")

# Access strategy configs
for strategy in config.strategies:
    print(f"{strategy.name}: {len(strategy.instruments)} instruments")
```

## Validation Rules

The configuration is validated using Pydantic with the following rules:

1. **Capital Allocation**: Sum of all `capital_allocation` values must equal 1.0 (100%)
1. **Strategy Names**: Must be unique within a configuration
1. **Dates**: `end_date` must be after `start_date`
1. **Instruments**: Each strategy must have at least one instrument
1. **Reallocation**: If multi-strategy, can optionally specify reallocation policy

## Strategy-Specific Configuration

Strategies can define custom configuration classes by subclassing `StrategyConfig`:

```python
from pydantic import Field, field_validator
from qtrader.api.strategy import StrategyConfig

class SMAConfig(StrategyConfig):
    """Configuration for SMA Crossover strategy."""

    fast_period: int = Field(gt=0, description="Fast MA period")
    slow_period: int = Field(gt=0, description="Slow MA period")

    @field_validator("slow_period")
    @classmethod
    def slow_must_be_greater(cls, v: int, info) -> int:
        fast = info.data.get("fast_period")
        if fast and v <= fast:
            raise ValueError("slow_period must be > fast_period")
        return v
```

Then in YAML:

```yaml
strategies:
  - name: "SMA_Tech"
    config:
      fast_period: 20
      slow_period: 50  # Validated: must be > fast_period
```

## Examples

### Single Strategy (100% allocation)

- **File**: `single_strategy_sma.yaml`
- **Strategy**: SMA Crossover on 5 tech stocks
- **Features**: Simple configuration, no reallocation

### Multi-Strategy with Reallocation

- **File**: `multi_strategy_tech_vs_etf.yaml`
- **Strategies**:
  - SMA Crossover on 8 tech stocks (60% initial)
  - Buy & Hold SPY (40% initial)
- **Features**:
  - Separate universes per strategy
  - Quarterly reallocation based on Sharpe ratio
  - Min/max allocation constraints

## Migration from Dict-Based Config

**Old (Python dict):**

```python
config = {
    "start_date": "2020-01-01",
    "end_date": "2023-12-31",
    "initial_capital": 100000,
    "commission_pct": 0.001,
    "instruments": ["AAPL", "MSFT"],  # Single universe
}
```

**New (YAML):**

```yaml
start_date: "2020-01-01"
end_date: "2023-12-31"
initial_capital: 100000
execution:
  commission_pct: 0.001

strategies:
  - name: "MyStrategy"
    module_path: "my_module"
    class_name: "MyStrategy"
    capital_allocation: 1.0
    instruments: ["AAPL", "MSFT"]  # Per-strategy universe
    config: {}
```

## See Also

- **Code**: `src/qtrader/config/backtest_config.py` - Pydantic models
- **Code**: `src/qtrader/config/reallocation.py` - Reallocation policies
- **Docs**: `docs/architecture.md` - Architecture overview
