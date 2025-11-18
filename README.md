# QTrader

A Python backtesting framework for quantitative trading strategies.

> 📦 **This is the QTrader package repository** for developers. After installing (`pip install qtrader`), use `qtrader init-project` to create your **trading project** with configs, strategies, and data.

## What It Does

QTrader lets you backtest trading strategies with real market data. It handles data loading, order execution simulation, portfolio tracking, and performance reporting.

## Features

- Extensible data adapter system with built-in Yahoo CSV support
- Realistic execution simulation with slippage and commissions
- Portfolio and position management
- Corporate actions handling (splits, dividends)
- Event-driven architecture
- 22 technical indicators across 5 categories (moving averages, momentum, volatility, volume, trend)
- Comprehensive test coverage (1664 tests, 80% coverage)

## Quick Start

### Installation

```bash
pip install qtrader
```

### Initialize a New Project

```bash
# Create a complete backtesting project with sample strategies and data
qtrader init-project my-trading-system
cd my-trading-system
```

> **Note:** This documentation refers to the **QTrader package repository**. When you run `qtrader init-project`, you get a **scaffolded project** with a different structure (configs, strategies, data). The templates and examples are in your new project, not in this repository.

This creates a ready-to-use project structure:

```
my-trading-system/
├── config/
│   ├── system.yaml              # System configuration
│   ├── data_sources.yaml        # Data source definitions
│   └── backtests/
│       ├── buy_hold.yaml        # Example: Buy and hold
│       └── sma_crossover.yaml   # Example: SMA crossover
├── library/
│   └── strategies/
│       ├── buy_and_hold.py     # Example strategy
│       └── sma_crossover.py    # Example strategy
├── data/
│   └── us-equity-yahoo-csv/
│       └── AAPL.csv            # Sample data (100 bars)
├── examples/
│   └── run_backtest.py         # Example scripts
└── run_backtest.py             # Simple runner script (copy)
```

### Run Your First Backtest

**Using the CLI (Recommended):**

```bash
# Inside your scaffolded project directory
cd my-trading-system

# Run the buy-and-hold example
qtrader backtest --file config/backtests/buy_hold.yaml

# Run the SMA crossover example
qtrader backtest --file config/backtests/sma_crossover.yaml

# See all available options
qtrader backtest --help
```

**Or use the helper script:**

```bash
# Alternative: use the included runner script
python run_backtest.py config/backtests/buy_hold.yaml
```

Results are saved to `output/backtests/{backtest_id}/` with performance metrics, equity curves, and trade history.

### Programmatic Usage

```python
from qtrader.engine import BacktestEngine
from qtrader.engine.config import BacktestConfig

# Load configuration
config = BacktestConfig.from_yaml("config/backtests/buy_hold.yaml")

# Run backtest
engine = BacktestEngine(config)
results = engine.run()

# View results
print(f"Final Portfolio Value: ${results.final_value:,.2f}")
print(f"Total Return: {results.total_return:.2%}")
```

## Indicator Library

QTrader includes 22 technical indicators across 5 categories:

- **Moving Averages** (7): SMA, EMA, WMA, DEMA, TEMA, HMA, SMMA
- **Momentum** (6): RSI, MACD, Stochastic, CCI, ROC, Williams %R
- **Volatility** (3): ATR, Bollinger Bands, Standard Deviation
- **Volume** (4): VWAP, OBV, A/D, CMF
- **Trend** (2): ADX, Aroon

All indicators support both stateful (streaming) and stateless (batch) computation modes.

For detailed documentation, formulas, parameters, and usage examples, see [Indicators Documentation](docs/packages/indicators/README.md).

## Project Structure

```
qtrader/
├── engine/         # Backtest orchestration
├── services/       # Core services (data, portfolio, execution, strategy)
├── events/         # Event system and event types
├── libraries/      # Indicators, strategies, risk policies
└── cli/            # Command-line interface
```

## Extending QTrader

QTrader is designed to be extended with custom strategies, indicators, data adapters, and risk policies. Use the scaffolding commands to generate template files with documentation and examples.

### Option 1: Generate Individual Components

Create individual custom components in your project:

```bash
# Navigate to your project
cd my-trading-system

# Generate templates for specific component types
qtrader init-library ./library --type strategy --type indicator

# This creates:
# library/strategies/template.py
# library/indicators/template.py
```

### Option 2: Generate Full Custom Library

Create a complete external library for reuse across projects:

```bash
# Create a full library structure
qtrader init-library ~/my-qtrader-extensions

# This creates:
# ~/my-qtrader-extensions/
# ├── strategies/
# ├── indicators/
# ├── adapters/
# ├── risk_policies/
# └── metrics/
```

### Configure Custom Libraries

After generating templates, configure `config/system.yaml` to use them:

```yaml
custom_libraries:
  strategies: "./library/strategies"      # Path to your strategies
  indicators: null                        # null = use built-in only
  adapters: null                          # null = use built-in only
  risk_policies: null                     # null = use built-in only
  metrics: null                           # null = use built-in only
```

**Important:** Set paths to `null` for components you don't customize. QTrader will use built-in components only.

### Example: Custom Strategy

The scaffolded templates show the structure. Here's a minimal example:

```python
# library/strategies/my_strategy.py
from qtrader.libraries.strategies import Strategy, StrategyConfig
from qtrader.services.strategy.models import SignalIntention

class MyStrategyConfig(StrategyConfig):
    name: str = "my_strategy"
    sma_period: int = 20
    # Add your parameters here

class MyStrategy(Strategy[MyStrategyConfig]):
    def on_bar(self, event, context):
        # Your trading logic
        sma = context.sma(symbol=event.symbol, period=self.config.sma_period)

        if event.close > sma and not context.has_position(event.symbol):
            context.emit_signal(
                symbol=event.symbol,
                intention=SignalIntention.OPEN_LONG,
                quantity=100
            )

CONFIG = MyStrategyConfig()  # Required for auto-discovery
```

**Reference:** See `src/qtrader/scaffold/library/strategies/sma_crossover.py` for a complete example with comments.

### Example: Custom Data Adapter

Integrate proprietary data sources:

```python
# library/adapters/my_adapter.py
from qtrader.services.data.adapters.protocol import IDataAdapter

class MyProprietaryAdapter(IDataAdapter):
    def read_bars(self, start_date: str, end_date: str):
        # Load from your proprietary database
        return self.db.query(...)

    def to_price_bar_event(self, bar) -> PriceBarEvent:
        # Convert to QTrader event format
        return PriceBarEvent(...)
```

**Reference:** See `src/qtrader/services/data/adapters/builtin/yahoo_csv.py` for a complete implementation.

## Development

### Run Tests

```bash
# All tests
make test

# With coverage
make test-coverage

# Quality checks
make qa
```

### Configuration

Backtests are configured via YAML files. See `config/` directory for examples.

## Documentation

Detailed documentation available in `docs/`:

- [Indicators Documentation](docs/packages/indicators/README.md) - Complete indicator reference
- [CLI Documentation](docs/packages/cli/backtest.md) - Command-line usage
- [Architecture Notes](docs/MULTI_STRATEGY_REFACTORING.md) - Design decisions

## Status

Active development. Core features implemented and tested.

## License

MIT License - see [LICENSE](LICENSE) file for details.
