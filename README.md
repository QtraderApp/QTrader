# QTrader

> Event-driven Python backtesting framework for quantitative trading strategies.

QTrader helps you design, test, and iterate on trading ideas using historical market data. It provides an extensible, typed, and composable environment focused on correctness, transparency, and reproducibility.

______________________________________________________________________

## Table of Contents

1. Intro & Philosophy
1. Architecture & Workflow (Event Driven)
1. User Guide (Getting Started)
1. CLI Reference (Essentials)
1. Extending (Strategies, Indicators, Libraries, Adapters)
1. Developer Guide (Source Layout, Testing, Quality, Principles)
1. Indicator Library Overview
1. Status & Roadmap
1. License

______________________________________________________________________

## 1. Intro & Philosophy

QTrader aims to:

- Separate concerns cleanly (data, strategy, execution, portfolio, reporting).
- Make strategy iteration fast: scaffold projects in seconds with `init-project`.
- Provide strong typing and validated configuration (pydantic) for safer refactors.
- Be transparent: everything is represented as explicit events you can inspect.
- Remain extensible: plug in custom data adapters, indicators, strategies, risk policies.

> 📦 This repository is the **package source**. End users create a *project* via `qtrader init-project`. The scaffolded project has its own structure (configs, strategies, data) distinct from this source tree.

______________________________________________________________________

## 2. Architecture & Workflow (Event Driven)

The engine processes a stream of domain events. Each service reacts deterministically and may emit new events. This enables fine-grained auditing and reproducibility.

### High-Level Components

- **Data Service**: Reads raw bars via adapters, emits `PriceBarEvent`.
- **Strategy Service**: Consumes market events, computes indicators, emits `SignalEvent` (intentions).
- **Execution Service**: Translates signals → orders, applies slippage/commission, emits `OrderEvent` / `FillEvent`.
- **Portfolio Service**: Updates positions, cash, P&L on fills.
- **Metrics/Reporting Service**: Aggregates performance and writes outputs.
- **Event Bus / Store**: Dispatches & persists ordered events for replay and inspection.

### Event Flow Diagram

```
           +------------------+
           |  Data Adapter(s) |  (CSV, API, Custom)
           +---------+--------+
                     |
                     v  PriceBarEvent
                +----+----+
                |  Bus    |  (dispatch)
                +----+----+
                     |
                     v
             +---------------+
             | Strategy Svc  |  (Indicators, context)
             +-------+-------+
                     |
             SignalEvent |
                     v
            +---------------+
            | Execution Svc |  (Routing, slippage, commission)
            +-------+-------+
                    |
             OrderEvent/FillEvent
                    v
            +---------------+
            | Portfolio Svc |  (Positions, cash, NAV)
            +-------+-------+
                    |
                 Metrics
                    v
            +---------------+
            | Reporting Svc |  (Results, artifacts)
            +---------------+
```

### Design Principles

- **Event Immutability**: Events are append-only; derived state is recomputable.
- **Typed Configuration**: Backtest & system configs validated at load time.
- **Deterministic Runs**: Same inputs → same sequence of events → same results.
- **Progressive Extension**: Start with defaults, selectively override pieces.

______________________________________________________________________

## 3. User Guide (Getting Started)

### Install

```bash
pip install qtrader
```

### Initialize a New Project

```bash
qtrader init-project my-trading-system
cd my-trading-system
```

Project scaffold (abbreviated):

```
my-trading-system/
├── config/
│   ├── system.yaml
│   ├── data_sources.yaml
│   └── backtests/
│       ├── buy_hold.yaml
│       └── sma_crossover.yaml
├── library/
│   └── strategies/
│       ├── buy_and_hold.py
│       └── sma_crossover.py
├── data/
│   └── us-equity-yahoo-csv/
│       └── AAPL.csv
├── output/
├── examples/
│   └── run_backtest.py
└── run_backtest.py
```

### Run a Backtest (CLI)

```bash
qtrader backtest config/backtests/buy_hold.yaml
qtrader backtest --file config/backtests/sma_crossover.yaml
qtrader backtest --help
```

Artifacts: `output/backtests/{backtest_id}/` (metrics, equity curve, trades, config snapshot).

### Programmatic API

```python
from qtrader.engine import BacktestEngine
from qtrader.engine.config import BacktestConfig

config = BacktestConfig.from_yaml("config/backtests/buy_hold.yaml")
engine = BacktestEngine(config)
results = engine.run()
print(results.final_value, results.total_return)
```

### Basic CLI Surface (Core Commands)

```bash
# Run a backtest
qtrader backtest --file config/backtests/sma_crossover.yaml

# Update Yahoo CSV data incrementally (auto symbol discovery)
qtrader data yahoo-update --days 365

# Generate component templates
qtrader init-library ./library --type strategy --type indicator

# Show data source names
qtrader data list
```

______________________________________________________________________

## 4. CLI Reference (Essentials)

| Command                                                      | Purpose                                      |
| ------------------------------------------------------------ | -------------------------------------------- |
| `qtrader init-project <path>`                                | Scaffold a new backtesting project           |
| `qtrader backtest --file <yaml>`                             | Run a configured backtest                    |
| `qtrader data yahoo-update [--days N] [--symbols AAPL MSFT]` | Download/refresh local Yahoo OHLCV CSVs      |
| `qtrader data list`                                          | List configured data adapters/sources        |
| `qtrader init-library <path> [--type ...]`                   | Generate template code for custom components |

Extended docs: `docs/packages/cli/backtest.md`.

______________________________________________________________________

## 5. Extending

### Strategies & Indicators

Use `qtrader init-library` to create template files then implement logic in `on_bar` / indicator calculate methods.

Minimal custom strategy example:

```python
from qtrader.libraries.strategies import Strategy, StrategyConfig
from qtrader.services.strategy.models import SignalIntention

class MyStrategyConfig(StrategyConfig):
    name: str = "my_strategy"
    sma_period: int = 20

class MyStrategy(Strategy[MyStrategyConfig]):
    def on_bar(self, event, context):
        sma = context.sma(symbol=event.symbol, period=self.config.sma_period)
        if event.close > sma and not context.has_position(event.symbol):
            context.emit_signal(symbol=event.symbol,
                                intention=SignalIntention.OPEN_LONG,
                                quantity=100)

CONFIG = MyStrategyConfig()  # Required for discovery
```

### Data Adapters

Implement the adapter protocol to load proprietary data and emit events.

```python
from qtrader.services.data.adapters.protocol import IDataAdapter

class MyAdapter(IDataAdapter):
    def read_bars(self, start_date: str, end_date: str):
        # return iterable of raw bar records
        ...
    def to_price_bar_event(self, bar):
        # convert to PriceBarEvent
        ...
```

### Custom Library Layout

```
my-qtrader-extensions/
├── strategies/
├── indicators/
├── adapters/
├── risk_policies/
└── metrics/
```

Configure paths in `config/system.yaml` (set to `null` for built-in only):

```yaml
custom_libraries:
  strategies: "./library/strategies"
  indicators: null
  adapters: null
  risk_policies: null
  metrics: null
```

______________________________________________________________________

## 6. Developer Guide

### Source Layout (Package Repository)

```
src/qtrader/
├── engine/      # Orchestration & backtest engine
├── services/    # data, strategy, execution, portfolio, reports
├── events/      # Event definitions & bus/store
├── libraries/   # Built-in indicators, strategies, risk policies
├── cli/         # Command-line interface
└── scaffold/    # Project & library templates distributed with package
```

### Quality & Tests

```bash
make qa            # Lint + format (ruff, isort, mdformat)
make test          # Run full test suite
make test-coverage # Run tests with coverage report
```

Current internal metrics (may differ in CI): ~1600+ tests, ~80% coverage.

### Principles

- Typed configs (pydantic) reduce runtime surprises.
- Indicators support streaming & batch modes.
- Services are cohesive; cross-service communication only via events.
- Deterministic sequencing enables replay & debugging.

### Contributing (Early Phase)

Open issues for feature proposals or architecture questions. Please include reproducible examples for bug reports (config YAML, sample data snippet, strategy code).

______________________________________________________________________

## 7. Indicator Library Overview

Categories & examples:

- **Moving Averages (7)**: SMA, EMA, WMA, DEMA, TEMA, HMA, SMMA
- **Momentum (6)**: RSI, MACD, Stochastic, CCI, ROC, Williams %R
- **Volatility (3)**: ATR, Bollinger Bands, StdDev
- **Volume (4)**: VWAP, OBV, Acc/Dist, CMF
- **Trend (2)**: ADX, Aroon

Indicators are accessible through the strategy context (e.g., `context.sma(...)`).

See: `docs/packages/indicators/README.md` for parameters & formulas.

______________________________________________________________________

## 8. Status & Roadmap

Status: Active development (beta). Core single-strategy backtesting stable.

Planned / Evaluating:

- Multi-strategy portfolio coordination.
- Streaming / live paper feed integration.
- Enhanced risk & order routing models.
- More built-in adapters (Parquet, SQL, APIs).
- Scenario & walk-forward utilities.

______________________________________________________________________

## 9. License

MIT License. See [LICENSE](LICENSE).

______________________________________________________________________

Enjoy building! If something feels unclear or friction-heavy, open an issue early—we iterate fast during beta.
