# Library Registry Design

## Philosophy: Registry-Based Component Loading

QTrader uses a **registry approach** for loading strategies, indicators, metrics, and risk policies. This enables:

1. **Clean configuration** - Reference components by name, not file paths
1. **Built-in + Custom** - System provides built-ins, users add customs
1. **Automatic discovery** - System scans library paths and registers all valid implementations
1. **ABC contracts** - All components implement abstract base classes

______________________________________________________________________

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        System Startup                            │
│                                                                   │
│  1. Load system.yaml (custom_libraries paths)                   │
│  2. Scan buildin/ directories → register built-in components    │
│  3. Scan custom_libraries/ paths → register custom components   │
│  4. Create registries: StrategyRegistry, IndicatorRegistry, etc. │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Backtest Configuration                      │
│                                                                   │
│  strategies:                                                     │
│    - strategy_id: "buy_and_hold"  # ← Registry name             │
│      universe: ["AAPL"]                                          │
│      data_sources: ["algoseek-us-equity-1d-unadjusted"]         │
│      config:                                                     │
│        warmup_bars: 0                                            │
│                                                                   │
│  risk_policy:                                                    │
│    name: "naive"  # ← Registry name                             │
│    config:                                                       │
│      max_pct_position_size: 0.90                                 │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Runtime Lookup                              │
│                                                                   │
│  StrategyService:                                                │
│    strategy = StrategyRegistry.get("buy_and_hold")              │
│    # Returns: BuyAndHoldStrategy instance                        │
│                                                                   │
│  RiskService (Portfolio Manager):                                │
│    policy = RiskPolicyRegistry.get("naive")                     │
│    # Returns: NaiveRiskPolicy instance                           │
└─────────────────────────────────────────────────────────────────┘
```

______________________________________________________________________

## Component Types & Responsibilities

### 1. Strategies

**Responsibility:** Generate trading signals based on market data

**Where:**

- Built-in: `src/qtrader/libraries/strategies/buildin/`
- Custom: Path in `system.yaml → custom_libraries.strategies`

**ABC Contract:**

```python
class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        """Process new bar and generate signals."""
        pass

    @abstractmethod
    def warmup_bars_required(self) -> int:
        """Number of bars needed before strategy starts trading."""
        pass
```

**Strategy DOES:**

- Receive market data (PriceBarEvent)
- Use indicators for analysis
- Generate signals (`context.emit_signal(...)`)
- Choose which data sources to consume

**Strategy DOES NOT:**

- Know about portfolio state (that's PortfolioService)
- Know about risk limits (that's RiskService/Portfolio Manager)
- Execute orders (that's ExecutionService)

**Config Example:**

```yaml
strategies:
  - strategy_id: "sma_crossover"  # Registry name
    universe: ["AAPL", "MSFT"]  # Symbols to trade
    data_sources: ["algoseek-us-equity-1d-adjusted"]  # Data feeds
    config:  # Strategy-specific overrides
      fast_period: 10
      slow_period: 20
      warmup_bars: 21  # Strategy declares its own warmup
```

______________________________________________________________________

### 2. Indicators

**Responsibility:** Calculate technical analysis values from price history

**Where:**

- Built-in: `src/qtrader/libraries/indicators/buildin/`
- Custom: Path in `system.yaml → custom_libraries.indicators`

**ABC Contract:**

```python
class BaseIndicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """Calculate indicator from historical data."""
        pass

    @abstractmethod
    def update(self, bar: Bar) -> float:
        """Update indicator with new bar."""
        pass
```

**Usage:**

```python
# In strategy
from qtrader.libraries.indicators.buildin import SMA

class MyStrategy(BaseStrategy):
    def __init__(self):
        self.sma = SMA(period=20)

    def on_bar(self, event, context):
        bars = context.get_bars(event.symbol, 20)
        sma_value = self.sma.calculate(bars)
```

______________________________________________________________________

### 3. Risk Policies

**Responsibility:** Decide position sizing and risk limits at portfolio level

**Where:**

- Built-in: `src/qtrader/libraries/risk_policies/buildin/`
- Custom: Path in `system.yaml → custom_libraries.risk_policies`

**ABC Contract:**

```python
class BaseRiskPolicy(ABC):
    @abstractmethod
    def evaluate_signal(
        self,
        signal: SignalEvent,
        portfolio_state: PortfolioState
    ) -> OrderDecision:
        """Evaluate signal against portfolio state and risk limits."""
        pass

    @abstractmethod
    def calculate_position_size(
        self,
        signal: SignalEvent,
        portfolio_state: PortfolioState,
        price: Decimal
    ) -> int:
        """Calculate position size based on risk parameters."""
        pass
```

**Important:** Risk policies operate at **Portfolio Manager level**, not strategy level.

**Flow:**

```
Strategy → SignalEvent("BUY AAPL, confidence=100%")
          ↓
Portfolio Manager (RiskService):
  1. Check current portfolio state
  2. Apply risk policy (e.g., "naive")
  3. Calculate position size (respecting max_pct_position_size)
  4. Check risk limits (drawdown, concentration)
  5. Emit OrderEvent or reject signal
          ↓
ExecutionService → Fill order
```

**Config Example:**

```yaml
risk_policy:
  name: "naive"  # Registry name
  config:  # Policy overrides
    max_pct_position_size: 0.90  # Max 90% of equity per position
    max_drawdown: 0.25  # Max 25% drawdown
```

______________________________________________________________________

### 4. Metrics (Performance)

**Responsibility:** Calculate performance statistics from backtest results

**Where:**

- Built-in: `src/qtrader/libraries/performance/buildin/`
- Custom: Path in `system.yaml → custom_libraries.metrics`

**ABC Contract:**

```python
class BaseMetric(ABC):
    @abstractmethod
    def compute(self, results: BacktestResult) -> float:
        """Compute metric from backtest results."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name for reporting."""
        pass
```

**Usage:**

```python
# ReportingService (future)
metrics = MetricRegistry.get_all()
for metric in metrics:
    value = metric.compute(backtest_result)
    print(f"{metric.name}: {value}")
```

______________________________________________________________________

## Configuration Structure

### System Configuration (system.yaml)

**Purpose:** Define WHERE to find custom libraries

```yaml
# System-wide settings
custom_libraries:
  risk_policies: "path/to/custom/risk/policies"
  indicators: "path/to/custom/indicators"
  strategies: "path/to/custom/strategies"
  metrics: "path/to/custom/metrics"

# Service configurations
execution:
  fill_policy: "conservative"
  commission:
    model: "per_share"
    per_share: 0.0005
    minimum: 1.00

portfolio:
  lot_method_long: "fifo"
  lot_method_short: "lifo"

data:
  sources_config: "config/data_sources.yaml"
  default_mode: "adjusted"
  default_timezone: "America/New_York"
```

______________________________________________________________________

### Backtest Configuration (e.g., test_apple_run.yaml)

**Purpose:** Define WHAT to run (dates, universe, strategies)

```yaml
# Backtest parameters
start_date: "2019-01-02"
end_date: "2021-12-31"
initial_equity: 100_000
universe: ["AAPL", "MSFT", "GOOGL"]  # DataService loads all
replay_speed: 1.0  # 1 sec/bar for visualization

# Data selection
data:
  dataset: "algoseek-us-equity-1d-unadjusted"

# Strategies (reference by registry name)
strategies:
  - strategy_id: "buy_and_hold"  # Registry name (not file path!)
    universe: ["AAPL"]  # Strategy trades subset of backtest universe
    data_sources: ["algoseek-us-equity-1d-unadjusted"]  # Which data feeds
    config:  # Strategy-specific config
      hold_period_days: null  # null = forever
      warmup_bars: 0

  - strategy_id: "sma_crossover"
    universe: ["MSFT", "GOOGL"]
    data_sources: ["algoseek-us-equity-1d-unadjusted"]
    config:
      fast_period: 10
      slow_period: 20
      warmup_bars: 21  # This strategy needs warmup

# Risk policy (Portfolio Manager level)
risk_policy:
  name: "naive"  # Registry name
  config:
    max_pct_position_size: 0.90
    max_drawdown: 0.25
```

______________________________________________________________________

### Strategy Configuration (inside strategy file)

**Purpose:** Default strategy parameters (can be overridden in backtest YAML)

```python
# examples/strategies/buy_and_hold.py

from qtrader.libraries.strategies.base import BaseStrategy

class BuyAndHoldConfig:
    """Default configuration for Buy and Hold strategy."""
    hold_period_days: int | None = None  # null = forever
    warmup_bars: int = 0

class BuyAndHoldStrategy(BaseStrategy):
    """Simple buy and hold strategy."""

    def __init__(self, config: BuyAndHoldConfig):
        self.config = config

    def warmup_bars_required(self) -> int:
        return self.config.warmup_bars

    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        # Check if we already have position
        position = context.get_position(event.symbol)
        if position is None or position.quantity == 0:
            # Buy signal with 100% confidence
            context.emit_signal(
                symbol=event.symbol,
                direction="BUY",
                confidence=1.0
            )
```

______________________________________________________________________

## Key Design Decisions

### A. Universe at Two Levels ✅

**Backtest Universe:** Symbols DataService loads from all configured data sources

```yaml
universe: ["AAPL", "MSFT", "GOOGL", "TSLA"]  # Load ALL these
```

**Strategy Universe:** Symbols each strategy trades (subset of backtest universe)

```yaml
strategies:
  - strategy_id: "momentum"
    universe: ["AAPL", "MSFT"]  # Trades only these
  - strategy_id: "mean_reversion"
    universe: ["GOOGL", "TSLA"]  # Trades only these
```

**Validation:** `strategy.universe ⊆ backtest.universe` (enforced at config load)

______________________________________________________________________

### B. Registry Approach ✅

**Why Registry?**

- Clean configs: `strategy_id: "buy_and_hold"` not `path: ./strategies/buy_and_hold.py`
- Automatic discovery: System scans paths and registers all valid implementations
- Built-in + Custom: Both treated uniformly
- Type safety: Registry validates ABC compliance

**How It Works:**

1. System startup: Scan `buildin/` and `custom_libraries/` paths
1. For each file: Import, check if implements ABC, register by name
1. Runtime: Lookup by name from registry
1. Config validation: Ensure referenced names exist in registry

______________________________________________________________________

### C. Strategy Chooses Data Sources ✅

**System loads, strategy chooses:**

```yaml
# System config: Define available data sources
data:
  sources_config: "config/data_sources.yaml"

# Backtest config: Load specific datasets
universe: ["AAPL"]  # Load from configured sources

# Strategy config: Choose which feeds to consume
strategies:
  - strategy_id: "news_momentum"
    universe: ["AAPL"]
    data_sources:
      - "algoseek-us-equity-1d-adjusted"  # Price data
      - "news-sentiment-feed"  # News data
    config:
      sentiment_threshold: 0.7
```

**Use Case:** Strategy combines multiple data feeds (prices + news) to generate signals.

______________________________________________________________________

### D. Risk at Portfolio Level ✅

**Strategy:** Generates signals ("BUY AAPL, confidence=100%")

**Portfolio Manager (RiskService):** Applies risk policy

- Checks current portfolio state
- Applies position sizing rules
- Enforces risk limits (drawdown, concentration)
- Decides whether to emit OrderEvent

**Flow:**

```
Strategy: "I think we should buy AAPL"
         ↓ SignalEvent
Portfolio Manager: "Let me check..."
  - Current equity: $100,000
  - Max position: 90% → $90,000
  - Price: $150
  - Max shares: 600
  - Decision: Buy 600 shares
         ↓ OrderEvent
ExecutionService: Fill order
```

______________________________________________________________________

### E. Warmup at Strategy Level ✅

**Why?** Different strategies need different warmup periods:

- Buy and Hold: 0 bars (no indicator warmup)
- SMA Crossover (10, 20): 21 bars
- RSI (14): ~28 bars

**How:**

```yaml
strategies:
  - strategy_id: "buy_and_hold"
    config:
      warmup_bars: 0

  - strategy_id: "sma_crossover"
    config:
      warmup_bars: 21  # Needs 20 bars for slow SMA + 1
```

**Implementation:**

```python
class BaseStrategy(ABC):
    @abstractmethod
    def warmup_bars_required(self) -> int:
        """Number of bars needed before trading."""
        pass
```

**Engine Behavior:**

- Loads warmup data for each strategy
- During warmup: Indicators calculate but no signals emitted
- After warmup: Strategy starts generating signals

______________________________________________________________________

### F. Replay Speed ✅

**Already Implemented:**

```yaml
replay_speed: 1.0  # 1 second per bar
replay_speed: 0.0  # Full speed (default)
```

**Usage:**

```python
# In BacktestEngine
if self.config.replay_speed > 0:
    time.sleep(self.config.replay_speed)
```

**Purpose:** Slow down event loop for visualization/debugging (view logs at human-readable pace).

______________________________________________________________________

## Implementation Roadmap

### Phase 1: ABC Contracts (Week 1)

1. Define `BaseStrategy` ABC
1. Define `BaseIndicator` ABC
1. Define `BaseRiskPolicy` ABC
1. Define `BaseMetric` ABC

### Phase 2: Registry System (Week 1-2)

1. Create `Registry` base class
1. Implement `StrategyRegistry`
1. Implement `IndicatorRegistry`
1. Implement `RiskPolicyRegistry`
1. Implement `MetricRegistry`
1. Add system startup: scan paths, register components

### Phase 3: Built-in Implementations (Week 2-3)

1. Indicators: SMA, EMA, RSI, MACD, Bollinger Bands
1. Risk Policies: Naive, Fixed Fraction, Vol Target
1. Strategies: Buy and Hold, SMA Crossover (examples)
1. Metrics: Sharpe, Drawdown, Sortino, Win Rate

### Phase 4: Integration (Week 3-4)

1. Update StrategyService to use StrategyRegistry
1. Update RiskService to use RiskPolicyRegistry
1. Add ReportingService using MetricRegistry
1. Update config loaders to validate registry names
1. Add strategy universe validation

______________________________________________________________________

## Examples

### Example 1: Simple Backtest (Buy and Hold)

```yaml
# test_apple_run.yaml
start_date: "2019-01-02"
end_date: "2021-12-31"
initial_equity: 100_000
universe: ["AAPL"]

data:
  dataset: "algoseek-us-equity-1d-unadjusted"

strategies:
  - strategy_id: "buy_and_hold"
    universe: ["AAPL"]
    data_sources: ["algoseek-us-equity-1d-unadjusted"]
    config:
      warmup_bars: 0

risk_policy:
  name: "naive"
  config:
    max_pct_position_size: 0.90
```

### Example 2: Multi-Strategy Backtest

```yaml
# multi_strategy.yaml
start_date: "2020-01-01"
end_date: "2023-12-31"
initial_equity: 1_000_000
universe: ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]

data:
  dataset: "schwab-us-equity-1d-adjusted"

strategies:
  - strategy_id: "momentum"
    universe: ["AAPL", "MSFT", "GOOGL"]  # Tech stocks
    data_sources: ["schwab-us-equity-1d-adjusted"]
    config:
      lookback: 20
      warmup_bars: 21

  - strategy_id: "mean_reversion"
    universe: ["TSLA", "NVDA"]  # Volatile stocks
    data_sources: ["schwab-us-equity-1d-adjusted"]
    config:
      z_score_threshold: 2.0
      lookback: 50
      warmup_bars: 51

risk_policy:
  name: "vol_target"
  config:
    target_volatility: 0.15  # 15% annualized
    max_pct_position_size: 0.20  # Max 20% per position
```

### Example 3: Custom Strategy with Multiple Data Sources

```yaml
# news_momentum.yaml
start_date: "2023-01-01"
end_date: "2023-12-31"
initial_equity: 500_000
universe: ["AAPL", "MSFT"]

data:
  dataset: "algoseek-us-equity-1d-adjusted"

strategies:
  - strategy_id: "news_momentum"  # Custom strategy
    universe: ["AAPL", "MSFT"]
    data_sources:
      - "algoseek-us-equity-1d-adjusted"  # Price data
      - "news-sentiment-feed"  # Custom news source
    config:
      price_momentum_period: 20
      sentiment_threshold: 0.7
      warmup_bars: 21

risk_policy:
  name: "kelly"  # Custom risk policy
  config:
    kelly_fraction: 0.5  # Half-Kelly
    max_leverage: 1.0
```

______________________________________________________________________

## Summary

✅ **Registry approach** for clean component loading\
✅ **Two-level universe** (backtest-level + strategy-level)\
✅ **Strategy chooses data sources** from system-loaded feeds\
✅ **Risk at portfolio level** (Portfolio Manager applies policies)\
✅ **Warmup at strategy level** (each strategy declares needs)\
✅ **Replay speed** already implemented for visualization\
✅ **Clean config structure** (system.yaml + backtest YAML + strategy file)

**Next Steps:**

1. Implement ABC contracts for all component types
1. Build registry system with automatic discovery
1. Create built-in implementations (indicators, risk policies, metrics)
1. Update services to use registries
1. Add validation (strategy.universe ⊆ backtest.universe, registry name checks)
