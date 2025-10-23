# Configuration Alignment - Decisions Summary

**Date:** 2025-10-23\
**Context:** Architecture alignment phase, defining config structure before running engine

______________________________________________________________________

## Key Decisions Made

### A. Universe Structure ✅

**Decision:** Simple list (not double-nested)

```yaml
# ✅ CORRECT
universe: ["AAPL", "MSFT", "GOOGL"]

# ❌ WRONG (was initial proposal)
universe: [["AAPL"], ["MSFT"]]
```

**Rationale:** Simpler, cleaner. No need for grouping at universe level.

______________________________________________________________________

### B. Registry Approach for Libraries ✅

**Decision:** Reference components by name, not file path

**Before (rejected):**

```yaml
strategies:
  - path: ./buy_and_hold.py  # ❌ File path approach
    strategy_id: buy_and_hold
```

**After (agreed):**

```yaml
strategies:
  - strategy_id: "buy_and_hold"  # ✅ Registry name
    universe: ["AAPL"]
    data_sources: ["algoseek-us-equity-1d-unadjusted"]
```

**How Registry Works:**

1. System startup: Scan `custom_libraries` paths from `system.yaml`
1. Register all valid implementations (check ABC compliance)
1. Runtime: Lookup by name from registry
1. Applies to: strategies, indicators, metrics, risk policies

**System Config:**

```yaml
custom_libraries:
  strategies: "path/to/custom/strategies"  # System scans this
  indicators: "path/to/custom/indicators"
  risk_policies: "path/to/custom/risk/policies"
  metrics: "path/to/custom/metrics"
```

______________________________________________________________________

### C. Strategy Data Source Selection ✅

**Decision:** System loads data, strategy chooses which feeds to consume

**Architecture:**

```
DataService: Loads ALL data sources configured in data_sources.yaml
            ↓
Strategy: Chooses which feeds to consume from available sources
```

**Use Case Example:**

```yaml
# Strategy using multiple data sources
strategies:
  - strategy_id: "news_momentum"
    universe: ["AAPL"]
    data_sources:
      - "algoseek-us-equity-1d-adjusted"  # Price data
      - "news-sentiment-feed"              # News data
    config:
      sentiment_threshold: 0.7
```

**Rationale:**

- Flexibility: Different strategies can use different data combinations
- Reusability: Same data loaded once, used by multiple strategies
- Clear intent: Strategy declares what data it needs

______________________________________________________________________

### D. Risk Policies at Portfolio Level ✅

**Decision:** Risk policies operate at Portfolio Manager (RiskService) level, NOT strategy level

**Flow:**

```
Strategy: Generates signal ("BUY AAPL, confidence=100%")
         ↓
Portfolio Manager (RiskService):
  1. Check current portfolio state
  2. Load risk_policy from registry
  3. Calculate position size
  4. Check risk limits (drawdown, concentration)
  5. Emit OrderEvent or reject
         ↓
ExecutionService: Fill order
```

**Config:**

```yaml
# Portfolio Manager level (in backtest YAML)
risk_policy:
  name: "naive"  # Registry name
  config:
    max_pct_position_size: 0.90
    max_drawdown: 0.25
```

**Strategy DOES:**

- Generate signals with confidence level
- Indicate "I think we should buy/sell"

**Strategy DOES NOT:**

- Know current portfolio state
- Calculate position sizes
- Apply risk limits

**Portfolio Manager DOES:**

- Apply risk policy to signals
- Calculate actual position sizes
- Enforce risk limits
- Decide whether to trade

______________________________________________________________________

### E. System Config Structure ✅

**Decision:** Clean dictionary format for `custom_libraries`

**Before (rejected):**

```yaml
custom_libraries:
  - custom_risk_library_path: "path"     # ❌ Array of single-key dicts
  - custom_indicators_library_path: "path"
```

**After (agreed):**

```yaml
custom_libraries:
  risk_policies: "path/to/custom/risk/policies"  # ✅ Clean dict
  indicators: "path/to/custom/indicators"
  strategies: "path/to/custom/strategies"
  metrics: "path/to/custom/metrics"
```

**Rationale:** Cleaner, more readable, standard YAML practice

______________________________________________________________________

### F. Warmup at Strategy Level ✅

**Decision:** Each strategy declares its own `warmup_bars` requirement

**Why?** Different strategies need different warmup:

- Buy and Hold: 0 bars (no indicators)
- SMA Crossover (10, 20): 21 bars
- RSI (14): ~28 bars

**Config:**

```yaml
strategies:
  - strategy_id: "buy_and_hold"
    config:
      warmup_bars: 0  # No warmup needed

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

- During warmup: Load data, calculate indicators, NO signals
- After warmup: Strategy starts generating signals

**Rationale:** Strategy knows what indicators it uses, so it knows warmup needs

______________________________________________________________________

### G. Replay Speed ✅

**Status:** Already implemented ✅

**Config:**

```yaml
replay_speed: 1.0  # 1 second per bar (for visualization)
replay_speed: 0.0  # Full speed (default, for production backtests)
```

**Implementation:**

```python
# In BacktestEngine
if self.config.replay_speed > 0:
    time.sleep(self.config.replay_speed)
```

**Purpose:** Slow down event loop to view logs at human-readable pace during development

______________________________________________________________________

## Three Configuration Levels

### 1. System Configuration (system.yaml)

**Purpose:** HOW the system operates (fixed across backtests)

**Contains:**

- Service configurations (execution, portfolio, data)
- Commission/slippage models
- Custom library paths
- Logging settings

**Example:**

```yaml
custom_libraries:
  strategies: "path/to/custom/strategies"
  risk_policies: "path/to/custom/risk/policies"

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
```

______________________________________________________________________

### 2. Backtest Configuration (e.g., test_apple_run.yaml)

**Purpose:** WHAT to test (varies per backtest)

**Contains:**

- Dates, universe, equity
- Strategy selections (from registry)
- Risk policy selection (from registry)
- Dataset selection

**Example:**

```yaml
start_date: "2019-01-02"
end_date: "2021-12-31"
initial_equity: 100_000
universe: ["AAPL", "MSFT"]

data:
  dataset: "algoseek-us-equity-1d-unadjusted"

strategies:
  - strategy_id: "buy_and_hold"  # Registry name
    universe: ["AAPL"]
    data_sources: ["algoseek-us-equity-1d-unadjusted"]
    config:
      warmup_bars: 0

risk_policy:
  name: "naive"
  config:
    max_pct_position_size: 0.90
```

______________________________________________________________________

### 3. Strategy Configuration (inside strategy .py file)

**Purpose:** Default strategy parameters (can be overridden in backtest YAML)

**Contains:**

- Strategy-specific parameters
- Default warmup requirements
- Indicator settings

**Example:**

```python
# examples/strategies/buy_and_hold.py

class BuyAndHoldConfig:
    """Default configuration."""
    hold_period_days: int | None = None  # null = forever
    warmup_bars: int = 0

class BuyAndHoldStrategy(BaseStrategy):
    def __init__(self, config: BuyAndHoldConfig):
        self.config = config

    def warmup_bars_required(self) -> int:
        return self.config.warmup_bars
```

**Override in backtest YAML:**

```yaml
strategies:
  - strategy_id: "buy_and_hold"
    config:
      hold_period_days: 30  # Override: hold for 30 days, then sell
      warmup_bars: 0
```

______________________________________________________________________

## Updated Pydantic Models

### StrategyConfigItem

```python
class StrategyConfigItem(BaseModel):
    """Configuration for a single strategy."""

    strategy_id: str  # Registry name (not file path!)
    universe: list[str]  # Symbols to trade (subset of backtest)
    data_sources: list[str]  # Data feeds to consume
    config: dict[str, Any]  # Strategy-specific overrides
```

### RiskPolicyConfig

```python
class RiskPolicyConfig(BaseModel):
    """Risk policy configuration."""

    name: str  # Registry name
    config: dict[str, Any]  # Policy-specific overrides
```

### BacktestConfig

```python
class BacktestConfig(BaseModel):
    """Backtest run configuration."""

    # Backtest parameters
    start_date: datetime
    end_date: datetime
    initial_equity: Decimal
    universe: list[str]  # Symbols to load
    replay_speed: float = 0.0

    # Data, strategies, risk
    data: DataConfig
    strategies: list[StrategyConfigItem]
    risk_policy: RiskPolicyConfig
```

______________________________________________________________________

## Validation Rules

### 1. Strategy Universe Validation

```python
# Enforce: strategy.universe ⊆ backtest.universe
for strategy in config.strategies:
    if not set(strategy.universe).issubset(set(config.universe)):
        raise ValueError(
            f"Strategy {strategy.strategy_id} universe {strategy.universe} "
            f"not subset of backtest universe {config.universe}"
        )
```

### 2. Registry Name Validation

```python
# Ensure strategy_id exists in registry
if not StrategyRegistry.exists(strategy.strategy_id):
    raise ValueError(f"Strategy '{strategy.strategy_id}' not found in registry")

# Ensure risk_policy exists in registry
if not RiskPolicyRegistry.exists(config.risk_policy.name):
    raise ValueError(f"Risk policy '{config.risk_policy.name}' not found in registry")
```

### 3. Data Source Validation

```python
# Ensure data sources configured in system
for source in strategy.data_sources:
    if not DataSourceRegistry.exists(source):
        raise ValueError(f"Data source '{source}' not configured in system")
```

______________________________________________________________________

## Next Steps

### Immediate (Before Running Engine)

1. ✅ Config models updated (`StrategyConfigItem`, `RiskPolicyConfig`, `BacktestConfig`)
1. ✅ Config files updated (`system.yaml`, `test_apple_run.yaml`)
1. ✅ Example strategy documented (`buy_and_hold.py`)

### Phase 5.5: Library ABCs & Registry (1-2 weeks)

1. Define ABC contracts (`BaseStrategy`, `BaseIndicator`, `BaseRiskPolicy`, `BaseMetric`)
1. Implement registry system with automatic discovery
1. Create built-in implementations
1. Update services to use registries
1. Add validation logic

### Testing

1. Run engine with DataService (test with current setup)
1. Implement registry system
1. Run full backtest with strategies and risk policies

______________________________________________________________________

## Files Modified

### Updated

- ✅ `config/system.yaml` - Clean `custom_libraries` format
- ✅ `config/test_apple_run.yaml` - Registry approach, clean structure
- ✅ `src/qtrader/engine/config.py` - New models (`RiskPolicyConfig`, updated `StrategyConfigItem`)
- ✅ `examples/strategies/buy_and_hold.py` - Proper example implementation

### Created

- ✅ `docs/LIBRARY_REGISTRY_DESIGN.md` - Comprehensive design document
- ✅ `docs/CONFIG_ALIGNMENT_DECISIONS.md` - This document

______________________________________________________________________

## Philosophy Summary

**"Registry approach for clean, extensible architecture"**

- **Built-in + Custom**: Both treated uniformly through registries
- **Reference by name**: `strategy_id: "buy_and_hold"` not `path: ./file.py`
- **ABC contracts**: All components implement abstract base classes
- **Automatic discovery**: System scans paths, registers valid implementations
- **Clean separation**: System config (HOW) vs Backtest config (WHAT) vs Strategy config (DEFAULTS)
- **Two-level universe**: Backtest loads data, strategy trades subset
- **Portfolio-level risk**: Strategy signals, Portfolio Manager applies risk
- **Strategy-level warmup**: Each strategy declares its needs
- **Multi-feed strategies**: Strategy chooses from available data sources

✅ **Ready to proceed with implementation**
