# Multi-Mode Architecture Design Decision

**Date:** October 8, 2025 **Status:** Approved **Decision Maker:** Architecture Review

______________________________________________________________________

## Context

During migration planning review, a critical requirement emerged: **different processing stages need different price adjustment modes**.

### The Problem

Original single-mode plan had one mode selected at data layer:

```yaml
data:
  price_series_mode: "adjusted"  # ONE mode for ALL stages
```

This creates conflicts:

| Stage                 | Needs          | Why                                    |
| --------------------- | -------------- | -------------------------------------- |
| **Signal Generation** | `adjusted`     | Split-consistent indicators (SMA, RSI) |
| **Execution/Fills**   | `unadjusted`   | Actual market prices for commissions   |
| **Performance**       | `total_return` | Includes dividend reinvestment         |

**Example conflict**: Using adjusted prices for fills would calculate commissions on split-adjusted prices, not actual traded prices. Using unadjusted for signals would break technical indicators across stock splits.

______________________________________________________________________

## Decision

**Implement Multi-Mode Bar Architecture**: Each bar contains all three adjustment modes, components select appropriate mode.

### Architecture

```python
class MultiModeBar(BaseModel):
    """Bar with all adjustment modes."""
    symbol: str
    trade_datetime: str
    unadjusted: CanonicalBar      # Actual traded prices
    adjusted: CanonicalBar         # Split-adjusted
    total_return: CanonicalBar     # Split + dividend adjusted

class BacktestConfig:
    signal_generation_mode: str = "adjusted"      # For strategies
    execution_mode: str = "unadjusted"            # For fills
    performance_mode: str = "total_return"        # For metrics
```

### Data Flow

```
Raw Data → VendorAdapter → AlgoseekPriceSeries
  → to_canonical_series() → Dict[3 modes]
  → PriceSeriesIterator → MultiModeBar
  → Strategy (uses adjusted)
  → ExecutionEngine (uses unadjusted)
  → Portfolio (uses total_return)
```

______________________________________________________________________

## Rationale

### 1. Correctness per Stage

**Strategy (adjusted mode)**:

```python
def on_bar(self, bar: MultiModeBar, ctx: Context):
    strategy_bar = bar.adjusted

    # SMA calculation unaffected by stock splits
    sma_50 = self.calculate_sma(strategy_bar.close, 50)

    # Signal based on split-consistent prices
    if strategy_bar.close > sma_50:
        return [Signal.BUY]
```

- ✅ Technical indicators work correctly across splits
- ✅ Thresholds remain valid after splits
- ✅ Chart patterns consistent

**Execution (unadjusted mode)**:

```python
def evaluate_fill(self, bar: MultiModeBar, order: Order):
    exec_bar = bar.unadjusted

    # Fill at actual traded price
    fill_price = exec_bar.high

    # Commission based on real market price
    commission = fill_price * order.shares * 0.001

    # Actual cash requirement
    cash_needed = fill_price * order.shares + commission
```

- ✅ Commissions calculated on actual traded prices
- ✅ Slippage reflects real market conditions
- ✅ Volume participation uses actual volumes
- ✅ Cash requirements match reality

**Performance (total_return mode)**:

```python
def calculate_metrics(self, bar: MultiModeBar):
    perf_bar = bar.total_return

    # Return includes dividend reinvestment
    total_return = (perf_bar.close - self.entry_price) / self.entry_price

    # Accurate comparison to benchmarks
    alpha = total_return - benchmark_return
```

- ✅ Includes dividend reinvestment
- ✅ Accurate performance attribution
- ✅ Fair benchmark comparisons

### 2. Single Data Load

- Load once from data source
- Transform to 3 modes simultaneously
- Stream all 3 modes together
- No duplicate I/O or computation

### 3. Configuration Flexibility

```yaml
# Default configuration (optimal for most cases)
data:
  signal_generation_mode: "adjusted"
  execution_mode: "unadjusted"
  performance_mode: "total_return"

# Alternative: Use unadjusted for everything (debugging)
# data:
#   signal_generation_mode: "unadjusted"
#   execution_mode: "unadjusted"
#   performance_mode: "unadjusted"
```

______________________________________________________________________

## Alternatives Considered

### Alternative 1: Single Mode (Original Plan) ❌

```yaml
data:
  price_series_mode: "adjusted"  # ONE mode
```

**Pros**:

- Simpler architecture
- Lower memory (1x vs 3x bars)

**Cons**:

- ❌ Can't optimize per stage
- ❌ Wrong commissions (adjusted prices)
- ❌ Wrong performance (no dividends)
- ❌ Must choose between correct signals OR correct fills

**Verdict**: Rejected - Correctness is non-negotiable

### Alternative 2: Multi-Pass with Cache ❌

Run backtest multiple times with different modes:

- Pass 1 (adjusted): Generate signals, cache
- Pass 2 (unadjusted): Execute fills
- Pass 3 (total_return): Calculate performance

**Pros**:

- Lower memory (1x bars at a time)

**Cons**:

- ❌ Complex caching logic
- ❌ Multiple data loads
- ❌ Hard to debug
- ❌ Can't handle dynamic decisions

**Verdict**: Rejected - Too complex

### Alternative 3: Multi-Mode Bar (Selected) ✅

**Pros**:

- ✅ Correct prices for each stage
- ✅ Single data load
- ✅ Simple component interface
- ✅ Configurable per stage
- ✅ Easy to test

**Cons**:

- Memory: 3x bars (manageable for typical backtests)
- Complexity: Components select mode (minimal)

**Verdict**: Accepted

______________________________________________________________________

## Consequences

### Positive

1. **Correctness**: Each stage uses optimal price series
1. **Flexibility**: Configuration per component
1. **Single Load**: No duplicate data loading
1. **Testability**: Easy to verify each mode

### Negative

1. **Memory**: 3x bars in memory vs 1x

   - **Mitigation**: Streaming iterator (one timestamp at a time)
   - **Impact**: For 1 year daily data: ~3 MB vs 1 MB (negligible)

1. **Component Complexity**: Must select mode

   - **Mitigation**: Default modes in config
   - **Impact**: One line: `bar.adjusted` or `bar.get_bar(config.mode)`

______________________________________________________________________

## Implementation

### Phase 2: Iterator Infrastructure

1. Create `MultiModeBar` model
1. Update `PriceSeriesIterator` to yield `MultiModeBar`
1. Update configuration schema

### Phase 4: Backtest Engine

1. Update event loop to pass `MultiModeBar`
1. Each component selects appropriate mode

### Phase 5-6: Component Updates

1. Strategy: Use `bar.adjusted`
1. Execution: Use `bar.unadjusted`
1. Portfolio: Use `bar.total_return` for performance

______________________________________________________________________

## Examples

### Example 1: Stock Split Scenario

**AAPL 4:1 split on 2020-08-31**

```python
# Before split: $499.23
# After split: $129.04

# Strategy (adjusted mode)
strategy_bar = bar.adjusted
print(strategy_bar.close)  # $124.81 (pre-split adjusted to post-split)

# Execution (unadjusted mode)
exec_bar = bar.unadjusted
print(exec_bar.close)  # $499.23 (actual traded price)

# Commission calculation
commission = exec_bar.close * 100 * 0.001  # $49.92 (correct!)
# If using adjusted: 124.81 * 100 * 0.001 = $12.48 (WRONG!)
```

### Example 2: Dividend Scenario

**AAPL $0.82 dividend on 2020-08-07**

```python
# Strategy (adjusted mode)
strategy_bar = bar.adjusted
# Dividend recorded but prices not adjusted for it
# (split-adjusted only)

# Performance (total_return mode)
perf_bar = bar.total_return
# Prices adjusted for dividend reinvestment
# Accurate return calculation
```

______________________________________________________________________

## Validation

### Test Cases

1. ✅ Strategy generates same signals pre/post split (using adjusted)
1. ✅ Commissions match actual traded prices (using unadjusted)
1. ✅ Performance includes dividend returns (using total_return)
1. ✅ Memory usage acceptable (\<10 MB for 1 year daily)
1. ✅ Configuration per stage works

______________________________________________________________________

## Decision Summary

| Aspect          | Single Mode    | Multi-Mode           |
| --------------- | -------------- | -------------------- |
| **Correctness** | ❌ Compromised | ✅ Optimal per stage |
| **Memory**      | ✅ 1x          | ⚠️ 3x (manageable)   |
| **Flexibility** | ❌ Limited     | ✅ Configurable      |
| **Complexity**  | ✅ Simple      | ⚠️ Mode selection    |

**Final Decision**: **Multi-Mode Architecture** ✅

The correctness benefits outweigh the memory cost. Modern systems can easily handle 3x memory for typical backtests, and streaming keeps memory bounded.

______________________________________________________________________

## Approval

- [x] Architecture review completed
- [x] Performance impact assessed
- [x] Implementation plan updated
- [x] Migration plan reflects multi-mode design

**Status**: Approved for implementation

______________________________________________________________________

**References**:

- `docs/DATA_LAYER_MIGRATION_PLAN.md` - Full migration plan
- `docs/DATA_LAYER_MIGRATION_SUMMARY.md` - Executive summary
- Issue: "Signal generation vs execution price series requirements"
