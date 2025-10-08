# Data Layer Migration: Before vs After

## Quick Comparison

### Bar Model

**BEFORE (OLD)**:

```python
class Bar(NamedTuple):
    ts: datetime
    symbol: str
    unadjusted: PriceSeries       # Nested NamedTuple
    capital_adjusted: PriceSeries
    total_return: PriceSeries
    dividend: Optional[Dividend]
    split: Optional[Split]

# Usage in strategy - hardcoded series selection
def on_bar(self, bar: Bar, ctx: Context):
    close = bar.capital_adjusted.close  # Must select series
```

**AFTER (NEW)**:

```python
class MultiModeBar(BaseModel):
    """Bar with all adjustment modes."""
    symbol: str
    trade_datetime: str
    unadjusted: CanonicalBar      # Actual traded prices
    adjusted: CanonicalBar         # Split-adjusted
    total_return: CanonicalBar     # Split + dividend adjusted

    def get_bar(self, mode: str) -> CanonicalBar:
        """Component selects appropriate mode."""

class CanonicalBar(BaseModel):
    """Single price series."""
    trade_datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    dividend: Optional[Decimal]

# Usage - each component selects optimal mode
def on_bar(self, bar: MultiModeBar, ctx: Context):
    # Strategy uses adjusted (split-consistent indicators)
    strategy_bar = bar.adjusted
    close = strategy_bar.close

# Execution uses unadjusted (realistic fills)
def evaluate_fill(self, bar: MultiModeBar, order: Order):
    exec_bar = bar.unadjusted
    fill_price = exec_bar.high  # Actual traded price!

# Performance uses total_return (includes dividends)
def calculate_return(self, bar: MultiModeBar):
    perf_bar = bar.total_return
    return_pct = (perf_bar.close - self.entry) / self.entry
```

______________________________________________________________________

### Data Loading

**BEFORE (OLD)**:

```python
# Adapter does everything
adapter = AlgoseekOHLCAdapter(config, instrument)
bars: List[Bar] = list(adapter.read_bars(config))

# Returns complex multi-series bars (500+ lines of logic)
for bar in bars:
    price = bar.capital_adjusted.close
```

**AFTER (NEW)**:

```python
# Step 1: Adapter returns vendor model
adapter = AlgoseekVendorAdapter(config)
vendor_bars = list(adapter.read_bars(symbol, start, end))

# Step 2: Transform to canonical (all 3 modes)
series = AlgoseekPriceSeries(symbol=symbol, bars=vendor_bars)
canonical = series.to_canonical_series()
# Returns: {
#   "unadjusted": CanonicalPriceSeries(...),
#   "adjusted": CanonicalPriceSeries(...),
#   "total_return": CanonicalPriceSeries(...)
# }

# Step 3: Create multi-mode iterator
iterator = PriceSeriesIterator(canonical)

# Step 4: Stream multi-mode bars
for multi_bar in iterator:
    # Each component selects its mode
    strategy_bar = multi_bar.adjusted
    exec_bar = multi_bar.unadjusted
    perf_bar = multi_bar.total_return
```

______________________________________________________________________

### Execution Engine

**BEFORE (OLD)**:

```python
def on_bar(self, bar: Bar, ...) -> List[Fill]:
    # Must select correct series
    high = bar.unadjusted.high  # For realistic fills
    low = bar.unadjusted.low
```

**AFTER (NEW)**:

```python
def on_bar(self, bar: MultiModeBar, ...) -> List[Fill]:
    # Select unadjusted mode for realistic fills
    exec_bar = bar.unadjusted
    high = exec_bar.high  # Actual traded price
    low = exec_bar.low

    # Calculate commission on actual price (not adjusted!)
    commission = exec_bar.close * order.shares * 0.001
```

**Why unadjusted for execution?**

- Commissions based on actual traded prices
- Realistic slippage estimates
- Accurate cash requirements

______________________________________________________________________

### Portfolio Valuation

**BEFORE (OLD)**:

```python
def update_bar(self, bar: Bar):
    # Must select series for valuation
    close = bar.capital_adjusted.close
```

**AFTER (NEW)**:

```python
def update_bar(self, bar: MultiModeBar):
    # Use unadjusted for current valuation
    valuation_bar = bar.unadjusted
    close = valuation_bar.close

def calculate_return(self, bar: MultiModeBar):
    # Use total_return for accurate performance
    perf_bar = bar.total_return
    return_pct = (perf_bar.close - self.entry) / self.entry
```

**Why different modes?**

- **Valuation**: `unadjusted` - Current market value at actual prices
- **Performance**: `total_return` - Includes dividend reinvestment for accurate returns

______________________________________________________________________

### Configuration

**BEFORE (OLD)**:

```yaml
# No configuration - series selection hardcoded in each component
execution:
  # Uses bar.unadjusted implicitly

portfolio:
  # Uses bar.capital_adjusted implicitly

strategy:
  # Uses bar.capital_adjusted implicitly
```

**AFTER (NEW)**:

```yaml
data:
  # Mode per component for optimal correctness
  signal_generation_mode: "adjusted"      # Strategy indicators (split-safe)
  execution_mode: "unadjusted"            # Realistic fills (actual prices)
  performance_mode: "total_return"        # Accurate returns (with dividends)

# Why different modes?
# - Strategy: adjusted prices for split-consistent technical indicators
# - Execution: unadjusted prices for realistic fills and commissions
# - Performance: total_return for accurate returns including dividends
```

______________________________________________________________________

## Key Benefits

### 1. Optimal Mode per Component

**Problem**: Different stages need different adjustment modes

**Solution**: Multi-mode architecture

| Component       | Mode           | Why                                 |
| --------------- | -------------- | ----------------------------------- |
| **Strategy**    | `adjusted`     | SMA, RSI work across stock splits   |
| **Execution**   | `unadjusted`   | Commissions on actual traded prices |
| **Performance** | `total_return` | Returns include dividends           |

**Example - AAPL 4:1 split (2020-08-31)**:

```python
# Before split: $499.23, After: $129.04

# Strategy uses adjusted
strategy_bar = multi_bar.adjusted
print(strategy_bar.close)  # $124.81 (pre-split adjusted to post-split)
# SMA calculation unaffected by split ✅

# Execution uses unadjusted
exec_bar = multi_bar.unadjusted
print(exec_bar.close)  # $499.23 (actual traded price before split)
commission = exec_bar.close * 100 * 0.001  # $49.92 (correct!)
# If using adjusted: 124.81 * 100 * 0.001 = $12.48 (WRONG!) ❌

# Performance uses total_return
perf_bar = multi_bar.total_return
# Includes dividend reinvestment for accurate returns ✅
```

### 2. Single Data Load

- Load once from data source
- Transform to all 3 modes simultaneously
- Stream all 3 modes together (MultiModeBar)
- No duplicate I/O or computation

### 3. Configuration-Driven

- Explicit mode per component in YAML
- No hardcoded series selection in code
- Easy to change mode for debugging/testing

### 4. Type Safety

- Pydantic models with validation
- Clear interfaces: `MultiModeBar` → `CanonicalBar`
- IDE autocomplete and type checking

______________________________________________________________________

## Benefits Summary

| Aspect               | Before                       | After                     | Improvement    |
| -------------------- | ---------------------------- | ------------------------- | -------------- |
| **Bar Model**        | Complex (7 fields, 3 nested) | Container + 3 Simple bars | Cleaner design |
| **Strategy Code**    | `bar.capital_adjusted.close` | `bar.adjusted.close`      | Explicit mode  |
| **Adapter Code**     | 500+ lines (mixed logic)     | ~100 lines (pure loading) | -80% LOC       |
| **Memory Usage**     | All 3 series loaded          | All 3 modes (streaming)   | Same (bounded) |
| **Mode Selection**   | Runtime (scattered)          | Config per component      | Flexible       |
| **Correctness**      | One mode for all stages      | Optimal mode per stage    | ✅ Better      |
| **Vendor Isolation** | Mixed with transformation    | Clean separation          | Better SoC     |

**Key Insight**: We accept 3x bars in memory because correctness requires different modes per component. Memory impact is minimal with streaming (one timestamp at a time).

______________________________________________________________________

## Migration Impact

### Files to Update: ~50 files

**Core (10 files)**:

- `src/qtrader/models/bar.py` → DELETE (old Bar)
- `src/qtrader/models/canonical_bar.py` → ADD MultiModeBar
- `src/qtrader/adapters/algoseek.py` → REFACTOR (returns AlgoseekBar)
- `src/qtrader/data/iterator.py` → UPDATE (yields MultiModeBar)
- `src/qtrader/api/backtest.py` → UPDATE (passes MultiModeBar)
- `src/qtrader/api/strategy.py` → UPDATE (receives MultiModeBar)
- `src/qtrader/execution/engine.py` → UPDATE (uses unadjusted mode)
- `src/qtrader/models/portfolio.py` → UPDATE (uses total_return for performance)
- `src/qtrader/models/position.py` → UPDATE (mode-aware valuation)
- `config/qtrader.yaml` → UPDATE (add mode per component)

**Tests (~40 files)**:

- All unit tests (400+ tests)
- All integration tests (70+ tests)
- Golden tests

### Lines of Code Changes: ~5,000 LOC

- **Deleted**: ~1,500 LOC (old Bar model, complex adapter logic)
- **Added**: ~1,500 LOC (MultiModeBar, iterator infrastructure, mode selection)
- **Modified**: ~2,500 LOC (test updates, strategy updates, component mode selection)
- **Net**: +500 LOC (but significantly better architecture and correctness)

**Trade-off**: Slightly more code for significantly better correctness and flexibility.

______________________________________________________________________

## Timeline: 17 Days

| Week       | Focus             | Deliverable                                 |
| ---------- | ----------------- | ------------------------------------------- |
| **Week 1** | Infrastructure    | Iterator, adapters, backtest engine         |
| **Week 2** | Execution & Tests | Execution engine, portfolio, test migration |
| **Week 3** | Polish            | Documentation, cleanup, validation          |

______________________________________________________________________

## Key Validation

### Data Correctness ✅

- **Phase 1 Complete**: Data layer validated
- **Golden Output**: $0.82 AAPL dividend (100% accurate)
- **Test Coverage**: 13 unit + 6 integration tests passing
- **Formula Verified**: Against official AAPL data

### Architecture ✅

- **Vendor Isolation**: AlgoseekBar → CanonicalBar
- **Single Responsibility**: Adapter loads, PriceSeries transforms
- **Iterator Pattern**: Memory efficient streaming (MultiModeBar)
- **Multi-Mode**: Each bar contains all 3 adjustment modes
- **Configuration**: Mode selection per component (signal/execution/performance)

______________________________________________________________________

## Questions?

**See full docs**:

- `docs/MULTI_MODE_ARCHITECTURE_DECISION.md` - Multi-mode design rationale
- `docs/DATA_LAYER_MIGRATION_PLAN.md` - Complete implementation plan
- `docs/DATA_LAYER_MIGRATION_SUMMARY.md` - Executive summary
- `DATA_LAYER_MIGRATION.md` - Phase 1 completion report
- `DIVIDEND_CALCULATION_BUG_REPORT.md` - Data layer validation
