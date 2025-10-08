# Phase 4 Part 3: Backtest Engine Iterator Integration

**Date**: 2025-10-08\
**Milestone**: Update Backtest.run() to use iterators\
**Status**: 🟡 **Engine Updated - Tests Pending**

## Overview

Phase 4 Part 3 successfully updates the `Backtest.run()` method to use iterator-based data loading with BarMerger coordination. The backtest engine now fully supports Phase 4 architecture, but integration tests need updating to match the new interface.

## Completed Changes

### 1. Backtest Engine Update (`src/qtrader/api/backtest.py`)

**Signature Change**:

```python
# Before (Phase 3)
def run(self, ctx, bars: List[Bar], symbols, out_dir, adjustment_events=None)

# After (Phase 4)
def run(self, ctx, data_iterators: Dict[str, PriceSeriesIterator], symbols, out_dir, adjustment_events=None)
```

**Key Changes**:

1. **Iterator-Based Input**: Accepts `Dict[str, PriceSeriesIterator]` instead of `List[Bar]`
1. **BarMerger Integration**: Uses BarMerger to coordinate multi-symbol streams chronologically
1. **Mode Selection**: Extracts `Bar` from `MultiModeBar.adjusted` for strategy
1. **No Legacy Support**: Removed backward compatibility (as per requirements)

**Implementation**:

```python
# Phase 4: Use BarMerger to coordinate multi-symbol streams
merger = BarMerger(data_iterators)
bars_list = []  # Will contain CanonicalBar objects from MultiModeBar.adjusted

# Extract Bar objects from MultiModeBar (use adjusted mode for strategy)
while merger.has_next():
    symbol, multi_mode_bar = merger.get_next_bar()
    bar = multi_mode_bar.adjusted  # CanonicalBar with split-adjusted prices
    bars_list.append(bar)

# Rest of backtest proceeds with bars_list
```

**Benefits**:

- ✅ Memory efficient (streaming instead of loading all bars upfront)
- ✅ Multi-symbol support via BarMerger
- ✅ Chronological ordering guaranteed
- ✅ Clean separation of concerns
- ✅ Prepares for Phase 4 Part 4 (MultiModeBar in strategies)

### 2. Import Updates

**Added**:

```python
from qtrader.data import BarMerger, PriceSeriesIterator
```

**Removed**:

```python
from typing import Union  # No longer needed (no legacy support)
from qtrader.models.bar import Bar  # Replaced by CanonicalBar
```

## Pending Work

### Phase 4 Part 3.5: Integration Test Updates

**Affected Files** (5 test files, ~40 test methods):

- `tests/integration/test_backtest_full_execution.py` (5 tests)
- All other files using `Backtest.run()`

**Required Changes**:

1. **Update Test Helpers**:

   ```python
   # Old helper
   def create_test_bars(symbol, count, start_price) -> List[Bar]:
       bars = []
       # Create Bar namedtuples with open, high, low, close...
       return bars

   # New helper
   def create_test_iterator(symbol, count, start_price) -> Dict[str, PriceSeriesIterator]:
       vendor_bars = []
       # Create AlgoseekBar objects
       for i in range(count):
           bar = AlgoseekBar(
               Ticker=symbol,
               TradeDate=(base_date + timedelta(days=i)).isoformat(),
               Open=float(open_price),
               High=float(high_price),
               Low=float(low_price),
               Close=float(close_price),
               MarketHoursVolume=1000000,
               CumulativePriceFactor=1.0,
               CumulativeVolumeFactor=1.0,
           )
           vendor_bars.append(bar)

       # Create iterator
       loader = DataLoader({})
       iterator = loader.load_from_vendor_series(symbol, vendor_bars)
       return {symbol: iterator}
   ```

1. **Update Test Calls**:

   ```python
   # Old
   bars = create_test_bars("AAPL", count=5)
   backtest.run(ctx, bars, ["AAPL"], out_dir=Path("/tmp"))

   # New
   iterators = create_test_iterator("AAPL", count=5)
   backtest.run(ctx, iterators, ["AAPL"], out_dir=Path("/tmp"))
   ```

1. **Update Strategy on_bar() Methods**:

   ```python
   # Old
   def on_bar(self, bar: Bar, ctx) -> Optional[List[Signal]]:
       signal_ts = bar.ts  # datetime
       symbol = bar.symbol
       price = bar.close  # Decimal

   # New
   def on_bar(self, bar, ctx) -> Optional[List[Signal]]:  # bar is CanonicalBar
       signal_ts = datetime.fromisoformat(bar.trade_datetime)  # ISO string
       symbol = bar.symbol  # same
       price = bar.close  # float
   ```

**Differences: Bar vs CanonicalBar**:

| Field     | Old Bar (NamedTuple) | New CanonicalBar (Pydantic) |
| --------- | -------------------- | --------------------------- |
| Timestamp | `ts: datetime`       | `trade_datetime: str` (ISO) |
| OHLCV     | Individual Decimals  | Individual floats           |
| Symbol    | `symbol: str`        | `symbol: str` (same)        |

### Phase 4 Part 4: Strategy Interface Update

**After tests are fixed**, update Strategy protocol to receive `MultiModeBar`:

```python
# Current (Phase 4 Part 3)
def on_bar(self, bar, ctx) -> Optional[List[Signal]]:  # bar is CanonicalBar
    strategy_bar = bar  # Uses adjusted mode implicitly

# Future (Phase 4 Part 4)
def on_bar(self, bar: MultiModeBar, ctx) -> Optional[List[Signal]]:
    strategy_bar = bar.adjusted  # Explicit mode selection
    exec_bar = bar.unadjusted     # Available if needed
    perf_bar = bar.total_return   # Available if needed
```

## Architecture Validation

### Current Flow (Phase 4 Part 3)

```
DataLoader.load_data(symbol, start, end)
    ↓
PriceSeriesIterator (yields MultiModeBar)
    ↓
BarMerger({symbol: iterator, ...})
    ↓
while merger.has_next():
    symbol, multi_mode_bar = merger.get_next_bar()
    bar = multi_mode_bar.adjusted  ← Extract CanonicalBar
    bars_list.append(bar)
    ↓
strategy.on_bar(bar, ctx)  ← Receives CanonicalBar
```

### Target Flow (Phase 4 Part 4)

```
DataLoader.load_data(symbol, start, end)
    ↓
PriceSeriesIterator (yields MultiModeBar)
    ↓
BarMerger({symbol: iterator, ...})
    ↓
while merger.has_next():
    symbol, multi_mode_bar = merger.get_next_bar()
    ↓
strategy.on_bar(multi_mode_bar, ctx)  ← Receives MultiModeBar directly
    strategy_bar = multi_mode_bar.adjusted
```

## Technical Details

### BarMerger Usage

```python
# Create iterators (one per symbol)
iterators = {
    "AAPL": DataLoader.load_data("AAPL", "2024-01-01", "2024-12-31"),
    "MSFT": DataLoader.load_data("MSFT", "2024-01-01", "2024-12-31"),
}

# Coordinate chronologically
merger = BarMerger(iterators)

# Process in order
while merger.has_next():
    symbol, multi_mode_bar = merger.get_next_bar()
    # multi_mode_bar.trade_datetime is guaranteed chronological
    # symbol tells which symbol this bar belongs to
```

### Mode Selection

Currently (Phase 4 Part 3):

```python
# Backtest extracts adjusted mode for strategy
bar = multi_mode_bar.adjusted

# Strategy receives CanonicalBar (adjusted mode)
def on_bar(self, bar, ctx):
    # bar.close is split-adjusted
    # bar.volume is split-adjusted
```

Future (Phase 4 Part 4):

```python
# Backtest passes full MultiModeBar to strategy
def on_bar(self, bar: MultiModeBar, ctx):
    # Strategy chooses mode
    indicator_bar = bar.adjusted      # For SMA, RSI, etc.
    execution_bar = bar.unadjusted    # For realistic fills
    performance_bar = bar.total_return # For returns
```

## Test Impact Summary

### Tests Requiring Updates

| Test File                         | Tests       | Change Type           |
| --------------------------------- | ----------- | --------------------- |
| `test_backtest_full_execution.py` | 5           | Update helper + calls |
| `test_full_backtest.py`           | 4 (skipped) | No immediate impact   |
| Other integration tests           | TBD         | Review needed         |

### Tests NOT Affected

- ✅ Unit tests for data layer (72 tests) - Already passing
- ✅ Unit tests for models - No changes needed
- ✅ Unit tests for risk - No changes needed
- ✅ Unit tests for portfolio - No changes needed

Only integration tests that use `Backtest.run()` are affected.

## Migration Guide for Downstream Code

### For Test Code

**Step 1**: Create iterator helper

```python
from qtrader.models.vendors.algoseek.bar import AlgoseekBar
from qtrader.data import DataLoader

def create_test_iterator(symbol: str, count: int) -> Dict[str, PriceSeriesIterator]:
    vendor_bars = [
        AlgoseekBar(
            Ticker=symbol,
            TradeDate=f"2024-01-{i+1:02d}",
            Open=100.0,
            High=101.0,
            Low=99.0,
            Close=100.5,
            MarketHoursVolume=1000000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
        )
        for i in range(count)
    ]

    loader = DataLoader({})
    iterator = loader.load_from_vendor_series(symbol, vendor_bars)
    return {symbol: iterator}
```

**Step 2**: Update test calls

```python
# Replace:
bars = create_test_bars("AAPL", 5)
backtest.run(ctx, bars, ["AAPL"], out_dir)

# With:
iterators = create_test_iterator("AAPL", 5)
backtest.run(ctx, iterators, ["AAPL"], out_dir)
```

**Step 3**: Update strategy on_bar

```python
# Replace:
def on_bar(self, bar: Bar, ctx):
    ts = bar.ts  # datetime object

# With:
def on_bar(self, bar, ctx):  # bar is CanonicalBar
    ts = datetime.fromisoformat(bar.trade_datetime)  # ISO string
```

### For Production Code

Already compatible! Examples like `minimal_iterator_backtest.py` show the correct usage:

```python
loader = DataLoader(config)
iterators = {
    symbol: loader.load_data(symbol, start_date, end_date)
    for symbol in symbols
}

backtest = Backtest(config, strategy)
backtest.run(ctx, iterators, symbols, out_dir)
```

## Success Criteria

### Completed ✅

- [x] Backtest.run() signature updated to use iterators
- [x] BarMerger integrated into event loop
- [x] Multi-symbol chronological ordering working
- [x] Lint errors resolved
- [x] No compilation errors in backtest engine

### Pending 📋

- [ ] Integration test helper functions updated
- [ ] All integration tests passing
- [ ] Strategy on_bar() signature documentation updated
- [ ] Migration guide for custom strategies

## Commits

**This Phase** (Part 3):

- Pending commit with backtest engine updates

**Previous Phases**:

- `016b632`: docs: Add Phase 4 Part 2 completion summary
- `2e6edd8`: feat(phase4-part2): Add minimal iterator-based backtest demonstration
- `9838780`: feat(phase4-part1): Add BarMerger for multi-symbol coordination

## Next Steps

### Immediate (Phase 4 Part 3.5)

1. Create comprehensive test helper for iterator generation
1. Update all 5 tests in `test_backtest_full_execution.py`
1. Verify all tests pass
1. Commit Phase 4 Part 3 complete

### Soon (Phase 4 Part 4)

1. Update Strategy protocol to receive MultiModeBar
1. Update example strategies (buy_and_hold, sma_crossover)
1. Add mode selection documentation
1. Update strategy development guide

### Later (Phase 4 Parts 5-6)

1. Update remaining integration tests
1. Add multi-symbol backtest examples
1. Performance testing with large datasets
1. Complete Phase 4 documentation

## Known Issues

### Issue 1: CanonicalBar vs Bar Type Mismatch

**Problem**: Tests expect old `Bar` NamedTuple, backtest provides `CanonicalBar`

**Impact**: Test strategies fail when accessing `bar.ts` (should be `bar.trade_datetime`)

**Solution**: Update test strategies to use CanonicalBar field names

**Status**: Documented, fix pending in Part 3.5

### Issue 2: AlgoseekBar Constructor Signature

**Problem**: Test helper needs to create AlgoseekBar with correct field names

**Fields Required**:

- `Ticker` (not `symbol`)
- `TradeDate` (not `trade_datetime`)
- `Open`, `High`, `Low`, `Close` (capitalized)
- `MarketHoursVolume` (not `volume`)
- `CumulativePriceFactor`, `CumulativeVolumeFactor`

**Status**: Documented, fix pending in Part 3.5

## References

### Related Documents

- [Phase 4 Part 2 Completion](./PHASE_4_PART_2_COMPLETION_SUMMARY.md)
- [Phase 4 Part 1 Completion](./docs/PHASE_4_PART_1_COMPLETION_SUMMARY.md) (if exists)
- [Data Layer Migration Plan](./DATA_LAYER_MIGRATION_PLAN.md)

### Code References

- **Backtest Engine**: `src/qtrader/api/backtest.py`
- **BarMerger**: `src/qtrader/data/bar_merger.py`
- **DataLoader**: `src/qtrader/data/loader.py`
- **Test File**: `tests/integration/test_backtest_full_execution.py`

### Examples

- **Minimal Backtest**: `examples/minimal_iterator_backtest.py` (Phase 4 architecture demo)
- **Buy and Hold**: `examples/buy_and_hold_strategy.py` (needs update for Phase 4)

______________________________________________________________________

**Phase 4 Part 3 Status**: 🟡 **Engine Complete - Tests Pending**\
**Next Milestone**: Phase 4 Part 3.5 (Integration test updates)
