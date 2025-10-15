# CLI Status: DataService Migration

## Current Status: ❌ NOT Updated

The CLI (`src/qtrader/cli.py`) has **NOT yet been updated** to use the new DataService from Phase 1.

## What the CLI Currently Does

### `raw-data` Command

**Current implementation:**

```python
# Direct adapter access (old way)
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)
bars = list(adapter.read_bars(start_date, end_date))

# Problem: Gets vendor-specific bars (AlgoseekBar, SchwabBar)
# Requires hasattr() checks to handle different bar types
```

**Issues with current approach:**

- ❌ Uses low-level adapter API directly
- ❌ Gets vendor-specific bar types (AlgoseekBar vs SchwabBar)
- ❌ Requires vendor-specific display code with `hasattr()` checks
- ❌ Doesn't use the new lego architecture
- ❌ Not consistent with Phase 1 DataService

## What Should Be Done

### Update CLI to Use DataService

**Recommended implementation:**

```python
# Use DataService (new way)
from qtrader.services.data import DataService
from qtrader.config.data_config import DataConfig, BarSchemaConfig

# Configure service
config = DataConfig(
    source_tag=f"{source}-adjusted",
    bar_schema=bar_schema,
)

# Load data
service = DataService(config)
iterator = service.load_symbol(symbol, start_date, end_date)
bars = list(iterator)

# Benefits: Gets canonical MultiBar objects
# - multi_bar.unadjusted    → raw prices
# - multi_bar.adjusted      → split-adjusted
# - multi_bar.total_return  → total return
```

**Benefits:**

- ✅ Vendor-agnostic canonical Bar objects
- ✅ No vendor-specific code needed
- ✅ All three adjustment modes available
- ✅ Consistent with lego architecture
- ✅ Better error handling and logging
- ✅ Can be easily mocked for testing

## Files Status

| File                                 | Uses DataService? | Status          |
| ------------------------------------ | ----------------- | --------------- |
| `src/qtrader/cli.py`                 | ❌ No             | Needs update    |
| `examples/data_service_example.py`   | ✅ Yes            | Up to date      |
| `examples/data_source_selection.py`  | ✅ Yes            | Up to date      |
| `examples/buy_and_hold_strategy.py`  | ⚠️ Partial        | Uses old runner |
| `examples/sma_crossover_strategy.py` | ⚠️ Partial        | Uses old runner |

## Migration Plan

### Step 1: Update `raw-data` Command

See `docs/lego_architecture/CLI_MIGRATION.md` for detailed code.

**Changes needed:**

1. Add DataService imports
1. Replace adapter creation with DataService
1. Update bar display to use canonical Bar objects
1. Remove vendor-specific code (hasattr checks)
1. Add `--mode` option to select adjustment mode

**Estimated effort:** 30 minutes

### Step 2: Test Updated Command

```bash
# Test with Algoseek
qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31

# Test with different modes
qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31 --mode adjusted
qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31 --mode total_return
```

**Estimated effort:** 15 minutes

### Step 3: Update Examples (Optional)

`buy_and_hold_strategy.py` and `sma_crossover_strategy.py` currently use an old runner pattern. These could be updated after Phase 5 (BacktestEngine) is complete.

**Priority:** Low (examples still work, just not using new architecture)

## Recommendation

**Should you update the CLI now?**

**Option A: Update now (Recommended)**

- ✅ CLI uses consistent architecture
- ✅ Validates DataService works in real usage
- ✅ Better user experience (adjustment modes)
- ⏱️ Quick update (~45 minutes)

**Option B: Update later**

- ✅ Focus on Phase 2 (PortfolioService)
- ❌ CLI inconsistent with architecture
- ❌ Miss opportunity to validate DataService

**My recommendation:** Update now. It's a quick win that:

1. Validates Phase 1 DataService works in production
1. Makes CLI consistent with architecture
1. Provides better UX (adjustment modes)
1. Sets good precedent for future phases

## Summary

**Current State:**

- CLI uses direct adapter access (old pattern)
- Gets vendor-specific bars requiring type-specific code

**Desired State:**

- CLI uses DataService (new pattern)
- Gets canonical MultiBar objects with clean interface

**Action Required:**

- Update `src/qtrader/cli.py` to use DataService
- Test with real data
- Optional: Update strategy examples

**Effort:** ~1 hour total **Priority:** Medium (not blocking Phase 2, but recommended)
