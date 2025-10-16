# Bug Fixes and Improvements Summary

## Issues Fixed

### 1. ✅ Token Cache Path Bug

**Issue:** `SchwabOHLCAdapter` crashed when `token_cache_path: null` in config

```
TypeError: argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'NoneType'
```

**Root Cause:** Config had `token_cache_path: null` (YAML null → Python `None`), but adapter tried to do `Path(config["token_cache_path"])` without checking for `None`.

**Fix:** Updated `SchwabOHLCAdapter.__init__()` to handle `None`:

```python
# Handle token_cache_path: if None or not set, use default
token_cache_path = config.get("token_cache_path", None)
if token_cache_path:
    oauth_config["token_cache_path"] = Path(token_cache_path)
else:
    # Use default: ~/.qtrader/schwab_tokens.json
    oauth_config["token_cache_path"] = Path.home() / ".qtrader" / "schwab_tokens.json"
```

**File:** `src/qtrader/adapters/schwab.py` (lines 317-323)

______________________________________________________________________

### 2. ✅ DatasetUpdater Missing Instrument Argument

**Issue:** CLI commands crashed with:

```
TypeError: SchwabOHLCAdapter.__init__() missing 1 required positional argument: 'instrument'
```

**Root Cause:** After architectural refactor, adapters require `(config, instrument)` but `DatasetUpdater` was instantiating with just `adapter_class(self.adapter_config)`.

**Fix:** Refactored `DatasetUpdater` to create adapters per-symbol on-demand:

```python
# OLD: Create one adapter for all symbols (broken)
self.adapter = adapter_class(self.adapter_config)

# NEW: Create adapter per symbol when needed
def update_symbol(self, symbol: str, ...):
    instrument = Instrument(symbol=symbol)
    adapter = self.resolver.resolve_by_dataset(self.dataset_name, instrument)
    return adapter.update_to_latest(dry_run=dry_run)
```

**Files:**

- `src/qtrader/data/dataset_updater.py` (lines 120-160, 220-260)

**Key Changes:**

1. Removed adapter instantiation from `__init__()`
1. Added `_create_adapter_for_symbol()` helper method
1. Create fresh adapter per symbol in `update_symbol()`
1. Check adapter class (not instance) for update support

______________________________________________________________________

### 3. ✅ Cache Info Enhancement

**Issue:** `qtrader data cache-info` only showed symbol names, no date ranges or bar counts.

**Enhancement:** Added metadata display:

- Start Date
- End Date
- Number of Bars
- Last Update timestamp

**Implementation:**

```python
# Read .metadata.json for each symbol
metadata_file = cache_root / symbol / ".metadata.json"
if metadata_file.exists():
    with open(metadata_file) as f:
        metadata = json.load(f)

    date_range = metadata.get("date_range", {})
    start_date = date_range.get("start", "N/A")
    end_date = date_range.get("end", "N/A")
    row_count = metadata.get("row_count", "N/A")
    last_update = metadata.get("last_update", "N/A")

    table.add_row(symbol, start_date, end_date, str(row_count), last_update)
```

**File:** `src/qtrader/cli.py` (lines 385-415)

**Output Example:**

```
                         Cached Symbols
┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Start Date ┃ End Date   ┃ Bars ┃ Last Update         ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ 2025-09-16 │ 2025-10-16 │   23 │ 2025-10-16 17:29:14 │
│ GOOGL  │ 2019-12-31 │ 2025-01-30 │ 1278 │ 2025-10-15 22:54:13 │
│ MSFT   │ 2019-12-31 │ 2025-01-30 │ 1278 │ 2025-10-15 22:54:13 │
└────────┴────────────┴────────────┴──────┴─────────────────────┘
```

______________________________________________________________________

## Testing Results

### ✅ Test Script (scripts/test_schwab_aapl.py)

```bash
python scripts/test_schwab_aapl.py
```

**Results:**

- ✅ Environment check passes
- ✅ Adapter initializes with minimal `Instrument("AAPL")`
- ✅ Token cache defaults to `~/.qtrader/schwab_tokens.json`
- ✅ Cache hit works (23 bars loaded)
- ✅ Metadata correctly displayed
- ✅ All 6 tests pass

### ✅ CLI Cache Info

```bash
qtrader data cache-info --dataset schwab-us-equity-1d-adjusted
```

**Results:**

- ✅ Shows 3 cached symbols
- ✅ Displays date ranges for each
- ✅ Shows bar counts
- ✅ Shows last update timestamps
- ✅ No crashes

### ✅ CLI Update (Dry Run)

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL --dry-run
```

**Results:**

- ✅ Creates adapter dynamically per symbol
- ✅ Progress bar works
- ✅ Summary table displays correctly
- ✅ Dry-run mode works
- ✅ No crashes

______________________________________________________________________

## Architecture Impact

### Instrument Model Changes

**Before (duplicated):**

```python
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
```

**After (minimal):**

```python
instrument = Instrument("AAPL")  # Just symbol!
adapter = resolver.resolve_by_dataset("schwab-us-equity-1d-adjusted", instrument)
```

### DatasetUpdater Pattern

**Before (one adapter for all symbols):**

```python
class DatasetUpdater:
    def __init__(self, dataset):
        self.adapter = adapter_class(config)  # BROKEN - needs instrument

    def update_symbol(self, symbol):
        self.adapter.update_to_latest(symbol=symbol)  # Doesn't exist
```

**After (adapter per symbol):**

```python
class DatasetUpdater:
    def __init__(self, dataset):
        self.dataset_name = dataset
        self.resolver = DataSourceResolver()
        # No adapter instantiation here

    def _create_adapter_for_symbol(self, symbol):
        instrument = Instrument(symbol=symbol)
        return self.resolver.resolve_by_dataset(self.dataset_name, instrument)

    def update_symbol(self, symbol, dry_run=False):
        adapter = self._create_adapter_for_symbol(symbol)
        return adapter.update_to_latest(dry_run=dry_run)
```

______________________________________________________________________

## Files Modified

1. **src/qtrader/adapters/schwab.py**

   - Fixed token_cache_path handling (None → default path)
   - Lines 317-323

1. **src/qtrader/data/dataset_updater.py**

   - Refactored to create adapters per-symbol
   - Removed adapter from __init__
   - Added \_create_adapter_for_symbol() helper
   - Lines 120-260

1. **src/qtrader/cli.py**

   - Enhanced cache-info to show metadata
   - Added date range, bar count, last update columns
   - Lines 385-415

______________________________________________________________________

## Summary

All issues resolved:

- ✅ Token cache path bug fixed (handles None gracefully)
- ✅ DatasetUpdater refactored (creates adapters per-symbol)
- ✅ Cache info enhanced (shows date ranges and bar counts)
- ✅ All CLI commands working
- ✅ Test script passes all 6 tests
- ✅ No regressions

The system is now fully functional with the new minimal Instrument architecture!
