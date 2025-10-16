# Complete Architectural Fix - Implementation Summary

## Objective

Eliminate duplication between dataset config and Instrument metadata. Make dataset config the single source of truth for provider, asset type, and other metadata.

## Problem Statement

**Before:**

```python
# Config already says this is Schwab equity data
schwab-us-equity-1d-adjusted:
  provider: schwab
  asset_class: equity

# But we specified it AGAIN in Instrument
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
```

**Duplication issues:**

1. Same information in two places (config + instrument)
1. DataService had to hardcode/guess instrument metadata
1. Ambiguous which is source of truth
1. Adapters only used symbol anyway

## Solution

**After:**

```python
# Config is the ONLY place that specifies provider/asset_class
schwab-us-equity-1d-adjusted:
  provider: schwab
  asset_class: equity

# Instrument is minimal - just the ticker
instrument = Instrument("AAPL")

# Dataset is specified explicitly when resolving
adapter = resolver.resolve_by_dataset("schwab-us-equity-1d-adjusted", instrument)
```

**Benefits:**

1. ✅ No duplication - config is single source of truth
1. ✅ Explicit dataset parameter - no guessing
1. ✅ User responsibility for correct ticker per dataset
1. ✅ Supports complex symbol mappings (futures: XYT vs XYT1)

______________________________________________________________________

## Changes Made

### 1. Updated Instrument Model

**File:** `src/qtrader/models/instrument.py`

**BEFORE:**

```python
class Instrument(NamedTuple):
    symbol: str
    instrument_type: InstrumentType  # REMOVED - duplicates config
    data_source: DataSource          # REMOVED - duplicates config
    frequency: Optional[str] = None
    metadata: Dict[str, Any] = {}
```

**AFTER:**

```python
class Instrument(NamedTuple):
    symbol: str                      # Just the ticker!
    frequency: Optional[str] = None  # Override dataset default
    metadata: Dict[str, Any] = {}    # Custom attributes
```

**Philosophy:**

- User provides correct ticker for dataset
- Dataset config provides all metadata
- No duplication!

### 2. Added Explicit Dataset Resolution

**File:** `src/qtrader/adapters/resolver.py`

**NEW METHOD (preferred):**

```python
def resolve_by_dataset(self, dataset: str, instrument: Instrument):
    """
    Resolve using explicit dataset name.

    Examples:
        >>> instrument = Instrument("AAPL")
        >>> adapter = resolver.resolve_by_dataset(
        ...     "schwab-us-equity-1d-adjusted",
        ...     instrument
        ... )
    """
    if dataset not in self.sources:
        raise KeyError(f"Dataset '{dataset}' not configured")

    config = self.sources[dataset]
    return self._create_adapter(dataset, config, instrument)
```

**DEPRECATED METHOD (backward compat):**

```python
def resolve(self, instrument: Instrument):
    """DEPRECATED - use resolve_by_dataset() instead."""
    logger.warning("Use resolve_by_dataset() with explicit dataset")
    # Tries to infer from instrument.metadata['data_source']
    # Fails with clear error if new Instrument API used
```

### 3. Updated DataService

**File:** `src/qtrader/services/data/service.py`

**BEFORE:**

```python
def __init__(self, config, resolver=None):
    # Had to guess dataset from config.source_selector

def get_instrument(self, symbol):
    # Hardcoded InstrumentType.EQUITY
    return Instrument(symbol, InstrumentType.EQUITY, self._guess_source())
```

**AFTER:**

```python
def __init__(self, config, dataset=None, resolver=None):
    """
    Args:
        dataset: Explicit dataset name (e.g., "schwab-us-equity-1d-adjusted")
                 Required! If None, will try to infer with warning.
    """
    self.dataset = dataset
    if not self.dataset:
        logger.warning("DataService initialized without explicit dataset")
        self.dataset = self._infer_dataset_from_selector(config.source_selector)

def get_instrument(self, symbol):
    # Returns minimal instrument (symbol only)
    return Instrument(symbol=symbol)
```

**Key changes:**

- Requires explicit `dataset` parameter (or logs warning)
- No more hardcoded EQUITY assumptions
- Returns minimal Instrument

### 4. Updated Test Script

**File:** `scripts/test_schwab_aapl.py`

**BEFORE:**

```python
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

config = resolver.sources["schwab-us-equity-1d-adjusted"]

instrument = Instrument(
    symbol="AAPL",
    instrument_type=InstrumentType.EQUITY,  # Redundant!
    data_source=DataSource.SCHWAB,          # Redundant!
)

adapter = SchwabOHLCAdapter(config=config, instrument=instrument)
```

**AFTER:**

```python
from qtrader.models.instrument import Instrument

dataset = "schwab-us-equity-1d-adjusted"  # Explicit!
config = resolver.sources[dataset]

instrument = Instrument("AAPL")  # Minimal!

adapter = resolver.resolve_by_dataset(dataset, instrument)  # Clean!
```

______________________________________________________________________

## API Changes Summary

### Creating Instruments

| Old API                                                                           | New API                                |
| --------------------------------------------------------------------------------- | -------------------------------------- |
| `Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)`                    | `Instrument("AAPL")`                   |
| `Instrument("BTCUSD", InstrumentType.CRYPTO, DataSource.BINANCE, frequency="1m")` | `Instrument("BTCUSD", frequency="1m")` |

### Resolving to Adapters

| Old API                                       | New API                                            |
| --------------------------------------------- | -------------------------------------------------- |
| `resolver.resolve(instrument)` *(deprecated)* | `resolver.resolve_by_dataset(dataset, instrument)` |
| *Inferred from instrument.data_source*        | *Explicit dataset parameter*                       |

### DataService Initialization

| Old API                                          | New API                                                       |
| ------------------------------------------------ | ------------------------------------------------------------- |
| `DataService(config)` *(implicit, logs warning)* | `DataService(config, dataset="schwab-us-equity-1d-adjusted")` |
| *Guessed from config.source_selector*            | *Explicit dataset parameter*                                  |

______________________________________________________________________

## Migration Path

### For Test Scripts

```python
# Step 1: Make dataset explicit
dataset = "schwab-us-equity-1d-adjusted"

# Step 2: Use minimal Instrument
instrument = Instrument("AAPL")  # No InstrumentType, no DataSource

# Step 3: Use resolve_by_dataset()
adapter = resolver.resolve_by_dataset(dataset, instrument)
```

### For Strategy Code

**Will require future updates:**

```yaml
# Old multi-strategy config
strategies:
  - name: momentum
    instruments:
      - symbol: AAPL
        instrument_type: equity      # TO REMOVE
        data_source: schwab          # TO REMOVE

# New multi-strategy config
strategies:
  - name: momentum
    dataset: schwab-us-equity-1d-adjusted  # Single source of truth!
    instruments:
      - symbol: AAPL  # Just ticker
      - symbol: MSFT
```

______________________________________________________________________

## Backward Compatibility

### What Breaks Immediately?

**Old Instrument creation:**

```python
# This will FAIL (InstrumentType/DataSource removed)
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
# Error: No parameter named 'instrument_type'
```

**Fix:**

```python
instrument = Instrument("AAPL")  # Clean!
```

### What Works (with warnings)?

**Old resolve() method:**

```python
# This logs deprecation warning but tries to work
adapter = resolver.resolve(instrument)
# Warning: resolve() is deprecated. Use resolve_by_dataset()
```

**Old DataService without dataset:**

```python
# This logs warning and tries to infer
service = DataService(config)
# Warning: DataService initialized without explicit dataset
```

______________________________________________________________________

## Testing

### Test Script Ready

```bash
# Set credentials
export SCHWAB_API_KEY="your_client_id"
export SCHWAB_API_SECRET="your_client_secret"

# Run test
python scripts/test_schwab_aapl.py
```

**Expected output:**

```
✓ Dataset: schwab-us-equity-1d-adjusted
✓ Adapter initialized for AAPL
  (Config provides: provider=schwab, asset_class=equity)
✓ Fetched X bars
✓ Cache created
✓ Smart caching fully functional!
```

### Unit Tests Needed

- [ ] Test `Instrument()` with minimal args
- [ ] Test `resolve_by_dataset()` with various datasets
- [ ] Test `DataService(config, dataset=...)`
- [ ] Test backward compat warnings
- [ ] Test error messages for missing fields

______________________________________________________________________

## Next Steps

### Immediate (Ready to Test)

1. **Run real Schwab test:**

   ```bash
   python scripts/test_schwab_aapl.py
   ```

1. **Verify cache creation:**

   ```bash
   ls -la data/us-equity-daily-adjusted-schwab/AAPL/
   ```

1. **Test incremental update:**

   ```bash
   qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL
   ```

### Short-term (Next Week)

1. **Update CLI commands** to use `--dataset` parameter
1. **Update example strategies** to use new Instrument API
1. **Add unit tests** for new architecture
1. **Update docs** with migration guide

### Medium-term (Phase 2)

1. **Update multi-strategy configs** to specify dataset per strategy
1. **Add InstrumentRegistry** for multi-source symbol lookup
1. **Remove deprecated methods** (breaking change for v2.0)

______________________________________________________________________

## Design Philosophy

### Core Principles

1. **Single Source of Truth**

   - Config file specifies provider, asset class, adjustments
   - Instrument just specifies symbol (and optional overrides)
   - No duplication!

1. **Explicit Over Implicit**

   - Always specify dataset explicitly
   - No guessing or inference
   - Clear what data source is being used

1. **User Responsibility**

   - User provides correct ticker for each dataset
   - Different providers use different naming (AAPL vs @AAPL vs AAPL.NAS)
   - System doesn't try to map symbols across providers

1. **Minimal Instrument**

   - Just symbol + optional frequency override
   - Metadata dict for custom attributes
   - Dataset provides all else

### Why This Matters

**Before (duplication):**

- "Give me AAPL from Schwab... oh wait, config already says Schwab"
- "Is instrument_type needed? Config already says equity"
- "DataService has to guess - is this equity or futures?"

**After (clean):**

- "Give me AAPL from schwab-us-equity-1d-adjusted dataset"
- "Dataset config tells me everything about provider/asset/adjustments"
- "User chose the right ticker for this dataset"

______________________________________________________________________

## Summary

### What We Built

✅ **Eliminated duplication** - config is single source of truth\
✅ **Explicit dataset resolution** - `resolve_by_dataset()`\
✅ **Minimal Instrument model** - just symbol + overrides\
✅ **DataService with explicit dataset** - no more guessing\
✅ **Updated test script** - demonstrates clean architecture\
✅ **Migration guide** - comprehensive docs for updating code\
✅ **Backward compatibility** - deprecated methods still work (with warnings)

### Ready to Test

The test script is ready! Just provide Schwab credentials and run:

```bash
export SCHWAB_API_KEY="..."
export SCHWAB_API_SECRET="..."
python scripts/test_schwab_aapl.py
```

This will test the complete new architecture with real data.
