# Data Architecture Migration Guide

## Breaking Changes in Instrument Model

### What Changed?

**OLD Instrument (duplicated metadata):**

```python
instrument = Instrument(
    symbol="AAPL",
    instrument_type=InstrumentType.EQUITY,  # Duplicates config
    data_source=DataSource.SCHWAB,          # Duplicates config
)
```

**NEW Instrument (minimal, no duplication):**

```python
# Just the symbol!
instrument = Instrument(symbol="AAPL")

# Optional: frequency override
instrument = Instrument(symbol="AAPL", frequency="1m")

# Optional: custom metadata
instrument = Instrument(
    symbol="ES_Z24",
    metadata={"contract_month": "2024-12"}
)
```

### Why?

1. **Config is the source of truth** - data_sources.yaml already specifies provider and asset_class
1. **Eliminated duplication** - why specify "Schwab equity" twice?
1. **User responsibility** - users provide correct ticker for each dataset (Schwab uses "AAPL", IQFeed might use "AAPL.NAS")
1. **Simpler API** - explicit dataset parameter, no inference/guessing

______________________________________________________________________

## Migration Steps

### 1. Update Direct Adapter Usage

**BEFORE:**

```python
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

resolver = DataSourceResolver()
config = resolver.sources["schwab-us-equity-1d-adjusted"]

instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
adapter = SchwabOHLCAdapter(config=config, instrument=instrument)
```

**AFTER:**

```python
from qtrader.models.instrument import Instrument

resolver = DataSourceResolver()
dataset = "schwab-us-equity-1d-adjusted"  # Explicit!
config = resolver.sources[dataset]

instrument = Instrument("AAPL")  # Minimal!
adapter = resolver.resolve_by_dataset(dataset, instrument)  # New method
```

### 2. Update DataService Usage

**BEFORE:**

```python
from qtrader.services.data import DataService

service = DataService(config)  # Guessed dataset from config
instrument = service.get_instrument("AAPL")  # Returned hardcoded EQUITY type
```

**AFTER:**

```python
from qtrader.services.data import DataService

service = DataService(config, dataset="schwab-us-equity-1d-adjusted")  # Explicit!
instrument = service.get_instrument("AAPL")  # Returns minimal Instrument
```

### 3. Update Strategy Configs

**BEFORE (multi-strategy YAML):**

```yaml
strategies:
  - name: momentum
    instruments:
      - symbol: AAPL
        instrument_type: equity     # Redundant
        data_source: schwab         # Redundant
      - symbol: MSFT
        instrument_type: equity
        data_source: schwab
```

**AFTER:**

```yaml
strategies:
  - name: momentum
    dataset: schwab-us-equity-1d-adjusted  # Single source of truth!
    instruments:
      - symbol: AAPL  # Just the ticker
      - symbol: MSFT  # Clean!
```

### 4. Update CLI Commands

**BEFORE:**

```bash
# Implicit source inference
qtrader data raw --symbol AAPL --source schwab
```

**AFTER:**

```bash
# Explicit dataset
qtrader data raw --symbol AAPL --dataset schwab-us-equity-1d-adjusted
```

______________________________________________________________________

## Code Changes Summary

### Modified Files

1. **src/qtrader/models/instrument.py**

   - Removed: `instrument_type`, `data_source` fields
   - Kept: `symbol`, `frequency`, `metadata`
   - Simplified docstring and examples

1. **src/qtrader/adapters/resolver.py**

   - Added: `resolve_by_dataset(dataset, instrument)` - NEW PREFERRED METHOD
   - Deprecated: `resolve(instrument)` - old API with warning
   - Changed: Explicit dataset parameter instead of inferring from instrument.data_source

1. **src/qtrader/services/data/service.py**

   - Added: `dataset` parameter to `__init__()` - REQUIRED (or inferred with warning)
   - Added: `_infer_dataset_from_selector()` helper for backward compatibility
   - Changed: `get_instrument()` returns minimal Instrument (symbol only)
   - Removed: `_get_data_source()` method (no longer needed)

1. **scripts/test_schwab_aapl.py**

   - Updated: Uses new Instrument(symbol) API
   - Updated: Uses resolve_by_dataset() instead of direct adapter instantiation
   - Updated: Explicit dataset variable throughout

______________________________________________________________________

## Backward Compatibility

### What Still Works (with warnings)?

**OLD Instrument model still exists in git history**

- If you have old code using InstrumentType/DataSource, it will fail with clear error
- Error message explains how to migrate

**OLD resolve() method deprecated**

- Still exists but logs warning
- Will be removed in future version
- Use resolve_by_dataset() instead

**OLD DataService without dataset parameter**

- Still works but logs warning
- Attempts to infer dataset from config.source_selector
- May fail if inference is ambiguous

______________________________________________________________________

## Testing Migration

### Test Script Pattern

```python
#!/usr/bin/env python3
"""Test new architecture with explicit dataset."""

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import Instrument

def main():
    # 1. Choose dataset explicitly
    dataset = "schwab-us-equity-1d-adjusted"

    # 2. Get config
    resolver = DataSourceResolver()
    config = resolver.sources[dataset]

    # 3. Create minimal instrument
    instrument = Instrument("AAPL")

    # 4. Resolve to adapter with explicit dataset
    adapter = resolver.resolve_by_dataset(dataset, instrument)

    # 5. Use adapter
    bars = adapter.read_bars(start_date="2024-01-01", end_date="2024-12-31")
    print(f"Fetched {len(list(bars))} bars")

if __name__ == "__main__":
    main()
```

### Unit Test Pattern

```python
def test_explicit_dataset_resolution():
    """Test new explicit dataset resolution."""
    resolver = DataSourceResolver()
    instrument = Instrument("AAPL")

    # Use explicit dataset (no guessing!)
    adapter = resolver.resolve_by_dataset("schwab-us-equity-1d-adjusted", instrument)

    assert adapter is not None
    assert adapter.instrument.symbol == "AAPL"
```

______________________________________________________________________

## Benefits of New Architecture

### 1. No Duplication

- Config says: "schwab-us-equity-1d-adjusted has provider=schwab, asset_class=equity"
- Instrument says: "AAPL"
- **No redundancy!**

### 2. Explicit is Better Than Implicit

```python
# OLD: Where does EQUITY come from? Where does SCHWAB come from?
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)

# NEW: Clear! Dataset is the source of truth
dataset = "schwab-us-equity-1d-adjusted"
instrument = Instrument("AAPL")
adapter = resolver.resolve_by_dataset(dataset, instrument)
```

### 3. Supports Complex Symbol Mappings

```python
# Futures contract naming varies by provider
# User's responsibility to provide correct ticker for each dataset

# Schwab dataset might use:
instrument_schwab = Instrument("ES_Z24")

# IQFeed dataset might use:
instrument_iqfeed = Instrument("@ESZ24")

# Each dataset config knows how to handle its own naming convention
```

### 4. Future-Proof for Multi-Dataset Strategies

```yaml
# Strategy can use multiple datasets cleanly
strategies:
  - name: cross_market
    datasets:
      equities: schwab-us-equity-1d-adjusted
      futures: iqfeed-futures-1d
      crypto: binance-spot-1h
    instruments:
      - symbol: AAPL
        dataset: equities  # Explicit mapping
      - symbol: "@ESZ24"
        dataset: futures
      - symbol: BTCUSDT
        dataset: crypto
```

______________________________________________________________________

## Common Migration Issues

### Issue 1: "Instrument has no attribute 'data_source'"

**Cause:** Trying to use old Instrument API

**Fix:**

```python
# OLD
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
adapter = resolver.resolve(instrument)  # Deprecated!

# NEW
instrument = Instrument("AAPL")
adapter = resolver.resolve_by_dataset("schwab-us-equity-1d-adjusted", instrument)
```

### Issue 2: "DataService initialized without explicit dataset"

**Cause:** Not passing dataset parameter

**Fix:**

```python
# OLD
service = DataService(config)  # Implicit, logs warning

# NEW
service = DataService(config, dataset="schwab-us-equity-1d-adjusted")  # Explicit!
```

### Issue 3: ImportError for InstrumentType/DataSource

**Cause:** Still importing deprecated enums

**Fix:**

```python
# OLD
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

# NEW
from qtrader.models.instrument import Instrument  # That's it!

# Note: InstrumentType and DataSource enums still exist for internal use
# but are not needed for creating Instruments anymore
```

______________________________________________________________________

## Rollout Plan

### Phase 1: Core Changes (DONE)

- ✅ Updated Instrument model
- ✅ Added resolve_by_dataset() method
- ✅ Updated DataService with dataset parameter
- ✅ Updated test script

### Phase 2: Update All Usage (IN PROGRESS)

- [ ] Update CLI commands
- [ ] Update example strategies
- [ ] Update backtest configs
- [ ] Update documentation

### Phase 3: Deprecate Old API (FUTURE)

- [ ] Add deprecation warnings everywhere
- [ ] Update all tests
- [ ] Migration guide in docs

### Phase 4: Remove Old API (FUTURE)

- [ ] Remove resolve() method
- [ ] Remove old DataService inference
- [ ] Breaking change release (v2.0.0)

______________________________________________________________________

## Questions?

**Q: Can I still use multiple data sources in one strategy?** A: Yes! Strategy config specifies dataset per instrument:

```yaml
instruments:
  - symbol: AAPL
    dataset: schwab-us-equity-1d-adjusted
  - symbol: BTCUSDT
    dataset: binance-spot-1h
```

**Q: What about futures with complex naming?** A: User provides correct ticker for each dataset. Dataset config handles provider-specific naming.

**Q: Do I need to update all code at once?** A: No. Old API still works (with warnings). Migrate incrementally.

**Q: When will old API be removed?** A: Not before v2.0.0. Plenty of deprecation warning period.
