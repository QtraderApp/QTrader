# Data Workflow Duplication Analysis

## Problem Statement

The current architecture stores the same information in multiple places:

### 1. Config File (`data_sources.yaml`)

```yaml
schwab-us-equity-1d-adjusted:
  provider: schwab          # ← Says it's Schwab
  asset_class: equity       # ← Says it's equity
  frequency: 1d
  adapter: schwabOHLC
```

### 2. Instrument Object

```python
instrument = Instrument(
    symbol="AAPL",
    instrument_type=InstrumentType.EQUITY,  # ← Says it's equity AGAIN
    data_source=DataSource.SCHWAB,          # ← Says it's Schwab AGAIN
)
```

**This is redundant!** The config already knows it's Schwab equity data. Why specify it again?

______________________________________________________________________

## Current Data Flow (Redundant)

### Path 1: Test Scripts (Direct Adapter Usage)

```
User specifies:
  ├─ Config: "schwab-us-equity-1d-adjusted" (has provider=schwab, asset_class=equity)
  └─ Instrument: (symbol="AAPL", type=EQUITY, source=SCHWAB)
         ↓
  Adapter gets BOTH config AND instrument
         ↓
  Adapter only uses: instrument.symbol
         ↓
  Result: instrument_type and data_source are WASTED
```

### Path 2: DataService (via Resolver)

```
User calls: service.get_instrument("AAPL")
         ↓
  DataService creates:
    Instrument("AAPL", EQUITY, SCHWAB)  # ← Hardcoded assumptions!
         ↓
  Resolver.resolve(instrument)
         ↓
  Uses instrument.data_source → finds config
         ↓
  Config ALREADY has provider=schwab, asset_class=equity
         ↓
  Result: Instrument duplicates config metadata
```

### Path 3: Strategy (High-Level)

```
Strategy: "Give me AAPL"
         ↓
  DataService.get_instrument("AAPL")
         ↓
  Returns: Instrument("AAPL", EQUITY, SCHWAB)  # ← Where did EQUITY and SCHWAB come from?
         ↓
  Problem: DataService just guesses/hardcodes!
```

______________________________________________________________________

## Root Cause Analysis

### What's Wrong?

1. **Config is the source of truth** for:

   - Provider (schwab, algoseek, etc.)
   - Asset class (equity, crypto, futures)
   - Frequency (1d, 1m)

1. **But Instrument duplicates this metadata** for no clear reason

1. **Adapters don't use the duplicated metadata** - only use symbol

1. **DataService has no way to look up metadata** - just hardcodes EQUITY

### Why Did This Happen?

Looking at the design history:

**Phase 1 Goal (from specs/phase01.md):**

> "Decouple logical instruments from physical data sources"

**Original Vision:**

```python
# Strategy specifies WHAT it wants (logical)
instruments = [
    Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK),
    Instrument("BTCUSD", InstrumentType.CRYPTO, DataSource.BINANCE),
]

# Config specifies HOW to get it (physical)
# - algoseek → AlgoseekOHLCAdapter with specific paths
# - binance → BinanceAdapter with API keys
```

**The idea was:**

- Instrument = "I want Apple stock from Algoseek"
- Resolver = "Ah, Algoseek equities go to algoseek-us-equity-1d-unadjusted config"
- Config = "Use AlgoseekOHLCAdapter with path X"

**But reality diverged:**

- Config already specifies provider + asset_class
- So Instrument metadata is redundant
- And DataService can't infer it anyway (just hardcodes)

______________________________________________________________________

## Proposed Solutions

### Option 1: Minimal Instrument (Symbol Only) ⭐ RECOMMENDED

**Core Idea:** Instrument only specifies symbol. All metadata comes from config/registry.

```python
# Simplified Instrument
class Instrument(NamedTuple):
    symbol: str
    frequency: Optional[str] = None  # Can override dataset default
    metadata: Dict[str, Any] = {}    # Custom tags

# No more instrument_type or data_source!
```

**Workflow:**

```python
# 1. Test Scripts (Direct)
config = resolver.sources["schwab-us-equity-1d-adjusted"]
instrument = Instrument("AAPL")  # That's it!
adapter = SchwabOHLCAdapter(config=config, instrument=instrument)

# 2. DataService (Indirect)
class DataService:
    def __init__(self, dataset: str):
        self.dataset = dataset  # e.g., "schwab-us-equity-1d-adjusted"
        self.config = resolver.sources[dataset]

    def get_instrument(self, symbol: str) -> Instrument:
        # No guessing needed - just create minimal instrument
        return Instrument(symbol=symbol)

    def get_bars(self, symbol: str):
        instrument = Instrument(symbol)
        # Pass the DATASET to resolver, not data_source enum
        adapter = resolver.resolve_by_dataset(self.dataset, instrument)
        return adapter.read_bars(...)

# 3. Strategy (High-Level)
# Strategy config specifies dataset
strategy_config = {
    "dataset": "schwab-us-equity-1d-adjusted",  # Source of truth!
    "symbols": ["AAPL", "MSFT"],
}

class MyStrategy(Strategy):
    def on_bar(self, bar, ctx):
        # Don't care about instrument metadata
        # Just use the data
        ...
```

**Advantages:**

- ✅ No duplication - config is single source of truth
- ✅ Simpler Instrument model - just symbol + optional overrides
- ✅ DataService doesn't guess - uses explicit dataset
- ✅ Adapters already only use symbol
- ✅ Clear separation: dataset = config key, instrument = symbol

**Disadvantages:**

- ❌ Can't mix data sources in one strategy (but do we need this?)
- ❌ Multi-source strategies need different approach

______________________________________________________________________

### Option 2: Config-Derived Instrument

**Core Idea:** Instrument still has metadata, but it's populated FROM config, not specified by user.

```python
class Instrument(NamedTuple):
    symbol: str
    instrument_type: InstrumentType  # Derived from config
    data_source: DataSource          # Derived from config
    frequency: Optional[str] = None
    metadata: Dict[str, Any] = {}

class DataSourceResolver:
    def create_instrument(self, symbol: str, dataset: str) -> Instrument:
        """Create instrument with metadata from config."""
        config = self.sources[dataset]

        return Instrument(
            symbol=symbol,
            instrument_type=InstrumentType(config["asset_class"]),
            data_source=DataSource(config["provider"]),
            frequency=config.get("frequency"),
        )

    def resolve_by_dataset(self, dataset: str, instrument: Instrument):
        """Resolve using explicit dataset name."""
        config = self.sources[dataset]
        return self._create_adapter(dataset, config, instrument)
```

**Usage:**

```python
# Test script
instrument = resolver.create_instrument("AAPL", "schwab-us-equity-1d-adjusted")
# Returns: Instrument("AAPL", EQUITY, SCHWAB) from config
adapter = resolver.resolve_by_dataset("schwab-us-equity-1d-adjusted", instrument)

# DataService
class DataService:
    def get_instrument(self, symbol: str) -> Instrument:
        return self.resolver.create_instrument(symbol, self.dataset)
```

**Advantages:**

- ✅ Instrument has full metadata for logging
- ✅ Single source of truth (config)
- ✅ Can validate symbol belongs to dataset type
- ✅ Backward compatible with existing Instrument model

**Disadvantages:**

- ❌ More complex API
- ❌ Still carrying metadata adapters don't use

______________________________________________________________________

### Option 3: Instrument Registry (Future-Proof)

**Core Idea:** Lookup symbol → get instrument metadata from registry.

```python
class InstrumentRegistry:
    """Maps symbols to their metadata."""

    def __init__(self, config_path: str = "config/instruments.yaml"):
        self.instruments = self._load_registry(config_path)

    def lookup(self, symbol: str) -> InstrumentInfo:
        """
        Look up instrument metadata.

        Returns:
            InstrumentInfo with:
            - instrument_type (equity, crypto, etc.)
            - exchange
            - sector
            - default_data_source
            - etc.
        """
        if symbol in self.instruments:
            return self.instruments[symbol]

        # Try to infer from symbol
        if symbol.endswith("USD"):
            return InstrumentInfo(symbol, InstrumentType.CRYPTO, ...)
        else:
            return InstrumentInfo(symbol, InstrumentType.EQUITY, ...)

class DataService:
    def __init__(self, dataset: str):
        self.dataset = dataset
        self.registry = InstrumentRegistry()

    def get_instrument(self, symbol: str) -> Instrument:
        # Look up in registry (not hardcoded!)
        info = self.registry.lookup(symbol)

        return Instrument(
            symbol=symbol,
            instrument_type=info.instrument_type,
            data_source=self._get_data_source(),
        )
```

**Advantages:**

- ✅ Proper metadata lookup (not hardcoded)
- ✅ Can support multi-asset strategies
- ✅ Extensible for future features
- ✅ Single source of truth (registry)

**Disadvantages:**

- ❌ More infrastructure (registry file, lookup logic)
- ❌ Maintenance overhead (keeping registry updated)
- ❌ Still duplicates what config already knows

______________________________________________________________________

## Recommendation: Option 1 (Minimal Instrument) + Registry for Multi-Source

### Phase 1: Simplify Current Single-Source Use Case

**For current code (single dataset per backtest):**

```python
# 1. Simplify Instrument
class Instrument(NamedTuple):
    symbol: str
    frequency: Optional[str] = None  # Override dataset default
    metadata: Dict[str, Any] = {}

# 2. Update DataService to require explicit dataset
class DataService:
    def __init__(self, dataset: str):
        """
        Initialize data service for a specific dataset.

        Args:
            dataset: Dataset name from data_sources.yaml
                    (e.g., "schwab-us-equity-1d-adjusted")
        """
        self.dataset = dataset
        self.resolver = DataSourceResolver()
        self.config = self.resolver.sources[dataset]

    def get_bars(self, symbol: str, ...):
        instrument = Instrument(symbol)
        adapter = self.resolver.resolve_by_dataset(self.dataset, instrument)
        return adapter.read_bars(...)

# 3. Update resolver
class DataSourceResolver:
    def resolve_by_dataset(self, dataset: str, instrument: Instrument):
        """Resolve using explicit dataset name (no guessing)."""
        config = self.sources[dataset]
        return self._create_adapter(dataset, config, instrument)

# 4. Strategy config becomes explicit
backtest_config = {
    "dataset": "schwab-us-equity-1d-adjusted",  # Single source of truth
    "symbols": ["AAPL", "MSFT", "GOOGL"],
    ...
}

# 5. Test scripts
config = resolver.sources["schwab-us-equity-1d-adjusted"]
instrument = Instrument("AAPL")
adapter = SchwabOHLCAdapter(config=config, instrument=instrument)
```

**Migration:**

1. Add `resolve_by_dataset()` to DataSourceResolver ✅ NEW METHOD
1. Update DataService to take `dataset` parameter ✅ BREAKING CHANGE
1. Simplify Instrument model (optional - can keep old for backward compat)
1. Update all strategy configs to specify `dataset` instead of per-symbol data_source

______________________________________________________________________

### Phase 2: Add Registry for Multi-Source Strategies

**For future multi-asset/multi-source:**

```python
class InstrumentRegistry:
    def lookup(self, symbol: str) -> InstrumentMetadata:
        """Get instrument metadata including best data source."""
        ...

class MultiSourceDataService:
    def __init__(self):
        self.registry = InstrumentRegistry()
        self.resolver = DataSourceResolver()

    def get_bars(self, symbol: str, ...):
        # Look up best dataset for this symbol
        metadata = self.registry.lookup(symbol)
        dataset = self._choose_dataset(symbol, metadata)

        instrument = Instrument(symbol)
        adapter = self.resolver.resolve_by_dataset(dataset, instrument)
        return adapter.read_bars(...)
```

______________________________________________________________________

## Action Plan

### Immediate Actions (For Test Script)

**Option A: Keep current redundancy (quick fix)**

```python
# Just accept the duplication for now
instrument = Instrument(
    symbol="AAPL",
    instrument_type=InstrumentType.EQUITY,
    data_source=DataSource.SCHWAB,
)
# Add comment explaining it's redundant but harmless
```

**Option B: Create helper (cleaner)**

```python
def create_test_instrument(symbol: str, config: Dict) -> Instrument:
    """Create instrument from symbol and config (no duplication)."""
    return Instrument(
        symbol=symbol,
        instrument_type=InstrumentType(config["asset_class"]),
        data_source=DataSource(config["provider"]),
    )

# Usage
instrument = create_test_instrument("AAPL", config)
adapter = SchwabOHLCAdapter(config=config, instrument=instrument)
```

### Medium-Term (1-2 weeks)

1. **Add `resolve_by_dataset()` method** to DataSourceResolver

   - Takes explicit dataset name instead of inferring from instrument.data_source
   - More predictable, less magic

1. **Update DataService constructor** to take dataset parameter

   - `DataService(dataset="schwab-us-equity-1d-adjusted")`
   - No more hardcoded EQUITY assumptions

1. **Deprecate instrument.data_source usage in resolver**

   - Keep for backward compatibility
   - Log warning about using explicit dataset instead

1. **Update documentation** explaining the new pattern

### Long-Term (Phase 2)

1. **Implement InstrumentRegistry** for multi-source lookup
1. **Simplify Instrument model** (symbol only + optional overrides)
1. **Remove data_source/instrument_type from Instrument** (breaking change)
1. **Create MultiSourceDataService** for advanced use cases

______________________________________________________________________

## Summary

**Current Problem:**

- Config already specifies provider + asset_class
- Instrument duplicates this metadata
- DataService can't look it up (just hardcodes)
- Adapters don't use it (only use symbol)

**Root Cause:**

- Mixed responsibilities: config holds physical details, instrument holds logical details
- But the "logical" details (equity, schwab) are actually in the config too
- So we're specifying the same thing twice

**Solution:**

1. **Short-term:** Accept duplication, add helper to derive from config
1. **Medium-term:** Make dataset explicit, deprecate data_source inference
1. **Long-term:** Add registry for true multi-source support

**For your test script:** Use Option B (helper function) to derive Instrument from config:

```python
def create_test_instrument(symbol: str, config: Dict) -> Instrument:
    """Derive instrument metadata from config (single source of truth)."""
    return Instrument(
        symbol=symbol,
        instrument_type=InstrumentType(config["asset_class"]),
        data_source=DataSource(config["provider"]),
    )
```

This makes it clear that config is the source of truth, and we're just copying metadata for the Instrument API.
