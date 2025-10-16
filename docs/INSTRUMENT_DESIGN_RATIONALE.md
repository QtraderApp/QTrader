# Instrument Model Design - When Are Fields Actually Used?

## Question

Since adapters like `SchwabOHLCAdapter` only use `instrument.symbol`, why do we need `instrument_type` and `data_source`?

## Answer: Separation of Concerns

The `Instrument` class serves **different purposes at different layers**:

### Layer 1: Direct Adapter Usage (What You're Doing)

**File:** `scripts/test_schwab_aapl.py`

```python
# You already know the dataset config explicitly
config = resolver.sources["schwab-us-equity-1d-adjusted"]

# You create the adapter directly
instrument = Instrument(symbol="AAPL", instrument_type=InstrumentType.EQUITY, data_source=DataSource.SCHWAB)
adapter = SchwabOHLCAdapter(config=config, instrument=instrument)
```

**What adapter uses:**

- ✅ `instrument.symbol` - for API calls, cache directory, logging

**What adapter doesn't use:**

- ❌ `instrument.instrument_type` - not needed
- ❌ `instrument.data_source` - not needed (you already passed config)

**Conclusion: For direct adapter usage, only `symbol` matters.**

______________________________________________________________________

### Layer 2: DataSourceResolver (Indirect Adapter Usage)

**File:** `src/qtrader/adapters/resolver.py`

```python
# Strategy doesn't know which adapter to use
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)

# Resolver maps data_source enum → dataset config → adapter
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)  # Uses instrument.data_source!
```

**What resolver uses:**

- ✅ `instrument.data_source` - maps to `data_sources.yaml` entry
- ❌ `instrument.instrument_type` - not used (yet - see future plans)

**Code in resolver.py line 342:**

```python
def resolve(self, instrument: Instrument):
    source_name = instrument.data_source.value  # <-- USES data_source

    if source_name not in self.sources:
        # Try backward compatibility lookup
        matching_sources = [
            name for name, config in self.sources.items()
            if config.get("provider") == source_name
        ]
```

**Purpose:** Enables strategies to say "give me AAPL from Schwab" without knowing:

- Which specific dataset (`schwab-us-equity-1d-adjusted` vs `schwab-us-equity-1m-unadjusted`)
- Connection details
- Cache paths
- OAuth credentials

______________________________________________________________________

### Layer 3: DataService (High-Level API)

**File:** `src/qtrader/services/data/service.py` line 273

```python
def get_instrument(self, symbol: str) -> Instrument:
    """Build instrument for symbol."""
    instrument = Instrument(
        symbol=symbol,
        instrument_type=InstrumentType.EQUITY,  # <-- Currently hardcoded
        data_source=self._get_data_source(),    # <-- From config
    )

    # Log it
    logger.debug(
        "data_service.get_instrument",
        instrument_type=instrument.instrument_type,  # <-- USES for logging
        data_source=instrument.data_source,          # <-- USES for logging
    )
```

**What DataService uses:**

- ✅ `instrument.instrument_type` - logging, future registry lookup
- ✅ `instrument.data_source` - logging, passed to resolver

**TODO comment at line 272:**

```python
# TODO: Get instrument type and data source from registry in Phase 2
# For now, assume EQUITY and use config source
```

**Purpose:**

- Centralized instrument creation
- Instrument registry lookups (planned)
- Logging and observability

______________________________________________________________________

### Layer 4: Strategy Code (End User)

**File:** Strategy implementations

```python
class MyStrategy(Strategy):
    def configure(self):
        # User doesn't want to know about adapters, configs, etc.
        self.aapl = self.add_instrument("AAPL")  # That's it!
        self.tsla = self.add_instrument("TSLA")
```

**What strategy sees:**

- Just the symbol string
- DataService handles everything else

______________________________________________________________________

## Design Rationale

### Why Keep All Three Fields?

1. **Abstraction Layers:**

   - **Low-level** (adapters): Only need symbol
   - **Mid-level** (resolver): Need data_source for mapping
   - **High-level** (service): Need both for registry/logging

1. **Future-Proofing:**

   ```python
   # Phase 2 TODO: Instrument Registry
   class InstrumentRegistry:
       def lookup(self, symbol: str) -> Instrument:
           """
           Look up instrument metadata from database/config.

           Returns Instrument with correct:
           - instrument_type (EQUITY vs CRYPTO vs FUTURE)
           - data_source (best source for this symbol)
           - metadata (sector, exchange, etc.)
           """
   ```

1. **Multi-Asset Support:**

   ```python
   # Future: Different asset classes need different handling
   if instrument.instrument_type == InstrumentType.FUTURE:
       # Need contract specs, rollover dates
       pass
   elif instrument.instrument_type == InstrumentType.CRYPTO:
       # Need exchange info, trading pairs
       pass
   ```

1. **Data Source Routing:**

   ```python
   # Future: Smart routing based on symbol + type
   if symbol in ["BTC", "ETH"]:
       data_source = DataSource.BINANCE
   elif symbol in ["ES", "NQ"]:
       data_source = DataSource.FUTURES_DB
   else:
       data_source = DataSource.SCHWAB
   ```

______________________________________________________________________

## Simplified Instrument for Test Scripts?

**Yes!** You could create a simpler factory:

```python
def create_test_instrument(symbol: str) -> Instrument:
    """
    Create instrument for testing.

    Uses minimal fields since we're bypassing resolver.
    """
    return Instrument(
        symbol=symbol,
        instrument_type=InstrumentType.EQUITY,  # Doesn't matter for adapter
        data_source=DataSource.SCHWAB,          # Doesn't matter for adapter
    )

# Usage
instrument = create_test_instrument("AAPL")
adapter = SchwabOHLCAdapter(config=config, instrument=instrument)
```

**Or even simpler** - make fields optional with defaults:

```python
# Proposed change to Instrument class
class Instrument(NamedTuple):
    symbol: str
    instrument_type: InstrumentType = InstrumentType.EQUITY      # Default
    data_source: DataSource = DataSource.ALGOSEEK                # Default
    frequency: Optional[str] = None
    metadata: Dict[str, Any] = {}
```

Then your test becomes:

```python
instrument = Instrument("AAPL")  # That's it!
```

______________________________________________________________________

## Recommendation for Your Test Script

### Current (Explicit)

```python
instrument = Instrument(
    symbol="AAPL",
    instrument_type=InstrumentType.EQUITY,
    data_source=DataSource.SCHWAB,
)
```

**Pros:**

- Clear and explicit
- Shows all fields
- Educational for understanding the model

**Cons:**

- Verbose for simple tests
- Fields not actually used by adapter

### Alternative (Minimal)

```python
# Add helper function to test script
def test_instrument(symbol: str) -> Instrument:
    """Create instrument for direct adapter testing."""
    return Instrument(symbol, InstrumentType.EQUITY, DataSource.SCHWAB)

# Usage
instrument = test_instrument("AAPL")
adapter = SchwabOHLCAdapter(config=config, instrument=instrument)
```

**Pros:**

- Cleaner test code
- Still complete Instrument
- Documents that these fields aren't used here

______________________________________________________________________

## Summary

| Layer                          | Uses symbol? | Uses instrument_type? | Uses data_source? |
| ------------------------------ | ------------ | --------------------- | ----------------- |
| **Adapter** (Schwab, Algoseek) | ✅ Yes       | ❌ No                 | ❌ No             |
| **DataSourceResolver**         | ❌ No        | ❌ No                 | ✅ Yes (mapping)  |
| **DataService**                | ✅ Yes       | ✅ Yes (logging)      | ✅ Yes (logging)  |
| **Strategy**                   | ✅ Yes       | ❌ No (hidden)        | ❌ No (hidden)    |

**For your test script:** Only `symbol` matters since you're using adapters directly.

**For production code:** All three fields enable the abstraction layers.

**Best practice:** Keep all three fields, but document that only `symbol` is used by adapters.
