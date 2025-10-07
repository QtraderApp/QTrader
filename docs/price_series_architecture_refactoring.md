# Price Series Architecture Refactoring

## Current Issues

### 1. Mixed Responsibilities

`AlgoseekPriceSeries.to_canonical_series()` handles three different concerns:

- Data transformation (unadjusted → canonical)
- Corporate action adjustment (splits)
- Financial computation (total return with dividend reinvestment)

### 2. Look-Ahead Bias in "Adjusted" Series

The backward adjustment for split-adjusted prices requires knowing ALL future corporate events. This is acceptable for historical analysis but creates conceptual confusion about "no look-ahead" bias.

### 3. Total Return is a Financial Strategy, Not a Data Model

Computing total return involves:

- Investment simulation logic (buy 1 share at t=0)
- Dividend reinvestment strategy
- Performance tracking

This belongs in a separate layer, not in the data model.

## Proposed Architecture

### Layer 1: Data Models (Pure Data Representation)

```
┌─────────────────────────────────────────────────────────────┐
│ Data Layer (Pydantic Models)                                │
│                                                              │
│  CanonicalBar                                               │
│  - Simple OHLCV container                                   │
│  - No business logic                                        │
│  - Validation only                                          │
│                                                              │
│  CanonicalPriceSeries                                       │
│  - mode: str (for labeling only)                           │
│  - bars: list[CanonicalBar]                                │
│  - Minimal validation                                       │
│                                                              │
│  AlgoseekBar (vendor-specific)                             │
│  - Raw vendor fields                                        │
│  - Extract corporate events (get_split_ratio, etc.)        │
│  - OHLC validation                                          │
│                                                              │
│  AlgoseekPriceSeries (vendor-specific)                     │
│  - bars: list[AlgoseekBar]                                 │
│  - NO transformation methods                                │
└─────────────────────────────────────────────────────────────┘
```

### Layer 2: Price Adjustment Service (Corporate Actions)

```
┌─────────────────────────────────────────────────────────────┐
│ Price Adjustment Service                                     │
│                                                              │
│  class PriceAdjuster:                                       │
│                                                              │
│    @staticmethod                                            │
│    def to_unadjusted(vendor_series) -> CanonicalPriceSeries│
│      - Direct mapping, no adjustment                        │
│                                                              │
│    @staticmethod                                            │
│    def to_split_adjusted(vendor_series, reference_date)    │
│      -> CanonicalPriceSeries                                │
│      - Backward adjustment to reference_date               │
│      - Uses CumulativeVolumeFactor                          │
│      - Explicit about look-ahead requirement                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer 3: Financial Computation Service (Investment Simulation)

```
┌─────────────────────────────────────────────────────────────┐
│ Total Return Calculator                                      │
│                                                              │
│  class TotalReturnCalculator:                               │
│                                                              │
│    @staticmethod                                            │
│    def compute_total_return(                                │
│        unadjusted_series: CanonicalPriceSeries,            │
│        corporate_events: list[CorporateEvent]               │
│    ) -> CanonicalPriceSeries:                               │
│      - Forward compounding (no look-ahead)                  │
│      - Simulates: buy 1 share at t=0                       │
│      - Dividends reinvested                                 │
│      - Splits automatically applied                         │
│      - Volume in starting-date units                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Layer 4: Corporate Events Extractor

```
┌─────────────────────────────────────────────────────────────┐
│ Corporate Events Service                                     │
│                                                              │
│  class CorporateEvent:                                      │
│    date: str                                                │
│    event_type: str  # "split" | "dividend"                 │
│    split_ratio: Optional[Decimal]                           │
│    dividend_amount: Optional[Decimal]                       │
│                                                              │
│  class CorporateEventExtractor:                             │
│                                                              │
│    @staticmethod                                            │
│    def extract_from_algoseek(                               │
│        bars: list[AlgoseekBar]                              │
│    ) -> list[CorporateEvent]:                               │
│      - Centralized event extraction                         │
│      - Reusable across adjustment methods                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Benefits of This Refactoring

### 1. Separation of Concerns

- **Data Models**: Pure data representation + validation
- **Adjusters**: Corporate action mathematics
- **Calculators**: Financial simulation logic
- **Extractors**: Vendor-specific event parsing

### 2. Testability

Each layer can be tested independently:

```python
# Test data model validation
bar = CanonicalBar(...)

# Test adjustment math
adjusted = PriceAdjuster.to_split_adjusted(series, reference_date="2024-12-31")

# Test total return logic
tr_series = TotalReturnCalculator.compute_total_return(unadjusted, events)
```

### 3. Flexibility

- Easy to add new adjustment methods (e.g., capital-adjusted with special dividends)
- Easy to add new vendor formats (just implement event extractor)
- Easy to modify Total Return strategy (e.g., dividend tax handling)

### 4. Clear Responsibility

```python
# Data layer: WHAT the data is
bar = CanonicalBar(...)

# Service layer: HOW to transform it
adjusted = PriceAdjuster.to_split_adjusted(...)

# Business layer: WHY we're transforming it (investment simulation)
total_return = TotalReturnCalculator.compute_total_return(...)
```

### 5. Explicit About Look-Ahead Bias

```python
# Clearly requires full series (look-ahead)
def to_split_adjusted(vendor_series, reference_date: str):
    """
    Compute split-adjusted prices relative to reference_date.

    **REQUIRES LOOK-AHEAD**: Must have complete corporate event history
    up to and including reference_date to correctly adjust historical prices.

    Use this for historical analysis, NOT for backtesting without proper
    handling of the look-ahead requirement.
    """
    ...

# Explicitly no look-ahead
def compute_total_return(unadjusted_series, corporate_events):
    """
    Compute total return series using forward compounding.

    **NO LOOK-AHEAD**: Processes bars sequentially, applying only events
    that have occurred up to the current bar. Suitable for backtesting.
    """
    ...
```

## Migration Path

### Phase 1: Extract Services (Non-Breaking)

1. Create `PriceAdjuster` class with static methods
1. Create `TotalReturnCalculator` class
1. Create `CorporateEventExtractor` class
1. Keep `to_canonical_series()` method, but delegate to services internally

### Phase 2: Update Callers

1. Update backtester to call services directly
1. Update examples to use service layer
1. Add deprecation warning to `to_canonical_series()`

### Phase 3: Remove Old Method

1. Remove `to_canonical_series()` from `AlgoseekPriceSeries`
1. Models become pure data containers

## Recommended File Structure

```
src/qtrader/
├── models/
│   ├── bar.py                    # CanonicalBar
│   ├── price_series.py           # CanonicalPriceSeries
│   └── vendors/
│       └── algoseek/
│           ├── bar.py            # AlgoseekBar
│           └── price_series.py   # AlgoseekPriceSeries (no business logic)
│
├── services/
│   ├── corporate_events/
│   │   ├── extractor.py          # CorporateEventExtractor
│   │   └── models.py             # CorporateEvent
│   │
│   ├── price_adjustment/
│   │   └── adjuster.py           # PriceAdjuster
│   │
│   └── total_return/
│       └── calculator.py         # TotalReturnCalculator
```

## Example Usage (After Refactoring)

```python
from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries
from qtrader.services.corporate_events import CorporateEventExtractor
from qtrader.services.price_adjustment import PriceAdjuster
from qtrader.services.total_return import TotalReturnCalculator

# 1. Load raw vendor data
vendor_bars = [AlgoseekBar(**row) for row in data]
vendor_series = AlgoseekPriceSeries(instrument=instr, bars=vendor_bars)

# 2. Extract corporate events (once)
events = CorporateEventExtractor.extract_from_algoseek(vendor_bars)

# 3. Generate series as needed
unadjusted = PriceAdjuster.to_unadjusted(vendor_series)
adjusted = PriceAdjuster.to_split_adjusted(vendor_series, reference_date="2024-12-31")
total_return = TotalReturnCalculator.compute_total_return(unadjusted, events)

# Clear separation of concerns!
```

## Open Questions

1. **Should we support "as-of" split adjustment?**

   - Current: Adjust to latest split (reference_date = last bar date)
   - Alternative: Allow user to specify any reference date
   - Use case: Compare prices "as of 2020-08-30" (pre-split basis)

1. **How to handle dividend tax in Total Return?**

   - Current: Assumes 100% reinvestment (no tax)
   - Future: Add `tax_rate` parameter?

1. **Should Total Return support different reinvestment strategies?**

   - Current: Reinvest immediately at close price
   - Alternative: Reinvest at open of next day, or with delay

1. **Volume adjustment in Total Return: always in starting-date units?**

   - Current: Yes (divide by cumulative_split_ratio)
   - Alternative: Offer both unadjusted and adjusted volume options?

## Decision Required

**Should we proceed with this refactoring before integrating into backtester?**

- ✅ **Pros**: Clean architecture, testable, flexible
- ⚠️ **Cons**: More upfront work, more files to manage

My recommendation: **Yes, refactor now** before the backtester depends on the current structure. It will be much harder to refactor later when other components depend on `to_canonical_series()`.
