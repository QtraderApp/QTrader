# Price Series Architecture

## Overview

This document describes the architecture for handling price data from multiple vendors with different adjustment methodologies.

## Core Principles

1. **Vendor Agnostic Canonical Models**: Final consumption models (`CanonicalBar`, `CanonicalPriceSeries`) contain no vendor-specific fields
1. **Series-Level Processing**: Adjustments computed on entire series, not bar-by-bar (backward adjustment from last bar)
1. **Three Adjustment Modes**: All systems produce three price series: `unadjusted`, `adjusted`, `total_return`

## Data Models

### CanonicalBar (Vendor Agnostic)

```python
class CanonicalBar:
    trade_datetime: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    dividend: Optional[Decimal]  # Split-adjusted dividend per share
```

**Key Points:**

- NO `split_ratio` field - splits are baked into prices based on mode
- `dividend` is split-adjusted (e.g., $1 dividend before 2:1 split shows as $0.50)
- Represents a bar in ONE specific adjustment mode

### CanonicalPriceSeries (Vendor Agnostic)

```python
class CanonicalPriceSeries:
    mode: str  # "unadjusted" | "adjusted" | "total_return"
    instrument: Instrument
    bars: list[CanonicalBar]
```

**Adjustment Modes:**

- **`unadjusted`**: Raw prices as traded, volume as reported, dividends as paid
- **`adjusted`**: Split-adjusted prices & volume, dividends split-adjusted
- **`total_return`**: Split + dividend adjusted prices, no separate dividends (embedded)

### AlgoseekBar (Vendor Specific)

```python
class AlgoseekBar:
    TradeDate: str
    Ticker: str
    Open: float
    High: float
    Low: float
    Close: float
    MarketHoursVolume: int
    CumulativePriceFactor: float
    CumulativeVolumeFactor: float
    AdjustmentFactor: Optional[float]
    AdjustmentReason: Optional[str]
```

**Key Methods:**

- `validate_ohlc()`: Returns list of warnings (non-strict validation)
- `is_split()`: Check if bar has split event
- `is_dividend()`: Check if bar has dividend event
- `get_split_ratio()`: Extract split ratio from adjustment factor
- `get_dividend_amount()`: Extract dividend from adjustment factor & close price

### AlgoseekPriceSeries (Vendor Specific)

```python
class AlgoseekPriceSeries:
    instrument: Instrument
    bars: list[AlgoseekBar]
```

**Key Method:**

```python
def to_canonical_series() -> dict[str, CanonicalPriceSeries]:
    """
    Compute all three canonical series from raw Algoseek data.

    Returns:
        {
            "unadjusted": CanonicalPriceSeries(...),
            "adjusted": CanonicalPriceSeries(...),
            "total_return": CanonicalPriceSeries(...)
        }
    """
```

## Processing Flow

```
┌─────────────────────┐
│  Raw Vendor Data    │
│  (Database/Files)   │
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│  AlgoseekBar        │  ← Parse raw data
│  (Vendor Schema)    │     Validate OHLC (warnings)
└──────────┬──────────┘
           │
           v
┌─────────────────────┐
│ AlgoseekPriceSeries │  ← Collect bars
│ (Raw Unadjusted)    │     Validate series consistency
└──────────┬──────────┘
           │
           │ .to_canonical_series()
           v
┌─────────────────────────────────────────┐
│          3 CanonicalPriceSeries         │
│                                         │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐
│  │Unadjusted  │  │ Adjusted   │  │Total Return│
│  └────────────┘  └────────────┘  └────────────┘
│                                         │
│  Each contains list[CanonicalBar]      │
└─────────────────────────────────────────┘
           │
           v
┌─────────────────────┐
│   Backtester        │  ← Select mode via config
│   (Consumes one)    │     e.g., "adjusted"
└─────────────────────┘
```

## Adjustment Calculations (Algoseek)

### Backward Adjustment Approach

Adjustments are computed **backward** from the last (most recent) bar:

```python
last_price_factor = bars[-1].CumulativePriceFactor
last_volume_factor = bars[-1].CumulativeVolumeFactor

for bar in bars:
    # Adjusted (split-adjusted)
    vol_ratio = last_volume_factor / bar.CumulativeVolumeFactor
    adjusted_price = bar.Open * vol_ratio
    adjusted_volume = bar.MarketHoursVolume / vol_ratio

    # Total Return (split + dividend adjusted)
    price_ratio = last_price_factor / bar.CumulativePriceFactor
    total_return_price = bar.Open * price_ratio
```

### Why Backward?

- Most recent bar has no future adjustments → `factor = 1.0`
- Historical bars adjusted to be comparable with present
- Avoids "future bias" of knowing adjustments that hadn't happened yet
- Industry standard for backtesting

## Validation Strategy

### Vendor-Specific Validation (AlgoseekBar)

```python
def validate_ohlc(self) -> list[str]:
    """Return list of warnings, don't raise exceptions."""
    warnings = []

    if self.high < self.low:
        warnings.append(f"High < Low: {self.high} < {self.low}")

    # Use tolerance for minor violations
    tolerance = 0.01  # 1%
    if self.high < self.open * (1 - tolerance):
        warnings.append(f"High < Open (beyond tolerance)")

    return warnings
```

**Design Decision**: Return warnings list instead of raising exceptions

- Allows processing to continue with data quality issues
- Warnings logged for analysis
- Severe violations can still halt processing

### Series-Level Validation

```python
@model_validator(mode='after')
def validate_time_series(self) -> 'AlgoseekPriceSeries':
    """Validate series consistency."""
    # All bars same ticker
    # Chronological order
    # No duplicates
    # Etc.
```

## Usage Example

```python
# 1. Parse raw data
raw_data = fetch_from_database(ticker="AAPL", start="2020-08-01")
algoseek_bars = [AlgoseekBar(**row) for row in raw_data]

# 2. Create vendor-specific series
vendor_series = AlgoseekPriceSeries(
    instrument=Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK),
    bars=algoseek_bars
)

# 3. Compute all canonical series
canonical_series = vendor_series.to_canonical_series()

# 4. Select mode for backtesting
backtest_data = canonical_series["adjusted"]  # or "unadjusted" or "total_return"

# 5. Run backtest
backtester.run(price_series=backtest_data)
```

## Benefits

### Separation of Concerns

- Vendor logic isolated in vendor-specific classes
- Backtester only sees canonical models
- Easy to add new vendors (implement vendor bar + series)

### Series-Level Processing

- Correct handling of backward adjustments
- No bar-by-bar state management
- Validates entire series before processing

### Three Series Available

- Unadjusted: realistic fills, slippage analysis
- Adjusted: standard backtesting, comparability
- Total Return: performance attribution, benchmarking

### Data Quality

- Warnings instead of exceptions for minor violations
- Severe issues still halt processing
- Full audit trail of data quality issues

## Future Enhancements

1. **Other Vendors**: Create `<Vendor>Bar` and `<Vendor>PriceSeries` classes
1. **Caching**: Cache canonical series to avoid recomputation
1. **Partial Updates**: Incremental updates for new bars
1. **Quality Metrics**: Aggregate warning statistics per series
