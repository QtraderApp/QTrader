# Phase 4 Part 3: Split Processing Implementation

**Date**: October 8, 2025\
**Status**: ✅ IMPLEMENTED (Needs Testing)

## Summary

Implemented Phase 2 architecture compliance for backtest execution:

- **Execution uses unadjusted prices** (realistic fills and commissions)
- **Strategy uses adjusted prices** (split-consistent indicators)
- **Split detection and processing** (updates position quantities and cost basis)
- **Dividend payments use unadjusted amounts** (matches unadjusted positions)

## Changes Made

### 1. Import SplitProcessor

**File**: `src/qtrader/api/backtest.py`

```python
from qtrader.execution.split_processor import SplitProcessor
```

### 2. Initialize SplitProcessor

**File**: `src/qtrader/api/backtest.py` (lines 64-66)

```python
# Split processor for handling corporate actions
self.split_processor: Optional[SplitProcessor] = None

# Track previous adjustment ratios for split detection
self._prev_adjustment_ratios: Dict[str, Decimal] = {}
```

**File**: `src/qtrader/api/backtest.py` (lines 148-150)

```python
# Initialize split processor
self.split_processor = SplitProcessor(ctx.portfolio.positions)
```

### 3. Extract Unadjusted Bars for Execution

**File**: `src/qtrader/api/backtest.py` (lines 238-244)

```python
for bar_idx in range(start_idx, len(bars_list)):
    bar, multi_mode_bar = bars_list[bar_idx]  # Unpack both

    # Extract unadjusted bar for execution (Phase 2 architecture)
    unadjusted_bar = multi_mode_bar.unadjusted
    next_unadjusted_bar = (
        bars_list[bar_idx + 1][1].unadjusted if bar_idx + 1 < len(bars_list) else None
    )
```

### 4. Split Detection and Processing

**File**: `src/qtrader/api/backtest.py` (lines 253-286)

```python
# Detect and process splits (before dividends and trading)
# Compare unadjusted/adjusted price ratio to detect splits
adjustment_ratio = unadjusted_bar.close / bar.close
prev_ratio = self._prev_adjustment_ratios.get(bar.symbol)

if prev_ratio is not None:
    # Check if ratio changed significantly (indicates split)
    ratio_change = adjustment_ratio / prev_ratio
    # Allow 0.5% tolerance for price movements
    if abs(ratio_change - Decimal("1")) > Decimal("0.005"):
        # Split detected!
        logger.info(
            "backtest.split_detected",
            symbol=bar.symbol,
            date=bar.ts.isoformat(),
            prev_ratio=float(prev_ratio),
            curr_ratio=float(adjustment_ratio),
            split_ratio=float(ratio_change),
        )

        # Process split (updates position qty and cost basis)
        split_result = self.split_processor.process_split(
            symbol=bar.symbol,
            adjustment_factor=Decimal("1") / ratio_change,  # Convert to AlgoSeek format
            current_price=unadjusted_bar.close,
        )

        if split_result.get("processed"):
            logger.info(
                "backtest.split_processed",
                symbol=bar.symbol,
                **split_result,
            )

# Store current ratio for next iteration
self._prev_adjustment_ratios[bar.symbol] = adjustment_ratio
```

### 5. Use Unadjusted for Execution

**File**: `src/qtrader/api/backtest.py` (lines 363-366)

```python
# Process bar through execution engine (Phase 2 architecture: use unadjusted)
# Execution uses unadjusted prices for realistic fills and commissions
# Positions track real share quantities (updated by split processing)
fills = self.execution_engine.on_bar(unadjusted_bar, next_bar=next_unadjusted_bar)
```

### 6. Updated Dividend Comment

**File**: `src/qtrader/api/backtest.py` (lines 288-292)

```python
# Process dividend cash payment (if bar has dividend on ex-date)
# Use UNADJUSTED dividend amount for cash payments (Phase 2 architecture)
# - Portfolio positions are in real shares (updated by split processing)
# - Dividends are actual dollars per share paid at that time
# - Example: After 4:1 split, 4 shares × $0.205 unadjusted = $0.82 total
if unadjusted_bar.dividend is not None:
```

## Accounting Example

### Timeline: AAPL 4:1 Split

**2020-08-01: Buy 1 share**

- Strategy sees: adjusted $125 (split-adjusted)
- Strategy signal: "Buy 1 share"
- Execution fills at: unadjusted $500 (actual traded price)
- Position: 1 share @ $500 cost basis
- Cash: -$500.50 (including commission)
- **Commission**: $500 × 0.001 = $0.50 ✓ (on actual price)

**2020-08-07: Dividend**

- Unadjusted bar: dividend = $0.82/share
- Position: 1 share
- Payment: 1 × $0.82 = $0.82
- Cash: +$0.82

**2020-08-31: Split 4:1 Detected**

- Adjustment ratio changes: 502/125.5 = 4.0 → 129/129 = 1.0
- Ratio change: 1.0 / 4.0 = 0.25 (25% = 4:1 split detected)
- SplitProcessor updates position:
  - Quantity: 1 → 4 shares
  - Cost basis: $500/share → $125/share
  - Total cost: $500 (preserved ✓)
- Position after split: 4 shares @ $125/share
- Market value: 4 × $129 = $516

**2020-09-20: Sell all**

- Position: 4 shares @ $125 cost basis
- Strategy sees: adjusted $130
- Strategy signal: "Sell 4 shares" (closes position)
- Execution fills: 4 shares @ $130 unadjusted
- Cash: +$520 (before commission)
- Commission: 4 × $130 × 0.001 = $0.52
- Net proceeds: $519.48

### Final Accounting

```
Cash Flow:
  Initial:     $10,000.00
  Buy:            -500.50  (1 share @ $500 + commission)
  Dividend:         +0.82  (1 share × $0.82)
  Sell:           +519.48  (4 shares @ $130 - commission)
  Final:       $10,019.80

Position Tracking:
  Buy:     1 @ $500
  Split:   4 @ $125 (cost basis preserved: $500 total)
  Sell:    0 (flat)

P&L:
  Buy cost:   -$500.50
  Sell:       +$519.48
  Dividend:      +0.82
  Net P&L:    +$19.80 ✓
```

## Phase 2 Architecture Compliance

### ✅ Verified

1. **Strategy uses adjusted**: `signals = self.strategy.on_bar(bar, ctx)` where `bar = multi_mode_bar.adjusted`
1. **Execution uses unadjusted**: `fills = self.execution_engine.on_bar(unadjusted_bar, next_bar=next_unadjusted_bar)`
1. **Dividends use unadjusted**: `if unadjusted_bar.dividend is not None`
1. **Split processing**: Detects and processes splits to update position quantities
1. **Commissions realistic**: Calculated on actual unadjusted prices
1. **Cost basis preserved**: Through split processing

### Configuration Honored

```yaml
data:
  signal_generation_mode: "adjusted"    # ✅ Strategy uses adjusted
  execution_mode: "unadjusted"          # ✅ Execution uses unadjusted
  performance_mode: "total_return"      # ⏳ Future: performance metrics
```

## Split Detection Logic

**Method**: Compare `unadjusted_price / adjusted_price` ratio between bars

```python
current_ratio = unadjusted_bar.close / adjusted_bar.close
prev_ratio = stored from previous bar

if abs(current_ratio / prev_ratio - 1.0) > 0.005:  # 0.5% tolerance
    # Split detected!
    split_ratio = current_ratio / prev_ratio
    process_split(split_ratio)
```

**Example**:

- Before split: $500 unadj / $125 adj = 4.0 ratio
- After split: $129 unadj / $129 adj = 1.0 ratio
- Change: 1.0 / 4.0 = 0.25 → 4:1 split detected

## Testing Status

### ✅ Lint

- All lint checks passing
- No unused imports
- No unused variables

### ⏳ Unit Tests

- 321 tests passing
- 16 tests failing (use old `Bar` type, need migration)
- 23 tests error (use old `Bar` type, need migration)

### ⏳ Integration Tests

- Need to create split accounting test
- Need to verify with real data containing splits
- Need to compare with golden output

## Next Steps

1. **Create split accounting test** with synthetic data
1. **Update failing tests** to use `CanonicalBar` instead of `Bar`
1. **Test with real AAPL data** around 2020-08-31 split
1. **Verify against golden output** from previous runs
1. **Document Phase 4 Part 3 completion**

## Files Modified

- `src/qtrader/api/backtest.py`: Main implementation (split detection, unadjusted execution)
- `tests/test_split_accounting.py`: New test file (needs fixes)

## Documentation Created

- `docs/PHASE_4_PART_3_SPLIT_IMPLEMENTATION.md`: This file
