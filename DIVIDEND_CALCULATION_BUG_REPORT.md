# 🐛 Dividend Calculation Bug Report

**Date**: October 8, 2025\
**Severity**: 🔴 **CRITICAL** - Incorrect dividend amounts in all output\
**Status**: ✅ **VERIFIED** - Root cause identified, fix not yet applied

______________________________________________________________________

## Executive Summary

The current implementation incorrectly calculates dividend amounts by treating Algoseek's `AdjustmentFactor` as a dollar amount, when it's actually a **price adjustment ratio**. This results in wrong dividend values in unadjusted, adjusted, and total return series.

**Example**: AAPL dividend on 2020-08-07

- ❌ Current output: $0.9982 (wrong)
- ✅ Correct value: $0.82 (official AAPL dividend)
- 📊 Error magnitude: 21.7% too high

______________________________________________________________________

## 1. Problem Description

### Current Implementation (WRONG)

**File**: `src/qtrader/models/vendors/algoseek/bar.py`

```python
def get_dividend_amount(self) -> Optional[Decimal]:
    """Extract dividend amount if AdjustmentReason indicates a dividend."""
    if self.is_dividend() and self.AdjustmentFactor:
        # ❌ WRONG: AdjustmentFactor is NOT a dollar amount
        return Decimal(str(self.AdjustmentFactor)).quantize(
            Decimal("0.0001"), rounding=ROUND_HALF_UP
        )
    return None
```

**Issue**: Treats `AdjustmentFactor = 0.998200215` as `$0.9982`

### What AdjustmentFactor Actually Represents

For dividends, `AdjustmentFactor` is a **price adjustment ratio**, not a dollar amount:

```
AdjustmentFactor = (Close[T-1] - Dividend) / Close[T-1]
```

Rearranging to solve for dividend:

```
Dividend = (1 - AdjustmentFactor) × Close[T-1]
```

______________________________________________________________________

## 2. Verification with Real Data

### Raw Algoseek Data (AAPL 2020-08-05 to 2020-08-10)

| Date           | Open   | High   | Low    | Close      | AdjFactor       | AdjReason   | CumPriceFactor |
| -------------- | ------ | ------ | ------ | ---------- | --------------- | ----------- | -------------- |
| 2020-08-03     | 433.00 | 446.54 | 431.57 | 435.75     | None            | None        | 8.085970041    |
| 2020-08-04     | 436.55 | 443.15 | 433.55 | 438.66     | None            | None        | 8.085970041    |
| 2020-08-05     | 437.75 | 441.53 | 435.59 | 440.25     | None            | None        | 8.085970041    |
| 2020-08-06     | 441.65 | 457.65 | 439.19 | **455.61** | None            | None        | 8.085970041    |
| **2020-08-07** | 453.00 | 454.70 | 441.18 | **444.45** | **0.998200215** | **CashDiv** | 8.100549288    |
| 2020-08-10     | 450.06 | 455.10 | 440.00 | 450.91     | None            | None        | 8.100549288    |

### Calculations

#### ❌ Current (Wrong) Method

```
dividend = AdjustmentFactor
dividend = 0.998200215
Result: $0.9982
```

#### ✅ Correct Method (User's Discovery)

```
dividend = (1 - AdjustmentFactor) × Close[T-1]
dividend = (1 - 0.998200215) × 455.61
dividend = 0.001799785 × 455.61
Result: $0.8200
```

#### 📊 Verification

- **Official AAPL dividend** (2020-08-07): **$0.82**
- **Our calculation**: **$0.82**
- **Match**: ✅ **PERFECT**

______________________________________________________________________

## 3. Key Insights

### Understanding AdjustmentFactor

The `AdjustmentFactor` field has **different semantics** depending on the corporate event type:

| Event Type   | AdjustmentFactor Meaning | Example                 |
| ------------ | ------------------------ | ----------------------- |
| **Split**    | Inverse of split ratio   | 4:1 split → 0.25        |
| **Dividend** | Price adjustment ratio   | $0.82 div → 0.998200215 |

**Same field, different semantics!**

### Ex-Dividend Date Semantics

```
T-1 (Aug 6): Close = $455.61  [Last day to own for dividend]
T   (Aug 7): AdjustmentFactor = 0.998200215  [Ex-dividend date]
             Dividend calculated from T-1 close
```

**Critical**: The dividend on date T is calculated from the **previous day's close** (T-1), not the current day's close (T).

**Why?** Shareholders who own the stock at the end of T-1 receive the dividend. The price adjusts downward on T to reflect the dividend payout.

### Price Impact Analysis

```
Close[T-1] - Close[T] = $455.61 - $444.45 = $11.16
Calculated dividend = $0.82
```

**Note**: The price drop ($11.16) ≠ dividend ($0.82) because market conditions also affect the price. The dividend adjustment is theoretical for historical price series.

______________________________________________________________________

## 4. Impact Assessment

### Affected Components

#### ✅ **ALREADY CORRECT** (No changes needed)

- `AlgoseekBar.is_dividend()` - Detection works correctly
- `AlgoseekBar.is_split()` - Detection works correctly
- Split ratio calculation - Correct (uses `1 / AdjustmentFactor`)

#### ❌ **INCORRECT** (Needs fix)

- `AlgoseekBar.get_dividend_amount()` - Returns wrong dollar amount
- `golden_output_new.json` - Contains wrong dividend value (0.9982)
- All test expectations using dividend amounts
- Documentation referencing $1.00 dividend

### Test Impact

All tests currently **PASS** because they were written against the wrong implementation:

```python
# Current test (wrong but passing)
assert bar.get_dividend_amount() == Decimal("1.00")  # Actually returns wrong value
```

After fix, these tests will **FAIL** until updated with correct expectations.

______________________________________________________________________

## 5. Mathematical Proof

### Algoseek's Price Adjustment Formula

For a dividend, prices before the ex-dividend date need to be adjusted downward:

```
AdjustedPrice[T-1] = UnadjustedPrice[T-1] × AdjustmentFactor
```

This ensures price continuity across the dividend event.

### Deriving the Dividend Formula

Given:

```
AdjustmentFactor = (Close[T-1] - Dividend) / Close[T-1]
```

Multiply both sides by `Close[T-1]`:

```
AdjustmentFactor × Close[T-1] = Close[T-1] - Dividend
```

Rearrange:

```
Dividend = Close[T-1] - (AdjustmentFactor × Close[T-1])
Dividend = Close[T-1] × (1 - AdjustmentFactor)
```

**Therefore**:

```
Dividend = (1 - AdjustmentFactor) × Close[T-1]
```

### Verification with AAPL Data

```
Close[T-1] = 455.61
AdjustmentFactor = 0.998200215
Dividend = (1 - 0.998200215) × 455.61
Dividend = 0.001799785 × 455.61
Dividend = 0.82000044365
Dividend ≈ $0.82 ✅
```

______________________________________________________________________

## 6. Required Changes

### Change 1: Fix `get_dividend_amount()` Method

**File**: `src/qtrader/models/vendors/algoseek/bar.py`

```python
def get_dividend_amount(self) -> Optional[Decimal]:
    """
    Extract dividend amount if AdjustmentReason indicates a dividend.

    For dividends, Algoseek's AdjustmentFactor represents a price adjustment ratio.
    The dividend amount must be calculated from the previous day's close price:

        Dividend = (1 - AdjustmentFactor) × Close[T-1]

    Where:
        - AdjustmentFactor appears on ex-dividend date (T)
        - Close[T-1] is the previous day's closing price
        - Dividend is paid to shareholders holding at end of T-1

    Returns:
        Decimal dividend amount per share in dollars, or None if no dividend
    """
    if self.is_dividend() and self.AdjustmentFactor:
        # CORRECT: Calculate from adjustment ratio
        # Note: This requires access to previous bar's close price
        # This method signature needs to accept previous_close parameter
        raise NotImplementedError(
            "get_dividend_amount() requires previous bar's close price. "
            "Use get_dividend_adjustment_factor() instead and calculate "
            "dividend at the series level where T-1 close is available."
        )
    return None

def get_dividend_adjustment_factor(self) -> Optional[Decimal]:
    """
    Get the dividend adjustment factor for this bar.

    Returns the raw AdjustmentFactor which can be used to calculate
    the dividend amount when combined with the previous close price:

        Dividend = (1 - factor) × Close[T-1]

    Returns:
        Decimal adjustment factor, or None if no dividend
    """
    if self.is_dividend() and self.AdjustmentFactor:
        return Decimal(str(self.AdjustmentFactor))
    return None
```

### Change 2: Update Series-Level Dividend Calculation

**File**: `src/qtrader/models/vendors/algoseek/series.py`

The dividend calculation must happen at the **series level** where we have access to the previous bar:

```python
def _extract_corporate_events(self) -> None:
    """Extract corporate events from bars."""
    prev_bar = None

    for bar in self.bars:
        if bar.is_dividend():
            if prev_bar is None:
                raise ValueError(
                    f"Cannot calculate dividend for first bar on {bar.TradeDate}: "
                    f"requires previous close price"
                )

            # CORRECT: Use previous close
            adjustment_factor = Decimal(str(bar.AdjustmentFactor))
            dividend_amount = (Decimal("1") - adjustment_factor) * Decimal(str(prev_bar.Close))

            self.dividends.append(
                DividendEvent(
                    date=bar.TradeDate.date(),
                    amount=dividend_amount.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
                )
            )

        if bar.is_split():
            # Split calculation remains unchanged
            split_ratio = Decimal("1") / Decimal(str(bar.AdjustmentFactor))
            self.splits.append(
                SplitEvent(
                    date=bar.TradeDate.date(),
                    ratio=split_ratio.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                )
            )

        prev_bar = bar  # Track previous bar
```

### Change 3: Update Tests

All test expectations for dividend amounts need updating. Example:

```python
# OLD (wrong)
assert dividend_event.amount == Decimal("1.00")

# NEW (correct - for AAPL 2020-08-07)
# Dividend = (1 - 0.998200215) × 455.61 = 0.82
assert dividend_event.amount == Decimal("0.82")
```

### Change 4: Regenerate Golden Output

```bash
python src/qtrader/1playground/generate_golden_output.py
```

Expected changes in `golden_output_new.json`:

```json
{
  "trade_datetime": "2020-08-07",
  "dividend": "0.8200"  // Changed from "0.9982"
}
```

______________________________________________________________________

## 7. Testing Strategy

### Phase 1: Verification Tests (Before Fix)

```bash
# Create verification script
python scripts/verify_dividend_bug.py
```

Should output:

```
❌ Current: $0.9982
✅ Correct: $0.82
Error: 21.7%
```

### Phase 2: Apply Fix

1. Update `AlgoseekBar.get_dividend_amount()` → remove or deprecate
1. Add `AlgoseekBar.get_dividend_adjustment_factor()`
1. Update `AlgoseekPriceSeries._extract_corporate_events()`
1. Update all tests with correct values

### Phase 3: Validation

```bash
# Run all tests (will fail initially)
pytest tests/unit/models/test_vendors_algoseek.py -v

# Update test expectations
# Re-run tests (should pass)

# Regenerate golden output
python src/qtrader/1playground/generate_golden_output.py

# Validate against known AAPL dividend
python src/qtrader/1playground/test_validation_script.py
```

### Phase 4: Integration Tests

```bash
# Full test suite
pytest tests/ -v

# Check all corporate event scenarios
pytest tests/integration/test_data_layer_corporate_events.py -v
```

______________________________________________________________________

## 8. Recommendations

### Immediate Actions

1. ✅ **VERIFIED**: Root cause confirmed with real data
1. 🔄 **PENDING**: Apply fix to `AlgoseekBar` and `AlgoseekPriceSeries`
1. 🔄 **PENDING**: Update all test expectations
1. 🔄 **PENDING**: Regenerate golden output
1. 🔄 **PENDING**: Update documentation

### Architecture Improvement

**Current problem**: `get_dividend_amount()` on a single bar cannot calculate dividend correctly because it needs the previous bar's close.

**Solution**: Move dividend calculation to series level where we have sequential access to bars.

```python
# Bar level: Return adjustment factor only
def get_dividend_adjustment_factor(self) -> Optional[Decimal]:
    """Returns the raw adjustment factor for dividend calculation."""

# Series level: Calculate actual dividend amount
def _extract_corporate_events(self) -> None:
    """Calculate dividends using prev_bar.Close."""
```

This matches the **semantic truth**: dividends are a property of the **time series**, not individual bars.

______________________________________________________________________

## 9. Documentation Updates Needed

### Files to Update

1. `docs/DATA_LAYER_TEST_COVERAGE.md`

   - Update dividend calculation formula
   - Update test expectations

1. `DATA_LAYER_VALIDATION_SUMMARY.md`

   - Correct the bug fix description
   - Update validation metrics

1. `src/qtrader/1playground/README.md`

   - Update dividend formula

1. `QUICK_REFERENCE.md`

   - Correct dividend extraction example

### Docstring Updates

Update all docstrings that mention dividend calculation to use correct formula:

```python
"""
For dividends:
    Dividend = (1 - AdjustmentFactor) × Close[T-1]

Where:
    - T is the ex-dividend date (when AdjustmentFactor appears)
    - Close[T-1] is the previous day's closing price
    - AdjustmentFactor is a price adjustment ratio (not dollar amount)
"""
```

______________________________________________________________________

## 10. Acceptance Criteria

✅ Fix is complete when:

1. [ ] `get_dividend_amount()` correctly calculates dividend from previous close
1. [ ] All unit tests pass with correct expectations
1. [ ] Integration tests pass with real AAPL data
1. [ ] Golden output shows `$0.82` for AAPL 2020-08-07 dividend
1. [ ] Validation script confirms 100% match
1. [ ] Documentation updated with correct formula
1. [ ] No regression in split ratio calculation

______________________________________________________________________

## Appendix A: Complete Test Case

```python
def test_aapl_dividend_2020_08_07():
    """Test AAPL dividend calculation with real data."""

    # Setup
    bars = load_aapl_bars_2020_08_03_to_2020_08_10()
    series = AlgoseekPriceSeries(bars=bars)

    # Find dividend event
    dividend_events = [d for d in series.dividends if d.date == date(2020, 8, 7)]
    assert len(dividend_events) == 1

    dividend = dividend_events[0]

    # Verify
    assert dividend.amount == Decimal("0.82")  # Official AAPL dividend
    assert dividend.date == date(2020, 8, 7)   # Ex-dividend date

    # Verify calculation matches formula
    aug_6_bar = [b for b in bars if b.TradeDate.date() == date(2020, 8, 6)][0]
    aug_7_bar = [b for b in bars if b.TradeDate.date() == date(2020, 8, 7)][0]

    expected = (Decimal("1") - Decimal(str(aug_7_bar.AdjustmentFactor))) * Decimal(str(aug_6_bar.Close))
    assert abs(dividend.amount - expected) < Decimal("0.01")
```

______________________________________________________________________

## Appendix B: Formula Reference Card

### Dividend Calculation

```
Dividend = (1 - AdjustmentFactor) × Close[T-1]

Where:
  T = Ex-dividend date (current bar)
  T-1 = Previous trading day
  AdjustmentFactor = Price adjustment ratio on date T
  Close[T-1] = Closing price on date T-1
```

### Split Calculation (Unchanged)

```
SplitRatio = 1 / AdjustmentFactor

Examples:
  4:1 forward split: AdjustmentFactor = 0.25 → Ratio = 4.0
  1:5 reverse split: AdjustmentFactor = 5.0 → Ratio = 0.2
```

### Why Different?

- **Splits**: `AdjustmentFactor` IS the inverse of split ratio (by design)
- **Dividends**: `AdjustmentFactor` IS the price adjustment ratio (needs calculation)

**Root cause**: Same field (`AdjustmentFactor`), different semantics based on `AdjustmentReason`.

______________________________________________________________________

**Status**: ✅ Analysis complete, ready for implementation\
**Next Step**: Apply fixes to codebase\
**Estimated Impact**: ~2-3 hours (code + tests + documentation)
