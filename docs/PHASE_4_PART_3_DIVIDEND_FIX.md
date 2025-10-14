# Critical Fix: Dividend Payment with Split-Adjusted Prices

## Problem Identified

**User Discovery**: Dividend cash payments were using split-adjusted dividend amounts, causing incorrect cash calculations when trading with adjusted prices.

## The Bug

### Scenario from golden_output_new.json

**Timeline**:

- **2020-08-01**: Buy 100 shares at $108.25 (adjusted price, post-split equivalent)
- **2020-08-07**: Dividend ex-date
  - Unadjusted: $0.82/share
  - Adjusted: $0.205/share (split-adjusted for future 4:1 split)
- **2020-08-31**: 4:1 split happens

**Wrong Calculation** (before fix):

```python
dividend_payment = bar.dividend * position.qty  # bar from adjusted mode
                 = $0.205 * 100
                 = $20.50 ❌
```

**Correct Calculation** (after fix):

```python
dividend_payment = unadjusted_bar.dividend * position.qty
                 = $0.82 * 100
                 = $82.00 ✓
```

## Why This Happens

### Price Adjustment Modes

The data layer provides three modes:

1. **Unadjusted**: Actual historical prices and dividends

   - 2020-08-07: close = $444.45, dividend = $0.82
   - 2020-08-31: close = $129.04 (after 4:1 split)

1. **Adjusted**: Backward-adjusted for splits

   - 2020-08-07: close = $111.11, dividend = $0.205 ($0.82 / 4)
   - 2020-08-31: close = $129.04 (same)

1. **Total Return**: Dividend reinvested automatically

   - No cash dividends, embedded in price

### The Mismatch

**Backtest was doing**:

- Trading: Uses `adjusted` prices ($111.11) ✓
- Positions: Based on adjusted price world (100 shares) ✓
- Dividends: Uses `adjusted` dividends ($0.205) ❌

**Problem**: Position quantities aren't split-adjusted!

- We buy 100 shares at $111.11 (adjusted)
- Those 100 shares represent 100 shares in the position tracker
- When dividend pays, it should be $0.82 × 100 (unadjusted), not $0.205 × 100

## The Fix

### What Changed

**Before** (line 115-122):

```python
while merger.has_next():
    symbol, multi_mode_bar = merger.get_next_bar()
    bar = multi_mode_bar.adjusted  # Only kept adjusted
    bars_list.append(bar)  # Lost access to unadjusted!
```

**After** (line 115-124):

```python
while merger.has_next():
    symbol, multi_mode_bar = merger.get_next_bar()
    bar = multi_mode_bar.adjusted
    bars_list.append((bar, multi_mode_bar))  # Keep both!
```

**Dividend Processing** (line 239-286):

```python
# Use UNADJUSTED dividend for cash payments
unadjusted_bar = multi_mode_bar.unadjusted

if unadjusted_bar.dividend is not None:
    position = ctx.portfolio.positions.get_position(bar.symbol)
    if position and not position.is_flat():
        if position.qty > 0:
            ctx.portfolio.apply_long_dividend(
                symbol=bar.symbol,
                dividend_per_share=unadjusted_bar.dividend,  # ← Real cash amount!
                ts=bar.ts,
            )
```

### Why This Works

1. **Strategy gets adjusted prices**: `bar` from `multi_mode_bar.adjusted`

   - Indicator calculations use split-adjusted prices
   - Backtests are consistent across time

1. **Dividends use unadjusted amounts**: `unadjusted_bar.dividend`

   - Real cash payment at that point in time
   - Example: Company pays $0.82/share on 2020-08-07
   - You own 100 shares → You get $82, regardless of future splits

1. **Position quantities match**: `position.qty`

   - When you buy 100 shares, position.qty = 100
   - No position adjustment for splits (we removed SplitProcessor)
   - Dividend payment: $0.82 × 100 = $82 ✓

## Verification with Golden Output

### Data from golden_output_new.json

**Dividend Event (2020-08-07)**:

```json
{
  "unadjusted": {
    "trade_datetime": "2020-08-07",
    "close": 444.45,
    "dividend": "0.8200"  ← Actual payment
  },
  "adjusted": {
    "trade_datetime": "2020-08-07",
    "close": 111.11,
    "dividend": "0.2050000000000000439285714286"  ← Split-adjusted
  }
}
```

**Split Event (2020-08-31)**:

```json
{
  "metadata": {
    "splits": [
      {
        "date": "2020-08-31",
        "ratio": "4.00:1",
        "factor": 4.0
      }
    ]
  }
}
```

### Example Calculation

**Scenario**: Buy 100 shares on 2020-08-03

**Trade Execution** (uses adjusted):

- Price: $108.25 (adjusted)
- Shares: 100
- Cost: $10,825

**Dividend Payment** (2020-08-07):

- Unadjusted dividend: $0.82/share
- Position: 100 shares
- **Payment: $0.82 × 100 = $82.00** ✓

**After Split** (2020-08-31):

- No position adjustment needed
- Trading continues with adjusted prices
- Future dividends still use unadjusted amounts

## Impact on Portfolio Snapshots

### Before (incorrect)

```python
snapshot = {
    "dividend_per_share": 0.205,  # Split-adjusted (wrong for cash payment)
}
```

### After (correct)

```python
snapshot = {
    "dividend_per_share_unadjusted": 0.82,  # Actual cash payment ✓
    "dividend_per_share_adjusted": 0.205,   # For analysis
}
```

## Other Considerations

### What About Splits?

**Question**: Don't positions need split adjustment?

**Answer**: No, because we're using adjusted prices throughout:

- Buy 100 shares at $108.25 (adjusted) = $10,825
- After 4:1 split: Still worth ~$10,825 in adjusted-price world
- No position quantity adjustment needed
- Prices stay consistent in adjusted mode

### What About Executions?

**Current**: All executions use adjusted prices from `bar`

- This is correct for strategy decisions
- Commissions calculated on adjusted notional
- No issue with split timing

**Future** (Phase 4 Part 4): Executions might use unadjusted prices

- More realistic execution simulation
- Slippage based on actual historical prices
- Will require execution engine updates

## Files Modified

### Core Change

- `src/qtrader/api/backtest.py`: Keep both adjusted and unadjusted bars in loop
  - Line 113-124: Store tuples `(adjusted_bar, multi_mode_bar)`
  - Line 230-234: Unpack both in main loop
  - Line 239-286: Use `multi_mode_bar.unadjusted.dividend` for cash
  - Line 363-367: Log both dividend amounts in snapshots

### Changes Summary

- **+15 lines**: Keep MultiModeBar references
- **+8 lines**: Extract unadjusted dividend
- **+5 lines**: Enhanced logging with both dividend amounts
- **Changed logic**: Use unadjusted dividend for cash, adjusted for strategy

## Testing Recommendations

### Unit Test Scenario

```python
def test_dividend_payment_with_future_split():
    """Verify dividend uses unadjusted amount even with future splits."""
    # Create bars with split AFTER dividend
    bars = [
        # Buy on Aug 3 (adjusted price)
        create_bar("2020-08-03", close=108.25),

        # Dividend on Aug 7 (pre-split)
        create_bar_with_dividend(
            date="2020-08-07",
            close_adjusted=111.11,
            close_unadjusted=444.45,
            dividend_unadjusted=0.82,
            dividend_adjusted=0.205,
        ),

        # Split on Aug 31
        create_bar("2020-08-31", close=129.04),  # Post-split price
    ]

    # Buy 100 shares on Aug 3
    strategy.on_bar(bars[0])  # Generates BUY signal
    # position.qty = 100

    # Process dividend on Aug 7
    backtest.process_dividend(bars[1])

    # Verify payment
    dividend_payment = portfolio.cash.get_transactions()[-1].amount
    assert dividend_payment == Decimal("82.00")  # $0.82 × 100 shares
    # NOT $20.50 ($0.205 × 100)
```

### Integration Test

Update existing tests that verify dividend payments:

- Check they use unadjusted amounts
- Verify snapshots show both amounts
- Confirm cash ledger has correct entries

## Migration Notes

### For Test Writers

**Old approach** (broken):

```python
# Tests were checking adjusted dividend
assert bar.dividend == Decimal("0.205")
assert payment == Decimal("20.50")  # Wrong!
```

**New approach** (correct):

```python
# Tests must check unadjusted dividend
assert multi_mode_bar.unadjusted.dividend == Decimal("0.82")
assert payment == Decimal("82.00")  # Correct!
```

### For Strategy Writers

**No changes needed!**

- Strategies still receive adjusted bars
- All prices are split-adjusted
- Indicators work as before
- Dividend cash appears automatically in portfolio

## Conclusion

This was a **critical bug** that would have caused:

- ❌ Incorrect cash balances (4x too low for 4:1 splits)
- ❌ Wrong portfolio equity calculations
- ❌ Invalid backtest results
- ❌ Misleading performance metrics

The fix ensures:

- ✅ Correct dividend cash payments
- ✅ Accurate portfolio tracking
- ✅ Proper cash ledger entries
- ✅ Valid backtest results

**Credit**: User identified this edge case by carefully analyzing the golden output data and recognizing the split-adjustment timing issue. Excellent attention to detail!

______________________________________________________________________

**Status**: ✅ Fixed\
**Lint**: ✅ Passing\
**Tests**: ⏳ Integration tests need verification\
**Phase**: 4 Part 3 (Dividend Fix)
