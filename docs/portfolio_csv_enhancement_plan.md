# Portfolio CSV Enhancement Plan

## Current Issues Analysis

### Issue 1: Dividend Cash Credits Not Tracked ❌ MISSING

**Problem**: Dividends are processed and add cash to the portfolio, but the portfolio snapshot doesn't distinguish dividend cash credits from other cash movements.

**Evidence**:

```csv
# Your Expected CSV (Feb 8, 2019):
2/8/2019 0:00,AAPL,...,cash_credits=414.08,...

# Current Implementation:
2019-02-08,AAPL,...,cash_credits=0.0,...
```

**Root Cause**: We calculate `cash_change = current_cash - prev_cash` and separate into debits/credits, but we don't track the SOURCE of the cash change (dividend vs. trading).

______________________________________________________________________

### Issue 2: Stock Splits Not Handled ❌ CRITICAL BUG

**Problem**: When a 4:1 stock split occurs (2020-08-31), the position quantity is NOT adjusted.

**Evidence from Data**:

```csv
2020-08-28: Close=$499.23, 569 shares, CumulativePriceFactor=8.10
2020-08-31: Open=$127.67 (÷4), AdjustmentFactor=0.25 (4:1 split), CumulativePriceFactor=32.40
```

**What Should Happen**:

- Position qty: 569 → 2,276 shares (×4)
- Avg cost: $144.01 → $36.00 (÷4)
- Market value: UNCHANGED ($113,000)

**Current Behavior**: Position remains 569 shares, causing MASSIVE calculation errors!

______________________________________________________________________

### Issue 3: Dataset Structure Understanding ✅ CONFIRMED

**AlgoSeek Adjustment Model**:

```
Prices: Already adjusted for ALL historical events
CumulativePriceFactor: Multiplier to work backwards to original prices
AdjustmentFactor: Single-event adjustment on specific date
  - Cash Dividend: Factor < 1.0 (e.g., 0.9957 = $0.73 dividend)
  - Stock Split: Factor = split ratio (e.g., 0.25 = 4:1 split)
AdjustmentReason: "CashDiv" or "Subdiv"
```

**Calculation Examples**:

**Dividend (Feb 8, 2019)**:

```
Close price: $170.41 (already adjusted)
AdjustmentFactor: 0.995729
Dividend per share = Close × (1 - AdjustmentFactor) = $170.41 × 0.004271 = $0.7276
Total dividend = $0.7276 × 569 shares = $414.08
```

**Split (Aug 31, 2020)**:

```
AdjustmentFactor: 0.25 (4:1 split)
Split ratio = 1 / 0.25 = 4
New shares = 569 × 4 = 2,276
New avg cost = $144.01 / 4 = $36.00
```

______________________________________________________________________

## Recommended Solution

### Phase 1: Add Bar OHLC + Adjustment Data to Portfolio CSV ✅ FOUNDATION

**Add to each portfolio snapshot row**:

```csv
TradeDate,Ticker,Open,High,Low,Close,MarketHoursVolume,
CumulativePriceFactor,CumulativeVolumeFactor,AdjustmentFactor,AdjustmentReason,SecId,
Signal,Size,Initial_Cash,Cash_Credits,Cash_Debits,Final_Cash,
Initial_Portfolio_Value,Mark_To_Market_Adjustment,Final_Portfolio_Value,Positions
```

**Why**: Having raw bar data + adjustment factors in CSV allows:

1. Manual verification of calculations
1. Post-processing analysis of splits/dividends
1. Complete transparency of data used for each decision

**Changes Required**:

- Modify `backtest.py` snapshot collection to include bar OHLC data
- Modify `cli.py` CSV export to include all bar fields
- Store adjustment event info if available

______________________________________________________________________

### Phase 2: Track Dividend Source Separately ✅ ENHANCEMENT

**Current Logic**:

```python
# Calculate daily changes
cash_change = current_cash - prev_cash
cash_debits = min(0.0, cash_change)  # Negative only
cash_credits = max(0.0, cash_change)  # Positive only
```

**Problem**: This captures NET cash change but doesn't distinguish:

- Trading activity (buy/sell)
- Dividends received
- Commissions paid

**Solution**: Track cash sources separately:

```python
# Track cash movements by source
trading_cash_flow = sum(fill.cash_impact for fill in fills_this_bar)
dividend_cash_flow = sum(div_result['amount'] for div_result in dividend_results if div_result['processed'])
commission_total = sum(fill.fees for fill in fills_this_bar)

# Separate into categories
cash_debits_trading = min(0.0, trading_cash_flow)
cash_credits_trading = 0.0  # Sales not implemented yet
cash_credits_dividends = max(0.0, dividend_cash_flow)
cash_debits_commissions = -commission_total
```

**CSV Columns**:

```csv
Cash_Credits_Trading,Cash_Credits_Dividends,Cash_Debits_Trading,Cash_Debits_Commissions
```

______________________________________________________________________

### Phase 3: Implement Stock Split Handler ⚠️ CRITICAL

**Missing Component**: No split handling logic exists anywhere in codebase!

**Required Changes**:

**3.1. Create Split Processor** (similar to DividendProcessor):

```python
class SplitProcessor:
    """Handle stock split adjustments to positions."""

    def process_split(self, event: AdjustmentEvent) -> Dict[str, Any]:
        """
        Process a stock split event.

        Args:
            event: Adjustment event with AdjustmentFactor
                   (e.g., 0.25 for 4:1 split)

        Returns:
            Dict with processing results:
            - processed: bool
            - symbol: str
            - split_ratio: float (e.g., 4.0 for 4:1)
            - old_qty: Decimal
            - new_qty: Decimal
            - old_avg_cost: Decimal
            - new_avg_cost: Decimal
        """
        symbol = event.symbol
        position = self.portfolio.positions.get_position(symbol)

        if position.is_flat():
            return {"processed": False, "reason": "No position"}

        # Calculate split ratio: 0.25 factor = 4:1 split
        split_ratio = 1 / event.adjustment_factor

        # Adjust position
        old_qty = position.quantity
        old_avg_cost = position.avg_price

        new_qty = old_qty * Decimal(str(split_ratio))
        new_avg_cost = old_avg_cost / Decimal(str(split_ratio))

        # Update position directly
        position._quantity = new_qty
        position._avg_price = new_avg_cost

        return {
            "processed": True,
            "symbol": symbol,
            "split_ratio": split_ratio,
            "old_qty": old_qty,
            "new_qty": new_qty,
            "old_avg_cost": old_avg_cost,
            "new_avg_cost": new_avg_cost,
        }
```

**3.2. Integrate into Backtest Loop**:

```python
# Process adjustment events (dividends AND splits)
if self.adjustment_processor and bar.ts not in processed_adjustment_dates:
    # Get adjustment events for this date
    events = self.adjustment_processor.get_events(bar.ts)

    for event in events:
        if event.adjustment_reason == "CashDiv":
            result = self.dividend_processor.process_ex_date(bar.ts)
            dividend_results.extend(result)
        elif event.adjustment_reason == "Subdiv":
            result = self.split_processor.process_split(event)
            split_results.append(result)

    processed_adjustment_dates.add(bar.ts)
```

**3.3. Add Split Data to Snapshot**:

```csv
Split_Ratio,Pre_Split_Qty,Post_Split_Qty
```

______________________________________________________________________

### Phase 4: Enhanced Portfolio CSV Format 🎯 TARGET

**Complete CSV Structure**:

```csv
# Bar Data
TradeDate,Ticker,Open,High,Low,Close,MarketHoursVolume,
CumulativePriceFactor,CumulativeVolumeFactor,AdjustmentFactor,AdjustmentReason,SecId,

# Trading Activity
Signal,Order_ID,Order_Type,Order_Qty,Fill_Price,Fill_Qty,Commission,

# Cash Flow Detail
Initial_Cash,
Cash_Credits_Dividends,Cash_Credits_Sales,
Cash_Debits_Purchases,Cash_Debits_Commissions,
Final_Cash,

# Position Tracking
Initial_Portfolio_Value,Mark_To_Market_Adjustment,Final_Portfolio_Value,
Position_Qty,Position_Avg_Cost,Position_Market_Price,

# Split/Dividend Events
Split_Ratio,Dividend_Per_Share,

# Summary
Total_Value,Num_Positions
```

______________________________________________________________________

## Implementation Priority

### 🔥 P0 - CRITICAL (Must Fix Immediately)

**Phase 3: Stock Split Handler**

- Current backtest results are WRONG after 2020-08-31
- Buy-and-hold strategy shows -36% return but should show positive return
- Position quantity is incorrect (569 instead of 2,276)

### ⚠️ P1 - HIGH (Important for Accuracy)

**Phase 2: Dividend Source Tracking**

- Portfolio CSV shows cash movements but doesn't explain source
- Cash flow analysis requires manual cross-reference with fills.csv
- User explicitly requested separate dividend tracking

### ✅ P2 - MEDIUM (Nice to Have)

**Phase 1: Bar OHLC + Adjustment Data**

- Improves CSV transparency and auditability
- Allows post-processing analysis
- Matches user's expected format

______________________________________________________________________

## Testing Plan

### Test Case 1: Dividend Processing

```python
# Verify Feb 8, 2019 dividend
expected_dividend = 569 * 0.7276 = $414.08
assert portfolio_snapshot['cash_credits_dividends'] == 414.08
assert portfolio_snapshot['dividend_per_share'] == 0.7276
```

### Test Case 2: Stock Split Processing

```python
# Verify Aug 31, 2020 split
expected_new_qty = 569 * 4 = 2276
expected_new_avg_cost = 144.01 / 4 = 36.0025
assert position.quantity == 2276
assert position.avg_price == 36.0025
assert portfolio_value_unchanged  # Market value same before/after
```

### Test Case 3: End-to-End Accuracy

```python
# Run full backtest 2019-2023
# Verify final position qty accounts for split
# Verify final return is positive (not -36%)
```

______________________________________________________________________

## Expected Impact

### Before Fix (Current State)

```
Final Portfolio (Dec 2023):
  Position: 569 shares @ unknown price
  Cash: $18,057.31
  Equity: $45,665.19
  Total: $63,722.50
  P&L: -$36,277.50 (-36.28%) ❌ WRONG
```

### After Fix (Expected)

```
Final Portfolio (Dec 2023):
  Position: 2,276 shares (569 × 4 from split) @ ~$193 per share
  Cash: $18,057.31 + dividends received
  Equity: ~$439,000 (2,276 × $193)
  Total: ~$457,000
  P&L: ~+$357,000 (+357%) ✅ CORRECT
```

**Dividends Expected** (21 dividend events × ~$0.73/share × post-split shares):

```
~21 dividends × $0.73 × average shares held = ~$34,000 in dividends
```

______________________________________________________________________

## Next Steps

1. **Verify current split handling** - Check if AlgoSeekAdapter or Position class has any split logic
1. **Implement SplitProcessor** - Create new class to handle split events
1. **Test with buy_and_hold_strategy.py** - Verify correct handling through 2020-08-31
1. **Update portfolio CSV** - Add all requested fields
1. **Document dataset structure** - Create guide for AlgoSeek adjustment model

______________________________________________________________________

## Questions for User

1. **Split Implementation**: Should we modify Position directly or create a new "adjustment" system?
1. **Dividend Tracking**: Do you want dividend data from AlgoSeek adjustment factors OR from separate dividend calendar?
1. **CSV Format**: Your expected format has Signal/Size columns - should these show the ACTIVE signal or the fill that occurred?
1. **Historical Adjustments**: Should we verify CumulativePriceFactor calculations or trust AlgoSeek's adjusted prices?
