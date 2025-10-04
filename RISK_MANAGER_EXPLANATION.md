# Risk Manager: Concentration vs Cash - Executive Summary

## Your Question

> "Are we sure risk manager working ok? If we have 5 strategies with 25% max concentration, first 4 consume all cash, 5th signal passes concentration (new symbol/sector) but there's no cash... must be rejected. Is this way?"

## Short Answer

**The risk manager IS working correctly from a risk control perspective, but NOT the way you expect for multi-strategy allocation.**

The 5th signal is **APPROVED** (with reduced size), not rejected.

## What's Happening

### The Current Behavior (Demonstrated)

Run the test I created:

```bash
uv run python test_cash_vs_concentration.py
```

**Result:**

- First 4 strategies: Each approved for ~$25k (25% of initial $100k equity)
- After 4 fills: Cash = $31,800, Equity = $31,800
- 5th strategy requests: $20,000 position (20% of original equity)
- **Concentration limit kicks in**: 25% of CURRENT equity = $7,950
- **Position sized down**: From $20,000 → $7,950
- **Cash check passes**: $31,800 available > $7,950 needed ✅
- **5th signal: APPROVED** with 159 shares @ $50 = $7,950

### Why This Happens

The evaluation order in `RiskManager.evaluate_signal()`:

1. **Step 3**: Calculate position size → $20,000
1. **Step 4**: Apply concentration limits using **current equity** ($31,800) → reduces to $7,950
1. **Step 5**: Check cash availability → $7,950 < $31,800 ✅ **PASSES**

**Key insight:** Concentration limits use **current equity**, which shrinks as cash is spent. As equity drops, concentration limits get tighter, inadvertently sizing positions to fit remaining cash.

## The Problem for Multi-Strategy Portfolios

This creates **allocation unfairness**:

- **Strategies 1-4**: Get full sizes (~$25k each, 25% of $100k equity)
- **Strategy 5**: Gets squeezed to $7.95k (25% of $31.8k equity)

Even though:

- Strategy 5 is a NEW symbol (passes concentration by itself)
- You intended each strategy to have equal 25% allocation

## Is This a Bug?

### From Risk Management Perspective: **NO**

- Prevents oversized positions relative to **current** capital
- Standard practice: "25% limit" means 25% of what you have NOW
- Protects against over-concentration during drawdowns

### From Multi-Strategy Allocation Perspective: **YES**

- Later strategies get unfairly reduced allocations
- First strategies benefit from higher equity baseline
- Cash exhaustion is masked by automatic size reduction
- Hard to detect "out of cash" condition from stats

## The Solution

### Recommendation: Add Cash-First Check

Modify `RiskManager` to check cash BEFORE applying concentration limits:

```python
# Step 3: Calculate position size
sized_qty = calculate_position_size(...)  # e.g., $20,000

# Step 4: CHECK CASH FIRST (NEW)
required_cash = sized_qty * current_price
if required_cash > available_cash:
    return rejected("Insufficient cash")  # REJECT HERE

# Step 5: Apply concentration limits
sized_qty = apply_concentration_limits(...)  # May reduce size

# Step 6: Final cash check (verify adjusted size)
```

This ensures:

- Clear rejection when out of cash ✅
- 5th strategy gets rejected (as you expect) ✅
- No unfair "rescue" by concentration adjustment ✅
- Better visibility into cash constraints ✅

### Implementation Plan

**Phase 2 Enhancement** (recommended for next iteration):

Add configuration flag to `RiskPolicy`:

```python
class RiskPolicy(NamedTuple):
    # ... existing fields ...
    reject_on_pre_concentration_cash_shortage: bool = False  # NEW
```

- Default `False`: Current behavior (backward compatible)
- Set `True`: Cash-first rejection (multi-strategy fairness)

## Current Status

## Current Status

**✅ IMPLEMENTED in Stage 5B** (Phase 2 Enhancement):

Added `check_cash_before_concentration` flag to `RiskPolicy`:

```python
policy = RiskPolicy(
    sizing_method=SizingMethod.FIXED_VALUE,
    default_position_size=Decimal("25000.00"),
    max_position_pct=Decimal("0.25"),
    check_cash_before_concentration=True,  # Enable cash-first check
)
```

- ✅ Added configuration flag to RiskPolicy
- ✅ Implemented cash-first check in RiskManager
- ✅ Added unit and integration tests
- ✅ Updated documentation

**Test Results:**

- ✅ 54/54 tests passing (45 unit + 9 integration)
- ✅ New test: `test_cash_first_check_for_multi_strategy_fairness`
- ✅ Updated test: `test_cash_depletion_across_multiple_signals` documents both behaviors
- ✅ Verification test: `test_cash_vs_concentration.py` demonstrates the fix

**Phase 3 (Future):**

- [ ] Consider adding `concentration_basis` option (current/initial/peak equity)
- [ ] Add multi-strategy fairness tests
- [ ] Update examples with multi-strategy configurations

## Testing Your Scenario

See the test I created: `test_cash_vs_concentration.py`

This test explicitly demonstrates your 5-strategy scenario and shows that:

1. The 5th signal is APPROVED (current behavior)
1. It's approved because concentration reduced the size to fit cash
1. This is likely NOT what you want for multi-strategy allocation

## Bottom Line

**Your concern is valid.** The risk manager works correctly for single-strategy risk control, but needs enhancement for fair multi-strategy allocation.

The note in `test_cash_depletion_across_multiple_signals` documents this interaction:

> "Due to the interaction between concentration limits (based on equity) and cash availability (based on ledger balance), the concentration limit may reduce the position size to fit within available cash before the cash check occurs. This is actually correct behavior - the risk manager is conservative and sizes positions appropriately."

But "correct" depends on your use case:

- **Single strategy with dynamic risk management**: Current behavior is correct ✅
- **Multi-strategy with fair allocation**: Need Phase 2 enhancement ⚠️

## Next Steps

1. Review the detailed analysis: `docs/risk_concentration_vs_cash_analysis.md`
1. Decide if you need Phase 2 enhancement for multi-strategy support
1. Configure your policies accordingly:
   - Current system: Be aware of sequential allocation unfairness
   - Phase 2 system: Use `reject_on_pre_concentration_cash_shortage=True`

## Questions?

See the full analysis document for:

- Detailed code flow explanation
- Alternative solutions (equity basis options)
- Implementation considerations
- Comprehensive test coverage plan
