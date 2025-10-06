# Stage 6B: Shorting & Accruals - Implementation Plan

**Status:** In Progress **Start Date:** October 6, 2025 **Estimated Completion:** 6-8 hours

## Overview

Complete the shorting and accruals system by integrating dividend event detection and processing into the backtest loop.

## ✅ Already Implemented (Stages 1-5)

### Portfolio (models/portfolio.py)

- ✅ `apply_short_dividend()` - Debit cash for short positions
- ✅ `apply_borrow_cost()` - Daily borrow cost accrual
- ✅ Tests: `test_portfolio_short_dividend`, `test_portfolio_borrow_cost`

### Risk Management (risk/)

- ✅ `RiskPolicy.allow_shorting` - Enable/disable shorts
- ✅ `RiskManager.evaluate_signal()` - Reject shorts when disabled
- ✅ Short position sizing and validation
- ✅ Tests: `test_reject_short_when_not_allowed`

### Execution (execution/)

- ✅ `ExecutionConfig.borrow_rate_annual` - Configurable borrow rate
- ✅ `ExecutionEngine.on_end_of_day()` - Calls apply_borrow_cost
- ✅ Short fill handling (SELL orders create/increase short positions)

### Data (data/)

- ✅ `AdjustmentEvent` model for corporate actions
- ✅ Algoseek adapter parses adjustment metadata
- ✅ CSV adapter supports adjustment events

## ❌ Missing Components for Stage 6B

### 1. Dividend Calculator (NEW)

**File:** `src/qtrader/execution/dividend_calculator.py`

**Purpose:** Calculate dividend per share from price factors

**Classes:**

```python
class DividendCalculator:
    """Calculate dividend amount from adjustment factors."""

    @staticmethod
    def calculate_from_factors(
        close_before: Decimal,
        close_after: Decimal,
        price_factor: Decimal,
    ) -> Decimal:
        """
        Calculate dividend per share from adjustment.

        Formula: div = close_after * (price_factor - 1)

        Where price_factor = close_before / close_after
        """
```

**Tests:** `tests/unit/execution/test_dividend_calculator.py`

### 2. Dividend Processor (NEW)

**File:** `src/qtrader/execution/dividend_processor.py`

**Purpose:** Detect and process dividend events during backtest

**Classes:**

```python
class DividendProcessor:
    """Process dividend events during backtest."""

    def __init__(self, portfolio: Portfolio, adjustment_events: Dict[str, List[AdjustmentEvent]]):
        self.portfolio = portfolio
        self.events_by_date = self._index_by_date(adjustment_events)

    def process_ex_date(self, ts: datetime) -> List[Dict]:
        """
        Process dividend ex-dates for current bar.

        Returns list of processed dividend events.
        """
```

**Tests:** `tests/unit/execution/test_dividend_processor.py`

### 3. Backtest Integration (MODIFY)

**File:** `src/qtrader/api/backtest.py`

**Changes:**

1. Accept `adjustment_events` from data adapter
1. Initialize `DividendProcessor` if events present
1. Call `dividend_processor.process_ex_date()` before EOD
1. Track dividend metadata in results

**New Method:**

```python
def _process_dividends(self, bar: Bar) -> List[Dict]:
    """Process dividend ex-dates for current bar."""
    if self.dividend_processor:
        return self.dividend_processor.process_ex_date(bar.ts)
    return []
```

### 4. Integration Tests (NEW)

**File:** `tests/integration/test_shorting_accruals.py`

**Test Scenarios:**

1. Short dividend debited on ex-date
1. Borrow cost accrues daily on shorts
1. Combined: borrow + dividend on same day
1. Long position → no dividend debit
1. No position → no dividend debit
1. Multiple short positions with dividends

## Implementation Order

### Phase 1: Dividend Calculator (1 hour) ✅ COMPLETE

- [x] Create `DividendCalculator` class
- [x] Implement `calculate_from_factors()` method
- [x] Add validation and error handling
- [x] Write 20 unit tests (all passing)
- [x] Formula corrected: `div = close_after * (cum_price_factor - 1)`
- [x] Commit: "feat(execution): Add dividend calculator with adjustment factors"

### Phase 2: Dividend Processor (2 hours) ✅ COMPLETE

- [x] Create `DividendProcessor` class
- [x] Implement event indexing by date
- [x] Implement `process_ex_date()` method
- [x] Add logging for dividend processing
- [x] Write 17 unit tests (all passing)
- [x] Handles multiple dividends on same date
- [x] Tracks processing statistics
- [x] Commit: "feat(execution): Add dividend processor for ex-date handling"
- [ ] Test with multiple symbols

### Phase 3: Backtest Integration (2 hours)

- [ ] Modify backtest to accept adjustment events
- [ ] Initialize dividend processor
- [ ] Add dividend processing to event loop
- [ ] Track dividend metadata in results
- [ ] Update backtest tests

### Phase 4: Integration Tests (2 hours)

- [ ] Create end-to-end shorting tests
- [ ] Test dividend + borrow cost scenarios
- [ ] Test with real data (AAPL, MSFT dividends)
- [ ] Verify cash flows are correct
- [ ] Test edge cases (cover before ex-date, etc.)

### Phase 5: Documentation (1 hour)

- [ ] Document dividend calculation method
- [ ] Add shorting examples
- [ ] Update architecture diagrams
- [ ] Create Stage 6B completion summary

## Technical Specifications

### Dividend Calculation Formula

**From Algoseek Adjustment Factors:**

```
Given:
- close_before: Close price day before ex-date
- close_after: Close price on ex-date
- cum_price_factor: Cumulative adjustment factor (= close_before / close_after)

Calculation:
div_per_share = close_after * (cum_price_factor - 1)

Explanation:
- cum_price_factor represents the ratio: close_before / close_after
- When a dividend is paid, close_after drops by the dividend amount
- (factor - 1) gives the fractional dividend yield
- Multiplying by close_after recovers the dividend amount

Example (AAPL):
- 2023-02-09 (before): $152.55
- 2023-02-10 (ex-date): $152.32
- cum_price_factor: 1.001508 (= 152.55 / 152.32)
- div = 152.32 * (1.001508 - 1) = 152.32 * 0.001508 = $0.23/share
```

### Ex-Date Processing Flow

```python
# In backtest event loop
for bar in bars:
    # 1. Process bar (strategy, execution)
    strategy.on_bar(bar, ctx)
    engine.evaluate_intrabar(bar)
    engine.end_of_bar(bar)

    # 2. Apply fills
    portfolio.apply_fills()

    # 3. Process dividends (BEFORE EOD)
    dividends = dividend_processor.process_ex_date(bar.ts)

    # 4. Accrue borrow costs (EOD)
    portfolio.apply_borrow_cost(borrow_rate, bar.ts)

    # 5. Snapshot
    results.snapshot(bar.ts)
```

### Error Handling

**Missing Data:**

- If adjustment event has no price factors → log warning, skip
- If symbol not in portfolio → skip (no short position)
- If dividend calculation yields negative → log error, skip

**Validation:**

- Dividend amount must be positive
- Ex-date must be valid datetime
- Symbol must exist in universe

## Success Criteria

### Functional Requirements

- [x] Borrow costs accrue daily on shorts (already implemented)
- [ ] Short dividends debited on ex-date
- [ ] Dividend amount calculated from adjustment factors
- [ ] No dividends debited for long positions
- [ ] No dividends debited when no position

### Test Coverage

- [x] Portfolio borrow cost tests (existing)
- [x] Portfolio short dividend tests (existing)
- [ ] Dividend calculator tests (10+ tests)
- [ ] Dividend processor tests (15+ tests)
- [ ] Integration tests (6+ scenarios)

### Performance

- Dividend processing < 1ms per bar
- No memory leaks with large event sets
- Efficient date-based event lookup

### Documentation

- Dividend calculation methodology
- Shorting best practices
- Example strategies with shorts
- Stage 6B completion summary

## Dependencies

**External:**

- None (all internal)

**Internal:**

- Stage 1: Data adapters (AdjustmentEvent) ✅
- Stage 2: Portfolio, Position ✅
- Stage 3-4: Execution engine ✅
- Stage 5: Risk management ✅

## Testing Strategy

### Unit Tests

- **Dividend Calculator:** 10 tests

  - Valid calculation with factors
  - Zero dividend cases
  - Edge cases (small adjustments)
  - Validation errors

- **Dividend Processor:** 15 tests

  - Event indexing
  - Ex-date detection
  - Short position filtering
  - Multiple symbols
  - No events case

### Integration Tests

- **Full Workflow:** 6+ tests
  - Short dividend + borrow cost
  - Multiple ex-dates
  - Real AAPL data (20 dividends)
  - Real MSFT data (20 dividends)
  - Cover position before ex-date
  - Flat position (no dividend)

### Regression Tests

- Run all 402 existing tests
- Verify no breaking changes
- Maintain 87%+ coverage

## Timeline

| Phase     | Duration    | Deliverable                |
| --------- | ----------- | -------------------------- |
| Phase 1   | 1 hour      | DividendCalculator + tests |
| Phase 2   | 2 hours     | DividendProcessor + tests  |
| Phase 3   | 2 hours     | Backtest integration       |
| Phase 4   | 2 hours     | Integration tests          |
| Phase 5   | 1 hour      | Documentation              |
| **Total** | **8 hours** | **Stage 6B Complete**      |

## Post-Implementation

### Validation

1. Run full test suite (should have 440+ passing)
1. Verify coverage stays above 85%
1. Run pre-commit hooks
1. Create logical git commits

### Documentation

1. Update implementation plan
1. Create Stage 6B summary
1. Update architecture diagrams
1. Add examples to docs

### Next Steps

- **Stage 7:** Strategy framework (registry, multi-strategy)
- **Stage 8:** Golden tests (full system validation)

______________________________________________________________________

**Ready to begin implementation!** 🚀
