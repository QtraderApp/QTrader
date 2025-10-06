# Dividend Receipts for Long Positions — Specification

**Owner:** Javier\
**Version:** 1.0\
**Date:** October 6, 2025\
**Status:** Draft — Pending Approval\
**Related:** Stage 6B Shorting & Accruals, Phase01 Specification

______________________________________________________________________

## 1. Purpose & Context

### 1.1 Problem Statement

**Current State:** Stage 6B implements dividend handling for SHORT positions only:

- When a short position is held through an ex-dividend date, the portfolio pays the dividend (cash debit)
- Implemented via `Portfolio.apply_short_dividend()` and `CashLedger.debit()`
- Transaction type: `DIVIDEND` (cost/expense)

**Gap:** Long positions do NOT receive dividend income:

- When a long position is held through an ex-dividend date, no cash is credited
- Missing: `Portfolio.apply_long_dividend()` method
- Missing: `DIVIDEND_RECEIVED` transaction type
- Result: **Total return calculations are incomplete and asymmetric**

### 1.2 Business Rationale

**Total Return = Price Return + Dividend Return**

For U.S. equities:

- **S&P 500 Historical Context:** ~50% of total returns come from reinvested dividends over 50+ year periods
- **Dividend Yield Range:** 0% (growth stocks) to 8%+ (high-yield stocks)
- **Backtesting Standard:** Industry practice requires accurate total return calculation

**Without dividend receipts:**

- Long positions show artificially low returns
- Cannot fairly compare dividend-paying vs growth stocks
- Cannot accurately attribute performance to alpha vs dividend income
- Asymmetric model: tracks costs (shorts pay) but not income (longs receive)

### 1.3 Objectives

1. **Complete Total Return Calculation:** Track both dividend costs (shorts) and income (longs)
1. **Symmetric Implementation:** Mirror existing short dividend logic for long positions
1. **Transaction Transparency:** Separate transaction type for dividend receipts enables performance attribution
1. **Minimal Complexity:** Leverage existing infrastructure (CashLedger, DividendProcessor, event indexing)

### 1.4 Non-Goals (This Phase)

- ❌ Dividend reinvestment (DRIP) — would require order generation, separate feature
- ❌ Special dividends vs regular dividends — both treated identically as cash
- ❌ Tax withholding on dividends — assume gross dividend paid
- ❌ Payment date vs ex-date timing — use ex-date for immediate cash impact
- ❌ Qualified vs ordinary dividends — tax considerations out of scope

______________________________________________________________________

## 2. Requirements

### 2.1 Functional Requirements

**FR1: Long Position Dividend Detection**

- GIVEN a long position (qty > 0) exists at end-of-day prior to ex-date
- WHEN ex-date is encountered during backtest
- THEN dividend amount MUST be calculated and credited to cash

**FR2: Dividend Calculation Consistency**

- Dividend per share calculation MUST use identical formula as short dividends:
  ```
  dividend_per_share = close_after * (cumulative_price_factor - 1)
  ```
- Same calculation for both longs and shorts ensures consistency

**FR3: Cash Credit Transaction**

- Portfolio MUST credit cash ledger with: `shares * dividend_per_share`
- Transaction type MUST be `DIVIDEND_RECEIVED` (distinct from `DIVIDEND` for shorts)
- Transaction timestamp MUST be ex-date
- Transaction description MUST include symbol and per-share amount

**FR4: Position State Requirements**

- Only process positions with `qty > 0` at ex-date timestamp
- Positions closed before ex-date receive NO dividend
- Positions opened after ex-date receive NO dividend (trade date after ex-date)
- Partial fills: credit dividend for actual shares held

**FR5: Event Type Filtering**

- Process only `event_type == "cashdiv"` (cash dividends)
- Ignore stock dividends, splits, spin-offs (not cash events)
- Consistent with existing short dividend logic

### 2.2 Non-Functional Requirements

**NFR1: Performance**

- No measurable performance degradation vs shorts-only implementation
- Event indexing by ex-date already exists (no additional indexing cost)
- O(1) position lookup per ex-date

**NFR2: Maintainability**

- Code symmetry: `apply_long_dividend()` mirrors `apply_short_dividend()`
- Reuse existing test patterns and fixtures
- Clear separation between long/short dividend logic

**NFR3: Testability**

- Unit tests: DividendProcessor handles longs correctly
- Unit tests: Portfolio.apply_long_dividend() credits cash
- Integration tests: End-to-end backtest with long dividends
- Regression tests: Existing short dividend tests unchanged

**NFR4: Data Requirements**

- Uses existing `AdjustmentEvent` data structure
- No new data fields required
- Backward compatible: works with existing adjustment event datasets

______________________________________________________________________

## 3. Technical Design

### 3.1 Architecture Overview

```
DividendProcessor
    ├─ _index_by_date()          [NO CHANGE - already filters "cashdiv"]
    ├─ process_ex_date()         [NO CHANGE - iterates positions]
    └─ _calculate_dividend()     [MODIFIED - add long position branch]
            │
            ├─ if qty < 0 → Portfolio.apply_short_dividend()   [EXISTING]
            │                       └─ CashLedger.debit()       [EXISTING]
            │
            └─ if qty > 0 → Portfolio.apply_long_dividend()    [NEW]
                                    └─ CashLedger.credit()      [EXISTING, unused for dividends]
```

**Infrastructure Status:**

- ✅ `CashLedger.credit()` exists, tested, ready to use
- ✅ `DividendCalculator.calculate_from_factors()` works for both long/short
- ✅ `DividendProcessor._index_by_date()` already filters cash dividends
- ✅ Event loop in `Backtest.run()` calls `process_ex_date()` once per timestamp

**Required Changes:**

- 🆕 Add `Portfolio.apply_long_dividend()` method
- 🆕 Add `TransactionType.DIVIDEND_RECEIVED` enum value
- 🔧 Modify `DividendProcessor._calculate_dividend()` to handle `qty > 0`
- 📋 Add tests for long dividends (mirror existing short tests)
- 📋 Update documentation with examples

### 3.2 Data Model Changes

**src/qtrader/models/ledger.py**

```python
class TransactionType(Enum):
    """Types of cash transactions in ledger."""
    FILL = "fill"                      # Existing
    DIVIDEND = "dividend"              # Existing - for SHORT positions (cost)
    DIVIDEND_RECEIVED = "div_received" # NEW - for LONG positions (income)
    BORROW_COST = "borrow_cost"        # Existing
    DEPOSIT = "deposit"                # Existing
    WITHDRAWAL = "withdrawal"          # Existing
```

**Rationale:** Separate transaction type enables:

- Performance attribution: isolate dividend income from price returns
- Transaction filtering: query all dividend receipts
- Tax reporting: distinguish dividend income from capital gains
- Debugging: clearly identify dividend vs other cash flows

### 3.3 Portfolio Changes

**src/qtrader/models/portfolio.py**

**New Method:**

```python
def apply_long_dividend(
    self,
    symbol: str,
    dividend_per_share: Decimal,
    timestamp: datetime
) -> None:
    """
    Apply dividend receipt for a LONG position.

    Called when a long position is held through an ex-dividend date.
    Credits cash with the dividend amount.

    Args:
        symbol: Security identifier
        dividend_per_share: Dividend amount per share (always positive)
        timestamp: Ex-dividend date

    Raises:
        ValueError: If position does not exist or quantity <= 0

    Example:
        >>> portfolio.apply_long_dividend("AAPL", Decimal("0.25"), ex_date)
        # Position: 100 shares AAPL
        # Cash credited: 100 * 0.25 = $25.00
    """
    position = self.positions.get(symbol)
    if not position:
        raise ValueError(f"No position exists for {symbol}")

    if position.qty <= 0:
        raise ValueError(
            f"Cannot apply long dividend to non-long position: "
            f"{symbol} qty={position.qty}"
        )

    total_dividend = abs(position.qty) * dividend_per_share

    self.cash.credit(
        amount=total_dividend,
        timestamp=timestamp,
        type=TransactionType.DIVIDEND_RECEIVED,
        description=f"{symbol} dividend: {position.qty} shares @ ${dividend_per_share}/share"
    )
```

**Design Notes:**

- **Symmetry:** Nearly identical signature to `apply_short_dividend()`
- **Validation:** Same checks as short dividends (position exists, correct direction)
- **Credit vs Debit:** Only difference is `credit()` instead of `debit()`
- **Description:** Clear transaction description for audit trail

### 3.4 DividendProcessor Changes

**src/qtrader/execution/dividend_processor.py**

**Modified Method: `_calculate_dividend()`**

```python
def _calculate_dividend(
    self,
    symbol: str,
    event: AdjustmentEvent,
    position: Position,
    timestamp: datetime
) -> None:
    """
    Calculate and apply dividend for a position (SHORT or LONG).

    Args:
        symbol: Security identifier
        event: Adjustment event with price factors
        position: Current position in the security
        timestamp: Ex-dividend date
    """
    # Calculate dividend per share (same formula for shorts and longs)
    dividend_per_share = DividendCalculator.calculate_from_factors(
        close_after=event.metadata["close_after"],
        cumulative_price_factor=event.px_factor
    )

    # Validate calculation
    if dividend_per_share <= 0:
        self.logger.warning(
            f"Invalid dividend amount for {symbol}: {dividend_per_share}"
        )
        return

    # Apply dividend based on position direction
    if position.qty < 0:
        # SHORT position: PAY dividend (debit cash)
        self.portfolio.apply_short_dividend(
            symbol=symbol,
            dividend_per_share=dividend_per_share,
            timestamp=timestamp
        )
        self.logger.info(
            f"Applied short dividend: {symbol} "
            f"{position.qty} shares @ ${dividend_per_share}/share"
        )

    elif position.qty > 0:
        # LONG position: RECEIVE dividend (credit cash)
        self.portfolio.apply_long_dividend(
            symbol=symbol,
            dividend_per_share=dividend_per_share,
            timestamp=timestamp
        )
        self.logger.info(
            f"Applied long dividend: {symbol} "
            f"{position.qty} shares @ ${dividend_per_share}/share"
        )
    # Note: qty == 0 should never occur (filtered in process_ex_date)
```

**Changes from Current Implementation:**

- Add `elif position.qty > 0:` branch for long positions
- Call `apply_long_dividend()` instead of `apply_short_dividend()`
- Log long dividend application separately for debugging
- Keep short dividend logic unchanged

### 3.5 Algorithm & Control Flow

**No changes to main control flow:**

```
Backtest.run()
    for each timestamp:
        1. Process bars (price updates)
        2. Generate signals (strategy)
        3. Create orders (risk manager)
        4. Execute orders (execution engine)
        5. **Process dividends for ex-date** ← Enhanced here
        6. Apply borrow costs (end of day)
```

**DividendProcessor.process_ex_date() logic:**

```
1. Lookup events for ex_date in self._events_by_date
2. For each (symbol, event):
    a. Check if position exists in portfolio
    b. If yes AND qty != 0:
        - Calculate dividend_per_share (DividendCalculator)
        - If qty < 0: apply_short_dividend() [EXISTING]
        - If qty > 0: apply_long_dividend() [NEW]
    c. Update statistics (success/skip counters)
3. Return processing statistics
```

**Edge Cases:**

- **No position on ex-date:** Skip (logged as skipped)
- **Position closed before ex-date:** No dividend (position check fails)
- **Position opened after ex-date:** No dividend (not in portfolio on ex-date)
- **Position qty == 0:** Should never occur (portfolio removes zero positions)

______________________________________________________________________

## 4. Data Specifications

### 4.1 Input Data (No Changes)

**AdjustmentEvent Structure:**

```python
AdjustmentEvent(
    ts=datetime(2024, 2, 9, 9, 30, tzinfo=UTC),  # Ex-date
    symbol="AAPL",
    event_type="cashdiv",                        # Cash dividend only
    px_factor=Decimal("0.9975"),                 # Cumulative factor
    vol_factor=Decimal("1.0"),                   # Unchanged
    metadata={
        "close_after": Decimal("187.50"),        # Close on ex-date
        "amount": Decimal("0.25")                # Per-share dividend (optional)
    }
)
```

**Required Fields:**

- `ts`: Ex-dividend date (timezone-aware)
- `event_type`: Must be "cashdiv" (filtered during init)
- `px_factor`: Cumulative price adjustment factor
- `metadata["close_after"]`: Close price on ex-date

**Calculation Formula (Same as Shorts):**

```python
dividend_per_share = close_after * (cumulative_price_factor - 1)
```

**Example:**

```
close_after = $187.50
px_factor = 0.9975
dividend = 187.50 * (0.9975 - 1) = 187.50 * (-0.0025) = -$0.46875

Absolute value: $0.46875 per share
Long: +$0.46875 per share (credit)
Short: -$0.46875 per share (debit)
```

### 4.2 Output Data (Transaction Ledger)

**Cash Ledger Transaction Example:**

```python
# SHORT position dividend (EXISTING)
Transaction(
    timestamp=datetime(2024, 2, 9, 16, 0, tzinfo=UTC),
    type=TransactionType.DIVIDEND,
    amount=Decimal("-46.88"),  # Negative = debit
    balance=Decimal("9953.12"),
    description="AAPL dividend: -100 shares @ $0.46875/share"
)

# LONG position dividend (NEW)
Transaction(
    timestamp=datetime(2024, 2, 9, 16, 0, tzinfo=UTC),
    type=TransactionType.DIVIDEND_RECEIVED,
    amount=Decimal("46.88"),   # Positive = credit
    balance=Decimal("10046.88"),
    description="AAPL dividend: 100 shares @ $0.46875/share"
)
```

**Transaction Attributes:**

- `timestamp`: Ex-date (when dividend is credited/debited)
- `type`: `DIVIDEND_RECEIVED` for longs, `DIVIDEND` for shorts
- `amount`: Total dividend (shares × dividend_per_share), signed
- `balance`: Cash balance after transaction
- `description`: Human-readable with symbol, quantity, per-share amount

______________________________________________________________________

## 5. Validation & Success Criteria

### 5.1 Test Coverage Requirements

**Unit Tests (Portfolio):**

- ✅ `test_portfolio_long_dividend_credits_cash`
- ✅ `test_portfolio_long_dividend_requires_long_position`
- ✅ `test_portfolio_long_dividend_rejects_short_position`
- ✅ `test_portfolio_long_dividend_rejects_missing_position`

**Unit Tests (DividendProcessor):**

- ✅ `test_processor_calculates_long_dividend`
- ✅ `test_processor_applies_both_long_and_short_dividends`
- ✅ `test_processor_skips_zero_position`
- ✅ `test_processor_logs_long_dividend_application`

**Integration Tests (Backtest):**

- ✅ `test_long_position_receives_dividend_on_ex_date`
- ✅ `test_long_position_closed_before_ex_date_no_dividend`
- ✅ `test_long_position_opened_after_ex_date_no_dividend`
- ✅ `test_multiple_dividends_over_time_long_position`
- ✅ `test_mixed_portfolio_long_and_short_dividends`

**Regression Tests:**

- ✅ All 46 existing Stage 6B tests must pass unchanged
- ✅ No performance degradation (< 5% slowdown)

### 5.2 Acceptance Criteria

**AC1: Functional Correctness**

- [ ] Long positions receive dividend credits on ex-date
- [ ] Short positions still pay dividends (no regression)
- [ ] Closed positions receive no dividends
- [ ] Positions opened after ex-date receive no dividends
- [ ] Cash balance reflects both dividend costs and income

**AC2: Transaction Transparency**

- [ ] `DIVIDEND_RECEIVED` transactions appear in cash ledger
- [ ] Transaction descriptions clearly identify symbol and amounts
- [ ] Can filter/query dividend income separately from costs

**AC3: Calculation Accuracy**

- [ ] Same dividend_per_share formula for longs and shorts
- [ ] Total dividend = shares × dividend_per_share (exact to cent)
- [ ] Matches Algoseek adjustment factor calculations

**AC4: Code Quality**

- [ ] Google-style docstrings for new methods
- [ ] Type hints on all function signatures
- [ ] Consistent with existing code style (PEP 8)
- [ ] No TODO/FIXME comments in production code

**AC5: Documentation**

- [ ] Updated STAGE_6B_IMPLEMENTATION_PLAN.md with extension
- [ ] Code examples in docstrings
- [ ] Architecture diagram updated (if applicable)
- [ ] This specification reviewed and approved

______________________________________________________________________

## 6. Examples & Use Cases

### 6.1 Example 1: Single Dividend Receipt

**Scenario:** Buy 200 shares MSFT, hold through ex-date

**Setup:**

```python
# Initial state
cash = $10,000
position = 200 shares MSFT @ $400/share (cost basis: $80,000)

# Dividend event
event = AdjustmentEvent(
    ts=datetime(2024, 8, 14, tzinfo=UTC),
    symbol="MSFT",
    event_type="cashdiv",
    px_factor=Decimal("0.99875"),
    metadata={"close_after": Decimal("400.00")}
)
```

**Calculation:**

```python
dividend_per_share = 400.00 * (0.99875 - 1) = 400.00 * (-0.00125) = -$0.50
absolute_value = $0.50 per share
total_dividend = 200 shares × $0.50 = $100.00
```

**Result:**

```python
# Cash ledger transaction
Transaction(
    timestamp=datetime(2024, 8, 14, 16, 0, tzinfo=UTC),
    type=TransactionType.DIVIDEND_RECEIVED,
    amount=Decimal("100.00"),
    balance=Decimal("10100.00"),
    description="MSFT dividend: 200 shares @ $0.50/share"
)

# Final state
cash = $10,100 (received $100 dividend)
position = 200 shares MSFT @ $400/share (cost basis unchanged)
```

### 6.2 Example 2: Mixed Long/Short Portfolio

**Scenario:** Long AAPL, Short MSFT, both pay dividends on same date

**Setup:**

```python
# Positions
long_aapl = 100 shares @ $180/share
short_msft = -50 shares @ $400/share

# Dividend events (same ex-date)
aapl_event = AdjustmentEvent(
    ts=datetime(2024, 8, 14, tzinfo=UTC),
    symbol="AAPL",
    event_type="cashdiv",
    px_factor=Decimal("0.9975"),
    metadata={"close_after": Decimal("180.00")}
)

msft_event = AdjustmentEvent(
    ts=datetime(2024, 8, 14, tzinfo=UTC),
    symbol="MSFT",
    event_type="cashdiv",
    px_factor=Decimal("0.99875"),
    metadata={"close_after": Decimal("400.00")}
)
```

**Calculations:**

```python
# AAPL (LONG): Receive dividend
aapl_div = 180.00 * (0.9975 - 1) = -$0.45
aapl_total = 100 × $0.45 = +$45.00 (credit)

# MSFT (SHORT): Pay dividend
msft_div = 400.00 * (0.99875 - 1) = -$0.50
msft_total = 50 × $0.50 = -$25.00 (debit)

# Net cash flow
net = +$45.00 - $25.00 = +$20.00
```

**Result:**

```python
# Two transactions in cash ledger
Transaction(
    type=TransactionType.DIVIDEND_RECEIVED,
    amount=Decimal("45.00"),
    description="AAPL dividend: 100 shares @ $0.45/share"
)

Transaction(
    type=TransactionType.DIVIDEND,
    amount=Decimal("-25.00"),
    description="MSFT dividend: -50 shares @ $0.50/share"
)

# Net effect: cash increased by $20
```

### 6.3 Example 3: Quarterly Dividends

**Scenario:** Hold KO (Coca-Cola) for 1 year, receive 4 quarterly dividends

**Setup:**

```python
position = 1000 shares KO
dividend_per_quarter = $0.46/share
quarters = ["2024-02-09", "2024-05-10", "2024-08-09", "2024-11-08"]
```

**Expected Outcome:**

```python
# 4 dividend receipts over the year
q1_dividend = 1000 × $0.46 = $460
q2_dividend = 1000 × $0.46 = $460
q3_dividend = 1000 × $0.46 = $460
q4_dividend = 1000 × $0.46 = $460

# Annual dividend income
total_annual_dividend = $1,840

# If KO price unchanged over year:
# Total return = 0% (price) + $1,840 (dividends) = +1.84% on $100k position
```

### 6.4 Example 4: Position Timing Edge Cases

**Scenario:** Test position entry/exit timing relative to ex-date

```python
# Ex-date: 2024-08-14
# Dividend: $0.50/share

# Case A: Buy before ex-date, hold through → RECEIVE dividend ✅
buy_date = 2024-08-13
hold_through_ex_date = True
result = +$50 credit (100 shares × $0.50)

# Case B: Buy on ex-date → NO dividend ❌
buy_date = 2024-08-14
result = $0 (position opened on/after ex-date)

# Case C: Sell before ex-date → NO dividend ❌
buy_date = 2024-08-01
sell_date = 2024-08-13
result = $0 (position closed before ex-date)

# Case D: Buy after ex-date → NO dividend ❌
buy_date = 2024-08-15
result = $0 (position didn't exist on ex-date)
```

______________________________________________________________________

## 7. Risk Assessment

### 7.1 Technical Risks

**Risk: Incorrect Dividend Direction**

- **Impact:** High — could credit instead of debit or vice versa
- **Mitigation:** Comprehensive unit tests for both long/short cases
- **Detection:** Integration tests validate cash balance changes

**Risk: Double-Counting Dividends**

- **Impact:** High — dividends applied multiple times
- **Mitigation:** Existing duplicate prevention in Backtest.run()
- **Detection:** Process statistics track success/skip counts

**Risk: Position Timing Edge Cases**

- **Impact:** Medium — dividends applied when shouldn't be
- **Mitigation:** Same position existence check as shorts
- **Detection:** Integration tests for all timing scenarios

### 7.2 Data Quality Risks

**Risk: Missing Adjustment Events**

- **Impact:** Medium — dividends not processed
- **Mitigation:** Data validation layer (out of scope for this change)
- **Detection:** Compare dividend income vs expected yield

**Risk: Incorrect Price Factors**

- **Impact:** Medium — wrong dividend amounts
- **Mitigation:** Use vendor-provided factors (Algoseek)
- **Detection:** Spot check against known dividend amounts

### 7.3 Performance Risks

**Risk: Degraded Backtest Performance**

- **Impact:** Low — additional branch in hot loop
- **Mitigation:** O(1) operation, symmetric to existing shorts logic
- **Detection:** Benchmark before/after (expect < 5% slowdown)

______________________________________________________________________

## 8. Open Questions & Decisions

### 8.1 Resolved Decisions

**Q1: Should we use payment date or ex-date for cash impact?**

- **Decision:** Ex-date (immediate impact on ex-date)
- **Rationale:** Consistent with short dividend timing, simpler model
- **Alternative:** Payment date (more realistic but adds complexity)

**Q2: Should we create separate transaction type for long dividends?**

- **Decision:** YES — `DIVIDEND_RECEIVED` distinct from `DIVIDEND`
- **Rationale:** Enables performance attribution, clearer audit trail
- **Alternative:** Reuse `DIVIDEND` with signed amounts (less clear)

**Q3: Should we implement dividend reinvestment (DRIP)?**

- **Decision:** NO — out of scope for this phase
- **Rationale:** Requires order generation logic, separate feature
- **Future:** Can add DRIP flag in Strategy or ExecutionConfig

**Q4: Should we handle special dividends differently?**

- **Decision:** NO — treat all cash dividends identically
- **Rationale:** Adjustment events don't distinguish regular vs special
- **Future:** Could add metadata field if needed for attribution

### 8.2 Open Questions (Require Approval)

**Q5: Should we add dividend income to performance metrics?**

- **Current:** Equity = cash + unrealized_pnl (price-based only)
- **Proposal:** Add explicit dividend tracking for Sharpe/return calculations
- **Status:** ⏸️ Defer to Phase 2 (Performance Metrics stage)

**Q6: Should we support custom dividend schedules?**

- **Context:** Algoseek provides historical dividends, but what about synthetic/future dividends?
- **Proposal:** Add `CustomDividendSchedule` adapter for testing
- **Status:** ⏸️ Defer unless specific use case emerges

______________________________________________________________________

## 9. References

### 9.1 Internal Documents

- **STAGE_6B_IMPLEMENTATION_PLAN.md** — Current implementation status
- **docs/specs/phase01.md** — Original backtest specification
- **docs/architecture.md** — System architecture overview

### 9.2 External References

- **Investopedia: Ex-Dividend Date** — <https://www.investopedia.com/terms/e/ex-dividend.asp>
- **SEC: Dividend Payment Process** — Timeline and mechanics
- **Algoseek: Adjustment Factors** — Vendor documentation for px_factor calculations
- **QuantConnect: Dividend Handling** — Industry reference implementation

### 9.3 Related Code

- `src/qtrader/models/portfolio.py:171` — `apply_short_dividend()` (reference impl)
- `src/qtrader/models/ledger.py:64,103` — `debit()` and `credit()` methods
- `src/qtrader/execution/dividend_processor.py:~160` — `_calculate_dividend()`
- `src/qtrader/execution/dividend_calculator.py` — Dividend calculation formulas

______________________________________________________________________

## 10. Approval & Sign-Off

**Specification Status:** 🟡 Draft — Awaiting Review

**Reviewers:**

- [ ] **Owner (Javier):** Technical design, implementation feasibility
- [ ] **QA/Testing:** Test coverage adequacy
- [ ] **Documentation:** Clarity and completeness

**Approval Criteria:**

1. All sections complete with no TBD placeholders
1. Test coverage plan comprehensive (10+ tests)
1. Risk assessment identifies all major concerns
1. Examples demonstrate all key scenarios
1. Open questions resolved or deferred with justification

**Next Steps After Approval:**

1. Create `STAGE_6B_EXTENSION_IMPLEMENTATION_PLAN.md`
1. Implement Phase 1: Portfolio & Ledger changes (30 min)
1. Implement Phase 2: DividendProcessor changes (30 min)
1. Implement Phase 3: Unit tests (1 hour)
1. Implement Phase 4: Integration tests (30 min)
1. Update documentation (30 min)
1. Commit and mark Stage 6B fully complete

______________________________________________________________________

**Document Version History:**

- v1.0 (2025-10-06): Initial draft specification
