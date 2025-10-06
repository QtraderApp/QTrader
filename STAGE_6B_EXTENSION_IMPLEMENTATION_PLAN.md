# Stage 6B Extension: Dividend Receipts for Long Positions — Implementation Plan

**Status:** 🟡 Ready to Implement\
**Start Date:** October 6, 2025\
**Estimated Duration:** 2-3 hours\
**Related Documents:**

- `docs/specs/dividend_receipts_long_positions.md` — Technical specification
- `STAGE_6B_IMPLEMENTATION_PLAN.md` — Original Stage 6B (shorts only)

______________________________________________________________________

## Executive Summary

**Objective:** Extend Stage 6B dividend system to track dividend receipts for LONG positions, completing total return calculations.

**Current State:** Stage 6B implements dividend payments for SHORT positions only (cash debit). Long positions do not receive dividend income (cash credit), making total return calculations incomplete and asymmetric.

**Scope:** Add support for long position dividend receipts using existing infrastructure (CashLedger, DividendProcessor, event indexing).

**Effort Estimate:** 2-3 hours

- Phase 1: Code changes (1 hour)
- Phase 2: Unit tests (1 hour)
- Phase 3: Integration tests (30 minutes)
- Phase 4: Documentation (30 minutes)

**Success Criteria:**

- ✅ Long positions receive dividend credits on ex-date
- ✅ Short positions continue to pay dividends (no regression)
- ✅ All 448+ existing tests pass
- ✅ 10+ new tests for long dividends
- ✅ Transaction ledger shows both `DIVIDEND` (shorts) and `DIVIDEND_RECEIVED` (longs)

______________________________________________________________________

## Phase 1: Core Implementation (1 hour)

### 1.1 Add DIVIDEND_RECEIVED Transaction Type

**File:** `src/qtrader/models/ledger.py`\
**Estimated Time:** 5 minutes

**Change:**

```python
class TransactionType(Enum):
    """Types of cash transactions in ledger."""
    FILL = "fill"
    DIVIDEND = "dividend"              # For SHORT positions (cost)
    DIVIDEND_RECEIVED = "div_received" # NEW: For LONG positions (income)
    BORROW_COST = "borrow_cost"
    DEPOSIT = "deposit"
    WITHDRAWAL = "withdrawal"
```

**Validation:**

- [ ] Enum value added
- [ ] No conflicts with existing transaction types
- [ ] String value follows naming convention

______________________________________________________________________

### 1.2 Implement Portfolio.apply_long_dividend()

**File:** `src/qtrader/models/portfolio.py`\
**Estimated Time:** 20 minutes\
**Location:** Add after `apply_short_dividend()` method (around line 195)

**Implementation:**

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

**Validation:**

- [ ] Method signature matches `apply_short_dividend()` (consistency)
- [ ] Validates position exists and qty > 0
- [ ] Uses `cash.credit()` instead of `debit()`
- [ ] Transaction type is `DIVIDEND_RECEIVED`
- [ ] Description format matches short dividend description
- [ ] Google-style docstring with example
- [ ] Type hints on all parameters

**Testing Checklist:**

- [ ] Call with valid long position → cash increases
- [ ] Call with short position → raises ValueError
- [ ] Call with missing position → raises ValueError
- [ ] Call with qty=0 position → raises ValueError

______________________________________________________________________

### 1.3 Extend DividendProcessor.\_calculate_dividend()

**File:** `src/qtrader/execution/dividend_processor.py`\
**Estimated Time:** 15 minutes\
**Location:** Modify `_calculate_dividend()` method (around line 160)

**Current Implementation:**

```python
def _calculate_dividend(
    self,
    symbol: str,
    event: AdjustmentEvent,
    position: Position,
    timestamp: datetime
) -> None:
    """Calculate and apply dividend for a SHORT position."""
    dividend_per_share = DividendCalculator.calculate_from_factors(
        close_after=event.metadata["close_after"],
        cumulative_price_factor=event.px_factor
    )

    if dividend_per_share <= 0:
        self.logger.warning(...)
        return

    # Only handles SHORT positions
    if position.qty < 0:
        self.portfolio.apply_short_dividend(...)
        self.logger.info(...)
```

**Modified Implementation:**

```python
def _calculate_dividend(
    self,
    symbol: str,
    event: AdjustmentEvent,
    position: Position,
    timestamp: datetime
) -> None:
    """Calculate and apply dividend for a position (SHORT or LONG)."""
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

**Changes Summary:**

- Update docstring: "SHORT or LONG" instead of "SHORT position"
- Add comment: "same formula for shorts and longs"
- Add `elif position.qty > 0:` branch for long positions
- Call `apply_long_dividend()` in new branch
- Add separate logger.info() for long dividends
- Keep all existing short dividend logic unchanged

**Validation:**

- [ ] Docstring updated
- [ ] Calculation logic unchanged (works for both directions)
- [ ] Long position branch added with correct method call
- [ ] Short position logic unchanged (no regression)
- [ ] Logger messages distinguish long vs short
- [ ] Comment explains qty == 0 never occurs

______________________________________________________________________

### 1.4 Verification Commands

**Run after Phase 1 implementation:**

```bash
# Check syntax (should pass with no errors)
python -m py_compile src/qtrader/models/ledger.py
python -m py_compile src/qtrader/models/portfolio.py
python -m py_compile src/qtrader/execution/dividend_processor.py

# Quick smoke test (existing tests should still pass)
pytest tests/unit/models/test_portfolio.py::test_portfolio_short_dividend -v
pytest tests/unit/execution/test_dividend_processor.py -v -k "short"

# Check for import errors
python -c "from qtrader.models.ledger import TransactionType; print(TransactionType.DIVIDEND_RECEIVED)"
python -c "from qtrader.models.portfolio import Portfolio; print(hasattr(Portfolio, 'apply_long_dividend'))"
```

**Expected Output:**

- No compilation errors
- All existing short dividend tests pass
- DIVIDEND_RECEIVED enum value accessible
- apply_long_dividend method exists

______________________________________________________________________

## Phase 2: Unit Tests (1 hour)

### 2.1 Portfolio Unit Tests

**File:** `tests/unit/models/test_portfolio.py`\
**Estimated Time:** 30 minutes\
**Location:** Add after `test_portfolio_short_dividend` tests

**Tests to Implement:**

#### Test 1: Basic Long Dividend Receipt

```python
def test_portfolio_long_dividend_credits_cash(context):
    """Test long position receives dividend credit."""
    # Setup: Buy 100 shares, initial cash $10,000
    context.portfolio.apply_fill(
        order_id="o1",
        fill_id="f1",
        symbol="AAPL",
        qty=100,
        price=Decimal("180.00"),
        timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc),
        fees=Decimal("1.00")
    )

    initial_cash = context.portfolio.cash.balance

    # Apply dividend: $0.25/share
    context.portfolio.apply_long_dividend(
        symbol="AAPL",
        dividend_per_share=Decimal("0.25"),
        timestamp=datetime(2024, 2, 9, tzinfo=timezone.utc)
    )

    # Assert: Cash increased by 100 * 0.25 = $25
    expected_dividend = Decimal("25.00")
    assert context.portfolio.cash.balance == initial_cash + expected_dividend

    # Assert: Transaction recorded
    transactions = context.portfolio.cash.transactions
    div_tx = [tx for tx in transactions if tx.type == TransactionType.DIVIDEND_RECEIVED]
    assert len(div_tx) == 1
    assert div_tx[0].amount == expected_dividend
    assert "AAPL" in div_tx[0].description
```

#### Test 2: Requires Long Position

```python
def test_portfolio_long_dividend_requires_long_position(context):
    """Test long dividend fails if no long position exists."""
    # No position opened

    with pytest.raises(ValueError, match="No position exists"):
        context.portfolio.apply_long_dividend(
            symbol="AAPL",
            dividend_per_share=Decimal("0.25"),
            timestamp=datetime(2024, 2, 9, tzinfo=timezone.utc)
        )
```

#### Test 3: Rejects Short Position

```python
def test_portfolio_long_dividend_rejects_short_position(context):
    """Test long dividend fails for short position."""
    # Setup: Short 100 shares
    context.portfolio.apply_fill(
        order_id="o1",
        fill_id="f1",
        symbol="AAPL",
        qty=-100,  # Short
        price=Decimal("180.00"),
        timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc),
        fees=Decimal("1.00")
    )

    with pytest.raises(ValueError, match="non-long position"):
        context.portfolio.apply_long_dividend(
            symbol="AAPL",
            dividend_per_share=Decimal("0.25"),
            timestamp=datetime(2024, 2, 9, tzinfo=timezone.utc)
        )
```

#### Test 4: Partial Fill Dividend

```python
def test_portfolio_long_dividend_partial_position(context):
    """Test dividend calculated for actual shares held."""
    # Setup: Buy 75 shares (partial fill)
    context.portfolio.apply_fill(
        order_id="o1",
        fill_id="f1",
        symbol="AAPL",
        qty=75,
        price=Decimal("180.00"),
        timestamp=datetime(2024, 2, 1, tzinfo=timezone.utc),
        fees=Decimal("1.00")
    )

    initial_cash = context.portfolio.cash.balance

    # Apply dividend
    context.portfolio.apply_long_dividend(
        symbol="AAPL",
        dividend_per_share=Decimal("0.25"),
        timestamp=datetime(2024, 2, 9, tzinfo=timezone.utc)
    )

    # Assert: Dividend for 75 shares, not 100
    expected_dividend = Decimal("18.75")  # 75 * 0.25
    assert context.portfolio.cash.balance == initial_cash + expected_dividend
```

**Validation Checklist:**

- [ ] All 4+ tests implemented
- [ ] Tests use existing context fixture
- [ ] Assertions check cash balance changes
- [ ] Assertions check transaction ledger
- [ ] Error cases tested (no position, short position)
- [ ] Edge cases tested (partial positions)

______________________________________________________________________

### 2.2 DividendProcessor Unit Tests

**File:** `tests/unit/execution/test_dividend_processor.py`\
**Estimated Time:** 30 minutes\
**Location:** Add after existing short dividend tests

**Tests to Implement:**

#### Test 1: Process Long Dividend

```python
def test_processor_calculates_long_dividend(mock_portfolio):
    """Test processor applies long dividend correctly."""
    # Setup: Long position
    position = Position(symbol="AAPL", qty=100, cost_basis=Decimal("18000.00"))
    mock_portfolio.positions = {"AAPL": position}

    # Setup: Dividend event
    event = AdjustmentEvent(
        ts=datetime(2024, 2, 9, 9, 30, tzinfo=timezone.utc),
        symbol="AAPL",
        event_type="cashdiv",
        px_factor=Decimal("0.9975"),
        vol_factor=Decimal("1.0"),
        metadata={"close_after": Decimal("180.00")}
    )

    processor = DividendProcessor(
        portfolio=mock_portfolio,
        adjustment_events=[event]
    )

    # Execute
    stats = processor.process_ex_date(datetime(2024, 2, 9, tzinfo=timezone.utc))

    # Assert: apply_long_dividend called
    assert mock_portfolio.apply_long_dividend.called
    assert stats["success_count"] == 1
```

#### Test 2: Mixed Long/Short Portfolio

```python
def test_processor_applies_both_long_and_short_dividends(mock_portfolio):
    """Test processor handles mixed portfolio correctly."""
    # Setup: Both long and short positions
    long_pos = Position(symbol="AAPL", qty=100, cost_basis=Decimal("18000.00"))
    short_pos = Position(symbol="MSFT", qty=-50, cost_basis=Decimal("20000.00"))
    mock_portfolio.positions = {
        "AAPL": long_pos,
        "MSFT": short_pos
    }

    # Setup: Both have dividends on same date
    events = [
        AdjustmentEvent(
            ts=datetime(2024, 2, 9, 9, 30, tzinfo=timezone.utc),
            symbol="AAPL",
            event_type="cashdiv",
            px_factor=Decimal("0.9975"),
            vol_factor=Decimal("1.0"),
            metadata={"close_after": Decimal("180.00")}
        ),
        AdjustmentEvent(
            ts=datetime(2024, 2, 9, 9, 30, tzinfo=timezone.utc),
            symbol="MSFT",
            event_type="cashdiv",
            px_factor=Decimal("0.99875"),
            vol_factor=Decimal("1.0"),
            metadata={"close_after": Decimal("400.00")}
        )
    ]

    processor = DividendProcessor(
        portfolio=mock_portfolio,
        adjustment_events=events
    )

    # Execute
    stats = processor.process_ex_date(datetime(2024, 2, 9, tzinfo=timezone.utc))

    # Assert: Both methods called
    assert mock_portfolio.apply_long_dividend.called
    assert mock_portfolio.apply_short_dividend.called
    assert stats["success_count"] == 2
```

#### Test 3: Logger Messages

```python
def test_processor_logs_long_dividend_application(mock_portfolio, caplog):
    """Test processor logs long dividend events."""
    position = Position(symbol="AAPL", qty=100, cost_basis=Decimal("18000.00"))
    mock_portfolio.positions = {"AAPL": position}

    event = AdjustmentEvent(
        ts=datetime(2024, 2, 9, 9, 30, tzinfo=timezone.utc),
        symbol="AAPL",
        event_type="cashdiv",
        px_factor=Decimal("0.9975"),
        vol_factor=Decimal("1.0"),
        metadata={"close_after": Decimal("180.00")}
    )

    processor = DividendProcessor(
        portfolio=mock_portfolio,
        adjustment_events=[event]
    )

    with caplog.at_level(logging.INFO):
        processor.process_ex_date(datetime(2024, 2, 9, tzinfo=timezone.utc))

    # Assert: Log message contains "long dividend"
    assert any("Applied long dividend" in record.message for record in caplog.records)
```

**Validation Checklist:**

- [ ] All 3+ tests implemented
- [ ] Tests use mock portfolio fixture
- [ ] Both long and short cases tested
- [ ] Statistics validated (success_count)
- [ ] Logger output verified

______________________________________________________________________

## Phase 3: Integration Tests (30 minutes)

### 3.1 Backtest Integration Tests

**File:** `tests/integration/test_backtest_dividends.py`\
**Estimated Time:** 30 minutes\
**Location:** Add after existing short dividend integration tests

**Tests to Implement:**

#### Test 1: Long Position Receives Dividend

```python
def test_long_position_receives_dividend_on_ex_date():
    """Test long position receives dividend credit on ex-date."""
    # Setup: Simple buy and hold strategy
    # ... (similar structure to existing tests)

    # Assert:
    # - Cash balance increased by dividend amount
    # - Transaction type is DIVIDEND_RECEIVED
    # - Final equity = initial equity + dividends (if price unchanged)
```

#### Test 2: Position Closed Before Ex-Date

```python
def test_long_position_closed_before_ex_date_no_dividend():
    """Test closed position does not receive dividend."""
    # Setup: Buy then sell before ex-date
    # Assert: No dividend received
```

#### Test 3: Position Opened After Ex-Date

```python
def test_long_position_opened_after_ex_date_no_dividend():
    """Test position opened after ex-date receives no dividend."""
    # Setup: Buy after ex-date
    # Assert: No dividend received
```

#### Test 4: Multiple Dividends Over Time

```python
def test_multiple_dividends_over_time_long_position():
    """Test quarterly dividends over multiple ex-dates."""
    # Setup: Hold for 4 quarters
    # Assert: 4 dividend receipts, correct total
```

#### Test 5: Mixed Long/Short Portfolio

```python
def test_mixed_portfolio_long_and_short_dividends():
    """Test portfolio with both long and short positions."""
    # Setup: Long AAPL, Short MSFT
    # Assert:
    # - Long position receives dividend (credit)
    # - Short position pays dividend (debit)
    # - Both transactions in ledger
    # - Net cash flow correct
```

**Validation Checklist:**

- [ ] All 5+ integration tests implemented
- [ ] Tests cover all timing edge cases
- [ ] Tests validate cash balance changes
- [ ] Tests check transaction ledger contents
- [ ] Tests use realistic dividend amounts

______________________________________________________________________

## Phase 4: Documentation & Cleanup (30 minutes)

### 4.1 Update STAGE_6B_IMPLEMENTATION_PLAN.md

**Estimated Time:** 15 minutes

**Add section at end:**

````markdown
## Stage 6B Extension: Long Position Dividends

**Completion Date:** October 6, 2025
**Commits:** 1 commit
**Tests Added:** 10+ tests

### Implementation Summary

Extended dividend system to handle LONG position dividend receipts, completing
total return calculations. Long positions now receive dividend income (cash credit)
symmetrically with short positions paying dividends (cash debit).

### Changes

**Models:**
- Added `TransactionType.DIVIDEND_RECEIVED` for long dividend income
- Added `Portfolio.apply_long_dividend()` method (mirrors apply_short_dividend)

**Execution:**
- Modified `DividendProcessor._calculate_dividend()` to handle qty > 0
- Added logging for long dividend application

**Tests:**
- 4 Portfolio unit tests (long dividend credit, validation)
- 3 DividendProcessor unit tests (long calculation, mixed portfolio)
- 5 Integration tests (timing, multiple dividends, mixed portfolio)

### Technical Highlights

**Symmetric Implementation:**
- Same dividend calculation formula for longs and shorts
- Mirrored method signatures (apply_long_dividend vs apply_short_dividend)
- Consistent transaction descriptions and logging

**Total Return Completion:**
```python
# Before: Only tracked costs (shorts pay dividends)
short_dividend = -100 shares × $0.50 = -$50.00 (debit)

# After: Now tracks both costs and income
short_dividend = -100 shares × $0.50 = -$50.00 (debit)
long_dividend  = +200 shares × $0.50 = +$100.00 (credit)
net_cash_flow  = +$50.00
````

**Examples:**

*Single Long Position:*

```python
# Position: 100 shares AAPL @ $180
# Ex-date: 2024-02-09
# Dividend: $0.25/share

Result:
- Cash credited: $25.00
- Transaction: DIVIDEND_RECEIVED
- Description: "AAPL dividend: 100 shares @ $0.25/share"
```

*Mixed Portfolio:*

```python
# Position: Long 100 AAPL, Short 50 MSFT
# Both ex-date: 2024-08-14
# AAPL dividend: $0.45/share
# MSFT dividend: $0.50/share

Result:
- AAPL credit: +$45.00 (long receives)
- MSFT debit: -$25.00 (short pays)
- Net cash: +$20.00
```

### Test Results

```
Stage 6B Extension Tests: 12 passed
Stage 6B Original Tests: 46 passed (no regressions)
Total Project Tests: 460 passed, 10 skipped

Coverage:
- Portfolio.apply_long_dividend: 100%
- DividendProcessor long branch: 100%
- Integration scenarios: 100%
```

### Success Criteria ✅

- [x] Long positions receive dividend credits on ex-date
- [x] Short positions still pay dividends (no regression)
- [x] Closed positions receive no dividends
- [x] Positions opened after ex-date receive no dividends
- [x] Transaction ledger distinguishes long vs short dividends
- [x] All existing tests pass
- [x] 10+ new tests added
- [x] Documentation updated

### Commit History

```
abc1234 feat(execution): Add long position dividend receipts
        - Add DIVIDEND_RECEIVED transaction type
        - Implement Portfolio.apply_long_dividend()
        - Extend DividendProcessor for qty > 0
        - Add 12 comprehensive tests
        - Complete total return calculation system
```

### Next Steps

Stage 6B is now fully complete with symmetric dividend handling for both long and short positions. Total return calculations accurately reflect both price returns and dividend income/costs.

**Recommended Next Stage:** Stage 7 - Strategy Framework

````

---

### 4.2 Update Code Comments

**Estimated Time:** 10 minutes

**Files to update:**

1. **src/qtrader/execution/dividend_processor.py**
   - Update module docstring to mention both long and short dividends
   - Update class docstring with examples of both

2. **src/qtrader/models/portfolio.py**
   - Update class docstring to mention dividend receipts
   - Cross-reference apply_long_dividend and apply_short_dividend in docstrings

3. **tests/integration/test_backtest_dividends.py**
   - Update module docstring with long dividend examples

---

### 4.3 Verify Pre-commit Hooks

**Estimated Time:** 5 minutes

```bash
# Run pre-commit hooks manually
pre-commit run --all-files

# Expected checks:
# - Black formatting
# - Flake8 linting
# - isort import sorting
# - mypy type checking
# - mdformat markdown formatting
# - Trailing whitespace removal
````

**Fix any issues:**

- Format code: `black src/ tests/`
- Sort imports: `isort src/ tests/`
- Type hints: Add missing annotations

______________________________________________________________________

## Phase 5: Testing & Validation (Integrated)

### 5.1 Unit Test Execution

```bash
# Run new portfolio tests
pytest tests/unit/models/test_portfolio.py -v -k "long_dividend"

# Run new processor tests
pytest tests/unit/execution/test_dividend_processor.py -v -k "long"

# Expected: 7+ new tests passing
```

### 5.2 Integration Test Execution

```bash
# Run new integration tests
pytest tests/integration/test_backtest_dividends.py -v

# Expected: 9+ existing + 5+ new = 14+ tests passing
```

### 5.3 Full Regression Test

```bash
# Run entire test suite
pytest tests/ -v --tb=short

# Expected: 460+ tests passing, 10 skipped
# Increase from 448 (Stage 6B baseline) by ~12 new tests
```

### 5.4 Coverage Report

```bash
# Generate coverage report for changed files
pytest tests/ \
  --cov=src/qtrader/models/portfolio \
  --cov=src/qtrader/models/ledger \
  --cov=src/qtrader/execution/dividend_processor \
  --cov-report=html \
  --cov-report=term

# Expected coverage:
# - portfolio.py: 95%+ (new method covered)
# - ledger.py: 100% (enum only)
# - dividend_processor.py: 95%+ (new branch covered)
```

### 5.5 Manual Validation Scenarios

**Scenario 1: Compare Long vs Short Dividends**

```python
# Test script: verify symmetric behavior
from qtrader.models.portfolio import Portfolio

# Setup identical positions (opposite directions)
portfolio1 = Portfolio(...)  # Long 100 shares
portfolio2 = Portfolio(...)  # Short 100 shares

# Apply same dividend
dividend = Decimal("0.50")
portfolio1.apply_long_dividend("TEST", dividend, ex_date)
portfolio2.apply_short_dividend("TEST", dividend, ex_date)

# Verify: Cash changes equal magnitude, opposite sign
assert portfolio1.cash.balance == initial + 50.00
assert portfolio2.cash.balance == initial - 50.00
```

**Scenario 2: Quarterly Dividend Stream**

```python
# Test 4 quarterly dividends over 1 year
# Verify cumulative dividend income matches expected annual yield
```

______________________________________________________________________

## Timeline & Milestones

### Estimated Schedule

| Phase     | Task                            | Duration     | Dependencies   |
| --------- | ------------------------------- | ------------ | -------------- |
| 1.1       | Add DIVIDEND_RECEIVED enum      | 5 min        | None           |
| 1.2       | Implement apply_long_dividend() | 20 min       | 1.1            |
| 1.3       | Extend \_calculate_dividend()   | 15 min       | 1.2            |
| 1.4       | Verify implementation           | 20 min       | 1.1-1.3        |
| 2.1       | Portfolio unit tests            | 30 min       | 1.2            |
| 2.2       | Processor unit tests            | 30 min       | 1.3            |
| 3.1       | Integration tests               | 30 min       | 1.1-1.3        |
| 4.1       | Update Stage 6B plan            | 15 min       | All tests pass |
| 4.2       | Update code comments            | 10 min       | 4.1            |
| 4.3       | Pre-commit validation           | 5 min        | 4.2            |
| **Total** |                                 | **2h 40min** |                |

### Milestones

**M1: Core Implementation Complete** (60 min)

- [ ] All code changes committed
- [ ] No syntax errors
- [ ] Existing tests still pass

**M2: Unit Tests Complete** (60 min)

- [ ] 7+ new unit tests passing
- [ ] Coverage > 95% for new code
- [ ] No test failures

**M3: Integration Tests Complete** (30 min)

- [ ] 5+ integration tests passing
- [ ] All timing edge cases covered
- [ ] Full regression passes (460+ tests)

**M4: Documentation Complete** (30 min)

- [ ] Implementation plan updated
- [ ] Code comments updated
- [ ] Pre-commit hooks pass
- [ ] Ready for commit

______________________________________________________________________

## Risk Management

### Technical Risks

| Risk                                           | Impact | Probability | Mitigation                         |
| ---------------------------------------------- | ------ | ----------- | ---------------------------------- |
| Break existing short dividend tests            | High   | Low         | Run regression after each change   |
| Incorrect dividend direction (credit vs debit) | High   | Low         | Unit tests validate cash direction |
| Double-counting dividends                      | High   | Low         | Use existing duplicate prevention  |
| Missing edge cases                             | Medium | Medium      | Comprehensive integration tests    |
| Type annotation errors                         | Low    | Low         | mypy validation                    |

### Process Risks

| Risk                                    | Impact | Probability | Mitigation                             |
| --------------------------------------- | ------ | ----------- | -------------------------------------- |
| Scope creep (add DRIP, tax withholding) | Medium | Medium      | Strict scope adherence, defer features |
| Insufficient test coverage              | Medium | Low         | Coverage reports > 95%                 |
| Documentation drift                     | Low    | Medium      | Update docs concurrently with code     |

______________________________________________________________________

## Success Criteria (Final Validation)

### Functional Requirements

- [ ] **FR1:** Long positions receive dividend credits on ex-date
- [ ] **FR2:** Dividend calculation uses identical formula for longs/shorts
- [ ] **FR3:** Cash ledger contains DIVIDEND_RECEIVED transactions
- [ ] **FR4:** Position state requirements enforced (qty > 0 on ex-date)
- [ ] **FR5:** Only "cashdiv" events processed (no splits/stock dividends)

### Non-Functional Requirements

- [ ] **NFR1:** No measurable performance degradation (< 5% slowdown)
- [ ] **NFR2:** Code symmetry maintained (apply_long_dividend mirrors apply_short_dividend)
- [ ] **NFR3:** Test coverage > 95% for new code
- [ ] **NFR4:** No new data requirements (uses existing AdjustmentEvent)

### Quality Gates

- [ ] All 460+ tests pass (448 existing + 12+ new)
- [ ] No linting errors (flake8, black, isort)
- [ ] No type errors (mypy --strict)
- [ ] Pre-commit hooks pass
- [ ] Coverage reports generated
- [ ] Documentation updated

### Deliverables Checklist

**Code:**

- [ ] src/qtrader/models/ledger.py (DIVIDEND_RECEIVED enum)
- [ ] src/qtrader/models/portfolio.py (apply_long_dividend method)
- [ ] src/qtrader/execution/dividend_processor.py (modified \_calculate_dividend)

**Tests:**

- [ ] tests/unit/models/test_portfolio.py (4+ new tests)
- [ ] tests/unit/execution/test_dividend_processor.py (3+ new tests)
- [ ] tests/integration/test_backtest_dividends.py (5+ new tests)

**Documentation:**

- [ ] STAGE_6B_IMPLEMENTATION_PLAN.md (extension section added)
- [ ] docs/specs/dividend_receipts_long_positions.md (specification approved)
- [ ] Code docstrings updated

**Validation:**

- [ ] Full test suite passing
- [ ] Coverage reports > 95%
- [ ] Pre-commit hooks passing
- [ ] Git commit with clear message

______________________________________________________________________

## Commit Strategy

### Single Atomic Commit

**Commit Message:**

```
feat(execution): Add long position dividend receipts

Complete total return calculation by adding dividend income tracking
for long positions, symmetrically with existing short dividend costs.

Changes:
- Add TransactionType.DIVIDEND_RECEIVED for long dividend income
- Implement Portfolio.apply_long_dividend() (mirrors apply_short_dividend)
- Extend DividendProcessor._calculate_dividend() to handle qty > 0
- Add 12 comprehensive tests (4 unit portfolio, 3 unit processor, 5 integration)

Long positions now receive dividend credits on ex-date, enabling accurate
total return attribution (price return + dividend return).

Examples:
- Long 100 shares @ $0.50/share → +$50 cash credit
- Short 100 shares @ $0.50/share → -$50 cash debit
- Mixed portfolio correctly applies both

Tests: 460 passed (+12 new), 10 skipped
Coverage: 95%+ for all modified files

Closes Stage 6B Extension
Refs: docs/specs/dividend_receipts_long_positions.md
```

**Files in Commit:**

```
src/qtrader/models/ledger.py
src/qtrader/models/portfolio.py
src/qtrader/execution/dividend_processor.py
tests/unit/models/test_portfolio.py
tests/unit/execution/test_dividend_processor.py
tests/integration/test_backtest_dividends.py
STAGE_6B_IMPLEMENTATION_PLAN.md
```

______________________________________________________________________

## Post-Implementation Actions

### 1. Verification

```bash
# Confirm clean working directory
git status

# Verify commit
git log -1 --stat

# Confirm test count
pytest tests/ --collect-only | grep "tests collected"
```

### 2. Performance Benchmark

```bash
# Run backtest with dividends (before/after comparison)
# Expected: < 5% slowdown vs shorts-only implementation
```

### 3. Generate Reports

```bash
# Coverage report
pytest tests/ --cov=src/qtrader --cov-report=html

# Open htmlcov/index.html and verify > 95% for modified files
```

### 4. Update Project Board

- [x] Mark "Long Position Dividends" task complete
- [x] Update Stage 6B status: "Fully Complete"
- [x] Create "Stage 7: Strategy Framework" task

______________________________________________________________________

## Appendix

### A. Related Files Reference

**Modified Files:**

- `src/qtrader/models/ledger.py` (line 7, add enum)
- `src/qtrader/models/portfolio.py` (line 195, add method)
- `src/qtrader/execution/dividend_processor.py` (line 160, add branch)

**Test Files:**

- `tests/unit/models/test_portfolio.py` (add 4+ tests)
- `tests/unit/execution/test_dividend_processor.py` (add 3+ tests)
- `tests/integration/test_backtest_dividends.py` (add 5+ tests)

**Documentation:**

- `STAGE_6B_IMPLEMENTATION_PLAN.md` (add extension section)
- `docs/specs/dividend_receipts_long_positions.md` (specification)

### B. External References

**Dividend Mechanics:**

- Ex-Dividend Date: Position must exist before ex-date to receive dividend
- Record Date: Position must be on record 2 days after ex-date (T+2 settlement)
- Payment Date: Actual payment date (not modeled in this implementation)

**Industry Standards:**

- Total Return = Price Return + Dividend Return
- S&P 500: ~50% of total returns from dividends (50+ year period)
- Typical U.S. equity dividend yield: 1.5-2.5% annually

**Testing Patterns:**

- AAA Pattern: Arrange, Act, Assert
- Fixture reuse: context fixture for portfolio tests
- Mock isolation: DividendProcessor tests use mock portfolio

______________________________________________________________________

**Plan Status:** 🟡 Ready to Implement\
**Next Action:** Begin Phase 1 implementation (estimated 1 hour)

______________________________________________________________________

**Document Version History:**

- v1.0 (2025-10-06): Initial implementation plan
