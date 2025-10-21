# Week 3 Summary: Corporate Actions + Fees

**Status**: ✅ COMPLETE\
**Date**: January 2025\
**Tests**: 22 new tests, 721 total passing\
**Coverage**: 90%\
**Quality**: MyPy clean, Ruff clean

## Overview

Week 3 implemented corporate action processing (stock splits, dividends) and comprehensive fee accrual mechanisms (borrow fees, margin interest) for the Portfolio Service. All implementations follow lot-based accounting principles with full ledger tracking.

## Implementations

### 1. Stock Splits (`process_split`)

**Purpose**: Process stock splits and reverse splits while preserving position value.

**Features**:

- Regular splits (e.g., 4-for-1: ratio = 4.0)
- Reverse splits (e.g., 1-for-4: ratio = 0.25)
- Adjusts all lots for symbol:
  - `quantity *= ratio`
  - `entry_price /= ratio`
- Preserves total position value
- Handles immutable Lot objects via rebuild
- Updates position aggregates (quantity, avg_price, market_value)
- Creates SPLIT ledger entries

**Technical Notes**:

- Lot immutability requires rebuilding objects with new values
- Lot tracker: removes old lots, adds adjusted lots
- Zero or negative ratios rejected with validation errors

**Code Location**: `src/qtrader/services/portfolio/service.py:656-773` (118 lines)

______________________________________________________________________

### 2. Cash Dividends (`process_dividend`)

**Purpose**: Process cash dividend payments to/from portfolio.

**Features**:

- Long positions: Receive cash (positive flow)
- Short positions: Pay cash (negative flow)
- Cash flow = quantity × amount_per_share
- Tracks cumulative metrics:
  - `_total_dividends_received` (long positions)
  - `_total_dividends_paid` (short positions)
- Creates DIVIDEND ledger entries

**Technical Notes**:

- Negative amounts rejected
- Missing positions raise errors
- Sign of quantity determines cash flow direction

**Code Location**: `src/qtrader/services/portfolio/service.py:775-838` (64 lines)

______________________________________________________________________

### 3. Borrow Fee Accrual

**Purpose**: Daily accrual of stock borrow fees on short positions.

**Features**:

- Applied to short positions only
- Daily accrual: `abs(market_value) × annual_rate ÷ day_count`
- Symbol-specific rates via `config.borrow_rate_by_symbol`
- Falls back to `config.default_borrow_rate_apr`
- Debits cash (expense)
- Tracks `_total_borrow_fees` cumulative
- Creates BORROW_FEE ledger entries per symbol

**Example Calculation**:

```
Short Position: -100 shares @ $150 = -$15,000 market value
Borrow Rate: 5% APR
Day Count: 360
Daily Fee: $15,000 × 0.05 ÷ 360 = $2.0833...
```

**Code Location**: `src/qtrader/services/portfolio/service.py:583-619` (in `mark_to_market`)

______________________________________________________________________

### 4. Margin Interest Accrual

**Purpose**: Daily accrual of interest on negative cash (margin borrowing).

**Features**:

- Applied to negative cash only
- Daily accrual: `abs(cash) × annual_rate ÷ day_count`
- Rate: `config.margin_rate_apr`
- Debits cash (makes more negative)
- Compounds naturally (interest-on-interest)
- Tracks `_total_margin_interest` cumulative
- Creates MARGIN_INTEREST ledger entries

**Example Calculation**:

```
Negative Cash: -$50,100
Margin Rate: 7% APR
Day Count: 360
Daily Interest: $50,100 × 0.07 ÷ 360 = $9.7417...
New Cash: -$50,100 - $9.7417 = -$50,109.7417
```

**Code Location**: `src/qtrader/services/portfolio/service.py:621-641` (in `mark_to_market`)

______________________________________________________________________

### 5. Mark-to-Market Orchestration

**Purpose**: Comprehensive end-of-day processing of all fees and valuations.

**Process**:

1. Price updates (via `update_prices()` before calling)
1. Accrue borrow fees on all short positions
1. Accrue margin interest on negative cash (if applicable)
1. Create ledger entries for all accruals
1. Update cumulative tracking metrics

**Technical Notes**:

- Borrow fees applied FIRST, then margin interest
- Margin interest calculation uses cash AFTER borrow fees
- Single call per EOD (not multiple times per day)
- Comprehensive logging of all operations

**Code Location**: `src/qtrader/services/portfolio/service.py:559-641` (82 lines)

______________________________________________________________________

## Test Coverage

### Corporate Actions Tests (`test_corporate_actions.py`)

**12 tests** covering splits and dividends:

#### Stock Splits (7 tests)

- ✅ `test_split_long_position_regular` - 4-for-1 split, value preservation
- ✅ `test_split_short_position_regular` - 2-for-1 on shorts
- ✅ `test_reverse_split` - 1-for-4 reverse split
- ✅ `test_split_with_multiple_lots` - Multi-lot consistency
- ✅ `test_split_invalid_ratio_zero` - Error handling
- ✅ `test_split_invalid_ratio_negative` - Error handling
- ✅ `test_split_no_position` - Error handling

#### Dividends (5 tests)

- ✅ `test_dividend_long_position` - Cash in, cumulative tracking
- ✅ `test_dividend_short_position` - Cash out, cumulative tracking
- ✅ `test_dividend_multiple_symbols` - Multiple symbols
- ✅ `test_dividend_invalid_negative_amount` - Error handling
- ✅ `test_dividend_no_position` - Error handling

______________________________________________________________________

### Fee Accrual Tests (`test_fees.py`)

**10 tests** covering borrow fees, margin interest, and mark-to-market:

#### Borrow Fees (4 tests)

- ✅ `test_borrow_fee_short_position` - Daily accrual calculation
- ✅ `test_borrow_fee_no_charge_on_long` - Long exemption
- ✅ `test_borrow_fee_multiple_short_positions` - Multiple shorts
- ✅ `test_borrow_fee_custom_rate` - Symbol-specific rates

#### Margin Interest (3 tests)

- ✅ `test_margin_interest_negative_cash` - Daily accrual
- ✅ `test_margin_interest_no_charge_on_positive_cash` - Positive exemption
- ✅ `test_margin_interest_accumulates` - Multi-day compounding

#### Mark-to-Market (3 tests)

- ✅ `test_mark_to_market_combined_fees` - Borrow + margin together
- ✅ `test_mark_to_market_no_fees_all_positive` - No fees scenario
- ✅ `test_mark_to_market_creates_ledger_entries` - Ledger tracking

______________________________________________________________________

### Integration Tests

**Existing**: `tests/integration/services/test_corporate_actions.py`

- 11 tests for corporate action detection via DataService
- Tests real Algoseek data: AAPL 4:1 split, quarterly dividends
- Validates CorporateActionEvent structure
- Chronological ordering verification

______________________________________________________________________

## Quality Metrics

### Test Results

```
Portfolio Tests: 85/85 passing (Week 1 + Week 2 + Week 3)
Total Tests: 721/721 passing
Coverage: 90%
```

### Code Quality

- ✅ MyPy: No errors (strict mode)
- ✅ Ruff: All checks passed
- ✅ Pre-commit hooks: All passed

### Lines of Code

- **Implementation**: 228 lines added to `service.py`
  - `process_split`: 118 lines
  - `process_dividend`: 64 lines
  - `mark_to_market`: 82 lines (replaced 13-line stub)
- **Tests**: 850 lines total
  - `test_corporate_actions.py`: 400 lines (12 tests)
  - `test_fees.py`: 450 lines (10 tests)

______________________________________________________________________

## Git Commits

Three logically separated commits following conventional commit format:

1. **feat(portfolio): add corporate actions and fee accruals (Week 3)**

   - Commit: `c44e0fe`
   - Files: `service.py` (+228 lines)
   - Implementation of all 5 methods

1. **test(portfolio): add corporate actions unit tests (Week 3)**

   - Commit: `b847e46`
   - Files: `test_corporate_actions.py` (new, +400 lines)
   - 12 tests for splits and dividends

1. **test(portfolio): add fee accrual unit tests (Week 3)**

   - Commit: `07a8af9`
   - Files: `test_fees.py` (new, +450 lines)
   - 10 tests for fees and mark-to-market

______________________________________________________________________

## Technical Highlights

### Lot Immutability Pattern

```python
# Can't mutate frozen Pydantic models, must rebuild
adjusted_lots = [
    Lot(
        lot_id=existing_lot.lot_id,
        symbol=existing_lot.symbol,
        quantity=existing_lot.quantity * ratio,  # New value
        entry_price=existing_lot.entry_price / ratio,  # New value
        timestamp=existing_lot.timestamp,
        side=existing_lot.side,
    )
    for existing_lot in lots
]
```

### Fee Calculation Order

```python
# 1. Borrow fees applied first
cash -= borrow_fee

# 2. Margin interest calculated on NEW cash balance (includes borrow fee)
margin_interest = abs(cash_after_borrow) * rate / day_count
cash -= margin_interest
```

### Symbol-Specific Configuration

```python
config = PortfolioConfig(
    default_borrow_rate_apr=Decimal("0.05"),  # 5% default
    borrow_rate_by_symbol={
        "AAPL": Decimal("0.15"),  # 15% for AAPL
        "GME": Decimal("0.80"),   # 80% for GME (hard to borrow)
    },
)
```

______________________________________________________________________

## Next Steps: Week 4

**Focus**: State management, persistence, and polish

**Planned Features**:

1. State snapshots (save/load portfolio state)
1. Position history tracking
1. Performance metrics (Sharpe, returns, drawdowns)
1. Trade analytics (win rate, avg P&L)
1. Portfolio rebalancing utilities
1. Risk metrics (VaR, beta, correlation)
1. Comprehensive documentation
1. Example strategies using all features

**Target**: Production-ready Portfolio Service with full lifecycle management

______________________________________________________________________

## Lessons Learned

1. **Immutable Models**: Pydantic's frozen models require rebuilding rather than mutation
1. **Fee Ordering**: Order matters when multiple fees affect cash balance
1. **Decimal Arithmetic**: Essential for financial calculations (no float rounding errors)
1. **Test-Driven Development**: Writing tests first caught edge cases immediately
1. **Comprehensive Logging**: Debug-level logs invaluable for tracking complex operations

______________________________________________________________________

## Summary

Week 3 delivers production-ready corporate action processing and fee accruals:

- ✅ Stock splits (regular and reverse)
- ✅ Cash dividends (long receive, short pay)
- ✅ Borrow fee accruals (symbol-specific rates)
- ✅ Margin interest accruals (compounds naturally)
- ✅ Complete mark-to-market orchestration
- ✅ 22 comprehensive unit tests
- ✅ 90% code coverage
- ✅ Type-safe (MyPy strict mode)
- ✅ Three logically separated commits

**Phase 2 Week 3: COMPLETE** ✅
