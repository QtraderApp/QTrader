# Phase 2: PortfolioService (v3 - Final Specification)

## Overview

**Goal:** Create a standalone PortfolioService that maintains accurate portfolio state through deterministic event processing, lot-based accounting, and complete audit trail.

**Duration:** 4 weeks **Complexity:** Medium-High **Priority:** ⭐⭐⭐ Critical - Core accounting and state management

## Design Philosophy

### Core Principles

1. **Treat All Data as Unadjusted** - Portfolio sees prices "as of that point in time"
1. **No Feed Awareness** - Portfolio doesn't distinguish between data sources
1. **User Responsibility** - If using adjusted feeds without corporate actions, user accepts limitations
1. **Deterministic Replay** - Reapplying events produces identical state
1. **Complete Audit Trail** - All state changes recorded in ledger
1. **Lot-Based Accounting** - FIFO for longs, LIFO for shorts
1. **Decimal Precision** - All money/shares use `Decimal` type

### What Portfolio Service Does

✅ Track positions (long and short) ✅ Manage cash balance ✅ Process fills with lot accounting ✅ Calculate realized and unrealized P&L ✅ Process corporate actions (splits, dividends) ✅ Accrue fees and interest daily ✅ Maintain complete ledger ✅ Provide portfolio state snapshots

### What Portfolio Service Does NOT Do

❌ Create orders or fills (that's ExecutionService) ❌ Load market data (that's DataService) ❌ Make risk decisions (that's RiskService) ❌ Determine if cash can go negative (that's RiskService) ❌ Know about data feed types (treats all as unadjusted)

______________________________________________________________________

## Functional Requirements

### Must Support (Phase 2)

#### Position Management

- [x] Long positions with FIFO lot accounting
- [x] Short positions with LIFO lot accounting
- [x] Long → Flat → Short transitions (same symbol)
- [x] Multiple lots per symbol
- [x] Position history (keep positions at zero)

#### Transaction Processing

- [x] Buy long (open/add to position)
- [x] Sell long (close/reduce position, FIFO)
- [x] Sell short (open/add to short position)
- [x] Buy to cover (close/reduce short, LIFO)
- [x] Full fills only (no partial fills)

#### Fees & Commissions

- [x] Entry fees: per-share
- [x] Exit fees: per-share
- [x] Entry fees: percentage of notional value
- [x] Exit fees: percentage of notional value
- [x] Commissions tracked as separate expense (NOT in cost basis)

#### Cash Flows

- [x] Dividend income (long positions receive cash)
- [x] Dividend expense (short positions pay cash)
- [x] Borrow fees on short positions (daily accrual)
- [x] Margin interest on negative cash (daily accrual)

#### Corporate Actions

- [x] Stock splits (e.g., 1 → 4 split)
- [x] Reverse splits (e.g., 4 → 1 split)
- [x] Splits on long positions
- [x] Splits on short positions
- [x] Adjust quantity and cost basis to preserve total value

#### Mark-to-Market

- [x] Daily mark-to-market valuation
- [x] Update position values with current prices
- [x] Calculate unrealized P&L
- [x] Use last known price if current price unavailable

#### Ledger & Audit

- [x] Record all events in ledger
- [x] Timestamp every entry
- [x] Track entry type (FILL, DIVIDEND, SPLIT, FEE, INTEREST)
- [x] Enable deterministic replay

### Explicitly Excluded (Phase 2)

#### Deferred to Later

- [ ] Stock dividends (non-cash) - Defer
- [ ] Partial fills - Defer (Phase 3 or later)
- [ ] Mergers/acquisitions - Defer
- [ ] Spinoffs - Defer
- [ ] Rights issues/warrants - Defer
- [ ] Extended hours trading - Defer
- [ ] Multi-currency support - Defer (USD only)
- [ ] Wash sale tracking - Defer
- [ ] Delisting/bankruptcy - Defer
- [ ] Restricted cash tracking - NOT NEEDED

______________________________________________________________________

## Data Model

### Core Entities

#### Lot (Position Building Block)

```python
from decimal import Decimal
from datetime import datetime
from enum import Enum

class LotSide(str, Enum):
    LONG = "long"
    SHORT = "short"

class Lot(BaseModel):
    """
    Individual lot representing a single trade.

    For FIFO/LIFO accounting. Multiple lots per symbol.
    """
    lot_id: str  # Unique identifier
    symbol: str
    side: LotSide
    quantity: Decimal  # Positive for long, negative for short
    entry_price: Decimal  # Price per share when opened
    entry_timestamp: datetime
    entry_fill_id: str  # Reference to fill that created this lot

    # Fees allocated to this lot (NOT in cost basis)
    entry_commission: Decimal = Decimal("0")

    # For P&L calculation
    realized_pnl: Decimal = Decimal("0")  # Accumulated as lot closes

    class Config:
        frozen = True  # Immutable after creation
```

#### Position (Aggregate View)

```python
class Position(BaseModel):
    """
    Aggregate position for a symbol.

    Derived from lots, not primary storage.
    """
    symbol: str
    quantity: Decimal  # Total shares (positive=long, negative=short)
    lots: list[Lot]  # All open lots for this position

    # Aggregate values
    total_cost: Decimal  # Sum of (lot.quantity * lot.entry_price)
    avg_price: Decimal  # total_cost / quantity

    # Current valuation
    current_price: Decimal | None
    market_value: Decimal  # quantity * current_price
    unrealized_pnl: Decimal  # market_value - total_cost

    # Metadata
    last_updated: datetime

    @property
    def side(self) -> Literal["long", "short", "flat"]:
        if self.quantity > 0:
            return "long"
        elif self.quantity < 0:
            return "short"
        else:
            return "flat"
```

#### LedgerEntry

```python
class LedgerEntryType(str, Enum):
    FILL = "fill"
    DIVIDEND = "dividend"
    SPLIT = "split"
    BORROW_FEE = "borrow_fee"
    MARGIN_INTEREST = "margin_interest"
    COMMISSION = "commission"
    MARK_TO_MARKET = "mark_to_market"

class LedgerEntry(BaseModel):
    """
    Single entry in portfolio ledger.

    Records all economic events affecting portfolio.
    """
    entry_id: str  # Unique ID
    timestamp: datetime
    entry_type: LedgerEntryType

    # Transaction details (if applicable)
    symbol: str | None
    quantity: Decimal | None  # Shares affected
    price: Decimal | None  # Price per share

    # Cash flow
    cash_flow: Decimal  # Positive = cash in, Negative = cash out

    # Fees (tracked separately from cost basis)
    commission: Decimal = Decimal("0")

    # P&L (if applicable)
    realized_pnl: Decimal | None = None

    # References
    fill_id: str | None = None
    lot_ids: list[str] = []  # Lots affected by this entry

    # Metadata
    description: str
    metadata: dict = {}
```

#### PortfolioState

```python
class PortfolioState(BaseModel):
    """
    Immutable snapshot of portfolio at a point in time.

    Used for:
    - Risk evaluation
    - Reporting
    - Historical analysis
    """
    timestamp: datetime

    # Cash
    cash: Decimal

    # Positions
    positions: dict[str, Position]  # symbol → Position

    # Valuations
    equity: Decimal  # Total portfolio value
    market_value: Decimal  # Sum of all position values

    # P&L
    realized_pnl: Decimal  # Total realized (from inception)
    unrealized_pnl: Decimal  # Current unrealized
    total_pnl: Decimal  # realized + unrealized

    # Exposures
    long_exposure: Decimal  # Sum of long position values
    short_exposure: Decimal  # Sum of short position values (absolute)
    net_exposure: Decimal  # long - short
    gross_exposure: Decimal  # long + short

    # Leverage
    leverage: Decimal  # gross_exposure / equity (if equity > 0)

    # Fees & Interest (cumulative)
    total_commissions: Decimal
    total_borrow_fees: Decimal
    total_margin_interest: Decimal
    total_dividends_received: Decimal
    total_dividends_paid: Decimal
```

______________________________________________________________________

## Service Interface

```python
from typing import Protocol, Optional
from decimal import Decimal
from datetime import datetime

class IPortfolioService(Protocol):
    """
    Portfolio service interface.

    Implements lot-based accounting with complete audit trail.
    """

    # ==================== Fill Processing ====================

    def apply_fill(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> None:
        """
        Apply fill to portfolio.

        Processing:
        1. Determine if opening or closing position
        2. For closes: Match lots using FIFO (long) or LIFO (short)
        3. Calculate realized P&L for closed lots
        4. Update cash balance
        5. Record in ledger

        Args:
            fill_id: Unique identifier for this fill
            timestamp: When fill occurred
            symbol: Ticker symbol
            side: "buy" or "sell"
            quantity: Number of shares (positive)
            price: Price per share
            commission: Commission paid (separate from cost basis)

        Raises:
            ValueError: If attempting to close more than available
        """
        ...

    # ==================== Market Data ====================

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """
        Update mark-to-market prices.

        Updates position market values and unrealized P&L.
        Does NOT create ledger entries (use mark_to_market() for that).

        Args:
            prices: Dict mapping symbol → current price
        """
        ...

    def mark_to_market(self, timestamp: datetime) -> None:
        """
        Perform end-of-day mark-to-market.

        1. Update all position values with current prices
        2. Calculate unrealized P&L
        3. Accrue borrow fees on shorts
        4. Accrue margin interest on negative cash
        5. Create ledger entries for fees/interest

        Args:
            timestamp: Time of mark (typically end of day)
        """
        ...

    # ==================== Corporate Actions ====================

    def process_split(
        self,
        symbol: str,
        split_date: datetime,
        ratio: Decimal,
    ) -> None:
        """
        Process stock split or reverse split.

        Adjusts all lots for this symbol:
        - quantity = quantity * ratio
        - entry_price = entry_price / ratio
        - Total value preserved

        Works for both long and short positions.

        Args:
            symbol: Symbol splitting
            split_date: Date of split
            ratio: Split ratio (4.0 for 4-for-1, 0.25 for 1-for-4 reverse)

        Example:
            1-for-4 split: ratio = 4.0
            4-for-1 reverse: ratio = 0.25
        """
        ...

    def process_dividend(
        self,
        symbol: str,
        ex_date: datetime,
        amount_per_share: Decimal,
    ) -> None:
        """
        Process cash dividend.

        For long positions: Cash increases (income)
        For short positions: Cash decreases (expense)

        Args:
            symbol: Symbol paying dividend
            ex_date: Ex-dividend date
            amount_per_share: Dividend per share
        """
        ...

    # ==================== Queries ====================

    def get_position(self, symbol: str) -> Position | None:
        """
        Get current position for symbol.

        Returns None if no position (or position is flat).

        Args:
            symbol: Ticker symbol

        Returns:
            Position or None
        """
        ...

    def get_positions(self) -> dict[str, Position]:
        """
        Get all current positions.

        Returns:
            Dict mapping symbol → Position (includes flat positions if keep_history=True)
        """
        ...

    def get_cash(self) -> Decimal:
        """
        Get current cash balance.

        Can be negative (margin loan).

        Returns:
            Current cash
        """
        ...

    def get_equity(self) -> Decimal:
        """
        Calculate total portfolio equity.

        Equity = Cash + Sum(position market values)

        Returns:
            Total equity
        """
        ...

    def get_state(self) -> PortfolioState:
        """
        Get complete portfolio state snapshot.

        Returns:
            Immutable state for reporting, risk, etc.
        """
        ...

    def get_ledger(
        self,
        since: datetime | None = None,
        entry_types: list[LedgerEntryType] | None = None,
    ) -> list[LedgerEntry]:
        """
        Get ledger entries.

        Args:
            since: Only return entries after this time (optional)
            entry_types: Filter by entry type (optional)

        Returns:
            List of ledger entries
        """
        ...

    def get_realized_pnl(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> Decimal:
        """
        Get realized P&L.

        Args:
            symbol: Specific symbol or None for total
            since: Since this timestamp or None for all time

        Returns:
            Total realized P&L
        """
        ...

    def get_unrealized_pnl(
        self,
        symbol: str | None = None,
    ) -> Decimal:
        """
        Get unrealized P&L.

        Args:
            symbol: Specific symbol or None for total

        Returns:
            Current unrealized P&L
        """
        ...
```

______________________________________________________________________

## Implementation Plan

### Week 1: Core Service + Ledger

**Goal:** Working service with basic fill processing and ledger

#### Tasks

1. **Create Service Structure**

   - [ ] Create `src/qtrader/services/portfolio/` directory
   - [ ] Create `__init__.py`
   - [ ] Create `interface.py` with `IPortfolioService` protocol
   - [ ] Create `models.py` with Lot, Position, LedgerEntry, PortfolioState
   - [ ] Create `service.py` with `PortfolioService` class skeleton

1. **Implement Ledger**

   - [ ] Create `ledger.py` with `Ledger` class
   - [ ] Support adding entries
   - [ ] Support querying entries (by time, type, symbol)
   - [ ] Ensure chronological ordering

1. **Basic Fill Processing**

   - [ ] Implement `apply_fill()` for opening long positions
   - [ ] Implement `apply_fill()` for opening short positions
   - [ ] Create lots on position open
   - [ ] Update cash balance
   - [ ] Record fills in ledger

1. **Basic Queries**

   - [ ] Implement `get_position()`
   - [ ] Implement `get_positions()`
   - [ ] Implement `get_cash()`
   - [ ] Implement `get_equity()`

1. **Testing**

   - [ ] Unit tests for ledger
   - [ ] Unit tests for simple fills (open positions)
   - [ ] Unit tests for queries
   - [ ] Achieve 80%+ coverage

**Deliverable:** Can open long/short positions, track cash, query state

______________________________________________________________________

### Week 2: Lot Accounting + P&L

**Goal:** FIFO/LIFO lot matching and accurate P&L calculation

#### Tasks

1. **Lot Tracker**

   - [ ] Create `lot_tracker.py` with `LotTracker` class
   - [ ] Implement FIFO queue for long positions
   - [ ] Implement LIFO stack for short positions
   - [ ] Handle partial lot closes
   - [ ] Track remaining quantity per lot

1. **Closing Positions**

   - [ ] Implement closing long positions (FIFO matching)
   - [ ] Implement closing short positions (LIFO matching)
   - [ ] Calculate realized P&L per lot
   - [ ] Handle commissions as separate expense
   - [ ] Record realized P&L in ledger

1. **Position Transitions**

   - [ ] Handle long → flat → short (same symbol)
   - [ ] Handle short → flat → long (same symbol)
   - [ ] Ensure lot queues are properly cleared/restarted

1. **P&L Calculations**

   - [ ] Implement `get_realized_pnl()`
   - [ ] Implement `get_unrealized_pnl()`
   - [ ] Implement mark-to-market valuation
   - [ ] Update position market values

1. **Testing**

   - [ ] Unit tests for FIFO lot matching
   - [ ] Unit tests for LIFO lot matching
   - [ ] Unit tests for realized P&L
   - [ ] Unit tests for position transitions
   - [ ] Property-based tests for lot accounting invariants
   - [ ] Achieve 85%+ coverage

**Deliverable:** Full position lifecycle with accurate P&L

______________________________________________________________________

### Week 3: Corporate Actions + Fees

**Goal:** Split handling and fee/interest accruals

#### Tasks

1. **Stock Splits**

   - [ ] Implement `process_split()`
   - [ ] Adjust lot quantities and prices
   - [ ] Preserve total position value
   - [ ] Handle splits on long positions
   - [ ] Handle splits on short positions
   - [ ] Record splits in ledger

1. **Dividends**

   - [ ] Implement `process_dividend()`
   - [ ] Credit cash for long positions
   - [ ] Debit cash for short positions
   - [ ] Calculate based on current position quantity
   - [ ] Record dividends in ledger
   - [ ] Track cumulative dividends received/paid

1. **Borrow Fees**

   - [ ] Calculate borrow fees on short positions
   - [ ] Daily accrual: `short_value * annual_rate * (days/360)`
   - [ ] Apply fees during `mark_to_market()`
   - [ ] Debit cash for fees
   - [ ] Record in ledger
   - [ ] Track cumulative borrow fees

1. **Margin Interest**

   - [ ] Calculate interest on negative cash
   - [ ] Daily accrual: `negative_cash * annual_rate * (days/360)`
   - [ ] Apply interest during `mark_to_market()`
   - [ ] Debit cash for interest (makes it more negative)
   - [ ] Record in ledger
   - [ ] Track cumulative margin interest

1. **Mark-to-Market**

   - [ ] Implement `mark_to_market()` orchestration
   - [ ] Update prices
   - [ ] Calculate unrealized P&L
   - [ ] Accrue borrow fees
   - [ ] Accrue margin interest
   - [ ] Create ledger entries

1. **Testing**

   - [ ] Unit tests for splits (regular and reverse)
   - [ ] Unit tests for dividends (long and short)
   - [ ] Unit tests for borrow fee calculation
   - [ ] Unit tests for margin interest calculation
   - [ ] Unit tests for mark-to-market process
   - [ ] Integration tests with real-world scenarios
   - [ ] Achieve 90%+ coverage

**Deliverable:** Complete corporate actions and fee handling

______________________________________________________________________

### Week 4: State Management + Polish

**Goal:** Portfolio state snapshots, testing, documentation

#### Tasks

1. **Portfolio State**

   - [ ] Implement `get_state()` comprehensive snapshot
   - [ ] Calculate all exposures (long, short, net, gross)
   - [ ] Calculate leverage
   - [ ] Include cumulative metrics
   - [ ] Make state immutable (frozen Pydantic model)

1. **State History** (Optional)

   - [ ] Store historical state snapshots
   - [ ] Query state at specific timestamp
   - [ ] Export to Parquet for analysis

1. **Edge Cases**

   - [ ] Handle missing prices gracefully (use last known)
   - [ ] Handle zero equity (leverage undefined)
   - [ ] Handle position history (keep flat positions)
   - [ ] Validate all inputs (quantities, prices > 0)

1. **Configuration**

   - [ ] Create `PortfolioConfig` model
   - [ ] Initial cash
   - [ ] Borrow fee rate (default)
   - [ ] Margin interest rate
   - [ ] Lot method (for future extension)
   - [ ] Keep position history (True/False)

1. **Mock Implementation**

   - [ ] Create `tests/mocks/portfolio_service.py`
   - [ ] `MockPortfolioService` with canned states
   - [ ] Useful for testing ExecutionService, RiskService later

1. **Comprehensive Testing**

   - [ ] Complete unit test coverage (>90%)
   - [ ] Integration tests with real data
   - [ ] Property-based tests (Hypothesis)
   - [ ] Performance tests (large portfolios)
   - [ ] Test scenarios:
     - [ ] Multi-day backtest with all features
     - [ ] Long-only strategy
     - [ ] Market neutral (long/short)
     - [ ] High-turnover strategy
     - [ ] Corporate actions during holds

1. **Documentation**

   - [ ] API documentation (docstrings)
   - [ ] Architecture documentation
   - [ ] User guide with examples
   - [ ] Testing guide
   - [ ] Accounting methodology doc

1. **Validation**

   - [ ] Run full test suite
   - [ ] MyPy type checking passes
   - [ ] Ruff linting passes
   - [ ] Coverage ≥90%
   - [ ] All acceptance criteria met

**Deliverable:** Production-ready PortfolioService with complete documentation

______________________________________________________________________

## Accounting Rules

### Lot Matching

#### Long Positions (FIFO - First In, First Out)

```
Lots: [Lot1: 100@$10, Lot2: 50@$12, Lot3: 75@$11]

Sell 125 shares:
1. Close Lot1: 100 shares @ $10 (oldest)
2. Partial close Lot2: 25 shares @ $12
3. Remaining: [Lot2: 25@$12, Lot3: 75@$11]
```

#### Short Positions (LIFO - Last In, First Out)

```
Lots: [Lot1: -100@$50, Lot2: -50@$48, Lot3: -75@$52]

Cover 125 shares:
1. Close Lot3: 75 shares @ $52 (newest)
2. Close Lot2: 50 shares @ $48
3. Remaining: [Lot1: -100@$50]
```

### Cash Flows

#### Buy Long

```
Cash flow = -(quantity × price + commission)
Example: Buy 100 @ $10, commission $5
Cash: -$1,005
```

#### Sell Long

```
Cash flow = +(quantity × price - commission)
Example: Sell 100 @ $12, commission $5
Cash: +$1,195
Realized P&L: ($12 - $10) × 100 - $5 - $5(entry) = $190
```

#### Sell Short

```
Cash flow = +(quantity × price - commission)
Example: Short 100 @ $50, commission $5
Cash: +$4,995
```

#### Buy to Cover

```
Cash flow = -(quantity × price + commission)
Example: Cover 100 @ $48, commission $5
Cash: -$4,805
Realized P&L: ($50 - $48) × 100 - $5 - $5(entry) = $190
```

### Corporate Actions

#### Stock Split

```
Before: 100 shares @ $100/share cost = $10,000 total
Split: 4-for-1 (ratio = 4.0)
After: 400 shares @ $25/share cost = $10,000 total
```

#### Reverse Split

```
Before: 400 shares @ $25/share cost = $10,000 total
Reverse: 1-for-4 (ratio = 0.25)
After: 100 shares @ $100/share cost = $10,000 total
```

#### Dividend (Long)

```
Position: +100 shares
Dividend: $2/share
Cash: +$200 (dividend income)
```

#### Dividend (Short)

```
Position: -100 shares
Dividend: $2/share
Cash: -$200 (dividend expense)
```

### Fees & Interest

#### Borrow Fee (Daily)

```
Short position value: $15,000
Annual borrow rate: 5%
Days held: 1
Borrow fee = $15,000 × 0.05 × (1/360) = $2.08
Cash: -$2.08
```

#### Margin Interest (Daily)

```
Cash balance: -$10,000 (borrowed from broker)
Annual margin rate: 7%
Days: 1
Interest = $10,000 × 0.07 × (1/360) = $1.94
Cash: -$11.94 (more negative)
```

______________________________________________________________________

## Validation Criteria

### Functional Requirements

- [ ] ✅ Can process buy and sell fills
- [ ] ✅ Can open long and short positions
- [ ] ✅ FIFO lot matching for longs works correctly
- [ ] ✅ LIFO lot matching for shorts works correctly
- [ ] ✅ Realized P&L calculated accurately
- [ ] ✅ Unrealized P&L calculated accurately
- [ ] ✅ Handles position transitions (long→short, short→long)
- [ ] ✅ Processes stock splits correctly
- [ ] ✅ Processes dividends correctly (long and short)
- [ ] ✅ Accrues borrow fees daily
- [ ] ✅ Accrues margin interest daily
- [ ] ✅ Tracks commissions separately
- [ ] ✅ Maintains complete ledger
- [ ] ✅ Provides accurate portfolio state

### Technical Requirements

- [ ] ✅ Implements `IPortfolioService` protocol
- [ ] ✅ Zero dependencies on execution/data/strategy layers
- [ ] ✅ All methods have type hints
- [ ] ✅ All public methods have docstrings
- [ ] ✅ MyPy passes with no errors
- [ ] ✅ Ruff linting passes
- [ ] ✅ Test coverage ≥ 90%
- [ ] ✅ All tests pass (577+ tests)
- [ ] ✅ Mock implementation available

### Invariants (Must Always Hold)

- [ ] ✅ Total equity = Cash + Sum(position market values)
- [ ] ✅ Total realized P&L = Sum of all closed lot P&Ls - Total commissions - Total fees
- [ ] ✅ Position quantity = Sum of lot quantities
- [ ] ✅ Split preserves total position value
- [ ] ✅ Ledger is chronologically ordered
- [ ] ✅ Cannot close more shares than available
- [ ] ✅ All cash flows recorded in ledger
- [ ] ✅ All Decimal arithmetic (no floats)

______________________________________________________________________

## Configuration

```yaml
# config/portfolio.yaml

portfolio:
  # Starting capital
  initial_cash: 100000.00

  # Fee rates (can be overridden per fill)
  default_commission_per_share: 0.00
  default_commission_pct: 0.00

  # Borrow fees (short selling)
  default_borrow_rate_apr: 0.05  # 5% annual

  # Margin interest (negative cash)
  margin_rate_apr: 0.07  # 7% annual

  # Accounting
  lot_method:
    long: fifo
    short: lifo

  # Position history
  keep_position_history: true  # Keep positions at zero for reporting

  # Day count convention
  day_count: 360  # For interest calculations
```

______________________________________________________________________

## Testing Strategy

### Unit Tests

```python
# tests/unit/services/portfolio/test_service.py

class TestFillProcessing:
    def test_open_long_position()
    def test_close_long_position_fifo()
    def test_partial_close_long_position()
    def test_open_short_position()
    def test_close_short_position_lifo()
    def test_position_transition_long_to_short()

class TestCashCalculations:
    def test_cash_flow_buy_long()
    def test_cash_flow_sell_long()
    def test_cash_flow_sell_short()
    def test_cash_flow_cover_short()
    def test_commission_handling()

class TestPnLCalculations:
    def test_realized_pnl_single_lot()
    def test_realized_pnl_multiple_lots()
    def test_unrealized_pnl()
    def test_total_pnl()

class TestCorporateActions:
    def test_stock_split_long()
    def test_stock_split_short()
    def test_reverse_split()
    def test_dividend_long()
    def test_dividend_short()

class TestFeesAndInterest:
    def test_borrow_fee_calculation()
    def test_margin_interest_calculation()
    def test_mark_to_market_process()

class TestLedger:
    def test_ledger_chronological_order()
    def test_ledger_query_by_type()
    def test_ledger_query_by_time()
    def test_deterministic_replay()

class TestEdgeCases:
    def test_missing_price_uses_last_known()
    def test_zero_equity_leverage()
    def test_zero_position_handling()
    def test_large_split_ratio()
```

### Integration Tests

```python
# tests/integration/services/test_portfolio_integration.py

class TestPortfolioIntegration:
    def test_long_only_strategy()
    def test_market_neutral_strategy()
    def test_high_turnover_strategy()
    def test_corporate_actions_during_holds()
    def test_multi_day_backtest()
```

### Property-Based Tests (Hypothesis)

```python
# tests/unit/services/portfolio/test_portfolio_properties.py

from hypothesis import given, strategies as st

@given(
    initial_cash=st.decimals(min_value=1000, max_value=1000000),
    fills=st.lists(st.tuples(...))
)
def test_equity_invariant(initial_cash, fills):
    """Equity always equals cash + position values"""
    portfolio = PortfolioService(initial_cash)
    # Apply fills...
    assert portfolio.get_equity() == portfolio.get_cash() + sum_position_values()

@given(...)
def test_split_preserves_value(symbol, ratio):
    """Stock split preserves total position value"""
    # ...
```

______________________________________________________________________

## Next Phase

👉 **[Phase 3: ExecutionService](phase3_execution_service.md)**

After PortfolioService is complete, Phase 3 will:

- Create ExecutionService for order management
- Simulate realistic fills (slippage, partial fills)
- Integrate with PortfolioService (apply fills)
- Add fill simulation modes (instant, realistic, historical)

______________________________________________________________________

## Appendix

### Example Usage

```python
from qtrader.services import PortfolioService
from qtrader.config import PortfolioConfig
from decimal import Decimal
from datetime import datetime

# Configure
config = PortfolioConfig(
    initial_cash=Decimal("100000"),
    margin_rate_apr=Decimal("0.07"),
    default_borrow_rate_apr=Decimal("0.05"),
)

# Create service
portfolio = PortfolioService(config)

# Day 1: Buy long
portfolio.apply_fill(
    fill_id="fill_001",
    timestamp=datetime(2020, 1, 2, 9, 30),
    symbol="AAPL",
    side="buy",
    quantity=Decimal("100"),
    price=Decimal("75.00"),
    commission=Decimal("1.00"),
)

print(f"Cash: ${portfolio.get_cash()}")  # $92,499.00
print(f"Equity: ${portfolio.get_equity()}")  # $100,000.00 (if price unchanged)

# Update price
portfolio.update_prices({"AAPL": Decimal("76.00")})
print(f"Unrealized P&L: ${portfolio.get_unrealized_pnl()}")  # $100.00

# Day 2: Sell half
portfolio.apply_fill(
    fill_id="fill_002",
    timestamp=datetime(2020, 1, 3, 15, 0),
    symbol="AAPL",
    side="sell",
    quantity=Decimal("50"),
    price=Decimal("76.50"),
    commission=Decimal("1.00"),
)

print(f"Realized P&L: ${portfolio.get_realized_pnl()}")  # $73.00
# Calculation: ($76.50 - $75.00) × 50 - $1.00(exit) - $0.50(half of entry) = $73.00

# Day 3: Stock split 4-for-1
portfolio.process_split(
    symbol="AAPL",
    split_date=datetime(2020, 1, 4),
    ratio=Decimal("4.0"),
)

position = portfolio.get_position("AAPL")
print(f"Quantity: {position.quantity}")  # 200 shares (was 50)
print(f"Avg Price: ${position.avg_price}")  # $18.75 (was $75.00)

# Day 4: Dividend
portfolio.process_dividend(
    symbol="AAPL",
    ex_date=datetime(2020, 1, 5),
    amount_per_share=Decimal("0.20"),
)

print(f"Cash: ${portfolio.get_cash()}")  # +$40.00 (200 shares × $0.20)

# End of day: Mark-to-market
portfolio.mark_to_market(timestamp=datetime(2020, 1, 5, 16, 0))

# Get state
state = portfolio.get_state()
print(f"Total Equity: ${state.equity}")
print(f"Total P&L: ${state.total_pnl}")
print(f"Leverage: {state.leverage}")
```

______________________________________________________________________

**Phase Status:** 📝 Ready to Implement **Dependencies:** None (can start immediately) **Last Updated:** October 16, 2025
