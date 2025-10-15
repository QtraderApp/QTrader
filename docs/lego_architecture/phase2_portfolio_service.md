# Phase 2: Extract PortfolioService

## Overview

**Goal:** Create a standalone PortfolioService that handles position tracking, cash management, and equity calculations with zero knowledge of how fills are created or how data is sourced.

**Duration:** 2-3 weeks **Complexity:** Medium **Priority:** High - Core business logic isolation

## Current State (Master Branch)

### What Exists

Current `Portfolio` model in `src/qtrader/models/portfolio.py`:

```python
class Portfolio(BaseModel):
    """Portfolio with positions and cash."""
    initial_cash: Decimal
    cash: Decimal
    positions: Dict[str, "Position"]
    equity: Decimal

    # Methods are mixed with data
    def update_prices(self, prices: Dict[str, Decimal]): ...
    def apply_fill(self, fill: Fill): ...
    # etc.
```

**Problems:**

- Portfolio is a Pydantic model (data + behavior mixed)
- Directly manipulated by ExecutionEngine
- Position tracking logic spread across multiple files
- Hard to test independently
- No clear interface

## Target Architecture

### Service Interface

```python
# src/qtrader/services/portfolio/interface.py

from decimal import Decimal
from typing import Dict, Optional, Protocol

from qtrader.models.ledger import LedgerEntry
from qtrader.models.order import Fill
from qtrader.models.portfolio import Portfolio, PortfolioState
from qtrader.models.position import Position


class IPortfolioService(Protocol):
    """
    Portfolio service interface.

    Responsibilities:
    - Track positions (long/short)
    - Manage cash balance
    - Calculate equity and valuations
    - Record transactions in ledger
    - Provide portfolio snapshots

    Does NOT:
    - Create orders or fills
    - Load market data
    - Execute trades
    - Make risk decisions
    """

    def apply_fill(self, fill: Fill) -> None:
        """
        Apply fill to portfolio.

        Updates:
        - Position quantity and cost basis
        - Cash balance (adjusted for cost + commission)
        - Ledger entry

        Args:
            fill: Fill to apply

        Raises:
            ValueError: If fill would violate constraints
        """
        ...

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        """
        Update mark-to-market prices.

        Args:
            prices: Dict mapping symbol → current price
        """
        ...

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get current position for symbol.

        Args:
            symbol: Ticker symbol

        Returns:
            Position or None if no position
        """
        ...

    def get_cash(self) -> Decimal:
        """Get current cash balance."""
        ...

    def get_equity(self) -> Decimal:
        """Get total equity (cash + positions market value)."""
        ...

    def get_state(self) -> PortfolioState:
        """
        Get complete portfolio state snapshot.

        Returns:
            Immutable state for risk evaluation, reporting, etc.
        """
        ...

    def get_ledger_entries(self) -> list[LedgerEntry]:
        """Get all ledger entries (transaction history)."""
        ...

    def process_dividend(self, symbol: str, amount: Decimal, ex_date: str) -> None:
        """
        Process dividend payment.

        Args:
            symbol: Symbol paying dividend
            amount: Dividend per share
            ex_date: Ex-dividend date
        """
        ...

    def process_split(self, symbol: str, ratio: Decimal, split_date: str) -> None:
        """
        Process stock split.

        Args:
            symbol: Symbol splitting
            ratio: Split ratio (e.g., 2.0 for 2-for-1)
            split_date: Split effective date
        """
        ...


class PortfolioState(BaseModel):
    """
    Immutable portfolio state snapshot.

    Used for:
    - Risk evaluation
    - Reporting
    - Historical snapshots
    """
    timestamp: datetime
    cash: Decimal
    equity: Decimal
    positions: Dict[str, Position]
    leverage: Decimal
    long_exposure: Decimal
    short_exposure: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
```

### Service Implementation

```python
# src/qtrader/services/portfolio/service.py

from decimal import Decimal
from typing import Dict, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.models.ledger import Ledger, LedgerEntry, LedgerEntryType
from qtrader.models.order import Fill
from qtrader.models.position import Position, PositionTracker
from qtrader.services.portfolio.interface import IPortfolioService, PortfolioState

logger = LoggerFactory.get_logger()


class PortfolioService:
    """
    Concrete portfolio service implementation.

    Manages all portfolio state using internal PositionTracker and Ledger.
    """

    def __init__(self, initial_cash: Decimal):
        """
        Initialize portfolio service.

        Args:
            initial_cash: Starting cash balance
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.position_tracker = PositionTracker()
        self.ledger = Ledger()
        self._current_prices: Dict[str, Decimal] = {}

        logger.info(
            "portfolio_service.initialized",
            initial_cash=float(initial_cash),
        )

    def apply_fill(self, fill: Fill) -> None:
        """Apply fill to portfolio."""
        # Update position
        self.position_tracker.update_from_fill(fill)

        # Update cash
        cash_impact = self._calculate_cash_impact(fill)
        self.cash += cash_impact

        # Record in ledger
        entry = LedgerEntry(
            timestamp=fill.timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=fill.symbol,
            quantity=fill.qty,
            price=fill.price,
            commission=fill.commission,
            cash_impact=cash_impact,
        )
        self.ledger.add_entry(entry)

        logger.info(
            "portfolio_service.fill_applied",
            symbol=fill.symbol,
            side=fill.side.value,
            qty=fill.qty,
            price=float(fill.price),
            commission=float(fill.commission),
            cash=float(self.cash),
        )

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        """Update mark prices."""
        self._current_prices.update(prices)

        # Update position market values
        for symbol, price in prices.items():
            position = self.position_tracker.get_position(symbol)
            if position:
                self.position_tracker.update_price(symbol, price)

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for symbol."""
        return self.position_tracker.get_position(symbol)

    def get_cash(self) -> Decimal:
        """Get cash balance."""
        return self.cash

    def get_equity(self) -> Decimal:
        """Calculate total equity."""
        positions_value = sum(
            p.market_value for p in self.position_tracker.positions.values()
        )
        return self.cash + positions_value

    def get_state(self) -> PortfolioState:
        """Get portfolio state snapshot."""
        positions = self.position_tracker.get_all_positions()

        # Calculate exposures
        long_exp = sum(p.market_value for p in positions.values() if p.qty > 0)
        short_exp = sum(abs(p.market_value) for p in positions.values() if p.qty < 0)

        return PortfolioState(
            timestamp=datetime.now(),
            cash=self.cash,
            equity=self.get_equity(),
            positions=positions,
            leverage=(long_exp + short_exp) / self.get_equity() if self.get_equity() > 0 else Decimal(0),
            long_exposure=long_exp,
            short_exposure=short_exp,
            gross_exposure=long_exp + short_exp,
            net_exposure=long_exp - short_exp,
        )

    def _calculate_cash_impact(self, fill: Fill) -> Decimal:
        """Calculate cash impact of fill."""
        # Buy: cash decreases (negative)
        # Sell: cash increases (positive)
        # Always subtract commission
        notional = fill.qty * fill.price
        if fill.side == OrderSide.BUY:
            return -(notional + fill.commission)
        else:
            return notional - fill.commission
```

## Implementation Tasks

### Week 1: Interface & Core Service

#### Task 1.1: Create Service Structure

- [ ] Create `src/qtrader/services/portfolio/` directory
- [ ] Create `interface.py` with `IPortfolioService` protocol
- [ ] Create `service.py` with `PortfolioService` implementation
- [ ] Create `state.py` with `PortfolioState` model

#### Task 1.2: Extract Position Tracking Logic

- [ ] Review current `PositionTracker` in `models/position.py`
- [ ] Ensure PositionTracker is service-friendly (internal helper)
- [ ] Move position logic into PortfolioService if needed
- [ ] Keep Position as pure data model

#### Task 1.3: Implement Core Methods

- [ ] Implement `apply_fill()`
- [ ] Implement `update_prices()`
- [ ] Implement `get_position()`
- [ ] Implement `get_cash()`
- [ ] Implement `get_equity()`
- [ ] Implement `get_state()`

### Week 2: Corporate Actions & Testing

#### Task 2.1: Corporate Actions Support

- [ ] Implement `process_dividend()`
- [ ] Implement `process_split()`
- [ ] Add ledger entries for corporate actions
- [ ] Test with historical corporate action data

#### Task 2.2: Unit Tests

- [ ] Create `tests/unit/services/portfolio/test_service.py`
- [ ] Test fill application (buy/sell)
- [ ] Test cash calculations
- [ ] Test equity calculations
- [ ] Test position tracking
- [ ] Test corporate actions
- [ ] Achieve > 90% coverage

#### Task 2.3: Mock Implementation

- [ ] Create `tests/mocks/portfolio_service.py`
- [ ] Implement `MockPortfolioService`
- [ ] Provide canned states for testing

### Week 3: Integration & Polish

#### Task 3.1: Integration Tests

- [ ] Test with real fill sequences
- [ ] Test multi-symbol portfolios
- [ ] Test edge cases (zero cash, short positions)
- [ ] Performance testing (large portfolios)

#### Task 3.2: Documentation

- [ ] API documentation
- [ ] Usage examples
- [ ] Corporate actions guide
- [ ] Testing guide

## Validation Criteria

- [ ] ✅ Implements `IPortfolioService` protocol
- [ ] ✅ Zero dependencies on execution/data/strategy
- [ ] ✅ Can track long and short positions
- [ ] ✅ Correctly handles cash flows
- [ ] ✅ Processes dividends and splits
- [ ] ✅ Test coverage ≥ 90%
- [ ] ✅ All tests pass
- [ ] ✅ MyPy clean
- [ ] ✅ Documentation complete

## Next Phase

👉 **[Phase 3: Extract ExecutionService](phase3_execution_service.md)**

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** None (can start immediately) **Last Updated:** October 15, 2025
