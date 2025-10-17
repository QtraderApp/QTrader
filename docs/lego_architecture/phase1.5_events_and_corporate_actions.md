# Phase 1.5: Events & Corporate Actions

## Overview

**Goal:** Complete the data layer with corporate action detection and establish event-driven architecture foundation for the entire QTrader system.

**Duration:** 3-4 weeks **Complexity:** Medium **Priority:** ⭐⭐⭐ Critical - Foundation for all subsequent phases

**Why Phase 1.5?**

- Portfolio needs corporate actions to maintain accurate positions
- Event-driven architecture enables loose coupling between services
- Better to build foundation now than retrofit later
- Services can work both standalone (direct API) and event-driven

______________________________________________________________________

## Design Principles

### 1. **Event-Driven Architecture**

```
Services publish events → EventBus → Services subscribe to events
- Loose coupling (services don't call each other directly)
- Single event, multiple consumers
- Extensibility (add new services without modifying existing)
- Audit trail (all events logged)
```

### 2. **Dual-Mode Services**

```
Each service works TWO ways:
1. Direct API: For testing, simple use cases
2. Event-driven: For production backtesting

Example:
  portfolio.apply_fill(fill)           # Direct
  event_bus.publish(FillEvent(fill))   # Event-driven → portfolio handles it
```

### 3. **DataService Owns ALL Market Data**

```
Corporate actions are market data → DataService provides them
- Prices: DataService.load_symbol()
- Corporate actions: DataService.get_corporate_actions()
- Portfolio just processes what it's told
```

______________________________________________________________________

## Functional Requirements

### Corporate Actions from Algoseek

#### What Algoseek Provides

**Data Structure:**

```python
class AlgoseekBar:
    # ... price fields ...
    AdjustmentFactor: float | None       # Corporate action ratio
    AdjustmentReason: str | None         # Type of action
    CumulativePriceFactor: float         # For dividend adjustments
    CumulativeVolumeFactor: float        # For split adjustments
```

**AdjustmentReason Values:**

| Reason          | Meaning                       | Event Type       |
| --------------- | ----------------------------- | ---------------- |
| `CashDiv`       | Cash dividend                 | Dividend (cash)  |
| `ScriptDiv`     | Stock dividend                | Dividend (stock) |
| `ScriptDivDiff` | Script dividend differential  | Dividend         |
| `Subdiv`        | Subdivision (forward split)   | Split            |
| `Cons`          | Consolidation (reverse split) | Split            |
| `BonusSame`     | Bonus issue                   | Split            |

**Detection Logic (Already Implemented):**

```python
# In AlgoseekBar
def is_dividend(self) -> bool:
    return self.AdjustmentReason in ("CashDiv", "ScriptDiv", "ScriptDivDiff")

def is_split(self) -> bool:
    return self.AdjustmentReason in ("Subdiv", "BonusSame", "ScriptDiv", "Cons")

def get_dividend_amount(self, previous_close: float) -> Decimal | None:
    """
    Calculate dividend from adjustment factor:
    Dividend = (1 - AdjustmentFactor) × Close[T-1]
    """
    if self.is_dividend() and self.AdjustmentFactor:
        adjustment_factor = Decimal(str(self.AdjustmentFactor))
        prev_close = Decimal(str(previous_close))
        dividend = (Decimal("1") - adjustment_factor) * prev_close
        return dividend.quantize(Decimal("0.0001"))
    return None

def get_split_ratio(self) -> Decimal | None:
    """
    Calculate split ratio:
    Split ratio = 1 / AdjustmentFactor

    Examples:
    - 4-for-1 split: AdjustmentFactor=0.25 → ratio=4.0
    - 1-for-4 reverse: AdjustmentFactor=4.0 → ratio=0.25
    """
    if self.is_split() and self.AdjustmentFactor:
        split_ratio = Decimal("1") / Decimal(str(self.AdjustmentFactor))
        return split_ratio.quantize(Decimal("0.01"))
    return None
```

#### Scope for Phase 1.5

**✅ Must Support:**

- Cash dividends (`CashDiv`)
- Forward splits (`Subdiv`)
- Reverse splits (`Cons`)

**⏸️ Defer to Later:**

- Stock dividends (`ScriptDiv`) - More complex, less common
- Bonus issues (`BonusSame`) - Regional specific
- Script dividend differential - Edge case

______________________________________________________________________

## Event Model

### Core Event Types

```python
# src/qtrader/events/events.py

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal
from uuid import uuid4

from qtrader.models.bar import Bar
from qtrader.models.order import Order, Fill


@dataclass(frozen=True)
class Event:
    """
    Base event class.

    All events are immutable (frozen) for safety.
    """
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""


# ============================================
# Market Data Events
# ============================================

@dataclass(frozen=True)
class MarketDataEvent(Event):
    """Base class for market data events"""
    symbol: str = ""


@dataclass(frozen=True)
class PriceBarEvent(MarketDataEvent):
    """
    Price bar received from data feed.

    Published by: DataService
    Consumed by: StrategyContext, Analytics, Reporting
    """
    event_type: str = "price_bar"
    bar: Bar | None = None


@dataclass(frozen=True)
class CorporateActionEvent(MarketDataEvent):
    """
    Corporate action occurred.

    Published by: DataService
    Consumed by: PortfolioService, Analytics, Reporting
    """
    event_type: str = "corporate_action"
    action_type: Literal["dividend", "split"] = "dividend"
    effective_date: datetime | None = None

    # For dividends
    dividend_amount: Decimal | None = None
    dividend_type: Literal["cash", "stock"] = "cash"
    ex_date: datetime | None = None

    # For splits
    split_ratio: Decimal | None = None


# ============================================
# Trading Events
# ============================================

@dataclass(frozen=True)
class OrderEvent(Event):
    """
    Order created by strategy.

    Published by: StrategyContext
    Consumed by: ExecutionService, Analytics, Reporting
    """
    event_type: str = "order"
    order: Order | None = None


@dataclass(frozen=True)
class FillEvent(Event):
    """
    Order filled by execution engine.

    Published by: ExecutionService
    Consumed by: PortfolioService, Analytics, Reporting
    """
    event_type: str = "fill"
    fill: Fill | None = None


# ============================================
# Portfolio Events
# ============================================

@dataclass(frozen=True)
class PositionChangedEvent(Event):
    """
    Position changed (fill applied or corporate action).

    Published by: PortfolioService
    Consumed by: RiskService, Analytics, Reporting
    """
    event_type: str = "position_changed"
    symbol: str = ""
    old_quantity: Decimal | None = None
    new_quantity: Decimal | None = None
    reason: Literal["fill", "split", "dividend"] = "fill"


@dataclass(frozen=True)
class CashChangedEvent(Event):
    """
    Cash balance changed.

    Published by: PortfolioService
    Consumed by: RiskService, Analytics
    """
    event_type: str = "cash_changed"
    old_cash: Decimal | None = None
    new_cash: Decimal | None = None
    reason: str = ""


# ============================================
# Risk Events
# ============================================

@dataclass(frozen=True)
class RiskViolationEvent(Event):
    """
    Risk limit violated.

    Published by: RiskService
    Consumed by: ExecutionService, Reporting
    """
    event_type: str = "risk_violation"
    violation_type: str = ""
    message: str = ""


# ============================================
# Backtest Control Events
# ============================================

@dataclass(frozen=True)
class BacktestStartedEvent(Event):
    """Backtest started"""
    event_type: str = "backtest_started"
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass(frozen=True)
class BacktestEndedEvent(Event):
    """Backtest ended"""
    event_type: str = "backtest_ended"
    total_bars: int = 0
    total_fills: int = 0


@dataclass(frozen=True)
class BarCloseEvent(Event):
    """
    Bar closed (end of bar processing).

    Marks end of processing for current bar.
    Triggers mark-to-market, fee accrual, etc.

    Published by: BacktestEngine
    Consumed by: PortfolioService, Analytics
    """
    event_type: str = "bar_close"
    current_time: datetime | None = None
```

______________________________________________________________________

## EventBus Architecture

### Interface

```python
# src/qtrader/events/event_bus.py

from typing import Protocol, Callable
from collections import defaultdict
from datetime import datetime

from qtrader.events.events import Event
from qtrader.config.logging_config import LoggerFactory

logger = LoggerFactory.get_logger()


class IEventBus(Protocol):
    """Event bus interface for pub/sub messaging"""

    def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.

        Args:
            event: Event to publish
        """
        ...

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
        priority: int = 0
    ) -> None:
        """
        Subscribe to event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle event
            priority: Handler priority (higher = called first)
        """
        ...

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> None:
        """
        Unsubscribe from event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        ...

    def get_history(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int | None = None
    ) -> list[Event]:
        """
        Get event history.

        Args:
            event_type: Filter by event type (None = all)
            since: Filter by timestamp (None = all time)
            limit: Max events to return (None = no limit)

        Returns:
            List of events matching filters
        """
        ...

    def clear_history(self) -> None:
        """Clear event history"""
        ...


class EventBus:
    """
    Synchronous event bus for backtesting.

    Features:
    - Synchronous, deterministic processing (critical for backtesting)
    - Multiple subscribers per event type
    - Priority-based handler ordering
    - Complete event history for replay/debugging
    - Error isolation (one handler failure doesn't stop others)

    Thread Safety: NOT thread-safe (backtesting is single-threaded)
    """

    def __init__(self, max_history: int = 100000):
        """
        Initialize event bus.

        Args:
            max_history: Max events to keep in history (0 = unlimited)
        """
        self._subscribers: dict[str, list[tuple[int, Callable]]] = defaultdict(list)
        self._event_history: list[Event] = []
        self._max_history = max_history

        logger.info("event_bus.initialized", max_history=max_history)

    def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.

        Synchronous processing - handlers called in priority order.
        If a handler raises exception, it's logged but doesn't stop other handlers.
        """
        # Store in history
        self._event_history.append(event)
        if self._max_history > 0 and len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history:]

        # Get handlers for this event type
        handlers = self._subscribers.get(event.event_type, [])

        # Sort by priority (higher first)
        handlers = sorted(handlers, key=lambda x: x[0], reverse=True)

        # Call each handler
        for priority, handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "event_bus.handler_error",
                    event_type=event.event_type,
                    event_id=event.event_id,
                    handler=handler.__name__,
                    error=str(e),
                    exc_info=True
                )

        logger.debug(
            "event_bus.published",
            event_type=event.event_type,
            event_id=event.event_id,
            num_handlers=len(handlers)
        )

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
        priority: int = 0
    ) -> None:
        """Subscribe to event type"""
        self._subscribers[event_type].append((priority, handler))

        logger.info(
            "event_bus.subscribed",
            event_type=event_type,
            handler=handler.__name__,
            priority=priority
        )

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None]
    ) -> None:
        """Unsubscribe from event type"""
        handlers = self._subscribers.get(event_type, [])
        self._subscribers[event_type] = [
            (p, h) for p, h in handlers if h != handler
        ]

        logger.info(
            "event_bus.unsubscribed",
            event_type=event_type,
            handler=handler.__name__
        )

    def get_history(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int | None = None
    ) -> list[Event]:
        """Get event history with filters"""
        events = self._event_history

        # Filter by type
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        # Filter by time
        if since:
            events = [e for e in events if e.timestamp >= since]

        # Limit results
        if limit:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """Clear event history"""
        self._event_history.clear()
        logger.info("event_bus.history_cleared")
```

______________________________________________________________________

## DataService Enhancement

### Corporate Actions API

```python
# Additions to src/qtrader/services/data/interface.py

from qtrader.events.events import CorporateActionEvent

class IDataService(Protocol):
    """Enhanced data service interface"""

    # Existing methods
    def load_symbol(...) -> Iterator[Bar]: ...
    def load_universe(...) -> dict[str, Iterator[Bar]]: ...
    def get_instrument(...) -> Instrument: ...

    # NEW: Corporate actions
    def get_corporate_actions(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> list[CorporateActionEvent]:
        """
        Get corporate actions for symbol in date range.

        Returns events in chronological order.
        Empty list if data source doesn't provide corp actions.

        Args:
            symbol: Ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of CorporateActionEvent
        """
        ...
```

### Implementation

```python
# Additions to src/qtrader/services/data/service.py

class DataService:
    """Enhanced with corporate actions"""

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> list[CorporateActionEvent]:
        """Get corporate actions from adapter"""

        # Get adapter config
        adapter_config = self._build_adapter_config(symbol)

        # Create adapter
        adapter_class = self.resolver.get_adapter_class(adapter_config.adapter)
        adapter = adapter_class(adapter_config)

        # Check if adapter supports corporate actions
        if not hasattr(adapter, 'get_corporate_actions'):
            logger.warning(
                "data_service.no_corporate_actions_support",
                symbol=symbol,
                adapter=type(adapter).__name__,
                message="Adapter does not support corporate actions"
            )
            return []

        # Get corporate actions
        try:
            actions = adapter.get_corporate_actions(symbol, start_date, end_date)

            logger.info(
                "data_service.corporate_actions_loaded",
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                num_actions=len(actions)
            )

            return actions

        except Exception as e:
            logger.error(
                "data_service.corporate_actions_error",
                symbol=symbol,
                error=str(e),
                exc_info=True
            )
            return []
```

______________________________________________________________________

## Algoseek Adapter Enhancement

```python
# Additions to src/qtrader/adapters/algoseek.py

from qtrader.events.events import CorporateActionEvent

class AlgoseekOHLCVendorAdapter:
    """Enhanced with corporate action detection"""

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> list[CorporateActionEvent]:
        """
        Extract corporate actions from Algoseek bars.

        Process:
        1. Read bars for date range
        2. Check each bar for AdjustmentReason
        3. Extract dividend/split details
        4. Create CorporateActionEvent

        Returns:
            List of CorporateActionEvent in chronological order
        """
        # Read bars
        bars_iter = self.read_bars(symbol, start_date, end_date)
        bars = list(bars_iter)  # Materialize to access previous bar

        events = []

        for i, bar in enumerate(bars):
            # Skip if no adjustment
            if not bar.AdjustmentReason:
                continue

            # Handle dividends
            if bar.is_dividend():
                # Need previous close to calculate dividend amount
                if i == 0:
                    logger.warning(
                        "algoseek.dividend_no_previous_close",
                        symbol=symbol,
                        date=bar.Date,
                        message="Cannot calculate dividend amount (no previous bar)"
                    )
                    continue

                previous_close = bars[i - 1].Close
                dividend_amount = bar.get_dividend_amount(previous_close)

                if dividend_amount:
                    event = CorporateActionEvent(
                        timestamp=datetime.combine(bar.Date, datetime.min.time()),
                        symbol=symbol,
                        action_type="dividend",
                        effective_date=datetime.combine(bar.Date, datetime.min.time()),
                        ex_date=datetime.combine(bar.Date, datetime.min.time()),
                        dividend_amount=dividend_amount,
                        dividend_type="cash"
                    )
                    events.append(event)

            # Handle splits
            elif bar.is_split():
                split_ratio = bar.get_split_ratio()

                if split_ratio:
                    event = CorporateActionEvent(
                        timestamp=datetime.combine(bar.Date, datetime.min.time()),
                        symbol=symbol,
                        action_type="split",
                        effective_date=datetime.combine(bar.Date, datetime.min.time()),
                        split_ratio=split_ratio
                    )
                    events.append(event)

        logger.info(
            "algoseek.corporate_actions_extracted",
            symbol=symbol,
            num_dividends=sum(1 for e in events if e.action_type == "dividend"),
            num_splits=sum(1 for e in events if e.action_type == "split"),
            total_events=len(events)
        )

        return events
```

______________________________________________________________________

## Implementation Plan

### Week 1: Event Infrastructure

**Goal:** Working event bus with core event types

#### Tasks

- [ ] Create `src/qtrader/events/` directory
- [ ] Implement `events.py` with all event classes
- [ ] Implement `event_bus.py` with IEventBus and EventBus
- [ ] Add unit tests for EventBus
  - [ ] Test publish/subscribe
  - [ ] Test priority ordering
  - [ ] Test error isolation
  - [ ] Test history
- [ ] Test coverage ≥ 90%

**Deliverable:** Working EventBus with tests

______________________________________________________________________

### Week 2: DataService Corporate Actions

**Goal:** DataService can detect and return corporate actions

#### Tasks

- [ ] Enhance `IDataService` interface with `get_corporate_actions()`
- [ ] Implement `get_corporate_actions()` in `DataService`
- [ ] Enhance `AlgoseekOHLCVendorAdapter` with corporate action detection
- [ ] Add unit tests
  - [ ] Test dividend detection
  - [ ] Test split detection
  - [ ] Test with real Algoseek sample data
  - [ ] Test edge cases (no previous bar, missing fields)
- [ ] Integration test with actual data files
- [ ] Test coverage ≥ 90%

**Deliverable:** DataService returns corporate actions from Algoseek

______________________________________________________________________

### Week 3: Event Publishing (Optional - Can Defer)

**Goal:** DataService publishes events to EventBus

#### Tasks

- [ ] Add optional `event_bus` parameter to DataService
- [ ] Publish `PriceBarEvent` when loading bars
- [ ] Publish `CorporateActionEvent` when detected
- [ ] Add tests for event publishing
- [ ] Update documentation

**Deliverable:** DataService can publish to EventBus

**Note:** This can be deferred to Phase 5 (BacktestEngine) when we actually need event orchestration.

______________________________________________________________________

### Week 4: Documentation & Polish

**Goal:** Complete documentation and examples

#### Tasks

- [ ] API documentation for events
- [ ] Event bus usage guide
- [ ] Corporate action detection guide
- [ ] Example: Using corporate actions
- [ ] Example: Using event bus
- [ ] Update Phase 2 spec with event integration points
- [ ] Run full test suite (all 577+ tests still pass)
- [ ] MyPy clean
- [ ] Coverage report

**Deliverable:** Production-ready event infrastructure

______________________________________________________________________

## Validation Criteria

### Functional

- [ ] ✅ EventBus publishes and delivers events
- [ ] ✅ Multiple subscribers can receive same event
- [ ] ✅ Priority ordering works correctly
- [ ] ✅ Error in one handler doesn't stop others
- [ ] ✅ DataService detects dividends from Algoseek
- [ ] ✅ DataService detects splits from Algoseek
- [ ] ✅ Corporate actions returned in chronological order
- [ ] ✅ Works with missing/incomplete data gracefully

### Technical

- [ ] ✅ All event classes are immutable (frozen dataclasses)
- [ ] ✅ EventBus is deterministic (synchronous)
- [ ] ✅ All methods have type hints
- [ ] ✅ All public methods have docstrings
- [ ] ✅ MyPy passes
- [ ] ✅ Ruff linting passes
- [ ] ✅ Test coverage ≥ 90%
- [ ] ✅ All Phase 1 tests still pass (no regression)

### Integration

- [ ] ✅ Can process real Algoseek sample data
- [ ] ✅ Detects AAPL 4-for-1 split (2020-08-31)
- [ ] ✅ Detects AAPL dividends correctly
- [ ] ✅ Handles edge cases (first bar, missing data)

______________________________________________________________________

## Event Flow Example

### Backtest Scenario

```python
# Setup
event_bus = EventBus()
data_service = DataService(config, event_bus=event_bus)
portfolio = PortfolioService(initial_cash=100000, event_bus=event_bus)

# Portfolio subscribes to corporate actions
event_bus.subscribe("corporate_action", portfolio.handle_corporate_action)

# Run backtest
for date in date_range(start, end):
    # DataService loads data and detects corporate actions
    bars = data_service.load_symbol("AAPL", date, date)
    corp_actions = data_service.get_corporate_actions("AAPL", date, date)

    # Publish corporate actions
    for action in corp_actions:
        event_bus.publish(action)
        # → Portfolio receives event
        # → Portfolio processes split/dividend

    # Update prices
    for bar in bars:
        event_bus.publish(PriceBarEvent(symbol="AAPL", bar=bar))
        # → Strategy receives event
        # → Strategy makes decision

    # Strategy places order
    order = strategy.generate_signal()
    event_bus.publish(OrderEvent(order=order))
    # → Execution receives event

    # Execution fills order
    fill = execution.simulate_fill(order)
    event_bus.publish(FillEvent(fill=fill))
    # → Portfolio receives event
    # → Portfolio updates position
    # → Analytics receives event
    # → Reporting receives event

# Audit trail
all_events = event_bus.get_history()
dividends = event_bus.get_history(event_type="corporate_action")
fills = event_bus.get_history(event_type="fill")
```

______________________________________________________________________

## What Events to Publish?

### Phase 1.5 (Now)

**Market Data:**

- `CorporateActionEvent` - When detected from data

**Control:**

- (None yet - BacktestEngine will add later)

### Phase 2 (Portfolio)

**Portfolio:**

- `PositionChangedEvent` - When position changes
- `CashChangedEvent` - When cash changes

### Phase 3 (Execution)

**Trading:**

- `FillEvent` - When order filled

### Phase 5 (BacktestEngine)

**Control:**

- `BacktestStartedEvent`
- `BacktestEndedEvent`
- `BarCloseEvent` - End of bar processing
- `PriceBarEvent` - New bar received

### Phase 6 (Strategy)

**Trading:**

- `OrderEvent` - When strategy creates order

### Phase 4 (Risk)

**Risk:**

- `RiskViolationEvent` - When risk limit violated

______________________________________________________________________

## Migration Path

### Phase 1 (Current) → Phase 1.5

**No Breaking Changes:**

- DataService API stays the same
- New method added: `get_corporate_actions()`
- EventBus is optional (can be None)
- All existing tests pass

### Phase 1.5 → Phase 2

**Portfolio Enhancement:**

```python
# Phase 2: Portfolio can work both ways
portfolio = PortfolioService(initial_cash=100000)

# Option 1: Direct API (for testing)
portfolio.apply_fill(fill)
portfolio.process_split(symbol, ratio, date)

# Option 2: Event-driven (for production)
portfolio = PortfolioService(initial_cash=100000, event_bus=event_bus)
event_bus.publish(FillEvent(fill))
event_bus.publish(CorporateActionEvent(...))
# Portfolio handles events automatically
```

______________________________________________________________________

## Next Phase

👉 **[Phase 2: PortfolioService](phase2_portfolio_service_v3.md)** (with event support)

After Phase 1.5:

- Portfolio can subscribe to `CorporateActionEvent`
- Portfolio can subscribe to `FillEvent`
- Portfolio can publish `PositionChangedEvent`
- Works standalone OR event-driven

______________________________________________________________________

**Phase Status:** 📝 Ready to Implement **Dependencies:** Phase 1 (DataService) ✅ Complete **Last Updated:** October 17, 2025
