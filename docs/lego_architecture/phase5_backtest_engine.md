# Phase 5: BacktestEngine Implementation (Event-Driven Orchestrator)

## Overview

**Goal:** Create a pure event-driven orchestration engine (\<200 lines) that coordinates all services via EventBus. No business logic, no direct service calls—just event publishing and timing control.

**Start Date:** October 21, 2025\
**Duration:** 2-3 weeks\
**Status:** 🚧 **IN PROGRESS**\
**Complexity:** High\
**Priority:** Critical - Ties everything together

**Key Principle:** BacktestEngine publishes orchestration events. Services subscribe, process, and publish their own events. Engine never touches service state directly.

______________________________________________________________________

## Architecture Philosophy

**Event-Driven Everything:**

- BacktestEngine = Pure orchestrator (publishes timing/coordination events)
- Services = Independent subscribers/publishers
- No direct service calls
- No shared state
- Services own their internal state

**Master Config Pattern:**

- Single `backtest.yaml` configuration
- Each service gets its config section
- Services instantiated via `from_config()` factory methods
- Easy to version, test, and swap configurations

**Service State Ownership:**

- Each service maintains internal state
- Services publish state change events when relevant
- Other services cache what they need
- BacktestEngine never reads or modifies service state

______________________________________________________________________

## Target Architecture

### Engine Structure

```python
class BacktestEngine:
    """
    Pure event-driven orchestration engine.

    Responsibilities:
    - Publish timing/coordination events (BarEvent, triggers)
    - Manage simulation time progression
    - Coordinate warmup phase
    - Generate final reports

    Does NOT:
    - Call service methods directly
    - Access service state
    - Contain business logic
    - Make trading decisions
    """

    def __init__(
        self,
        event_bus: EventBus,
        data_service: IDataService,
        portfolio_service: IPortfolioService,
        execution_service: IExecutionService,
        risk_service: IRiskService,
        strategy_service: IStrategyService,
        start_date: datetime,
        end_date: datetime,
        warmup_bars: int = 0,
    ):
        """All services injected, communicate via event_bus only."""
        self._event_bus = event_bus
        self._data = data_service
        self._start = start_date
        self._end = end_date
        self._warmup_bars = warmup_bars

        # Services are wired but only accessed for iteration
        # All communication happens through event_bus

    @classmethod
    def from_config(cls, config_path: str) -> "BacktestEngine":
        """
        Factory method: Config → Services → Engine

        Loads master config, instantiates all services with their
        config sections, wires them to event bus, returns engine.
        """
        pass

    def run(self) -> BacktestResult:
        """
        Execute backtest event loop.

        Returns summary results. Detailed logs/reports handled by
        services or reporter service.
        """
        pass
```

### Master Configuration Structure

```yaml
# config/backtest.yaml
backtest:
  start_date: "2020-01-01"
  end_date: "2020-12-31"
  initial_capital: 1000000
  warmup_bars: 100
  universe: ["AAPL", "GOOGL", "MSFT"]

data:
  source: "schwab"  # or "algoseek"
  data_path: "data/us-equity-daily-adjusted-schwab"

portfolio:
  initial_capital: 1000000  # redundant with backtest, or derive?
  commission_model: "fixed"
  commission_rate: 0.001
  slippage_model: "fixed"
  slippage_bps: 5

risk:
  cash_buffer_pct: 0.02
  budgets:
    - strategy_id: "momentum_v1"
      capital_weight: 0.6
    - strategy_id: "mean_reversion"
      capital_weight: 0.4
  sizing:
    momentum_v1:
      fraction: 0.03
    mean_reversion:
      fraction: 0.02
  concentration:
    max_position_pct: 0.10
  leverage:
    max_gross: 2.0
    max_net: 1.0

execution:
  fill_policy: "next_bar"  # or "immediate", "realistic"
  commission_model: "fixed"
  slippage_model: "fixed"

strategies:
  - path: "strategies/momentum_v1.py"
    strategy_id: "momentum_v1"
    config:
      lookback: 20
      threshold: 0.05
  - path: "strategies/mean_reversion.py"
    strategy_id: "mean_reversion"
    config:
      window: 30
      entry_z: 2.0
      exit_z: 0.5
```

## Event Loop Flow (Event-Driven)

### High-Level Flow

```
1. Load Configuration
   - Parse backtest.yaml
   - Instantiate services via from_config()
   - Wire all services to EventBus

2. Initialization Phase
   - Publish BacktestStartEvent
   - Services initialize (load data, set up state)

3. Warmup Phase (if warmup_bars > 0)
   - Iterate through warmup bars
   - Publish BarEvent for each bar
   - Services process bars (build indicators)
   - NO signal generation (strategies lack data)

4. Main Trading Loop
   - For each bar in [start_date, end_date]:
     a. Publish BarEvent (per symbol)
     b. Publish ValuationTriggerEvent
     c. Publish RiskEvaluationTriggerEvent
     d. (Services handle everything via subscriptions)

5. Finalization Phase
   - Publish BacktestEndEvent
   - Services finalize (close positions, generate reports)
   - Return BacktestResult

6. Reporting
   - Services publish or save their outputs
   - Optional: ReportingService aggregates
```

### Detailed Event Flow (Per Bar)

```python
# BacktestEngine main loop (simplified)
for ts, multi_bar in self._data.get_bars(self._start, self._end):
    # Step 1: Publish bar events (all services receive)
    for symbol, bar in multi_bar.items():
        self._event_bus.publish(
            BarEvent(
                symbol=symbol,
                bar=bar,
                ts=ts,
                is_warmup=False  # or True during warmup
            )
        )

    # What happens automatically via subscriptions:
    # → PortfolioService.on_bar(): updates position values with latest prices
    # → ExecutionService.on_bar(): checks pending orders, attempts fills
    #     - If filled: publishes FillEvent
    # → StrategyService.on_bar(): routes bar to each strategy
    #     - Each strategy processes bar independently
    #     - If signal conditions met: publishes SignalEvent
    # → RiskService.on_signal(): buffers incoming signals

    # → PortfolioService.on_fill(): applies fills to positions

    # Step 2: Trigger portfolio valuation
    self._event_bus.publish(ValuationTriggerEvent(ts=ts))

    # → PortfolioService.on_valuation_trigger(): calculates portfolio metrics
    #     - Publishes PortfolioStateEvent with equity, positions, cash, etc.
    # → RiskService.on_portfolio_state(): caches latest portfolio state

    # Step 3: Trigger risk evaluation (batch process all buffered signals)
    self._event_bus.publish(RiskEvaluationTriggerEvent(ts=ts))

    # → RiskService.on_risk_evaluation_trigger():
    #     - Evaluates all buffered signals at once
    #     - Sizes positions based on allocated capital
    #     - Checks concentration & leverage limits
    #     - Publishes OrderApprovedEvent (per approved signal)
    #     - Publishes OrderRejectedEvent (per rejected signal)
    #     - Clears signal buffer

    # → ExecutionService.on_order_approved():
    #     - Creates OrderEvent
    #     - Adds to pending order book
    #     - Will fill on next bar (when symbol's BarEvent arrives)
```

### Service Responsibilities

**BacktestEngine:**

- Iterate through time (bars)
- Publish `BarEvent`, `ValuationTriggerEvent`, `RiskEvaluationTriggerEvent`
- Manage warmup flag
- Return final results

**DataService:**

- Provide bar iterator (`get_bars()`)
- Load data from disk/cache
- Handle bar alignment (already implemented)

**StrategyService (NEW):**

- Load external strategy `.py` files
- Instantiate multiple `Strategy` objects
- Subscribe to `BarEvent`, route to each strategy
- Strategies publish `SignalEvent` when conditions met

**RiskService (Phase 4 - already implemented):**

- Subscribe to `SignalEvent` → buffer signals
- Subscribe to `PortfolioStateEvent` → cache state
- Subscribe to `RiskEvaluationTriggerEvent` → batch evaluate signals
- Publish `OrderApprovedEvent` or `OrderRejectedEvent`

**ExecutionService (Phase 3 - needs updates):**

- Subscribe to `OrderApprovedEvent` → create order, add to pending book
- Subscribe to `BarEvent` → check pending orders, attempt fills
- Publish `FillEvent` when orders filled
- Apply commission/slippage models

**PortfolioService (Phase 2 - needs updates):**

- Subscribe to `BarEvent` → update position values
- Subscribe to `FillEvent` → apply fills to positions
- Subscribe to `ValuationTriggerEvent` → calculate metrics
- Publish `PortfolioStateEvent` with current state

______________________________________________________________________

## New Events for Phase 5

### Orchestration Events

```python
@dataclass(frozen=True)
class BacktestStartEvent(Event):
    """Published at backtest initialization."""
    event_type: str = "backtest_start"
    ts: datetime = field(default_factory=datetime.now)
    start_date: datetime = ...
    end_date: datetime = ...
    initial_capital: Decimal = ...
    universe: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class BacktestEndEvent(Event):
    """Published at backtest completion."""
    event_type: str = "backtest_end"
    ts: datetime = field(default_factory=datetime.now)
    final_equity: Decimal = ...
    total_return: float = ...

@dataclass(frozen=True)
class ValuationTriggerEvent(Event):
    """Triggers portfolio valuation and metric calculation."""
    event_type: str = "valuation_trigger"
    ts: datetime = field(default_factory=datetime.now)

@dataclass(frozen=True)
class PortfolioStateEvent(Event):
    """
    Published by PortfolioService after valuation.
    Contains current portfolio snapshot.
    """
    event_type: str = "portfolio_state"
    ts: datetime = field(default_factory=datetime.now)
    equity: Decimal = ...
    cash: Decimal = ...
    gross_exposure: Decimal = ...
    net_exposure: Decimal = ...
    positions: dict[str, Position] = field(default_factory=dict)
```

### Updated BarEvent

```python
@dataclass(frozen=True)
class BarEvent(Event):
    """
    Published by BacktestEngine for each symbol/bar.
    Enhanced with warmup flag.
    """
    event_type: str = "bar"
    symbol: str = ""
    bar: Bar = ...
    ts: datetime = field(default_factory=datetime.now)
    is_warmup: bool = False  # NEW: True during warmup phase
```

______________________________________________________________________

## Implementation Plan

### Week 1: Core Engine & Configuration (Days 1-5)

**Day 1: Master Config Loader**

- Create `BacktestConfig` dataclass
- Implement `load_backtest_config(path: str) -> BacktestConfig`
- Validation for all config sections
- Tests for config loading

**Day 2: Service Factory Pattern**

- Each service gets `from_config(config_section, event_bus)` class method
- Update existing services (Portfolio, Execution, Risk)
- Test service instantiation from config

**Day 3: StrategyService Implementation**

- `IStrategyService` interface
- `StrategyService` loads external `.py` files
- Instantiate multiple `Strategy` objects
- Route `BarEvent` to each strategy
- Tests with mock strategies

**Day 4: BacktestEngine Core**

- `BacktestEngine` class with event loop
- `from_config()` factory method
- Main `run()` method skeleton
- Basic iteration through bars
- Publish `BarEvent` per symbol

**Day 5: Warmup Phase Logic**

- Implement warmup bar iteration
- Set `is_warmup=True` on BarEvents
- Prevent signal generation during warmup
- Tests for warmup behavior

### Week 2: Service Updates & Event Flow (Days 6-10)

**Day 6: PortfolioService Updates**

- Subscribe to `BarEvent` → update position values
- Subscribe to `ValuationTriggerEvent` → calculate metrics
- Publish `PortfolioStateEvent`
- Tests for valuation trigger

**Day 7: ExecutionService Updates**

- Internal order book (`_pending_orders`)
- Subscribe to `OrderApprovedEvent` → queue orders
- Subscribe to `BarEvent` → attempt fills
- Publish `FillEvent` on successful fills
- Tests for order book management

**Day 8: RiskService Integration**

- Verify Phase 4 batch evaluation still works
- Subscribe to `PortfolioStateEvent` → cache state
- End-of-bar batch evaluation triggered by `RiskEvaluationTriggerEvent`
- Integration tests with real portfolio state

**Day 9: Event Flow Integration**

- Wire all services together
- Test complete event flow: Bar → Strategy → Signal → Risk → Order → Fill → Portfolio
- Verify event ordering and timing
- Debug any race conditions

**Day 10: Multi-Strategy Coordination**

- Test with 2+ strategies simultaneously
- Verify signal buffering and batch evaluation
- Test capital allocation across strategies
- Verify independent strategy execution

### Week 3: Reporting, Testing & Polish (Days 11-15)

**Day 11: Reporting & Results**

- `BacktestResult` dataclass
- Collect metrics from services
- Generate summary report
- Export trades log (optional)

**Day 12: Integration Tests**

- End-to-end backtest scenarios
- Single strategy test
- Multi-strategy test
- Warmup phase validation
- Order approval/rejection flows

**Day 13: Performance & Optimization**

- Profile event loop
- Optimize hot paths
- Benchmark vs Phase 3 engine
- Memory usage analysis

**Day 14: Documentation**

- Update all service docstrings
- Create usage examples
- Configuration guide
- Migration guide from old engine

**Day 15: Final Testing & Polish**

- Edge case testing
- Error handling validation
- Logging audit
- Code review and cleanup

______________________________________________________________________

## Service Updates Required

### PortfolioService (Phase 2)

**New Methods:**

- `on_bar(event: BarEvent)` - Update position values
- `on_valuation_trigger(event: ValuationTriggerEvent)` - Calculate metrics, publish state

**Changes:**

- Remove direct API calls
- Add event subscriptions in `__init__`
- Publish `PortfolioStateEvent` instead of returning state

### ExecutionService (Phase 3)

**New Attributes:**

- `_pending_orders: dict[str, OrderEvent]` - Internal order book

**New Methods:**

- `on_order_approved(event: OrderApprovedEvent)` - Queue approved orders
- `on_bar(event: BarEvent)` - Check pending orders, attempt fills

**Changes:**

- Remove `submit_order()` direct API
- Add event subscriptions in `__init__`
- Publish `FillEvent` instead of returning fills

### RiskService (Phase 4)

**New Subscriptions:**

- `PortfolioStateEvent` → cache portfolio state (already done in Phase 4)

**Verification:**

- Ensure batch evaluation still works with `RiskEvaluationTriggerEvent`
- Test with real `PortfolioStateEvent` data

### StrategyService (NEW)

**Purpose:** Load and orchestrate multiple external strategy files

**Responsibilities:**

- Load `.py` files containing `Strategy` + `StrategyConfig`
- Instantiate strategy objects
- Route `BarEvent` to each strategy
- Strategies publish `SignalEvent` independently

**Interface:**

```python
class IStrategyService(Protocol):
    def on_bar(self, event: BarEvent) -> None:
        """Route bar to all strategies."""

    def on_backtest_start(self, event: BacktestStartEvent) -> None:
        """Initialize all strategies."""

    def on_backtest_end(self, event: BacktestEndEvent) -> None:
        """Finalize all strategies."""
```

### DataService (Phase 1)

**No Changes Required:**

- Already provides `get_bars()` iterator
- Already handles bar alignment
- Just needs to be accessed by BacktestEngine

______________________________________________________________________

## Design Decisions & Rationale

### Why Event-Driven?

1. **Decoupling:** Services never depend on each other directly
1. **Testability:** Each service can be tested in isolation
1. **Extensibility:** Add new services without modifying existing ones
1. **Clarity:** Event flow is explicit and auditable
1. **Consistency:** Matches Phase 4 RiskService pattern

### Why Batch Signal Evaluation?

1. **Consistency:** All signals evaluated with same portfolio state
1. **Atomic Time:** All events at timestamp `ts` are consistent
1. **Realistic:** Real trading processes orders at discrete times
1. **Performance:** Batch operations more efficient than per-signal
1. **Already Works:** Phase 4 RiskService proven architecture

### Why Master Config Pattern?

1. **Single Source of Truth:** All settings in one place
1. **Version Control:** Easy to track backtest configurations
1. **Testability:** Swap configs for different test scenarios
1. **Documentation:** Config file is self-documenting
1. **Flexibility:** Each service handles its own config section

### Why Services Own State?

1. **Encapsulation:** Services control their internal state
1. **Independence:** No shared mutable state between services
1. **Testability:** Services can be tested with mock state
1. **Clarity:** State ownership is explicit
1. **Safety:** No accidental state corruption

### Why StrategyService?

1. **Orchestration:** Manages multiple strategies uniformly
1. **Loading:** Handles external `.py` file loading
1. **Routing:** Distributes bars to all strategies
1. **Independence:** Strategies don't know about each other
1. **Scalability:** Easy to add/remove strategies

______________________________________________________________________

## Validation Criteria

### Functionality

- [ ] ✅ End-to-end backtest executes successfully
- [ ] ✅ Multi-strategy support (2+ strategies simultaneously)
- [ ] ✅ Warmup phase prevents signal generation
- [ ] ✅ Event ordering is correct (Bar → Valuation → Risk Evaluation)
- [ ] ✅ All services communicate only via events
- [ ] ✅ Order approval/rejection flow works correctly
- [ ] ✅ Fills applied to portfolio correctly

### Architecture

- [ ] ✅ BacktestEngine < 200 lines
- [ ] ✅ Zero direct service method calls
- [ ] ✅ Zero business logic in engine
- [ ] ✅ All services instantiated from config
- [ ] ✅ Services own their internal state

### Testing

- [ ] ✅ Unit tests for BacktestEngine
- [ ] ✅ Integration tests with all services
- [ ] ✅ Multi-strategy integration test
- [ ] ✅ Warmup phase test
- [ ] ✅ Event flow validation tests
- [ ] ✅ Config loading tests

### Quality

- [ ] ✅ All type hints correct
- [ ] ✅ All docstrings complete
- [ ] ✅ Logging at appropriate levels
- [ ] ✅ Error handling comprehensive
- [ ] ✅ Code coverage > 85%

### Performance

- [ ] ✅ Backtest completes in reasonable time
- [ ] ✅ Memory usage stable
- [ ] ✅ No event queue backlog
- [ ] ✅ Performance within 20% of manual implementation

______________________________________________________________________

## Known Limitations (Phase 5 Scope)

**Out of Scope:**

- Live trading mode (Phase 6+)
- Real-time data feeds (Phase 6+)
- Distributed backtesting (Future)
- Advanced reporting/visualization (Future)
- Strategy optimization/parameter search (Future)
- Risk metrics calculation (Future - separate service)
- Performance attribution (Future - separate service)

**These are intentional scope decisions for Phase 5 MVP. Focus is on clean event-driven architecture.**

______________________________________________________________________

## Next Phases

### Phase 6: Strategy Context Refactoring

- Simplify strategy API
- Remove old context object
- Pure event-driven strategies
- Strategy lifecycle management

### Phase 7: Live Trading Adapter

- Real-time event processing
- Broker integration
- Order management system
- Risk monitoring

### Phase 8: Advanced Reporting

- Performance analytics service
- Risk metrics calculation
- Trade attribution
- Visualization/dashboards

______________________________________________________________________

**Phase Status:** 🚧 **IN PROGRESS**\
**Dependencies:** Phases 1-4 complete\
**Started:** October 21, 2025\
**Target Completion:** Mid-November 2025
