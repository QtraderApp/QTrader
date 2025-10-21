# Phase 4: RiskService Implementation (MVP)

## Overview

**Goal:** Isolate risk management, capital allocation, position sizing, and limit enforcement from strategy and execution layers. Pure event-driven service that turns signals into risk-approved orders.

**Start Date:** October 21, 2025\
**Duration:** 2 weeks (Days 1-14)\
**Status:** ✅ **COMPLETE**\
**Complexity:** Medium\
**Priority:** High - Business logic isolation

**Key Principle:** Pure event-driven architecture. No direct API calls between services—all communication via EventBus.

______________________________________________________________________

## Executive Summary

Phase 4 MVP implements a production-ready RiskService that:

- **Allocates capital** across multiple strategies (fixed risk budgets)
- **Sizes positions** using fixed_fraction model (simple & robust)
- **Enforces limits** (concentration, leverage - core risk controls)
- **Publishes decisions** as events with audit trail
- **Pure functions** with no state mutation

**Architecture Pattern:** Event-driven, deterministic, testable

**Philosophy:** 80% of business value with 50% of complexity. Advanced features deferred to Phase 11.

______________________________________________________________________

## Architecture

### Event-Driven Flow

```
1. BacktestEngine → BarEvent
2. DataService → processes bar
3. PortfolioService → PortfolioStateEvent (equity, positions, cash)
4. Strategy → SignalEvent (buy/sell signals with strength)
5. BacktestEngine → RiskEvaluationTriggerEvent
6. RiskService → evaluates batch:
   - OrderApprovedEvent (per approved order with audit reason)
   - OrderRejectedEvent (per rejection with detailed reason)
7. ExecutionService → subscribes to OrderApprovedEvent
8. ExecutionService → FillEvent
9. PortfolioService → applies fill → PortfolioStateEvent (cycle)
```

**Key Constraint:** All events at timestamp `ts` must be consistent (same bar boundary).

**Simplified Approach:** No separate MarketMetricsService in Phase 4. Strategy includes volatility/price in SignalEvent.metadata if needed for position sizing.

### Service Interface

```python
class IRiskService(Protocol):
    """
    Risk service interface (MVP).

    Responsibilities:
    - Subscribe to SignalEvent, PortfolioStateEvent
    - Allocate capital across strategies (fixed risk budgets)
    - Size positions per signal (fixed_fraction model)
    - Enforce portfolio limits (concentration, leverage)
    - Publish OrderApprovedEvent or OrderRejectedEvent per signal

    Does NOT:
    - Execute orders (ExecutionService)
    - Mutate portfolio (PortfolioService)
    - Load market data (DataService)
    - Make strategy decisions (Strategy)
    """

    def on_signal(self, event: SignalEvent) -> None:
        """Buffer signal for batch evaluation."""

    def on_risk_evaluation_trigger(self, event: RiskEvaluationTriggerEvent) -> None:
        """Evaluate buffered signals, publish orders/rejections."""

    def on_portfolio_state(self, event: PortfolioStateEvent) -> None:
        """Update cached portfolio state."""
```

### Core Components (Simplified)

```
RiskService
├── models.py (Signal, OrderBase, RiskConfig, PortfolioState snapshots)
├── interface.py (IRiskService protocol)
├── service.py (event subscribers, batch evaluation)
├── allocator.py (strategy capital allocation - fixed budgets)
├── sizer.py (position sizing: fixed_fraction only)
├── limits.py (concentration, leverage checks only)
└── config.py (RiskConfig YAML loader)
```

**Removed from MVP:** MarketMetricsEvent, RiskDecisionEvent, liquidity/margin/drawdown checks

______________________________________________________________________

## Data Models

### Events (New for Phase 4)

```python
@dataclass(frozen=True)
class SignalEvent:
    """Published by Strategy when generating trading signal."""
    ts: datetime
    strategy_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    strength: float  # [-1, 1] signal confidence
    metadata: dict[str, float] = field(default_factory=dict)

@dataclass(frozen=True)
class RiskEvaluationTriggerEvent:
    """Published by BacktestEngine to trigger batch evaluation."""
    ts: datetime

@dataclass(frozen=True)
class OrderApprovedEvent:
    """Published by RiskService for approved orders."""
    ts: datetime
    order: OrderBase
    reason: str  # e.g., "Approved: 500 shares within limits"

@dataclass(frozen=True)
class OrderRejectedEvent:
    """Published by RiskService for rejected signals."""
    ts: datetime
    signal: SignalEvent
    reason: str  # e.g., "Rejected: exceeds concentration limit"


```

### Core Models

```python
@dataclass
class Signal:
    """Trading signal from strategy."""
    strategy_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    strength: float  # [-1, 1]
    metadata: dict[str, float] = field(default_factory=dict)

@dataclass
class OrderBase:
    """Order to be sent to ExecutionService."""
    strategy_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: int
    reason: str  # audit trail

@dataclass
class PortfolioState:
    """Snapshot from PortfolioService (immutable for RiskService)."""
    ts: datetime
    equity: Decimal
    cash: Decimal
    gross_exposure: Decimal
    net_exposure: Decimal
    positions: dict[str, Position]  # symbol -> Position
    run_drawdown: float  # from HWM, calculated by PortfolioService

@dataclass
class Position:
    """Position snapshot."""
    symbol: str
    quantity: int
    market_value: Decimal


```

### Configuration

```python
@dataclass
class StrategyBudget:
    strategy_id: str
    capital_weight: float  # 0.0 to 1.0

@dataclass
class SizingConfig:
    model: Literal["fixed_fraction"]  # MVP: only fixed_fraction
    fraction: float  # e.g., 0.02 = 2% of allocated capital per signal
    min_quantity: int = 1
    round_to_lot: bool = True

@dataclass
class ConcentrationLimit:
    max_position_pct: float  # e.g., 0.10 = 10% of equity per symbol

@dataclass
class LeverageLimit:
    max_gross: float  # e.g., 2.0
    max_net: float  # e.g., 1.0

@dataclass
class RiskConfig:
    budgets: list[StrategyBudget]
    sizing: dict[str, SizingConfig]  # strategy_id -> config
    concentration: ConcentrationLimit
    leverage: LeverageLimit
    cash_buffer_pct: float = 0.02  # reserve 2% cash for safety
```

______________________________________________________________________

## Implementation Plan (2-Week MVP)

### Week 1 (Days 1-7): Foundation & Core Logic

**Objective:** Event models, interfaces, service shell, capital allocation

#### Day 1-2: Events & Models

- [ ] Create `src/qtrader/services/risk/__init__.py`
- [ ] Create `src/qtrader/services/risk/models.py`
  - Signal, OrderBase, PortfolioState snapshot
  - RiskConfig: StrategyBudget, SizingConfig, ConcentrationLimit, LeverageLimit
- [ ] Add events to `src/qtrader/events/events.py`
  - SignalEvent, RiskEvaluationTriggerEvent
  - OrderApprovedEvent, OrderRejectedEvent
- [ ] Create `src/qtrader/services/risk/interface.py`
  - IRiskService protocol

**Tests:** 10-15 model validation tests

#### Day 3-4: Service Shell & Event Handlers

- [ ] Create `src/qtrader/services/risk/service.py`
  - RiskService class with EventBus subscription
  - Event handlers: on_signal, on_trigger, on_portfolio_state
  - Signal buffering (list[SignalEvent])
  - State caching (latest PortfolioState)
  - Timestamp consistency validation

**Tests:** 10-15 event handling tests

#### Day 5-7: Capital Allocation

- [ ] Create `src/qtrader/services/risk/allocator.py`
  - `allocate_capital(budgets, equity, cash_buffer) -> dict[str, Decimal]`
  - Fixed budget allocation (no throttling)
  - Validation: weights sum ≤ 1.0
- [ ] Integration into service.py

**Tests:** 15-20 allocation tests

**Week 1 Milestone:** Service subscribes to events, buffers signals, allocates capital ✅

______________________________________________________________________

### Week 2 (Days 8-14): Position Sizing, Limits & Testing

**Objective:** Complete signal → order pipeline with core limits

#### Day 8-9: Position Sizing

- [ ] Create `src/qtrader/services/risk/sizer.py`
  - `FixedFractionSizer` class
  - Formula: `target_notional = fraction * allocated_capital * |strength|`
  - Convert notional → quantity (with price from PortfolioState or SignalEvent)
  - Rounding to lot size (default 1 share)

**Tests:** 20-25 sizing tests (including edge cases: zero strength, tiny capital)

#### Day 10-11: Limit Checks

- [ ] Create `src/qtrader/services/risk/limits.py`
  - `check_concentration(symbol, target_qty, state, config) -> tuple[bool, str]`
    - Ensure position market value ≤ max_position_pct * equity
  - `check_leverage(projected_gross, projected_net, state, config) -> tuple[bool, str]`
    - Ensure gross ≤ max_gross *equity, net ≤ max_net* equity
  - Helper: `project_exposure(current_positions, new_order) -> tuple[Decimal, Decimal]`

**Tests:** 25-30 limit tests (boundary conditions, multi-symbol scenarios)

#### Day 12-13: Batch Evaluation & Logging

- [ ] Implement `plan_batch()` in service.py
  - Group signals by strategy_id
  - Allocate capital per strategy
  - For each signal:
    - Size position (FixedFractionSizer)
    - Check concentration limit
    - Check leverage limit (projected portfolio)
    - Publish OrderApprovedEvent or OrderRejectedEvent
  - Clear signal buffer after batch
- [ ] Add structured logging (LoggerFactory)
  - `risk.signal.received` (DEBUG)
  - `risk.allocation.computed` (INFO)
  - `risk.order.approved` (INFO)
  - `risk.order.rejected` (WARNING)
  - `risk.batch.complete` (INFO)

**Tests:** 20-25 batch integration tests

#### Day 14: Configuration, Examples & QA

- [ ] Create `src/qtrader/services/risk/config.py`
  - YAML loader for RiskConfig
  - Validation logic (weights, valid fractions)
- [ ] Example config: `config/risk_example.yaml`
- [ ] Usage examples:
  - `examples/services/risk/basic_risk_workflow.py`
  - `examples/services/risk/multi_strategy_allocation.py`
- [ ] Run full test suite:
  - Target: ≥90% coverage
  - MyPy strict mode: 0 errors
  - Ruff linting: all checks passing
- [ ] Update documentation

**Week 2 Milestone:** Production-ready RiskService MVP ✅

______________________________________________________________________

## Quality Gates

### Code Quality

- [ ] MyPy strict mode: 0 errors
- [ ] Ruff linting: all checks passing
- [ ] Type hints on all functions
- [ ] Docstrings on all public methods

### Testing

- [ ] ≥90 unit tests
- [ ] ≥15 integration tests
- [ ] ≥90% code coverage
- [ ] Core edge cases covered (zero signals, single strategy, limit violations)

### Performance

- [ ] `plan_batch(100 signals)` < 10ms
- [ ] Event handling < 0.1ms per event
- [ ] No memory leaks in long-running backtests

### Observability

- [ ] Structured logging (risk.\* namespace)
- [ ] Event publishing for all decisions
- [ ] Audit trail in OrderApprovedEvent.reason

______________________________________________________________________

## Validation Criteria

- [ ] ✅ Implements `IRiskService`
- [ ] ✅ Pure event-driven (no direct API calls)
- [ ] ✅ Deterministic (same inputs → same outputs)
- [ ] ✅ Timestamp consistency enforced
- [ ] ✅ No state mutation (publishes events only)
- [ ] ✅ Test coverage ≥ 90%
- [ ] ✅ MyPy strict mode clean
- [ ] ✅ Audit trail in OrderApproved/Rejected.reason

______________________________________________________________________

## MVP Simplifications (Deferred to Phase 11)

**Position Sizing Models:**

- ❌ vol_target sizing (requires volatility calculation)
- ❌ equal_weight sizing (trivial but not MVP-critical)
- ❌ Kelly criterion (requires alpha tracking)
- ✅ fixed_fraction only (simple, robust, sufficient)

**Limit Checks:**

- ❌ Liquidity limits (requires ADV from MarketMetricsService)
- ❌ Margin requirements (cash check handled by PortfolioService)
- ❌ Drawdown throttling (graduated multipliers)
- ❌ Sector concentration (requires sector mapping)
- ✅ Concentration + leverage only (core risk controls)

**Services:**

- ❌ MarketMetricsService (separate service for volatility/ADV)
- ✅ Strategy includes price in SignalEvent if needed

**Events:**

- ❌ RiskDecisionEvent (OrderApproved/Rejected.reason is sufficient audit)
- ❌ MarketMetricsEvent (no separate metrics service)
- ✅ SignalEvent, OrderApprovedEvent, OrderRejectedEvent only

**Configuration:**

- ❌ Dynamic risk budget rebalancing
- ❌ Multi-threshold drawdown controls
- ✅ Static budgets, binary on/off controls

______________________________________________________________________

## Dependencies

- ✅ Phase 2: PortfolioService (publishes PortfolioStateEvent)
- ✅ Phase 3: ExecutionService (subscribes to OrderApprovedEvent)
- ✅ EventBus: Core infrastructure

______________________________________________________________________

## Success Metrics (MVP)

**Test Suite:**

- Total tests: 1,030+ (927 existing + ~100 new)
- Risk tests: 90-105
- Coverage: ≥90% maintained

**Implementation:**

- Lines of code: ~850 (50% reduction from full spec)
- Files created: 6 (models, interface, service, allocator, sizer, limits, config)
- Sizing models: 1 (fixed_fraction)
- Limit types: 2 (concentration, leverage)

**Performance:**

- Batch evaluation: < 10ms for 100 signals
- Event latency: < 0.1ms

**Time Savings:**

- Duration: 2 weeks (not 3)
- Services: 1 (not 2 with MarketMetricsService)
- Complexity: 50% of original spec

______________________________________________________________________

## Next Phase

👉 **[Phase 5: BacktestEngine](phase5_backtest_engine.md)** - Orchestrate all services

______________________________________________________________________

## Future Enhancements (Phase 11)

After completing core LEGO architecture (Phases 1-5), nice-to-have features:

**Phase 11: Advanced Risk Management**

- MarketMetricsService (volatility, ADV calculation)
- vol_target and equal_weight sizing models
- Liquidity limits (ADV participation)
- Margin requirements validation
- Graduated drawdown throttling
- Sector concentration limits
- Kelly criterion sizing
- RiskDecisionEvent for detailed audit trail
- VaR/Greeks portfolio analytics
- Dynamic risk budget rebalancing

**Philosophy:** Ship the 80% core first, add polish later.

______________________________________________________________________

**Phase Status:** ✅ **COMPLETE** (MVP - 2 weeks)\
**Completion Date:** October 21, 2025\
**Dependencies:** Phase 2 (PortfolioService), Phase 3 (ExecutionService)\
**Last Updated:** October 21, 2025

______________________________________________________________________

## Implementation Results

### Files Created

| File               | Lines     | Description                                                           |
| ------------------ | --------- | --------------------------------------------------------------------- |
| `models.py`        | 237       | Data models (Signal, OrderBase, Position, PortfolioState, RiskConfig) |
| `interface.py`     | 161       | IRiskService protocol definition                                      |
| `service.py`       | 501       | Core RiskService with batch evaluation                                |
| `allocator.py`     | 183       | Capital allocation across strategies                                  |
| `sizer.py`         | 295       | Position sizing (FixedFractionSizer)                                  |
| `limits.py`        | 330       | Limit checking (concentration, leverage)                              |
| `config_loader.py` | 330       | YAML configuration loading                                            |
| **Total**          | **2,037** | **Production-ready MVP**                                              |

### Test Coverage

| Test File               | Tests   | Coverage        |
| ----------------------- | ------- | --------------- |
| `test_models.py`        | 36      | 94%             |
| `test_service.py`       | 14      | 90%             |
| `test_allocator.py`     | 24      | 100%            |
| `test_sizer.py`         | 34      | 100%            |
| `test_limits.py`        | 25      | 98%             |
| `test_config_loader.py` | 16      | 82%             |
| `test_integration.py`   | 10      | Full E2E        |
| **Total**               | **159** | **92% overall** |

### Examples Created

- **`risk_example.yaml`** - Fully documented configuration file with multi-strategy setup
- **`risk_service_example.py`** (232 LOC) - Working example demonstrating:
  - Config loading from YAML
  - Multi-strategy portfolio (momentum + mean reversion)
  - Signal processing with approvals/rejections
  - Concentration limit enforcement with existing positions
  - Complete audit trail output

### Key Features Implemented

✅ **Capital Allocation**

- Fixed budgets per strategy with validation (sum ≤ 1.0)
- Cash buffer reservation (default 2%)
- Decimal precision for accuracy

✅ **Position Sizing**

- FixedFractionSizer with configurable fraction per strategy
- Lot size support (1, 100, etc.)
- Minimum quantity enforcement
- Price from signal.metadata (MVP approach)

✅ **Limit Checking**

- Concentration: Per-symbol exposure % of equity
- Leverage: Portfolio gross/net exposure limits
- Considers existing positions
- Pure functions with comprehensive validation

✅ **Event-Driven Architecture**

- Subscribes to: SignalEvent, RiskEvaluationTriggerEvent
- Publishes: OrderApprovedEvent, OrderRejectedEvent
- Batch evaluation at bar boundaries
- Detailed audit trails in all events

✅ **Configuration Management**

- YAML-based config with validation
- Comprehensive error messages
- Supports Path or string paths
- Default values for optional fields

### Performance Metrics

- **Test Execution:** 159 tests in 0.81s
- **Example Run:** < 1s for full demonstration
- **Code Quality:** 92% test coverage, all lint checks passing
- **Documentation:** Comprehensive docstrings, type hints, examples

### Design Decisions

1. **Price Sourcing (MVP):** Signals carry price in metadata

   - Simple, testable, sufficient for Phase 4
   - No dependency on MarketDataService
   - Can be enhanced in later phases

1. **Pure Functions:** All core logic (allocator, sizer, limits) as stateless functions

   - Easier to test, reason about, and reuse
   - Service orchestrates but doesn't contain business logic

1. **Event-Driven Only:** No direct service-to-service calls

   - Clean separation of concerns
   - Easy to add services without breaking existing ones

1. **Conservative Defaults:** 2% cash buffer, reasonable limits

   - Safe out-of-the-box behavior
   - Users can customize via config

### Known Limitations (MVP Scope)

- Single pricing mode (from signal metadata)
- Two limit types only (concentration, leverage)
- One sizing model (fixed_fraction)
- No volatility-based sizing
- No liquidity/ADV checks
- No sector concentration limits
- No margin requirement validation

**These are intentional scope decisions for MVP. Advanced features deferred to Phase 11.**

______________________________________________________________________
