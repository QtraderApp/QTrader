# Phase 3: ExecutionService - Detailed Implementation Plan

**Branch**: `feature/lego-phase3-execution-service` **Duration**: 3-4 weeks **Start Date**: October 21, 2025 **Dependencies**: Phase 2 (PortfolioService) ✅ Complete

## Executive Summary

Extract order execution logic into a standalone, testable service that:

- Validates and tracks orders
- Simulates realistic fills based on market data
- Calculates commissions and slippage
- Returns Fill objects for portfolio application
- **Does NOT** modify portfolio directly (separation of concerns)

## Week-by-Week Breakdown

### Week 1: Foundation & Models (Days 1-7)

**Objective**: Create core data models and service interface

#### Day 1-2: Models & Types

- [ ] **Task 1.1**: Create `src/qtrader/services/execution/models.py`

  - Order models (Market, Limit, Stop, MOC)
  - Fill model with commission/slippage details
  - OrderState enum (PENDING, SUBMITTED, PARTIAL, FILLED, CANCELLED, REJECTED, EXPIRED)
  - OrderSide enum (BUY, SELL)
  - OrderType enum (MARKET, LIMIT, STOP, MARKET_ON_CLOSE)
  - TimeInForce enum (DAY, GTC, IOC, FOK)
  - FillDecision dataclass (should_fill, fill_price, reason, next_bar flag)

- [ ] **Task 1.2**: Create `src/qtrader/services/execution/interface.py`

  ```python
  class IExecutionService(Protocol):
      def submit_order(self, order: Order) -> str  # Returns order_id
      def on_bar(self, symbol: str, bar: Bar, ts: datetime) -> list[Fill]
      def get_pending_orders(self, symbol: Optional[str] = None) -> list[Order]
      def get_filled_orders(self, symbol: Optional[str] = None) -> list[Order]
      def cancel_order(self, order_id: str) -> bool
      def get_order(self, order_id: str) -> Optional[Order]
  ```

- [ ] **Task 1.3**: Create order validation logic

  - Validate order parameters (qty > 0, limit_price for LIMIT orders, etc.)
  - Validate order state transitions
  - Document validation rules

**Deliverables**: Models, interface, validation logic **Tests**: 15-20 unit tests for models and validation **QA Check**: MyPy clean, Ruff clean, tests passing

#### Day 3-4: Fill Policy Foundation

- [ ] **Task 1.4**: Create `src/qtrader/services/execution/fill_policy.py`

  - FillPolicy class with configuration
  - Conservative mode implementation:
    - Market: fills at next bar open
    - MOC: fills at current bar close ± slippage
    - Limit Buy: if low ≤ limit, fill at min(limit, close)
    - Limit Sell: if high ≥ limit, fill at max(limit, close)
    - Stop Buy: if high ≥ stop, fill at max(stop, close) + slippage
    - Stop Sell: if low ≤ stop, fill at min(stop, close) - slippage

- [ ] **Task 1.5**: Implement `evaluate_order()` method

  - Route to appropriate evaluation based on order type
  - Return FillDecision with price and reason
  - Handle close-only bars (skip limit/stop evaluation)

**Deliverables**: FillPolicy with all order types **Tests**: 25-30 unit tests covering all order types and edge cases **QA Check**: All fill policy tests passing

#### Day 5-7: Service Implementation

- [ ] **Task 1.6**: Create `src/qtrader/services/execution/service.py`

  - ExecutionService class implementing IExecutionService
  - Order tracking (pending_orders, filled_orders dicts)
  - submit_order() with validation
  - on_bar() to process pending orders
  - Order state management

- [ ] **Task 1.7**: Implement basic fill generation

  - Generate Fill objects from FillDecision
  - Calculate commissions
  - Update order state (PENDING → FILLED)
  - Track filled orders

**Deliverables**: Basic ExecutionService with order lifecycle **Tests**: 20-25 unit tests for service methods **QA Check**: Tests passing, coverage ≥ 85%

**Week 1 Milestone**: Core execution service with basic fill simulation ✅

______________________________________________________________________

### Week 2: Advanced Fill Logic (Days 8-14)

**Objective**: Add sophisticated fill policies and partial fills

#### Day 8-9: Commission Calculator

- [ ] **Task 2.1**: Create `src/qtrader/services/execution/commission.py`

  - CommissionCalculator class
  - Per-share model: `qty * per_share_rate + ticket_min`
  - Per-trade model: `flat_fee` or `percentage * notional`
  - Tiered rates support (volume brackets)
  - Commission caps (maximum per order)

- [ ] **Task 2.2**: Integration with fill generation

  - Calculate commission for each fill
  - Include commission in Fill object
  - Document commission models

**Deliverables**: Commission calculator with multiple models **Tests**: 15-20 unit tests for commission calculations **QA Check**: All edge cases covered

#### Day 10-11: Slippage Models

- [ ] **Task 2.3**: Create `src/qtrader/services/execution/slippage.py`

  - Fixed BPS slippage (current implementation)
  - Volume-based slippage (higher for large orders)
  - Spread-based slippage (half-spread model)
  - Time-of-day slippage (market open/close)

- [ ] **Task 2.4**: Integrate slippage with fill policy

  - Apply slippage based on order type
  - Document slippage calculation
  - Configuration options

**Deliverables**: Slippage models integrated **Tests**: 15-20 unit tests for slippage **QA Check**: Tests passing

#### Day 12-14: Partial Fills & Volume Limits

- [ ] **Task 2.5**: Implement volume participation limits

  - `max_participation` config (e.g., 10% of bar volume)
  - Split orders that exceed limit
  - Partial fill tracking (remaining_qty on Order)
  - Queue unfilled quantity for next bar

- [ ] **Task 2.6**: Implement fill queueing

  - `queue_bars` config (max bars to carry unfilled qty)
  - Expire orders after queue_bars exhausted
  - Update order state (PARTIAL → FILLED or EXPIRED)

- [ ] **Task 2.7**: Add order expiry logic

  - DAY orders expire at end of day
  - GTC orders persist across days
  - IOC orders fill or cancel immediately
  - FOK orders fill completely or cancel

**Deliverables**: Partial fills, volume limits, order expiry **Tests**: 30-35 unit tests covering partial fill scenarios **QA Check**: Complex multi-bar scenarios tested

**Week 2 Milestone**: Advanced fill logic with realistic constraints ✅

______________________________________________________________________

### Week 3: Testing & Polish (Days 15-21)

**Objective**: Comprehensive testing and edge case handling

#### Day 15-16: Unit Test Coverage

- [ ] **Task 3.1**: Comprehensive unit tests

  - All order types (Market, Limit, Stop, MOC)
  - All order states and transitions
  - Edge cases (zero volume bars, extreme prices)
  - Invalid input handling

- [ ] **Task 3.2**: Mock integration tests

  - ExecutionService with MockPortfolioService
  - ExecutionService with MockDataService
  - Verify no direct portfolio manipulation

**Target**: 90% test coverage minimum **Tests**: 50+ unit tests total **QA Check**: All tests passing, coverage report

#### Day 17-18: Integration Tests

- [ ] **Task 3.3**: Integration with PortfolioService

  - Real PortfolioService + ExecutionService
  - Multi-symbol scenarios
  - Corporate actions during pending orders
  - Round-trip: submit order → fill → apply_fill

- [ ] **Task 3.4**: Integration with DataService

  - Real data from DataService
  - Handle missing bars
  - Handle close-only bars
  - Market halts and gaps

**Deliverables**: Integration test suite **Tests**: 20-25 integration tests **QA Check**: All integration tests passing

#### Day 19-21: Documentation & Examples

- [ ] **Task 3.5**: API Documentation

  - Comprehensive docstrings (Google style)
  - Usage examples in docstrings
  - Architecture notes (separation of concerns)

- [ ] **Task 3.6**: Usage examples

  - Create `examples/execution/` directory
  - Example: Simple market order flow
  - Example: Limit order with partial fills
  - Example: Mock execution for strategy testing

- [ ] **Task 3.7**: Week 3 Summary Document

  - Implementation summary
  - Test coverage report
  - Performance notes
  - Known limitations

**Deliverables**: Complete documentation and examples **QA Check**: Documentation reviewed, examples tested

**Week 3 Milestone**: Production-ready ExecutionService ✅

______________________________________________________________________

### Week 4: Performance & Final Polish (Days 22-28)

**Objective**: Optimize, benchmark, and prepare for Phase 4

#### Day 22-23: Performance Optimization

- [ ] **Task 4.1**: Profile execution performance

  - Benchmark order submission (target: \<1ms)
  - Benchmark on_bar processing (target: \<5ms for 100 pending orders)
  - Identify bottlenecks

- [ ] **Task 4.2**: Optimize hot paths

  - Cache fill policy decisions where possible
  - Optimize order lookup (consider OrderedDict for pending_orders)
  - Minimize allocations in on_bar()

**Deliverables**: Performance benchmarks and optimizations **Benchmarks**: Document performance metrics **QA Check**: No performance regressions

#### Day 24-25: Error Handling & Logging

- [ ] **Task 4.3**: Comprehensive error handling

  - Invalid order parameters
  - Order state conflicts
  - Missing market data
  - Portfolio errors during fill application

- [ ] **Task 4.4**: Structured logging

  - Log all order submissions (DEBUG level)
  - Log all fills (INFO level)
  - Log rejections and cancellations (WARNING level)
  - Log errors with context (ERROR level)

**Deliverables**: Robust error handling and logging **QA Check**: Error scenarios tested

#### Day 26-27: Final QA & Documentation

- [ ] **Task 4.5**: Final QA suite

  - Run full test suite (740+ tests from previous phases)
  - Run execution tests (150+ new tests)
  - Verify 90% coverage maintained
  - MyPy strict mode passing
  - Ruff linting clean

- [ ] **Task 4.6**: Phase 3 completion documentation

  - Update `phase3_execution_service.md` with COMPLETE status
  - Create `WEEK4_SUMMARY.md` with statistics
  - Document API changes
  - Migration guide (if needed)

**Deliverables**: Final QA report and documentation **QA Check**: All quality gates passing

#### Day 28: Commit & Merge Preparation

- [ ] **Task 4.7**: Create conventional commits

  - Commit 1: `feat(execution): add models and interface`
  - Commit 2: `feat(execution): implement fill policy and service`
  - Commit 3: `feat(execution): add commission and slippage models`
  - Commit 4: `test(execution): add comprehensive test suite`
  - Commit 5: `docs(execution): add Week 3/4 summaries`

- [ ] **Task 4.8**: Merge to lego-architecture

  - Merge `feature/lego-phase3-execution-service` → `feature/lego-architecture`
  - Delete phase 3 branch
  - Push to origin

**Deliverables**: Clean git history, merge complete **QA Check**: All tests passing after merge

**Week 4 Milestone**: Phase 3 Complete, ready for Phase 4 (RiskService) ✅

______________________________________________________________________

## Directory Structure

```
src/qtrader/services/execution/
├── __init__.py              # Exports: IExecutionService, ExecutionService, Order, Fill
├── interface.py             # IExecutionService protocol
├── service.py               # ExecutionService implementation
├── models.py                # Order, Fill, OrderState, etc.
├── fill_policy.py           # FillPolicy class
├── commission.py            # CommissionCalculator
├── slippage.py              # Slippage models
└── config.py                # ExecutionConfig

tests/unit/services/execution/
├── __init__.py
├── test_models.py           # Order/Fill model tests
├── test_fill_policy.py      # Fill policy tests (all order types)
├── test_commission.py       # Commission calculator tests
├── test_slippage.py         # Slippage model tests
├── test_service.py          # ExecutionService tests
└── test_integration.py      # Integration with Portfolio

tests/integration/services/
└── test_execution_portfolio.py  # End-to-end execution tests

examples/execution/
├── basic_market_order.py
├── limit_order_workflow.py
├── partial_fills_example.py
└── mock_execution_testing.py
```

## Key Decisions & Trade-offs

### 1. **No Direct Portfolio Manipulation**

- ExecutionService returns Fill objects
- Caller applies fills to portfolio
- **Benefit**: Clean separation, easier testing
- **Trade-off**: Slight API verbosity

### 2. **Conservative Fill Policy Default**

- Pessimistic assumptions (market at open, limit at worst case)
- **Benefit**: More realistic backtests
- **Trade-off**: May underestimate performance

### 3. **Volume Participation Limits**

- Default 10% max participation
- Partial fills for large orders
- **Benefit**: Realistic market impact modeling
- **Trade-off**: More complex fill logic

### 4. **Order Queueing**

- Carry unfilled quantity for N bars (default 3)
- Expire after queue exhausted
- **Benefit**: Realistic order persistence
- **Trade-off**: More state to track

## Quality Gates

### Code Quality

- [ ] MyPy strict mode: 0 errors
- [ ] Ruff linting: all checks passing
- [ ] Test coverage: ≥ 90%
- [ ] Docstring coverage: 100% public APIs
- [ ] No circular imports

### Functional Requirements

- [ ] All order types supported (Market, Limit, Stop, MOC)
- [ ] All order states handled correctly
- [ ] Commission calculation accurate
- [ ] Slippage applied correctly
- [ ] Partial fills working
- [ ] Volume limits enforced

### Performance Requirements

- [ ] Order submission: < 1ms
- [ ] on_bar() with 100 pending orders: < 5ms
- [ ] Memory: < 100MB for 10,000 orders

### Testing Requirements

- [ ] 150+ unit tests
- [ ] 20+ integration tests
- [ ] All edge cases covered
- [ ] Mock usage examples provided

## Success Metrics

**Test Suite**:

- Total tests: 890+ (740 existing + 150 new)
- Execution tests: 150+
- Coverage: 90%+ maintained

**Implementation**:

- Lines of code: ~1,500-2,000
- Public APIs: 6-8 main classes
- Configuration options: 15-20

**Documentation**:

- API documentation: Complete
- Usage examples: 4+
- Weekly summaries: 2 (Week 3 + Week 4)

## Risks & Mitigations

| Risk                        | Impact | Mitigation                                         |
| --------------------------- | ------ | -------------------------------------------------- |
| **Complex fill logic**      | High   | Start with conservative mode, add optimistic later |
| **Performance bottlenecks** | Medium | Profile early, optimize hot paths                  |
| **State management bugs**   | High   | Comprehensive state transition tests               |
| **Commission edge cases**   | Medium | Test with real broker commission structures        |
| **Partial fill complexity** | High   | Detailed test scenarios, clear documentation       |

## Dependencies

### Required (Phase 2)

- ✅ PortfolioService with apply_fill()
- ✅ Portfolio models (Position, LedgerEntry)
- ✅ Bar models (for price data)

### Optional (Future)

- Phase 4: RiskService (will consume ExecutionService)
- Phase 5: BacktestEngine (will orchestrate ExecutionService)

## Post-Phase 3 Enhancements (Phase 4+)

These are **NOT** in Phase 3 scope:

- [ ] Optimistic fill mode
- [ ] Order routing (multiple venues)
- [ ] Order amendments (modify price/qty)
- [ ] Advanced order types (trailing stop, iceberg)
- [ ] Market impact models (beyond simple volume limits)
- [ ] Live trading adapter (real broker integration)

## Phase 3 Completion Checklist

- [ ] All Week 1 tasks complete
- [ ] All Week 2 tasks complete
- [ ] All Week 3 tasks complete
- [ ] All Week 4 tasks complete
- [ ] 890+ tests passing
- [ ] 90% coverage maintained
- [ ] MyPy clean
- [ ] Ruff clean
- [ ] Documentation complete
- [ ] Commits created (5 conventional commits)
- [ ] Merged to feature/lego-architecture
- [ ] Branch deleted
- [ ] Ready for Phase 4 (RiskService)

______________________________________________________________________

**Document Status**: Initial Implementation Plan **Last Updated**: October 21, 2025 **Author**: QTrader Development Team **Next Review**: After Week 1 completion
