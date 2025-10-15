# Phase 3: Extract ExecutionService

## Overview

**Goal:** Isolate order execution logic (validation, fill simulation, commission) from orchestration and portfolio manipulation.

**Duration:** 3-4 weeks **Complexity:** High **Priority:** High - Complex business logic

## Target Architecture

### Service Interface

````python
class IExecutionService(Protocol):
    """
    Execution service interface.

    Responsibilities:
    - Validate orders
    - Simulate fills based on market data
    - Calculate commissions and slippage
    - Track order state (pending, filled, expired)
    - Apply fill policies

    Does NOT:
    - Modify portfolio directly (returns fills for application)
    - Load market data
    - Make trading decisions
    ```

## Key Methods

- `submit_order(order: OrderBase) -> None`
- `on_bar(symbol: str, bar: Bar, ts: datetime) -> List[Fill]`
- `get_pending_orders() -> List[OrderBase]`
- `get_filled_orders() -> List[OrderBase]`
- `cancel_order(order_id: str) -> None`

## Implementation Tasks

### Week 1-2: Core Execution Logic

- [ ] Define `IExecutionService` protocol
- [ ] Extract order validation
- [ ] Extract fill simulation
- [ ] Implement fill policies (aggressive, conservative, VWAP)

### Week 3: Commission & Slippage

- [ ] Commission calculator integration
- [ ] Slippage models
- [ ] Partial fill support
- [ ] Volume participation limits

### Week 4: Testing & Integration

- [ ] Unit tests (with mock portfolio, mock data)
- [ ] Integration tests with real portfolio service
- [ ] Performance benchmarks
- [ ] Documentation

## Validation Criteria

- [ ] ✅ Implements `IExecutionService`
- [ ] ✅ Zero direct portfolio manipulation
- [ ] ✅ Uses `IPortfolioService` interface only
- [ ] ✅ Can test without real portfolio
- [ ] ✅ Test coverage ≥ 90%

## Next Phase

👉 **[Phase 4: Extract RiskService](phase4_risk_service.md)**

---

**Phase Status:** 📝 Planning
**Dependencies:** Phase 2 (PortfolioService)
**Last Updated:** October 15, 2025
````
