# Phase 4: Extract RiskService

## Overview

**Goal:** Separate risk management, position sizing, and limit enforcement from execution and strategy layers.

**Duration:** 2-3 weeks **Complexity:** Medium **Priority:** High - Business logic isolation

## Target Architecture

### Service Interface

```python
class IRiskService(Protocol):
    """
    Risk service interface.

    Responsibilities:
    - Evaluate trading signals
    - Calculate position sizes
    - Enforce portfolio limits (concentration, leverage)
    - Check margin requirements
    - Generate orders from approved signals

    Does NOT:
    - Execute orders
    - Manage positions
    - Load market data
    - Make strategy decisions
    """
```

## Key Methods

- `evaluate_signal(signal: Signal, state: PortfolioState) -> Optional[OrderBase]`
- `check_concentration_limit(symbol: str, qty: int, state: PortfolioState) -> bool`
- `check_leverage_limit(order: OrderBase, state: PortfolioState) -> bool`
- `calculate_position_size(signal: Signal, state: PortfolioState) -> int`

## Implementation Tasks

### Week 1: Interface & Core Logic

- [ ] Define `IRiskService` protocol
- [ ] Extract signal evaluation logic
- [ ] Implement position sizing methods

### Week 2: Limits & Constraints

- [ ] Concentration limits
- [ ] Leverage limits
- [ ] Drawdown controls
- [ ] Cash requirements

### Week 3: Testing & Integration

- [ ] Unit tests with mock portfolio states
- [ ] Integration tests
- [ ] Documentation

## Validation Criteria

- [ ] ✅ Implements `IRiskService`
- [ ] ✅ Works with `PortfolioState` (immutable)
- [ ] ✅ No direct portfolio manipulation
- [ ] ✅ Test coverage ≥ 90%

## Next Phase

👉 **[Phase 5: Rebuild BacktestEngine](phase5_backtest_engine.md)**

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** Phase 2 (PortfolioService) **Last Updated:** October 15, 2025
