# Phase 5: Rebuild BacktestEngine

## Overview

**Goal:** Transform BacktestEngine from a 600+ line monolith with mixed concerns into a clean < 200 line orchestration layer that coordinates services.

**Duration:** 3-5 weeks **Complexity:** High **Priority:** Critical - Ties everything together

## Target Architecture

### Engine Structure

```python
class BacktestEngine:
    """
    Orchestration-only backtest engine.

    Responsibilities:
    - Coordinate services via interfaces
    - Implement event loop
    - Manage warmup phase
    - Generate reports

    Does NOT:
    - Contain business logic
    - Directly manipulate data structures
    - Know about implementation details
    """

    def __init__(
        self,
        data: IDataService,
        portfolio: IPortfolioService,
        execution: IExecutionService,
        risk: IRiskService,
        strategy: IStrategyService,
    ):
        """All services injected via interfaces."""
        self.data = data
        self.portfolio = portfolio
        self.execution = execution
        self.risk = risk
        self.strategy = strategy
```

## Event Loop Flow

```
1. data.load_universe(symbols, start, end) → iterators
2. strategy.on_init(ctx)
3. Warmup phase (if enabled)
4. strategy.on_start(ctx)
5. Main loop:
   for multi_bar in merger:
       portfolio.update_prices({symbol: price})
       signals = strategy.on_bar(ctx)
       orders = [risk.evaluate_signal(s, portfolio.get_state()) for s in signals]
       for order in orders:
           execution.submit_order(order)
       fills = execution.on_bar(symbol, bar, ts)
       for fill in fills:
           portfolio.apply_fill(fill)
6. strategy.on_end(ctx)
7. Generate reports
```

## Implementation Tasks

### Week 1: Dependency Injection Framework

- [ ] Design service injection pattern
- [ ] Create service factory/builder
- [ ] Configuration → services mapping

### Week 2-3: Event Loop Refactoring

- [ ] Remove business logic from engine
- [ ] Delegate to services via interfaces
- [ ] Implement clean event loop
- [ ] Warmup phase coordination

### Week 4: Reporting & Output

- [ ] Portfolio snapshots
- [ ] Performance metrics
- [ ] Trade log
- [ ] CSV/JSON export

### Week 5: Testing & Integration

- [ ] Integration tests with all services
- [ ] End-to-end backtest scenarios
- [ ] Performance benchmarks
- [ ] Documentation

## Validation Criteria

- [ ] ✅ BacktestEngine < 200 lines
- [ ] ✅ All services injected via interfaces
- [ ] ✅ Zero business logic in engine
- [ ] ✅ All integration tests pass
- [ ] ✅ Performance within 10% of original

## Next Phase

👉 **[Phase 6: Strategy Context Refactoring](phase6_strategy_context.md)**

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** Phases 1-4 complete **Last Updated:** October 15, 2025
