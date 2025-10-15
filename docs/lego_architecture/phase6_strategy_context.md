# Phase 6: Strategy Context Refactoring

## Overview

**Goal:** Refactor Strategy Context to be a clean facade over services, providing a simple user-facing API while hiding complexity.

**Duration:** 2-3 weeks **Complexity:** Medium **Priority:** Medium - User experience

## Target Architecture

### Context as Facade

```python
class Context:
    """
    User-facing strategy context.

    Wraps services and provides convenience methods.
    Hides internal complexity from strategy authors.
    """

    def __init__(
        self,
        portfolio: IPortfolioService,
        risk: IRiskService,
        data: IDataService,
    ):
        """Inject services via interfaces."""
        self._portfolio = portfolio
        self._risk = risk
        self._data = data

    # Convenience methods
    def get_position(self, symbol: str) -> Optional[Position]:
        """Delegate to portfolio service."""
        return self._portfolio.get_position(symbol)

    def get_cash(self) -> Decimal:
        """Delegate to portfolio service."""
        return self._portfolio.get_cash()

    def submit_signal(self, signal: Signal) -> None:
        """Delegate to risk service for evaluation."""
        state = self._portfolio.get_state()
        order = self._risk.evaluate_signal(signal, state)
        if order:
            self._execution.submit_order(order)
```

## Strategy API

```python
class Strategy(ABC):
    """Base strategy class."""

    @abstractmethod
    def on_init(self, ctx: Context) -> None:
        """Initialize strategy."""
        pass

    @abstractmethod
    def on_bar(self, ctx: Context) -> List[Signal]:
        """Process bar, return signals."""
        pass

    @abstractmethod
    def on_end(self, ctx: Context) -> None:
        """Cleanup."""
        pass
```

## Implementation Tasks

### Week 1: Context Refactoring

- [ ] Define new Context API
- [ ] Implement service delegation
- [ ] Add convenience methods
- [ ] Remove direct service exposure

### Week 2: Strategy Migration

- [ ] Update Strategy base class
- [ ] Migrate example strategies
- [ ] Add migration guide
- [ ] **No backward compatibility needed** - update all strategies with new API

### Week 3: Testing & Documentation

- [ ] Unit tests for Context
- [ ] Integration tests with strategies
- [ ] User documentation
- [ ] Migration examples

## Validation Criteria

- [ ] ✅ Context delegates to services
- [ ] ✅ Clean user-facing API
- [ ] ✅ Example strategies working
- [ ] ✅ Documentation complete

## Completion

After Phase 6, the lego architecture is **complete**! 🎉

All services are independent, testable, and composable.

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** Phase 5 (BacktestEngine) **Last Updated:** October 15, 2025
