# QTrader Lego Architecture Initiative

## Vision

Transform QTrader from a monolithic architecture with tight coupling between layers into a modular "lego-style" architecture where each layer is an independent service that can be tested, replaced, and composed with minimal dependencies.

## Core Principles

1. **Service Boundaries** - Each service has ONE clear responsibility
1. **Interface Contracts** - Services communicate via protocols/abstract interfaces
1. **Dependency Injection** - Services are injected at runtime, not imported directly
1. **Models as Contracts** - Shared Pydantic models define data shapes between services
1. **No Cross-Layer Knowledge** - Services don't import each other's implementations
1. **Independent Testing** - Each service can be tested in complete isolation

## Goals

### Primary Goals

- ✅ Test data layer without execution engine
- ✅ Swap data sources (Algoseek ↔ Schwab) with minimal code changes
- ✅ Replace execution logic without touching data layer
- ✅ Add new strategies without modifying core services
- ✅ Prepare for future live trading integration

### Success Metrics

- Each service has < 3 dependencies on other services
- 100% of services can be mocked for testing
- < 5 lines of code to swap implementations
- Test coverage > 90% per service
- Clear public interfaces documented for all services

## Architecture Comparison

### Current Architecture (Master Branch)

```
BacktestEngine (600+ lines)
  ├─ imports ExecutionEngine
  ├─ imports Portfolio (direct manipulation)
  ├─ imports RiskManager
  ├─ imports DataLoader
  └─ imports Strategy
      └─ Everything tightly coupled
```

**Problems:**

- Can't test components independently
- Hard to swap implementations
- Circular dependencies
- Business logic mixed with orchestration
- Future features (live trading) require massive refactoring

### Lego Architecture (Target)

```
BacktestEngine (< 200 lines, orchestration only)
  ├─ IDataService (interface)
  │   └─ DataService (implementation)
  ├─ IPortfolioService (interface)
  │   └─ PortfolioService (implementation)
  ├─ IExecutionService (interface)
  │   └─ ExecutionService (implementation)
  ├─ IRiskService (interface)
  │   └─ RiskService (implementation)
  └─ IStrategyService (interface)
      └─ StrategyService (implementation)

Services ONLY know about:
  - Their own interface
  - Shared models (Bar, Order, Portfolio, etc.)
  - Configuration
```

**Benefits:**

- Independent testing with mocks
- Easy to swap implementations
- Clear separation of concerns
- Business logic in services, orchestration in engine
- Future-proof for live trading, multiple brokers, etc.

## Implementation Phases

### Phase 1: Extract DataService ⭐ START HERE

**Duration:** 1-2 weeks **Foundation:** Use complete data layer from feature/schwab-integration branch **Goal:** Prove the concept with the cleanest, most self-contained layer

[📋 Phase 1 Implementation Plan](phase1_data_service.md)

### Phase 2: Extract PortfolioService

**Duration:** 2-3 weeks **Goal:** Separate position tracking from execution logic

[📋 Phase 2 Implementation Plan](phase2_portfolio_service.md)

### Phase 3: Extract ExecutionService

**Duration:** 3-4 weeks **Goal:** Isolate order execution and fill simulation

[📋 Phase 3 Implementation Plan](phase3_execution_service.md)

### Phase 4: Extract RiskService

**Duration:** 2-3 weeks **Goal:** Separate risk management and position sizing

[📋 Phase 4 Implementation Plan](phase4_risk_service.md)

### Phase 5: Rebuild BacktestEngine

**Duration:** 3-5 weeks **Goal:** Thin orchestration layer using all services

[📋 Phase 5 Implementation Plan](phase5_backtest_engine.md)

### Phase 6: Strategy Context Refactoring

**Duration:** 2-3 weeks **Goal:** Clean user-facing API wrapping services

[📋 Phase 6 Implementation Plan](phase6_strategy_context.md)

## Total Estimated Timeline

**15-20 weeks** (approximately 4-5 months)

**Can be parallelized:** Phases 2-4 have some independence once Phase 1 proves the pattern.

## Non-Goals (Out of Scope)

- ❌ Full microservices architecture (too heavy)
- ❌ Network-based service communication
- ❌ Event bus/message queue (start simple)
- ❌ Async/parallel processing (initially)
- ❌ Backward compatibility with old API (clean break is acceptable)

## Technology Choices

### Interface Definition

- **Python Protocols** (from `typing` module)
- Lightweight, no inheritance required
- Duck typing friendly
- Type checking support via MyPy

### Dependency Injection

- **Constructor injection** pattern
- Simple and explicit
- No framework needed (keep it lightweight)

### Testing

- **pytest** with fixtures
- Mock services via interfaces
- Integration tests with real services
- Property-based testing where applicable

### Models

- **Pydantic v2** (already in use)
- Immutable data classes
- Validation built-in
- Clear contracts

## Migration Strategy

### Strangler Fig Pattern (Recommended)

Gradually replace old system while keeping it functional:

1. Create new service alongside old code
1. Route new code paths to service
1. Gradually migrate old code
1. Remove old code when fully replaced
1. Repeat for next service

### Greenfield Approach (Alternative)

Given the willingness to rebuild:

1. Create feature/lego-architecture branch ✅ DONE
1. Build services from scratch (clean slate)
1. Comprehensive testing per service
1. Port strategies to new architecture
1. Switch when ready

**Chosen:** Greenfield approach on feature/lego-architecture branch

## Risk Management

### Technical Risks

| Risk                 | Mitigation                                        |
| -------------------- | ------------------------------------------------- |
| Over-engineering     | Start simple, add complexity only when needed     |
| Performance overhead | Profile early, optimize hot paths                 |
| Learning curve       | Document interfaces thoroughly, provide examples  |
| Scope creep          | Stick to defined phases, resist feature additions |

### Project Risks

| Risk                  | Mitigation                             |
| --------------------- | -------------------------------------- |
| Timeline slippage     | Break phases into weekly milestones    |
| Loss of functionality | Comprehensive integration tests        |
| Breaking changes      | Accept clean break, document migration |
| Testing burden        | TDD approach, write tests first        |

## Success Criteria

Before merging to master, ALL criteria must be met:

- [ ] All 6 phases complete
- [ ] Test coverage ≥ 90% for all services
- [ ] All integration tests passing
- [ ] Performance within 10% of original
- [ ] Documentation complete for all services
- [ ] Example strategies migrated and working
- [ ] MyPy type checking passing
- [ ] Code review approved

## Next Steps

1. ✅ Create feature/lego-architecture branch (DONE)
1. ✅ Review schwab-integration data layer (DONE)
1. 📖 Read Phase 1 implementation plan
1. 🚀 Begin Phase 1: Extract DataService

## References

- [Data Layer Architecture](../data_layer_architecture.md)
- [Algoseek Guide](../data_vendors/algoseek_guide.md)
- [Schwab Guide](../data_vendors/schwab_guide.md)
- Original evaluation document in conversation history

______________________________________________________________________

**Document Status:** Initial Draft **Last Updated:** October 15, 2025 **Author:** QTrader Development Team **Branch:** feature/lego-architecture
