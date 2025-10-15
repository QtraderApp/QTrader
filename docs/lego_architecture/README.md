# Lego Architecture - Implementation Plans

This directory contains the implementation plans for refactoring QTrader into a service-based "lego" architecture where each layer is an independent, testable service with minimal cross-layer dependencies.

## 📋 Quick Navigation

- **[Overview](overview.md)** - Architecture vision, principles, and roadmap
- **[Phase 1: DataService](phase1_data_service.md)** - Extract data layer as standalone service ⭐ START HERE
- **[Phase 2: PortfolioService](phase2_portfolio_service.md)** - Separate position tracking and cash management
- **[Phase 3: ExecutionService](phase3_execution_service.md)** - Isolate order execution logic
- **[Phase 4: RiskService](phase4_risk_service.md)** - Extract risk management and position sizing
- **[Phase 5: BacktestEngine](phase5_backtest_engine.md)** - Rebuild as thin orchestration layer
- **[Phase 6: Strategy Context](phase6_strategy_context.md)** - Create clean user-facing API
- **[Phase 7: IndicatorService](phase7_indicator_service.md)** - Extract technical indicators as independent service
- **[Phase 8: AnalyticsService](phase8_analytics_service.md)** - Performance metrics calculation
- **[Phase 9: ReportingService](phase9_reporting_service.md)** - Format and display results
- **[Phase 10: Configuration](phase10_configuration.md)** - Centralized configuration management

## 🎯 Goals

Transform QTrader from a monolithic architecture with tight coupling to a modular "lego" architecture where:

- ✅ Each service has ONE clear responsibility
- ✅ Services communicate via abstract interfaces (Protocols)
- ✅ Services can be tested independently
- ✅ Services can be swapped/replaced easily
- ✅ Dependencies flow in one direction (no circular deps)
- ✅ Business logic separated from orchestration

## 🏗️ Architecture Principles

### 1. Service Boundaries

Each service is a "lego piece" that:

- Has a clear, focused responsibility
- Exposes a Protocol interface
- Depends only on models and config
- Does NOT import other services
- Can be tested in isolation

### 2. Communication via Interfaces

```python
# NOT this (tight coupling)
from qtrader.execution.engine import ExecutionEngine
engine = ExecutionEngine(...)

# THIS (loose coupling)
from qtrader.services.execution import IExecutionService
engine: IExecutionService = create_execution_service(...)
```

### 3. Dependency Injection

Services are injected at runtime:

```python
backtest = BacktestEngine(
    data=DataService(config),
    portfolio=PortfolioService(config),
    execution=ExecutionService(config),
    risk=RiskService(config),
)
```

### 4. Models as Contracts

Pydantic models define the "language" services speak:

- Services depend on models
- Models DON'T depend on services
- Models are pure data (no business logic)

## 📊 Current Status

| Phase                     | Status      | Est. Duration | Priority    |
| ------------------------- | ----------- | ------------- | ----------- |
| Phase 1: DataService      | 📝 Planning | 1-2 weeks     | ⭐ Critical |
| Phase 2: PortfolioService | 📝 Planning | 2-3 weeks     | High        |
| Phase 3: ExecutionService | 📝 Planning | 3-4 weeks     | High        |
| Phase 4: RiskService      | 2-3 weeks   | Medium        | 📝 Planned  |
| Phase 5: BacktestEngine   | 3-5 weeks   | High          | 📝 Planned  |
| Phase 6: Strategy Context | 2-3 weeks   | Medium        | 📝 Planned  |
| Phase 7: IndicatorService | 2-3 weeks   | High          | 📝 Planned  |
| Phase 8: AnalyticsService | 2 weeks     | High          | 📝 Planned  |
| Phase 9: ReportingService | 1-2 weeks   | High          | 📝 Planned  |
| Phase 10: Configuration   | 1-2 weeks   | Medium        | 📝 Planned  |

**Total Estimated Duration:** 20-30 weeks (5-7.5 months)

## 🚀 Getting Started

### For Implementers

1. **Start with [Overview](overview.md)** to understand the architecture vision
1. **Read [Phase 1: DataService](phase1_data_service.md)** in detail
1. **Create feature branch:** `feature/lego-phase1-data-service`
1. **Follow the implementation tasks** in the phase document
1. **Submit PR** when validation criteria are met

### For Reviewers

1. Check that service implements the defined Protocol
1. Verify zero dependencies on other services (except models)
1. Ensure test coverage ≥ 90%
1. Confirm all validation criteria met
1. Review migration impact on existing code

### For Users

Once complete, the new architecture will:

- Make testing strategies easier (mock services)
- Enable live trading (swap execution service)
- Support multiple data sources (swap data service)
- Improve performance (optimize services independently)
- Simplify maintenance (change one service at a time)

## 📝 Document Structure

Each phase document contains:

- **Overview** - Goal, duration, complexity, priority
- **Current State** - What exists today, dependencies
- **Target Architecture** - Interface definition, implementation example
- **Implementation Tasks** - Week-by-week breakdown with checklists
- **Validation Criteria** - Functional, technical, performance requirements
- **Testing Strategy** - Unit, integration, mock usage examples
- **Migration Path** - How to transition from old to new
- **Success Metrics** - Measurable outcomes
- **Risks & Mitigations** - Potential issues and solutions
- **Dependencies** - What this phase needs and what it enables

## 🔗 Related Documentation

- **[Architecture Overview](../architecture.md)** - Current master branch architecture
- **[Data Layer Architecture](../data_layer_architecture.md)** - Detailed data layer design
- **[Schwab Integration](../data_vendors/schwab_guide.md)** - Schwab data source guide
- **[Algoseek Integration](../data_vendors/algoseek_guide.md)** - Algoseek data source guide

## 💡 Key Decisions

### No Backward Compatibility Required

We've decided to **NOT maintain backward compatibility** during refactoring:

- Clean break is acceptable
- Update all dependent code when services change
- Simpler implementation, clearer architecture
- Faster development

### Start from feature/schwab-integration Branch

The lego-architecture branch is based on `feature/schwab-integration` because:

- It has the most complete data layer implementation
- Supports multiple data sources (Algoseek, Schwab)
- Already has clean adapter pattern
- Well-tested and documented

### Phased Approach

We're using a phased approach rather than big-bang rewrite:

- Prove concept with Phase 1 (DataService)
- Validate approach before continuing
- Each phase delivers value independently
- Can pause/adjust between phases

## 🤝 Contributing

When implementing a phase:

1. Create feature branch from `feature/lego-architecture`
1. Follow the implementation tasks in phase document
1. Write tests FIRST (TDD approach)
1. Document as you go
1. Meet all validation criteria
1. Get code review
1. Merge back to `feature/lego-architecture`

## 📞 Questions?

If you have questions about:

- **Architecture decisions** - See [overview.md](overview.md)
- **Specific phase** - See that phase's document
- **Implementation details** - Check existing similar code in `feature/schwab-integration`
- **Testing approach** - See phase document's Testing Strategy section

______________________________________________________________________

**Branch:** `feature/lego-architecture` **Based On:** `feature/schwab-integration` **Target Merge:** `master` (after all phases complete) **Last Updated:** October 15, 2025
