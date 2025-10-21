# ROAD MAP and TODOs

## Current Phase: Phase 2 - Portfolio Service

**Focus:** Position tracking, cash management, and corporate action processing

**Timeline:** 3-4 weeks

**Key Deliverables:**

- Position tracking with cost basis
- Cash management with transaction history
- Corporate action processing (splits, dividends)
- Fill processing and P&L calculation
- Mark-to-market valuation

______________________________________________________________________

## Completed Phases

### ✅ Phase 1: Data Service (Complete)

- Multi-mode bar support (unadjusted, adjusted, total_return)
- Algoseek OHLC adapter
- Schwab API adapter
- Data loader and iterator infrastructure
- 577+ unit tests

### ✅ Phase 1.5: Events & Corporate Actions (Complete)

**Week 1-2 Completed:**

- Event infrastructure (EventBus, 14 event types)
- Corporate action detection from Algoseek
- `DataService.get_corporate_actions()` API
- 42 event unit tests + 11 corporate action integration tests
- 607 total tests passing

**Week 3 Deferred:**

- Event publishing deferred to Phase 5 (BacktestEngine orchestration)

______________________________________________________________________

## Upcoming Phases

### 📝 Phase 2: Portfolio Service (Next - In Progress)

**Duration:** 3-4 weeks

**Objectives:**

- Position and cash management
- Corporate action processing
- Fill processing
- P&L tracking
- Mark-to-market

**Dependencies:** Phase 1 ✅, Phase 1.5 ✅

### Phase 3: Execution Service

**Duration:** 2-3 weeks

**Objectives:**

- Simulated fill generation
- Commission/slippage models
- Multiple fill policies
- Order validation

**Dependencies:** Phase 2

### Phase 4: Risk Management Service

**Duration:** 2-3 weeks

**Objectives:**

- Position limits
- Leverage limits
- Drawdown limits
- Signal validation and rejection

**Dependencies:** Phase 2

### Phase 5: Backtest Engine

**Duration:** 3-4 weeks

**Objectives:**

- Event-driven orchestration
- Bar-by-bar simulation
- Event bus integration
- Performance metrics

**Dependencies:** Phase 2, 3, 4

### Phase 6: Strategy Context

**Duration:** 2-3 weeks

**Objectives:**

- Strategy base class
- Indicator calculation
- Signal generation
- Portfolio access

**Dependencies:** Phase 5

______________________________________________________________________

## Technical Debt & Infrastructure

### High Priority

- [ ] Add automatic update for algoseek daily dataset
  - Update SecMaster
  - Daily OHLC update
  - Command: `qtrader data update --dataset algoseek-us-equity-1d`

### Medium Priority

- [ ] Add coverage reporting to CI/CD
- [ ] Add performance benchmarks
- [ ] Document event flow patterns

### Low Priority

- [ ] Add example backtests
- [ ] Add Jupyter notebook tutorials

______________________________________________________________________

## Quality Standards (Maintained Across All Phases)

- ✅ Type hints on all public APIs (MyPy clean)
- ✅ Docstrings on all public methods
- ✅ Test coverage ≥ 90%
- ✅ Ruff linting passes
- ✅ No regressions in existing tests

______________________________________________________________________

**Last Updated:** October 21, 2025
