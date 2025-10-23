# Architecture Alignment Analysis

**Date:** October 23, 2025\
**Purpose:** Align evolved architectural vision with current implementation\
**Status:** 🔍 Understanding Phase - No coding yet

______________________________________________________________________

## 1. Current Implementation State (Phases 1-5 ✅)

### ✅ What We Have Built

**Package Structure:**

```
src/qtrader/
├── contracts/          # Immutable data models (Bar, Position, Order, etc.)
├── engine/            # BacktestEngine (299 LOC, pure orchestrator)
├── events/            # EventBus + Event definitions
├── libraries/         # 🆕 NEW STRUCTURE
│   ├── indicators/    # Empty base.py (needs ABC)
│   ├── strategies/    # Empty base.py (needs ABC)
│   ├── performance/   # Empty base.py (needs ABC - was "metrics")
│   └── risk_policies/ # Empty (needs ABC)
├── services/
│   ├── data/          # DataService ✅
│   ├── strategy/      # StrategyService ✅
│   ├── portfolio_manager/  # RiskService ✅ (YOUR: Portfolio Manager)
│   ├── ledger/        # PortfolioService ✅ (YOUR: Ledger Service)
│   ├── execution/     # ExecutionService ✅
│   └── reports/       # Empty placeholder
└── system/            # Config, Logging
```

### ✅ Implemented Services

| Your Diagram Name     | Current Code Name  | Implementation Status                                                      |
| --------------------- | ------------------ | -------------------------------------------------------------------------- |
| **Data Service**      | `DataService`      | ✅ Complete - streams bars, publishes PriceBarEvent                        |
| **Strategy Service**  | `StrategyService`  | ✅ Complete - loads strategies, routes bars, publishes SignalEvent         |
| **Portfolio Manager** | `RiskService`      | ✅ Complete - risk policies, position sizing, publishes OrderApprovedEvent |
| **Ledger Service**    | `PortfolioService` | ✅ Complete - position tracking, accounting, publishes PortfolioStateEvent |
| **Execution Service** | `ExecutionService` | ✅ Complete - simulates fills, publishes FillEvent                         |
| **Reporting Service** | (missing)          | ❌ Empty placeholder at `services/reports/`                                |

### ✅ Libraries Created (Empty Placeholders)

```
libraries/
├── indicators/         # 🆕 Created, empty base.py
├── strategies/         # 🆕 Created, empty base.py
├── performance/        # 🆕 Created, empty base.py (was "metrics" in diagram)
└── risk_policies/      # 🆕 Created, empty __init__.py
```

**Status:** Structure exists but ABC contracts not yet implemented.

______________________________________________________________________

## 2. Your Evolved Architectural Vision

### Key Insights from Diagrams

#### A. **Not Everything Goes Through Event Bus** ✅ CORRECT

- ✅ **Direct connections:** Config files, Libraries, Data sources, Outputs
- ✅ **Event Bus only for runtime:** Service-to-service communication during backtest

#### B. **Built-in vs Custom Libraries** 🎯 NEW CONCEPT

```
Built-in Libraries (qtrader.libraries.*)
├── indicators/buildin/     # 🆕 SMA, EMA, RSI, MACD
├── strategies/buildin/     # 🆕 Example strategies
├── performance/buildin/    # 🆕 Sharpe, Drawdown, Win Rate
└── risk_policies/buildin/  # 🆕 Fixed fraction, vol target

Custom Libraries (user-provided)
├── user_indicators.py      # Implements BaseIndicator ABC
├── user_strategies.py      # Implements BaseStrategy ABC
├── user_metrics.py         # Implements BaseMetric ABC
└── user_risk_policies.py   # Implements BaseRiskPolicy ABC
```

**Your Vision:** Users can extend QTrader without modifying codebase, following ABC interfaces.

#### C. **Service Naming Clarity** ✅ ALIGNED

- Portfolio Manager = RiskService ✅ (decides WHAT to trade)
- Ledger Service = PortfolioService ✅ (records WHAT happened)

#### D. **Missing: Reporting Service** ❌ GAP

Your diagram shows full Reporting Service:

- Consumes: SnapshotEvent from Ledger
- Uses: Metrics Library (direct import)
- Outputs: Console, JSON (direct write)

______________________________________________________________________

## 3. Gaps & Misalignments

### 🔴 Critical Gaps

#### Gap 1: ABC Contracts Not Implemented

**Problem:** Empty placeholder files, no abstract base classes defined.

**Your Vision Files:**

```python
# libraries/indicators/base.py - EMPTY ❌
# Should have:
class BaseIndicator(ABC):
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.Series:
        pass

    @abstractmethod
    def update(self, bar: Bar) -> float:
        pass

# libraries/strategies/base.py - EMPTY ❌
# Should have:
class BaseStrategy(ABC):
    @abstractmethod
    def on_bar(self, event: PriceBarEvent) -> None:
        pass

# libraries/performance/base.py - EMPTY ❌
# Should have:
class BaseMetric(ABC):
    @abstractmethod
    def compute(self, results: BacktestResult) -> float:
        pass
```

**Impact:** Users cannot extend QTrader yet. No built-in indicators/metrics exist.

#### Gap 2: Reporting Service Missing

**Problem:** `services/reports/` is empty placeholder.

**Your Vision:**

- ReportingService consumes SnapshotEvent
- Uses BaseMetric implementations (Sharpe, Drawdown, etc.)
- Outputs Console and JSON reports

**Current State:**

- BacktestEngine returns basic BacktestResult
- No metric calculations
- No formatted reports
- Examples use print() statements

#### Gap 3: Built-in Library Implementations Missing

**Problem:** No built-in indicators, strategies, metrics, or risk policies exist.

**Your Vision:**

```
libraries/
├── indicators/buildin/
│   ├── sma.py           # ❌ Missing
│   ├── ema.py           # ❌ Missing
│   ├── rsi.py           # ❌ Missing
│   └── bollinger.py     # ❌ Missing
├── performance/buildin/
│   ├── sharpe.py        # ❌ Missing
│   ├── drawdown.py      # ❌ Missing
│   └── win_rate.py      # ❌ Missing
└── risk_policies/buildin/
    ├── fixed_fraction.py  # ❌ Missing (exists in RiskService, not library)
    └── vol_target.py      # ❌ Missing
```

**Current State:**

- Indicators: Users implement in strategy code
- Metrics: Not implemented anywhere
- Risk policies: Hardcoded in RiskService (not pluggable)

### 🟡 Naming Inconsistencies

#### Issue 1: "performance" vs "metrics"

- **Your Diagram:** "Metrics Library"
- **Current Code:** `libraries/performance/`

**Question:** Should we rename to `libraries/metrics/` for clarity?

#### Issue 2: Service Folder vs Class Naming

- **Folder:** `services/portfolio_manager/`
- **Class:** `RiskService`
- **Your Diagram:** "Portfolio Manager"

**Current State:** Confusing! Folder name doesn't match class name.

**Options:**

1. Rename folder: `services/portfolio_manager/` → `services/risk/` (match class)
1. Rename class: `RiskService` → `PortfolioManagerService` (match folder)
1. Add alias/comment explaining Portfolio Manager = RiskService

#### Issue 3: PortfolioService vs LedgerService

- **Folder:** `services/ledger/`
- **Class:** `PortfolioService`
- **Your Diagram:** "Ledger Service"

**Same issue as above!**

______________________________________________________________________

## 4. Lego Architecture Documentation Outdated

### Phase Status Corrections Needed

**README.md says:**

```markdown
| Phase 1: DataService      | 📝 Planning | 1-2 weeks | ⭐ Critical |
| Phase 2: PortfolioService | 📝 Planning | 2-3 weeks | High        |
| Phase 3: ExecutionService | 📝 Planning | 3-4 weeks | High        |
| Phase 4: RiskService      | 📝 Planned  | 2-3 weeks | Medium      |
| Phase 5: BacktestEngine   | 📝 Planned  | 3-5 weeks | High        |
```

**Reality:**

```markdown
| Phase 1: DataService      | ✅ COMPLETE | 2 days    | ⭐ Critical |
| Phase 2: PortfolioService | ✅ COMPLETE | 3 days    | High        |
| Phase 3: ExecutionService | ✅ COMPLETE | 2 days    | High        |
| Phase 4: RiskService      | ✅ COMPLETE | 2 days    | Medium      |
| Phase 5: BacktestEngine   | ✅ COMPLETE | 2 days    | High        |
| Phase 6: Strategy Context | ❓ UNCLEAR  | TBD       | Medium      |
| Phase 7: IndicatorService | 🔄 RETHINK  | TBD       | High        |
| Phase 8: AnalyticsService | 🔄 RETHINK  | TBD       | High        |
| Phase 9: ReportingService | 🆕 NEXT     | TBD       | High        |
| Phase 10: Configuration   | ✅ PARTIAL  | TBD       | Medium      |
```

### Phases 6-10: Need Revision Based on New Vision

#### Phase 6: Strategy Context

**Original Plan:** User-facing API wrapper for strategies.

**Your Vision Impact:**

- Strategies should implement `BaseStrategy` ABC
- Load from external files (already done by StrategyService)
- Access indicators via library imports (not yet built)

**Question:** Do we still need Strategy Context or is ABC + StrategyService enough?

#### Phase 7: IndicatorService

**Original Plan:** Centralized service for technical indicators.

**Your Vision Impact:**

- Indicators are a **library**, not a service!
- Built-in indicators: `qtrader.libraries.indicators.buildin.*`
- Custom indicators: User implements `BaseIndicator`
- Strategies import indicators directly (not via service)

**Conclusion:** Phase 7 should be "Indicator Library Implementation", not "IndicatorService".

#### Phase 8: AnalyticsService → Performance Metrics Library

**Original Plan:** Service for performance metrics.

**Your Vision Impact:**

- Metrics/Performance is a **library**, not a service!
- ReportingService **uses** metrics library
- Built-in metrics: `qtrader.libraries.performance.buildin.*`
- Custom metrics: User implements `BaseMetric`

**Conclusion:** Phase 8 should be "Performance Library Implementation", merged into Phase 9.

#### Phase 9: ReportingService

**Original Plan:** Format and display results.

**Your Vision:** CRITICAL! This is the missing piece.

**Requirements:**

- Service that consumes SnapshotEvent
- Imports metrics library (built-in + custom)
- Calculates performance metrics (Sharpe, drawdown, etc.)
- Outputs console reports (formatted tables)
- Outputs JSON reports (for further analysis)

**Status:** Should be **NEXT PRIORITY** after library ABCs.

#### Phase 10: Configuration

**Original Plan:** Centralized config management.

**Current State:** Already partially implemented:

- SystemConfig (system.yaml)
- BacktestConfig (backtest.yaml)
- Service-specific configs (DataConfig, RiskConfig, etc.)

**Remaining:**

- Config validation improvements
- Better error messages
- Config inheritance/overrides

______________________________________________________________________

## 5. Recommended Path Forward

### 🎯 Phase 5.5: Library ABCs & Built-ins (NEW)

**Duration:** 1-2 weeks\
**Priority:** HIGH - Enables extensibility

**Tasks:**

1. **Define ABC Contracts** (Week 1, Days 1-2)

   ```python
   # libraries/indicators/base.py
   - BaseIndicator ABC (calculate, update methods)

   # libraries/strategies/base.py
   - BaseStrategy ABC (on_bar, on_signal methods)

   # libraries/performance/base.py
   - BaseMetric ABC (compute method)

   # libraries/risk_policies/base.py
   - BaseRiskPolicy ABC (evaluate, size methods)
   ```

1. **Implement Built-in Indicators** (Week 1, Days 3-5)

   ```python
   libraries/indicators/buildin/
   - sma.py, ema.py, rsi.py, macd.py, bollinger.py
   - All implement BaseIndicator
   - Unit tests for each
   ```

1. **Implement Built-in Metrics** (Week 2, Days 1-3)

   ```python
   libraries/performance/buildin/
   - sharpe.py, drawdown.py, sortino.py, calmar.py
   - win_rate.py, profit_factor.py
   - All implement BaseMetric
   - Unit tests for each
   ```

1. **Refactor Risk Policies** (Week 2, Days 4-5)

   ```python
   libraries/risk_policies/buildin/
   - Extract FixedFractionSizer from RiskService
   - Create fixed_fraction.py implementing BaseRiskPolicy
   - Create vol_target.py, kelly.py
   - RiskService becomes pluggable (loads policies from library)
   ```

1. **Documentation & Examples**

   ```python
   examples/custom_libraries/
   - my_indicator.py (implements BaseIndicator)
   - my_strategy.py (implements BaseStrategy)
   - my_metric.py (implements BaseMetric)
   - README.md explaining how to extend
   ```

### 🎯 Phase 6: ReportingService (REVISED)

**Duration:** 1 week\
**Priority:** HIGH - User-facing feature

**Tasks:**

1. **Create ReportingService** (Days 1-3)

   - Subscribe to SnapshotEvent from Ledger
   - Import metrics from `libraries.performance`
   - Calculate all metrics (Sharpe, drawdown, etc.)
   - Format console output (rich tables)
   - Generate JSON reports

1. **Console Reporter** (Days 4-5)

   - Real-time progress bar (already exists in CLI)
   - Summary table after backtest
   - Trade-by-trade output (optional flag)

1. **JSON Reporter** (Days 4-5)

   - Full backtest results to JSON file
   - Trade history
   - Equity curve data points
   - All calculated metrics

### 🎯 Phase 7-10: Optional/Deferred

- Phase 7 (IndicatorService): ❌ SKIP - Indicators are library, not service
- Phase 8 (AnalyticsService): ❌ SKIP - Merged into Phase 6
- Phase 9: ✅ DONE as Phase 6 Revised
- Phase 10 (Configuration): ✅ PARTIAL - Polish later

______________________________________________________________________

## 6. Key Questions for Alignment

### Naming & Structure

**Q1:** Should we rename folders to match class names?

```
Option A: Keep current (confusing)
services/portfolio_manager/service.py:RiskService
services/ledger/service.py:PortfolioService

Option B: Rename folders to match classes
services/risk/service.py:RiskService
services/portfolio/service.py:PortfolioService

Option C: Rename classes to match folders (your vision)
services/portfolio_manager/service.py:PortfolioManagerService
services/ledger/service.py:LedgerService

Option D: Add comments/aliases
# services/portfolio_manager/service.py
class RiskService:  # aka Portfolio Manager
```

**Q2:** Performance vs Metrics naming?

```
Option A: Keep current
libraries/performance/

Option B: Rename to match diagram
libraries/metrics/
```

### Architecture Decisions

**Q3:** Indicator Library vs IndicatorService?

```
Your Vision: Library (direct import)
Original Plan: Service (via EventBus)

Which approach for:
- Technical indicators?
- Custom user indicators?
```

**Q4:** Risk Policy Pluggability?

```
Current: FixedFractionSizer hardcoded in RiskService
Your Vision: BaseRiskPolicy ABC, pluggable policies

Should RiskService:
A) Keep hardcoded sizers (simple, works now)
B) Load policies from library (extensible, your vision)
C) Hybrid (built-in + optional custom)
```

**Q5:** Strategy Loading Mechanism?

```
Current: StrategyService loads .py files dynamically
Your Vision: BaseStrategy ABC

Should strategies:
A) Be Python files with on_bar() function (current)
B) Be classes implementing BaseStrategy (your vision)
C) Support both?
```

### Implementation Priority

**Q6:** What to build first?

```
Option A: Library ABCs → Built-ins → ReportingService
Option B: ReportingService with basic metrics → Then library ABCs
Option C: Rename/restructure → Then ABCs → Then ReportingService
```

**Q7:** How important is extensibility NOW?

```
If HIGH: Build full ABC system with built-ins (Phase 5.5)
If MEDIUM: Basic built-in indicators/metrics, defer ABCs
If LOW: Focus on ReportingService with hardcoded metrics
```

______________________________________________________________________

## 7. Documentation Updates Needed

### Update Lego Architecture Docs

1. **README.md**

   - Mark Phases 1-5 as COMPLETE ✅
   - Update durations (days not weeks)
   - Revise Phases 6-10 based on new vision

1. **Phase Documents**

   - Add "COMPLETE" badges to phase1-5 docs
   - Revise phase6: Strategy Context (or skip)
   - Revise phase7: Indicator Library (not service)
   - Revise phase8: Merge into phase9
   - Revise phase9: ReportingService (priority)
   - Revise phase10: Config polish

1. **New Documents**

   - `ARCHITECTURE_VISION.md` - Your evolved vision with diagrams
   - `LIBRARY_EXTENSION_GUIDE.md` - How users extend with ABCs
   - `SERVICE_NAMING.md` - Explain Portfolio Manager = RiskService

### Update Diagrams

1. **architecture.md**

   - Add library layer with ABC contracts
   - Show built-in vs custom distinction
   - Clarify ReportingService as next priority

1. **high_level_architecture.md**

   - Update service names to match code
   - Add "Missing: ReportingService" callout
   - Show library ABC structure

______________________________________________________________________

## 8. Next Steps (Your Decision)

### Option A: Discussion First (Recommended) ✅

**Current Phase:** Understanding alignment

**Actions:**

1. ✅ Review this document
1. ❓ Answer key questions (Q1-Q7)
1. ❓ Clarify priorities
1. ❓ Agree on naming/structure
1. → Then proceed to implementation

### Option B: Start Building

**Jump to:** Phase 5.5 (Library ABCs)

**Risk:** May need rework if vision not aligned

### Option C: Documentation First

**Focus:** Update all lego architecture docs

**Benefit:** Clear roadmap before coding

______________________________________________________________________

## 9. Summary

### ✅ What's Working Well

1. Event-driven architecture is solid
1. Services are properly separated
1. Phases 1-5 complete and working
1. Direct vs EventBus separation is correct
1. Your architectural vision is clear and mature

### 🔴 What Needs Attention

1. Library ABC contracts not implemented (critical)
1. ReportingService missing (user-facing gap)
1. Built-in indicators/metrics don't exist
1. Service naming confusion (folders vs classes)
1. Lego architecture docs outdated

### 🎯 Recommended Next Actions

1. **Align on naming** (Q1, Q2)
1. **Clarify library approach** (Q3, Q4, Q5)
1. **Agree on priority** (Q6, Q7)
1. **Update documentation** (lego architecture phases)
1. **Implement Phase 5.5** (Library ABCs + Built-ins)
1. **Implement Phase 6 Revised** (ReportingService)

______________________________________________________________________

**Ready for discussion! What's your take on the questions and recommended path?** 🚀
