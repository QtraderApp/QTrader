# QTrader Lego Architecture - Final Folder Structure

This document shows the complete folder structure after all 10 phases are implemented. It explains what's shared, what's service-specific, and how everything fits together.

## Complete Folder Structure (After Phase 10)

```
src/qtrader/
├── __init__.py
├── cli.py                          # CLI entry point (uses services)
├── py.typed                        # Type checking marker
│
├── models/                         # SHARED: Data contracts used by ALL services
│   ├── __init__.py
│   ├── bar.py                      # Bar, PriceSeries (canonical OHLCV)
│   ├── multi_bar.py                # MultiBar (3 adjustment modes)
│   ├── instrument.py               # Instrument, DataSource, InstrumentType
│   ├── order.py                    # Order, OrderType, OrderStatus
│   ├── position.py                 # Position (holdings)
│   ├── trade.py                    # Trade (filled orders)
│   ├── ledger.py                   # Ledger (cash + positions)
│   ├── portfolio.py                # Portfolio (aggregate view)
│   ├── performance.py              # PerformanceMetrics, PerformanceReport
│   └── vendors/                    # Vendor-specific models
│       ├── algoseek.py             # AlgoseekBar, AlgoseekPriceSeries
│       └── schwab.py               # SchwabBar (when implemented)
│
├── adapters/                       # SHARED: Data source adapters (used by DataService)
│   ├── __init__.py
│   ├── algoseek.py                 # AlgoseekOHLCVendorAdapter
│   ├── schwab.py                   # SchwabOHLCVendorAdapter (future)
│   └── resolver.py                 # DataSourceResolver (maps logical → physical)
│
├── data/                           # SHARED: Data layer infrastructure (used by DataService)
│   ├── __init__.py
│   ├── loader.py                   # DataLoader (coordinates adapter + transform)
│   ├── iterator.py                 # PriceSeriesIterator (streaming)
│   └── bar_merger.py               # BarMerger (multi-symbol alignment)
│
├── config/                         # SHARED: Configuration models (used by ALL services)
│   ├── __init__.py
│   ├── base_config.py              # BaseConfig (Phase 10)
│   ├── data_config.py              # DataConfig (validation, schema)
│   ├── portfolio_config.py         # PortfolioConfig (Phase 2)
│   ├── execution_config.py         # ExecutionConfig (commission, slippage)
│   ├── risk_config.py              # RiskConfig (limits, constraints)
│   ├── backtest_config.py          # BacktestConfig (orchestration)
│   ├── strategy_config.py          # StrategyConfig (user settings)
│   ├── indicator_config.py         # IndicatorConfig (Phase 7)
│   ├── analytics_config.py         # AnalyticsConfig (Phase 8)
│   ├── reporting_config.py         # ReportingConfig (Phase 9)
│   ├── logging_config.py           # LoggingConfig
│   └── system_config.py            # SystemConfig (top-level)
│
├── services/                       # LEGO SERVICES: Independent, composable, testable
│   ├── __init__.py
│   │
│   ├── data/                       # Phase 1: DataService ✅ COMPLETE
│   │   ├── __init__.py
│   │   ├── interface.py            # IDataService, IDataAdapter protocols
│   │   └── service.py              # DataService implementation
│   │
│   ├── portfolio/                  # Phase 2: PortfolioService (NEXT)
│   │   ├── __init__.py
│   │   ├── interface.py            # IPortfolioService protocol
│   │   ├── service.py              # PortfolioService implementation
│   │   └── tracker.py              # PositionTracker helper
│   │
│   ├── execution/                  # Phase 3: ExecutionService
│   │   ├── __init__.py
│   │   ├── interface.py            # IExecutionService, IFillSimulator protocols
│   │   ├── service.py              # ExecutionService implementation
│   │   ├── fill_simulator.py      # FillSimulator (slippage, commission)
│   │   └── order_manager.py       # OrderManager (queue, status tracking)
│   │
│   ├── risk/                       # Phase 4: RiskService
│   │   ├── __init__.py
│   │   ├── interface.py            # IRiskService protocol
│   │   ├── service.py              # RiskService implementation
│   │   ├── checks.py               # Risk checks (position size, concentration, etc.)
│   │   └── validators.py           # Order validators
│   │
│   ├── backtest/                   # Phase 5: BacktestEngine
│   │   ├── __init__.py
│   │   ├── interface.py            # IBacktestEngine protocol
│   │   ├── engine.py               # BacktestEngine (coordinates all services)
│   │   └── runner.py               # BacktestRunner (CLI/programmatic interface)
│   │
│   ├── strategy/                   # Phase 6: StrategyContext
│   │   ├── __init__.py
│   │   ├── interface.py            # IStrategy, IStrategyContext protocols
│   │   ├── context.py              # StrategyContext (user-facing API)
│   │   └── base_strategy.py       # BaseStrategy (abstract base for users)
│   │
│   ├── indicators/                 # Phase 7: IndicatorService
│   │   ├── __init__.py
│   │   ├── interface.py            # IIndicatorService, IIndicator protocols
│   │   ├── service.py              # IndicatorService (caching, computation)
│   │   ├── library/                # Built-in indicators
│   │   │   ├── __init__.py
│   │   │   ├── sma.py              # SimpleMovingAverage
│   │   │   ├── ema.py              # ExponentialMovingAverage
│   │   │   ├── rsi.py              # RelativeStrengthIndex
│   │   │   ├── macd.py             # MACD
│   │   │   ├── bollinger.py        # BollingerBands
│   │   │   └── atr.py              # AverageTrueRange
│   │   └── builder.py              # IndicatorBuilder (factory)
│   │
│   ├── analytics/                  # Phase 8: AnalyticsService
│   │   ├── __init__.py
│   │   ├── interface.py            # IAnalyticsService protocol
│   │   ├── service.py              # AnalyticsService implementation
│   │   ├── metrics/                # Metric calculators
│   │   │   ├── __init__.py
│   │   │   ├── returns.py          # Total return, CAGR, annualized
│   │   │   ├── risk.py             # Volatility, Sharpe, Sortino, Calmar
│   │   │   ├── drawdown.py         # Max drawdown, drawdown duration
│   │   │   ├── trading.py          # Win rate, profit factor, avg win/loss
│   │   │   └── kelly.py            # Kelly criterion
│   │   └── calculator.py           # MetricsCalculator (orchestrates metrics)
│   │
│   └── reporting/                  # Phase 9: ReportingService
│       ├── __init__.py
│       ├── interface.py            # IReportingService protocol
│       ├── service.py              # ReportingService implementation
│       ├── formatters/             # Output formatters
│       │   ├── __init__.py
│       │   ├── console.py          # Console output (tables, colors)
│       │   ├── json.py             # JSON export
│       │   ├── csv.py              # CSV export (trades, positions)
│       │   └── html.py             # HTML report generation
│       └── visualizations/         # Plot generators
│           ├── __init__.py
│           ├── equity_curve.py     # Equity curve plot
│           ├── drawdown.py         # Drawdown chart
│           ├── returns.py          # Returns distribution
│           └── positions.py        # Position exposure over time
│
├── api/                            # SHARED: External API integrations (future)
│   ├── __init__.py
│   ├── schwab_api.py               # Schwab API client (future)
│   └── polygon_api.py              # Polygon.io API client (future)
│
├── auth/                           # SHARED: Authentication (future live trading)
│   ├── __init__.py
│   └── schwab_auth.py              # OAuth for Schwab (future)
│
└── playground/                     # Temporary/experimental code
    └── (various experiments)
```

## Key Architectural Principles

### 1. Models Are Shared Contracts

**Location:** `src/qtrader/models/`

**Purpose:** Define data structures that ALL services understand

**Who Uses Them:**

- ✅ **DataService** - Creates Bar, PriceSeries, MultiBar
- ✅ **PortfolioService** - Creates Position, Ledger, Portfolio
- ✅ **ExecutionService** - Creates Order, Trade
- ✅ **RiskService** - Reads Position, Portfolio, Order
- ✅ **BacktestEngine** - Coordinates all models
- ✅ **StrategyContext** - Exposes models to user strategies
- ✅ **IndicatorService** - Reads Bar data
- ✅ **AnalyticsService** - Creates PerformanceMetrics
- ✅ **ReportingService** - Reads all models for display

**Key Point:** Models are **immutable contracts**. Services communicate by passing these models, not by sharing internal state.

### 2. Config Is Shared Infrastructure

**Location:** `src/qtrader/config/`

**Purpose:** Type-safe configuration using Pydantic

**Structure:**

```python
# Top-level config
SystemConfig
  ├── DataConfig          # For DataService
  ├── PortfolioConfig     # For PortfolioService
  ├── ExecutionConfig     # For ExecutionService
  ├── RiskConfig          # For RiskService
  ├── BacktestConfig      # For BacktestEngine
  ├── StrategyConfig      # For StrategyContext
  ├── IndicatorConfig     # For IndicatorService
  ├── AnalyticsConfig     # For AnalyticsService
  ├── ReportingConfig     # For ReportingService
  └── LoggingConfig       # System-wide logging
```

**NOT a service** - Configuration is infrastructure, not business logic

### 3. Adapters Are Shared by DataService

**Location:** `src/qtrader/adapters/`

**Purpose:** Vendor-specific data access

**Who Uses Them:**

- ✅ **DataService ONLY** - Wraps adapters with clean interface
- ❌ **Other services** - Never touch adapters directly

**Key Point:** Adapters are implementation details of DataService. Other services use `IDataService`, not adapters.

### 4. Data Layer Is Shared by DataService

**Location:** `src/qtrader/data/`

**Purpose:** Data loading, transformation, streaming

**Who Uses Them:**

- ✅ **DataService ONLY** - Uses DataLoader, PriceSeriesIterator
- ❌ **Other services** - Never touch data layer directly

**Key Point:** Data layer is implementation detail. Other services get iterators from `IDataService`.

### 5. API/Auth Are Shared (Future)

**Location:** `src/qtrader/api/`, `src/qtrader/auth/`

**Purpose:** External API integrations for live trading

**Current Status:** Mostly empty, for future use

**Who Will Use Them:**

- ✅ **DataService** - May use Schwab API for live data
- ✅ **ExecutionService** - May use broker API for live orders
- ❌ **Other services** - Isolated from external APIs

### 6. Services Are Independent Legos

**Location:** `src/qtrader/services/`

**Purpose:** Business logic, composable, testable

**Key Principles:**

1. **Each service has ONE responsibility**

   - DataService: Data loading
   - PortfolioService: Position tracking
   - ExecutionService: Order execution
   - RiskService: Risk validation
   - etc.

1. **Services communicate via Protocol interfaces**

   - Enables dependency injection
   - Enables mocking for tests
   - Enables swapping implementations

1. **Services are independent**

   - Can be tested in isolation
   - Can be reused in different contexts
   - Can be deployed separately (future microservices)

1. **Services use models as contracts**

   - Pass immutable data structures
   - No shared mutable state
   - Clear data flow

## Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                         Models                               │
│  (Bar, Order, Position, Trade, Portfolio, Instrument, etc.)  │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │ (all services depend on models)
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                                                               │
│  ┌─────────────┐    ┌──────────────┐    ┌────────────┐     │
│  │ DataService │    │   Portfolio  │    │ Execution  │     │
│  │   Phase 1   │    │   Service    │    │  Service   │     │
│  │     ✅      │    │   Phase 2    │    │  Phase 3   │     │
│  └─────────────┘    └──────────────┘    └────────────┘     │
│         │                   │                    │           │
│         └───────────────────┴────────────────────┘           │
│                             │                                │
│                             ▼                                │
│                    ┌─────────────────┐                       │
│                    │   RiskService   │                       │
│                    │     Phase 4     │                       │
│                    └─────────────────┘                       │
│                             │                                │
│                             ▼                                │
│                    ┌─────────────────┐                       │
│                    │ BacktestEngine  │                       │
│                    │     Phase 5     │                       │
│                    └─────────────────┘                       │
│                             │                                │
│                             ▼                                │
│                    ┌─────────────────┐                       │
│                    │ StrategyContext │                       │
│                    │     Phase 6     │                       │
│                    │  (User-facing)  │                       │
│                    └─────────────────┘                       │
│                                                               │
│  Supporting Services (can be used independently):            │
│                                                               │
│  ┌────────────┐  ┌──────────────┐  ┌───────────────┐       │
│  │ Indicator  │  │  Analytics   │  │   Reporting   │       │
│  │  Service   │  │   Service    │  │    Service    │       │
│  │  Phase 7   │  │   Phase 8    │  │    Phase 9    │       │
│  └────────────┘  └──────────────┘  └───────────────┘       │
│        │                 │                   │               │
│        └─────────────────┴───────────────────┘               │
│                          │                                   │
│                          ▼                                   │
│                    (Used by any service/strategy)            │
│                                                               │
└───────────────────────────────────────────────────────────────┘

Infrastructure (used by all):
- Config (Phase 10)
- Logging
- Error handling
```

## What's Shared vs Service-Specific

### Always Shared (Used by Multiple Services)

| Component      | Location    | Used By          | Purpose              |
| -------------- | ----------- | ---------------- | -------------------- |
| **Models**     | `models/`   | ALL services     | Data contracts       |
| **Config**     | `config/`   | ALL services     | Configuration        |
| **Adapters**   | `adapters/` | DataService only | Data source adapters |
| **Data Layer** | `data/`     | DataService only | Loading, streaming   |
| **API**        | `api/`      | Data/Execution   | External APIs        |
| **Auth**       | `auth/`     | API clients      | Authentication       |

### Service-Specific (Never Shared)

| Component            | Location               | Responsibility    | Exports             |
| -------------------- | ---------------------- | ----------------- | ------------------- |
| **DataService**      | `services/data/`       | Data loading      | `IDataService`      |
| **PortfolioService** | `services/portfolio/`  | Position tracking | `IPortfolioService` |
| **ExecutionService** | `services/execution/`  | Order execution   | `IExecutionService` |
| **RiskService**      | `services/risk/`       | Risk validation   | `IRiskService`      |
| **BacktestEngine**   | `services/backtest/`   | Orchestration     | `IBacktestEngine`   |
| **StrategyContext**  | `services/strategy/`   | User API          | `IStrategyContext`  |
| **IndicatorService** | `services/indicators/` | Indicators        | `IIndicatorService` |
| **AnalyticsService** | `services/analytics/`  | Metrics           | `IAnalyticsService` |
| **ReportingService** | `services/reporting/`  | Output            | `IReportingService` |

## Example: How Services Interact

### Scenario: Running a backtest

```python
# 1. Load configuration
config = SystemConfig.from_yaml("config/qtrader.yaml")

# 2. Initialize services (dependency injection)
data_service = DataService(config.data)
portfolio_service = PortfolioService(config.portfolio)
execution_service = ExecutionService(config.execution)
risk_service = RiskService(config.risk, portfolio_service)
indicator_service = IndicatorService(config.indicators, data_service)
analytics_service = AnalyticsService(config.analytics)
reporting_service = ReportingService(config.reporting)

# 3. Create backtest engine (coordinates services)
engine = BacktestEngine(
    data_service=data_service,
    portfolio_service=portfolio_service,
    execution_service=execution_service,
    risk_service=risk_service,
    analytics_service=analytics_service,
    reporting_service=reporting_service,
    config=config.backtest,
)

# 4. Create strategy context (user-facing API)
context = StrategyContext(
    data_service=data_service,
    portfolio_service=portfolio_service,
    indicator_service=indicator_service,
    config=config.strategy,
)

# 5. User defines strategy
class MyStrategy(BaseStrategy):
    def on_bar(self, bar: MultiBar) -> None:
        # Use context to access services
        sma = self.context.indicators.sma(bar.symbol, 20)
        position = self.context.portfolio.get_position(bar.symbol)

        if bar.adjusted.close > sma and position is None:
            self.context.buy(bar.symbol, 100)

# 6. Run backtest
strategy = MyStrategy()
results = engine.run(strategy, start_date, end_date)

# 7. Get report
report = reporting_service.generate_report(results)
print(report)
```

### Data Flow

```
DataService → MultiBar → StrategyContext → Strategy decides
                                              ↓
                                         Order created
                                              ↓
                                    RiskService validates
                                              ↓
                                 ExecutionService simulates fill
                                              ↓
                                    Trade recorded
                                              ↓
                            PortfolioService updates positions
                                              ↓
                           AnalyticsService calculates metrics
                                              ↓
                          ReportingService formats output
```

## Benefits of This Structure

### 1. Clear Separation of Concerns

- Each service has one job
- Easy to understand what each service does
- Easy to find where functionality lives

### 2. Testability

- Services can be tested in isolation
- Mocks via Protocol interfaces
- No need for complex test setup

### 3. Reusability

- Services can be used in different contexts
- Example: IndicatorService can be used outside backtests
- Example: AnalyticsService can analyze external portfolio

### 4. Maintainability

- Changes to one service don't affect others
- Clear interfaces make refactoring safe
- Easy to add new services

### 5. Scalability

- Services can be split into microservices later
- Easy to parallelize (e.g., run multiple backtests)
- Easy to cache/optimize individual services

## Migration Path

### Current State (Before Lego Architecture)

```
src/qtrader/
├── models/              ✅ Already good
├── data/                ✅ Already good
├── adapters/            ✅ Already good
└── (everything else mixed together)
```

### Phase 1 (Current - DataService)

```
src/qtrader/
├── models/              ✅ Shared
├── data/                ✅ Shared (used by DataService)
├── adapters/            ✅ Shared (used by DataService)
├── config/              ✅ Shared
└── services/
    └── data/            ✅ NEW: DataService
```

### Final State (After Phase 10)

```
src/qtrader/
├── models/              ✅ Shared (immutable contracts)
├── config/              ✅ Shared (typed configuration)
├── data/                ✅ Shared (DataService implementation)
├── adapters/            ✅ Shared (DataService implementation)
├── api/                 ✅ Shared (external integrations)
├── auth/                ✅ Shared (authentication)
└── services/            ✅ All 9 services implemented
    ├── data/
    ├── portfolio/
    ├── execution/
    ├── risk/
    ├── backtest/
    ├── strategy/
    ├── indicators/
    ├── analytics/
    └── reporting/
```

## Summary

**Shared Components:**

- `models/` - Data contracts (used by everyone)
- `config/` - Configuration (used by everyone)
- `adapters/` - Data adapters (implementation detail of DataService)
- `data/` - Data loading (implementation detail of DataService)
- `api/` - External APIs (used by Data/Execution services)
- `auth/` - Authentication (used by API clients)

**Service Components:**

- `services/*/` - Independent, testable business logic
- Each service exports a Protocol interface
- Services communicate via dependency injection
- Services use models as immutable contracts

**Key Insight:** Most code is shared (models, config, infrastructure). Services are thin orchestration layers that coordinate shared components with clean interfaces.

This structure enables:

- ✅ Independent development
- ✅ Comprehensive testing
- ✅ Clear responsibilities
- ✅ Easy maintenance
- ✅ Future scalability
