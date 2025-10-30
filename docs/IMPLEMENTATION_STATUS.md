# QTrader Implementation Status

**Last Updated:** October 30, 2025\
**Branch:** `feature/lego-architecture`\
**Status:** Core Infrastructure Complete, Services Integration In Progress

______________________________________________________________________

## Table of Contents

1. [Vision & Architecture](#vision--architecture)
1. [Implementation Status](#implementation-status)
1. [Completed Components](#completed-components)
1. [Current State](#current-state)
1. [Roadmap](#roadmap)
1. [How It Works](#how-it-works)
1. [Key Design Decisions](#key-design-decisions)

______________________________________________________________________

## Vision & Architecture

### The Big Picture

QTrader is an **event-driven backtesting and trading system** built on a microservices-inspired architecture. The system processes market data through a pipeline of specialized services, each responsible for a specific domain concern.

**Core Philosophy:**

- **Event-Driven**: All communication happens via immutable events on an event bus
- **Language-Agnostic Contracts**: JSON Schema contracts enable polyglot implementations
- **Single Responsibility**: Each service has one clear job
- **Deterministic Replay**: Complete audit trail for debugging and compliance
- **Type-Safe**: Pydantic models with JSON Schema validation

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Event Bus                                   │
│                    (Pub/Sub Message Broker)                         │
└─────────────────────────────────────────────────────────────────────┘
         ▲              ▲              ▲              ▲              ▲
         │              │              │              │              │
         │              │              │              │              │
    ┌────▼────┐    ┌───▼────┐    ┌───▼────┐    ┌───▼─────┐   ┌────▼────┐
    │  Data   │    │Strategy│    │Manager │    │Execution│   │Portfolio│
    │ Service │    │Service │    │Service │    │ Service │   │ Service │
    └─────────┘    └────────┘    └────────┘    └─────────┘   └─────────┘
         │              │              │              │              │
         ▼              ▼              ▼              ▼              ▼
    PriceBar       Signal         Order           Fill         Portfolio
     Event         Event          Event          Event           State
```

**Event Flow:**

1. **DataService** → Publishes `PriceBarEvent` with OHLCV data
1. **StrategyService** → Consumes bars, publishes `SignalEvent` (trading intent)
1. **ManagerService** → Consumes signals, applies risk/sizing, publishes `OrderEvent`
1. **ExecutionService** → Consumes orders, simulates fills, publishes `FillEvent`
1. **PortfolioService** → Consumes fills, updates positions, publishes `PortfolioStateEvent`

______________________________________________________________________

## Implementation Status

### ✅ Completed (Production Ready)

| Component            | Status      | Location                                | Tests              |
| -------------------- | ----------- | --------------------------------------- | ------------------ |
| **Event System**     | ✅ Complete | `src/qtrader/events/`                   | 145 tests          |
| **Contracts**        | ✅ Complete | `src/qtrader/contracts/`                | 22 schema tests    |
| **EventBus**         | ✅ Complete | `src/qtrader/events/event_bus.py`       | 24 tests           |
| **BacktestEngine**   | ✅ Complete | `src/qtrader/engine/backtest_engine.py` | Integration tested |
| **DataService**      | ✅ Complete | `src/qtrader/services/data/`            | Full coverage      |
| **StrategyService**  | ✅ Complete | `src/qtrader/services/strategy/`        | Full coverage      |
| **PortfolioService** | ✅ Complete | `src/qtrader/services/portfolio/`       | 89 tests           |

### 🚧 In Progress

| Component            | Status         | Blockers                 | ETA      |
| -------------------- | -------------- | ------------------------ | -------- |
| **ManagerService**   | 🔴 Not Started | Need implementation plan | 2-3 days |
| **ExecutionService** | 🟡 Partial     | Needs event integration  | 2-3 days |

### 📋 Planned

| Component              | Priority | Dependencies   | Description               |
| ---------------------- | -------- | -------------- | ------------------------- |
| **RiskService**        | High     | ManagerService | Pre-trade risk evaluation |
| **ReportingService**   | Medium   | All services   | Performance analytics     |
| **LiveTradingAdapter** | Low      | All services   | Broker integration        |

______________________________________________________________________

## Completed Components

### 1. Event System ✅

**Location:** [`src/qtrader/events/`](../src/qtrader/events/)

**Key Classes:**

- `BaseEvent` - Common envelope pattern (event_id, occurred_at, source_service)
- `ValidatedEvent` - JSON Schema validation against contracts
- `EventBus` - Pub/sub message broker with topic-based subscriptions

**Implemented Events:**

- `PriceBarEvent` - OHLCV market data bars
- `SignalEvent` - Trading signals from strategies
- `OrderEvent` - Order instructions to execution
- `FillEvent` - Order execution confirmations
- `ConsolidatedPortfolioEvent` - Complete portfolio snapshots
- `CorporateActionEvent` - Splits, dividends, etc.
- `ValuationTriggerEvent` - End-of-day marking

**Features:**

- Immutable events (frozen Pydantic models)
- Automatic UUID generation
- ISO 8601 timestamps
- Correlation/causation tracking for distributed tracing
- Type-safe with full IDE support

**Reference:** See [`src/qtrader/events/events.py`](../src/qtrader/events/events.py) for all event definitions.

______________________________________________________________________

### 2. Contracts (JSON Schemas) ✅

**Location:** [`src/qtrader/contracts/`](../src/qtrader/contracts/)

Language-agnostic event contracts using JSON Schema Draft 2020-12.

**Directory Structure:**

```
contracts/
├── schemas/
│   ├── envelope.v1.json              # Common envelope pattern
│   ├── data/
│   │   ├── bar.v1.json               # OHLCV bars with adjustment factors
│   │   └── corporate_action.v1.json  # Corporate events
│   ├── strategy/
│   │   └── signal.v1.json            # Trading signals
│   ├── manager/
│   │   └── order.v1.json             # Order instructions
│   ├── execution/
│   │   └── fill.v1.json              # Fill confirmations
│   └── portfolio/
│       └── consolidated_portfolio.v1.json  # Portfolio snapshots
└── examples/
    └── [matching example files]
```

**Key Features:**

- Service-based organization (by event producer)
- Semantic versioning in filenames (`contract.v1.json`, `contract.v2.json`)
- Decimal-as-string for financial precision
- Strict typing with `additionalProperties: false`
- Complete field documentation

**Validation:** All events validate against their schemas at runtime via `ValidatedEvent` base class.

**Reference:** See [`src/qtrader/contracts/README.md`](../src/qtrader/contracts/README.md) for contract documentation.

______________________________________________________________________

### 3. EventBus ✅

**Location:** [`src/qtrader/events/event_bus.py`](../src/qtrader/events/event_bus.py)

In-memory pub/sub message broker for service coordination.

**Features:**

- **Topic-based subscriptions:** Services subscribe to specific event types
- **Type-safe publishing:** Only `BaseEvent` instances accepted
- **Async-ready:** Designed for future async/await support
- **Lifecycle management:** Register/unregister subscribers dynamically
- **Complete audit trail:** All events stored in order

**Usage Pattern:**

```python
# Services subscribe to topics
event_bus.subscribe("bar", strategy_service.on_bar)
event_bus.subscribe("signal", manager_service.on_signal)

# Services publish events
event_bus.publish(PriceBarEvent(...))
event_bus.publish(SignalEvent(...))
```

**Testing:** 24 unit tests covering subscriptions, publishing, wildcards, and edge cases.

______________________________________________________________________

### 4. BacktestEngine ✅

**Location:** [`src/qtrader/engine/backtest_engine.py`](../src/qtrader/engine/backtest_engine.py)

Orchestrates the backtest simulation by coordinating all services.

**Responsibilities:**

1. **Service Initialization:** Creates and wires all services with EventBus
1. **Data Loading:** Manages historical data sources (CSV, Parquet, Arrow)
1. **Time Progression:** Advances simulation time bar-by-bar
1. **Event Coordination:** Ensures correct event ordering and delivery
1. **Result Collection:** Gathers performance metrics and state snapshots

**Key Methods:**

- `from_config()` - Factory method for configuration-based setup
- `run()` - Main simulation loop
- `process_bar()` - Single bar processing with event flow
- `get_results()` - Extract performance data and equity curves

**Configuration:** YAML-based configuration in [`config/system.yaml`](../config/system.yaml)

______________________________________________________________________

### 5. DataService ✅

**Location:** [`src/qtrader/services/data/`](../src/qtrader/services/data/)

Streams historical market data and publishes bar events.

**Features:**

- **Multiple data sources:** CSV, Parquet, PyArrow Table
- **Corporate actions:** Automatic splits and dividend processing
- **Adjustment factors:** Supports both adjusted and unadjusted prices
- **Multi-symbol support:** Handles portfolio-level data streaming
- **Warmup period:** Provides initial data for indicator calculation

**Event Publishing:**

- `PriceBarEvent` - For each OHLCV bar
- `CorporateActionEvent` - For splits, dividends, etc.

**Usage:**

```python
data_service = DataService(event_bus=event_bus, config=data_config)
data_service.load_data(symbol="AAPL", start_date="2020-01-01")
data_service.stream_bar(datetime(2020, 1, 2))  # Publishes PriceBarEvent
```

**Reference:** See [`src/qtrader/services/data/service.py`](../src/qtrader/services/data/service.py)

______________________________________________________________________

### 6. StrategyService ✅

**Location:** [`src/qtrader/services/strategy/`](../src/qtrader/services/strategy/)

Executes trading strategies and emits signals based on market conditions.

**Architecture:**

- **Strategy Protocol:** Interface that all strategies must implement
- **Event-Driven:** Strategies react to `PriceBarEvent` from data service
- **Context Object:** Strategies emit signals via `Context.emit_signal()`
- **Multi-Strategy:** Supports multiple concurrent strategies

**Signal Emission:**

```python
class MyStrategy:
    def on_bar(self, bar: PriceBarEvent, context: Context):
        # Strategy logic
        if conditions_met:
            context.emit_signal(
                symbol="AAPL",
                intention=SignalIntention.OPEN_LONG,
                price=bar.bar.close,
                confidence=0.85,
                stop_loss=Decimal("140.00"),
                take_profit=Decimal("160.00")
            )
```

**Built-in Strategies:**

- `BuyAndHoldStrategy` - Simple buy on first bar
- `SMADualCrossStrategy` - Moving average crossover
- Custom strategies via protocol implementation

**Reference:** See [`src/qtrader/services/strategy/service.py`](../src/qtrader/services/strategy/service.py)

______________________________________________________________________

### 7. PortfolioService ✅

**Location:** [`src/qtrader/services/portfolio/`](../src/qtrader/services/portfolio/)

Manages positions, cash, P&L, and portfolio accounting with complete audit trail.

**Features:**

- **Lot-Based Accounting:** FIFO for longs, LIFO for shorts
- **Realized & Unrealized P&L:** Accurate profit tracking
- **Corporate Actions:** Automatic position adjustments for splits/dividends
- **Fee Accrual:** Commissions, borrow fees, margin interest
- **Complete Ledger:** Full transaction history for compliance
- **Multi-Currency:** Support for FX positions

**Key Components:**

- `PortfolioService` - Main service implementation ([`service.py`](../src/qtrader/services/portfolio/service.py))
- `LotTracker` - FIFO/LIFO lot matching ([`lot_tracker.py`](../src/qtrader/services/portfolio/lot_tracker.py))
- `Position`, `Lot`, `LedgerEntry` - Core models ([`models.py`](../src/qtrader/services/portfolio/models.py))
- `IPortfolioService` - Protocol interface ([`interface.py`](../src/qtrader/services/portfolio/interface.py))

**State Management:**

- `get_state()` - Immutable portfolio snapshot
- `get_snapshot()` - Serializable state for persistence
- `restore_from_snapshot()` - Deterministic replay

**Testing:** 89 comprehensive unit tests covering all accounting scenarios.

**Reference:** See [`src/qtrader/services/portfolio/__init__.py`](../src/qtrader/services/portfolio/__init__.py)

______________________________________________________________________

## Current State

### What's Working Now

You can run a **complete end-to-end backtest** with:

1. ✅ Historical data loading (CSV/Parquet)
1. ✅ Strategy execution with signal generation
1. ✅ **Manual order creation** (bypassing Manager - temporary)
1. ✅ Fill simulation (basic execution engine)
1. ✅ Portfolio accounting with P&L tracking
1. ✅ Performance reporting

**Example:** [`basic_run_example.py`](../basic_run_example.py)

### Current Limitations

**Missing Services:**

1. ❌ **ManagerService** - Currently strategies must create orders manually
1. ❌ **RiskService** - No pre-trade risk checks or position sizing

**Workarounds in Place:**

- Strategies use `Context.emit_order()` directly (temporary)
- Fixed position sizing (100 shares)
- No portfolio-level risk limits

### What This Means

The **core infrastructure is complete and battle-tested**, but the **decision-making layer** (Manager + Risk) needs implementation to unlock full capabilities like:

- Dynamic position sizing based on portfolio equity
- Risk-adjusted position sizing (e.g., 1% portfolio risk per trade)
- Multi-strategy coordination and allocation
- Portfolio-level exposure limits

______________________________________________________________________

## Roadmap

### Phase 1: ManagerService (Next - 2-3 days)

**Goal:** Implement signal-to-order translation with basic position sizing.

**Tasks:**

1. Create `ManagerService` class with event subscriptions
1. Implement `on_signal()` handler to consume `SignalEvent`
1. Add basic position sizing logic:
   - Fixed quantity mode (100 shares)
   - Fixed equity percentage mode (10% of portfolio)
1. Create `OrderEvent` and publish to EventBus
1. Wire into BacktestEngine
1. Add unit tests (target: 30+ tests)

**Outcome:** Strategies emit signals → Manager creates sized orders → Execution fills

**Reference:** [`docs/lego_architecture/MANAGER_SERVICE_PLAN.md`](lego_architecture/MANAGER_SERVICE_PLAN.md) (to be created)

______________________________________________________________________

### Phase 2: ExecutionService Integration (2-3 days)

**Goal:** Refactor existing execution engine to event-driven architecture.

**Current State:**

- `ExecutionEngine` exists but uses old models
- Located in [`src/qtrader/engine/execution_engine.py`](../src/qtrader/engine/execution_engine.py)

**Tasks:**

1. Create `ExecutionService` with EventBus integration
1. Subscribe to `OrderEvent` from ManagerService
1. Subscribe to `PriceBarEvent` for fill evaluation
1. Implement fill policies:
   - Market orders (fill at bar close)
   - Limit orders (fill if price reached)
   - Stop orders (trigger then fill)
1. Publish `FillEvent` on execution
1. Add commission calculation
1. Add slippage simulation
1. Wire into BacktestEngine
1. Add unit tests (target: 40+ tests)

**Outcome:** Complete order-to-fill lifecycle with realistic execution simulation

**Reference:** [`docs/lego_architecture/EXECUTION_SERVICE_PLAN.md`](lego_architecture/EXECUTION_SERVICE_PLAN.md) (to be created)

______________________________________________________________________

### Phase 3: RiskService (High Priority - 3-4 days)

**Goal:** Pre-trade risk evaluation and position sizing.

**Responsibilities:**

1. Evaluate signals against risk policy (YAML config)
1. Calculate position sizes using various algorithms:
   - Fixed equity percentage
   - Volatility targeting
   - Kelly Criterion
   - Risk parity
1. Check portfolio-level limits:
   - Max position size per asset
   - Max drawdown limits
   - Sector concentration limits
   - Leverage limits
1. Approve/reject signals with detailed reasons

**Integration Point:** Manager calls RiskService before creating orders

**Configuration:** [`src/qtrader/libraries/risk/buildin/naive.yaml`](../src/qtrader/libraries/risk/buildin/naive.yaml)

**Reference:** [`src/qtrader/libraries/risk/`](../src/qtrader/libraries/risk/) (existing foundation)

______________________________________________________________________

### Phase 4: ReportingService (Medium Priority - 2-3 days)

**Goal:** Performance analytics and result visualization.

**Features:**

- Equity curve generation
- Drawdown analysis
- Sharpe/Sortino ratios
- Trade statistics (win rate, avg profit/loss)
- Position-level attribution
- Strategy-level attribution
- Export to JSON/CSV/HTML

**Integration:** Consumes all events for comprehensive analysis

______________________________________________________________________

### Phase 5: Live Trading Support (Future)

**Goal:** Adapt for real-time trading with broker integration.

**Changes Needed:**

1. **Real-time EventBus:** Replace in-memory with message queue (Redis/Kafka)
1. **Broker Adapters:** Interactive Brokers, Alpaca, etc.
1. **Order Management:** Track live order states
1. **Position Reconciliation:** Sync with broker
1. **Market Data Streaming:** Replace historical replay

**Note:** Architecture is designed for this - event contracts are language/platform agnostic.

______________________________________________________________________

## How It Works

### Backtest Execution Flow

1. **Initialization:**

   ```
   BacktestEngine.from_config()
   ├── Create EventBus
   ├── Initialize DataService(event_bus)
   ├── Initialize StrategyService(event_bus)
   ├── Initialize PortfolioService(event_bus)
   └── Load configurations from YAML
   ```

1. **Simulation Loop:**

   ```
   For each bar in historical data:
   │
   ├─ DataService publishes PriceBarEvent
   │  └─ EventBus delivers to subscribers
   │
   ├─ StrategyService receives bar
   │  ├─ Update indicators
   │  ├─ Evaluate conditions
   │  └─ Emit SignalEvent (if triggered)
   │
   ├─ [FUTURE] ManagerService receives signal
   │  ├─ Apply risk checks
   │  ├─ Calculate position size
   │  └─ Emit OrderEvent
   │
   ├─ [FUTURE] ExecutionService receives order
   │  ├─ Evaluate fill conditions
   │  └─ Emit FillEvent
   │
   └─ PortfolioService receives fill
      ├─ Update positions via lot tracking
      ├─ Calculate P&L
      └─ Publish PortfolioStateEvent
   ```

1. **Result Collection:**

   ```
   BacktestEngine.get_results()
   ├── Extract equity curve
   ├── Calculate performance metrics
   └── Generate trade log
   ```

### Event Delivery Guarantees

**Current Implementation (In-Memory):**

- ✅ Ordered delivery (FIFO per topic)
- ✅ Synchronous processing (no race conditions)
- ✅ Complete event history
- ⚠️ No persistence (events lost on crash)
- ⚠️ No replay (must rerun entire backtest)

**Future (Production):**

- Event sourcing with persistent log
- Checkpoint/resume capability
- Distributed processing support

______________________________________________________________________

## Key Design Decisions

### 1. Why Event-Driven Architecture?

**Benefits:**

- **Loose Coupling:** Services don't know about each other, only events
- **Testability:** Each service can be tested in isolation
- **Debuggability:** Complete event log for replay and analysis
- **Extensibility:** Add new services without modifying existing ones
- **Polyglot:** Can rewrite services in any language (Rust, C++, etc.)

**Trade-offs:**

- More complex than direct function calls
- Requires discipline in event design
- Harder to trace execution flow initially

### 2. Why JSON Schema Contracts?

**Benefits:**

- **Language Agnostic:** Python, Rust, TypeScript all validate the same way
- **Versioning:** Clear breaking vs non-breaking changes
- **Documentation:** Schema is self-documenting
- **Validation:** Runtime checks prevent bugs from propagating

**Trade-offs:**

- More files to maintain (schema + example + event class)
- Slower than pure Python (validation overhead)

**Decision:** Worth it for correctness and future-proofing.

### 3. Why Separate Manager from Execution?

**Manager (Strategic):**

- "Should we trade?" (risk checks)
- "How much?" (position sizing)
- "What type?" (market vs limit)

**Execution (Tactical):**

- "Can we fill?" (market conditions)
- "At what price?" (slippage simulation)
- "When?" (order matching)

**Benefit:** Clean separation of concerns, realistic simulation.

### 4. Why Immutable Events?

**Benefits:**

- **Thread Safety:** No race conditions
- **Audit Trail:** Events can't be changed after publishing
- **Replay:** Deterministic results from same event sequence
- **Debugging:** Know exactly what happened and when

**Implementation:** Pydantic models with `frozen=True`

### 5. Why Lot-Based Portfolio Accounting?

**Benefits:**

- **Tax Accuracy:** FIFO/LIFO for cost basis
- **Corporate Actions:** Correct position adjustments
- **Split Handling:** Track individual purchase lots
- **Audit Trail:** Complete transaction history

**Trade-off:** More complex than position-only tracking, but necessary for realism.

______________________________________________________________________

## Getting Started

### Run a Simple Backtest

```bash
# Activate environment
cd /home/javier/Projects/QTrader
uv sync

# Run basic example
uv run python basic_run_example.py
```

### Configuration Files

- **System:** [`config/system.yaml`](../config/system.yaml) - Backtest parameters
- **Portfolio:** [`config/portfolio.yaml`](../config/portfolio.yaml) - Initial capital, fees
- **Data Sources:** [`config/data_sources.yaml`](../config/data_sources.yaml) - Historical data paths
- **Risk Policy:** [`src/qtrader/libraries/risk/buildin/naive.yaml`](../src/qtrader/libraries/risk/buildin/naive.yaml)

### Directory Guide

```
QTrader/
├── src/qtrader/
│   ├── events/              # Event system and all event classes
│   ├── contracts/           # JSON Schema contracts
│   ├── engine/              # BacktestEngine orchestration
│   ├── services/            # All domain services
│   │   ├── data/            # Market data streaming
│   │   ├── strategy/        # Trading strategy execution
│   │   ├── portfolio/       # Position and P&L accounting
│   │   ├── execution/       # Order execution (needs refactor)
│   │   └── manager/         # Signal-to-order (TO DO)
│   └── libraries/           # Reusable components
│       ├── indicators/      # Technical indicators
│       └── risk/            # Risk management
├── tests/                   # Comprehensive test suite
│   ├── unit/                # Unit tests per service
│   └── integration/         # End-to-end tests
├── config/                  # YAML configurations
├── docs/                    # Documentation
└── examples/                # Example strategies and usage
```

______________________________________________________________________

## Contributing

### Adding a New Service

1. Create service directory: `src/qtrader/services/my_service/`
1. Define event contracts: `src/qtrader/contracts/schemas/my_service/*.json`
1. Implement service with EventBus integration
1. Add unit tests: `tests/unit/services/my_service/`
1. Update BacktestEngine to wire the service
1. Document in this file

### Adding a New Event

1. Create JSON Schema: `src/qtrader/contracts/schemas/{service}/{event}.v1.json`
1. Create example: `src/qtrader/contracts/examples/{service}/{event}.v1.example.json`
1. Add Pydantic class in `src/qtrader/events/events.py`
1. Add validation test: `tests/unit/events/test_{event}_event.py`
1. Export in `src/qtrader/events/__init__.py`

______________________________________________________________________

## Questions?

- **Architecture Decisions:** See [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- **Event Contracts:** See [`src/qtrader/contracts/README.md`](../src/qtrader/contracts/README.md)
- **API Documentation:** See [`docs/API_DESIGN.md`](API_DESIGN.md)

______________________________________________________________________

**Next Steps:** Implement ManagerService to complete the core event flow.
