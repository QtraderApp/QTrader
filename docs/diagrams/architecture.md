# QTrader Architecture Diagrams

This document provides high-level architecture diagrams for the QTrader backtesting engine.

**Legend:**

- 🟢 Green: Implemented components (Stages 1-5A complete)
- 🟡 Yellow: In progress or planned (Stage 5B-8)
- 🔵 Blue: External dependencies or data sources

## System Architecture Overview

```mermaid
graph TB
    subgraph "External Data Sources 🔵"
        PARQUET[("Parquet Files<br/>(Algoseek)")]:::external
        CSV[("CSV Files<br/>(Security Master)")]:::external
    end

    subgraph "Data Adapters Layer 🟢"
        PARQUET_ADAPTER["AlgoseekParquetAdapter<br/>✅ Implemented"]:::implemented
        CSV_ADAPTER["CSVAdapter<br/>✅ Implemented"]:::implemented
        BASE_ADAPTER["BaseAdapter Protocol<br/>✅ Implemented"]:::implemented
    end

    subgraph "Core Data Models 🟢"
        BAR["Bar (OHLCV)<br/>✅ Implemented"]:::implemented
        ADJ["AdjustmentEvent<br/>✅ Implemented"]:::implemented
        ORDER["Order Models<br/>✅ Implemented"]:::implemented
        POSITION["Position<br/>✅ Implemented"]:::implemented
        PORTFOLIO["Portfolio<br/>✅ Implemented"]:::implemented
        LEDGER["Ledger<br/>✅ Implemented"]:::implemented
    end

    subgraph "Execution Engine 🟢"
        ENGINE["BacktestEngine<br/>✅ Implemented"]:::implemented
        FILL_POLICY["Fill Policies<br/>✅ Market, MOC, Limit, Stop"]:::implemented
        COMMISSION["Commission Models<br/>✅ Implemented"]:::implemented
        PARTICIPATION["Volume Participation<br/>✅ Implemented"]:::implemented
    end

    subgraph "Risk Management 🟡"
        SIGNAL["Signal Model<br/>🔄 In Progress (Stage 5B)"]:::planned
        RISK_MGR["RiskManager<br/>🔄 In Progress (Stage 5B)"]:::planned
        RISK_POLICY["RiskPolicy Config<br/>🔄 In Progress (Stage 5B)"]:::planned
    end

    subgraph "Strategy Layer 🟡"
        INDICATORS["Indicator Framework<br/>🔄 Planned (Stage 6A)"]:::planned
        WARMUP["Indicator Warmup<br/>🔄 Planned (Stage 6A)"]:::planned
        STRATEGY["Strategy Base Class<br/>✅ Implemented"]:::implemented
        CONTEXT["Context API<br/>✅ Implemented"]:::implemented
    end

    subgraph "Risk & Accounting 🟡"
        SHORT["Shorting Support<br/>🔄 Planned (Stage 6B)"]:::planned
        ACCRUAL["Borrow Fee Accruals<br/>🔄 Planned (Stage 6B)"]:::planned
    end

    subgraph "Output & Reporting 🟡"
        RESULTS["Run Results<br/>🔄 Planned (Stage 6B)"]:::planned
        METRICS["Performance Metrics<br/>🔄 Planned (Stage 8)"]:::planned
    end

    subgraph "Public API & CLI 🟡"
        PUBLIC_API["Public API<br/>🔄 Planned (Stage 7)"]:::planned
        CLI["CLI Interface<br/>🔄 Planned (Stage 7)"]:::planned
        CONFIG["Configuration System<br/>✅ Implemented"]:::implemented
    end

    %% Data flow
    PARQUET --> PARQUET_ADAPTER
    CSV --> CSV_ADAPTER
    PARQUET_ADAPTER --> BAR
    PARQUET_ADAPTER --> ADJ
    CSV_ADAPTER --> BAR

    %% Core models connections
    BAR --> ENGINE
    ORDER --> ENGINE
    ENGINE --> FILL_POLICY
    ENGINE --> PARTICIPATION
    ENGINE --> COMMISSION
    ENGINE --> POSITION
    ENGINE --> PORTFOLIO
    ENGINE --> LEDGER

    %% Risk management flow (NEW)
    STRATEGY --> SIGNAL
    SIGNAL --> RISK_MGR
    RISK_POLICY --> RISK_MGR
    PORTFOLIO --> RISK_MGR
    RISK_MGR --> ORDER

    %% Strategy connections
    INDICATORS --> WARMUP
    WARMUP --> STRATEGY
    CONTEXT --> STRATEGY
    CONTEXT --> RISK_MGR

    %% Risk connections
    SHORT --> ENGINE
    ACCRUAL --> LEDGER

    %% Output connections
    ENGINE --> RESULTS
    RESULTS --> METRICS

    %% API connections
    CONFIG --> PUBLIC_API
    CONFIG --> RISK_POLICY
    PUBLIC_API --> CLI
    CLI --> ENGINE
    PUBLIC_API --> ENGINE

    %% Styling
    classDef implemented fill:#90EE90,stroke:#2d5016,stroke-width:2px,color:#000
    classDef planned fill:#FFD700,stroke:#8B7500,stroke-width:2px,color:#000
    classDef external fill:#87CEEB,stroke:#4682B4,stroke-width:2px,color:#000
```

## Event Loop Architecture

```mermaid
sequenceDiagram
    participant User
    participant Engine as BacktestEngine<br/>🟢
    participant Strategy as User Strategy<br/>🟢
    participant Indicators as Indicator Manager<br/>🟡
    participant RiskMgr as RiskManager<br/>🟡
    participant Portfolio as Portfolio<br/>🟢
    participant Ledger as Ledger<br/>🟢

    User->>Engine: Initialize with config
    activate Engine

    Engine->>Strategy: on_init()
    Note over Strategy,Indicators: Register custom indicators<br/>🟡 Planned (Stage 6A)
    Strategy-->>Indicators: Register indicators

    alt Warmup Enabled 🟡
        Engine->>Indicators: Warmup Phase
        loop N warmup bars
            Engine->>Indicators: Feed historical bar
            Note over Indicators: Populate indicator state<br/>No strategy calls
        end
        Indicators-->>Engine: Warmup complete
    end

    Engine->>Strategy: on_start()
    Note over Strategy: Trading begins

    loop For each bar in dataset
        Engine->>Strategy: on_bar(ctx)
        activate Strategy

        Strategy->>Indicators: Get indicator values
        Indicators-->>Strategy: Return values (or None)

        Strategy->>Portfolio: Query positions
        Portfolio-->>Strategy: Position info

        Strategy->>Strategy: Generate trading signals
        Note over Strategy: Analyze conditions<br/>Create Signal objects

        Strategy->>RiskMgr: Submit signals (via ctx)
        deactivate Strategy
        activate RiskMgr

        loop For each signal
            RiskMgr->>Portfolio: Query equity & positions
            Portfolio-->>RiskMgr: Portfolio state

            RiskMgr->>RiskMgr: Evaluate signal<br/>1. Validate direction<br/>2. Check constraints<br/>3. Calculate size<br/>4. Apply limits<br/>5. Check cash

            alt Signal approved
                RiskMgr->>Engine: Submit sized order
                Note over RiskMgr,Engine: Signal converted to Order<br/>with approved quantity
            else Signal rejected
                Note over RiskMgr: Log rejection reason<br/>Continue to next signal
            end
        end
        deactivate RiskMgr

        Engine->>Engine: Process orders with fill policies

        opt Orders filled
            Engine->>Ledger: Record fills, update cash
            Engine->>Portfolio: Update positions
            Engine->>Strategy: on_fill(fill)
        end
    end

    Engine->>Strategy: on_end()
    Engine->>Ledger: Finalize accounting
    Engine-->>User: Return results
    deactivate Engine
```

## Risk Management Flow (Stage 5B - In Progress)

```mermaid
sequenceDiagram
    participant Strategy as User Strategy<br/>🟢
    participant Context as Context API<br/>🟢
    participant RiskMgr as RiskManager<br/>🟡
    participant Policy as RiskPolicy<br/>🟡
    participant Portfolio as Portfolio<br/>🟢
    participant Engine as ExecutionEngine<br/>🟢

    Note over Strategy: Generate trading intent

    Strategy->>Strategy: Analyze market conditions
    activate Strategy

    Strategy->>Context: Return List[Signal]
    Note over Strategy,Context: Signal = intent (what to trade)<br/>NOT sized order (how much)
    deactivate Strategy

    loop For each Signal
        Context->>RiskMgr: evaluate_signal(signal, current_price)
        activate RiskMgr

        RiskMgr->>Policy: Check allow_shorting
        Policy-->>RiskMgr: Policy rules

        RiskMgr->>Portfolio: Query current positions
        Portfolio-->>RiskMgr: Position state

        RiskMgr->>Portfolio: Query equity & cash
        Portfolio-->>RiskMgr: Financial state

        RiskMgr->>RiskMgr: 1. Validate direction
        RiskMgr->>RiskMgr: 2. Check portfolio constraints
        RiskMgr->>RiskMgr: 3. Calculate size (sizing_method)
        RiskMgr->>RiskMgr: 4. Apply concentration limits
        RiskMgr->>RiskMgr: 5. Check cash availability

        alt Signal Approved
            RiskMgr-->>Context: RiskDecision(approved=True, sized_qty=X)
            deactivate RiskMgr

            Context->>RiskMgr: signal_to_order(signal, decision)
            activate RiskMgr
            RiskMgr-->>Context: Sized Order
            deactivate RiskMgr

            Context->>Engine: submit_order(order)
            Note over Engine: Order enters execution queue

        else Signal Rejected
            RiskMgr-->>Context: RiskDecision(approved=False, reason=X)
            deactivate RiskMgr
            Note over Context: Log rejection, continue
        end
    end

    Note over Engine: Process fills (existing flow)
```

## Data Adapter Architecture

```mermaid
graph LR
    subgraph "Vendor Data Sources 🔵"
        ALGOSEEK[("Algoseek<br/>Parquet")]:::external
        CUSTOM_CSV[("Custom<br/>CSV")]:::external
        FUTURE[("Future Vendors<br/>IEX, Polygon, etc.")]:::external
    end

    subgraph "Adapter Layer 🟢"
        BASE[DataAdapter Protocol<br/>✅ Implemented]:::implemented

        ALGOSEEK_ADAPTER["AlgoseekParquetAdapter<br/>✅ Implemented<br/><br/>• DuckDB-based<br/>• Handles partitions<br/>• Adjustment events"]:::implemented

        CSV_ADAPTER["CSVAdapter<br/>✅ Implemented<br/><br/>• Security master<br/>• Configurable schema<br/>• Symbol mapping"]:::implemented

        FUTURE_ADAPTER["Future Adapters<br/>🔄 Planned<br/><br/>• IEX Cloud<br/>• Polygon.io<br/>• Custom formats"]:::planned
    end

    subgraph "Canonical Models 🟢"
        BAR["Bar (OHLCV)<br/>✅ Implemented<br/><br/>ts, symbol<br/>open, high, low<br/>close, volume"]:::implemented

        ADJ["AdjustmentEvent<br/>✅ Implemented<br/><br/>ts, symbol<br/>event_type<br/>px_factor, vol_factor"]:::implemented

        MODE["DataMode<br/>✅ Implemented<br/><br/>ADJUSTED<br/>UNADJUSTED<br/>SPLIT_ADJUSTED"]:::implemented
    end

    subgraph "Engine 🟢"
        ENGINE["BacktestEngine<br/>✅ Implemented"]:::implemented
    end

    ALGOSEEK -->|read_bars| ALGOSEEK_ADAPTER
    ALGOSEEK -->|read_adjustments| ALGOSEEK_ADAPTER
    CUSTOM_CSV -->|read_bars| CSV_ADAPTER
    FUTURE -->|read_bars| FUTURE_ADAPTER

    ALGOSEEK_ADAPTER -->|implements| BASE
    CSV_ADAPTER -->|implements| BASE
    FUTURE_ADAPTER -->|implements| BASE

    BASE -->|emits| BAR
    BASE -->|emits| ADJ
    BASE -->|declares| MODE

    BAR --> ENGINE
    ADJ --> ENGINE
    MODE --> ENGINE

    classDef implemented fill:#90EE90,stroke:#2d5016,stroke-width:2px,color:#000
    classDef planned fill:#FFD700,stroke:#8B7500,stroke-width:2px,color:#000
    classDef external fill:#87CEEB,stroke:#4682B4,stroke-width:2px,color:#000
```

## Order Execution Flow

```mermaid
flowchart TD
    START([Strategy submits order]):::implemented

    ORDER_VALIDATE{Order validation}:::implemented
    ORDER_REJECT[Reject: Invalid order]:::implemented

    RISK_CHECK{Risk checks<br/>Cash/Equity}:::implemented
    RISK_REJECT[Reject: Insufficient funds]:::implemented

    QUEUE[Add to order queue]:::implemented

    FILL_POLICY{Fill Policy Type}:::implemented

    MARKET[Market Fill<br/>✅ Stage 3]:::implemented
    MOC[Market-on-Close<br/>✅ Stage 3]:::implemented
    LIMIT[Limit Order<br/>✅ Stage 4]:::implemented
    STOP[Stop Order<br/>✅ Stage 4]:::implemented
    PARTICIPATION[Volume Participation<br/>🔄 Stage 5]:::planned

    PARTIAL{Partial fill?}:::planned

    COMMISSION[Calculate commission]:::implemented

    LEDGER_UPDATE[Update ledger<br/>Cash, PnL, Costs]:::implemented
    PORTFOLIO_UPDATE[Update portfolio<br/>Positions]:::implemented

    ON_FILL[Call strategy.on_fill]:::implemented

    CONTINUE{More bars?}:::implemented
    NEXT_BAR[Process next bar]:::implemented
    END_STRATEGY[Call strategy.on_end]:::implemented
    FINAL([Return results]):::planned

    START --> ORDER_VALIDATE
    ORDER_VALIDATE -->|Invalid| ORDER_REJECT
    ORDER_VALIDATE -->|Valid| RISK_CHECK

    RISK_CHECK -->|Failed| RISK_REJECT
    RISK_CHECK -->|Passed| QUEUE

    QUEUE --> FILL_POLICY

    FILL_POLICY -->|Market| MARKET
    FILL_POLICY -->|MOC| MOC
    FILL_POLICY -->|Limit| LIMIT
    FILL_POLICY -->|Stop| STOP
    FILL_POLICY -->|Participation| PARTICIPATION

    MARKET --> COMMISSION
    MOC --> COMMISSION
    LIMIT --> COMMISSION
    STOP --> COMMISSION
    PARTICIPATION --> PARTIAL

    PARTIAL -->|Yes| COMMISSION
    PARTIAL -->|No fill| CONTINUE

    COMMISSION --> LEDGER_UPDATE
    LEDGER_UPDATE --> PORTFOLIO_UPDATE
    PORTFOLIO_UPDATE --> ON_FILL

    ON_FILL --> CONTINUE
    ORDER_REJECT --> CONTINUE
    RISK_REJECT --> CONTINUE

    CONTINUE -->|Yes| NEXT_BAR
    CONTINUE -->|No| END_STRATEGY

    NEXT_BAR --> START
    END_STRATEGY --> FINAL

    classDef implemented fill:#90EE90,stroke:#2d5016,stroke-width:2px,color:#000
    classDef planned fill:#FFD700,stroke:#8B7500,stroke-width:2px,color:#000
```

## Indicator Framework Architecture (Planned - Stage 6A)

```mermaid
graph TB
    subgraph "Strategy Layer 🟡"
        STRATEGY["User Strategy<br/>🟡 Planned"]:::planned
        ON_INIT["on_init() hook<br/>Register indicators"]:::planned
        ON_START["on_start() hook<br/>Trading begins"]:::planned
        ON_BAR["on_bar() hook<br/>Make decisions"]:::planned
    end

    subgraph "Indicator Manager 🟡"
        MANAGER["IndicatorManager<br/>🟡 Planned"]:::planned
        WARMUP_SYS["Warmup System<br/>🟡 Planned"]:::planned
        AUTO_DETECT["Auto-detect lookback<br/>🟡 Planned"]:::planned
    end

    subgraph "Built-in Indicators 🟡"
        SMA["SMA<br/>Simple Moving Avg"]:::planned
        EMA["EMA<br/>Exponential MA"]:::planned
        BB["Bollinger Bands"]:::planned
        RSI["RSI<br/>Relative Strength"]:::planned
        MACD["MACD<br/>Moving Avg Conv/Div"]:::planned
        ATR["ATR<br/>Average True Range"]:::planned
    end

    subgraph "Helper Functions 🟡"
        CROSS["Crossover Detection"]:::planned
        THRESHOLD["Threshold Detection"]:::planned
        DIVERGENCE["Divergence Analysis"]:::planned
        HISTOGRAM["Histogram Analysis"]:::planned
        TREND["Trend Detection"]:::planned
    end

    subgraph "Custom Indicators 🟡"
        CUSTOM["User Custom<br/>Indicators"]:::planned
        BASE_IND["BaseIndicator<br/>compute(), warmup()"]:::planned
    end

    subgraph "Context API 🟢"
        CTX["Context<br/>✅ Implemented"]:::implemented
        IND_ACCESS["ctx.ind property<br/>🟡 Planned"]:::planned
        BAR_HISTORY["ctx.get_bar_history()<br/>✅ Implemented"]:::implemented
    end

    subgraph "Data Flow 🟢"
        BARS["Historical Bars<br/>✅ Implemented"]:::implemented
    end

    BARS --> WARMUP_SYS

    STRATEGY --> ON_INIT
    ON_INIT --> MANAGER
    MANAGER --> |Register| SMA
    MANAGER --> |Register| EMA
    MANAGER --> |Register| BB
    MANAGER --> |Register| RSI
    MANAGER --> |Register| MACD
    MANAGER --> |Register| ATR

    CUSTOM --> BASE_IND
    BASE_IND --> MANAGER

    WARMUP_SYS --> AUTO_DETECT
    AUTO_DETECT --> MANAGER
    MANAGER --> ON_START

    ON_START --> ON_BAR
    ON_BAR --> IND_ACCESS
    IND_ACCESS --> CTX
    CTX --> BAR_HISTORY

    IND_ACCESS --> |Get values| MANAGER
    MANAGER --> |Return| SMA
    MANAGER --> |Return| EMA
    MANAGER --> |Return| BB

    CTX --> CROSS
    CTX --> THRESHOLD
    CTX --> DIVERGENCE

    classDef implemented fill:#90EE90,stroke:#2d5016,stroke-width:2px,color:#000
    classDef planned fill:#FFD700,stroke:#8B7500,stroke-width:2px,color:#000
```

## Implementation Progress

### ✅ Completed (Stages 1-5A)

| Component                      | Stage | Status                      |
| ------------------------------ | ----- | --------------------------- |
| **Data Models**                | 1     | ✅ Complete                 |
| - Bar (OHLCV)                  | 1     | ✅ 8/8 tests passing        |
| - Data Adapters                | 1     | ✅ Algoseek Parquet, CSV    |
| - Configuration                | 1     | ✅ YAML-based config        |
| **Order & Ledger**             | 2     | ✅ Complete                 |
| - Order Models                 | 2     | ✅ Market, Limit, Stop, MOC |
| - Portfolio                    | 2     | ✅ Position tracking        |
| - Ledger                       | 2     | ✅ Cash, PnL, costs         |
| **Execution Engine**           | 3-4   | ✅ Complete                 |
| - Market & MOC fills           | 3     | ✅ Implemented              |
| - Limit & Stop fills           | 4     | ✅ Implemented              |
| - Commission models            | 3-4   | ✅ Per-share + ticket min   |
| **Strategy Base**              | 3-4   | ✅ Complete                 |
| - Strategy protocol            | 3     | ✅ Lifecycle hooks          |
| - Context API                  | 3     | ✅ Portfolio access         |
| **Volume Participation**       | 5A    | ✅ Complete                 |
| - Participation fills          | 5A    | ✅ Partial fills            |
| - Volume limits                | 5A    | ✅ Max % of bar volume      |
| - Residual queuing             | 5A    | ✅ Multi-bar fills          |
| - High participation guardrail | 5A    | ✅ Safety checks            |

**Total Tests Passing:** 177 tests (36 Stage 1 + 55 Stage 2 + 86 Stages 3-5A)

### 🔄 In Progress (Stage 5B)

| Component                 | Stage | Status                           |
| ------------------------- | ----- | -------------------------------- |
| **Risk Management**       | 5B    | 🔄 In Progress (Days 17-19)      |
| - Signal model            | 5B    | 🔄 Trading intent representation |
| - RiskPolicy config       | 5B    | 🔄 Configuration system          |
| - RiskManager             | 5B    | 🔄 Evaluation & sizing logic     |
| - Position sizing methods | 5B    | 🔄 4 basic methods (Phase 1)     |
| - Concentration limits    | 5B    | 🔄 Max position %, max count     |
| - Leverage constraints    | 5B    | 🔄 Gross/net exposure limits     |
| - Strategy integration    | 5B    | 🔄 Signal-based workflow         |

**Expected Tests:** 43 tests (35 unit + 8 integration)

### 🔄 Planned (Stages 6A-8)

| Component                | Stage | Status                          |
| ------------------------ | ----- | ------------------------------- |
| **Indicators Framework** | 6A    | 🔄 Planned (Days 20-23)         |
| - Base indicator class   | 6A    | 🔄 compute(), warmup(), reset() |
| - Built-in indicators    | 6A    | 🔄 SMA, EMA, BB, RSI, MACD, ATR |
| - Helper functions       | 6A    | 🔄 13 utility functions         |
| - Warmup system          | 6A    | 🔄 Auto/explicit warmup         |
| - on_init() hook         | 6A    | 🔄 Pre-warmup registration      |
| **Shorting & Accruals**  | 6B    | 🔄 Planned (Days 24-27)         |
| - Short selling          | 6B    | 🔄 Borrow/return logic          |
| - Borrow fees            | 6B    | 🔄 Daily accruals               |
| - Run results            | 6B    | 🔄 JSON/CSV output              |
| **Public API & CLI**     | 7     | 🔄 Planned (Days 28-32)         |
| - Public API             | 7     | 🔄 pip installable              |
| - CLI interface          | 7     | 🔄 qtrader backtest             |
| - Documentation          | 7     | 🔄 API docs, examples           |
| **Golden Baselines**     | 8     | 🔄 Planned (Days 33-37)         |
| - Buy & Hold             | 8     | 🔄 Reference strategy           |
| - SMA Cross              | 8     | 🔄 Indicator validation         |
| - CI Integration         | 8     | 🔄 Automated validation         |

## Component Dependencies

```mermaid
graph TD
    S1[Stage 1: Data & Models<br/>✅ Complete]:::implemented
    S2[Stage 2: Orders & Ledger<br/>✅ Complete]:::implemented
    S3[Stage 3: Market/MOC Exec<br/>✅ Complete]:::implemented
    S4[Stage 4: Limit/Stop Exec<br/>✅ Complete]:::implemented
    S5A[Stage 5A: Participation<br/>✅ Complete]:::implemented
    S5B[Stage 5B: Risk Mgmt<br/>🔄 In Progress]:::planned
    S6A[Stage 6A: Indicators<br/>🔄 Planned]:::planned
    S6B[Stage 6B: Shorting<br/>🔄 Planned]:::planned
    S7[Stage 7: API & CLI<br/>🔄 Planned]:::planned
    S8[Stage 8: Golden Tests<br/>🔄 Planned]:::planned

    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5A
    S5A --> S5B
    S5B --> S6A
    S5B --> S6B
    S6A --> S6B
    S6B --> S7
    S7 --> S8
    S6A --> S8

    classDef implemented fill:#90EE90,stroke:#2d5016,stroke-width:2px,color:#000
    classDef planned fill:#FFD700,stroke:#8B7500,stroke-width:2px,color:#000
```

## Notes

- **Green (🟢)**: Fully implemented and tested (Stages 1-5A complete, 177 tests passing)
- **Yellow (🟡)**: Specification complete, implementation in progress or planned (Stage 5B-8)
- **Blue (🔵)**: External dependencies or data sources

The system follows a **ports & adapters** architecture with clear separation between:

1. Data ingestion (adapters normalize vendor formats)
1. Core models (vendor-agnostic Bar contract)
1. Execution engine (deterministic event loop)
1. Strategy layer (user-defined trading logic)
1. Output & reporting (results, metrics, audit trails)

For detailed specifications, see:

- Technical Specification: `docs/specs/phase01.md`
- Implementation Plan: `docs/implementation_plan_phase01.md`
