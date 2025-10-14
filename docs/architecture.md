# QTrader Architecture Overview

**High-level system architecture and component diagrams**

For detailed architecture diagrams with implementation status visualization, see:

**[📊 Architecture Diagrams](diagrams/architecture.md)**

This includes:

- **System Architecture Overview** - Complete component hierarchy with color-coded implementation status
- **Event Loop Architecture** - Sequence diagram showing strategy lifecycle and warmup phases
- **Data Adapter Architecture** - Vendor normalization to canonical Bar model
- **Order Execution Flow** - End-to-end order processing with fill policies
- **Indicator Framework Architecture** - Planned indicator system (Stage 6A)
- **Implementation Progress** - Detailed status tables for all 8 stages
- **Component Dependencies** - Stage dependency graph

## Quick Reference

### Implementation Status (Color-Coded)

- 🟢 **Green**: Implemented & tested (Stages 1-6A complete)
- 🟡 **Yellow**: Planned but not yet implemented (Stages 6B-8)
- 🔵 **Blue**: External dependencies or data sources

### Current Progress

**Completed:** Stages 1-6A (314 tests passing)

- ✅ Data models & adapters (Bar, MultiBar, Order, Portfolio, Ledger, Position)
- ✅ Execution engine (Market, MOC, Limit, Stop orders with fill policies)
- ✅ Commission models & split processing
- ✅ Strategy base class & Context API
- ✅ Volume participation with partial fills
- ✅ **Risk Management System (Stage 5B)**
  - Signal model (trading intent with direction, type, sizing method)
  - RiskManager (portfolio-scoped risk evaluation)
  - Position sizing methods (Fixed, Percent, Risk-based, Kelly)
  - Concentration & leverage limits
  - Strategy integration (signal-based workflow)
- ✅ **Indicator Framework (Stage 6A)**
  - Base indicator class with state management
  - Indicator manager with warmup coordination
  - Helper utilities for indicator calculations
  - Momentum indicators (MACD, RSI)
  - Trend indicators (SMA, EMA)
  - Volatility indicators (ATR, Bollinger Bands)

**Planned:** Stages 6B-8

- 🔄 Shorting & borrow fees (Stage 6B)
- 🔄 Public API & CLI refinement (Stage 7)
- 🔄 Golden baseline tests (Stage 8)

## Module Structure

### Core Components

**📁 models/** - Core data models (Pydantic-based, immutable)

- `bar.py` - Bar and PriceSeries models (OHLCV data representation)
- `multi_bar.py` - MultiBar (unadjusted/adjusted/total_return modes)
- `order.py` - Order models (Market, Limit, Stop, MOC)
- `portfolio.py` - Portfolio with cash/equity tracking
- `position.py` - Position and PositionTracker (average cost method)
- `ledger.py` - Transaction ledger for fills and corporate events
- `instrument.py` - Instrument metadata (symbol, type, data source)
- `vendors/algoseek/` - Algoseek-specific bar and price series models

**📁 data/** - Data loading and iteration

- `loader.py` - DataLoader (loads parquet/CSV data with filtering)
- `iterator.py` - PriceSeriesIterator (time-ordered multi-symbol iteration)
- `bar_merger.py` - Corporate event adjustment logic (splits, dividends)

**📁 adapters/** - Vendor data normalization

- `algoseek.py` - AlgoseekAdapter (parquet → MultiBar conversion)
- `adjustments.py` - Corporate action adjustment calculations
- `resolver.py` - DataSourceResolver (vendor adapter registry)

**📁 execution/** - Order execution and fills

- `engine.py` - ExecutionEngine (order validation, fill simulation)
- `fill_policy.py` - Fill policies (aggressive, conservative, VWAP)
- `commission.py` - Commission calculators (flat, percent, tiered)
- `config.py` - ExecutionConfig (slippage, commission settings)
- `split_processor.py` - Position adjustment for stock splits
- `warmup.py` - Warmup mode (load historical bars without execution)

**📁 api/** - User-facing interfaces

- `strategy.py` - Strategy base class (lifecycle hooks)
- `context.py` - Context API (order submission, data access)
- `backtest.py` - Backtest orchestration (event loop, warmup)

**📁 risk/** - Risk management and position sizing

- `signal.py` - Signal model (trading intent: direction, type, weight)
- `policy.py` - RiskPolicy (portfolio limits, sizing method)
- `sizing.py` - Position sizing functions (Fixed, Percent, Risk, Kelly)
- `manager.py` - RiskManager (signal evaluation, limit enforcement)

**📁 indicators/** - Technical indicators

- `base.py` - Indicator base class (state management, warmup)
- `manager.py` - IndicatorManager (registration, lifecycle)
- `helpers.py` - Common calculation utilities
- `momentum/` - MACD, RSI
- `trend/` - SMA, EMA
- `volatility/` - ATR, Bollinger Bands

**📁 config/** - Configuration management

- `data_config.py` - DataConfig (file paths, bar schema)
- `logging_config.py` - Structured logging with LoggerFactory
- `system_config.py` - System-wide settings

**📁 cli/** - Command-line interface

- `cli.py` - Click-based CLI entry point

### Test Coverage

**📁 tests/unit/** - 27 test modules, 314 tests

- `adapters/` - Data adapter tests
- `data/` - Data loading and iteration tests
- `execution/` - Order execution and fill policy tests
- `indicators/momentum/` - MACD, RSI tests
- `indicators/trend/` - SMA, EMA tests
- `indicators/volatility/` - ATR, Bollinger Bands tests
- `models/` - Core model validation tests
- `risk/` - Signal, policy, sizing, manager tests

**📁 tests/integration/** - End-to-end workflow tests

- Full backtest execution tests
- Risk workflow integration
- Data layer with corporate events
- Split accounting validation

## Related Documentation

- **Technical Specification:** [specs/phase01.md](specs/phase01.md)
- **Implementation Plan:** [implementation_plan_phase01.md](implementation_plan_phase01.md)
- **Logging Guide:** [logging.md](logging.md)
