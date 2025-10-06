# QTrader Phase 1 — Implementation Plan

**Version:** 3.1 **Date:** October 6, 2025 **Status:** Stage 6B Complete (including Long Dividend Extension) **Reference:** `docs/specs/phase01.md` v1.0

______________________________________________________________________

## 📊 Status Dashboard

### Completed Stages ✅

| Stage | Component              | Tests | Status      |
| ----- | ---------------------- | ----- | ----------- |
| 1     | Data Models & Adapters | 36    | ✅ COMPLETE |
| 2     | Orders & Ledger        | 55    | ✅ COMPLETE |
| 3     | Execution (Market/MOC) | 128   | ✅ COMPLETE |
| 4     | Execution (Limit/Stop) | 147   | ✅ COMPLETE |
| 5A    | Volume Participation   | 10    | ✅ COMPLETE |
| 5B    | Risk Management        | 54    | ✅ COMPLETE |
| 6A    | Indicators Framework   | 54    | ✅ COMPLETE |
| 6B    | Shorting & Accruals    | 46    | ✅ COMPLETE |
| 6B-X  | Long Dividends (Ext)   | +7    | ✅ COMPLETE |

**Total Tests:** 455 passing (+7), 10 skipped **Code Coverage:** 95%+ **Next:** Stage 7 (Backtest Runner)

### Architecture Highlights

- ✅ Vendor-agnostic Bar model (pure OHLCV)
- ✅ Adjustment events tracked separately (AdjustmentEvent)
- ✅ Signal-based risk management (portfolio-scoped)
- ✅ Decimal precision throughout (no float errors)
- ✅ Conservative fill model (no look-ahead bias)
- ✅ Comprehensive indicators (SMA, EMA, BB, RSI, ATR, MACD)
- ✅ Dividend processing for BOTH long and short positions
- ✅ Type-safe position handling (no circular dependencies)

______________________________________________________________________

## 🎯 Quick Navigation

- **Stage 6B Extension (Complete)** → [Section 4](#4-stage-6b-extension-long-position-dividends)
- **Future Stages** → [Section 5](#5-future-stages-7-8)
- **Project Structure** → [Appendix A](#appendix-a-project-structure)
- **Development Workflow** → [Appendix B](#appendix-b-development-workflow)

______________________________________________________________________

## 1. Completed Stages (1-6B)

### Stage 1: Data Models & Adapters ✅

**Duration:** 3 days | **Tests:** 36 passing

**Key Deliverables:**

- Canonical Bar model (vendor-agnostic OHLCV)
- AdjustmentEvent model (separate from Bar)
- Algoseek Parquet adapter (DuckDB-based)
- CSV adapter (for security master)
- Bar validator with OHLC policies
- Configuration schema (bar_schema, adjustment_schema)

**Architecture Decision:**

- Bar = Universal contract (works with ANY vendor)
- Adjustment metadata stored separately
- Adapters declare DataMode (adjusted/unadjusted/split_adjusted)

**Files:** 7 core files, 6 test files **Lines:** ~1,200 core + 800 tests

______________________________________________________________________

### Stage 2: Orders & Ledger ✅

**Duration:** 3 days | **Tests:** 55 passing

**Key Deliverables:**

- Order model (Market, MOC, Limit, Stop)
- Position tracker (open, add, reduce, close, flip)
- Cash ledger (Decimal precision, transaction history)
- Order state machine (SUBMITTED → FILLED/EXPIRED/CANCELED)
- Partial fill tracking

**Key Features:**

- Immutable order/position patterns
- Realized PnL on reduce/close
- Unrealized PnL tracked
- Margin support (negative balances allowed)

**Files:** 3 core files, 3 test files **Lines:** ~650 core + 760 tests

______________________________________________________________________

### Stage 3: Execution Engine (Market/MOC) ✅

**Duration:** 4 days | **Tests:** 128 passing

**Key Deliverables:**

- ExecutionEngine with event loop
- FillPolicy (conservative rules)
- CommissionCalculator (per-share + ticket minimum)
- Market orders fill at next bar open
- MOC orders fill at bar close with slippage
- Portfolio integration (atomic updates)

**Key Achievement:**

- Conservative fill model (no look-ahead bias)
- Decimal precision maintained
- Comprehensive logging (structlog)

**Files:** 5 core files, 9 test files **Lines:** ~900 core + 650 tests

______________________________________________________________________

### Stage 4: Execution Engine (Limit/Stop) ✅

**Duration:** 3 days | **Tests:** 147 passing (+19 from Stage 3)

**Key Deliverables:**

- Limit order evaluation (conservative touch rules)
- Stop order evaluation (conservative touch rules)
- Close-only bar handling (malformed OHLC)
- DAY TIF expiration (end of day)
- Stop slippage modeling

**Conservative Touch Rules:**

- Limit Buy: if `low ≤ limit` → fill at `min(limit, close)`
- Limit Sell: if `high ≥ limit` → fill at `max(limit, close)`
- Stop Buy: if `high ≥ stop` → fill at `max(stop, close)` + slippage
- Stop Sell: if `low ≤ stop` → fill at `min(stop, close)` - slippage

**Files:** Modified 2 core files, added 19 tests **Lines:** +300 core + 450 tests

______________________________________________________________________

### Stage 5A: Volume Participation ✅

**Duration:** 3 days | **Tests:** 10 passing

**Key Deliverables:**

- Participation cap calculation
- Partial fill tracking with residuals
- Residual queue management
- High participation guardrail (warns + clamps)

**Key Features:**

- Large orders split across multiple bars
- Residuals carried forward
- Queue expiration after N bars

**Files:** 2 core files, 2 test files **Lines:** ~300 core + 200 tests

______________________________________________________________________

### Stage 5B: Risk Management System ✅

**Duration:** 10 hours | **Tests:** 54 passing

**Architectural Change:**

- Strategies emit **Signals** (intent) not Orders
- RiskManager evaluates → produces sized Orders
- Portfolio-scoped (supports multiple strategies)

**Key Deliverables:**

- Signal model (SignalType, SignalDirection)
- RiskPolicy configuration
- RiskManager with evaluation logic
- 4 sizing methods:
  - FIXED_QUANTITY
  - FIXED_VALUE
  - PORTFOLIO_PERCENT (default)
  - RISK_PERCENT (with stop loss)
- Concentration limits (max_position_pct, max_positions)
- Leverage constraints (max_gross_exposure, max_net_exposure)
- Cash reserve enforcement

**Key Achievement:**

- Risk management in place BEFORE complex strategies
- Multi-strategy fair allocation
- Cash-first checking for fairness

**Files:** 4 core files, 6 test files, 1 example **Lines:** ~920 core + 1,630 tests

**Deferred to Phase 2:**

- VOLATILITY_TARGET sizing (requires ATR)
- KELLY_CRITERION sizing (requires P&L history)
- EQUAL_RISK_CONTRIBUTION (requires correlation matrix)
- Sector concentration limits
- Daily loss limits

______________________________________________________________________

### Stage 6A: Indicators Framework ✅

**Duration:** 14 hours | **Tests:** 54 passing

**Key Deliverables:**

- Base Indicator class (registration, caching)
- 6 Built-in indicators:
  - SMA (Simple Moving Average)
  - EMA (Exponential Moving Average)
  - BollingerBands (upper/lower/middle)
  - RSI (Relative Strength Index)
  - ATR (Average True Range)
  - MACD (Moving Average Convergence Divergence)
- 13 Helper functions (max, min, avg, std, crossover, etc.)
- IndicatorManager (caching per indicator+params)
- Context integration (`ctx.ind.sma(20)`)
- Warmup system:
  - Auto-detects max lookback period
  - Lifecycle: `on_init()` → warmup → `on_start()` → `on_bar()`
  - CLI support: `--warmup` and `--warmup-bars N`
  - Indicators always valid after warmup

**Key Achievement:**

- No None handling needed when warmup enabled
- Indicators work seamlessly in strategies
- Custom indicators easily registered

**Files:** 3 core files, 5 test files **Lines:** ~850 core + 620 tests

**Example Usage:**

```python
class SMACrossover(Strategy):
    def on_bar(self, bar: Bar, ctx: Context):
        fast = ctx.ind.sma(20)  # No None check needed with warmup
        slow = ctx.ind.sma(50)

        if ctx.crossover(fast, slow):
            ctx.submit_signal(Signal.long("SPY"))
```

______________________________________________________________________

### Stage 6B: Shorting & Accruals ✅

**Duration:** 8 hours | **Tests:** 46 passing | **Status:** COMPLETE

**Key Deliverables:**

- **DividendCalculator** - Calculate dividend per share from adjustment factors

  - Formula: `div = close_after * (cumulative_price_factor - 1)`
  - 20 unit tests

- **DividendProcessor** - Process dividend events during backtests

  - Event indexing by ex-date (O(1) lookup)
  - Filters cash dividends only
  - Processes SHORT positions exclusively
  - 17 unit tests

- **Backtest Integration** - Dividend processing in event loop

  - Optional `adjustment_events` parameter
  - Processes dividends once per timestamp
  - Duplicate prevention for same-timestamp bars
  - 5 integration tests

- **Integration Tests** - End-to-end dividend scenarios

  - Position timing vs ex-date validation
  - Multiple dividends over time (quarterly)
  - Non-cash events filtered (stock splits)
  - 4 comprehensive tests

**Architecture:**

```
Backtest Event Loop:
  1. Process bar (price updates)
  2. Generate signals (strategy)
  3. Create orders (risk manager)
  4. Execute orders (execution engine)
  5. Process dividends (ex-date) ← SHORT POSITIONS ONLY
  6. Apply borrow costs (end of day)
```

**Current Limitation:**

- Only SHORT positions pay dividends (cost/debit)
- LONG positions do NOT receive dividends (income/credit)
- **Result:** Total return calculations incomplete

**Files:** 2 core files, 4 test files **Lines:** ~460 core + 580 tests

**Commits:**

```
89581d2 feat(execution): Add dividend calculator with adjustment factors
8efeb55 feat(execution): Add dividend processor for ex-date handling
77f1131 feat(api): Integrate dividend processing into backtest
2c26768 test(integration): Add comprehensive end-to-end shorting tests
b5b1fca docs: Complete Stage 6B implementation plan with summary
```

______________________________________________________________________

## 4. Stage 6B Extension: Long Position Dividends

**Status:** ✅ COMPLETE (October 6, 2025) **Duration:** Completed in 1.5 hours **Objective:** Complete total return calculations by adding dividend income for LONG positions

### Implementation Summary

**Commit:** `0a748c5` - feat(execution): Add long position dividend receipts

**Changes Delivered:**

- ✅ Added `TransactionType.DIVIDEND_RECEIVED` for long dividend income
- ✅ Implemented `Portfolio.apply_long_dividend()` method
- ✅ Extended `DividendProcessor._process_single_event()` for symmetric handling
- ✅ Added 7 new tests (4 portfolio unit, 3 processor unit)
- ✅ Updated 3 existing tests for new behavior
- ✅ Fixed type annotations (removed unnecessary None checks)

**Test Results:**

- **Total Tests:** 455 passing (+7 new), 10 skipped
- **Coverage:** >95% for all modified files
- **Runtime:** 1.17 seconds (full suite)

**Architecture:**

```python
# DividendProcessor now symmetric
if position.qty < 0:
    # SHORT: Pay dividend (debit cash)
    portfolio.apply_short_dividend(...)  # TransactionType.DIVIDEND
elif position.qty > 0:
    # LONG: Receive dividend (credit cash)
    portfolio.apply_long_dividend(...)   # TransactionType.DIVIDEND_RECEIVED
```

**Files Modified:**

- `src/qtrader/models/portfolio.py` (+30 lines, 2 methods)
- `src/qtrader/execution/dividend_processor.py` (+20 lines, docstrings)
- `tests/unit/models/test_portfolio.py` (+130 lines, 4 tests)
- `tests/unit/execution/test_dividend_processor.py` (+160 lines, 3 tests)
- `tests/integration/test_backtest_dividends.py` (1 test updated)

### Why This Matters

**Total Return = Price Return + Dividend Return**

- S&P 500: ~50% of total returns from reinvested dividends (50+ year period)
- Current system now tracks both costs (shorts pay) AND income (longs receive)
- Enables fair comparison of dividend-paying vs growth stocks
- Complete symmetric dividend processing model

### Key Implementation Details

**Code Changes:**

```python
# Portfolio: Symmetric dividend handling
def apply_short_dividend(...):
    """Debit cash for short dividend (qty < 0)."""
    self.cash.debit(amount=..., transaction_type="DIVIDEND")

def apply_long_dividend(...):
    """Credit cash for long dividend (qty > 0)."""
    self.cash.credit(amount=..., transaction_type="DIVIDEND_RECEIVED")

# DividendProcessor: Process both directions
if position.qty < 0:
    portfolio.apply_short_dividend(...)
elif position.qty > 0:
    portfolio.apply_long_dividend(...)
```

**Type Safety Improvements:**

Fixed circular dependency by removing unnecessary `None` checks:

```python
# Before (type error):
position = self.positions.get_position(symbol)
if position and position.qty < 0:  # ❌ Position depends on itself

# After (type safe):
position = self.positions.get_position(symbol)
if position.qty < 0:  # ✅ get_position() never returns None
```

### Examples

**Example 1: Single Long Position**

```python
# Position: 200 shares MSFT @ $400
# Ex-date: 2024-08-14
# Dividend: $0.50/share

Result:
- Cash credited: 200 × $0.50 = $100.00
- Transaction type: DIVIDEND_RECEIVED
- Log: "Long dividend on MSFT: 200 shares @ $0.50/share"
```

**Example 2: Mixed Portfolio (Real Backtest)**

```python
# Long 100 AAPL @ $180 (dividend: $0.45/share)
# Short 50 MSFT @ $400 (dividend: $0.50/share)
# Same ex-date

Result:
- AAPL credit: +$45.00 (long receives)
- MSFT debit: -$25.00 (short pays)
- Net cash impact: +$20.00
- Both processed in single ex-date cycle
```

### Success Criteria (All Met ✅)

**Functional:**

- ✅ Long positions receive dividend credits on ex-date
- ✅ Short positions still pay dividends (no regression)
- ✅ Cash balance reflects both costs and income
- ✅ Closed/new positions receive no dividends

**Technical:**

- ✅ All 455 tests pass (448 existing + 7 new)
- ✅ No performance degradation (1.17s full suite)
- ✅ Code coverage >95% for all modified files
- ✅ Pre-commit hooks pass (ruff, isort, mdformat)

### After Completion

Stage 6B now provides:

- ✅ Complete dividend tracking (both costs and income)
- ✅ Accurate total return calculations
- ✅ Support for mixed long/short portfolios
- ✅ Type-safe position handling (no circular dependencies)
- ✅ Transaction-level audit trail (DIVIDEND vs DIVIDEND_RECEIVED)
- ✅ Maintain transaction-level transparency

______________________________________________________________________

## 5. Future Stages (7-8)

### Stage 6C: Instrument Abstraction & Data Source Resolver (In Progress)

**Duration:** 2-3 weeks | **Priority:** HIGH | **Status:** Planning Complete

**Objective:** Replace file path-based configuration with Instrument abstraction for better scalability and multi-source support.

**Key Components:**

1. **Instrument Model** (`src/qtrader/models/instrument.py`)

   - Instrument class (symbol, type, source, frequency, metadata)
   - InstrumentType enum (EQUITY, CRYPTO, FUTURE, FOREX, SIGNAL)
   - DataSource enum (ALGOSEEK, DATABASE, IQFEED, BINANCE, CSV_FILE, API)

1. **DataSourceResolver** (`src/qtrader/adapters/resolver.py`)

   - Load data_sources.yaml configuration
   - Map DataSource enum to adapter classes
   - Instantiate adapters with instrument context
   - Environment variable substitution (${VAR})

1. **Adapter Refactoring**

   - Update AlgoseekParquetAdapter to accept Instrument
   - Update CSVAdapter to accept Instrument
   - Remove hardcoded path logic from adapters
   - Symbol → SecId lookup via security master

1. **Configuration Changes**

   - Create data_sources.yaml (system-wide config)
   - Update strategy config pattern (use instruments list)
   - Remove data_paths and symbols from backtest_config
   - Add frequency per-instrument with global default

1. **CLI Updates**

   - Load data_sources.yaml at startup
   - Pass Instrument objects to backtest runner
   - Update \_load_data_files() helper
   - Update config extraction helpers

1. **Testing**

   - Update 1-2 integration tests to use Instrument pattern
   - Update example strategies
   - Test multi-source scenarios
   - Test environment variable substitution

**Implementation Plan:**

- ✅ Phase 0: Documentation updated (spec + implementation plan)
- ⏳ Phase 1: Create Instrument model and enums (2 days)
- ⏳ Phase 2: Create DataSourceResolver (2 days)
- ⏳ Phase 3: Refactor adapters (3 days)
- ⏳ Phase 4: Update backtest runner (2 days)
- ⏳ Phase 5: Update CLI and examples (2 days)
- ⏳ Phase 6: Testing and validation (3 days)

**No Backward Compatibility:** Project is pre-production; clean break from data_paths pattern.

**Files to Modify:**

- `src/qtrader/models/instrument.py` (NEW)
- `src/qtrader/adapters/resolver.py` (NEW)
- `src/qtrader/adapters/algoseek_parquet.py` (REFACTOR)
- `src/qtrader/adapters/csv_adapter.py` (REFACTOR)
- `src/qtrader/api/backtest.py` (UPDATE)
- `src/qtrader/cli.py` (UPDATE)
- `examples/sma_crossover_strategy.py` (UPDATE)
- `config/data_sources.yaml` (NEW)
- `tests/integration/test_backtest_full_execution.py` (UPDATE 1-2 tests)

### Stage 7: Public API & CLI (Planned)

**Duration:** 5 days | **Priority:** HIGH | **Status:** 90% Complete

**Note:** Stage 7 core implementation and CLI are complete. Remaining work includes optional debugging features after Stage 6C.

**Completed:**

- ✅ Strategy base class and protocol
- ✅ Context with order submission API
- ✅ Backtest runner (full run mode)
- ✅ CLI implementation (self-contained strategy pattern)
- ✅ Config extraction and override system

**Remaining (Optional):**

- ⏳ Interactive debugging with `Backtest.next_bar()`
- ⏳ Debug output files (indicators.csv, bars.csv)
- ⏳ YAML config file loading

______________________________________________________________________

### Stage 8: Golden Baselines & Validation (Planned)

**Duration:** 5 days | **Priority:** HIGH

**Key Components:**

- Buy-and-Hold strategy (AAPL, MSFT, AMZN)
- SMA Crossover strategy (MSFT)
- Golden file generator scripts
- Golden validation tests
- CI checks for determinism

**Process:**

1. Implement reference strategies
1. Debug bar-by-bar with `Backtest.next_bar()`
1. Verify with `--debug-output`
1. Review results together
1. Commit golden files
1. Automate validation in CI

______________________________________________________________________

## Appendix A: Project Structure

```
src/qtrader/
├── api/                       # Public API
│   ├── strategy.py           # Strategy protocol
│   ├── context.py            # Context for strategies
│   ├── backtest.py           # Backtest runner
│   └── indicators.py         # Indicator framework
├── models/                    # Core models
│   ├── bar.py               # Bar, AdjustmentEvent, enums
│   ├── order.py             # Order types
│   ├── position.py          # Position tracking
│   └── ledger.py            # Cash ledger
├── execution/                 # Execution engine
│   ├── engine.py            # Main engine
│   ├── fill_policy.py       # Fill rules
│   ├── dividend_calculator.py
│   └── dividend_processor.py
├── risk/                      # Risk management
│   ├── signal.py            # Signal model
│   ├── policy.py            # RiskPolicy
│   ├── manager.py           # RiskManager
│   └── sizing.py            # Sizing methods
├── adapters/                  # Data adapters (private)
│   ├── algoseek_parquet.py
│   └── csv_adapter.py
└── config/                    # Configuration
    ├── data_config.py
    └── engine_config.py

tests/
├── unit/                      # Unit tests by component
├── integration/               # Integration tests
└── goldens/                   # Golden baselines (future)
```

______________________________________________________________________

## Appendix B: Development Workflow

### Testing

```bash
# Run all tests
make test

# Run specific module
pytest tests/unit/execution/ -v

# With coverage
pytest --cov=qtrader --cov-report=html

# Fast (no coverage)
make test-fast
```

### Code Quality

```bash
# Format code
make format

# Lint
make lint

# Type check
make type-check

# All checks
make qa
```

### Git Workflow

1. Create branch: `git checkout -b feature/name`
1. Implement with TDD
1. Run `make qa`
1. Commit (pre-commit hooks auto-format)
1. PR review
1. Merge to master

______________________________________________________________________

## Appendix C: Dependencies

```toml
[project]
name = "qtrader"
version = "0.1.0"
requires-python = ">=3.13"

dependencies = [
    "duckdb>=1.4.0",        # Parquet reading
    "pandas>=2.3.2",        # Data manipulation
    "pyarrow>=21.0.0",      # Parquet support
    "click>=8.0.0",         # CLI
    "pydantic>=2.11.9",     # Config validation
    "pyyaml>=6.0",          # YAML loading
    "pytz>=2024.1",         # Timezones
    "structlog>=24.4.0",    # Logging
]

[project.scripts]
qtrader = "qtrader.cli:main"
```

______________________________________________________________________

**Document Version:** 3.1 (October 6, 2025) **Status:** Stage 6B Extension Complete - Ready for Stage 7
