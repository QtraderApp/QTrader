# QTrader Phase 1 — Implementation Plan

**Version:** 4.0 | **Date:** October 6, 2025 | **Status:** Stage 7 Complete - Production Ready

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
| 6B    | Shorting & Dividends   | 53    | ✅ COMPLETE |
| 6C    | Instrument Abstraction | 23    | ✅ COMPLETE |
| 7     | Backtest Runner & CLI  | 110   | ✅ COMPLETE |

**Total Tests:** 470 passing, 10 skipped\
**Code Coverage:** 96%\
**Status:** Production Ready\
**Next:** Stage 8 (Golden Baselines)

### Architecture Highlights

- ✅ Vendor-agnostic Bar model (pure OHLCV)
- ✅ Adjustment events tracked separately (AdjustmentEvent)
- ✅ Signal-based risk management (portfolio-scoped)
- ✅ Decimal precision throughout (no float errors)
- ✅ Conservative fill model (no look-ahead bias)
- ✅ Comprehensive indicators (SMA, EMA, BB, RSI, ATR, MACD)
- ✅ Symmetric dividend processing (long receives, short pays)
- ✅ Instrument abstraction (logical vs physical separation)
- ✅ Multi-source data support (Algoseek, CSV, extensible)
- ✅ Production-ready CLI (self-contained strategies)
- ✅ Complete event loop (warmup → signals → risk → execution → fills)

______________________________________________________________________

## 🎯 Quick Navigation

- **What's Next** → [Stage 8: Golden Baselines](#stage-8-golden-baselines-next)
- **Recent Stages** → [Stage 6C](#stage-6c-instrument-abstraction-complete) | [Stage 7](#stage-7-backtest-runner--cli-complete)
- **Project Structure** → [Appendix A](#appendix-a-project-structure)
- **Development Workflow** → [Appendix B](#appendix-b-development-workflow)

______________________________________________________________________

## Stages 1-5B: Foundation (Complete)

**Stages 1-5B** built the core foundation (detailed docs in previous versions):

- **Stage 1:** Data models (Bar, AdjustmentEvent) + adapters (Algoseek, CSV)
- **Stage 2:** Orders (Market, MOC, Limit, Stop) + Position + Ledger
- **Stage 3:** Execution engine + fill policies + commissions
- **Stage 4:** Limit/Stop order evaluation (conservative touch rules)
- **Stage 5A:** Volume participation + partial fills
- **Stage 5B:** Risk management (signals → sized orders) + 4 sizing methods

**Total:** 438 tests passing | **Coverage:** 95%+

______________________________________________________________________

## Stage 6A: Indicators Framework (Complete)

**Duration:** 14 hours | **Tests:** 54 passing | **Status:** COMPLETE

**Key Deliverables:**

- Base Indicator class with registration and caching
- 6 built-in indicators: SMA, EMA, BollingerBands, RSI, ATR, MACD
- 13 helper functions (max, min, avg, std, crossover, etc.)
- IndicatorManager with per-symbol caching
- Context integration (`ctx.ind.sma(20)`)
- Warmup system: auto-detects lookback, runs `on_init()` → warmup → `on_start()` → trading loop
- CLI support: `--warmup` and `--warmup-bars N`

**Key Achievement:** Indicators always valid after warmup (no None handling needed in strategies)

**Files:** 3 core files, 5 test files | **Lines:** ~850 core + 620 tests

______________________________________________________________________

## Stage 6B: Shorting & Dividends (Complete)

**Duration:** 10 hours | **Tests:** 53 passing | **Status:** COMPLETE

**Key Deliverables:**

- **DividendCalculator:** Calculates dividend/share from adjustment factors
  - Formula: `div = close_after * (cumulative_price_factor - 1)`
- **DividendProcessor:** Processes dividend events on ex-dates
  - Event indexing by ex-date (O(1) lookup)
  - Filters cash dividends only
  - **Symmetric handling:** SHORT pays (debit), LONG receives (credit)
- **Portfolio Integration:** Added transaction types
  - `TransactionType.DIVIDEND` (short pays)
  - `TransactionType.DIVIDEND_RECEIVED` (long receives)
- **Backtest Integration:** Optional `adjustment_events` parameter

**Architecture:** Dividends processed in event loop after fills, before borrow costs

**Total Return Support:** Complete tracking of both dividend costs (shorts) and income (longs)

**Files:** 2 core files, 4 test files | **Lines:** ~460 core + 580 tests

______________________________________________________________________

## Stage 6C: Instrument Abstraction (Complete)

**Duration:** 2 weeks | **Tests:** 23 passing | **Status:** COMPLETE (October 6, 2025)

**Objective:** Replace file path-based configuration with logical instrument specification

### What Changed

**Before (Path-based):**

```python
backtest_config = {
    "data_paths": ["/data/AAPL.parquet", "/data/MSFT.parquet"],
    "symbols": ["AAPL", "MSFT"]
}
```

**After (Instrument-based):**

```python
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

backtest_config = {
    "instruments": [
        Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK, "1d"),
        Instrument("MSFT", InstrumentType.EQUITY, DataSource.ALGOSEEK, "1d")
    ]
}
```

### Key Components

1. **Instrument Model** (`src/qtrader/models/instrument.py`)

   - `Instrument` NamedTuple: symbol, type, source, frequency, metadata
   - `InstrumentType` enum: EQUITY, CRYPTO, FUTURE, FOREX, SIGNAL
   - `DataSource` enum: ALGOSEEK, DATABASE, IQFEED, BINANCE, CSV_FILE, API

1. **DataSourceResolver** (`src/qtrader/adapters/resolver.py`)

   - Maps `DataSource` enum → adapter classes
   - Loads `data_sources.yaml` config (system-wide)
   - Environment variable substitution (`${VAR_NAME}`)
   - Config search: explicit → ./config → ~/.qtrader

1. **Adapter Refactoring** (BREAKING CHANGE)

   - **Before:** Stateless - `adapter.read_bars(path, config)`
   - **After:** Stateful - `adapter = Adapter(config, instrument); adapter.read_bars(config)`
   - Symbol resolution: Ticker → SecId (via security master)
   - Path construction from templates in config

1. **CLI Integration** (`src/qtrader/cli.py`)

   - Loads `data_sources.yaml` at startup
   - New function: `_load_data_from_instruments()`
   - Validates `instruments` in backtest config
   - Displays instrument details in verbose mode

### Benefits

- ✅ Separation of concerns: "what to trade" vs "where to get data"
- ✅ Multi-source support: Mix Algoseek + CSV + Database in single backtest
- ✅ Cleaner configuration: Instrument objects replace file paths
- ✅ Extensibility: Add new sources via config (no code changes)
- ✅ Environment flexibility: Dev/prod configs via env vars

**Files Modified:** 7 files | **Lines:** +1,473 insertions, -339 deletions | **Coverage:** 96%

**Commits:** 7 logically separated commits (feat, refactor, test, docs, chore)

**Detailed Documentation:** `docs/stage6c_completion_summary.md`

______________________________________________________________________

## Stage 7: Backtest Runner & CLI (Complete)

**Duration:** 5 days | **Tests:** 110 passing | **Status:** COMPLETE (October 6, 2025)

**Objective:** Production-ready backtest runner with complete event loop and CLI

### Complete Trading Loop

Implemented 5-phase event loop in `src/qtrader/api/backtest.py`:

1. **Initialization:** `strategy.on_init(ctx)` - Register custom indicators
1. **Warmup:** Build indicator state (don't call `on_bar()`)
1. **Start:** `strategy.on_start(ctx)` - After warmup completes
1. **Trading Loop (NEW - 150 lines):**
   - Update context state (date, symbol, price)
   - Add bar to history
   - Process dividends (if ex-date)
   - `strategy.on_bar()` → get signals
   - For each signal:
     - `ctx.evaluate_signal()` → RiskDecision
     - `ctx.signal_to_order()` → OrderBase
     - `execution_engine.submit_order()`
   - `execution_engine.on_bar()` → fills
   - `strategy.on_fill()` for each fill
   - Save indicator state for crossovers
   - Snapshot portfolio every 10 bars
1. **Finalization:** `strategy.on_end()`

**Key Achievement:** Full signal → risk → order → fill → portfolio flow with comprehensive metadata

### Production CLI

**Design Philosophy:** Self-contained strategy files (code + config in one place)

**Commands:**

```bash
# Main command
qtrader backtest --strategy examples/sma_crossover_strategy.py [options]

# Validation command (deprecated, use Instrument pattern)
qtrader validate-data --data <path> --symbols <list>
```

**Key Features:**

- ✅ Minimal required arguments (only `--strategy`)
- ✅ Auto-generated output directories with timestamps
- ✅ Configuration overrides via `--set KEY=VALUE`
- ✅ Debug mode: `--debug` (CSV exports for bars, indicators)
- ✅ Verbose logging: `--verbose`
- ✅ DataSourceResolver integration
- ✅ Instrument-based data loading

**Strategy File Pattern:**

```python
# examples/sma_crossover_strategy.py
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

# Strategy parameters
config = {"fast": 20, "slow": 50}

# Complete backtest config
backtest_config = {
    "instruments": [
        Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK, "1d")
    ],
    "initial_cash": 100_000,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
}

class SMACrossover(Strategy):
    def on_bar(self, bar, ctx):
        fast = ctx.ind.sma(config["fast"])
        slow = ctx.ind.sma(config["slow"])

        if ctx.crossover(fast, slow):
            ctx.submit_signal(Signal.long(bar.symbol))
        elif ctx.crossunder(fast, slow):
            ctx.submit_signal(Signal.close(bar.symbol))
```

**Files:** Complete rewrite of `src/qtrader/cli.py` (~785 lines)

**Detailed Documentation:** `docs/stage7_summary.md`, `docs/cli_usage.md`

______________________________________________________________________

## Stage 8: Golden Baselines (NEXT)

**Duration:** 5-7 days | **Priority:** HIGH | **Status:** Ready to start

### Objective

Establish deterministic golden baselines for regression testing and validation. Create reference strategies with known-good results that must pass in CI.

### Implementation Plan

**Phase 1: Reference Strategies (2 days)**

1. **Buy-and-Hold Strategy**

   - Symbols: AAPL, MSFT, AMZN
   - Period: 2023-01-01 to 2023-12-31
   - Initial allocation: Equal weight
   - Expected behavior: One-time purchases, hold to end
   - Validates: Basic execution, portfolio tracking, dividends

1. **SMA Crossover Strategy**

   - Symbol: MSFT
   - Parameters: 20/50 SMA
   - Period: 2023-01-01 to 2023-12-31
   - Expected behavior: 15-25 round trips
   - Validates: Indicators, signals, risk management, multiple fills

**Phase 2: Golden File Generation (1 day)**

Create `scripts/generate_goldens.py` to generate golden files:

```python
def generate_golden(strategy_name, strategy_class, config):
    """Run backtest, save results to tests/goldens/"""
    result = Backtest.run(strategy_class, config)

    golden = {
        "metadata": {
            "strategy": strategy_name,
            "generated": datetime.now().isoformat(),
            "version": "1.0"
        },
        "config": config,
        "results": {
            "final_cash": float(result.final_cash),
            "final_equity": float(result.final_equity),
            "total_return": float(result.total_return_pct),
            "num_trades": result.num_trades,
            "num_fills": result.num_fills,
            "commissions_paid": float(result.total_commissions),
        },
        "final_positions": result.final_positions,
        "key_snapshots": result.snapshots[::len(result.snapshots)//10],
    }

    with open(f"tests/goldens/{strategy_name}_golden.json", "w") as f:
        json.dump(golden, f, indent=2)
```

**Phase 3: Validation Tests (1 day)**

```python
# tests/integration/goldens/test_buy_and_hold.py
def test_buy_and_hold_matches_golden():
    """Verify buy-and-hold results match golden file."""
    golden = load_golden("buy_and_hold")
    result = Backtest.run(BuyAndHold, golden["config"])

    assert_close(result.final_cash, golden["results"]["final_cash"])
    assert_close(result.final_equity, golden["results"]["final_equity"])
    assert result.num_trades == golden["results"]["num_trades"]
```

**Phase 4: Debug Bar-by-Bar (1-2 days)**

Add interactive debugging capability:

```python
# For troubleshooting golden mismatches
backtest = Backtest(strategy, config)
backtest.init()

while backtest.has_next():
    bar = backtest.next_bar()  # Step one bar
    print(f"Date: {bar.ts}, Portfolio: {backtest.portfolio.equity}")
    # Inspect state after each bar
```

**Phase 5: CI Integration (1 day)**

```yaml
# .github/workflows/test.yml
- name: Validate Golden Baselines
  run: pytest tests/integration/goldens/ -v --strict-markers
```

### Success Criteria

- ✅ 2 reference strategies implemented
- ✅ Golden files generated and committed
- ✅ Validation tests pass (exact match on key metrics)
- ✅ CI enforces golden validation on all PRs
- ✅ Documentation for adding new goldens

### Benefits

- **Regression Detection:** Catch unintended behavior changes
- **Determinism Proof:** Same inputs = same outputs every time
- **Onboarding:** Clear examples of correct usage
- **Debugging:** Known-good baselines for troubleshooting
- **Confidence:** Production readiness validation

### Files to Create

```
tests/integration/goldens/
  __init__.py
  test_buy_and_hold.py          (new)
  test_sma_crossover.py         (new)
  buy_and_hold_golden.json      (new)
  sma_crossover_golden.json     (new)

scripts/
  generate_goldens.py           (new)

examples/
  buy_and_hold_strategy.py      (new)
```

### Estimated Effort

- Reference strategies: 2 days
- Golden generation: 1 day
- Validation tests: 1 day
- Debug bar-by-bar: 1-2 days
- CI integration: 1 day
- **Total: 5-7 days**

______________________________________________________________________

## Future Stages (Phase 2)

After Stage 8, these are candidates for Phase 2:

### Performance & Optimization

- Parallel backtest execution (multiple strategies)
- Efficient indicator caching strategies
- Memory-mapped data loading for large datasets

### Advanced Features

- Multi-currency support (FX conversion)
- Futures and options support
- Intraday backtesting (minute/tick data)
- Portfolio rebalancing utilities
- Transaction cost analysis (TCA)

### Analytics & Reporting

- Tearsheet generation (Sharpe, Sortino, drawdown)
- Attribution analysis (performance decomposition)
- Risk metrics (VaR, CVaR, beta)
- HTML/PDF report generation
- Interactive web dashboard

### Risk Management Enhancements

- VOLATILITY_TARGET sizing
- KELLY_CRITERION sizing
- EQUAL_RISK_CONTRIBUTION
- Sector concentration limits
- Daily loss limits
- Correlation-based diversification

### Live Trading (Future)

- Broker integration (IBKR, Alpaca)
- Order management system (OMS)
- Real-time data feeds
- Position reconciliation
- Trade blotter

______________________________________________________________________

## Appendix A: Project Structure

```
src/qtrader/
├── api/                          # Public API
│   ├── strategy.py              # Strategy base class
│   ├── context.py               # Context for strategies
│   └── backtest.py              # Backtest runner ✅ Complete
├── models/                       # Core models
│   ├── bar.py                   # Bar, AdjustmentEvent
│   ├── instrument.py            # Instrument ✅ New in 6C
│   ├── order.py                 # Order types
│   ├── position.py              # Position tracking
│   ├── portfolio.py             # Portfolio with dividends
│   └── ledger.py                # Cash ledger
├── execution/                    # Execution engine
│   ├── engine.py                # Main engine
│   ├── fill_policy.py           # Fill rules
│   ├── commission.py            # Commission calc
│   ├── dividend_calculator.py   # Dividend math
│   └── dividend_processor.py    # Dividend events
├── risk/                         # Risk management
│   ├── signal.py                # Signal model
│   ├── policy.py                # RiskPolicy
│   ├── manager.py               # RiskManager
│   └── sizing.py                # Sizing methods
├── indicators/                   # Indicators framework
│   ├── base.py                  # Base Indicator
│   ├── manager.py               # IndicatorManager
│   ├── helpers.py               # Helper functions
│   ├── momentum/                # RSI, MACD
│   ├── trend/                   # SMA, EMA
│   └── volatility/              # BB, ATR
├── adapters/                     # Data adapters
│   ├── base.py                  # DataAdapter protocol
│   ├── resolver.py              # DataSourceResolver ✅ New in 6C
│   ├── algoseek_parquet.py      # Algoseek adapter (refactored)
│   └── csv_adapter.py           # CSV adapter (refactored)
├── config/                       # Configuration
│   ├── data_config.py
│   └── logging_config.py
├── validation/                   # Data validation
│   └── bar_validator.py
└── cli.py                        # CLI ✅ Complete rewrite in 7

config/
└── data_sources.yaml             # System-wide config ✅ New in 6C

tests/
├── unit/                         # 300+ unit tests
├── integration/                  # 160+ integration tests
└── goldens/                      # Golden baselines (Stage 8)

examples/
├── sma_crossover_strategy.py     # Updated for 6C/7
└── buy_and_hold_strategy.py      # Stage 8 (todo)
```

______________________________________________________________________

## Appendix B: Development Workflow

### Testing

```bash
# Run all tests
make test

# Run specific suite
pytest tests/unit/execution/ -v
pytest tests/integration/ -v

# With coverage
pytest --cov=qtrader --cov-report=html

# Fast (no coverage)
make test-fast

# Single test
pytest tests/unit/models/test_portfolio.py::test_apply_long_dividend -v
```

### Code Quality

```bash
# Format code (CAUTION: mdformat may corrupt markdown files)
make format

# Lint
make lint

# Type check
make type-check

# All checks (format + lint + type + test)
make qa
```

### Git Workflow

1. Create branch: `git checkout -b feature/name`
1. Implement with TDD
1. Run `make qa` (must pass)
1. Commit (pre-commit hooks auto-format)
1. Push and create PR
1. Merge to master after review

### Running Backtests

```bash
# Simple example
qtrader backtest --strategy examples/sma_crossover_strategy.py

# With overrides
qtrader backtest --strategy examples/sma_crossover_strategy.py \
  --set initial_cash=50000 \
  --set start_date=2024-01-01

# Debug mode
qtrader backtest --strategy examples/sma_crossover_strategy.py \
  --debug --verbose

# Custom output directory
qtrader backtest --strategy examples/sma_crossover_strategy.py \
  --out ./my_results/
```

______________________________________________________________________

**Document Version:** 4.0 (October 6, 2025) | **Status:** Stage 7 Complete - Ready for Stage 8
