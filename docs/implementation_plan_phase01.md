# QTrader Phase 1 — Implementation Plan

**Version:** 3.0 **Date:** October 6, 2025 **Status:** Stage 6B Complete (with Long Dividend Extension Ready) **Reference:** `docs/specs/phase01.md` v1.0

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

**Total Tests:** 530 passing, 10 skipped **Code Coverage:** 94% **Next:** Stage 6B Extension (Long Dividends) - 2-3 hours OR Stage 7

### Architecture Highlights

- ✅ Vendor-agnostic Bar model (pure OHLCV)
- ✅ Adjustment events tracked separately (AdjustmentEvent)
- ✅ Signal-based risk management (portfolio-scoped)
- ✅ Decimal precision throughout (no float errors)
- ✅ Conservative fill model (no look-ahead bias)
- ✅ Comprehensive indicators (SMA, EMA, BB, RSI, ATR, MACD)
- ✅ Dividend processing for SHORT positions
- 🟡 Long position dividends (READY to implement)

______________________________________________________________________

## 🎯 Quick Navigation

- **Stage 6B Extension** → [Section 4](#stage-6b-extension-long-position-dividends)
- **Future Stages** → [Section 5](#future-stages-7-8)
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

**Status:** 🟡 READY TO IMPLEMENT **Duration:** 2-3 hours **Objective:** Complete total return calculations by adding dividend income for LONG positions

### Why This Matters

**Total Return = Price Return + Dividend Return**

- S&P 500: ~50% of total returns from reinvested dividends (50+ year period)
- Current system only tracks costs (shorts pay) but not income (longs receive)
- Cannot fairly compare dividend-paying vs growth stocks
- Asymmetric model is incomplete

### What's Ready

**Existing Infrastructure:** ✅

- `CashLedger.credit()` exists and tested
- `DividendCalculator` works for both directions
- `DividendProcessor` already indexes events
- Event processing framework complete

**Required Changes:** 🆕

- 1 new enum value: `TransactionType.DIVIDEND_RECEIVED`
- 1 new method: `Portfolio.apply_long_dividend()`
- 1 conditional branch: `elif position.qty > 0` in processor
- 12 new tests (4 portfolio, 3 processor, 5 integration)

### Implementation Plan

**Phase 1: Core Implementation (1 hour)**

1. **Add DIVIDEND_RECEIVED enum** (5 min)

   ```python
   # src/qtrader/models/ledger.py
   class TransactionType(Enum):
       DIVIDEND = "dividend"              # SHORT positions (cost)
       DIVIDEND_RECEIVED = "div_received" # NEW: LONG positions (income)
   ```

1. **Implement Portfolio.apply_long_dividend()** (20 min)

   ```python
   # src/qtrader/models/portfolio.py
   def apply_long_dividend(
       self, symbol: str, dividend_per_share: Decimal, timestamp: datetime
   ) -> None:
       """Apply dividend receipt for LONG position (credits cash)."""
       position = self.positions.get(symbol)
       if not position or position.qty <= 0:
           raise ValueError("No long position")

       total = abs(position.qty) * dividend_per_share
       self.cash.credit(
           amount=total,
           timestamp=timestamp,
           type=TransactionType.DIVIDEND_RECEIVED,
           description=f"{symbol} dividend: {position.qty} @ ${dividend_per_share}"
       )
   ```

1. **Extend DividendProcessor** (15 min)

   ```python
   # src/qtrader/execution/dividend_processor.py
   def _calculate_dividend(...):
       dividend_per_share = DividendCalculator.calculate_from_factors(...)

       if position.qty < 0:
           # SHORT: Pay dividend (debit)
           self.portfolio.apply_short_dividend(...)
       elif position.qty > 0:
           # LONG: Receive dividend (credit) ← NEW
           self.portfolio.apply_long_dividend(...)
   ```

1. **Verify** (20 min)

   - Syntax checks
   - Existing tests still pass

**Phase 2: Unit Tests (1 hour)**

- Portfolio tests (4):

  - Credits cash correctly
  - Requires long position
  - Rejects short position
  - Handles partial positions

- Processor tests (3):

  - Calculates long dividend
  - Handles mixed portfolio (long + short)
  - Logs correctly

**Phase 3: Integration Tests (30 min)**

- End-to-end scenarios (5):
  - Long receives dividend on ex-date
  - Position closed before ex-date (no dividend)
  - Position opened after ex-date (no dividend)
  - Multiple quarterly dividends
  - Mixed long/short portfolio (both processed)

**Phase 4: Documentation (30 min)**

- Update STAGE_6B plan with completion
- Add examples to docstrings

### Examples

**Example 1: Single Long Position**

```python
# Position: 200 shares MSFT @ $400
# Ex-date: 2024-08-14
# Dividend: $0.50/share

Result:
- Cash credited: 200 × $0.50 = $100.00
- Transaction type: DIVIDEND_RECEIVED
- Description: "MSFT dividend: 200 shares @ $0.50/share"
```

**Example 2: Mixed Portfolio**

```python
# Long 100 AAPL @ $180 (dividend: $0.45/share)
# Short 50 MSFT @ $400 (dividend: $0.50/share)
# Same ex-date

Result:
- AAPL credit: +$45.00 (long receives)
- MSFT debit: -$25.00 (short pays)
- Net cash: +$20.00
```

### Success Criteria

**Functional:**

- [ ] Long positions receive dividend credits on ex-date
- [ ] Short positions still pay dividends (no regression)
- [ ] Cash balance reflects both costs and income
- [ ] Closed/new positions receive no dividends

**Technical:**

- [ ] All 542+ tests pass (530 existing + 12 new)
- [ ] No performance degradation
- [ ] Code coverage > 95% for new code
- [ ] Pre-commit hooks pass

**Commit:**

```
feat(execution): Add long position dividend receipts

Complete total return calculation by adding dividend income tracking
for long positions, symmetrically with existing short dividend costs.

Changes:
- Add TransactionType.DIVIDEND_RECEIVED for long dividend income
- Implement Portfolio.apply_long_dividend() (mirrors apply_short_dividend)
- Extend DividendProcessor._calculate_dividend() to handle qty > 0
- Add 12 comprehensive tests (4 unit portfolio, 3 unit processor, 5 integration)

Tests: 542 passed (+12 new), 10 skipped
Coverage: 95%+ for all modified files

Closes Stage 6B Extension
```

### After Completion

Stage 6B will:

- ✅ Track dividend costs (shorts pay)
- ✅ Track dividend income (longs receive)
- ✅ Enable accurate total return calculations
- ✅ Support mixed long/short portfolios
- ✅ Maintain transaction-level transparency

______________________________________________________________________

## 5. Future Stages (7-8)

### Stage 7: Public API & CLI (Planned)

**Duration:** 5 days | **Priority:** HIGH

**Key Components:**

- Strategy base class and protocol
- Context with order submission API
- Context debug API (debug_state, debug_orders, debug_fills)
- Backtest runner (full run + step-by-step mode)
- CLI implementation
- Config file loading (YAML)

**Debugging Features:**

- Standard Python debugging (pdb, VS Code, PyCharm)
- Interactive backtesting with `Backtest.next_bar()`
- Debug output files (bars.csv, indicators.csv, portfolio_snapshots.csv)
- Structured logging with levels

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

**Document Version:** 3.0 (October 6, 2025) **Status:** Ready for Stage 6B Extension OR Stage 7 implementation
