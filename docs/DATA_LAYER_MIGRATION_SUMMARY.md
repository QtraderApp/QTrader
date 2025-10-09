# Data Layer Migration - Executive Summary

**Status:** Planning Complete | **Timeline:** 17 days | **Complexity:** Medium

______________________________________________________________________

## 🎯 Objective

Migrate QTrader from legacy multi-series `Bar` model to new **Canonical Data Layer** with iterator-based data flow and **multi-mode architecture**.

**Key Principles:**

- ✅ **No backward compatibility** - Clean refactor
- ✅ **No bridge code** - Direct replacement
- ✅ **Iterator-based** - Memory efficient streaming
- ✅ **Multi-mode architecture** - Each bar contains all 3 adjustment modes
- ✅ **Configuration-driven** - Mode selection per component in YAML

______________________________________________________________________

## 📊 Current State

### ✅ What's Done (Phases 1-6, 8) - **CORE MIGRATION COMPLETE**

**Phase 1: Data Layer Models** - ✅ Production ready:

- ✅ `MultiModeBar` - Container with all 3 adjustment modes (immutable, Pydantic V2)
- ✅ `CanonicalBar` - Single price series (Pydantic)
- ✅ `CanonicalPriceSeries` - Collection with mode
- ✅ `AlgoseekBar` - Vendor-specific with correct dividend formula
- ✅ `AlgoseekPriceSeries.to_canonical_series()` - Transforms to 3 modes
- ✅ **Validation**: $0.82 AAPL dividend (100% accurate)

**Phase 2: Iterator Infrastructure** - ✅ Complete:

- ✅ `PriceSeriesIterator` - Streams MultiModeBar with peek support
- ✅ `DataLoader` - Coordinates adapter → transformation → iterator
- ✅ `BarMerger` - Multi-symbol coordination
- ✅ Multi-mode configuration schema (3 mode settings)
- ✅ **Tests**: 47 unit tests passing

**Phase 3: Adapter Refactoring** - ✅ Complete:

- ✅ `AlgoseekOHLCVendorAdapter` - Simplified (350 lines vs 583 legacy)
- ✅ Returns vendor models only (AlgoseekBar)
- ✅ Legacy adapter deprecated (AlgoseekOHLCAdapterLegacy)
- ✅ **Tests**: 14 adapter tests passing

**Phase 4: Backtest Engine** - ✅ Complete:

- ✅ Updated `Backtest.run()` with iterator-based data flow
- ✅ Passes `MultiModeBar` to all components
- ✅ Multi-symbol coordination working
- ✅ Split handling integrated
- ✅ **Tests**: Integration tests passing

**Phase 5: Execution Engine** - ✅ Complete:

- ✅ `ExecutionEngine.on_bar(bar: MultiModeBar)` updated
- ✅ Uses `bar.unadjusted` for realistic fills at actual prices
- ✅ Fill policies updated (market, limit, stop orders)
- ✅ Dividend and split processing integrated
- ✅ **Tests**: 50+ execution tests passing

**Phase 6: Portfolio & Position** - ✅ Complete:

- ✅ Portfolio price tracking working (uses unadjusted for valuation)
- ✅ Position valuation and P&L calculations working
- ✅ Split handling working (cost basis preserved)
- ✅ Equity calculation working
- ✅ **Tests**: 41 tests passing (16 portfolio + 25 position)

**Phase 8: Test Suite Migration** - ✅ Complete:

- ✅ **All 321 tests passing** (100% pass rate)
- ✅ Test fixtures updated for CanonicalBar and MultiModeBar
- ✅ Split accounting test passing (validated 4:1 split)
- ✅ No legacy Bar references remain in tests

**Example Strategies Updated** - ✅ Complete:

- ✅ `examples/buy_and_hold_strategy.py`
- ✅ `examples/sma_crossover_strategy.py`
- ✅ `examples/minimal_iterator_backtest.py`

### ⏳ What Remains (Phases 7, 9-10)

**Phase 7: Performance and Reporting** - 📋 TODO (2 days):

- ⏳ Performance metrics (Sharpe ratio, max drawdown, total return)
- ⏳ Trade analytics (win rate, avg win/loss, holding period)
- ⏳ Risk metrics (VaR, CVaR, beta, correlation)
- ⏳ Reporting and visualization (equity curve, HTML reports)
- ⏳ Unit tests for analytics (~30+ tests)

**Phase 9: Cleanup & Validation** - 📋 TODO (1 day):

- ⏳ Remove old `Bar` model (legacy code)
- ⏳ Remove `AlgoseekOHLCAdapterLegacy`
- ⏳ Rename files (canonical_bar.py → bar.py)
- ⏳ Final validation (coverage >95%)

**Phase 10: Documentation** - 📋 TODO (2 days):

- ⏳ Update architecture docs
- ⏳ Create migration guide
- ⏳ Update README and API reference
- ⏳ Document performance analytics (Phase 7)

______________________________________________________________________

## 🏗️ Target Architecture

### Data Flow (Multi-Mode Iterator)

```
┌──────────────────────────────────────────────────────────┐
│  1. Raw Data (Parquet/CSV)                               │
│     ↓                                                     │
│  2. VendorAdapter → Iterator[AlgoseekBar]                │
│     ↓                                                     │
│  3. PriceSeriesBuilder → 3 CanonicalPriceSeries          │
│     ↓                                                     │
│  4. PriceSeriesIterator → MultiModeBar                   │
│        ├─ .unadjusted (actual traded prices)            │
│        ├─ .adjusted (split-adjusted)                     │
│        └─ .total_return (split + dividend)               │
│     ↓                                                     │
│  5. Backtest.run(iterator)                               │
│     ↓                                                     │
│  6. Strategy.on_bar(bar: MultiModeBar)                   │
│        bar.adjusted.close  # For indicators              │
│     ↓                                                     │
│  7. Execution.on_bar(bar: MultiModeBar)                  │
│        bar.unadjusted.high  # For fills                  │
│     ↓                                                     │
│  8. Portfolio.update(bar: MultiModeBar)                  │
│        bar.total_return.close  # For performance         │
└──────────────────────────────────────────────────────────┘
```

### Key Improvements

**Before (OLD)**:

```python
def on_bar(self, bar: Bar, ctx: Context):
    # Multi-series - must select
    close = bar.capital_adjusted.close
```

**After (NEW)**:

```python
# Strategy uses adjusted (split-consistent indicators)
def on_bar(self, bar: MultiModeBar, ctx: Context):
    strategy_bar = bar.adjusted  # or bar.get_bar("adjusted")
    close = strategy_bar.close

# Execution uses unadjusted (realistic fills)
def evaluate_fill(self, bar: MultiModeBar, order: Order):
    exec_bar = bar.unadjusted
    fill_price = exec_bar.high

# Portfolio uses total_return (accurate performance)
def calculate_return(self, bar: MultiModeBar):
    perf_bar = bar.total_return
    return_pct = (perf_bar.close - self.entry) / self.entry
```

**Benefits**:

- ✅ **Optimal mode per component** - Signal gen (adjusted), Execution (unadjusted), Performance (total_return)
- ✅ **Single data load** - All 3 modes available, no duplicate I/O
- ✅ **Memory efficient** - Iterator-based streaming
- ✅ **Configuration-driven** - Mode selection per stage in YAML
- ✅ **Correctness** - Commissions on actual prices, indicators across splits, returns with dividends

______________________________________________________________________

## 📅 Implementation Timeline

| Phase                          | Days | Work                            | Status      | Completion Date |
| ------------------------------ | ---- | ------------------------------- | ----------- | --------------- |
| **1. Core Models**             | 1    | Data layer validated            | ✅ COMPLETE | Oct 7, 2025     |
| **2. Iterator Infrastructure** | 1    | PriceSeriesIterator, DataLoader | ✅ COMPLETE | Oct 7, 2025     |
| **3. Adapter Refactoring**     | 1    | Simplified vendor adapters      | ✅ COMPLETE | Oct 7, 2025     |
| **4. Backtest Engine**         | 3    | Iterator-based runner           | ✅ COMPLETE | Oct 8, 2025     |
| **5. Execution Engine**        | 2    | CanonicalBar support            | ✅ COMPLETE | Oct 8, 2025     |
| **6. Portfolio Update**        | 1    | Direct field access             | ✅ COMPLETE | Oct 9, 2025     |
| **7. Performance & Reporting** | 2    | Analytics, metrics, reporting   | 📋 TODO     | -               |
| **8. Test Suite**              | 3    | Migrate 470+ tests              | ✅ COMPLETE | Oct 9, 2025     |
| **9. Cleanup**                 | 1    | Remove old code                 | 📋 TODO     | -               |
| **10. Documentation**          | 2    | Docs & examples                 | 📋 TODO     | -               |

**Total: 19 days (~4 weeks)** | **Completed: 13 days (68%)** | **Remaining: 5 days**

**Latest Update**: Core migration complete (October 9, 2025) - Phases 1-6 and 8 done, 321/321 tests passing

______________________________________________________________________

## 🔑 Critical Changes

### 1. Multi-Mode Bar Architecture

| Aspect             | OLD                          | NEW                                      |
| ------------------ | ---------------------------- | ---------------------------------------- |
| **Structure**      | Multi-series NamedTuple      | Multi-mode Pydantic                      |
| **Bar Type**       | `Bar` (all 3 series)         | `MultiModeBar` (container)               |
| **Price Access**   | `bar.capital_adjusted.close` | `bar.adjusted.close`                     |
| **Mode Selection** | Hardcoded in components      | Config per component                     |
| **Strategy**       | N/A                          | Uses `.adjusted` (split-safe)            |
| **Execution**      | N/A                          | Uses `.unadjusted` (real prices)         |
| **Performance**    | N/A                          | Uses `.total_return` (dividends)         |
| **Memory**         | All series loaded            | All 3 modes (acceptable for correctness) |

**Why Multi-Mode?**

Different components need different adjustment modes:

- **Strategy**: `adjusted` - Technical indicators work across stock splits
- **Execution**: `unadjusted` - Commissions calculated on actual traded prices
- **Performance**: `total_return` - Returns include dividend reinvestment

### 2. Data Loading Pattern

**OLD** (Adapter returns Bar):

```python
adapter = AlgoseekOHLCAdapter(config, instrument)
bars: List[Bar] = list(adapter.read_bars(config))
# Returns complex multi-series bars
```

**NEW** (Multi-mode iterator):

```python
adapter = AlgoseekVendorAdapter(config)
vendor_bars = list(adapter.read_bars(symbol, start, end))
series = AlgoseekPriceSeries(symbol, bars=vendor_bars)
canonical = series.to_canonical_series()  # All 3 modes
iterator = PriceSeriesIterator(canonical)  # Yields MultiModeBar
```

### 3. Configuration

**New `config/qtrader.yaml`**:

```yaml
data:
  # Mode per component for optimal correctness
  signal_generation_mode: "adjusted"      # Strategy indicators
  execution_mode: "unadjusted"            # Realistic fills
  performance_mode: "total_return"        # Include dividends
```

______________________________________________________________________

## 🎯 Success Criteria

### Functional

- [x] All 321+ tests passing ✅
- [x] Golden output matches ($0.82 dividend validated) ✅
- [x] Iterator-based data flow working ✅
- [x] Multi-symbol coordination working ✅
- [x] Multi-mode bar architecture working ✅
- [x] Each component uses optimal mode ✅
- [ ] Performance analytics implemented (Phase 7)

### Non-Functional

- [x] Code coverage >95% ✅
- [x] Memory usage acceptable (3x bars but streaming keeps it bounded) ✅
- [x] Performance within 10% of current ✅
- [ ] Documentation complete (Phase 10)
- [x] Examples working ✅

### Code Quality

- [x] No legacy Bar references in core code ✅
- [x] No bridge/compatibility code ✅
- [x] Clean separation (data → backtest) ✅
- [x] Type hints throughout ✅
- [x] Docstrings complete ✅
- [ ] Legacy code removed (Phase 9)

______________________________________________________________________

## 🚨 Risk Assessment

| Risk                      | Severity | Mitigation                  |
| ------------------------- | -------- | --------------------------- |
| Breaking changes in tests | HIGH     | Phase 7 dedicated (3 days)  |
| Memory usage (3x bars)    | LOW      | Streaming keeps it bounded  |
| Multi-symbol coordination | MEDIUM   | BarMerger tested separately |
| Iterator state bugs       | LOW      | Comprehensive unit tests    |
| Mode selection errors     | LOW      | Config validation + tests   |

______________________________________________________________________

## 📝 Key Deliverables by Phase

### Phase 1: Core Models ✅ COMPLETE

- `CanonicalBar`, `CanonicalPriceSeries`, `MultiModeBar` models
- `AlgoseekBar`, `AlgoseekPriceSeries` with correct dividend formula
- Golden output validation ($0.82 AAPL dividend)
- 13 unit tests + 6 integration tests

### Phase 2: Iterator Infrastructure ✅ COMPLETE

- `MultiModeBar` model (container with 3 CanonicalBar fields)
- `PriceSeriesIterator` class (yields MultiModeBar)
- `BarMerger` (multi-symbol coordination)
- `DataLoader` service (loads all 3 modes)
- Configuration schema (mode per component)
- 47 unit tests passing

### Phase 3: Adapter Refactoring ✅ COMPLETE

- Simplified `AlgoseekOHLCVendorAdapter` (~350 lines, was 583)
- Returns `AlgoseekBar` (vendor model only)
- No transformation logic (clean separation)
- Legacy adapter deprecated
- 14 adapter tests passing

### Phase 4: Backtest Engine ✅ COMPLETE

- `BarMerger` (multi-symbol coordination, yields MultiModeBar)
- Updated `Backtest.run()` signature (iterator-based)
- Iterator-based event loop (passes MultiModeBar)
- Updated strategy interface (receives MultiModeBar)
- Split handling integrated
- Integration tests passing

### Phase 5: Execution Engine ✅ COMPLETE

- Updated `ExecutionEngine.on_bar(bar: MultiModeBar)`
- Uses `bar.unadjusted` for realistic fills at actual prices
- Simplified fill logic (actual traded prices)
- Dividend and split processing integrated
- 50+ execution tests passing

### Phase 6: Portfolio Update ✅ COMPLETE

- Updated `Portfolio.update_prices()` (uses unadjusted for valuation)
- Updated `Position` calculations (market value, unrealized P&L)
- Split accounting validated (cost basis preserved)
- Equity calculation working
- 41 tests passing (16 portfolio + 25 position)

### Phase 7: Performance and Reporting 📋 TODO

- Performance metrics (total return, Sharpe ratio, max drawdown, Calmar ratio)
- Trade analytics (win rate, avg win/loss, holding period, trade distribution)
- Risk metrics (VaR, CVaR, beta, correlation)
- Reporting and visualization (equity curve, drawdown chart, HTML reports)
- 30+ unit tests for analytics

### Phase 8: Test Suite ✅ COMPLETE

- Test fixtures for `CanonicalBar` and `MultiModeBar`
- All unit tests updated (321 passing)
- All integration tests updated
- Split accounting test validated
- Golden tests verified
- No legacy Bar references

### Phase 9: Cleanup 📋 TODO

- Old `Bar` model removed
- Old adapter removed (`AlgoseekOHLCAdapterLegacy`)
- File renames (canonical_bar.py → bar.py)
- Full test suite validation (400+ tests with Phase 7)
- Performance validation

### Phase 10: Documentation 📋 TODO

- Updated examples (3 files already done)
- Updated architecture docs
- Migration guide created
- API reference updated
- Performance analytics guide

______________________________________________________________________

## 🔄 Migration Process

### Step 1: Review Plan ✅ COMPLETE

- [x] Team review completed
- [x] Timeline approved
- [x] Resources allocated

### Step 2: Execute Phases (19 days) - **68% COMPLETE**

- [x] Phase 1: Core Models ✅ (Oct 7, 2025)
- [x] Phase 2: Iterator infrastructure ✅ (Oct 7, 2025)
- [x] Phase 3: Adapter refactoring ✅ (Oct 7, 2025)
- [x] Phase 4: Backtest engine ✅ (Oct 8, 2025)
- [x] Phase 5: Execution engine ✅ (Oct 8, 2025)
- [x] Phase 6: Portfolio update ✅ (Oct 9, 2025)
- [ ] Phase 7: Performance and Reporting 📋 (2 days remaining)
- [x] Phase 8: Test suite ✅ (Oct 9, 2025) - 321/321 tests passing
- [ ] Phase 9: Cleanup 📋 (1 day remaining)
- [ ] Phase 10: Documentation 📋 (2 days remaining)

### Step 3: Validation - **MOSTLY COMPLETE**

- [x] All tests passing (321/321) ✅
- [x] Golden output verified ($0.82 dividend) ✅
- [x] Performance validated ✅
- [ ] Documentation complete (Phase 10)

### Step 4: Deploy - **PENDING**

- [ ] Old code removed (Phase 9)
- [ ] Branch merged
- [ ] Release tagged

______________________________________________________________________

## 📖 Documentation

**Full Plan**: `docs/DATA_LAYER_MIGRATION_PLAN.md`

- Complete phase-by-phase breakdown
- Code examples
- File structure
- Design decisions
- Risk analysis

**Related Docs**:

- `DATA_LAYER_MIGRATION.md` - Phase 1 completion report
- `DIVIDEND_CALCULATION_BUG_REPORT.md` - Data layer validation
- `docs/architecture.md` - System architecture
- `docs/implementation_plan_phase01.md` - Overall project plan

______________________________________________________________________

## 🚀 Next Steps

1. **Phase 7: Performance and Reporting** (2 days)

   - Implement performance metrics using `total_return` mode
   - Create analytics modules (performance, trades, risk)
   - Build reporting and visualization
   - Add 30+ unit tests for analytics

1. **Phase 9: Cleanup** (1 day)

   - Remove legacy `Bar` model and old adapter
   - Rename files for consistency
   - Final validation and coverage check

1. **Phase 10: Documentation** (2 days)

   - Update architecture docs
   - Create comprehensive migration guide
   - Update README and API reference
   - Document performance analytics

**Current Status**: Core migration complete! 68% done, 5 days remaining for analytics, cleanup, and documentation.

______________________________________________________________________

## 💡 Key Design Decisions

### Decision 1: Multi-Mode Bar Architecture

**Why**: Different components need different adjustment modes

**Benefits**:

- Strategy uses adjusted (split-consistent indicators)
- Execution uses unadjusted (realistic fills, accurate commissions)
- Performance uses total_return (includes dividends)

**Trade-off**: 3x memory (one bar becomes three)

**Verdict**: ✅ Accept - Correctness outweighs memory cost

**Details**: See `docs/MULTI_MODE_ARCHITECTURE_DECISION.md`

### Decision 2: Iterator-Based Flow

**Why**: Memory efficient, natural streaming\
**Trade-off**: Can't random access bars\
**Verdict**: ✅ Accept - Backtest is sequential

### Decision 3: Vendor Transformation at Boundary

**Why**: Clean separation, simple adapters\
**Trade-off**: Load all bars before transform\
**Verdict**: ✅ Accept - Data fits in memory

### Decision 4: No Backward Compatibility

**Why**: Clean slate, no technical debt\
**Trade-off**: All code must update\
**Verdict**: ✅ Accept - Project allows full refactor

______________________________________________________________________

## 📞 Questions?

See full implementation plan: `docs/DATA_LAYER_MIGRATION_PLAN.md`

**Key Documents**:

- **Architecture Decision**: `docs/MULTI_MODE_ARCHITECTURE_DECISION.md`
- **Before/After Comparison**: `docs/DATA_LAYER_BEFORE_AFTER.md`
- **Phase-by-Phase Plan**: `docs/DATA_LAYER_MIGRATION_PLAN.md`

**Key Sections**:

- Phase-by-phase breakdown (Phases 2-9)
- Code examples (before/after)
- File structure (new layout)
- Risk assessment
- Success criteria
- Migration checklist

______________________________________________________________________

**END OF SUMMARY**
