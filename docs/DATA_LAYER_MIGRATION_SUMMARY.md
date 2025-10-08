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

### ✅ What's Done (Phase 1)

**Data Layer Models** - Production ready:

- `MultiModeBar` - Container with all 3 adjustment modes (designed, not yet implemented)
- `CanonicalBar` - Single price series (Pydantic)
- `CanonicalPriceSeries` - Collection with mode
- `AlgoseekBar` - Vendor-specific with correct dividend formula
- `AlgoseekPriceSeries.to_canonical_series()` - Transforms to 3 modes
- **Validation**: $0.82 AAPL dividend (100% accurate)
- **Tests**: 13 unit + 6 integration passing

### ❌ What Needs Migration

**Old Architecture** (~50 files):

- `Bar` model (multi-series NamedTuple)
- `AlgoseekOHLCAdapter` (returns old Bar)
- `ExecutionEngine` (expects old Bar)
- `Portfolio` (uses bar.capital_adjusted.close)
- `Strategy.on_bar()` (receives old Bar)
- All tests (~470 tests)

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

| Phase                          | Days | Work                            | Status  |
| ------------------------------ | ---- | ------------------------------- | ------- |
| **1. Core Models**             | 1    | Data layer validated            | ✅ DONE |
| **2. Iterator Infrastructure** | 2    | PriceSeriesIterator, DataLoader | 📋 TODO |
| **3. Adapter Refactoring**     | 2    | Simplified vendor adapters      | 📋 TODO |
| **4. Backtest Engine**         | 3    | Iterator-based runner           | 📋 TODO |
| **5. Execution Engine**        | 2    | CanonicalBar support            | 📋 TODO |
| **6. Portfolio Update**        | 1    | Direct field access             | 📋 TODO |
| **7. Test Suite**              | 3    | Migrate 470+ tests              | 📋 TODO |
| **8. Documentation**           | 2    | Docs & examples                 | 📋 TODO |
| **9. Cleanup**                 | 1    | Remove old code                 | 📋 TODO |

**Total: 17 days (~3.5 weeks)**

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

- [ ] All 470+ tests passing
- [ ] Golden output matches ($0.82 dividend validated)
- [ ] Iterator-based data flow working
- [ ] Multi-symbol coordination working
- [ ] Multi-mode bar architecture working
- [ ] Each component uses optimal mode

### Non-Functional

- [ ] Code coverage >95%
- [ ] Memory usage acceptable (3x bars but streaming keeps it bounded)
- [ ] Performance within 10% of current
- [ ] Documentation complete
- [ ] Examples working

### Code Quality

- [ ] No legacy Bar references
- [ ] No bridge/compatibility code
- [ ] Clean separation (data → backtest)
- [ ] Type hints throughout
- [ ] Docstrings complete

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

### Phase 2: Iterator Infrastructure

- `MultiModeBar` model (container with 3 CanonicalBar fields)
- `PriceSeriesIterator` class (yields MultiModeBar)
- `DataLoader` service (loads all 3 modes)
- Configuration schema (mode per component)
- 25+ unit tests

### Phase 3: Adapter Refactoring

- Simplified `AlgoseekVendorAdapter` (~100 lines, was 500+)
- Returns `AlgoseekBar` (vendor model)
- No transformation logic
- 10+ adapter tests

### Phase 4: Backtest Engine

- `BarMerger` (multi-symbol coordination, yields MultiModeBar)
- Updated `Backtest.run()` signature
- Iterator-based event loop (passes MultiModeBar)
- Updated strategy interface (receives MultiModeBar)
- 15+ integration tests

### Phase 5: Execution Engine

- Updated `ExecutionEngine.on_bar(bar: MultiModeBar)`
- Uses `bar.unadjusted` for realistic fills
- Simplified fill logic (actual traded prices)
- 50+ tests updated

### Phase 6: Portfolio Update

- Updated `Portfolio.update_bar(bar: MultiModeBar)`
- Uses `bar.unadjusted` for valuation
- Uses `bar.total_return` for performance
- Updated `Position` calculations
- 40+ tests updated

### Phase 7: Test Suite

- Test fixtures for `CanonicalBar`
- All unit tests updated (400+)
- All integration tests updated (70+)
- Golden tests regenerated

### Phase 8: Documentation

- Updated examples (3 files)
- Updated architecture docs
- Migration guide created
- API reference updated

### Phase 9: Cleanup

- Old `Bar` model removed
- Old adapter removed
- Full test suite validation
- Performance validation

______________________________________________________________________

## 🔄 Migration Process

### Step 1: Review Plan

- [ ] Team review completed
- [ ] Timeline approved
- [ ] Resources allocated

### Step 2: Execute Phases (17 days)

- [ ] Phase 2: Iterator infrastructure
- [ ] Phase 3: Adapter refactoring
- [ ] Phase 4: Backtest engine
- [ ] Phase 5: Execution engine
- [ ] Phase 6: Portfolio update
- [ ] Phase 7: Test suite
- [ ] Phase 8: Documentation
- [ ] Phase 9: Cleanup

### Step 3: Validation

- [ ] All tests passing
- [ ] Golden output verified
- [ ] Performance validated
- [ ] Documentation complete

### Step 4: Deploy

- [ ] Old code removed
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

1. **Today**: Review this summary and full plan
1. **Tomorrow**: Start Phase 2 (Iterator Infrastructure)
1. **Daily**: Track progress, run tests
1. **Week 1**: Complete Phases 2-4 (infrastructure + backtest)
1. **Week 2**: Complete Phases 5-7 (execution + tests)
1. **Week 3**: Complete Phases 8-9 (docs + cleanup)

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
