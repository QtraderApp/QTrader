# Data Layer Migration - Executive Summary

**Status:** Planning Complete | **Timeline:** 17 days | **Complexity:** Medium

______________________________________________________________________

## 🎯 Objective

Migrate QTrader from legacy multi-series `Bar` model to new **Canonical Data Layer** with iterator-based data flow.

**Key Principles:**

- ✅ **No backward compatibility** - Clean refactor
- ✅ **No bridge code** - Direct replacement
- ✅ **Iterator-based** - Memory efficient streaming
- ✅ **Configuration-driven** - Mode selection in YAML

______________________________________________________________________

## 📊 Current State

### ✅ What's Done (Phase 1)

**Data Layer Models** - Production ready:

- `CanonicalBar` - Single series bar (Pydantic)
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

### Data Flow (Iterator-Based)

```
┌──────────────────────────────────────────────────┐
│  1. Raw Data (Parquet/CSV)                       │
│     ↓                                             │
│  2. VendorAdapter → Iterator[AlgoseekBar]        │
│     ↓                                             │
│  3. PriceSeriesBuilder → 3 CanonicalPriceSeries  │
│     ↓                                             │
│  4. Mode Selection (config: "adjusted")          │
│     ↓                                             │
│  5. PriceSeriesIterator → CanonicalBar           │
│     ↓                                             │
│  6. Backtest.run(iterator)                       │
│     ↓                                             │
│  7. Strategy.on_bar(bar: CanonicalBar)           │
│          bar.close  # Direct access!             │
└──────────────────────────────────────────────────┘
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
def on_bar(self, bar: CanonicalBar, ctx: Context):
    # Single series - direct access
    close = bar.close
```

**Benefits**:

- ✅ Simpler strategy code (no series selection)
- ✅ Memory efficient (iterator vs load-all)
- ✅ Clean vendor separation
- ✅ Configuration-driven mode selection

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

### 1. Bar Model Simplification

| Aspect             | OLD                          | NEW                    |
| ------------------ | ---------------------------- | ---------------------- |
| **Structure**      | Multi-series NamedTuple      | Single-series Pydantic |
| **Price Access**   | `bar.capital_adjusted.close` | `bar.close`            |
| **Mode Selection** | Runtime (code)               | Config-time (YAML)     |
| **Memory**         | All series loaded            | Single series          |

### 2. Data Loading Pattern

**OLD** (Adapter returns Bar):

```python
adapter = AlgoseekOHLCAdapter(config, instrument)
bars: List[Bar] = list(adapter.read_bars(config))
# Returns complex multi-series bars
```

**NEW** (Adapter returns vendor model):

```python
adapter = AlgoseekVendorAdapter(config)
vendor_bars = list(adapter.read_bars(symbol, start, end))
series = AlgoseekPriceSeries(symbol, bars=vendor_bars)
canonical = series.to_canonical_series()  # 3 modes
iterator = PriceSeriesIterator(canonical["adjusted"])
```

### 3. Configuration

**New `config/qtrader.yaml`**:

```yaml
data:
  price_series_mode: "adjusted"  # unadjusted | adjusted | total_return
```

______________________________________________________________________

## 🎯 Success Criteria

### Functional

- [ ] All 470+ tests passing
- [ ] Golden output matches ($0.82 dividend validated)
- [ ] Iterator-based data flow working
- [ ] Multi-symbol coordination working
- [ ] All 3 modes selectable

### Non-Functional

- [ ] Code coverage >95%
- [ ] Memory usage ≤ current (iterator should reduce)
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
| Performance regression    | MEDIUM   | Phase 9 validation          |
| Multi-symbol coordination | MEDIUM   | BarMerger tested separately |
| Iterator state bugs       | LOW      | Comprehensive unit tests    |

______________________________________________________________________

## 📝 Key Deliverables by Phase

### Phase 2: Iterator Infrastructure

- `PriceSeriesIterator` class (peek support)
- `DataLoader` service
- Configuration schema
- 20+ unit tests

### Phase 3: Adapter Refactoring

- Simplified `AlgoseekVendorAdapter` (~100 lines, was 500+)
- Returns `AlgoseekBar` (vendor model)
- No transformation logic
- 10+ adapter tests

### Phase 4: Backtest Engine

- `BarMerger` (multi-symbol coordination)
- Updated `Backtest.run()` signature
- Iterator-based event loop
- Updated strategy interface
- 15+ integration tests

### Phase 5: Execution Engine

- Updated `ExecutionEngine.on_bar()`
- Direct field access (bar.high, bar.low)
- Simplified fill logic
- 50+ tests updated

### Phase 6: Portfolio Update

- Updated `Portfolio.update_bar()`
- Updated `Position` valuation
- Direct field access
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

### Decision 1: Single Series per Bar

**Why**: Simpler code, configuration-driven mode selection\
**Trade-off**: Can't mix modes in same backtest\
**Verdict**: ✅ Accept - Cleaner architecture

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

**Key Sections**:

- Phase-by-phase breakdown (Phases 2-9)
- Code examples (before/after)
- File structure (new layout)
- Risk assessment
- Success criteria
- Migration checklist

______________________________________________________________________

**END OF SUMMARY**
