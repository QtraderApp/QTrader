# Test Skipping Explanation

## Summary

When you run `make test`, you see **10 tests skipped** (marked with 's' in the output). These are **intentional placeholder tests** for future features that haven't been implemented yet.

## Skipped Tests Breakdown

### Total Tests: 242 collected

- **Passed: 232** ✅
- **Skipped: 10** ⏭️ (intentional placeholders)

### Skipped Tests by Category

#### 1. Golden File Tests (4 skipped)

**Location:** `tests/integration/goldens/`

**File:** `test_buy_and_hold.py` (3 skipped)

- `test_buy_and_hold_aapl_golden_placeholder`
- `test_buy_and_hold_msft_golden_placeholder`
- `test_buy_and_hold_amzn_golden_placeholder`

**File:** `test_sma_cross.py` (1 skipped)

- `test_sma_cross_msft_golden_placeholder`

**Reason:** "Requires Strategy API, Backtest runner, and golden file generation"

**What they will test:**

- Deterministic output verification
- Regression testing (compare against reference outputs)
- NAV, fills, orders CSV validation
- Ensures no unexpected calculation changes

#### 2. Data Pipeline Tests (2 skipped)

**Location:** `tests/integration/test_data_pipeline.py`

**Tests:**

- `test_data_pipeline_placeholder`
- `test_multi_dataset_alignment_placeholder`

**Reason:** "Adapter integration tests require full data loading implementation"

**What they will test:**

- Full data loading workflow (Files → Adapter → Bar Objects)
- Bar validation and normalization
- Date/symbol filtering
- Multi-dataset alignment strategies

#### 3. Full Backtest Tests (4 skipped)

**Location:** `tests/integration/test_full_backtest.py`

**Tests:**

- `test_buy_and_hold_aapl_placeholder`
- `test_multi_symbol_rotation_placeholder`
- `test_participation_with_real_volumes_placeholder`
- `test_limit_orders_with_real_ohlc_placeholder`

**Reason:** "Requires Strategy API and Backtest runner (future stages)"

**What they will test:**

- Complete backtest workflows with fixture data
- Buy-and-hold strategy execution
- Multi-symbol portfolio rotation
- Participation rate capping with real volumes
- Limit order execution with real OHLC data

## Why They're Skipped

These tests are **placeholders** for future functionality that requires:

1. **Strategy API** (Stage 6+)

   - `Context` API implementation
   - `on_bar()` callback mechanism
   - Strategy protocol execution

1. **Backtest Runner** (Stage 6+)

   - Event loop integration
   - Data feed orchestration
   - Output generation (NAV, fills, orders CSVs)

1. **Golden File Infrastructure** (Stage 7+)

   - Reference output generation
   - Deterministic testing framework
   - Regression detection system

## Current Test Coverage

Despite the skipped tests, **you have excellent coverage**:

```
Total Coverage: 85% (1,617 lines covered out of 1,896)
```

**Fully tested modules:**

- ✅ Risk Management (90-100% coverage)
- ✅ Order Workflow (100% coverage)
- ✅ Execution Engine (86-100% coverage)
- ✅ Models (95-100% coverage)
- ✅ Position Tracking (95% coverage)
- ✅ Cash Ledger (100% coverage)
- ✅ Adapters (80-100% coverage)
- ✅ Configuration (89-99% coverage)

**Lower coverage (expected - not fully used yet):**

- ⏳ Context API (26%) - awaiting Strategy integration
- ⏳ Backtest API (55%) - awaiting runner implementation
- ⏳ Bar Validator (63%) - partially implemented
- ⏳ CSV Adapter (60%) - basic functionality working

## How to See Detailed Skip Reasons

Run with verbose output:

```bash
# See all skip reasons
make test-fast

# Or with pytest directly
uv run pytest -v

# Show skipped test details
uv run pytest -v -rs
```

## When Will They Be Implemented?

### Stage 6: Event Loop & Strategy Integration

- Implement Strategy API (`Context`, `on_bar`)
- Build backtest runner
- Enable full backtest tests ✅

### Stage 7: Output & Golden Files

- Implement output generation (NAV, fills, orders)
- Create golden file infrastructure
- Enable golden file regression tests ✅

### Future: Data Pipeline Enhancement

- Enhanced adapter integration tests
- Multi-dataset alignment
- Advanced data validation

## Is This Normal?

**Yes!** This is a **best practice** in test-driven development:

✅ **Write placeholder tests early** to document requirements ✅ **Skip them** until dependencies are ready ✅ **Prevents forgetting** what needs to be tested ✅ **Clear tracking** of remaining work

## Summary

- **10 skipped tests** = **10 planned future features**
- **232 passing tests** = **Current implementation is solid**
- **85% code coverage** = **Excellent quality**
- **All core functionality tested** = **Production-ready Stage 5B**

The skipped tests are a **roadmap**, not a problem. They show that the project is well-planned with clear testing goals for future stages.

## How to Remove Warnings

If you want a cleaner test output (hide skipped tests), you can:

```bash
# Run without showing skipped tests
uv run pytest --cov --cov-report=term-missing --cov-report=html -q

# Or update makefile to hide skips
# Add -q flag to pytest command
```

But keeping them visible is actually **good practice** - it reminds you what's planned!
