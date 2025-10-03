# Test Restructuring Plan

## Current Status

- **172 unit tests** (100% passing, 93% coverage)
- **0 integration tests**
- Tests organized by module: `tests/{module}/test_{file}.py`
- No golden file regression tests

## Objectives

1. ✅ Restructure to separate unit tests from integration tests
1. ✅ Add integration tests for end-to-end workflows
1. ✅ Add golden file regression tests
1. ✅ Follow industry best practices

## Proposed Structure

```
tests/
├── unit/                           # Fast, isolated component tests
│   ├── __init__.py
│   ├── adapters/
│   │   ├── __init__.py
│   │   ├── test_algoseek_parquet.py
│   │   └── test_csv_adapter.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── test_data_config.py
│   │   └── test_logging_config.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── test_commission.py
│   │   ├── test_engine.py
│   │   ├── test_fill_policy.py
│   │   ├── test_limit_stop.py
│   │   └── test_participation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── test_bar.py
│   │   ├── test_ledger.py
│   │   ├── test_order.py
│   │   ├── test_portfolio.py
│   │   └── test_position.py
│   └── validation/
│       ├── __init__.py
│       └── test_bar_validator.py
│
├── integration/                    # End-to-end workflow tests
│   ├── __init__.py
│   ├── conftest.py                # Integration test fixtures
│   ├── test_data_pipeline.py      # Adapter → Bars → Validation
│   ├── test_order_workflow.py     # Orders → Fills → Portfolio
│   ├── test_full_backtest.py      # Complete backtest scenarios
│   └── goldens/                   # Golden file regression tests
│       ├── __init__.py
│       ├── test_buy_and_hold.py
│       ├── test_sma_cross.py
│       └── data/                  # Expected outputs
│           ├── buy_and_hold_aapl_nav.csv
│           ├── buy_and_hold_aapl_fills.csv
│           └── sma_cross_msft_nav.csv
│
└── README.md                       # Updated test documentation
```

## Benefits

### Unit Tests (`tests/unit/`)

- ✅ **Fast** - Run in \<3 seconds
- ✅ **Isolated** - Mock dependencies, test single components
- ✅ **Focused** - One class/function per test
- ✅ **Developer-friendly** - Run during TDD cycle
- ✅ **Easy debugging** - Clear failure points

### Integration Tests (`tests/integration/`)

- ✅ **Realistic** - Use real data and full workflows
- ✅ **Comprehensive** - Test component interactions
- ✅ **Confidence** - Catch integration bugs
- ✅ **Documentation** - Show how system works end-to-end
- ✅ **Regression protection** - Golden files prevent unwanted changes

## Implementation Plan

### Phase 1: Restructure Existing Tests (No Changes to Code)

**Time**: 30 minutes **Risk**: Low (move files only)

1. Create `tests/unit/` directory structure
1. Move all existing 172 tests to `tests/unit/`
1. Update `__init__.py` files
1. Run `pytest tests/unit/` to verify all pass
1. Update `tests/README.md`

### Phase 2: Add Integration Test Infrastructure

**Time**: 1 hour **Risk**: Low (new files only)

1. Create `tests/integration/` directory
1. Add `conftest.py` with shared fixtures:
   - Real data adapter fixtures
   - Portfolio fixtures
   - Engine fixtures
1. Add `__init__.py` files

### Phase 3: Add Integration Tests (NEW)

**Time**: 2-3 hours **Risk**: Medium (may find bugs)

1. **test_data_pipeline.py**: Adapter → Bar conversion with real data

   - Load AAPL from fixture
   - Verify Bar normalization
   - Test date filtering
   - Test symbol filtering

1. **test_order_workflow.py**: Order → Fill → Portfolio updates

   - Submit orders on synthetic bars
   - Verify fills generated correctly
   - Verify portfolio cash/positions updated
   - Test multiple order types

1. **test_full_backtest.py**: Complete backtest scenarios

   - Buy-and-hold with real data
   - Multi-symbol trading
   - Participation capping with real volumes
   - Round-trip trades with PnL

### Phase 4: Golden File Regression Tests

**Time**: 2-3 hours **Risk**: Low (reference tests)

1. **tests/integration/goldens/test_buy_and_hold.py**

   - Run buy-and-hold on AAPL, MSFT, AMZN
   - Compare NAV, fills, orders against golden files
   - Store expected outputs in `data/` directory

1. **tests/integration/goldens/test_sma_cross.py**

   - Implement simple SMA crossover strategy
   - Run on MSFT (fixture data)
   - Compare results against golden files

## Pytest Configuration Updates

Update `pyproject.toml`:

```toml
[tool.pytest.ini_options]
# Default: run all tests
testpaths = ["tests"]

# Markers for selective test runs
markers = [
    "unit: Fast unit tests (isolated components)",
    "integration: Slower integration tests (end-to-end)",
    "golden: Golden file regression tests",
    "slow: Tests that take >1s",
]

# Configure coverage
addopts = [
    "--cov=src/qtrader",
    "--cov-report=html",
    "--cov-report=term-missing",
    "-v",
]
```

## Running Tests

```bash
# Run everything (unit + integration)
pytest tests/

# Run only unit tests (fast, for TDD)
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only golden file tests
pytest tests/integration/goldens/

# Run specific test file
pytest tests/unit/execution/test_participation.py

# Run tests by marker
pytest -m unit           # Fast unit tests
pytest -m integration    # Slower integration tests
pytest -m "not slow"     # Skip slow tests
```

## Makefile Updates

Update `makefile`:

```makefile
.PHONY: test-unit test-integration test-golden test-all

test-unit:
	@echo "ℹ️  Running unit tests (fast)..."
	uv run pytest tests/unit/ -v

test-integration:
	@echo "ℹ️  Running integration tests..."
	uv run pytest tests/integration/ -v --ignore=tests/integration/goldens/

test-golden:
	@echo "ℹ️  Running golden file regression tests..."
	uv run pytest tests/integration/goldens/ -v

test-all: test-unit test-integration test-golden
	@echo "✅ All test suites passed"

# Update existing 'test' target to run everything
test: setup test-all
```

## Success Metrics

### Phase 1 Complete ✅

- All 172 existing tests moved to `tests/unit/`
- All tests still pass
- No code changes required
- CI/CD still works

### Phase 2 Complete ✅

- Integration test infrastructure in place
- Shared fixtures available
- Directory structure created

### Phase 3 Complete ✅

- 5-10 integration tests added
- Tests cover full data flow
- Tests use real fixture data
- All tests pass

### Phase 4 Complete ✅

- Golden file tests implemented
- Buy-and-hold results stored
- SMA cross results stored
- Deterministic outputs verified

## Timeline

- **Phase 1**: 30 minutes (low risk)
- **Phase 2**: 1 hour (low risk)
- **Phase 3**: 2-3 hours (medium risk, may find bugs)
- **Phase 4**: 2-3 hours (low risk)

**Total**: 5-7 hours

## Next Steps

1. Get approval for restructuring approach
1. Execute Phase 1 (move files)
1. Verify all tests still pass
1. Execute Phases 2-4 incrementally
1. Update documentation

## Professional Best Practices ✅

This approach follows industry standards from:

- **Django**: `tests/unit/` and `tests/integration/`
- **FastAPI**: Separate unit and integration test directories
- **pytest**: Markers for selective test runs
- **Google Testing Blog**: Test pyramid (many unit tests, fewer integration tests)
- **Martin Fowler**: Test categorization by speed and scope

## Questions to Resolve

1. ✅ Proceed with restructuring? **Recommended: YES**
1. ✅ Add integration tests now or after Stage 6? **Recommended: NOW** (easier before more features)
1. ✅ Golden file tests: Use CSV or JSON outputs? **Recommended: CSV** (matches spec)
1. ✅ Store golden files in git or separate artifact storage? **Recommended: Git** (small files, deterministic)
