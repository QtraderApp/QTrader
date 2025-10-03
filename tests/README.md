# QTrader Test Suite

This directory contains all tests for the QTrader project, organized to mirror the source structure in `src/qtrader/`.

## Structure

```
tests/
├── adapters/          # Tests for data adapters
│   ├── test_algoseek_parquet.py
│   └── test_csv_adapter.py
├── api/               # Tests for public API (Strategy, Context, Backtest)
├── config/            # Tests for configuration
│   └── test_data_config.py
├── models/            # Tests for data models
│   ├── test_bar.py
│   ├── test_ledger.py
│   ├── test_order.py
│   └── test_position.py
└── validation/        # Tests for data validation
    └── test_bar_validator.py
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run specific module tests
pytest tests/models/
pytest tests/adapters/

# Run specific test file
pytest tests/models/test_order.py

# Run with verbose output
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/qtrader --cov-report=html
```

## Test Organization Principles

1. **Mirror source structure**: Each source module has a corresponding test module

   - `src/qtrader/models/order.py` → `tests/models/test_order.py`
   - `src/qtrader/adapters/csv_adapter.py` → `tests/adapters/test_csv_adapter.py`

1. **One test file per source file**: Keep tests focused and easy to find

1. **Test file naming**: Prefix with `test_` (pytest convention)

1. **Package structure**: Each test directory is a Python package with `__init__.py`

## Current Test Coverage

- **Stage 1 Complete**: Data foundation (adapters, models, validation) - 36 tests
- **Stage 2 Complete**: Order models & ledger foundation - 55 tests
- **Total**: 91 tests passing ✅

## Adding New Tests

When creating a new source module, create a corresponding test file:

```python
# Example: Adding src/qtrader/execution/engine.py

# 1. Create test file: tests/execution/test_engine.py
# 2. Add __init__.py: tests/execution/__init__.py
# 3. Write tests following existing patterns
```

## Test Fixtures

Common fixtures are defined in `conftest.py` files at appropriate levels:

- Root-level `conftest.py` for project-wide fixtures
- Module-level `conftest.py` for module-specific fixtures
