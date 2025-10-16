# CLI Refactoring Summary

## Problem

The original CLI (`cli.py`) had become a 420-line monolith with only 3 commands, mixing:

- Click decorators (CLI framework)
- Business logic (update orchestration, cache scanning)
- UI/formatting (Rich tables, progress bars)
- Error handling and validation

This created several issues:

1. **Poor separation of concerns** - CLI doing too much
1. **Not testable** - Business logic trapped in CLI commands
1. **Not reusable** - Can't use update logic outside CLI
1. **Not scalable** - Adding backtest/portfolio commands would create 2000+ line file

## Solution

Refactored into a clean **3-layer architecture**:

```
src/qtrader/
├── cli/                     # Layer 1: Thin CLI orchestration
│   ├── __init__.py          #   5 lines
│   ├── main.py              #  20 lines - root CLI group
│   ├── commands/
│   │   ├── __init__.py      #   5 lines
│   │   └── data.py          # 348 lines - data commands (thin)
│   └── ui/
│       ├── __init__.py      #  15 lines
│       ├── formatters.py    # 140 lines - Rich table formatting
│       └── progress.py      #  24 lines - Progress bars
├── services/
│   └── data/
│       └── update_service.py  # 154 lines - business logic
```

**Total: 711 lines** (vs 420 before, but with better separation)

## Architecture Layers

### Layer 1: CLI Commands (Thin Orchestration)

**`cli/commands/data.py`** (348 lines)

- Click decorators and argument parsing
- Calls services to do actual work
- Uses UI helpers for formatting
- Error handling and user feedback

**Responsibilities:**

- Parse CLI arguments
- Create service instances
- Call service methods
- Format results with UI helpers
- Handle errors and display messages

**What it does NOT do:**

- Business logic
- Direct cache access
- Table formatting (delegates to UI layer)
- Progress bar creation (delegates to UI layer)

### Layer 2: Business Logic (Services)

**`services/data/update_service.py`** (154 lines)

- Wraps DatasetUpdater with convenience methods
- Handles symbol source priority logic
- Reads cache metadata
- No CLI or UI dependencies

**Responsibilities:**

- Orchestrate updates using DatasetUpdater
- Determine symbol sources (--symbols > universe > cache)
- Read cache metadata
- Return plain data (no formatting)

**What it does NOT do:**

- CLI argument parsing
- Rich table formatting
- Progress bars
- Error display

### Layer 3: UI Helpers (Presentation)

**`cli/ui/formatters.py`** (140 lines) - Table formatters

- Creates Rich tables
- Adds rows with proper formatting
- No business logic

**`cli/ui/progress.py`** (24 lines) - Progress bars

- Creates Rich Progress instances
- Configured for specific use cases

## Benefits

### ✅ Separation of Concerns

- **CLI**: Thin orchestration layer
- **Service**: Testable business logic
- **UI**: Reusable formatters

### ✅ Testability

**Before:**

```python
# Can't test update logic without CLI framework
def update_dataset(dataset: str, symbols: str, ...):  # CLI command
    # 167 lines of mixed logic
```

**After:**

```python
# Test service directly
service = UpdateService("schwab-us-equity-1d-adjusted")
symbols, desc = service.get_symbols_to_update()
assert symbols == ["AAPL", "GOOGL"]
```

### ✅ Reusability

**Before:** Update logic trapped in CLI

**After:**

```python
# Use in scripts
from qtrader.services import UpdateService

service = UpdateService("schwab-us-equity-1d-adjusted")
for result in service.update_symbols(["AAPL", "TSLA"]):
    print(f"{result.symbol}: {result.bars_added} bars")

# Use in API endpoints
# Use in notebooks
# Use in scheduled jobs
```

### ✅ Scalability

**Before:** Adding 5 more command groups → 2000+ line monolith

**After:** Each command group in separate file

```
cli/commands/
├── data.py       # 348 lines (data management)
├── backtest.py   # ~300 lines (backtesting) - future
├── portfolio.py  # ~200 lines (portfolio analysis) - future
├── risk.py       # ~150 lines (risk metrics) - future
└── report.py     # ~200 lines (reporting) - future
```

Total would be ~1200 lines spread across 5 focused files, not a 2000+ line monolith.

### ✅ Maintainability

- **Easier to find code** (logical grouping by domain)
- **Easier to modify** (changes localized to one layer)
- **Easier to onboard** (clear structure, single responsibility)

## Code Metrics

### Before Refactoring

| File      | Lines   | Responsibilities                           |
| --------- | ------- | ------------------------------------------ |
| cli.py    | 420     | CLI + Business Logic + UI + Error Handling |
| **Total** | **420** | **Everything mixed together**              |

### After Refactoring

| File                            | Lines   | Responsibilities               |
| ------------------------------- | ------- | ------------------------------ |
| **CLI Layer**                   |         |                                |
| cli/main.py                     | 20      | Root CLI group                 |
| cli/commands/data.py            | 348     | Data commands (orchestration)  |
| **Business Logic Layer**        |         |                                |
| services/data/update_service.py | 154     | Update orchestration logic     |
| **UI Layer**                    |         |                                |
| cli/ui/formatters.py            | 140     | Rich table formatting          |
| cli/ui/progress.py              | 24      | Progress bar helpers           |
| **Supporting**                  |         |                                |
| cli/\*\*/\_\_init\_\_.py        | 25      | Package initialization         |
| **Total**                       | **711** | **Clean separation, reusable** |

### Improvement Metrics

- **Separation**: 3 clear layers (CLI, Service, UI)
- **Reusability**: Service layer usable outside CLI
- **Testability**: Can unit test service without CLI framework
- **Scalability**: Easy to add new command groups without bloat

## Migration Impact

### ✅ Zero Breaking Changes

- Entry point unchanged: `qtrader = "qtrader.cli:main"`
- All CLI commands work identically
- All 298 tests pass
- 90% code coverage maintained

### File Changes

**Added:**

```
src/qtrader/
├── cli/                      # NEW package
│   ├── __init__.py
│   ├── main.py
│   ├── commands/
│   │   ├── __init__.py
│   │   └── data.py
│   └── ui/
│       ├── __init__.py
│       ├── formatters.py
│       └── progress.py
└── services/data/
    └── update_service.py     # NEW service
```

**Removed:**

```
src/qtrader/cli.py            # 420 lines → refactored
```

## Future Extensions

### Easy to Add New Command Groups

**Backtest Commands** (`cli/commands/backtest.py`):

```python
@click.group("backtest")
def backtest_group():
    """Backtesting commands"""
    pass

@backtest_group.command("run")
@click.option("--strategy", required=True)
def run_backtest(strategy: str):
    service = BacktestService()
    results = service.run(strategy)
    # Use UI formatters for display
```

**Portfolio Commands** (`cli/commands/portfolio.py`):

```python
@click.group("portfolio")
def portfolio_group():
    """Portfolio analysis commands"""
    pass
```

### Service Layer Can Be Extended

```python
# New service for backtesting
class BacktestService:
    def run(self, strategy: str) -> BacktestResult:
        # Business logic here
        pass

# Reuse UI helpers
from qtrader.cli.ui import create_backtest_summary_table
```

## Testing Strategy

### Service Layer (Unit Tests)

```python
def test_update_service_symbol_priority():
    service = UpdateService("test-dataset")

    # Test explicit symbols priority
    symbols, desc = service.get_symbols_to_update(["AAPL"])
    assert symbols == ["AAPL"]
    assert "symbols" in desc

def test_update_service_cache_metadata():
    service = UpdateService("test-dataset")
    start, end, count = service.get_cache_metadata("AAPL")
    assert start is not None
```

### CLI Layer (Integration Tests)

```python
from click.testing import CliRunner

def test_cli_update_command():
    runner = CliRunner()
    result = runner.invoke(update_dataset, ["--dataset", "test", "--dry-run"])
    assert result.exit_code == 0
    assert "symbols" in result.output
```

## Conclusion

This refactoring transforms a growing 420-line monolith into a **clean, scalable, 3-layer architecture**:

1. **CLI Layer**: Thin orchestration (Click commands)
1. **Service Layer**: Reusable business logic
1. **UI Layer**: Shared formatting components

**Result:**

- ✅ Better separation of concerns
- ✅ Testable business logic
- ✅ Reusable services (CLI, API, scripts, notebooks)
- ✅ Scalable for future commands (backtest, portfolio, risk, etc.)
- ✅ Zero breaking changes, all tests pass

The codebase is now ready to scale to 10+ command groups without becoming an unmaintainable mess.
