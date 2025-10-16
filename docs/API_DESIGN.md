# QTrader Service API Design

## Clean Service API

The QTrader services package exposes a clean, top-level API for easy imports:

```python
# ✅ Clean API - Import from top-level services package
from qtrader.services import DataService, IDataService, IDataAdapter

# ❌ Verbose - Don't need to import from submodules
# from qtrader.services.data.service import DataService
# from qtrader.services.data.interface import IDataService
```

## Architecture Benefits

### 1. **Simplicity for Users**

```python
# Simple, clean imports
from qtrader.services import DataService

service = DataService(config)
```

### 2. **Protocol-Based Design**

```python
# Type hints use protocols for loose coupling
from qtrader.services import IDataService

def run_backtest(data_service: IDataService):
    """Works with any IDataService implementation."""
    pass
```

### 3. **Dependency Injection Ready**

```python
# Easy to inject for testing
from qtrader.services import IDataService, DataService

def create_data_service(config: dict) -> IDataService:
    """Factory function for DI."""
    return DataService(config)
```

## Package Structure

```
src/qtrader/services/
├── __init__.py          # Exposes clean API: DataService, IDataService, IDataAdapter
├── data/
│   ├── __init__.py      # Submodule exports
│   ├── interface.py     # IDataService, IDataAdapter protocols
│   └── service.py       # DataService implementation
└── (future services...)
    ├── execution/
    ├── portfolio/
    └── strategy/
```

## Design Principles

### No Separate `api` Module Needed

The `__init__.py` at each level serves as the public API:

- **Top-level** (`services/__init__.py`): Exposes main service classes
- **Submodule** (`services/data/__init__.py`): Exposes data-specific components
- **Implementation** (`services/data/service.py`): Internal implementation

### Why This Works

1. **Python convention**: `__init__.py` is the standard way to define package APIs
1. **Single source of truth**: No separate `api.py` module that duplicates exports
1. **Clarity**: Import path matches logical hierarchy
1. **Maintenance**: Fewer files to update when adding services

## Example Usage

### Basic Usage

```python
from qtrader.services import DataService

service = DataService({
    "adapter": {...},
    "data_sources": {...}
})

# Load and iterate
for multibar in service.load("AAPL", "2024-01-01", "2024-12-31"):
    print(f"Date: {multibar.adjusted.trade_datetime}")
    print(f"Close: {multibar.adjusted.close}")
```

### Testing with Protocols

```python
from qtrader.services import IDataService

class MockDataService(IDataService):
    """Test implementation."""
    def load(self, symbol, start_date, end_date):
        # Mock data
        pass

def test_with_mock():
    service: IDataService = MockDataService()
    # Test code...
```

### Future Services (Phase 2+)

```python
# As we add more services, they follow the same pattern
from qtrader.services import (
    DataService,          # Phase 1 ✅
    ExecutionService,     # Phase 4
    PortfolioService,     # Phase 6
    StrategyService,      # Phase 7
)
```

## Type Safety

The `__all__` annotation ensures MyPy compliance:

```python
# services/__init__.py
__all__: list[str] = [
    "DataService",
    "IDataService",
    "IDataAdapter",
]
```

This provides:

- **Type checking**: MyPy validates the exports
- **IDE support**: Autocomplete and go-to-definition work correctly
- **Documentation**: Clear contract of what's public vs internal

## Conclusion

**Yes, DataService should be exposed at the services level** for a clean API.

**No, we don't need a separate `api` module** - `__init__.py` serves this purpose perfectly and follows Python conventions.

This approach:

- ✅ Makes imports clean and simple
- ✅ Follows Python package conventions
- ✅ Enables protocol-based dependency injection
- ✅ Scales as we add more services
- ✅ Provides full type safety
