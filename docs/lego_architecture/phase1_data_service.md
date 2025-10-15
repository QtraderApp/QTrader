# Phase 1: Extract DataService

## Overview

**Goal:** Create a standalone, independently testable DataService that handles all data loading, transformation, and streaming with zero dependencies on execution, portfolio, or strategy layers.

**Duration:** 1-2 weeks **Complexity:** Low (foundation already exists) **Priority:** ⭐ Critical - Proves the lego architecture concept

## Current State

### What We Have (feature/schwab-integration branch)

✅ **Clean data layer structure:**

```
src/qtrader/
  data/
    loader.py          # DataLoader class
    iterator.py        # PriceSeriesIterator
    bar_merger.py      # BarMerger for multi-symbol
  adapters/
    algoseek.py        # AlgoseekOHLCVendorAdapter
    schwab.py          # SchwabOHLCVendorAdapter (planned)
    resolver.py        # DataSourceResolver
  models/
    bar.py             # Bar, PriceSeries
    multi_bar.py       # MultiBar (3 adjustment modes)
    instrument.py      # Instrument, DataSource enum
    vendors/
      algoseek.py      # AlgoseekBar, AlgoseekPriceSeries
```

✅ **Key features already implemented:**

- Multi-vendor adapter pattern
- Three adjustment modes (unadjusted, split-adjusted, total return)
- Iterator-based streaming
- Corporate actions handling (splits, dividends)
- Configuration-driven data loading
- Type-safe with Pydantic models

### Current Dependencies

**Good (acceptable):**

```python
from qtrader.models.bar import Bar, PriceSeries
from qtrader.models.multi_bar import MultiBar
from qtrader.models.instrument import Instrument, DataSource
from qtrader.config.data_config import DataConfig
```

**Bad (must remove):**

```python
# Currently NONE! Data layer is already clean 🎉
```

## Target Architecture

### Service Interface

```python
# src/qtrader/services/data/interface.py

from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, List, Optional, Protocol

from qtrader.models.bar import PriceSeries
from qtrader.models.instrument import Instrument
from qtrader.data.iterator import PriceSeriesIterator


class IDataService(Protocol):
    """
    Data service interface for loading and streaming price data.

    Responsibilities:
    - Load historical data for symbols
    - Transform to canonical format with adjustment modes
    - Stream data via iterators
    - Provide instrument metadata

    Does NOT:
    - Execute orders
    - Manage portfolio
    - Calculate indicators
    - Make trading decisions
    """

    def load_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        data_source: Optional[str] = None,
    ) -> PriceSeriesIterator:
        """
        Load data for single symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start_date: Start of date range
            end_date: End of date range (inclusive)
            data_source: Optional override for data source

        Returns:
            Iterator yielding MultiBar instances

        Raises:
            ValueError: If symbol not found or invalid date range
            FileNotFoundError: If data files missing
        """
        ...

    def load_universe(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        *,
        data_source: Optional[str] = None,
    ) -> Dict[str, PriceSeriesIterator]:
        """
        Load data for multiple symbols.

        Args:
            symbol: List of ticker symbols
            start_date: Start of date range
            end_date: End of date range (inclusive)
            data_source: Optional override for data source

        Returns:
            Dict mapping symbol → iterator

        Raises:
            ValueError: If any symbol not found
        """
        ...

    def get_instrument(self, symbol: str) -> Instrument:
        """
        Get instrument metadata.

        Args:
            symbol: Ticker symbol

        Returns:
            Instrument with metadata

        Raises:
            ValueError: If symbol not found
        """
        ...

    def list_available_symbols(
        self,
        data_source: Optional[str] = None,
    ) -> List[str]:
        """
        List all available symbols.

        Args:
            data_source: Filter by data source (None = all)

        Returns:
            List of available symbols
        """
        ...


class IDataAdapter(Protocol):
    """
    Adapter interface for vendor-specific data sources.

    Implementations: AlgoseekOHLCVendorAdapter, SchwabOHLCVendorAdapter, etc.
    """

    def read_bars(
        self,
        start_date: str,
        end_date: str,
    ) -> List:  # Vendor-specific bar type
        """Read raw bars from vendor data source."""
        ...

    def to_canonical_series(self, bars: List) -> Dict[str, PriceSeries]:
        """Transform vendor bars to canonical series with all adjustment modes."""
        ...
```

### Service Implementation

```python
# src/qtrader/services/data/service.py

from datetime import date
from typing import Dict, List, Optional

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.config.data_config import DataConfig
from qtrader.data.iterator import PriceSeriesIterator
from qtrader.data.loader import DataLoader
from qtrader.models.instrument import DataSource, Instrument, InstrumentType
from qtrader.services.data.interface import IDataService


class DataService:
    """
    Concrete implementation of data service.

    Delegates to DataLoader and adapters for actual data loading.
    Provides clean interface for consumers.
    """

    def __init__(
        self,
        config: DataConfig,
        resolver: Optional[DataSourceResolver] = None,
    ):
        """
        Initialize data service.

        Args:
            config: Data configuration
            resolver: Data source resolver (creates default if None)
        """
        self.config = config
        self.resolver = resolver or DataSourceResolver(config.sources)
        self.loader = DataLoader(config.to_dict())

    def load_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        data_source: Optional[str] = None,
    ) -> PriceSeriesIterator:
        """Load data for single symbol."""
        instrument = self._build_instrument(symbol, data_source)
        return self.loader.load(
            instrument,
            start_date.isoformat(),
            end_date.isoformat(),
        )

    def load_universe(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        *,
        data_source: Optional[str] = None,
    ) -> Dict[str, PriceSeriesIterator]:
        """Load data for multiple symbols."""
        return {
            symbol: self.load_symbol(symbol, start_date, end_date, data_source=data_source)
            for symbol in symbols
        }

    def get_instrument(self, symbol: str) -> Instrument:
        """Get instrument metadata."""
        # Default to Algoseek, equity
        # Can be enhanced with symbol lookup
        return Instrument(
            symbol=symbol,
            instrument_type=InstrumentType.EQUITY,
            data_source=DataSource.ALGOSEEK,
        )

    def list_available_symbols(
        self,
        data_source: Optional[str] = None,
    ) -> List[str]:
        """List available symbols."""
        # Placeholder - implement based on symbol map
        raise NotImplementedError("Symbol listing not yet implemented")

    def _build_instrument(self, symbol: str, data_source: Optional[str]) -> Instrument:
        """Build instrument with optional data source override."""
        if data_source:
            ds = DataSource[data_source.upper()]
        else:
            ds = DataSource.ALGOSEEK  # Default

        return Instrument(
            symbol=symbol,
            instrument_type=InstrumentType.EQUITY,
            data_source=ds,
        )
```

## Implementation Tasks

### Week 1: Interface Definition & Service Extraction

#### Task 1.1: Create Service Structure

- [ ] Create `src/qtrader/services/` directory
- [ ] Create `src/qtrader/services/__init__.py`
- [ ] Create `src/qtrader/services/data/` directory
- [ ] Create `src/qtrader/services/data/__init__.py`

**Files:**

```
src/qtrader/services/
  __init__.py
  data/
    __init__.py
    interface.py       # IDataService protocol
    service.py         # DataService implementation
```

#### Task 1.2: Define IDataService Protocol

- [ ] Create `interface.py` with `IDataService` protocol
- [ ] Define all method signatures with full type hints
- [ ] Add comprehensive docstrings
- [ ] Document exceptions that can be raised
- [ ] Add usage examples in docstrings

**Acceptance Criteria:**

- MyPy passes with no errors
- All methods have type hints
- Docstrings follow Google style
- Protocol has zero implementation code

#### Task 1.3: Implement DataService

- [ ] Create `service.py` with `DataService` class
- [ ] Implement all `IDataService` methods
- [ ] Delegate to existing `DataLoader`
- [ ] Add structured logging
- [ ] Handle errors gracefully

**Acceptance Criteria:**

- Implements `IDataService` protocol
- All methods functional
- Zero dependencies on execution/portfolio/strategy
- Logging at appropriate levels

#### Task 1.4: Update Existing Components

- [ ] Keep `DataLoader` as internal helper (don't delete)
- [ ] Keep adapters unchanged (already clean)
- [ ] Update `__init__.py` exports

**Files to Update:**

- `src/qtrader/services/data/__init__.py` - export IDataService, DataService
- `src/qtrader/services/__init__.py` - export data service

### Week 2: Testing & Documentation

#### Task 2.1: Unit Tests

- [ ] Create `tests/unit/services/data/test_service.py`
- [ ] Test `load_symbol()` with various inputs
- [ ] Test `load_universe()` with multiple symbols
- [ ] Test error handling (missing files, invalid dates)
- [ ] Test with real Algoseek data
- [ ] Achieve > 90% coverage

**Test Structure:**

```python
# tests/unit/services/data/test_service.py

import pytest
from datetime import date
from qtrader.services.data import DataService, IDataService
from qtrader.config.data_config import DataConfig


class TestDataService:
    """Test DataService implementation."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return DataConfig(
            root_path="data/",
            sources={"algoseek": {...}},
        )

    @pytest.fixture
    def service(self, config) -> IDataService:
        """Create service instance."""
        return DataService(config)

    def test_implements_interface(self, service):
        """Verify service implements IDataService."""
        assert isinstance(service, IDataService)  # Protocol check

    def test_load_symbol_success(self, service):
        """Test loading single symbol."""
        iterator = service.load_symbol(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 12, 31),
        )
        bars = list(iterator)
        assert len(bars) > 0
        assert bars[0].symbol == "AAPL"

    def test_load_universe_success(self, service):
        """Test loading multiple symbols."""
        iterators = service.load_universe(
            ["AAPL", "MSFT"],
            date(2020, 1, 1),
            date(2020, 12, 31),
        )
        assert len(iterators) == 2
        assert "AAPL" in iterators
        assert "MSFT" in iterators

    def test_load_symbol_invalid_date(self, service):
        """Test error handling for invalid dates."""
        with pytest.raises(ValueError):
            service.load_symbol(
                "AAPL",
                date(2020, 12, 31),  # end before start
                date(2020, 1, 1),
            )

    # ... more tests
```

#### Task 2.2: Integration Tests

- [ ] Create `tests/integration/services/test_data_service_integration.py`
- [ ] Test with real Algoseek data files
- [ ] Test multi-symbol loading
- [ ] Test date range edge cases
- [ ] Test memory usage with large datasets

#### Task 2.3: Mock Interface for Testing

- [ ] Create `tests/mocks/data_service.py`
- [ ] Implement `MockDataService` for use by other tests
- [ ] Provide canned data for testing

**Mock Structure:**

```python
# tests/mocks/data_service.py

from datetime import date
from typing import Dict, List, Optional

from qtrader.data.iterator import PriceSeriesIterator
from qtrader.models.instrument import Instrument
from qtrader.services.data.interface import IDataService


class MockDataService(IDataService):
    """
    Mock data service for testing.

    Returns canned data without requiring real files.
    Useful for testing execution, portfolio, strategy layers.
    """

    def __init__(self, canned_data: Optional[Dict] = None):
        self.canned_data = canned_data or {}

    def load_symbol(self, symbol, start_date, end_date, *, data_source=None):
        """Return canned iterator."""
        if symbol in self.canned_data:
            return self.canned_data[symbol]
        return self._generate_random_data(symbol, start_date, end_date)

    def load_universe(self, symbols, start_date, end_date, *, data_source=None):
        """Return dict of canned iterators."""
        return {s: self.load_symbol(s, start_date, end_date) for s in symbols}

    def get_instrument(self, symbol):
        """Return mock instrument."""
        return Instrument(symbol=symbol, ...)

    def list_available_symbols(self, data_source=None):
        """Return mock symbol list."""
        return list(self.canned_data.keys())

    def _generate_random_data(self, symbol, start, end):
        """Generate random bars for testing."""
        # Implementation...
        pass
```

#### Task 2.4: Documentation

- [ ] Add comprehensive docstrings to all classes/methods
- [ ] Create usage examples
- [ ] Document configuration options
- [ ] Add troubleshooting section

**Documentation Files:**

```
docs/lego_architecture/
  services/
    data_service.md          # Usage guide
    data_service_api.md      # API reference
    data_service_testing.md  # Testing guide
```

## Validation Criteria

### Functional Requirements

- [ ] ✅ Can load single symbol data
- [ ] ✅ Can load multiple symbol data
- [ ] ✅ Supports all adjustment modes (unadjusted, adjusted, total_return)
- [ ] ✅ Handles corporate actions (splits, dividends)
- [ ] ✅ Validates date ranges
- [ ] ✅ Provides instrument metadata
- [ ] ✅ Handles missing data gracefully

### Technical Requirements

- [ ] ✅ Implements `IDataService` protocol
- [ ] ✅ Zero dependencies on execution/portfolio/strategy
- [ ] ✅ All public methods have type hints
- [ ] ✅ MyPy passes with no errors
- [ ] ✅ Test coverage ≥ 90%
- [ ] ✅ All tests pass
- [ ] ✅ Structured logging implemented
- [ ] ✅ Documentation complete

### Performance Requirements

- [ ] ✅ Load 1 symbol, 1 year: < 1 second
- [ ] ✅ Load 10 symbols, 1 year: < 5 seconds
- [ ] ✅ Memory usage: < 500MB for 100 symbols, 1 year

## Testing Strategy

### Unit Tests (Isolated)

```python
def test_data_service_without_execution():
    """Prove data service works independently."""
    service = DataService(config)
    iterator = service.load_symbol("AAPL", start, end)

    # No execution, portfolio, or strategy needed!
    bars = list(iterator)
    assert len(bars) > 0
```

### Integration Tests (With Dependencies)

```python
def test_data_service_with_real_files():
    """Test with actual Algoseek data."""
    config = DataConfig.from_yaml("config/data_sources.yaml")
    service = DataService(config)

    # Load real data
    iterator = service.load_symbol("AAPL", date(2020, 1, 1), date(2020, 1, 31))
    bars = list(iterator)

    # Verify data quality
    assert all(b.adjusted.close > 0 for b in bars)
    assert all(b.unadjusted.close > 0 for b in bars)
```

### Mock Usage (For Other Layers)

```python
def test_execution_service_with_mock_data():
    """Future test showing how execution will use mock data."""
    mock_data = MockDataService(canned_data={
        "AAPL": generate_test_bars(),
    })

    # Execution service doesn't need real data service!
    execution = ExecutionService(data=mock_data, ...)
    # Test execution logic in isolation
```

## Migration Path

### Step 1: Create Service (No Breaking Changes)

- Service lives alongside existing `DataLoader`
- Nothing breaks in master branch code
- Can test independently

### Step 2: Update Internal Usage

- Update `BacktestEngine` to use `IDataService`
- Inject service via constructor
- **No backward compatibility** - clean break acceptable

### Step 3: Remove Old Code

- Immediately after service is tested and working
- Remove direct `DataLoader` usage from public APIs
- Keep `DataLoader` only as internal implementation detail within `DataService`

## Success Metrics

After Phase 1 completion:

- [ ] ✅ `DataService` exists and implements `IDataService`
- [ ] ✅ All tests pass (unit + integration)
- [ ] ✅ Coverage ≥ 90% for `DataService`
- [ ] ✅ `MockDataService` available for other layer tests
- [ ] ✅ Documentation complete
- [ ] ✅ Can demonstrate: "Load data without any other services running"
- [ ] ✅ Performance benchmarks met
- [ ] ✅ MyPy passes
- [ ] ✅ Code review approved

## Risks & Mitigations

| Risk                           | Impact | Mitigation                                      |
| ------------------------------ | ------ | ----------------------------------------------- |
| Existing DataLoader is complex | Medium | Keep it as internal helper, delegate to it      |
| Performance regression         | Medium | Benchmark before/after, optimize if needed      |
| Breaking changes               | Low    | Service is new, nothing depends on it yet       |
| Incomplete interface           | Low    | Start minimal, extend as needed in later phases |

## Dependencies

### Depends On

- ✅ Clean data layer (already exists in feature/schwab-integration)
- ✅ Pydantic models (already exist)
- ✅ Configuration system (already exists)

### Blocks

- Phase 2: PortfolioService (will use `MockDataService` for testing)
- Phase 3: ExecutionService (will use `MockDataService` for testing)
- Phase 5: BacktestEngine (will inject `IDataService`)

## Next Phase

Once Phase 1 is complete and validated:

👉 **[Phase 2: Extract PortfolioService](phase2_portfolio_service.md)**

______________________________________________________________________

**Phase Status:** 📝 Planning **Start Date:** TBD **Target Completion:** 2 weeks from start **Assigned To:** TBD **Last Updated:** October 15, 2025
