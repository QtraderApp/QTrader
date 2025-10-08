# Data Layer Migration Implementation Plan

**Version:** 1.0\
**Date:** October 8, 2025\
**Status:** Planning Phase

______________________________________________________________________

## Executive Summary

Migrate QTrader from the legacy `Bar` model (multi-series NamedTuple) to the new **Canonical Data Layer** architecture (vendor-agnostic models with proper separation). This is a **complete refactoring** with:

- ✅ **No backward compatibility** - Clean slate design
- ✅ **No bridge code** - Direct replacement
- ✅ **Full downstream refactoring** - All components updated
- ✅ **Iterator-based data flow** - Memory efficient streaming

**Key Achievement:** Data layer is already validated and producing correct results (dividend bug fixed, golden output matches).

______________________________________________________________________

## Current State Analysis

### What Works ✅

**New Data Layer** (`src/qtrader/models/`):

- ✅ `CanonicalBar` - Vendor-agnostic OHLC bar (Pydantic)
- ✅ `CanonicalPriceSeries` - Collection with mode (unadjusted/adjusted/total_return)
- ✅ `AlgoseekBar` - Vendor-specific raw bar (correct dividend formula)
- ✅ `AlgoseekPriceSeries` - Vendor collection with transformation
- ✅ `to_canonical_series()` - Produces all 3 modes with **correct math**
- ✅ Golden output validated: **$0.82 AAPL dividend** (100% accurate)

### What Needs Migration ❌

**Old Bar Model** (`src/qtrader/models/bar.py`):

```python
# OLD - Complex NamedTuple with 3 embedded series
class Bar(NamedTuple):
    ts: datetime
    symbol: str
    unadjusted: PriceSeries      # Nested NamedTuple
    capital_adjusted: PriceSeries
    total_return: PriceSeries
    dividend: Optional[Dividend]
    split: Optional[Split]
```

**Current Adapter** (`src/qtrader/adapters/algoseek.py`):

- ❌ Returns old `Bar` model
- ❌ Builds all 3 series inline (200+ lines)
- ❌ Mixes data loading with transformation
- ❌ No vendor abstraction

**Downstream Consumers**:

- ❌ `ExecutionEngine` expects old `Bar`
- ❌ `Portfolio` uses `bar.capital_adjusted.close`
- ❌ `Strategy.on_bar()` receives old `Bar`
- ❌ All tests use old `Bar` fixtures

______________________________________________________________________

## Target Architecture

### Data Flow (Iterator-Based)

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER (New)                           │
└─────────────────────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  1. Raw Data Source (Parquet/CSV/Database)     │
    └────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  2. VendorAdapter (e.g., AlgoseekAdapter)      │
    │     - Reads raw data                           │
    │     - Parses to vendor model (AlgoseekBar)     │
    │     - Returns Iterator[AlgoseekBar]            │
    └────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  3. PriceSeriesBuilder                         │
    │     - Collects vendor bars                     │
    │     - Builds AlgoseekPriceSeries               │
    │     - Transforms to 3 CanonicalPriceSeries     │
    │     - Returns Dict[mode, CanonicalPriceSeries] │
    └────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  4. PriceSeriesIterator                        │
    │     - Wraps selected CanonicalPriceSeries      │
    │     - Yields CanonicalBar one at a time        │
    │     - Iterator[CanonicalBar]                   │
    └────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   BACKTEST ENGINE (Updated)                     │
└─────────────────────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  5. Backtest.run()                             │
    │     for bar in price_series_iterator:          │
    │         strategy.on_bar(bar, ctx)              │
    └────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  6. Strategy.on_bar(bar: CanonicalBar)         │
    │     - Access bar.close, bar.high, etc.         │
    │     - Return signals                           │
    └────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  7. ExecutionEngine.on_bar(bar)                │
    │     - Evaluate orders against bar.high/low     │
    │     - Generate fills                           │
    └────────────────────────────────────────────────┘
                            ▼
    ┌────────────────────────────────────────────────┐
    │  8. Portfolio.update(bar)                      │
    │     - Update positions with bar.close          │
    │     - Calculate portfolio value                │
    └────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Multi-Mode Bar Model**: Each bar contains ALL adjustment modes

   - `MultiModeBar` contains: unadjusted, adjusted, total_return
   - Each component selects appropriate mode for its purpose
   - Single data load, multiple uses
   - Configuration specifies which mode each stage uses

1. **Iterator-Based**: Streaming data flow

   - Memory efficient streaming (one timestamp at a time)
   - Natural backtest progression
   - Easy to add filters/transformations

1. **Vendor Isolation**: Transformation at boundary

   - Vendor adapters return vendor models
   - `to_canonical_series()` converts to canonical
   - Backtest engine never sees vendor models

1. **Configuration-Driven Mode Selection per Stage**:

   ```yaml
   data:
     # Each stage uses optimal mode for its purpose
     signal_generation_mode: "adjusted"      # For indicators/strategies (split-adjusted)
     execution_mode: "unadjusted"            # For fills (actual traded prices)
     performance_mode: "total_return"        # For metrics (includes dividends)
   ```

   **Rationale**:

   - **Signals**: Use adjusted prices for consistent technical indicators across splits
   - **Execution**: Use unadjusted prices for realistic fills at actual market prices
   - **Performance**: Use total_return to include dividend reinvestment in returns

______________________________________________________________________

## Implementation Phases

### Phase 1: Core Models (1 day) ✅ COMPLETE

**Status**: Already implemented and validated

- ✅ `CanonicalBar` model
- ✅ `CanonicalPriceSeries` model
- ✅ `AlgoseekBar` with correct dividend formula
- ✅ `AlgoseekPriceSeries.to_canonical_series()`
- ✅ Tests passing (13 unit, 6 integration)
- ✅ Golden output validated ($0.82 AAPL dividend)

**No work needed** - Data layer is production-ready.

______________________________________________________________________

### Phase 2: Iterator Infrastructure (2 days)

**Objective**: Build streaming data layer with multi-mode support.

#### 2.0 Create `MultiModeBar` Model

**File**: `src/qtrader/models/multi_mode_bar.py`

```python
"""Multi-mode bar containing all adjustment modes."""

from pydantic import BaseModel
from qtrader.models.canonical_bar import CanonicalBar


class MultiModeBar(BaseModel):
    """
    Bar with all adjustment modes.

    Allows each component to select appropriate mode:
    - Strategy: adjusted (for indicators)
    - Execution: unadjusted (for fills)
    - Performance: total_return (for metrics)
    """

    symbol: str
    trade_datetime: str
    unadjusted: CanonicalBar
    adjusted: CanonicalBar
    total_return: CanonicalBar

    class Config:
        frozen = True  # Immutable

    def get_bar(self, mode: str) -> CanonicalBar:
        """Get bar for specific mode."""
        if mode == "unadjusted":
            return self.unadjusted
        elif mode == "adjusted":
            return self.adjusted
        elif mode == "total_return":
            return self.total_return
        else:
            raise ValueError(f"Invalid mode: {mode}")
```

**Tests**: `tests/unit/models/test_multi_mode_bar.py`

- Test all modes accessible
- Test get_bar() method
- Test immutability

#### 2.1 Create `PriceSeriesIterator` Class

**File**: `src/qtrader/data/price_series_iterator.py`

```python
"""Price series iterator for streaming multi-mode bars."""

from typing import Iterator, Optional, Dict
from qtrader.models.canonical_bar import CanonicalBar, CanonicalPriceSeries
from qtrader.models.multi_mode_bar import MultiModeBar


class PriceSeriesIterator:
    """
    Iterator wrapper for multi-mode price series.

    Provides streaming access to bars with all adjustment modes.
    Supports peek (look at next bar without consuming).
    """

    def __init__(self, series_dict: Dict[str, CanonicalPriceSeries]):
        """
        Initialize iterator.

        Args:
            series_dict: Dict with keys 'unadjusted', 'adjusted', 'total_return'
        """
        self.series_dict = series_dict
        self.symbol = series_dict["unadjusted"].symbol
        self._index = 0
        self._peeked: Optional[MultiModeBar] = None

    def __iter__(self) -> Iterator[MultiModeBar]:
        """Return iterator."""
        return self

    def __next__(self) -> MultiModeBar:
        """Get next multi-mode bar."""
        # If we peeked, return peeked bar
        if self._peeked is not None:
            bar = self._peeked
            self._peeked = None
            return bar

        # Otherwise get next from series
        unadj_bars = self.series_dict["unadjusted"].bars
        if self._index >= len(unadj_bars):
            raise StopIteration

        # Build MultiModeBar from all three series
        multi_bar = MultiModeBar(
            symbol=self.symbol,
            trade_datetime=unadj_bars[self._index].trade_datetime,
            unadjusted=self.series_dict["unadjusted"].bars[self._index],
            adjusted=self.series_dict["adjusted"].bars[self._index],
            total_return=self.series_dict["total_return"].bars[self._index],
        )

        self._index += 1
        return multi_bar

    def peek(self) -> Optional[MultiModeBar]:
        """
        Peek at next bar without consuming.

        Returns:
            Next MultiModeBar or None if at end
        """
        if self._peeked is not None:
            return self._peeked

        unadj_bars = self.series_dict["unadjusted"].bars
        if self._index >= len(unadj_bars):
            return None

        self._peeked = MultiModeBar(
            symbol=self.symbol,
            trade_datetime=unadj_bars[self._index].trade_datetime,
            unadjusted=self.series_dict["unadjusted"].bars[self._index],
            adjusted=self.series_dict["adjusted"].bars[self._index],
            total_return=self.series_dict["total_return"].bars[self._index],
        )
        self._index += 1
        return self._peeked

    def has_next(self) -> bool:
        """Check if more bars available."""
        unadj_bars = self.series_dict["unadjusted"].bars
        return self._peeked is not None or self._index < len(unadj_bars)

    def reset(self) -> None:
        """Reset iterator to beginning."""
        self._index = 0
        self._peeked = None
```

**Tests**: `tests/unit/data/test_price_series_iterator.py`

- Test iteration with MultiModeBar
- Test peek without consume
- Test has_next
- Test reset
- Test empty series
- Test mode access (unadjusted/adjusted/total_return)

#### 2.2 Create `DataLoader` Service

**File**: `src/qtrader/data/loader.py`

```python
"""Data loading service - coordinates adapter and transformation."""

from pathlib import Path
from typing import Dict, List
from qtrader.models.canonical_bar import CanonicalPriceSeries
from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries
from qtrader.data.price_series_iterator import PriceSeriesIterator


class DataLoader:
    """
    Coordinates data loading and transformation.

    Responsibilities:
    1. Call vendor adapter to get raw bars
    2. Build vendor price series
    3. Transform to canonical series (all 3 modes)
    4. Return iterator with multi-mode bars
    """

    def __init__(self, config: Dict):
        """
        Initialize data loader.

        Args:
            config: Data configuration dict (mode selection moved to components)
        """
        self.config = config

    def load_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> PriceSeriesIterator:
        """
        Load data for symbol and return multi-mode iterator.

        Args:
            symbol: Ticker symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            PriceSeriesIterator yielding MultiModeBar (all 3 modes)

        Steps:
        1. Load raw vendor bars (from adapter)
        2. Build AlgoseekPriceSeries
        3. Transform to canonical series (all 3 modes)
        4. Return iterator with all modes
        """
        # Step 1: Load raw bars from adapter
        # (Adapter integration done in Phase 3)
        raw_bars: List[AlgoseekBar] = self._load_from_adapter(
            symbol, start_date, end_date
        )

        # Step 2: Build vendor series
        vendor_series = AlgoseekPriceSeries(symbol=symbol, bars=raw_bars)

        # Step 3: Transform to canonical (all 3 modes)
        canonical_series_dict = vendor_series.to_canonical_series()

        # Step 4: Return iterator with all modes
        # Components will select mode based on their config
        return PriceSeriesIterator(canonical_series_dict)

    def _load_from_adapter(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> List[AlgoseekBar]:
        """Load raw bars from adapter (stub for Phase 3)."""
        raise NotImplementedError("Phase 3: Adapter integration")
```

**Tests**: `tests/unit/data/test_data_loader.py`

- Test iterator returns MultiModeBar
- Test all 3 modes present
- Test with mock adapter

#### 2.3 Update Configuration Schema

**File**: `config/qtrader.yaml`

```yaml
data:
  # Mode selection per component (each uses optimal series for its purpose)
  signal_generation_mode: "adjusted"      # For indicators/strategies (split-adjusted)
  execution_mode: "unadjusted"            # For fills (actual traded prices)
  performance_mode: "total_return"        # For metrics (includes dividends)

  # Mode descriptions:
  # - unadjusted: Raw prices as traded (for realistic fills, commissions)
  # - adjusted: Split-adjusted (for consistent indicators across splits)
  # - total_return: Split + dividend adjusted (for accurate performance metrics)
```

**Deliverables**:

- ✅ `MultiModeBar` model with all 3 modes
- ✅ `PriceSeriesIterator` yielding MultiModeBar
- ✅ `DataLoader` service (loads all modes)
- ✅ Configuration schema (mode per component)
- ✅ Unit tests (25+ tests)

______________________________________________________________________

### Phase 3: Adapter Refactoring (2 days)

**Objective**: Simplify adapters to return vendor models only.

#### 3.1 Create New `AlgoseekAdapter` (Simplified)

**File**: `src/qtrader/adapters/algoseek_vendor_adapter.py`

```python
"""Algoseek vendor adapter - returns AlgoseekBar objects."""

from typing import Iterator, List
from pathlib import Path
import duckdb
from qtrader.models.vendors.algoseek import AlgoseekBar


class AlgoseekVendorAdapter:
    """
    Algoseek vendor adapter - parses raw data to AlgoseekBar.

    Responsibilities:
    - Read parquet/CSV files
    - Parse timestamps
    - Validate data
    - Return Iterator[AlgoseekBar]

    Does NOT:
    - Perform adjustments
    - Transform to canonical
    - Business logic
    """

    def __init__(self, config: dict):
        """
        Initialize adapter.

        Args:
            config: Adapter configuration
        """
        self.root_path = Path(config["root_path"])
        self.symbol_map = config.get("symbol_map")

    def read_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Iterator[AlgoseekBar]:
        """
        Read raw bars from data source.

        Args:
            symbol: Ticker symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Yields:
            AlgoseekBar objects in chronological order
        """
        # Use DuckDB to read parquet efficiently
        conn = duckdb.connect(":memory:")

        # Build query
        secid = self._get_secid(symbol)
        parquet_path = self.root_path / f"SecId={secid}" / "*.parquet"

        query = f"""
        SELECT *
        FROM read_parquet('{parquet_path}')
        WHERE TradeDate >= '{start_date}'
          AND TradeDate <= '{end_date}'
        ORDER BY TradeDate
        """

        # Execute and yield bars
        result = conn.execute(query).fetchall()
        columns = [desc[0] for desc in conn.description]

        for row in result:
            row_dict = dict(zip(columns, row))
            yield AlgoseekBar(**row_dict)

        conn.close()

    def _get_secid(self, symbol: str) -> int:
        """Map symbol to SecId (stub)."""
        # Load from symbol_map CSV
        raise NotImplementedError()
```

**Key Changes**:

- ✅ Returns `AlgoseekBar` (vendor model)
- ✅ No transformation logic
- ✅ Pure data loading
- ✅ ~100 lines (was 500+)

#### 3.2 Update `DataLoader` Integration

**File**: `src/qtrader/data/loader.py` (update)

```python
def _load_from_adapter(
    self,
    symbol: str,
    start_date: str,
    end_date: str
) -> List[AlgoseekBar]:
    """Load raw bars from adapter."""
    from qtrader.adapters.algoseek_vendor_adapter import AlgoseekVendorAdapter

    adapter = AlgoseekVendorAdapter(self.config["adapter"])
    raw_bars = list(adapter.read_bars(symbol, start_date, end_date))
    return raw_bars
```

#### 3.3 Deprecate Old Adapter

**Action**:

- Rename `src/qtrader/adapters/algoseek.py` → `algoseek_legacy.py`
- Add deprecation warning
- Keep for reference during migration

**Deliverables**:

- ✅ New simplified `AlgoseekVendorAdapter`
- ✅ Integration with `DataLoader`
- ✅ Tests for adapter (10+ tests)
- ✅ Old adapter deprecated

______________________________________________________________________

### Phase 4: Backtest Engine Update (3 days)

**Objective**: Update backtest runner to use iterator-based data flow.

#### 4.1 Update `Backtest.run()` Signature

**File**: `src/qtrader/api/backtest.py`

**OLD**:

```python
def run(
    self,
    ctx: Context,
    bars: List[Bar],  # OLD: All bars loaded
    symbols: List[str],
    ...
):
```

**NEW**:

```python
def run(
    self,
    ctx: Context,
    data_iterators: Dict[str, PriceSeriesIterator],  # NEW: Iterator per symbol
    symbols: List[str],
    ...
):
    """
    Run backtest with iterator-based data flow.

    Args:
        ctx: Context with portfolio and risk manager
        data_iterators: Dict mapping symbol -> PriceSeriesIterator
        symbols: List of symbols
    """
```

#### 4.2 Update Event Loop

**File**: `src/qtrader/api/backtest.py`

```python
# Multi-symbol coordination
class BarMerger:
    """Merge multiple symbol iterators by timestamp."""

    def __init__(self, iterators: Dict[str, PriceSeriesIterator]):
        self.iterators = iterators
        self.current_bars = {}
        self._prime_iterators()

    def _prime_iterators(self):
        """Read first bar from each iterator."""
        for symbol, iterator in self.iterators.items():
            try:
                self.current_bars[symbol] = next(iterator)
            except StopIteration:
                pass

    def get_next_bar(self) -> tuple[str, MultiModeBar]:
        """Get next bar across all symbols (earliest timestamp)."""
        if not self.current_bars:
            raise StopIteration

        # Find earliest timestamp
        earliest_symbol = min(
            self.current_bars.keys(),
            key=lambda s: self.current_bars[s].trade_datetime
        )

        bar = self.current_bars[earliest_symbol]

        # Advance that iterator
        try:
            self.current_bars[earliest_symbol] = next(
                self.iterators[earliest_symbol]
            )
        except StopIteration:
            del self.current_bars[earliest_symbol]

        return earliest_symbol, bar


# In Backtest.run()
bar_merger = BarMerger(data_iterators)

try:
    while True:
        symbol, multi_bar = bar_merger.get_next_bar()

        # Each component selects its preferred mode
        # Strategy gets adjusted for indicators
        signals = self.strategy.on_bar(multi_bar, ctx)

        # Execution gets unadjusted for realistic fills
        fills = self.execution_engine.on_bar(multi_bar, signals)

        # Portfolio can use total_return for performance
        self.portfolio.update(multi_bar, fills)

except StopIteration:
    # All bars processed
    pass
```

#### 4.3 Update Strategy Interface

**File**: `src/qtrader/api/strategy.py`

**OLD**:

```python
def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
    # bar has: unadjusted, capital_adjusted, total_return
    price = bar.capital_adjusted.close  # OLD
```

**NEW**:

```python
def on_bar(self, bar: MultiModeBar, ctx: Context) -> Optional[List[Signal]]:
    # Strategy uses adjusted mode (configured in ctx.config)
    strategy_bar = bar.get_bar(ctx.config.signal_generation_mode)
    # or directly: strategy_bar = bar.adjusted

    price = strategy_bar.close  # Simpler access!

    # Example: SMA calculation uses adjusted prices
    if strategy_bar.close > self.sma_50:
        return [Signal.BUY]
```

**Migration Impact**:

- ✅ Strategy uses adjusted mode (optimal for indicators)
- ✅ Execution uses unadjusted mode (optimal for fills)
- ✅ Each component gets appropriate price series
- ✅ Configuration specifies mode per stage

**Deliverables**:

- ✅ Updated `Backtest.run()` with iterator flow
- ✅ `BarMerger` for multi-symbol coordination
- ✅ Updated strategy interface
- ✅ Updated example strategies (3 files)
- ✅ Integration tests (15+ tests)

______________________________________________________________________

### Phase 5: Execution Engine Update (2 days)

**Objective**: Update execution engine to work with `CanonicalBar`.

#### 5.1 Update `ExecutionEngine.on_bar()`

**File**: `src/qtrader/execution/engine.py`

**OLD**:

```python
def on_bar(
    self,
    bar: Bar,  # OLD: Multi-series bar
    next_bar: Optional[Bar] = None,
    ...
) -> List[Fill]:
    # Access bar.unadjusted.high for fills
    high = bar.unadjusted.high
```

**NEW**:

```python
def on_bar(
    self,
    bar: MultiModeBar,  # NEW: Multi-mode bar
    next_bar: Optional[MultiModeBar] = None,
    ...
) -> List[Fill]:
    # Execution uses unadjusted mode (configured in ctx.config)
    exec_bar = bar.get_bar(self.config.execution_mode)
    # or directly: exec_bar = bar.unadjusted

    high = exec_bar.high  # Actual traded high

    # Fill at actual market prices (not adjusted prices)
    # This ensures realistic commission/slippage calculations
    fill_price = exec_bar.high
```

**Key Changes**:

- Execution explicitly uses unadjusted mode
- Fills based on actual traded prices
- Commissions calculated on real market values
- Simpler code with direct field access

**Rationale**:

- **Why unadjusted for execution?**
  - Commissions per share are based on actual traded price
  - Slippage should reflect real market conditions
  - Order size validation uses actual volumes
  - Position sizing based on real cash requirements

#### 5.2 Update Dividend Processing

**File**: `src/qtrader/execution/dividend_processor.py`

**OLD**:

```python
def process_bar(self, bar: Bar):
    if bar.dividend:
        # Process dividend
```

**NEW**:

```python
def process_bar(self, bar: CanonicalBar):
    if bar.dividend:
        # Process dividend (same logic, simpler type)
```

**Deliverables**:

- ✅ Updated `ExecutionEngine`
- ✅ Updated `DividendProcessor`
- ✅ Updated fill policies (3 files)
- ✅ Unit tests updated (50+ tests)

______________________________________________________________________

### Phase 6: Portfolio & Position Update (1 day)

**Objective**: Update portfolio components for `CanonicalBar`.

#### 6.1 Update `Portfolio.update_bar()`

**File**: `src/qtrader/models/portfolio.py`

**OLD**:

```python
def update_bar(self, bar: Bar):
    close = bar.capital_adjusted.close  # OLD
```

**NEW**:

```python
def update_bar(self, bar: MultiModeBar):
    # Portfolio can use different modes for different purposes:

    # 1. Position valuation: Use unadjusted for current market value
    valuation_bar = bar.unadjusted
    position_value = self.position.shares * valuation_bar.close

    # 2. Performance tracking: Use total_return for accurate returns
    performance_bar = bar.get_bar(self.config.performance_mode)
    # or directly: performance_bar = bar.total_return

    # Total return includes dividend reinvestment
    total_return = (performance_bar.close - self.entry_price) / self.entry_price
```

**Rationale**:

- **Position Valuation**: Use unadjusted (current market value)
- **Performance Metrics**: Use total_return (includes dividends)
- **P&L Calculations**: Can use adjusted (split-consistent)

#### 6.2 Update `Position` Valuation

**File**: `src/qtrader/models/position.py`

Similar simplification - direct field access.

**Deliverables**:

- ✅ Updated `Portfolio`
- ✅ Updated `Position`
- ✅ Unit tests updated (40+ tests)

______________________________________________________________________

### Phase 7: Test Suite Migration (3 days)

**Objective**: Update all tests to use new models.

#### 7.1 Create Test Fixtures

**File**: `tests/fixtures/canonical_bars.py`

```python
"""Canonical bar fixtures for testing."""

from qtrader.models.canonical_bar import CanonicalBar


def create_test_bar(
    trade_datetime: str = "2024-01-01",
    open_price: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 102.0,
    volume: int = 1000000,
    dividend: Optional[Decimal] = None,
) -> CanonicalBar:
    """Create test canonical bar."""
    return CanonicalBar(
        trade_datetime=trade_datetime,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        dividend=dividend,
    )
```

#### 7.2 Update Test Files

**Strategy**:

1. Update unit tests (50 files, ~400 tests)

   - Replace `Bar` fixtures with `CanonicalBar`
   - Update assertions
   - Remove series selection logic

1. Update integration tests (10 files, ~70 tests)

   - Update end-to-end flows
   - Verify iterator integration

1. Update golden tests

   - Regenerate with new models
   - Verify output format

**Deliverables**:

- ✅ All unit tests passing (~400 tests)
- ✅ All integration tests passing (~70 tests)
- ✅ Golden tests regenerated

______________________________________________________________________

### Phase 8: Documentation & Examples (2 days)

**Objective**: Update all documentation and examples.

#### 8.1 Update Examples

**Files**:

- `examples/buy_and_hold_strategy.py`
- `examples/sma_crossover_strategy.py`
- `examples/risk_signal_example.py`

**Changes**:

```python
# OLD
def on_bar(self, bar: Bar, ctx: Context):
    close = bar.capital_adjusted.close

# NEW
def on_bar(self, bar: CanonicalBar, ctx: Context):
    close = bar.close  # Simpler!
```

#### 8.2 Update Documentation

**Files**:

- `docs/architecture.md`
- `docs/implementation_plan_phase01.md`
- `README.md`
- `QUICK_REFERENCE.md`

**Add**:

- Data layer architecture diagram
- Iterator usage guide
- Mode selection guide
- Migration examples

#### 8.3 Create Migration Guide

**File**: `docs/MIGRATION_GUIDE_V2.md`

Content:

- Before/after code examples
- Breaking changes list
- Configuration updates
- Common migration patterns

**Deliverables**:

- ✅ Updated examples (3 files)
- ✅ Updated documentation (5 files)
- ✅ Migration guide created
- ✅ API reference updated

______________________________________________________________________

### Phase 9: Cleanup & Validation (1 day)

**Objective**: Remove old code and validate system.

#### 9.1 Remove Old Code

**Delete**:

- `src/qtrader/models/bar.py` (old Bar/PriceSeries)
- `src/qtrader/adapters/algoseek_legacy.py`
- Old adapter tests

**Rename**:

- `canonical_bar.py` → `bar.py` (becomes primary)
- `algoseek_vendor_adapter.py` → `algoseek.py`

#### 9.2 Run Full Test Suite

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/qtrader --cov-report=html

# Verify golden output
python scripts/generate_goldens.py
```

#### 9.3 Performance Validation

**Metrics**:

- Memory usage (should be lower with iterator)
- Speed (should be comparable)
- Test coverage (maintain >95%)

**Deliverables**:

- ✅ Old code removed
- ✅ All tests passing (470+ tests)
- ✅ Coverage >95%
- ✅ Performance validated

______________________________________________________________________

## Timeline Summary

| Phase                      | Duration | Deliverables         | Status      |
| -------------------------- | -------- | -------------------- | ----------- |
| 1. Core Models             | 1 day    | Data layer models    | ✅ COMPLETE |
| 2. Iterator Infrastructure | 2 days   | Iterator, DataLoader | 📋 TODO     |
| 3. Adapter Refactoring     | 2 days   | Simplified adapters  | 📋 TODO     |
| 4. Backtest Engine         | 3 days   | Updated runner       | 📋 TODO     |
| 5. Execution Engine        | 2 days   | Updated execution    | 📋 TODO     |
| 6. Portfolio Update        | 1 day    | Updated portfolio    | 📋 TODO     |
| 7. Test Suite              | 3 days   | All tests passing    | 📋 TODO     |
| 8. Documentation           | 2 days   | Docs & examples      | 📋 TODO     |
| 9. Cleanup                 | 1 day    | Remove old code      | 📋 TODO     |

**Total**: 17 days (~3.5 weeks)

______________________________________________________________________

## Risk Assessment

### Technical Risks

| Risk                      | Severity | Mitigation                          |
| ------------------------- | -------- | ----------------------------------- |
| Breaking changes in tests | HIGH     | Phase 7 dedicated to test migration |
| Performance regression    | MEDIUM   | Phase 9 performance validation      |
| Multi-symbol coordination | MEDIUM   | BarMerger tested separately         |
| Iterator state management | LOW      | Comprehensive unit tests            |

### Business Risks

| Risk                      | Severity | Mitigation                |
| ------------------------- | -------- | ------------------------- |
| Extended development time | LOW      | Clear 17-day timeline     |
| Incomplete migration      | LOW      | Phase-by-phase validation |
| Documentation gaps        | LOW      | Phase 8 dedicated to docs |

______________________________________________________________________

## Success Criteria

### Functional Requirements

- ✅ All 470+ tests passing
- ✅ Golden output matches ($0.82 AAPL dividend validated)
- ✅ Iterator-based data flow working
- ✅ Multi-symbol backtests supported
- ✅ All 3 modes selectable (unadjusted/adjusted/total_return)

### Non-Functional Requirements

- ✅ Code coverage >95%
- ✅ Memory usage ≤ current (iterator should reduce)
- ✅ Performance within 10% of current
- ✅ Documentation complete
- ✅ Examples working

### Code Quality

- ✅ No legacy `Bar` model references
- ✅ No bridge/compatibility code
- ✅ Clean separation: data layer → backtest engine
- ✅ Type hints throughout
- ✅ Docstrings on all public APIs

______________________________________________________________________

## Migration Checklist

### Pre-Migration

- [x] Data layer validated
- [x] Golden output verified ($0.82 dividend)
- [x] Implementation plan reviewed
- [ ] Team alignment on timeline

### During Migration

- [ ] Phase 2: Iterator infrastructure
- [ ] Phase 3: Adapter refactoring
- [ ] Phase 4: Backtest engine update
- [ ] Phase 5: Execution engine update
- [ ] Phase 6: Portfolio update
- [ ] Phase 7: Test suite migration
- [ ] Phase 8: Documentation
- [ ] Phase 9: Cleanup

### Post-Migration

- [ ] All tests passing
- [ ] Golden output regenerated
- [ ] Performance validated
- [ ] Documentation complete
- [ ] Examples updated
- [ ] Old code removed

______________________________________________________________________

## Appendix A: Key Design Decisions

### Decision 1: Multi-Mode Bar Model

**Rationale**:

- Different stages need different modes:
  - **Strategy**: adjusted (split-consistent indicators)
  - **Execution**: unadjusted (realistic fills at actual prices)
  - **Performance**: total_return (includes dividend reinvestment)
- Single data load, multiple uses
- Each component selects optimal mode
- Configuration-driven per stage

**Trade-off**:

- Memory: 3x bars in memory (vs 1x in single-mode)
- Complexity: Components must select appropriate mode

**Verdict**: ✅ Accept - Flexibility and correctness outweigh memory cost

**Example Use Case**:

```yaml
# Strategy generates signals using adjusted prices (split-consistent)
signal_generation_mode: "adjusted"

# Execution fills orders at unadjusted prices (actual market prices)
execution_mode: "unadjusted"

# Performance calculated with total return (includes dividends)
performance_mode: "total_return"
```

This allows:

- SMA crossover strategy to work correctly across stock splits
- Commission calculations based on actual traded prices
- Accurate performance metrics including dividend reinvestment

### Decision 2: Iterator-Based Flow

**Rationale**:

- Memory efficient
- Natural streaming
- Easy to add transformations

**Trade-off**:

- Can't random access bars
- Must coordinate multi-symbol iterators

**Verdict**: ✅ Accept - Backtest is sequential anyway

### Decision 3: Vendor Transformation at Boundary

**Rationale**:

- Clean separation of concerns
- Vendor adapters stay simple
- Transformation logic in one place

**Trade-off**:

- Load all bars before transformation
- Can't stream transformation

**Verdict**: ✅ Accept - Data sets fit in memory

### Decision 4: No Backward Compatibility

**Rationale**:

- Clean slate design
- No technical debt
- Simpler implementation

**Trade-off**:

- All code must be updated
- No gradual migration

**Verdict**: ✅ Accept - Project scope allows clean refactor

______________________________________________________________________

## Appendix B: Code Examples

### Before (OLD)

```python
# Strategy
def on_bar(self, bar: Bar, ctx: Context):
    # Multi-series bar - must select
    close = bar.capital_adjusted.close

# Execution
def evaluate_order(self, bar: Bar):
    # Must select series - but which one for fills?
    high = bar.unadjusted.high  # For realistic fills

# Adapter
def read_bars(self) -> Iterator[Bar]:
    # Returns complex multi-series bar
    return Bar(
        ts=ts,
        symbol=symbol,
        unadjusted=PriceSeries(...),
        capital_adjusted=PriceSeries(...),
        total_return=PriceSeries(...),
    )
```

### After (NEW)

```python
# Strategy (uses adjusted for indicators)
def on_bar(self, bar: MultiModeBar, ctx: Context):
    # Select adjusted mode for consistent indicators across splits
    strategy_bar = bar.adjusted
    close = strategy_bar.close

    # SMA calculation unaffected by stock splits
    if close > self.sma_50:
        return [Signal.BUY]

# Execution (uses unadjusted for fills)
def evaluate_fill(self, bar: MultiModeBar, order: Order):
    # Select unadjusted mode for realistic fills at actual prices
    exec_bar = bar.unadjusted
    fill_price = exec_bar.high

    # Commissions based on actual traded price
    commission = fill_price * order.shares * 0.001

# Portfolio (uses total_return for performance)
def calculate_return(self, bar: MultiModeBar):
    # Select total_return mode for accurate performance including dividends
    perf_bar = bar.total_return
    return_value = (perf_bar.close - self.entry_price) / self.entry_price

# Adapter (returns vendor model only)
def read_bars(self) -> Iterator[AlgoseekBar]:
    # Returns vendor model - transformation happens at data layer
    yield AlgoseekBar(...)

# Data Layer (transforms to all modes)
loader = DataLoader(config)
iterator = loader.load_data(symbol, start, end)
# Returns MultiModeBar iterator with all 3 modes
# Each component selects appropriate mode
```

### Benefits of Multi-Mode Architecture

1. **Optimal Mode per Stage**:

   - Signals: `adjusted` (split-consistent indicators)
   - Execution: `unadjusted` (realistic fills)
   - Performance: `total_return` (includes dividends)

1. **Single Data Load**:

   - Load once, use three ways
   - No duplicate data loading
   - Memory efficient streaming

1. **Configuration Flexibility**:

   ```yaml
   signal_generation_mode: "adjusted"
   execution_mode: "unadjusted"
   performance_mode: "total_return"
   ```

1. **Clear Separation of Concerns**:

   - Strategy focuses on signals (adjusted prices)
   - Execution focuses on fills (actual prices)
   - Portfolio tracks performance (total return)

______________________________________________________________________

## Appendix C: File Structure

### New Structure

```
src/qtrader/
├── models/
│   ├── bar.py                    # CanonicalBar (renamed from canonical_bar.py)
│   ├── price_series.py           # CanonicalPriceSeries
│   └── vendors/
│       └── algoseek/
│           ├── bar.py            # AlgoseekBar
│           └── price_series.py   # AlgoseekPriceSeries
│
├── data/                         # NEW: Data layer services
│   ├── __init__.py
│   ├── loader.py                 # DataLoader
│   └── price_series_iterator.py  # PriceSeriesIterator
│
├── adapters/
│   ├── base.py
│   └── algoseek.py               # AlgoseekVendorAdapter (simplified)
│
├── api/
│   ├── backtest.py               # Updated for iterator flow
│   ├── strategy.py               # Updated for CanonicalBar
│   └── context.py
│
├── execution/
│   ├── engine.py                 # Updated for CanonicalBar
│   ├── dividend_processor.py     # Updated
│   └── ...
│
└── ...
```

### Deleted Files

```
src/qtrader/models/bar.py              # OLD multi-series Bar
src/qtrader/adapters/algoseek_legacy.py # OLD complex adapter
```

______________________________________________________________________

## Next Steps

1. **Review this plan** with team
1. **Phase 2: Start iterator infrastructure** (2 days)
1. **Daily standup** to track progress
1. **Phase-by-phase validation** (run tests after each phase)

______________________________________________________________________

**END OF IMPLEMENTATION PLAN**
