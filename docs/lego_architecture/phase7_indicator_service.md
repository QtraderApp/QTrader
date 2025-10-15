# Phase 7: IndicatorService

## Overview

**Goal:** Extract technical indicators into an independent service with caching, state management, and a clean interface for strategy consumption.

**Duration:** 2-3 weeks **Complexity:** Medium **Priority:** High - Critical for strategy functionality

## Rationale: Why Separate Service?

### Why NOT Inside Strategy (Phase 6)?

- ❌ Indicators have complex state management (caching, warmup)
- ❌ Multiple strategies might share indicator computations
- ❌ Testing strategy logic separately from indicator logic is valuable
- ❌ Mixing concerns: Strategy = WHAT to trade, Indicators = HOW to analyze

### Why Separate Service?

- ✅ Clear separation of concerns
- ✅ Independent testing of indicator calculations
- ✅ Reusable across strategies (performance optimization)
- ✅ Easier to add custom indicators
- ✅ Can cache computations across strategies
- ✅ Clean interface for indicator management

## Current State (Master Branch)

```
src/qtrader/indicators/
  __init__.py
  base.py              # Indicator base class with caching
  manager.py           # IndicatorManager for convenience
  helpers.py           # Common calculation utilities
  momentum/
    __init__.py
    macd.py            # MACD indicator
    rsi.py             # RSI indicator
  trend/
    __init__.py
    sma.py             # Simple Moving Average
    ema.py             # Exponential Moving Average
  volatility/
    __init__.py
    atr.py             # Average True Range
    bollinger.py       # Bollinger Bands
```

**Current Dependencies:**

```python
from qtrader.api.context import Context  # For bar history access
from qtrader.models.bar import Bar       # For price data
```

**Current Usage:**

```python
class MyStrategy(Strategy):
    def on_bar(self, ctx: Context):
        sma_50 = ctx.indicators.sma("AAPL", 50)
        rsi = ctx.indicators.rsi("AAPL", 14)
        if rsi < 30 and price > sma_50:
            # Buy signal
```

## Target Architecture

### Service Interface

```python
# src/qtrader/services/indicators/interface.py

from abc import abstractmethod
from typing import Optional, Protocol, TypeVar

from qtrader.models.bar import Bar

T = TypeVar("T")


class IIndicator(Protocol[T]):
    """
    Interface for all indicators.

    Indicators compute technical analysis values from price history.
    Implementations handle caching and state management.
    """

    @abstractmethod
    def compute(self, symbol: str, bars: list[Bar]) -> Optional[T]:
        """
        Compute indicator value from bar history.

        Args:
            symbol: Symbol to compute for
            bars: Historical bars (oldest first)

        Returns:
            Indicator value or None if insufficient data
        """
        ...

    @abstractmethod
    def warmup(self, symbol: str, bars: list[Bar]) -> None:
        """
        Warmup indicator with historical data.

        Args:
            symbol: Symbol to warmup for
            bars: Historical bars for warmup
        """
        ...

    @abstractmethod
    def reset(self, symbol: str) -> None:
        """
        Reset indicator state for symbol.

        Args:
            symbol: Symbol to reset
        """
        ...


class IIndicatorService(Protocol):
    """
    Indicator service interface.

    Provides access to built-in and custom indicators with
    automatic caching and state management.

    Responsibilities:
    - Compute technical indicators
    - Cache indicator values
    - Manage indicator lifecycle (warmup, reset)
    - Register custom indicators

    Does NOT:
    - Make trading decisions (that's Strategy)
    - Load price data (that's DataService via Context)
    - Manage portfolio (that's PortfolioService)
    """

    # Moving Averages
    def sma(
        self, symbol: str, bars: list[Bar], period: int, field: str = "close"
    ) -> Optional[float]:
        """Simple Moving Average."""
        ...

    def ema(
        self, symbol: str, bars: list[Bar], period: int, field: str = "close"
    ) -> Optional[float]:
        """Exponential Moving Average."""
        ...

    # Momentum Indicators
    def rsi(
        self, symbol: str, bars: list[Bar], period: int = 14
    ) -> Optional[float]:
        """Relative Strength Index."""
        ...

    def macd(
        self,
        symbol: str,
        bars: list[Bar],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Optional[dict]:
        """MACD indicator (returns dict with macd, signal, histogram)."""
        ...

    # Volatility Indicators
    def atr(
        self, symbol: str, bars: list[Bar], period: int = 14
    ) -> Optional[float]:
        """Average True Range."""
        ...

    def bollinger_bands(
        self,
        symbol: str,
        bars: list[Bar],
        period: int = 20,
        num_std: float = 2.0,
    ) -> Optional[dict]:
        """Bollinger Bands (returns dict with upper, middle, lower)."""
        ...

    # Custom Indicators
    def register_indicator(self, name: str, indicator: IIndicator) -> None:
        """
        Register custom indicator.

        Args:
            name: Unique identifier for indicator
            indicator: Indicator implementation
        """
        ...

    def get_indicator(self, name: str, symbol: str, bars: list[Bar]) -> Optional:
        """
        Get custom indicator value.

        Args:
            name: Indicator identifier
            symbol: Symbol to compute for
            bars: Bar history

        Returns:
            Indicator value or None
        """
        ...

    # Lifecycle Management
    def warmup_all(self, symbol: str, bars: list[Bar]) -> None:
        """Warmup all indicators for symbol."""
        ...

    def reset_all(self, symbol: str) -> None:
        """Reset all indicators for symbol."""
        ...
```

### Service Implementation

```python
# src/qtrader/services/indicators/service.py

from typing import Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.indicators.base import Indicator
from qtrader.indicators.momentum import MACDIndicator, RSI
from qtrader.indicators.trend import EMA, SMA
from qtrader.indicators.volatility import ATR, BollingerBandsIndicator
from qtrader.models.bar import Bar
from qtrader.services.indicators.interface import IIndicator, IIndicatorService

logger = LoggerFactory.get_logger()


class IndicatorService:
    """
    Concrete implementation of indicator service.

    Manages built-in and custom indicators with caching.
    Delegates computation to indicator implementations.
    """

    def __init__(self):
        """Initialize indicator service."""
        self._indicators: dict[str, IIndicator] = {}
        self._custom_indicators: dict[str, IIndicator] = {}

        logger.info("indicator_service.initialized")

    def sma(
        self, symbol: str, bars: list[Bar], period: int, field: str = "close"
    ) -> Optional[float]:
        """Simple Moving Average."""
        key = f"sma_{period}_{field}"
        if key not in self._indicators:
            self._indicators[key] = SMA(period, field)
        return self._indicators[key].compute(symbol, bars)

    def ema(
        self, symbol: str, bars: list[Bar], period: int, field: str = "close"
    ) -> Optional[float]:
        """Exponential Moving Average."""
        key = f"ema_{period}_{field}"
        if key not in self._indicators:
            self._indicators[key] = EMA(period, field)
        return self._indicators[key].compute(symbol, bars)

    def rsi(
        self, symbol: str, bars: list[Bar], period: int = 14
    ) -> Optional[float]:
        """Relative Strength Index."""
        key = f"rsi_{period}"
        if key not in self._indicators:
            self._indicators[key] = RSI(period)
        return self._indicators[key].compute(symbol, bars)

    def macd(
        self,
        symbol: str,
        bars: list[Bar],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Optional[dict]:
        """MACD indicator."""
        key = f"macd_{fast}_{slow}_{signal}"
        if key not in self._indicators:
            self._indicators[key] = MACDIndicator(fast, slow, signal)
        result = self._indicators[key].compute(symbol, bars)
        if result:
            return {
                "macd": result.macd,
                "signal": result.signal,
                "histogram": result.histogram,
            }
        return None

    def atr(
        self, symbol: str, bars: list[Bar], period: int = 14
    ) -> Optional[float]:
        """Average True Range."""
        key = f"atr_{period}"
        if key not in self._indicators:
            self._indicators[key] = ATR(period)
        return self._indicators[key].compute(symbol, bars)

    def bollinger_bands(
        self,
        symbol: str,
        bars: list[Bar],
        period: int = 20,
        num_std: float = 2.0,
    ) -> Optional[dict]:
        """Bollinger Bands."""
        key = f"bb_{period}_{num_std}"
        if key not in self._indicators:
            self._indicators[key] = BollingerBandsIndicator(period, num_std)
        result = self._indicators[key].compute(symbol, bars)
        if result:
            return {
                "upper": result.upper,
                "middle": result.middle,
                "lower": result.lower,
            }
        return None

    def register_indicator(self, name: str, indicator: IIndicator) -> None:
        """Register custom indicator."""
        if name in self._custom_indicators:
            logger.warning(
                "indicator_service.indicator_overwrite",
                name=name,
                reason="Indicator already registered",
            )
        self._custom_indicators[name] = indicator
        logger.info("indicator_service.indicator_registered", name=name)

    def get_indicator(self, name: str, symbol: str, bars: list[Bar]) -> Optional:
        """Get custom indicator value."""
        if name not in self._custom_indicators:
            logger.error(
                "indicator_service.indicator_not_found",
                name=name,
                available=list(self._custom_indicators.keys()),
            )
            return None
        return self._custom_indicators[name].compute(symbol, bars)

    def warmup_all(self, symbol: str, bars: list[Bar]) -> None:
        """Warmup all indicators."""
        for indicator in self._indicators.values():
            indicator.warmup(symbol, bars)
        for indicator in self._custom_indicators.values():
            indicator.warmup(symbol, bars)

        logger.debug(
            "indicator_service.warmup_complete",
            symbol=symbol,
            num_indicators=len(self._indicators) + len(self._custom_indicators),
        )

    def reset_all(self, symbol: str) -> None:
        """Reset all indicators."""
        for indicator in self._indicators.values():
            indicator.reset(symbol)
        for indicator in self._custom_indicators.values():
            indicator.reset(symbol)

        logger.debug("indicator_service.reset_complete", symbol=symbol)
```

### Integration with Context

```python
# src/qtrader/api/context.py (Phase 6)

class Context:
    """User-facing strategy context."""

    def __init__(
        self,
        # ... other services ...
        indicators: IIndicatorService,
    ):
        self._indicators = indicators

    # Convenience methods for indicators
    def sma(self, symbol: str, period: int, field: str = "close") -> Optional[float]:
        """Simple Moving Average."""
        bars = self.get_bar_history(symbol, period)
        return self._indicators.sma(symbol, bars, period, field)

    def rsi(self, symbol: str, period: int = 14) -> Optional[float]:
        """Relative Strength Index."""
        bars = self.get_bar_history(symbol, period * 2)  # RSI needs extra bars
        return self._indicators.rsi(symbol, bars, period)
```

## Implementation Tasks

### Week 1: Setup & Core Infrastructure

- [ ] Create service structure
  - `src/qtrader/services/indicators/__init__.py`
  - `src/qtrader/services/indicators/interface.py`
  - `src/qtrader/services/indicators/service.py`
- [ ] Copy indicators from master branch
  - `src/qtrader/indicators/` (all files)
- [ ] Define `IIndicator` and `IIndicatorService` protocols
- [ ] Refactor indicator base class if needed (remove Context dependency)
- [ ] Implement `IndicatorService` class

### Week 2: Indicator Migration & Testing

- [ ] Update all indicators to work with service
  - SMA, EMA
  - RSI, MACD
  - ATR, Bollinger Bands
- [ ] Write unit tests for each indicator
- [ ] Write integration tests for service
- [ ] Test caching behavior
- [ ] Test warmup/reset functionality

### Week 3: Integration & Documentation

- [ ] Integrate with Context (Phase 6)
- [ ] Add convenience methods to Context
- [ ] Migration guide for strategies
- [ ] API documentation
- [ ] Performance benchmarks

## Testing Strategy

### Unit Tests (Indicator Logic)

```python
def test_sma_calculation():
    """Test SMA computes correctly."""
    service = IndicatorService()
    bars = [create_bar(close=i) for i in range(1, 11)]

    sma = service.sma("AAPL", bars, period=5)

    assert sma == 8.0  # (6+7+8+9+10)/5
```

### Integration Tests (Service Behavior)

```python
def test_indicator_caching():
    """Test indicators are cached per params."""
    service = IndicatorService()
    bars = generate_bars(100)

    # First call creates indicator
    sma1 = service.sma("AAPL", bars, period=20)

    # Second call uses cached indicator
    sma2 = service.sma("AAPL", bars, period=20)

    # Verify same indicator instance used
    assert len(service._indicators) == 1
```

### Mock Usage (In Strategy Tests)

```python
def test_strategy_with_mock_indicators():
    """Test strategy logic with mock indicators."""
    mock_indicators = MockIndicatorService()
    mock_indicators.set_sma("AAPL", 50.0)
    mock_indicators.set_rsi("AAPL", 30.0)

    ctx = Context(indicators=mock_indicators, ...)
    strategy = MyStrategy()

    signals = strategy.on_bar(ctx)

    assert len(signals) == 1  # Should generate buy signal
```

## Validation Criteria

- [ ] ✅ All indicators from master ported
- [ ] ✅ `IIndicatorService` protocol defined
- [ ] ✅ Service implements protocol
- [ ] ✅ Zero dependencies on execution/portfolio/risk
- [ ] ✅ Only depends on: models (Bar), config
- [ ] ✅ Context integration complete
- [ ] ✅ All tests pass (unit + integration)
- [ ] ✅ Test coverage ≥ 90%
- [ ] ✅ Documentation complete
- [ ] ✅ Performance benchmarks met

## Migration Path

### Step 1: Copy Indicators

```bash
git checkout feature/lego-architecture
git checkout master -- src/qtrader/indicators
```

### Step 2: Create Service

- Define interface
- Implement service
- No breaking changes yet

### Step 3: Update Context (Phase 6)

- Inject `IIndicatorService`
- Add convenience methods
- Strategies use `ctx.sma()` etc.

### Step 4: Cleanup

- Remove old IndicatorManager if redundant
- Update all example strategies

## Success Metrics

- [ ] ✅ Can test indicator logic without strategy
- [ ] ✅ Can mock indicators in strategy tests
- [ ] ✅ Caching works correctly
- [ ] ✅ Warmup improves performance
- [ ] ✅ Custom indicators can be registered
- [ ] ✅ Context API is clean and intuitive

## Dependencies

### Depends On

- Phase 1: DataService (for bar history access via Context)
- Phase 6: Strategy Context (indicators injected into Context)

### Blocks

- Strategy implementation (strategies need indicators)

## Risks & Mitigations

| Risk                     | Mitigation                       |
| ------------------------ | -------------------------------- |
| Performance overhead     | Cache aggressively, benchmark    |
| Complex state management | Thorough testing of warmup/reset |
| Breaking API changes     | Migration guide, examples        |

## Next Phase

👉 **[Phase 8: ReportingService](phase8_reporting_service.md)**

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** Phase 1 (DataService), Phase 6 (Context) **Estimated Duration:** 2-3 weeks **Last Updated:** October 15, 2025
