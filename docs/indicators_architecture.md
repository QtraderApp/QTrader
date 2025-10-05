# Indicators Module Architecture

## Overview

The indicators module is organized by **category** to improve maintainability, discoverability, and scalability. The structure separates **infrastructure** (framework code) from **built-in indicators** (domain-specific implementations).

## Directory Structure

```
src/qtrader/indicators/
├── base.py                  # Infrastructure: Base Indicator class
├── manager.py               # Infrastructure: IndicatorManager (caching)
├── helpers.py               # Infrastructure: 13 helper functions
├── __init__.py              # Public exports
│
├── trend/                   # Category: Trend Indicators
│   ├── __init__.py
│   ├── sma.py              # Simple Moving Average
│   └── ema.py              # Exponential Moving Average
│
├── volatility/              # Category: Volatility Indicators
│   ├── __init__.py
│   ├── atr.py              # Average True Range
│   └── bollinger_bands.py  # Bollinger Bands
│
└── momentum/                # Category: Momentum Indicators
    ├── __init__.py
    ├── rsi.py              # Relative Strength Index
    └── macd.py             # Moving Average Convergence Divergence
```

## Design Principles

### 1. Separation of Concerns

- **Infrastructure** (root level): Framework code that all indicators use

  - `base.py`: Abstract `Indicator[T]` class with lifecycle hooks
  - `manager.py`: `IndicatorManager` for caching and convenience methods
  - `helpers.py`: 13 utility functions for pattern detection

- **Built-in Indicators** (subdirectories): Domain-specific implementations

  - Organized by category (trend, volatility, momentum)
  - Each indicator in its own file
  - Category `__init__.py` exports indicators

### 2. Category-Based Organization

Indicators are grouped by their primary purpose:

- **Trend**: Indicators that identify and measure trends

  - SMA, EMA

- **Volatility**: Indicators that measure price volatility

  - ATR, Bollinger Bands

- **Momentum**: Indicators that measure price momentum/strength

  - RSI, MACD

### 3. Scalability

Adding new indicators is straightforward:

1. Choose appropriate category (or create new one)
1. Create new file in category directory
1. Export from category `__init__.py`
1. Export from main `__init__.py`

Example - adding Stochastic Oscillator:

```python
# src/qtrader/indicators/momentum/stochastic.py
class StochasticOscillator(Indicator[float]):
    ...

# src/qtrader/indicators/momentum/__init__.py
from qtrader.indicators.momentum.stochastic import StochasticOscillator
__all__ = [..., "StochasticOscillator"]

# src/qtrader/indicators/__init__.py
from qtrader.indicators.momentum import ..., StochasticOscillator
__all__ = [..., "StochasticOscillator"]
```

## Public API

All indicators are exposed through `qtrader.api`:

```python
from qtrader.api import (
    # Infrastructure
    Indicator,
    IndicatorManager,

    # Trend
    SMA,
    EMA,

    # Volatility
    ATR,
    BollingerBandsIndicator,
    BollingerBands,

    # Momentum
    RSI,
    MACDIndicator,
    MACD,
    MACDResult,

    # Helpers
    crossed_above,
    crossed_below,
    # ... 11 more helpers
)
```

## Usage Example

```python
from qtrader.api import Strategy, Context

class MyStrategy(Strategy):
    def on_bar(self, bar, ctx: Context):
        # Access indicators via ctx.ind (IndicatorManager)

        # Trend indicators
        sma_20 = ctx.ind.sma(bar.symbol, 20)
        ema_50 = ctx.ind.ema(bar.symbol, 50)

        # Volatility indicators
        atr = ctx.ind.atr(bar.symbol, 14)
        bb = ctx.ind.bollinger_bands(bar.symbol, 20)

        # Momentum indicators
        rsi = ctx.ind.rsi(bar.symbol, 14)
        macd = ctx.ind.macd(bar.symbol)

        # Use helper functions
        ctx._track_indicator(bar.symbol, 'rsi', rsi)
        if ctx.crossed_above_threshold(bar.symbol, 'rsi', 30):
            # RSI crossed above 30 (oversold exit)
            ...
```

## Benefits

### 1. Maintainability

- Each indicator in its own file (easier to test and modify)
- Clear separation between infrastructure and implementations
- Category organization makes code navigation intuitive

### 2. Discoverability

- Users can easily find indicators by category
- Clear module structure in IDE/editor
- Related indicators grouped together

### 3. Testability

- Can test individual indicators in isolation
- Test infrastructure separately from implementations
- Category-based test organization

### 4. Extensibility

- Easy to add new indicators to existing categories
- Easy to add new categories
- Infrastructure remains unchanged when adding indicators

## Future Extensions

Potential new categories:

- **volume/**: Volume-based indicators (OBV, VWAP, Volume Profile)
- **statistical/**: Statistical indicators (Correlation, Covariance, Z-Score)
- **pattern/**: Pattern recognition (Candlestick patterns, Chart patterns)
- **custom/**: User-contributed indicators

## Testing Structure

Tests will mirror the indicators structure:

```
tests/unit/indicators/
├── test_base.py              # Test Indicator base class
├── test_manager.py           # Test IndicatorManager
├── test_helpers.py           # Test helper functions
├── trend/
│   ├── test_sma.py
│   └── test_ema.py
├── volatility/
│   ├── test_atr.py
│   └── test_bollinger_bands.py
└── momentum/
    ├── test_rsi.py
    └── test_macd.py
```

## Migration Notes

- All indicators still exposed through `qtrader.api` (no breaking changes)
- Internal structure changed, but public API remains the same
- Example strategies work without modification
