# Custom Indicator & Strategy Implementation Summary

## Overview

Successfully implemented a complete custom indicator (Bollinger Bands) and a custom strategy that uses it, demonstrating the full extensibility pattern for the QTrader library system.

## What Was Built

### 1. Custom Indicator: Bollinger Bands (`my_library/indicators/bollinger_bands.py`)

**Features:**

- Volatility indicator with upper/middle/lower bands
- Uses SMA composition (demonstrates indicator reuse)
- Returns multi-value dict: `{"upper": float, "middle": float, "lower": float}`
- Additional computed properties:
  - `bandwidth`: Volatility measure `(upper - lower) / middle`
  - `percent_b`: Price position within bands `(price - lower) / (upper - lower)`

**Implementation Highlights:**

```python
class BollingerBands(BaseIndicator):
    def __init__(self, period=20, num_std=2.0, price_field="close"):
        self._sma = SMA(period=period)  # Composition pattern
        self._prices = deque(maxlen=period)

    def calculate(self, bars) -> list[dict[str, float] | None]:
        # Batch calculation with std dev

    def update(self, bar) -> dict[str, float] | None:
        # Incremental update
        middle = self._sma.value
        std_dev = (variance ** 0.5)
        return {
            "upper": middle + std_dev * num_std,
            "middle": middle,
            "lower": middle - std_dev * num_std
        }

    @property
    def bandwidth(self) -> float | None:
        # Volatility measure

    @property
    def percent_b(self) -> float | None:
        # Price position (0-1 scale, can exceed range)
```

**Documentation:**

- Complete theory explanation
- Formulas and calculations
- Usage examples
- Trading interpretation
- Typical parameter settings

### 2. Custom Strategy: Bollinger Breakout (`my_library/strategies/bollinger_breakout.py`)

**Strategy Logic:**

- **Entry Signals:**

  - BUY when %B < oversold_threshold (default: 0.0 = below lower band)
  - SELL when %B > overbought_threshold (default: 1.0 = above upper band)

- **Exit Signals:**

  - EXIT when %B between 0.4-0.6 (mean reversion to middle band)

- **Volatility Filter:**

  - Only trade when bandwidth > min_bandwidth (default: 2%)
  - Prevents trading in low-volatility environments

**Configuration (Process/Parameter Separation):**

```python
class BollingerBreakoutConfig(BaseStrategyConfig):
    # Identity
    name: str = ...
    display_name: str = ...

    # Indicator parameters
    bb_period: int = 20
    bb_num_std: float = 2.0

    # Entry thresholds
    oversold_threshold: float = 0.0
    overbought_threshold: float = 1.0

    # Risk filters
    min_bandwidth: float = 0.02  # 2%

    # Position sizing
    position_size_pct: float = 0.1  # 10% of portfolio
```

**Lifecycle Hooks:**

```python
def setup(self, context):
    # Logs configuration
    # Validates thresholds
    # Raises errors if invalid setup

def teardown(self, context):
    # Logs final statistics
    # Reports final positions
```

### 3. Test Suite: Bollinger Bands (`examples/indicators/test_bollinger_bands.py`)

**Test Coverage:**

1. **Basic Calculation** - OHLCV processing, band computation
1. **Stateless Batch** - `calculate()` method, warmup period
1. **Volatility Detection** - Bandwidth analysis across regimes
1. **Trading Signals** - %B-based signal generation
1. **Parameter Sensitivity** - Different period/std configurations
1. **Reset Functionality** - State management

**Results:**

```
================================================================================
TEST 1: Basic Bollinger Bands Calculation
================================================================================
Processing 50 bars with period=20, num_std=2.0

Bar 49:
  Price:  108.30
  Upper:  111.06
  Middle: 107.93
  Lower:  104.81
  Width:  0.0579
  %B:     0.5584

================================================================================
TEST 3: Volatility Detection (Bandwidth Analysis)
================================================================================
LOW Volatility:
  Bandwidth: 0.0444 (4.44%)

NORMAL Volatility:
  Bandwidth: 0.0579 (5.79%)

HIGH Volatility:
  Bandwidth: 0.1059 (10.59%)

All tests completed successfully!
```

## Key Design Patterns Demonstrated

### 1. Indicator Composition

```python
# BollingerBands reuses built-in SMA
self._sma = SMA(period=period, price_field=price_field)
```

### 2. Multi-Value Returns

```python
# Instead of single float, return dict
return {
    "upper": upper_band,
    "middle": middle_band,
    "lower": lower_band
}
```

### 3. Additional Properties

```python
# Derived metrics beyond base interface
@property
def bandwidth(self) -> float | None:
    return (upper - lower) / middle

@property
def percent_b(self) -> float | None:
    return (close - lower) / (upper - lower)
```

### 4. Process/Parameter Separation

```python
# Strategy class = PROCESS (algorithm)
class BollingerBreakoutStrategy(BaseStrategy):
    def on_bar(self, event, context):
        # Logic uses config values
        if percent_b < self._config.oversold_threshold:
            ...

# Config class = PARAMETERS (tunable values)
class BollingerBreakoutConfig(BaseStrategyConfig):
    oversold_threshold: float = 0.0  # Easily changed for optimization
```

### 5. Type Safety

```python
# Proper type hints for type checker
self._config: BollingerBreakoutConfig = config

# Handle None values from warmup
if bar is None:
    return
```

## File Structure

```
my_library/
├── indicators/
│   ├── __init__.py              # Exports BollingerBands
│   └── bollinger_bands.py       # Full implementation (269 lines)
│       ├── BollingerBands class
│       ├── calculate() method
│       ├── update() method
│       ├── bandwidth property
│       └── percent_b property
│
└── strategies/
    ├── __init__.py              # Exports BollingerBreakoutConfig, BollingerBreakoutStrategy
    └── bollinger_breakout.py    # Full implementation (218 lines)
        ├── BollingerBreakoutConfig class
        ├── BollingerBreakoutStrategy class
        ├── setup() lifecycle hook
        ├── teardown() lifecycle hook
        └── on_bar() trading logic

examples/indicators/
└── test_bollinger_bands.py      # Comprehensive test suite (280 lines)
    ├── 6 test scenarios
    ├── create_sample_bars() helper
    └── Validation of all features
```

## Integration with QTrader Architecture

### Indicator Integration

- ✅ Inherits from `BaseIndicator` ABC
- ✅ Implements all required methods (calculate, update, reset, value, is_ready)
- ✅ Type-safe (returns `list[dict[str, float] | None]`)
- ✅ Stateful and stateless modes supported
- ✅ Follows warmup contract (returns None until ready)

### Strategy Integration

- ✅ Inherits from `BaseStrategy` ABC
- ✅ Uses `BaseStrategyConfig` for parameters
- ✅ Implements lifecycle hooks (setup, teardown)
- ✅ Process/parameter separation pattern
- ✅ Type-safe event handling (`PriceBarEvent`)
- ✅ Proper None handling (bar can be None)

### Built-in Library Composition

- ✅ BollingerBands uses built-in SMA internally
- ✅ Demonstrates code reuse across library
- ✅ Shows how custom indicators can leverage existing components

## Trading Interpretation

### Bollinger Bands Theory

- **Upper Band**: Middle + (2 × StdDev) - Resistance level
- **Middle Band**: SMA(20) - Trend baseline
- **Lower Band**: Middle - (2 × StdDev) - Support level

### Signal Generation

- **%B < 0.0**: Price below lower band (oversold, potential BUY)
- **%B > 1.0**: Price above upper band (overbought, potential SELL)
- **0.4 < %B < 0.6**: Price near middle (mean reversion EXIT)

### Volatility Filter

- **High Bandwidth**: Bands wide apart (high volatility, good for breakouts)
- **Low Bandwidth**: Bands narrow (low volatility, "squeeze", avoid trading)
- **Squeeze → Expansion**: Classic setup for breakout trades

## Next Steps

### Immediate Enhancements

1. **More Custom Indicators**:

   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - ATR (Average True Range)
   - Stochastic Oscillator

1. **Strategy Improvements**:

   - Add position sizing based on bandwidth (larger positions in high volatility)
   - Implement stop-loss based on ATR
   - Add trend filter (only trade with trend)

1. **Backtesting**:

   - Create config YAML for strategy
   - Run backtest with actual market data
   - Analyze performance metrics
   - Optimize parameters

### Registry System

1. **Auto-Discovery**:

   - Scan `my_library/indicators` and `my_library/strategies`
   - Register by name
   - Validate ABC compliance

1. **Dynamic Loading**:

   - Load indicators/strategies from config
   - Support multiple instances with different parameters
   - Enable A/B testing

1. **Validation**:

   - Check all required methods implemented
   - Verify type signatures
   - Test warmup behavior

## Lessons Learned

### Type System

- **Issue**: Pylance doesn't support covariant config types in strategies
- **Solution**: Use `self._config: SpecificConfig = config` for type safety
- **Benefit**: Full IDE support with proper type hints

### Indicator Composition

- **Pattern**: Custom indicators can use built-in indicators
- **Example**: BollingerBands uses SMA internally
- **Benefit**: Code reuse, DRY principle, easier maintenance

### Multi-Value Indicators

- **Pattern**: Return dict instead of single float
- **Example**: `{"upper": ..., "middle": ..., "lower": ...}`
- **Benefit**: More expressive, avoids separate indicator instances

### Process/Parameter Separation

- **Critical Insight**: Strategy class = algorithm, Config class = parameters
- **Benefit**: Same strategy, different configs = parameter optimization
- **Example**: Change oversold_threshold without touching strategy code

## Conclusion

Successfully demonstrated complete custom indicator and strategy implementation:

- ✅ BaseIndicator ABC compliance
- ✅ BaseStrategy ABC compliance
- ✅ Indicator composition (BB uses SMA)
- ✅ Multi-value returns (dict support)
- ✅ Additional properties (bandwidth, %B)
- ✅ Process/parameter separation
- ✅ Lifecycle hooks (setup/teardown)
- ✅ Type safety throughout
- ✅ Comprehensive testing
- ✅ Full documentation

The pattern is ready for users to create their own indicators and strategies following the same structure.
