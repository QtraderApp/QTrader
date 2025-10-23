# Signal-Based Strategy Architecture

## Overview

QTrader strategies follow a **signal-based architecture** that separates concerns between strategy logic, risk management, and execution.

## Key Principles

### 1. Separation of Concerns

**Strategies emit signals, not orders.**

- **Strategy**: Analyzes market data and declares trading intent with confidence levels
- **RiskService**: Evaluates signals, calculates position sizes, applies risk limits
- **ExecutionService**: Places and manages orders

## Signal Confidence

Strategies emit signals with confidence levels between `0.0` and `1.0`:

```python
context.emit_signal(
    timestamp=bar.trade_datetime,
    strategy_id=self.config.name,
    symbol="AAPL",
    intention=SignalIntention.OPEN_LONG,  # or CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT
    confidence=0.75,                      # 0.0 to 1.0
    reason="Oversold on RSI",             # optional explanation
    metadata={...}                        # optional data
)
```

### 3. No Position Management in Strategies

❌ **Wrong** - Strategy tracks positions and calculates sizes:

```python
class BadStrategy(BaseStrategy):
    def __init__(self, config):
        self._positions = {}  # DON'T DO THIS
        self._position_size_pct = 0.1  # DON'T DO THIS

    def on_bar(self, event, context):
        if signal:
            size = self._position_size_pct * portfolio_value
            context.place_order(symbol, size)  # WRONG
```

✅ **Right** - Strategy emits signals with confidence:

```python
class GoodStrategy(BaseStrategy):
    def on_bar(self, event, context):
        if signal:
            confidence = calculate_confidence(signal_strength)
            context.emit_signal(
                symbol=symbol,
                direction="BUY",
                confidence=confidence,
                reason="Strategy logic triggered"
            )
```

## Context Interface

Strategies receive a `Context` object that provides:

### Signal Emission

```python
context.emit_signal(
    timestamp: datetime,
    strategy_id: str,
    symbol: str,
    intention: SignalIntention | str,  # OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT
    confidence: float,  # 0.0 to 1.0
    reason: str | None = None,
    metadata: dict | None = None
) -> Signal
```

### Data Access

```python
# Get current position (read-only)
position = context.get_position(symbol)

# Get historical bars
bars = context.get_bars(symbol, n=20)

# Get current price
price = context.get_price(symbol)
```

## SignalEvent Structure

Signals are defined in the `qtrader.contracts.strategies` module:

```python
from qtrader.contracts.strategies import Signal, SignalIntention

@dataclass(frozen=True)
class Signal(BaseModel):
    timestamp: datetime
    strategy_id: str
    symbol: str
    intention: SignalIntention  # OPEN_LONG, CLOSE_LONG, OPEN_SHORT, CLOSE_SHORT
    confidence: float  # [0.0, 1.0]
    reason: Optional[str]
    metadata: Optional[Dict[str, Any]]

class SignalIntention(str, Enum):
    OPEN_LONG = "OPEN_LONG"      # Initiate/add to long position
    CLOSE_LONG = "CLOSE_LONG"    # Close/reduce long position
    OPEN_SHORT = "OPEN_SHORT"    # Initiate/add to short position
    CLOSE_SHORT = "CLOSE_SHORT"  # Close/reduce short position
```

## Example: Bollinger Breakout Strategy

```python
from qtrader.libraries.strategies.base import BaseStrategy, BaseStrategyConfig
from qtrader.events.events import PriceBarEvent

class BollingerBreakoutConfig(BaseStrategyConfig):
    name: str = "bb_breakout"
    display_name: str = "Bollinger Breakout"
    bb_period: int = 20
    bb_num_std: float = 2.0
    max_confidence: float = 0.9  # Cap signal confidence

class BollingerBreakoutStrategy(BaseStrategy):
    def __init__(self, config: BollingerBreakoutConfig):
        super().__init__(config)
        self._bb = BollingerBands(config.bb_period, config.bb_num_std)

    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        # Update indicator
        bands = self._bb.update(event.bar)

        if not self._bb.is_ready:
            return

        # Calculate confidence from signal strength
        percent_b = self._bb.percent_b
        bandwidth = self._bb.bandwidth

        # Emit OPEN_LONG signal when price below lower band
        if percent_b < 0.0:
            distance = abs(percent_b)
            confidence = self._calculate_confidence(distance, bandwidth)

            context.emit_signal(
                timestamp=event.bar.trade_datetime,
                strategy_id=self.config.name,
                symbol=event.symbol,
                intention=SignalIntention.OPEN_LONG,
                confidence=confidence,
                reason=f"Oversold: %B={percent_b:.3f}",
                metadata={
                    "percent_b": percent_b,
                    "bandwidth": bandwidth,
                    "price": event.bar.close,
                }
            )

        # Emit OPEN_SHORT signal when price above upper band
        elif percent_b > 1.0:
            distance = percent_b - 1.0
            confidence = self._calculate_confidence(distance, bandwidth)

            context.emit_signal(
                timestamp=event.bar.trade_datetime,
                strategy_id=self.config.name,
                symbol=event.symbol,
                intention=SignalIntention.OPEN_SHORT,
                confidence=confidence,
                reason=f"Overbought: %B={percent_b:.3f}",
                metadata={
                    "percent_b": percent_b,
                    "bandwidth": bandwidth,
                    "price": event.bar.close,
                }
            )

    def _calculate_confidence(self, distance: float, bandwidth: float) -> float:
        """Calculate signal confidence from distance and volatility."""
        # More extreme = higher confidence
        base_confidence = min(abs(distance) * 2.0, 1.0)

        # Higher volatility = more trending = more confident
        volatility_factor = min(bandwidth / 0.05, 1.0)

        # Combine factors
        confidence = base_confidence * (0.7 + 0.3 * volatility_factor)

        # Cap at max_confidence
        return min(confidence, self._config.max_confidence)
```

## Benefits

1. **Testability**: Strategies can be tested in isolation without risk/execution dependencies
1. **Composability**: Multiple strategies can emit signals independently
1. **Flexibility**: Risk management can be changed without modifying strategies
1. **Clarity**: Clear separation between "what to trade" and "how much to trade"
1. **Reusability**: Same strategy can be used with different risk profiles

## Strategy Registry

Strategies are auto-discovered via the registry system:

```python
from qtrader.libraries.registry import StrategyRegistry

# Discover strategies
registry = StrategyRegistry()
registry.discover(
    buildin_path=Path("src/qtrader/libraries/strategies/buildin"),
    custom_paths=[Path("my_library/strategies")]
)

# List available
print(registry.list_names())
# ['bollingerbreakoutstrategy', ...]

# Get strategy class
BollingerBreakout = registry.get("bollingerbreakoutstrategy")
config = BollingerBreakoutConfig(...)
strategy = BollingerBreakout(config)
```

## Next Steps

1. Implement risk sizing logic in `RiskService`
1. Connect signals to order placement in `ExecutionService`
1. Add built-in strategies to `libraries/strategies/buildin/`
1. Implement multi-strategy signal aggregation
