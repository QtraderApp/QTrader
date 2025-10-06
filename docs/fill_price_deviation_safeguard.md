# Fill Price Deviation Safeguard

## Summary

Implemented a safeguard that protects against excessive price deviation between signal generation and order execution. This addresses the quantitative trading issue where position sizing at signal price may not reflect actual fill price.

## Problem Statement

**Scenario:**

- Bar 1 (2019-01-02): Close = $157.92
- Strategy generates BUY signal, sizes position: 90,000 / 157.92 = **569 shares**
- Bar 2 (2019-01-03): Open = $144.01
- Order fills at $144.01
- Actual cost: 569 × 144.01 = **$81,942** (only 82% of intended $90,000)

**The issue:** Overnight gap of **8.81%** means actual capital deployed differs significantly from intended.

## Solution

### Design Decision: Keep Realistic Sizing

We chose **Option 3**: Signal at bar N close, size at bar N close, fill at bar N+1 open.

**Rationale:**

- This mirrors real trading (you size based on what you know)
- No look-ahead bias
- Realistic behavior
- Gaps are part of trading

**Safeguard:** Add a check that **rejects orders** when fill price deviates too much from signal price.

## Implementation

### 1. Added Configuration Parameter

**RiskPolicy** (`src/qtrader/risk/policy.py`):

```python
max_fill_price_deviation_pct: Optional[Decimal] = Decimal("0.10")  # 10% default
```

**ExecutionConfig** (`src/qtrader/execution/config.py`):

```python
max_fill_price_deviation_pct: Optional[Decimal] = Decimal("0.10")  # 10% default
```

### 2. Store Signal Price in Orders

**OrderBase** (`src/qtrader/models/order.py`):

```python
signal_price: Optional[Decimal] = None  # Price when signal was generated
```

**Risk Manager** (`src/qtrader/risk/manager.py`):

```python
# When converting signal to order:
order = Order(
    ...,
    signal_price=current_price,  # Store for deviation checks
)
```

### 3. Check Deviation Before Fill

**Execution Engine** (`src/qtrader/execution/engine.py`):

```python
def _check_fill_price_deviation(
    self, order: OrderBase, fill_price: Decimal, max_deviation_pct: Decimal
) -> tuple[bool, str, float]:
    """Check if fill price deviates too much from signal price."""
    if order.signal_price is None:
        return (True, "No signal price to check", 0.0)

    deviation = abs(fill_price - order.signal_price) / order.signal_price
    deviation_pct = float(deviation)

    if deviation > max_deviation_pct:
        return (False, f"Fill price deviates {deviation_pct:.2%} from signal price", deviation_pct)

    return (True, "Fill price within tolerance", deviation_pct)
```

Applied in `on_bar()` before generating fill:

```python
if decision.should_fill:
    # Check fill price deviation safeguard
    if order.signal_price is not None and self.config.max_fill_price_deviation_pct is not None:
        deviation_check = self._check_fill_price_deviation(
            order, decision.fill_price, self.config.max_fill_price_deviation_pct
        )
        if not deviation_check[0]:
            # REJECT and CANCEL order
            logger.warning("execution_engine.fill_rejected_price_deviation", ...)
            orders_to_remove.append(order_id)
            updated_order = order.with_state(OrderState.CANCELED)
            continue
```

### 4. CLI Integration

**CLI** (`src/qtrader/cli.py`):

```python
# Extract from backtest_config
max_fill_price_dev = backtest_config.get("max_fill_price_deviation_pct")
if max_fill_price_dev is not None and not isinstance(max_fill_price_dev, Decimal):
    max_fill_price_dev = Decimal(str(max_fill_price_dev))
elif max_fill_price_dev is None:
    max_fill_price_dev = Decimal("0.10")  # Default 10%

exec_config = ExecutionConfig(
    ...,
    max_fill_price_deviation_pct=max_fill_price_dev,
)
```

## Usage

### Strategy Configuration

```python
backtest_config = {
    "instruments": [...],
    "initial_cash": 100000.0,
    "position_size": 0.90,
    ...
    "max_fill_price_deviation_pct": 0.05,  # Reject if >5% deviation (strict)
    # Or use default 0.10 (10%) by omitting this line
}
```

### Test Results

**Test 1: Default 10% tolerance (buy_and_hold_strategy.py)**

- Signal price: $157.92
- Fill price: $144.01
- Deviation: 8.81% < 10% ✓
- Result: **Order FILLED** (569 shares @ $144.01)

**Test 2: Strict 5% tolerance (buy_and_hold_strict.py)**

- Signal price: $157.92
- Fill price: $144.01
- Deviation: 8.81% > 5% ✗
- Result: **Order CANCELED**
- Log: `execution_engine.fill_rejected_price_deviation`

## Benefits

1. **Protects against overnight gaps**: Prevents execution when prices gap significantly
1. **Configurable tolerance**: Strategies can set their own risk tolerance
1. **No look-ahead bias**: Maintains realistic backtesting
1. **Clear logging**: Warns when orders are rejected due to deviation
1. **Fail-safe default**: 10% tolerance protects against extreme events by default

## Trade-offs

**Pros:**

- Realistic sizing (no future knowledge)
- Protects against excessive slippage
- Configurable per-strategy

**Cons:**

- May reject valid signals during volatile periods
- Adds complexity to execution logic
- Requires careful tolerance tuning

## Recommendations

- **Conservative strategies**: Use strict tolerance (5%)
- **Aggressive strategies**: Use relaxed tolerance (15-20%)
- **Default**: 10% is reasonable for most strategies
- **Disable**: Set to `None` to disable check entirely
- **Monitor**: Review rejection logs to tune tolerance

## Files Modified

1. `src/qtrader/risk/policy.py` - Added `max_fill_price_deviation_pct`
1. `src/qtrader/execution/config.py` - Added `max_fill_price_deviation_pct`
1. `src/qtrader/models/order.py` - Added `signal_price` field
1. `src/qtrader/risk/manager.py` - Store signal price in orders
1. `src/qtrader/execution/engine.py` - Implement deviation check
1. `src/qtrader/cli.py` - Extract config parameter
1. `src/qtrader/api/backtest.py` - Fixed metadata (filled_orders count)
1. `examples/buy_and_hold_strict.py` - Test strategy with strict tolerance

## Future Enhancements

1. **Historical volatility-based tolerance**: Adjust tolerance based on symbol's typical overnight gaps
1. **Adaptive sizing**: Recalculate size at fill price (requires look-ahead flag)
1. **Partial fills**: Allow partial execution at degraded prices
1. **Alert-only mode**: Log warnings but don't reject (for analysis)
