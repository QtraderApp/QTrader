# Risk Management System - User Guide

**Phase: Stage 5B - Risk Management**\
**Status: Complete (53/53 tests passing)**\
**Date: October 2024**

## Overview

The Risk Management system provides centralized position sizing, concentration limits, and cash management for trading strategies. It follows a **Signal-based workflow** where strategies express trading intent (WHAT to trade) and the RiskManager determines position size (HOW MUCH to trade).

## Architecture

```
Strategy → Signal → RiskManager → Order → ExecutionEngine
         (intent)   (sizing)      (filled)
```

### Key Components

1. **Signal**: Trading intent without position sizing
1. **RiskPolicy**: Configuration for sizing methods and limits
1. **RiskManager**: Evaluates signals and creates sized orders
1. **Context**: Strategy API for portfolio access and signal evaluation

## Signal Model

Signals represent **trading intent** before position sizing:

```python
from qtrader.risk import Signal, SignalType, SignalDirection
from datetime import datetime, timezone
from decimal import Decimal

signal = Signal(
    signal_id="entry_AAPL_001",
    strategy_ts=datetime(2024, 1, 15, 9, 30, tzinfo=timezone.utc),
    symbol="AAPL",
    signal_type=SignalType.ENTRY_LONG,
    direction=SignalDirection.LONG,

    # Sizing hints (optional - RiskManager may override)
    target_qty=None,           # Specific quantity
    target_weight=Decimal("0.10"),  # 10% of portfolio
    target_value=None,         # Dollar amount

    # Order preferences
    order_type=OrderType.MARKET,
    limit_price=None,
    stop_price=None,

    # Risk context
    conviction=Decimal("0.85"),  # 0.0-1.0 confidence
    urgency="normal",            # normal|high|low
    metadata={"reason": "Momentum breakout"},
)

# Validate before submission
signal.validate()  # Raises ValueError if invalid
```

### Signal Types

- `ENTRY_LONG`: Open long position
- `ENTRY_SHORT`: Open short position
- `EXIT_LONG`: Close long position
- `EXIT_SHORT`: Close short position
- `REBALANCE`: Adjust existing position

### Signal Direction

- `LONG`: Target long position
- `SHORT`: Target short position
- `FLAT`: Close all positions

## Risk Policy

Configures sizing methods and limits:

```python
from qtrader.risk import RiskPolicy, SizingMethod
from decimal import Decimal

policy = RiskPolicy(
    # Position sizing method
    sizing_method=SizingMethod.PORTFOLIO_PERCENT,
    default_position_size=Decimal("0.10"),  # 10% of portfolio

    # Concentration limits
    max_position_pct=Decimal("0.15"),  # Max 15% per position
    max_positions=10,                   # Max 10 concurrent positions

    # Leverage constraints
    max_gross_exposure=Decimal("1.0"),  # 100% gross (long + short)
    max_net_exposure=Decimal("1.0"),    # 100% net (long - short)
    allow_shorting=False,               # Long only

    # Safety margins
    cash_reserve_pct=Decimal("0.05"),   # Keep 5% cash reserve

    # Rejection policies
    reject_on_insufficient_cash=True,
    reject_on_concentration_breach=True,
    reject_on_leverage_breach=True,
)

# Validate policy
policy.validate()  # Raises ValueError if invalid
```

### Multi-Strategy Allocation: Cash-First Check

**New in Stage 5B**: `check_cash_before_concentration` parameter

**The Problem:** In multi-strategy portfolios, concentration limits are based on **current equity** (which shrinks as cash is spent). As equity drops, concentration limits get tighter, inadvertently sizing positions to fit available cash. Later strategies get approved with reduced sizes instead of being rejected.

**Example:**

- 5 strategies, each with 25% max concentration
- After 4 strategies fill ($100k → $20k cash), equity drops
- 5th strategy: Concentration limit now 25% of $20k = $5k (down from $25k)
- Result: Approved with small size instead of rejected

**Solution:** Set `check_cash_before_concentration=True` to check cash **before** concentration adjustment:

```python
# Default behavior: Dynamic risk management (single strategy)
policy_dynamic = RiskPolicy(
    sizing_method=SizingMethod.FIXED_VALUE,
    default_position_size=Decimal("25000.00"),
    max_position_pct=Decimal("0.25"),
    check_cash_before_concentration=False,  # Default
)
# Result: Signals approved with reduced sizes to fit cash

# Fair allocation: Multi-strategy fairness
policy_fair = RiskPolicy(
    sizing_method=SizingMethod.FIXED_VALUE,
    default_position_size=Decimal("25000.00"),
    max_position_pct=Decimal("0.25"),
    check_cash_before_concentration=True,  # Enable cash-first
)
# Result: Signals rejected with "Insufficient cash" message
```

**When to Use:**

- `False` (default): Single strategy, dynamic risk management, conservative sizing
- `True`: Multiple strategies, fair allocation, clear rejection signals

See `docs/risk_concentration_vs_cash_analysis.md` for detailed analysis.

### Sizing Methods (Phase 1)

1. **FIXED_QUANTITY**: Fixed number of shares

   - `default_position_size`: Number of shares (e.g., 100)

1. **FIXED_VALUE**: Fixed dollar amount

   - `default_position_size`: Dollar amount (e.g., 10000.00)

1. **PORTFOLIO_PERCENT**: Percentage of equity

   - `default_position_size`: 0.0-1.0 (e.g., 0.10 = 10%)

1. **RISK_PERCENT**: Risk-based sizing (requires stop)

   - `default_position_size`: 0.0-1.0 (e.g., 0.02 = risk 2% per trade)
   - Signal must include `stop_price`

### Phase 2 Methods (Deferred)

- `VOLATILITY_TARGET`: Size based on volatility
- `KELLY_CRITERION`: Optimal Kelly sizing
- `EQUAL_RISK_CONTRIBUTION`: Risk parity

## Risk Manager

Evaluates signals and creates sized orders:

```python
from qtrader.risk import RiskManager
from qtrader.models import Portfolio

# Create portfolio
portfolio = Portfolio(initial_cash=Decimal("100000.00"))

# Create risk manager
manager = RiskManager(policy=policy, portfolio=portfolio)

# Evaluate signal
decision = manager.evaluate_signal(signal, current_price=Decimal("150.00"))

if decision.approved:
    print(f"Approved: {decision.sized_qty} shares")
    print(f"Limits applied: {decision.applied_limits}")

    # Convert to order
    order = manager.signal_to_order(signal, decision, Decimal("150.00"))
    # Submit order to ExecutionEngine...
else:
    print(f"Rejected: {decision.reason}")

# Get statistics
stats = manager.get_stats()
print(f"Approval rate: {stats['approval_rate']}")
```

### Evaluation Process

RiskManager evaluates signals in 6 steps:

1. **Validate Signal**: Check signal validity
1. **Check Direction**: Verify short selling allowed if needed
1. **Check Constraints**: Portfolio-level limits (leverage, exposure)
1. **Calculate Size**: Apply sizing method
1. **Apply Concentration**: Enforce max_position_pct, max_positions
1. **Check Cash**: Verify sufficient funds (including reserve)

## Strategy Integration

### Phase 2: Signal-Based API

Strategies now return signals from `on_bar()`:

```python
from typing import List, Optional
from qtrader.api.context import Context
from qtrader.models.bar import Bar
from qtrader.risk import Signal, SignalType, SignalDirection

class MyStrategy:
    def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
        """Return list of signals (or None)."""
        signals = []

        # Get current position
        position = ctx.get_position(bar.symbol)

        # Generate signal based on logic
        if self.should_enter(bar) and position.qty == 0:
            signal = Signal(
                signal_id=f"entry_{bar.symbol}_{bar.ts.isoformat()}",
                strategy_ts=bar.ts,
                symbol=bar.symbol,
                signal_type=SignalType.ENTRY_LONG,
                direction=SignalDirection.LONG,
                target_weight=Decimal("0.10"),  # 10% position
                conviction=Decimal("0.75"),
                urgency="normal",
            )
            signals.append(signal)

        return signals if signals else None
```

### Context API

Access portfolio state and risk management:

```python
# Portfolio queries
equity = ctx.get_equity()
cash = ctx.get_cash()
position = ctx.get_position("AAPL")

# Signal evaluation
decision = ctx.evaluate_signal(signal)
if decision.approved:
    order = ctx.signal_to_order(signal, decision)
```

## Configuration Examples

### Conservative (10% per position, 5 max)

```python
policy = RiskPolicy(
    sizing_method=SizingMethod.PORTFOLIO_PERCENT,
    default_position_size=Decimal("0.10"),
    max_position_pct=Decimal("0.12"),
    max_positions=5,
    cash_reserve_pct=Decimal("0.10"),  # 10% reserve
    allow_shorting=False,
)
```

### Aggressive (20% per position, 10 max)

```python
policy = RiskPolicy(
    sizing_method=SizingMethod.PORTFOLIO_PERCENT,
    default_position_size=Decimal("0.20"),
    max_position_pct=Decimal("0.25"),
    max_positions=10,
    cash_reserve_pct=Decimal("0.05"),  # 5% reserve
    allow_shorting=True,
    max_gross_exposure=Decimal("1.5"),  # 150% gross
)
```

### Fixed Dollar Amount

```python
policy = RiskPolicy(
    sizing_method=SizingMethod.FIXED_VALUE,
    default_position_size=Decimal("10000.00"),  # $10k per position
    max_position_pct=Decimal("0.20"),
    cash_reserve_pct=Decimal("0.05"),
)
```

### Risk-Based Sizing

```python
policy = RiskPolicy(
    sizing_method=SizingMethod.RISK_PERCENT,
    default_position_size=Decimal("0.02"),  # Risk 2% per trade
    max_position_pct=Decimal("0.15"),
    cash_reserve_pct=Decimal("0.05"),
)

# Signal must include stop_price
signal = Signal(
    signal_type=SignalType.ENTRY_LONG,
    direction=SignalDirection.LONG,
    stop_price=Decimal("145.00"),  # Required for RISK_PERCENT
    # ... other fields
)
```

## Testing

### Unit Tests (45 tests)

```bash
# All unit tests
uv run pytest tests/unit/risk/ -v

# Specific component
uv run pytest tests/unit/risk/test_manager.py -v
```

### Integration Tests (8 tests)

```bash
# All integration tests
uv run pytest tests/integration/test_risk_workflow.py -v
```

### All Risk Tests (53 tests)

```bash
uv run pytest tests/unit/risk/ tests/integration/test_risk_workflow.py -v
```

## Example Usage

See `examples/risk_signal_example.py` for complete working example:

```bash
uv run python examples/risk_signal_example.py
```

## Migration Guide

### From Direct Order Submission

**Before** (Phase 1):

```python
def on_bar(self, bar, ctx):
    if self.should_buy():
        ctx.buy_market(qty=100)  # Fixed quantity
```

**After** (Phase 2):

```python
def on_bar(self, bar, ctx) -> Optional[List[Signal]]:
    if self.should_buy():
        signal = Signal(
            signal_id=...,
            strategy_ts=bar.ts,
            symbol=bar.symbol,
            signal_type=SignalType.ENTRY_LONG,
            direction=SignalDirection.LONG,
            # No qty - RiskManager sizes it!
        )
        return [signal]
    return None
```

## Best Practices

1. **Validate Signals**: Always call `signal.validate()` before evaluation
1. **Check Decisions**: Always check `decision.approved` before creating orders
1. **Use Hints**: Provide `target_weight` or `target_qty` as hints to RiskManager
1. **Set Conviction**: Use conviction (0.0-1.0) to express signal strength
1. **Monitor Stats**: Check `manager.get_stats()` for approval rates
1. **Test Policies**: Validate policies with `policy.validate()`
1. **Reserve Cash**: Always set `cash_reserve_pct > 0` to handle edge cases

## Implementation Details

### Code Structure

```
src/qtrader/risk/
├── __init__.py      # Package exports
├── signal.py        # Signal model (150 lines)
├── policy.py        # RiskPolicy config (120 lines)
├── sizing.py        # Sizing methods (230 lines)
└── manager.py       # RiskManager (420 lines)
```

### Key Design Decisions

1. **Signal-First**: Strategies express intent, not sizing
1. **Centralized**: Single RiskManager per portfolio
1. **Flexible Sizing**: Multiple methods (4 Phase 1 + 3 Phase 2)
1. **Conservative**: Concentration limits always applied
1. **Portfolio-Scoped**: Multi-strategy ready
1. **Validation**: Early validation prevents runtime errors
1. **Logging**: Comprehensive logging for audit trail

### Performance Considerations

- Signal validation: O(1)
- Size calculation: O(1)
- Concentration check: O(n) where n = number of positions
- Cash check: O(1)
- Overall: O(n) per signal evaluation

### Thread Safety

**Not thread-safe**. RiskManager should be used from single thread. For multi-strategy scenarios, each strategy should have its own RiskManager instance or use a thread-safe wrapper.

## Troubleshooting

### Signal Rejected: "Shorting not allowed"

- Set `allow_shorting=True` in policy
- Or change signal direction to LONG/FLAT

### Signal Rejected: "Insufficient cash"

- Reduce `default_position_size`
- Reduce `cash_reserve_pct`
- Add capital to portfolio

### Signal Rejected: "Maximum positions reached"

- Increase `max_positions`
- Exit existing positions
- Set `max_positions=None` for unlimited

### Signal Rejected: "concentration limits"

- Increase `max_position_pct`
- Reduce `default_position_size`
- Set `reject_on_concentration_breach=False` (not recommended)

### Sized Quantity Less Than Expected

- Check `applied_limits` in RiskDecision
- Concentration limit may have reduced size
- Verify `max_position_pct` is appropriate

## Phase 2 Roadmap

Future enhancements (not yet implemented):

1. **Advanced Sizing Methods**:

   - Volatility targeting
   - Kelly criterion
   - Equal risk contribution

1. **Dynamic Limits**:

   - Time-of-day adjustments
   - Volatility-based limits
   - Correlation-aware concentration

1. **Pre-trade Analysis**:

   - Trade impact estimation
   - Opportunity cost calculation
   - Portfolio optimization

1. **Risk Metrics**:

   - VaR calculation
   - Expected shortfall
   - Sharpe optimization

1. **Multi-Strategy**:

   - Strategy-level limits
   - Cross-strategy correlation
   - Risk allocation

## References

- Implementation Plan: `docs/implementation_plan_phase01.md`
- Risk Design: `docs/risk_management_design.md`
- Architecture: `docs/architecture.md`
- Unit Tests: `tests/unit/risk/`
- Integration Tests: `tests/integration/test_risk_workflow.py`
- Example: `examples/risk_signal_example.py`

## Support

For questions or issues:

1. Check test examples in `tests/unit/risk/` and `tests/integration/`
1. Review example strategy in `examples/risk_signal_example.py`
1. Check logs for detailed rejection reasons
1. Validate signals and policies before evaluation
