# Audit Trail Architecture

**Status**: Implemented\
**Phase**: Phase 1 Complete\
**Last Updated**: 2024-03-15

## Overview

Complete audit trail implementation enabling full traceability from strategy signals through order execution. Every order can be traced back to its originating signal and strategy.

## Event Flow & Linking

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           AUDIT TRAIL FLOW                              │
└─────────────────────────────────────────────────────────────────────────┘

Strategy Service                Manager Service              Execution Service
     ↓                                 ↓                            ↓
┌────────────┐                  ┌────────────┐              ┌────────────┐
│SignalEvent │                  │OrderEvent  │              │FillEvent   │
├────────────┤                  ├────────────┤              ├────────────┤
│signal_id ──┼──────────────────→intent_id   │              │            │
│strategy_id─┼──────────────────→source_──   │              │            │
│            │                  │  strategy_id│              │            │
│timestamp   │                  │idempotency_─┼──────┐       │            │
│symbol      │                  │  key        │      │       │            │
│intention   │                  │order_id ────┼──────┼───────→source_──  │
│price       │                  │timestamp    │      │       │  order_id  │
│confidence  │                  │symbol       │      │       │timestamp   │
│            │                  │side         │      │       │symbol      │
└────────────┘                  │quantity     │      │       │fill_price  │
                                │order_type   │      │       └────────────┘
                                └────────────┘      │
                                                    │
                                          Used for replay
                                          protection and
                                          duplicate detection
```

## Field Mappings

### SignalEvent → OrderEvent

| SignalEvent Field | OrderEvent Field          | Purpose                           |
| ----------------- | ------------------------- | --------------------------------- |
| `signal_id`       | `intent_id`               | Links order to originating signal |
| `strategy_id`     | `source_strategy_id`      | Strategy attribution              |
| `symbol`          | `symbol`                  | Instrument identifier             |
| `timestamp`       | (Used in idempotency_key) | Temporal ordering                 |

### OrderEvent → FillEvent

| OrderEvent Field | FillEvent Field   | Purpose                         |
| ---------------- | ----------------- | ------------------------------- |
| `order_id`       | `source_order_id` | Links fill to originating order |
| `symbol`         | `symbol`          | Instrument identifier           |
| `side`           | `side`            | Buy/sell direction              |

## Key Fields

### SignalEvent.signal_id (NEW)

**Type**: `str` (required)\
**Format**: UUID or timestamp-based identifier\
**Purpose**: Unique identifier for each signal, used for audit trail linking

**Example**: `"signal-550e8400-e29b-41d4-a716-446655440001"`

**Usage**:

- Links to `OrderEvent.intent_id` to trace which signal triggered which order
- Enables deduplication of signals in distributed systems
- Provides unique reference for debugging and analysis

**Generation Pattern** (recommended):

```python
signal_id = f"signal-{uuid.uuid4()}"
# OR
signal_id = f"signal-{strategy_id}-{timestamp.isoformat()}"
```

### SignalEvent.strategy_id (EXISTING)

**Type**: `str` (required)\
**Format**: Human-readable strategy identifier\
**Purpose**: Identifies which strategy generated the signal

**Example**: `"sma_crossover"`, `"buy_and_hold"`, `"momentum_reversal"`

**Usage**:

- Copied to `OrderEvent.source_strategy_id` for multi-strategy attribution
- Enables per-strategy performance analysis
- Used in idempotency key generation

### OrderEvent.intent_id (NEW - Phase 1)

**Type**: `str` (required)\
**Format**: Must match `SignalEvent.signal_id`\
**Purpose**: Links order back to originating signal for complete audit trail

**Example**: `"signal-550e8400-e29b-41d4-a716-446655440001"`

**Usage**:

- References `SignalEvent.signal_id` to trace signal → order relationship
- Enables analysis: "Which signals resulted in orders?"
- Enables debugging: "Why was this order placed?"

### OrderEvent.idempotency_key (NEW - Phase 1)

**Type**: `str` (required)\
**Format**: `{strategy_id}-{signal_id}-{timestamp}`\
**Purpose**: Replay protection - prevents duplicate orders from same signal

**Example**: `"sma_crossover-signal-550e8400-e29b-41d4-a716-446655440001-2024-03-15T14:35:22.123Z"`

**Usage**:

- ManagerService checks for existing orders with same idempotency_key
- Prevents duplicate order submission if signal is reprocessed
- Enables safe event replay in distributed systems

**Generation Pattern**:

```python
idempotency_key = f"{signal.strategy_id}-{signal.signal_id}-{signal.timestamp}"
```

### OrderEvent.source_strategy_id (EXISTING)

**Type**: `str` (optional)\
**Format**: Must match `SignalEvent.strategy_id`\
**Purpose**: Strategy attribution for multi-strategy portfolios

**Example**: `"sma_crossover"`

**Usage**:

- Copied from `SignalEvent.strategy_id`
- Enables per-strategy P&L analysis
- Enables per-strategy risk limits

## Multi-Strategy Scenario

In a system with multiple strategies, the audit trail ensures proper attribution:

```
Strategy: "sma_crossover" (SMA Crossover)
  ↓
  SignalEvent(
    signal_id="signal-001",
    strategy_id="sma_crossover",
    symbol="AAPL",
    intention="OPEN_LONG"
  )
  ↓
  OrderEvent(
    order_id="order-001",
    intent_id="signal-001",           ← Links to signal
    source_strategy_id="sma_crossover", ← Strategy attribution
    idempotency_key="sma_crossover-signal-001-2024-03-15T14:35:22.123Z"
  )
  ↓
  FillEvent(
    fill_id="fill-001",
    source_order_id="order-001",      ← Links to order
    strategy_id="sma_crossover"       ← Preserved attribution
  )

Strategy: "momentum_reversal" (Momentum Reversal)
  ↓
  SignalEvent(
    signal_id="signal-002",
    strategy_id="momentum_reversal",
    symbol="MSFT",
    intention="OPEN_SHORT"
  )
  ↓
  OrderEvent(
    order_id="order-002",
    intent_id="signal-002",           ← Links to signal
    source_strategy_id="momentum_reversal", ← Different strategy
    idempotency_key="momentum_reversal-signal-002-2024-03-15T14:36:10.456Z"
  )
  ↓
  FillEvent(
    fill_id="fill-002",
    source_order_id="order-002",
    strategy_id="momentum_reversal"
  )
```

## Audit Trail Queries

### Query 1: Find all orders from a specific signal

```python
signal_id = "signal-550e8400-e29b-41d4-a716-446655440001"
orders = db.query(OrderEvent).filter(OrderEvent.intent_id == signal_id).all()
```

### Query 2: Find all fills from a specific order

```python
order_id = "order-2024-03-15-001"
fills = db.query(FillEvent).filter(FillEvent.source_order_id == order_id).all()
```

### Query 3: Trace complete signal → order → fill chain

```python
def trace_signal_to_fills(signal_id: str):
    # Find orders from signal
    orders = db.query(OrderEvent).filter(OrderEvent.intent_id == signal_id).all()

    # Find fills from orders
    fills = []
    for order in orders:
        order_fills = db.query(FillEvent).filter(FillEvent.source_order_id == order.order_id).all()
        fills.extend(order_fills)

    return orders, fills
```

### Query 4: Per-strategy performance analysis

```python
def strategy_performance(strategy_id: str, start_date: str, end_date: str):
    # Find all fills attributed to strategy
    fills = db.query(FillEvent).filter(
        FillEvent.strategy_id == strategy_id,
        FillEvent.timestamp >= start_date,
        FillEvent.timestamp <= end_date
    ).all()

    # Calculate P&L
    total_pl = sum(fill.net_value for fill in fills)
    return total_pl
```

## Idempotency & Replay Protection

### Problem: Duplicate Orders

In distributed systems or during replay scenarios, the same signal might be processed multiple times. Without idempotency protection, this would result in duplicate orders.

### Solution: Idempotency Key

The `idempotency_key` field ensures that even if a signal is reprocessed, only one order is created:

```python
def process_signal(signal: SignalEvent) -> Optional[OrderEvent]:
    # Generate idempotency key
    idempotency_key = f"{signal.strategy_id}-{signal.signal_id}-{signal.timestamp}"

    # Check if order already exists
    existing_order = db.query(OrderEvent).filter(
        OrderEvent.idempotency_key == idempotency_key
    ).first()

    if existing_order:
        logger.info(f"Order already exists for signal {signal.signal_id}, skipping")
        return existing_order  # Return existing order, don't create duplicate

    # Create new order
    order = OrderEvent(
        order_id=generate_order_id(),
        intent_id=signal.signal_id,
        idempotency_key=idempotency_key,
        source_strategy_id=signal.strategy_id,
        symbol=signal.symbol,
        ...
    )

    db.add(order)
    db.commit()
    return order
```

### Replay Safety

If the event bus replays events (e.g., after crash recovery):

1. **Signal replayed**: Same `SignalEvent` processed again
1. **Idempotency key generated**: Same key as original
1. **Duplicate detected**: ManagerService finds existing order with same key
1. **No duplicate order**: System skips order creation

This ensures **exactly-once semantics** for order creation, even with **at-least-once** event delivery.

## Implementation Status

### Phase 1: Complete ✅

- [x] Added `signal_id` to `SignalEvent` (required field)
- [x] Added `intent_id` to `OrderEvent` (links to `signal_id`)
- [x] Added `idempotency_key` to `OrderEvent` (replay protection)
- [x] Updated JSON Schema contracts for validation
- [x] Updated Pydantic event models
- [x] Created 18 contract validation tests (all passing)
- [x] Updated 13 existing signal event tests (all passing)
- [x] Updated 24 existing order event tests (all passing)
- [x] Updated example JSON files with proper linking

**Test Results**: 69 tests passing (34 signal + 35 order)

### Phase 2: Pending (Risk Tools Library)

ManagerService will generate `idempotency_key` and `intent_id` when converting signals to orders:

```python
class ManagerService:
    def process_signal(self, signal: SignalEvent) -> OrderEvent:
        # Generate order with proper linking
        order = OrderEvent(
            order_id=self._generate_order_id(),
            intent_id=signal.signal_id,  # Link to signal
            idempotency_key=f"{signal.strategy_id}-{signal.signal_id}-{signal.timestamp}",
            source_strategy_id=signal.strategy_id,  # Strategy attribution
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=signal.symbol,
            side=self._intention_to_side(signal.intention),
            quantity=self._calculate_quantity(signal),  # From risk tools
            order_type="market",
            source_service="manager_service",
        )
        return order
```

## Contract Versions

| Event       | Schema File               | Version | Status     |
| ----------- | ------------------------- | ------- | ---------- |
| SignalEvent | `strategy/signal.v1.json` | v1      | Updated ✅ |
| OrderEvent  | `manager/order.v1.json`   | v1      | Updated ✅ |
| FillEvent   | `execution/fill.v1.json`  | v1      | Stable     |

## Benefits

1. **Complete Traceability**: Every order can be traced to its signal and strategy
1. **Replay Protection**: Idempotency prevents duplicate orders
1. **Multi-Strategy Support**: Proper attribution for portfolio analysis
1. **Debugging**: "Why was this order placed?" → trace back to signal
1. **Performance Analysis**: "How profitable is strategy X?" → filter by strategy_id
1. **Audit Compliance**: Full event history for regulatory requirements

## Next Steps

See `IMPLEMENTATION_PLAN.md` for remaining phases:

- Phase 2: Risk Tools Library (extract pure functions)
- Phase 3: ManagerService Refactor (implement linking logic)
- Phase 4: BacktestEngine Integration
- Phase 5: Integration Testing
- Phase 6: Documentation & Cleanup
