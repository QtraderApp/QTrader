# Multi-Strategy Position Tracking Refactoring

## Overview

This document describes the refactoring of the Portfolio Service to support multiple strategies holding positions in the same symbol independently, with proper attribution and isolation.

## Problem Statement

**Original Issue**: The portfolio service was keying positions by symbol only:

```python
_positions: dict[str, Position]  # Keyed by symbol
```

**Impact**: When multiple strategies traded the same symbol:

1. Only the first strategy's `strategy_id` was stamped on the position
1. Subsequent fills from other strategies collapsed into the same position
1. Attribution was lost, breaking strategy group snapshots
1. ManagerService per-strategy risk checks became unreliable

## Solution

### Core Data Structure Changes

Changed position and lot tracker dictionaries to use `(strategy_id, symbol)` tuple keys:

```python
# Before
_positions: dict[str, Position]
_lot_tracker: dict[str, LotTracker]

# After
_positions: dict[tuple[str, str], Position]  # (strategy_id, symbol)
_lot_tracker: dict[tuple[str, str], LotTracker]  # (strategy_id, symbol)
```

**Key Format**: `(strategy_id, symbol)` where `strategy_id` defaults to `"unattributed"` if None

### Modified Files

#### 1. `src/qtrader/services/portfolio/service.py`

**Data Structures** (Lines 80-82):

- Updated `_positions` and `_lot_tracker` type annotations

**Position Management Methods**:

- `_has_long_position()`, `_has_short_position()`: Added `strategy_id` parameter
- `_open_long_position()`, `_open_short_position()`: Use tuple keying `(strat_id, symbol)`
- `_close_long_position()`, `_close_short_position()`: Updated signatures and keying
- `apply_fill()`: Passes strategy_id to all position operations

**Query Methods**:

- `get_position()`: Added optional `strategy_id` parameter
- `get_positions()`: Returns `dict[tuple[str, str], Position]`
- `get_all_lots()`: Added optional `strategy_id` filter

**Price Updates**:

- `update_prices()`: Iterates over all strategies for each symbol
- `mark_to_market()`: Updated to unpack tuple keys

**Corporate Actions**:

- `process_split()`: Applies to all strategies holding the symbol
- `process_dividend()`: Applies proportionally to all strategies

**State Management**:

- `get_snapshot()`: Serializes keys as `"strategy_id:symbol"` strings
- `restore_from_snapshot()`: Parses keys back to tuples
- `_publish_portfolio_state()`: Groups positions by strategy correctly

#### 2. `src/qtrader/services/portfolio/models.py`

**PortfolioState Model** (Line 298):

```python
positions: dict[tuple[str, str], Position]  # Was dict[str, Position]
```

#### 3. Test Files Updated

- `tests/unit/services/portfolio/test_service.py`: Updated assertions to use tuple keys
- `tests/unit/services/portfolio/test_state_management.py`: Updated snapshot/restore tests
- **NEW**: `tests/unit/services/portfolio/test_multi_strategy.py`: 8 comprehensive multi-strategy tests

## API Changes

### Backward Compatibility

**Breaking Changes**:

1. `get_positions()` now returns `dict[tuple[str, str], Position]` instead of `dict[str, Position]`
1. Snapshot format changed: keys are now `"strategy_id:symbol"` instead of `"symbol"`

**Additions**:

- `get_position(symbol, strategy_id=None)`: Optional strategy_id parameter
- `get_all_lots(symbol=None, strategy_id=None)`: Optional strategy_id filter

### Migration Guide

**Before**:

```python
positions = portfolio.get_positions()
aapl_position = positions["AAPL"]
```

**After**:

```python
# Option 1: Access by tuple key
positions = portfolio.get_positions()
aapl_position = positions[("unattributed", "AAPL")]

# Option 2: Use get_position with strategy_id
aapl_position = portfolio.get_position("AAPL", "strategy_a")

# Option 3: Iterate over all positions
for (strategy_id, symbol), position in positions.items():
    print(f"{strategy_id} holds {position.quantity} of {symbol}")
```

## Test Coverage

### Original Tests: 110 tests (all passing)

- Position management
- Lot tracking (FIFO/LIFO)
- Corporate actions
- Fees and commissions
- State snapshots
- Query methods

### New Multi-Strategy Tests: 8 tests (all passing)

1. **test_two_strategies_same_symbol_independent_positions**: Verifies two strategies can independently hold the same symbol
1. **test_strategy_close_does_not_affect_other_strategy**: Closing one strategy's position leaves others intact
1. **test_unattributed_strategy_default**: Fills without strategy_id use "unattributed"
1. **test_snapshot_preserves_strategy_attribution**: Snapshot correctly serializes multi-strategy positions
1. **test_restore_preserves_strategy_attribution**: Restore correctly deserializes multi-strategy positions
1. **test_corporate_actions_apply_to_all_strategies**: Stock splits apply to all strategies holding symbol
1. **test_dividends_apply_to_all_strategies**: Dividends apply proportionally to all strategies
1. **test_get_all_lots_filtered_by_strategy**: Lot queries can filter by strategy

**Total Test Suite**: 118 tests, 100% passing

## Benefits

1. **Proper Attribution**: Each strategy's positions are completely isolated
1. **Accurate Strategy Groups**: Portfolio state events contain correct per-strategy positions
1. **Reliable Risk Checks**: ManagerService can accurately assess per-strategy exposure
1. **Corporate Actions**: Splits and dividends correctly apply across all strategies
1. **Backward Compatible API**: Most existing code works with minimal changes

## Performance Considerations

- Tuple keys have minimal memory overhead
- Dictionary lookups remain O(1)
- Iteration over positions unchanged in complexity
- Snapshot serialization adds minimal string formatting

## Future Enhancements

Potential improvements:

1. Add strategy-level PnL tracking
1. Add per-strategy risk metrics in portfolio state
1. Support strategy hierarchies (parent/child strategies)
1. Add strategy-level allocation limits

## Conclusion

This refactoring successfully addresses the multi-strategy attribution bug while maintaining backward compatibility where possible. The comprehensive test suite ensures correctness and provides confidence for future development.
