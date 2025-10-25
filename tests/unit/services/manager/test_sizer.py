"""Tests for position sizing (sizer.py)."""

from decimal import Decimal

import pytest

from qtrader.services.manager.models import Signal
from qtrader.services.manager.sizer import FixedFractionSizer, size_position

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def signal_buy_full() -> Signal:
    """Signal with full strength (1.0) for BUY."""
    return Signal(
        strategy_id="test_strategy",
        symbol="AAPL",
        side="BUY",
        strength=1.0,
    )


@pytest.fixture
def signal_buy_half() -> Signal:
    """Signal with half strength (0.5) for BUY."""
    return Signal(
        strategy_id="test_strategy",
        symbol="AAPL",
        side="BUY",
        strength=0.5,
    )


@pytest.fixture
def signal_sell_full() -> Signal:
    """Signal with full strength (1.0) for SELL."""
    return Signal(
        strategy_id="test_strategy",
        symbol="AAPL",
        side="SELL",
        strength=1.0,
    )


@pytest.fixture
def signal_zero_strength() -> Signal:
    """Signal with zero strength."""
    return Signal(
        strategy_id="test_strategy",
        symbol="AAPL",
        side="BUY",
        strength=0.0,
    )


# =============================================================================
# Tests: size_position() Function
# =============================================================================


def test_size_position_basic_buy(signal_buy_full: Signal) -> None:
    """Test basic position sizing for BUY signal."""
    # 10000 * 0.02 * 1.0 = 200 notional / 150 price = 1.33 shares → 1 share
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 1


def test_size_position_basic_sell(signal_sell_full: Signal) -> None:
    """Test basic position sizing for SELL signal (same as BUY)."""
    # Direction doesn't affect sizing, only signal.side matters
    quantity = size_position(
        signal=signal_sell_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 1


def test_size_position_half_strength(signal_buy_half: Signal) -> None:
    """Test position sizing with 50% signal strength."""
    # 10000 * 0.02 * 0.5 = 100 notional / 150 price = 0.67 shares → 0 shares
    quantity = size_position(
        signal=signal_buy_half,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 0


def test_size_position_zero_strength(signal_zero_strength: Signal) -> None:
    """Test position sizing with zero strength returns 0."""
    quantity = size_position(
        signal=signal_zero_strength,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 0


def test_size_position_large_capital(signal_buy_full: Signal) -> None:
    """Test position sizing with large capital."""
    # 100000 * 0.10 * 1.0 = 10000 notional / 450 price = 22.22 shares → 22 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("100000"),
        current_price=Decimal("450"),
        fraction=Decimal("0.10"),
    )
    assert quantity == 22


def test_size_position_small_capital(signal_buy_full: Signal) -> None:
    """Test position sizing with small capital."""
    # 1000 * 0.02 * 1.0 = 20 notional / 150 price = 0.13 shares → 0 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("1000"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 0


def test_size_position_lot_size_100(signal_buy_full: Signal) -> None:
    """Test position sizing with lot_size=100 (e.g., options contracts)."""
    # 100000 * 0.10 * 1.0 = 10000 notional / 450 price = 22.22 shares
    # Round to 100: floor(22.22 / 100) * 100 = 0 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("100000"),
        current_price=Decimal("450"),
        fraction=Decimal("0.10"),
        lot_size=100,
    )
    assert quantity == 0

    # Larger capital to get to 100 shares
    # 1000000 * 0.10 * 1.0 = 100000 notional / 450 price = 222.22 shares
    # Round to 100: floor(222.22 / 100) * 100 = 200 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("1000000"),
        current_price=Decimal("450"),
        fraction=Decimal("0.10"),
        lot_size=100,
    )
    assert quantity == 200


def test_size_position_min_quantity_enforced(signal_buy_full: Signal) -> None:
    """Test position sizing enforces minimum quantity."""
    # 10000 * 0.02 * 1.0 = 200 notional / 150 price = 1.33 shares → 1 share
    # But min_quantity=5, so return 0
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
        min_quantity=5,
    )
    assert quantity == 0


def test_size_position_min_quantity_met(signal_buy_full: Signal) -> None:
    """Test position sizing when minimum quantity is met."""
    # 100000 * 0.02 * 1.0 = 2000 notional / 150 price = 13.33 shares → 13 shares
    # min_quantity=5, so return 13
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("100000"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
        min_quantity=5,
    )
    assert quantity == 13


def test_size_position_zero_price(signal_buy_full: Signal) -> None:
    """Test position sizing with zero price returns 0."""
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("0"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 0


def test_size_position_precision(signal_buy_full: Signal) -> None:
    """Test position sizing maintains Decimal precision."""
    # 10000.5678 * 0.0234 * 1.0 = 234.0132852 notional / 12.345 price
    # = 18.955... shares → 18 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000.5678"),
        current_price=Decimal("12.345"),
        fraction=Decimal("0.0234"),
    )
    assert quantity == 18


def test_size_position_negative_strength_uses_abs(signal_buy_full: Signal) -> None:
    """Test position sizing uses absolute value of strength."""
    # Create signal with negative strength (should work same as positive)
    signal = Signal(
        strategy_id="test_strategy",
        symbol="AAPL",
        side="BUY",
        strength=-0.8,
    )
    # 10000 * 0.10 * 0.8 = 800 notional / 100 price = 8 shares
    quantity = size_position(
        signal=signal,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("100"),
        fraction=Decimal("0.10"),
    )
    assert quantity == 8


# =============================================================================
# Tests: size_position() Validation
# =============================================================================


def test_size_position_invalid_allocated_capital_negative(
    signal_buy_full: Signal,
) -> None:
    """Test size_position raises ValueError for negative allocated capital."""
    with pytest.raises(ValueError, match="allocated_capital must be non-negative"):
        size_position(
            signal=signal_buy_full,
            allocated_capital=Decimal("-1000"),
            current_price=Decimal("150"),
            fraction=Decimal("0.02"),
        )


def test_size_position_invalid_price_negative(signal_buy_full: Signal) -> None:
    """Test size_position raises ValueError for negative price."""
    with pytest.raises(ValueError, match="current_price must be non-negative"):
        size_position(
            signal=signal_buy_full,
            allocated_capital=Decimal("10000"),
            current_price=Decimal("-150"),
            fraction=Decimal("0.02"),
        )


def test_size_position_invalid_fraction_negative(signal_buy_full: Signal) -> None:
    """Test size_position raises ValueError for negative fraction."""
    with pytest.raises(ValueError, match="fraction must be non-negative"):
        size_position(
            signal=signal_buy_full,
            allocated_capital=Decimal("10000"),
            current_price=Decimal("150"),
            fraction=Decimal("-0.02"),
        )


def test_size_position_invalid_lot_size_zero(signal_buy_full: Signal) -> None:
    """Test size_position raises ValueError for lot_size=0."""
    with pytest.raises(ValueError, match="lot_size must be positive"):
        size_position(
            signal=signal_buy_full,
            allocated_capital=Decimal("10000"),
            current_price=Decimal("150"),
            fraction=Decimal("0.02"),
            lot_size=0,
        )


def test_size_position_invalid_lot_size_negative(signal_buy_full: Signal) -> None:
    """Test size_position raises ValueError for negative lot_size."""
    with pytest.raises(ValueError, match="lot_size must be positive"):
        size_position(
            signal=signal_buy_full,
            allocated_capital=Decimal("10000"),
            current_price=Decimal("150"),
            fraction=Decimal("0.02"),
            lot_size=-1,
        )


def test_size_position_invalid_min_quantity_negative(signal_buy_full: Signal) -> None:
    """Test size_position raises ValueError for negative min_quantity."""
    with pytest.raises(ValueError, match="min_quantity must be non-negative"):
        size_position(
            signal=signal_buy_full,
            allocated_capital=Decimal("10000"),
            current_price=Decimal("150"),
            fraction=Decimal("0.02"),
            min_quantity=-1,
        )


# =============================================================================
# Tests: FixedFractionSizer Class
# =============================================================================


def test_fixed_fraction_sizer_initialization() -> None:
    """Test FixedFractionSizer initialization."""
    sizer = FixedFractionSizer(
        fraction=Decimal("0.02"),
        lot_size=1,
        min_quantity=0,
    )
    assert sizer.fraction == Decimal("0.02")
    assert sizer.lot_size == 1
    assert sizer.min_quantity == 0


def test_fixed_fraction_sizer_defaults() -> None:
    """Test FixedFractionSizer uses default lot_size and min_quantity."""
    sizer = FixedFractionSizer(fraction=Decimal("0.02"))
    assert sizer.fraction == Decimal("0.02")
    assert sizer.lot_size == 1
    assert sizer.min_quantity == 0


def test_fixed_fraction_sizer_basic_sizing(signal_buy_full: Signal) -> None:
    """Test FixedFractionSizer basic position sizing."""
    sizer = FixedFractionSizer(fraction=Decimal("0.02"))
    quantity = sizer.size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("150"),
    )
    assert quantity == 1


def test_fixed_fraction_sizer_with_lot_size(signal_buy_full: Signal) -> None:
    """Test FixedFractionSizer with custom lot_size."""
    sizer = FixedFractionSizer(fraction=Decimal("0.10"), lot_size=100)
    # 1000000 * 0.10 * 1.0 = 100000 notional / 450 price = 222.22 shares → 200
    quantity = sizer.size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("1000000"),
        current_price=Decimal("450"),
    )
    assert quantity == 200


def test_fixed_fraction_sizer_with_min_quantity(signal_buy_full: Signal) -> None:
    """Test FixedFractionSizer with custom min_quantity."""
    sizer = FixedFractionSizer(fraction=Decimal("0.02"), min_quantity=5)
    # 10000 * 0.02 * 1.0 = 200 notional / 150 price = 1.33 shares → 1 share
    # But min_quantity=5, so return 0
    quantity = sizer.size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("150"),
    )
    assert quantity == 0


def test_fixed_fraction_sizer_repr() -> None:
    """Test FixedFractionSizer string representation."""
    sizer = FixedFractionSizer(
        fraction=Decimal("0.02"),
        lot_size=100,
        min_quantity=10,
    )
    repr_str = repr(sizer)
    assert "FixedFractionSizer" in repr_str
    assert "0.02" in repr_str
    assert "100" in repr_str
    assert "10" in repr_str


# =============================================================================
# Tests: FixedFractionSizer Validation
# =============================================================================


def test_fixed_fraction_sizer_invalid_fraction_zero() -> None:
    """Test FixedFractionSizer raises ValueError for fraction=0."""
    with pytest.raises(ValueError, match="fraction must be in"):
        FixedFractionSizer(fraction=Decimal("0"))


def test_fixed_fraction_sizer_invalid_fraction_negative() -> None:
    """Test FixedFractionSizer raises ValueError for negative fraction."""
    with pytest.raises(ValueError, match="fraction must be in"):
        FixedFractionSizer(fraction=Decimal("-0.02"))


def test_fixed_fraction_sizer_invalid_fraction_too_high() -> None:
    """Test FixedFractionSizer raises ValueError for fraction > 1."""
    with pytest.raises(ValueError, match="fraction must be in"):
        FixedFractionSizer(fraction=Decimal("1.01"))


def test_fixed_fraction_sizer_invalid_lot_size_zero() -> None:
    """Test FixedFractionSizer raises ValueError for lot_size=0."""
    with pytest.raises(ValueError, match="lot_size must be positive"):
        FixedFractionSizer(fraction=Decimal("0.02"), lot_size=0)


def test_fixed_fraction_sizer_invalid_lot_size_negative() -> None:
    """Test FixedFractionSizer raises ValueError for negative lot_size."""
    with pytest.raises(ValueError, match="lot_size must be positive"):
        FixedFractionSizer(fraction=Decimal("0.02"), lot_size=-1)


def test_fixed_fraction_sizer_invalid_min_quantity_negative() -> None:
    """Test FixedFractionSizer raises ValueError for negative min_quantity."""
    with pytest.raises(ValueError, match="min_quantity must be non-negative"):
        FixedFractionSizer(fraction=Decimal("0.02"), min_quantity=-1)


# =============================================================================
# Tests: Edge Cases
# =============================================================================


def test_size_position_very_small_price(signal_buy_full: Signal) -> None:
    """Test position sizing with very small price (penny stock)."""
    # 10000 * 0.02 * 1.0 = 200 notional / 0.01 price = 20000 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("0.01"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 20000


def test_size_position_very_large_price(signal_buy_full: Signal) -> None:
    """Test position sizing with very large price (expensive stock)."""
    # 10000 * 0.02 * 1.0 = 200 notional / 10000 price = 0.02 shares → 0 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("10000"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 0


def test_size_position_fraction_one(signal_buy_full: Signal) -> None:
    """Test position sizing with fraction=1.0 (all capital)."""
    # 10000 * 1.0 * 1.0 = 10000 notional / 100 price = 100 shares
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("10000"),
        current_price=Decimal("100"),
        fraction=Decimal("1.0"),
    )
    assert quantity == 100


def test_size_position_zero_capital(signal_buy_full: Signal) -> None:
    """Test position sizing with zero capital returns 0."""
    quantity = size_position(
        signal=signal_buy_full,
        allocated_capital=Decimal("0"),
        current_price=Decimal("150"),
        fraction=Decimal("0.02"),
    )
    assert quantity == 0
