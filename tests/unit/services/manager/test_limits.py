"""Tests for risk limit checking (limits.py)."""

from decimal import Decimal

import pytest

from qtrader.services.manager.limits import check_all_limits, check_concentration_limit, check_leverage_limits
from qtrader.services.manager.models import ConcentrationLimit, LeverageLimit, OrderBase, Position

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def order_buy_100_aapl() -> OrderBase:
    """Order to BUY 100 shares of AAPL."""
    return OrderBase(
        strategy_id="test_strategy",
        symbol="AAPL",
        side="BUY",
        quantity=100,
        reason="Test order",
    )


@pytest.fixture
def order_sell_50_aapl() -> OrderBase:
    """Order to SELL 50 shares of AAPL."""
    return OrderBase(
        strategy_id="test_strategy",
        symbol="AAPL",
        side="SELL",
        quantity=50,
        reason="Test order",
    )


@pytest.fixture
def order_buy_200_spy() -> OrderBase:
    """Order to BUY 200 shares of SPY."""
    return OrderBase(
        strategy_id="test_strategy",
        symbol="SPY",
        side="BUY",
        quantity=200,
        reason="Test order",
    )


@pytest.fixture
def positions_empty() -> list[Position]:
    """Empty position list."""
    return []


@pytest.fixture
def positions_with_aapl() -> list[Position]:
    """Positions with 50 shares of AAPL."""
    return [
        Position(
            symbol="AAPL",
            quantity=50,
            market_value=Decimal("7500"),  # 50 * 150 = 7500
        )
    ]


@pytest.fixture
def positions_multiple() -> list[Position]:
    """Positions with AAPL and SPY."""
    return [
        Position(symbol="AAPL", quantity=100, market_value=Decimal("15000")),
        Position(symbol="SPY", quantity=500, market_value=Decimal("225000")),
    ]


# =============================================================================
# Tests: check_concentration_limit() - Within Limit
# =============================================================================


def test_concentration_limit_within_limit_new_position(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test concentration check passes for new position within limit."""
    # 100 shares * 150 price = 15000 exposure / 100000 equity = 15%
    # Limit is 20%, so within limit
    limit = ConcentrationLimit(max_position_pct=0.20)
    violation = check_concentration_limit(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


def test_concentration_limit_within_limit_add_to_position(
    order_buy_100_aapl: OrderBase,
    positions_with_aapl: list[Position],
) -> None:
    """Test concentration check passes when adding to existing position."""
    # Current: 50 shares, Proposed: 50 + 100 = 150 shares
    # 150 * 150 price = 22500 exposure / 100000 equity = 22.5%
    # Limit is 25%, so within limit
    limit = ConcentrationLimit(max_position_pct=0.25)
    violation = check_concentration_limit(
        order=order_buy_100_aapl,
        current_positions=positions_with_aapl,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


def test_concentration_limit_within_limit_reduce_position(
    order_sell_50_aapl: OrderBase,
    positions_with_aapl: list[Position],
) -> None:
    """Test concentration check passes when reducing position."""
    # Current: 50 shares, Proposed: 50 - 50 = 0 shares (flat)
    # 0 * 150 price = 0 exposure / 100000 equity = 0%
    # Always within limit
    limit = ConcentrationLimit(max_position_pct=0.10)
    violation = check_concentration_limit(
        order=order_sell_50_aapl,
        current_positions=positions_with_aapl,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


# =============================================================================
# Tests: check_concentration_limit() - At Limit
# =============================================================================


def test_concentration_limit_at_limit(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test concentration check passes when exactly at limit."""
    # 100 shares * 150 price = 15000 exposure / 100000 equity = 15%
    # Limit is exactly 15%, so at limit (should pass)
    limit = ConcentrationLimit(max_position_pct=0.15)
    violation = check_concentration_limit(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


# =============================================================================
# Tests: check_concentration_limit() - Exceeds Limit
# =============================================================================


def test_concentration_limit_exceeds_limit_new_position(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test concentration check fails for new position exceeding limit."""
    # 100 shares * 150 price = 15000 exposure / 100000 equity = 15%
    # Limit is 10%, so exceeds limit
    limit = ConcentrationLimit(max_position_pct=0.10)
    violation = check_concentration_limit(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is not None
    assert violation.limit_type == "concentration"
    assert violation.symbol == "AAPL"
    assert violation.proposed_pct == pytest.approx(0.15, abs=0.001)
    assert violation.limit_pct == 0.10
    assert "AAPL" in violation.message
    assert "15" in violation.message  # 15%


def test_concentration_limit_exceeds_limit_add_to_position(
    order_buy_100_aapl: OrderBase,
    positions_with_aapl: list[Position],
) -> None:
    """Test concentration check fails when adding to position exceeds limit."""
    # Current: 50 shares, Proposed: 50 + 100 = 150 shares
    # 150 * 150 price = 22500 exposure / 100000 equity = 22.5%
    # Limit is 20%, so exceeds limit
    limit = ConcentrationLimit(max_position_pct=0.20)
    violation = check_concentration_limit(
        order=order_buy_100_aapl,
        current_positions=positions_with_aapl,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is not None
    assert violation.limit_type == "concentration"
    assert violation.symbol == "AAPL"
    assert violation.proposed_pct == pytest.approx(0.225, abs=0.001)
    assert violation.limit_pct == 0.20


# =============================================================================
# Tests: check_concentration_limit() - Edge Cases
# =============================================================================


def test_concentration_limit_zero_equity(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test concentration check handles zero equity."""
    limit = ConcentrationLimit(max_position_pct=0.10)
    violation = check_concentration_limit(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("0"),
        current_price=Decimal("150"),
        limit=limit,
    )
    # Should return violation (cannot have position with zero equity)
    assert violation is not None
    assert violation.proposed_pct == float("inf")
    assert "zero equity" in violation.message


def test_concentration_limit_flattening_position_with_zero_equity(
    order_sell_50_aapl: OrderBase,
    positions_with_aapl: list[Position],
) -> None:
    """Test concentration check allows flattening position even with zero equity."""
    # Selling entire position (50 shares) → proposed = 0 shares
    limit = ConcentrationLimit(max_position_pct=0.10)
    violation = check_concentration_limit(
        order=order_sell_50_aapl,
        current_positions=positions_with_aapl,
        equity=Decimal("0"),  # Zero equity but flattening is OK
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None  # Flattening should be allowed


# =============================================================================
# Tests: check_leverage_limits() - Within Limits
# =============================================================================


def test_leverage_limits_within_limits_new_position(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test leverage check passes for new position within limits."""
    # Proposed gross: 100 * 150 = 15000 / 100000 equity = 0.15 (15%)
    # Proposed net: same (all long)
    # Limits: gross=2.0, net=1.0, so within limits
    limit = LeverageLimit(max_gross=2.0, max_net=1.0)
    violation = check_leverage_limits(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


def test_leverage_limits_within_limits_with_existing_positions(
    order_buy_100_aapl: OrderBase,
    positions_multiple: list[Position],
) -> None:
    """Test leverage check passes with existing positions."""
    # Existing: AAPL (15000) + SPY (225000) = 240000 gross
    # Proposed: AAPL increases by 100 * 150 = 15000
    # New AAPL position: 100 + 100 = 200 shares * 150 = 30000
    # Proposed gross: SPY (225000) + AAPL (30000) = 255000 / 100000 = 2.55
    # Proposed net: same (all long) = 255000 / 100000 = 2.55
    # Limits: gross=3.0 (OK), net=3.0 (OK), so within limits
    limit = LeverageLimit(max_gross=3.0, max_net=3.0)
    violation = check_leverage_limits(
        order=order_buy_100_aapl,
        current_positions=positions_multiple,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


def test_leverage_limits_within_limits_reduce_position(
    order_sell_50_aapl: OrderBase,
    positions_with_aapl: list[Position],
) -> None:
    """Test leverage check passes when reducing position."""
    # Current: 50 shares (7500), Proposed: 0 shares (0)
    # Gross/net both reduce, so always within limits
    limit = LeverageLimit(max_gross=1.0, max_net=1.0)
    violation = check_leverage_limits(
        order=order_sell_50_aapl,
        current_positions=positions_with_aapl,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


# =============================================================================
# Tests: check_leverage_limits() - At Limits
# =============================================================================


def test_leverage_limits_at_gross_limit(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test leverage check passes when exactly at gross limit."""
    # Proposed gross: 100 * 150 = 15000 / 100000 equity = 0.15
    # Limit is exactly 0.15, so at limit (should pass)
    limit = LeverageLimit(max_gross=0.15, max_net=1.0)
    violation = check_leverage_limits(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is None


# =============================================================================
# Tests: check_leverage_limits() - Exceeds Limits
# =============================================================================


def test_leverage_limits_exceeds_gross_limit(
    order_buy_100_aapl: OrderBase,
    positions_multiple: list[Position],
) -> None:
    """Test leverage check fails when gross leverage exceeded."""
    # Existing: AAPL (15000) + SPY (225000) = 240000
    # Proposed AAPL: (100 + 100) * 150 = 30000
    # Proposed gross: SPY (225000) + AAPL (30000) = 255000 / 100000 = 2.55
    # Limit is 2.0, so exceeds gross limit
    limit = LeverageLimit(max_gross=2.0, max_net=10.0)
    violation = check_leverage_limits(
        order=order_buy_100_aapl,
        current_positions=positions_multiple,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is not None
    assert violation.limit_type == "leverage"
    assert violation.symbol == "PORTFOLIO"
    assert violation.proposed_pct == pytest.approx(2.55, abs=0.01)
    assert violation.limit_pct == 2.0
    assert "Gross leverage" in violation.message


def test_leverage_limits_exceeds_net_limit(
    order_buy_100_aapl: OrderBase,
    positions_multiple: list[Position],
) -> None:
    """Test leverage check fails when net leverage exceeded."""
    # All positions are long, so net = gross
    # Proposed gross/net: 255000 / 100000 = 2.55
    # Gross limit is 10.0 (OK), net limit is 2.0, so exceeds net limit
    limit = LeverageLimit(max_gross=10.0, max_net=2.0)
    violation = check_leverage_limits(
        order=order_buy_100_aapl,
        current_positions=positions_multiple,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is not None
    assert violation.limit_type == "leverage"
    assert violation.symbol == "PORTFOLIO"
    assert violation.proposed_pct == pytest.approx(2.55, abs=0.01)
    assert violation.limit_pct == 2.0
    assert "Net leverage" in violation.message


def test_leverage_limits_gross_checked_before_net(
    order_buy_100_aapl: OrderBase,
    positions_multiple: list[Position],
) -> None:
    """Test that gross leverage violation is reported first (more severe)."""
    # Proposed gross/net: 255000 / 100000 = 2.55
    # Both limits are 2.0, so both exceeded, but gross should be reported first
    limit = LeverageLimit(max_gross=2.0, max_net=2.0)
    violation = check_leverage_limits(
        order=order_buy_100_aapl,
        current_positions=positions_multiple,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    assert violation is not None
    assert "Gross leverage" in violation.message  # Gross reported first


# =============================================================================
# Tests: check_leverage_limits() - Edge Cases
# =============================================================================


def test_leverage_limits_zero_equity(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test leverage check handles zero equity."""
    limit = LeverageLimit(max_gross=2.0, max_net=1.0)
    violation = check_leverage_limits(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("0"),
        current_price=Decimal("150"),
        limit=limit,
    )
    # Should return violation (cannot have position with zero equity)
    assert violation is not None
    assert violation.proposed_pct == float("inf")
    assert "zero equity" in violation.message


def test_leverage_limits_short_position() -> None:
    """Test leverage limits with short positions."""
    # Short 100 AAPL (qty = -100)
    order = OrderBase(
        strategy_id="test",
        symbol="AAPL",
        side="SELL",
        quantity=100,
        reason="short",
    )
    positions: list[Position] = []

    # Proposed: -100 shares * 150 = -15000 exposure
    # Gross: 15000 (absolute), Net: 15000 (absolute of net)
    # 15000 / 100000 = 0.15
    limit = LeverageLimit(max_gross=0.10, max_net=1.0)
    violation = check_leverage_limits(
        order=order,
        current_positions=positions,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    # Exceeds gross limit (0.15 > 0.10)
    assert violation is not None
    assert "Gross leverage" in violation.message


def test_leverage_limits_long_short_portfolio() -> None:
    """Test leverage limits with both long and short positions."""
    # Long 500 SPY (225000), Short 100 AAPL
    positions = [
        Position(symbol="SPY", quantity=500, market_value=Decimal("225000")),
        Position(symbol="AAPL", quantity=-50, market_value=Decimal("-7500")),
    ]

    # Buy 50 more AAPL (reducing short to 0)
    order = OrderBase(
        strategy_id="test",
        symbol="AAPL",
        side="BUY",
        quantity=50,
        reason="cover short",
    )

    # Current gross: 225000 + 7500 = 232500
    # Proposed AAPL: -50 + 50 = 0 (flat)
    # Proposed gross: 225000 + 0 = 225000 / 100000 = 2.25
    # Proposed net: 225000 + 0 = 225000 / 100000 = 2.25
    limit = LeverageLimit(max_gross=2.0, max_net=10.0)
    violation = check_leverage_limits(
        order=order,
        current_positions=positions,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        limit=limit,
    )
    # Exceeds gross limit (2.25 > 2.0)
    assert violation is not None
    assert "Gross leverage" in violation.message


# =============================================================================
# Tests: check_all_limits()
# =============================================================================


def test_check_all_limits_no_violations(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test check_all_limits returns empty list when no violations."""
    violations = check_all_limits(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        concentration_limit=ConcentrationLimit(max_position_pct=0.20),
        leverage_limit=LeverageLimit(max_gross=2.0, max_net=1.0),
    )
    assert len(violations) == 0


def test_check_all_limits_concentration_violation_only(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test check_all_limits returns only concentration violation."""
    # 100 * 150 = 15000 / 100000 = 15% (exceeds 10% concentration)
    # But 15% < 2.0 leverage, so leverage OK
    violations = check_all_limits(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        concentration_limit=ConcentrationLimit(max_position_pct=0.10),
        leverage_limit=LeverageLimit(max_gross=2.0, max_net=1.0),
    )
    assert len(violations) == 1
    assert violations[0].limit_type == "concentration"


def test_check_all_limits_leverage_violation_only(
    order_buy_100_aapl: OrderBase,
    positions_multiple: list[Position],
) -> None:
    """Test check_all_limits returns only leverage violation."""
    # Proposed gross: 255000 / 100000 = 2.55 (exceeds 2.0 leverage)
    # AAPL concentration: 30000 / 100000 = 30% (but limit is 40%, so OK)
    violations = check_all_limits(
        order=order_buy_100_aapl,
        current_positions=positions_multiple,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        concentration_limit=ConcentrationLimit(max_position_pct=0.40),
        leverage_limit=LeverageLimit(max_gross=2.0, max_net=10.0),
    )
    assert len(violations) == 1
    assert violations[0].limit_type == "leverage"


def test_check_all_limits_both_violations(
    order_buy_100_aapl: OrderBase,
    positions_multiple: list[Position],
) -> None:
    """Test check_all_limits returns both violations."""
    # AAPL concentration: 30000 / 100000 = 30% (exceeds 20%)
    # Gross leverage: 255000 / 100000 = 2.55 (exceeds 2.0)
    violations = check_all_limits(
        order=order_buy_100_aapl,
        current_positions=positions_multiple,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        concentration_limit=ConcentrationLimit(max_position_pct=0.20),
        leverage_limit=LeverageLimit(max_gross=2.0, max_net=10.0),
    )
    assert len(violations) == 2
    assert violations[0].limit_type == "concentration"
    assert violations[1].limit_type == "leverage"


def test_check_all_limits_skip_concentration_if_none(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test check_all_limits skips concentration check if limit is None."""
    # Even though concentration would be violated (15% > 10%),
    # it's not checked because concentration_limit=None
    violations = check_all_limits(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        concentration_limit=None,
        leverage_limit=LeverageLimit(max_gross=2.0, max_net=1.0),
    )
    assert len(violations) == 0


def test_check_all_limits_skip_leverage_if_none(
    order_buy_100_aapl: OrderBase,
    positions_multiple: list[Position],
) -> None:
    """Test check_all_limits skips leverage check if limit is None."""
    # Even though leverage would be violated (2.55 > 2.0),
    # it's not checked because leverage_limit=None
    violations = check_all_limits(
        order=order_buy_100_aapl,
        current_positions=positions_multiple,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        concentration_limit=ConcentrationLimit(max_position_pct=0.40),
        leverage_limit=None,
    )
    assert len(violations) == 0


def test_check_all_limits_skip_both_if_none(
    order_buy_100_aapl: OrderBase,
    positions_empty: list[Position],
) -> None:
    """Test check_all_limits skips all checks if both limits are None."""
    violations = check_all_limits(
        order=order_buy_100_aapl,
        current_positions=positions_empty,
        equity=Decimal("100000"),
        current_price=Decimal("150"),
        concentration_limit=None,
        leverage_limit=None,
    )
    assert len(violations) == 0
