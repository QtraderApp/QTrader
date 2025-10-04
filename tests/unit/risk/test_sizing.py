"""Tests for position sizing methods."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.models.portfolio import Portfolio
from qtrader.risk.policy import RiskPolicy, SizingMethod
from qtrader.risk.signal import Signal, SignalDirection, SignalType
from qtrader.risk.sizing import calculate_position_size


@pytest.fixture
def portfolio():
    """Create a portfolio with $100,000 cash."""
    return Portfolio(initial_cash=Decimal("100000.00"))


@pytest.fixture
def signal_long():
    """Create a basic LONG signal."""
    return Signal(
        signal_id="sig-001",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_LONG,
        direction=SignalDirection.LONG,
    )


def test_fixed_quantity_with_signal_hint(portfolio, signal_long):
    """Test FIXED_QUANTITY sizing uses signal target_qty."""
    policy = RiskPolicy(sizing_method=SizingMethod.FIXED_QUANTITY)
    signal = signal_long._replace(target_qty=100)

    qty = calculate_position_size(
        signal=signal,
        policy=policy,
        portfolio=portfolio,
        current_price=Decimal("150.00"),
    )

    assert qty == 100


def test_fixed_quantity_with_policy_default(portfolio, signal_long):
    """Test FIXED_QUANTITY sizing uses policy default if no signal hint."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.FIXED_QUANTITY,
        default_position_size=Decimal("200"),  # Treated as quantity
    )

    qty = calculate_position_size(
        signal=signal_long,
        policy=policy,
        portfolio=portfolio,
        current_price=Decimal("150.00"),
    )

    assert qty == 200


def test_fixed_value_with_signal_hint(portfolio, signal_long):
    """Test FIXED_VALUE sizing uses signal target_value."""
    policy = RiskPolicy(sizing_method=SizingMethod.FIXED_VALUE)
    signal = signal_long._replace(target_value=Decimal("15000.00"))
    current_price = Decimal("150.00")

    qty = calculate_position_size(
        signal=signal,
        policy=policy,
        portfolio=portfolio,
        current_price=current_price,
    )

    # 15000 / 150 = 100 shares
    assert qty == 100


def test_fixed_value_with_policy_default(portfolio, signal_long):
    """Test FIXED_VALUE sizing uses policy default if no signal hint."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.FIXED_VALUE,
        default_position_size=Decimal("10000.00"),  # Treated as dollar value
    )
    current_price = Decimal("100.00")

    qty = calculate_position_size(
        signal=signal_long,
        policy=policy,
        portfolio=portfolio,
        current_price=current_price,
    )

    # 10000 / 100 = 100 shares
    assert qty == 100


def test_portfolio_percent_with_signal_hint(portfolio, signal_long):
    """Test PORTFOLIO_PERCENT sizing uses signal target_weight."""
    policy = RiskPolicy(sizing_method=SizingMethod.PORTFOLIO_PERCENT)
    signal = signal_long._replace(target_weight=Decimal("0.10"))  # 10% of equity
    current_price = Decimal("100.00")

    qty = calculate_position_size(
        signal=signal,
        policy=policy,
        portfolio=portfolio,
        current_price=current_price,
    )

    # Equity = 100,000
    # 10% = 10,000
    # 10,000 / 100 = 100 shares
    assert qty == 100


def test_portfolio_percent_with_policy_default(portfolio, signal_long):
    """Test PORTFOLIO_PERCENT sizing uses policy default if no signal hint."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.PORTFOLIO_PERCENT,
        default_position_size=Decimal("0.05"),  # 5% of equity
    )
    current_price = Decimal("100.00")

    qty = calculate_position_size(
        signal=signal_long,
        policy=policy,
        portfolio=portfolio,
        current_price=current_price,
    )

    # Equity = 100,000
    # 5% = 5,000
    # 5,000 / 100 = 50 shares
    assert qty == 50


def test_risk_percent_long_position(portfolio, signal_long):
    """Test RISK_PERCENT sizing for LONG position with stop loss."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.RISK_PERCENT,
        default_position_size=Decimal("0.02"),  # Risk 2% of equity
    )
    current_price = Decimal("100.00")
    stop_price = Decimal("95.00")  # $5 risk per share

    signal = signal_long._replace(stop_price=stop_price)

    qty = calculate_position_size(
        signal=signal,
        policy=policy,
        portfolio=portfolio,
        current_price=current_price,
    )

    # Equity = 100,000
    # Risk amount = 2% = 2,000
    # Risk per share = 100 - 95 = 5
    # Qty = 2,000 / 5 = 400 shares
    assert qty == 400


def test_risk_percent_short_position(portfolio):
    """Test RISK_PERCENT sizing for SHORT position with stop loss."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.RISK_PERCENT,
        default_position_size=Decimal("0.01"),  # Risk 1% of equity
        allow_shorting=True,
    )
    current_price = Decimal("100.00")
    stop_price = Decimal("105.00")  # $5 risk per share (stop above entry)

    signal = Signal(
        signal_id="sig-002",
        strategy_ts=datetime(2023, 1, 1, 9, 30, tzinfo=pytz.UTC),
        symbol="AAPL",
        signal_type=SignalType.ENTRY_SHORT,
        direction=SignalDirection.SHORT,
        stop_price=stop_price,
    )

    qty = calculate_position_size(
        signal=signal,
        policy=policy,
        portfolio=portfolio,
        current_price=current_price,
    )

    # Equity = 100,000
    # Risk amount = 1% = 1,000
    # Risk per share = 105 - 100 = 5
    # Qty = 1,000 / 5 = 200 shares
    assert qty == 200


def test_risk_percent_requires_stop_price(portfolio, signal_long):
    """Test RISK_PERCENT sizing requires stop_price."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.RISK_PERCENT,
        default_position_size=Decimal("0.02"),
    )

    with pytest.raises(ValueError, match="RISK_PERCENT sizing requires stop_price"):
        calculate_position_size(
            signal=signal_long,  # No stop_price
            policy=policy,
            portfolio=portfolio,
            current_price=Decimal("100.00"),
        )


def test_risk_percent_invalid_stop_for_long(portfolio, signal_long):
    """Test RISK_PERCENT rejects stop above entry for LONG."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.RISK_PERCENT,
        default_position_size=Decimal("0.02"),
    )
    signal = signal_long._replace(stop_price=Decimal("110.00"))  # Above entry

    with pytest.raises(ValueError, match="stop_price.*must be below current_price"):
        calculate_position_size(
            signal=signal,
            policy=policy,
            portfolio=portfolio,
            current_price=Decimal("100.00"),
        )


def test_zero_equity_returns_zero_qty(signal_long):
    """Test sizing returns 0 when equity is zero."""
    # Create portfolio with zero cash
    portfolio = Portfolio(initial_cash=Decimal("0.00"))

    policy = RiskPolicy(
        sizing_method=SizingMethod.PORTFOLIO_PERCENT,
        default_position_size=Decimal("0.05"),
    )

    qty = calculate_position_size(
        signal=signal_long,
        policy=policy,
        portfolio=portfolio,
        current_price=Decimal("100.00"),
    )

    assert qty == 0


def test_phase2_method_falls_back_to_portfolio_percent(portfolio, signal_long):
    """Test Phase 2 sizing methods fallback to PORTFOLIO_PERCENT."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.VOLATILITY_TARGET,  # Phase 2 method
        default_position_size=Decimal("0.05"),
    )

    qty = calculate_position_size(
        signal=signal_long,
        policy=policy,
        portfolio=portfolio,
        current_price=Decimal("100.00"),
    )

    # Should fallback to PORTFOLIO_PERCENT logic
    # Equity = 100,000
    # 5% = 5,000
    # 5,000 / 100 = 50 shares
    assert qty == 50


def test_invalid_price_raises_error(portfolio, signal_long):
    """Test sizing raises error for invalid price."""
    policy = RiskPolicy(sizing_method=SizingMethod.FIXED_VALUE)

    with pytest.raises(ValueError, match="Invalid current_price"):
        calculate_position_size(
            signal=signal_long,
            policy=policy,
            portfolio=portfolio,
            current_price=Decimal("0.00"),  # Invalid
        )
