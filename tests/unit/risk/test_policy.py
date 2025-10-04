"""Tests for RiskPolicy model."""

from decimal import Decimal

import pytest

from qtrader.risk.policy import RiskPolicy, SizingMethod


def test_risk_policy_defaults():
    """Test RiskPolicy with default values."""
    policy = RiskPolicy()

    assert policy.sizing_method == SizingMethod.PORTFOLIO_PERCENT
    assert policy.default_position_size == Decimal("0.05")
    assert policy.max_position_pct == Decimal("0.20")
    assert policy.max_positions is None
    assert policy.max_gross_exposure == Decimal("1.0")
    assert policy.max_net_exposure == Decimal("1.0")
    assert policy.allow_shorting is False
    assert policy.cash_reserve_pct == Decimal("0.05")
    assert policy.reject_on_insufficient_cash is True
    assert policy.reject_on_concentration_breach is True
    assert policy.reject_on_leverage_breach is True


def test_risk_policy_custom_values():
    """Test RiskPolicy with custom values."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.FIXED_QUANTITY,
        default_position_size=Decimal("100"),
        max_position_pct=Decimal("0.10"),
        max_positions=5,
        max_gross_exposure=Decimal("1.5"),
        max_net_exposure=Decimal("0.8"),
        allow_shorting=True,
        cash_reserve_pct=Decimal("0.10"),
        reject_on_insufficient_cash=False,
    )

    assert policy.sizing_method == SizingMethod.FIXED_QUANTITY
    assert policy.default_position_size == Decimal("100")
    assert policy.max_position_pct == Decimal("0.10")
    assert policy.max_positions == 5
    assert policy.allow_shorting is True


def test_risk_policy_validation_position_size_range():
    """Test policy validation enforces valid position size range for percentage methods."""
    policy = RiskPolicy(
        sizing_method=SizingMethod.PORTFOLIO_PERCENT,
        default_position_size=Decimal("1.5"),
    )

    with pytest.raises(ValueError, match="default_position_size must be 0.0-1.0"):
        policy.validate()


def test_risk_policy_validation_allows_large_fixed_values():
    """Test policy validation allows large values for FIXED_QUANTITY and FIXED_VALUE."""
    # Should not raise for FIXED_QUANTITY
    policy_qty = RiskPolicy(
        sizing_method=SizingMethod.FIXED_QUANTITY,
        default_position_size=Decimal("1000"),  # 1000 shares
    )
    policy_qty.validate()  # Should not raise

    # Should not raise for FIXED_VALUE
    policy_val = RiskPolicy(
        sizing_method=SizingMethod.FIXED_VALUE,
        default_position_size=Decimal("50000.00"),  # $50,000
    )
    policy_val.validate()  # Should not raise


def test_risk_policy_validation_max_position_pct_range():
    """Test policy validation enforces valid max position pct range."""
    policy = RiskPolicy(max_position_pct=Decimal("1.5"))

    with pytest.raises(ValueError, match="max_position_pct must be 0.0-1.0"):
        policy.validate()


def test_risk_policy_validation_cash_reserve_range():
    """Test policy validation enforces valid cash reserve range."""
    policy = RiskPolicy(cash_reserve_pct=Decimal("1.5"))

    with pytest.raises(ValueError, match="cash_reserve_pct must be 0.0-1.0"):
        policy.validate()


def test_risk_policy_validation_gross_exposure_range():
    """Test policy validation enforces valid gross exposure range."""
    policy = RiskPolicy(max_gross_exposure=Decimal("15.0"))

    with pytest.raises(ValueError, match="max_gross_exposure must be 0.0-10.0"):
        policy.validate()


def test_risk_policy_validation_net_exposure_vs_gross():
    """Test policy validation enforces net <= gross exposure."""
    policy = RiskPolicy(
        max_gross_exposure=Decimal("1.0"),
        max_net_exposure=Decimal("1.5"),  # Invalid: > gross
    )

    with pytest.raises(ValueError, match="max_net_exposure must be"):
        policy.validate()


def test_risk_policy_validation_max_positions_positive():
    """Test policy validation enforces positive max_positions."""
    policy = RiskPolicy(max_positions=-1)

    with pytest.raises(ValueError, match="max_positions must be positive"):
        policy.validate()


def test_risk_policy_validation_accepts_phase2_methods():
    """Test policy validation accepts Phase 2 methods (with warning in manager)."""
    # Phase 2 methods should validate without error
    # (will be handled with fallback in RiskManager)
    policy = RiskPolicy(sizing_method=SizingMethod.VOLATILITY_TARGET)
    policy.validate()  # Should not raise

    policy = RiskPolicy(sizing_method=SizingMethod.KELLY_CRITERION)
    policy.validate()  # Should not raise

    policy = RiskPolicy(sizing_method=SizingMethod.EQUAL_RISK_CONTRIBUTION)
    policy.validate()  # Should not raise
