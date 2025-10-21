"""Tests for risk service data models."""

from datetime import datetime
from decimal import Decimal

import pytest

from qtrader.services.risk.models import (
    ConcentrationLimit,
    LeverageLimit,
    OrderBase,
    PortfolioState,
    Position,
    RiskConfig,
    Signal,
    SizingConfig,
    StrategyBudget,
)

# ============================================================================
# Signal Tests
# ============================================================================


def test_signal_valid_creation():
    """Test creating a valid signal."""
    signal = Signal(
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )
    assert signal.strategy_id == "momentum_v1"
    assert signal.symbol == "AAPL"
    assert signal.side == "BUY"
    assert signal.strength == 0.75
    assert signal.metadata == {}


def test_signal_with_metadata():
    """Test signal with metadata."""
    signal = Signal(
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.5,
        metadata={"price": 150.0, "volatility": 0.25},
    )
    assert signal.metadata["price"] == 150.0
    assert signal.metadata["volatility"] == 0.25


def test_signal_invalid_strength_high():
    """Test signal rejects strength > 1."""
    with pytest.raises(ValueError, match="strength must be in"):
        Signal(
            strategy_id="test",
            symbol="AAPL",
            side="BUY",
            strength=1.5,
        )


def test_signal_invalid_strength_low():
    """Test signal rejects strength < -1."""
    with pytest.raises(ValueError, match="strength must be in"):
        Signal(
            strategy_id="test",
            symbol="AAPL",
            side="BUY",
            strength=-1.5,
        )


def test_signal_zero_strength_valid():
    """Test signal accepts zero strength (no conviction)."""
    signal = Signal(
        strategy_id="test",
        symbol="AAPL",
        side="BUY",
        strength=0.0,
    )
    assert signal.strength == 0.0


def test_signal_empty_strategy_id():
    """Test signal rejects empty strategy_id."""
    with pytest.raises(ValueError, match="strategy_id cannot be empty"):
        Signal(
            strategy_id="",
            symbol="AAPL",
            side="BUY",
            strength=0.5,
        )


def test_signal_empty_symbol():
    """Test signal rejects empty symbol."""
    with pytest.raises(ValueError, match="symbol cannot be empty"):
        Signal(
            strategy_id="test",
            symbol="",
            side="BUY",
            strength=0.5,
        )


def test_signal_invalid_side():
    """Test signal rejects invalid side."""
    with pytest.raises(ValueError, match="side must be 'BUY' or 'SELL'"):
        Signal(
            strategy_id="test",
            symbol="AAPL",
            side="HOLD",  # pyright: ignore[reportArgumentType]
            strength=0.5,
        )


# ============================================================================
# OrderBase Tests
# ============================================================================


def test_orderbase_valid_creation():
    """Test creating a valid order."""
    order = OrderBase(
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        quantity=500,
        reason="Approved: 500 shares, 2% of allocated capital",
    )
    assert order.strategy_id == "momentum_v1"
    assert order.symbol == "AAPL"
    assert order.side == "BUY"
    assert order.quantity == 500
    assert "Approved" in order.reason


def test_orderbase_invalid_quantity_zero():
    """Test order rejects zero quantity."""
    with pytest.raises(ValueError, match="quantity must be positive"):
        OrderBase(
            strategy_id="test",
            symbol="AAPL",
            side="BUY",
            quantity=0,
            reason="test",
        )


def test_orderbase_invalid_quantity_negative():
    """Test order rejects negative quantity."""
    with pytest.raises(ValueError, match="quantity must be positive"):
        OrderBase(
            strategy_id="test",
            symbol="AAPL",
            side="BUY",
            quantity=-100,
            reason="test",
        )


def test_orderbase_empty_reason():
    """Test order rejects empty reason."""
    with pytest.raises(ValueError, match="reason cannot be empty"):
        OrderBase(
            strategy_id="test",
            symbol="AAPL",
            side="BUY",
            quantity=100,
            reason="",
        )


# ============================================================================
# Position Tests
# ============================================================================


def test_position_valid_long():
    """Test creating a long position."""
    pos = Position(
        symbol="AAPL",
        quantity=100,
        market_value=Decimal("15000.00"),
    )
    assert pos.symbol == "AAPL"
    assert pos.quantity == 100
    assert pos.market_value == Decimal("15000.00")


def test_position_valid_short():
    """Test creating a short position."""
    pos = Position(
        symbol="TSLA",
        quantity=-50,
        market_value=Decimal("-10000.00"),
    )
    assert pos.quantity == -50
    assert pos.market_value < 0


def test_position_empty_symbol():
    """Test position rejects empty symbol."""
    with pytest.raises(ValueError, match="symbol cannot be empty"):
        Position(
            symbol="",
            quantity=100,
            market_value=Decimal("1000"),
        )


# ============================================================================
# PortfolioState Tests
# ============================================================================


def test_portfoliostate_valid_creation():
    """Test creating a valid portfolio state."""
    state = PortfolioState(
        ts=datetime(2020, 1, 2, 16, 0),
        equity=Decimal("1000000"),
        cash=Decimal("500000"),
        gross_exposure=Decimal("500000"),
        net_exposure=Decimal("500000"),
        positions={},
    )
    assert state.equity == Decimal("1000000")
    assert state.cash == Decimal("500000")
    assert len(state.positions) == 0


def test_portfoliostate_with_positions():
    """Test portfolio state with positions."""
    pos1 = Position("AAPL", 100, Decimal("15000"))
    pos2 = Position("GOOGL", 50, Decimal("7500"))

    state = PortfolioState(
        ts=datetime(2020, 1, 2, 16, 0),
        equity=Decimal("1000000"),
        cash=Decimal("977500"),
        gross_exposure=Decimal("22500"),
        net_exposure=Decimal("22500"),
        positions={"AAPL": pos1, "GOOGL": pos2},
    )
    assert len(state.positions) == 2
    assert state.positions["AAPL"].quantity == 100


def test_portfoliostate_negative_equity():
    """Test portfolio state rejects negative equity."""
    with pytest.raises(ValueError, match="equity cannot be negative"):
        PortfolioState(
            ts=datetime(2020, 1, 2, 16, 0),
            equity=Decimal("-1000"),
            cash=Decimal("0"),
            gross_exposure=Decimal("0"),
            net_exposure=Decimal("0"),
            positions={},
        )


def test_portfoliostate_negative_gross_exposure():
    """Test portfolio state rejects negative gross exposure."""
    with pytest.raises(ValueError, match="gross_exposure cannot be negative"):
        PortfolioState(
            ts=datetime(2020, 1, 2, 16, 0),
            equity=Decimal("1000000"),
            cash=Decimal("1000000"),
            gross_exposure=Decimal("-100"),
            net_exposure=Decimal("0"),
            positions={},
        )


def test_portfoliostate_negative_cash_allowed():
    """Test portfolio state allows negative cash (margin account)."""
    state = PortfolioState(
        ts=datetime(2020, 1, 2, 16, 0),
        equity=Decimal("1000000"),
        cash=Decimal("-50000"),  # Borrowed cash
        gross_exposure=Decimal("1050000"),
        net_exposure=Decimal("1050000"),
        positions={},
    )
    assert state.cash == Decimal("-50000")


# ============================================================================
# Configuration Tests
# ============================================================================


def test_strategybudget_valid():
    """Test creating a valid strategy budget."""
    budget = StrategyBudget(
        strategy_id="momentum_v1",
        capital_weight=0.3,
    )
    assert budget.strategy_id == "momentum_v1"
    assert budget.capital_weight == 0.3


def test_strategybudget_invalid_weight_high():
    """Test budget rejects weight > 1."""
    with pytest.raises(ValueError, match="capital_weight must be in"):
        StrategyBudget(
            strategy_id="test",
            capital_weight=1.5,
        )


def test_strategybudget_invalid_weight_negative():
    """Test budget rejects negative weight."""
    with pytest.raises(ValueError, match="capital_weight must be in"):
        StrategyBudget(
            strategy_id="test",
            capital_weight=-0.1,
        )


def test_sizingconfig_valid():
    """Test creating a valid sizing config."""
    config = SizingConfig(
        model="fixed_fraction",
        fraction=0.02,
    )
    assert config.model == "fixed_fraction"
    assert config.fraction == 0.02


def test_sizingconfig_invalid_model():
    """Test sizing config rejects invalid model."""
    with pytest.raises(ValueError, match="Phase 4 MVP only supports"):
        SizingConfig(
            model="vol_target",  # pyright: ignore[reportArgumentType]
            fraction=0.02,
        )


def test_sizingconfig_invalid_fraction_zero():
    """Test sizing config rejects zero fraction."""
    with pytest.raises(ValueError, match="fraction must be in"):
        SizingConfig(
            model="fixed_fraction",
            fraction=0.0,
        )


def test_sizingconfig_invalid_fraction_high():
    """Test sizing config rejects fraction > 1."""
    with pytest.raises(ValueError, match="fraction must be in"):
        SizingConfig(
            model="fixed_fraction",
            fraction=1.5,
        )


def test_concentrationlimit_valid():
    """Test creating a valid concentration limit."""
    limit = ConcentrationLimit(max_position_pct=0.10)
    assert limit.max_position_pct == 0.10


def test_concentrationlimit_invalid_zero():
    """Test concentration limit rejects zero."""
    with pytest.raises(ValueError, match="max_position_pct must be in"):
        ConcentrationLimit(max_position_pct=0.0)


def test_leveragelimit_valid():
    """Test creating a valid leverage limit."""
    limit = LeverageLimit(max_gross=2.0, max_net=1.0)
    assert limit.max_gross == 2.0
    assert limit.max_net == 1.0


def test_leveragelimit_invalid_gross_zero():
    """Test leverage limit rejects zero gross."""
    with pytest.raises(ValueError, match="max_gross must be positive"):
        LeverageLimit(max_gross=0.0, max_net=1.0)


def test_leveragelimit_invalid_net_negative():
    """Test leverage limit rejects negative net."""
    with pytest.raises(ValueError, match="max_net must be positive"):
        LeverageLimit(max_gross=2.0, max_net=-1.0)


def test_riskconfig_valid():
    """Test creating a valid risk config."""
    budgets = [
        StrategyBudget("momentum_v1", 0.3),
        StrategyBudget("mean_reversion_v1", 0.2),
    ]
    sizing = {
        "momentum_v1": SizingConfig("fixed_fraction", 0.02),
        "mean_reversion_v1": SizingConfig("fixed_fraction", 0.015),
    }
    config = RiskConfig(
        budgets=budgets,
        sizing=sizing,
        concentration=ConcentrationLimit(0.10),
        leverage=LeverageLimit(2.0, 1.0),
    )
    assert len(config.budgets) == 2
    assert len(config.sizing) == 2


def test_riskconfig_budgets_exceed_100():
    """Test risk config rejects budgets > 100%."""
    budgets = [
        StrategyBudget("strat1", 0.6),
        StrategyBudget("strat2", 0.6),
    ]
    sizing = {
        "strat1": SizingConfig("fixed_fraction", 0.02),
        "strat2": SizingConfig("fixed_fraction", 0.02),
    }
    with pytest.raises(ValueError, match="Budget weights sum to"):
        RiskConfig(
            budgets=budgets,
            sizing=sizing,
            concentration=ConcentrationLimit(0.10),
            leverage=LeverageLimit(2.0, 1.0),
        )


def test_riskconfig_missing_sizing():
    """Test risk config rejects missing sizing for budget."""
    budgets = [
        StrategyBudget("momentum_v1", 0.3),
        StrategyBudget("mean_reversion_v1", 0.2),
    ]
    sizing = {
        "momentum_v1": SizingConfig("fixed_fraction", 0.02),
        # Missing mean_reversion_v1
    }
    with pytest.raises(ValueError, match="have budgets but no sizing config"):
        RiskConfig(
            budgets=budgets,
            sizing=sizing,
            concentration=ConcentrationLimit(0.10),
            leverage=LeverageLimit(2.0, 1.0),
        )


def test_riskconfig_invalid_cash_buffer():
    """Test risk config rejects invalid cash buffer."""
    budgets = [StrategyBudget("test", 0.5)]
    sizing = {"test": SizingConfig("fixed_fraction", 0.02)}

    with pytest.raises(ValueError, match="cash_buffer_pct must be in"):
        RiskConfig(
            budgets=budgets,
            sizing=sizing,
            concentration=ConcentrationLimit(0.10),
            leverage=LeverageLimit(2.0, 1.0),
            cash_buffer_pct=0.6,  # > 50%
        )
