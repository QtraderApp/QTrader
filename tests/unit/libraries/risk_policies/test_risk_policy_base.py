"""
Unit tests for qtrader.libraries.risk_policies.base module.

Tests the BaseRiskPolicy abstract class:
- Abstract method requirements
- Name property
- batch_evaluate default implementation
- OrderDecision and PortfolioState contracts

Following unittest.prompt.md guidelines:
- Descriptive test names
- Arrange-Act-Assert pattern
- pytest fixtures
- Focus on contract compliance
"""

from datetime import datetime
from decimal import Decimal

import pytest

from qtrader.events.events import SignalEvent
from qtrader.libraries.risk.base import BaseRiskPolicy, OrderDecision, PortfolioState

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_signal() -> SignalEvent:
    """Create sample signal for testing."""
    return SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test", symbol="AAPL", side="BUY", strength=0.8)


@pytest.fixture
def sample_portfolio() -> PortfolioState:
    """Create sample portfolio state for testing."""
    return PortfolioState(
        current_equity=Decimal("100000"),
        cash_available=Decimal("50000"),
        leverage=1.0,
        drawdown=0.0,
        positions={},
    )


# ============================================================================
# Test Fixtures - Concrete Implementation
# ============================================================================


class MockRiskConfig:
    """Mock config for testing."""

    max_position_size: float = 0.9


class ConcreteRiskPolicy(BaseRiskPolicy):
    """Minimal concrete implementation for testing."""

    def __init__(self, config: MockRiskConfig):
        self.config = config

    def evaluate_signal(self, signal: SignalEvent, portfolio: PortfolioState, price: Decimal) -> OrderDecision:
        """Basic evaluation."""
        return OrderDecision(
            approved=True, symbol=signal.symbol, side=signal.side, quantity=100, reason="Test approval"
        )

    def calculate_position_size(self, signal: SignalEvent, portfolio: PortfolioState, price: Decimal) -> int:
        """Basic position sizing."""
        return 100

    def batch_evaluate(
        self, signals: list[SignalEvent], portfolio: PortfolioState, prices: dict[str, Decimal]
    ) -> list[OrderDecision]:
        """Use default implementation from base class."""
        decisions = []
        for signal in signals:
            price = prices.get(signal.symbol)
            if price is None:
                decisions.append(
                    OrderDecision(
                        approved=False, symbol=signal.symbol, side=signal.side, quantity=0, reason="Price not available"
                    )
                )
            else:
                decisions.append(self.evaluate_signal(signal, portfolio, price))
        return decisions


class IncompleteRiskPolicy(BaseRiskPolicy):
    """Policy missing required methods."""

    pass


# ============================================================================
# Test PortfolioState Dataclass
# ============================================================================


class TestPortfolioState:
    """Test PortfolioState dataclass."""

    def test_portfolio_state_creation_with_all_fields(self) -> None:
        """PortfolioState should be created with all required fields."""
        # Arrange & Act
        state = PortfolioState(
            current_equity=Decimal("100000.00"),
            cash_available=Decimal("50000.00"),
            leverage=1.5,
            drawdown=0.05,
            positions={"AAPL": {"qty": 100}},
        )

        # Assert
        assert state.current_equity == Decimal("100000.00")
        assert state.cash_available == Decimal("50000.00")
        assert state.leverage == 1.5
        assert state.drawdown == 0.05
        assert "AAPL" in state.positions

    def test_portfolio_state_is_dataclass(self) -> None:
        """PortfolioState should be a dataclass."""
        # Arrange & Act
        from dataclasses import is_dataclass

        # Assert
        assert is_dataclass(PortfolioState)


# ============================================================================
# Test OrderDecision Dataclass
# ============================================================================


class TestOrderDecision:
    """Test OrderDecision dataclass."""

    def test_order_decision_approved_creation(self) -> None:
        """OrderDecision for approved signal should be created."""
        # Arrange & Act
        decision = OrderDecision(approved=True, symbol="AAPL", side="BUY", quantity=100, reason="Strong signal")

        # Assert
        assert decision.approved is True
        assert decision.symbol == "AAPL"
        assert decision.side == "BUY"
        assert decision.quantity == 100
        assert decision.reason == "Strong signal"

    def test_order_decision_rejected_creation(self) -> None:
        """OrderDecision for rejected signal should be created."""
        # Arrange & Act
        decision = OrderDecision(approved=False, symbol="AAPL", side="BUY", quantity=0, reason="Risk limit exceeded")

        # Assert
        assert decision.approved is False
        assert decision.quantity == 0
        assert "risk" in decision.reason.lower()

    def test_order_decision_is_dataclass(self) -> None:
        """OrderDecision should be a dataclass."""
        # Arrange & Act
        from dataclasses import is_dataclass

        # Assert
        assert is_dataclass(OrderDecision)


# ============================================================================
# Test BaseRiskPolicy Abstract Interface
# ============================================================================


class TestBaseRiskPolicyAbstractInterface:
    """Test BaseRiskPolicy abstract method enforcement."""

    def test_cannot_instantiate_base_risk_policy_directly(self) -> None:
        """BaseRiskPolicy is abstract and cannot be instantiated."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError) as exc_info:
            BaseRiskPolicy(MockRiskConfig())  # type: ignore[abstract]

        assert "abstract" in str(exc_info.value).lower()

    def test_cannot_instantiate_incomplete_policy(self) -> None:
        """Policy missing abstract methods cannot be instantiated."""
        # Arrange & Act & Assert
        with pytest.raises(TypeError) as exc_info:
            IncompleteRiskPolicy(MockRiskConfig())  # type: ignore[abstract]

        assert "abstract" in str(exc_info.value).lower()

    def test_concrete_policy_can_be_instantiated(self) -> None:
        """Concrete policy implementing all methods can be instantiated."""
        # Arrange
        config = MockRiskConfig()

        # Act
        policy = ConcreteRiskPolicy(config)

        # Assert
        assert isinstance(policy, BaseRiskPolicy)
        assert policy.config == config


# ============================================================================
# Test BaseRiskPolicy Required Methods
# ============================================================================


class TestBaseRiskPolicyRequiredMethods:
    """Test that concrete policies provide required methods."""

    def test_init_method_required(self) -> None:
        """Concrete policy must implement __init__."""
        # Arrange
        config = MockRiskConfig()

        # Act
        policy = ConcreteRiskPolicy(config)

        # Assert
        assert hasattr(policy, "__init__")
        assert policy.config == config

    def test_evaluate_signal_method_required(self) -> None:
        """Concrete policy must implement evaluate_signal."""
        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())

        # Act & Assert
        assert hasattr(policy, "evaluate_signal")
        assert callable(policy.evaluate_signal)

    def test_calculate_position_size_method_required(self) -> None:
        """Concrete policy must implement calculate_position_size."""
        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())

        # Act & Assert
        assert hasattr(policy, "calculate_position_size")
        assert callable(policy.calculate_position_size)

    def test_batch_evaluate_method_exists(self) -> None:
        """batch_evaluate method should exist (has default implementation)."""
        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())

        # Act & Assert
        assert hasattr(policy, "batch_evaluate")
        assert callable(policy.batch_evaluate)


# ============================================================================
# Test BaseRiskPolicy Name Property
# ============================================================================


class TestBaseRiskPolicyNameProperty:
    """Test the name property default implementation."""

    def test_name_property_strips_risk_policy_suffix(self) -> None:
        """name property should remove 'RiskPolicy' suffix."""

        # Arrange
        class TestRiskPolicy(ConcreteRiskPolicy):
            pass

        policy = TestRiskPolicy(MockRiskConfig())

        # Act
        name = policy.name

        # Assert
        assert name == "test"

    def test_name_property_strips_policy_suffix(self) -> None:
        """name property should remove 'Policy' suffix."""

        # Arrange
        class NaivePolicy(ConcreteRiskPolicy):
            pass

        policy = NaivePolicy(MockRiskConfig())

        # Act
        name = policy.name

        # Assert
        assert name == "naive"

    def test_name_property_converts_camelcase_to_snake_case(self) -> None:
        """name property should convert CamelCase to snake_case."""

        # Arrange
        class VolTargetRiskPolicy(ConcreteRiskPolicy):
            pass

        policy = VolTargetRiskPolicy(MockRiskConfig())

        # Act
        name = policy.name

        # Assert
        assert name == "vol_target"

    def test_name_property_can_be_overridden(self) -> None:
        """name property can be overridden in subclass."""

        # Arrange
        class CustomPolicy(ConcreteRiskPolicy):
            @property
            def name(self) -> str:
                return "custom_name"

        policy = CustomPolicy(MockRiskConfig())

        # Act
        name = policy.name

        # Assert
        assert name == "custom_name"


# ============================================================================
# Test evaluate_signal Method
# ============================================================================


class TestEvaluateSignalMethod:
    """Test evaluate_signal method behavior."""

    def test_evaluate_signal_returns_order_decision(self) -> None:
        """evaluate_signal should return OrderDecision."""
        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())
        from datetime import datetime

        signal = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test", symbol="AAPL", side="BUY", strength=0.8)
        portfolio = PortfolioState(
            current_equity=Decimal("100000"),
            cash_available=Decimal("50000"),
            leverage=1.0,
            drawdown=0.0,
            positions={},
        )
        price = Decimal("150.00")

        # Act
        decision = policy.evaluate_signal(signal, portfolio, price)

        # Assert
        assert isinstance(decision, OrderDecision)
        assert decision.symbol == "AAPL"

    def test_evaluate_signal_receives_signal_portfolio_price(self) -> None:
        """evaluate_signal should receive signal, portfolio state, and price."""
        from datetime import datetime

        # Arrange
        class TrackingPolicy(ConcreteRiskPolicy):
            def evaluate_signal(self, signal: SignalEvent, portfolio: PortfolioState, price: Decimal) -> OrderDecision:
                self.received_signal = signal
                self.received_portfolio = portfolio
                self.received_price = price
                return super().evaluate_signal(signal, portfolio, price)

        policy = TrackingPolicy(MockRiskConfig())
        signal = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test", symbol="AAPL", side="BUY", strength=0.8)
        portfolio = PortfolioState(
            current_equity=Decimal("100000"),
            cash_available=Decimal("50000"),
            leverage=1.0,
            drawdown=0.0,
            positions={},
        )
        price = Decimal("150.00")

        # Act
        policy.evaluate_signal(signal, portfolio, price)

        # Assert
        assert hasattr(policy, "received_signal")
        assert policy.received_signal == signal
        assert policy.received_portfolio == portfolio
        assert policy.received_price == price


# ============================================================================
# Test calculate_position_size Method
# ============================================================================


class TestCalculatePositionSizeMethod:
    """Test calculate_position_size method."""

    def test_calculate_position_size_returns_int(self) -> None:
        """calculate_position_size should return integer quantity."""
        from datetime import datetime

        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())
        signal = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test", symbol="AAPL", side="BUY", strength=0.8)
        portfolio = PortfolioState(
            current_equity=Decimal("100000"),
            cash_available=Decimal("50000"),
            leverage=1.0,
            drawdown=0.0,
            positions={},
        )
        price = Decimal("150.00")

        # Act
        size = policy.calculate_position_size(signal, portfolio, price)

        # Assert
        assert isinstance(size, int)
        assert size >= 0


# ============================================================================
# Test batch_evaluate Default Implementation
# ============================================================================


class TestBatchEvaluateMethod:
    """Test batch_evaluate default implementation."""

    def test_batch_evaluate_returns_list_of_decisions(self) -> None:
        """batch_evaluate should return list of OrderDecisions."""
        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())
        signal1 = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test1", symbol="AAPL", side="BUY", strength=0.8)
        signal2 = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test2", symbol="MSFT", side="BUY", strength=0.7)

        portfolio = PortfolioState(
            current_equity=Decimal("100000"),
            cash_available=Decimal("50000"),
            leverage=1.0,
            drawdown=0.0,
            positions={},
        )
        prices = {"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")}

        # Act
        decisions = policy.batch_evaluate([signal1, signal2], portfolio, prices)

        # Assert
        assert isinstance(decisions, list)
        assert len(decisions) == 2
        assert all(isinstance(d, OrderDecision) for d in decisions)

    def test_batch_evaluate_empty_signals_returns_empty_list(self) -> None:
        """batch_evaluate with empty signals should return empty list."""
        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())
        portfolio = PortfolioState(
            current_equity=Decimal("100000"),
            cash_available=Decimal("50000"),
            leverage=1.0,
            drawdown=0.0,
            positions={},
        )
        prices: dict[str, Decimal] = {}

        # Act
        decisions = policy.batch_evaluate([], portfolio, prices)

        # Assert
        assert decisions == []

    def test_batch_evaluate_missing_price_rejects_signal(self) -> None:
        """batch_evaluate should reject signal if price not available."""
        # Arrange
        policy = ConcreteRiskPolicy(MockRiskConfig())
        signal = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test", symbol="AAPL", side="BUY", strength=0.8)

        portfolio = PortfolioState(
            current_equity=Decimal("100000"),
            cash_available=Decimal("50000"),
            leverage=1.0,
            drawdown=0.0,
            positions={},
        )
        prices: dict[str, Decimal] = {}  # Missing AAPL price

        # Act
        decisions = policy.batch_evaluate([signal], portfolio, prices)

        # Assert
        assert len(decisions) == 1
        assert decisions[0].approved is False
        assert "price" in decisions[0].reason.lower()

    def test_batch_evaluate_calls_evaluate_signal_for_each(self) -> None:
        """batch_evaluate default implementation calls evaluate_signal for each signal."""

        # Arrange
        class CountingPolicy(ConcreteRiskPolicy):
            def __init__(self, config: MockRiskConfig):
                super().__init__(config)
                self.evaluate_count = 0

            def evaluate_signal(self, signal: SignalEvent, portfolio: PortfolioState, price: Decimal) -> OrderDecision:
                self.evaluate_count += 1
                return super().evaluate_signal(signal, portfolio, price)

        policy = CountingPolicy(MockRiskConfig())
        signal1 = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test1", symbol="AAPL", side="BUY", strength=0.8)
        signal2 = SignalEvent(ts=datetime(2024, 1, 1), strategy_id="test2", symbol="MSFT", side="BUY", strength=0.7)

        portfolio = PortfolioState(
            current_equity=Decimal("100000"),
            cash_available=Decimal("50000"),
            leverage=1.0,
            drawdown=0.0,
            positions={},
        )
        prices = {"AAPL": Decimal("150.00"), "MSFT": Decimal("300.00")}

        # Act
        policy.batch_evaluate([signal1, signal2], portfolio, prices)

        # Assert
        assert policy.evaluate_count == 2


# ============================================================================
# Test Documentation
# ============================================================================


class TestRiskPolicyDocumentation:
    """Test that base classes have proper documentation."""

    def test_base_risk_policy_has_docstring(self) -> None:
        """BaseRiskPolicy should have comprehensive docstring."""
        # Arrange & Act & Assert
        assert BaseRiskPolicy.__doc__ is not None
        assert len(BaseRiskPolicy.__doc__) > 100

    def test_evaluate_signal_has_docstring(self) -> None:
        """evaluate_signal method should be documented."""
        # Arrange & Act & Assert
        assert BaseRiskPolicy.evaluate_signal.__doc__ is not None

    def test_batch_evaluate_has_docstring(self) -> None:
        """batch_evaluate method should be documented."""
        # Arrange & Act & Assert
        assert BaseRiskPolicy.batch_evaluate.__doc__ is not None
