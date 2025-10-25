"""Tests for RiskService event handling."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock

import pytest

from qtrader.events.event_bus import EventBus
from qtrader.events.events import OrderRejectedEvent, RiskEvaluationTriggerEvent, SignalEvent
from qtrader.services.manager.models import (
    ConcentrationLimit,
    LeverageLimit,
    PortfolioState,
    Position,
    RiskConfig,
    SizingConfig,
    StrategyBudget,
)
from qtrader.services.manager.service import RiskService


@pytest.fixture
def event_bus():
    """Create event bus for testing."""
    return EventBus()


@pytest.fixture
def risk_config():
    """Create minimal risk config for testing."""
    return RiskConfig(
        budgets=[
            StrategyBudget("momentum_v1", 0.3),
            StrategyBudget("mean_reversion_v1", 0.2),
        ],
        sizing={
            "momentum_v1": SizingConfig("fixed_fraction", 0.02),
            "mean_reversion_v1": SizingConfig("fixed_fraction", 0.015),
        },
        concentration=ConcentrationLimit(0.10),
        leverage=LeverageLimit(2.0, 1.0),
    )


@pytest.fixture
def risk_service(risk_config, event_bus):
    """Create RiskService for testing."""
    return RiskService(risk_config, event_bus)


@pytest.fixture
def portfolio_state():
    """Create sample portfolio state."""
    return PortfolioState(
        ts=datetime(2020, 1, 2, 16, 0),
        equity=Decimal("1000000"),
        cash=Decimal("500000"),
        gross_exposure=Decimal("500000"),
        net_exposure=Decimal("500000"),
        positions={"AAPL": Position("AAPL", 100, Decimal("15000"))},
    )


# ============================================================================
# Signal Handling Tests
# ============================================================================


def test_on_signal_buffers_signal(risk_service):
    """Test that on_signal buffers signal."""
    signal = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )

    risk_service.on_signal(signal)

    assert len(risk_service._signal_buffer) == 1
    assert risk_service._signal_buffer[0] == signal


def test_on_signal_multiple_signals_same_bar(risk_service):
    """Test buffering multiple signals from same bar."""
    ts = datetime(2020, 1, 2, 16, 0)

    signal1 = SignalEvent(ts=ts, strategy_id="momentum_v1", symbol="AAPL", side="BUY", strength=0.75)
    signal2 = SignalEvent(ts=ts, strategy_id="momentum_v1", symbol="GOOGL", side="BUY", strength=0.5)
    signal3 = SignalEvent(ts=ts, strategy_id="mean_reversion_v1", symbol="TSLA", side="SELL", strength=0.6)

    risk_service.on_signal(signal1)
    risk_service.on_signal(signal2)
    risk_service.on_signal(signal3)

    assert len(risk_service._signal_buffer) == 3


def test_on_signal_sets_current_ts(risk_service):
    """Test that first signal sets current timestamp."""
    signal = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )

    assert risk_service._current_ts is None
    risk_service.on_signal(signal)
    assert risk_service._current_ts == signal.ts


def test_on_signal_rejects_inconsistent_timestamp(risk_service):
    """Test that signal with different timestamp is rejected."""
    signal1 = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )
    signal2 = SignalEvent(
        ts=datetime(2020, 1, 3, 16, 0),  # Different bar!
        strategy_id="momentum_v1",
        symbol="GOOGL",
        side="BUY",
        strength=0.5,
    )

    risk_service.on_signal(signal1)

    with pytest.raises(ValueError, match="timestamp.*does not match"):
        risk_service.on_signal(signal2)


def test_on_signal_zero_strength_valid(risk_service):
    """Test that zero strength signal is accepted."""
    signal = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.0,  # No conviction
    )

    risk_service.on_signal(signal)
    assert len(risk_service._signal_buffer) == 1


# ============================================================================
# Portfolio State Handling Tests
# ============================================================================


def test_on_portfolio_state_caches_state(risk_service, portfolio_state):
    """Test that on_portfolio_state caches state."""
    assert risk_service._portfolio_state is None

    risk_service.on_portfolio_state(portfolio_state)

    assert risk_service._portfolio_state is not None
    assert risk_service._portfolio_state.equity == Decimal("1000000")
    assert risk_service._portfolio_state.cash == Decimal("500000")


def test_on_portfolio_state_updates_existing_cache(risk_service, portfolio_state):
    """Test that portfolio state cache is updated."""
    # Cache initial state
    risk_service.on_portfolio_state(portfolio_state)
    assert risk_service._portfolio_state.equity == Decimal("1000000")

    # Update state
    new_state = PortfolioState(
        ts=datetime(2020, 1, 3, 16, 0),
        equity=Decimal("1100000"),  # Increased
        cash=Decimal("550000"),
        gross_exposure=Decimal("550000"),
        net_exposure=Decimal("550000"),
        positions={},
    )

    risk_service.on_portfolio_state(new_state)

    assert risk_service._portfolio_state.equity == Decimal("1100000")
    assert risk_service._portfolio_state.ts == datetime(2020, 1, 3, 16, 0)


def test_on_portfolio_state_with_positions(risk_service):
    """Test portfolio state with multiple positions."""
    state = PortfolioState(
        ts=datetime(2020, 1, 2, 16, 0),
        equity=Decimal("1000000"),
        cash=Decimal("977500"),
        gross_exposure=Decimal("22500"),
        net_exposure=Decimal("22500"),
        positions={
            "AAPL": Position("AAPL", 100, Decimal("15000")),
            "GOOGL": Position("GOOGL", 50, Decimal("7500")),
        },
    )

    risk_service.on_portfolio_state(state)

    assert len(risk_service._portfolio_state.positions) == 2
    assert "AAPL" in risk_service._portfolio_state.positions
    assert "GOOGL" in risk_service._portfolio_state.positions


# ============================================================================
# Risk Evaluation Tests (Scaffold)
# ============================================================================


def test_on_risk_evaluation_trigger_no_signals(risk_service, event_bus):
    """Test evaluation with no signals does nothing."""
    trigger = RiskEvaluationTriggerEvent(ts=datetime(2020, 1, 2, 16, 0))

    # Should not raise, just return early
    risk_service.on_risk_evaluation_trigger(trigger)


def test_on_risk_evaluation_trigger_no_portfolio_state(risk_service, event_bus):
    """Test evaluation without portfolio state rejects all signals."""
    signal = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )
    trigger = RiskEvaluationTriggerEvent(ts=datetime(2020, 1, 2, 16, 0))

    # Mock event bus to capture published events
    published_events = []
    event_bus.publish = Mock(side_effect=lambda e: published_events.append(e))
    risk_service._event_bus = event_bus

    risk_service.on_signal(signal)
    risk_service.on_risk_evaluation_trigger(trigger)

    # Should publish rejection
    assert len(published_events) == 1
    assert isinstance(published_events[0], OrderRejectedEvent)
    assert "No portfolio state" in published_events[0].reason


def test_on_risk_evaluation_trigger_clears_buffer(risk_service, portfolio_state, event_bus):
    """Test that evaluation clears signal buffer."""
    signal = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )
    trigger = RiskEvaluationTriggerEvent(ts=datetime(2020, 1, 2, 16, 0))

    event_bus.publish = Mock()
    risk_service._event_bus = event_bus

    risk_service.on_portfolio_state(portfolio_state)
    risk_service.on_signal(signal)

    assert len(risk_service._signal_buffer) == 1

    risk_service.on_risk_evaluation_trigger(trigger)

    assert len(risk_service._signal_buffer) == 0


def test_on_risk_evaluation_trigger_resets_current_ts(risk_service, portfolio_state, event_bus):
    """Test that evaluation resets current timestamp."""
    signal = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )
    trigger = RiskEvaluationTriggerEvent(ts=datetime(2020, 1, 2, 16, 0))

    event_bus.publish = Mock()
    risk_service._event_bus = event_bus

    risk_service.on_portfolio_state(portfolio_state)
    risk_service.on_signal(signal)

    assert risk_service._current_ts is not None

    risk_service.on_risk_evaluation_trigger(trigger)

    assert risk_service._current_ts is None


def test_on_risk_evaluation_trigger_scaffold_rejects_all(risk_service, portfolio_state, event_bus):
    """Test that signals without price metadata are rejected."""
    signals = [
        SignalEvent(
            ts=datetime(2020, 1, 2, 16, 0),
            strategy_id="momentum_v1",
            symbol="AAPL",
            side="BUY",
            strength=0.75,
            metadata={},  # No price!
        ),
        SignalEvent(
            ts=datetime(2020, 1, 2, 16, 0),
            strategy_id="momentum_v1",
            symbol="GOOGL",
            side="BUY",
            strength=0.5,
            metadata={},  # No price!
        ),
    ]
    trigger = RiskEvaluationTriggerEvent(ts=datetime(2020, 1, 2, 16, 0))

    published_events = []
    event_bus.publish = Mock(side_effect=lambda e: published_events.append(e))
    risk_service._event_bus = event_bus

    risk_service.on_portfolio_state(portfolio_state)
    for signal in signals:
        risk_service.on_signal(signal)

    risk_service.on_risk_evaluation_trigger(trigger)

    # Should publish 2 rejections (missing price)
    assert len(published_events) == 2
    assert all(isinstance(e, OrderRejectedEvent) for e in published_events)
    assert all("price" in e.reason.lower() for e in published_events)


def test_publish_rejection_helper(risk_service, event_bus):
    """Test _publish_rejection helper method."""
    signal = SignalEvent(
        ts=datetime(2020, 1, 2, 16, 0),
        strategy_id="momentum_v1",
        symbol="AAPL",
        side="BUY",
        strength=0.75,
    )

    published_events = []
    event_bus.publish = Mock(side_effect=lambda e: published_events.append(e))
    risk_service._event_bus = event_bus

    risk_service._publish_rejection(signal, signal.ts, "Test rejection")

    assert len(published_events) == 1
    event = published_events[0]
    assert isinstance(event, OrderRejectedEvent)
    assert event.strategy_id == "momentum_v1"
    assert event.symbol == "AAPL"
    assert event.side == "BUY"
    assert event.reason == "Test rejection"
