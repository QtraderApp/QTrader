"""
Unit tests for event classes.

Tests event creation, immutability, and validation.
"""

import unittest
from datetime import datetime
from decimal import Decimal

from qtrader.events.events import (
    BacktestEndedEvent,
    BacktestStartedEvent,
    BarCloseEvent,
    CashChangedEvent,
    CorporateActionEvent,
    Event,
    FillEvent,
    OrderEvent,
    PositionChangedEvent,
    PriceBarEvent,
    RiskViolationEvent,
    SignalEvent,
)


class TestEventBase(unittest.TestCase):
    """Test base Event class."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = Event()
        self.assertIsNotNone(event.event_id)
        self.assertIsInstance(event.timestamp, datetime)
        self.assertEqual(event.event_type, "")

    def test_event_immutability(self):
        """Test that events are immutable (frozen)."""
        event = Event()
        with self.assertRaises(AttributeError):
            event.event_type = "modified"  # pyright: ignore[reportAttributeAccessIssue]

    def test_event_with_custom_values(self):
        """Test event creation with custom values."""
        timestamp = datetime(2020, 1, 1, 12, 0)
        event = Event(
            event_id="test_id",
            timestamp=timestamp,
            event_type="test_event",
        )
        self.assertEqual(event.event_id, "test_id")
        self.assertEqual(event.timestamp, timestamp)
        self.assertEqual(event.event_type, "test_event")


class TestCorporateActionEvent(unittest.TestCase):
    """Test CorporateActionEvent."""

    def test_dividend_event(self):
        """Test dividend event creation."""
        event = CorporateActionEvent(
            symbol="AAPL",
            action_type="dividend",
            effective_date=datetime(2020, 2, 7),
            ex_date=datetime(2020, 2, 7),
            dividend_amount=Decimal("0.77"),
            dividend_type="cash",
        )
        self.assertEqual(event.event_type, "corporate_action")
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.action_type, "dividend")
        self.assertEqual(event.dividend_amount, Decimal("0.77"))
        self.assertEqual(event.dividend_type, "cash")

    def test_split_event(self):
        """Test split event creation."""
        event = CorporateActionEvent(
            symbol="AAPL",
            action_type="split",
            effective_date=datetime(2020, 8, 31),
            split_ratio=Decimal("4.0"),
        )
        self.assertEqual(event.event_type, "corporate_action")
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.action_type, "split")
        self.assertEqual(event.split_ratio, Decimal("4.0"))

    def test_corporate_action_immutability(self):
        """Test that corporate action events are immutable."""
        event = CorporateActionEvent(symbol="AAPL", action_type="split")
        with self.assertRaises(AttributeError):
            event.symbol = "MSFT"  # pyright: ignore[reportAttributeAccessIssue]


class TestPriceBarEvent(unittest.TestCase):
    """Test PriceBarEvent."""

    def test_price_bar_event_creation(self):
        """Test price bar event creation (without full Bar for now)."""
        # For now, test event creation without complex Bar model
        # Full Bar integration will be tested in integration tests
        event = PriceBarEvent(symbol="AAPL", bar=None)
        self.assertEqual(event.event_type, "price_bar")
        self.assertEqual(event.symbol, "AAPL")
        self.assertIsNone(event.bar)  # Bar can be None

    def test_price_bar_event_immutability(self):
        """Test that price bar events are immutable."""
        event = PriceBarEvent(symbol="AAPL")
        with self.assertRaises(AttributeError):
            event.symbol = "MSFT"  # pyright: ignore[reportAttributeAccessIssue]


class TestSignalEvent(unittest.TestCase):
    """Test SignalEvent."""

    def test_signal_event_creation(self):
        """Test signal event creation (Phase 4 spec)."""
        event = SignalEvent(
            strategy_id="mean_reversion",
            symbol="AAPL",
            side="BUY",
            strength=0.85,
            metadata={"price": 150.0, "volatility": 0.25},
        )
        self.assertEqual(event.event_type, "signal")
        self.assertEqual(event.strategy_id, "mean_reversion")
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.side, "BUY")
        self.assertEqual(event.strength, 0.85)
        self.assertEqual(event.metadata["price"], 150.0)


class TestOrderEvent(unittest.TestCase):
    """Test OrderEvent."""

    def test_order_event_creation(self):
        """Test order event creation."""
        event = OrderEvent(
            order_id="order_123",
            signal_id="sig_123",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            order_type="market",
        )
        self.assertEqual(event.event_type, "order")
        self.assertEqual(event.order_id, "order_123")
        self.assertEqual(event.signal_id, "sig_123")
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.side, "buy")
        self.assertEqual(event.quantity, Decimal("100"))


class TestFillEvent(unittest.TestCase):
    """Test FillEvent."""

    def test_fill_event_creation(self):
        """Test fill event creation."""
        event = FillEvent(
            fill_id="fill_123",
            order_id="order_123",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("75.50"),
            commission=Decimal("1.00"),
        )
        self.assertEqual(event.event_type, "fill")
        self.assertEqual(event.fill_id, "fill_123")
        self.assertEqual(event.order_id, "order_123")
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.quantity, Decimal("100"))
        self.assertEqual(event.price, Decimal("75.50"))


class TestPositionChangedEvent(unittest.TestCase):
    """Test PositionChangedEvent."""

    def test_position_changed_event(self):
        """Test position changed event creation."""
        event = PositionChangedEvent(
            symbol="AAPL",
            old_quantity=Decimal("0"),
            new_quantity=Decimal("100"),
            reason="fill",
        )
        self.assertEqual(event.event_type, "position_changed")
        self.assertEqual(event.symbol, "AAPL")
        self.assertEqual(event.old_quantity, Decimal("0"))
        self.assertEqual(event.new_quantity, Decimal("100"))
        self.assertEqual(event.reason, "fill")


class TestCashChangedEvent(unittest.TestCase):
    """Test CashChangedEvent."""

    def test_cash_changed_event(self):
        """Test cash changed event creation."""
        event = CashChangedEvent(
            old_cash=Decimal("100000"),
            new_cash=Decimal("92500"),
            change_amount=Decimal("-7500"),
            reason="fill_buy",
        )
        self.assertEqual(event.event_type, "cash_changed")
        self.assertEqual(event.old_cash, Decimal("100000"))
        self.assertEqual(event.new_cash, Decimal("92500"))
        self.assertEqual(event.change_amount, Decimal("-7500"))


class TestRiskViolationEvent(unittest.TestCase):
    """Test RiskViolationEvent."""

    def test_risk_violation_event(self):
        """Test risk violation event creation."""
        event = RiskViolationEvent(
            violation_type="position_limit_exceeded",
            severity="error",
            message="Position size exceeds 10% of portfolio",
            symbol="AAPL",
        )
        self.assertEqual(event.event_type, "risk_violation")
        self.assertEqual(event.violation_type, "position_limit_exceeded")
        self.assertEqual(event.severity, "error")
        self.assertIn("10%", event.message)


class TestBacktestEvents(unittest.TestCase):
    """Test backtest control events."""

    def test_backtest_started_event(self):
        """Test backtest started event."""
        event = BacktestStartedEvent(
            backtest_id="bt_123",
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            config={"initial_cash": 100000},
        )
        self.assertEqual(event.event_type, "backtest_started")
        self.assertEqual(event.backtest_id, "bt_123")
        self.assertEqual(event.config["initial_cash"], 100000)

    def test_backtest_ended_event(self):
        """Test backtest ended event."""
        event = BacktestEndedEvent(
            backtest_id="bt_123",
            total_bars=250,
            total_fills=50,
            duration_seconds=12.5,
        )
        self.assertEqual(event.event_type, "backtest_ended")
        self.assertEqual(event.total_bars, 250)
        self.assertEqual(event.total_fills, 50)

    def test_bar_close_event(self):
        """Test bar close event."""
        event = BarCloseEvent(
            current_time=datetime(2020, 1, 2, 16, 0),
            bar_number=1,
        )
        self.assertEqual(event.event_type, "bar_close")
        self.assertEqual(event.bar_number, 1)
        self.assertIsNotNone(event.current_time)


if __name__ == "__main__":
    unittest.main()
