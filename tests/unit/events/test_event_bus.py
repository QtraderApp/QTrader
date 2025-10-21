"""
Unit tests for EventBus.

Tests publish/subscribe, priority ordering, error isolation, and history management.
"""

import unittest
from datetime import datetime, timedelta
from decimal import Decimal

from qtrader.events.event_bus import EventBus
from qtrader.events.events import CorporateActionEvent, Event, FillEvent, OrderEvent, SignalEvent


class TestEventBusBasic(unittest.TestCase):
    """Test basic EventBus functionality."""

    def setUp(self):
        """Create fresh EventBus for each test."""
        self.bus = EventBus()
        self.events_received = []

    def test_publish_without_subscribers(self):
        """Test publishing event with no subscribers (should not error)."""
        event = Event(event_type="test")
        self.bus.publish(event)  # Should not raise

    def test_subscribe_and_publish(self):
        """Test basic subscribe and publish."""

        def handler(event):
            self.events_received.append(event)

        self.bus.subscribe("test", handler)
        event = Event(event_type="test")
        self.bus.publish(event)

        self.assertEqual(len(self.events_received), 1)
        self.assertEqual(self.events_received[0].event_id, event.event_id)

    def test_multiple_handlers_same_event(self):
        """Test multiple handlers for same event type."""
        calls = []

        def handler1(event):
            calls.append("handler1")

        def handler2(event):
            calls.append("handler2")

        self.bus.subscribe("test", handler1)
        self.bus.subscribe("test", handler2)

        event = Event(event_type="test")
        self.bus.publish(event)

        self.assertEqual(len(calls), 2)
        self.assertIn("handler1", calls)
        self.assertIn("handler2", calls)

    def test_different_event_types(self):
        """Test that handlers only receive subscribed event types."""
        fill_events = []
        order_events = []

        def fill_handler(event):
            fill_events.append(event)

        def order_handler(event):
            order_events.append(event)

        self.bus.subscribe("fill", fill_handler)
        self.bus.subscribe("order", order_handler)

        fill = FillEvent(fill_id="f1", order_id="o1", symbol="AAPL")
        order = OrderEvent(order_id="o1", symbol="AAPL")

        self.bus.publish(fill)
        self.bus.publish(order)

        self.assertEqual(len(fill_events), 1)
        self.assertEqual(len(order_events), 1)
        self.assertEqual(fill_events[0].event_type, "fill")
        self.assertEqual(order_events[0].event_type, "order")


class TestEventBusPriority(unittest.TestCase):
    """Test priority-based handler ordering."""

    def setUp(self):
        """Create fresh EventBus for each test."""
        self.bus = EventBus()
        self.call_order = []

    def test_priority_ordering(self):
        """Test that handlers are called in priority order (highest first)."""

        def low_priority(event):
            self.call_order.append("low")

        def medium_priority(event):
            self.call_order.append("medium")

        def high_priority(event):
            self.call_order.append("high")

        # Subscribe in random order
        self.bus.subscribe("test", medium_priority, priority=10)
        self.bus.subscribe("test", low_priority, priority=1)
        self.bus.subscribe("test", high_priority, priority=100)

        event = Event(event_type="test")
        self.bus.publish(event)

        # Should be called in priority order
        self.assertEqual(self.call_order, ["high", "medium", "low"])

    def test_same_priority_fifo(self):
        """Test that handlers with same priority are called in subscription order."""

        def handler1(event):
            self.call_order.append("h1")

        def handler2(event):
            self.call_order.append("h2")

        def handler3(event):
            self.call_order.append("h3")

        # All same priority
        self.bus.subscribe("test", handler1, priority=10)
        self.bus.subscribe("test", handler2, priority=10)
        self.bus.subscribe("test", handler3, priority=10)

        event = Event(event_type="test")
        self.bus.publish(event)

        # Should maintain subscription order
        self.assertEqual(self.call_order, ["h1", "h2", "h3"])

    def test_default_priority_zero(self):
        """Test that default priority is 0."""

        def default_handler(event):
            self.call_order.append("default")

        def high_handler(event):
            self.call_order.append("high")

        self.bus.subscribe("test", high_handler, priority=10)
        self.bus.subscribe("test", default_handler)  # No priority = 0

        event = Event(event_type="test")
        self.bus.publish(event)

        self.assertEqual(self.call_order, ["high", "default"])


class TestEventBusErrorIsolation(unittest.TestCase):
    """Test error isolation between handlers."""

    def setUp(self):
        """Create fresh EventBus for each test."""
        self.bus = EventBus()
        self.successful_calls = []

    def test_handler_error_doesnt_stop_others(self):
        """Test that exception in one handler doesn't stop other handlers."""

        def failing_handler(event):
            self.successful_calls.append("before_fail")
            raise ValueError("Handler failed!")

        def success_handler1(event):
            self.successful_calls.append("success1")

        def success_handler2(event):
            self.successful_calls.append("success2")

        self.bus.subscribe("test", success_handler1, priority=100)
        self.bus.subscribe("test", failing_handler, priority=50)
        self.bus.subscribe("test", success_handler2, priority=10)

        event = Event(event_type="test")
        # Should not raise despite failing_handler exception
        self.bus.publish(event)

        # All non-failing handlers should have been called
        self.assertIn("success1", self.successful_calls)
        self.assertIn("success2", self.successful_calls)
        self.assertIn("before_fail", self.successful_calls)

    def test_multiple_failing_handlers(self):
        """Test that multiple failures are isolated."""

        def fail1(event):
            raise ValueError("Fail 1")

        def fail2(event):
            raise RuntimeError("Fail 2")

        def success(event):
            self.successful_calls.append("success")

        self.bus.subscribe("test", fail1)
        self.bus.subscribe("test", success)
        self.bus.subscribe("test", fail2)

        event = Event(event_type="test")
        self.bus.publish(event)  # Should not raise

        self.assertIn("success", self.successful_calls)


class TestEventBusUnsubscribe(unittest.TestCase):
    """Test unsubscribe functionality."""

    def setUp(self):
        """Create fresh EventBus for each test."""
        self.bus = EventBus()
        self.calls = []

    def test_unsubscribe_handler(self):
        """Test unsubscribing a handler."""

        def handler(event):
            self.calls.append("handler")

        self.bus.subscribe("test", handler)
        event = Event(event_type="test")
        self.bus.publish(event)
        self.assertEqual(len(self.calls), 1)

        # Unsubscribe and publish again
        self.bus.unsubscribe("test", handler)
        self.bus.publish(event)
        self.assertEqual(len(self.calls), 1)  # No new calls

    def test_unsubscribe_nonexistent_handler(self):
        """Test unsubscribing handler that was never subscribed (should not error)."""

        def handler(event):
            pass

        self.bus.unsubscribe("test", handler)  # Should not raise

    def test_unsubscribe_one_of_multiple(self):
        """Test unsubscribing one handler while others remain."""

        def handler1(event):
            self.calls.append("h1")

        def handler2(event):
            self.calls.append("h2")

        self.bus.subscribe("test", handler1)
        self.bus.subscribe("test", handler2)

        event = Event(event_type="test")
        self.bus.publish(event)
        self.assertEqual(len(self.calls), 2)

        # Unsubscribe handler1
        self.bus.unsubscribe("test", handler1)
        self.calls.clear()
        self.bus.publish(event)

        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0], "h2")


class TestEventBusHistory(unittest.TestCase):
    """Test event history functionality."""

    def setUp(self):
        """Create fresh EventBus for each test."""
        self.bus = EventBus(max_history=100)

    def test_history_stores_events(self):
        """Test that published events are stored in history."""
        event1 = Event(event_type="test")
        event2 = Event(event_type="test")

        self.bus.publish(event1)
        self.bus.publish(event2)

        history = self.bus.get_history()
        self.assertEqual(len(history), 2)

    def test_history_filter_by_type(self):
        """Test filtering history by event type."""
        fill = FillEvent(fill_id="f1", order_id="o1", symbol="AAPL")
        order = OrderEvent(order_id="o1", symbol="AAPL")

        self.bus.publish(fill)
        self.bus.publish(order)

        fill_history = self.bus.get_history(event_type="fill")
        order_history = self.bus.get_history(event_type="order")

        self.assertEqual(len(fill_history), 1)
        self.assertEqual(len(order_history), 1)
        self.assertEqual(fill_history[0].event_type, "fill")

    def test_history_filter_by_time(self):
        """Test filtering history by timestamp."""
        now = datetime.now()
        past = now - timedelta(hours=1)

        old_event = Event(event_type="test", timestamp=past)
        new_event = Event(event_type="test", timestamp=now)

        self.bus.publish(old_event)
        self.bus.publish(new_event)

        recent = self.bus.get_history(since=now - timedelta(minutes=30))
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].timestamp, now)

    def test_history_limit(self):
        """Test limiting history results."""
        for i in range(10):
            self.bus.publish(Event(event_type="test"))

        limited = self.bus.get_history(limit=5)
        self.assertEqual(len(limited), 5)

    def test_history_max_size(self):
        """Test that history is bounded by max_history."""
        bus = EventBus(max_history=5)

        # Publish more events than max_history
        for i in range(10):
            bus.publish(Event(event_type="test", event_id=f"event_{i}"))

        history = bus.get_history()
        self.assertEqual(len(history), 5)
        # Should keep most recent events
        self.assertEqual(history[0].event_id, "event_5")
        self.assertEqual(history[-1].event_id, "event_9")

    def test_clear_history(self):
        """Test clearing history."""
        self.bus.publish(Event(event_type="test"))
        self.bus.publish(Event(event_type="test"))

        self.assertEqual(len(self.bus.get_history()), 2)

        self.bus.clear_history()
        self.assertEqual(len(self.bus.get_history()), 0)

    def test_history_unlimited(self):
        """Test unlimited history (max_history=0)."""
        bus = EventBus(max_history=0)

        # Publish many events
        for i in range(1000):
            bus.publish(Event(event_type="test"))

        history = bus.get_history()
        self.assertEqual(len(history), 1000)


class TestEventBusUtilities(unittest.TestCase):
    """Test utility methods."""

    def setUp(self):
        """Create fresh EventBus for each test."""
        self.bus = EventBus()

    def test_get_subscriber_count(self):
        """Test getting subscriber count."""

        def handler1(event):
            pass

        def handler2(event):
            pass

        self.assertEqual(self.bus.get_subscriber_count("test"), 0)

        self.bus.subscribe("test", handler1)
        self.assertEqual(self.bus.get_subscriber_count("test"), 1)

        self.bus.subscribe("test", handler2)
        self.assertEqual(self.bus.get_subscriber_count("test"), 2)

    def test_get_all_event_types(self):
        """Test getting all subscribed event types."""

        def handler(event):
            pass

        self.assertEqual(len(self.bus.get_all_event_types()), 0)

        self.bus.subscribe("fill", handler)
        self.bus.subscribe("order", handler)

        event_types = self.bus.get_all_event_types()
        self.assertEqual(len(event_types), 2)
        self.assertIn("fill", event_types)
        self.assertIn("order", event_types)


class TestEventBusIntegration(unittest.TestCase):
    """Integration tests with real event scenarios."""

    def test_portfolio_fills_scenario(self):
        """Test realistic scenario: Portfolio processes fills."""
        bus = EventBus()

        # Mock portfolio state
        portfolio_state = {"cash": Decimal("100000"), "position": Decimal("0")}

        def handle_fill(event):
            """Simulate portfolio processing fill."""
            if event.side == "buy":
                portfolio_state["cash"] -= event.quantity * event.price + event.commission
                portfolio_state["position"] += event.quantity
            else:
                portfolio_state["cash"] += event.quantity * event.price - event.commission
                portfolio_state["position"] -= event.quantity

        bus.subscribe("fill", handle_fill)

        # Buy 100 shares at $75
        buy_fill = FillEvent(
            fill_id="f1",
            order_id="o1",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("75.00"),
            commission=Decimal("1.00"),
        )
        bus.publish(buy_fill)

        self.assertEqual(portfolio_state["position"], Decimal("100"))
        self.assertEqual(portfolio_state["cash"], Decimal("92499.00"))

        # Sell 50 shares at $76
        sell_fill = FillEvent(
            fill_id="f2",
            order_id="o2",
            symbol="AAPL",
            side="sell",
            quantity=Decimal("50"),
            price=Decimal("76.00"),
            commission=Decimal("1.00"),
        )
        bus.publish(sell_fill)

        self.assertEqual(portfolio_state["position"], Decimal("50"))
        self.assertEqual(portfolio_state["cash"], Decimal("96298.00"))

    def test_corporate_action_processing(self):
        """Test realistic scenario: Portfolio processes split."""
        bus = EventBus()

        # Mock portfolio state
        portfolio_state = {"position": Decimal("100"), "avg_price": Decimal("75.00")}

        def handle_corporate_action(event):
            """Simulate portfolio processing corporate action."""
            if event.action_type == "split":
                portfolio_state["position"] *= event.split_ratio
                portfolio_state["avg_price"] /= event.split_ratio

        bus.subscribe("corporate_action", handle_corporate_action)

        # 4-for-1 split
        split_event = CorporateActionEvent(
            symbol="AAPL",
            action_type="split",
            split_ratio=Decimal("4.0"),
            effective_date=datetime(2020, 8, 31),
        )
        bus.publish(split_event)

        self.assertEqual(portfolio_state["position"], Decimal("400"))
        self.assertEqual(portfolio_state["avg_price"], Decimal("18.75"))

    def test_multi_service_event_consumption(self):
        """Test multiple services consuming same event."""
        bus = EventBus()

        # Track which services processed the event
        services_called = []

        def portfolio_handler(event):
            services_called.append("portfolio")

        def analytics_handler(event):
            services_called.append("analytics")

        def reporting_handler(event):
            services_called.append("reporting")

        # All services subscribe to fills
        bus.subscribe("fill", portfolio_handler, priority=100)  # Portfolio first
        bus.subscribe("fill", analytics_handler, priority=50)
        bus.subscribe("fill", reporting_handler, priority=10)

        fill = FillEvent(
            fill_id="f1",
            order_id="o1",
            symbol="AAPL",
            side="buy",
            quantity=Decimal("100"),
            price=Decimal("75.00"),
        )
        bus.publish(fill)

        # All services should have been called in priority order
        self.assertEqual(services_called, ["portfolio", "analytics", "reporting"])

    def test_signal_to_order_flow(self):
        """Test complete signal → risk → order → execution flow."""
        bus = EventBus()

        # Track the complete flow
        flow_events = []

        def strategy_generates_signal():
            """Strategy publishes signal."""
            signal = SignalEvent(
                signal_id="sig_123",
                symbol="AAPL",
                direction="buy",
                quantity=Decimal("150"),  # Strategy wants 150
                signal_strength=0.85,
                strategy_name="momentum",
            )
            bus.publish(signal)
            flow_events.append(("signal", signal))

        def risk_evaluates_signal(event: SignalEvent):
            """Risk manager validates and creates order."""
            # Simulate risk check: reduce quantity to 100 (risk limit)
            approved_quantity = min(event.quantity, Decimal("100"))

            if approved_quantity > 0:
                order = OrderEvent(
                    order_id="ord_123",
                    signal_id=event.signal_id,
                    symbol=event.symbol,
                    side=event.direction,
                    quantity=approved_quantity,
                    order_type="market",
                )
                bus.publish(order)
                flow_events.append(("order", order))

        def execution_fills_order(event: OrderEvent):
            """Execution service fills order."""
            fill = FillEvent(
                fill_id="fill_123",
                order_id=event.order_id,
                symbol=event.symbol,
                side=event.side,
                quantity=event.quantity,
                price=Decimal("75.50"),
                commission=Decimal("1.00"),
            )
            bus.publish(fill)
            flow_events.append(("fill", fill))

        def portfolio_processes_fill(event: FillEvent):
            """Portfolio updates position."""
            flow_events.append(("portfolio_updated", event.fill_id))

        # Wire up the flow
        bus.subscribe("signal", risk_evaluates_signal)
        bus.subscribe("order", execution_fills_order)
        bus.subscribe("fill", portfolio_processes_fill)

        # Trigger the flow
        strategy_generates_signal()

        # Verify complete flow (events append in handler, not in publish order)
        self.assertEqual(len(flow_events), 4)
        # Events are appended as they're handled, which happens nested during publish
        # So order will be: portfolio_updated (innermost), fill, order, signal (outermost)
        # Let's verify we have all the right event types
        event_types = [e[0] for e in flow_events]
        self.assertIn("signal", event_types)
        self.assertIn("order", event_types)
        self.assertIn("fill", event_types)
        self.assertIn("portfolio_updated", event_types)

        # Verify signal and order data
        signal_event = next(e[1] for e in flow_events if e[0] == "signal")
        order_event = next(e[1] for e in flow_events if e[0] == "order")
        self.assertEqual(signal_event.quantity, Decimal("150"))
        self.assertEqual(order_event.quantity, Decimal("100"))  # Risk-adjusted


if __name__ == "__main__":
    unittest.main()
