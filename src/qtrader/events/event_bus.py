"""
EventBus implementation for QTrader event-driven architecture.

Provides synchronous, deterministic publish/subscribe infrastructure
optimized for backtesting. Events are dispatched in priority order with
error isolation and complete history tracking.

Key Features:
- Synchronous execution (no async/threading)
- Deterministic ordering (priority-based)
- Error isolation (one handler failure doesn't stop others)
- Event history for replay and debugging
- Memory-bounded history (configurable)
"""

from collections import defaultdict
from datetime import datetime
from typing import Callable, Protocol

# TODO: Re-enable after EventStore is rebuilt for new event system
# from qtrader.events.event_store import EventStore
from qtrader.events.events import BaseEvent as Event
from qtrader.system import LoggerFactory

logger = LoggerFactory.get_logger()


class IEventBus(Protocol):
    """
    Event bus interface for publish/subscribe messaging.

    Enables loose coupling between services by allowing them to communicate
    via events rather than direct method calls. Services publish events when
    something interesting happens, and other services subscribe to receive
    those events.

    Benefits:
    - Services don't need to know about each other
    - One event can have multiple consumers
    - Easy to add new services without modifying existing ones
    - Complete audit trail of all events
    - Enables replay for debugging and testing

    Usage:
        >>> bus = EventBus()
        >>>
        >>> # Service subscribes to events
        >>> def handle_fill(event: FillEvent):
        ...     portfolio.apply_fill(event)
        >>> bus.subscribe("fill", handle_fill, priority=10)
        >>>
        >>> # Service publishes events
        >>> fill_event = FillEvent(...)
        >>> bus.publish(fill_event)  # Handler called synchronously
    """

    def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.

        Calls all registered handlers for this event type synchronously
        in priority order. If a handler raises an exception, it is logged
        but other handlers continue to be called.

        Args:
            event: Event to publish

        Example:
            >>> bus.publish(PriceBarEvent(symbol="AAPL", bar=bar))
        """
        ...

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
        priority: int = 0,
    ) -> None:
        """
        Subscribe to event type.

        Registers a handler to be called when events of the specified type
        are published. Handlers with higher priority are called first.

        Args:
            event_type: Type of event to subscribe to (e.g., 'fill', 'price_bar')
            handler: Callback function to handle event
            priority: Handler priority (higher = called first, default=0)

        Example:
            >>> def my_handler(event):
            ...     print(f"Received: {event}")
            >>> bus.subscribe("fill", my_handler, priority=10)
        """
        ...

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> None:
        """
        Unsubscribe from event type.

        Removes a previously registered handler. If handler was not
        subscribed, this is a no-op.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove

        Example:
            >>> bus.unsubscribe("fill", my_handler)
        """
        ...

    def get_history(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """
        Get event history with optional filters.

        Useful for debugging, replay, and analysis. Returns events in
        chronological order.

        Args:
            event_type: Filter by event type (None = all types)
            since: Filter by timestamp (None = all time)
            limit: Max events to return (None = no limit)

        Returns:
            List of events matching filters

        Example:
            >>> # Get all fills in last hour
            >>> fills = bus.get_history(
            ...     event_type="fill",
            ...     since=datetime.now() - timedelta(hours=1)
            ... )
        """
        ...

    def clear_history(self) -> None:
        """
        Clear event history.

        Useful for starting a new backtest with clean state.

        Example:
            >>> bus.clear_history()  # Start fresh
        """
        ...


class EventBus:
    """
    Synchronous event bus for deterministic backtesting.

    Features:
    - Synchronous execution: publish() blocks until all handlers complete
    - Deterministic ordering: Handlers called in priority order (highest first)
    - Error isolation: One handler failure doesn't stop others
    - Event history: All events stored for replay/debugging
    - Memory bounded: History capped to prevent memory issues

    Thread Safety: NOT thread-safe (backtesting is single-threaded)

    Performance: Optimized for in-memory, single-threaded backtesting.
    Suitable for millions of events.

    Example:
        >>> # Setup
        >>> bus = EventBus(max_history=100_000)
        >>>
        >>> # Subscribe with priority
        >>> bus.subscribe("fill", portfolio.handle_fill, priority=100)
        >>> bus.subscribe("fill", analytics.record_fill, priority=50)
        >>>
        >>> # Publish (handlers called in priority order)
        >>> bus.publish(FillEvent(...))
        >>>
        >>> # Query history
        >>> recent_fills = bus.get_history(event_type="fill", limit=10)
    """

    def __init__(self, max_history: int = 100_000):
        """
        Initialize event bus.

        Args:
            max_history: Maximum events to keep in history (0 = unlimited).
                        When limit reached, oldest events are discarded.
                        Default 100k events ≈ 40 years of daily backtesting.
        """
        # Store handlers by event type: {event_type: [(priority, handler), ...]}
        self._subscribers: dict[str, list[tuple[int, Callable]]] = defaultdict(list)

        # Event history (chronological order)
        self._event_history: list[Event] = []
        self._max_history = max_history

        # DEBUG: Internal initialization
        logger.debug(
            "event_bus.initialized",
            max_history=max_history,
        )

    def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.

        Processing order:
        1. Add event to history
        2. Get handlers for this event type
        3. Sort handlers by priority (highest first)
        4. Call each handler synchronously
        5. If handler raises exception, log it but continue

        This ensures:
        - Deterministic execution (same events → same order)
        - Error isolation (one failure doesn't cascade)
        - Complete audit trail (all events in history)

        Args:
            event: Event to publish
        """
        # Store in history (with memory bounding)
        self._event_history.append(event)
        if self._max_history > 0 and len(self._event_history) > self._max_history:
            # Keep only most recent events
            self._event_history = self._event_history[-self._max_history :]

        # Get handlers for this event type
        handlers = self._subscribers.get(event.event_type, [])

        # Sort by priority (higher first)
        sorted_handlers = sorted(handlers, key=lambda x: x[0], reverse=True)

        # Call each handler (with error isolation)
        for priority, handler in sorted_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "event_bus.handler_error",
                    event_type=event.event_type,
                    event_id=event.event_id,
                    handler=handler.__name__,
                    error=str(e),
                )
                # Continue to next handler (error isolation)

        logger.debug(
            "event_bus.published",
            event_type=event.event_type,
            event_id=event.event_id,
            num_handlers=len(sorted_handlers),
        )

    def subscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
        priority: int = 0,
    ) -> None:
        """
        Subscribe to event type.

        Handlers with higher priority are called first. Within same priority,
        order is determined by subscription order (FIFO).

        Args:
            event_type: Type of event to subscribe to
            handler: Callback function to handle event
            priority: Handler priority (higher = called first, default=0)
        """
        self._subscribers[event_type].append((priority, handler))

        # DEBUG: Subscription details (internal plumbing)
        logger.debug(
            "event_bus.subscribed",
            event_type=event_type,
            handler=handler.__name__,
            priority=priority,
            total_handlers=len(self._subscribers[event_type]),
        )

    def unsubscribe(
        self,
        event_type: str,
        handler: Callable[[Event], None],
    ) -> None:
        """
        Unsubscribe from event type.

        Removes handler from subscriber list. If handler was not subscribed,
        this is a no-op (idempotent).

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler to remove
        """
        if event_type not in self._subscribers:
            return

        # Remove all entries for this handler (may have different priorities)
        original_count = len(self._subscribers[event_type])
        self._subscribers[event_type] = [(p, h) for p, h in self._subscribers[event_type] if h != handler]
        removed_count = original_count - len(self._subscribers[event_type])

        if removed_count > 0:
            # DEBUG: Unsubscription details
            logger.debug(
                "event_bus.unsubscribed",
                event_type=event_type,
                handler=handler.__name__,
                removed_count=removed_count,
            )

    def get_history(
        self,
        event_type: str | None = None,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> list[Event]:
        """
        Get event history with optional filters.

        Filters are applied in order:
        1. Filter by event type (if specified)
        2. Filter by timestamp (if specified)
        3. Limit results (if specified)

        Args:
            event_type: Filter by event type (None = all types)
            since: Filter by timestamp (None = all time)
            limit: Max events to return (None = no limit)

        Returns:
            List of events in chronological order
        """
        events = self._event_history

        # Filter by type
        if event_type is not None:
            events = [e for e in events if e.event_type == event_type]

        # Filter by time
        if since is not None:
            events = [e for e in events if e.occurred_at >= since]

        # Limit results (take most recent)
        if limit is not None:
            events = events[-limit:]

        return events

    def clear_history(self) -> None:
        """
        Clear event history.

        Useful for starting a new backtest run with clean state.
        Does not affect subscriptions.
        """
        self._event_history.clear()
        # DEBUG: History management
        logger.debug("event_bus.history_cleared")

    def get_subscriber_count(self, event_type: str) -> int:
        """
        Get number of subscribers for event type.

        Useful for debugging and monitoring.

        Args:
            event_type: Event type to check

        Returns:
            Number of registered handlers for this event type
        """
        return len(self._subscribers.get(event_type, []))

    def get_all_event_types(self) -> list[str]:
        """
        Get list of all event types with subscribers.

        Useful for debugging and monitoring.

        Returns:
            List of event types that have at least one subscriber
        """
        return list(self._subscribers.keys())
