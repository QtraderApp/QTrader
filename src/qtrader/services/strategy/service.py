"""
Strategy Service Implementation.

Orchestrates strategy instances, routing PriceBarEvents to appropriate strategies
based on their universe filters. Provides Context for strategies to interact with
the backtest engine.
"""

from typing import TYPE_CHECKING

import structlog

from qtrader.events.event_bus import EventBus
from qtrader.events.events import FillEvent, PriceBarEvent
from qtrader.services.strategy.context import Context

if TYPE_CHECKING:
    from qtrader.libraries.strategies import Strategy

logger = structlog.get_logger(__name__)


class StrategyService:
    """
    Orchestrates multiple strategy instances.

    Responsibilities:
    - Manage strategy lifecycle (setup, on_bar, on_position_filled, teardown)
    - Route PriceBarEvents to strategies based on universe filtering
    - Route FillEvents to strategies for position tracking (event-driven)
    - Provide Context for strategies to emit signals and query data
    - Handle strategy exceptions gracefully (log and continue)

    Event Flow:
    - Subscribes to: PriceBarEvent, FillEvent
    - Strategies publish: SignalEvent (via context.emit_signal)

    Universe Filtering:
    - Each strategy has a universe: list[str] in config
    - Empty list = all symbols
    - Non-empty = only those symbols
    - Example: strategy.universe = ["AAPL", "MSFT"]
    - Applies to both bars and fills
    """

    def __init__(self, event_bus: EventBus, strategies: dict[str, "Strategy"]) -> None:
        """
        Initialize strategy service.

        Args:
            event_bus: Event bus for publishing/subscribing to events
            strategies: Dict mapping strategy name to strategy instance
        """
        self._event_bus = event_bus
        self._strategies = strategies
        self._contexts: dict[str, Context] = {}
        self._strategy_metrics: dict[str, dict] = {}
        self._quarantined: set[str] = set()  # Track strategies that failed setup
        self._subscription_tokens: list = []  # Track subscriptions for cleanup

        # Create Context for each strategy
        for name, strategy in strategies.items():
            self._contexts[name] = Context(
                strategy_id=name,
                event_bus=event_bus,
            )
            self._strategy_metrics[name] = {
                "bars_processed": 0,
                "signals_emitted": 0,
                "errors": 0,
            }

        # Subscribe to bar events (store token for cleanup)
        token = self._event_bus.subscribe("bar", self.on_bar)  # type: ignore[arg-type]
        self._subscription_tokens.append(token)

        # Subscribe to fill events for position tracking (store token for cleanup)
        token = self._event_bus.subscribe("fill", self.on_fill)  # type: ignore[arg-type]
        self._subscription_tokens.append(token)

        logger.info(
            "strategy.service.initialized",
            strategy_count=len(strategies),
            strategy_names=list(strategies.keys()),
        )

    def setup(self) -> None:
        """
        Initialize all strategies (call setup method).

        Called once before processing any bars. Failures are logged and tracked
        but do NOT abort the entire service - healthy strategies continue trading.
        Failed strategies are quarantined (no further event routing).

        Quarantined strategies:
        - Are added to self._quarantined set on setup failure
        - Do not receive bar or fill events
        - Are marked as "quarantined": true in get_metrics()
        """
        for name, strategy in self._strategies.items():
            try:
                context = self._contexts[name]
                strategy.setup(context)
                # Remove from quarantine on successful setup (handles retry scenarios)
                self._quarantined.discard(name)
                logger.info(
                    "strategy.service.setup_complete",
                    strategy=name,
                )
            except Exception as e:
                logger.error(
                    "strategy.service.setup_failed",
                    strategy=name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._strategy_metrics[name]["errors"] += 1
                self._quarantined.add(name)  # Quarantine failed strategy
                logger.warning(
                    "strategy.service.strategy_quarantined",
                    strategy=name,
                    reason="setup_failed",
                )
                # Continue with other strategies - don't raise

    def teardown(self) -> None:
        """
        Cleanup all strategies (call teardown method).

        Called once after all bars processed. Also unsubscribes from event bus
        to prevent duplicate handlers if service is recreated.
        """
        # Unsubscribe from event bus first
        for token in self._subscription_tokens:
            try:
                token.unsubscribe()
            except Exception as e:
                logger.warning(
                    "strategy.service.unsubscribe_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )

        # Then teardown strategies
        for name, strategy in self._strategies.items():
            try:
                context = self._contexts[name]
                strategy.teardown(context)
                logger.info(
                    "strategy.service.teardown_complete",
                    strategy=name,
                    metrics=self._strategy_metrics[name],
                )
            except Exception as e:
                logger.warning(
                    "strategy.service.teardown_failed",
                    strategy=name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._strategy_metrics[name]["errors"] += 1

    def on_bar(self, event: PriceBarEvent) -> None:
        """
        Route PriceBarEvent to strategies based on universe filtering.

        Caches bar in each strategy's context before calling on_bar(),
        enabling strategies to query historical data via context.get_bars().

        Quarantined strategies (failed setup) are skipped.

        Args:
            event: Price bar event to route to strategies
        """
        for name, strategy in self._strategies.items():
            # Skip quarantined strategies
            if name in self._quarantined:
                continue

            # Universe filtering
            if strategy.config.universe and event.symbol not in strategy.config.universe:
                continue  # Skip this strategy for this symbol

            # Cache bar for historical queries
            context = self._contexts[name]
            context.cache_bar(event)

            # Route bar to strategy
            try:
                strategy.on_bar(event, context)
                self._strategy_metrics[name]["bars_processed"] += 1
            except Exception as e:
                logger.error(
                    "strategy.service.on_bar_error",
                    strategy=name,
                    symbol=event.symbol,
                    timestamp=event.timestamp,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._strategy_metrics[name]["errors"] += 1

    def on_fill(self, event: FillEvent) -> None:
        """
        Route FillEvent to strategies for position tracking.

        Enables event-driven position tracking via strategy.on_position_filled().

        Routing Logic:
        1. If event.strategy_id is populated, route ONLY to that strategy
        2. If event.strategy_id is None/empty, fall back to universe filtering
           (route to all strategies that trade this symbol)

        Quarantined strategies (failed setup) are skipped in both paths.

        This prevents multi-strategy portfolios from corrupting state when
        multiple strategies trade the same symbol - each fill goes only to
        the strategy that originated it.

        Args:
            event: Fill event from execution service
        """
        # Strategy-specific routing: if strategy_id is populated, route only to that strategy
        if event.strategy_id:
            # Skip quarantined strategies
            if event.strategy_id in self._quarantined:
                return

            if event.strategy_id not in self._strategies:
                logger.warning(
                    "strategy.service.on_fill_unknown_strategy",
                    strategy_id=event.strategy_id,
                    symbol=event.symbol,
                    fill_id=event.fill_id,
                    message="Fill references unknown strategy, skipping",
                )
                return

            # Route to the specific strategy
            strategy = self._strategies[event.strategy_id]
            context = self._contexts[event.strategy_id]
            try:
                strategy.on_position_filled(event, context)
            except Exception as e:
                logger.error(
                    "strategy.service.on_fill_error",
                    strategy=event.strategy_id,
                    symbol=event.symbol,
                    fill_id=event.fill_id,
                    side=event.side,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._strategy_metrics[event.strategy_id]["errors"] += 1
            return

        # Fallback: universe filtering when strategy_id is not provided
        # Route to all strategies that trade this symbol
        for name, strategy in self._strategies.items():
            # Skip quarantined strategies
            if name in self._quarantined:
                continue

            # Universe filtering - only route fills for symbols in strategy universe
            if strategy.config.universe and event.symbol not in strategy.config.universe:
                continue  # Skip this strategy for this symbol

            # Route fill to strategy's on_position_filled handler
            context = self._contexts[name]
            try:
                strategy.on_position_filled(event, context)
            except Exception as e:
                logger.error(
                    "strategy.service.on_fill_error",
                    strategy=name,
                    symbol=event.symbol,
                    fill_id=event.fill_id,
                    side=event.side,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._strategy_metrics[name]["errors"] += 1

    def get_metrics(self) -> dict[str, dict]:
        """
        Get metrics for all strategies.

        Returns:
            Dict mapping strategy name to metrics dict including:
            - bars_processed: Number of bars successfully processed
            - signals_emitted: Number of signals emitted
            - errors: Number of errors encountered
            - quarantined: True if strategy failed setup and is inactive
        """
        # Update signals_emitted from context before returning
        for name, context in self._contexts.items():
            self._strategy_metrics[name]["signals_emitted"] = context.signal_count
            self._strategy_metrics[name]["quarantined"] = name in self._quarantined

        return dict(self._strategy_metrics)
