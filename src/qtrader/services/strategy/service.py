"""
Strategy Service Implementation.

Orchestrates strategy instances, routing PriceBarEvents to appropriate strategies
based on their universe filters. Provides Context for strategies to interact with
the backtest engine.
"""

from typing import TYPE_CHECKING

import structlog

from qtrader.events.event_bus import EventBus
from qtrader.events.events import PriceBarEvent
from qtrader.services.strategy.context import Context

if TYPE_CHECKING:
    from qtrader.libraries.strategies import Strategy

logger = structlog.get_logger(__name__)


class StrategyService:
    """
    Orchestrates multiple strategy instances.

    Responsibilities:
    - Manage strategy lifecycle (setup, on_bar, teardown)
    - Route PriceBarEvents to strategies based on universe filtering
    - Provide Context for strategies to emit signals and query data
    - Handle strategy exceptions gracefully (log and continue)

    Event Flow:
    - Subscribes to: PriceBarEvent
    - Strategies publish: SignalEvent (via context.emit_signal)

    Universe Filtering:
    - Each strategy has a universe: list[str] in config
    - Empty list = all symbols
    - Non-empty = only those symbols
    - Example: strategy.universe = ["AAPL", "MSFT"]
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

        # Subscribe to bar events
        self._event_bus.subscribe("bar", self.on_bar)  # type: ignore[arg-type]

        logger.info(
            "strategy.service.initialized",
            strategy_count=len(strategies),
            strategy_names=list(strategies.keys()),
        )

    def setup(self) -> None:
        """
        Initialize all strategies (call setup method).

        Called once before processing any bars.
        """
        for name, strategy in self._strategies.items():
            try:
                context = self._contexts[name]
                strategy.setup(context)
                logger.info(
                    "strategy.service.setup_complete",
                    strategy=name,
                    warmup_bars=strategy.warmup_bars,
                )
            except Exception as e:
                logger.error(
                    "strategy.service.setup_failed",
                    strategy=name,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                self._strategy_metrics[name]["errors"] += 1
                raise

    def teardown(self) -> None:
        """
        Cleanup all strategies (call teardown method).

        Called once after all bars processed.
        """
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

        Args:
            event: Price bar event to route to strategies
        """
        for name, strategy in self._strategies.items():
            # Universe filtering
            if strategy.config.universe and event.symbol not in strategy.config.universe:
                continue  # Skip this strategy for this symbol

            # Route bar to strategy
            try:
                context = self._contexts[name]
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

    def get_metrics(self) -> dict[str, dict]:
        """
        Get metrics for all strategies.

        Returns:
            Dict mapping strategy name to metrics dict
        """
        # Update signals_emitted from context before returning
        for name, context in self._contexts.items():
            self._strategy_metrics[name]["signals_emitted"] = context._signal_count

        return dict(self._strategy_metrics)
