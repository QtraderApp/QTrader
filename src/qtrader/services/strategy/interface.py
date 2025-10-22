"""
Strategy Service Interface.

Defines the protocol for loading and orchestrating external strategy instances.
"""

from typing import Protocol

from qtrader.events.event_bus import EventBus


class IStrategyService(Protocol):
    """Protocol for strategy orchestration service."""

    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize strategy service.

        Args:
            event_bus: Event bus for publishing/subscribing to events
        """
        ...

    def load_strategy(self, strategy_path: str, strategy_id: str, config: dict) -> None:
        """
        Load an external strategy from a Python file.

        Args:
            strategy_path: Path to .py file containing strategy class
            strategy_id: Unique identifier for this strategy instance
            config: Configuration dictionary passed to strategy constructor

        Raises:
            ValueError: If strategy file cannot be loaded or is invalid
        """
        ...
