"""
Strategy Service Implementation.

Loads and orchestrates external strategy instances, routing BarEvents
to all loaded strategies.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any

from qtrader.events.event_bus import EventBus
from qtrader.events.events import PriceBarEvent


class StrategyService:
    """
    Orchestrates multiple external strategy instances.

    Responsibilities:
    - Load strategy classes from external .py files
    - Instantiate strategies with their configurations
    - Route BarEvents to all strategies
    - Strategies publish SignalEvent when conditions are met

    Event Flow:
    - Subscribes to: BarEvent
    - Strategies publish: SignalEvent (each strategy decides when)
    """

    def __init__(self, event_bus: EventBus) -> None:
        """
        Initialize strategy service.

        Args:
            event_bus: Event bus for publishing/subscribing to events
        """
        self._event_bus = event_bus
        self._strategies: dict[str, Any] = {}  # strategy_id -> strategy instance

        # Subscribe to bar events
        self._event_bus.subscribe("bar", self.on_bar)  # type: ignore[arg-type]

    @classmethod
    def from_config(cls, strategies_config: list[dict[str, Any]], event_bus: EventBus) -> "StrategyService":
        """
        Factory method to create service from configuration.

        Args:
            strategies_config: List of strategy configurations, each with:
                - path: Path to .py file containing strategy class
                - strategy_id: Unique identifier for this strategy
                - config: Configuration dict passed to strategy constructor
            event_bus: Event bus for communication

        Returns:
            Configured StrategyService instance

        Raises:
            ValueError: If any strategy fails to load
        """
        service = cls(event_bus=event_bus)

        for strategy_config in strategies_config:
            service.load_strategy(
                strategy_path=strategy_config["path"],
                strategy_id=strategy_config["strategy_id"],
                config=strategy_config.get("config", {}),
            )

        return service

    def load_strategy(self, strategy_path: str, strategy_id: str, config: dict) -> None:
        """
        Load an external strategy from a Python file.

        The strategy file must contain a class named 'Strategy' with:
        - __init__(self, strategy_id: str, event_bus: EventBus, config: dict)
        - on_bar(self, event: PriceBarEvent) method

        Args:
            strategy_path: Path to .py file containing strategy class
            strategy_id: Unique identifier for this strategy instance
            config: Configuration dictionary passed to strategy constructor

        Raises:
            ValueError: If strategy file cannot be loaded or is invalid
        """
        if strategy_id in self._strategies:
            raise ValueError(f"Strategy '{strategy_id}' already loaded")

        path = Path(strategy_path)
        if not path.exists():
            raise ValueError(f"Strategy file not found: {strategy_path}")

        if not path.suffix == ".py":
            raise ValueError(f"Strategy file must be .py: {strategy_path}")

        # Load module dynamically
        try:
            spec = importlib.util.spec_from_file_location(f"strategy_{strategy_id}", path)
            if spec is None or spec.loader is None:
                raise ValueError(f"Failed to load module from {strategy_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[f"strategy_{strategy_id}"] = module
            spec.loader.exec_module(module)

            # Get Strategy class from module
            if not hasattr(module, "Strategy"):
                raise ValueError(f"Strategy file must contain a class named 'Strategy': {strategy_path}")

            strategy_class = getattr(module, "Strategy")

            # Instantiate strategy
            strategy_instance = strategy_class(strategy_id=strategy_id, event_bus=self._event_bus, config=config)

            self._strategies[strategy_id] = strategy_instance

        except Exception as e:
            raise ValueError(f"Failed to load strategy from {strategy_path}: {e}")

    def on_bar(self, event: PriceBarEvent) -> None:
        """
        Route BarEvent to all loaded strategies.

        Each strategy decides whether to generate a signal based on:
        - Bar data (price, volume, etc.)
        - Is warmup phase (should not generate signals)
        - Internal indicator state

        Args:
            event: Bar event to route to strategies
        """
        for strategy in self._strategies.values():
            if hasattr(strategy, "on_bar"):
                strategy.on_bar(event)
