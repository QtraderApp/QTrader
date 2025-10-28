"""
Broken strategy - for testing error handling.

This file has import errors and should be skipped gracefully.
"""

# This import will fail
from nonexistent_module import something_broken  # pyright: ignore[reportMissingImports]  # noqa: F401

from qtrader.events.events import PriceBarEvent
from qtrader.libraries.strategies import Context, Strategy, StrategyConfig


class BrokenConfig(StrategyConfig):
    name: str = "broken_strategy"
    display_name: str = "Broken Strategy"
    warmup_bars: int = 0


CONFIG = BrokenConfig()


class BrokenStrategy(Strategy):
    def __init__(self, config: BrokenConfig):
        self.config = config

    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        pass
