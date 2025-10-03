"""Base Strategy class for user strategies."""

from typing import Protocol


class Strategy(Protocol):
    """
    Base strategy protocol for user-defined trading strategies.

    Users subclass this and implement on_bar() at minimum.
    """

    def on_start(self, ctx) -> None:
        """Called once before first bar. Optional."""
        pass

    def on_bar(self, bar, ctx) -> None:
        """
        Called for each bar in the dataset. Required.

        Args:
            bar: Current Bar object
            ctx: Context for accessing indicators and submitting orders
        """
        ...

    def on_fill(self, fill, ctx) -> None:
        """Called after each fill. Optional."""
        pass

    def on_end(self, ctx) -> None:
        """Called once after last bar. Optional."""
        pass
