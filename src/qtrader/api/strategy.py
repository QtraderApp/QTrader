"""Base Strategy class for user strategies."""

from typing import List, Optional, Protocol

from qtrader.risk import Signal


class Strategy(Protocol):
    """
    Base strategy protocol for user-defined trading strategies.

    Users subclass this and implement on_bar() at minimum.

    Phase 2: Strategies now return Signal objects instead of submitting orders directly.
    The RiskManager evaluates signals and creates appropriately sized orders.
    """

    def on_start(self, ctx) -> None:
        """Called once before first bar. Optional."""
        pass

    def on_bar(self, bar, ctx) -> Optional[List[Signal]]:
        """
        Called for each bar in the dataset. Required.

        Args:
            bar: Current Bar object
            ctx: Context for accessing indicators and portfolio state

        Returns:
            List of Signal objects (or None/empty list if no signals)

        Note: Signals represent trading INTENT without position sizing.
        The RiskManager will evaluate signals and create appropriately sized orders.
        """
        ...

    def on_fill(self, fill, ctx) -> None:
        """Called after each fill. Optional."""
        pass

    def on_end(self, ctx) -> None:
        """Called once after last bar. Optional."""
        pass
