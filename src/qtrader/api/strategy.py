"""Base Strategy class for user strategies."""

from typing import List, Optional, Protocol

from qtrader.risk import Signal


class Strategy(Protocol):
    """
    Base strategy protocol for user-defined trading strategies.

    Users subclass this and implement on_bar() at minimum.

    Phase 2: Strategies now return Signal objects instead of submitting orders directly.
    The RiskManager evaluates signals and creates appropriately sized orders.

    Phase 3: Added on_init() hook for custom indicator registration before warmup.
    """

    def on_init(self, ctx) -> None:
        """
        Called once before warmup (if enabled) or first bar.

        Use this to register custom indicators that need to be included
        in warmup period lookback calculation.

        Optional hook.

        Args:
            ctx: Context for indicator registration

        Example:
            def on_init(self, ctx):
                ctx.ind.register("momentum", CustomMomentum(period=20))
        """
        pass

    def on_start(self, ctx) -> None:
        """
        Called once after warmup completes (if enabled) or before first bar.

        When warmup is enabled, all indicators will have valid values
        when this is called.

        Optional hook.

        Args:
            ctx: Context for setup operations
        """
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
