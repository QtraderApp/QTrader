"""Base Strategy class for user strategies."""

from typing import TYPE_CHECKING, List, Optional, Protocol

from pydantic import BaseModel

from qtrader.risk import Signal

if TYPE_CHECKING:
    from qtrader.data import MultiBar


class StrategyConfig(BaseModel):
    """Base class for strategy-specific configuration.

    Strategies should subclass this to define their parameters with type safety
    and validation. Provides automatic validation, IDE autocomplete, and
    self-documenting configuration.

    Example:
        >>> class SMAConfig(StrategyConfig):
        ...     fast_period: int = Field(50, gt=0, description="Fast SMA period")
        ...     slow_period: int = Field(200, gt=0, description="Slow SMA period")
        ...
        ...     @validator("slow_period")
        ...     def slow_gt_fast(cls, v, values):
        ...         if "fast_period" in values and v <= values["fast_period"]:
        ...             raise ValueError("slow_period must be > fast_period")
        ...         return v

        >>> config = SMAConfig(fast_period=50, slow_period=200)
        >>> # Validation happens automatically

    Benefits:
        - Type safety: Catch configuration errors at load time
        - IDE support: Autocomplete and type hints
        - Validation: Pydantic validators ensure constraints
        - Documentation: Field descriptions serve as inline docs
        - Serialization: Easy conversion to/from YAML, JSON
    """

    class Config:
        frozen = True  # Immutable after creation
        validate_assignment = True  # Validate on attribute changes
        extra = "forbid"  # Reject unknown fields
        use_enum_values = True  # Use enum values instead of enum objects


class Strategy(Protocol):
    """
    Base strategy protocol for user-defined trading strategies.

    Users subclass this and implement on_bar() at minimum.

    Phase 2: Strategies now return Signal objects instead of submitting orders directly.
    The RiskManager evaluates signals and creates appropriately sized orders.

    Phase 3: Added on_init() hook for custom indicator registration before warmup.

    Phase 5: Strategies now receive MultiBar with all adjustment modes available.
    Strategies explicitly select the mode they want to use (typically 'adjusted').
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

    def on_bar(self, bar: "MultiBar", ctx) -> Optional[List[Signal]]:
        """
        Called for each bar in the dataset. Required.

        Args:
            bar: MultiBar with all adjustment modes (unadjusted, adjusted, total_return)
            ctx: Context for accessing indicators and portfolio state

        Returns:
            List of Signal objects (or None/empty list if no signals)

        Note: Signals represent trading INTENT without position sizing.
        The RiskManager will evaluate signals and create appropriately sized orders.

        Example:
            def on_bar(self, bar: MultiBar, ctx) -> Optional[List[Signal]]:
                # Use adjusted prices for indicators (consistent across splits)
                adjusted_bar = bar.adjusted
                price = adjusted_bar.close

                # Or explicitly get a mode
                adjusted_bar = bar.get_bar("adjusted")

                # Generate signals...
                return [signal]
        """
        ...

    def on_fill(self, fill, ctx) -> None:
        """Called after each fill. Optional."""
        pass

    def on_end(self, ctx) -> None:
        """Called once after last bar. Optional."""
        pass
