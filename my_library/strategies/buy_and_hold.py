"""
Buy and Hold Strategy.

The simplest possible strategy - buy on the first bar and hold forever.
Demonstrates minimal strategy implementation with no indicators or complex logic.
"""

from pydantic import ConfigDict

from qtrader.contracts.strategies import SignalIntention
from qtrader.events.events import PriceBarEvent
from qtrader.libraries.strategies import BaseStrategy, BaseStrategyConfig, Context


class BuyAndHoldConfig(BaseStrategyConfig):
    """
    Configuration for Buy and Hold Strategy.

    This is the simplest possible config - just identity fields.
    No additional parameters needed since the strategy has no logic to tune.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Identity
    name: str = "buy_and_hold"
    display_name: str = "Buy and Hold"

    # Metadata
    description: str = "Simplest possible strategy - buy on first bar and hold forever"
    author: str = "QTrader Team"
    created: str = "2024-10-23"
    updated: str = "2024-10-23"
    version: str = "1.0.0"

    # Warmup
    warmup_bars: int = 0  # No warmup needed

    # Signal confidence (always max confidence)
    confidence: float = 1.0


class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold Strategy.

    Strategy Logic:
    - On first bar (t=0), emit OPEN_LONG signal with maximum confidence
    - Hold forever (no more signals)

    This demonstrates:
    1. Minimal strategy implementation
    2. No indicators needed
    3. No warmup required
    4. Single signal emission
    5. Stateful tracking (bought flag)
    """

    def __init__(self, config: BuyAndHoldConfig):
        """Initialize strategy with configuration."""
        super().__init__(config)
        self.config = config
        self._config = config
        self._bought = False  # Track if we've bought

    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        """
        Process price bar - buy on first bar only.

        Args:
            event: Price bar event
            context: Strategy context for signal emission
        """
        # Skip if already bought
        if self._bought:
            return

        bar = event.bar
        if bar is None:
            return

        # Buy on first bar
        context.emit_signal(
            timestamp=bar.trade_datetime,
            strategy_id=self.config.name,
            symbol=event.symbol,
            intention=SignalIntention.OPEN_LONG,
            confidence=self._config.confidence,
            reason="Buy and hold - initial purchase",
            metadata={"price": bar.close},
        )

        # Mark as bought
        self._bought = True
