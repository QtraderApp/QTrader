"""
SMA Crossover Strategy - Demonstrates Phase 4 Context Enhancement.

This strategy showcases the new Context capabilities:
- get_bars() for historical data access
- get_price() for current price
- Stateful decision making without internal state management

Strategy Logic:
- Calculate fast (20-period) and slow (50-period) Simple Moving Averages
- OPEN_LONG when fast SMA crosses above slow SMA (golden cross)
- CLOSE_LONG when fast SMA crosses below slow SMA (death cross)

Key Features:
- Uses context.get_bars() to access historical prices
- No internal state storage - all state via Context
- Demonstrates proper separation: Strategy decides WHAT, Manager decides HOW MUCH
"""

from decimal import Decimal

from qtrader.events.events import PriceBarEvent
from qtrader.libraries.strategies import Context, Strategy, StrategyConfig
from qtrader.services.strategy.models import SignalIntention


class SMAConfig(StrategyConfig):
    """Configuration for SMA Crossover strategy."""

    name: str = "sma_crossover"
    display_name: str = "SMA Crossover"
    universe: list[str] = []  # Apply to all symbols by default

    # Strategy-specific parameters
    fast_period: int = 20  # Fast SMA period
    slow_period: int = 50  # Slow SMA period (strategy needs this many bars minimum)
    confidence: Decimal = Decimal("0.75")  # Signal confidence


# Export config for auto-discovery
CONFIG = SMAConfig()


class SMACrossover(Strategy):
    """
    SMA Crossover Strategy using Phase 4 Context enhancements.

    Demonstrates:
    - Historical bar access via context.get_bars()
    - Technical indicator calculation (SMA)
    - Stateful decisions without internal state
    - Proper signal emission with metadata
    """

    def __init__(self, config: SMAConfig):
        """
        Initialize SMA Crossover strategy.

        Args:
            config: Strategy configuration with SMA periods
        """
        # Type assertion for IDE/type checker
        assert isinstance(config, SMAConfig)
        self.config = config
        # Note: No internal state for positions or prices!
        # Context provides all the state we need

    def setup(self, context: Context) -> None:
        """
        Strategy setup (called once before first bar).

        Args:
            context: Strategy execution context
        """
        pass  # No setup needed - Context handles bar caching

    def teardown(self, context: Context) -> None:
        """
        Strategy teardown (called once after last bar).

        Args:
            context: Strategy execution context
        """
        pass  # No cleanup needed

    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        """
        Process bar and generate signals based on SMA crossover.

        Args:
            event: Current price bar
            context: Strategy execution context
        """
        symbol = event.symbol

        # Get historical bars for SMA calculation
        # Note: No need to check event.is_warmup - get_bars() handles insufficient data
        # Need slow_period bars to calculate both SMAs
        bars = context.get_bars(symbol, n=self.config.slow_period)

        # Wait until we have enough data
        if bars is None or len(bars) < self.config.slow_period:
            return

        # Calculate fast SMA (most recent fast_period bars)
        fast_bars = bars[-self.config.fast_period :]
        fast_prices = [bar.close for bar in fast_bars]
        fast_sma = sum(fast_prices) / len(fast_prices)

        # Calculate slow SMA (all bars)
        slow_prices = [bar.close for bar in bars]
        slow_sma = sum(slow_prices) / len(slow_prices)

        # Get previous bars for crossover detection
        prev_bars = context.get_bars(symbol, n=self.config.slow_period + 1)
        if prev_bars is None or len(prev_bars) < self.config.slow_period + 1:
            return  # Not enough history for crossover detection

        # Calculate previous SMAs
        prev_fast_bars = prev_bars[-(self.config.fast_period + 1) : -1]
        prev_fast_prices = [bar.close for bar in prev_fast_bars]
        prev_fast_sma = sum(prev_fast_prices) / len(prev_fast_prices)

        prev_slow_bars = prev_bars[:-1]
        prev_slow_prices = [bar.close for bar in prev_slow_bars]
        prev_slow_sma = sum(prev_slow_prices) / len(prev_slow_prices)

        # Detect crossovers
        golden_cross = prev_fast_sma <= prev_slow_sma and fast_sma > slow_sma
        death_cross = prev_fast_sma >= prev_slow_sma and fast_sma < slow_sma

        # Get current price for signal
        current_price = context.get_price(symbol)
        if current_price is None:
            return

        # Generate signals on crossovers
        if golden_cross:
            # Fast SMA crossed above slow SMA - bullish signal
            context.emit_signal(
                timestamp=event.timestamp,
                symbol=symbol,
                intention=SignalIntention.OPEN_LONG,
                price=current_price,
                confidence=self.config.confidence,
                reason=f"Golden cross: fast SMA ({fast_sma:.2f}) > slow SMA ({slow_sma:.2f})",
                metadata={
                    "fast_sma": float(fast_sma),
                    "slow_sma": float(slow_sma),
                    "prev_fast_sma": float(prev_fast_sma),
                    "prev_slow_sma": float(prev_slow_sma),
                    "crossover_type": "golden",
                },
            )

        elif death_cross:
            # Fast SMA crossed below slow SMA - bearish signal
            context.emit_signal(
                timestamp=event.timestamp,
                symbol=symbol,
                intention=SignalIntention.CLOSE_LONG,
                price=current_price,
                confidence=self.config.confidence,
                reason=f"Death cross: fast SMA ({fast_sma:.2f}) < slow SMA ({slow_sma:.2f})",
                metadata={
                    "fast_sma": float(fast_sma),
                    "slow_sma": float(slow_sma),
                    "prev_fast_sma": float(prev_fast_sma),
                    "prev_slow_sma": float(prev_slow_sma),
                    "crossover_type": "death",
                },
            )

        # Note: No position checking here!
        # Strategy declares INTENT (OPEN_LONG/CLOSE_LONG)
        # RiskManager/PositionSizer decides:
        # - IF to take the signal (based on risk limits)
        # - HOW MUCH to trade (based on buying power, position limits)
        # - WHETHER we're already in a position (PortfolioService)
