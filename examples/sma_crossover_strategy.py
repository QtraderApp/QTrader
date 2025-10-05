"""
Example: SMA Crossover strategy with indicators.

Demonstrates:
- Using built-in indicators (SMA)
- Indicator tracking for crossover detection
- Signal-based trading with risk management
"""

from datetime import datetime
from typing import List, Optional

from qtrader.api import Context, Strategy
from qtrader.models.bar import Bar
from qtrader.risk import Signal, SignalDirection, SignalType


class SMACrossover(Strategy):
    """
    Simple Moving Average crossover strategy.

    Buys when fast SMA crosses above slow SMA (bullish).
    Sells when fast SMA crosses below slow SMA (bearish).
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize strategy.

        Args:
            fast_period: Fast SMA period (default 20)
            slow_period: Slow SMA period (default 50)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_counter = 0

    def on_init(self, ctx: Context) -> None:
        """Called before warmup. Can register custom indicators here."""
        # Built-in indicators are auto-registered
        pass

    def on_start(self, ctx: Context) -> None:
        """Called after warmup completes (if enabled)."""
        print(f"Strategy started at {ctx.current_date}")

    def on_bar(self, bar: Bar, ctx: Context) -> Optional[List[Signal]]:
        """
        Called for each bar.

        Computes SMAs and checks for crossovers.
        """
        # Compute indicators
        fast_sma = ctx.ind.sma(bar.symbol, self.fast_period)
        slow_sma = ctx.ind.sma(bar.symbol, self.slow_period)

        # Need both indicators to be valid
        if fast_sma is None or slow_sma is None:
            return None

        # Track indicators for crossover detection
        ctx._track_indicator(bar.symbol, "fast_sma", fast_sma)
        ctx._track_indicator(bar.symbol, "slow_sma", slow_sma)

        signals = []

        # Check for bullish crossover
        if ctx.crossed_above(bar.symbol, "fast_sma", "slow_sma"):
            self.signal_counter += 1
            signals.append(
                Signal(
                    signal_id=f"sma_xover_{self.signal_counter}",
                    strategy_ts=datetime.now(),
                    symbol=bar.symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    direction=SignalDirection.LONG,
                    metadata={
                        "fast_sma": fast_sma,
                        "slow_sma": slow_sma,
                        "reason": "Bullish crossover",
                    },
                )
            )

        # Check for bearish crossover
        elif ctx.crossed_below(bar.symbol, "fast_sma", "slow_sma"):
            self.signal_counter += 1
            signals.append(
                Signal(
                    signal_id=f"sma_xover_{self.signal_counter}",
                    strategy_ts=datetime.now(),
                    symbol=bar.symbol,
                    signal_type=SignalType.EXIT_LONG,
                    direction=SignalDirection.FLAT,
                    metadata={
                        "fast_sma": fast_sma,
                        "slow_sma": slow_sma,
                        "reason": "Bearish crossover",
                    },
                )
            )

        return signals if signals else None

    def on_fill(self, fill, ctx: Context) -> None:
        """Called after fills."""
        pass

    def on_end(self, ctx: Context) -> None:
        """Called after backtest completes."""
        print("Strategy completed")
