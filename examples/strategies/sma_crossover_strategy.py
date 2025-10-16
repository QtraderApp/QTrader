"""
Example: SMA Crossover strategy with indicators.

Demonstrates:
- Using StrategyConfig for type-safe configuration
- Using built-in indicators (SMA)
- Indicator tracking for crossover detection
- Signal-based trading with risk management
- YAML-based backtest configuration
"""

from datetime import datetime
from typing import List, Optional

from pydantic import Field, field_validator

from qtrader.api import Context, Strategy
from qtrader.api.strategy import StrategyConfig
from qtrader.data import MultiBar
from qtrader.risk import Signal, SignalDirection, SignalType


class SMAConfig(StrategyConfig):
    """Configuration for SMA Crossover strategy.

    This is loaded from YAML and provides type safety, validation,
    and IDE autocomplete support.
    """

    fast_period: int = Field(20, gt=0, description="Fast SMA period")
    slow_period: int = Field(50, gt=0, description="Slow SMA period")

    @field_validator("slow_period")
    @classmethod
    def slow_must_be_greater(cls, v: int, info) -> int:
        """Ensure slow period is greater than fast period."""
        fast = info.data.get("fast_period")
        if fast and v <= fast:
            raise ValueError(f"slow_period ({v}) must be > fast_period ({fast})")
        return v


class SMACrossoverStrategy(Strategy):
    """
    Simple Moving Average crossover strategy.

    Buys when fast SMA crosses above slow SMA (bullish).
    Sells when fast SMA crosses below slow SMA (bearish).

    Configuration is provided via SMAConfig, which is loaded from
    the YAML backtest configuration file.
    """

    def __init__(self, fast_period: int = 20, slow_period: int = 50):
        """
        Initialize strategy.

        Args:
            fast_period: Fast SMA period (default 20)
            slow_period: Slow SMA period (default 50)

        Note: When used with YAML config, parameters are passed from
              the strategy_config section automatically.
        """
        # Validate parameters using the config class
        self.config = SMAConfig(fast_period=fast_period, slow_period=slow_period)
        self.signal_counter = 0

    def on_init(self, ctx: Context) -> None:
        """Called before warmup. Can register custom indicators here."""
        # Built-in indicators are auto-registered
        pass

    def on_start(self, ctx: Context) -> None:
        """Called after warmup completes (if enabled)."""
        print(f"SMA Crossover strategy started at {ctx.current_date}")
        print(f"  Fast period: {self.config.fast_period}")
        print(f"  Slow period: {self.config.slow_period}")

    def on_bar(self, bar: MultiBar, ctx: Context) -> Optional[List[Signal]]:
        """
        Called for each bar.

        Computes SMAs and checks for crossovers.

        Args:
            bar: MultiBar with all adjustment modes
            ctx: Context with indicators and portfolio state

        Returns:
            List of signals if crossover detected, else None
        """
        # Use adjusted prices for indicators (consistent across splits)
        adjusted_bar = bar.adjusted
        symbol = bar.symbol

        # Compute indicators using config values
        fast_sma = ctx.ind.sma(symbol, self.config.fast_period)
        slow_sma = ctx.ind.sma(symbol, self.config.slow_period)

        # Need both indicators to be valid
        if fast_sma is None or slow_sma is None:
            return None

        # Track indicators for crossover detection
        ctx._track_indicator(symbol, "fast_sma", fast_sma)
        ctx._track_indicator(symbol, "slow_sma", slow_sma)

        signals = []

        # Check for bullish crossover
        if ctx.crossed_above(symbol, "fast_sma", "slow_sma"):
            self.signal_counter += 1
            signals.append(
                Signal(
                    signal_id=f"sma_xover_{self.signal_counter}",
                    strategy_ts=datetime.fromisoformat(adjusted_bar.trade_datetime),
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    direction=SignalDirection.LONG,
                    metadata={
                        "fast_sma": fast_sma,
                        "slow_sma": slow_sma,
                        "fast_period": self.config.fast_period,
                        "slow_period": self.config.slow_period,
                        "reason": "Bullish crossover",
                    },
                )
            )

        # Check for bearish crossover
        elif ctx.crossed_below(symbol, "fast_sma", "slow_sma"):
            self.signal_counter += 1
            signals.append(
                Signal(
                    signal_id=f"sma_xover_{self.signal_counter}",
                    strategy_ts=datetime.fromisoformat(adjusted_bar.trade_datetime),
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    direction=SignalDirection.FLAT,
                    metadata={
                        "fast_sma": fast_sma,
                        "slow_sma": slow_sma,
                        "fast_period": self.config.fast_period,
                        "slow_period": self.config.slow_period,
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
        print("SMA Crossover strategy completed")
        print(f"Total signals generated: {self.signal_counter}")
