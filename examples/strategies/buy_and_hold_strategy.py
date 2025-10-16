"""
Example: Buy-and-Hold strategy for golden baseline testing.

Demonstrates:
- Using StrategyConfig for type-safe configuration
- Simple buy-and-hold logic
- Equal weight allocation
- YAML-based backtest configuration

This strategy:
- Buys equal weight of all symbols on first bar
- Holds positions until end of backtest
- Optionally rebalances (configurable)

Used as a golden baseline to validate:
- Basic execution (market orders)
- Portfolio tracking over time
- Dividend processing (for long positions)
- Position management
- Commission calculations
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional

from pydantic import Field

from qtrader.api import Context, Strategy
from qtrader.api.strategy import StrategyConfig
from qtrader.data import MultiBar
from qtrader.risk import Signal, SignalDirection, SignalType


class BuyHoldConfig(StrategyConfig):
    """Configuration for Buy and Hold strategy.

    This is loaded from YAML and provides type safety, validation,
    and IDE autocomplete support.
    """

    rebalance: bool = Field(False, description="Whether to rebalance positions periodically")
    target_weight: Decimal = Field(
        Decimal("0.90"), gt=0, le=1, description="Target weight for positions (fraction of capital)"
    )


class BuyAndHoldStrategy(Strategy):
    """
    Buy-and-hold strategy for regression testing.

    Buys all symbols on first bar with equal weight allocation,
    then holds until end of backtest.

    Configuration is provided via BuyHoldConfig, which is loaded from
    the YAML backtest configuration file.
    """

    def __init__(self, rebalance: bool = False, target_weight: Decimal = Decimal("0.90")):
        """
        Initialize buy-and-hold strategy.

        Args:
            rebalance: Whether to rebalance positions (default False)
            target_weight: Target weight for positions (default 0.90)

        Note: When used with YAML config, parameters are passed from
              the strategy_config section automatically.
        """
        self.config = BuyHoldConfig(rebalance=rebalance, target_weight=target_weight)
        self.symbols_to_buy: set[str] = set()
        self.signal_counter = 0

    def on_init(self, ctx: Context) -> None:
        """Called before warmup. No indicators needed."""
        pass

    def on_start(self, ctx: Context) -> None:
        """Called after warmup completes (if enabled)."""
        print(f"Buy-and-Hold strategy started at {ctx.current_date}")
        print(f"  Rebalance: {self.config.rebalance}")
        print(f"  Target weight: {self.config.target_weight}")

    def on_bar(self, bar: MultiBar, ctx: Context) -> Optional[List[Signal]]:
        """
        Generate buy signals on first bar for each symbol.

        Args:
            bar: MultiBar with all adjustment modes available
            ctx: Context with portfolio and market data

        Returns:
            List of buy signals (one per symbol) on first bar, None thereafter
        """
        # Use adjusted prices for decision making (consistent across splits)
        adjusted_bar = bar.adjusted
        symbol = bar.symbol

        # Buy all symbols on their first bar
        if symbol in self.symbols_to_buy:
            self.symbols_to_buy.remove(symbol)
            self.signal_counter += 1

            signal = Signal(
                signal_id=f"BH-{self.signal_counter}",
                strategy_ts=datetime.fromisoformat(adjusted_bar.trade_datetime),
                symbol=symbol,
                signal_type=SignalType.ENTRY_LONG,
                direction=SignalDirection.LONG,
                target_weight=self.config.target_weight,
                metadata={
                    "price": adjusted_bar.close,
                    "reason": "Initial entry",
                },
            )

            print(f"Buy signal: {symbol} @ ${adjusted_bar.close:.2f} on {adjusted_bar.trade_datetime}")
            return [signal]

        return None

    def on_fill(self, fill, ctx: Context) -> None:
        """Called after fills."""
        print(f"Filled: {fill.symbol} {fill.qty} shares @ {fill.price}")

    def on_end(self, ctx: Context) -> None:
        """Called after backtest completes."""
        assert ctx.portfolio is not None  # Portfolio always exists during backtest
        print("Buy-and-Hold strategy completed")
        print(f"Final equity: ${ctx.portfolio.get_equity():,.2f}")
        print(f"Total signals generated: {self.signal_counter}")
