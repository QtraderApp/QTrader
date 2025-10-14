"""
Example: Buy-and-Hold strategy for golden baseline testing.

This strategy:
- Buys equal weight of all symbols on first bar
- Holds positions until end of backtest
- Does not rebalance

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

from qtrader.api import Context, Strategy
from qtrader.data import MultiBar
from qtrader.models.instrument import DataSource, Instrument, InstrumentType
from qtrader.risk import Signal, SignalDirection, SignalType

# Strategy configuration (no parameters needed for buy-and-hold)
config = {
    "rebalance": False,  # Never rebalance
}

# Backtest configuration for golden baseline
backtest_config = {
    # Data configuration - multiple symbols for diversification
    "instruments": [
        Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK),
    ],
    # Portfolio configuration
    "initial_cash": 100_000.0,
    "position_size": 0.90,
    "max_position_pct": 1.00,
    "allow_shorting": False,
    # Execution configuration
    "max_fill_price_deviation_pct": Decimal("0.1"),
    "max_participation": 0.10,
    # Warmup configuration
    "warmup": False,
    "warmup_bars": 0,
    # Date range
    "start_date": "2019-01-01",
    "end_date": "2023-12-29",
}


class BuyAndHold(Strategy):
    """
    Buy-and-hold strategy for regression testing.

    Buys all symbols on first bar with equal weight allocation,
    then holds until end of backtest.
    """

    def __init__(self, rebalance: bool = False):
        """
        Initialize buy-and-hold strategy.

        Args:
            rebalance: Ignored for this strategy (always False).
        """
        self.symbols_to_buy: set[str] = set()
        self.signal_counter = 0

    def on_init(self, ctx: Context) -> None:
        """Called before warmup. No indicators needed."""
        pass

    def on_start(self, ctx: Context) -> None:
        """Called after warmup completes (if enabled)."""
        print(f"Buy-and-Hold strategy started at {ctx.current_date}")

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
                target_weight=Decimal("0.90"),  # Use 90% of available capital
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
