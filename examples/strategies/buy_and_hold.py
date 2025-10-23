"""
Buy and Hold Strategy - Example Implementation

Demonstrates separation of PROCESS (strategy logic) from PARAMETERS (config values):
- BuyAndHoldConfig: Defines tunable parameters (including name and display_name)
- BuyAndHoldStrategy: Defines trading algorithm

This separation enables:
- Parameter optimization without code changes
- Same strategy with different configs
- Clear distinction between logic and values

Registry Name: Defined in config (default: "buy_and_hold")
Display Name: Defined in config (default: "Buy and Hold Strategy")
"""

from qtrader.events.events import PriceBarEvent
from qtrader.libraries.strategies import BaseStrategy, BaseStrategyConfig, Context


class BuyAndHoldConfig(BaseStrategyConfig):
    """
    Configuration for Buy and Hold strategy.

    This defines the PARAMETERS (tunable values), not the logic.

    Parameters:
        name: Strategy identifier for registry
        display_name: Human-readable name for UI/reports
        hold_period_days: Number of days to hold (None = hold forever)
        warmup_bars: Number of bars for warmup (0 = no warmup needed)

    These can be overridden in backtest YAML:
        strategies:
          - strategy_id: "buy_and_hold"
            config:
              name: "buy_and_hold_30d"
              display_name: "Buy & Hold (30 Days)"
              hold_period_days: 30  # Override: hold 30 days then sell
              warmup_bars: 0
    """

    name: str = "buy_and_hold"
    display_name: str = "Buy and Hold Strategy"
    hold_period_days: int | None = None  # None = hold forever
    warmup_bars: int = 0  # No warmup needed

    class Config:
        """Enforce strict validation for this strategy's config."""

        extra = "forbid"  # Reject typos in YAML configs


class BuyAndHoldStrategy(BaseStrategy):
    """
    Simple Buy and Hold strategy.

    This defines the PROCESS (trading algorithm), not the parameters.

    Strategy Logic:
    - On first bar: Buy the symbol with 100% confidence
    - On subsequent bars: Hold (do nothing)
    - Optional: Sell after hold_period_days (if configured)

    Risk Management:
    - Strategy only generates signals ("I want to buy")
    - Portfolio Manager (RiskService) decides actual position size
    - Portfolio Manager applies risk_policy from backtest config

    Key Design:
    - Strategy class = WHAT TO DO (the algorithm)
    - Config class = SPECIFIC VALUES (tunable parameters)
    - Same strategy, different configs = different behaviors
    """

    def __init__(self, config: BuyAndHoldConfig):
        """
        Initialize strategy with configuration.

        Args:
            config: Buy and hold configuration with tunable parameters
        """
        self.config = config
        self._positions_initiated = set()  # Track symbols we've bought
        self._entry_dates = {}  # Track when we entered (for hold_period)
        self._bar_count = 0  # Track bars processed

    def setup(self, context: "Context") -> None:
        """
        Optional setup phase before trading starts.

        Demonstrates:
        - Config validation
        - Logging initialization
        """
        # Validate configuration
        if self.config.hold_period_days is not None and self.config.hold_period_days <= 0:
            raise ValueError("hold_period_days must be positive")

        # Log strategy initialization
        print(f"[{self.name}] Setup complete")
        if self.config.hold_period_days:
            print(f"  Hold period: {self.config.hold_period_days} days")
        else:
            print(f"  Hold period: Forever (classic buy & hold)")

    def teardown(self, context: "Context") -> None:
        """
        Optional cleanup phase after trading ends.

        Demonstrates:
        - Final statistics logging
        - State summary
        """
        print(f"[{self.name}] Teardown")
        print(f"  Total bars processed: {self._bar_count}")
        print(f"  Symbols traded: {len(self._positions_initiated)}")

    def warmup_bars_required(self) -> int:
        """
        This strategy needs no warmup bars.

        Returns:
            0 (no indicators to warm up)
        """
        return self.config.warmup_bars

    def on_bar(self, event: PriceBarEvent, context: "Context") -> None:
        """
        Process new bar.

        Algorithm:
        1. Check if we already have position
        2. If not, generate buy signal
        3. If hold_period set and exceeded, generate sell signal

        Args:
            event: Price bar event with OHLCV data
            context: Strategy context for checking positions and emitting signals
        """
        self._bar_count += 1  # Track for teardown stats
        symbol = event.symbol

        # Check if we already initiated position for this symbol
        if symbol not in self._positions_initiated:
            # Check current position (in case we're resuming)
            position = context.get_position(symbol)
            if position is not None and position.quantity > 0:
                self._positions_initiated.add(symbol)
                self._entry_dates[symbol] = event.timestamp
                return

            # Generate buy signal with 100% confidence
            # Portfolio Manager will decide actual position size based on risk_policy
            context.emit_signal(
                symbol=symbol,
                direction="BUY",
                confidence=1.0,  # 100% confidence
                reason="Initial buy for buy-and-hold strategy",
            )

            self._positions_initiated.add(symbol)
            self._entry_dates[symbol] = event.timestamp
            return

        # Check if we should sell (if hold_period configured)
        if self.config.hold_period_days is not None:
            entry_date = self._entry_dates.get(symbol)
            if entry_date is not None:
                days_held = (event.timestamp - entry_date).days

                if days_held >= self.config.hold_period_days:
                    # Time to sell
                    context.emit_signal(
                        symbol=symbol,
                        direction="SELL",
                        confidence=1.0,
                        reason=f"Holding period of {self.config.hold_period_days} days reached",
                    )

                    # Reset for potential re-entry
                    self._positions_initiated.remove(symbol)
                    del self._entry_dates[symbol]
