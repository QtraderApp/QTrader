"""
Bollinger Breakout Strategy.

Trend-following strategy that trades breakouts of Bollinger Bands with confirmation
from volume and RSI indicators.
"""

from pydantic import ConfigDict

from qtrader.events.events import PriceBarEvent
from qtrader.libraries.indicators import BollingerBands
from qtrader.libraries.strategies import Context, Strategy, StrategyConfig
from qtrader.services.strategy.models import SignalIntention


class BollingerBreakoutConfig(StrategyConfig):
    """
    Configuration for Bollinger Bands Breakout Strategy.

    This separates the tunable parameters from the strategy logic,
    enabling parameter optimization without changing the algorithm.
    """

    # Pydantic V2 config
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    # Identity
    name: str = "bollinger_breakout"
    display_name: str = "Bollinger Breakout Strategy"

    # Metadata
    description: str = (
        "Mean reversion strategy using Bollinger Bands. "
        "Opens long positions when price breaks below lower band (oversold), "
        "opens short positions when price breaks above upper band (overbought). "
        "Uses bandwidth as volatility filter and confidence scaling."
    )
    author: str = "QTrader Team"
    created: str = "2024-10-15"
    updated: str = "2024-10-23"
    version: str = "1.1.0"

    # Warmup
    warmup_bars: int = 30  # Enough for BB indicator to warm up

    # Bollinger Bands parameters
    bb_period: int = 20
    bb_num_std: float = 2.0

    # Entry thresholds
    oversold_threshold: float = 0.0
    overbought_threshold: float = 1.0

    # Volatility filter
    min_bandwidth: float = 0.02

    # Signal confidence scaling
    max_confidence: float = 0.9


class BollingerBreakoutStrategy(Strategy):
    """
    Bollinger Bands Breakout Strategy.

    Entry Logic:
    - BUY signal when %B < oversold_threshold (price below lower band)
    - SELL signal when %B > overbought_threshold (price above upper band)

    Volatility Filter:
    - Only trade when bandwidth > min_bandwidth (sufficient volatility)

    Signal Confidence:
    - Higher confidence when further from bands (stronger signal)
    - Scaled by bandwidth (higher volatility = higher confidence)

    This demonstrates:
    1. Using custom indicator (BollingerBands) from my_library
    2. Multi-value indicator usage (upper/middle/lower bands)
    3. Additional indicator properties (bandwidth, %B)
    4. Signal emission with confidence levels
    5. Process/parameter separation (config defines thresholds)

    Note:
    - Strategy emits SIGNALS, not orders
    - RiskService handles position sizing
    - ExecutionService handles order placement
    """

    def __init__(self, config: BollingerBreakoutConfig):
        """Initialize strategy with configuration."""
        super().__init__(config)
        self.config = config  # Store config (Strategy requirement)
        self._config = config  # Private typed reference for type checker

        # Initialize indicators
        self._bb = BollingerBands(
            period=config.bb_period,
            num_std=config.bb_num_std,
            price_field="close",
        )

    @property
    def typed_config(self) -> BollingerBreakoutConfig:
        """Return config with correct type for type checker."""
        assert isinstance(self.config, BollingerBreakoutConfig)
        return self.config

    def setup(self, context: Context) -> None:
        """
        Setup called once before backtesting starts.

        Use for:
        - Validation
        - Loading models
        - Initializing state
        - Logging configuration
        """
        # Log strategy configuration
        print(f"\n{'=' * 80}")
        print(f"Strategy: {self._config.display_name} v{self._config.version}")
        print(f"{'=' * 80}")

        # Metadata
        if self._config.description:
            print("\nDescription:")
            print(f"  {self._config.description}")
        if self._config.author:
            print(f"\nAuthor: {self._config.author}")
        if self._config.created:
            print(f"Created: {self._config.created} | Updated: {self._config.updated}")

        # Parameters
        print("\nBollinger Bands Settings:")
        print(f"  Period: {self._config.bb_period}")
        print(f"  Std Dev: {self._config.bb_num_std}")
        print("\nSignal Thresholds:")
        print(f"  Oversold (OPEN_LONG):  %B < {self._config.oversold_threshold}")
        print(f"  Overbought (OPEN_SHORT): %B > {self._config.overbought_threshold}")
        print("\nVolatility Filter:")
        print(f"  Min Bandwidth: {self._config.min_bandwidth:.2%}")
        print("\nSignal Confidence:")
        print(f"  Max Confidence: {self._config.max_confidence:.1%}")
        print(f"{'=' * 80}\n")

        # Validate thresholds
        if self._config.oversold_threshold >= self._config.overbought_threshold:
            raise ValueError(
                f"oversold_threshold ({self._config.oversold_threshold}) "
                f"must be < overbought_threshold ({self._config.overbought_threshold})"
            )

    def teardown(self, context: Context) -> None:
        """
        Teardown called once after backtesting completes.

        Use for:
        - Final statistics
        - Cleanup
        - Saving results
        - Logging summary
        """
        print(f"\n{'=' * 80}")
        print(f"Strategy Complete: {self.config.display_name}")
        print(f"{'=' * 80}\n")

    def on_bar(self, event: PriceBarEvent, context: Context) -> None:
        """
        Process price bar and emit trading signals.

        Strategy Logic:
        1. Update Bollinger Bands with new bar
        2. Wait for indicator warmup
        3. Apply volatility filter (bandwidth > min_bandwidth)
        4. Emit signals with confidence levels:
           - BUY when %B < oversold_threshold (price below lower band)
           - SELL when %B > overbought_threshold (price above upper band)

        Signal Confidence Calculation:
        - Base confidence from distance to bands (%B value)
        - Adjusted by volatility (bandwidth)
        - Capped at max_confidence

        Args:
            event: Price bar event containing OHLCV data
            context: Runtime context for emitting signals
        """
        symbol = event.symbol
        bar = event.bar

        if bar is None:
            return

        # Update Bollinger Bands
        bands = self._bb.update(bar)

        # Wait for warmup
        if not self._bb.is_ready or bands is None:
            return

        # Get current metrics
        bandwidth = self._bb.bandwidth
        percent_b = self._bb.percent_b

        if bandwidth is None or percent_b is None:
            return

        # Apply volatility filter
        if bandwidth < self._config.min_bandwidth:
            return

        # Generate OPEN_LONG signal: Price below lower band (oversold)
        if percent_b < self._config.oversold_threshold:
            # Calculate confidence based on how far below lower band
            distance = abs(percent_b)  # How far below 0
            confidence = self._calculate_confidence(distance, bandwidth)

            context.emit_signal(
                timestamp=bar.trade_datetime,
                strategy_id=self.config.name,
                symbol=symbol,
                intention=SignalIntention.OPEN_LONG,
                confidence=confidence,
                reason=f"Oversold: %B={percent_b:.3f}, BW={bandwidth:.2%}",
                metadata={
                    "percent_b": percent_b,
                    "bandwidth": bandwidth,
                    "price": bar.close,
                    "lower_band": bands["lower"],
                    "upper_band": bands["upper"],
                },
            )

        # Generate OPEN_SHORT signal: Price above upper band (overbought)
        elif percent_b > self._config.overbought_threshold:
            # Calculate confidence based on how far above upper band
            distance = percent_b - 1.0  # How far above 1.0
            confidence = self._calculate_confidence(distance, bandwidth)

            context.emit_signal(
                timestamp=bar.trade_datetime,
                strategy_id=self.config.name,
                symbol=symbol,
                intention=SignalIntention.OPEN_SHORT,
                confidence=confidence,
                reason=f"Overbought: %B={percent_b:.3f}, BW={bandwidth:.2%}",
                metadata={
                    "percent_b": percent_b,
                    "bandwidth": bandwidth,
                    "price": bar.close,
                    "lower_band": bands["lower"],
                    "upper_band": bands["upper"],
                },
            )

    def _calculate_confidence(self, distance: float, bandwidth: float) -> float:
        """
        Calculate signal confidence from distance and volatility.

        Confidence Factors:
        - Distance: How far price is from bands (higher = stronger signal)
        - Bandwidth: Market volatility (higher = more confident in trends)

        Formula:
        - Base confidence from distance (capped at 1.0)
        - Volatility adjustment factor from bandwidth
        - Combined and capped at max_confidence

        Args:
            distance: Absolute distance from normal range (0.0+)
            bandwidth: Current Bollinger Band bandwidth (0.0-1.0)

        Returns:
            Confidence level between 0.0 and max_confidence
        """
        # Base confidence from distance (more extreme = higher confidence)
        # Scale by 2.0 so %B of -0.5 gives full confidence
        base_confidence = min(abs(distance) * 2.0, 1.0)

        # Volatility factor (higher bandwidth = more trending = more confident)
        # Use 0.05 (5%) as reference bandwidth
        volatility_factor = min(bandwidth / 0.05, 1.0)

        # Combine: 70% base + 30% volatility adjustment
        confidence = base_confidence * (0.7 + 0.3 * volatility_factor)

        # Cap at configured maximum
        return min(confidence, self._config.max_confidence)
