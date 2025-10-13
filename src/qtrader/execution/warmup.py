"""Warmup system for indicators before trading starts."""

from typing import TYPE_CHECKING, List

from qtrader.config.logging_config import LoggerFactory

if TYPE_CHECKING:
    from qtrader.api.context import Context
    from qtrader.models.canonical_bar import CanonicalBar

logger = LoggerFactory.get_logger()


class WarmupDetector:
    """
    Detects the maximum lookback period from registered indicators.

    This ensures sufficient historical data is processed before trading starts
    so that indicators always return valid values (never None).
    """

    @staticmethod
    def detect_max_lookback(ctx: "Context") -> int:
        """
        Detect maximum lookback period from all registered indicators.

        Scans all indicators in the IndicatorManager and returns the largest
        lookback period required. This becomes the warmup period.

        Args:
            ctx: Context containing indicator manager

        Returns:
            Maximum lookback period in bars (0 if no indicators)

        Example:
            If strategy uses SMA(20), SMA(50), and RSI(14), returns 50.
        """
        max_lookback = 0

        # Check standard indicators
        for key, indicator in ctx.ind._indicators.items():
            lookback = WarmupDetector._get_indicator_lookback(indicator)
            if lookback > max_lookback:
                max_lookback = lookback
                logger.debug(
                    "warmup.lookback_detected",
                    indicator=key,
                    lookback=lookback,
                )

        # Check custom indicators
        for name, indicator in ctx.ind._custom_indicators.items():
            lookback = WarmupDetector._get_indicator_lookback(indicator)
            if lookback > max_lookback:
                max_lookback = lookback
                logger.debug(
                    "warmup.lookback_detected",
                    indicator=f"custom_{name}",
                    lookback=lookback,
                )

        logger.info(
            "warmup.max_lookback_detected",
            max_lookback=max_lookback,
            indicator_count=len(ctx.ind._indicators) + len(ctx.ind._custom_indicators),
        )

        return max_lookback

    @staticmethod
    def _get_indicator_lookback(indicator) -> int:
        """
        Extract lookback period from an indicator instance.

        Args:
            indicator: Indicator instance

        Returns:
            Lookback period in bars
        """
        # Try common attribute names
        if hasattr(indicator, "period"):
            return int(indicator.period)

        # For MACD, use slow period + signal period
        if hasattr(indicator, "slow") and hasattr(indicator, "signal_period"):
            return int(indicator.slow) + int(indicator.signal_period)

        # For indicators with multiple periods, use the maximum
        if hasattr(indicator, "periods") and isinstance(indicator.periods, (list, tuple)):
            return int(max(indicator.periods))

        # Default to 0 if we can't determine
        return 0


class WarmupProcessor:
    """
    Processes warmup bars to initialize indicators.

    During warmup:
    - Bars are processed to build indicator state
    - Strategy on_bar() is NOT called
    - After warmup completes, all indicators return valid values
    """

    def __init__(self, warmup_bars: int, enable_warmup: bool = True):
        """
        Initialize warmup processor.

        Args:
            warmup_bars: Number of bars to use for warmup
            enable_warmup: Whether warmup is enabled
        """
        self.warmup_bars = warmup_bars
        self.enable_warmup = enable_warmup
        self.warmup_complete = False
        self.bars_processed = 0

        if self.enable_warmup:
            logger.info(
                "warmup.initialized",
                warmup_bars=warmup_bars,
                enabled=True,
            )
        else:
            logger.info(
                "warmup.disabled",
                enabled=False,
            )

    def should_skip_bar(self, bar_index: int) -> bool:
        """
        Check if current bar should be skipped during warmup.

        Args:
            bar_index: Index of current bar (0-based)

        Returns:
            True if bar should be skipped (still in warmup), False otherwise
        """
        if not self.enable_warmup:
            return False

        if self.warmup_complete:
            return False

        return bar_index < self.warmup_bars

    def process_warmup_bar(self, ctx: "Context", symbol: str, bar: "CanonicalBar", symbols: List[str]) -> None:
        """
        Process a single warmup bar.

        Adds bar to history and computes all registered indicators
        WITHOUT calling strategy on_bar().

        Args:
            ctx: Context to update
            symbol: Symbol for the bar
            bar: CanonicalBar to process
            symbols: List of symbols to process indicators for
        """
        if not self.enable_warmup or self.warmup_complete:
            return

        # Add bar to history
        ctx._add_bar_to_history(symbol, bar)

        # Compute all registered indicators for all symbols
        # This builds up their internal state
        for symbol in symbols:
            # Process standard indicators
            for key, indicator in ctx.ind._indicators.items():
                try:
                    _ = indicator.compute(symbol, ctx)
                except Exception as e:
                    logger.warning(
                        "warmup.indicator_compute_error",
                        indicator=key,
                        symbol=symbol,
                        error=str(e),
                    )

            # Process custom indicators
            for name, indicator in ctx.ind._custom_indicators.items():
                try:
                    _ = indicator.compute(symbol, ctx)
                except Exception as e:
                    logger.warning(
                        "warmup.custom_indicator_compute_error",
                        indicator=name,
                        symbol=symbol,
                        error=str(e),
                    )

        # Save indicator state for next bar
        ctx._save_indicator_state()

        self.bars_processed += 1

        if self.bars_processed % 10 == 0:
            logger.debug(
                "warmup.progress",
                bars_processed=self.bars_processed,
                warmup_bars=self.warmup_bars,
                progress_pct=round(self.bars_processed / self.warmup_bars * 100, 1),
            )

    def complete_warmup(self) -> None:
        """Mark warmup as complete."""
        if self.enable_warmup and not self.warmup_complete:
            self.warmup_complete = True
            logger.info(
                "warmup.complete",
                bars_processed=self.bars_processed,
                warmup_bars=self.warmup_bars,
            )

    def get_metadata(self) -> dict:
        """
        Get warmup metadata for run.json.

        Returns:
            Dictionary with warmup metadata
        """
        return {
            "enabled": self.enable_warmup,
            "warmup_bars": self.warmup_bars,
            "bars_processed": self.bars_processed,
            "complete": self.warmup_complete,
        }
