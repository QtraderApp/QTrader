"""Backtest runner and config loader."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.execution.warmup import WarmupDetector, WarmupProcessor

if TYPE_CHECKING:
    from qtrader.api.context import Context
    from qtrader.api.strategy import Strategy
    from qtrader.execution.config import ExecutionConfig
    from qtrader.models.bar import Bar

logger = LoggerFactory.get_logger()


def load_config(path: Path):
    """Load configuration from YAML file."""
    raise NotImplementedError("Stage 1: Stub only")


class Backtest:
    """
    Backtest runner with warmup support.

    Manages complete backtest lifecycle:
    1. strategy.on_init(ctx) - Register indicators
    2. Warmup phase (if enabled) - Build indicator state
    3. strategy.on_start(ctx) - Post-warmup setup
    4. Main loop - Process bars with strategy.on_bar()
    5. strategy.on_end(ctx) - Cleanup

    Stub for Stage 1. Full implementation in later stages.
    """

    def __init__(self, config: "ExecutionConfig", strategy: "Strategy"):
        """
        Initialize backtest runner.

        Args:
            config: Execution configuration including warmup settings
            strategy: Strategy instance to run
        """
        self.config = config
        self.strategy = strategy
        self.warmup_metadata: Optional[Dict[str, Any]] = None

    def run(self, ctx: "Context", bars: List["Bar"], symbols: List[str], out_dir: Path):
        """
        Run backtest with warmup support.

        Args:
            ctx: Context instance
            bars: All bars to process
            symbols: List of symbols in backtest
            out_dir: Output directory for results

        Returns:
            Dict with run metadata including warmup info
        """
        logger.info(
            "backtest.starting",
            symbols=symbols,
            total_bars=len(bars),
            warmup_enabled=self.config.warmup,
            warmup_bars=self.config.warmup_bars,
        )

        # Phase 1: Initialize strategy
        if hasattr(self.strategy, "on_init"):
            logger.info("backtest.calling_on_init")
            self.strategy.on_init(ctx)

        # Phase 2: Warmup (if enabled)
        warmup_processor = None
        if self.config.warmup:
            warmup_bars = self.config.warmup_bars or WarmupDetector.detect_max_lookback(ctx)

            if warmup_bars > 0:
                logger.info(
                    "backtest.warmup_starting",
                    warmup_bars=warmup_bars,
                    auto_detected=self.config.warmup_bars is None,
                )

                warmup_processor = WarmupProcessor(warmup_bars=warmup_bars, enable_warmup=True)

                # Process warmup bars
                for bar_idx, bar in enumerate(bars):
                    if not warmup_processor.should_skip_bar(bar_idx):
                        break

                    # Process bar for indicators only (don't call on_bar)
                    warmup_processor.process_warmup_bar(ctx, bar, symbols)

                # Mark warmup complete
                warmup_processor.complete_warmup()
                self.warmup_metadata = warmup_processor.get_metadata()

                logger.info("backtest.warmup_complete", **self.warmup_metadata)
            else:
                logger.info("backtest.warmup_skipped", reason="No indicators requiring warmup")
                self.warmup_metadata = {"enabled": True, "warmup_bars": 0, "bars_processed": 0, "complete": True}

        # Phase 3: Call on_start after warmup
        if hasattr(self.strategy, "on_start"):
            logger.info("backtest.calling_on_start")
            self.strategy.on_start(ctx)

        # Phase 4: Main trading loop
        start_idx = warmup_processor.warmup_bars if warmup_processor else 0
        logger.info("backtest.trading_loop_starting", start_idx=start_idx, total_bars=len(bars))

        for bar_idx, bar in enumerate(bars):
            if bar_idx < start_idx:
                continue  # Skip warmup bars

            # Call strategy on_bar
            _ = self.strategy.on_bar(bar, ctx)

            # Process signals (implementation in later stages)
            # This would call risk manager, create orders, execute fills, etc.

        # Phase 5: End callback
        if hasattr(self.strategy, "on_end"):
            logger.info("backtest.calling_on_end")
            self.strategy.on_end(ctx)

        logger.info("backtest.complete", bars_processed=len(bars) - start_idx)

        # Return run metadata
        metadata: Dict[str, Any] = {"total_bars": len(bars), "trading_bars": len(bars) - start_idx}

        if self.warmup_metadata:
            metadata["warmup"] = self.warmup_metadata

        return metadata


def run_backtest(config_path: Path, strategy_class, out_dir: Path):
    """Convenience function to run backtest."""
    raise NotImplementedError("Stage 1: Stub only")
