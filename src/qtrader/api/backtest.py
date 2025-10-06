"""Backtest runner and config loader."""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.execution.dividend_processor import DividendProcessor
from qtrader.execution.engine import ExecutionEngine
from qtrader.execution.warmup import WarmupDetector, WarmupProcessor

if TYPE_CHECKING:
    from qtrader.api.context import Context
    from qtrader.api.strategy import Strategy
    from qtrader.execution.config import ExecutionConfig
    from qtrader.models.bar import AdjustmentEvent, Bar
    from qtrader.models.order import Fill

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
        self.dividend_processor: Optional[DividendProcessor] = None
        self.dividend_metadata: Dict[str, Any] = {}
        self.execution_engine: Optional[ExecutionEngine] = None

        # Track all fills for reporting
        self.all_fills: List["Fill"] = []

        # Track portfolio snapshots (for debugging/output)
        self.portfolio_snapshots: List[Dict[str, Any]] = []

    def run(
        self,
        ctx: "Context",
        bars: List["Bar"],
        symbols: List[str],
        out_dir: Path,
        adjustment_events: Optional[Dict[str, List["AdjustmentEvent"]]] = None,
    ):
        """
        Run backtest with complete execution loop.

        Implements canonical event loop from spec:
        1. strategy.on_init() - Register indicators
        2. Warmup phase (if enabled) - Build indicator state
        3. strategy.on_start() - Post-warmup setup
        4. Main trading loop:
           - Process dividends (if ex-date)
           - Call strategy.on_bar() -> get signals
           - Evaluate signals through risk manager
           - Convert approved signals to orders
           - Submit orders to execution engine
           - Process bar (intrabar fills, end-of-bar)
           - Apply fills to portfolio
           - EOD accruals (borrow costs)
           - Snapshot portfolio state
        5. strategy.on_end() - Cleanup

        Args:
            ctx: Context instance with risk_manager and portfolio
            bars: All bars to process (sorted by timestamp)
            symbols: List of symbols in backtest
            out_dir: Output directory for results
            adjustment_events: Optional dict mapping symbol -> list of adjustment events
                              (for dividend processing)

        Returns:
            Dict with run metadata including warmup, dividend, and execution info
        """
        logger.info(
            "backtest.starting",
            symbols=symbols,
            total_bars=len(bars),
            warmup_enabled=self.config.warmup,
            warmup_bars=self.config.warmup_bars,
            has_adjustment_events=adjustment_events is not None,
        )

        # Validate context has required components
        if ctx.portfolio is None:
            raise RuntimeError("Context must have portfolio configured")

        # Risk manager is optional (backward compatibility with tests)
        # If strategy returns signals, risk manager is required (checked later)

        # Initialize execution engine
        self.execution_engine = ExecutionEngine(
            portfolio=ctx.portfolio,
            config=self.config,
        )

        # Phase 1: Initialize strategy
        if hasattr(self.strategy, "on_init"):
            logger.info("backtest.calling_on_init")
            self.strategy.on_init(ctx)

        # Initialize dividend processor if adjustment events provided
        if adjustment_events:
            self.dividend_processor = DividendProcessor(ctx.portfolio, adjustment_events)
            logger.info(
                "backtest.dividend_processor_initialized",
                **self.dividend_processor.get_stats(),
            )

        # Phase 2: Warmup (if enabled)
        warmup_processor = None
        start_idx = 0

        if self.config.warmup:
            warmup_bars = self.config.warmup_bars or WarmupDetector.detect_max_lookback(ctx)

            if warmup_bars > 0:
                logger.info(
                    "backtest.warmup_starting",
                    warmup_bars=warmup_bars,
                    auto_detected=self.config.warmup_bars is None,
                )

                warmup_processor = WarmupProcessor(warmup_bars=warmup_bars, enable_warmup=True)

                # Process warmup bars (indicators only, no trading)
                for bar_idx, bar in enumerate(bars):
                    if not warmup_processor.should_skip_bar(bar_idx):
                        break

                    # Add bar to context history for indicators
                    ctx._add_bar_to_history(bar)

                    # Process bar for indicators only (don't call on_bar)
                    warmup_processor.process_warmup_bar(ctx, bar, symbols)

                # Mark warmup complete
                warmup_processor.complete_warmup()
                self.warmup_metadata = warmup_processor.get_metadata()
                start_idx = warmup_bars

                logger.info("backtest.warmup_complete", **self.warmup_metadata)
            else:
                logger.info("backtest.warmup_skipped", reason="No indicators requiring warmup")
                self.warmup_metadata = {"enabled": True, "warmup_bars": 0, "bars_processed": 0, "complete": True}

        # Phase 3: Call on_start after warmup
        if hasattr(self.strategy, "on_start"):
            logger.info("backtest.calling_on_start")
            self.strategy.on_start(ctx)

        # Phase 4: Main trading loop
        logger.info("backtest.trading_loop_starting", start_idx=start_idx, total_bars=len(bars))

        # Track processed dividend dates to avoid duplicate processing
        processed_dividend_dates = set()

        # Track bars processed for next_bar lookhead
        bars_list = list(bars)  # Ensure we can index

        for bar_idx in range(start_idx, len(bars_list)):
            bar = bars_list[bar_idx]
            next_bar = bars_list[bar_idx + 1] if bar_idx + 1 < len(bars_list) else None

            # Update context state for this bar
            ctx.current_date = bar.ts
            ctx.current_symbol = bar.symbol
            ctx.current_price = bar.close

            # Add bar to context history
            ctx._add_bar_to_history(bar)

            # Process dividends on ex-date (if dividend processor initialized)
            if self.dividend_processor and bar.ts not in processed_dividend_dates:
                dividend_results = self.dividend_processor.process_ex_date(bar.ts)
                if dividend_results:
                    processed_dividend_dates.add(bar.ts)
                    logger.debug(
                        "backtest.dividends_processed",
                        date=bar.ts.isoformat(),
                        symbol=bar.symbol,
                        count=len(dividend_results),
                        processed=sum(1 for r in dividend_results if r["processed"]),
                    )

            # Call strategy.on_bar() - returns signals
            signals = self.strategy.on_bar(bar, ctx)

            # Process signals through risk manager
            if signals:
                if ctx.risk_manager is None:
                    logger.warning(
                        "backtest.signals_ignored",
                        count=len(signals),
                        reason="No risk manager configured - signals cannot be converted to orders",
                    )
                    continue  # Skip signal processing

                for signal in signals:
                    try:
                        # Evaluate signal
                        decision = ctx.evaluate_signal(signal)

                        if decision.approved:
                            # Convert to order
                            order = ctx.signal_to_order(signal, decision)

                            # Submit to execution engine
                            self.execution_engine.submit_order(order, bar.ts)

                            logger.debug(
                                "backtest.order_submitted",
                                order_id=order.order_id,
                                symbol=order.symbol,
                                side=order.side.value,
                                qty=order.qty,
                                order_type=order.order_type.value,
                            )
                        else:
                            logger.debug(
                                "backtest.signal_rejected",
                                signal_id=signal.signal_id,
                                symbol=signal.symbol,
                                reason=decision.reason,
                            )
                    except Exception as e:
                        logger.error(
                            "backtest.signal_processing_error",
                            signal_id=signal.signal_id,
                            error=str(e),
                        )

            # Process bar through execution engine
            # This handles: intrabar evaluation (limit/stop), participation, partials, MOC fills
            fills = self.execution_engine.on_bar(bar, next_bar=next_bar)

            # Track fills
            self.all_fills.extend(fills)

            # Call strategy on_fill for each fill
            if fills and hasattr(self.strategy, "on_fill"):
                for fill in fills:
                    self.strategy.on_fill(fill, ctx)

            # Save indicator state for crossover detection
            ctx._save_indicator_state()

            # Snapshot portfolio at end of day (optional - for debugging)
            if bar_idx % 10 == 0 or bar_idx == len(bars_list) - 1:  # Every 10 bars or last bar
                snapshot = {
                    "timestamp": bar.ts.isoformat(),
                    "symbol": bar.symbol,
                    "cash": float(ctx.portfolio.cash.get_balance()),
                    "equity": float(ctx.portfolio.get_equity()),
                    "positions": len(
                        [p for p in ctx.portfolio.positions.get_all_positions().values() if not p.is_flat()]
                    ),
                }
                self.portfolio_snapshots.append(snapshot)

        # Phase 5: End callback
        if hasattr(self.strategy, "on_end"):
            logger.info("backtest.calling_on_end")
            self.strategy.on_end(ctx)

        logger.info(
            "backtest.complete",
            bars_processed=len(bars) - start_idx,
            total_fills=len(self.all_fills),
            final_cash=float(ctx.portfolio.cash.get_balance()),
            final_equity=float(ctx.portfolio.get_equity()),
        )

        # Return run metadata
        metadata: Dict[str, Any] = {
            "total_bars": len(bars),
            "trading_bars": len(bars) - start_idx,
            "total_fills": len(self.all_fills),
            "final_cash": float(ctx.portfolio.cash.get_balance()),
            "final_equity": float(ctx.portfolio.get_equity()),
        }

        if self.warmup_metadata:
            metadata["warmup"] = self.warmup_metadata

        if self.dividend_processor:
            metadata["dividends"] = self.dividend_processor.get_stats()

        if self.execution_engine:
            metadata["execution"] = {
                "pending_orders": len(self.execution_engine.pending_orders),
                "filled_orders": len(
                    [o for o in self.execution_engine.pending_orders.values() if o.state.value == "filled"]
                ),
            }

        return metadata


def run_backtest(config_path: Path, strategy_class, out_dir: Path):
    """Convenience function to run backtest."""
    raise NotImplementedError("Stage 1: Stub only")
