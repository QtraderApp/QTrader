"""Backtest runner and config loader."""

from datetime import datetime
from decimal import Decimal
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
        self.split_processor: Optional[Any] = None  # SplitProcessor (imported conditionally)
        self.dividend_metadata: Dict[str, Any] = {}
        self.execution_engine: Optional[ExecutionEngine] = None

        # Track all fills for reporting
        self.all_fills: List["Fill"] = []

        # Track portfolio snapshots (for debugging/output)
        self.portfolio_snapshots: List[Dict[str, Any]] = []

        # Track previous bar values for daily changes
        self._prev_cash: Optional[Decimal] = None
        self._prev_portfolio_value: Optional[Decimal] = None

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

        # Store adjustment events for split processing
        self.adjustment_events = adjustment_events

        # Initialize dividend and split processors if adjustment events provided
        if adjustment_events:
            # Dividend processor for cash dividends
            self.dividend_processor = DividendProcessor(ctx.portfolio, adjustment_events)
            logger.info(
                "backtest.dividend_processor_initialized",
                **self.dividend_processor.get_stats(),
            )

            # Split processor for stock splits/reverse splits
            from qtrader.execution.split_processor import SplitProcessor

            self.split_processor = SplitProcessor(ctx.portfolio.positions)
            logger.info("backtest.split_processor_initialized")

            # Index ALL events by date (not just dividends)
            self.events_by_date: Dict[datetime, List["AdjustmentEvent"]] = {}
            for symbol, events in adjustment_events.items():
                for event in events:
                    if event.ts not in self.events_by_date:
                        self.events_by_date[event.ts] = []
                    self.events_by_date[event.ts].append(event)

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

        # Track bars for indexing (needed for initial snapshot and next_bar lookhead)
        bars_list = list(bars)  # Ensure we can index

        # Capture initial portfolio snapshot before trading begins
        initial_cash = ctx.portfolio.cash.get_balance()
        initial_snapshot = {
            "timestamp": bars_list[start_idx].ts.isoformat() if start_idx < len(bars_list) else None,
            "symbol": bars_list[start_idx].symbol if start_idx < len(bars_list) else None,
            # Cash tracking
            "initial_cash": float(initial_cash),
            "cash_debits": 0.0,
            "cash_credits": 0.0,
            "end_cash": float(initial_cash),
            # Portfolio tracking
            "initial_portfolio_value": 0.0,
            "daily_mtm": 0.0,
            "end_portfolio_value": 0.0,
            # Account summary
            "total_value": float(initial_cash),
            "num_positions": 0,
        }
        self.portfolio_snapshots.append(initial_snapshot)

        # Initialize tracking variables with initial values
        self._prev_cash = initial_cash
        self._prev_portfolio_value = Decimal("0")

        # Phase 4: Main trading loop
        logger.info("backtest.trading_loop_starting", start_idx=start_idx, total_bars=len(bars))

        # Track processed dividend dates to avoid duplicate processing
        processed_dividend_dates = set()

        # Track previous bar for dividend calculation (need close price before ex-date)
        prev_bar = None

        for bar_idx in range(start_idx, len(bars_list)):
            bar = bars_list[bar_idx]
            next_bar = bars_list[bar_idx + 1] if bar_idx + 1 < len(bars_list) else None

            # Update context state for this bar
            ctx.current_date = bar.ts
            ctx.current_symbol = bar.symbol
            ctx.current_price = bar.close

            # Add bar to context history
            ctx._add_bar_to_history(bar)

            # Process corporate actions (dividends and splits) on ex-date
            if hasattr(self, "events_by_date") and bar.ts not in processed_dividend_dates:
                # Get ALL adjustment events for this date
                events = self.events_by_date.get(bar.ts, [])

                for event in events:
                    if event.event_type == "CashDiv":
                        # Process cash dividend
                        # Pass previous bar's close price for dividend calculation
                        close_prices: Dict[str, Decimal] = {}
                        if prev_bar and prev_bar.symbol == bar.symbol:
                            close_prices[bar.symbol] = prev_bar.close

                        if self.dividend_processor is not None:
                            dividend_results = self.dividend_processor.process_ex_date(bar.ts, close_prices)
                            if dividend_results:
                                logger.debug(
                                    "backtest.dividends_processed",
                                    date=bar.ts.isoformat(),
                                    symbol=bar.symbol,
                                    count=len(dividend_results),
                                    processed=sum(1 for r in dividend_results if r["processed"]),
                                )
                    elif event.event_type == "Subdiv" and hasattr(self, "split_processor"):
                        # Process stock split/reverse split
                        # metadata should have 'AdjustmentFactor' from config
                        logger.info(
                            "backtest.split_event_detected",
                            date=bar.ts.isoformat(),
                            symbol=bar.symbol,
                            event_type=event.event_type,
                            px_factor=float(event.px_factor),
                            metadata=str(event.metadata),
                        )

                        # Get adjustment_factor from metadata if available
                        # metadata_fields=["AdjustmentFactor"] should put it in metadata dict
                        split_adjustment_factor: Decimal
                        if "AdjustmentFactor" in event.metadata:
                            split_adjustment_factor = Decimal(str(event.metadata["AdjustmentFactor"]))
                        elif "adjustment_factor" in event.metadata:
                            split_adjustment_factor = Decimal(str(event.metadata["adjustment_factor"]))
                        else:
                            # Fallback: px_factor might be the adjustment factor
                            logger.warning(
                                "backtest.split_no_adjustment_factor",
                                date=bar.ts.isoformat(),
                                metadata=str(event.metadata),
                                msg="Using px_factor as adjustment_factor",
                            )
                            split_adjustment_factor = event.px_factor

                        if self.split_processor is not None:
                            split_result = self.split_processor.process_split(
                                symbol=bar.symbol,
                                adjustment_factor=split_adjustment_factor,
                                current_price=bar.close,
                            )
                            if split_result["processed"]:
                                logger.info(
                                    "backtest.split_processed",
                                    date=bar.ts.isoformat(),
                                    **{k: v for k, v in split_result.items() if k != "processed"},
                                )

                # Mark date as processed to avoid reprocessing
                if events:
                    processed_dividend_dates.add(bar.ts)

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

            # Snapshot portfolio at end of day with detailed cash flow tracking
            if True:  # Every bar (change to `bar_idx % N == 0` for every N bars)
                # Get current values
                current_cash = ctx.portfolio.cash.get_balance()

                # Calculate portfolio market value (sum of all position values)
                portfolio_market_value = Decimal("0")
                all_positions = ctx.portfolio.positions.get_all_positions()
                for symbol, position in all_positions.items():
                    if not position.is_flat():
                        current_price = ctx.portfolio._current_prices.get(symbol, Decimal("0"))
                        portfolio_market_value += position.market_value(current_price)

                # Get adjustment event for this bar (if any)
                adjustment_event: Optional["AdjustmentEvent"] = None
                adjustment_factor: Optional[float] = None
                adjustment_reason: Optional[str] = None
                cumulative_price_factor: Optional[float] = None
                cumulative_volume_factor: Optional[float] = None
                if hasattr(self, "events_by_date") and bar.ts in self.events_by_date:
                    events = self.events_by_date[bar.ts]
                    if events:
                        # Take first event for this date
                        adjustment_event = events[0]
                        adjustment_reason = adjustment_event.event_type
                        cumulative_price_factor = float(adjustment_event.px_factor)
                        cumulative_volume_factor = float(adjustment_event.vol_factor)
                        # Get individual adjustment factor from metadata
                        if "AdjustmentFactor" in adjustment_event.metadata:
                            adjustment_factor = float(adjustment_event.metadata["AdjustmentFactor"])

                # Get fills for this bar
                bar_fills = [f for f in fills if f.symbol == bar.symbol]
                fill_info: Dict[str, Any] = {}
                if bar_fills:
                    # Aggregate fills for this bar
                    fill = bar_fills[0]  # Take first fill
                    fill_info = {
                        "signal": getattr(fill, "signal_id", None),
                        "order_id": fill.order_id,
                        "order_type": "",  # Fill doesn't store order_type, would need order tracking
                        "fill_qty": fill.qty,
                        "fill_price": float(fill.price),
                        "commission": float(fill.fees),
                    }

                # Get position for this symbol
                position_opt = all_positions.get(bar.symbol)
                position_qty = position_opt.qty if position_opt and not position_opt.is_flat() else 0
                position_avg_cost = (
                    float(position_opt.avg_price) if position_opt and not position_opt.is_flat() else 0.0
                )

                # Total account value
                total_value = current_cash + portfolio_market_value

                # Calculate daily changes
                cash_change = float(current_cash - self._prev_cash) if self._prev_cash is not None else 0.0
                portfolio_change = (
                    float(portfolio_market_value - self._prev_portfolio_value)
                    if self._prev_portfolio_value is not None
                    else 0.0
                )

                # Separate cash debits (negative) and credits (positive)
                cash_debits = min(0.0, cash_change)  # Negative values only
                cash_credits = max(0.0, cash_change)  # Positive values only

                snapshot = {
                    # Bar OHLC data
                    "timestamp": bar.ts.isoformat(),
                    "symbol": bar.symbol,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": bar.volume,
                    # Adjustment factors
                    "cumulative_price_factor": cumulative_price_factor,
                    "cumulative_volume_factor": cumulative_volume_factor,
                    "adjustment_factor": adjustment_factor,
                    "adjustment_reason": adjustment_reason,
                    # Fill information
                    "signal": fill_info.get("signal"),
                    "order_id": fill_info.get("order_id"),
                    "order_type": fill_info.get("order_type"),
                    "fill_qty": fill_info.get("fill_qty", 0),
                    "fill_price": fill_info.get("fill_price", 0.0),
                    "commission": fill_info.get("commission", 0.0),
                    # Cash flow tracking
                    "initial_cash": float(self._prev_cash) if self._prev_cash is not None else float(current_cash),
                    "cash_debits": cash_debits,
                    "cash_credits": cash_credits,
                    "end_cash": float(current_cash),
                    # Portfolio tracking
                    "initial_portfolio_value": float(self._prev_portfolio_value)
                    if self._prev_portfolio_value is not None
                    else 0.0,
                    "daily_mtm": portfolio_change,  # Mark-to-market P&L
                    "end_portfolio_value": float(portfolio_market_value),
                    # Position details
                    "position_qty": position_qty,
                    "position_avg_cost": position_avg_cost,
                    # Account summary
                    "total_value": float(total_value),
                    "num_positions": len([p for p in all_positions.values() if not p.is_flat()]),
                }
                self.portfolio_snapshots.append(snapshot)

                # Save for next bar
                self._prev_cash = current_cash
                self._prev_portfolio_value = portfolio_market_value

            # Save current bar as prev_bar for dividend calculation
            prev_bar = bar

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
                "filled_orders": len(self.execution_engine.filled_orders),
            }

        return metadata


def run_backtest(config_path: Path, strategy_class, out_dir: Path):
    """Convenience function to run backtest."""
    raise NotImplementedError("Stage 1: Stub only")
