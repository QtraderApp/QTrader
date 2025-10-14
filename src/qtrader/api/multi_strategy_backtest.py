"""Multi-strategy backtest engine.

This module extends the single-strategy Backtest class to support:
1. Multiple strategies with separate universes and capital allocations
2. Per-strategy portfolio tracking and risk management
3. Dynamic capital reallocation based on performance
4. Consolidated reporting across all strategies
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, List

from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.config.backtest_config import BacktestConfig, StrategyAllocation
from qtrader.config.logging_config import LoggerFactory
from qtrader.data import BarMerger, PriceSeriesIterator
from qtrader.execution.config import ExecutionConfig
from qtrader.models.portfolio import Portfolio
from qtrader.models.strategy_metrics import StrategyMetrics
from qtrader.risk.manager import RiskManager
from qtrader.risk.policy import RiskPolicy

logger = LoggerFactory.get_logger()


class StrategyRunner:
    """Manages execution of a single strategy within multi-strategy backtest."""

    def __init__(
        self,
        allocation: StrategyAllocation,
        strategy_instance,
        context: Context,
        backtest: Backtest,
    ):
        """
        Initialize strategy runner.

        Args:
            allocation: Strategy configuration
            strategy_instance: Instantiated strategy
            context: Strategy-specific context
            backtest: Backtest engine for this strategy
        """
        self.allocation = allocation
        self.strategy = strategy_instance
        self.context = context
        self.backtest = backtest
        self.name = allocation.name

        # Track strategy-specific symbols
        self.symbols = [inst["symbol"] for inst in allocation.instruments]

    def process_bar(self, multi_bar, bar_ts: datetime) -> None:
        """
        Process a bar for this strategy if it's in the strategy's universe.

        Args:
            multi_bar: MultiBar object
            bar_ts: Bar timestamp
        """
        symbol = multi_bar.symbol

        # Only process if this symbol is in strategy's universe
        if symbol not in self.symbols:
            return

        # Update context for this bar
        self.context.current_date = bar_ts
        self.context.current_symbol = symbol
        self.context.current_price = Decimal(str(multi_bar.unadjusted.close))

        # Call strategy.on_bar() - implementation depends on strategy phase
        # For now, strategies still receive MultiBar and return signals
        signals = self.strategy.on_bar(multi_bar, self.context)

        # Process signals through this strategy's risk manager
        if signals and self.context.risk_manager:
            for signal in signals:
                try:
                    decision = self.context.evaluate_signal(signal)
                    if decision.approved:
                        order = self.context.signal_to_order(signal, decision)
                        # Submit with strategy name for attribution
                        if self.backtest.execution_engine:
                            self.backtest.execution_engine.submit_order(order, bar_ts)
                            logger.debug(
                                "multi_strategy.order_submitted",
                                strategy=self.name,
                                order_id=order.order_id,
                                symbol=order.symbol,
                                side=order.side.value,
                                qty=order.qty,
                            )
                except Exception as e:
                    logger.error(
                        "multi_strategy.signal_processing_error",
                        strategy=self.name,
                        signal_id=signal.signal_id,
                        error=str(e),
                    )


class MultiStrategyBacktest:
    """
    Multi-strategy backtest engine.

    Manages multiple strategies with:
    - Separate capital allocations
    - Independent universes
    - Per-strategy risk management
    - Dynamic reallocation
    - Consolidated reporting
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize multi-strategy backtest.

        Args:
            config: Complete backtest configuration
        """
        self.config = config
        self.strategy_runners: List[StrategyRunner] = []
        self.portfolio: Portfolio | None = None
        self.start_time: datetime | None = None

    def run(
        self,
        data_iterators: Dict[str, PriceSeriesIterator],
        out_dir: Path,
    ) -> Dict:
        """
        Run multi-strategy backtest.

        Args:
            data_iterators: Dict mapping symbol -> PriceSeriesIterator
            out_dir: Output directory for results

        Returns:
            Dict with run metadata
        """
        logger.info(
            "multi_strategy.starting",
            strategies=len(self.config.strategies),
            initial_capital=float(self.config.initial_capital),
            start_date=self.config.start_date.isoformat(),
            end_date=self.config.end_date.isoformat(),
        )

        self.start_time = datetime.now()

        # Initialize portfolio in multi-strategy mode
        self.portfolio = Portfolio(
            initial_cash=self.config.initial_capital,
            multi_strategy=True,
        )

        # Allocate capital to each strategy
        for strategy_alloc in self.config.strategies:
            strategy_capital = self.config.initial_capital * strategy_alloc.initial_allocation_pct
            self.portfolio.allocate_to_strategy(strategy_alloc.name, strategy_capital)

            logger.info(
                "multi_strategy.capital_allocated",
                strategy=strategy_alloc.name,
                capital=float(strategy_capital),
                allocation_pct=float(strategy_alloc.initial_allocation_pct * 100),
            )

        # Initialize each strategy
        for strategy_alloc in self.config.strategies:
            # Load strategy class
            StrategyClass = strategy_alloc.load_strategy_class()

            # Instantiate strategy with config
            strategy_instance = StrategyClass(**strategy_alloc.strategy_config)

            # Create strategy-specific risk manager
            # Use strategy-specific settings if available, else fall back to global defaults
            if strategy_alloc.risk:
                position_size = strategy_alloc.risk.position_size or self.config.risk.default_position_size
                max_position_pct = strategy_alloc.risk.max_position_pct or self.config.risk.default_max_position_pct
                allow_shorting = (
                    strategy_alloc.risk.allow_shorting
                    if strategy_alloc.risk.allow_shorting is not None
                    else self.config.risk.default_allow_shorting
                )
            else:
                position_size = self.config.risk.default_position_size
                max_position_pct = self.config.risk.default_max_position_pct
                allow_shorting = self.config.risk.default_allow_shorting

            risk_policy = RiskPolicy(
                default_position_size=position_size,
                max_position_pct=max_position_pct,
                allow_shorting=allow_shorting,
            )
            risk_manager = RiskManager(portfolio=self.portfolio, policy=risk_policy)

            # Create strategy-specific context
            ctx = Context(
                risk_manager=risk_manager,
                portfolio=self.portfolio,
            )

            # Create execution config
            exec_config = ExecutionConfig(
                warmup=self.config.execution.warmup,
                warmup_bars=self.config.execution.warmup_bars,
            )

            # Create backtest engine for this strategy
            backtest = Backtest(exec_config, strategy_instance)

            # Create strategy runner
            runner = StrategyRunner(
                allocation=strategy_alloc,
                strategy_instance=strategy_instance,
                context=ctx,
                backtest=backtest,
            )

            self.strategy_runners.append(runner)

            logger.info(
                "multi_strategy.strategy_initialized",
                strategy=strategy_alloc.name,
                strategy_class=StrategyClass.__name__,
                symbols=len(runner.symbols),
            )

        # Get all unique symbols across strategies (for logging)
        _ = self.config.all_symbols  # Mark as intentionally unused

        # Use BarMerger to coordinate multi-symbol streams
        merger = BarMerger(data_iterators)
        bars_list = []

        while merger.has_next():
            _, multi_bar = merger.get_next_bar()
            bars_list.append(multi_bar)

        logger.info(
            "multi_strategy.bars_loaded",
            total_bars=len(bars_list),
            **merger.get_stats(),
        )

        # Initialize all strategies (call on_init)
        for runner in self.strategy_runners:
            if hasattr(runner.strategy, "on_init"):
                logger.info("multi_strategy.calling_on_init", strategy=runner.name)
                runner.strategy.on_init(runner.context)

        # TODO: Handle warmup phase for each strategy
        # For now, skip warmup in multi-strategy mode

        # Call on_start for all strategies
        for runner in self.strategy_runners:
            if hasattr(runner.strategy, "on_start"):
                logger.info("multi_strategy.calling_on_start", strategy=runner.name)
                runner.strategy.on_start(runner.context)

        # Main trading loop
        logger.info("multi_strategy.trading_loop_starting", total_bars=len(bars_list))

        for bar_idx, multi_bar in enumerate(bars_list):
            bar_ts = datetime.fromisoformat(multi_bar.adjusted.trade_datetime)

            # Update portfolio prices for all symbols
            self.portfolio.update_prices({multi_bar.symbol: Decimal(str(multi_bar.unadjusted.close))})

            # Process dividends and splits (portfolio-level)
            self._process_corporate_actions(multi_bar, bar_ts)

            # Process bar through each strategy
            for runner in self.strategy_runners:
                runner.process_bar(multi_bar, bar_ts)

            # Process execution for all strategies
            # Each strategy's execution engine handles its own orders
            for runner in self.strategy_runners:
                if runner.backtest.execution_engine:
                    # Get next bar for lookahead
                    next_bar = bars_list[bar_idx + 1].unadjusted if bar_idx + 1 < len(bars_list) else None

                    # Process this bar through execution engine
                    fills = runner.backtest.execution_engine.on_bar(
                        multi_bar.unadjusted,
                        symbol=multi_bar.symbol,
                        ts=bar_ts,
                        next_bar=next_bar,
                    )

                    # Apply fills to portfolio with strategy attribution
                    for fill in fills:
                        self.portfolio.apply_fill(
                            symbol=fill.symbol,
                            side=fill.side,
                            qty=fill.qty,
                            fill_price=fill.price,
                            commission=fill.fees,
                            ts=fill.execution_ts,
                            order_id=fill.order_id,
                            fill_id=fill.fill_id,
                            strategy_name=runner.name,
                        )

                        # Notify strategy
                        if hasattr(runner.strategy, "on_fill"):
                            runner.strategy.on_fill(fill, runner.context)

            # Check for reallocation (simplified - would check frequency)
            # TODO: Implement reallocation logic based on policy

        # Call on_end for all strategies
        for runner in self.strategy_runners:
            if hasattr(runner.strategy, "on_end"):
                logger.info("multi_strategy.calling_on_end", strategy=runner.name)
                runner.strategy.on_end(runner.context)

        # Generate final report
        duration = (datetime.now() - self.start_time).total_seconds()

        # Collect per-strategy metrics
        strategy_metrics = self.portfolio.get_all_strategy_metrics()

        logger.info(
            "multi_strategy.complete",
            duration_sec=duration,
            total_bars=len(bars_list),
            strategies=len(self.strategy_runners),
        )

        # Build metadata
        metadata = {
            "duration_sec": duration,
            "total_bars": len(bars_list),
            "strategies": len(self.strategy_runners),
            "initial_capital": float(self.config.initial_capital),
            "final_equity": float(self.portfolio.get_equity()),
            "strategy_metrics": {name: self._metrics_to_dict(metrics) for name, metrics in strategy_metrics.items()},
        }

        return metadata

    def _process_corporate_actions(self, multi_bar, bar_ts: datetime) -> None:
        """Process dividends and splits (stub for now)."""
        # TODO: Implement dividend and split processing
        pass

    def _metrics_to_dict(self, metrics: StrategyMetrics) -> Dict:
        """Convert StrategyMetrics to dict for reporting."""
        return {
            "capital": float(metrics.current_capital),
            "equity": float(metrics.equity),
            "total_return": float(metrics.total_return),
            "sharpe_ratio": float(metrics.sharpe_ratio) if metrics.sharpe_ratio else None,
            "max_drawdown": float(metrics.max_drawdown),
            "win_rate": float(metrics.win_rate),
            "total_trades": metrics.total_trades,
        }
