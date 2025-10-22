"""
Backtest Engine Implementation.

Pure event-driven orchestrator that coordinates all services via EventBus.
No business logic, no direct service calls, just event publishing.
"""

import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta

import structlog

from qtrader.backtest.config import BacktestConfig
from qtrader.events.event_bus import EventBus
from qtrader.events.events import Event, RiskEvaluationTriggerEvent, ValuationTriggerEvent
from qtrader.services.data.service import DataService
from qtrader.services.execution.service import ExecutionService
from qtrader.services.portfolio.service import PortfolioService
from qtrader.services.risk.service import RiskService
from qtrader.services.strategy.service import StrategyService
from qtrader.system.config import get_system_config

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: date
    end_date: date
    initial_capital: float
    final_capital: float
    total_return: float
    num_trades: int
    duration: timedelta


class BacktestEngine:
    """
    Pure event-driven backtesting orchestrator.

    Responsibilities:
    - Load configuration and instantiate services
    - Iterate through historical data
    - Publish BarEvent for each symbol/bar
    - Trigger portfolio valuation (ValuationTriggerEvent)
    - Trigger risk evaluation (RiskEvaluationTriggerEvent)
    - Return BacktestResult

    Phase Ordering for Quantitative Correctness:
    =============================================
    For each timestamp, strict sequential phases ensure causal consistency:

    Phase 1: MarketData
        - DataService publishes PriceBarEvent for ALL symbols at timestamp T
        - Services update their internal state (indicators, caches)
        - NO signals generated yet (prevents race conditions)

    Phase 2: Signals (implicit)
        - StrategyService processes all bars
        - Generates trading signals based on updated state
        - Signals buffered in RiskService (not executed immediately)

    Phase 3: Valuation (barrier)
        - Engine publishes ValuationTriggerEvent
        - PortfolioService calculates equity, positions, P&L
        - Creates CONSISTENT SNAPSHOT of portfolio state
        - Critical: All bars processed BEFORE valuation starts

    Phase 4: RiskEvaluation (barrier)
        - Engine publishes RiskEvaluationTriggerEvent
        - RiskService processes ALL buffered signals in single batch
        - Cross-strategy netting: A buys 100 + B sells 100 = net 0
        - Position limits checked against consistent portfolio snapshot
        - Approved orders published as OrderEvent

    Phase 5: Execution (next cycle)
        - ExecutionService fills orders at next bar's prices
        - Fill → Portfolio → updated equity → Risk sees new state
        - Strict causal ordering: no stale state

    Quant Gotchas Solved:
    =====================
    1. Race Conditions:
       - Without barriers, Strategy A and B signals arrive in arbitrary order
       - Risk calculations use inconsistent snapshots
       - Solution: ValuationTriggerEvent = clock tick, all strategies sync

    2. Wash Trades:
       - Without coordinator, A buys 100 + B sells 100 = 2 fills (double commissions)
       - Solution: RiskService nets across strategies BEFORE execution

    3. Stale Portfolio State:
       - If Risk reads portfolio while Fill still processing → wrong leverage
       - Solution: Strict phases: Fill → Portfolio update → Risk evaluation

    4. Indicator Lookahead:
       - If Strategy A generates signal before all bars arrive → uses partial data
       - Solution: All PriceBarEvents published BEFORE strategies evaluate

    Production Considerations:
    ==========================
    Backtest Mode (current):
        - Synchronous, deterministic, single-threaded
        - Barriers are function calls (no async needed)
        - Perfect for research: reproducible, debuggable

    Production Mode (future):
        - Async event bus with bounded queues
        - Backpressure: drop old bars during market bursts (open/close)
        - State versioning: tag snapshots with clock tick
        - Monitor queue depths: alert if Risk buffer > 1000 signals

    Event Publishing Pattern:
    =========================
    For each timestamp:
        1. Publish PriceBarEvent (per symbol) - services update state
        2. Publish ValuationTriggerEvent - portfolio calculates metrics
        3. Publish RiskEvaluationTriggerEvent - risk processes signals in batch

    Services own their business logic. Engine has ZERO business logic.

    Target: <200 LOC, no business rules, pure orchestration.
    """

    def __init__(
        self,
        config: BacktestConfig,
        event_bus: EventBus,
        data_service: DataService,
        portfolio_service: PortfolioService,
        execution_service: ExecutionService,
        risk_service: RiskService,
        strategy_service: StrategyService,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            event_bus: Event bus for publishing events
            data_service: Data service for loading historical bars
            portfolio_service: Portfolio service for position tracking
            execution_service: Execution service for order fills
            risk_service: Risk service for signal evaluation
            strategy_service: Strategy service orchestrating strategies
        """
        self.config = config
        self._event_bus = event_bus
        self._data_service = data_service
        self._portfolio_service = portfolio_service
        self._execution_service = execution_service
        self._risk_service = risk_service
        self._strategy_service = strategy_service

        logger.info(
            "backtest.engine.initialized",
            start_date=config.start_date,
            end_date=config.end_date,
            warmup_bars=config.warmup_bars,
            universe_size=len(config.universe),
        )

    @classmethod
    def from_config(cls, config: BacktestConfig) -> "BacktestEngine":
        """
        Factory method to create engine from configuration.

        Loads SystemConfig for service configurations, uses BacktestConfig
        for run parameters (dates, universe, strategies).

        Args:
            config: Backtest configuration loaded from YAML

        Returns:
            Configured BacktestEngine instance

        Raises:
            ValueError: If configuration is invalid or services fail to initialize
        """
        # Load system configuration
        system_config = get_system_config()

        # Create event bus
        event_bus = EventBus()

        # Create data service using factory method (with EventBus for event-driven mode)
        data_service = DataService.from_config(
            config_dict=asdict(system_config.data),
            dataset=config.data.dataset,
            event_bus=event_bus,
        )

        # Create portfolio service from system config
        portfolio_service = PortfolioService.from_config(
            config_dict=asdict(system_config.portfolio), event_bus=event_bus
        )

        # Create execution service from system config
        execution_service = ExecutionService.from_config(
            config_dict=asdict(system_config.execution), event_bus=event_bus
        )

        # Create risk service from system config
        risk_service = RiskService.from_config(config_dict=asdict(system_config.risk), event_bus=event_bus)

        # Create strategy service
        strategy_service = StrategyService.from_config(
            strategies_config=[s.model_dump() for s in config.strategies],
            event_bus=event_bus,
        )

        return cls(
            config=config,
            event_bus=event_bus,
            data_service=data_service,
            portfolio_service=portfolio_service,
            execution_service=execution_service,
            risk_service=risk_service,
            strategy_service=strategy_service,
        )

    def run(self) -> BacktestResult:
        """
        Run the backtest with strict phase ordering for quantitative correctness.

        Phase-Based Event Flow:
        =======================
        1. Warmup Phase (if warmup_bars > 0):
           - Stream warmup_bars worth of data with is_warmup=True
           - Strategies build indicators but don't generate signals
           - No orders executed during warmup

        2. Main Phase (for each timestamp T):
           The engine enforces strict sequential phases to prevent race conditions
           and ensure causally consistent portfolio state:

           Phase 1: MarketData
           -------------------
           - DataService publishes PriceBarEvent for ALL symbols at T
           - StrategyService updates indicators for ALL strategies
           - RiskService buffers any signals generated
           - NO orders executed yet (prevents using partial data)

           Phase 2: Valuation (barrier)
           ----------------------------
           - Engine publishes ValuationTriggerEvent(ts=T)
           - This is a CLOCK TICK BARRIER: all data for T arrived
           - PortfolioService calculates equity, positions, P&L
           - Creates CONSISTENT SNAPSHOT for risk decisions
           - Critical: happens AFTER all bars processed

           Phase 3: RiskEvaluation (barrier)
           ---------------------------------
           - Engine publishes RiskEvaluationTriggerEvent(ts=T)
           - RiskService processes ALL buffered signals in single batch
           - Cross-strategy netting applied:
             * Strategy A: BUY 100 AAPL
             * Strategy B: SELL 100 AAPL
             * Net order: 0 AAPL (no wash trade, no commissions)
           - Position limit checks use consistent portfolio snapshot
           - Approved orders published as OrderEvent

           Phase 4: Execution (next cycle at T+1)
           --------------------------------------
           - ExecutionService fills orders at T+1 bar prices
           - Fills → Portfolio updates → Risk sees new state
           - Strict causal ordering: Fill → Portfolio → Risk

        3. Results Collection:
           - Query final portfolio state from PortfolioService
           - Count executed trades from ExecutionService
           - Calculate performance metrics (return, Sharpe, etc.)

        Quantitative Correctness Guarantees:
        ====================================
        1. Consistent Snapshots:
           - Valuation happens AFTER all bars for timestamp
           - Risk decisions use single consistent portfolio state
           - No race conditions between Strategy A and B

        2. No Wash Trades:
           - RiskService nets signals across ALL strategies
           - A buys 100 + B sells 100 = net 0 (zero commissions)
           - Without netting: 2 fills, double commissions, tax complexity

        3. Causal Ordering:
           - Fill event → Portfolio update → Equity recalc → Risk sees new state
           - Never: Risk uses stale equity while Fill processing
           - Deterministic: same inputs → same outputs (for research)

        4. No Lookahead:
           - Strategies only see data up to T when generating signals for T
           - Orders executed at T+1 prices (realistic slippage model)
           - Warmup phase builds indicators without affecting backtest

        Implementation Notes:
        =====================
        Synchronous Execution (Backtest):
            - Single-threaded, deterministic
            - Barriers are simple function calls
            - Perfect for research: reproducible, debuggable

        Async Ready (Production):
            - Same phase pattern works with async event bus
            - Barriers become async coordination points
            - Bounded queues prevent memory explosion
            - Backpressure policy: drop old bars during market bursts

        Performance:
            - O(N*M) where N=days, M=universe_size
            - Typical: 1000 days * 500 symbols = 500K bars
            - Bottleneck: indicator calculations (pandas/numpy)
            - Target: >100 bars/sec with 50+ indicators

        Returns:
            BacktestResult with performance metrics and trade log

        Raises:
            RuntimeError: If backtest execution fails (data load, service error, etc.)

        Example:
            >>> engine = BacktestEngine.from_config(config)
            >>> result = engine.run()
            >>> print(f"Return: {result.total_return:.2%}")
            >>> print(f"Trades: {result.num_trades}")
        """
        start_time = datetime.now()
        logger.info(
            "backtest.starting",
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            universe_size=len(self.config.universe),
            warmup_bars=self.config.warmup_bars,
        )

        try:
            # Step 1: Warmup Phase
            if self.config.warmup_bars > 0:
                logger.info(
                    "backtest.warmup_phase.starting",
                    warmup_bars=self.config.warmup_bars,
                )

                # Calculate warmup start date (warmup_bars days before start_date)
                warmup_start = self.config.start_date - timedelta(days=self.config.warmup_bars * 2)  # *2 for weekends
                warmup_end = self.config.start_date - timedelta(days=1)

                # Stream warmup data (DataService publishes PriceBarEvent with is_warmup=True)
                self._data_service.stream_universe(
                    symbols=self.config.universe,
                    start_date=warmup_start,
                    end_date=warmup_end,
                    is_warmup=True,
                )

                logger.info("backtest.warmup_phase.complete")

            # Step 2: Main Event Loop
            logger.info(
                "backtest.main_phase.starting",
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            # Load data and get synchronized iterator
            # DataService.stream_universe() will:
            # 1. Load all data for universe
            # 2. Synchronize by timestamp
            # 3. Publish PriceBarEvent for each symbol/timestamp
            # We need to track timestamps to publish trigger events

            # For Phase 5a, we'll use a callback approach
            # Store current timestamp to publish triggers after bars
            self._current_timestamp: datetime | None = None
            self._bar_count = 0

            # Subscribe to price_bar events to track timestamps
            def track_timestamp(event: Event) -> None:
                if hasattr(event, "timestamp") and hasattr(event, "is_warmup") and not event.is_warmup:
                    new_ts = event.timestamp
                    # If timestamp changed, publish triggers for previous timestamp
                    if self._current_timestamp is not None and new_ts != self._current_timestamp:
                        # All bars for previous timestamp published, trigger valuation and risk
                        self._event_bus.publish(ValuationTriggerEvent(ts=self._current_timestamp))
                        self._event_bus.publish(RiskEvaluationTriggerEvent(ts=self._current_timestamp))

                        # Apply replay speed delay if configured (for visualization/debugging)
                        if self.config.replay_speed > 0:
                            time.sleep(self.config.replay_speed)

                    self._current_timestamp = new_ts
                    self._bar_count += 1

            # Subscribe to track timestamps
            self._event_bus.subscribe("price_bar", track_timestamp, priority=1000)  # High priority

            # Stream data (publishes PriceBarEvent for each symbol/bar)
            self._data_service.stream_universe(
                symbols=self.config.universe,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                is_warmup=False,
            )

            # Publish triggers for final timestamp
            if self._current_timestamp is not None:
                self._event_bus.publish(ValuationTriggerEvent(ts=self._current_timestamp))  # type: ignore[unreachable]
                self._event_bus.publish(RiskEvaluationTriggerEvent(ts=self._current_timestamp))

            logger.info(
                "backtest.main_phase.complete",
                bars_processed=self._bar_count,
            )

            # Step 3: Collect Results
            logger.info("backtest.collecting_results")

            # Get final portfolio state
            final_equity = self._portfolio_service.get_equity()

            # Get trade count from execution service
            # ExecutionService tracks fills, count them
            num_fills = len(self._execution_service.get_filled_orders())

            # Calculate performance metrics
            initial_capital = float(self.config.initial_capital)
            final_capital = float(final_equity)
            total_return = (final_capital - initial_capital) / initial_capital if initial_capital > 0 else 0.0

            duration = datetime.now() - start_time

            logger.info(
                "backtest.completed",
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                num_fills=num_fills,
                duration_seconds=duration.total_seconds(),
            )

            return BacktestResult(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                initial_capital=initial_capital,
                final_capital=final_capital,
                total_return=total_return,
                num_trades=num_fills,
                duration=duration,
            )

        except Exception as e:
            logger.error(
                "backtest.failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Backtest execution failed: {e}") from e
