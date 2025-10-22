"""
Backtest Engine Implementation.

Pure event-driven orchestrator that coordinates all services via EventBus.
No business logic, no direct service calls, just event publishing.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta

import structlog

from qtrader.backtest.config import BacktestConfig
from qtrader.events.event_bus import EventBus
from qtrader.events.events import RiskEvaluationTriggerEvent, ValuationTriggerEvent
from qtrader.services.data.service import DataService
from qtrader.services.execution.service import ExecutionService
from qtrader.services.portfolio.service import PortfolioService
from qtrader.services.risk.service import RiskService
from qtrader.services.strategy.service import StrategyService

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

    Event Publishing Pattern:
    For each timestamp:
        1. Publish BarEvent (per symbol) - services update state
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

        Args:
            config: Backtest configuration loaded from YAML

        Returns:
            Configured BacktestEngine instance

        Raises:
            ValueError: If configuration is invalid or services fail to initialize
        """
        # Create event bus
        event_bus = EventBus()

        # Create data service using factory method (with EventBus for event-driven mode)
        data_service = DataService.from_config(
            config_dict=config.data.model_dump(),
            dataset=config.data.dataset,
            event_bus=event_bus,
        )

        # Create portfolio service
        portfolio_service = PortfolioService.from_config(config_dict=config.portfolio.model_dump(), event_bus=event_bus)

        # Create execution service
        execution_service = ExecutionService.from_config(config_dict=config.execution.model_dump(), event_bus=event_bus)

        # Create risk service
        risk_service = RiskService.from_config(config_dict=config.risk.model_dump(), event_bus=event_bus)

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
        Run the backtest.

        Event Flow:
        1. Warmup Phase (if warmup_bars > 0):
           - Stream warmup_bars worth of data with is_warmup=True
           - Strategies build indicators but don't generate signals
        2. Main Phase:
           - Stream data with is_warmup=False
           - For each timestamp:
             a. DataService publishes PriceBarEvent (per symbol)
             b. Engine publishes ValuationTriggerEvent
             c. Engine publishes RiskEvaluationTriggerEvent
        3. Results Collection:
           - Query final portfolio state
           - Count executed trades
           - Calculate performance metrics

        Returns:
            BacktestResult with performance metrics and trade log

        Raises:
            RuntimeError: If backtest execution fails
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
            def track_timestamp(event):
                if hasattr(event, "timestamp") and not event.is_warmup:
                    new_ts = event.timestamp
                    # If timestamp changed, publish triggers for previous timestamp
                    if self._current_timestamp is not None and new_ts != self._current_timestamp:
                        # All bars for previous timestamp published, trigger valuation and risk
                        self._event_bus.publish(ValuationTriggerEvent(ts=self._current_timestamp))
                        self._event_bus.publish(RiskEvaluationTriggerEvent(ts=self._current_timestamp))
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
