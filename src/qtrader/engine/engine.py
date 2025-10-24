"""
Backtest Engine Implementation.

Minimal implementation focused on DataService only.
Other services suspended until refactored.

Pure event-driven orchestrator that coordinates services via EventBus.
"""

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import structlog

from qtrader.engine.config import BacktestConfig
from qtrader.events.event_bus import EventBus
from qtrader.events.event_store import EventStore, InMemoryEventStore, SQLiteEventStore
from qtrader.services.data.service import DataService
from qtrader.system.config import get_system_config

logger = structlog.get_logger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    start_date: date
    end_date: date
    bars_processed: int
    duration: timedelta


class BacktestEngine:
    """
    Minimal event-driven backtesting orchestrator.

    Current Phase: DataService only
    ================================
    Responsibilities:
    - Load configuration
    - Create EventBus and EventStore
    - Initialize DataService
    - Stream historical data (publish PriceBarEvent)
    - Track basic metrics

    Suspended Services (to be added incrementally):
    - PortfolioService
    - ExecutionService
    - RiskService
    - StrategyService

    Event Flow (Current):
    =====================
    1. Load data sources from config
    2. Stream bars for each symbol/timestamp
    3. Publish PriceBarEvent for each bar
    4. EventStore persists all events
    5. Return basic results (bars processed, duration)

    Future Event Flow (After Service Refactoring):
    ==============================================
    For each timestamp T:
        Phase 1: MarketData
            - DataService publishes PriceBarEvent for ALL symbols at T
            - Services update internal state

        Phase 2: Valuation (barrier)
            - Engine publishes ValuationTriggerEvent(ts=T)
            - Portfolio calculates equity, positions

        Phase 3: RiskEvaluation (barrier)
            - Engine publishes RiskEvaluationTriggerEvent(ts=T)
            - Risk processes signals, creates orders

        Phase 4: Execution (next cycle)
            - Execution fills orders at T+1 prices
    """

    def __init__(
        self,
        config: BacktestConfig,
        event_bus: EventBus,
        data_service: DataService,
        event_store: EventStore | None = None,
        results_dir: Path | None = None,
    ) -> None:
        """
        Initialize backtest engine.

        Args:
            config: Backtest configuration
            event_bus: Event bus for publishing events
            data_service: Data service for loading historical bars
            event_store: Optional persistence backend
            results_dir: Optional directory for run artifacts
        """
        self.config = config
        self._event_bus = event_bus
        self._data_service = data_service
        self._event_store = event_store
        self._results_dir = results_dir

        # Get all symbols from data sources
        all_symbols = config.all_symbols

        logger.info(
            "backtest.engine.initialized",
            start_date=config.start_date,
            end_date=config.end_date,
            universe_size=len(all_symbols),
            data_sources=len(config.data.sources),
            event_store=getattr(event_store, "__class__", type(None)).__name__,
            results_dir=str(results_dir) if results_dir else None,
        )

    @classmethod
    def from_config(cls, config: BacktestConfig) -> "BacktestEngine":
        """
        Factory method to create engine from configuration.

        Loads SystemConfig for service configurations, uses BacktestConfig
        for run parameters (dates, data sources).

        Args:
            config: Backtest configuration loaded from YAML

        Returns:
            Configured BacktestEngine instance

        Raises:
            ValueError: If configuration is invalid or services fail to initialize
        """
        # Load system configuration
        system_config = get_system_config()

        # Initialize logging from system config
        from qtrader.system.log_system import LoggerFactory

        LoggerFactory.configure(system_config.logging.to_logger_config())

        # Create event bus
        event_bus = EventBus()

        # Initialize event store and attach to bus for full stream persistence
        output_cfg = system_config.output
        run_started_at = datetime.now()
        results_base = Path(output_cfg.default_results_dir)

        if output_cfg.organize_by_date:
            results_base = (
                results_base
                / run_started_at.strftime("%Y")
                / run_started_at.strftime("%m")
                / run_started_at.strftime("%d")
            )
        results_base.mkdir(parents=True, exist_ok=True)

        if output_cfg.use_timestamps:
            run_label = run_started_at.strftime(output_cfg.timestamp_format)
        else:
            # Use data source names and date range
            source_names = [s.name for s in config.data.sources]
            run_label = f"{source_names[0]}_{config.start_date:%Y%m%d}_{config.end_date:%Y%m%d}"

        results_dir = results_base / run_label
        results_dir.mkdir(parents=True, exist_ok=True)

        event_store: EventStore
        store_path = results_dir / "events.sqlite"
        try:
            event_store = SQLiteEventStore(store_path)
            logger.info(
                "backtest.engine.event_store_initialized",
                backend="SQLiteEventStore",
                path=str(store_path),
            )
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.warning(
                "backtest.engine.event_store_fallback",
                backend="InMemoryEventStore",
                reason=str(exc),
            )
            event_store = InMemoryEventStore()

        event_bus.attach_store(event_store)

        # Create data service using factory method
        # Use first data source for now (TODO: support multiple sources)
        first_source = config.data.sources[0]
        # TODO: Align EventBus with IEventBus protocol to resolve type incompatibility
        data_service = DataService.from_config(
            config_dict=asdict(system_config.data),
            dataset=first_source.name,  # Use data source name as dataset
            event_bus=event_bus,  # type: ignore
        )

        # Services suspended - will add back incrementally
        # portfolio_service = PortfolioService.from_config(...)
        # execution_service = ExecutionService.from_config(...)
        # risk_service = RiskService.from_config(...)
        # strategy_service = StrategyService.from_config(...)

        return cls(
            config=config,
            event_bus=event_bus,
            data_service=data_service,
            event_store=event_store,
            results_dir=results_dir,
        )

    def run(self) -> BacktestResult:
        """
        Run the backtest - stream data and publish events.

        Current Implementation (DataService only):
        =========================================
        1. Stream historical data for all symbols in date range
        2. DataService publishes PriceBarEvent for each bar
        3. EventStore persists all events
        4. Return basic metrics (bars processed, duration)

        Future (After Service Refactoring):
        ==================================
        - Add warmup phase for strategies
        - Publish ValuationTriggerEvent after all bars for timestamp
        - Publish RiskEvaluationTriggerEvent for signal processing
        - Collect portfolio metrics and trade statistics
        - Generate comprehensive results

        Returns:
            BacktestResult with basic metrics

        Raises:
            RuntimeError: If backtest execution fails
        """
        start_time = datetime.now()

        # Get all symbols from all data sources
        all_symbols = list(self.config.all_symbols)

        logger.info(
            "backtest.starting",
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            universe_size=len(all_symbols),
            data_sources=[s.name for s in self.config.data.sources],
        )

        try:
            # Main event loop - stream data
            logger.info(
                "backtest.main_phase.starting",
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            # Track bar count
            self._bar_count = 0

            # Subscribe to price_bar events to count them
            def count_bars(event) -> None:
                self._bar_count += 1

            self._event_bus.subscribe("bar", count_bars, priority=1000)

            # Stream data from first data source
            # TODO: Support multiple data sources when DataService is fully refactored
            first_source = self.config.data.sources[0]
            source_symbols = first_source.universe

            logger.info(
                "backtest.streaming_data",
                source=first_source.name,
                symbols=source_symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            # Load and stream data for each symbol
            for symbol in source_symbols:
                logger.debug(
                    "backtest.loading_symbol",
                    symbol=symbol,
                    source=first_source.name,
                )

                # DataService will publish PriceBarEvent for each bar
                # TODO: Use stream_universe once DataService is refactored
                # For now, load data using existing methods
                try:
                    iterator = self._data_service.load_symbol(
                        symbol=symbol,
                        start_date=self.config.start_date,
                        end_date=self.config.end_date,
                    )

                    # Iterate and publish (DataService should handle this)
                    for multi_bar in iterator:
                        # Event already published by DataService
                        pass

                except Exception as e:
                    logger.warning(
                        "backtest.symbol_load_failed",
                        symbol=symbol,
                        error=str(e),
                    )
                    continue

            logger.info(
                "backtest.main_phase.complete",
                bars_processed=self._bar_count,
            )

            # Collect results
            duration = datetime.now() - start_time

            logger.info(
                "backtest.completed",
                bars_processed=self._bar_count,
                duration_seconds=duration.total_seconds(),
            )

            return BacktestResult(
                start_date=self.config.start_date,
                end_date=self.config.end_date,
                bars_processed=self._bar_count,
                duration=duration,
            )

        except Exception as e:
            logger.error(
                "backtest.failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise RuntimeError(f"Backtest execution failed: {e}") from e
