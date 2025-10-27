"""
Backtest Engine Implementation.

Minimal implementation focused on DataService only.
Other services suspended until refactored.

Pure event-driven orchestrator that coordinates services via EventBus.
"""

from dataclasses import dataclass
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
    Event-driven backtesting orchestrator.

    Architecture: Phase 1 - DataService Foundation
    ==============================================
    This is a minimal but complete implementation focusing on the data layer.
    The engine coordinates data streaming and event publishing via EventBus,
    with full event persistence through EventStore.

    Current Capabilities:
    - Load and validate backtest configuration
    - Create and manage EventBus with EventStore persistence
    - Initialize DataService with proper dataset configuration
    - Stream historical data with timestamp synchronization
    - Track execution metrics (bars processed, duration)

    Intentional Limitations:
    - No portfolio tracking (PortfolioService suspended)
    - No order execution simulation (ExecutionService suspended)
    - No risk management (RiskService suspended)
    - No strategy signals (StrategyService suspended)

    These services will be incrementally reintegrated following the
    lego architecture pattern once refactoring is complete.

    Event Flow (Current Phase):
    ===========================
    For each timestamp T across all symbols:
        1. DataService publishes PriceBarEvent(symbol=A, timestamp=T)
        2. DataService publishes PriceBarEvent(symbol=B, timestamp=T)
        3. ...all symbols at T before advancing to T+1
        4. EventStore persists all events

    Future Event Flow (After Service Integration):
    ==============================================
    For each timestamp T:
        Phase 1: MarketData
            - DataService publishes PriceBarEvent for ALL symbols at T
            - Services update internal state (prices, positions)

        Phase 2: Valuation (barrier)
            - Engine publishes ValuationTriggerEvent(ts=T)
            - PortfolioService calculates equity, positions, valuations

        Phase 3: RiskEvaluation (barrier)
            - Engine publishes RiskEvaluationTriggerEvent(ts=T)
            - RiskService processes signals from strategies
            - RiskService creates sized orders within risk limits

        Phase 4: Execution (next cycle)
            - ExecutionService fills orders at T+1 prices
            - FillEvent updates portfolio positions

    Resource Management:
    ====================
    The engine manages lifecycle of EventStore (SQLite or in-memory).
    Call shutdown() to properly close resources, or use as context manager:

        with BacktestEngine.from_config(config) as engine:
            result = engine.run()
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
        self._bar_count = 0  # Initialize for tracking bars processed

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
            if len(source_names) == 1:
                source_label = source_names[0]
            elif len(source_names) <= 3:
                source_label = "+".join(source_names)
            else:
                # For many sources, use count instead of names
                source_label = f"{len(source_names)}_sources"
            run_label = f"{source_label}_{config.start_date:%Y%m%d}_{config.end_date:%Y%m%d}"

        results_dir = results_base / run_label
        results_dir.mkdir(parents=True, exist_ok=True)

        event_store: EventStore
        store_filename = output_cfg.event_store.filename
        store_path = results_dir / store_filename

        # Create event store based on configured backend
        backend_type = output_cfg.event_store.backend
        try:
            if backend_type == "sqlite":
                event_store = SQLiteEventStore(store_path)
                logger.info(
                    "backtest.engine.event_store_initialized",
                    backend="SQLiteEventStore",
                    path=str(store_path),
                )
            else:  # memory
                event_store = InMemoryEventStore()
                logger.info(
                    "backtest.engine.event_store_initialized",
                    backend="InMemoryEventStore",
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
        # Extract provider from dataset name (format: provider-asset-freq-variant)
        # e.g., "algoseek-us-equity-1d-unadjusted" -> provider="algoseek"
        first_source = config.data.sources[0]
        dataset = first_source.name

        # Build config dict that from_config() expects
        # Provider will be extracted from dataset name inside from_config()
        config_dict = {
            "dataset": dataset,
        }

        # TODO: Align EventBus with IEventBus protocol to resolve type incompatibility
        data_service = DataService.from_config(
            config_dict=config_dict,
            dataset=dataset,
            event_bus=event_bus,
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

    def shutdown(self) -> None:
        """
        Clean up resources (close EventStore).

        Call this method to properly close SQLite connections and release
        file handles. Important for long-lived daemons or repeated runs.

        Examples:
            >>> engine = BacktestEngine.from_config(config)
            >>> try:
            ...     result = engine.run()
            ... finally:
            ...     engine.shutdown()
        """
        if self._event_store is not None:
            try:
                self._event_store.close()
                logger.debug("backtest.engine.event_store_closed")
            except Exception as e:
                logger.warning(
                    "backtest.engine.event_store_close_failed",
                    error=str(e),
                )

    def __enter__(self) -> "BacktestEngine":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - ensures cleanup."""
        self.shutdown()

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

            # Reset bar count for this run
            self._bar_count = 0

            # Subscribe to price_bar events to count them
            # Keep handler reference for cleanup
            def count_bars(event) -> None:
                self._bar_count += 1

            self._event_bus.subscribe("bar", count_bars, priority=1000)

            # Stream data from single configured source
            # Note: BacktestConfig validation enforces single source until multi-source
            # streaming is implemented (see config.py validate_single_source)
            first_source = self.config.data.sources[0]
            source_symbols = first_source.universe

            logger.info(
                "backtest.streaming_data",
                source=first_source.name,
                symbols=source_symbols,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            # Use stream_universe to ensure all symbols publish bars at each timestamp
            # before advancing (critical for cross-symbol strategies and risk barriers)
            #
            # MEMORY WARNING: Current DataService.stream_universe implementation loads
            # all bars into a timestamp_bars dict before publishing (see service.py:575-582).
            # For large universes or long date ranges, this can consume significant RAM.
            #
            # Estimated memory: ~500 bytes/bar * symbols * trading_days
            # Example: 100 symbols * 252 days * 500 bytes = ~12.6 MB (manageable)
            #          1000 symbols * 2520 days * 500 bytes = ~1.26 GB (high)
            #
            # TODO: Refactor DataService to use heap-merge streaming for incremental
            #       publishing instead of buffering all bars in memory.
            try:
                self._data_service.stream_universe(
                    symbols=list(source_symbols),
                    start_date=self.config.start_date,
                    end_date=self.config.end_date,
                    is_warmup=False,
                    strict=False,  # Continue if some symbols fail to load
                )
            except Exception as e:
                logger.error(
                    "backtest.data_stream_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise
            finally:
                # Always unsubscribe to prevent handler accumulation on re-runs
                self._event_bus.unsubscribe("bar", count_bars)

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
