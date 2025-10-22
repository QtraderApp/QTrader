"""Data service implementation.

Provides concrete implementation of IDataService that coordinates
data loading from vendor adapters using DataLoader and DataSourceResolver.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import structlog

from qtrader.events.event_bus import IEventBus
from qtrader.events.events import PriceBarEvent
from qtrader.models.instrument import Instrument
from qtrader.services.data.adapters.resolver import DataSourceResolver
from qtrader.services.data.data_config import DataConfig
from qtrader.services.data.loaders.iterator import PriceSeriesIterator
from qtrader.services.data.loaders.loader import DataLoader

logger = structlog.get_logger()


class DataService:
    """
    Concrete implementation of data service.

    Delegates to DataLoader and adapters for actual data loading.
    Provides clean interface for consumers (strategies, backtests).

    Attributes:
        config: Data configuration
        resolver: Data source resolver for adapter lookup
        loader: DataLoader for coordinating data loading
        _instrument_cache: Cache of Instrument objects by symbol

    Examples:
        >>> # Initialize with config
        >>> from qtrader.config import AssetClass, DataSourceSelector
        >>> selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        >>> config = DataConfig(
        ...     mode="adjusted",
        ...     bar_schema=bar_schema,
        ...     source_selector=selector
        ... )
        >>> service = DataService(config)
        >>>
        >>> # Load single symbol
        >>> iterator = service.load_symbol(
        ...     "AAPL",
        ...     date(2020, 1, 1),
        ...     date(2020, 12, 31)
        ... )
        >>> for multi_bar in iterator:
        ...     print(multi_bar.adjusted.close)
        >>>
        >>> # Load universe
        >>> iterators = service.load_universe(
        ...     ["AAPL", "MSFT", "GOOGL"],
        ...     date(2020, 1, 1),
        ...     date(2020, 12, 31)
        ... )

    Notes:
        - Wraps existing DataLoader for backward compatibility
        - Adds clean interface for future service consumers
        - Caches instrument metadata for performance
        - Handles data source resolution and adapter selection
    """

    def __init__(
        self,
        config: DataConfig,
        dataset: Optional[str] = None,
        resolver: Optional[DataSourceResolver] = None,
        event_bus: Optional[IEventBus] = None,
    ):
        """
        Initialize data service.

        Args:
            config: Data configuration
            dataset: Dataset name from data_sources.yaml (e.g., "schwab-us-equity-1d-adjusted").
                    If None, will try to infer from config.source_selector (legacy behavior).
            resolver: Data source resolver (creates default if None)
            event_bus: Optional EventBus for publishing PriceBarEvent and CorporateActionEvent.
                      If None, DataService operates in non-event mode (pull-based).

        Examples:
            >>> # Explicit dataset (RECOMMENDED)
            >>> service = DataService(config, dataset="schwab-us-equity-1d-adjusted")
            >>>
            >>> # With EventBus (event-driven mode)
            >>> from qtrader.events.event_bus import EventBus
            >>> bus = EventBus()
            >>> service = DataService(config, dataset="schwab-us-equity-1d-adjusted", event_bus=bus)
            >>>
            >>> # With custom resolver
            >>> resolver = DataSourceResolver("config/custom_sources.yaml")
            >>> service = DataService(config, dataset="schwab-us-equity-1d-adjusted", resolver=resolver)
            >>>
            >>> # Legacy mode (infers from source_selector)
            >>> service = DataService(config)  # Will log warning
        """
        self.config = config
        self.resolver = resolver or DataSourceResolver()
        self._event_bus = event_bus

        # Store explicit dataset if provided
        self.dataset = dataset

        if not self.dataset:
            logger.warning(
                "DataService initialized without explicit dataset. "
                "This is deprecated. Pass dataset parameter explicitly. "
                "Attempting to infer from config.source_selector..."
            )
            # Try to infer from source_selector (legacy)
            if hasattr(config, "source_selector") and config.source_selector:
                # Try to find matching dataset
                self.dataset = self._infer_dataset_from_selector(config.source_selector)
            else:
                logger.error("Cannot infer dataset - no dataset parameter and no source_selector in config")

        # Create DataLoader with dict config (includes adapter config)
        # DataLoader now accepts both DataConfig and dict
        config_dict = {
            "adapter": self._build_adapter_config(),
        }
        self.loader = DataLoader(config_dict)

        # Cache for instrument objects
        self._instrument_cache: Dict[str, Instrument] = {}

        logger.info(
            "data_service.initialized",
            mode=config.mode,
            source=config.source_selector.to_tag(),
        )

    @classmethod
    def from_config(
        cls,
        config_dict: dict,
        dataset: str,
        resolver: Optional[DataSourceResolver] = None,
        event_bus: Optional[IEventBus] = None,
    ) -> "DataService":
        """
        Factory method to create DataService from backtest configuration.

        This bridges the gap between backtest DataConfig (simple source + path)
        and the service-level DataConfig (complex with source_selector, mode, etc.).

        Args:
            config_dict: Dict from backtest DataConfig.model_dump()
                        Expected keys: 'source', 'data_path', 'dataset'
            dataset: Dataset name from data_sources.yaml
                    (e.g., "schwab-us-equity-1d-adjusted")
            resolver: Optional DataSourceResolver instance
            event_bus: Optional EventBus for event-driven operation

        Returns:
            Configured DataService instance

        Examples:
            >>> from qtrader.backtest.config import BacktestConfig
            >>> backtest_config = BacktestConfig(...)
            >>> service = DataService.from_config(
            ...     config_dict=backtest_config.data.model_dump(),
            ...     dataset="schwab-us-equity-1d-adjusted",
            ...     event_bus=event_bus
            ... )

        Note:
            This method creates a service-level DataConfig internally by inferring
            the proper source_selector and mode from the dataset name.
        """
        from qtrader.services.data.data_config import BarSchemaConfig
        from qtrader.services.data.data_config import DataConfig as ServiceDataConfig
        from qtrader.services.data.data_source_selector import AssetClass, DataSourceSelector

        source = config_dict.get("source", "schwab")

        # Infer asset class and mode from dataset name
        # Dataset format: {provider}-{asset_class}-{timeframe}-{mode}
        # e.g., "schwab-us-equity-1d-adjusted" or "algoseek-us-equity-1d-ohlc"

        # Determine asset class (default to EQUITY)
        asset_class = AssetClass.EQUITY
        if "equity" in dataset.lower():
            asset_class = AssetClass.EQUITY
        elif "option" in dataset.lower():
            asset_class = AssetClass.OPTIONS
        elif "future" in dataset.lower():
            asset_class = AssetClass.FUTURES

        # Determine mode (adjusted vs ohlc)
        mode = "adjusted" if "adjusted" in dataset.lower() else "ohlc"

        # Create source selector
        source_selector = DataSourceSelector(
            provider=source,
            asset_class=asset_class,
        )

        # Create minimal bar schema (will be overridden by resolver/loader)
        bar_schema = BarSchemaConfig(
            ts="trade_datetime",
            symbol="symbol",
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
        )

        # Create service-level DataConfig
        service_config = ServiceDataConfig(
            mode=mode,
            source_selector=source_selector,
            bar_schema=bar_schema,
        )

        # Create resolver if not provided
        if resolver is None:
            resolver = DataSourceResolver()

        logger.info(
            "data_service.from_config",
            source=source,
            dataset=dataset,
            mode=mode,
            asset_class=asset_class.value,
        )

        return cls(
            config=service_config,
            dataset=dataset,
            resolver=resolver,
            event_bus=event_bus,
        )

    def load_symbol(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        data_source: Optional[str] = None,
    ) -> PriceSeriesIterator:
        """
        Load data for single symbol.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start_date: Start of date range
            end_date: End of date range (inclusive)
            data_source: Optional override for data source

        Returns:
            Iterator yielding MultiBar instances

        Raises:
            ValueError: If symbol not found or invalid date range
            FileNotFoundError: If data files missing

        Examples:
            >>> iterator = service.load_symbol(
            ...     "AAPL",
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31)
            ... )
            >>> first_bar = next(iterator)
            >>> print(first_bar.adjusted.close)
        """
        if start_date > end_date:
            raise ValueError(f"Invalid date range: {start_date} > {end_date}")

        logger.info(
            "data_service.load_symbol",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            data_source=data_source,
        )

        # Use DataLoader to load data
        iterator = self.loader.load_data(
            symbol,
            start_date.isoformat(),
            end_date.isoformat(),
        )

        logger.info(
            "data_service.load_symbol.complete",
            symbol=symbol,
        )

        return iterator

    def load_universe(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        *,
        data_source: Optional[str] = None,
        strict: bool = False,
    ) -> Dict[str, PriceSeriesIterator]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start of date range
            end_date: End of date range (inclusive)
            data_source: Optional override for data source
            strict: If True, raise ValueError if any symbol fails to load.
                   If False (default), log warnings and continue with successful symbols.

        Returns:
            Dict mapping symbol → iterator (only successful symbols)

        Raises:
            ValueError: If any symbol not found and strict=True

        Examples:
            >>> # Lenient mode (default) - continues with successful symbols
            >>> iterators = service.load_universe(
            ...     ["AAPL", "INVALID", "MSFT"],
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31)
            ... )
            >>> # Returns {"AAPL": ..., "MSFT": ...}, logs warning for INVALID
            >>>
            >>> # Strict mode - fails fast on any error
            >>> iterators = service.load_universe(
            ...     ["AAPL", "INVALID", "MSFT"],
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31),
            ...     strict=True
            ... )
            >>> # Raises ValueError immediately when INVALID fails
        """
        if start_date > end_date:
            raise ValueError(f"Invalid date range: {start_date} > {end_date}")

        logger.info(
            "data_service.load_universe",
            symbol_count=len(symbols),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            data_source=data_source,
        )

        # Load each symbol independently
        iterators: Dict[str, PriceSeriesIterator] = {}
        failed_symbols: List[str] = []

        for symbol in symbols:
            try:
                iterator = self.load_symbol(
                    symbol,
                    start_date,
                    end_date,
                    data_source=data_source,
                )
                iterators[symbol] = iterator
            except (ValueError, FileNotFoundError) as e:
                logger.warning(
                    "data_service.load_universe.symbol_failed",
                    symbol=symbol,
                    error=str(e),
                )
                failed_symbols.append(symbol)

        if failed_symbols:
            if strict:
                # Strict mode: raise on any failure
                raise ValueError(
                    f"Failed to load {len(failed_symbols)} symbol(s) in strict mode: {failed_symbols}. "
                    f"Successfully loaded {len(iterators)} symbols."
                )
            else:
                # Lenient mode: continue with successful symbols
                logger.warning(
                    "data_service.load_universe.partial_success",
                    failed_count=len(failed_symbols),
                    failed_symbols=failed_symbols,
                )

        logger.info(
            "data_service.load_universe.complete",
            success_count=len(iterators),
            failed_count=len(failed_symbols),
        )

        return iterators

    def stream_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        is_warmup: bool = False,
    ) -> None:
        """
        Load bars and publish PriceBarEvent for each bar (event-driven mode).

        This method combines loading and event publishing for a single symbol.
        Requires EventBus to be configured during initialization.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL')
            start_date: Start of date range
            end_date: End of date range (inclusive)
            is_warmup: If True, publishes bars with is_warmup=True flag.
                      Strategies should NOT generate signals during warmup.

        Raises:
            ValueError: If EventBus not configured
            ValueError: If symbol not found or invalid date range
            FileNotFoundError: If data files missing

        Examples:
            >>> # Load warmup bars (no signals)
            >>> service.stream_bars("AAPL", date(2019, 1, 1), date(2019, 12, 31), is_warmup=True)
            >>>
            >>> # Load live bars (generate signals)
            >>> service.stream_bars("AAPL", date(2020, 1, 1), date(2020, 12, 31), is_warmup=False)

        Flow:
            DataService loads bar → Publishes PriceBarEvent →
            PortfolioService updates prices →
            ExecutionService checks fills →
            StrategyService sees bar → Publishes SignalEvent

        Note:
            This method blocks until all bars are processed. Use in BacktestEngine event loop.
        """
        if self._event_bus is None:
            raise ValueError(
                "EventBus not configured. Initialize DataService with event_bus parameter "
                "or use load_symbol() for non-event mode."
            )

        logger.info(
            "data_service.stream_bars",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            is_warmup=is_warmup,
        )

        # Load data
        iterator = self.load_symbol(symbol, start_date, end_date)

        # Stream bars as events
        bar_count = 0
        for multi_bar in iterator:
            # Publish event with adjusted bar (canonical mode for strategies)
            event = PriceBarEvent(
                symbol=symbol,
                bar=multi_bar.adjusted,  # Use adjusted mode for strategies
                timestamp=multi_bar.adjusted.trade_datetime,
                is_warmup=is_warmup,
            )
            self._event_bus.publish(event)
            bar_count += 1

        logger.info(
            "data_service.stream_bars.complete",
            symbol=symbol,
            bar_count=bar_count,
            is_warmup=is_warmup,
        )

    def stream_universe(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        *,
        is_warmup: bool = False,
        strict: bool = False,
    ) -> None:
        """
        Load bars for multiple symbols and publish PriceBarEvent for each bar.

        Synchronizes iterators to publish all bars for a given timestamp together
        before moving to next timestamp. This ensures proper event ordering.

        Args:
            symbols: List of ticker symbols
            start_date: Start of date range
            end_date: End of date range (inclusive)
            is_warmup: If True, all bars published with is_warmup=True
            strict: If True, raise ValueError if any symbol fails to load

        Raises:
            ValueError: If EventBus not configured
            ValueError: If any symbol not found and strict=True

        Examples:
            >>> # Stream universe in warmup mode
            >>> service.stream_universe(
            ...     ["AAPL", "MSFT", "GOOGL"],
            ...     date(2019, 1, 1),
            ...     date(2019, 12, 31),
            ...     is_warmup=True
            ... )
            >>>
            >>> # Stream universe in live mode
            >>> service.stream_universe(
            ...     ["AAPL", "MSFT", "GOOGL"],
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31),
            ...     is_warmup=False
            ... )

        Event Ordering:
            For each timestamp T:
            1. Publish PriceBarEvent(AAPL, T)
            2. Publish PriceBarEvent(MSFT, T)
            3. Publish PriceBarEvent(GOOGL, T)
            4. Move to next timestamp T+1

        Note:
            Bars are aligned by timestamp. If a symbol is missing data for a timestamp,
            that symbol is skipped for that timestamp.
        """
        if self._event_bus is None:
            raise ValueError(
                "EventBus not configured. Initialize DataService with event_bus parameter "
                "or use load_universe() for non-event mode."
            )

        logger.info(
            "data_service.stream_universe",
            symbol_count=len(symbols),
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            is_warmup=is_warmup,
        )

        # Load iterators for all symbols
        iterators = self.load_universe(symbols, start_date, end_date, strict=strict)

        # Create dict to track active iterators
        active_iterators = {symbol: iter(iterator) for symbol, iterator in iterators.items()}

        # Track bars by timestamp for synchronization
        timestamp_bars: Dict[datetime, Dict[str, Any]] = {}

        # Collect all bars from all iterators
        for symbol, iterator in active_iterators.items():
            try:
                for multi_bar in iterator:
                    ts = multi_bar.adjusted.trade_datetime
                    if ts not in timestamp_bars:
                        timestamp_bars[ts] = {}
                    timestamp_bars[ts][symbol] = multi_bar
            except StopIteration:
                continue

        # Publish bars in timestamp order
        total_bars = 0
        for ts in sorted(timestamp_bars.keys()):
            bars_at_ts = timestamp_bars[ts]

            # Publish bar for each symbol at this timestamp
            for symbol, multi_bar in sorted(bars_at_ts.items()):  # Sort for determinism
                event = PriceBarEvent(
                    symbol=symbol,
                    bar=multi_bar.adjusted,  # Use adjusted mode for strategies
                    timestamp=ts,
                    is_warmup=is_warmup,
                )
                self._event_bus.publish(event)
                total_bars += 1

        logger.info(
            "data_service.stream_universe.complete",
            symbol_count=len(active_iterators),
            total_bars=total_bars,
            unique_timestamps=len(timestamp_bars),
            is_warmup=is_warmup,
        )

    def get_instrument(self, symbol: str) -> Instrument:
        """
        Get instrument for symbol.

        Creates minimal instrument (symbol only) since dataset provides
        all metadata (provider, asset type, etc.).

        Args:
            symbol: Ticker symbol

        Returns:
            Instrument with symbol

        Examples:
            >>> service = DataService(config, dataset="schwab-us-equity-1d-adjusted")
            >>> instrument = service.get_instrument("AAPL")
            >>> print(instrument.symbol)  # 'AAPL'
        """
        # Check cache first
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        # Build minimal instrument (symbol only)
        # Dataset provides all metadata (no duplication)
        instrument = Instrument(symbol=symbol)

        # Cache it
        self._instrument_cache[symbol] = instrument

        logger.debug(
            "data_service.get_instrument",
            symbol=symbol,
            dataset=self.dataset,
        )

        return instrument

    def list_available_symbols(
        self,
        data_source: Optional[str] = None,
    ) -> List[str]:
        """
        List all available symbols.

        Reads symbols from the symbol_map file configured in the adapter.
        For Algoseek, this is typically the equity_security_master CSV.

        Args:
            data_source: Filter by data source (currently unused, reserved for future multi-source support)

        Returns:
            List of available symbols (sorted)

        Raises:
            FileNotFoundError: If symbol_map file not found
            ValueError: If symbol_map not configured in adapter

        Examples:
            >>> symbols = service.list_available_symbols()
            >>> print(f"Found {len(symbols)} symbols")
            >>> print(symbols[:5])  # First 5 symbols
        """
        logger.info(
            "data_service.list_available_symbols",
            data_source=data_source,
        )

        # Get adapter config
        adapter_config = self._build_adapter_config()

        # Check if symbol_map is configured
        if "symbol_map" not in adapter_config:
            logger.error(
                "data_service.list_available_symbols.no_symbol_map",
                adapter_config_keys=list(adapter_config.keys()),
            )
            raise ValueError(
                "Symbol map not configured in adapter config. Add 'symbol_map' path to your data source configuration."
            )

        # Read symbol map file
        from pathlib import Path

        import pandas as pd

        symbol_map_path = Path(adapter_config["symbol_map"])

        if not symbol_map_path.exists():
            logger.error(
                "data_service.list_available_symbols.file_not_found",
                path=str(symbol_map_path),
            )
            raise FileNotFoundError(f"Symbol map file not found: {symbol_map_path}")

        try:
            # Read CSV
            df = pd.read_csv(symbol_map_path)

            # Extract Symbol/Tickers column (case-insensitive, try multiple variations)
            symbol_col = None
            for col in df.columns:
                col_lower = col.lower()
                if col_lower in ("symbol", "symbols", "ticker", "tickers"):
                    symbol_col = col
                    break

            if symbol_col is None:
                raise ValueError(
                    f"No symbol column found in {symbol_map_path}. "
                    f"Expected 'Symbol', 'Tickers', 'Ticker', or similar. "
                    f"Available columns: {list(df.columns)}"
                )

            # Get unique symbols - handle multi-valued cells (e.g., "AAPL,APPL")
            symbols_raw = df[symbol_col].dropna().unique()
            symbols_set: set[str] = set()

            for val in symbols_raw:
                # Handle comma-separated tickers in single cell
                if "," in str(val):
                    symbols_set.update(s.strip() for s in str(val).split(","))
                else:
                    symbols_set.add(str(val).strip())

            # Sort and return
            symbols = sorted(symbols_set)

            logger.info(
                "data_service.list_available_symbols.complete",
                symbol_count=len(symbols),
                source_file=str(symbol_map_path),
            )

            return symbols

        except Exception as e:
            logger.error(
                "data_service.list_available_symbols.error",
                error=str(e),
                path=str(symbol_map_path),
            )
            raise

    def _infer_dataset_from_selector(self, selector) -> Optional[str]:
        """
        Try to infer dataset name from source_selector (legacy support).

        Args:
            selector: DataSourceSelector from config

        Returns:
            Dataset name if found, None otherwise
        """
        try:
            # Try to find matching dataset
            for source_name, source_config in self.resolver.sources.items():
                if selector.matches(source_config):
                    logger.info(
                        "data_service.inferred_dataset",
                        dataset=source_name,
                        selector=selector.to_tag(),
                    )
                    return source_name

            # If no match, try provider name
            provider = selector.provider
            if provider:
                # Look for dataset with matching provider
                for source_name, source_config in self.resolver.sources.items():
                    if source_config.get("provider") == provider:
                        logger.info(
                            "data_service.inferred_dataset_by_provider",
                            dataset=source_name,
                            provider=provider,
                        )
                        return source_name
        except Exception as e:
            logger.error(
                "data_service.dataset_inference_failed",
                error=str(e),
            )

        return None

    def _build_adapter_config(self) -> Dict:
        """
        Build adapter configuration dict from DataConfig.

        Returns:
            Configuration dict for adapter initialization

        Notes:
            - Converts Pydantic config to dict format expected by adapters
            - Looks up adapter settings using source_selector
            - This is a temporary bridge until Phase 2
        """
        # Use source_selector to find matching source
        try:
            # Find matching source by selector criteria
            matching_sources = []
            for source_name, source_config in self.resolver.sources.items():
                if self.config.source_selector.matches(source_config):
                    matching_sources.append((source_name, source_config))

            if matching_sources:
                # Use first match
                source_name, adapter_config = matching_sources[0]
                logger.debug(
                    "data_service.adapter_config_resolved",
                    source_name=source_name,
                    selector=self.config.source_selector.to_tag(),
                )
                return adapter_config.copy()

            # If no match by selector, try backward compatibility with provider name
            provider = self.config.source_selector.provider
            if provider:
                for source_name, source_config in self.resolver.sources.items():
                    if source_config.get("provider") == provider:
                        logger.debug(
                            "data_service.adapter_config_resolved_by_provider",
                            source_name=source_name,
                            provider=provider,
                        )
                        return source_config.copy()
        except Exception as e:
            logger.warning(
                "data_service.adapter_config_lookup_failed",
                error=str(e),
                selector=self.config.source_selector.to_tag(),
            )

        # Fallback to basic config
        logger.warning(
            "data_service.using_fallback_adapter_config",
            selector=self.config.source_selector.to_tag(),
        )
        return {
            "adapter": "algoseekOHLC",
            "root_path": "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample",
        }

    def get_corporate_actions(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        *,
        publish_events: bool = True,
    ) -> list:
        """
        Get corporate actions for symbol in date range.

        Returns events in chronological order.
        Empty list if data source doesn't provide corp actions.

        Args:
            symbol: Ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
            publish_events: If True and EventBus configured, publishes CorporateActionEvent
                          for each action. Default True.

        Returns:
            List of CorporateActionEvent

        Examples:
            >>> # Get actions without publishing (pull mode)
            >>> actions = service.get_corporate_actions(
            ...     "AAPL",
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31),
            ...     publish_events=False
            ... )
            >>> for action in actions:
            ...     if action.action_type == "dividend":
            ...         print(f"Dividend: ${action.dividend_amount}")
            ...     elif action.action_type == "split":
            ...         print(f"Split: {action.split_ratio}:1")
            >>>
            >>> # Get actions with event publishing (event-driven mode)
            >>> service.get_corporate_actions("AAPL", date(2020, 1, 1), date(2020, 12, 31))
            >>> # Publishes CorporateActionEvent for each action
        """
        if start_date > end_date:
            raise ValueError(f"Invalid date range: {start_date} > {end_date}")

        logger.info(
            "data_service.get_corporate_actions",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            publish_events=publish_events and self._event_bus is not None,
        )

        # Get instrument and create adapter
        instrument = self.get_instrument(symbol)
        adapter_config = self._build_adapter_config()

        # Import adapter dynamically (AlgoseekOHLCVendorAdapter)
        from qtrader.services.data.adapters.algoseek import AlgoseekOHLCVendorAdapter

        adapter = AlgoseekOHLCVendorAdapter(adapter_config, instrument)

        # Check if adapter supports corporate actions
        if not hasattr(adapter, "get_corporate_actions"):
            logger.warning(
                "data_service.get_corporate_actions.not_supported",
                symbol=symbol,
                adapter=type(adapter).__name__,
            )
            return []

        # Get corporate actions from adapter
        try:
            actions: list = adapter.get_corporate_actions(
                start_date.isoformat(),
                end_date.isoformat(),
            )

            # Publish events if requested and EventBus configured
            if publish_events and self._event_bus is not None:
                for action in actions:
                    # action is already a CorporateActionEvent from adapter
                    self._event_bus.publish(action)

                logger.info(
                    "data_service.get_corporate_actions.published",
                    symbol=symbol,
                    count=len(actions),
                )

            logger.info(
                "data_service.get_corporate_actions.complete",
                symbol=symbol,
                count=len(actions),
            )

            return actions

        except Exception as e:
            logger.error(
                "data_service.get_corporate_actions.error",
                symbol=symbol,
                error=str(e),
            )
            raise
