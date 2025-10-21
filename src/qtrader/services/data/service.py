"""Data service implementation.

Provides concrete implementation of IDataService that coordinates
data loading from vendor adapters using DataLoader and DataSourceResolver.
"""

from datetime import date
from typing import Dict, List, Optional

import structlog

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.config import DataConfig
from qtrader.data.iterator import PriceSeriesIterator
from qtrader.data.loader import DataLoader
from qtrader.models.instrument import Instrument

logger = structlog.get_logger(__name__)


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
    ):
        """
        Initialize data service.

        Args:
            config: Data configuration
            dataset: Dataset name from data_sources.yaml (e.g., "schwab-us-equity-1d-adjusted").
                    If None, will try to infer from config.source_selector (legacy behavior).
            resolver: Data source resolver (creates default if None)

        Examples:
            >>> # Explicit dataset (RECOMMENDED)
            >>> service = DataService(config, dataset="schwab-us-equity-1d-adjusted")
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

        # Convert config to dict for DataLoader (legacy interface)
        # TODO: Update DataLoader to accept DataConfig directly in Phase 2
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
    ) -> Dict[str, PriceSeriesIterator]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start of date range
            end_date: End of date range (inclusive)
            data_source: Optional override for data source

        Returns:
            Dict mapping symbol → iterator

        Raises:
            ValueError: If any symbol not found

        Examples:
            >>> iterators = service.load_universe(
            ...     ["AAPL", "MSFT", "GOOGL"],
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31)
            ... )
            >>> for symbol, iterator in iterators.items():
            ...     print(f"Processing {symbol}")
            ...     for bar in iterator:
            ...         print(bar.adjusted.close)
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
            # For now, continue with successful symbols
            # TODO: Add strict mode in Phase 2 to fail on any missing symbol
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

        Args:
            data_source: Filter by data source (None = all)

        Returns:
            List of available symbols

        Examples:
            >>> symbols = service.list_available_symbols()
            >>> print(f"Found {len(symbols)} symbols")
        """
        # TODO: Implement symbol discovery in Phase 2
        # For now, this requires registry or scanning data files
        logger.warning(
            "data_service.list_available_symbols.not_implemented",
            data_source=data_source,
        )
        raise NotImplementedError(
            "Symbol discovery will be implemented in Phase 2. For now, symbols must be provided explicitly."
        )

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
    ) -> list:
        """
        Get corporate actions for symbol in date range.

        Returns events in chronological order.
        Empty list if data source doesn't provide corp actions.

        Args:
            symbol: Ticker symbol
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of CorporateActionEvent

        Examples:
            >>> actions = service.get_corporate_actions(
            ...     "AAPL",
            ...     date(2020, 1, 1),
            ...     date(2020, 12, 31)
            ... )
            >>> for action in actions:
            ...     if action.action_type == "dividend":
            ...         print(f"Dividend: ${action.dividend_amount}")
            ...     elif action.action_type == "split":
            ...         print(f"Split: {action.split_ratio}:1")
        """
        if start_date > end_date:
            raise ValueError(f"Invalid date range: {start_date} > {end_date}")

        logger.info(
            "data_service.get_corporate_actions",
            symbol=symbol,
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        # Get instrument and create adapter
        instrument = self.get_instrument(symbol)
        adapter_config = self._build_adapter_config()

        # Import adapter dynamically (AlgoseekOHLCVendorAdapter)
        from qtrader.adapters.algoseek import AlgoseekOHLCVendorAdapter

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
