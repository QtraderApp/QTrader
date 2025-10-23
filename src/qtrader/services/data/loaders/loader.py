"""
Data loading service - coordinates adapter and transformation.

This module provides the DataLoader class which coordinates data loading from
vendor adapters and transformation to canonical multi-mode format. It serves
as the main entry point for loading price data in the QTrader system.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Union, cast

from qtrader.contracts.data import DataSource, Instrument
from qtrader.services.data.adapters.algoseek import AlgoseekOHLCVendorAdapter
from qtrader.services.data.adapters.models.algoseek import AlgoseekBar, AlgoseekPriceSeries
from qtrader.services.data.loaders.iterator import PriceSeriesIterator

if TYPE_CHECKING:
    from qtrader.services.data.config import DataConfig


class DataLoader:
    """
    Coordinates data loading and transformation.

    The DataLoader is responsible for:
    1. Loading raw bars from vendor adapters
    2. Building vendor-specific price series
    3. Transforming to canonical series (all 3 adjustment modes)
    4. Returning an iterator that yields multi-mode bars

    This provides a clean separation between vendor-specific data access
    and the canonical data model used throughout the backtest engine.

    Attributes:
        config: Data configuration dictionary

    Examples:
        >>> # Load data for backtest
        >>> config = {"data_path": "/path/to/data"}
        >>> loader = DataLoader(config)
        >>> iterator = loader.load_data("AAPL", "2020-01-01", "2020-12-31")
        >>>
        >>> # Stream multi-mode bars
        >>> for multi_bar in iterator:
        ...     strategy_bar = multi_bar.adjusted
        ...     exec_bar = multi_bar.unadjusted
        ...     perf_bar = multi_bar.total_return

    Notes:
        - Phase 2: Basic structure with stub adapter
        - Phase 3: Full adapter integration
        - Configuration no longer includes price_series_mode
          (components select mode individually)
    """

    def __init__(self, config: Union["DataConfig", Dict[str, Any]]) -> None:
        """
        Initialize data loader.

        Args:
            config: Data configuration (DataConfig object or dict for backward compatibility)

        Examples:
            >>> # Using DataConfig (RECOMMENDED)
            >>> from qtrader.config import DataConfig
            >>> data_config = DataConfig(...)
            >>> loader = DataLoader(data_config)
            >>>
            >>> # Using dict (legacy, for backward compatibility)
            >>> config = {"adapter": {"root_path": "/data/equity", ...}}
            >>> loader = DataLoader(config)
        """
        # Handle both DataConfig and dict
        if isinstance(config, dict):
            self.config: Union["DataConfig", Dict[str, Any]] = config
            self._adapter_config: Dict[str, Any] = config.get("adapter", {})
        else:
            # DataConfig object - extract adapter config
            self.config = config
            # DataConfig doesn't have direct adapter config, will need to build it
            self._adapter_config = self._extract_adapter_config(config)

    def _extract_adapter_config(self, config: "DataConfig") -> Dict:
        """
        Extract adapter configuration from DataConfig.

        Args:
            config: DataConfig instance

        Returns:
            Dict with adapter configuration
        """
        # For now, DataConfig doesn't carry adapter config directly
        # This will be improved when we refactor adapter resolution
        # For now, return empty dict and let adapter resolution happen elsewhere
        return {}

    def load_data(self, symbol: str, start_date: str, end_date: str) -> PriceSeriesIterator:
        """
        Load data for symbol and return multi-mode iterator.

        This is the main entry point for loading price data. It coordinates
        the entire pipeline from raw vendor data to canonical multi-mode format.

        Args:
            symbol: Ticker symbol
            start_date: Start date (ISO format, e.g., '2020-01-01')
            end_date: End date (ISO format, e.g., '2020-12-31')

        Returns:
            PriceSeriesIterator yielding MultiBar (all 3 modes)

        Raises:
            ValueError: If adapter configuration missing
            FileNotFoundError: If data source not found

        Examples:
            >>> config = {"adapter": {"root_path": "data/...", ...}}
            >>> loader = DataLoader(config)
            >>> iterator = loader.load_data("AAPL", "2020-01-01", "2020-12-31")
            >>> first_bar = next(iterator)
            >>> print(first_bar.adjusted.close)  # Split-adjusted close

        Process:
            1. Load raw vendor bars (from adapter)
            2. Build vendor-specific PriceSeries (AlgoseekPriceSeries or SchwabPriceSeries)
            3. Transform to canonical series (all 3 modes)
            4. Return iterator with all modes available

        Notes:
            - All 3 modes loaded once (single data read)
            - Components select mode based on their purpose
            - Iterator provides memory-efficient streaming
            - Supports multiple vendors (Algoseek, Schwab, etc.)
        """
        # Step 1: Load raw bars from adapter
        raw_bars, data_source = self._load_from_adapter(symbol, start_date, end_date)

        # Step 2: Build vendor-specific series (currently only Algoseek)
        vendor_series = AlgoseekPriceSeries(symbol=symbol, bars=cast(List[AlgoseekBar], raw_bars))

        # Step 3: Transform to canonical (all 3 modes)
        canonical_series_dict = vendor_series.to_canonical_series()

        # Step 4: Return iterator with all modes
        # Components will select mode based on their config:
        # - Strategy: adjusted (split-consistent indicators)
        # - Execution: unadjusted (realistic fills)
        # - Performance: total_return (cumulative returns)
        return PriceSeriesIterator(canonical_series_dict)

    def _load_from_adapter(self, symbol: str, start_date: str, end_date: str) -> tuple[List[AlgoseekBar], DataSource]:
        """
        Load raw bars from vendor adapter.

        This is an internal method that handles the adapter layer. It returns
        raw vendor bars without any transformation. The transformation to
        canonical format happens in the caller (load_data).

        This method is responsible ONLY for adapter coordination, not
        transformation to canonical format.

        Args:
            symbol: Ticker symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            Tuple of (raw_bars, data_source):
                - raw_bars: List of vendor-specific bars (AlgoseekBar)
                - data_source: DataSource enum indicating the source

        Raises:
            ValueError: If adapter configuration missing
            FileNotFoundError: If data source not found

        Notes:
            - Adapter configured from self.config["adapter"]
            - Returns raw vendor bars (no transformation)
            - Transformation happens in load_data() via to_canonical_series()
            - Supports multiple vendors (currently Algoseek)
        """

        # Validate adapter configuration
        if not self._adapter_config:
            # Try to get from dict-style config for backward compatibility
            if isinstance(self.config, dict):
                if "adapter" not in self.config:
                    raise ValueError("Adapter configuration missing from config")
                self._adapter_config = self.config["adapter"]
            else:
                raise ValueError("Adapter configuration not available. Use DataService instead of DataLoader directly.")

        # Determine data source from adapter config
        adapter_name = self._adapter_config.get("adapter", "algoseekOHLC")

        # Map adapter name to DataSource enum
        if "algoseek" in adapter_name.lower():
            data_source = DataSource.ALGOSEEK
        else:
            # Default to ALGOSEEK
            data_source = DataSource.ALGOSEEK

        # Create minimal instrument (new API - just symbol)
        instrument = Instrument(symbol=symbol)

        # Initialize adapter and load bars
        algoseek_adapter = AlgoseekOHLCVendorAdapter(self._adapter_config, instrument)
        algoseek_bars = list(algoseek_adapter.read_bars(start_date, end_date))
        return algoseek_bars, data_source

    def load_data_from_series(self, vendor_series: AlgoseekPriceSeries) -> PriceSeriesIterator:
        """
        Load data from pre-built vendor series.

        Useful for testing and when vendor series already constructed.

        Args:
            vendor_series: AlgoseekPriceSeries with raw data

        Returns:
            PriceSeriesIterator yielding MultiBar

        Examples:
            >>> # For testing with golden data
            >>> vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=raw_bars)
            >>> iterator = loader.load_data_from_series(vendor_series)
        """
        # Transform to canonical (all 3 modes)
        canonical_series_dict = vendor_series.to_canonical_series()

        # Return iterator with all modes
        return PriceSeriesIterator(canonical_series_dict)
