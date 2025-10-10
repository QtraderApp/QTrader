"""
Data loading service - coordinates adapter and transformation.

This module provides the DataLoader class which coordinates data loading from
vendor adapters and transformation to canonical multi-mode format. It serves
as the main entry point for loading price data in the QTrader system.
"""

from typing import Dict, List

from qtrader.adapters.algoseek import AlgoseekOHLCVendorAdapter
from qtrader.data.iterator import PriceSeriesIterator
from qtrader.models.canonical_bar import CanonicalPriceSeries
from qtrader.models.instrument import DataSource, Instrument, InstrumentType
from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries


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

    def __init__(self, config: Dict) -> None:
        """
        Initialize data loader.

        Args:
            config: Data configuration dict
                   (mode selection moved to individual components)

        Examples:
            >>> config = {
            ...     "data_path": "/data/equity",
            ...     "vendor": "algoseek"
            ... }
            >>> loader = DataLoader(config)
        """
        self.config = config

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
            PriceSeriesIterator yielding MultiModeBar (all 3 modes)

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
            2. Build AlgoseekPriceSeries
            3. Transform to canonical series (all 3 modes)
            4. Return iterator with all modes available

        Notes:
            - All 3 modes loaded once (single data read)
            - Components select mode based on their purpose
            - Iterator provides memory-efficient streaming
        """
        # Step 1: Load raw bars from adapter
        raw_bars: List[AlgoseekBar] = self._load_from_adapter(symbol, start_date, end_date)

        # Step 2: Build vendor series
        vendor_series = AlgoseekPriceSeries(symbol=symbol, bars=raw_bars)

        # Step 3: Transform to canonical (all 3 modes)
        canonical_series_dict: Dict[str, CanonicalPriceSeries]
        canonical_series_dict = vendor_series.to_canonical_series()

        # Step 4: Return iterator with all modes
        # Components will select mode based on their config:
        # - Strategy: adjusted (split-consistent indicators)
        # - Execution: unadjusted (realistic fills)
        # - Performance: total_return (includes dividends)
        return PriceSeriesIterator(canonical_series_dict)

    def _load_from_adapter(self, symbol: str, start_date: str, end_date: str) -> List[AlgoseekBar]:
        """
        Load raw bars from adapter.

        This method integrates with the vendor adapter to load raw AlgoseekBar
        objects. The adapter handles data access while this layer handles
        transformation to canonical format.

        Args:
            symbol: Ticker symbol
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of AlgoseekBar objects (raw vendor data)

        Raises:
            ValueError: If adapter configuration missing
            FileNotFoundError: If data source not found

        Notes:
            - Adapter configured from self.config["adapter"]
            - Returns raw vendor bars (no transformation)
            - Transformation happens in load_data() via to_canonical_series()
        """

        # Validate adapter configuration
        if "adapter" not in self.config:
            raise ValueError("Adapter configuration missing from config")

        # Create instrument for adapter
        # TODO: Get instrument type from config/registry
        instrument = Instrument(symbol, InstrumentType.EQUITY, DataSource.ALGOSEEK)

        # Initialize adapter
        adapter = AlgoseekOHLCVendorAdapter(self.config["adapter"], instrument)

        # Load raw bars (iterator → list for now)
        # TODO: Consider keeping as iterator for memory efficiency
        raw_bars = list(adapter.read_bars(start_date, end_date))

        return raw_bars

    def load_data_from_series(self, vendor_series: AlgoseekPriceSeries) -> PriceSeriesIterator:
        """
        Load data from pre-built vendor series.

        Useful for testing and when vendor series already constructed.

        Args:
            vendor_series: AlgoseekPriceSeries with raw data

        Returns:
            PriceSeriesIterator yielding MultiModeBar

        Examples:
            >>> # For testing with golden data
            >>> vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=raw_bars)
            >>> iterator = loader.load_data_from_series(vendor_series)
        """
        # Transform to canonical (all 3 modes)
        canonical_series_dict = vendor_series.to_canonical_series()

        # Return iterator with all modes
        return PriceSeriesIterator(canonical_series_dict)
