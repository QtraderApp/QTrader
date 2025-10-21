"""Data service interface definition.

Defines Protocol interfaces for data service and adapter implementations.
These protocols enable dependency injection and make services independently
testable through mocking.
"""

from datetime import date
from typing import Dict, List, Optional, Protocol

from qtrader.data.iterator import PriceSeriesIterator
from qtrader.models.bar import PriceSeries
from qtrader.models.instrument import Instrument


class IDataService(Protocol):
    """
    Data service interface for loading and streaming price data.

    Responsibilities:
    - Load historical data for symbols
    - Transform to canonical format with adjustment modes
    - Stream data via iterators
    - Provide instrument metadata

    Does NOT:
    - Execute orders
    - Manage portfolio
    - Calculate indicators
    - Make trading decisions

    Examples:
        >>> # Single symbol loading
        >>> service: IDataService = DataService(config)
        >>> iterator = service.load_symbol(
        ...     "AAPL",
        ...     date(2020, 1, 1),
        ...     date(2020, 12, 31)
        ... )
        >>> for multi_bar in iterator:
        ...     print(multi_bar.adjusted.close)
        >>>
        >>> # Multiple symbols (universe)
        >>> iterators = service.load_universe(
        ...     ["AAPL", "MSFT", "GOOGL"],
        ...     date(2020, 1, 1),
        ...     date(2020, 12, 31)
        ... )
        >>> for symbol, iterator in iterators.items():
        ...     print(f"Processing {symbol}")
    """

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
        """
        ...

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
        """
        ...

    def get_instrument(self, symbol: str) -> Instrument:
        """
        Get instrument metadata.

        Args:
            symbol: Ticker symbol

        Returns:
            Instrument with metadata

        Raises:
            ValueError: If symbol not found
        """
        ...

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
        """
        ...

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
        """
        ...


class IDataAdapter(Protocol):
    """
    Adapter interface for vendor-specific data sources.

    Implementations: AlgoseekOHLCVendorAdapter, SchwabOHLCVendorAdapter, etc.

    Responsibilities:
    - Read raw bars from vendor data source
    - Transform to canonical PriceSeries with all adjustment modes
    - Handle vendor-specific quirks (file formats, column names, etc.)

    Examples:
        >>> adapter: IDataAdapter = AlgoseekOHLCVendorAdapter(config, instrument)
        >>> raw_bars = adapter.read_bars("2020-01-01", "2020-12-31")
        >>> canonical = adapter.to_canonical_series(raw_bars)
        >>> # canonical has keys: 'unadjusted', 'adjusted', 'total_return'
    """

    def read_bars(
        self,
        start_date: str,
        end_date: str,
    ) -> List:  # Vendor-specific bar type (AlgoseekBar, etc.)
        """
        Read raw bars from vendor data source.

        Args:
            start_date: Start date in ISO format (YYYY-MM-DD)
            end_date: End date in ISO format (YYYY-MM-DD)

        Returns:
            List of vendor-specific bar instances

        Raises:
            FileNotFoundError: If data files not found
            ValueError: If date range invalid
        """
        ...

    def to_canonical_series(self, bars: List) -> Dict[str, PriceSeries]:
        """
        Transform vendor bars to canonical series with all adjustment modes.

        Args:
            bars: List of vendor-specific bars

        Returns:
            Dict with keys 'unadjusted', 'adjusted', 'total_return',
            each mapping to a canonical PriceSeries

        Notes:
            - All three series must have matching lengths and timestamps
            - adjusted: Split-adjusted prices
            - total_return: Split + dividend adjusted prices
        """
        ...
