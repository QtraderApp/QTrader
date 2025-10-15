"""
Instrument model and related types.

Provides logical abstraction for tradable instruments, decoupled from physical
data sources. Enables strategies to work with multiple asset types and data
sources without exposing implementation details.
"""

from enum import Enum
from typing import Any, Dict, NamedTuple, Optional


class InstrumentType(Enum):
    """Asset class classification."""

    EQUITY = "equity"
    CRYPTO = "crypto"
    FUTURE = "future"
    FOREX = "forex"
    SIGNAL = "signal"


class DataSource(Enum):
    """
    Logical data source identifier.

    Maps to physical adapters via data_sources.yaml configuration.
    Allows environment-specific configuration (dev/prod).
    """

    ALGOSEEK = "algoseek"
    SCHWAB = "schwab"
    CSV_FILE = "csv_file"


class Instrument(NamedTuple):
    """
    Logical instrument specification.

    Represents a tradable instrument independent of how/where data is stored.
    The DataSourceResolver maps this logical instrument to a physical adapter.

    Examples:
        >>> # Single equity from Algoseek
        >>> apple = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)

        >>> # Crypto with custom frequency
        >>> btc = Instrument(
        ...     "BTCUSD",
        ...     InstrumentType.CRYPTO,
        ...     DataSource.BINANCE,
        ...     frequency="1m"
        ... )

        >>> # Signal with metadata
        >>> sentiment = Instrument(
        ...     "NEWS_SENTIMENT",
        ...     InstrumentType.SIGNAL,
        ...     DataSource.DATABASE,
        ...     metadata={"provider": "RavenPack", "lag_days": 1}
        ... )
    """

    symbol: str
    instrument_type: InstrumentType
    data_source: DataSource
    frequency: Optional[str] = None  # e.g., "1d", "1m" (None = use global default)
    metadata: Dict[str, Any] = {}  # Custom attributes (provider, lag, etc.)

    def __repr__(self) -> str:
        """Human-readable representation."""
        freq = f"@{self.frequency}" if self.frequency else ""
        return f"Instrument({self.symbol}{freq}, {self.instrument_type.value}, {self.data_source.value})"
