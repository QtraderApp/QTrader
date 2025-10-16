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
    Minimal instrument specification.

    Represents a tradable instrument by symbol only. Dataset configuration
    (provider, asset type, etc.) is specified separately via dataset name.

    This design:
    - Eliminates duplication between config and instrument metadata
    - Makes dataset the single source of truth for provider/asset type
    - Puts responsibility on user to provide correct ticker for each dataset
    - Supports complex symbol mappings (e.g., futures: XYT vs XYT1)

    Philosophy:
        User specifies: "Give me bars for symbol AAPL from dataset schwab-us-equity-1d"
        Not: "Give me bars for AAPL (equity, from Schwab)" - that duplicates what dataset already specifies

    Examples:
        >>> # Basic instrument (just symbol)
        >>> instrument = Instrument("AAPL")

        >>> # With custom frequency (overrides dataset default)
        >>> instrument = Instrument("BTCUSD", frequency="1m")

        >>> # With metadata (custom attributes)
        >>> instrument = Instrument(
        ...     "ES_Z24",
        ...     metadata={"contract_month": "2024-12", "exchange": "CME"}
        ... )

    Note:
        Dataset is specified separately when resolving to adapter:
        >>> adapter = resolver.resolve_by_dataset("schwab-us-equity-1d-adjusted", instrument)

        This makes it explicit: dataset config is the source of truth,
        instrument is just the symbol + optional overrides.
    """

    symbol: str
    frequency: Optional[str] = None  # Override dataset default frequency
    metadata: Dict[str, Any] = {}  # Custom attributes (exchange, contract, etc.)

    def __repr__(self) -> str:
        """Human-readable representation."""
        freq = f"@{self.frequency}" if self.frequency else ""
        meta = f" {self.metadata}" if self.metadata else ""
        return f"Instrument({self.symbol}{freq}{meta})"
