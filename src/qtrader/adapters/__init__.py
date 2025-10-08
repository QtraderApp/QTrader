"""Data adapters for normalizing vendor data to canonical Bar."""

from qtrader.adapters.algoseek import AlgoseekOHLCVendorAdapter
from qtrader.adapters.algoseek_legacy import AlgoseekOHLCAdapterLegacy
from qtrader.adapters.base import DataAdapter
from qtrader.adapters.csv_adapter import CSVAdapter

__all__ = [
    "DataAdapter",
    "AlgoseekOHLCAdapterLegacy",  # Legacy - will be removed in Phase 9
    "AlgoseekOHLCVendorAdapter",  # New simplified OHLC adapter
    "CSVAdapter",
]
