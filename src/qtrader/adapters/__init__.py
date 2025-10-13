"""Data adapters for normalizing vendor data to canonical Bar."""

from qtrader.adapters.algoseek import AlgoseekOHLCVendorAdapter
from qtrader.adapters.base import DataAdapter
from qtrader.adapters.csv_adapter import CSVAdapter

__all__ = [
    "DataAdapter",
    "AlgoseekOHLCVendorAdapter",  # New simplified OHLC adapter
    "CSVAdapter",  # Legacy - to be migrated
]
