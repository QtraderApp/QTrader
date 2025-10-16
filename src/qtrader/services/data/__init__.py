"""Data service for loading and streaming price data.

This module provides the DataService implementation which coordinates
data loading from vendor adapters and provides clean streaming access
to price data for backtesting and analysis.
"""

from qtrader.services.data.interface import IDataAdapter, IDataService
from qtrader.services.data.service import DataService

__all__ = [
    "IDataService",
    "IDataAdapter",
    "DataService",
]
