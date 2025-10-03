"""Data adapters for normalizing vendor data to canonical Bar."""

from qtrader.adapters.algoseek_parquet import AlgoseekParquetAdapter
from qtrader.adapters.base import DataAdapter
from qtrader.adapters.csv_adapter import CSVAdapter

__all__ = [
    "DataAdapter",
    "AlgoseekParquetAdapter",
    "CSVAdapter",
]
