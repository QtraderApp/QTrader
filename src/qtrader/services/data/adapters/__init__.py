"""Data adapters for normalizing vendor data to canonical Bar."""

from qtrader.services.data.adapters.algoseek import AlgoseekOHLCVendorAdapter

# Re-export models for convenience
from qtrader.services.data.adapters.models.algoseek import AlgoseekBar, AlgoseekPriceSeries
from qtrader.services.data.adapters.models.schwab import SchwabBar, SchwabPriceSeries
from qtrader.services.data.adapters.schwab import SchwabOHLCAdapter

__all__ = [
    "AlgoseekOHLCVendorAdapter",
    "SchwabOHLCAdapter",
    "AlgoseekBar",
    "AlgoseekPriceSeries",
    "SchwabBar",
    "SchwabPriceSeries",
]
