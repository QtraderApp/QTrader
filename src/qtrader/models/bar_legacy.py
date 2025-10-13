"""
LEGACY: Old Bar model with embedded PriceSeries.

This module contains the legacy Bar model that has been replaced by the new
CanonicalBar + MultiModeBar architecture. It is kept temporarily for:
- CSVAdapter (to be migrated)
- BaseAdapter protocol (to be updated)

DO NOT USE IN NEW CODE. Use CanonicalBar and MultiModeBar instead.

This file will be deleted after CSVAdapter migration is complete.
"""

from datetime import datetime
from decimal import Decimal
from typing import NamedTuple, Optional


class PriceSeries(NamedTuple):
    """
    LEGACY: OHLCV data for a specific adjustment mode.

    Replaced by CanonicalBar in new architecture.
    """

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class Dividend(NamedTuple):
    """
    LEGACY: Cash dividend event on a bar.

    Replaced by dividend fields in CanonicalBar.
    """

    ex_date: datetime
    amount_per_share: Decimal
    payment_date: Optional[datetime] = None


class Split(NamedTuple):
    """
    LEGACY: Stock split or reverse split event on a bar.

    Replaced by split fields in CanonicalBar.
    """

    ex_date: datetime
    ratio: Decimal
    from_factor: int = 1
    to_factor: int = 1


class Bar(NamedTuple):
    """
    LEGACY: Canonical bar with multiple price series.

    Replaced by MultiModeBar (which contains CanonicalBars for each mode).

    DO NOT USE IN NEW CODE.
    """

    ts: datetime
    symbol: str
    unadjusted: PriceSeries
    capital_adjusted: PriceSeries
    total_return: PriceSeries
    dividend: Optional[Dividend] = None
    split: Optional[Split] = None
