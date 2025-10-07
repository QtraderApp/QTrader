"""Core data model for OHLCV bars and adjustment metadata."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, NamedTuple, Optional


class PriceSeries(NamedTuple):
    """
    OHLCV data for a specific adjustment mode.

    Each bar provides multiple price series (unadjusted, capital_adjusted, total_return).
    Volume is included per series because adjustments affect volume differently.

    Attributes:
        open: Opening price (Decimal for precision)
        high: High price (Decimal for precision)
        low: Low price (Decimal for precision)
        close: Closing price (Decimal for precision)
        volume: Volume in shares/contracts (adjusted for this series)
    """

    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


class Dividend(NamedTuple):
    """
    Cash dividend event on a bar.

    Cash ledger watches this attribute and automatically credits long positions
    or debits short positions on ex-date.

    Attributes:
        ex_date: Ex-dividend date (when stock trades without dividend)
        amount_per_share: Dividend amount per share (Decimal for precision)
        payment_date: Optional payment date (when cash is distributed)
    """

    ex_date: datetime
    amount_per_share: Decimal
    payment_date: Optional[datetime] = None


class Split(NamedTuple):
    """
    Stock split or reverse split event on a bar.

    Attributes:
        ex_date: Effective date of the split
        ratio: Split ratio (e.g., 0.25 for 4:1 split, 2.0 for 1:2 reverse split)
        from_factor: Original shares (e.g., 1 in "1-for-4" split)
        to_factor: New shares (e.g., 4 in "1-for-4" split)
    """

    ex_date: datetime
    ratio: Decimal
    from_factor: int = 1
    to_factor: int = 1


class Bar(NamedTuple):
    """
    Canonical bar with multiple price series.

    This is vendor-agnostic, asset-agnostic, and frequency-agnostic.
    Works with equities, futures, crypto, forex at any timeframe.

    Each bar contains three price series for different use cases:
    - unadjusted: Raw execution prices (for realistic fills, participation monitoring)
    - capital_adjusted: Split-adjusted only (standard backtesting)
    - total_return: Split + dividend adjusted (benchmarking, performance attribution)

    Adapters are responsible for transforming vendor data into all three series.
    Downstream components select which series to use via configuration.

    Attributes:
        ts: Timezone-aware timestamp of the bar
        symbol: Ticker/contract identifier (e.g., "AAPL", "ESH24")
        unadjusted: Raw OHLCV prices and actual volume
        capital_adjusted: Split-adjusted OHLCV (dividends not embedded)
        total_return: Split + dividend adjusted OHLCV (fully adjusted)
        dividend: Optional dividend event on this bar (if ex-date falls on this bar)
        split: Optional split event on this bar (if ex-date falls on this bar)
    """

    ts: datetime
    symbol: str
    unadjusted: PriceSeries
    capital_adjusted: PriceSeries
    total_return: PriceSeries
    dividend: Optional[Dividend] = None
    split: Optional[Split] = None


class AdjustmentEvent(NamedTuple):
    """
    Corporate action metadata for analysis and validation.

    NOT used by execution engine. Stored for:
    - Audit trail (data provenance)
    - Performance attribution (dividend-adjusted returns)
    - Data validation (detect missing adjustments)

    Attributes:
        ts: Event timestamp (ex-date for dividends)
        symbol: Ticker symbol
        event_type: Type of corporate action (CashDiv, Split, StockDiv, SpinOff)
        px_factor: Cumulative price adjustment factor
        vol_factor: Cumulative volume adjustment factor
        metadata: Vendor-specific details (amount, ratio, etc.)
    """

    ts: datetime
    symbol: str
    event_type: str
    px_factor: Decimal
    vol_factor: Decimal
    metadata: dict[str, Any]


class BarFrequency(Enum):
    """Supported bar frequencies for backtesting."""

    MIN_1 = "1m"
    MIN_5 = "5m"
    MIN_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"


class DataMode(Enum):
    """
    DEPRECATED: Data adjustment mode.

    No longer needed - Bar now contains all three adjustment modes.
    Kept for backward compatibility with existing code.
    Will be removed in future version.

    Use price series selection in configuration instead:
    - execution.price_series: "unadjusted"
    - portfolio.price_series: "capital_adjusted"
    - performance.price_series: "total_return"
    """

    ADJUSTED = "adjusted"
    UNADJUSTED = "unadjusted"
    SPLIT_ADJUSTED = "split_adjusted"


class OHLCPolicy(Enum):
    """
    Policies for handling malformed OHLC bars.

    STRICT_RAISE: Raise error on first malformed bar (fail fast)
    WARN_SKIP_BAR: Log warning and skip the bar entirely (no fills, orders remain pending)
    WARN_USE_CLOSE_ONLY: Log warning, allow bar but disable limit/stop evaluation (close-only mode)
    """

    STRICT_RAISE = "strict_raise"
    WARN_SKIP_BAR = "warn_skip_bar"
    WARN_USE_CLOSE_ONLY = "warn_use_close_only"
