"""Core data model for OHLCV bars and adjustment metadata."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, NamedTuple


class Bar(NamedTuple):
    """
    Canonical OHLCV bar - the ONLY contract consumed by execution engine.

    This is vendor-agnostic, asset-agnostic, and frequency-agnostic.
    Works with equities, futures, crypto, forex at any timeframe.

    All vendor data is normalized to this format at the adapter boundary.
    Vendor-specific fields (adjustments, bid/ask, etc.) are stored separately.

    Attributes:
        ts: Timezone-aware timestamp of the bar
        symbol: Ticker/contract identifier (e.g., "AAPL", "ESH24")
        open: Opening price (Decimal for precision)
        high: High price (Decimal for precision)
        low: Low price (Decimal for precision)
        close: Closing price (Decimal for precision)
        volume: Volume in shares/contracts (integer)
    """

    ts: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


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
    Data adjustment mode - declares how prices in Bar are adjusted.

    ADJUSTED: Total-return adjusted (dividends + splits embedded in OHLCV)
    UNADJUSTED: Raw trade prices (no adjustments applied)
    SPLIT_ADJUSTED: Only splits adjusted, dividends not embedded
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
