"""Core enums and types for OHLCV bars."""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, NamedTuple


class AdjustmentEvent(NamedTuple):
    """Corporate action metadata for analysis and validation.

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
    """DEPRECATED: Data adjustment mode.

    No longer needed - MultiModeBar now contains all three adjustment modes.
    Kept for backward compatibility with legacy adapters (CSV, base protocol).
    Will be removed after adapter migration.

    Use CanonicalBar and MultiModeBar instead:
    - MultiModeBar.unadjusted
    - MultiModeBar.adjusted
    - MultiModeBar.total_return
    """

    ADJUSTED = "adjusted"
    UNADJUSTED = "unadjusted"
    SPLIT_ADJUSTED = "split_adjusted"


class OHLCPolicy(Enum):
    """Policies for handling malformed OHLC bars.

    STRICT_RAISE: Raise error on first malformed bar (fail fast)
    WARN_SKIP_BAR: Log warning and skip the bar entirely (no fills, orders remain pending)
    WARN_USE_CLOSE_ONLY: Log warning, allow bar but disable limit/stop evaluation (close-only mode)
    """

    STRICT_RAISE = "strict_raise"
    WARN_SKIP_BAR = "warn_skip_bar"
    WARN_USE_CLOSE_ONLY = "warn_use_close_only"
