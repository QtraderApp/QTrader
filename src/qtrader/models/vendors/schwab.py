"""
Schwab vendor-specific bar model.

This module defines the Schwab-specific bar structure for price history data.
Schwab API returns split-adjusted prices only (no unadjusted or total return data).
"""

import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from qtrader.models.bar import Bar, PriceSeries


class SchwabBar(BaseModel):
    """
    Schwab OHLC Bar schema - vendor specific.

    This represents split-adjusted bars from Schwab's Price History API.
    Schwab returns only adjusted prices (split-adjusted, but NOT dividend-adjusted).

    Data Characteristics:
    - ADJUSTED prices (split-adjusted only)
    - NO unadjusted prices available
    - NO dividend data available
    - NO total return data available
    - Intraday and daily data supported

    API Response Format:
        {
            "candles": [
                {
                    "datetime": 1609459200000,  # Unix timestamp in milliseconds
                    "open": 132.43,
                    "high": 133.61,
                    "low": 131.72,
                    "close": 132.05,
                    "volume": 143301900
                },
                ...
            ],
            "symbol": "AAPL",
            "empty": false
        }

    Key Fields:
        timestamp: Trading timestamp (Unix milliseconds, converted to datetime)
        open: Split-adjusted opening price
        high: Split-adjusted high price
        low: Split-adjusted low price
        close: Split-adjusted closing price
        volume: Split-adjusted trading volume

    Notes:
        - Includes OHLC validation with 5% tolerance
        - Timestamp auto-converted from Unix milliseconds to datetime
        - Volume is optional (may be 0 for some periods)
        - This is vendor-specific; use adapters to convert to Bar
    """

    timestamp: Any = Field(..., description="Trading timestamp (accepts datetime, Unix ms, or ISO string)")
    open: float = Field(..., description="Split-adjusted opening price")
    high: float = Field(..., description="Split-adjusted high price")
    low: float = Field(..., description="Split-adjusted low price")
    close: float = Field(..., description="Split-adjusted closing price")
    volume: int = Field(default=0, description="Split-adjusted trading volume")

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime.datetime:
        """
        Parse timestamp from Unix timestamp (milliseconds) or ISO string.

        Schwab API returns Unix timestamps in milliseconds.
        This validator handles conversion to datetime.datetime.

        Args:
            v: Unix timestamp (int/float in milliseconds), ISO string, or datetime

        Returns:
            datetime.datetime object

        Raises:
            ValueError: If input cannot be parsed
        """
        if isinstance(v, datetime.datetime):
            # Already datetime
            return v
        elif isinstance(v, (int, float)):
            # Unix timestamp in milliseconds (Schwab API format)
            return datetime.datetime.fromtimestamp(v / 1000.0, tz=datetime.timezone.utc)
        elif isinstance(v, str):
            # ISO string format
            return datetime.datetime.fromisoformat(v)
        else:
            raise ValueError(f"Cannot parse datetime from type {type(v)}: {v}")

    @model_validator(mode="after")
    def validate_ohlc(self) -> "SchwabBar":
        """
        Validate OHLC relationships with tolerance for minor violations.

        Returns warnings but allows continuation if violations are not severe.
        Severe violations (High < Low) will raise an error.

        Tolerance is needed because API data can have minor inconsistencies.

        Returns:
            Self if validation passes

        Raises:
            ValueError: If High < Low (severe violation)
        """
        warnings = []
        tolerance = 0.05  # 5% tolerance for minor violations
        tolerance_multiplier = 1 + tolerance

        # Severe: High must be >= Low (no tolerance)
        if self.high < self.low:
            raise ValueError(
                f"[{self.timestamp}] SEVERE: High ({self.high}) < Low ({self.low}). This is a data integrity issue."
            )

        # Minor violations (log but allow)
        if self.high < self.open / tolerance_multiplier:
            warnings.append(f"High ({self.high}) < Open ({self.open}) [exceeds {tolerance * 100}% tolerance]")

        if self.high < self.close / tolerance_multiplier:
            warnings.append(f"High ({self.high}) < Close ({self.close}) [exceeds {tolerance * 100}% tolerance]")

        if self.low > self.open * tolerance_multiplier:
            warnings.append(f"Low ({self.low}) > Open ({self.open}) [exceeds {tolerance * 100}% tolerance]")

        if self.low > self.close * tolerance_multiplier:
            warnings.append(f"Low ({self.low}) > Close ({self.close}) [exceeds {tolerance * 100}% tolerance]")

        if warnings:
            print(f"⚠️  [{self.timestamp}] OHLC warnings:")
            for warning in warnings:
                print(f"    - {warning}")

        return self


class SchwabPriceSeries(BaseModel):
    """
    Schwab OHLCV Price Series - vendor specific.

    Holds split-adjusted bars from Schwab Price History API.
    Schwab only provides adjusted prices (split-adjusted, not dividend-adjusted).

    Data Limitations:
    - Only ADJUSTED series available (split-adjusted)
    - NO unadjusted data available
    - NO dividend data available
    - NO total return series available

    The to_canonical_series() method returns:
    1. adjusted: Split-adjusted prices from Schwab API
    2. unadjusted: None (not available)
    3. total_return: None (not available)

    This creates a "partial MultiBar" where only adjusted data is populated.

    Attributes:
        symbol: Ticker symbol
        bars: List of Schwab bars (chronologically ordered)
    """

    symbol: str = Field(..., description="Ticker symbol")
    bars: list[SchwabBar] = Field(..., description="List of split-adjusted Schwab bars")

    def to_canonical_series(
        self,
    ) -> dict[str, Optional[PriceSeries]]:
        """
        Convert Schwab bars to canonical price series.

        Schwab only provides split-adjusted prices, so:
        - adjusted: Available (direct conversion)
        - unadjusted: None (not provided by Schwab)
        - total_return: None (cannot compute without dividends)

        This creates a "partial MultiBar" structure where only adjusted
        data is available.

        Returns:
            Dictionary with keys: 'unadjusted', 'adjusted', 'total_return'
            - 'adjusted': PriceSeries with split-adjusted data
            - 'unadjusted': None
            - 'total_return': None
        """
        if not self.bars:
            return {
                "unadjusted": None,
                "adjusted": PriceSeries(mode="adjusted", symbol=self.symbol, bars=[]),
                "total_return": None,
            }

        # Convert Schwab bars to canonical Bar objects
        adjusted_bars = []
        for bar in self.bars:
            canonical_bar = Bar(
                trade_datetime=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                dividend=None,  # Schwab doesn't provide dividend data in price history
            )
            adjusted_bars.append(canonical_bar)

        return {
            "unadjusted": None,
            "adjusted": PriceSeries(mode="adjusted", symbol=self.symbol, bars=adjusted_bars),
            "total_return": None,
        }
