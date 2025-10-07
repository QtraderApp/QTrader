"""
Algoseek vendor-specific bar model.

This module defines the Algoseek-specific bar structure with vendor fields
for corporate action tracking and adjustment factors.
"""

import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from pydantic import BaseModel, field_validator, model_validator


class AlgoseekBar(BaseModel):
    """
    Algoseek Unadjusted OHLC Bar schema - vendor specific.

    This represents raw, unadjusted bars as stored in Algoseek database.
    Contains vendor-specific fields for adjustment tracking.

    Algoseek stores:
    - UNADJUSTED prices (raw as-traded)
    - Cumulative adjustment factors (for both price and volume)
    - Adjustment events (splits, dividends) with metadata

    Key Fields:
        TradeDate: Trading date (parsed from DuckDB Timestamp or string)
        Ticker: Stock symbol
        Open/High/Low/Close: Unadjusted OHLC prices
        MarketHoursVolume: Actual volume traded
        CumulativePriceFactor: Cumulative factor for price+dividend adjustments
        CumulativeVolumeFactor: Cumulative factor for split adjustments only
        AdjustmentFactor: Adjustment on this date (if any)
        AdjustmentReason: Type of adjustment (CashDiv, Subdiv, etc.)

    Corporate Event Detection:
        - is_split(): Returns True if this bar has a split event
        - is_dividend(): Returns True if this bar has a dividend event
        - get_split_ratio(): Extracts split ratio (e.g., 4.0 for 4:1 split)
        - get_dividend_amount(): Extracts dividend amount per share

    Notes:
        - Includes OHLC validation with 10% tolerance for adjustment artifacts
        - TradeDate is auto-converted from DuckDB Timestamp to string
        - This is vendor-specific; use adapters to convert to CanonicalBar
    """

    TradeDate: datetime.datetime
    Ticker: str
    Open: float
    High: float
    Low: float
    Close: float
    MarketHoursVolume: int
    CumulativePriceFactor: float
    CumulativeVolumeFactor: float
    AdjustmentFactor: Optional[float] = None
    AdjustmentReason: Optional[str] = None

    @field_validator("TradeDate", mode="before")
    @classmethod
    def parse_trade_date(cls, v):
        """
        Parse TradeDate from DuckDB Timestamp or string.

        DuckDB returns Timestamp objects, not strings. This validator
        handles both cases and converts to datetime.datetime.

        Args:
            v: Timestamp object or string

        Returns:
            datetime.datetime object
        """
        if hasattr(v, "date"):
            # DuckDB Timestamp object
            return datetime.datetime.combine(v.date(), datetime.time())
        elif isinstance(v, str):
            # String format
            return datetime.datetime.fromisoformat(v)
        elif isinstance(v, datetime.datetime):
            # Already datetime
            return v
        else:
            raise ValueError(f"Cannot parse TradeDate from type {type(v)}: {v}")

    @model_validator(mode="after")
    def validate_ohlc(self) -> "AlgoseekBar":
        """
        Validate OHLC relationships with tolerance for minor violations.

        Returns warnings but allows continuation if violations are not severe.
        Severe violations (High < Low) will raise an error.

        Tolerance is needed because Algoseek adjustment calculations can
        create minor OHLC violations due to rounding.

        Returns:
            Self if validation passes

        Raises:
            ValueError: If High < Low (severe violation)
        """
        warnings = []
        tolerance = 0.10  # 10% tolerance for Algoseek adjustment artifacts
        tolerance_multiplier = 1 + tolerance

        # Severe: High must be >= Low (no tolerance)
        if self.High < self.Low:
            raise ValueError(
                f"[{self.TradeDate}] SEVERE: High ({self.High}) < Low ({self.Low}). This is a data integrity issue."
            )

        # Minor violations (log but allow)
        if self.High < self.Open / tolerance_multiplier:
            warnings.append(f"High ({self.High}) < Open ({self.Open}) [exceeds {tolerance * 100}% tolerance]")

        if self.High < self.Close / tolerance_multiplier:
            warnings.append(f"High ({self.High}) < Close ({self.Close}) [exceeds {tolerance * 100}% tolerance]")

        if self.Low > self.Open * tolerance_multiplier:
            warnings.append(f"Low ({self.Low}) > Open ({self.Open}) [exceeds {tolerance * 100}% tolerance]")

        if self.Low > self.Close * tolerance_multiplier:
            warnings.append(f"Low ({self.Low}) > Close ({self.Close}) [exceeds {tolerance * 100}% tolerance]")

        if warnings:
            print(f"⚠️  [{self.TradeDate}] {self.Ticker} OHLC warnings:")
            for warning in warnings:
                print(f"    - {warning}")

        return self

    def is_dividend(self) -> bool:
        """
        Check if this bar represents a dividend event.

        Returns:
            True if AdjustmentReason indicates a dividend
        """
        return self.AdjustmentReason in ("CashDiv", "ScriptDiv", "ScriptDivDiff")

    def is_split(self) -> bool:
        """
        Check if this bar represents a split event.

        Returns:
            True if AdjustmentReason indicates a split
        """
        return self.AdjustmentReason in ("Subdiv", "BonusSame", "ScriptDiv", "Cons")

    def get_dividend_amount(self) -> Optional[Decimal]:
        """
        Extract dividend amount if AdjustmentReason indicates a dividend.

        For dividends, Algoseek's AdjustmentFactor represents:
            new_price = old_price * adjustment_factor

        Therefore:
            dividend = old_price * (1 - adjustment_factor)

        Returns:
            Decimal dividend amount per share, or None if no dividend
        """
        if self.is_dividend() and self.AdjustmentFactor:
            # For dividends: new_price = old_price * adjustment_factor
            # Therefore: dividend = old_price * (1 - adjustment_factor)
            div_pct = Decimal("1") - Decimal(str(self.AdjustmentFactor))
            div_amount = Decimal(str(self.Close)) * div_pct
            return div_amount.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return None

    def get_split_ratio(self) -> Optional[Decimal]:
        """
        Extract split ratio if AdjustmentReason indicates a split.

        For splits, Algoseek's AdjustmentFactor is the inverse ratio:
            e.g., 0.25 for a 4:1 split

        Therefore:
            split_ratio = 1 / adjustment_factor

        Returns:
            Decimal split ratio (e.g., 4.00 for 4:1 split), or None if no split
        """
        if self.is_split() and self.AdjustmentFactor:
            # For splits: adjustment_factor is the inverse ratio
            # e.g., 0.25 for a 4:1 split
            split_ratio = Decimal("1") / Decimal(str(self.AdjustmentFactor))
            return split_ratio.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return None
