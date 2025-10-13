"""
Canonical bar model - vendor-agnostic OHLCV data representation.

This module defines the canonical (standardized) bar model used throughout QTrader.
All vendor-specific data must be transformed into this format by vendor adapters.

The canonical model supports three adjustment modes:
- unadjusted: Raw prices as traded
- adjusted: Split-adjusted only (capital changes)
- total_return: Split + dividend adjusted (investment simulation)
"""

from decimal import Decimal
from typing import ClassVar, Optional

from pydantic import BaseModel, Field, model_validator


class Bar(BaseModel):
    """
    Canonical OHLC Bar - vendor agnostic.

    This represents a single bar in one of the three adjustment modes:
    - unadjusted: raw prices as traded
    - adjusted: split-adjusted only (capital adjusted)
    - total_return: split + dividend adjusted

    The dividend field contains the split-adjusted dividend amount per share.
    For example, if a stock paid $1 dividend before a 2:1 split, the historical
    bar would show dividend=$0.50 (adjusted to current split terms).

    Notes:
        - All prices must be positive
        - High >= Low (strictly enforced)
        - Volume >= 0
        - Dividend only present on ex-date
        - This is immutable (frozen after creation)
    """

    # Instance fields
    trade_datetime: str = Field(..., description="Trade datetime (ISO format)")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: int = Field(..., ge=0, description="Volume")
    dividend: Optional[Decimal] = Field(
        default=None, ge=0, description="Split-adjusted dividend amount per share (if any)"
    )

    model_config = {"frozen": True}  # Make immutable

    @model_validator(mode="after")
    def validate_ohlc(self) -> "Bar":
        """
        Validate OHLC relationships.

        Enforces: High >= Low (strict)

        Note: We don't enforce High >= Open/Close or Low <= Open/Close
        because vendor data may have minor violations due to adjustment artifacts.
        These should be caught by vendor-specific validators.

        Raises:
            ValueError: If High < Low
        """
        if self.high < self.low:
            raise ValueError(f"[{self.trade_datetime}] OHLC violation: High ({self.high}) < Low ({self.low})")
        return self


class PriceSeries(BaseModel):
    """
    Canonical OHLCV Price Series for a specific adjustment mode.

    This is vendor-agnostic and represents the final, validated time series
    that the backtester consumes.

    Each series has a mode that indicates the adjustment methodology:
    - unadjusted: Raw prices, no adjustments
    - adjusted: Split-adjusted (backward to post-split basis)
    - total_return: Forward compounding with dividend reinvestment

    Attributes:
        mode: Adjustment mode identifier
        symbol: Ticker symbol
        bars: List of canonical bars (chronologically ordered)

    Notes:
        - Bars should be in chronological order (oldest first)
        - All bars must have same symbol
        - Mode must be one of the valid modes
    """

    # Class variable for valid modes
    VALID_MODES: ClassVar[set[str]] = {"unadjusted", "adjusted", "total_return"}

    # Instance fields
    mode: str = Field(..., description="Adjustment mode (unadjusted|adjusted|total_return)")
    symbol: str = Field(..., description="Ticker symbol")
    bars: list[Bar] = Field(..., description="List of canonical bars")

    model_config = {"frozen": True}  # Make immutable

    @model_validator(mode="after")
    def validate_mode(self) -> "PriceSeries":
        """
        Validate that mode is one of the valid modes.

        Raises:
            ValueError: If mode is not in VALID_MODES
        """
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {self.VALID_MODES}")
        return self

    @model_validator(mode="after")
    def validate_bars(self) -> "PriceSeries":
        """
        Validate bars consistency.

        Note: We don't enforce chronological order here to allow flexibility
        in how bars are loaded. Consumers should sort if needed.
        """
        if not self.bars:
            return self

        # Could add more validations here if needed:
        # - Check chronological order
        # - Check for gaps
        # - Check for duplicates

        return self
