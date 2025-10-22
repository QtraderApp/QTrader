"""
Algoseek vendor-specific bar model.

This module defines the Algoseek-specific bar structure with vendor fields
for corporate action tracking and adjustment factors.
"""

import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from qtrader.contracts.data import Bar, PriceSeries


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
        - This is vendor-specific; use adapters to convert to Bar
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

    def get_dividend_amount(self, previous_close: float) -> Optional[Decimal]:
        """
        Extract dividend amount if AdjustmentReason indicates a dividend.

        For dividends, Algoseek's AdjustmentFactor represents a price adjustment ratio,
        not a dollar amount. The dividend must be calculated from the previous close:

            Dividend = (1 - AdjustmentFactor) × Close[T-1]

        Where:
            - AdjustmentFactor appears on ex-dividend date (T)
            - Close[T-1] is the previous trading day's closing price
            - Dividend is paid to shareholders holding at end of T-1

        Args:
            previous_close: The closing price from the previous trading day

        Returns:
            Decimal dividend amount per share in dollars, or None if no dividend

        Example:
            AAPL 2020-08-07:
            - AdjustmentFactor = 0.998200215
            - Previous Close (Aug 6) = $455.61
            - Dividend = (1 - 0.998200215) × 455.61 = $0.82
        """
        if self.is_dividend() and self.AdjustmentFactor:
            # Calculate dividend from adjustment factor and previous close
            adjustment_factor = Decimal(str(self.AdjustmentFactor))
            prev_close = Decimal(str(previous_close))
            dividend = (Decimal("1") - adjustment_factor) * prev_close
            return dividend.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return None

    def get_split_ratio(self) -> Optional[Decimal]:
        """
        Extract split ratio if AdjustmentReason indicates a split.

        For splits, Algoseek's AdjustmentFactor is the INVERSE of the split ratio:
            - 0.25 for a 4:1 forward split (1/4 = 0.25)
            - 5.0 for a 1:5 reverse split (1/0.2 = 5.0)

        Therefore: split_ratio = 1 / AdjustmentFactor

        Returns:
            Decimal split ratio (e.g., 4.00 for 4:1 split), or None if no split
        """
        if self.is_split() and self.AdjustmentFactor:
            # AdjustmentFactor is the inverse of the split ratio
            split_ratio = Decimal("1") / Decimal(str(self.AdjustmentFactor))
            return split_ratio.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return None


class AlgoseekPriceSeries(BaseModel):
    """
    Algoseek OHLCV Price Series - vendor specific.

    Holds raw unadjusted bars from Algoseek database.
    Provides method to compute all 3 canonical series (unadjusted, adjusted, total_return).

    The transformation logic:
    1. Unadjusted: Direct conversion (raw prices)
    2. Adjusted: Backward adjustment using CumulativeVolumeFactor
       - Prices divided by split ratio
       - Volume multiplied by split ratio
       - Reference point: last bar (most recent split state)
    3. Total Return: Forward compounding (no look-ahead)
       - TR_t = TR_{t-1} * (UnAdj_t * SplitRatio_t + Div_t) / UnAdj_{t-1}
       - Simulates buying 1 share at t=0 with dividend reinvestment
       - Volume in starting-date units (÷ cumulative_split_ratio)

    Note: Computing adjusted series requires the complete time series because
    backward adjustment must start from the last bar. This introduces some
    "future bias" in the sense that you need to know all future corporate events
    to correctly adjust historical prices. This is inherent to the adjustment process.

    Attributes:
        symbol: Ticker symbol
        bars: List of raw Algoseek bars (chronologically ordered)
    """

    symbol: str = Field(..., description="Ticker symbol")
    bars: list[AlgoseekBar] = Field(..., description="List of raw Algoseek bars")

    def to_canonical_series(self) -> dict[str, PriceSeries]:
        """
        Compute all 3 canonical price series from raw Algoseek bars.

        Process:
        1. Unadjusted: Direct conversion (raw prices)
        2. Adjusted (split-adjusted): Backward adjustment using CumulativeVolumeFactor
        3. Total Return: Forward compounding with dividend reinvestment

        Returns:
            Dictionary with keys: 'unadjusted', 'adjusted', 'total_return'
            Each value is a PriceSeries object
        """
        if not self.bars:
            return {
                "unadjusted": PriceSeries(mode="unadjusted", symbol=self.symbol, bars=[]),
                "adjusted": PriceSeries(mode="adjusted", symbol=self.symbol, bars=[]),
                "total_return": PriceSeries(mode="total_return", symbol=self.symbol, bars=[]),
            }

        # Get the last bar's cumulative factors (reference point for backward adjustment)
        last_bar = self.bars[-1]
        last_volume_factor = Decimal(str(last_bar.CumulativeVolumeFactor))

        unadjusted_bars = []
        adjusted_bars = []
        total_return_bars = []

        # For Total Return: forward compounding (no look-ahead)
        # TR_0 = UnAdj_0, then TR_t = TR_{t-1} * (UnAdj_t * SplitRatio_t + Div_t) / UnAdj_{t-1}
        prev_unadj_close: Optional[Decimal] = None
        prev_tr_close: Optional[Decimal] = None
        cumulative_split_ratio = Decimal("1.0")  # Track cumulative splits for volume adjustment
        prev_bar: Optional[AlgoseekBar] = None  # Track previous bar for dividend calculation

        for bar in self.bars:
            trade_datetime = bar.TradeDate  # Keep as datetime object

            # Calculate dividend amount (requires previous bar's close)
            dividend_amount: Optional[Decimal] = None
            if bar.is_dividend() and prev_bar is not None:
                dividend_amount = bar.get_dividend_amount(prev_bar.Close)

            # 1. Unadjusted (raw prices, no adjustment)
            unadj_bar = Bar(
                trade_datetime=trade_datetime,
                open=bar.Open,
                high=bar.High,
                low=bar.Low,
                close=bar.Close,
                volume=bar.MarketHoursVolume,
                dividend=dividend_amount,
            )
            unadjusted_bars.append(unadj_bar)

            # 2. Adjusted (split-adjusted only, backward to post-split basis)
            # If split on this date, apply to all prior dates
            # Formula: adjusted_price = unadjusted_price / split_ratio
            vol_factor_ratio = last_volume_factor / Decimal(str(bar.CumulativeVolumeFactor))

            # Adjust dividend amount for splits (divide by split ratio)
            adjusted_dividend: Optional[Decimal] = None
            if dividend_amount is not None:
                adjusted_dividend = dividend_amount / vol_factor_ratio

            adjusted_bar = Bar(
                trade_datetime=trade_datetime,
                open=float(Decimal(str(bar.Open)) / vol_factor_ratio),
                high=float(Decimal(str(bar.High)) / vol_factor_ratio),
                low=float(Decimal(str(bar.Low)) / vol_factor_ratio),
                close=float(Decimal(str(bar.Close)) / vol_factor_ratio),
                volume=int(Decimal(str(bar.MarketHoursVolume)) * vol_factor_ratio),
                dividend=adjusted_dividend,
            )
            adjusted_bars.append(adjusted_bar)

            # 3. Total Return (forward compounding, no look-ahead)
            # TR_t = TR_{t-1} * (UnAdj_t * SplitRatio_t + Div_t) / UnAdj_{t-1}

            # Get split ratio for this bar (if any) and accumulate for volume
            split_ratio = bar.get_split_ratio()
            if split_ratio is None:
                split_ratio = Decimal("1.0")
            cumulative_split_ratio *= split_ratio

            if prev_tr_close is None:
                # First bar: TR_0 = UnAdj_0
                tr_close = Decimal(str(bar.Close))
            else:
                # Use the dividend we already calculated
                dividend = dividend_amount if dividend_amount is not None else Decimal("0")

                # TR_t = TR_{t-1} * (UnAdj_t * SplitRatio_t + Div_t) / UnAdj_{t-1}
                unadj_close = Decimal(str(bar.Close))
                numerator = unadj_close * split_ratio + dividend
                if prev_unadj_close is not None:
                    tr_close = prev_tr_close * numerator / prev_unadj_close
                else:
                    tr_close = Decimal(str(bar.Close))

            # For OHLC in total return, we need to scale by the same ratio as close
            # (This assumes intraday there are no corporate events, which is reasonable)
            if prev_tr_close is None:
                tr_ratio = Decimal("1.0")
            else:
                tr_ratio = tr_close / Decimal(str(bar.Close))

            total_return_bar = Bar(
                trade_datetime=trade_datetime,
                open=float(Decimal(str(bar.Open)) * tr_ratio),
                high=float(Decimal(str(bar.High)) * tr_ratio),
                low=float(Decimal(str(bar.Low)) * tr_ratio),
                close=float(tr_close),
                volume=int(
                    Decimal(str(bar.MarketHoursVolume)) / cumulative_split_ratio
                ),  # Volume in starting-date units
                dividend=None,  # Dividends embedded in TR prices
            )
            total_return_bars.append(total_return_bar)

            # Update for next iteration
            prev_unadj_close = Decimal(str(bar.Close))
            prev_tr_close = tr_close
            prev_bar = bar  # Track previous bar for dividend calculation

        return {
            "unadjusted": PriceSeries(mode="unadjusted", symbol=self.symbol, bars=unadjusted_bars),
            "adjusted": PriceSeries(mode="adjusted", symbol=self.symbol, bars=adjusted_bars),
            "total_return": PriceSeries(mode="total_return", symbol=self.symbol, bars=total_return_bars),
        }
