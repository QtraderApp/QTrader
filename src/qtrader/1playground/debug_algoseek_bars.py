import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import ClassVar, Optional

import duckdb
from pydantic import BaseModel, Field, model_validator

from qtrader.models.instrument import DataSource, Instrument, InstrumentType

# Path to AAPL data
data_path = Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample-complete/SecId=33449")
parquet_pattern = str(data_path / "*.parquet")

# Connect to DuckDB
con = duckdb.connect(":memory:")

# Query AAPL data around the split in August 2020
query = f"""
SELECT
    TradeDate,
    Ticker,
    Open,
    High,
    Low,
    Close,
    MarketHoursVolume,
    CumulativePriceFactor,
    CumulativeVolumeFactor,
    AdjustmentFactor,
    AdjustmentReason
FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
WHERE TradeDate BETWEEN '2020-08-01' AND '2020-09-03'
ORDER BY TradeDate
"""

instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)


# These are CANONICAL models for bars and OHLC price time series with built-in validation
class CanonicalBar(BaseModel):
    """
    Canonical OHLC Bar - vendor agnostic.

    This represents a single bar in one of the three adjustment modes:
    - unadjusted: raw prices as traded
    - adjusted: split-adjusted only (capital adjusted)
    - total_return: split + dividend adjusted

    The dividend field contains the split-adjusted dividend amount per share.
    For example, if a stock paid $1 dividend before a 2:1 split, the historical
    bar would show dividend=$0.50 (adjusted to current split terms).
    """

    # Instance fields
    trade_datetime: str = Field(..., description="Trade datetime")
    open: float = Field(..., gt=0, description="Open price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Close price")
    volume: int = Field(..., ge=0, description="Volume")
    dividend: Optional[Decimal] = Field(
        default=None, ge=0, description="Split-adjusted dividend amount per share (if any)"
    )


class CanonicalPriceSeries(BaseModel):
    """
    Canonical OHLCV Price Series for a specific adjustment mode.

    This is vendor-agnostic and represents the final, validated time series
    that the backtester consumes.
    """

    # Class variable for valid modes
    VALID_MODES: ClassVar[set[str]] = {"unadjusted", "adjusted", "total_return"}

    # Instance fields
    mode: str = Field(..., description="Adjustment mode (unadjusted|adjusted|total_return)")
    instrument: Instrument = Field(..., description="Instrument details")
    bars: list[CanonicalBar] = Field(..., description="List of canonical bars")

    @model_validator(mode="after")
    def validate_mode(self) -> "CanonicalPriceSeries":
        """Validate that mode is one of the valid modes."""
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {self.VALID_MODES}")
        return self


# These are VENDOR-SPECIFIC models for raw Algoseek bars and price series
class AlgoseekBar(BaseModel):
    """
    Algoseek Unadjusted OHLC Bar schema - vendor specific.

    This represents raw, unadjusted bars as stored in Algoseek database.
    Contains vendor-specific fields for adjustment tracking.
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

    @model_validator(mode="after")
    def validate_ohlc(self) -> "AlgoseekBar":
        """
        Validate OHLC relationships with tolerance for minor violations.

        Returns warnings but allows continuation if violations are not severe.
        Severe violations (High < Low) will raise an error.
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

    def get_dividend_amount(self) -> Optional[Decimal]:
        """Extract dividend amount if AdjustmentReason indicates a dividend."""
        if self.AdjustmentReason in ("CashDiv", "ScriptDiv", "ScriptDivDiff") and self.AdjustmentFactor:
            # For dividends: new_price = old_price * adjustment_factor
            # Therefore: dividend = old_price * (1 - adjustment_factor)
            div_pct = Decimal("1") - Decimal(str(self.AdjustmentFactor))
            div_amount = Decimal(str(self.Close)) * div_pct
            return div_amount.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
        return None

    def get_split_ratio(self) -> Optional[Decimal]:
        """Extract split ratio if AdjustmentReason indicates a split."""
        if self.AdjustmentReason in ("Subdiv", "BonusSame", "ScriptDiv", "Cons") and self.AdjustmentFactor:
            # For splits: adjustment_factor is the inverse ratio
            # e.g., 0.25 for a 4:1 split
            split_ratio = Decimal("1") / Decimal(str(self.AdjustmentFactor))
            return split_ratio.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        return None


class AlgoseekPriceSeries(BaseModel):
    """
    Algoseek OHLCV Price Series - vendor specific.

    Holds raw unadjusted bars from Algoseek database.
    Provides method to compute all 3 canonical series (unadjusted, adjusted, total_return).

    Note: Computing adjusted series requires the complete time series because
    backward adjustment must start from the last bar. This introduces some
    "future bias" in the sense that you need to know all future corporate events
    to correctly adjust historical prices. This is inherent to the adjustment process.
    """

    instrument: Instrument = Field(..., description="Instrument details")
    bars: list[AlgoseekBar] = Field(..., description="List of raw Algoseek bars")

    def to_canonical_series(self) -> dict[str, CanonicalPriceSeries]:
        """
        Compute all 3 canonical price series from raw Algoseek bars.

        Process:
        1. Unadjusted: Direct conversion (raw prices)
        2. Adjusted (split-adjusted): Backward adjustment using CumulativeVolumeFactor
        3. Total Return: Backward adjustment using CumulativePriceFactor

        Returns:
            Dictionary with keys: 'unadjusted', 'adjusted', 'total_return'
        """
        if not self.bars:
            return {
                "unadjusted": CanonicalPriceSeries(mode="unadjusted", instrument=self.instrument, bars=[]),
                "adjusted": CanonicalPriceSeries(mode="adjusted", instrument=self.instrument, bars=[]),
                "total_return": CanonicalPriceSeries(mode="total_return", instrument=self.instrument, bars=[]),
            }

        # Get the last bar's cumulative factors (reference point for backward adjustment)
        last_bar = self.bars[-1]
        last_volume_factor = Decimal(str(last_bar.CumulativeVolumeFactor))

        unadjusted_bars = []
        adjusted_bars = []
        total_return_bars = []

        # For Total Return: forward compounding (no look-ahead)
        # TR_0 = UnAdj_0, then TR_t = TR_{t-1} * (UnAdj_t * SplitRatio_t + Div_t) / UnAdj_{t-1}
        prev_unadj_close = None
        prev_tr_close = None
        cumulative_split_ratio = Decimal("1.0")  # Track cumulative splits for volume adjustment

        for bar in self.bars:
            trade_date = bar.TradeDate.date().isoformat()

            # 1. Unadjusted (raw prices, no adjustment)
            unadj_bar = CanonicalBar(
                trade_datetime=trade_date,
                open=bar.Open,
                high=bar.High,
                low=bar.Low,
                close=bar.Close,
                volume=bar.MarketHoursVolume,
                dividend=bar.get_dividend_amount(),
            )
            unadjusted_bars.append(unadj_bar)

            # 2. Adjusted (split-adjusted only, backward to post-split basis)
            # If split on this date, apply to all prior dates
            # Formula: adjusted_price = unadjusted_price / split_ratio
            vol_factor_ratio = last_volume_factor / Decimal(str(bar.CumulativeVolumeFactor))

            # Adjust dividend amount for splits (divide by split ratio)
            adjusted_dividend = None
            dividend_amount = bar.get_dividend_amount()
            if dividend_amount is not None:
                adjusted_dividend = dividend_amount / vol_factor_ratio

            adjusted_bar = CanonicalBar(
                trade_datetime=trade_date,
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
                # Get dividend for this bar (if any)
                dividend = bar.get_dividend_amount()
                if dividend is None:
                    dividend = Decimal("0")

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

            total_return_bar = CanonicalBar(
                trade_datetime=trade_date,
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

        return {
            "unadjusted": CanonicalPriceSeries(mode="unadjusted", instrument=self.instrument, bars=unadjusted_bars),
            "adjusted": CanonicalPriceSeries(mode="adjusted", instrument=self.instrument, bars=adjusted_bars),
            "total_return": CanonicalPriceSeries(
                mode="total_return", instrument=self.instrument, bars=total_return_bars
            ),
        }


# Execute the query and fetch results
data = con.execute(query).fetchdf()
con.close()

# Step 1: Parse raw vendor bars
print("=" * 100)
print("STEP 1: Parsing raw Algoseek bars...")
print("=" * 100)
vendor_bars = [AlgoseekBar(**row.to_dict()) for index, row in data.iterrows()]
print(f"✅ Fetched {len(vendor_bars)} raw Algoseek bars for {instrument.symbol}")

# Step 2: Create vendor-specific price series
print("\n" + "=" * 100)
print("STEP 2: Creating AlgoseekPriceSeries...")
print("=" * 100)
algoseek_series = AlgoseekPriceSeries(instrument=instrument, bars=vendor_bars)
print(f"✅ Created AlgoseekPriceSeries with {len(algoseek_series.bars)} bars")

# Step 3: Compute all 3 canonical series (requires complete time series for backward adjustment)
print("\n" + "=" * 100)
print("STEP 3: Computing canonical series (unadjusted, adjusted, total_return)...")
print("=" * 100)
canonical_series = algoseek_series.to_canonical_series()

print("\n✅ Canonical Series Generated:")
print(f"   - Unadjusted:    {len(canonical_series['unadjusted'].bars)} bars")
print(f"   - Adjusted:      {len(canonical_series['adjusted'].bars)} bars")
print(f"   - Total Return:  {len(canonical_series['total_return'].bars)} bars")

# Step 4: Show sample data from each series
print("\n" + "=" * 100)
print("STEP 4: Sample data from 2020-08-07 to 2020-09-02 (includes dividend + split)...")
print("=" * 100)

# Show data from 2020-08-07 to 2020-09-02
start_date = "2020-08-07"
end_date = "2020-09-02"

for series_name, series in canonical_series.items():
    print(f"\n{series_name.upper()}:")
    print("-" * 100)

    for bar in series.bars:
        if start_date <= bar.trade_datetime <= end_date:
            print(
                f"  {bar.trade_datetime} | O:{bar.open:8.2f} H:{bar.high:8.2f} L:{bar.low:8.2f} C:{bar.close:8.2f} V:{bar.volume:12,}",
                end="",
            )
            if bar.dividend:
                print(f" | Div: ${bar.dividend:.4f}", end="")
            print()

print("\n" + "=" * 100)
print("✅ Processing complete!")
print("=" * 100)
