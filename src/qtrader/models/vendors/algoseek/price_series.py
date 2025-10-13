"""
Algoseek vendor-specific price series model.

This module contains the AlgoseekPriceSeries class which holds raw unadjusted
bars and provides transformation to canonical series.
"""

from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field

from qtrader.models.bar import Bar, PriceSeries

from .bar import AlgoseekBar


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
            trade_date = bar.TradeDate.date().isoformat()

            # Calculate dividend amount (requires previous bar's close)
            dividend_amount: Optional[Decimal] = None
            if bar.is_dividend() and prev_bar is not None:
                dividend_amount = bar.get_dividend_amount(prev_bar.Close)

            # 1. Unadjusted (raw prices, no adjustment)
            unadj_bar = Bar(
                trade_datetime=trade_date,
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
            prev_bar = bar  # Track previous bar for dividend calculation

        return {
            "unadjusted": PriceSeries(mode="unadjusted", symbol=self.symbol, bars=unadjusted_bars),
            "adjusted": PriceSeries(mode="adjusted", symbol=self.symbol, bars=adjusted_bars),
            "total_return": PriceSeries(mode="total_return", symbol=self.symbol, bars=total_return_bars),
        }
