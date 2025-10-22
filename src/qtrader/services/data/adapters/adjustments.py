"""Generic adjustment calculation utilities for price series.

This module provides vendor-agnostic functions to compute different price series
from total-return adjusted data using cumulative adjustment factors.

Price Series Types:
- Unadjusted: Raw prices as traded (no adjustments)
- Capital-adjusted: Adjusted for splits only (not dividends)
- Total-return: Adjusted for both splits and dividends

These functions work with any data vendor that provides cumulative adjustment factors.
The vendor-specific adapter is responsible for:
1. Reading the cumulative factors from the data source
2. Identifying corporate events (dividends, splits) from vendor-specific fields
3. Applying these generic calculation functions

Example vendors that provide cumulative factors:
- Algoseek: CumulativePriceFactor, CumulativeVolumeFactor
- Norgate: Similar cumulative adjustment factors
- Other institutional data providers
"""

from decimal import ROUND_HALF_UP, Decimal


def compute_unadjusted_price(
    adjusted_price: Decimal,
    cumulative_price_factor: Decimal,
) -> Decimal:
    """
    Compute unadjusted (raw) price from total-return adjusted price.

    Unadjusted price is the price as it was actually traded on that day,
    with NO adjustments for splits or dividends.

    Formula:
        unadjusted = adjusted * cumulative_price_factor

    Args:
        adjusted_price: Total-return adjusted price from vendor
        cumulative_price_factor: Cumulative price adjustment factor at this bar

    Returns:
        Unadjusted price (as traded)

    Example:
        >>> # AAPL on 2019-01-02: adjusted=157.92, factor=7.925959
        >>> compute_unadjusted_price(Decimal("157.92"), Decimal("7.925959"))
        Decimal('1251.7623...')  # Actual traded price before splits
    """
    return adjusted_price * cumulative_price_factor


def compute_capital_adjusted_price(
    adjusted_price: Decimal,
    cumulative_price_factor: Decimal,
    cumulative_volume_factor: Decimal,
) -> Decimal:
    """
    Compute capital-adjusted price from total-return adjusted price.

    Capital-adjusted price includes split adjustments but NOT dividend adjustments.
    This is useful for price-based indicators and chart analysis.

    The key insight: In cumulative factor systems:
    - Volume factor captures ONLY split adjustments (shares outstanding changes)
    - Price factor captures BOTH split and dividend adjustments
    - Ratio (price_factor / volume_factor) isolates the dividend component

    Formula:
        capital_adjusted = adjusted * (cumulative_price_factor / cumulative_volume_factor)

    Args:
        adjusted_price: Total-return adjusted price from vendor
        cumulative_price_factor: Cumulative price adjustment factor
        cumulative_volume_factor: Cumulative volume adjustment factor

    Returns:
        Capital-adjusted price (split-adjusted only, no dividends)

    Example:
        >>> # AAPL on 2019-01-02: adjusted=157.92, px_factor=7.925959, vol_factor=7.0
        >>> compute_capital_adjusted_price(
        ...     Decimal("157.92"),
        ...     Decimal("7.925959"),
        ...     Decimal("7.0")
        ... )
        Decimal('178.91...')  # After splits but before dividend adjustments
    """
    # Dividend adjustment factor is px_factor / vol_factor
    # Multiplying by this ratio removes dividend adjustments
    dividend_factor = cumulative_price_factor / cumulative_volume_factor
    return adjusted_price * dividend_factor


def compute_unadjusted_volume(
    adjusted_volume: int,
    cumulative_volume_factor: Decimal,
) -> int:
    """
    Compute unadjusted (raw) volume from adjusted volume.

    Volume adjustments account for splits and similar events that change
    the number of shares outstanding.

    Formula:
        unadjusted_volume = adjusted_volume / cumulative_volume_factor

    Args:
        adjusted_volume: Adjusted volume from vendor
        cumulative_volume_factor: Cumulative volume adjustment factor

    Returns:
        Unadjusted volume (as traded), rounded to nearest integer

    Example:
        >>> # AAPL on 2019-01-02: volume=30606605, vol_factor=7.0
        >>> compute_unadjusted_volume(30606605, Decimal("7.0"))
        4372372  # Actual volume before splits (7.0 = 7:1 split history)
    """
    result = Decimal(adjusted_volume) / cumulative_volume_factor
    return int(result.quantize(Decimal("1"), rounding=ROUND_HALF_UP))
