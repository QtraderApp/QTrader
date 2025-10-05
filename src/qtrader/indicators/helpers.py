"""
Indicator helper functions for common pattern detection.

Provides 13 utility functions across 5 categories:
- Crossover: crossed_above, crossed_below
- Threshold: crossed_above_threshold, crossed_below_threshold, above_threshold, below_threshold, between_thresholds
- Divergence: divergence_bullish, divergence_bearish
- Histogram: histogram_flipped_positive, histogram_flipped_negative
- Trend: is_increasing, is_decreasing
"""

from typing import Optional

# ============================================================================
# Crossover Detection
# ============================================================================


def crossed_above(
    curr1: Optional[float],
    curr2: Optional[float],
    prev1: Optional[float],
    prev2: Optional[float],
) -> bool:
    """
    Check if value1 crossed above value2.

    Returns True if:
    - prev1 <= prev2 (was below or equal)
    - curr1 > curr2 (now above)

    Args:
        curr1: Current value of first series
        curr2: Current value of second series
        prev1: Previous value of first series
        prev2: Previous value of second series

    Returns:
        True if crossover occurred, False otherwise

    Example:
        fast_sma = ctx.ind.sma(symbol, 20)
        slow_sma = ctx.ind.sma(symbol, 50)
        if crossed_above(fast_sma, slow_sma, prev_fast, prev_slow):
            # Bullish crossover
            ctx.buy_market(symbol, 100)
    """
    if curr1 is None or curr2 is None or prev1 is None or prev2 is None:
        return False

    return prev1 <= prev2 and curr1 > curr2


def crossed_below(
    curr1: Optional[float],
    curr2: Optional[float],
    prev1: Optional[float],
    prev2: Optional[float],
) -> bool:
    """
    Check if value1 crossed below value2.

    Returns True if:
    - prev1 >= prev2 (was above or equal)
    - curr1 < curr2 (now below)

    Args:
        curr1: Current value of first series
        curr2: Current value of second series
        prev1: Previous value of first series
        prev2: Previous value of second series

    Returns:
        True if crossover occurred, False otherwise

    Example:
        fast_sma = ctx.ind.sma(symbol, 20)
        slow_sma = ctx.ind.sma(symbol, 50)
        if crossed_below(fast_sma, slow_sma, prev_fast, prev_slow):
            # Bearish crossover
            ctx.sell_market(symbol, 100)
    """
    if curr1 is None or curr2 is None or prev1 is None or prev2 is None:
        return False

    return prev1 >= prev2 and curr1 < curr2


# ============================================================================
# Threshold Detection
# ============================================================================


def crossed_above_threshold(curr: Optional[float], prev: Optional[float], threshold: float) -> bool:
    """
    Check if value crossed above threshold.

    Returns True if:
    - prev <= threshold (was below or equal)
    - curr > threshold (now above)

    Args:
        curr: Current value
        prev: Previous value
        threshold: Threshold level

    Returns:
        True if crossed above threshold, False otherwise

    Example:
        rsi = ctx.ind.rsi(symbol, 14)
        if crossed_above_threshold(rsi, prev_rsi, 30):
            # RSI crossed above 30 (oversold exit)
            ctx.buy_market(symbol, 100)
    """
    if curr is None or prev is None:
        return False

    return prev <= threshold < curr


def crossed_below_threshold(curr: Optional[float], prev: Optional[float], threshold: float) -> bool:
    """
    Check if value crossed below threshold.

    Returns True if:
    - prev >= threshold (was above or equal)
    - curr < threshold (now below)

    Args:
        curr: Current value
        prev: Previous value
        threshold: Threshold level

    Returns:
        True if crossed below threshold, False otherwise

    Example:
        rsi = ctx.ind.rsi(symbol, 14)
        if crossed_below_threshold(rsi, prev_rsi, 70):
            # RSI crossed below 70 (overbought exit)
            ctx.sell_market(symbol, 100)
    """
    if curr is None or prev is None:
        return False

    return prev >= threshold > curr


def above_threshold(value: Optional[float], threshold: float) -> bool:
    """
    Check if value is currently above threshold.

    Args:
        value: Current value
        threshold: Threshold level

    Returns:
        True if value > threshold, False otherwise

    Example:
        rsi = ctx.ind.rsi(symbol, 14)
        if above_threshold(rsi, 70):
            # RSI is overbought
            pass
    """
    if value is None:
        return False

    return value > threshold


def below_threshold(value: Optional[float], threshold: float) -> bool:
    """
    Check if value is currently below threshold.

    Args:
        value: Current value
        threshold: Threshold level

    Returns:
        True if value < threshold, False otherwise

    Example:
        rsi = ctx.ind.rsi(symbol, 14)
        if below_threshold(rsi, 30):
            # RSI is oversold
            pass
    """
    if value is None:
        return False

    return value < threshold


def between_thresholds(value: Optional[float], lower: float, upper: float) -> bool:
    """
    Check if value is between two thresholds (inclusive).

    Args:
        value: Current value
        lower: Lower threshold
        upper: Upper threshold

    Returns:
        True if lower <= value <= upper, False otherwise

    Example:
        rsi = ctx.ind.rsi(symbol, 14)
        if between_thresholds(rsi, 40, 60):
            # RSI is in neutral zone
            pass
    """
    if value is None:
        return False

    return lower <= value <= upper


# ============================================================================
# Divergence Detection
# ============================================================================


def divergence_bullish(
    price_low1: Optional[float],
    price_low2: Optional[float],
    indicator_low1: Optional[float],
    indicator_low2: Optional[float],
) -> bool:
    """
    Detect bullish divergence.

    Bullish divergence occurs when:
    - Price makes a lower low (price_low2 < price_low1)
    - Indicator makes a higher low (indicator_low2 > indicator_low1)

    This suggests weakening downward momentum and potential reversal.

    Args:
        price_low1: First price low
        price_low2: Second price low (more recent)
        indicator_low1: First indicator low
        indicator_low2: Second indicator low (more recent)

    Returns:
        True if bullish divergence detected, False otherwise

    Example:
        # Detect bullish RSI divergence
        if divergence_bullish(price_low1, price_low2, rsi_low1, rsi_low2):
            # Potential bullish reversal
            ctx.buy_market(symbol, 100)
    """
    if price_low1 is None or price_low2 is None or indicator_low1 is None or indicator_low2 is None:
        return False

    return price_low2 < price_low1 and indicator_low2 > indicator_low1


def divergence_bearish(
    price_high1: Optional[float],
    price_high2: Optional[float],
    indicator_high1: Optional[float],
    indicator_high2: Optional[float],
) -> bool:
    """
    Detect bearish divergence.

    Bearish divergence occurs when:
    - Price makes a higher high (price_high2 > price_high1)
    - Indicator makes a lower high (indicator_high2 < indicator_high1)

    This suggests weakening upward momentum and potential reversal.

    Args:
        price_high1: First price high
        price_high2: Second price high (more recent)
        indicator_high1: First indicator high
        indicator_high2: Second indicator high (more recent)

    Returns:
        True if bearish divergence detected, False otherwise

    Example:
        # Detect bearish RSI divergence
        if divergence_bearish(price_high1, price_high2, rsi_high1, rsi_high2):
            # Potential bearish reversal
            ctx.sell_market(symbol, 100)
    """
    if price_high1 is None or price_high2 is None or indicator_high1 is None or indicator_high2 is None:
        return False

    return price_high2 > price_high1 and indicator_high2 < indicator_high1


# ============================================================================
# Histogram Detection
# ============================================================================


def histogram_flipped_positive(curr_histogram: Optional[float], prev_histogram: Optional[float]) -> bool:
    """
    Check if histogram flipped from negative to positive.

    Returns True if:
    - prev_histogram <= 0 (was negative or zero)
    - curr_histogram > 0 (now positive)

    Args:
        curr_histogram: Current histogram value
        prev_histogram: Previous histogram value

    Returns:
        True if histogram flipped positive, False otherwise

    Example:
        macd = ctx.ind.macd(symbol, 12, 26, 9)
        if macd and histogram_flipped_positive(macd.histogram, prev_histogram):
            # MACD bullish momentum shift
            ctx.buy_market(symbol, 100)
    """
    if curr_histogram is None or prev_histogram is None:
        return False

    return prev_histogram <= 0 < curr_histogram


def histogram_flipped_negative(curr_histogram: Optional[float], prev_histogram: Optional[float]) -> bool:
    """
    Check if histogram flipped from positive to negative.

    Returns True if:
    - prev_histogram >= 0 (was positive or zero)
    - curr_histogram < 0 (now negative)

    Args:
        curr_histogram: Current histogram value
        prev_histogram: Previous histogram value

    Returns:
        True if histogram flipped negative, False otherwise

    Example:
        macd = ctx.ind.macd(symbol, 12, 26, 9)
        if macd and histogram_flipped_negative(macd.histogram, prev_histogram):
            # MACD bearish momentum shift
            ctx.sell_market(symbol, 100)
    """
    if curr_histogram is None or prev_histogram is None:
        return False

    return prev_histogram >= 0 > curr_histogram


# ============================================================================
# Trend Detection
# ============================================================================


def is_increasing(values: list[Optional[float]], periods: int = 3) -> bool:
    """
    Check if indicator is trending upward over N periods.

    Returns True if each value is greater than the previous value
    for the specified number of periods.

    Args:
        values: List of indicator values (most recent last)
        periods: Number of periods to check (default 3)

    Returns:
        True if increasing trend detected, False otherwise

    Example:
        sma_values = [ctx.ind.sma(symbol, 20) for _ in range(3)]
        if is_increasing(sma_values):
            # SMA is in uptrend
            pass
    """
    if len(values) < periods:
        return False

    recent_values = values[-periods:]
    if any(v is None for v in recent_values):
        return False

    # Type narrowing - we know all values are float now
    float_values = [v for v in recent_values if v is not None]

    for i in range(1, len(float_values)):
        if float_values[i] <= float_values[i - 1]:
            return False

    return True


def is_decreasing(values: list[Optional[float]], periods: int = 3) -> bool:
    """
    Check if indicator is trending downward over N periods.

    Returns True if each value is less than the previous value
    for the specified number of periods.

    Args:
        values: List of indicator values (most recent last)
        periods: Number of periods to check (default 3)

    Returns:
        True if decreasing trend detected, False otherwise

    Example:
        sma_values = [ctx.ind.sma(symbol, 20) for _ in range(3)]
        if is_decreasing(sma_values):
            # SMA is in downtrend
            pass
    """
    if len(values) < periods:
        return False

    recent_values = values[-periods:]
    if any(v is None for v in recent_values):
        return False

    # Type narrowing - we know all values are float now
    float_values = [v for v in recent_values if v is not None]

    for i in range(1, len(float_values)):
        if float_values[i] >= float_values[i - 1]:
            return False

    return True
