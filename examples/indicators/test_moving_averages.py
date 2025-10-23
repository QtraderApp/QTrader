"""
Test Moving Average Indicators.

Demonstrates usage of all moving average indicators.
"""

from datetime import datetime, timedelta

from qtrader.contracts.data import Bar
from qtrader.libraries.indicators import DEMA, EMA, HMA, SMA, SMMA, TEMA, WMA


def create_sample_bars(num_bars: int = 50) -> list[Bar]:
    """
    Create sample price bars for testing.

    Args:
        num_bars: Number of bars to create

    Returns:
        List of Bar objects with synthetic price data
    """
    bars = []
    base_price = 100.0

    for i in range(num_bars):
        # Simple uptrend with noise
        close = base_price + i * 0.5 + (i % 5 - 2) * 0.3
        high = close + 0.5
        low = close - 0.5
        open_price = close - 0.1
        volume = 1000000

        timestamp = datetime(2024, 1, 1) + timedelta(days=i)

        bar = Bar(
            trade_datetime=timestamp,
            open=open_price,
            high=high,
            low=low,
            close=close,
            volume=volume,
        )
        bars.append(bar)

    return bars


def test_all_moving_averages():
    """Test all moving average implementations."""

    print("=" * 70)
    print("Moving Average Indicators Test")
    print("=" * 70)

    # Create sample data
    bars = create_sample_bars(50)
    period = 10

    # Initialize all indicators
    indicators = {
        "SMA": SMA(period=period),
        "EMA": EMA(period=period),
        "WMA": WMA(period=period),
        "DEMA": DEMA(period=period),
        "TEMA": TEMA(period=period),
        "HMA": HMA(period=period),
        "SMMA": SMMA(period=period),
    }

    print(f"\nTesting with {len(bars)} bars, period={period}")
    print(f"Price range: {bars[0].close:.2f} to {bars[-1].close:.2f}\n")

    # Test stateful updates
    print("=" * 70)
    print("STATEFUL MODE (incremental updates)")
    print("=" * 70)

    for name, indicator in indicators.items():
        print(f"\n{name}:")
        print(f"  Initial ready state: {indicator.is_ready}")

        # Update with all bars
        last_value = None
        ready_at = None

        for i, bar in enumerate(bars):
            value = indicator.update(bar)
            if value is not None:
                last_value = value
                if ready_at is None:
                    ready_at = i

        print(f"  Ready after {ready_at} bars")
        print(f"  Final value: {last_value:.4f}")
        print(f"  Is ready: {indicator.is_ready}")

    # Test stateless calculation
    print("\n" + "=" * 70)
    print("STATELESS MODE (batch calculation)")
    print("=" * 70)

    # Reset all indicators
    for indicator in indicators.values():
        indicator.reset()

    for name, indicator in indicators.items():
        values = indicator.calculate(bars)

        # Count non-None values
        valid_values = [v for v in values if v is not None]

        print(f"\n{name}:")
        print(f"  Total values: {len(values)}")
        print(f"  Valid values: {len(valid_values)}")
        print(f"  First valid: {valid_values[0]:.4f}")
        print(f"  Last valid: {valid_values[-1]:.4f}")

    # Test reset functionality
    print("\n" + "=" * 70)
    print("RESET FUNCTIONALITY")
    print("=" * 70)

    sma = SMA(period=5)

    # First run
    for bar in bars[:10]:
        sma.update(bar)

    value_before_reset = sma.value
    print(f"\nSMA value before reset: {value_before_reset:.4f}")
    print(f"Is ready before reset: {sma.is_ready}")

    # Reset
    sma.reset()
    print(f"Is ready after reset: {sma.is_ready}")
    print(f"Value after reset: {sma.value}")

    # Second run with same data
    for bar in bars[:10]:
        sma.update(bar)

    value_after_reset = sma.value
    print(f"SMA value after re-running: {value_after_reset:.4f}")
    print(f"Values match: {abs(value_before_reset - value_after_reset) < 0.0001}")

    # Test different price fields
    print("\n" + "=" * 70)
    print("DIFFERENT PRICE FIELDS")
    print("=" * 70)

    price_fields = ["open", "high", "low", "close"]
    period = 10

    for field in price_fields:
        sma = SMA(period=period, price_field=field)

        for bar in bars:
            sma.update(bar)

        print(f"\nSMA({period}) on {field:6s}: {sma.value:.4f}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_all_moving_averages()
