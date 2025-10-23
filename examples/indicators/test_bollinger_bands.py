"""
Test Bollinger Bands Indicator.

Demonstrates usage of the custom Bollinger Bands indicator including:
- Basic band calculation
- Bandwidth analysis (volatility detection)
- %B indicator (relative position)
- Trading signals
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from qtrader.contracts.data import Bar

# Add project root to path for my_library imports
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from my_library.indicators import BollingerBands  # noqa: E402


def create_sample_bars(num_bars: int = 100, volatility: str = "normal") -> list[Bar]:
    """
    Create sample price bars with different volatility patterns.

    Args:
        num_bars: Number of bars to create
        volatility: "low", "normal", or "high"

    Returns:
        List of Bar objects
    """
    bars = []
    base_price = 100.0

    # Volatility settings
    vol_multiplier = {"low": 0.3, "normal": 1.0, "high": 2.5}[volatility]

    for i in range(num_bars):
        # Create uptrend with varying volatility
        trend = i * 0.2
        noise = ((i % 7 - 3) * 0.5 + (i % 3 - 1) * 0.3) * vol_multiplier

        close = base_price + trend + noise
        high = close + (0.5 * vol_multiplier)
        low = close - (0.5 * vol_multiplier)
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


def test_bollinger_bands():
    """Test Bollinger Bands indicator with various scenarios."""

    print("=" * 80)
    print("Bollinger Bands Indicator Test")
    print("=" * 80)

    # Test 1: Basic Bollinger Bands calculation
    print("\n" + "=" * 80)
    print("TEST 1: Basic Bollinger Bands Calculation")
    print("=" * 80)

    bars = create_sample_bars(50, volatility="normal")
    bb = BollingerBands(period=20, num_std=2.0)

    print(f"\nProcessing {len(bars)} bars with period=20, num_std=2.0")
    print(f"Price range: {bars[0].close:.2f} to {bars[-1].close:.2f}\n")

    # Process bars
    for i, bar in enumerate(bars):
        bands = bb.update(bar)

        if bands is not None and i >= len(bars) - 5:
            print(f"Bar {i}:")
            print(f"  Price:  {bar.close:.2f}")
            print(f"  Upper:  {bands['upper']:.2f}")
            print(f"  Middle: {bands['middle']:.2f}")
            print(f"  Lower:  {bands['lower']:.2f}")
            print(f"  Width:  {bb.bandwidth:.4f}")
            print(f"  %B:     {bb.percent_b:.4f}")
            print()

    # Test 2: Stateless calculation
    print("=" * 80)
    print("TEST 2: Stateless Batch Calculation")
    print("=" * 80)

    bb2 = BollingerBands(period=20, num_std=2.0)
    all_bands = bb2.calculate(bars)

    valid_bands = [b for b in all_bands if b is not None]
    print(f"\nTotal bars: {len(all_bands)}")
    print(f"Valid bands: {len(valid_bands)}")
    print(f"Warmup period: {len(all_bands) - len(valid_bands)} bars")

    if valid_bands:
        print("\nFirst valid bands:")
        print(f"  Upper:  {valid_bands[0]['upper']:.2f}")
        print(f"  Middle: {valid_bands[0]['middle']:.2f}")
        print(f"  Lower:  {valid_bands[0]['lower']:.2f}")

        print("\nLast valid bands:")
        print(f"  Upper:  {valid_bands[-1]['upper']:.2f}")
        print(f"  Middle: {valid_bands[-1]['middle']:.2f}")
        print(f"  Lower:  {valid_bands[-1]['lower']:.2f}")

    # Test 3: Volatility detection with bandwidth
    print("\n" + "=" * 80)
    print("TEST 3: Volatility Detection (Bandwidth Analysis)")
    print("=" * 80)

    print("\nComparing different volatility regimes:\n")

    for vol_name in ["low", "normal", "high"]:
        bars_vol = create_sample_bars(50, volatility=vol_name)
        bb_vol = BollingerBands(period=20, num_std=2.0)

        # Process all bars
        for bar in bars_vol:
            bb_vol.update(bar)

        bandwidth = bb_vol.bandwidth
        bands = bb_vol.value

        if bandwidth and bands:
            print(f"{vol_name.upper()} Volatility:")
            print(f"  Bandwidth: {bandwidth:.4f} ({bandwidth * 100:.2f}%)")
            print(f"  Band range: {bands['upper'] - bands['lower']:.2f}")
            print()

    # Test 4: Trading signals with %B
    print("=" * 80)
    print("TEST 4: Trading Signals Using %B")
    print("=" * 80)

    bars_trading = create_sample_bars(60, volatility="normal")
    bb_trading = BollingerBands(period=20, num_std=2.0)

    signals = []

    for i, bar in enumerate(bars_trading):
        bands = bb_trading.update(bar)

        if bands is not None:
            percent_b = bb_trading.percent_b

            if percent_b is not None:
                # Generate trading signals
                if percent_b > 1.0:
                    signal = "SELL (overbought)"
                    signals.append((i, bar.close, signal, percent_b))
                elif percent_b < 0.0:
                    signal = "BUY (oversold)"
                    signals.append((i, bar.close, signal, percent_b))
                elif 0.4 <= percent_b <= 0.6:
                    signal = "NEUTRAL (near middle)"
                else:
                    signal = None

    print(f"\nGenerated {len(signals)} trading signals:")
    for bar_idx, price, signal, pb in signals[:10]:  # Show first 10
        print(f"  Bar {bar_idx}: {signal:25s} (Price: {price:.2f}, %B: {pb:.3f})")

    if len(signals) > 10:
        print(f"  ... and {len(signals) - 10} more signals")

    # Test 5: Different parameter settings
    print("\n" + "=" * 80)
    print("TEST 5: Different Parameter Settings")
    print("=" * 80)

    bars_params = create_sample_bars(50, volatility="normal")

    configs = [
        {"period": 10, "num_std": 1.5, "name": "Aggressive (10, 1.5)"},
        {"period": 20, "num_std": 2.0, "name": "Standard (20, 2.0)"},
        {"period": 50, "num_std": 2.5, "name": "Conservative (50, 2.5)"},
    ]

    print("\nComparing different settings on same data:\n")

    for config in configs:
        bb_param = BollingerBands(period=config["period"], num_std=config["num_std"])

        for bar in bars_params:
            bb_param.update(bar)

        bands = bb_param.value
        if bands:
            print(f"{config['name']}:")
            print(f"  Upper:  {bands['upper']:.2f}")
            print(f"  Middle: {bands['middle']:.2f}")
            print(f"  Lower:  {bands['lower']:.2f}")
            print(f"  Range:  {bands['upper'] - bands['lower']:.2f}")
            print()

    # Test 6: Reset functionality
    print("=" * 80)
    print("TEST 6: Reset Functionality")
    print("=" * 80)

    bb_reset = BollingerBands(period=10)

    # First run
    for bar in bars[:15]:
        bb_reset.update(bar)

    value_before = bb_reset.value
    print(f"\nValue before reset: {value_before}")
    print(f"Is ready before reset: {bb_reset.is_ready}")

    # Reset
    bb_reset.reset()
    print(f"\nIs ready after reset: {bb_reset.is_ready}")
    print(f"Value after reset: {bb_reset.value}")

    # Second run
    for bar in bars[:15]:
        bb_reset.update(bar)

    value_after = bb_reset.value
    print(f"\nValue after re-running: {value_after}")

    if value_before and value_after:
        match = all(abs(value_before[k] - value_after[k]) < 0.0001 for k in ["upper", "middle", "lower"])
        print(f"Values match: {match}")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_bollinger_bands()
