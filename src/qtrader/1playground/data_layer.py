"""
Data Layer Demonstration Script

This script demonstrates the complete data pipeline from raw Algoseek parquet files
to canonical bars. It shows how data flows through the system:

1. Raw Parquet Files (Algoseek format)
   ↓
2. AlgoseekOHLCVendorAdapter (reads raw bars)
   ↓
3. AlgoseekPriceSeries (vendor-specific series)
   ↓
4. PriceSeries (standardized format, 3 adjustment modes)
   ↓
5. PriceSeriesIterator (streams multi-mode bars)
   ↓
6. MultiBar (bar with all 3 adjustment modes)

Usage:
    python -m qtrader.1playground.data_layer

Output:
    - Displays raw Algoseek bars
    - Shows canonical bars in all 3 adjustment modes
    - Demonstrates how different components access different modes
"""

from qtrader.adapters.algoseek import AlgoseekOHLCVendorAdapter
from qtrader.data import DataLoader
from qtrader.models.instrument import DataSource, Instrument, InstrumentType


def main():
    """Demonstrate the data layer pipeline."""
    print("=" * 80)
    print("QTrader Data Layer Demonstration")
    print("=" * 80)
    print()

    # Configuration
    symbol = "AAPL"
    start_date = "2020-08-28"
    end_date = "2020-09-04"

    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print("Note: This period includes Apple's 4:1 stock split on 2020-08-31")
    print()

    # Step 1: Configure DataLoader
    print("-" * 80)
    print("Step 1: Configure DataLoader")
    print("-" * 80)

    config = {
        "adapter": {
            "root_path": "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample",
            "path_template": "{root_path}/SecId={secid}/*.parquet",
            "symbol_map": "data/equity_security_master_sample.csv",
        }
    }

    print(f"Root path: {config['adapter']['root_path']}")
    print(f"Path template: {config['adapter']['path_template']}")
    print(f"Symbol map: {config['adapter']['symbol_map']}")
    print()

    # Step 2: Load data
    print("-" * 80)
    print("Step 2: Load Data via DataLoader")
    print("-" * 80)

    loader = DataLoader(config)
    iterator = loader.load_data(symbol, start_date, end_date)

    print("✓ DataLoader initialized")
    print("✓ Data loaded and transformed to canonical format")
    print("✓ Iterator created with all 3 adjustment modes")
    print()

    # Step 3: Demonstrate the data pipeline
    print("-" * 80)
    print("Step 3: Stream Multi-Mode Bars")
    print("-" * 80)
    print()

    bar_count = 0
    split_detected = False
    for multi_bar in iterator:
        bar_count += 1

        # Show all bars to see the split effect
        print(f"Bar #{bar_count}: {multi_bar.trade_datetime}")
        print(f"  Symbol: {multi_bar.symbol}")
        print()

        # Show all 3 adjustment modes
        print("  Adjustment Modes:")
        print("    1. ADJUSTED (split-adjusted, for strategy indicators):")
        print(f"       Close: ${multi_bar.adjusted.close:.4f}")
        print(f"       Volume: {multi_bar.adjusted.volume:,}")
        print()

        print("    2. UNADJUSTED (raw prices, for execution fills):")
        print(f"       Close: ${multi_bar.unadjusted.close:.4f}")
        print(f"       Volume: {multi_bar.unadjusted.volume:,}")
        print()

        print("    3. TOTAL_RETURN (includes dividends, for performance):")
        print(f"       Close: ${multi_bar.total_return.close:.4f}")
        print(f"       Volume: {multi_bar.total_return.volume:,}")
        print()

        # Check if this is split day by comparing adjusted vs unadjusted
        ratio = float(multi_bar.unadjusted.close) / float(multi_bar.adjusted.close)
        if abs(ratio - 4.0) < 0.01:  # Close to 4:1 split
            split_detected = True
            print("  ⚠️  SPLIT DETECTED: 4:1 Stock Split")
            print("      Notice how unadjusted close is ~4x the adjusted close!")
            print()

        print()

    print(f"✓ Processed {bar_count} bars")
    if split_detected:
        print("✓ Stock split successfully detected and handled")
        print()
        print("KEY INSIGHT:")
        print("  Before split (2020-08-28):")
        print("    - ADJUSTED mode shows ~$125 (already adjusted for future 4:1 split)")
        print("    - UNADJUSTED mode shows ~$499 (actual historical price)")
        print("  After split (2020-08-31+):")
        print("    - Both modes converge (~$129-134) as no future splits to adjust for")
        print("    - TOTAL_RETURN mode includes dividend reinvestment assumptions")
    print()

    # Step 4: Demonstrate use cases
    print("-" * 80)
    print("Step 4: Use Cases for Each Adjustment Mode")
    print("-" * 80)
    print()

    # Reset iterator
    iterator = loader.load_data(symbol, start_date, end_date)
    first_bar = next(iterator)

    print("1. STRATEGY USE CASE (adjusted mode):")
    print("   - Compute indicators using split-adjusted prices")
    print("   - Ensures indicator continuity across splits")
    print(f"   - Example: SMA calculation uses close = ${first_bar.adjusted.close:.4f}")
    print()

    print("2. EXECUTION USE CASE (unadjusted mode):")
    print("   - Fill orders at realistic historical prices")
    print("   - No look-ahead bias from future splits")
    print(f"   - Example: Fill order at close = ${first_bar.unadjusted.close:.4f}")
    print()

    print("3. PERFORMANCE USE CASE (total_return mode):")
    print("   - Calculate accurate returns including dividends")
    print("   - Track true investor experience")
    print(f"   - Example: Return calculation uses close = ${first_bar.total_return.close:.4f}")
    print()

    # Step 5: Show raw data access
    print("-" * 80)
    print("Step 5: Raw Data Access (optional)")
    print("-" * 80)
    print()

    print("For advanced users, you can access raw Algoseek bars directly:")
    print()

    instrument = Instrument(symbol, InstrumentType.EQUITY, DataSource.ALGOSEEK)
    adapter = AlgoseekOHLCVendorAdapter(config["adapter"], instrument)

    raw_bars = list(adapter.read_bars(start_date, end_date))
    first_raw = raw_bars[0]

    print("Raw Algoseek Bar #1:")
    print(f"  Date: {first_raw.TradeDate}")
    print(f"  Ticker: {first_raw.Ticker}")
    print(f"  Open: ${first_raw.Open:.4f}")
    print(f"  High: ${first_raw.High:.4f}")
    print(f"  Low: ${first_raw.Low:.4f}")
    print(f"  Close: ${first_raw.Close:.4f}")
    print(f"  Volume: {first_raw.MarketHoursVolume:,}")
    print(f"  Cumulative Price Factor: {first_raw.CumulativePriceFactor:.6f}")
    print(f"  Cumulative Volume Factor: {first_raw.CumulativeVolumeFactor:.6f}")
    if first_raw.AdjustmentFactor:
        print(f"  Adjustment Factor: {first_raw.AdjustmentFactor}")
        print(f"  Adjustment Reason: {first_raw.AdjustmentReason}")
    print()

    # Summary
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("The data layer provides:")
    print("  ✓ Clean separation between vendor formats and canonical format")
    print("  ✓ Three adjustment modes for different use cases")
    print("  ✓ Memory-efficient streaming via iterators")
    print("  ✓ Automatic split/dividend handling")
    print("  ✓ Type-safe data access")
    print()
    print("Components access the mode they need:")
    print("  • Strategy → adjusted (consistent indicators)")
    print("  • Execution Engine → unadjusted (realistic fills)")
    print("  • Performance Tracker → total_return (accurate returns)")
    print()


if __name__ == "__main__":
    main()
