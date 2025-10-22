"""
Example: Using DataService for data loading.

This example demonstrates the new DataService interface from Phase 1
of the lego architecture. DataService provides a clean, testable interface
for loading and streaming price data.

The service:
- Loads data for single or multiple symbols
- Streams data via iterators
- Provides all three adjustment modes simultaneously
- Handles instrument metadata
"""

from datetime import date

from qtrader.services import DataService
from qtrader.services.data.config import BarSchemaConfig, DataConfig
from qtrader.services.data.source_selector import AssetClass, DataSourceSelector


def main() -> None:
    """Demonstrate DataService usage."""

    # 1. Configure data loading
    bar_schema = BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    # Configure data source selector
    selector = DataSourceSelector(
        provider="schwab",
        asset_class=AssetClass.EQUITY,
        frequency="1d",
    )

    config = DataConfig(
        mode="adjusted",
        frequency="1d",
        timezone="America/New_York",
        source_selector=selector,
        bar_schema=bar_schema,
    )

    # 2. Initialize DataService
    service = DataService(config)
    print("✓ DataService initialized")

    # 3. Load single symbol
    print("\n--- Loading single symbol ---")
    iterator = service.load_symbol(
        "AAPL",
        date(2020, 1, 1),
        date(2025, 1, 31),
    )

    # Stream bars
    bar_count = 0
    for multi_bar in iterator:
        bar_count += 1

        # Access different adjustment modes
        strategy_bar = multi_bar.adjusted  # For indicators/signals
        exec_bar = multi_bar.unadjusted  # For fills/execution
        perf_bar = multi_bar.total_return  # For performance/returns

        # Print first bar as example
        if bar_count == 1:
            print(f"Symbol: {multi_bar.symbol}")
            print(f"Date: {multi_bar.trade_datetime.date()}")
            print(f"Adjusted Close: ${strategy_bar.close:.2f}")
            print(f"Unadjusted Close: ${exec_bar.close:.2f}")
            print(f"Total Return Close: ${perf_bar.close:.2f}")
            if perf_bar.dividend:
                print(f"Dividend: ${perf_bar.dividend}")

            # Note: Schwab only provides adjusted data
            if "schwab" in config.source_selector.to_tag().lower():
                print("\n⚠️  Note: Schwab only provides split-adjusted data.")
                print("   All three modes show the same prices (adjusted).")
                print("   Unadjusted and total_return are not available from Schwab API.\n")

    print(f"✓ Loaded {bar_count} bars for AAPL")

    # 4. Load multiple symbols (universe)
    print("\n--- Loading universe ---")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    iterators = service.load_universe(
        symbols,
        date(2020, 1, 1),
        date(2025, 1, 31),
    )

    print(f"✓ Loaded data for {len(iterators)} symbols")

    # Process each symbol
    for symbol, iterator in iterators.items():
        bars = list(iterator)
        if bars:
            first_close = bars[0].adjusted.close
            last_close = bars[-1].adjusted.close
            pct_change = ((last_close - first_close) / first_close) * 100
            print(f"  {symbol}: {len(bars)} bars, {pct_change:+.2f}% change")

    # 5. Get instrument metadata
    print("\n--- Instrument metadata ---")
    instrument = service.get_instrument("AAPL")
    print(f"Symbol: {instrument.symbol}")
    print(f"Frequency: {instrument.frequency or 'default'}")
    print(f"Metadata: {instrument.metadata or 'none'}")

    # 6. Demonstrate error handling
    print("\n--- Error handling ---")
    try:
        service.load_symbol(
            "AAPL",
            date(2020, 12, 31),  # End before start
            date(2020, 1, 1),
        )
    except ValueError as e:
        print(f"✓ Caught expected error: {e}")

    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
