"""
Schwab Data Source Example
===========================

This example demonstrates how to use the Schwab data source in QTrader.

Key Features Demonstrated:
- Environment setup for Schwab API credentials
- OAuth authentication flow (first-time setup)
- Data caching for fast repeated access
- Viewing raw market data
- Understanding the cache-first architecture

Prerequisites:
- Schwab Developer Account (https://developer.schwab.com)
- API Key and Secret from Schwab Developer Portal
- Environment variables set (see .envrc.example)

Author: QTrader Team
Date: October 2025
"""

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import DataSource, Instrument, InstrumentType


def main():
    """
    Example: Loading Schwab data with caching.

    This demonstrates the complete workflow for accessing Schwab market data.
    """

    print("=" * 70)
    print("QTrader - Schwab Data Source Example")
    print("=" * 70)
    print()

    # Step 1: Create an Instrument
    # ----------------------------
    # Instruments define WHAT data you want (symbol + type + data source)
    print("Step 1: Creating Instrument for AAPL from Schwab")
    print("-" * 70)

    instrument = Instrument(
        symbol="AAPL",
        instrument_type=InstrumentType.EQUITY,
        data_source=DataSource.SCHWAB,  # Use Schwab as data source
    )

    print(f"✓ Instrument created: {instrument.symbol} ({instrument.instrument_type.value})")
    print(f"  Data Source: {instrument.data_source.value}")
    print()

    # Step 2: Resolve Data Source to Adapter
    # ----------------------------------------
    # The resolver reads config/data_sources.yaml and instantiates
    # the appropriate adapter (SchwabOHLCAdapter in this case)
    print("Step 2: Resolving Data Source")
    print("-" * 70)

    resolver = DataSourceResolver()
    adapter = resolver.resolve(instrument)

    print(f"✓ Adapter resolved: {type(adapter).__name__}")
    print("  Configuration loaded from: config/data_sources.yaml")
    print()

    # Step 3: Fetch Data (Cache-First Architecture)
    # -----------------------------------------------
    # First call: Cache miss → OAuth flow → API call → Cache write
    # Subsequent calls: Cache hit → Instant load from disk
    print("Step 3: Fetching Data (2024-01-01 to 2024-01-31)")
    print("-" * 70)
    print()
    print("Note: On first run, you'll see:")
    print("  1. OAuth flow (browser opens for authorization)")
    print("  2. Token cached in ~/.qtrader/schwab_tokens.json")
    print("  3. Data fetched from Schwab API (rate limited to 10 req/sec)")
    print("  4. Data cached in data/us-equity-daily-adjusted-schwab/")
    print()
    print("On subsequent runs:")
    print("  - Data loaded instantly from cache")
    print("  - No API calls needed")
    print("  - No OAuth re-authorization required")
    print()

    try:
        # Fetch bars
        bars = list(adapter.read_bars("2024-01-01", "2024-01-31"))

        print(f"✓ Successfully loaded {len(bars)} bars")
        print()

        # Step 4: Display Sample Data
        # ----------------------------
        if bars:
            print("Step 4: Sample Data (First 5 bars)")
            print("-" * 70)

            for i, bar in enumerate(bars[:5], 1):
                print(f"Bar {i}:")
                print(f"  Date:   {bar.timestamp.strftime('%Y-%m-%d')}")
                print(f"  Open:   ${bar.open:>8.2f}")
                print(f"  High:   ${bar.high:>8.2f}")
                print(f"  Low:    ${bar.low:>8.2f}")
                print(f"  Close:  ${bar.close:>8.2f}")
                print(f"  Volume: {bar.volume:>12,}")
                print()

            print("=" * 70)
            print("Success! ✅")
            print()
            print("Next Steps:")
            print("  - Try different date ranges")
            print("  - Try different symbols")
            print("  - View cached data in: data/us-equity-daily-adjusted-schwab/")
            print("  - Check metadata: data/us-equity-daily-adjusted-schwab/AAPL/.metadata.json")
            print("=" * 70)

    except FileNotFoundError as e:
        print()
        print("❌ Error: Configuration file not found")
        print()
        print("Solution:")
        print("  1. Copy .envrc.example to .envrc")
        print("  2. Edit .envrc with your Schwab API credentials")
        print("  3. Run: direnv allow  (or source .envrc)")
        print()
        print(f"Details: {e}")

    except KeyError as e:
        print()
        print("❌ Error: Environment variables not set")
        print()
        print("Required environment variables:")
        print("  - SCHWAB_API_KEY")
        print("  - SCHWAB_API_SECRET")
        print("  - SCHWAB_REDIRECT_URI (optional, defaults to https://127.0.0.1:8182)")
        print()
        print("Solution:")
        print("  1. Set variables in .envrc file")
        print("  2. Run: direnv allow  (or source .envrc)")
        print()
        print(f"Details: {e}")

    except Exception as e:
        print()
        print(f"❌ Error: {type(e).__name__}")
        print(f"   {e}")
        print()
        print("Common Issues:")
        print("  - Invalid API credentials → Check SCHWAB_API_KEY/SECRET")
        print("  - OAuth failed → Browser should open automatically")
        print("  - Rate limit exceeded → Wait 60 seconds and retry")
        print("  - Network error → Check internet connection")


def advanced_example():
    """
    Advanced Example: Using cached data for fast iteration.

    This shows how caching enables rapid backtesting and analysis.
    """

    print()
    print("=" * 70)
    print("Advanced Example: Cache Performance")
    print("=" * 70)
    print()

    import time

    # Create instrument
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)

    # Resolve adapter
    resolver = DataSourceResolver()
    adapter = resolver.resolve(instrument)

    # First call (may hit API)
    print("First call (may fetch from API)...")
    start_time = time.time()
    bars1 = list(adapter.read_bars("2024-01-01", "2024-12-31"))
    duration1 = time.time() - start_time
    print(f"✓ Loaded {len(bars1)} bars in {duration1:.2f}s")
    print()

    # Second call (definitely from cache)
    print("Second call (from cache)...")
    start_time = time.time()
    bars2 = list(adapter.read_bars("2024-01-01", "2024-12-31"))
    duration2 = time.time() - start_time
    print(f"✓ Loaded {len(bars2)} bars in {duration2:.2f}s")
    print()

    # Performance comparison
    if duration1 > 0 and duration2 > 0:
        speedup = duration1 / duration2
        print(f"Cache speedup: {speedup:.1f}x faster!")
        print()
        print("This is why caching matters for backtesting:")
        print("  - First run: Network latency + API processing")
        print("  - Cached runs: Disk I/O only (instant)")
        print("  - Enables rapid strategy iteration")

    print()
    print("=" * 70)


if __name__ == "__main__":
    # Run basic example
    main()

    # Uncomment to run advanced example:
    # advanced_example()
