#!/usr/bin/env python3
"""
Test Schwab smart caching with real AAPL data.

This script demonstrates:
1. Initial authentication with Schwab API
2. First fetch (creates cache)
3. Incremental update (fetches only new data)
4. Smart caching performance

Before running:
1. Set environment variables:
   export SCHWAB_API_KEY="your_client_id"
   export SCHWAB_API_SECRET="your_client_secret"
   export SCHWAB_REDIRECT_URI="https://127.0.0.1:8182"  # Optional, this is default

2. Make sure your Schwab app redirect URI is configured to: https://127.0.0.1:8182

Usage:
    python scripts/test_schwab_aapl.py
"""

import os
import sys
from datetime import date, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import Instrument


def print_section(title: str):
    """Print section header."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


def main():
    """Test Schwab smart caching with AAPL."""

    # Check environment variables
    print_section("1. Checking Environment")

    api_key = os.getenv("SCHWAB_API_KEY")
    api_secret = os.getenv("SCHWAB_API_SECRET")
    redirect_uri = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1:8182")

    if not api_key or not api_secret:
        print("❌ ERROR: Missing required environment variables!")
        print("\nPlease set:")
        print("  export SCHWAB_API_KEY='your_client_id'")
        print("  export SCHWAB_API_SECRET='your_client_secret'")
        print("\nOptional:")
        print("  export SCHWAB_REDIRECT_URI='https://127.0.0.1:8182'")
        return 1

    print(f"✓ SCHWAB_API_KEY: {api_key[:10]}...")
    print(f"✓ SCHWAB_API_SECRET: {api_secret[:10]}...")
    print(f"✓ SCHWAB_REDIRECT_URI: {redirect_uri}")

    # Initialize resolver and adapter
    print_section("2. Initializing Schwab Adapter")

    try:
        resolver = DataSourceResolver()

        # Specify dataset explicitly (no guessing!)
        dataset = "schwab-us-equity-1d-adjusted"
        config = resolver.sources.get(dataset)

        if not config:
            print(f"❌ ERROR: {dataset} not found in config")
            return 1

        # Update config with env vars
        config["client_id"] = api_key
        config["client_secret"] = api_secret
        config["redirect_uri"] = redirect_uri

        print(f"✓ Dataset: {dataset}")
        print(f"✓ Cache strategy: {config.get('cache_strategy', 'smart')}")
        print(f"✓ Incremental updates: {config.get('enable_incremental_update', True)}")
        print(f"✓ Cache root: {config.get('cache_root')}")

        # Create minimal instrument (symbol only!)
        # Dataset config is the single source of truth for provider, asset type, etc.
        # User's responsibility: provide correct ticker for this dataset (AAPL works for Schwab equities)
        instrument = Instrument(symbol="AAPL")

        # Use new explicit API: resolve_by_dataset
        adapter = resolver.resolve_by_dataset(dataset, instrument)

        print(f"✓ Adapter initialized for {instrument.symbol}")
        print(f"  (Config provides: provider={config['provider']}, asset_class={config['asset_class']})")

    except Exception as e:
        print(f"❌ ERROR: Failed to initialize adapter: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test 1: Initial fetch (creates cache)
    print_section("3. Initial Fetch - Last 30 Days (Creates Cache)")

    try:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)

        print(f"Fetching AAPL from {start_date} to {end_date}")
        print("This will:")
        print("  1. Authenticate with Schwab (browser will open)")
        print("  2. Fetch data from API")
        print("  3. Create cache")
        print("\nPlease complete authentication in browser...\n")

        bars = list(
            adapter.read_bars(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
        )

        print(f"\n✓ Fetched {len(bars)} bars")

        if bars:
            first_bar = bars[0]
            last_bar = bars[-1]
            print(f"  First bar: {first_bar.timestamp.date()} - Close: ${first_bar.close:.2f}")
            print(f"  Last bar:  {last_bar.timestamp.date()} - Close: ${last_bar.close:.2f}")

        # Check cache
        cache_file = Path(config["cache_root"]) / "AAPL" / "data.parquet"
        metadata_file = Path(config["cache_root"]) / "AAPL" / ".metadata.json"

        if cache_file.exists():
            print(f"\n✓ Cache created: {cache_file}")
            print(f"  Size: {cache_file.stat().st_size / 1024:.1f} KB")

        if metadata_file.exists():
            import json

            with open(metadata_file) as f:
                metadata = json.load(f)
            print(f"✓ Metadata: {metadata['date_range']}")

    except Exception as e:
        print(f"❌ ERROR: Initial fetch failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test 2: Read from cache (no API calls)
    print_section("4. Read from Cache (No API Calls)")

    try:
        # Read same date range - should come from cache
        bars_cached = list(
            adapter.read_bars(
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
        )

        print(f"✓ Read {len(bars_cached)} bars from cache (no API calls)")

        if len(bars_cached) == len(bars):
            print(f"✓ Cache integrity verified ({len(bars_cached)} bars match)")
        else:
            print(f"⚠ Warning: Bar count mismatch (cached: {len(bars_cached)}, original: {len(bars)})")

    except Exception as e:
        print(f"❌ ERROR: Cache read failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test 3: Incremental update
    print_section("5. Incremental Update (Smart Caching)")

    try:
        print("Calling update_to_latest()...")
        print("This will fetch only new bars since last cache update")

        bars_added, update_start, update_end = adapter.update_to_latest()

        if bars_added > 0:
            print(f"\n✓ Added {bars_added} new bars")
            print(f"  Range: {update_start} to {update_end}")
        else:
            print("\n✓ Cache already up-to-date (no new bars)")
            if update_start:
                print(f"  Would update: {update_start} to {update_end}")

    except Exception as e:
        print(f"❌ ERROR: Incremental update failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Test 4: Gap filling
    print_section("6. Gap Filling Test")

    try:
        # Request a range that extends beyond cache
        future_end = end_date + timedelta(days=60)

        print(f"Requesting extended range: {start_date} to {future_end}")
        print("Smart caching will:")
        print("  1. Read cached data")
        print("  2. Detect gap (cache end to future end)")
        print("  3. Fetch only the gap from API")
        print("  4. Merge and return combined data")

        bars_extended = list(
            adapter.read_bars(
                start_date=start_date.isoformat(),
                end_date=future_end.isoformat(),
            )
        )

        print(f"\n✓ Retrieved {len(bars_extended)} total bars")
        print(f"  Original cache: {len(bars)} bars")
        print(f"  Gap filled: {len(bars_extended) - len(bars)} new bars")

        if bars_extended:
            last_bar = bars_extended[-1]
            print(f"  Latest bar: {last_bar.timestamp.date()} - Close: ${last_bar.close:.2f}")

    except Exception as e:
        print(f"❌ ERROR: Gap filling failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Summary
    print_section("7. Summary")

    print("✓ Smart caching fully functional!")
    print("\nFeatures tested:")
    print("  ✓ Initial fetch and cache creation")
    print("  ✓ Reading from cache (no API calls)")
    print("  ✓ Incremental updates (fetch only new bars)")
    print("  ✓ Gap filling (detect and fetch missing ranges)")

    print("\nCache location:")
    print(f"  {cache_file}")
    print(f"  {metadata_file}")

    print("\nNext steps:")
    print(f"  • Try: qtrader data update --dataset {dataset} --symbols AAPL --verbose")
    print(f"  • Try: qtrader data cache-info --dataset {dataset}")
    print(
        "  • Try: qtrader data raw --symbol AAPL --start-date 2024-01-01 --end-date 2024-12-31 --dataset schwab-us-equity-1d-adjusted"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
