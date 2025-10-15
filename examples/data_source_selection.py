"""
Example: Data source selection in DataService.

This example demonstrates the different ways to specify which data source
to use (Algoseek, Schwab, CSV, etc.) when loading data with DataService.

There are 3 ways to control data source selection:
1. Via DataConfig.source_tag (main method)
2. Via data_sources.yaml configuration file
3. Via data_source parameter at runtime (future enhancement)
"""

from datetime import date

from qtrader.config.data_config import BarSchemaConfig, DataConfig
from qtrader.services.data import DataService


def example_1_algoseek() -> None:
    """Example 1: Load from Algoseek (local parquet files)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Loading from Algoseek")
    print("=" * 60)

    # Configure bar schema (maps vendor columns to canonical fields)
    bar_schema = BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    # Configure DataService with Algoseek source
    config = DataConfig(
        mode="adjusted",  # Which adjustment mode for strategy logic
        frequency="1d",  # Daily bars
        timezone="America/New_York",
        source_tag="algoseek-adjusted",  # ← KEY: This selects Algoseek
        bar_schema=bar_schema,
    )

    # Initialize service
    service = DataService(config)

    # Load data - will use Algoseek adapter automatically
    iterator = service.load_symbol("AAPL", date(2020, 1, 1), date(2020, 1, 31))

    # Verify source
    instrument = service.get_instrument("AAPL")
    print(f"✓ Data source: {instrument.data_source.value}")  # Shows 'algoseek'

    bar_count = sum(1 for _ in iterator)
    print(f"✓ Loaded {bar_count} bars from Algoseek")


def example_2_schwab() -> None:
    """Example 2: Load from Schwab API (requires API credentials)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Loading from Schwab API")
    print("=" * 60)

    # Same bar schema
    bar_schema = BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    # Configure DataService with Schwab source
    config = DataConfig(
        mode="adjusted",
        frequency="1d",
        timezone="America/New_York",
        source_tag="schwab-live",  # ← KEY: This selects Schwab
        bar_schema=bar_schema,
    )

    # Initialize service
    service = DataService(config)

    # Verify source will be Schwab
    instrument = service.get_instrument("AAPL")
    print(f"✓ Data source: {instrument.data_source.value}")  # Shows 'schwab'

    print("Note: Schwab adapter implementation coming in future phase")
    print("      Will use API with credentials from environment variables")


def example_3_understanding_source_tag() -> None:
    """Example 3: Understanding how source_tag works."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: How source_tag selection works")
    print("=" * 60)

    print("\nThe source_tag field has two parts:")
    print("  Format: '<source>-<description>'")
    print("  Example: 'algoseek-adjusted'")
    print()
    print("Part 1: Source name (before hyphen)")
    print("  - 'algoseek' → Uses Algoseek adapter")
    print("  - 'schwab'   → Uses Schwab adapter")
    print("  - 'csv'      → Uses CSV file adapter")
    print()
    print("Part 2: Description (after hyphen)")
    print("  - Just for documentation/clarity")
    print("  - Examples: 'adjusted', 'live', 'historical', 'cached'")
    print()
    print("How it works internally:")
    print("  1. DataService extracts source name from source_tag")
    print("     'algoseek-adjusted' → 'algoseek'")
    print()
    print("  2. Looks up config in data_sources.yaml:")
    print("     data_sources:")
    print("       algoseek:")
    print("         adapter: algoseekOHLC")
    print("         root_path: 'data/...'")
    print()
    print("  3. Creates appropriate adapter with config")
    print("  4. DataLoader uses adapter to load data")


def example_4_data_sources_yaml() -> None:
    """Example 4: Configuration via data_sources.yaml."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: data_sources.yaml configuration")
    print("=" * 60)

    print("\nThe data_sources.yaml file maps logical sources to physical adapters:")
    print()
    print("# config/data_sources.yaml")
    print("data_sources:")
    print()
    print("  # Algoseek: Local parquet files")
    print("  algoseek:")
    print("    adapter: algoseekOHLC")
    print("    root_path: 'data/us-equity-daily-ohlc-...'")
    print("    symbol_map: 'data/equity_security_master_sample.csv'")
    print()
    print("  # Schwab: API with caching")
    print("  schwab:")
    print("    adapter: schwabOHLC")
    print("    cache_root: 'data/us-equity-daily-adjusted-schwab'")
    print("    api:")
    print("      client_id: '${SCHWAB_API_KEY}'  # From environment")
    print()
    print("Benefits:")
    print("  ✓ Switch between dev/prod data by changing source_tag")
    print("  ✓ Environment-specific configuration (${ENV_VAR})")
    print("  ✓ No code changes needed to change data source")
    print("  ✓ Easy to add new data sources")


def example_5_multiple_sources() -> None:
    """Example 5: Using different sources for different symbols (future)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Multiple sources (future enhancement)")
    print("=" * 60)

    print("\nCurrently, DataService uses one source for all symbols.")
    print("Future enhancement will support per-symbol source selection:")
    print()
    print("# Load AAPL from Algoseek")
    print("aapl_iterator = service.load_symbol(")
    print("    'AAPL',")
    print("    date(2020, 1, 1),")
    print("    date(2020, 1, 31),")
    print("    data_source='algoseek',  # ← Future: override at runtime")
    print(")")
    print()
    print("# Load BTCUSD from crypto exchange")
    print("btc_iterator = service.load_symbol(")
    print("    'BTCUSD',")
    print("    date(2020, 1, 1),")
    print("    date(2020, 1, 31),")
    print("    data_source='coinbase',  # ← Future: different source")
    print(")")
    print()
    print("Status: Not yet implemented (Phase 2 or later)")


def example_6_practical_usage() -> None:
    """Example 6: Practical real-world usage patterns."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Practical usage patterns")
    print("=" * 60)

    print("\nPattern 1: Development with local data")
    print("  source_tag='algoseek-adjusted'")
    print("  → Uses local parquet files")
    print("  → Fast, no API calls")
    print("  → Good for backtesting")
    print()
    print("Pattern 2: Live trading with Schwab")
    print("  source_tag='schwab-live'")
    print("  → Uses Schwab API")
    print("  → Real-time data")
    print("  → Requires API credentials")
    print()
    print("Pattern 3: Testing with small dataset")
    print("  source_tag='csv-test'")
    print("  → Uses CSV files")
    print("  → Easy to create test data")
    print("  → Good for unit tests")
    print()
    print("Pattern 4: Production backtest with cached data")
    print("  source_tag='schwab-cached'")
    print("  → Uses Schwab API with local cache")
    print("  → First load hits API, subsequent loads use cache")
    print("  → Balance between fresh data and speed")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" DataService: Understanding Data Source Selection")
    print("=" * 70)

    # Example 1: Actually load from Algoseek
    example_1_algoseek()

    # Example 2: Show how to configure for Schwab
    example_2_schwab()

    # Example 3: Explain source_tag format
    example_3_understanding_source_tag()

    # Example 4: Explain data_sources.yaml
    example_4_data_sources_yaml()

    # Example 5: Future enhancements
    example_5_multiple_sources()

    # Example 6: Practical patterns
    example_6_practical_usage()

    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print("1. Set source_tag in DataConfig to select data source")
    print("2. source_tag format: '<source>-<description>'")
    print("3. Source name (before hyphen) must match data_sources.yaml")
    print("4. Configure adapters in config/data_sources.yaml")
    print("5. Use environment variables for API credentials")
    print("=" * 70)


if __name__ == "__main__":
    main()
