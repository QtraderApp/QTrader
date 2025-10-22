"""
DataService Example - Understanding Event-Driven Data Loading

This example demonstrates how the DataService works:
1. Loads historical price data from Algoseek (unadjusted)
2. Publishes PriceBarEvent for each bar via EventBus
3. Shows how to subscribe to these events

DataService is the ONLY service that reads from disk/API.
Other services (Strategy, Risk, Portfolio, Execution) subscribe to
PriceBarEvent and react to the data.

Key Concepts:
- DataService publishes events (doesn't know about subscribers)
- EventBus routes events to subscribers
- Events are published in chronological order
- All bars for timestamp T published before moving to T+1

Usage:
    python examples/services/data/data_service_example.py
"""

import sys
from datetime import date
from pathlib import Path

import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from qtrader.events.event_bus import EventBus
from qtrader.services.data.service import DataService
from qtrader.system import LoggerFactory, LoggingConfig

# Configure logging
LoggerFactory.configure(LoggingConfig(level="INFO", format="console"))
logger = LoggerFactory.get_logger()  # Auto-detects module name


def main():
    """
    Demonstrate DataService with EventBus.

    Shows:
    1. How to initialize DataService with EventBus
    2. How to subscribe to PriceBarEvent
    3. How data flows through the system
    4. Event ordering guarantees
    """
    logger.info("=== DataService Example ===")

    # Step 1: Load configuration
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    logger.info(
        "config.loaded",
        mode=config["data"]["default_mode"],
        timezone=config["data"]["default_timezone"],
    )

    # Step 2: Create EventBus
    # EventBus is the message broker - routes events from publishers to subscribers
    event_bus = EventBus()
    logger.info("event_bus.created")

    # Step 3: Subscribe to PriceBarEvent
    # This function will be called for EVERY bar that DataService publishes
    bar_count = {"count": 0, "symbols": set()}

    def handle_price_bar(event) -> None:
        """Handle incoming price bar events."""
        bar_count["count"] += 1
        bar_count["symbols"].add(event.symbol)

        # Print first 5 bars to show what we receive
        if bar_count["count"] <= 5:
            logger.info(
                "price_bar.received",
                symbol=event.symbol,
                date=event.bar.trade_datetime.isoformat(),
                open=float(event.bar.open),
                high=float(event.bar.high),
                low=float(event.bar.low),
                close=float(event.bar.close),
                volume=int(event.bar.volume),
                is_warmup=event.is_warmup,
            )

    # Subscribe to "price_bar" events
    event_bus.subscribe("price_bar", handle_price_bar)
    logger.info("event_bus.subscribed", event_type="price_bar")

    # Step 4: Create DataService
    # DataService needs dataset name from data_sources.yaml
    # We use Algoseek unadjusted data for this example
    from qtrader.services.data.config import BarSchemaConfig
    from qtrader.services.data.config import DataConfig as ServiceDataConfig
    from qtrader.services.data.source_selector import AssetClass, DataSourceSelector

    # Create source selector for Algoseek
    source_selector = DataSourceSelector(
        provider="algoseek",
        asset_class=AssetClass.EQUITY,
    )

    # Create bar schema (standard OHLCV)
    bar_schema = BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    # Create service-level DataConfig
    service_config = ServiceDataConfig(
        mode="unadjusted",  # Algoseek provides unadjusted data
        frequency="1d",
        timezone=config["data"]["default_timezone"],
        source_selector=source_selector,
        bar_schema=bar_schema,
    )

    # Create DataService with event bus
    data_service = DataService(
        config=service_config,
        dataset="algoseek-us-equity-1d-unadjusted",  # From data_sources.yaml
        event_bus=event_bus,
    )

    logger.info(
        "data_service.created",
        dataset="algoseek-us-equity-1d-unadjusted",
        mode=config["data"]["default_mode"],
    )

    # Step 5: Stream historical data
    # stream_universe() loads data and publishes PriceBarEvent for each bar
    # Events are published in timestamp order (all bars for T, then T+1, etc.)
    symbols = ["AAPL", "MSFT"]
    start_date = date(2023, 1, 3)  # First trading day of 2023
    end_date = date(2023, 1, 31)  # January 2023

    logger.info(
        "streaming.start",
        symbols=symbols,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    # This call:
    # 1. Loads data from disk (Algoseek parquet files)
    # 2. Synchronizes bars by timestamp
    # 3. Publishes PriceBarEvent for each bar
    # 4. Our handle_price_bar() function receives each event
    data_service.stream_universe(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        is_warmup=False,  # Not warmup data
    )

    # Step 6: Summary
    logger.info(
        "streaming.complete",
        total_bars=bar_count["count"],
        unique_symbols=len(bar_count["symbols"]),
        symbols=sorted(bar_count["symbols"]),
    )

    # Show what happened
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Dataset: algoseek-us-equity-1d-unadjusted")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Total bars received: {bar_count['count']}")
    print(f"Unique symbols: {len(bar_count['symbols'])}")
    print("\nHow it works:")
    print("1. DataService loads data from Algoseek parquet files")
    print("2. For each timestamp, publishes PriceBarEvent for all symbols")
    print("3. EventBus routes events to all subscribers")
    print("4. Our handle_price_bar() function processes each event")
    print("\nEvent Ordering:")
    print("- All bars for timestamp T published before T+1")
    print("- Ensures strategies see complete market snapshot")
    print("- No race conditions between symbols")
    print("=" * 60)


def demo_non_event_mode():
    """
    Demonstrate DataService without EventBus (pull-based mode).

    Sometimes you just want to load data without events.
    This is useful for:
    - Data exploration
    - Testing
    - Non-real-time analysis
    """
    logger.info("\n=== Non-Event Mode Example ===")

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create DataService WITHOUT EventBus
    from qtrader.services.data.config import BarSchemaConfig
    from qtrader.services.data.config import DataConfig as ServiceDataConfig
    from qtrader.services.data.source_selector import AssetClass, DataSourceSelector

    source_selector = DataSourceSelector(
        provider="algoseek",
        asset_class=AssetClass.EQUITY,
    )

    bar_schema = BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    service_config = ServiceDataConfig(
        mode="unadjusted",
        frequency="1d",
        timezone=config["data"]["default_timezone"],
        source_selector=source_selector,
        bar_schema=bar_schema,
    )

    data_service = DataService(
        config=service_config,
        dataset="algoseek-us-equity-1d-unadjusted",
        event_bus=None,  # No EventBus = pull-based mode
    )

    logger.info("data_service.created", mode="pull-based (no events)")

    # Load data for single symbol
    symbol = "AAPL"
    start_date = date(2023, 1, 3)
    end_date = date(2023, 1, 10)

    logger.info(
        "loading.start",
        symbol=symbol,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    # load_symbol() returns an iterator
    # No events published - you pull data as needed
    iterator = data_service.load_symbol(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
    )

    # Iterate through bars
    bars = list(iterator)
    logger.info("loading.complete", bars_loaded=len(bars))

    # Show first few bars
    print("\n" + "=" * 60)
    print(f"Loaded {len(bars)} bars for {symbol}")
    print("=" * 60)
    for i, multi_bar in enumerate(bars[:5], 1):
        bar = multi_bar.adjusted  # Use adjusted view
        print(
            f"{i}. {bar.trade_datetime}: O={bar.open:.2f} H={bar.high:.2f} "
            f"L={bar.low:.2f} C={bar.close:.2f} V={bar.volume:,}"
        )
    print("=" * 60)


def demo_universe_loading():
    """
    Demonstrate loading multiple symbols without events.

    Shows how to get iterators for multiple symbols and
    process them synchronously (same timestamp across symbols).
    """
    logger.info("\n=== Universe Loading Example ===")

    # Load config
    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create DataService without EventBus
    from qtrader.services.data.config import BarSchemaConfig
    from qtrader.services.data.config import DataConfig as ServiceDataConfig
    from qtrader.services.data.source_selector import AssetClass, DataSourceSelector

    source_selector = DataSourceSelector(
        provider="algoseek",
        asset_class=AssetClass.EQUITY,
    )

    bar_schema = BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    service_config = ServiceDataConfig(
        mode="unadjusted",
        frequency="1d",
        timezone=config["data"]["default_timezone"],
        source_selector=source_selector,
        bar_schema=bar_schema,
    )

    data_service = DataService(
        config=service_config,
        dataset="algoseek-us-equity-1d-unadjusted",
        event_bus=None,
    )

    # Load multiple symbols
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = date(2023, 1, 3)
    end_date = date(2023, 1, 5)

    logger.info(
        "universe.loading",
        symbols=symbols,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
    )

    # load_universe() returns dict of {symbol: iterator}
    iterators = data_service.load_universe(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
    )

    # Process each symbol
    print("\n" + "=" * 60)
    print("Universe Data")
    print("=" * 60)
    for symbol, iterator in iterators.items():
        bars = list(iterator)
        print(f"\n{symbol}: {len(bars)} bars")
        for multi_bar in bars[:3]:  # Show first 3
            bar = multi_bar.adjusted
            print(f"  {bar.trade_datetime}: Close=${bar.close:.2f} Volume={bar.volume:,}")
    print("=" * 60)


if __name__ == "__main__":
    # Run main event-driven example
    main()

    # Run additional examples
    demo_non_event_mode()
    demo_universe_loading()

    print("\n✅ DataService examples complete!")
    print("\nKey Takeaways:")
    print("1. DataService is the ONLY service that reads from disk/API")
    print("2. With EventBus: DataService publishes PriceBarEvent")
    print("3. Without EventBus: Use load_symbol() / load_universe() for pull-based access")
    print("4. Events published in timestamp order (all symbols at T, then T+1)")
    print("5. Other services (Strategy, Risk, etc.) subscribe to events")
    print("\nNext Steps:")
    print("- See examples/services/strategy/ for strategy examples")
    print("- See examples/services/risk/ for risk management examples")
    print("- See src/qtrader/backtest/engine.py for full backtest orchestration")
