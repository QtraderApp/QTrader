#!/usr/bin/env python3
"""
QTrader Basic Run Example - Minimal Backtest Setup

This example demonstrates the simplest possible backtest execution:
1. Load configuration
2. Create engine (automatically initializes EventBus, EventStore, DataService)
3. Run backtest
4. Review results

The engine handles all service orchestration internally.
"""

from pathlib import Path

from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine
from qtrader.system.config import get_system_config


def main():
    """Run a basic backtest with minimal setup."""

    print("=" * 80)
    print("QTrader - Basic Backtest Example")
    print("=" * 80)
    print()

    # ============================================================================
    # Step 1: Load Configuration
    # ============================================================================
    print("Step 1: Loading configuration from config/portfolio.yaml...")

    config_path = Path("config/portfolio.yaml")
    config = load_backtest_config(config_path)

    print(f"✓ Configuration loaded successfully")
    print(f"  Date Range: {config.start_date} to {config.end_date}")
    print(f"  Universe: {list(config.all_symbols)}")
    print(f"  Data Source: {config.data.sources[0].name}")
    print()

    # ============================================================================
    # Step 2: Create Engine
    # ============================================================================
    # The engine factory method (from_config) automatically:
    # - Creates EventBus for pub/sub messaging
    # - Initializes EventStore (SQLite) for event persistence
    # - Creates DataService with proper dataset configuration
    # - Sets up results directory with timestamps
    # - Configures logging

    print("Step 2: Creating backtest engine...")
    print("  (Initializing EventBus, EventStore, DataService...)")

    engine = BacktestEngine.from_config(config)

    print(f"✓ Engine initialized")
    print()

    # ============================================================================
    # Step 3: Run Backtest
    # ============================================================================
    # The run() method:
    # 1. Streams historical bars from DataService
    # 2. Publishes PriceBarEvent for each bar (timestamp-synchronized)
    # 3. EventStore persists all events to SQLite
    # 4. Returns metrics (bars processed, duration)

    print("Step 3: Running backtest...")
    print("  (Streaming historical data and publishing events...)")
    print()

    try:
        result = engine.run()

        print("✓ Backtest completed successfully!")
        print()

        # ========================================================================
        # Step 4: Review Results
        # ========================================================================
        print("=" * 80)
        print("BACKTEST RESULTS")
        print("=" * 80)
        print(f"Date Range:      {result.start_date} to {result.end_date}")
        print(f"Bars Processed:  {result.bars_processed:,}")
        print(f"Duration:        {result.duration}")
        print()

        # Check if event database was created
        if hasattr(engine, "_results_dir") and engine._results_dir:
            # Load system config to get event store filename
            system_config = get_system_config()
            event_store_filename = system_config.output.event_store.filename
            event_db = engine._results_dir / event_store_filename

            if event_db.exists():
                import os

                size_mb = os.path.getsize(event_db) / (1024 * 1024)
                print(f"Event Database:  {event_db}")
                print(f"Database Size:   {size_mb:.2f} MB")

        print("=" * 80)
        print()

        return 0

    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    finally:
        # ========================================================================
        # Step 5: Cleanup
        # ========================================================================
        # Always shutdown to close SQLite connections properly
        print("Step 5: Cleaning up resources...")
        engine.shutdown()
        print("✓ Engine shutdown complete")
        print()


if __name__ == "__main__":
    print()
    print("This example demonstrates the simplest possible backtest workflow.")
    print("All complexity is hidden inside BacktestEngine.from_config()")
    print()
    exit(main())
