#!/usr/bin/env python3
"""
Test script for Phase 3: BacktestEngine + StrategyService integration.

This script runs a simple backtest to verify:
1. StrategyRegistry discovers strategies from my_library/strategies
2. BacktestEngine instantiates strategies from portfolio.yaml
3. DataService streams bars to StrategyService
4. Strategies receive bars and emit signals
5. All events are logged
"""

from pathlib import Path

from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine


def main():
    print("=" * 80)
    print("Phase 3 Integration Test: BacktestEngine + StrategyService")
    print("=" * 80)
    print()

    # Load config
    config_path = Path("config/portfolio.yaml")
    print(f"Loading config from: {config_path}")
    config = load_backtest_config(config_path)

    print(f"  Start Date: {config.start_date}")
    print(f"  End Date: {config.end_date}")
    print(f"  Initial Equity: ${config.initial_equity:,.2f}")
    print(f"  Data Sources: {[s.name for s in config.data.sources]}")
    print(f"  Universe: {sorted(config.all_symbols)}")
    print(f"  Strategies: {[s.strategy_id for s in config.strategies]}")
    print()

    # Create engine
    print("Creating BacktestEngine...")
    with BacktestEngine.from_config(config) as engine:
        print("  ✓ Engine initialized")
        print()

        # Run backtest
        print("Running backtest...")
        result = engine.run()
        print()

        # Print results
        print("=" * 80)
        print("Backtest Complete!")
        print("=" * 80)
        print(f"  Bars Processed: {result.bars_processed}")
        print(f"  Duration: {result.duration.total_seconds():.2f} seconds")
        if result.bars_processed > 0:
            bars_per_sec = result.bars_processed / result.duration.total_seconds()
            print(f"  Performance: {bars_per_sec:,.0f} bars/second")
        print()

        # Print strategy metrics
        if engine._strategy_service:
            metrics = engine._strategy_service.get_metrics()
            print("Strategy Metrics:")
            print("-" * 80)
            for strategy_name, strategy_metrics in metrics.items():
                print(f"  {strategy_name}:")
                print(f"    Bars Processed: {strategy_metrics['bars_processed']}")
                print(f"    Signals Emitted: {strategy_metrics['signals_emitted']}")
                print(f"    Errors: {strategy_metrics['errors']}")
            print()

    print("✓ Test Complete!")


if __name__ == "__main__":
    main()
