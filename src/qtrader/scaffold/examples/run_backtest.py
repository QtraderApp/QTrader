#!/usr/bin/env python3
"""
Simple backtest runner for QTrader.

Usage:
    python run_backtest.py config/backtests/buy_hold.yaml
    python run_backtest.py config/backtests/sma_crossover.yaml
"""

import sys
from pathlib import Path

from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine


def main():
    """Run a backtest from a configuration file."""
    if len(sys.argv) < 2:
        print("Usage: python run_backtest.py <backtest_config.yaml>")
        print("\nExamples:")
        print("  python run_backtest.py config/backtests/buy_hold.yaml")
        print("  python run_backtest.py config/backtests/sma_crossover.yaml")
        sys.exit(1)

    config_path = Path(sys.argv[1])

    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading backtest configuration from: {config_path}")
    print("=" * 80)

    # Load backtest configuration (system config is loaded automatically)
    backtest_config = load_backtest_config(config_path)

    # Create and run engine
    engine = BacktestEngine.from_config(backtest_config)
    engine.run()

    print("\n" + "=" * 80)
    print("Backtest complete!")
    print(f"Results saved to: output/backtests/{backtest_config.backtest_id}/")


if __name__ == "__main__":
    main()
