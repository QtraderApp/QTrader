"""
Golden baseline generator for regression testing.

This script runs reference strategies and saves their results as golden files.
These golden files are used in tests to ensure deterministic behavior.

Usage:
    python scripts/generate_goldens.py [--strategy STRATEGY_NAME] [--verbose]

Examples:
    # Generate all golden baselines
    python scripts/generate_goldens.py

    # Generate specific strategy
    python scripts/generate_goldens.py --strategy buy_and_hold

    # Verbose output
    python scripts/generate_goldens.py --verbose
"""

import argparse
import importlib.util
import json
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Type

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.api.strategy import Strategy
from qtrader.config.logging_config import LoggerFactory
from qtrader.execution.config import ExecutionConfig
from qtrader.models.portfolio import Portfolio

logger = LoggerFactory.get_logger()


def decimal_to_float(obj: Any) -> Any:
    """Convert Decimal objects to float for JSON serialization."""
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: decimal_to_float(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [decimal_to_float(item) for item in obj]
    return obj


def load_strategy_module(strategy_path: Path):
    """
    Load strategy module from file path.

    Args:
        strategy_path: Path to strategy Python file

    Returns:
        Tuple of (strategy_class, config, backtest_config)
    """
    spec = importlib.util.spec_from_file_location("strategy_module", strategy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strategy from {strategy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Find Strategy class
    strategy_class = None
    for item_name in dir(module):
        item = getattr(module, item_name)
        if (
            isinstance(item, type)
            and hasattr(item, "on_bar")
            and item.__module__ == module.__name__
            and item_name not in ("Strategy",)
        ):
            strategy_class = item
            break

    if strategy_class is None:
        raise ValueError(f"No Strategy subclass found in {strategy_path}")

    config = getattr(module, "config", {})
    backtest_config = getattr(module, "backtest_config", {})

    return strategy_class, config, backtest_config


def load_data_and_run_backtest(
    strategy_class: Type[Strategy],
    backtest_config: Dict[str, Any],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run backtest with the given strategy and config.

    Args:
        strategy_class: Strategy class to instantiate
        backtest_config: Backtest configuration dict
        verbose: Enable verbose logging

    Returns:
        Dictionary with backtest results
    """
    from qtrader.config.data_config import BarSchemaConfig, DataConfig

    # Load data source resolver
    resolver = DataSourceResolver()

    # Get instruments from config
    instruments = backtest_config.get("instruments", [])
    if not instruments:
        raise ValueError("No instruments specified in backtest_config")

    # Create default bar schema config
    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="MarketHoursVolume",
    )
    data_config = DataConfig(bar_schema=bar_schema)

    # Load bars for all instruments
    all_bars = []

    for instrument in instruments:
        print(f"Loading data for {instrument.symbol} from {instrument.data_source.name}...")

        # Get adapter for this instrument
        adapter = resolver.resolve(instrument)

        # Read bars
        bars = list(adapter.read_bars(data_config))
        all_bars.extend(bars)

    # Sort all bars by timestamp
    all_bars.sort(key=lambda b: (b.ts, b.symbol))

    print(f"Loaded {len(all_bars)} bars for {len(instruments)} symbols")

    # Create execution config
    exec_config = ExecutionConfig(
        warmup=backtest_config.get("warmup", False),
        warmup_bars=backtest_config.get("warmup_bars"),
        max_participation=Decimal(str(backtest_config.get("max_participation", 0.10))),
    )

    # Create portfolio
    portfolio = Portfolio(initial_cash=Decimal(str(backtest_config.get("initial_cash", 100000.0))))

    # Create context with portfolio and risk manager
    from qtrader.risk.manager import RiskManager
    from qtrader.risk.policy import RiskPolicy

    risk_policy = RiskPolicy(
        default_position_size=Decimal(str(backtest_config.get("position_size", 0.10))),
        max_position_pct=Decimal(str(backtest_config.get("max_position_pct", 0.20))),
        allow_shorting=backtest_config.get("allow_shorting", False),
    )

    risk_manager = RiskManager(
        portfolio=portfolio,
        policy=risk_policy,
    )

    ctx = Context(
        portfolio=portfolio,
        risk_manager=risk_manager,
    )

    # Instantiate strategy
    strategy = strategy_class()

    # Create backtest runner
    backtest = Backtest(config=exec_config, strategy=strategy)

    # Create output directory
    out_dir = Path("backtest_results") / "golden_generation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run backtest
    print("\nRunning backtest...")
    symbols = [inst.symbol for inst in instruments]
    metadata = backtest.run(
        ctx=ctx,
        bars=all_bars,
        symbols=symbols,
        out_dir=out_dir,
    )

    # Extract results
    results = {
        "metadata": metadata,
        "final_cash": float(portfolio.cash.get_balance()),
        "final_equity": float(portfolio.get_equity()),
        "total_return_pct": float(
            ((portfolio.get_equity() - Decimal(str(backtest_config["initial_cash"])))
             / Decimal(str(backtest_config["initial_cash"]))) * 100
        ),
        "num_trades": len([fill for fill in backtest.all_fills if fill.qty != 0]),
        "num_fills": len(backtest.all_fills),
        "total_commissions": float(sum(fill.fees for fill in backtest.all_fills)),
        "final_positions": {
            symbol: {
                "quantity": float(pos.qty),
                "avg_price": float(pos.avg_price),
            }
            for symbol, pos in portfolio.positions.get_all_positions().items()
            if not pos.is_flat()
        },
        "snapshots": backtest.portfolio_snapshots,
    }

    if verbose:
        print("\n=== Results ===")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Trades: {results['num_trades']}")
        print(f"Fills: {results['num_fills']}")
        print(f"Commissions: ${results['total_commissions']:,.2f}")
        print(f"Final Positions: {len(results['final_positions'])}")

    return results


def generate_golden(
    strategy_name: str,
    strategy_path: Path,
    output_dir: Path,
    verbose: bool = False,
):
    """
    Generate golden baseline file for a strategy.

    Args:
        strategy_name: Name of the strategy (for output filename)
        strategy_path: Path to strategy Python file
        output_dir: Directory to save golden file
        verbose: Enable verbose output
    """
    print(f"\n{'='*60}")
    print(f"Generating golden baseline: {strategy_name}")
    print(f"{'='*60}")

    # Load strategy
    strategy_class, config, backtest_config = load_strategy_module(strategy_path)
    print(f"Loaded strategy: {strategy_class.__name__}")

    # Run backtest
    results = load_data_and_run_backtest(strategy_class, backtest_config, verbose)

    # Create golden file structure
    golden = {
        "metadata": {
            "strategy": strategy_name,
            "strategy_class": strategy_class.__name__,
            "generated": datetime.now().isoformat(),
            "version": "1.0",
            "description": strategy_class.__doc__.strip() if strategy_class.__doc__ else "",
        },
        "config": {
            "strategy_config": config,
            "backtest_config": {
                # Convert instruments to serializable format
                "instruments": [
                    {
                        "symbol": inst.symbol,
                        "type": inst.instrument_type.name,
                        "source": inst.data_source.name,
                        "frequency": inst.frequency,
                    }
                    for inst in backtest_config.get("instruments", [])
                ],
                "initial_cash": backtest_config.get("initial_cash"),
                "position_size": backtest_config.get("position_size"),
                "max_position_pct": backtest_config.get("max_position_pct"),
                "allow_shorting": backtest_config.get("allow_shorting"),
                "start_date": backtest_config.get("start_date"),
                "end_date": backtest_config.get("end_date"),
            },
        },
        "results": {
            "final_cash": results["final_cash"],
            "final_equity": results["final_equity"],
            "total_return_pct": results["total_return_pct"],
            "num_trades": results["num_trades"],
            "num_fills": results["num_fills"],
            "total_commissions": results["total_commissions"],
        },
        "final_positions": results["final_positions"],
        # Store key snapshots (every ~10% of the way through)
        "key_snapshots": (
            results["snapshots"][:: max(1, len(results["snapshots"]) // 10)]
            if results["snapshots"]
            else []
        ),
    }

    # Convert any remaining Decimals to floats
    golden = decimal_to_float(golden)

    # Save golden file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{strategy_name}_golden.json"

    with open(output_file, "w") as f:
        json.dump(golden, f, indent=2)

    print(f"\n✅ Golden file saved: {output_file}")
    print(f"   Final Equity: ${golden['results']['final_equity']:,.2f}")
    print(f"   Total Return: {golden['results']['total_return_pct']:.2f}%")
    print(f"   Trades: {golden['results']['num_trades']}")

    return golden


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate golden baseline files")
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["buy_and_hold", "sma_crossover", "all"],
        default="all",
        help="Strategy to generate golden for (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Define strategies to generate
    strategies = {
        "buy_and_hold": Path("examples/buy_and_hold_strategy.py"),
        "sma_crossover": Path("examples/sma_crossover_strategy.py"),
    }

    # Filter if specific strategy requested
    if args.strategy != "all":
        strategies = {args.strategy: strategies[args.strategy]}

    # Output directory
    output_dir = Path("tests/integration/goldens")

    # Generate goldens
    print(f"\n{'='*60}")
    print("Golden Baseline Generator")
    print(f"{'='*60}")
    print(f"Strategies: {', '.join(strategies.keys())}")
    print(f"Output directory: {output_dir}")

    generated = []
    for strategy_name, strategy_path in strategies.items():
        try:
            generate_golden(strategy_name, strategy_path, output_dir, args.verbose)
            generated.append(strategy_name)
        except Exception as e:
            print(f"\n❌ Error generating {strategy_name}: {e}")
            import traceback

            traceback.print_exc()

    # Summary
    print(f"\n{'='*60}")
    print(f"Generated {len(generated)}/{len(strategies)} golden baselines")
    print(f"{'='*60}")

    if len(generated) < len(strategies):
        sys.exit(1)


if __name__ == "__main__":
    main()
