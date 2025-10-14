"""QTrader CLI - Clean implementation."""

import ast
import importlib.util
import inspect
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional

import click

from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.api.multi_strategy_backtest import MultiStrategyBacktest
from qtrader.config.backtest_config import BacktestConfig
from qtrader.config.system_config import get_config
from qtrader.data import DataLoader
from qtrader.execution.config import ExecutionConfig
from qtrader.models.portfolio import Portfolio
from qtrader.risk.manager import RiskManager
from qtrader.risk.policy import RiskPolicy


def _load_module(path: str):
    """Load Python module from file path at runtime.

    This function performs dynamic module loading using Python's importlib machinery,
    allowing strategy files to be imported from arbitrary file paths without requiring
    them to be installed as packages or in the Python path.

    The loading process follows three steps:
    1. Create a module specification that describes how to load the module
    2. Create an empty module object from that specification
    3. Execute the module's code to populate it with classes and functions

    Args:
        path: Absolute or relative path to a Python file (.py)

    Returns:
        The loaded module object with all its attributes (classes, functions, variables)

    Raises:
        ImportError: If the file cannot be loaded (invalid path, no loader found, etc.)

    Example:
        >>> module = _load_module("strategies/my_strategy.py")
        >>> strategy_class = module.MyStrategy
        >>> config = module.config

    Note:
        The module is registered in sys.modules as "strategy", making it available
        for inspection and allowing relative imports within the loaded module to work.
        This also means loading a new strategy will replace the previous one.
    """
    spec = importlib.util.spec_from_file_location("strategy", path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy"] = module
    spec.loader.exec_module(module)
    return module


def _find_strategy_class(module):
    """Find and extract the Strategy class from a dynamically loaded module.

    This function uses Python's inspect module to locate a class that implements
    the Strategy interface by checking for the presence of the required on_bar method.
    It ensures only user-defined strategy classes (not imported ones) are detected.

    The detection process:
    1. Use inspect.getmembers to get all classes defined in the module
    2. Filter classes that have an on_bar method (Strategy interface requirement)
    3. Ensure the class was defined in this module (not imported from elsewhere)
    4. Validate that exactly one strategy class exists

    Args:
        module: A module object loaded via _load_module containing strategy code

    Returns:
        The Strategy class object (not an instance) that can be instantiated later

    Raises:
        ValueError: If no strategy class is found in the module
        ValueError: If multiple strategy classes are found (ambiguous)

    Example:
        >>> module = _load_module("strategies/my_strategy.py")
        >>> strategy_class = _find_strategy_class(module)
        >>> strategy = strategy_class(fast_period=10, slow_period=30)
        >>> strategy.on_bar(context, bar)

    Note:
        The function checks o.__module__ == module.__name__ to filter out imported
        classes like Strategy base class or helper classes from other modules.
        This ensures only classes defined directly in the strategy file are detected.

        Common reasons for "No Strategy class found":
        - Missing on_bar method in your strategy class
        - Strategy class imported from another file (not defined locally)
        - File contains only functions or configuration, no class

        Common reasons for "Multiple classes found":
        - Multiple strategy implementations in the same file (not recommended)
        - Helper classes that also implement on_bar method
        - Solution: Keep one strategy per file or rename helper methods
    """
    classes = [
        (n, o)
        for n, o in inspect.getmembers(module, inspect.isclass)
        if hasattr(o, "on_bar") and o.__module__ == module.__name__
    ]
    if not classes:
        raise ValueError("No Strategy class found")
    if len(classes) > 1:
        raise ValueError(f"Multiple classes found: {[n for n, _ in classes]}")
    return classes[0][1]


def _apply_overrides(config: Dict[str, Any], overrides: tuple) -> Dict[str, Any]:
    """Apply --set overrides to config."""

    config = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid: {override}")
        key, val = override.split("=", 1)
        try:
            config[key.strip()] = ast.literal_eval(val.strip())
        except (ValueError, SyntaxError):
            config[key.strip()] = val.strip()
    return config


@click.group()
@click.version_option(version="0.1.0")
def main():
    """QTrader - Quantitative Trading Backtest System"""
    pass


@main.command()
@click.option(
    "--strategy-file",
    type=click.Path(exists=True),
    help="Path to Python strategy file (legacy mode)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to YAML configuration file (recommended)",
)
@click.option("--set", "overrides", multiple=True, help="Override config: --set key=value")
@click.option("--output-dir", type=click.Path(), help="Output directory")
def backtest(
    strategy_file: Optional[str],
    config: Optional[str],
    overrides: tuple,
    output_dir: str,
):
    """Run backtest with strategy file or YAML config.

    Two modes are supported:

    1. YAML Config Mode (recommended):
       qtrader backtest --config config/backtests/single_strategy_sma.yaml

    2. Legacy Python File Mode (backward compatible):
       qtrader backtest --strategy-file examples/sma_crossover_strategy.py

    The YAML config mode supports multi-strategy backtests, type-safe configuration,
    and dynamic capital reallocation. See config/backtests/README.md for details.
    """
    click.echo("=" * 60)
    click.echo("QTrader Backtest Engine")
    click.echo("=" * 60)

    # Validate mutually exclusive options
    if not strategy_file and not config:
        click.echo("✗ Error: Must specify either --strategy-file or --config", err=True)
        sys.exit(1)

    if strategy_file and config:
        click.echo("✗ Error: Cannot use both --strategy-file and --config", err=True)
        sys.exit(1)

    # Route to appropriate mode
    if config:
        _run_yaml_backtest(config, output_dir)
    else:
        # strategy_file is guaranteed to be str here due to validation above
        _run_legacy_backtest(strategy_file, overrides, output_dir)  # type: ignore[arg-type]


def _run_yaml_backtest(config_path: str, output_dir: Optional[str]):
    """Run backtest using YAML configuration (multi-strategy support)."""
    click.echo("\n[YAML Config Mode: Type-Safe Multi-Strategy]")

    # Load and validate YAML config
    try:
        backtest_config = BacktestConfig.from_yaml(config_path)
        click.echo(f"✓ Config loaded: {config_path}")
        click.echo(f"  Strategies: {len(backtest_config.strategies)}")
        click.echo(f"  Period: {backtest_config.start_date} to {backtest_config.end_date}")
        click.echo(f"  Initial Capital: ${float(backtest_config.initial_capital):,.2f}")
    except Exception as e:
        click.echo(f"✗ Config validation failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Display strategy information
    click.echo("\nStrategies:")
    for strategy in backtest_config.strategies:
        click.echo(
            f"  - {strategy.name}: {len(strategy.instruments)} instruments, "
            f"{float(strategy.initial_allocation_pct) * 100:.0f}% allocation"
        )

    # Setup output directory
    if not output_dir:
        sys_config = get_config()
        ts = datetime.now().strftime(sys_config.output.timestamp_format)
        name = backtest_config.name or "multi_strategy"
        name_slug = name.replace(" ", "_").lower()
        output_dir = f"{sys_config.output.default_results_dir}/{name_slug}_{ts}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output: {out_path}")

    # Load data for all unique symbols across strategies
    click.echo("\nLoading data...")
    all_symbols = backtest_config.all_symbols

    try:
        config = {
            "adapter": {
                "root_path": "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample",
                "path_template": "{root_path}/SecId={secid}/*.parquet",
                "symbol_map": "data/equity_security_master_sample.csv",
            }
        }
        data_iterators = {}
        for symbol in all_symbols:
            loader = DataLoader({**config, "symbol": symbol})
            data_iterators[symbol] = loader.load_data(
                symbol, str(backtest_config.start_date), str(backtest_config.end_date)
            )
        click.echo(f"✓ Loaded {len(all_symbols)} symbol(s)")
    except Exception as e:
        click.echo(f"✗ Data loading failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Initialize multi-strategy backtest
    click.echo("\nInitializing multi-strategy backtest...")
    try:
        multi_backtest = MultiStrategyBacktest(backtest_config)
    except Exception as e:
        click.echo(f"✗ Initialization failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Run backtest
    click.echo("\nRunning backtest...")
    try:
        start = datetime.now()
        metadata = multi_backtest.run(data_iterators=data_iterators, out_dir=out_path)
        duration = (datetime.now() - start).total_seconds()

        click.echo("\n" + "=" * 60)
        click.echo("✓ Complete")
        click.echo("=" * 60)
        click.echo(f"\nDuration: {duration:.2f}s")
        click.echo(f"Bars: {metadata.get('total_bars', 0)}")
        click.echo(f"Strategies: {metadata.get('strategies', 0)}")

        initial = float(backtest_config.initial_capital)
        final = float(metadata.get("final_equity", initial))
        pnl = final - initial
        pnl_pct = (pnl / initial) * 100 if initial > 0 else 0

        click.echo(f"\nInitial Capital: ${initial:,.2f}")
        click.echo(f"Final Equity: ${final:,.2f}")
        click.echo(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")

        # Show per-strategy performance
        if "strategy_metrics" in metadata:
            click.echo("\nPer-Strategy Performance:")
            for name, metrics in metadata["strategy_metrics"].items():
                ret = metrics.get("total_return", 0) * 100
                click.echo(f"  {name}: {ret:+.2f}%")

        click.echo(f"\n✓ Results: {out_path}")

    except Exception as e:
        click.echo(f"\n✗ Backtest failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def _run_legacy_backtest(strategy_file: str, overrides: tuple, output_dir: Optional[str]):
    """Run backtest in legacy Python file mode (backward compatible)."""
    click.echo("\n[Legacy Mode: Python Strategy File]")

    # Load strategy
    try:
        module = _load_module(strategy_file)
        strategy_class = _find_strategy_class(module)
        strategy_config = getattr(module, "config", {})
        backtest_config = getattr(module, "backtest_config", {})
        click.echo(f"✓ Strategy: {strategy_class.__name__}")
    except Exception as e:
        click.echo(f"✗ Failed: {e}", err=True)
        sys.exit(1)

    # Apply overrides
    if overrides:
        strategy_config = _apply_overrides(strategy_config, overrides)

    # Validate
    if "instruments" not in backtest_config:
        click.echo("✗ Error: backtest_config must include 'instruments'", err=True)
        sys.exit(1)

    instruments = backtest_config["instruments"]
    symbols = [inst.symbol for inst in instruments]
    start_date = backtest_config.get("start_date", "2019-01-01")
    end_date = backtest_config.get("end_date", "2023-12-29")

    click.echo(f"\nSymbols: {', '.join(symbols)}")
    click.echo(f"Period: {start_date} to {end_date}")

    # Setup output
    if not output_dir:
        sys_config = get_config()
        ts = datetime.now().strftime(sys_config.output.timestamp_format)
        name = Path(strategy_file).stem
        output_dir = f"{sys_config.output.default_results_dir}/{name}_{ts}"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    click.echo(f"Output: {out_path}")

    # Load data
    click.echo("\nLoading data...")
    try:
        config = {
            "adapter": {
                "root_path": "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample",
                "path_template": "{root_path}/SecId={secid}/*.parquet",
                "symbol_map": "data/equity_security_master_sample.csv",
            }
        }
        data_iterators = {}
        for symbol in symbols:
            loader = DataLoader({**config, "symbol": symbol})
            data_iterators[symbol] = loader.load_data(symbol, start_date, end_date)
        click.echo(f"✓ Loaded {len(symbols)} symbol(s)")
    except Exception as e:
        click.echo(f"✗ Data loading failed: {e}", err=True)
        sys.exit(1)

    # Initialize components
    portfolio = Portfolio(initial_cash=Decimal(str(backtest_config.get("initial_cash", 100000))))
    risk_policy = RiskPolicy(
        default_position_size=Decimal(str(backtest_config.get("position_size", 5000))),
        max_position_pct=Decimal(str(backtest_config.get("max_position_pct", 0.10))),
        allow_shorting=backtest_config.get("allow_shorting", False),
    )
    risk_manager = RiskManager(portfolio=portfolio, policy=risk_policy)
    ctx = Context(portfolio=portfolio, risk_manager=risk_manager)
    exec_config = ExecutionConfig(
        warmup=backtest_config.get("warmup", False),
        warmup_bars=backtest_config.get("warmup_bars"),
    )

    # Initialize strategy
    try:
        strategy_instance = strategy_class(**strategy_config) if strategy_config else strategy_class()
    except TypeError:
        strategy_instance = strategy_class()

    # Run
    click.echo("\nRunning backtest...")
    bt = Backtest(exec_config, strategy_instance)
    try:
        start = datetime.now()
        metadata = bt.run(ctx=ctx, data_iterators=data_iterators, symbols=symbols, out_dir=out_path)
        duration = (datetime.now() - start).total_seconds()

        click.echo("\n" + "=" * 60)
        click.echo("✓ Complete")
        click.echo("=" * 60)
        click.echo(f"\nDuration: {duration:.2f}s")
        click.echo(f"Bars: {metadata.get('trading_bars', 0)}")
        click.echo(f"Fills: {metadata.get('total_fills', 0)}")

        final_cash = float(metadata.get("final_cash", 0))
        total_value = float(metadata.get("final_equity", 0))  # This is actually total portfolio value
        final_equity = total_value - final_cash
        initial = float(backtest_config.get("initial_cash", 100000))
        pnl = total_value - initial
        pnl_pct = (pnl / initial) * 100

        click.echo(f"\nCash: ${final_cash:,.2f}")
        click.echo(f"Equity: ${final_equity:,.2f}")
        click.echo(f"Total: ${total_value:,.2f}")
        click.echo(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        click.echo(f"\n✓ Results: {out_path}")

    except Exception as e:
        click.echo(f"\n✗ Backtest failed: {e}", err=True)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
