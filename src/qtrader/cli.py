"""QTrader CLI - Clean implementation."""

import importlib.util
import inspect
import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict

import click

from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.config.system_config import get_config
from qtrader.data import DataLoader
from qtrader.execution.config import ExecutionConfig
from qtrader.models.portfolio import Portfolio
from qtrader.risk.manager import RiskManager
from qtrader.risk.policy import RiskPolicy


def _load_module(path: str):
    """Load Python module from file path."""
    spec = importlib.util.spec_from_file_location("strategy", path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy"] = module
    spec.loader.exec_module(module)
    return module


def _find_strategy_class(module):
    """Find Strategy class (has on_bar method)."""
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
    import ast

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
@click.option("--strategy-file", required=True, type=click.Path(exists=True), help="Path to strategy file")
@click.option("--set", "overrides", multiple=True, help="Override config: --set key=value")
@click.option("--output-dir", type=click.Path(), help="Output directory")
def backtest(strategy_file: str, overrides: tuple, output_dir: str):
    """Run backtest with strategy file."""
    click.echo("=" * 60)
    click.echo("QTrader Backtest Engine")
    click.echo("=" * 60)

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
