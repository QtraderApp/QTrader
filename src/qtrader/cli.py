"""QTrader CLI - Clean implementation."""

import importlib.util
import inspect
import sys
from datetime import datetime
from typing import Any, Dict

import click
from rich.console import Console
from rich.table import Table

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import DataSource, Instrument, InstrumentType


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


@main.command("raw-data")
@click.option("--symbol", required=True, help="Symbol to load (e.g., AAPL)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--source",
    type=click.Choice(["algoseek"], case_sensitive=False),
    default="algoseek",
    help="Data source (currently only algoseek supported)",
)
def raw_data(symbol: str, start_date: str, end_date: str, source: str):
    """
    Browse raw unadjusted historical data bars interactively.

    Press ENTER to display next bar, CTRL+C to exit.

    Example:
        qtrader raw-data --symbol AAPL --start-date 2019-01-01 --end-date 2023-12-31 --source algoseek
    """
    console = Console()

    try:
        # Validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            console.print("[red]Error: Invalid date format. Use YYYY-MM-DD[/red]")
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

        # Create instrument
        data_source = DataSource.ALGOSEEK if source.lower() == "algoseek" else DataSource.ALGOSEEK
        instrument = Instrument(symbol=symbol, instrument_type=InstrumentType.EQUITY, data_source=data_source)

        # Resolve data source and create adapter
        console.print(f"[cyan]Loading data for {symbol} from {source}...[/cyan]")
        resolver = DataSourceResolver()
        adapter = resolver.resolve(instrument)

        # Load bars
        console.print(f"[cyan]Reading bars from {start_date} to {end_date}...[/cyan]")
        bars = list(adapter.read_bars(start_date, end_date))

        if not bars:
            console.print(f"[yellow]No data found for {symbol} between {start_date} and {end_date}[/yellow]")
            sys.exit(0)

        console.print(f"[green]Loaded {len(bars)} bars[/green]\n")
        console.print("[dim]Press ENTER to view next bar, CTRL+C to exit[/dim]\n")

        # Display bars one by one
        for idx, bar in enumerate(bars, 1):
            # Create table for this bar
            table = Table(title=f"Bar {idx}/{len(bars)} - {symbol}")
            table.add_column("Field", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")

            # Add bar data (convert datetime to string)
            trade_date_str = (
                bar.TradeDate.strftime("%Y-%m-%d") if hasattr(bar.TradeDate, "strftime") else str(bar.TradeDate)
            )
            table.add_row("Trade Date", trade_date_str)
            table.add_row("Open", f"${bar.Open:.2f}")
            table.add_row("High", f"${bar.High:.2f}")
            table.add_row("Low", f"${bar.Low:.2f}")
            table.add_row("Close", f"${bar.Close:.2f}")
            table.add_row("Volume", f"{bar.MarketHoursVolume:,}")

            # Add adjustment factors if present
            if hasattr(bar, "CumulativePriceFactor") and bar.CumulativePriceFactor is not None:
                table.add_row("Cumulative Price Factor", f"{bar.CumulativePriceFactor:.6f}")
            if hasattr(bar, "CumulativeVolumeFactor") and bar.CumulativeVolumeFactor is not None:
                table.add_row("Cumulative Volume Factor", f"{bar.CumulativeVolumeFactor:.6f}")
            if hasattr(bar, "AdjustmentFactor") and bar.AdjustmentFactor is not None:
                table.add_row("Adjustment Factor", f"{bar.AdjustmentFactor:.6f}")
            if hasattr(bar, "AdjustmentReason") and bar.AdjustmentReason:
                table.add_row("Adjustment Reason", bar.AdjustmentReason)

            # Display table
            console.print(table)

            # Wait for user input (except on last bar)
            if idx < len(bars):
                try:
                    input()
                except KeyboardInterrupt:
                    console.print("\n[yellow]Exiting...[/yellow]")
                    break
            else:
                console.print("\n[green]End of data[/green]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]Make sure the data files exist and the path is configured correctly.[/yellow]")
        sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
