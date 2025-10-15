"""QTrader CLI - Clean implementation."""

import importlib.util
import inspect
import sys
from datetime import datetime
from typing import Any, Dict

import click
from rich.console import Console
from rich.table import Table

from qtrader.config import BarSchemaConfig, DataConfig
from qtrader.services import DataService


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
    type=click.Choice(["algoseek", "schwab"], case_sensitive=False),
    default="algoseek",
    help="Data source",
)
def raw_data(symbol: str, start_date: str, end_date: str, source: str):
    """
    Browse raw unadjusted historical data bars interactively.

    Displays data exactly as provided by the source (unadjusted prices).
    Press ENTER to display next bar, CTRL+C to exit.

    Example:
        qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31
        qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31 --source schwab
    """
    console = Console()

    try:
        # Parse and validate dates
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d").date()
            end = datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as e:
            console.print("[red]Error: Invalid date format. Use YYYY-MM-DD[/red]")
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

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

        # Configure data service
        config = DataConfig(
            mode="adjusted",  # Internal processing mode
            frequency="1d",
            timezone="America/New_York",
            source_tag=f"{source.lower()}-adjusted",  # e.g., "algoseek-adjusted"
            bar_schema=bar_schema,
        )

        # Create service and load data
        console.print(f"[cyan]Loading data for {symbol} from {source}...[/cyan]")
        service = DataService(config)
        iterator = service.load_symbol(symbol, start, end)
        bars = list(iterator)

        if not bars:
            console.print(f"[yellow]No data found for {symbol} between {start_date} and {end_date}[/yellow]")
            sys.exit(0)

        console.print(f"[green]Loaded {len(bars)} bars[/green]")
        console.print("[dim]Displaying raw unadjusted prices[/dim]")
        console.print("[dim]Press ENTER to view next bar, CTRL+C to exit[/dim]\n")

        # Display bars one by one
        for idx, multi_bar in enumerate(bars, 1):
            # Always use unadjusted (raw) data
            bar = multi_bar.unadjusted

            # Create table for this bar
            table = Table(title=f"Bar {idx}/{len(bars)} - {symbol} (raw)")
            table.add_column("Field", style="cyan", no_wrap=True)
            table.add_column("Value", style="white")

            # Display bar data (uniform interface for all data sources)
            table.add_row("Date", str(multi_bar.trade_datetime.date()))
            table.add_row("Open", f"${bar.open:.2f}")
            table.add_row("High", f"${bar.high:.2f}")
            table.add_row("Low", f"${bar.low:.2f}")
            table.add_row("Close", f"${bar.close:.2f}")
            table.add_row("Volume", f"{bar.volume:,}")

            # Show dividend if present (on ex-dividend date)
            if bar.dividend:
                table.add_row("Dividend", f"${bar.dividend:.4f}", style="green bold")

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
