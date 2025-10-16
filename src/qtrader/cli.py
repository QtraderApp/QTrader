"""QTrader CLI - Clean implementation."""

import importlib.util
import inspect
import sys
from datetime import datetime
from typing import Any, Dict

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TimeElapsedColumn
from rich.table import Table

from qtrader.config import AssetClass, BarSchemaConfig, DataConfig, DataSourceSelector
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


@main.group("data")
def data_group():
    """Data management commands - browse, fetch, cache, and update market data"""
    pass


@data_group.command("raw")
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
        qtrader data raw --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31
        qtrader data raw --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31 --source schwab
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
        selector = DataSourceSelector(
            provider=source.lower(),  # e.g., "algoseek" or "schwab"
            asset_class=AssetClass.EQUITY,
        )
        config = DataConfig(
            mode="adjusted",  # Internal processing mode
            frequency="1d",
            timezone="America/New_York",
            source_selector=selector,
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


@data_group.command("update")
@click.option(
    "--dataset",
    required=True,
    help="Dataset identifier (e.g., schwab-us-equity-1d-adjusted)",
)
@click.option(
    "--symbols",
    help="Comma-separated list of symbols to update (default: all cached symbols)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be updated without making changes",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show detailed update progress",
)
def update_dataset(dataset: str, symbols: str, dry_run: bool, verbose: bool):
    """
    Update cached data to latest available.

    Incrementally updates cached data by fetching only new bars since
    last update. Works with any dataset that supports incremental updates.

    If no symbols specified, updates all symbols found in cache.

    Examples:
        # Update all symbols in Schwab dataset
        qtrader data update --dataset schwab-us-equity-1d-adjusted

        # Update specific symbols
        qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,TSLA,NVDA

        # Dry run (check what would be updated)
        qtrader data update --dataset schwab-us-equity-1d-adjusted --dry-run --verbose

        # Update Algoseek dataset
        qtrader data update --dataset algoseek-us-equity-1d-adjusted
    """
    from qtrader.data.dataset_updater import DatasetUpdater

    console = Console()

    try:
        # Parse symbols if provided
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None

        # Show mode
        mode_str = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]UPDATING[/green]"
        console.print(f"\n{mode_str} Dataset: [cyan]{dataset}[/cyan]\n")

        # Create updater
        updater = DatasetUpdater(dataset)

        # Get symbols to update (priority: --symbols > universe.csv > cached symbols)
        if symbol_list:
            # Explicit --symbols takes precedence
            symbols_to_update = symbol_list
            console.print(f"[cyan]Updating {len(symbols_to_update)} symbols (--symbols)...[/cyan]\n")
        elif updater.universe_symbols:
            # Use configured universe file
            symbols_to_update = updater.universe_symbols
            console.print(
                f"[cyan]Updating {len(symbols_to_update)} symbols from universe.csv "
                f"(full backfill + incremental)...[/cyan]\n"
            )
        else:
            # Fallback to scan cache
            symbols_to_update = updater._scan_cached_symbols()
            if not symbols_to_update:
                console.print("[yellow]No symbols found to update[/yellow]")
                console.print("[dim]Tip: Create universe.csv in cache directory or use --symbols[/dim]")
                return
            console.print(f"[cyan]Updating {len(symbols_to_update)} cached symbols...[/cyan]\n")

        # Update with progress bar
        results = []

        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Updating symbols...", total=len(symbols_to_update))

            # Update each symbol and show progress
            iterator = updater.update_symbols(symbols_to_update, dry_run=dry_run, verbose=verbose)

            for result in iterator:
                # Update progress bar description with current symbol
                status_emoji = "✓" if result.success else "✗"
                progress.update(task, description=f"[cyan]{status_emoji} {result.symbol}", advance=1)
                results.append(result)

        console.print()  # Add blank line after progress

        # Show results summary
        if not results:
            console.print("[yellow]No symbols found to update[/yellow]")
            return

        # Create summary table
        table = Table(title="Update Summary")
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Status", style="white")
        table.add_column("Bars Added", style="magenta", justify="right")
        table.add_column("Date Range", style="dim")

        successful = 0
        total_bars = 0
        errors = []

        for result in results:
            # Read full cached date range and bar count from metadata

            cache_root = updater._get_cache_root()
            start_date = end_date = row_count = "-"
            if cache_root is not None:
                metadata_file = cache_root / result.symbol / ".metadata.json"
                if metadata_file.exists():
                    import json

                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        date_range = metadata.get("date_range", {})
                        start_date = date_range.get("start", "-")
                        end_date = date_range.get("end", "-")
                        row_count = metadata.get("row_count", "-")
                    except Exception:
                        pass

            if result.success:
                successful += 1
                total_bars += result.bars_added

                if result.bars_added == 0:
                    status = "[green]✓ Current[/green]"
                    bars_str = "-"
                else:
                    status = "[green]✓ Updated[/green]"
                    bars_str = str(result.bars_added)
            else:
                status = "[red]✗ Error[/red]"
                bars_str = "-"
                errors.append((result.symbol, result.error))

            # Show full cached range and bar count
            table.add_row(result.symbol, status, bars_str, f"{start_date} to {end_date}", str(row_count))

        console.print(table)

        # Summary stats
        console.print(f"\n[green]Successful:[/green] {successful}/{len(results)}")
        console.print(f"[cyan]Total bars added:[/cyan] {total_bars:,}")

        if errors:
            console.print(f"\n[red]Errors ({len(errors)}):[/red]")
            for symbol, error in errors:
                console.print(f"  [red]•[/red] {symbol}: {error}")

        if dry_run:
            console.print("\n[yellow]This was a dry run. Use --no-dry-run to actually update data.[/yellow]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@data_group.command("cache-info")
@click.option(
    "--dataset",
    required=True,
    help="Dataset identifier (e.g., schwab-us-equity-1d-adjusted)",
)
def cache_info(dataset: str):
    """
    Show cache information for a dataset.

    Displays cached symbols, date ranges, and update status.

    Example:
        qtrader data cache-info --dataset schwab-us-equity-1d-adjusted
    """
    from qtrader.data.dataset_updater import DatasetUpdater

    console = Console()

    try:
        updater = DatasetUpdater(dataset)

        # Get cache info
        cache_root = updater._get_cache_root()
        if not cache_root or not cache_root.exists():
            console.print(f"[yellow]No cache found for dataset: {dataset}[/yellow]")
            return

        symbols = updater._scan_cached_symbols()
        if not symbols:
            console.print(f"[yellow]Cache directory empty: {cache_root}[/yellow]")
            return

        # Show summary
        console.print(f"\n[cyan]Dataset:[/cyan] {dataset}")
        console.print(f"[cyan]Cache location:[/cyan] {cache_root}")
        console.print(f"[cyan]Cached symbols:[/cyan] {len(symbols)}\n")

        # List symbols with metadata
        table = Table(title="Cached Symbols")
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Start Date", style="green")
        table.add_column("End Date", style="green")
        table.add_column("Bars", justify="right", style="yellow")
        table.add_column("Last Update", style="dim")

        for symbol in symbols:
            # Read metadata if available
            metadata_file = cache_root / symbol / ".metadata.json"

            if metadata_file.exists():
                import json

                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    date_range = metadata.get("date_range", {})
                    start_date = date_range.get("start", "N/A")
                    end_date = date_range.get("end", "N/A")
                    row_count = metadata.get("row_count", "N/A")
                    last_update = metadata.get("last_update", "N/A")

                    # Format last update (show just date/time, not full ISO)
                    if last_update != "N/A" and "T" in last_update:
                        last_update = last_update.split("T")[0] + " " + last_update.split("T")[1][:8]

                    table.add_row(symbol, start_date, end_date, str(row_count), last_update)
                except Exception as e:
                    table.add_row(symbol, "Error", "Error", "Error", str(e)[:20])
            else:
                # No metadata file - just show that cache exists
                table.add_row(symbol, "N/A", "N/A", "N/A", "No metadata")

        console.print(table)

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
