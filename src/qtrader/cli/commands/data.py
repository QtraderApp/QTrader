"""Data management commands - thin CLI orchestration layer."""

import sys
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table

from qtrader.cli.ui import (
    create_bar_table,
    create_cache_info_table,
    create_update_progress,
    create_update_summary_table,
)
from qtrader.cli.ui.formatters import add_bar_data, add_cache_info_row, add_update_result_row
from qtrader.services.data.adapters.resolver import DataSourceResolver
from qtrader.services.data.update_service import UpdateService


@click.group("data")
def data_group():
    """Data management commands - browse, fetch, cache, and update market data"""
    pass


@data_group.command("list")
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information about each dataset",
)
def list_datasets(verbose: bool):
    """
    List all available datasets configured in data_sources.yaml.

    Displays dataset names, providers, adapters, and asset classes.
    Use --verbose for additional configuration details.

    Example:
        qtrader data list
        qtrader data list --verbose
    """
    console = Console()

    try:
        # Load resolver to access configured datasets
        resolver = DataSourceResolver()

        # Get list of all datasets
        datasets = resolver.list_sources()

        if not datasets:
            console.print("[yellow]No datasets configured in data_sources.yaml[/yellow]")
            return

        # Display summary
        console.print(f"\n[cyan]Found {len(datasets)} configured dataset(s)[/cyan]")
        console.print(f"[dim]Configuration file: {resolver.config_path}[/dim]\n")

        # Create table
        table = Table(title="Available Datasets", show_header=True, header_style="bold cyan")
        table.add_column("Dataset Name", style="green", no_wrap=True)
        table.add_column("Provider", style="cyan")
        table.add_column("Adapter", style="yellow")
        table.add_column("Asset Class", style="magenta")

        if verbose:
            table.add_column("Frequency", style="blue")
            table.add_column("Cache", style="white")

        # Add rows for each dataset
        for dataset_name in sorted(datasets):
            config = resolver.get_source_config(dataset_name)

            provider = config.get("provider", "N/A")
            adapter = config.get("adapter", "N/A")
            asset_class = config.get("asset_class", "N/A")

            if verbose:
                frequency = config.get("frequency", "N/A")
                cache_status = "✓" if config.get("cache_root") else "✗"
                table.add_row(
                    dataset_name,
                    provider,
                    adapter,
                    asset_class,
                    frequency,
                    cache_status,
                )
            else:
                table.add_row(dataset_name, provider, adapter, asset_class)

        console.print(table)
        console.print()

        # Show helpful tips
        if verbose:
            console.print("[dim]Cache column: ✓ = caching enabled, ✗ = no cache[/dim]")
        else:
            console.print("[dim]Tip: Use --verbose for more details[/dim]")

    except FileNotFoundError as e:
        console.print(f"[red]Error: {e}[/red]")
        console.print("[yellow]No data_sources.yaml found. Create one in config/ directory.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)


@data_group.command("raw")
@click.option("--symbol", required=True, help="Symbol to load (e.g., AAPL)")
@click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end-date", required=True, help="End date (YYYY-MM-DD)")
@click.option(
    "--dataset",
    type=str,
    required=True,
    help="Dataset identifier (e.g., yahoo-us-equity-1d-csv)",
)
def raw_data(symbol: str, start_date: str, end_date: str, dataset: str):
    """
    Browse raw unadjusted historical data bars interactively.

    Displays data exactly as provided by the source.
    Press ENTER to display next bar, CTRL+C to exit.

    Example:
        Examples:
        # Fetch raw bars for AAPL
        qtrader data raw --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31 --dataset yahoo-us-equity-1d-csv
    """
    console = Console()

    try:
        # Parse and validate dates
        try:
            datetime.strptime(start_date, "%Y-%m-%d").date()
            datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError as e:
            console.print("[red]Error: Invalid date format. Use YYYY-MM-DD[/red]")
            console.print(f"[red]{e}[/red]")
            sys.exit(1)

        # Load data directly from adapter
        console.print(f"[cyan]Loading data for {symbol} from {dataset}...[/cyan]")

        from qtrader.services.data.models import Instrument

        resolver = DataSourceResolver()
        instrument = Instrument(symbol=symbol)
        adapter = resolver.resolve_by_dataset(dataset, instrument)

        # Read bars and convert to PriceBarEvents
        raw_bars = adapter.read_bars(start_date, end_date)
        bars = [adapter.to_price_bar_event(bar) for bar in raw_bars]

        if not bars:
            console.print(f"[yellow]No data found for {symbol} between {start_date} and {end_date}[/yellow]")
            sys.exit(0)

        console.print(f"[green]Loaded {len(bars)} bars[/green]")
        console.print("[dim]Displaying raw unadjusted prices[/dim]")
        console.print("[dim]Press ENTER to view next bar, CTRL+C to exit[/dim]\n")

        # Display bars one by one
        for idx, bar in enumerate(bars, 1):
            # Create table for this bar
            table = create_bar_table(symbol, idx, len(bars))

            # Prepare bar data from PriceBarEvent
            bar_data = {
                "date": bar.timestamp.split("T")[0],  # Extract date from ISO timestamp
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": bar.volume,
                "dividend": 0.0,  # PriceBarEvent doesn't have dividend field directly
            }

            # Add data to table
            add_bar_data(table, bar_data)

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
    "-d",
    "--dataset",
    required=True,
    help="Dataset identifier (e.g., yahoo-us-equity-1d-csv)",
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
@click.option(
    "--force-reprime",
    is_flag=True,
    help="Delete existing cache and re-prime from scratch (useful for adapters without incremental support)",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    help="Skip confirmation prompts (auto-confirm force-reprime deletion)",
)
def update_dataset(dataset: str, symbols: str, dry_run: bool, verbose: bool, force_reprime: bool, yes: bool):
    """
    Update cached data to latest available.

    Incrementally updates cached data by fetching only new bars since
    last update. Works with any dataset that supports incremental updates.

    If adapter doesn't support incremental updates, use --force-reprime to
    automatically delete cache and re-prime from scratch.

    If no symbols specified, updates all symbols found in cache.

    Examples:

        Examples:
        # Update dataset
        qtrader data update --dataset yahoo-us-equity-1d-csv

        # Force re-prime for adapter without incremental support
        qtrader data update --dataset my-custom-dataset --force-reprime
    """
    console = Console()

    try:
        # Parse symbols if provided
        symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None

        # Show mode
        mode_str = "[yellow]DRY RUN[/yellow]" if dry_run else "[green]UPDATING[/green]"
        reprime_str = " [yellow](FORCE RE-PRIME)[/yellow]" if force_reprime else ""
        console.print(f"\n{mode_str}{reprime_str} Dataset: [cyan]{dataset}[/cyan]\n")

        # Create update service
        service = UpdateService(dataset)

        # Force-reprime confirmation prompt
        if force_reprime and not dry_run and not yes:
            # Get cache directory path to show user what will be deleted
            cache_root = service.updater._get_cache_root()

            console.print("[yellow]⚠️  WARNING: Force re-prime will DELETE existing cache data[/yellow]")
            console.print(f"[dim]Cache directory: {cache_root}[/dim]\n")

            # Parse symbols to show what will be affected
            symbol_list_preview = [s.strip() for s in symbols.split(",")] if symbols else None
            if symbol_list_preview:
                console.print(f"[yellow]Symbols to re-prime: {', '.join(symbol_list_preview)}[/yellow]")
            else:
                console.print("[yellow]This will delete caches for ALL symbols in the dataset[/yellow]")

            console.print()

            if not click.confirm("Are you sure you want to continue?", default=False):
                console.print("[yellow]Aborted.[/yellow]")
                return

        if force_reprime and not dry_run:
            console.print("[yellow]⚠ Force re-prime: Will delete existing caches and re-download all data[/yellow]\n")

        # Get symbols to update (service handles priority logic)
        symbols_to_update, source_desc = service.get_symbols_to_update(symbol_list)

        if not symbols_to_update:
            console.print("[yellow]No symbols found to update[/yellow]")
            console.print("[dim]Tip: Create universe.csv in cache directory or use --symbols[/dim]")
            return

        console.print(f"[cyan]Updating {source_desc}...[/cyan]\n")

        # Update with progress bar
        results = []

        with create_update_progress(console) as progress:
            task = progress.add_task("[cyan]Updating symbols...", total=len(symbols_to_update))

            # Update each symbol and show progress
            for result in service.update_symbols(
                symbols_to_update, dry_run=dry_run, verbose=verbose, force_reprime=force_reprime
            ):
                status_emoji = "✓" if result.success else "✗"
                progress.update(task, description=f"[cyan]{status_emoji} {result.symbol}", advance=1)
                results.append(result)

        console.print()  # Add blank line after progress

        # Show results summary
        if not results:
            console.print("[yellow]No symbols found to update[/yellow]")
            return

        # Create summary table
        table = create_update_summary_table()

        successful = 0
        total_bars = 0
        errors = []

        for result in results:
            # Get full cached metadata
            start_date, end_date, row_count = service.get_cache_metadata(result.symbol)

            if result.success:
                successful += 1
                total_bars += result.bars_added
            else:
                errors.append((result.symbol, result.error))

            # Add row to table
            add_update_result_row(
                table,
                result.symbol,
                result.success,
                result.bars_added,
                start_date,
                end_date,
                row_count,
                result.error,
            )

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
    help="Dataset identifier (e.g., yahoo-us-equity-1d-csv)",
)
def cache_info(dataset: str):
    """
    Show cache information for a dataset.

    Displays cached symbols, date ranges, and update status.

    Examples:
        qtrader data cache-info --dataset yahoo-us-equity-1d-csv
    """
    console = Console()

    try:
        # Create update service
        service = UpdateService(dataset)

        # Get cache info
        cache_root = service.get_cache_root()
        if not cache_root or not cache_root.exists():
            console.print(f"[yellow]No cache found for dataset: {dataset}[/yellow]")
            return

        symbols = service.scan_cached_symbols()
        if not symbols:
            console.print(f"[yellow]Cache directory empty: {cache_root}[/yellow]")
            return

        # Show summary
        console.print(f"\n[cyan]Dataset:[/cyan] {dataset}")
        console.print(f"[cyan]Cache location:[/cyan] {cache_root}")
        console.print(f"[cyan]Cached symbols:[/cyan] {len(symbols)}\n")

        # Create table
        table = create_cache_info_table()

        # Add rows for each symbol
        for symbol in symbols:
            metadata = service.read_symbol_metadata(symbol, cache_root)
            add_cache_info_row(
                table,
                metadata["symbol"],
                metadata["start_date"],
                metadata["end_date"],
                str(metadata["row_count"]),
                metadata["last_update"],
            )

        console.print(table)

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        import traceback

        console.print(traceback.format_exc())
        sys.exit(1)
