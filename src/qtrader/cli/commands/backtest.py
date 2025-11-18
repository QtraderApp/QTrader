"""Backtest execution command."""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine
from qtrader.system.config import get_system_config, reload_system_config

console = Console()


@click.command("backtest")
@click.option(
    "--file",
    "-f",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to backtest configuration file (YAML)",
)
@click.option(
    "--silent",
    "-s",
    is_flag=True,
    help="Silent mode: no event display (fastest execution)",
)
@click.option(
    "--replay-speed",
    "-r",
    type=float,
    help="Override replay speed (-1=silent, 0=instant, >0=delay in seconds)",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Override start date (YYYY-MM-DD)",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m-%d"]),
    help="Override end date (YYYY-MM-DD)",
)
@click.option(
    "--log-level",
    "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Set logging level (DEBUG shows all initialization details)",
)
def backtest_command(
    config_file: Path,
    silent: bool,
    replay_speed: Optional[float],
    start_date: Optional[datetime],
    end_date: Optional[datetime],
    log_level: Optional[str],
):
    """
    Run a backtest from configuration file.

    Loads backtest configuration and executes the simulation. CLI options
    override config file values without modifying files.

    \b
    Examples:
        # Basic run with config defaults
        qtrader backtest --file config/portfolio.yaml

        # Silent mode (fastest execution)
        qtrader backtest -f config/portfolio.yaml --silent

        # Override replay speed
        qtrader backtest -f config/portfolio.yaml -r 0.5

        # Quick date range test
        qtrader backtest -f config/portfolio.yaml \\
            --start-date 2020-01-01 --end-date 2020-03-31

        # Debug mode (show all initialization logs)
        qtrader backtest -f config/portfolio.yaml -l debug

    \b
    Output:
        - Displays backtest progress and results
        - Event store location (if file-based backend in system.yaml)
        - See config/system.yaml for event_store settings (sqlite/parquet/memory)
    """
    try:
        # Header
        console.rule("[bold blue]QTrader Backtest[/bold blue]")
        console.print()

        # Load configuration
        console.print("[cyan]Loading configuration...[/cyan]")
        reload_system_config()
        config = load_backtest_config(config_file)

        # Apply log level override if specified
        if log_level:
            from typing import Literal, cast

            from qtrader.system import LoggerFactory

            system_config = get_system_config()
            # Type cast since click already validated the choice
            level = cast(Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], log_level.upper())
            system_config.logging.level = level
            LoggerFactory.configure(system_config.logging.to_logger_config())

        # Apply CLI overrides
        if silent:
            config.replay_speed = -1.0
            config.display_events = None
        elif replay_speed is not None:
            config.replay_speed = replay_speed

        if start_date:
            config.start_date = start_date

        if end_date:
            config.end_date = end_date

        # Display config summary
        console.print(f"  Backtest ID: [yellow]{config.backtest_id}[/yellow]")
        console.print(
            f"  Date Range: [yellow]{config.start_date.date()}[/yellow] to [yellow]{config.end_date.date()}[/yellow]"
        )
        console.print(f"  Universe: [magenta]{list(config.all_symbols)}[/magenta]")

        if config.replay_speed == -1.0:
            console.print("  Display: [dim]Silent mode (no events)[/dim]")
        elif config.replay_speed == 0:
            console.print(f"  Display: [yellow]{config.display_events}[/yellow] (instant)")
        else:
            console.print(f"  Display: [yellow]{config.display_events}[/yellow] ({config.replay_speed}s per event)")
        console.print()

        # Initialize engine
        with console.status("[cyan]Initializing backtest engine...[/cyan]"):
            engine = BacktestEngine.from_config(config)

        result = engine.run()

        console.print()
        console.print("[bold green]✓ Backtest completed successfully![/bold green]")
        console.print()

        # Display results
        console.rule("[bold green]RESULTS[/bold green]")
        console.print()
        console.print(f"[cyan]Date Range:[/cyan]      {result.start_date} to {result.end_date}")
        console.print(f"[cyan]Bars Processed:[/cyan]  {result.bars_processed:,}")
        console.print(f"[cyan]Duration:[/cyan]        {result.duration}")
        console.print()

        # Event store info
        system_config = get_system_config()
        backend_type = system_config.output.event_store.backend

        if backend_type == "memory":
            console.print("[cyan]Event Store:[/cyan]     memory (no files created)")
        elif hasattr(engine, "_results_dir") and engine._results_dir:
            console.print(f"[cyan]Results Dir:[/cyan]     {engine._results_dir}")

            event_store_filename = system_config.output.event_store.filename
            if "{backend}" in event_store_filename:
                extension_map = {"sqlite": "sqlite", "parquet": "parquet"}
                event_store_filename = event_store_filename.replace("{backend}", extension_map[backend_type])

            event_file = engine._results_dir / event_store_filename
            if event_file.exists():
                size_mb = os.path.getsize(event_file) / (1024 * 1024)
                console.print(f"[cyan]Event Store:[/cyan]     {event_file.name} ({size_mb:.2f} MB)")

        console.print()
        console.rule()
        console.print()

        # Cleanup
        engine.shutdown()

        sys.exit(0)

    except Exception as e:
        console.print()
        console.print(f"[bold red]✗ Backtest failed:[/bold red] {e}")
        import traceback

        console.print()
        console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)
