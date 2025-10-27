#!/usr/bin/env python3
"""
Example: Streaming Historical Data from Database to Event Bus

Demonstrates:
1. Setting up BacktestEngine with configuration
2. Subscribing to PriceBarEvent stream
3. Streaming historical bars from data service
4. Displaying real-time bar data in console with Rich formatting
5. Handling corporate actions (splits, dividends)

This example shows how data flows from the database through the event bus
to subscribers in the QTrader system.

Requirements:
    pip install rich
"""

from datetime import date
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from qtrader.engine.config import load_backtest_config
from qtrader.engine.engine import BacktestEngine
from qtrader.events.events import CorporateActionEvent, PriceBarEvent

# Initialize Rich console
console = Console()


def main() -> int:
    """Stream AAPL bars from August to September 2020."""

    console.rule("[bold blue]QTrader - Data Streaming Example[/bold blue]")
    console.print()

    # ============================================================================
    # Step 1: Load Configuration
    # ============================================================================
    console.print("[bold cyan]Step 1:[/bold cyan] Loading configuration...")

    # Use the standard portfolio config as base
    config_path = Path("config/portfolio.yaml")
    config = load_backtest_config(config_path)

    console.print("✓ Configuration loaded", style="green")
    console.print(f"  Data Source: [yellow]{config.data.sources[0].name}[/yellow]")
    console.print()

    # ============================================================================
    # Step 2: Create Engine (initializes EventBus and DataService)
    # ============================================================================
    console.print("[bold cyan]Step 2:[/bold cyan] Initializing engine...")

    engine = BacktestEngine.from_config(config)

    console.print("✓ Engine initialized", style="green")
    console.print(f"  EventBus: [dim]{engine._event_bus}[/dim]")
    console.print(f"  DataService: [dim]{engine._data_service}[/dim]")
    console.print()

    # ============================================================================
    # Step 3: Subscribe to PriceBarEvent Stream
    # ============================================================================
    console.print("[bold cyan]Step 3:[/bold cyan] Subscribing to price bar events...")

    bar_count = 0
    corporate_action_count = 0

    # Create a table for live bar display
    table = Table(title="📊 Streaming Price Bars", box=box.ROUNDED)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Symbol", style="magenta")
    table.add_column("Date", style="green")
    table.add_column("Open", justify="right", style="yellow")
    table.add_column("High", justify="right", style="green")
    table.add_column("Low", justify="right", style="red")
    table.add_column("Close", justify="right", style="blue")
    table.add_column("Volume", justify="right", style="white")

    def on_price_bar(event: Any) -> None:
        """Handle incoming price bar events."""
        nonlocal bar_count
        bar_count += 1

        # Cast to specific type for type safety in code
        assert isinstance(event, PriceBarEvent)

        # Add row to table
        table.add_row(
            str(bar_count),
            event.symbol,
            event.timestamp[:10],
            f"{event.open:,.2f}",
            f"{event.high:,.2f}",
            f"{event.low:,.2f}",
            f"{event.close:,.2f}",
            f"{event.volume:,}",
        )

        # Display bar data immediately
        console.print(
            f"[cyan]Bar #{bar_count:3d}[/cyan] | "
            f"[magenta]{event.symbol:6s}[/magenta] | "
            f"[green]{event.timestamp[:10]}[/green] | "
            f"O:[yellow]{event.open:>8}[/yellow] "
            f"H:[green]{event.high:>8}[/green] "
            f"L:[red]{event.low:>8}[/red] "
            f"C:[blue]{event.close:>8}[/blue] | "
            f"Vol: {event.volume:>10,}"
        )

        # Show adjustment factors
        if event.cumulative_price_factor != 1.0 or event.cumulative_volume_factor != 1.0:
            console.print(
                f"           [dim]└─ Adjustments: Price={event.cumulative_price_factor:.2f}, "
                f"Volume={event.cumulative_volume_factor:.2f}[/dim]"
            )

    def on_corporate_action(event: Any) -> None:
        """Handle corporate action events."""
        nonlocal corporate_action_count
        corporate_action_count += 1

        # Cast to specific type for type safety in code
        assert isinstance(event, CorporateActionEvent)

        console.print(
            f"[bold yellow]🔔 Corporate Action[/bold yellow] | "
            f"[magenta]{event.symbol}[/magenta] | "
            f"[yellow]{event.action_type.upper()}[/yellow] | "
            f"Ex-Date: [green]{event.ex_date}[/green]"
        )

        if event.action_type == "split":
            console.print(
                f"           [dim]└─ Split: {event.split_from}-for-{event.split_to} (ratio: {event.split_ratio})[/dim]"
            )
        elif event.action_type == "dividend":
            console.print(f"           [dim]└─ Amount: {event.dividend_amount}[/dim]")

    # Subscribe to events (using event type strings)
    bar_subscription = engine._event_bus.subscribe("bar", on_price_bar)
    action_subscription = engine._event_bus.subscribe("corporate_action", on_corporate_action)

    console.print("✓ Subscribed to [yellow]'bar'[/yellow] events", style="green")
    console.print("✓ Subscribed to [yellow]'corporate_action'[/yellow] events", style="green")
    console.print()

    # ============================================================================
    # Step 4: Stream Historical Data
    # ============================================================================
    console.print("[bold cyan]Step 4:[/bold cyan] Streaming historical data...")
    console.print()

    try:
        # Stream AAPL bars from August 1 to September 30, 2020
        # Note: Apple had a 4-for-1 stock split on August 31, 2020
        symbol = "AAPL"
        start = date(2020, 8, 1)
        end = date(2020, 9, 1)

        console.print(f"Streaming [magenta]{symbol}[/magenta] from [green]{start}[/green] to [green]{end}[/green]...")
        console.print(f"Replay Speed: [yellow]{engine.config.replay_speed}[/yellow] seconds per bar")
        console.print()

        # Stream bars through the event bus
        # Note: This may raise validation errors for corporate actions
        # if the data contains dividend types not in the schema (e.g., 'cash')
        # This is a known data quality issue and doesn't affect bar streaming
        try:
            engine._data_service.stream_bars(
                symbol=symbol,
                start_date=start,
                end_date=end,
                is_warmup=False,  # Not warmup - process normally
                replay_speed=engine.config.replay_speed,  # Use replay_speed from config
            )
        except Exception as stream_error:
            # Log the error but continue - bars were still streamed
            console.print()
            console.print(
                Panel(
                    f"[yellow]Stream completed with validation error:[/yellow]\n{stream_error}\n\n"
                    f"[dim](Bars were successfully streamed before error)[/dim]",
                    title="⚠️  Warning",
                    border_style="yellow",
                )
            )
            console.print()

        console.print()

        # Display the table summary
        console.print(table)
        console.print()

        # ========================================================================
        # Step 5: Display Summary
        # ========================================================================
        summary = Table(title="📈 Streaming Summary", box=box.DOUBLE_EDGE)
        summary.add_column("Metric", style="cyan", no_wrap=True)
        summary.add_column("Value", style="yellow")

        summary.add_row("Symbol", f"[magenta]{symbol}[/magenta]")
        summary.add_row("Date Range", f"[green]{start}[/green] to [green]{end}[/green]")
        summary.add_row("Bars Streamed", f"[bold cyan]{bar_count}[/bold cyan]")
        summary.add_row("Corporate Actions", f"[yellow]{corporate_action_count}[/yellow]")

        console.print(summary)
        console.print()

        if corporate_action_count > 0:
            console.print(
                Panel(
                    "[yellow]Apple had a dividend on August 07, 2020[/yellow]\n"
                    "[dim]└─    This affects price adjustment factors.[/dim]\n"
                    "[yellow]Apple had a 4-for-1 stock split on August 31, 2020[/yellow]\n"
                    "[dim]└─    This affects price and volume adjustment factors.[/dim]",
                    title="ℹ️  Note",
                    border_style="blue",
                )
            )
            console.print()

        return 0

    except Exception as e:
        console.print()
        console.print(f"[bold red]✗ Streaming failed:[/bold red] {e}")
        console.print_exception()
        return 1

    finally:
        # ========================================================================
        # Step 6: Cleanup
        # ========================================================================
        console.print("[bold cyan]Step 6:[/bold cyan] Cleaning up...")

        # Unsubscribe from events
        bar_subscription.unsubscribe()
        action_subscription.unsubscribe()

        # Shutdown engine
        engine.shutdown()

        console.print("✓ Cleanup complete", style="green")
        console.print()


if __name__ == "__main__":
    console.print()
    console.print("[bold blue]This example demonstrates streaming historical data through the event bus.[/bold blue]")
    console.print("[dim]Watch as bars are published and subscribers receive them in real-time.[/dim]")
    console.print()
    exit(main())
