""" CLI Migration Guide: Updating to use DataService

This document shows how to migrate CLI commands from direct adapter usage to the new DataService interface. """

# ============================================================================

# BEFORE: Direct adapter usage (current cli.py)

# ============================================================================

""" @main.command("raw-data") def raw_data(symbol: str, start_date: str, end_date: str, source: str): # OLD WAY: Direct adapter access resolver = DataSourceResolver() adapter = resolver.resolve(instrument) bars = list(adapter.read_bars(start_date, end_date)) # Vendor-specific bars

```
# Problem: Gets AlgoseekBar or SchwabBar (vendor-specific)
# Must handle different bar types with hasattr() checks
```

"""

# ============================================================================

# AFTER: Using DataService (recommended)

# ============================================================================

""" from qtrader.config.data_config import BarSchemaConfig, DataConfig from qtrader.services.data import DataService

@main.command("raw-data") def raw_data(symbol: str, start_date: str, end_date: str, source: str): # NEW WAY: Use DataService

```
# 1. Configure bar schema
bar_schema = BarSchemaConfig(
    ts="trade_datetime",
    symbol="symbol",
    open="open",
    high="high",
    low="low",
    close="close",
    volume="volume",
)

# 2. Configure data service
config = DataConfig(
    mode="adjusted",
    frequency="1d",
    timezone="America/New_York",
    source_tag=f"{source.lower()}-adjusted",  # algoseek-adjusted or schwab-adjusted
    bar_schema=bar_schema,
)

# 3. Create service
service = DataService(config)

# 4. Load data - gets canonical MultiBar objects
from datetime import datetime
start = datetime.strptime(start_date, "%Y-%m-%d").date()
end = datetime.strptime(end_date, "%Y-%m-%d").date()

iterator = service.load_symbol(symbol, start, end)
bars = list(iterator)  # List of MultiBar objects

# 5. Display - uniform interface, no vendor-specific code
for idx, multi_bar in enumerate(bars, 1):
    # Access unadjusted prices (raw data)
    raw_bar = multi_bar.unadjusted

    console.print(f"Bar {idx}/{len(bars)}")
    console.print(f"  Date: {multi_bar.trade_datetime.date()}")
    console.print(f"  Open: ${raw_bar.open:.2f}")
    console.print(f"  High: ${raw_bar.high:.2f}")
    console.print(f"  Low: ${raw_bar.low:.2f}")
    console.print(f"  Close: ${raw_bar.close:.2f}")
    console.print(f"  Volume: {raw_bar.volume:,}")

    # All three adjustment modes available
    # multi_bar.unadjusted   - Raw prices
    # multi_bar.adjusted     - Split-adjusted
    # multi_bar.total_return - Split + dividend adjusted
```

"""

# ============================================================================

# Benefits of Using DataService

# ============================================================================

"""

1. Vendor-agnostic interface

   - No need for hasattr() checks
   - Same code works for all data sources
   - Canonical Bar objects, not vendor-specific

1. All adjustment modes available

   - unadjusted (raw)
   - adjusted (split-adjusted)
   - total_return (total return)

1. Clean, testable code

   - Can mock IDataService for testing
   - Consistent with rest of architecture
   - Better error handling

1. Configuration-driven

   - Switch sources via config
   - No code changes needed
   - Environment-specific configs

1. Structured logging

   - DataService logs all operations
   - Better debugging
   - Production observability """

# ============================================================================

# Complete Updated CLI Command

# ============================================================================

UPDATED_RAW_DATA_COMMAND = """ @main.command("raw-data") @click.option("--symbol", required=True, help="Symbol to load (e.g., AAPL)") @click.option("--start-date", required=True, help="Start date (YYYY-MM-DD)") @click.option("--end-date", required=True, help="End date (YYYY-MM-DD)") @click.option( "--source", type=click.Choice(["algoseek", "schwab"], case_sensitive=False), default="algoseek", help="Data source", ) @click.option( "--mode", type=click.Choice(["unadjusted", "adjusted", "total_return"], case_sensitive=False), default="unadjusted", help="Adjustment mode to display", ) def raw_data(symbol: str, start_date: str, end_date: str, source: str, mode: str): """ Browse historical data bars interactively.

```
Press ENTER to display next bar, CTRL+C to exit.

Examples:
    # Raw unadjusted data
    qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-12-31

    # Split-adjusted data
    qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-12-31 --mode adjusted

    # Total return (with dividends)
    qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-12-31 --mode total_return
\"\"\"
from datetime import datetime
from qtrader.config.data_config import BarSchemaConfig, DataConfig
from qtrader.services.data import DataService
from rich.console import Console
from rich.table import Table

console = Console()

try:
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    # Configure data service
    bar_schema = BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )

    config = DataConfig(
        mode="adjusted",
        frequency="1d",
        timezone="America/New_York",
        source_tag=f"{source.lower()}-adjusted",
        bar_schema=bar_schema,
    )

    # Create service and load data
    console.print(f"[cyan]Loading data for {symbol} from {source}...[/cyan]")
    service = DataService(config)
    iterator = service.load_symbol(symbol, start, end)
    bars = list(iterator)

    if not bars:
        console.print(f"[yellow]No data found for {symbol}[/yellow]")
        return

    console.print(f"[green]Loaded {len(bars)} bars[/green]")
    console.print(f"[dim]Displaying {mode} prices[/dim]")
    console.print("[dim]Press ENTER to view next bar, CTRL+C to exit[/dim]\\n")

    # Display bars
    for idx, multi_bar in enumerate(bars, 1):
        # Get requested adjustment mode
        bar = multi_bar.get_bar(mode)

        # Create table
        table = Table(title=f"Bar {idx}/{len(bars)} - {symbol} ({mode})")
        table.add_column("Field", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        table.add_row("Date", str(multi_bar.trade_datetime.date()))
        table.add_row("Open", f"${bar.open:.2f}")
        table.add_row("High", f"${bar.high:.2f}")
        table.add_row("Low", f"${bar.low:.2f}")
        table.add_row("Close", f"${bar.close:.2f}")
        table.add_row("Volume", f"{bar.volume:,}")

        # Show dividend if present
        if bar.dividend:
            table.add_row("Dividend", f"${bar.dividend:.4f}", style="green")

        console.print(table)

        # Wait for input
        if idx < len(bars):
            try:
                input()
            except KeyboardInterrupt:
                console.print("\\n[yellow]Exiting...[/yellow]")
                break
        else:
            console.print("\\n[green]End of data[/green]")

except ValueError as e:
    console.print(f"[red]Error: {e}[/red]")
    sys.exit(1)
except Exception as e:
    console.print(f"[red]Unexpected error: {e}[/red]")
    import traceback
    console.print(traceback.format_exc())
    sys.exit(1)
```

"""

# ============================================================================

# Migration Checklist

# ============================================================================

""" To update CLI to use DataService:

\[ \] 1. Update imports in cli.py: - Add: from qtrader.config.data_config import BarSchemaConfig, DataConfig - Add: from qtrader.services.data import DataService - Remove: from qtrader.adapters.resolver import DataSourceResolver (optional)

\[ \] 2. Update raw-data command: - Replace adapter creation with DataService initialization - Replace adapter.read_bars() with service.load_symbol() - Update bar display logic to use canonical Bar objects - Remove vendor-specific bar handling (hasattr checks)

\[ \] 3. Add --mode option to command: - Allow users to select adjustment mode - Default to "unadjusted" for raw data

\[ \] 4. Test updated command: - Test with Algoseek source - Test with different modes (unadjusted, adjusted, total_return) - Verify error handling

\[ \] 5. Update other commands if they exist: - Any command that loads data should use DataService - Consistent interface across all commands

\[ \] 6. Update documentation: - Update help text to mention adjustment modes - Update examples in README """
