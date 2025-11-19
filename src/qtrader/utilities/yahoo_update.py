#!/usr/bin/env python3
"""
Yahoo Finance Data Updater.

Updates OHLCV data and dividends for symbols in data/us-equity-yahoo-csv using yfinance.
Performs incremental updates by checking existing data and only fetching new bars.

Features:
- Incremental updates: Only fetches data after the last available date
- Full refresh: Re-downloads all data if yfinance has earlier data than local CSV
- Dividend tracking: Maintains dividends_calendar.csv for all symbols
- Multi-symbol support: Updates all CSV files in the directory
- Error handling: Continues processing other symbols if one fails

CSV Format (per symbol):
    Date,Open,High,Low,Close,Adj Close,Volume
    2020-01-02,74.06,75.15,73.80,75.09,72.47,135480400

Dividends JSON Format (dividends_calendar.json):
    {
        "AAPL": [
            {"date": "2020-02-07", "amount": 0.77},
            {"date": "2020-05-08", "amount": 0.82}
        ]
    }

Universe JSON Format (universe.json):
    {
        "tickers": ["AAPL", "MSFT", "GOOGL"]
    }

Usage:
    # Update all symbols in data/us-equity-yahoo-csv/
    python -m qtrader.utilities.yahoo_update

    # Update specific symbols
    python -m qtrader.utilities.yahoo_update AAPL MSFT GOOGL

    # Update with date range
    python -m qtrader.utilities.yahoo_update --start 2020-01-01 --end 2024-12-31

    # Force full refresh (re-download all data)
    python -m qtrader.utilities.yahoo_update --full-refresh
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import yfinance as yf  # type: ignore[import-untyped]
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


def is_market_closed() -> bool:
    """
    Check if US market is closed for today.

    US market hours: 9:30 AM - 4:00 PM ET
    Returns True if current time is after 4:00 PM ET, False otherwise.
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return now_et >= market_close


def get_safe_end_date() -> str:
    """
    Get a safe end date that excludes today if market is still open.

    Returns:
        ISO date string (YYYY-MM-DD) - yesterday if market is open, today if closed
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))

    if is_market_closed():
        # Market is closed, we can include today's data
        return now_et.strftime("%Y-%m-%d")
    else:
        # Market is still open, use yesterday as end date
        yesterday = now_et - timedelta(days=1)
        return yesterday.strftime("%Y-%m-%d")


def get_existing_date_range(csv_path: Path) -> Optional[Tuple[str, str]]:
    """
    Get the date range of existing data in CSV.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Tuple of (first_date, last_date) as ISO strings, or None if file doesn't exist
    """
    if not csv_path.exists():
        return None

    try:
        with csv_path.open("r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if not rows:
                return None
            # CSV is expected to be sorted by date
            first_date = rows[0]["Date"]
            last_date = rows[-1]["Date"]
            return (first_date, last_date)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not read {csv_path.name}: {e}[/yellow]")
        return None


def fetch_yahoo_data(symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Fetch OHLCV and dividend data from Yahoo Finance.

    Args:
        symbol: Ticker symbol
        start_date: Start date (ISO format YYYY-MM-DD), or None for all available
        end_date: End date (ISO format YYYY-MM-DD), or None for safe date (excludes today if market open)

    Returns:
        Tuple of (price_df, dividends_series) or (None, None) on error
    """
    try:
        ticker = yf.Ticker(symbol)

        # Use safe end date if not provided (excludes today if market is still open)
        if end_date is None:
            end_date = get_safe_end_date()

        # Fetch historical data
        if start_date and end_date:
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=False, actions=True)
        elif start_date:
            hist = ticker.history(start=start_date, end=end_date, auto_adjust=False, actions=True)
        else:
            hist = ticker.history(period="max", auto_adjust=False, actions=True)

        if hist.empty:
            return None, None

        # Extract dividends (comes as part of history with actions=True)
        dividends = hist["Dividends"]
        dividends = dividends[dividends > 0]  # Filter out zero dividends

        # Keep only OHLCV columns
        price_data = hist[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Add Adj Close if available
        if "Close" in hist.columns:
            # Calculate adjusted close using split and dividend adjustments
            # yfinance provides this via the adjusted close calculation
            if "Adj Close" in hist.columns:
                price_data["Adj Close"] = hist["Adj Close"]
            else:
                # Fallback: use close as adj close
                price_data["Adj Close"] = hist["Close"]

        return price_data, dividends if not dividends.empty else None

    except Exception as e:
        console.print(f"[red]Error fetching {symbol}: {e}[/red]")
        return None, None


def merge_data(existing_path: Path, new_data: Any, full_refresh: bool = False) -> bool:
    """
    Merge new data with existing CSV or create new file.

    Args:
        existing_path: Path to existing CSV file
        new_data: DataFrame with new price data
        full_refresh: If True, replace entire file

    Returns:
        True if data was updated, False otherwise
    """
    if new_data is None or new_data.empty:
        return False

    try:
        # Ensure Date is a column (not index)
        if new_data.index.name == "Date" or "Date" not in new_data.columns:
            new_data = new_data.reset_index()

        # Format Date column to YYYY-MM-DD
        new_data["Date"] = new_data["Date"].dt.strftime("%Y-%m-%d")

        # Ensure correct column order
        columns = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        new_data = new_data[columns]

        if full_refresh or not existing_path.exists():
            # Write new file with correct column order
            new_data.to_csv(existing_path, index=False, columns=columns)
            return True

        # Read existing data
        existing_data = []
        with existing_path.open("r") as f:
            reader = csv.DictReader(f)
            existing_data = list(reader)

        if not existing_data:
            # Empty file, write new data
            new_data.to_csv(existing_path, index=False)
            return True

        # Get last date from existing data
        last_existing_date = existing_data[-1]["Date"]

        # Filter new data to only include dates after last existing
        new_data["Date"] = new_data["Date"].astype(str)
        new_rows = new_data[new_data["Date"] > last_existing_date]

        if new_rows.empty:
            return False  # No new data to add

        # Append new rows
        with existing_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"])
            for _, row in new_rows.iterrows():
                writer.writerow(
                    {
                        "Date": row["Date"],
                        "Open": row["Open"],
                        "High": row["High"],
                        "Low": row["Low"],
                        "Close": row["Close"],
                        "Adj Close": row.get("Adj Close", row["Close"]),
                        "Volume": int(row["Volume"]),
                    }
                )

        return True

    except Exception as e:
        console.print(f"[red]Error merging data for {existing_path.name}: {e}[/red]")
        return False


def update_dividends_calendar(dividends_path: Path, symbol: str, dividends: Any) -> None:
    """
    Update the dividends calendar JSON with new dividend data.

    Args:
        dividends_path: Path to dividends_calendar.json
        symbol: Ticker symbol
        dividends: Series with dividend data (index=date, value=amount)
    """
    if dividends is None or dividends.empty:
        return

    try:
        # Read existing dividends calendar
        calendar: Dict[str, List[Dict[str, Any]]] = {}
        if dividends_path.exists():
            with dividends_path.open("r") as f:
                calendar = json.load(f)

        # Get existing dividends for this symbol
        if symbol not in calendar:
            calendar[symbol] = []

        existing_dates = {entry["date"] for entry in calendar[symbol]}

        # Add new dividends
        new_count = 0
        for date, amount in dividends.items():
            date_str = date.strftime("%Y-%m-%d")
            if date_str not in existing_dates:
                calendar[symbol].append({"date": date_str, "amount": float(amount)})
                new_count += 1

        # Sort dividends by date for each symbol
        if new_count > 0:
            calendar[symbol] = sorted(calendar[symbol], key=lambda x: x["date"])

            # Write back to file
            with dividends_path.open("w") as f:
                json.dump(calendar, f, indent=2)

    except Exception as e:
        console.print(f"[yellow]Warning: Could not update dividends for {symbol}: {e}[/yellow]")


def update_symbol(
    symbol: str,
    data_dir: Path,
    dividends_path: Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    full_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Update data for a single symbol.

    Args:
        symbol: Ticker symbol
        data_dir: Directory containing CSV files
        dividends_path: Path to dividends_calendar.csv
        start_date: Override start date
        end_date: Override end date
        full_refresh: Force full data refresh

    Returns:
        Dict with update status and stats
    """
    csv_path = data_dir / f"{symbol}.csv"
    result = {
        "symbol": symbol,
        "success": False,
        "action": "none",
        "bars_added": 0,
        "dividends_added": 0,
    }

    # Determine date range
    fetch_start = start_date
    fetch_end = end_date

    if not full_refresh and not fetch_start:
        # Check existing data for incremental update
        existing_range = get_existing_date_range(csv_path)
        if existing_range:
            last_date = existing_range[1]
            # Fetch data starting from day after last date
            last_dt = datetime.strptime(last_date, "%Y-%m-%d")
            next_day = last_dt + timedelta(days=1)
            fetch_start = next_day.strftime("%Y-%m-%d")
            result["action"] = "incremental"

            # Check if start date is after safe end date (when market is open)
            safe_end = get_safe_end_date()
            if fetch_start > safe_end:
                # Already up to date (no new complete trading days)
                result["action"] = "none"
                result["success"] = True
                return result
        else:
            result["action"] = "new"
    else:
        result["action"] = "full_refresh" if full_refresh else "new"

    # Fetch data from Yahoo Finance
    price_data, dividends = fetch_yahoo_data(symbol, fetch_start, fetch_end)

    if price_data is None:
        return result

    # Check if Yahoo has earlier data than our CSV (need full refresh)
    if not full_refresh and csv_path.exists() and not price_data.empty:
        existing_range = get_existing_date_range(csv_path)
        if existing_range:
            yahoo_first = price_data.index[0].strftime("%Y-%m-%d")
            existing_first = existing_range[0]
            if yahoo_first < existing_first:
                console.print(
                    f"[yellow]{symbol}: Yahoo has data starting {yahoo_first}, local starts {existing_first}. Doing full refresh.[/yellow]"
                )
                price_data, dividends = fetch_yahoo_data(symbol, None, fetch_end)
                full_refresh = True
                result["action"] = "full_refresh"

    # Update price data
    if merge_data(csv_path, price_data, full_refresh):
        result["success"] = True
        result["bars_added"] = len(price_data)

    # Update dividends
    if dividends is not None:
        update_dividends_calendar(dividends_path, symbol, dividends)
        result["dividends_added"] = len(dividends)

    return result


def load_universe(universe_path: Path) -> Optional[List[str]]:
    """
    Load ticker symbols from universe.json.

    Args:
        universe_path: Path to universe.json

    Returns:
        List of ticker symbols, or None if file doesn't exist
    """
    if not universe_path.exists():
        return None

    try:
        with universe_path.open("r") as f:
            data = json.load(f)
            tickers = data.get("tickers", [])
            return list(tickers) if tickers else []
    except Exception as e:
        console.print(f"[yellow]Warning: Could not read universe.json: {e}[/yellow]")
        return None


def discover_symbols(data_dir: Path) -> List[str]:
    """
    Discover symbols from existing CSV files in data directory.

    Args:
        data_dir: Directory containing CSV files

    Returns:
        List of ticker symbols
    """
    symbols = []
    for csv_file in data_dir.glob("*.csv"):
        symbols.append(csv_file.stem)
    return sorted(symbols)


def main():
    """Main entry point for the update script."""
    parser = argparse.ArgumentParser(
        description="Update Yahoo Finance data for symbols in data/us-equity-yahoo-csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Update all symbols from universe.json
  python -m qtrader.utilities.yahoo_update

  # Update specific symbols
  python -m qtrader.utilities.yahoo_update AAPL MSFT

  # Update with date range
  python -m qtrader.utilities.yahoo_update --start 2020-01-01 --end 2024-12-31

  # Force full refresh
  python -m qtrader.utilities.yahoo_update --full-refresh

Note: If universe.json is not found, the script will discover symbols from existing CSV files.
        """,
    )
    parser.add_argument("symbols", nargs="*", help="Symbols to update (default: from universe.json or all CSV files)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--full-refresh", action="store_true", help="Re-download all data")
    parser.add_argument("--data-dir", default="data/us-equity-yahoo-csv", help="Data directory")

    args = parser.parse_args()

    # Resolve data directory
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        # Assume relative to project root
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent.parent.parent
        data_dir = project_root / args.data_dir

    if not data_dir.exists():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        sys.exit(1)

    universe_path = data_dir / "universe.json"
    dividends_path = data_dir / "dividends_calendar.json"

    # Determine symbols to update
    if args.symbols:
        symbols = args.symbols
    else:
        # Try to load from universe.json first
        symbols = load_universe(universe_path)
        if symbols is None:
            # Fallback: discover from existing CSV files
            console.print(f"[yellow]universe.json not found, discovering symbols from CSV files...[/yellow]")
            symbols = discover_symbols(data_dir)
            if not symbols:
                console.print(f"[yellow]No CSV files found in {data_dir}[/yellow]")
                sys.exit(0)
        elif not symbols:
            console.print(f"[yellow]universe.json exists but contains no tickers[/yellow]")
            sys.exit(0)

    console.print(f"\n[cyan]Yahoo Finance Data Updater[/cyan]")
    console.print(f"Data directory: {data_dir}")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Mode: {'Full Refresh' if args.full_refresh else 'Incremental Update'}")
    if args.start:
        console.print(f"Start date: {args.start}")
    if args.end:
        console.print(f"End date: {args.end}")
    else:
        safe_date = get_safe_end_date()
        if not is_market_closed():
            console.print(f"[yellow]Market is still open - excluding today's data (end date: {safe_date})[/yellow]")
        else:
            console.print(f"[green]Market is closed - including today's data (end date: {safe_date})[/green]")
    console.print()

    # Update symbols with progress bar
    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Updating symbols...", total=len(symbols))

        for symbol in symbols:
            progress.update(task, description=f"[cyan]Updating {symbol}...")
            result = update_symbol(
                symbol,
                data_dir,
                dividends_path,
                start_date=args.start,
                end_date=args.end,
                full_refresh=args.full_refresh,
            )
            results.append(result)
            progress.advance(task)

    # Display results
    console.print("\n[bold]Update Summary:[/bold]\n")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Symbol", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Action")
    table.add_column("Bars Added", justify="right")
    table.add_column("Dividends", justify="right")

    success_count = 0
    total_bars = 0
    total_dividends = 0

    for result in results:
        status = "✓" if result["success"] else "✗"

        table.add_row(
            result["symbol"],
            status,
            result["action"],
            str(result["bars_added"]),
            str(result["dividends_added"]),
        )

        if result["success"]:
            success_count += 1
            total_bars += result["bars_added"]
            total_dividends += result["dividends_added"]

    console.print(table)

    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"  Successful: {success_count}/{len(symbols)}")
    console.print(f"  Total bars added: {total_bars}")
    console.print(f"  Total dividends added: {total_dividends}")

    if dividends_path.exists():
        console.print(f"\n[dim]Dividends calendar: {dividends_path}[/dim]")


if __name__ == "__main__":
    main()
