"""
Golden Test Output Generator

This script generates the "golden" expected output from the current implementation
of debug_algoseek_bars.py. This output will be used to validate that the refactored
architecture produces identical results.

The test captures:
1. Unadjusted series (raw prices)
2. Adjusted series (split-adjusted, backward)
3. Total Return series (forward compounding with dividend reinvestment)

Test period: 2020-08-01 to 2020-09-03 (includes dividend on 8/7 and 4:1 split on 8/31)
"""

import json
from decimal import Decimal
from pathlib import Path

import duckdb
from debug_algoseek_bars import AlgoseekBar, AlgoseekPriceSeries

from qtrader.models.instrument import DataSource, Instrument, InstrumentType


def decimal_to_str(obj):
    """Convert Decimal to string for JSON serialization."""
    if isinstance(obj, Decimal):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def generate_golden_output():
    """Generate golden output from current implementation."""

    # Setup data path and query
    data_path = Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample-complete/SecId=33449")
    parquet_pattern = str(data_path / "*.parquet")

    con = duckdb.connect(":memory:")

    query = f"""
    SELECT
        TradeDate,
        Ticker,
        Open,
        High,
        Low,
        Close,
        MarketHoursVolume,
        CumulativePriceFactor,
        CumulativeVolumeFactor,
        AdjustmentFactor,
        AdjustmentReason
    FROM read_parquet('{parquet_pattern}', hive_partitioning=true)
    WHERE TradeDate BETWEEN '2020-08-01' AND '2020-09-03'
    ORDER BY TradeDate
    """

    data = con.execute(query).fetchdf()
    con.close()

    # Create instrument
    instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)

    # Parse vendor bars
    vendor_bars = [AlgoseekBar(**row.to_dict()) for index, row in data.iterrows()]

    # Create vendor series
    algoseek_series = AlgoseekPriceSeries(instrument=instrument, bars=vendor_bars)

    # Generate canonical series (current implementation)
    canonical_series = algoseek_series.to_canonical_series()

    # Extract golden data
    golden_data = {
        "metadata": {
            "description": "Golden test output for price series refactoring",
            "symbol": "AAPL",
            "start_date": "2020-08-01",
            "end_date": "2020-09-03",
            "corporate_events": {
                "dividend": {
                    "date": "2020-08-07",
                    "amount": "0.82",  # Before split adjustment
                    "adjusted_amount": "0.205",  # After 4:1 split
                },
                "split": {"date": "2020-08-31", "ratio": "4:1", "factor": 4.0},
            },
        },
        "series": {},
    }

    # Capture each series
    for mode_name, series in canonical_series.items():
        bars_data = []
        for bar in series.bars:
            bar_dict = {
                "trade_datetime": bar.trade_datetime,
                "open": round(bar.open, 2),
                "high": round(bar.high, 2),
                "low": round(bar.low, 2),
                "close": round(bar.close, 2),
                "volume": bar.volume,
            }
            if bar.dividend is not None:
                bar_dict["dividend"] = str(bar.dividend)
            bars_data.append(bar_dict)

        golden_data["series"][mode_name] = {"mode": series.mode, "bar_count": len(bars_data), "bars": bars_data}

    return golden_data


def save_golden_output(golden_data: dict, output_path: Path):
    """Save golden output to JSON file."""
    with open(output_path, "w") as f:
        json.dump(golden_data, f, indent=2, default=decimal_to_str)
    print(f"✅ Golden output saved to: {output_path}")


def display_summary(golden_data: dict):
    """Display summary of golden output."""
    print("\n" + "=" * 100)
    print("GOLDEN OUTPUT SUMMARY")
    print("=" * 100)

    metadata = golden_data["metadata"]
    print(f"\nSymbol: {metadata['symbol']}")
    print(f"Period: {metadata['start_date']} to {metadata['end_date']}")

    print("\nCorporate Events:")
    div = metadata["corporate_events"]["dividend"]
    print(f"  - Dividend on {div['date']}: ${div['amount']} (adjusted: ${div['adjusted_amount']})")
    split = metadata["corporate_events"]["split"]
    print(f"  - Split on {split['date']}: {split['ratio']} (factor: {split['factor']})")

    print("\nSeries Generated:")
    for mode_name, series_data in golden_data["series"].items():
        print(f"  - {mode_name}: {series_data['bar_count']} bars")

    # Show key validation points
    print("\n" + "-" * 100)
    print("KEY VALIDATION POINTS")
    print("-" * 100)

    # Find bars on key dates
    key_dates = ["2020-08-07", "2020-08-28", "2020-08-31", "2020-09-02"]

    for date in key_dates:
        print(f"\n{date}:")
        for mode_name, series_data in golden_data["series"].items():
            bar = next((b for b in series_data["bars"] if b["trade_datetime"] == date), None)
            if bar:
                div_str = f" | Div: ${bar['dividend']}" if "dividend" in bar else ""
                print(f"  {mode_name:15} | C: ${bar['close']:8.2f} V: {bar['volume']:12,}{div_str}")


if __name__ == "__main__":
    print("=" * 100)
    print("GENERATING GOLDEN OUTPUT FROM CURRENT IMPLEMENTATION")
    print("=" * 100)

    # Generate golden output
    golden_data = generate_golden_output()

    # Save to file
    output_path = Path("src/qtrader/1playground/golden_output.json")
    save_golden_output(golden_data, output_path)

    # Display summary
    display_summary(golden_data)

    print("\n" + "=" * 100)
    print("✅ GOLDEN OUTPUT GENERATION COMPLETE")
    print("=" * 100)
    print("\nThis file can now be used to validate the refactored architecture.")
    print("After refactoring, run the validation test to ensure outputs match.")
