"""
Test that the new data layer produces output matching golden output.

This test validates that moving the code from playground to the main
qtrader package didn't break anything.
"""

import sys
from pathlib import Path

import duckdb

from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries

# Add playground to path to access validation tools
# Note: Import after standard imports due to sys.path modification
playground_path = Path(__file__).parent
sys.path.insert(0, str(playground_path))

from test_validate_golden import validate_against_golden  # noqa: E402


def generate_test_data_from_new_implementation():
    """Generate test data using the new data layer implementation."""

    # Setup data path and query (same as golden output generation)
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

    # Parse vendor bars using new models
    vendor_bars = [AlgoseekBar(**row.to_dict()) for index, row in data.iterrows()]

    # Get symbol from first bar
    symbol = vendor_bars[0].Ticker if vendor_bars else "AAPL"

    # Create vendor series using new models
    algoseek_series = AlgoseekPriceSeries(symbol=symbol, bars=vendor_bars)

    # Generate canonical series (using new implementation)
    canonical_series = algoseek_series.to_canonical_series()

    # Convert to same structure as golden output
    test_data = {
        "metadata": {
            "description": "Test output from new data layer implementation",
            "symbol": symbol,
            "start_date": "2020-08-01",
            "end_date": "2020-09-03",
            "corporate_events": {
                "dividend": {"date": "2020-08-07", "amount": "0.82", "adjusted_amount": "0.205"},
                "split": {"date": "2020-08-31", "ratio": "4:1", "factor": 4.0},
            },
        },
        "series": {},
    }

    # Convert canonical series to test data format
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

        test_data["series"][mode_name] = {
            "mode": series.mode,
            "bar_count": len(bars_data),
            "bars": bars_data,
        }

    return test_data


if __name__ == "__main__":
    print("=" * 100)
    print("TESTING NEW DATA LAYER AGAINST GOLDEN OUTPUT")
    print("=" * 100)

    print("\nGenerating test data from new implementation...")
    test_data = generate_test_data_from_new_implementation()

    print(f"✅ Generated {len(test_data['series'])} series")
    for mode, series_data in test_data["series"].items():
        print(f"   - {mode}: {series_data['bar_count']} bars")

    print("\nValidating against golden output...")
    golden_path = Path("src/qtrader/1playground/golden_output.json")

    passed = validate_against_golden(test_data, golden_path)

    if passed:
        print("\n" + "=" * 100)
        print("🎉 SUCCESS! New data layer produces identical output to golden standard")
        print("=" * 100)
        sys.exit(0)
    else:
        print("\n" + "=" * 100)
        print("❌ FAILED! New data layer output doesn't match golden standard")
        print("=" * 100)
        sys.exit(1)
