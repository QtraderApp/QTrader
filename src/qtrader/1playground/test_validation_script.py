"""Test the validation script with newly generated golden output."""

import sys
from pathlib import Path

# Add playground to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import duckdb
from test_validate_golden import validate_against_golden

from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries

# Configuration
DATA_PATH = Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample-complete/SecId=33449")
START_DATE = "2020-08-01"
END_DATE = "2020-09-01"
SYMBOL = "AAPL"
GOLDEN_FILE = Path("src/qtrader/1playground/golden_output_new.json")

# Connect to DuckDB and query data
con = duckdb.connect(":memory:")
parquet_pattern = str(DATA_PATH / "*.parquet")

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
WHERE TradeDate BETWEEN '{START_DATE}' AND '{END_DATE}'
ORDER BY TradeDate
"""

result = con.execute(query)

# Parse into AlgoseekBar objects
bars = []
for row in result.fetchall():
    bar = AlgoseekBar(
        TradeDate=row[0],
        Ticker=row[1],
        Open=row[2],
        High=row[3],
        Low=row[4],
        Close=row[5],
        MarketHoursVolume=row[6],
        CumulativePriceFactor=row[7],
        CumulativeVolumeFactor=row[8],
        AdjustmentFactor=row[9],
        AdjustmentReason=row[10],
    )
    bars.append(bar)

# Transform to canonical series
series = AlgoseekPriceSeries(symbol=SYMBOL, bars=bars)
canonical = series.to_canonical_series()

# Build test data in same format as golden output
test_data = {
    "metadata": {"symbol": SYMBOL, "start_date": START_DATE, "end_date": END_DATE, "bar_count": len(bars)},
    "series": {},
}

# Convert each series to JSON-serializable format
for mode in ["unadjusted", "adjusted", "total_return"]:
    price_series = canonical[mode]

    bars_data = []
    for bar in price_series.bars:
        bar_dict = {
            "trade_datetime": bar.trade_datetime,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        # Include dividend if present
        if bar.dividend is not None:
            bar_dict["dividend"] = str(bar.dividend)

        bars_data.append(bar_dict)

    test_data["series"][mode] = {"mode": mode, "bar_count": len(bars_data), "bars": bars_data}

# Validate against golden output
print("\n" + "=" * 100)
print("TESTING VALIDATION SCRIPT")
print("=" * 100)
print(f"\nValidating test data against: {GOLDEN_FILE}")
print(f"Test data: {len(bars)} bars, 3 series\n")

passed = validate_against_golden(test_data, GOLDEN_FILE)

if passed:
    print("\n✅ Validation script works correctly!")
    print("✅ Test data matches golden output perfectly!")
else:
    print("\n❌ Validation failed - check errors above")
