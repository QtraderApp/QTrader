"""Generate golden output for price series testing.

Extracts data and generates golden output JSON with metadata and all three series
(unadjusted, adjusted, total_return) for validation testing.

Default: AAPL from 2020-08-01 to 2020-09-01 (includes dividend 8/7 and 4:1 split 8/31)
"""

import json
from pathlib import Path

import duckdb

from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries

# Configuration
DATA_PATH = Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample-complete/SecId=33449")
START_DATE = "2020-08-01"
END_DATE = "2020-09-01"
SYMBOL = "AAPL"
OUTPUT_FILE = Path("src/qtrader/1playground/golden_output_new.json")
CSV_OUTPUT_FILE = Path("src/qtrader/1playground/golden_output_new.csv")

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

# Detect corporate events
dividend_events = []
split_events = []
prev_bar = None
for bar in bars:
    if bar.is_dividend() and prev_bar:
        amt = bar.get_dividend_amount(prev_bar.Close)
        dividend_events.append({"date": bar.TradeDate.date().isoformat(), "amount": f"{amt:.2f}" if amt else None})
    if bar.is_split():
        ratio = bar.get_split_ratio()
        split_events.append(
            {
                "date": bar.TradeDate.date().isoformat(),
                "ratio": f"{ratio}:1" if ratio else None,
                "factor": float(ratio) if ratio else None,
            }
        )
    prev_bar = bar

print(f"\n{'=' * 80}")
print(f"Generating Golden Output: {SYMBOL} {START_DATE} to {END_DATE}")
print(f"Total bars: {len(bars)}")
print(f"{'=' * 80}\n")

# Transform to canonical series
series = AlgoseekPriceSeries(symbol=SYMBOL, bars=bars)
canonical = series.to_canonical_series()

# Build golden output structure
golden_output = {
    "metadata": {
        "description": "Golden test output for price series validation",
        "symbol": SYMBOL,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "bar_count": len(bars),
        "corporate_events": {"dividends": dividend_events, "splits": split_events},
    },
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

    golden_output["series"][mode] = {"mode": mode, "bar_count": len(bars_data), "bars": bars_data}

# Save to JSON file
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_FILE, "w") as f:
    json.dump(golden_output, f, indent=2)

print(f"✅ Golden output saved to: {OUTPUT_FILE}")
print(f"   - Metadata: {SYMBOL}, {len(bars)} bars")
print(f"   - Corporate events: {len(dividend_events)} dividends, {len(split_events)} splits")
print("   - Series: 3 (unadjusted, adjusted, total_return)")
print()

# Display summary for each series
for mode in ["unadjusted", "adjusted", "total_return"]:
    price_series = canonical[mode]

    print(f"\n{'=' * 80}")
    print(f"MODE: {mode.upper()}")
    print(f"{'=' * 80}")

    # Summary statistics
    first_close = price_series.bars[0].close
    last_close = price_series.bars[-1].close
    return_pct = ((last_close / first_close) - 1) * 100

    print(f"First close: ${first_close:.2f}")
    print(f"Last close:  ${last_close:.2f}")
    print(f"Return:      {return_pct:+.2f}%")
    print(f"Bars:        {len(price_series.bars)}")

print(f"\n{'=' * 80}")
print("CORPORATE EVENTS DETECTED")
print(f"{'=' * 80}")

for event in dividend_events:
    print(f"📊 {event['date']}: DIVIDEND ${event['amount']}")
for event in split_events:
    print(f"📈 {event['date']}: SPLIT {event['ratio']}")

print(f"\n✅ Golden output file: {OUTPUT_FILE}")
print("✅ Use test_validate_golden.py to validate against this output")
print()


# join all series into single CSV for easy inspection
print(f"Generating combined CSV output: {CSV_OUTPUT_FILE}")
with open(CSV_OUTPUT_FILE, "w") as f:
    f.write("mode,trade_datetime,open,high,low,close,volume,dividend\n")
    for mode in ["unadjusted", "adjusted", "total_return"]:
        price_series = canonical[mode]
        for bar in price_series.bars:
            f.write(
                f"{mode},{bar.trade_datetime},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume},"
                f"{str(bar.dividend) if bar.dividend else ''}\n"
            )

print(f"✅ CSV output file: {CSV_OUTPUT_FILE}")
print()
print(f"{'=' * 80}\n")
