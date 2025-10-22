# Algoseek Data Source Integration Guide

Complete guide to using Algoseek historical market data in QTrader.

> **📌 Critical Distinction**: Algoseek provides **unadjusted** raw price data (as-traded prices from the exchange). All price adjustments (split-adjusted, total return) are **transformations performed by QTrader** using the adjustment factors provided by Algoseek. The adapter always returns unadjusted data.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Data Structure](#data-structure)
- [Corporate Actions](#corporate-actions)
- [Adjustment Modes](#adjustment-modes)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

______________________________________________________________________

## Overview

The Algoseek integration provides access to high-quality historical US equity market data with comprehensive corporate action tracking.

> **⚠️ Important**: Algoseek provides **unadjusted** raw price data with adjustment factors. QTrader's transformation layer converts this into split-adjusted or total return prices as needed. The adapter always returns unadjusted prices.

Key features:

- ✅ **Unadjusted Raw Data** - As-traded prices from the exchange
- ✅ **Adjustment Factors Included** - Cumulative factors for transformations
- ✅ **Multi-Mode Transformations** - QTrader converts to split-adjusted or total return
- ✅ **Corporate Action Tracking** - Complete split and dividend history
- ✅ **Parquet Format** - Fast, efficient columnar storage
- ✅ **DuckDB Integration** - Lightning-fast queries with SQL
- ✅ **Long History** - Up to 20+ years of historical data
- ✅ **High Quality** - Institutional-grade market data

### Data Characteristics

| Feature                | Details                                                  |
| ---------------------- | -------------------------------------------------------- |
| **Raw Data Format**    | **Unadjusted prices** (as-traded on exchange)            |
| **Asset Classes**      | US Equities                                              |
| **Adjustment Factors** | Included (CumulativePriceFactor, CumulativeVolumeFactor) |
| **Transformations**    | Performed by QTrader (split-adjusted, total return)      |
| **Format**             | Parquet files (Hive partitioned by SecId)                |
| **Frequency**          | Daily OHLC + Volume                                      |
| **Historical Depth**   | 20+ years (depends on dataset subscription)              |
| **Query Engine**       | DuckDB (in-memory SQL)                                   |
| **Corporate Actions**  | Splits, dividends, stock dividends tracked               |

______________________________________________________________________

## Prerequisites

### 1. Algoseek Dataset

You need access to an Algoseek dataset, specifically:

- **US Equity Daily OHLC Standard Adjusted**
- Format: Parquet files with Hive partitioning
- Structure: `SecId=<id>/data.parquet`

### 2. Security Master File

A CSV file mapping symbols to SecIds:

```csv
Symbol,SecId,Name,Exchange,Type
AAPL,33449,Apple Inc.,XNAS,Common Stock
MSFT,59328,Microsoft Corporation,XNAS,Common Stock
...
```

### 3. System Requirements

- Python 3.11+
- DuckDB library (installed automatically)
- Sufficient disk space for Parquet files

______________________________________________________________________

## Setup

### Step 1: Organize Data Files

Place your Algoseek data in the following structure:

```
data/
└── us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample/
    ├── SecId=33449/
    │   └── data.parquet    # AAPL
    ├── SecId=59328/
    │   └── data.parquet    # MSFT
    └── ...
```

### Step 2: Place Security Master File

Place the security master CSV:

```
data/
└── equity_security_master_sample.csv
```

### Step 3: Verify Configuration

The Algoseek data source is pre-configured in `config/data_sources.yaml`:

```yaml
algoseek:
  adapter: algoseekOHLC
  root_path: "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample"
  mode: standard_adjusted
  path_template: "{root_path}/SecId={secid}/*.parquet"
  symbol_map: "data/equity_security_master_sample.csv"
```

______________________________________________________________________

## Usage

### CLI: View Raw Data

```bash
# View Apple stock data from 2020
qtrader raw-data --symbol AAPL --start-date 2020-01-01 --end-date 2020-12-31 --source algoseek
```

Output:

```
Loading data for AAPL from algoseek...
Reading bars from 2020-01-01 to 2020-12-31...
Loaded 253 bars

Press ENTER to view next bar, CTRL+C to exit

┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Field                    ┃ Value          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ Trade Date               │ 2020-01-02     │
│ Open                     │ $74.06         │
│ High                     │ $75.15         │
│ Low                      │ $73.80         │
│ Close                    │ $75.09         │
│ Volume                   │ 135,480,400    │
│ Cumulative Price Factor  │ 0.962345       │
│ Cumulative Volume Factor │ 1.000000       │
└──────────────────────────┴────────────────┘
```

### Python API

```python
from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import DataSource, Instrument, InstrumentType

# Create instrument
instrument = Instrument(
    symbol="AAPL",
    instrument_type=InstrumentType.EQUITY,
    data_source=DataSource.ALGOSEEK
)

# Resolve to adapter
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)

# Fetch raw unadjusted bars
bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

# Access bar data (all prices are unadjusted)
for bar in bars:
    print(f"{bar.TradeDate}: Close=${bar.Close:.2f}, Volume={bar.MarketHoursVolume:,}")
```

______________________________________________________________________

## Data Structure

### AlgoseekBar Fields

Each bar from Algoseek contains **unadjusted** prices plus adjustment factors:

> **Note**: All OHLC prices (Open, High, Low, Close) are **unadjusted** - these are the actual prices traded on the exchange. To get split-adjusted or total return prices, use QTrader's transformation layer (see [Adjustment Modes](#adjustment-modes)).

| Field                      | Type     | Description                                            |
| -------------------------- | -------- | ------------------------------------------------------ |
| **TradeDate**              | datetime | Trading date                                           |
| **Ticker**                 | str      | Stock symbol                                           |
| **Open**                   | float    | Unadjusted opening price                               |
| **High**                   | float    | Unadjusted high price                                  |
| **Low**                    | float    | Unadjusted low price                                   |
| **Close**                  | float    | Unadjusted closing price                               |
| **MarketHoursVolume**      | int      | Volume traded during market hours                      |
| **CumulativePriceFactor**  | float    | Cumulative adjustment factor for price (splits + divs) |
| **CumulativeVolumeFactor** | float    | Cumulative adjustment factor for volume (splits only)  |
| **AdjustmentFactor**       | float    | Adjustment factor on this specific date (if any)       |
| **AdjustmentReason**       | str      | Reason for adjustment (CashDiv, Subdiv, etc.)          |

### Adjustment Factors

Algoseek provides **unadjusted prices** along with two types of cumulative factors that QTrader uses for transformations:

1. **CumulativePriceFactor**: Accounts for splits AND dividends

   - Used by QTrader for total return calculations
   - Includes cash dividend reinvestment
   - Always ≤ 1.0

1. **CumulativeVolumeFactor**: Accounts for splits ONLY

   - Used by QTrader for split-adjusted prices
   - Preserves actual share counts
   - Multiplies with splits, divides with reverse splits

> **Key Point**: The adapter returns raw unadjusted data. QTrader's `AlgoseekPriceSeries.to_canonical()` method applies these factors to create adjusted and total return price series.

______________________________________________________________________

## Corporate Actions

### Detecting Corporate Actions

AlgoseekBar provides helper methods to detect events:

```python
bar = bars[0]

# Check for dividend
if bar.is_dividend():
    amount = bar.get_dividend_amount()
    print(f"Cash dividend: ${amount:.2f}")

# Check for split
if bar.is_split():
    ratio = bar.get_split_ratio()
    print(f"Split ratio: {ratio}:1")
```

### Dividend Types

| AdjustmentReason  | Description           | Example                     |
| ----------------- | --------------------- | --------------------------- |
| **CashDiv**       | Cash dividend         | $0.25 per share             |
| **ScriptDiv**     | Stock dividend        | Additional shares issued    |
| **ScriptDivDiff** | Stock dividend (diff) | Differential stock dividend |

### Split Types

| AdjustmentReason | Description   | Example                     |
| ---------------- | ------------- | --------------------------- |
| **Subdiv**       | Forward split | 4:1 split (4 new for 1 old) |
| **RevSub**       | Reverse split | 1:8 split (1 new for 8 old) |

### Corporate Action Examples

```python
# Example: Detect Apple's 4:1 split on 2020-08-31
bar = adapter.read_bars("2020-08-31", "2020-08-31")[0]

if bar.is_split():
    ratio = bar.get_split_ratio()
    print(f"Split detected: {ratio}:1")  # Output: 4.0:1
    print(f"Adjustment Factor: {bar.AdjustmentFactor}")  # 0.25
```

______________________________________________________________________

## Adjustment Modes

**Important**: The Algoseek adapter returns **unadjusted** data. All price adjustments are performed by QTrader's transformation layer using the provided adjustment factors.

Algoseek data can be transformed into three adjustment modes within QTrader:

### 1. Unadjusted (Raw Data)

Raw prices as traded on the exchange (default from adapter):

```python
# Unadjusted prices - this is what the adapter returns
bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))
bar = bars[0]
print(f"Unadjusted Close: ${bar.Close:.2f}")  # Raw price from exchange
```

**Use Cases:**

- Order execution simulation
- Intraday trading strategies
- Real-time price matching

### 2. Split-Adjusted (QTrader Transformation)

Adjusted for splits only (dividends NOT reinvested) - **transformation performed by QTrader**:

```python
from qtrader.services.data.adapters.models.algoseek import AlgoseekPriceSeries

# QTrader transforms unadjusted data to split-adjusted
series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
canonical_series = series.to_canonical()  # Transformation happens here

# Access split-adjusted mode
adjusted_bars = canonical_series.adjusted.bars
bar = adjusted_bars[0]
print(f"Split-Adjusted Close: ${bar.close:.2f}")  # Transformed by QTrader
```

**Use Cases:**

- Price chart visualization
- Technical analysis
- Historical price comparisons

### 3. Total Return (QTrader Transformation)

Adjusted for splits AND dividends (reinvested) - **transformation performed by QTrader**:

```python
# Access total return mode (also transformed by QTrader)
total_return_bars = canonical_series.total_return.bars
bar = total_return_bars[0]
print(f"Total Return Close: ${bar.close:.2f}")  # Transformed by QTrader
```

**Use Cases:**

- Portfolio performance calculation
- Dividend-adjusted backtesting
- Investment return analysis

### Adjustment Formula (Applied by QTrader)

```python
# Split-adjusted price (QTrader applies this transformation)
adjusted_price = unadjusted_price / CumulativeVolumeFactor

# Total return price (QTrader applies this transformation)
total_return_price = unadjusted_price / CumulativePriceFactor
```

______________________________________________________________________

## Troubleshooting

### Symbol Not Found

**Symptom**: `ValueError: Symbol not found in symbol map: XYZ`

**Solutions**:

1. Verify symbol exists in security master CSV
1. Check symbol spelling and capitalization
1. Confirm symbol was publicly traded during requested period
1. Review `data/equity_security_master_sample.csv` for available symbols

### Data Source Not Found

**Symptom**: `FileNotFoundError: Data source not found`

**Solutions**:

1. Verify data path in `config/data_sources.yaml`
1. Check Parquet files exist: `ls data/us-equity-daily-ohlc-*/SecId=*/*.parquet`
1. Ensure Hive partitioning structure: `SecId=XXXXX/data.parquet`
1. Confirm read permissions on data directory

### No Parquet Files Found

**Symptom**: `FileNotFoundError: No parquet files found`

**Solutions**:

1. Check SecId mapping is correct
1. Verify Parquet file naming: `data.parquet` (not `data_part_*.parquet`)
1. Confirm Hive partition format: `SecId={number}`
1. Review DuckDB compatibility with Parquet version

### OHLC Validation Warnings

**Symptom**: `⚠️ OHLC warnings: High < Open`

**Explanation**: Algoseek adjustment calculations can create minor OHLC violations due to rounding.

**Actions**:

- Warnings with \<10% tolerance are acceptable
- Severe violations (High < Low) will raise errors
- These are adjustment artifacts, not data quality issues

______________________________________________________________________

## Examples

### Example 1: Basic Data Loading

```python
from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import DataSource, Instrument, InstrumentType

# Setup
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)

# Load one year of data
bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))
print(f"Loaded {len(bars)} trading days")

# First and last bar
print(f"First: {bars[0].TradeDate} - ${bars[0].Close:.2f}")
print(f"Last:  {bars[-1].TradeDate} - ${bars[-1].Close:.2f}")
```

### Example 2: Corporate Action Detection

```python
# Load data around Apple's 4:1 split
bars = list(adapter.read_bars("2020-08-01", "2020-09-30"))

# Find corporate action events
events = []
for bar in bars:
    if bar.is_split():
        events.append({
            "date": bar.TradeDate,
            "type": "Split",
            "ratio": bar.get_split_ratio(),
            "factor": bar.AdjustmentFactor
        })
    elif bar.is_dividend():
        events.append({
            "date": bar.TradeDate,
            "type": "Dividend",
            "amount": bar.get_dividend_amount(),
            "factor": bar.AdjustmentFactor
        })

# Display events
for event in events:
    print(f"{event['date']}: {event['type']} - {event}")
```

### Example 3: Multi-Mode Price Comparison

```python
from qtrader.services.data.adapters.models.algoseek import AlgoseekPriceSeries

# Load raw unadjusted bars from adapter
bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))
print(f"Adapter returned {len(bars)} UNADJUSTED bars")

# QTrader transforms unadjusted data into all adjustment modes
series = AlgoseekPriceSeries(symbol="AAPL", bars=bars)
canonical_series = series.to_canonical()  # Transformations happen here!

# Compare first bar in each mode
date = bars[0].TradeDate
unadj = canonical_series.unadjusted.bars[0]  # Original raw data
adj = canonical_series.adjusted.bars[0]      # QTrader transformed (splits only)
total = canonical_series.total_return.bars[0]  # QTrader transformed (splits + divs)

print(f"Date: {date}")
print(f"Unadjusted (raw from Algoseek): ${unadj.close:.2f}")
print(f"Split-Adj (QTrader transform):  ${adj.close:.2f}")
print(f"Total Return (QTrader transform): ${total.close:.2f}")
```

### Example 4: Volume Analysis

```python
# Load bars
bars = list(adapter.read_bars("2020-01-01", "2020-12-31"))

# Calculate volume statistics
volumes = [bar.MarketHoursVolume for bar in bars]
avg_volume = sum(volumes) / len(volumes)
max_volume = max(volumes)
max_volume_date = bars[volumes.index(max_volume)].TradeDate

print(f"Average Daily Volume: {avg_volume:,.0f}")
print(f"Maximum Volume: {max_volume:,} on {max_volume_date}")

# Find high-volume days (> 2x average)
high_volume_days = [
    (bar.TradeDate, bar.MarketHoursVolume)
    for bar in bars
    if bar.MarketHoursVolume > avg_volume * 2
]

print(f"\nHigh-Volume Days ({len(high_volume_days)}):")
for date, volume in high_volume_days[:5]:
    print(f"  {date}: {volume:,}")
```

### Example 5: DuckDB Direct Query

```python
import duckdb

# For advanced users: Direct DuckDB query
conn = duckdb.connect(":memory:")

query = """
SELECT
    TradeDate,
    Close,
    MarketHoursVolume,
    AdjustmentReason
FROM read_parquet('data/us-equity-daily-ohlc-*/SecId=33449/*.parquet')
WHERE TradeDate >= '2020-01-01'
  AND TradeDate <= '2020-12-31'
  AND AdjustmentReason IS NOT NULL
ORDER BY TradeDate
"""

results = conn.execute(query).fetchall()
print(f"Found {len(results)} corporate action events")
```

______________________________________________________________________

## Additional Resources

- **Algoseek Website**: <https://www.algoseek.com>
- **QTrader Examples**: `examples/buy_and_hold_strategy.py`
- **AlgoseekBar Model**: `src/qtrader/models/vendors/algoseek.py`
- **Adapter Implementation**: `src/qtrader/adapters/algoseek.py`

______________________________________________________________________

## Performance Characteristics

### Query Speed

| Data Range | Bars Loaded | Load Time | Notes                   |
| ---------- | ----------- | --------- | ----------------------- |
| 1 month    | ~20 bars    | \<0.1s    | Instant                 |
| 1 year     | ~250 bars   | 0.1-0.2s  | DuckDB Parquet scan     |
| 5 years    | ~1250 bars  | 0.3-0.5s  | Multiple partition scan |
| 20 years   | ~5000 bars  | 1-2s      | Full historical load    |

### Storage Efficiency

- **Parquet Compression**: ~10x smaller than CSV
- **Per Symbol**: ~100KB per year (compressed)
- **Full Dataset**: Varies by universe size
- **Hive Partitioning**: Efficient SecId-based filtering

______________________________________________________________________

## Support

For issues or questions:

1. Check data file structure and permissions
1. Verify SecId mapping in security master CSV
1. Review DuckDB query logs
1. Inspect Parquet file metadata: `parquet-tools schema data.parquet`
1. Open an issue on GitHub

______________________________________________________________________

**Last Updated**: October 2025\
**Data Vendor**: Algoseek\
**Dataset**: US Equity Daily OHLC Standard Adjusted\
**QTrader Version**: 0.1.0
