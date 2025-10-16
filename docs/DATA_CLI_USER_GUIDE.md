# Data Management CLI - User Guide

This guide explains how to use QTrader's data management commands for browsing, updating, and maintaining cached market data.

## Overview

All data management commands are organized under `qtrader data`:

```bash
qtrader data --help
```

Commands:

- `qtrader data raw` - Browse raw historical data interactively
- `qtrader data update` - Update cached data to latest
- `qtrader data cache-info` - View cache status and contents

## Commands

### 1. Browse Raw Data (`qtrader data raw`)

Displays historical price data interactively, one bar at a time.

**Usage:**

```bash
qtrader data raw --symbol AAPL --start-date 2020-01-01 --end-date 2020-01-31
qtrader data raw --symbol TSLA --start-date 2024-01-01 --end-date 2024-12-31 --source schwab
```

**Options:**

- `--symbol` (required): Stock symbol (e.g., AAPL, TSLA)
- `--start-date` (required): Start date in YYYY-MM-DD format
- `--end-date` (required): End date in YYYY-MM-DD format
- `--source`: Data source - `algoseek` (default) or `schwab`

**Interactive Controls:**

- Press ENTER to view next bar
- Press CTRL+C to exit

**Example Output:**

```
Loading data for AAPL from schwab...
Loaded 20 bars
Displaying raw unadjusted prices
Press ENTER to view next bar, CTRL+C to exit

┏━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Field   ┃ Value       ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Date    │ 2020-01-02  │
│ Open    │ $297.68     │
│ High    │ $300.12     │
│ Low     │ $296.50     │
│ Close   │ $300.35     │
│ Volume  │ 135,480,400 │
└─────────┴─────────────┘
```

______________________________________________________________________

### 2. Update Cached Data (`qtrader data update`)

Updates cached data to latest available, fetching only new bars since last update.

**Usage:**

```bash
# Update all symbols in a dataset
qtrader data update --dataset schwab-us-equity-1d-adjusted

# Update specific symbols
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,TSLA,NVDA

# Dry run (check what would be updated)
qtrader data update --dataset schwab-us-equity-1d-adjusted --dry-run --verbose

# Update Algoseek dataset
qtrader data update --dataset algoseek-us-equity-1d-adjusted
```

**Options:**

- `--dataset` (required): Dataset identifier from `data_sources.yaml`
  - `schwab-us-equity-1d-adjusted` - Schwab daily adjusted data
  - `algoseek-us-equity-1d-adjusted` - Algoseek daily adjusted data
- `--symbols`: Comma-separated list of symbols (default: all cached symbols)
- `--dry-run`: Show what would be updated without making changes
- `--verbose`: Show detailed progress

**Example Output:**

```bash
$ qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,TSLA

UPDATING Dataset: schwab-us-equity-1d-adjusted

Updating 2 symbols...

⠹ ✓ TSLA ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2/2 100% 0:00:03

┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Status     ┃ Bars Added ┃ Date Range                ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ ✓ Updated  │          5 │ 2025-10-11 to 2025-10-15  │
│ TSLA   │ ✓ Updated  │          5 │ 2025-10-11 to 2025-10-15  │
└────────┴────────────┴────────────┴───────────────────────────┘

Successful: 2/2
Total bars added: 10
```

**Dry Run Example:**

```bash
$ qtrader data update --dataset schwab-us-equity-1d-adjusted --dry-run --verbose

DRY RUN Dataset: schwab-us-equity-1d-adjusted

Updating all cached symbols...

┏━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Status     ┃ Bars Added ┃ Date Range                ┃
┡━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ ✓ Current  │          - │ -                         │
│ TSLA   │ ✓ Updated  │          3 │ 2025-10-13 to 2025-10-15  │
│ NVDA   │ ✓ Updated  │          3 │ 2025-10-13 to 2025-10-15  │
└────────┴────────────┴────────────┴───────────────────────────┘

This was a dry run. Use --no-dry-run to actually update data.
```

______________________________________________________________________

### 3. View Cache Info (`qtrader data cache-info`)

Displays information about cached data for a dataset.

**Usage:**

```bash
qtrader data cache-info --dataset schwab-us-equity-1d-adjusted
qtrader data cache-info --dataset algoseek-us-equity-1d-adjusted
```

**Options:**

- `--dataset` (required): Dataset identifier from `data_sources.yaml`

**Example Output:**

```bash
$ qtrader data cache-info --dataset schwab-us-equity-1d-adjusted

Dataset: schwab-us-equity-1d-adjusted
Cache location: /home/user/.cache/qtrader/schwab
Cached symbols: 3

┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Symbol ┃ Cache File              ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ AAPL   │ AAPL/data.parquet       │
│ NVDA   │ NVDA/data.parquet       │
│ TSLA   │ TSLA/data.parquet       │
└────────┴─────────────────────────┘
```

______________________________________________________________________

## Smart Caching Feature

The `update` command uses **smart caching** to minimize API calls:

### How It Works

1. **Initial Fetch** (first time for a symbol):

   - Fetches all available history from API
   - Stores in cache with metadata

1. **Incremental Updates** (subsequent calls):

   - Reads last cached date from metadata
   - Fetches only new bars: `(last_date + 1)` to `today`
   - Merges with existing cache
   - Updates metadata

1. **Gap Filling**:

   - Detects missing date ranges
   - Fetches only gaps from API
   - Merges all data intelligently

### Performance Benefits

**Example: Daily update for 500 symbols**

❌ **Without smart caching:**

- Refetch entire year: 500 symbols × 252 bars = 126,000 API calls
- Time: ~3.5 hours (at 10 req/sec)

✅ **With smart caching:**

- Fetch only today: 500 symbols × 1 bar = 500 API calls
- Time: ~50 seconds (at 10 req/sec)

**Result: 99.6% reduction in API calls!**

______________________________________________________________________

## Configuration

Dataset configurations are in `config/data_sources.yaml`:

```yaml
data_sources:
  schwab-us-equity-1d-adjusted:
    adapter: schwabOHLC
    cache_root: "${HOME}/.cache/qtrader/schwab"
    cache_strategy: "smart"           # smart | simple | disabled
    enable_incremental_update: true   # Enable incremental updates
    update_mode: "manual"             # auto | manual
    force_refresh: false              # Ignore cache (for testing)
```

**Cache Strategies:**

1. **`smart`** (recommended):

   - Gap-filling
   - Incremental updates
   - Minimal API calls

1. **`simple`** (legacy):

   - All-or-nothing caching
   - Refetches if cache doesn't cover full range

1. **`disabled`**:

   - No caching
   - Always fetch from API

**Update Modes:**

- **`manual`**: Update only when `qtrader data update` is called
- **`auto`**: Auto-update cache before reading (use with caution)

______________________________________________________________________

## Workflow Examples

### Daily Update Workflow

```bash
# Morning routine: update all cached symbols
qtrader data update --dataset schwab-us-equity-1d-adjusted --verbose

# Or update just your watchlist
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,TSLA,NVDA,GOOGL
```

### Initial Setup Workflow

```bash
# 1. Browse data to verify source works
qtrader data raw --symbol AAPL --start-date 2024-01-01 --end-date 2024-01-31 --source schwab

# 2. This creates initial cache for AAPL (fetches all history)

# 3. Check cache status
qtrader data cache-info --dataset schwab-us-equity-1d-adjusted

# 4. Update to latest (incremental)
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL
```

### Backfill Workflow

```bash
# Browse data for multiple symbols (creates cache)
qtrader data raw --symbol TSLA --start-date 2020-01-01 --end-date 2024-12-31 --source schwab
qtrader data raw --symbol NVDA --start-date 2020-01-01 --end-date 2024-12-31 --source schwab

# Update all to latest
qtrader data update --dataset schwab-us-equity-1d-adjusted
```

______________________________________________________________________

## Troubleshooting

### No symbols found to update

**Problem:**

```
No symbols found to update
```

**Solution:**

- Cache is empty. Browse data first to create initial cache:
  ```bash
  qtrader data raw --symbol AAPL --start-date 2020-01-01 --end-date 2024-12-31 --source schwab
  ```

### Dataset not found

**Problem:**

```
Dataset 'xyz' not found. Available: [...]
```

**Solution:**

- Check available datasets in `config/data_sources.yaml`
- Use exact dataset name from configuration

### Adapter doesn't support updates

**Problem:**

```
Adapter 'xyz' does not support incremental updates
```

**Solution:**

- Only adapters with `update_to_latest()` method support incremental updates
- Currently supported: Schwab
- Coming soon: Algoseek

### API rate limits

**Problem:**

```
Rate limit exceeded
```

**Solution:**

- Schwab adapter has built-in rate limiting (10 req/sec)
- If updating many symbols, use `--verbose` to monitor progress
- Consider updating in smaller batches:
  ```bash
  qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,TSLA
  ```

______________________________________________________________________

## Dataset Support

| Dataset                  | Browse (`raw`) | Update         | Cache Strategy          |
| ------------------------ | -------------- | -------------- | ----------------------- |
| Schwab US Equity Daily   | ✅             | ✅             | Smart, Simple, Disabled |
| Algoseek US Equity Daily | ✅             | 🔄 Coming Soon | Simple                  |

Legend:

- ✅ Fully supported
- 🔄 In development
- ❌ Not supported

______________________________________________________________________

## Related Documentation

- [Schwab Smart Caching Implementation](./SCHWAB_SMART_CACHING_IMPLEMENTATION.md)
- [CLI Use Cases Compatibility](./CLI_USE_CASES_COMPATIBILITY.md)
- [Data Sources Configuration](../config/README.md)
