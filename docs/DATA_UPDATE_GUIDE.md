but can not# Data Update Guide

## Overview

The `qtrader data update` command automatically handles both **full backfill** (for new symbols) and **incremental updates** (for existing symbols) in a single command.

## Quick Start

### 1. Create Universe File

Create a CSV file with symbols you want to maintain:

```csv
SYMBOL
AAPL
GOOGL
MSFT
AMZN
TSLA
```

Place it in your cache directory: `data/us-equity-daily-adjusted-schwab/universe.csv`

### 2. Configure Dataset

The `universe_file` is already configured in `config/data_sources.yaml`:

```yaml
schwab-us-equity-1d-adjusted:
  cache_root: "data/us-equity-daily-adjusted-schwab"
  universe_file: "data/us-equity-daily-adjusted-schwab/universe.csv"  # Auto-detected
```

### 3. Run Update

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted
```

**That's it!** The command will:

- Read symbols from `universe.csv`
- For each symbol:
  - **No cache?** → Full backfill (all available history from API)
  - **Has cache?** → Incremental update (only new bars)

## Behavior

### Automatic Mode Detection

| Symbol State             | Action            | Data Fetched                                     |
| ------------------------ | ----------------- | ------------------------------------------------ |
| No cache exists          | **Full backfill** | All available history (queries API for earliest) |
| Cache exists, up-to-date | **No update**     | 0 bars (skips API call)                          |
| Cache exists, stale      | **Incremental**   | New bars since last update                       |

### Symbol Source Priority

1. **`--symbols AAPL,TSLA`** - Explicit CLI override
1. **`universe.csv`** - Configured universe file
1. **Scan cache** - Fallback (updates all cached symbols)

## Use Cases

### Daily Maintenance (Recommended)

```bash
# Single command for all updates (initial + incremental)
qtrader data update --dataset schwab-us-equity-1d-adjusted

# Schedule as cron job (weekdays at 6 PM)
0 18 * * 1-5 cd /path/to/QTrader && qtrader data update --dataset schwab-us-equity-1d-adjusted
```

### Add New Symbol

1. Add to `universe.csv`:

   ```csv
   SYMBOL
   AAPL
   GOOGL
   NEWCO  # New symbol
   ```

1. Run update:

   ```bash
   qtrader data update --dataset schwab-us-equity-1d-adjusted
   ```

1. Automatic full backfill for NEWCO, incremental for others

### Ad-hoc Updates

```bash
# Update specific symbols (override universe.csv)
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL,TSLA

# Dry run (see what would be updated)
qtrader data update --dataset schwab-us-equity-1d-adjusted --dry-run

# Verbose output
qtrader data update --dataset schwab-us-equity-1d-adjusted --verbose
```

### Check Cache Status

```bash
qtrader data cache-info --dataset schwab-us-equity-1d-adjusted
```

Output:

```
Symbol  Start Date   End Date     Bars    Last Update
AAPL    2014-10-16   2025-10-16   2756    2025-10-16 14:30:00
GOOGL   2014-10-16   2025-10-16   2756    2025-10-16 14:30:00
MSFT    2014-10-16   2025-10-16   2756    2025-10-16 14:30:00
```

## Configuration

### Full Configuration Example

```yaml
schwab-us-equity-1d-adjusted:
  # Required
  adapter: schwabOHLC
  cache_root: "data/us-equity-daily-adjusted-schwab"

  # Optional: Universe file (if not specified, scans cache)
  universe_file: "data/us-equity-daily-adjusted-schwab/universe.csv"

  # Caching behavior
  cache_strategy: "smart"              # Gap-filling enabled
  enable_incremental_update: true      # Allow incremental updates
  update_mode: "manual"                # CLI-controlled (not auto)

  # OAuth credentials
  client_id: "${SCHWAB_API_KEY}"
  client_secret: "${SCHWAB_API_SECRET}"
  redirect_uri: "${SCHWAB_REDIRECT_URI:-https://127.0.0.1:8182}"
  token_cache_path: null               # Default: ~/.qtrader/schwab_tokens.json

  # Rate limiting
  requests_per_second: 10
```

### Key Settings

- **`universe_file`**: Path to CSV with symbols to maintain (optional)
- **`cache_strategy: "smart"`**: Enables gap-filling and smart caching
- **`enable_incremental_update: true`**: Required for incremental updates
- **`update_mode: "manual"`**: Updates only via CLI (recommended)

## How It Works

### Smart Caching Logic

```python
for symbol in universe_symbols:
    if symbol_has_no_cache:
        # Full backfill - query API for earliest available date
        min_date, max_date = adapter.get_available_date_range()
        bars = adapter.read_bars(min_date, max_date)
        # → Caches ALL available history (e.g., MSFT: 30+ years, TSLA: 2010+)

    else:
        # Incremental update
        last_date = cache.get_last_date()

        if last_date >= today:
            # Already up to date - skip API call
            return 0 bars

        bars = adapter.update_to_latest(last_date + 1, today)
        # → Appends only truly new bars (deduplicates automatically)
```

### Full Backfill Details

When no cache exists:

- **Query API**: Gets earliest available date for the symbol (varies by ticker)
- **Fallback**: If API query fails, uses 20 years as safe default
- **End date**: Today
- **API calls**: 2 requests (1 for date range query, 1 for full data)
- **Result**: All available history cached (e.g., MSFT gets 30+ years, TSLA gets from 2010)
- **Example ranges**:
  - AAPL: ~1980 to today (40+ years)
  - MSFT: ~1986 to today (35+ years)
  - TSLA: ~2010 to today (15+ years)

### Incremental Update Details

When cache exists:

- **Start date**: Last cached date + 1 day
- **End date**: Today
- **API call**: Only if new bars available
- **Result**: New bars appended to cache

## Troubleshooting

### No symbols found to update

**Cause:** No `universe.csv` and no cached symbols

**Solution:**

```bash
# Create universe.csv with symbols
echo "SYMBOL
AAPL
GOOGL" > data/us-equity-daily-adjusted-schwab/universe.csv

# Run update
qtrader data update --dataset schwab-us-equity-1d-adjusted
```

### Symbol fails to backfill

**Cause:** Invalid symbol or API error

**Solution:**

- Verify symbol is valid for Schwab
- Check API credentials
- View error in verbose mode: `--verbose`

### Update says "0 bars added" but cache is empty

**Cause:** Symbol not in universe.csv and not in cache

**Solution:**

- Add symbol to `universe.csv`, or
- Use `--symbols` to specify explicitly

### Rate limit exceeded

**Cause:** Too many symbols, too frequent updates

**Solution:**

- Reduce `requests_per_second` in config
- Split universe into batches
- Don't run multiple updates simultaneously

## Best Practices

### ✅ Recommended

- **Use `universe.csv`** for production datasets
- **Single daily cron job** at market close (6 PM)
- **Monitor with `cache-info`** weekly
- **Keep symbols valid** (user responsibility)
- **Use `--dry-run`** before bulk operations

### ❌ Avoid

- Running updates during market hours (API load)
- Deleting cache unless necessary
- Backfilling 100+ symbols simultaneously (rate limits)
- Setting `update_mode: "auto"` (causes updates on every read)
- Using `force_refresh: true` (ignores cache, hits API every time)

## Migration from Old Backfill Script

If you were using `scripts/backfill_dataset.py`:

**Before:**

```bash
# Initial backfill
python scripts/backfill_dataset.py --dataset X --symbols AAPL --years 10

# Daily updates
qtrader data update --dataset X
```

**After:**

```bash
# Just create universe.csv and run (does both!)
qtrader data update --dataset schwab-us-equity-1d-adjusted
```

The backfill script has been **removed** as it's now redundant.

## Summary

### One Command, All Scenarios

```bash
qtrader data update --dataset schwab-us-equity-1d-adjusted
```

This single command handles:

- ✅ Initial full backfill (new symbols)
- ✅ Incremental updates (existing symbols)
- ✅ Gap filling (missing historical data)
- ✅ Daily maintenance
- ✅ API rate limiting
- ✅ Progress tracking

### Workflow

1. Create `universe.csv` in cache directory
1. Run `qtrader data update --dataset <name>`
1. Schedule as daily cron job
1. Done!

No separate backfill scripts, no complex workflows, just one simple command.
