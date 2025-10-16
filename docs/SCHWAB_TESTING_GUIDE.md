# Testing Schwab Smart Caching with Real Data

This guide walks you through testing the Schwab smart caching implementation with real AAPL data.

## Prerequisites

### 1. Schwab Developer Account

You'll need a Schwab developer account with API credentials:

1. Go to [Schwab Developer Portal](https://developer.schwab.com/)
1. Create an account (if you don't have one)
1. Create a new app to get:
   - **App Key** (client_id)
   - **Secret** (client_secret)

### 2. Configure Redirect URI

In your Schwab app settings:

- Set **Callback/Redirect URL** to: `https://127.0.0.1:8182`

This is required for the OAuth flow to work with the local HTTPS server.

### 3. Set Environment Variables

```bash
# Required
export SCHWAB_API_KEY="your_app_key_here"
export SCHWAB_API_SECRET="your_secret_here"

# Optional (defaults shown)
export SCHWAB_REDIRECT_URI="https://127.0.0.1:8182"
```

**Important:** Keep these credentials secure! Never commit them to git.

You can add these to your shell profile (`~/.zshrc` or `~/.bashrc`) or use a `.env` file.

## Running the Test

### Quick Test

```bash
cd /home/javier/Projects/QTrader
python scripts/test_schwab_aapl.py
```

### What the Test Does

The script performs 6 comprehensive tests:

1. **Environment Check**

   - Verifies SCHWAB_API_KEY and SCHWAB_API_SECRET are set
   - Shows configuration details

1. **Adapter Initialization**

   - Loads configuration from `data_sources.yaml`
   - Creates SchwabOHLCAdapter for AAPL

1. **Initial Fetch (Creates Cache)**

   - Fetches last 30 days of AAPL data
   - **Opens browser for authentication** (one-time)
   - Creates cache in `data/us-equity-daily-adjusted-schwab/AAPL/`
   - Saves metadata with date range

1. **Read from Cache**

   - Re-reads same date range
   - Verifies NO API calls (all from cache)
   - Confirms cache integrity

1. **Incremental Update**

   - Calls `update_to_latest()`
   - Fetches only new bars since last cache update
   - Updates metadata

1. **Gap Filling**

   - Requests extended date range (60 days into future)
   - Detects gap between cache and request
   - Fetches only the gap from API
   - Merges cached + new data

### Expected Output

```
================================================================================
  1. Checking Environment
================================================================================

✓ SCHWAB_API_KEY: XXXXXXXXXX...
✓ SCHWAB_API_SECRET: YYYYYYYYYY...
✓ SCHWAB_REDIRECT_URI: https://127.0.0.1:8182

================================================================================
  2. Initializing Schwab Adapter
================================================================================

✓ Dataset: schwab-us-equity-1d-adjusted
✓ Cache strategy: smart
✓ Incremental updates: True
✓ Cache root: data/us-equity-daily-adjusted-schwab
✓ Adapter initialized for AAPL

================================================================================
  3. Initial Fetch - Last 30 Days (Creates Cache)
================================================================================

Fetching AAPL from 2025-09-16 to 2025-10-16
This will:
  1. Authenticate with Schwab (browser will open)
  2. Fetch data from API
  3. Create cache

Please complete authentication in browser...

✓ Fetched 22 bars
  First bar: 2025-09-16 - Close: $225.77
  Last bar:  2025-10-16 - Close: $233.85

✓ Cache created: data/us-equity-daily-adjusted-schwab/AAPL/data.parquet
  Size: 3.2 KB
✓ Metadata: {'start': '2025-09-16', 'end': '2025-10-16'}

================================================================================
  4. Read from Cache (No API Calls)
================================================================================

✓ Read 22 bars from cache (no API calls)
✓ Cache integrity verified (22 bars match)

================================================================================
  5. Incremental Update (Smart Caching)
================================================================================

Calling update_to_latest()...
This will fetch only new bars since last cache update

✓ Cache already up-to-date (no new bars)

================================================================================
  6. Gap Filling Test
================================================================================

Requesting extended range: 2025-09-16 to 2025-12-15
Smart caching will:
  1. Read cached data
  2. Detect gap (cache end to future end)
  3. Fetch only the gap from API
  4. Merge and return combined data

✓ Retrieved 22 bars
  Original cache: 22 bars
  Gap filled: 0 new bars (future dates)
  Latest bar: 2025-10-16 - Close: $233.85

================================================================================
  7. Summary
================================================================================

✓ Smart caching fully functional!

Features tested:
  ✓ Initial fetch and cache creation
  ✓ Reading from cache (no API calls)
  ✓ Incremental updates (fetch only new bars)
  ✓ Gap filling (detect and fetch missing ranges)

Cache location:
  data/us-equity-daily-adjusted-schwab/AAPL/data.parquet
  data/us-equity-daily-adjusted-schwab/AAPL/.metadata.json

Next steps:
  • Try: qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL --verbose
  • Try: qtrader data cache-info --dataset schwab-us-equity-1d-adjusted
  • Try: qtrader data raw --symbol AAPL --start-date 2024-01-01 --end-date 2024-12-31 --source schwab
```

## Authentication Flow

### First Time Only

The first time you run the script, you'll see:

1. **Browser opens automatically** with Schwab login page
1. **Log in with your Schwab credentials**
1. **Authorize the app** (click "Allow")
1. Schwab redirects to `https://127.0.0.1:8182`
1. You'll see a **browser security warning** (self-signed cert - this is expected)
1. **Click "Advanced" → "Proceed"** (exact wording varies by browser)
1. You'll see: **"Authorization successful! You can close this window."**
1. Script continues automatically

### Token Caching

After first authentication:

- Access token saved to `~/.qtrader/schwab_tokens.json`
- Token auto-refreshes before expiry
- **No browser needed for subsequent runs** (until token expires)

Token file permissions: `600` (owner read/write only)

## Using the CLI Commands

Once the test passes, you can use the new CLI commands:

### View Cache Info

```bash
qtrader data cache-info --dataset schwab-us-equity-1d-adjusted
```

### Update Data

```bash
# Update AAPL to latest
qtrader data update --dataset schwab-us-equity-1d-adjusted --symbols AAPL --verbose

# Update all cached symbols
qtrader data update --dataset schwab-us-equity-1d-adjusted

# Dry run (check what would be updated)
qtrader data update --dataset schwab-us-equity-1d-adjusted --dry-run
```

### Browse Raw Data

```bash
qtrader data raw --symbol AAPL --start-date 2024-01-01 --end-date 2024-12-31 --source schwab
```

## Troubleshooting

### "Missing required environment variables"

**Problem:**

```
❌ ERROR: Missing required environment variables!
```

**Solution:**

```bash
export SCHWAB_API_KEY="your_app_key"
export SCHWAB_API_SECRET="your_secret"
```

### "Connection refused" or "Certificate error"

**Problem:** Browser can't connect to `https://127.0.0.1:8182`

**Solution:**

- Make sure nothing else is using port 8182
- The SSL certificate warning is normal (self-signed cert)
- Click "Advanced" → "Proceed" in browser

### "Invalid client credentials"

**Problem:**

```
❌ ERROR: 401 Unauthorized
```

**Solution:**

- Double-check your SCHWAB_API_KEY and SCHWAB_API_SECRET
- Make sure there are no extra spaces
- Verify credentials in Schwab Developer Portal

### "Redirect URI mismatch"

**Problem:**

```
Error: redirect_uri mismatch
```

**Solution:**

- In Schwab Developer Portal, set Callback URL to: `https://127.0.0.1:8182`
- Must match exactly (including `https://`)

### Token expired

**Problem:**

```
❌ ERROR: 401 Unauthorized (after previously working)
```

**Solution:**

```bash
# Delete cached token
rm ~/.qtrader/schwab_tokens.json

# Run test again (will re-authenticate)
python scripts/test_schwab_aapl.py
```

## Performance Verification

After running the test, check the logs for performance metrics:

**Initial fetch (30 days):**

- 1 API call
- ~22 bars (trading days)
- Creates cache: ~3 KB

**Cache read (same range):**

- 0 API calls
- Instant retrieval

**Incremental update (next day):**

- 1 API call (only new bar)
- Updates cache

**Gap filling (60 day extension):**

- 1 API call (only gap)
- Merges with cache

**Expected API call reduction: 99%+**

## Next Steps

After successful test:

1. **Add more symbols:**

   ```bash
   qtrader data raw --symbol TSLA --start-date 2024-01-01 --end-date 2024-12-31 --source schwab
   qtrader data raw --symbol NVDA --start-date 2024-01-01 --end-date 2024-12-31 --source schwab
   ```

1. **Set up daily update cron job:**

   ```bash
   # Add to crontab
   0 17 * * 1-5 cd /path/to/QTrader && qtrader data update --dataset schwab-us-equity-1d-adjusted
   ```

1. **Use in backtests:**

   - Your existing backtest scripts will automatically use the cache
   - No code changes needed!

## Security Notes

- **Never commit credentials** to git
- Token file is `chmod 600` (only you can read)
- SSL certificates are self-signed (local use only)
- Tokens auto-refresh (no password stored)

## Related Documentation

- [Schwab Smart Caching Implementation](../docs/SCHWAB_SMART_CACHING_IMPLEMENTATION.md)
- [Data CLI User Guide](../docs/DATA_CLI_USER_GUIDE.md)
- [Generic Data Update Completion](../docs/GENERIC_DATA_UPDATE_COMPLETION.md)
