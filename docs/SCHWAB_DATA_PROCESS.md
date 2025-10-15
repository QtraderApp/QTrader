# Schwab Data Process - Complete Guide

**Last Updated:** October 15, 2025 **Status:** ✅ Working (OAuth flow functional)

## Overview

QTrader integrates with Schwab's Market Data API to fetch historical OHLC (Open, High, Low, Close) bars for US equities. The process uses OAuth 2.0 for authentication and implements a cache-first strategy to minimize API calls.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User / Strategy                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              DataSourceResolver                              │
│  - Reads config/data_sources.yaml                           │
│  - Substitutes environment variables                         │
│  - Instantiates SchwabOHLCAdapter                           │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│             SchwabOHLCAdapter                                │
│  - OAuth authentication (via SchwabOAuthManager)            │
│  - API calls to /marketdata/v1/pricehistory                 │
│  - Rate limiting (10 req/sec)                               │
│  - Cache management (MetadataManager)                       │
│  - Returns Iterator[SchwabBar]                              │
└───────────────────────────┬─────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
    ┌─────────┐      ┌──────────┐     ┌──────────┐
    │  Cache  │      │ OAuth    │     │ Schwab   │
    │  (Disk) │      │ Manager  │     │   API    │
    └─────────┘      └──────────┘     └──────────┘
```

## Prerequisites

### 1. Schwab Developer Account

1. Go to <https://developer.schwab.com>
1. Create developer account
1. Create a new app:
   - **App Name:** QTrader (or your choice)
   - **Redirect URI:** `https://127.0.0.1:8182`
   - **Required APIs:** Market Data
1. Copy your **API Key** and **API Secret**

### 2. Environment Variables

Create or edit `.envrc` in your project root:

```bash
# Schwab API Credentials
export SCHWAB_API_KEY="your_api_key_here"
export SCHWAB_API_SECRET="your_api_secret_here"

# Optional: Override default redirect URI
# export SCHWAB_REDIRECT_URI="https://127.0.0.1:8182"
```

Then activate:

```bash
direnv allow  # If using direnv
# OR
source .envrc  # If using bash/zsh manually
```

### 3. Configuration File

Verify `config/data_sources.yaml` has the Schwab section:

```yaml
data_sources:
  schwab:
    adapter: schwabOHLC
    cache_root: "data/us-equity-daily-adjusted-schwab"
    mode: adjusted_only

    # OAuth credentials (required)
    client_id: "${SCHWAB_API_KEY}"
    client_secret: "${SCHWAB_API_SECRET}"

    # OAuth configuration (optional)
    redirect_uri: "${SCHWAB_REDIRECT_URI:-https://127.0.0.1:8182}"
    token_cache_path: "~/.qtrader/schwab_tokens.json"
    manual_mode: false

    # Rate limiting
    requests_per_second: 10
```

## OAuth Flow (First Time Only)

### Step 1: Script Initiates OAuth

When you first run a script that uses Schwab data:

```python
from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import DataSource, Instrument, InstrumentType

instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)
bars = list(adapter.read_bars("2024-01-01", "2024-01-31"))
```

### Step 2: Browser Opens Automatically

QTrader will:

1. ✅ Generate SSL certificates (stored in `~/.qtrader/ssl/`)
1. ✅ Start local HTTPS server on port 8182
1. ✅ Print authorization URL
1. ✅ Attempt to open browser automatically

You'll see output like:

```
======================================================================
SCHWAB OAUTH AUTHENTICATION REQUIRED
======================================================================

📋 Authorization URL:

https://api.schwabapi.com/v1/oauth/authorize?client_id=...

🔐 Steps:
  1. Click the URL above (or copy to browser)
  2. Log in to your Schwab account
  3. Authorize the application
  4. You'll be redirected to a local page (https://127.0.0.1:8182)

⚠️  Browser Security Warning:
  - You will see a security warning (expected)
  - This is because we use a self-signed certificate
  - Click 'Advanced' → 'Proceed to 127.0.0.1'

⏳ Waiting for authorization...
======================================================================
```

### Step 3: Authorize in Browser

1. **Log in** to your Schwab account
1. **Review permissions** (read-only market data)
1. **Click "Allow"**
1. **You'll be redirected** to `https://127.0.0.1:8182?code=...`

### Step 4: Handle SSL Warning

Because we use a self-signed certificate (required for localhost HTTPS):

- **Chrome/Edge:** Click "Advanced" → "Proceed to 127.0.0.1 (unsafe)"
- **Firefox:** Click "Advanced" → "Accept the Risk and Continue"
- **Safari:** Click "Show Details" → "Visit this website"

This is **safe** - it's your own local server.

### Step 5: Tokens Cached

Once authorized:

1. ✅ Authorization code captured
1. ✅ Access token + refresh token obtained
1. ✅ Tokens saved to `~/.qtrader/schwab_tokens.json`
1. ✅ Data fetch proceeds

**You won't need to authorize again!** Tokens are cached and auto-refreshed.

## Data Fetching Process

### Cache-First Strategy

```
┌─────────────────────────────────────────────────────────────┐
│  1. Request bars for AAPL 2024-01-01 to 2024-01-31         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │ Check cache    │
                   │ metadata       │
                   └────┬───────┬───┘
                        │       │
            Cache Hit   │       │   Cache Miss
              ┌─────────┘       └─────────┐
              ▼                           ▼
    ┌─────────────────┐       ┌──────────────────┐
    │ Load from disk  │       │ Fetch from API   │
    │ (instant)       │       │ (rate limited)   │
    └─────┬───────────┘       └────────┬─────────┘
          │                            │
          │                            ▼
          │                  ┌─────────────────┐
          │                  │ Write to cache  │
          │                  └────────┬────────┘
          │                           │
          └───────────┬───────────────┘
                      │
                      ▼
            ┌──────────────────┐
            │ Return bars      │
            └──────────────────┘
```

### Cache Structure

```
data/us-equity-daily-adjusted-schwab/
├── AAPL/
│   ├── data.parquet          # Cached bars
│   └── .metadata.json        # Cache metadata
├── TSLA/
│   ├── data.parquet
│   └── .metadata.json
└── ...
```

### Metadata Format

```json
{
  "symbol": "AAPL",
  "last_update": "2025-10-15T10:30:00Z",
  "date_range": {
    "start": "2019-01-01",
    "end": "2025-10-15"
  },
  "row_count": 1658,
  "frequency_type": "daily",
  "frequency": 1,
  "source": "schwab"
}
```

### Cache Invalidation

Cache is used when:

- ✅ Requested date range is subset of cached range
- ✅ Metadata file exists and is valid
- ✅ Data file exists and is readable

Cache is bypassed when:

- ❌ Requested date range extends beyond cached range
- ❌ Metadata file missing or corrupted
- ❌ Data file missing

## Rate Limiting

Schwab API limits:

- **10 requests per second** (enforced by adapter)
- **Token bucket algorithm** prevents bursts
- **Automatic backoff** on 429 errors

The adapter uses `RateLimiter` class:

```python
self.rate_limiter = RateLimiter(requests_per_second=10.0)

# Before each API call:
self.rate_limiter.wait()  # Blocks if needed
response = self.session.get(url, headers=headers)
```

## Data Characteristics

### What Schwab Returns

- **Frequency:** Daily bars (1 day)
- **Adjustments:** Split-adjusted ONLY
- **Dividends:** NOT adjusted (price doesn't drop on ex-div date)
- **Fields:** open, high, low, close, volume, datetime

### What QTrader Does

The adapter returns **raw Schwab data** as `SchwabBar` objects:

- ✅ Validates data format
- ✅ Converts timestamps to UTC
- ✅ Returns chronological iterator
- ❌ No additional adjustments
- ❌ No dividend handling (done in DataLoader)

Transformation to canonical format happens in `DataLoader`, not the adapter.

## Error Handling

### Common Errors

#### 1. Missing Environment Variables

```
KeyError: 'SCHWAB_API_KEY'
```

**Solution:**

```bash
# Set in .envrc
export SCHWAB_API_KEY="your_key"
export SCHWAB_API_SECRET="your_secret"

# Activate
direnv allow  # or source .envrc
```

#### 2. OAuth Timeout

```
RuntimeError: Failed to capture authorization code from callback
```

**Causes:**

- Took longer than 2 minutes to authorize
- Browser didn't open automatically
- Clicked "Deny" instead of "Allow"
- Network connectivity issues

**Solution:**

- Run script again
- Manually copy authorization URL from terminal
- Complete OAuth flow within 2 minutes
- Check firewall settings (port 8182)

#### 3. Invalid API Credentials

```
401 Unauthorized
```

**Solution:**

- Verify API Key and Secret in Schwab Developer Portal
- Check for typos in `.envrc`
- Ensure API is enabled for "Market Data"

#### 4. Rate Limit Exceeded

```
429 Too Many Requests
```

**Solution:**

- Wait 60 seconds
- Reduce `requests_per_second` in config
- Check for other scripts using Schwab API

#### 5. Symbol Not Found

```
404 Not Found
```

**Cause:** Symbol doesn't exist or is delisted

**Solution:**

- Verify symbol on Schwab website
- Check if symbol was renamed (e.g., FB → META)

## Testing

### Unit Tests

```bash
# Test adapter initialization
pytest tests/unit/adapters/test_schwab.py::TestSchwabOHLCAdapterInitialization -v

# Test API calls (mocked)
pytest tests/unit/adapters/test_schwab.py::TestSchwabOHLCAdapterAPICall -v

# Test caching logic
pytest tests/unit/adapters/test_schwab_caching.py -v
```

### Integration Test (Requires Real Credentials)

```bash
# Set environment variables first
export SCHWAB_API_KEY="your_key"
export SCHWAB_API_SECRET="your_secret"

# Run example
python examples/schwab_data_example.py
```

### Expected Output (First Run)

```
======================================================================
QTrader - Schwab Data Source Example
======================================================================

Step 1: Creating Instrument for AAPL from Schwab
----------------------------------------------------------------------
✓ Instrument created: AAPL (equity)
  Data Source: schwab

Step 2: Resolving Data Source
----------------------------------------------------------------------
✓ Adapter resolved: SchwabOHLCAdapter
  Configuration loaded from: config/data_sources.yaml

Step 3: Fetching Data (2024-01-01 to 2024-01-31)
----------------------------------------------------------------------
[OAuth flow happens here - browser opens]
✓ Successfully loaded 21 bars

Step 4: Sample Data (First 5 bars)
----------------------------------------------------------------------
Bar 1:
  Date:   2024-01-02
  Open:   $187.15
  High:   $189.58
  Low:    $186.40
  Close:  $189.25
  Volume: 82,488,900
...
```

### Expected Output (Subsequent Runs)

```
Step 3: Fetching Data (2024-01-01 to 2024-01-31)
----------------------------------------------------------------------
[No OAuth flow - loads from cache instantly]
✓ Successfully loaded 21 bars
```

## Performance

### First Run (Cache Miss)

- OAuth flow: ~30-60 seconds (one-time)
- API call: ~200-500ms per request
- Cache write: ~50ms
- **Total:** ~1 second for 1 year of data

### Subsequent Runs (Cache Hit)

- Cache read: ~10-50ms
- **Total:** ~50ms for 1 year of data
- **Speedup:** 20-100x faster

### Benchmark (1 Year of Data)

```python
import time
from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import Instrument, InstrumentType, DataSource

instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)

# First call (API)
start = time.time()
bars1 = list(adapter.read_bars("2024-01-01", "2024-12-31"))
print(f"First call: {time.time() - start:.2f}s")

# Second call (cache)
start = time.time()
bars2 = list(adapter.read_bars("2024-01-01", "2024-12-31"))
print(f"Second call: {time.time() - start:.2f}s")
```

**Output:**

```
First call: 0.85s   # API + cache write
Second call: 0.04s  # Cache read only
Speedup: 21x
```

## Troubleshooting Checklist

- [ ] Environment variables set (`SCHWAB_API_KEY`, `SCHWAB_API_SECRET`)
- [ ] Variables activated (`direnv allow` or `source .envrc`)
- [ ] Configuration file exists (`config/data_sources.yaml`)
- [ ] API credentials valid (check Schwab Developer Portal)
- [ ] OAuth completed successfully (tokens in `~/.qtrader/schwab_tokens.json`)
- [ ] Port 8182 not blocked by firewall
- [ ] Internet connection active
- [ ] Symbol exists and is valid

## Security Notes

### Token Storage

Tokens stored in `~/.qtrader/schwab_tokens.json`:

```json
{
  "access_token": "eyJ0eXAi...",
  "refresh_token": "4y8ZQ...",
  "expires_at": 1729025400.0,
  "token_type": "Bearer"
}
```

**Security:**

- ✅ File permissions: `0600` (owner read/write only)
- ✅ Stored outside project directory
- ✅ Tokens expire and auto-refresh
- ✅ No passwords stored (OAuth flow only)
- ❌ Not encrypted (consider encrypting sensitive files)

### SSL Certificates

Self-signed certificates in `~/.qtrader/ssl/`:

- `localhost.pem` - Certificate
- `localhost-key.pem` - Private key

**Security:**

- ✅ Generated locally (not from CA)
- ✅ Valid for localhost only
- ✅ 1-year expiration
- ⚠️ Browser warnings expected (safe to ignore for localhost)

## Next Steps

1. ✅ **Run example:** `python examples/schwab_data_example.py`
1. ✅ **Check cache:** View files in `data/us-equity-daily-adjusted-schwab/`
1. ✅ **Verify OAuth:** Check `~/.qtrader/schwab_tokens.json`
1. ✅ **Test in strategy:** Use `DataSource.SCHWAB` in backtests
1. ✅ **Monitor rate limits:** Check logs for `rate_limit` events

## Related Documentation

- [Schwab API Documentation](https://developer.schwab.com/market-data)
- [OAuth 2.0 Specification](https://oauth.net/2/)
- [QTrader Architecture](docs/architecture.md)
- [Data Layer Design](docs/DATA_LAYER_MIGRATION_SUMMARY.md)

## Support

For issues with Schwab integration:

1. Check this document first
1. Review error messages and logs
1. Search existing GitHub issues
1. Create new issue with:
   - Error message
   - Redacted logs
   - Steps to reproduce
   - Environment details (Python version, OS)

______________________________________________________________________

**Status:** ✅ Working **Last Tested:** October 15, 2025 **Python Version:** 3.13.3 **Schwab API Version:** v1
