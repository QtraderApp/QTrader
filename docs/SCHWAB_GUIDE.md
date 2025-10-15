# Schwab Data Source Integration Guide

Complete guide to using the Schwab API as a data source in QTrader.

## 📋 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Usage](#usage)
- [Caching](#caching)
- [Troubleshooting](#troubleshooting)
- [API Limits](#api-limits)
- [Examples](#examples)

______________________________________________________________________

## Overview

The Schwab integration provides access to split-adjusted historical price data from Schwab's Market Data API. Key features:

- ✅ **OAuth 2.0 Authentication** - Secure token management with auto-refresh
- ✅ **Cache-First Architecture** - Fast repeated access with intelligent caching
- ✅ **Rate Limiting** - Built-in 10 requests/second limit
- ✅ **Split-Adjusted Data** - Only adjusted prices (no unadjusted or total return modes)
- ✅ **Auto-Retry** - Exponential backoff for transient failures

### Data Characteristics

| Feature              | Details                                  |
| -------------------- | ---------------------------------------- |
| **Asset Classes**    | US Equities only                         |
| **Adjustment Mode**  | Split-adjusted only                      |
| **Frequencies**      | Daily (default), Minute, Weekly, Monthly |
| **Historical Depth** | Up to 20 years                           |
| **Rate Limit**       | 10 requests/second                       |
| **Caching**          | Parquet files with metadata              |

______________________________________________________________________

## Prerequisites

### 1. Schwab Developer Account

1. Visit <https://developer.schwab.com>
1. Sign up for a developer account
1. Create a new app in the Developer Portal
1. Note your:
   - **App Key** (Client ID)
   - **App Secret** (Client Secret)
   - **Redirect URI** (use `https://127.0.0.1:8182` for local development)

### 2. System Requirements

- Python 3.11+
- Internet connection (for OAuth and API calls)
- Browser (for initial OAuth authorization)

______________________________________________________________________

## Setup

### Step 1: Configure Environment Variables

Copy the example environment file:

```bash
cp .envrc.example .envrc
```

Edit `.envrc` with your credentials:

```bash
# Schwab API Credentials
export SCHWAB_API_KEY="your_app_key_here"
export SCHWAB_API_SECRET="your_app_secret_here"
export SCHWAB_REDIRECT_URI="https://127.0.0.1:8182"
```

### Step 2: Load Environment Variables

Using `direnv` (recommended):

```bash
direnv allow
```

Or manually:

```bash
source .envrc
```

### Step 3: Verify Configuration

The Schwab data source is pre-configured in `config/data_sources.yaml`:

```yaml
schwab:
  adapter: schwabOHLC
  cache_root: "data/us-equity-daily-adjusted-schwab"
  mode: adjusted_only

  api:
    base_url: "https://api.schwabapi.com"
    auth:
      client_id: "${SCHWAB_API_KEY}"
      client_secret: "${SCHWAB_API_SECRET}"
      redirect_uri: "${SCHWAB_REDIRECT_URI:-https://127.0.0.1:8182}"
      token_cache: "~/.qtrader/schwab_tokens.json"

    rate_limit:
      requests_per_second: 10
      backoff_base_delay: 1.0
      backoff_max_delay: 60.0

    fetch:
      chunk_size_days: 365
      max_years: 20
      timeout_seconds: 30
```

______________________________________________________________________

## Usage

### CLI: View Raw Data

```bash
# View Apple stock data from January 2024
qtrader raw-data --symbol AAPL --start-date 2024-01-01 --end-date 2024-01-31 --source schwab
```

First run output:

```
Loading data for AAPL from schwab...
[OAuth flow initiated - browser will open]
Authorization successful!
Token cached in ~/.qtrader/schwab_tokens.json
Reading bars from 2024-01-01 to 2024-01-31...
Loaded 21 bars

Press ENTER to view next bar, CTRL+C to exit

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Field        ┃ Value        ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ Timestamp    │ 2024-01-02   │
│ Open         │ $184.92      │
│ High         │ $186.95      │
│ Low          │ $183.82      │
│ Close        │ $185.64      │
│ Volume       │ 82,488,200   │
└──────────────┴──────────────┘
```

Subsequent runs (cached):

```
Loading data for AAPL from schwab...
Reading bars from 2024-01-01 to 2024-01-31...
Loaded 21 bars (from cache)
```

### Python API

```python
from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import DataSource, Instrument, InstrumentType

# Create instrument
instrument = Instrument(
    symbol="AAPL",
    instrument_type=InstrumentType.EQUITY,
    data_source=DataSource.SCHWAB
)

# Resolve to adapter
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)

# Fetch data
bars = list(adapter.read_bars("2024-01-01", "2024-12-31"))

# Use bars
for bar in bars:
    print(f"{bar.timestamp}: ${bar.close:.2f}")
```

See `examples/schwab_data_example.py` for more detailed examples.

______________________________________________________________________

## Caching

### Cache Structure

Cached data is stored in Parquet format with metadata:

```
data/us-equity-daily-adjusted-schwab/
├── AAPL/
│   ├── data.parquet           # Price data
│   └── .metadata.json          # Cache metadata
├── MSFT/
│   ├── data.parquet
│   └── .metadata.json
└── ...
```

### Metadata Format

`.metadata.json` contains:

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

### Cache Behavior

1. **Cache Hit** (data exists for requested range)

   - Data loaded from disk (instant)
   - No API call made
   - No OAuth required

1. **Cache Miss** (data doesn't exist or out of date)

   - OAuth token validated/refreshed
   - API called to fetch missing data
   - New data appended to cache
   - Metadata updated

1. **Cache Management**

   - Manual clearing: `rm -rf data/us-equity-daily-adjusted-schwab/SYMBOL`
   - Automatic invalidation: None (30-day freshness recommended)
   - Size: ~100KB per symbol per year

______________________________________________________________________

## Troubleshooting

### OAuth Issues

**Symptom**: Browser doesn't open or authorization fails

**Solutions**:

1. Check redirect URI matches your app configuration

1. Verify API credentials are correct

1. Try manual mode:

   ```python
   # In your code
   os.environ["SCHWAB_OAUTH_MANUAL"] = "true"
   ```

1. Delete stale tokens:

   ```bash
   rm ~/.qtrader/schwab_tokens.json
   ```

### Rate Limit Errors

**Symptom**: `429 Too Many Requests` errors

**Solutions**:

1. Wait 60 seconds for rate limit reset
1. Reduce concurrent requests
1. Use cached data when possible
1. Check for retry logic in logs

### Missing Data

**Symptom**: Empty response or partial data

**Solutions**:

1. Verify symbol is valid (US equities only)
1. Check date range (max 20 years historical)
1. Confirm market was open on requested dates
1. Review logs for API errors

### Configuration Errors

**Symptom**: `KeyError` for environment variables

**Solutions**:

1. Verify `.envrc` file exists and is sourced

1. Check environment variables:

   ```bash
   echo $SCHWAB_API_KEY
   echo $SCHWAB_API_SECRET
   ```

1. Re-load environment:

   ```bash
   direnv allow  # or source .envrc
   ```

______________________________________________________________________

## API Limits

### Rate Limits

| Limit Type           | Value    | Enforcement           |
| -------------------- | -------- | --------------------- |
| **Requests/Second**  | 10       | Built-in rate limiter |
| **Daily Requests**   | ~120,000 | Schwab API limit      |
| **Historical Depth** | 20 years | API restriction       |

### Quota Management

- Rate limiter uses token bucket algorithm
- Automatic exponential backoff on server errors
- No retry on client errors (4xx)
- Logs all rate limit events

### Best Practices

1. **Cache First**: Always use cached data when available
1. **Batch Wisely**: Fetch yearly chunks to minimize API calls
1. **Off-Peak**: Run large fetches during off-market hours
1. **Monitor Logs**: Watch for rate limit warnings

______________________________________________________________________

## Examples

### Example 1: Basic Data Fetch

```python
from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import DataSource, Instrument, InstrumentType

# Setup
instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.SCHWAB)
resolver = DataSourceResolver()
adapter = resolver.resolve(instrument)

# Fetch
bars = list(adapter.read_bars("2024-01-01", "2024-03-31"))
print(f"Loaded {len(bars)} bars")
```

### Example 2: Intraday Data

```python
# 5-minute bars
bars = list(adapter.read_bars(
    "2024-10-01",
    "2024-10-15",
    frequency_type="minute",
    frequency=5
))
```

### Example 3: Cache Performance Test

```python
import time

# First call (may hit API)
start = time.time()
bars1 = list(adapter.read_bars("2024-01-01", "2024-12-31"))
duration1 = time.time() - start
print(f"First call: {duration1:.2f}s")

# Second call (from cache)
start = time.time()
bars2 = list(adapter.read_bars("2024-01-01", "2024-12-31"))
duration2 = time.time() - start
print(f"Cached call: {duration2:.2f}s")
print(f"Speedup: {duration1/duration2:.1f}x")
```

### Example 4: Error Handling

```python
try:
    bars = list(adapter.read_bars("2024-01-01", "2024-12-31"))
except FileNotFoundError:
    print("Config file not found - check setup")
except KeyError as e:
    print(f"Missing environment variable: {e}")
except requests.HTTPError as e:
    if e.response.status_code == 429:
        print("Rate limit exceeded - wait 60 seconds")
    elif e.response.status_code == 401:
        print("OAuth failed - delete token cache and retry")
    else:
        print(f"API error: {e}")
```

______________________________________________________________________

## Additional Resources

- **Schwab API Docs**: <https://developer.schwab.com/products/trader-api--individual>
- **QTrader Examples**: `examples/schwab_data_example.py`
- **Integration Plan**: `docs/schwab_integration.md`
- **Phase Completion**: `docs/schwab_phase*_complete.md`

______________________________________________________________________

## Support

For issues or questions:

1. Check logs: Structured logging available via `LoggerFactory`
1. Review troubleshooting section above
1. Check Schwab API status: <https://developer.schwab.com/status>
1. Open an issue on GitHub

______________________________________________________________________

**Last Updated**: October 2025\
**API Version**: Schwab Trader API v1\
**QTrader Version**: 0.1.0
