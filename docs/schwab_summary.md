# Schwab Integration - Quick Summary

**Status:** ✅ Planning Complete - Ready for Implementation **Document:** [Full Details](./schwab_integration.md) **Estimated Time:** 15-21 hours (2-3 days)

______________________________________________________________________

## ✅ Your Approved Decisions

| #   | Question              | Decision                                            |
| --- | --------------------- | --------------------------------------------------- |
| 1   | MultiBar strategy?    | **Option 1** - Partial (None for unavailable modes) |
| 2   | OAuth callback?       | **HTTPS required** - `https://127.0.0.1:8182`       |
| 3   | Token storage?        | Plain JSON in `~/.qtrader/schwab_tokens.json`       |
| 4   | Rate limiting?        | **10 requests/second** with exponential backoff     |
| 5   | Validate vs Algoseek? | **No** - skip cross-validation                      |
| 6   | Cache strategy?       | Hard fail if API needed, show available dates       |
| 7   | Symbol mapping?       | Later (not now)                                     |
| 8   | Metadata format?      | **JSON** with best practices                        |
| 9   | Update strategy?      | On-demand + `--update-all` flag                     |

______________________________________________________________________

## 🏗️ What Gets Built

### New Files

```
src/qtrader/auth/
├── __init__.py
├── schwab_oauth.py         # OAuth with HTTPS callback
└── ssl_generator.py        # Self-signed cert generator

src/qtrader/adapters/
└── schwab.py              # Cache-first API adapter

src/qtrader/models/vendors/
└── schwab.py              # SchwabBar + SchwabPriceSeries

data/us-equity-daily-adjusted-schwab/
└── SYMBOL/
    ├── data.parquet       # Cached data
    └── .metadata.json     # Cache metadata

.qtrader/
├── ssl/
│   └── localhost.pem      # SSL certificate
└── schwab_tokens.json     # OAuth tokens
```

### Updated Files

```
config/data_sources.yaml   # Add schwab config
config/.envrc.example      # Add SCHWAB_* env vars
src/qtrader/cli.py         # Handle None modes gracefully
src/qtrader/adapters/resolver.py  # Register schwab adapter
```

______________________________________________________________________

## 🎯 Key Features

### 1. HTTPS OAuth (Required by Schwab)

- Self-signed SSL certificate auto-generated
- Browser security warning expected (normal for localhost)
- Token cached and auto-refreshed
- Secure permissions (chmod 600)

### 2. Intelligent Caching

```python
# Cache-first strategy
if all_dates_in_cache:
    return cached_data  # Fast path
elif some_dates_in_cache:
    fetch_missing_from_api()
    merge_and_cache()
else:
    fetch_all_from_api()
    cache()
```

### 3. Rate Limiting (10/sec)

- Token bucket algorithm
- Exponential backoff on errors
- Respects Schwab API limits

### 4. Partial MultiBar

```python
MultiBar(
    symbol="AAPL",
    trade_datetime="2023-01-15T00:00:00",
    unadjusted=None,      # Schwab doesn't provide
    adjusted=schwab_bar,   # ✓ Real data
    total_return=None      # Schwab doesn't provide
)
```

### 5. Smart Error Handling

```
❌ Error: API required but unavailable

Requested: 2019-01-01 to 2025-10-15
Available in cache: 2019-01-01 to 2023-12-31
Missing: 2024-01-01 to 2025-10-15 (285 days)

Solutions:
1. Use cached data: --end-date 2023-12-31
2. Authenticate with Schwab API to fetch missing data
3. Run: qtrader schwab-auth
```

______________________________________________________________________

## 📊 Usage Examples

### First Time (OAuth Required)

```bash
# Will trigger OAuth flow
qtrader raw-data --symbol AAPL --start-date 2019-01-01 --end-date 2023-12-31 --source schwab

# Output:
# 🔐 OAuth authentication required...
# Please visit: https://api.schwabapi.com/...
# ⏳ Waiting for callback...
# ✓ Token cached (valid for 30 minutes)
# ⏳ Fetching 1,258 bars from Schwab API...
# ✓ Cached for future use
```

### Subsequent Calls (Uses Cache)

```bash
# No API call - uses cache
qtrader raw-data --symbol AAPL --start-date 2019-01-01 --end-date 2023-12-31 --source schwab

# Output:
# ✓ Using cached data (1,258 bars)
# ⚠ Schwab provides adjusted data only
#   - Unadjusted mode: Not available
#   - Total return mode: Not available
```

### Update All Symbols

```bash
# Update all cached symbols
qtrader schwab-update --all

# Update specific symbol
qtrader schwab-update --symbol AAPL
```

______________________________________________________________________

## 🔧 Environment Setup

```bash
# .envrc
export SCHWAB_API_KEY="your_api_key_here"
export SCHWAB_API_SECRET="your_api_secret_here"
export SCHWAB_REDIRECT_URI="https://127.0.0.1:8182"
```

______________________________________________________________________

## 📝 Implementation Phases

| Phase | Task          | Time | Status            |
| ----- | ------------- | ---- | ----------------- |
| 1     | OAuth + SSL   | 2-3h | ⏳ Ready to start |
| 2     | Vendor Models | 2-3h | Pending           |
| 3     | Adapter Core  | 4-5h | Pending           |
| 4     | Caching Logic | 3-4h | Pending           |
| 5     | Integration   | 2-3h | Pending           |
| 6     | Polish        | 2-3h | Pending           |

**Total:** 15-21 hours

______________________________________________________________________

## 🚀 Next Steps

Ready to begin implementation!

**Option A: Start Now**

```bash
# Create feature branch
git checkout -b feature/schwab-integration

# Install dependencies
uv add cryptography  # For SSL cert generation

# Start Phase 1
# Create src/qtrader/auth/ directory and implement OAuth
```

**Option B: Review First**

Ask questions or request changes to the plan before starting.

______________________________________________________________________

## 📚 Reference

- **Full Plan:** [schwab_integration.md](./schwab_integration.md)
- **Demo Code:** `src/qtrader/playground/schwap_demo.py`
- **Algoseek Reference:** `src/qtrader/adapters/algoseek.py`

______________________________________________________________________

**Decision Required:** Should I proceed with Phase 1 (OAuth + SSL implementation)?
