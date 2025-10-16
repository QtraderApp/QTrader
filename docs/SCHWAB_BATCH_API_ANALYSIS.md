# Schwab API Batch Request Analysis

## Executive Summary

**Finding**: Schwab has **two different APIs** with different batching capabilities:

1. **REST API (Price History)**: ❌ **One symbol per request**
1. **Streaming API (WebSocket)**: ✅ **Multiple symbols per request**

## REST API - Price History (Current Implementation)

### Endpoint

```
GET /marketdata/v1/pricehistory
```

### Parameters

- `symbol`: **Single symbol** (e.g., "AAPL")
- `periodType`: day, month, year, ytd
- `period`: Number of periods
- `frequencyType`: minute, daily, weekly, monthly
- `frequency`: Frequency value

### Batching Capability

❌ **NO BATCH SUPPORT** - One symbol per API call

### Current Usage

```python
# File: src/qtrader/adapters/schwab.py
PRICE_HISTORY_ENDPOINT = "/marketdata/v1/pricehistory"

params = {
    "symbol": symbol,  # Single symbol only
    "periodType": period_type,
    "period": period,
    # ...
}
response = self._call_api(self.PRICE_HISTORY_ENDPOINT, params)
```

### Rate Limits

- **10 requests/second** per API key
- **Implication**: Fetching 500 symbols = 50 seconds minimum

## Streaming API - WebSocket (Not Currently Used)

### Service Types

- `LEVELONE_EQUITIES`: Level 1 quotes
- `CHART_EQUITY`: Chart candles (OHLC)
- `CHART_FUTURES`: Futures candles

### Batching Capability

✅ **FULL BATCH SUPPORT** - Comma-separated symbols

### Request Format

```json
{
  "service": "CHART_EQUITY",
  "command": "SUBS",
  "parameters": {
    "keys": "AAPL,TSLA,IBM,MSFT,GOOGL",  // Multiple symbols!
    "fields": "0,1,2,3,4,5"
  }
}
```

### Response Format

```json
{
  "data": [{
    "service": "CHART_EQUITY",
    "content": [
      {"key": "AAPL", "1": 150.25, "2": 151.00, ...},
      {"key": "TSLA", "1": 250.50, "2": 252.00, ...},
      {"key": "IBM", "1": 140.75, "2": 141.25, ...}
    ]
  }]
}
```

### Field Definitions (CHART_EQUITY)

| Field | Name        | Type   | Description                  |
| ----- | ----------- | ------ | ---------------------------- |
| 0     | key         | String | Symbol                       |
| 1     | Open Price  | double | Opening price for the minute |
| 2     | High Price  | double | Highest price for the minute |
| 3     | Low Price   | double | Lowest price for the minute  |
| 4     | Close Price | double | Closing price for the minute |
| 5     | Volume      | double | Total volume for the minute  |
| 6     | Sequence    | long   | Identifies the candle minute |
| 7     | Chart Time  | long   | Milliseconds since Epoch     |
| 8     | Chart Day   | int    | Day identifier               |

## Optimization Strategies

### Strategy 1: Parallel REST API Calls (Current Best Option)

**Approach**: Use Python's `asyncio` or `concurrent.futures` to parallelize REST API calls

**Pros**:

- Works with existing REST API
- No need to implement WebSocket protocol
- Simple to implement

**Cons**:

- Still limited by rate limit (10 req/sec)
- More API calls = higher costs

**Implementation**:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def fetch_multiple_symbols(symbols: List[str], start_date, end_date):
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(self._fetch_from_api, symbol, start_date, end_date)
            for symbol in symbols
        ]
        results = [f.result() for f in futures]
    return results
```

**Performance**:

- Current: 500 symbols × 1 req = 50 seconds (sequential)
- Optimized: 500 symbols ÷ 10 parallel = 5 seconds (10× faster)

### Strategy 2: WebSocket Streaming API (Future Enhancement)

**Approach**: Implement WebSocket client for historical data retrieval

**Pros**:

- True batch support (multiple symbols per request)
- More efficient for bulk operations
- Real-time streaming capability

**Cons**:

- Requires WebSocket implementation
- More complex authentication flow
- May not support historical date ranges (streaming focus)

**Implementation Complexity**: High

- OAuth flow integration
- WebSocket connection management
- Message parsing and assembly
- Reconnection logic

**Note**: The streaming API appears designed for **real-time data**, not historical bulk downloads. May not support `startDate` and `endDate` parameters.

### Strategy 3: Hybrid Approach (Recommended Long-term)

**Phase 1** (Immediate):

- Implement parallel REST API calls for historical data
- Use smart caching to minimize API calls
- Optimize with initial backfill strategy

**Phase 2** (Future):

- Implement WebSocket client for real-time updates
- Use streaming API for incremental daily updates
- Keep REST API for historical backfills

## Recommendations

### Immediate Actions (This Sprint)

1. **Parallel REST API Implementation**

   - Add async support to `SchwabOHLCAdapter`
   - Implement batch fetch method: `fetch_multiple_symbols()`
   - Respect rate limits with proper throttling
   - Add to CLI: `qtrader data batch-fetch --symbols AAPL,TSLA,MSFT`

1. **Smart Caching Optimization**

   - Implement initial backfill strategy (already planned)
   - Add incremental update from last bar
   - This will minimize API calls regardless of batching

1. **CLI Reorganization**

   - Move data commands under `qtrader data` group
   - Add batch operations support

### Future Enhancements (Next Quarter)

1. **WebSocket Implementation**

   - Research Schwab WebSocket historical data capabilities
   - Implement if historical date range queries supported
   - Otherwise, use for real-time updates only

1. **Intelligent Dispatcher**

   - Use REST API for historical backfills
   - Use WebSocket for daily incremental updates
   - Automatically choose optimal strategy

## Performance Comparison

### Scenario: Daily Update of 500 Symbols (1 new bar each)

| Strategy                       | API Calls | Time    | Bandwidth |
| ------------------------------ | --------- | ------- | --------- |
| Current (Sequential)           | 500       | 50 sec  | 500 KB    |
| Parallel REST (10 workers)     | 500       | 5 sec   | 500 KB    |
| WebSocket Batch (if supported) | 1-10      | \<1 sec | 50 KB     |
| Smart Cache + Parallel         | 10-50\*   | \<1 sec | 50 KB     |

\*Only uncached symbols need API calls

### Scenario: Initial Backfill of 500 Symbols (20 years each)

| Strategy                   | API Calls | Time   | Bandwidth |
| -------------------------- | --------- | ------ | --------- |
| Current (Sequential)       | 500       | 50 sec | 126 MB    |
| Parallel REST (10 workers) | 500       | 5 sec  | 126 MB    |
| WebSocket Batch            | N/A\*     | N/A    | N/A       |
| Smart Cache (one-time)     | 500       | 5 sec  | 126 MB    |

\*WebSocket may not support historical ranges

## Conclusion

1. **Schwab REST API does NOT support batch requests** - one symbol per call
1. **Schwab Streaming API DOES support multiple symbols** - but designed for real-time, not historical
1. **Best immediate solution**: Parallel REST API calls (10× speedup)
1. **Best long-term solution**: Smart caching + incremental updates reduces API calls by 99%+

The smart caching strategy we're implementing is **more important** than batch API support, as it reduces 500 daily API calls down to 0-50 (only new/missing symbols).

## Related Files

- Current implementation: `src/qtrader/adapters/schwab.py`
- API documentation: `docs/external/schawb_api.md`
- Smart caching plan: `docs/SCHWAB_SMART_CACHING_IMPLEMENTATION.md`
- CLI structure: `src/qtrader/cli.py`

## Next Steps

1. ✅ Document batch API capabilities (this document)
1. 🔄 Reorganize CLI to `qtrader data ...` structure
1. 🔄 Implement parallel REST API fetching
1. 🔄 Continue smart caching implementation
1. 📋 Research WebSocket historical data support
