# QTrader TODO

## 🔴 CRITICAL: Memory Optimization in DataService.stream_universe

**Priority:** HIGH (Performance/OOM Risk)\
**File:** `src/qtrader/services/data/service.py:570`\
**Status:** 🔍 TODO

### Problem

`stream_universe()` currently buffers ALL bars for ALL symbols into memory before publishing:

```python
# Current implementation (lines 571-583)
timestamp_bars: Dict[datetime, Dict[str, Any]] = {}

for symbol, iterator in active_iterators.items():
    for multi_bar in iterator:
        ts = multi_bar.adjusted.trade_datetime
        if ts not in timestamp_bars:
            timestamp_bars[ts] = {}
        timestamp_bars[ts][symbol] = multi_bar
```

**Memory Impact:**

- 100 symbols × 252 days × ~500 bytes/bar ≈ **12.6 MB**
- 1000 symbols × 252 days × ~500 bytes/bar ≈ **126 MB**
- 1000 symbols × 5 years (1260 days) ≈ **630 MB** ⚠️
- Large universes or long date ranges can cause **OOM crashes**

### Objective

Refactor to **stream incrementally** using heap-merge pattern:

1. Hold only N bars in memory (where N = number of symbols)
1. Process bars by timestamp without buffering entire dataset
1. Maintain current event ordering guarantee (all bars at timestamp T before T+1)
1. Preserve deterministic symbol ordering within each timestamp
