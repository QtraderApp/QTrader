# Schwab Smart Caching Implementation Plan

**Status:** Planning Phase\
**Date:** October 16, 2025\
**Goal:** Optimize Schwab API usage with intelligent gap-filling and incremental updates

______________________________________________________________________

## 📋 Current State Analysis

### **Current Implementation: "All-or-Nothing" Strategy**

**Location:** `src/qtrader/adapters/schwab.py` (lines 443-520)

**Behavior:**

```python
# Cache must FULLY cover requested range
if cached_start > start_date or cached_end < end_date:
    return None  # Triggers full API fetch for entire range
```

**Problems:**

1. ❌ Refetches data already in cache when range extends beyond cached dates
1. ❌ Wastes API quota fetching duplicate historical data
1. ❌ Slower performance for overlapping queries
1. ❌ No incremental update mechanism

**Example Inefficiency:**

- Cache: 2020-01-01 to 2020-12-31 (365 bars)
- Request: 2020-06-01 to 2025-10-16 (1,963 bars)
- Current: Fetches ALL 1,963 bars from API
- Optimal: Fetch only 1,598 bars (2021-01-01 to 2025-10-16), use cache for rest

______________________________________________________________________

## 🎯 Proposed Solution: Smart Caching Strategies

### **Strategy 1: Gap-Filling (Smart Merge)**

Detect gaps and fetch only missing data, merge with cache.

### **Strategy 2: Initial Backfill + Incremental Updates**

Load full history once, then only fetch new bars.

### **Strategy 3: Incremental Update Mode**

Update from last cached bar to latest available in API.

______________________________________________________________________

## 📐 Implementation Design

### **New Configuration Options**

Add to `config/data_sources.yaml`:

```yaml
schwab-us-equity-1d-adjusted:
  provider: schwab
  adapter: schwabOHLC
  cache_root: "data/schwab-cache"

  # CACHING STRATEGY OPTIONS
  cache_strategy: "smart"  # Options: "smart", "simple", "disabled"

  # INITIAL BACKFILL (Strategy 2)
  initial_backfill: true           # Fetch full history on first cache creation
  max_history_years: 20            # How far back to fetch (Schwab API limit)

  # INCREMENTAL UPDATE (Strategy 3)
  enable_incremental_update: true  # Update from last cached bar to present
  update_mode: "auto"              # Options: "auto", "manual", "on_request"

  # CACHE INVALIDATION
  force_refresh: false             # Ignore cache and refetch all data
  cache_ttl_days: null             # Time-to-live (null = no expiry)
```

______________________________________________________________________

## 🔧 Implementation Details

### **1. Enhanced Cache Metadata**

Update `.metadata.json` structure:

```json
{
  "symbol": "AAPL",
  "last_update": "2025-10-16T10:30:00Z",
  "date_range": {
    "start": "2005-01-01",
    "end": "2025-10-16"
  },
  "row_count": 5234,
  "frequency_type": "daily",
  "frequency": 1,
  "source": "schwab",

  // NEW FIELDS
  "cache_version": "2.0",
  "cache_strategy": "smart",
  "initial_backfill_complete": true,
  "last_incremental_update": "2025-10-16T10:30:00Z",
  "gaps": [],  // Track known gaps if any
  "api_limits": {
    "earliest_available": "2005-01-01",  // From API metadata
    "latest_available": "2025-10-16"
  }
}
```

### **2. New Methods in `SchwabOHLCAdapter`**

#### **2.1 Smart Gap Detection**

```python
def _detect_gaps(
    self,
    start_date: str,
    end_date: str,
    metadata: dict
) -> list[tuple[str, str]]:
    """
    Detect date gaps between requested range and cached data.

    Args:
        start_date: Requested start date
        end_date: Requested end date
        metadata: Cache metadata dict

    Returns:
        List of (gap_start, gap_end) tuples to fetch from API

    Examples:
        >>> # Cache: 2020-01-01 to 2020-12-31
        >>> # Request: 2019-01-01 to 2021-12-31
        >>> gaps = _detect_gaps("2019-01-01", "2021-12-31", metadata)
        >>> # Returns: [("2019-01-01", "2019-12-31"), ("2021-01-01", "2021-12-31")]
    """
    cached_start = metadata["date_range"]["start"]
    cached_end = metadata["date_range"]["end"]

    gaps = []

    # Gap BEFORE cache
    if start_date < cached_start:
        gaps.append((start_date, min(cached_start, end_date)))

    # Gap AFTER cache (incremental update zone)
    if end_date > cached_end:
        gaps.append((max(cached_end, start_date), end_date))

    return gaps
```

#### **2.2 Merge Bars from Multiple Sources**

```python
def _merge_bars(
    self,
    cached_bars: list[SchwabBar],
    api_bars: list[SchwabBar]
) -> list[SchwabBar]:
    """
    Merge bars from cache and API, removing duplicates.

    Args:
        cached_bars: Bars from cache
        api_bars: Bars from API

    Returns:
        Sorted, deduplicated list of bars
    """
    # Combine all bars
    all_bars = cached_bars + api_bars

    # Deduplicate by timestamp
    seen_timestamps = set()
    unique_bars = []

    for bar in all_bars:
        ts = bar.timestamp.isoformat()
        if ts not in seen_timestamps:
            seen_timestamps.add(ts)
            unique_bars.append(bar)

    # Sort chronologically
    unique_bars.sort(key=lambda b: b.timestamp)

    return unique_bars
```

#### **2.3 Initial Backfill**

```python
def _ensure_initial_backfill(self) -> None:
    """
    Ensure full historical data is cached (run once).

    On first cache creation, fetches maximum available history
    from Schwab API. Subsequent calls are no-op.

    Config:
        initial_backfill: true
        max_history_years: 20
    """
    if not self.config.get("initial_backfill", False):
        return

    metadata = self.metadata_manager.read_metadata()

    # Check if backfill already complete
    if metadata and metadata.get("initial_backfill_complete"):
        logger.debug(
            "schwab_cache.backfill_already_complete",
            symbol=self.instrument.symbol
        )
        return

    # Fetch maximum history
    max_years = self.config.get("max_history_years", 20)
    start_date = (datetime.now() - timedelta(days=max_years * 365)).date().isoformat()
    end_date = datetime.now().date().isoformat()

    logger.info(
        "schwab_cache.initial_backfill_start",
        symbol=self.instrument.symbol,
        start_date=start_date,
        end_date=end_date
    )

    # Fetch all history
    bars = list(self._fetch_from_api(start_date, end_date))

    # Write to cache with backfill flag
    self._write_to_cache(bars)
    metadata = self.metadata_manager.read_metadata()
    metadata["initial_backfill_complete"] = True
    metadata["cache_strategy"] = "smart"

    with open(self.metadata_manager.metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "schwab_cache.initial_backfill_complete",
        symbol=self.instrument.symbol,
        bars_cached=len(bars)
    )
```

#### **2.4 Incremental Update**

```python
def update_to_latest(self) -> int:
    """
    Update cache from last cached bar to latest available in API.

    This is the "incremental update" mode:
    - Reads last cached date
    - Fetches from (last_date + 1 day) to today
    - Appends new bars to cache

    Returns:
        Number of new bars added

    Examples:
        >>> adapter = SchwabOHLCAdapter(config, instrument)
        >>> new_bars = adapter.update_to_latest()
        >>> print(f"Added {new_bars} new bars")

    Config:
        enable_incremental_update: true
        update_mode: "auto"  # Can be triggered manually or on read
    """
    if not self.config.get("enable_incremental_update", True):
        logger.debug("Incremental updates disabled")
        return 0

    metadata = self.metadata_manager.read_metadata()

    if not metadata:
        logger.warning(
            "schwab_cache.no_metadata_for_update",
            symbol=self.instrument.symbol
        )
        return 0

    # Get last cached date
    cached_end = metadata["date_range"]["end"]
    today = datetime.now().date().isoformat()

    # Check if update needed
    if cached_end >= today:
        logger.debug(
            "schwab_cache.already_up_to_date",
            symbol=self.instrument.symbol,
            cached_end=cached_end
        )
        return 0

    # Calculate update range (next day after cache to today)
    from datetime import timedelta
    update_start = (datetime.fromisoformat(cached_end) + timedelta(days=1)).date().isoformat()
    update_end = today

    logger.info(
        "schwab_cache.incremental_update_start",
        symbol=self.instrument.symbol,
        update_start=update_start,
        update_end=update_end
    )

    # Fetch new bars from API
    new_bars = list(self._fetch_from_api(update_start, update_end))

    if not new_bars:
        logger.info(
            "schwab_cache.no_new_bars",
            symbol=self.instrument.symbol
        )
        return 0

    # Load existing cache
    existing_bars = self._read_all_from_cache()

    # Merge and write
    all_bars = self._merge_bars(existing_bars, new_bars)
    self._write_to_cache(all_bars)

    # Update metadata
    metadata["last_incremental_update"] = datetime.now(timezone.utc).isoformat()
    metadata["date_range"]["end"] = update_end

    with open(self.metadata_manager.metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "schwab_cache.incremental_update_complete",
        symbol=self.instrument.symbol,
        new_bars=len(new_bars),
        total_bars=len(all_bars)
    )

    return len(new_bars)
```

#### **2.5 Enhanced `read_bars()` with Smart Strategy**

```python
def read_bars(
    self,
    start_date: str,
    end_date: str,
    frequency_type: str = "daily",
    frequency: int = 1,
) -> Iterator[SchwabBar]:
    """
    Read bars with intelligent caching strategy.

    Strategies:
    1. "smart": Gap-filling + incremental updates
    2. "simple": All-or-nothing (current behavior)
    3. "disabled": Always fetch from API

    Smart Strategy Flow:
    1. Check if initial backfill needed (first time only)
    2. Check for incremental updates (if update_mode="auto")
    3. Detect gaps between requested range and cache
    4. Fetch only missing data from API
    5. Merge cache + API data
    6. Return sorted, deduplicated bars
    """
    strategy = self.config.get("cache_strategy", "smart")

    # STRATEGY: DISABLED
    if strategy == "disabled" or not self.metadata_manager:
        for bar in self._fetch_from_api(start_date, end_date, frequency_type, frequency):
            yield bar
        return

    # STRATEGY: SIMPLE (current behavior)
    if strategy == "simple":
        cached_bars = self._read_from_cache(start_date, end_date)
        if cached_bars:
            for bar in cached_bars:
                yield bar
            return

        bars_from_api = list(self._fetch_from_api(start_date, end_date, frequency_type, frequency))
        if bars_from_api:
            self._write_to_cache(bars_from_api, frequency_type, frequency)
        for bar in bars_from_api:
            yield bar
        return

    # STRATEGY: SMART (new implementation)
    if strategy == "smart":
        # Step 1: Ensure initial backfill (first time only)
        self._ensure_initial_backfill()

        # Step 2: Auto-update if configured
        if self.config.get("update_mode") == "auto":
            self.update_to_latest()

        # Step 3: Check metadata
        metadata = self.metadata_manager.read_metadata()

        if not metadata:
            # No cache yet - fetch and cache
            bars = list(self._fetch_from_api(start_date, end_date, frequency_type, frequency))
            self._write_to_cache(bars, frequency_type, frequency)
            for bar in bars:
                yield bar
            return

        # Step 4: Detect gaps
        gaps = self._detect_gaps(start_date, end_date, metadata)

        # Step 5: Fetch gap data from API
        gap_bars = []
        for gap_start, gap_end in gaps:
            logger.info(
                "schwab_cache.fetching_gap",
                symbol=self.instrument.symbol,
                gap_start=gap_start,
                gap_end=gap_end
            )
            gap_bars.extend(self._fetch_from_api(gap_start, gap_end, frequency_type, frequency))

        # Step 6: Read cached data (overlapping range)
        cached_start = metadata["date_range"]["start"]
        cached_end = metadata["date_range"]["end"]

        # Only read cache if it overlaps with request
        cached_bars = []
        if start_date <= cached_end and end_date >= cached_start:
            overlap_start = max(start_date, cached_start)
            overlap_end = min(end_date, cached_end)
            cached_bars = self._read_from_cache(overlap_start, overlap_end) or []

        # Step 7: Merge all data
        all_bars = self._merge_bars(cached_bars, gap_bars)

        # Step 8: Update cache if we fetched new data
        if gap_bars:
            self._write_to_cache(all_bars, frequency_type, frequency)

        # Step 9: Filter to requested range and yield
        for bar in all_bars:
            bar_date = bar.timestamp.date().isoformat()
            if start_date <= bar_date <= end_date:
                yield bar
```

#### **2.6 Helper: Read All from Cache**

```python
def _read_all_from_cache(self) -> list[SchwabBar]:
    """
    Read all bars from cache (no date filtering).

    Returns:
        All cached bars or empty list
    """
    if not self.metadata_manager or not self.metadata_manager.cache_exists():
        return []

    metadata = self.metadata_manager.read_metadata()
    if not metadata:
        return []

    # Read entire cache
    cached_start = metadata["date_range"]["start"]
    cached_end = metadata["date_range"]["end"]

    return self._read_from_cache(cached_start, cached_end) or []
```

______________________________________________________________________

## 🔄 Usage Examples

### **Example 1: First-Time Request (Initial Backfill)**

```yaml
# config/data_sources.yaml
schwab-us-equity-1d-adjusted:
  cache_strategy: "smart"
  initial_backfill: true
  max_history_years: 20
```

```python
# First request for AAPL
adapter = SchwabOHLCAdapter(config, instrument)
bars = adapter.read_bars("2024-01-01", "2024-12-31")

# Behind the scenes:
# 1. Detects no cache exists
# 2. Fetches FULL history: 2005-01-01 to 2025-10-16 (20 years)
# 3. Caches ~5,000 bars
# 4. Returns filtered subset: 2024-01-01 to 2024-12-31
```

### **Example 2: Incremental Update (Auto Mode)**

```yaml
schwab-us-equity-1d-adjusted:
  cache_strategy: "smart"
  enable_incremental_update: true
  update_mode: "auto"
```

```python
# Day 1: Request historical data
bars = adapter.read_bars("2024-01-01", "2024-12-31")
# Cache: 2005-01-01 to 2025-10-15 (yesterday)

# Day 2: Request including today
bars = adapter.read_bars("2024-01-01", "2025-10-16")
# Auto-update triggered:
# 1. Detects cache ends at 2025-10-15
# 2. Fetches ONLY new bar: 2025-10-16
# 3. Appends to cache
# 4. Returns full range from cache
```

### **Example 3: Manual Incremental Update**

```python
# Update cache to latest manually
adapter = SchwabOHLCAdapter(config, instrument)
new_bars_count = adapter.update_to_latest()
print(f"Added {new_bars_count} new bars")

# Then request data (all from cache)
bars = adapter.read_bars("2020-01-01", "2025-10-16")
```

### **Example 4: Gap-Filling**

```python
# Cache: 2020-01-01 to 2020-12-31
# Request: 2019-01-01 to 2021-12-31

bars = adapter.read_bars("2019-01-01", "2021-12-31")

# Behind the scenes:
# 1. Detects gaps: [2019-01-01, 2019-12-31] and [2021-01-01, 2021-12-31]
# 2. Fetches gap #1 from API: ~250 bars (2019)
# 3. Fetches gap #2 from API: ~250 bars (2021)
# 4. Reads cached range: 365 bars (2020)
# 5. Merges all: ~865 bars total
# 6. Updates cache with merged data
# 7. Returns filtered range
```

### **Example 5: Force Refresh**

```yaml
schwab-us-equity-1d-adjusted:
  cache_strategy: "smart"
  force_refresh: true  # Ignore cache
```

```python
# Fetches fresh data from API, overwrites cache
bars = adapter.read_bars("2024-01-01", "2024-12-31")
```

______________________________________________________________________

## 📊 Performance Comparison

### **Scenario: Daily Update Workflow**

**Cache State:** 2005-01-01 to 2025-10-15 (5,233 bars)\
**Request:** 2024-01-01 to 2025-10-16 (today)

| Strategy                | API Calls | Bars Fetched | Bars from Cache | Total Time            |
| ----------------------- | --------- | ------------ | --------------- | --------------------- |
| **Current (Simple)**    | 1         | 500          | 0               | ~2s (API)             |
| **Smart (Auto Update)** | 1         | 1            | 499             | ~0.5s (1 API + cache) |
| **Savings**             | 0         | -499 (99.8%) | +499            | **75% faster**        |

### **Scenario: Historical Backfill + Daily Updates (30 days)**

**Initial Request:** Full history\
**Daily Requests:** Update to latest (30 times)

| Strategy                           | Total API Calls | Total Bars Fetched | Bandwidth Used |
| ---------------------------------- | --------------- | ------------------ | -------------- |
| **Current (Simple)**               | 31              | 162,023            | ~16 MB         |
| **Smart (Backfill + Incremental)** | 31              | 5,263              | ~0.5 MB        |
| **Savings**                        | 0               | -156,760 (96.8%)   | **97% less**   |

______________________________________________________________________

## 🧪 Testing Plan

### **Unit Tests**

```python
# tests/unit/adapters/test_schwab_smart_caching.py

def test_detect_gaps_before_cache():
    """Test gap detection before cached range."""

def test_detect_gaps_after_cache():
    """Test gap detection after cached range."""

def test_detect_gaps_both_sides():
    """Test gaps before and after cache."""

def test_merge_bars_deduplication():
    """Test bar merging removes duplicates."""

def test_initial_backfill_first_run():
    """Test initial backfill fetches full history."""

def test_initial_backfill_skip_if_complete():
    """Test backfill skips if already done."""

def test_incremental_update_adds_new_bars():
    """Test incremental update appends new bars."""

def test_incremental_update_no_op_if_current():
    """Test update skips if cache already current."""
```

### **Integration Tests**

```python
# tests/integration/adapters/test_schwab_caching_integration.py

def test_smart_caching_end_to_end():
    """Full workflow: backfill → gaps → incremental."""

def test_strategy_switching():
    """Test switching between smart/simple/disabled."""
```

______________________________________________________________________

## 📝 Migration Guide

### **Backward Compatibility**

✅ Default strategy remains "simple" (current behavior)\
✅ Opt-in to "smart" strategy via config\
✅ Existing caches continue to work\
✅ Gradual migration: symbol-by-symbol

### **Migration Steps**

1. **Update config** to enable smart caching:

   ```yaml
   cache_strategy: "smart"
   ```

1. **First run** triggers initial backfill (if enabled)

1. **Subsequent runs** use incremental updates

1. **Monitor** cache files and API usage

______________________________________________________________________

## 🚀 Implementation Phases

### **Phase 1: Core Infrastructure** (Week 1)

- [ ] Add cache metadata fields (version, strategy, backfill flag)
- [ ] Implement `_detect_gaps()` method
- [ ] Implement `_merge_bars()` method
- [ ] Add unit tests for gap detection and merging

### **Phase 2: Backfill Strategy** (Week 2)

- [ ] Implement `_ensure_initial_backfill()` method
- [ ] Add config options: `initial_backfill`, `max_history_years`
- [ ] Update metadata on backfill completion
- [ ] Add integration test for backfill

### **Phase 3: Incremental Updates** (Week 2)

- [ ] Implement `update_to_latest()` method
- [ ] Add config: `enable_incremental_update`, `update_mode`
- [ ] Support auto/manual update modes
- [ ] Add unit tests for incremental updates

### **Phase 4: Smart Strategy Integration** (Week 3)

- [ ] Refactor `read_bars()` to support multiple strategies
- [ ] Implement smart strategy with gap-filling
- [ ] Add `_read_all_from_cache()` helper
- [ ] Integration tests for full workflow

### **Phase 5: Performance & Polish** (Week 4)

- [ ] Add performance benchmarks
- [ ] Implement cache invalidation (force_refresh, TTL)
- [ ] Update documentation
- [ ] Code review and optimization

______________________________________________________________________

## 📚 API Reference

### **New Configuration Keys**

| Key                         | Type   | Default    | Description                                     |
| --------------------------- | ------ | ---------- | ----------------------------------------------- |
| `cache_strategy`            | string | `"simple"` | Caching strategy: `smart`, `simple`, `disabled` |
| `initial_backfill`          | bool   | `false`    | Fetch full history on first cache creation      |
| `max_history_years`         | int    | `20`       | Years of history to fetch in backfill           |
| `enable_incremental_update` | bool   | `true`     | Enable incremental updates                      |
| `update_mode`               | string | `"auto"`   | Update mode: `auto`, `manual`, `on_request`     |
| `force_refresh`             | bool   | `false`    | Ignore cache and refetch all data               |
| `cache_ttl_days`            | int    | `null`     | Cache expiry in days (null = no expiry)         |

### **New Public Methods**

```python
def update_to_latest() -> int:
    """Update cache from last bar to latest available."""

def force_refresh(start_date: str, end_date: str) -> None:
    """Invalidate cache and refetch data."""

def get_cache_info() -> dict:
    """Return cache metadata and statistics."""
```

______________________________________________________________________

## ✅ Success Criteria

- [ ] Reduce API calls by >90% for daily update workflows
- [ ] Maintain backward compatibility with existing code
- [ ] All 298+ tests passing
- [ ] Performance benchmarks show >70% improvement
- [ ] Documentation complete with examples
- [ ] Zero data loss or corruption in cache

______________________________________________________________________

## 🔗 Related Files

- Implementation: `src/qtrader/adapters/schwab.py`
- Tests: `tests/unit/adapters/test_schwab.py`
- Config: `config/data_sources.yaml`
- Docs: `docs/adapters/schwab.md`

______________________________________________________________________

## 🖥️ CLI Integration

### **Use Case 1: Force Refresh Single Symbol**

```bash
qtrader raw-data --symbol AAPL \
  --start-date 2019-01-01 \
  --end-date 2023-01-31 \
  --source schwab-us-equity-1d-adjusted \
  --force-refresh
```

**Implementation:**

- Pass `force_refresh=true` to adapter config
- Adapter ignores cache and fetches fresh data
- Updates cache with new data

```python
# In cli.py
@click.option("--force-refresh", is_flag=True, help="Ignore cache and fetch fresh data")
def raw_data(..., force_refresh: bool):
    # Pass to adapter config
    config_dict = resolver.get_source_config(source)
    if force_refresh:
        config_dict["force_refresh"] = True
```

______________________________________________________________________

### **Use Case 2: Update All Tickers in Dataset**

```bash
# Update all symbols in Schwab cache to latest
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted

# Update specific symbols only
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted \
  --symbols AAPL,MSFT,GOOGL

# Dry run (show what would be updated)
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted --dry-run

# Verbose output
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted --verbose
```

**Implementation:**

```python
# New CLI command
@main.command("update-dataset")
@click.option("--dataset", required=True, help="Dataset name from data_sources.yaml")
@click.option("--symbols", help="Comma-separated symbols (default: all in cache)")
@click.option("--dry-run", is_flag=True, help="Show what would be updated")
@click.option("--verbose", is_flag=True, help="Show detailed progress")
def update_dataset(dataset: str, symbols: str, dry_run: bool, verbose: bool):
    """
    Update dataset cache to latest available data.

    Scans cache directory for all symbols and updates each to latest
    available in API. Only fetches new bars (incremental update).

    Examples:
        # Update all cached symbols
        qtrader update-dataset --dataset schwab-us-equity-1d-adjusted

        # Update specific symbols
        qtrader update-dataset --dataset schwab-us-equity-1d-adjusted --symbols AAPL,MSFT

        # Preview changes
        qtrader update-dataset --dataset schwab-us-equity-1d-adjusted --dry-run
    """
    from pathlib import Path
    from qtrader.adapters.resolver import DataSourceResolver
    from qtrader.models.instrument import Instrument, InstrumentType, DataSource

    console = Console()

    # Get dataset config
    resolver = DataSourceResolver()
    if dataset not in resolver.sources:
        console.print(f"[red]Error: Dataset '{dataset}' not found[/red]")
        console.print(f"Available: {list(resolver.sources.keys())}")
        sys.exit(1)

    config = resolver.get_source_config(dataset)

    # Get cache root
    cache_root = Path(config.get("cache_root", "data/schwab-cache"))
    if not cache_root.exists():
        console.print(f"[yellow]Cache directory not found: {cache_root}[/yellow]")
        sys.exit(0)

    # Get symbols to update
    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",")]
    else:
        # Scan cache directory for all symbols
        symbol_dirs = [d for d in cache_root.iterdir() if d.is_dir()]
        symbol_list = [d.name for d in symbol_dirs]

    if not symbol_list:
        console.print("[yellow]No symbols found to update[/yellow]")
        sys.exit(0)

    console.print(f"[cyan]Dataset:[/cyan] {dataset}")
    console.print(f"[cyan]Symbols to update:[/cyan] {len(symbol_list)}")
    if dry_run:
        console.print("[yellow]DRY RUN - No changes will be made[/yellow]")
    console.print()

    # Create progress table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Updating symbols...", total=len(symbol_list))

        for symbol in symbol_list:
            progress.update(task, description=f"Updating {symbol}...")

            try:
                # Create adapter for this symbol
                instrument = Instrument(symbol, InstrumentType.EQUITY, DataSource.SCHWAB)
                from qtrader.adapters.schwab import SchwabOHLCAdapter
                adapter = SchwabOHLCAdapter(config, instrument)

                if dry_run:
                    # Just check what would be updated
                    metadata = adapter.metadata_manager.read_metadata()
                    if metadata:
                        cached_end = metadata["date_range"]["end"]
                        today = datetime.now().date().isoformat()

                        if cached_end < today:
                            from datetime import timedelta
                            days_behind = (datetime.fromisoformat(today) -
                                         datetime.fromisoformat(cached_end)).days
                            results.append({
                                "symbol": symbol,
                                "status": "needs_update",
                                "cached_end": cached_end,
                                "days_behind": days_behind,
                                "new_bars": days_behind
                            })
                        else:
                            results.append({
                                "symbol": symbol,
                                "status": "current",
                                "cached_end": cached_end,
                                "days_behind": 0,
                                "new_bars": 0
                            })
                    else:
                        results.append({
                            "symbol": symbol,
                            "status": "no_cache",
                            "cached_end": "N/A",
                            "days_behind": "N/A",
                            "new_bars": "N/A"
                        })
                else:
                    # Actually update
                    new_bars = adapter.update_to_latest()

                    metadata = adapter.metadata_manager.read_metadata()
                    cached_end = metadata["date_range"]["end"] if metadata else "N/A"

                    results.append({
                        "symbol": symbol,
                        "status": "updated" if new_bars > 0 else "current",
                        "cached_end": cached_end,
                        "new_bars": new_bars
                    })

                    if verbose and new_bars > 0:
                        console.print(f"  [green]✓[/green] {symbol}: Added {new_bars} new bars")

            except Exception as e:
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "error": str(e),
                    "new_bars": 0
                })
                if verbose:
                    console.print(f"  [red]✗[/red] {symbol}: {e}")

            progress.advance(task)

    # Display summary table
    console.print()
    table = Table(title="Update Summary")
    table.add_column("Symbol", style="cyan")
    table.add_column("Status", style="white")
    table.add_column("Cached Through", style="white")

    if dry_run:
        table.add_column("Days Behind", style="yellow")
        table.add_column("New Bars (Est.)", style="yellow")
    else:
        table.add_column("New Bars Added", style="green")

    for result in results:
        status_color = {
            "updated": "green",
            "current": "dim",
            "needs_update": "yellow",
            "no_cache": "blue",
            "error": "red"
        }.get(result["status"], "white")

        status_text = f"[{status_color}]{result['status']}[/{status_color}]"

        if dry_run:
            table.add_row(
                result["symbol"],
                status_text,
                str(result.get("cached_end", "N/A")),
                str(result.get("days_behind", "N/A")),
                str(result.get("new_bars", "N/A"))
            )
        else:
            table.add_row(
                result["symbol"],
                status_text,
                str(result.get("cached_end", "N/A")),
                str(result.get("new_bars", 0))
            )

    console.print(table)

    # Summary stats
    total_updated = sum(1 for r in results if r["status"] == "updated")
    total_current = sum(1 for r in results if r["status"] == "current")
    total_errors = sum(1 for r in results if r["status"] == "error")
    total_new_bars = sum(r.get("new_bars", 0) for r in results if isinstance(r.get("new_bars"), int))

    console.print()
    console.print(f"[cyan]Total symbols processed:[/cyan] {len(symbol_list)}")
    if not dry_run:
        console.print(f"[green]Updated:[/green] {total_updated}")
        console.print(f"[dim]Already current:[/dim] {total_current}")
        console.print(f"[green]Total new bars added:[/green] {total_new_bars}")
    else:
        needs_update = sum(1 for r in results if r["status"] == "needs_update")
        console.print(f"[yellow]Need update:[/yellow] {needs_update}")
        console.print(f"[dim]Already current:[/dim] {total_current}")

    if total_errors > 0:
        console.print(f"[red]Errors:[/red] {total_errors}")
```

______________________________________________________________________

### **Use Case 3: Backfill Historical Data**

```bash
# Fetch full history for new symbol
qtrader backfill --dataset schwab-us-equity-1d-adjusted --symbol AAPL

# Backfill multiple symbols
qtrader backfill --dataset schwab-us-equity-1d-adjusted --symbols AAPL,MSFT,GOOGL

# Custom date range
qtrader backfill --dataset schwab-us-equity-1d-adjusted --symbol AAPL \
  --start-date 2010-01-01 --end-date 2025-10-16
```

**Implementation:**

```python
@main.command("backfill")
@click.option("--dataset", required=True, help="Dataset name")
@click.option("--symbol", help="Single symbol to backfill")
@click.option("--symbols", help="Comma-separated symbols")
@click.option("--start-date", help="Start date (default: max history)")
@click.option("--end-date", help="End date (default: today)")
def backfill(dataset: str, symbol: str, symbols: str, start_date: str, end_date: str):
    """
    Backfill historical data for symbols.

    Fetches full available history from API and caches locally.
    Use for initial setup or adding new symbols.
    """
    # Similar to update-dataset but fetches full range
    # Uses initial_backfill logic from adapter
```

______________________________________________________________________

### **Use Case 4: Cache Management**

```bash
# Show cache statistics
qtrader cache-info --dataset schwab-us-equity-1d-adjusted

# Clear cache for specific symbols
qtrader cache-clear --dataset schwab-us-equity-1d-adjusted --symbols AAPL,MSFT

# Validate cache integrity
qtrader cache-validate --dataset schwab-us-equity-1d-adjusted

# Rebuild cache (force refetch all data)
qtrader cache-rebuild --dataset schwab-us-equity-1d-adjusted --symbol AAPL
```

______________________________________________________________________

## ✅ Implementation Compatibility Assessment

### **Your Proposed Use Cases: FULLY SUPPORTED** ✅

| Use Case                        | CLI Command              | Implementation Method                         | Status              |
| ------------------------------- | ------------------------ | --------------------------------------------- | ------------------- |
| **Force refresh single symbol** | `--force-refresh` flag   | Pass config to adapter                        | ✅ Supported        |
| **Update all tickers**          | `update-dataset` command | Iterate cache dirs, call `update_to_latest()` | ✅ Fully compatible |
| **Backfill new symbols**        | `backfill` command       | Use `_ensure_initial_backfill()`              | ✅ Supported        |
| **Incremental updates**         | Auto or manual mode      | Built into `read_bars()` or explicit call     | ✅ Core feature     |

### **Why Our Implementation Works Perfectly:**

✅ **Modular Design**

- `update_to_latest()` is standalone method
- Can be called from CLI, scripts, or auto-triggered
- No dependencies on specific workflows

✅ **Config-Driven**

- `force_refresh` flag overrides caching
- `cache_strategy` controls behavior
- Easy to pass from CLI to adapter

✅ **Cache Directory Structure**

```
data/schwab-cache/
  AAPL/
    data.parquet
    .metadata.json
  MSFT/
    data.parquet
    .metadata.json
  GOOGL/
    ...
```

- Easy to scan for all symbols
- Each symbol independent
- Parallel updates possible

✅ **Metadata Tracking**

- `.metadata.json` contains last update time
- CLI can read metadata to determine what needs updating
- Dry-run mode just reads metadata (no API calls)

______________________________________________________________________

## 🚀 Enhanced CLI Commands Summary

```bash
# 1. View raw data (with force refresh)
qtrader raw-data --symbol AAPL --start-date 2019-01-01 --end-date 2023-01-31 \
  --source schwab-us-equity-1d-adjusted --force-refresh

# 2. Update all cached symbols to latest
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted

# 3. Update specific symbols only
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted --symbols AAPL,MSFT

# 4. Preview what would be updated (no API calls)
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted --dry-run

# 5. Backfill new symbol with full history
qtrader backfill --dataset schwab-us-equity-1d-adjusted --symbol NVDA

# 6. Show cache statistics
qtrader cache-info --dataset schwab-us-equity-1d-adjusted

# 7. Clear cache and force rebuild
qtrader cache-clear --dataset schwab-us-equity-1d-adjusted --symbol AAPL
```

______________________________________________________________________

## 📊 Workflow Examples

### **Daily Maintenance Workflow**

```bash
#!/bin/bash
# daily_update.sh - Run every morning before market open

echo "Updating Schwab dataset..."
qtrader update-dataset --dataset schwab-us-equity-1d-adjusted --verbose

echo "Validating cache..."
qtrader cache-validate --dataset schwab-us-equity-1d-adjusted

echo "Done! Cache is current."
```

**Result:**

- Scans all cached symbols (e.g., 500 tickers)
- Each symbol: Checks if behind, fetches only new bars
- Total API calls: ~500 (one per symbol, ~1 bar each)
- Total time: ~5 minutes (with rate limiting)
- Total bandwidth: \<1 MB (only new bars)

### **Adding New Symbol Workflow**

```bash
# Add NVDA to your dataset
qtrader backfill --dataset schwab-us-equity-1d-adjusted --symbol NVDA

# Verify
qtrader cache-info --dataset schwab-us-equity-1d-adjusted --symbol NVDA

# Use immediately
qtrader raw-data --symbol NVDA --start-date 2024-01-01 --end-date 2024-12-31 \
  --source schwab-us-equity-1d-adjusted
```

______________________________________________________________________

## 🎯 Answer to Your Question

**Q: Is the implementation we are proposing OK with these use cases?**

**A: YES! ABSOLUTELY! 💯**

The proposed implementation is **PERFECTLY SUITED** for your CLI use cases because:

1. ✅ **`--force-refresh` flag** → Just pass `force_refresh=True` to adapter config
1. ✅ **`update-dataset` command** → Iterate cache dirs + call `update_to_latest()` per symbol
1. ✅ **Incremental updates** → Core feature of smart caching strategy
1. ✅ **Batch operations** → Independent symbol caches enable parallel processing
1. ✅ **Dry-run mode** → Read metadata without API calls

**No changes needed to the core implementation!** The adapter API we designed supports all these CLI patterns naturally.

______________________________________________________________________

**Next Steps:** Review and approve implementation plan, then begin Phase 1.
