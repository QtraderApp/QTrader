# Data Source Selection Guide

## Quick Answer

**To use Algoseek (or any data source), set the `source_tag` in `DataConfig`:**

```python
from qtrader.config.data_config import DataConfig, BarSchemaConfig
from qtrader.services.data import DataService

# Configure for Algoseek
config = DataConfig(
    mode="adjusted",
    source_tag="algoseek-adjusted",  # ← This selects Algoseek
    bar_schema=bar_schema,
)

service = DataService(config)
```

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                     Your Code                                │
│                                                               │
│  config = DataConfig(                                        │
│      source_tag="algoseek-adjusted"  ← You specify this     │
│  )                                                            │
│                                                               │
│  service = DataService(config)                               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  DataService                                 │
│                                                               │
│  1. Extracts source name: "algoseek"                         │
│     (splits on "-" and takes first part)                     │
│                                                               │
│  2. Looks up in data_sources.yaml:                           │
│     algoseek → adapter config                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              config/data_sources.yaml                        │
│                                                               │
│  data_sources:                                               │
│    algoseek:                                                 │
│      adapter: algoseekOHLC                                   │
│      root_path: "data/..."                                   │
│      symbol_map: "data/..."                                  │
│                                                               │
│    schwab:                                                   │
│      adapter: schwabOHLC                                     │
│      api:                                                    │
│        client_id: "${SCHWAB_API_KEY}"                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                Appropriate Adapter                           │
│                                                               │
│  - AlgoseekOHLCVendorAdapter                                 │
│  - SchwabOHLCVendorAdapter (future)                          │
│  - CSVAdapter (future)                                       │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                   Data Files                                 │
│                                                               │
│  Algoseek: Local parquet files                               │
│  Schwab: API calls + cache                                   │
│  CSV: Local CSV files                                        │
└─────────────────────────────────────────────────────────────┘
```

## Available Sources

### Current Implementation

| source_tag          | Description                  | Location                        |
| ------------------- | ---------------------------- | ------------------------------- |
| `algoseek-adjusted` | Local Algoseek parquet files | `data/us-equity-daily-ohlc-...` |
| `schwab-live`       | Schwab API (coming soon)     | Schwab API + cache              |
| `csv-file`          | CSV files (future)           | Local CSV directory             |

### Configuration File: `config/data_sources.yaml`

```yaml
data_sources:
  # Algoseek - Local parquet files
  algoseek:
    adapter: algoseekOHLC
    root_path: "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample"
    symbol_map: "data/equity_security_master_sample.csv"

  # Schwab - API with OAuth
  schwab:
    adapter: schwabOHLC
    cache_root: "data/us-equity-daily-adjusted-schwab"
    api:
      client_id: "${SCHWAB_API_KEY}"      # From environment
      client_secret: "${SCHWAB_API_SECRET}"
```

## source_tag Format

```
<source>-<description>
   ↑         ↑
   |         └── Optional: For documentation/clarity
   |             Examples: "adjusted", "live", "cached", "test"
   |
   └── Required: Must match key in data_sources.yaml
       Examples: "algoseek", "schwab", "csv"
```

**Examples:**

- `algoseek-adjusted` → Uses `algoseek` adapter
- `schwab-live` → Uses `schwab` adapter
- `schwab-cached` → Uses `schwab` adapter (same adapter, different semantics)
- `csv-test` → Uses `csv` adapter

## Common Patterns

### Pattern 1: Development/Backtesting (Algoseek)

```python
config = DataConfig(
    source_tag="algoseek-adjusted",  # ← Local files, fast
    bar_schema=bar_schema,
)
```

**Use when:**

- Developing strategies
- Running backtests
- Don't need live data
- Want fast iteration

### Pattern 2: Live Trading (Schwab - Future)

```python
config = DataConfig(
    source_tag="schwab-live",  # ← Live API data
    bar_schema=bar_schema,
)
```

**Use when:**

- Live trading
- Need real-time data
- Have API credentials

### Pattern 3: Testing (CSV)

```python
config = DataConfig(
    source_tag="csv-test",  # ← Simple CSV files
    bar_schema=bar_schema,
)
```

**Use when:**

- Unit testing
- Creating golden test data
- Simple data format needed

## How to Add a New Data Source

1. **Update `config/data_sources.yaml`:**

```yaml
data_sources:
  my_new_source:
    adapter: myNewAdapter
    # ... adapter-specific config
```

2. **Create adapter** (if needed):

```python
# src/qtrader/adapters/my_new_adapter.py
class MyNewAdapter:
    def read_bars(self, start_date, end_date):
        # Load data from your source
        pass
```

3. **Use in code:**

```python
config = DataConfig(
    source_tag="my_new_source-description",
    bar_schema=bar_schema,
)
```

## Environment Variables

For sources requiring credentials (like Schwab):

```bash
# Set in .env or shell
export SCHWAB_API_KEY="your_key_here"
export SCHWAB_API_SECRET="your_secret_here"
```

The `data_sources.yaml` file uses `${VAR_NAME}` syntax to reference them:

```yaml
schwab:
  api:
    client_id: "${SCHWAB_API_KEY}"  # ← Replaced at runtime
```

## Future Enhancement: Runtime Override

Currently planned for Phase 2 or later:

```python
# Override source at runtime (per-symbol)
iterator = service.load_symbol(
    "AAPL",
    date(2020, 1, 1),
    date(2020, 1, 31),
    data_source="schwab",  # ← Override default
)
```

This will allow mixing data sources in a single backtest.

## Summary

**To select Algoseek:**

1. Set `source_tag="algoseek-adjusted"` in `DataConfig`
1. Ensure `config/data_sources.yaml` has `algoseek` configuration
1. DataService automatically uses the right adapter

**Key insight:** The source selection happens at configuration time (when creating `DataConfig`), not at runtime (when calling `load_symbol`).
