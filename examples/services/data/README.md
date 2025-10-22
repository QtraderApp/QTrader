# DataService Examples

This directory demonstrates how to use QTrader's `DataService` for loading and streaming historical price data.

## Overview

`DataService` is the **ONLY** service that reads from disk or external APIs. All other services (Strategy, Risk, Portfolio, Execution) subscribe to events published by DataService.

## Files

- **`config.yaml`** - Minimal data service configuration
- **`data_service_example.py`** - Comprehensive examples showing:
  - Event-driven mode (with EventBus)
  - Pull-based mode (without EventBus)
  - Universe loading (multiple symbols)

## Running the Example

```bash
python examples/services/data/data_service_example.py
```

## Key Concepts

### Event-Driven Mode (Recommended for Backtests)

```python
from qtrader.events.event_bus import EventBus
from qtrader.services.data.service import DataService

# Create EventBus
event_bus = EventBus()

# Subscribe to price_bar events
def handle_price_bar(event):
    print(f"{event.symbol}: ${event.bar.close}")

event_bus.subscribe("price_bar", handle_price_bar)

# Create DataService with EventBus
data_service = DataService(
    config=service_config,
    dataset="algoseek-us-equity-1d-unadjusted",
    event_bus=event_bus,
)

# Stream data - publishes PriceBarEvent for each bar
data_service.stream_universe(
    symbols=["AAPL", "MSFT"],
    start_date=date(2023, 1, 3),
    end_date=date(2023, 1, 31),
)
```

**What happens:**

1. DataService loads data from Algoseek parquet files
1. For each timestamp, publishes `PriceBarEvent` for all symbols
1. EventBus routes events to all subscribers
1. Your handler receives each event

**Event Ordering:**

- All bars for timestamp T published before T+1
- Ensures strategies see complete market snapshot
- No race conditions between symbols

### Pull-Based Mode (For Exploration)

```python
# Create DataService WITHOUT EventBus
data_service = DataService(
    config=service_config,
    dataset="algoseek-us-equity-1d-unadjusted",
    event_bus=None,  # No events
)

# Load data for single symbol
iterator = data_service.load_symbol(
    symbol="AAPL",
    start_date=date(2023, 1, 3),
    end_date=date(2023, 1, 10),
)

# Iterate through bars
for multi_bar in iterator:
    bar = multi_bar.adjusted
    print(f"{bar.trade_datetime}: Close=${bar.close}")
```

**Use cases:**

- Data exploration
- Testing
- Notebooks
- Non-real-time analysis

### Dataset Configuration

Datasets are defined in `config/data_sources.yaml`. The example uses:

```yaml
algoseek-us-equity-1d-unadjusted:
  provider: algoseek
  asset_class: equity
  frequency: 1d
  adjustment_mode: unadjusted
  adapter: algoseekOHLC
  root_path: "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample"
```

You can create your own datasets by:

1. Adding entry to `data_sources.yaml`
1. Pointing to your data location
1. Specifying the adapter (algoseek, schwab, etc.)

## Architecture

```
DataService
    ├── Loads data from disk/API
    ├── Uses DataLoader + Adapters
    ├── Publishes PriceBarEvent (if EventBus configured)
    └── Returns iterators (if no EventBus)

EventBus
    ├── Routes events to subscribers
    ├── Decouples DataService from other services
    └── Enables event-driven architecture

Other Services (Strategy, Risk, etc.)
    ├── Subscribe to PriceBarEvent
    ├── Never read from disk directly
    └── React to data as it arrives
```

## What DataService Does NOT Do

❌ **NO** trading logic ❌ **NO** risk management ❌ **NO** order execution ❌ **NO** portfolio tracking

✅ **ONLY** loads data and publishes events

## Next Steps

- **Strategy Examples**: `examples/services/strategy/`
- **Risk Examples**: `examples/services/risk/`
- **Full Backtest**: `src/qtrader/backtest/engine.py`
- **Documentation**: `docs/DATA_UPDATE_GUIDE.md`

## Troubleshooting

**"Symbol not found in symbol map"**

- Algoseek sample data only has AAPL, MSFT, (check `data/equity_security_master_sample.csv`)
- Use symbols from the sample data or add your own data

**"Dataset not found"**

- Check `config/data_sources.yaml` for available datasets
- Verify `root_path` points to existing data
- Ensure parquet files exist in expected location

**"EventBus not configured"**

- If using `stream_universe()`, you must pass `event_bus` to DataService
- For non-event mode, use `load_symbol()` or `load_universe()` instead
