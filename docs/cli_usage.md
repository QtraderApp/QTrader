# QTrader CLI Usage Guide

## Overview

The QTrader CLI provides a simple, self-contained way to run backtests. All configuration lives in the strategy file itself, making strategies portable and reproducible.

## Design Principles

1. **Self-Contained Strategies**: Each strategy file contains all necessary configuration
1. **Minimal CLI Arguments**: Only `--strategy` is required
1. **Configuration Override**: Use `--set` to override any parameter
1. **Sensible Defaults**: Output directory auto-generated with timestamp

## Strategy File Structure

A complete strategy file includes:

```python
# Strategy-specific parameters
config = {
    "fast_period": 20,
    "slow_period": 50,
}

# Backtest configuration (ALL settings)
backtest_config = {
    # Data configuration (REQUIRED)
    "data_paths": [
        "data/AAPL.parquet",
        "data/MSFT.parquet"
    ],
    "symbols": ["AAPL", "MSFT"],

    # Portfolio configuration
    "initial_cash": 100000.0,
    "position_size": 5000.0,
    "max_position_pct": 0.10,
    "allow_shorting": False,

    # Execution configuration
    "max_participation": 0.10,

    # Warmup configuration
    "warmup": True,
    "warmup_bars": None,  # None = auto-detect
}

# Strategy class
class MyStrategy(Strategy):
    def __init__(self, fast_period=20, slow_period=50):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def on_bar(self, bar, ctx):
        # Strategy logic here
        pass
```

## CLI Commands

### Basic Backtest

```bash
# Run with all config from strategy file
qtrader backtest --strategy strategies/sma_crossover.py

# Output: ./backtest_results/sma_crossover_20251006_143022/
```

### Custom Output Directory

```bash
qtrader backtest \
  --strategy strategies/sma_crossover.py \
  --out results/experiment_1
```

### Override Strategy Parameters

```bash
qtrader backtest \
  --strategy strategies/sma_crossover.py \
  --set fast_period=10 \
  --set slow_period=30
```

### Override Data and Symbols

```bash
# Test on different data
qtrader backtest \
  --strategy strategies/sma_crossover.py \
  --set 'data_paths=["data/TSLA.parquet"]' \
  --set 'symbols=["TSLA"]'
```

### Override Backtest Configuration

```bash
# Change initial cash and enable warmup
qtrader backtest \
  --strategy strategies/sma_crossover.py \
  --set initial_cash=200000 \
  --set warmup=true \
  --set warmup_bars=50
```

### Debug Mode

```bash
# Enable debug output (CSV exports)
qtrader backtest \
  --strategy strategies/sma_crossover.py \
  --debug \
  --verbose
```

## Configuration Precedence

1. **CLI `--set` overrides** (highest priority)
1. **Strategy file config**
1. **System defaults** (lowest priority)

## Required Configuration

The following must be specified in `backtest_config`:

- `data_paths`: List of data file paths (strings)
- `symbols`: List of symbols to trade (strings)

All other configuration has sensible defaults.

## Output

The CLI generates:

- **Console output**: Summary of configuration, progress, and results
- **Default output directory**: `./backtest_results/<strategy_name>_<timestamp>/`
- **Debug files** (with `--debug`):
  - `fills.csv`: All fill records
  - `portfolio_snapshots.csv`: Portfolio state over time

## Example Session

```bash
$ qtrader backtest --strategy examples/sma_crossover_strategy.py

============================================================
QTrader - Backtesting Engine
============================================================
Strategy file: sma_crossover_strategy.py
✓ Strategy loaded: SMACrossover

Strategy Configuration:
  fast_period: 20
  slow_period: 50

Backtest Configuration:
  Data: 1 file(s)
    - data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample/SecId=33127/data_0.parquet
  Symbols: AAPL
  Initial Cash: $100,000.00
  Position Size: $5,000.00
  Max Position %: 10.0%
  Allow Short: False
  Warmup: Enabled

✓ Data loaded: 252 bars
✓ Output directory: ./backtest_results/sma_crossover_strategy_20251006_143022

============================================================
Running backtest...
============================================================

============================================================
✓ Backtest Complete
============================================================
Duration: 0.12s
Bars Processed: 252
Total Fills: 8

Final Portfolio:
  Cash: $98,450.00
  Equity: $5,125.00
  Total Value: $103,575.00

P&L: $3,575.00 (+3.58%)

Results saved to: ./backtest_results/sma_crossover_strategy_20251006_143022
```

## Validate Data Command

Check data files without running a backtest:

```bash
qtrader validate-data \
  --data data/AAPL.parquet \
  --symbols AAPL
```

## Best Practices

1. **Version control strategy files**: Include `config` and `backtest_config` in git
1. **Use meaningful strategy names**: Files become output directory names
1. **Document configurations**: Add comments explaining parameter choices
1. **Test with --set first**: Override parameters before editing strategy file
1. **Use --debug for new strategies**: Verify fills and portfolio state

## Advanced Usage

### Parameter Sweeps

```bash
# Test multiple fast_period values
for fp in 10 15 20 25 30; do
  qtrader backtest \
    --strategy strategies/sma_crossover.py \
    --set fast_period=$fp \
    --out results/sweep_fp_${fp}
done
```

### Multiple Symbols

```bash
qtrader backtest \
  --strategy strategies/sma_crossover.py \
  --set 'symbols=["AAPL","MSFT","AMZN","GOOGL"]' \
  --set 'data_paths=["data/AAPL.parquet","data/MSFT.parquet","data/AMZN.parquet","data/GOOGL.parquet"]'
```

### Production Runs

```bash
# Full logging, debug output, custom directory
qtrader backtest \
  --strategy strategies/production_strategy.py \
  --out prod_runs/$(date +%Y%m%d) \
  --debug \
  --verbose
```
