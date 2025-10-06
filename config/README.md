# QTrader Configuration

This directory contains system-level configuration files for QTrader. These are infrastructure settings that apply to the entire framework, **NOT** strategy-specific parameters.

## Configuration Files

### `qtrader.yaml` - Main System Configuration ⚙️

The primary configuration file for QTrader system settings.

**Key Sections:**

- **output**: Control where and how backtest results are saved
- **logging**: Configure logging levels and formats
- **execution**: Default commission and slippage models
- **data**: Data loading and caching settings (references `data_sources.yaml`)
- **risk**: Default risk management policies
- **backtest**: Default backtest parameters
- **reporting**: Metrics and report generation settings

**Example Customization:**

```yaml
# Change default results directory
output:
  default_results_dir: "my_results"
  organize_by_date: true  # Creates my_results/2025-10-06/strategy_timestamp/

# Increase logging verbosity
logging:
  level: "DEBUG"
  log_to_file: true

# Change commission model
execution:
  commission:
    model: "fixed"
    fixed_amount: 5.00
```

### `data_sources.yaml` - Data Adapter Configuration 📊

Maps logical data sources to physical data adapters and their settings.

**Why separate from `qtrader.yaml`?**

1. **Security**: Often contains sensitive credentials (API keys, DB passwords)
1. **Reusability**: Can be shared across multiple projects via `~/.qtrader/data_sources.yaml`
1. **Modularity**: Different tools may need only data config or only system config
1. **Size**: Can grow large with many data sources without cluttering main config

**Example:**

```yaml
data_sources:
  algoseek:
    adapter: algoseek_parquet
    root_path: "data/us-equity-daily-ohlc"
    mode: standard_adjusted

  csv_file:
    adapter: csv_adapter
    root_path: "data/csv"

  # Sensitive credentials
  database:
    adapter: postgres_adapter
    connection_string: "${DATABASE_URL}"  # From environment variable
```

**Security Best Practice:**

```bash
# Add to .gitignore if it contains secrets
echo "config/data_sources.yaml" >> .gitignore

# Use user config for sensitive data
cp config/data_sources.yaml ~/.qtrader/data_sources.yaml
# Edit ~/.qtrader/data_sources.yaml with your credentials
```

## Configuration Precedence

Settings are loaded with the following priority (highest to lowest):

1. **CLI flags** - Direct command-line arguments (e.g., `--out results/`)
1. **Environment variables** - Use `${VAR_NAME}` syntax in YAML
1. **Project config** - `./config/qtrader.yaml` (this directory)
1. **User config** - `~/.qtrader/config.yaml` (your home directory)
1. **Built-in defaults** - Hardcoded defaults in the codebase

## Usage

### In Python Code

```python
from qtrader.config.system_config import get_config

# Load configuration
config = get_config()

# Access settings
print(f"Results directory: {config.output.default_results_dir}")
print(f"Commission model: {config.execution.commission.model}")

# Use in your code
output_dir = Path(config.output.default_results_dir)
```

### Via CLI

The CLI automatically loads configuration:

```bash
# Uses configured default_results_dir
python -m qtrader.cli backtest --strategy my_strategy.py

# Override with CLI flag (takes precedence)
python -m qtrader.cli backtest --strategy my_strategy.py --out custom_dir/
```

## Environment Variable Substitution

Use `${VAR_NAME}` to reference environment variables:

```yaml
data_sources:
  database:
    adapter: postgres_adapter
    connection_string: "${DATABASE_URL}"

integration:
  webhooks:
    on_complete_url: "${WEBHOOK_URL}"
```

Then set in your shell:

```bash
export DATABASE_URL="postgresql://user:pass@localhost/qtrader"
export WEBHOOK_URL="https://hooks.example.com/backtest-complete"
```

## User-Specific Configuration

Create `~/.qtrader/config.yaml` for personal settings that apply across all projects:

```bash
mkdir -p ~/.qtrader
cat > ~/.qtrader/config.yaml << EOF
output:
  default_results_dir: "/home/javier/trading_results"
  organize_by_date: true

logging:
  level: "DEBUG"
  log_to_file: true

execution:
  commission:
    per_share: 0.001  # Higher commission for conservative estimates
EOF
```

These settings will be used for all QTrader projects unless overridden by project-specific config.

## Important: Strategy vs. System Configuration

### ✅ System Configuration (config/qtrader.yaml)

Framework-level settings that apply to all backtests:

- Output directory structure
- Logging configuration
- Default commission models
- Data source mappings
- Risk management defaults
- Report formatting

### ✅ Strategy Configuration (in .py files)

Strategy-specific parameters that define the trading logic:

```python
# In examples/sma_crossover_strategy.py
config = {
    "fast_period": 10,
    "slow_period": 30,
}

backtest_config = {
    "instruments": [...],
    "initial_cash": 100000.0,
    "position_size": 0.20,
}
```

**Rule of Thumb:**

- If it affects **how the framework runs**, it goes in `config/qtrader.yaml`
- If it affects **what the strategy does**, it stays in the strategy `.py` file

## Common Configurations

### Save Results to a Custom Directory

```yaml
# config/qtrader.yaml
output:
  default_results_dir: "results"
  organize_by_date: true
```

Results will be saved to: `results/2025-10-06/strategy_name_timestamp/`

### Enable Debug Logging

```yaml
# config/qtrader.yaml
logging:
  level: "DEBUG"
  log_to_file: true
  log_dir: "logs"

development:
  debug_mode: true
  save_debug_snapshots: true
```

### Custom Commission Model

```yaml
# config/qtrader.yaml
execution:
  commission:
    model: "tiered"
    tiers:
      - max_shares: 1000
        per_share: 0.001
      - max_shares: 10000
        per_share: 0.0005
      - per_share: 0.0003
    minimum: 1.00
```

### Aggressive Slippage for Conservative Testing

```yaml
# config/qtrader.yaml
execution:
  slippage:
    market_order_bps: 10  # 10 basis points
    stop_order_bps: 15    # 15 basis points
    mode: "aggressive"
```

## Validation

Check your configuration is valid:

```python
from qtrader.config.system_config import get_config

try:
    config = get_config()
    print("✓ Configuration loaded successfully")
    print(f"  Results dir: {config.output.default_results_dir}")
    print(f"  Log level: {config.logging.level}")
except Exception as e:
    print(f"✗ Configuration error: {e}")
```

## See Also

- [CLI Usage Guide](../docs/CLI_QUICKSTART.md)
- [Architecture Documentation](../docs/architecture.md)
- [Data Sources Documentation](../docs/dataset_specs/)
