# Phase 10: Configuration Management

## Overview

**Goal:** Create a centralized configuration system that manages all service configurations with validation, defaults, and environment-specific overrides.

**Duration:** 1-2 weeks **Complexity:** Low **Priority:** Medium - Quality of life improvement

## Question: Are Configs a Service?

### Answer: NO, But Needs Structure

Configuration is **NOT a service** in the lego architecture sense because:

- ❌ Config doesn't have business logic
- ❌ Config doesn't transform data
- ❌ Config doesn't make decisions

However, config **DOES need** structured management:

- ✅ Centralized location
- ✅ Validation (Pydantic models)
- ✅ Environment-specific overrides
- ✅ Type safety
- ✅ Documentation

## Current State

### What Exists

Scattered config files and classes:

```
config/
  data_sources.yaml          # Data source configs
  qtrader.yaml              # System config
  backtests/                # Backtest-specific configs

src/qtrader/config/
  data_config.py            # DataConfig (Pydantic)
  logging_config.py         # LoggingConfig, LoggerFactory
  system_config.py          # System settings
  # Missing: ExecutionConfig, RiskConfig, etc.
```

### Problems

- ❌ Configs scattered across modules
- ❌ Inconsistent validation
- ❌ No centralized defaults
- ❌ Hard to override for testing
- ❌ No environment-specific configs

## Target Architecture

### Configuration Structure

```
src/qtrader/config/
  __init__.py                # Export all configs
  base.py                    # BaseConfig with common functionality
  data_config.py             # DataConfig (data sources, paths)
  execution_config.py        # ExecutionConfig (slippage, commission)
  risk_config.py             # RiskConfig (limits, sizing)
  strategy_config.py         # StrategyConfig (parameters)
  analytics_config.py        # AnalyticsConfig (risk-free rate, etc.)
  reporting_config.py        # ReportingConfig (output formats)
  system_config.py           # SystemConfig (logging, threading)
  loader.py                  # ConfigLoader (load from files)

config/
  default.yaml               # Default configuration
  development.yaml           # Development overrides
  production.yaml            # Production settings
  test.yaml                  # Test-specific configs
```

### Configuration Classes (Pydantic)

```python
# src/qtrader/config/base.py

from abc import ABC
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class BaseConfig(BaseModel, ABC):
    """
    Base configuration class.

    All service configs inherit from this.
    Provides common functionality like validation, defaults, serialization.
    """

    class Config:
        """Pydantic config."""

        frozen = True  # Immutable
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseConfig":
        """Create from dictionary with validation."""
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: Path) -> "BaseConfig":
        """Load from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def merge(self, overrides: Dict[str, Any]) -> "BaseConfig":
        """Create new config with overrides applied."""
        data = self.to_dict()
        data.update(overrides)
        return self.__class__.from_dict(data)
```

### Service-Specific Configs

```python
# src/qtrader/config/execution_config.py

from decimal import Decimal
from typing import Literal

from qtrader.config.base import BaseConfig


class ExecutionConfig(BaseConfig):
    """
    Execution service configuration.

    Controls order execution behavior, slippage, commissions, etc.
    """

    # Slippage
    moc_slip_bps: Decimal = Decimal("2.0")  # Market-on-close slippage (bps)
    stop_slip_bps: Decimal = Decimal("2.0")  # Stop order slippage (bps)

    # Commission
    per_share: Decimal = Decimal("0.005")  # Per-share commission
    ticket_min: Decimal = Decimal("1.0")  # Minimum commission per order

    # Volume participation
    max_participation: Decimal = Decimal("0.10")  # Max 10% of bar volume
    allow_high_participation: bool = False  # Require explicit opt-in > 20%
    queue_bars: int = 5  # Max bars to queue unfilled orders

    # Fill policies
    limit_mode: Literal["conservative", "aggressive"] = "conservative"
    stop_mode: Literal["conservative", "aggressive"] = "conservative"


# src/qtrader/config/risk_config.py

from decimal import Decimal
from typing import Literal

from qtrader.config.base import BaseConfig


class RiskConfig(BaseConfig):
    """
    Risk management configuration.

    Controls position sizing, portfolio limits, etc.
    """

    # Position sizing
    sizing_method: Literal["fixed", "percent", "risk_based", "kelly"] = "percent"
    default_size: Decimal = Decimal("0.05")  # 5% of portfolio

    # Portfolio limits
    max_position_size: Decimal = Decimal("0.20")  # Max 20% per position
    max_total_exposure: Decimal = Decimal("1.0")  # Max 100% exposure
    max_leverage: Decimal = Decimal("1.0")  # No leverage by default

    # Risk parameters
    max_portfolio_risk: Decimal = Decimal("0.02")  # Max 2% portfolio risk per trade
    risk_free_rate: Decimal = Decimal("0.02")  # 2% annual risk-free rate


# src/qtrader/config/analytics_config.py

from qtrader.config.base import BaseConfig


class AnalyticsConfig(BaseConfig):
    """
    Analytics service configuration.

    Parameters for metrics calculation.
    """

    risk_free_rate: float = 0.02  # 2% annual
    trading_days_per_year: int = 252
    confidence_level: float = 0.95  # For VaR calculations


# src/qtrader/config/reporting_config.py

from pathlib import Path
from typing import List, Literal

from qtrader.config.base import BaseConfig


class ReportingConfig(BaseConfig):
    """
    Reporting service configuration.

    Output formats, file paths, plot settings.
    """

    # Output settings
    output_dir: Path = Path("output/backtests")
    formats: List[Literal["console", "json", "csv", "html"]] = ["console", "json"]

    # Plot settings
    generate_plots: bool = True
    plot_format: Literal["png", "svg", "pdf"] = "png"
    plot_dpi: int = 300

    # Console formatting
    decimal_places: int = 2
    show_full_results: bool = True
```

### Configuration Loader

```python
# src/qtrader/config/loader.py

from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from qtrader.config.analytics_config import AnalyticsConfig
from qtrader.config.data_config import DataConfig
from qtrader.config.execution_config import ExecutionConfig
from qtrader.config.reporting_config import ReportingConfig
from qtrader.config.risk_config import RiskConfig
from qtrader.config.system_config import SystemConfig


class ConfigLoader:
    """
    Centralized configuration loader.

    Loads configs from YAML files with environment-specific overrides.

    Usage:
        loader = ConfigLoader()
        config = loader.load("config/default.yaml", env="development")

        # Access service configs
        data_config = config["data"]
        execution_config = config["execution"]
    """

    # Config class registry
    CONFIG_CLASSES = {
        "data": DataConfig,
        "execution": ExecutionConfig,
        "risk": RiskConfig,
        "analytics": AnalyticsConfig,
        "reporting": ReportingConfig,
        "system": SystemConfig,
    }

    def __init__(self, config_dir: Path = Path("config")):
        """
        Initialize config loader.

        Args:
            config_dir: Directory containing config files
        """
        self.config_dir = config_dir

    def load(
        self,
        config_file: str = "default.yaml",
        env: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Load configuration with environment overrides.

        Args:
            config_file: Main config file (default.yaml)
            env: Environment name (development, production, test)
            overrides: Additional overrides (highest priority)

        Returns:
            Dict of service configs (already validated Pydantic models)

        Example:
            config = loader.load("default.yaml", env="development")
            execution_config: ExecutionConfig = config["execution"]
        """
        # Load base config
        base_path = self.config_dir / config_file
        base_data = self._load_yaml(base_path)

        # Load environment overrides
        if env:
            env_path = self.config_dir / f"{env}.yaml"
            if env_path.exists():
                env_data = self._load_yaml(env_path)
                base_data = self._deep_merge(base_data, env_data)

        # Apply manual overrides
        if overrides:
            base_data = self._deep_merge(base_data, overrides)

        # Validate and create Pydantic configs
        configs = {}
        for service_name, config_class in self.CONFIG_CLASSES.items():
            if service_name in base_data:
                configs[service_name] = config_class.from_dict(base_data[service_name])
            else:
                # Use defaults
                configs[service_name] = config_class()

        return configs

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML file."""
        with open(path) as f:
            return yaml.safe_load(f) or {}

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override into base."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
```

### Example Configuration File

```yaml
# config/default.yaml

data:
  sources:
    algoseek:
      root_path: "data/us-equity-daily-ohlc"
      path_template: "{root_path}/SecId={secid}/*.parquet"
      symbol_map: "data/equity_security_master.csv"

execution:
  moc_slip_bps: 2.0
  stop_slip_bps: 2.0
  per_share: 0.005
  ticket_min: 1.0
  max_participation: 0.10
  queue_bars: 5
  limit_mode: "conservative"

risk:
  sizing_method: "percent"
  default_size: 0.05
  max_position_size: 0.20
  max_total_exposure: 1.0

analytics:
  risk_free_rate: 0.02
  trading_days_per_year: 252

reporting:
  output_dir: "output/backtests"
  formats: ["console", "json", "csv"]
  generate_plots: true
  plot_format: "png"

system:
  log_level: "INFO"
  num_workers: 4
```

```yaml
# config/development.yaml (overrides for development)

system:
  log_level: "DEBUG"

reporting:
  show_full_results: true

execution:
  allow_high_participation: true  # Allow testing high participation
```

```yaml
# config/test.yaml (overrides for testing)

data:
  sources:
    algoseek:
      root_path: "tests/fixtures/data"

system:
  log_level: "WARNING"  # Less noise in tests

reporting:
  generate_plots: false  # Faster tests
```

## Usage in Services

### Service Initialization

```python
# Services receive typed config objects

# OLD (scattered, unvalidated)
service = ExecutionEngine(
    moc_slip_bps=2.0,
    per_share=0.005,
    ticket_min=1.0,
    # ... many parameters
)

# NEW (typed, validated)
config = loader.load("default.yaml", env="development")
execution_config: ExecutionConfig = config["execution"]
service = ExecutionService(execution_config)
```

### Testing with Overrides

```python
# Test with custom config
def test_execution_with_high_slippage():
    config = ExecutionConfig(
        moc_slip_bps=Decimal("10.0"),  # Test high slippage
        allow_high_participation=True,
    )

    service = ExecutionService(config)
    # Test behavior...
```

## Implementation Tasks

### Week 1: Core Infrastructure

- [ ] Create `BaseConfig` class
- [ ] Create service-specific configs:
  - [ ] `ExecutionConfig`
  - [ ] `RiskConfig`
  - [ ] `AnalyticsConfig`
  - [ ] `ReportingConfig`
- [ ] Create `ConfigLoader`
- [ ] Write unit tests for configs

### Week 2: Integration & Migration

- [ ] Create default config files (YAML)
- [ ] Create environment-specific configs
- [ ] Update all services to use typed configs
- [ ] Migration guide
- [ ] Documentation

## Validation Criteria

- [ ] ✅ All service configs use Pydantic models
- [ ] ✅ Config validation catches errors early
- [ ] ✅ Environment-specific overrides work
- [ ] ✅ Easy to test with custom configs
- [ ] ✅ Documentation with all options
- [ ] ✅ No more scattered config parameters

## Benefits

### Type Safety

```python
# Compile-time type checking
config: ExecutionConfig = ...
service = ExecutionService(config)  # Type-safe!
```

### Validation

```python
# Invalid config caught early
config = ExecutionConfig(
    max_participation=2.0  # ERROR: Must be <= 1.0
)
# Pydantic raises ValidationError
```

### Environment Management

```bash
# Development
qtrader backtest --config default.yaml --env development

# Production
qtrader backtest --config default.yaml --env production

# Testing
pytest  # Automatically uses test.yaml
```

### Documentation

```python
# Self-documenting with Pydantic
print(ExecutionConfig.model_json_schema())
# Shows all fields, types, defaults, descriptions
```

## Success Metrics

- [ ] ✅ Zero hardcoded config values in services
- [ ] ✅ All configs validated at startup
- [ ] ✅ Easy to add new config options
- [ ] ✅ Tests use isolated configs
- [ ] ✅ Clear documentation of all options

______________________________________________________________________

**Phase Status:** 📝 Planning **Dependencies:** All service phases **Estimated Duration:** 1-2 weeks **Last Updated:** October 15, 2025
