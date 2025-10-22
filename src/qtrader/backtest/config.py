"""
Backtest Configuration Models.

Philosophy: Clean separation of concerns
- system.yaml: ALL service configurations (execution, risk, portfolio, data, etc.)
- backtest YAML: ONLY run parameters (dates, universe, capital) + strategies

This module provides BacktestConfig for per-run parameters.
Services get their configuration from SystemConfig (system.yaml).
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data configuration for backtest run.

    Specifies which dataset to use. Dataset details (source, path, etc.)
    are defined in data_sources.yaml and managed by SystemConfig.
    """

    dataset: str = Field(
        ...,
        description="Dataset name from data_sources.yaml (e.g., 'schwab-us-equity-1d-adjusted')",
    )


class StrategyConfigItem(BaseModel):
    """Configuration for a single strategy."""

    path: str = Field(..., description="Path to strategy .py file")
    strategy_id: str = Field(..., description="Unique strategy identifier")
    config: dict[str, Any] = Field(default_factory=dict, description="Strategy-specific config")


class BacktestConfig(BaseModel):
    """Backtest run configuration.

    Contains ONLY per-run parameters:
    - Dates, universe, capital (what to test)
    - Strategy configurations (which strategies)
    - Dataset selection (which data)

    Service configurations (execution, risk, portfolio) come from SystemConfig (system.yaml).

    Example YAML:
        ```yaml
        start_date: 2020-01-01
        end_date: 2023-12-31
        initial_capital: 100000
        universe: [AAPL, MSFT, GOOGL]
        warmup_bars: 20
        replay_speed: 0.0  # Full speed (default). Use 1.0 for 1 sec/bar visualization

        data:
          dataset: schwab-us-equity-1d-adjusted

        strategies:
          - path: strategies/momentum.py
            strategy_id: momentum_20
            config:
              lookback: 20
        ```
    """

    # Backtest parameters
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(..., description="Starting capital")
    warmup_bars: int = Field(default=0, description="Number of warmup bars")
    universe: list[str] = Field(..., description="List of symbols to trade")
    replay_speed: float = Field(
        default=0.0,
        ge=0.0,
        description="Replay speed in seconds per bar (0.0 = full speed, 1.0 = 1 sec/bar). "
        "Only applies to historical backtests for visualization/debugging.",
    )

    # Data and strategies
    data: DataConfig = Field(..., description="Dataset selection")
    strategies: list[StrategyConfigItem] = Field(..., description="Strategy configurations")

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: datetime, info) -> datetime:
        """Validate end_date is after start_date."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @field_validator("warmup_bars")
    @classmethod
    def validate_warmup(cls, v: int) -> int:
        """Validate warmup bars is non-negative."""
        if v < 0:
            raise ValueError("warmup_bars must be non-negative")
        return v

    @field_validator("universe")
    @classmethod
    def validate_universe(cls, v: list[str]) -> list[str]:
        """Validate universe is not empty."""
        if not v:
            raise ValueError("universe cannot be empty")
        return v


class ConfigLoadError(Exception):
    """Raised when config loading fails."""

    pass


def load_backtest_config(config_path: str | Path) -> BacktestConfig:
    """
    Load and validate backtest configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated BacktestConfig object

    Raises:
        ConfigLoadError: If file not found, invalid YAML, or validation fails

    Example:
        >>> config = load_backtest_config("my_backtest.yaml")
        >>> print(f"Running backtest from {config.start_date} to {config.end_date}")
        >>> print(f"Universe: {config.universe}")
        >>> print(f"Strategies: {[s.strategy_id for s in config.strategies]}")
    """
    path = Path(config_path)

    # Check file exists
    if not path.exists():
        raise ConfigLoadError(f"Config file not found: {path}")

    # Load YAML
    try:
        with path.open() as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in {path}: {e}") from e

    if not isinstance(raw_config, dict):
        raise ConfigLoadError(f"Config must be a YAML dictionary, got {type(raw_config)}")

    # Validate and construct BacktestConfig
    try:
        config = BacktestConfig(**raw_config)
    except Exception as e:
        raise ConfigLoadError(f"Config validation failed: {e}") from e

    return config
