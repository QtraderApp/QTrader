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


class DataSourceConfig(BaseModel):
    """Configuration for a single data source with its universe.

    Allows specifying different symbols for different data sources.
    Example: Load AAPL prices from one source, AAPL news from another.
    """

    name: str = Field(..., description="Data source name from data_sources.yaml")
    universe: list[str] = Field(..., description="Symbols to load from this data source")


class DataConfig(BaseModel):
    """Data configuration for backtest run.

    Specifies which data sources to use and what symbols to load from each.
    Data source details (adapter, path, etc.) are defined in data_sources.yaml.
    """

    sources: list[DataSourceConfig] = Field(..., description="Data sources with their universes")


class StrategyConfigItem(BaseModel):
    """Configuration for a single strategy.

    Strategies are referenced by registry name (not file path).
    System loads custom strategies from custom_libraries.strategies path.
    """

    strategy_id: str = Field(..., description="Strategy name from registry (buildin or custom)")
    universe: list[str] = Field(..., description="Symbols this strategy trades (must be subset of backtest universe)")
    data_sources: list[str] = Field(
        ...,
        description="Data sources this strategy uses (e.g., ['algoseek-us-equity-1d-unadjusted', 'news-feed'])",
    )
    config: dict[str, Any] = Field(default_factory=dict, description="Strategy-specific config overrides")


class RiskPolicyConfig(BaseModel):
    """Risk policy configuration.

    Risk policies are referenced by registry name (buildin or custom).
    Applied at Portfolio Manager level, not strategy level.
    """

    name: str = Field(..., description="Risk policy name from registry")
    config: dict[str, Any] = Field(default_factory=dict, description="Policy-specific config overrides")


class BacktestConfig(BaseModel):
    """Backtest run configuration.

    Contains ONLY per-run parameters:
    - Dates, equity (what to test)
    - Data sources with per-source universes
    - Strategy configurations (which strategies to run)
    - Risk policy (Portfolio Manager level)

    Service configurations (execution, portfolio, logging) come from SystemConfig (system.yaml).

    Example YAML:
        ```yaml
        start_date: 2020-01-01
        end_date: 2023-12-31
        initial_equity: 100000
        replay_speed: 0.0  # Full speed (default). Use 1.0 for 1 sec/bar

        data:
          sources:
            - name: algoseek-us-equity-1d-adjusted
              universe: [AAPL, MSFT, GOOGL]  # Load these from algoseek
            - name: news-sentiment-feed
              universe: [AAPL]  # Load news only for AAPL

        strategies:
          - strategy_id: momentum_20  # Registry name
            universe: [AAPL, MSFT]  # Strategy trades subset of loaded symbols
            data_sources: [algoseek-us-equity-1d-adjusted]  # Which feeds to use
            config:
              lookback: 20
              warmup_bars: 21  # Strategy-specific warmup

        risk_policy:
          name: naive  # Registry name
          config:
            max_pct_position_size: 0.30
        ```
    """

    # Backtest parameters
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_equity: Decimal = Field(..., description="Starting equity")
    replay_speed: float = Field(
        default=0.0,
        ge=0.0,
        description="Replay speed in seconds per bar (0.0 = full speed, 1.0 = 1 sec/bar). "
        "For visualization/debugging only.",
    )

    # Data, strategies, and risk
    data: DataConfig = Field(..., description="Data sources with per-source universes")
    strategies: list[StrategyConfigItem] = Field(..., description="Strategy configurations")
    risk_policy: RiskPolicyConfig = Field(..., description="Risk policy (Portfolio Manager level)")

    @property
    def all_symbols(self) -> set[str]:
        """Get all symbols across all data sources."""
        symbols = set()
        for source in self.data.sources:
            symbols.update(source.universe)
        return symbols

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, v: datetime, info) -> datetime:
        """Validate end_date is after start_date."""
        if "start_date" in info.data and v <= info.data["start_date"]:
            raise ValueError("end_date must be after start_date")
        return v

    @field_validator("strategies")
    @classmethod
    def validate_strategy_universe(cls, v: list[StrategyConfigItem], info) -> list[StrategyConfigItem]:
        """Validate strategy universes are subsets of loaded symbols."""
        if "data" not in info.data:
            return v

        # Get all symbols from all data sources
        all_symbols = set()
        for source in info.data["data"].sources:
            all_symbols.update(source.universe)

        # Validate each strategy
        for strategy in v:
            strategy_symbols = set(strategy.universe)
            if not strategy_symbols.issubset(all_symbols):
                missing = strategy_symbols - all_symbols
                raise ValueError(
                    f"Strategy '{strategy.strategy_id}' universe contains symbols not in data sources: {missing}"
                )

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
