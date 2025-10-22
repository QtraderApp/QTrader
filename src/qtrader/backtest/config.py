"""
Backtest Configuration Models.

Provides typed configuration structures for backtest engine and all services.
Master config pattern: single YAML file configures all services.
"""

from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class DataConfig(BaseModel):
    """Data service configuration."""

    source: str = Field(..., description="Data source: 'schwab', 'algoseek', etc.")
    data_path: str = Field(..., description="Path to data directory")
    dataset: str = Field(
        ...,
        description="Dataset name from data_sources.yaml (e.g., 'schwab-us-equity-1d-adjusted')",
    )

    @field_validator("source")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """Validate data source."""
        allowed = ["schwab", "algoseek"]
        if v not in allowed:
            raise ValueError(f"source must be one of {allowed}, got {v}")
        return v


class PortfolioConfig(BaseModel):
    """Portfolio service configuration."""

    initial_capital: Decimal = Field(..., description="Starting capital")
    commission_model: str = Field(default="fixed", description="Commission model type")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    slippage_model: str = Field(default="fixed", description="Slippage model type")
    slippage_bps: float = Field(default=5.0, description="Slippage in basis points")

    @field_validator("commission_model")
    @classmethod
    def validate_commission_model(cls, v: str) -> str:
        """Validate commission model."""
        allowed = ["fixed", "tiered", "zero"]
        if v not in allowed:
            raise ValueError(f"commission_model must be one of {allowed}, got {v}")
        return v

    @field_validator("slippage_model")
    @classmethod
    def validate_slippage_model(cls, v: str) -> str:
        """Validate slippage model."""
        allowed = ["fixed", "volume", "none"]
        if v not in allowed:
            raise ValueError(f"slippage_model must be one of {allowed}, got {v}")
        return v


class RiskBudgetConfig(BaseModel):
    """Risk budget allocation for a strategy."""

    strategy_id: str = Field(..., description="Strategy identifier")
    capital_weight: float = Field(..., description="Fraction of capital allocated (0-1)")

    @field_validator("capital_weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        """Validate capital weight is in [0, 1]."""
        if not 0 <= v <= 1:
            raise ValueError(f"capital_weight must be between 0 and 1, got {v}")
        return v


class PositionSizingConfig(BaseModel):
    """Position sizing configuration for a strategy."""

    fraction: float = Field(..., description="Fraction of allocated capital per position")

    @field_validator("fraction")
    @classmethod
    def validate_fraction(cls, v: float) -> float:
        """Validate sizing fraction."""
        if not 0 < v <= 1:
            raise ValueError(f"fraction must be between 0 and 1, got {v}")
        return v


class ConcentrationLimitConfig(BaseModel):
    """Concentration limit configuration."""

    max_position_pct: float = Field(..., description="Max position size as % of equity")

    @field_validator("max_position_pct")
    @classmethod
    def validate_max_position(cls, v: float) -> float:
        """Validate concentration limit."""
        if not 0 < v <= 1:
            raise ValueError(f"max_position_pct must be between 0 and 1, got {v}")
        return v


class LeverageLimitConfig(BaseModel):
    """Leverage limit configuration."""

    max_gross: float = Field(..., description="Max gross leverage")
    max_net: float = Field(..., description="Max net leverage")

    @field_validator("max_gross", "max_net")
    @classmethod
    def validate_leverage(cls, v: float) -> float:
        """Validate leverage limits."""
        if v < 0:
            raise ValueError(f"leverage limit must be non-negative, got {v}")
        return v


class RiskConfig(BaseModel):
    """Risk service configuration."""

    cash_buffer_pct: float = Field(default=0.02, description="Cash buffer percentage")
    budgets: list[RiskBudgetConfig] = Field(..., description="Strategy capital allocations")
    sizing: dict[str, PositionSizingConfig] = Field(..., description="Position sizing per strategy")
    concentration: ConcentrationLimitConfig = Field(..., description="Concentration limits")
    leverage: LeverageLimitConfig = Field(..., description="Leverage limits")

    @field_validator("cash_buffer_pct")
    @classmethod
    def validate_cash_buffer(cls, v: float) -> float:
        """Validate cash buffer."""
        if not 0 <= v <= 1:
            raise ValueError(f"cash_buffer_pct must be between 0 and 1, got {v}")
        return v

    @field_validator("budgets")
    @classmethod
    def validate_budgets_sum(cls, v: list[RiskBudgetConfig]) -> list[RiskBudgetConfig]:
        """Validate budget weights sum to <= 1."""
        total = sum(b.capital_weight for b in v)
        if total > 1.0:
            raise ValueError(f"budget weights sum to {total}, must be <= 1.0")
        return v


class ExecutionConfig(BaseModel):
    """Execution service configuration."""

    fill_policy: str = Field(default="next_bar", description="Fill timing policy")
    commission_model: str = Field(default="fixed", description="Commission model")
    slippage_model: str = Field(default="fixed", description="Slippage model")

    @field_validator("fill_policy")
    @classmethod
    def validate_fill_policy(cls, v: str) -> str:
        """Validate fill policy."""
        allowed = ["next_bar", "immediate", "realistic"]
        if v not in allowed:
            raise ValueError(f"fill_policy must be one of {allowed}, got {v}")
        return v


class StrategyConfigItem(BaseModel):
    """Configuration for a single strategy."""

    path: str = Field(..., description="Path to strategy .py file")
    strategy_id: str = Field(..., description="Unique strategy identifier")
    config: dict[str, Any] = Field(default_factory=dict, description="Strategy-specific config")


class BacktestConfig(BaseModel):
    """Master backtest configuration."""

    # Backtest parameters
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")
    initial_capital: Decimal = Field(..., description="Starting capital")
    warmup_bars: int = Field(default=0, description="Number of warmup bars")
    universe: list[str] = Field(..., description="List of symbols to trade")

    # Service configurations
    data: DataConfig = Field(..., description="Data service config")
    portfolio: PortfolioConfig = Field(..., description="Portfolio service config")
    risk: RiskConfig = Field(..., description="Risk service config")
    execution: ExecutionConfig = Field(..., description="Execution service config")
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
        >>> config = load_backtest_config("config/backtest.yaml")
        >>> print(f"Running backtest from {config.start_date} to {config.end_date}")
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
