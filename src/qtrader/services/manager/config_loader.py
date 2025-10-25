"""Configuration loader for RiskService.

Loads RiskConfig from YAML files with validation and helpful error messages.

Example YAML structure:
    budgets:
      - strategy_id: momentum_v1
        capital_weight: 0.6
      - strategy_id: mean_reversion_v1
        capital_weight: 0.4

    sizing:
      momentum_v1:
        model: fixed_fraction
        fraction: 0.03
      mean_reversion_v1:
        model: fixed_fraction
        fraction: 0.02

    concentration:
      max_position_pct: 0.10  # 10% per symbol

    leverage:
      max_gross: 2.0  # 200% gross exposure
      max_net: 1.0    # 100% net exposure

    cash_buffer_pct: 0.02  # Reserve 2% cash
"""

from pathlib import Path
from typing import Any

import yaml

from qtrader.services.manager.models import ConcentrationLimit, LeverageLimit, RiskConfig, SizingConfig, StrategyBudget


class ConfigLoadError(Exception):
    """Raised when config file cannot be loaded or is invalid."""

    pass


def load_risk_config(config_path: str | Path) -> RiskConfig:
    """Load RiskConfig from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated RiskConfig instance

    Raises:
        ConfigLoadError: If file not found, invalid YAML, or validation fails

    Example:
        >>> config = load_risk_config("config/risk.yaml")
        >>> print(config.budgets[0].strategy_id)
        momentum_v1
    """
    config_path = Path(config_path)

    # Check file exists
    if not config_path.exists():
        raise ConfigLoadError(f"Config file not found: {config_path}")

    # Load YAML
    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"Invalid YAML in {config_path}: {e}") from e
    except Exception as e:
        raise ConfigLoadError(f"Failed to read {config_path}: {e}") from e

    if not isinstance(data, dict):
        raise ConfigLoadError(f"Config must be a YAML dict, got {type(data).__name__}")

    # Parse and validate
    try:
        return _parse_config(data, config_path)
    except Exception as e:
        raise ConfigLoadError(f"Failed to parse config from {config_path}: {e}") from e


def _parse_config(data: dict[str, Any], config_path: Path) -> RiskConfig:
    """Parse raw YAML dict into RiskConfig.

    Args:
        data: Raw YAML data
        config_path: Original file path (for error messages)

    Returns:
        RiskConfig instance

    Raises:
        ConfigLoadError: If required fields missing or invalid
    """
    # Required fields
    required = ["budgets", "sizing", "concentration", "leverage"]
    missing = [f for f in required if f not in data]
    if missing:
        raise ConfigLoadError(f"Missing required fields in {config_path}: {', '.join(missing)}")

    # Parse budgets
    try:
        budgets = _parse_budgets(data["budgets"])
    except Exception as e:
        raise ConfigLoadError(f"Invalid budgets in {config_path}: {e}") from e

    # Parse sizing
    try:
        sizing = _parse_sizing(data["sizing"])
    except Exception as e:
        raise ConfigLoadError(f"Invalid sizing in {config_path}: {e}") from e

    # Parse concentration limit
    try:
        concentration = _parse_concentration(data["concentration"])
    except Exception as e:
        raise ConfigLoadError(f"Invalid concentration in {config_path}: {e}") from e

    # Parse leverage limit
    try:
        leverage = _parse_leverage(data["leverage"])
    except Exception as e:
        raise ConfigLoadError(f"Invalid leverage in {config_path}: {e}") from e

    # Optional: cash buffer (default 2%)
    cash_buffer_pct = data.get("cash_buffer_pct", 0.02)

    # Create RiskConfig (validation happens in __post_init__)
    try:
        return RiskConfig(
            budgets=budgets,
            sizing=sizing,
            concentration=concentration,
            leverage=leverage,
            cash_buffer_pct=cash_buffer_pct,
        )
    except Exception as e:
        raise ConfigLoadError(f"Config validation failed: {e}") from e


def _parse_budgets(budgets_data: Any) -> list[StrategyBudget]:
    """Parse budgets section.

    Args:
        budgets_data: Raw budgets data from YAML

    Returns:
        List of StrategyBudget instances

    Raises:
        ValueError: If budgets invalid
    """
    if not isinstance(budgets_data, list):
        raise ValueError(f"budgets must be a list, got {type(budgets_data).__name__}")

    if not budgets_data:
        raise ValueError("budgets cannot be empty")

    budgets = []
    for i, budget_dict in enumerate(budgets_data):
        if not isinstance(budget_dict, dict):
            raise ValueError(f"Budget {i} must be a dict, got {type(budget_dict).__name__}")

        # Required fields
        if "strategy_id" not in budget_dict:
            raise ValueError(f"Budget {i} missing strategy_id")
        if "capital_weight" not in budget_dict:
            raise ValueError(f"Budget {i} missing capital_weight")

        try:
            budget = StrategyBudget(
                strategy_id=budget_dict["strategy_id"],
                capital_weight=float(budget_dict["capital_weight"]),
            )
            budgets.append(budget)
        except Exception as e:
            raise ValueError(f"Budget {i} validation failed: {e}") from e

    return budgets


def _parse_sizing(sizing_data: Any) -> dict[str, SizingConfig]:
    """Parse sizing section.

    Args:
        sizing_data: Raw sizing data from YAML

    Returns:
        Dict mapping strategy_id -> SizingConfig

    Raises:
        ValueError: If sizing invalid
    """
    if not isinstance(sizing_data, dict):
        raise ValueError(f"sizing must be a dict, got {type(sizing_data).__name__}")

    if not sizing_data:
        raise ValueError("sizing cannot be empty")

    sizing = {}
    for strategy_id, config_dict in sizing_data.items():
        if not isinstance(config_dict, dict):
            raise ValueError(f"Sizing for '{strategy_id}' must be a dict, got {type(config_dict).__name__}")

        # Required fields
        if "model" not in config_dict:
            raise ValueError(f"Sizing for '{strategy_id}' missing model")
        if "fraction" not in config_dict:
            raise ValueError(f"Sizing for '{strategy_id}' missing fraction")

        try:
            config = SizingConfig(
                model=config_dict["model"],
                fraction=float(config_dict["fraction"]),
            )
            sizing[strategy_id] = config
        except Exception as e:
            raise ValueError(f"Sizing for '{strategy_id}' validation failed: {e}") from e

    return sizing


def _parse_concentration(concentration_data: Any) -> ConcentrationLimit:
    """Parse concentration section.

    Args:
        concentration_data: Raw concentration data from YAML

    Returns:
        ConcentrationLimit instance

    Raises:
        ValueError: If concentration invalid
    """
    if not isinstance(concentration_data, dict):
        raise ValueError(f"concentration must be a dict, got {type(concentration_data).__name__}")

    if "max_position_pct" not in concentration_data:
        raise ValueError("concentration missing max_position_pct")

    try:
        return ConcentrationLimit(max_position_pct=float(concentration_data["max_position_pct"]))
    except Exception as e:
        raise ValueError(f"Concentration validation failed: {e}") from e


def _parse_leverage(leverage_data: Any) -> LeverageLimit:
    """Parse leverage section.

    Args:
        leverage_data: Raw leverage data from YAML

    Returns:
        LeverageLimit instance

    Raises:
        ValueError: If leverage invalid
    """
    if not isinstance(leverage_data, dict):
        raise ValueError(f"leverage must be a dict, got {type(leverage_data).__name__}")

    if "max_gross" not in leverage_data:
        raise ValueError("leverage missing max_gross")
    if "max_net" not in leverage_data:
        raise ValueError("leverage missing max_net")

    try:
        return LeverageLimit(
            max_gross=float(leverage_data["max_gross"]),
            max_net=float(leverage_data["max_net"]),
        )
    except Exception as e:
        raise ValueError(f"Leverage validation failed: {e}") from e
