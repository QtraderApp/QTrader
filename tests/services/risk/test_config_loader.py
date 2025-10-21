"""Tests for RiskService configuration loader.

Tests YAML config loading, validation, and error handling.
"""

import tempfile
from pathlib import Path

import pytest

from qtrader.services.risk.config_loader import ConfigLoadError, load_risk_config
from qtrader.services.risk.models import RiskConfig


@pytest.fixture
def valid_config_yaml() -> str:
    """Return valid YAML config string."""
    return """
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
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0

cash_buffer_pct: 0.02
"""


@pytest.fixture
def minimal_config_yaml() -> str:
    """Return minimal valid config (no optional cash_buffer_pct)."""
    return """
budgets:
  - strategy_id: strategy_a
    capital_weight: 1.0

sizing:
  strategy_a:
    model: fixed_fraction
    fraction: 0.05

concentration:
  max_position_pct: 0.15

leverage:
  max_gross: 1.5
  max_net: 1.0
"""


def test_load_valid_config(valid_config_yaml: str):
    """Test loading valid config from YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(valid_config_yaml)
        temp_path = Path(f.name)

    try:
        config = load_risk_config(temp_path)

        # Check type
        assert isinstance(config, RiskConfig)

        # Check budgets
        assert len(config.budgets) == 2
        assert config.budgets[0].strategy_id == "momentum_v1"
        assert config.budgets[0].capital_weight == 0.6
        assert config.budgets[1].strategy_id == "mean_reversion_v1"
        assert config.budgets[1].capital_weight == 0.4

        # Check sizing
        assert len(config.sizing) == 2
        assert "momentum_v1" in config.sizing
        assert config.sizing["momentum_v1"].model == "fixed_fraction"
        assert config.sizing["momentum_v1"].fraction == 0.03
        assert "mean_reversion_v1" in config.sizing
        assert config.sizing["mean_reversion_v1"].fraction == 0.02

        # Check concentration
        assert config.concentration.max_position_pct == 0.10

        # Check leverage
        assert config.leverage.max_gross == 2.0
        assert config.leverage.max_net == 1.0

        # Check cash buffer
        assert config.cash_buffer_pct == 0.02

    finally:
        temp_path.unlink()


def test_load_minimal_config(minimal_config_yaml: str):
    """Test loading minimal config without optional fields."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(minimal_config_yaml)
        temp_path = Path(f.name)

    try:
        config = load_risk_config(temp_path)

        # Should use default cash_buffer_pct
        assert config.cash_buffer_pct == 0.02

        # Check other fields loaded correctly
        assert len(config.budgets) == 1
        assert config.budgets[0].strategy_id == "strategy_a"
        assert config.concentration.max_position_pct == 0.15
        assert config.leverage.max_gross == 1.5

    finally:
        temp_path.unlink()


def test_load_file_not_found():
    """Test error when config file doesn't exist."""
    with pytest.raises(ConfigLoadError, match="Config file not found"):
        load_risk_config("/nonexistent/path/config.yaml")


def test_load_invalid_yaml():
    """Test error on malformed YAML."""
    bad_yaml = """
budgets:
  - strategy_id: test
    capital_weight: [this is not closed
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="Invalid YAML"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_not_dict():
    """Test error when YAML is not a dict."""
    not_dict_yaml = """
- just
- a
- list
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(not_dict_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="Config must be a YAML dict"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_missing_required_field():
    """Test error when required field is missing."""
    # Missing 'leverage' section
    incomplete_yaml = """
budgets:
  - strategy_id: test
    capital_weight: 1.0

sizing:
  test:
    model: fixed_fraction
    fraction: 0.05

concentration:
  max_position_pct: 0.10
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(incomplete_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="Missing required fields.*leverage"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_invalid_budgets_not_list():
    """Test error when budgets is not a list."""
    bad_yaml = """
budgets: "not a list"

sizing:
  test:
    model: fixed_fraction
    fraction: 0.05

concentration:
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="Invalid budgets.*must be a list"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_empty_budgets():
    """Test error when budgets list is empty."""
    bad_yaml = """
budgets: []

sizing:
  test:
    model: fixed_fraction
    fraction: 0.05

concentration:
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="budgets cannot be empty"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_budget_missing_field():
    """Test error when budget entry missing required field."""
    bad_yaml = """
budgets:
  - strategy_id: test
    # Missing capital_weight

sizing:
  test:
    model: fixed_fraction
    fraction: 0.05

concentration:
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="Budget.*missing capital_weight"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_invalid_sizing_not_dict():
    """Test error when sizing is not a dict."""
    bad_yaml = """
budgets:
  - strategy_id: test
    capital_weight: 1.0

sizing: "not a dict"

concentration:
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="Invalid sizing.*must be a dict"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_sizing_missing_field():
    """Test error when sizing config missing required field."""
    bad_yaml = """
budgets:
  - strategy_id: test
    capital_weight: 1.0

sizing:
  test:
    model: fixed_fraction
    # Missing fraction

concentration:
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="Sizing.*missing fraction"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_concentration_missing_field():
    """Test error when concentration missing max_position_pct."""
    bad_yaml = """
budgets:
  - strategy_id: test
    capital_weight: 1.0

sizing:
  test:
    model: fixed_fraction
    fraction: 0.05

concentration:
  wrong_field: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="concentration missing max_position_pct"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_leverage_missing_field():
    """Test error when leverage missing required field."""
    bad_yaml = """
budgets:
  - strategy_id: test
    capital_weight: 1.0

sizing:
  test:
    model: fixed_fraction
    fraction: 0.05

concentration:
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  # Missing max_net
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="leverage missing max_net"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_validation_error_in_model():
    """Test error when model validation fails (e.g., invalid capital_weight)."""
    bad_yaml = """
budgets:
  - strategy_id: test
    capital_weight: 1.5  # Invalid: > 1.0

sizing:
  test:
    model: fixed_fraction
    fraction: 0.05

concentration:
  max_position_pct: 0.10

leverage:
  max_gross: 2.0
  max_net: 1.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(bad_yaml)
        temp_path = Path(f.name)

    try:
        with pytest.raises(ConfigLoadError, match="validation failed"):
            load_risk_config(temp_path)
    finally:
        temp_path.unlink()


def test_load_with_path_object(valid_config_yaml: str):
    """Test loading with Path object instead of string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(valid_config_yaml)
        temp_path = Path(f.name)

    try:
        # Pass as Path object
        config = load_risk_config(temp_path)
        assert isinstance(config, RiskConfig)
        assert len(config.budgets) == 2

    finally:
        temp_path.unlink()


def test_load_with_string_path(valid_config_yaml: str):
    """Test loading with string path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(valid_config_yaml)
        temp_path = Path(f.name)

    try:
        # Pass as string
        config = load_risk_config(str(temp_path))
        assert isinstance(config, RiskConfig)
        assert len(config.budgets) == 2

    finally:
        temp_path.unlink()
