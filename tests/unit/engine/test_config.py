"""Tests for backtest configuration loading and validation.

After refactoring, BacktestConfig only contains run parameters.
Service configurations (Risk, Execution, Portfolio) are tested in system/test_config.py.
"""

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from qtrader.engine.config import (
    BacktestConfig,
    ConfigLoadError,
    DataConfig,
    DataSourceConfig,
    RiskPolicyConfig,
    StrategyConfigItem,
    load_backtest_config,
)


class TestDataSourceConfig:
    """Tests for DataSourceConfig."""

    def test_valid_config(self):
        """Test valid data source config."""
        config = DataSourceConfig(name="algoseek-us-equity-1d-unadjusted", universe=["AAPL", "MSFT"])
        assert config.name == "algoseek-us-equity-1d-unadjusted"
        assert len(config.universe) == 2
        assert "AAPL" in config.universe


class TestDataConfig:
    """Tests for DataConfig."""

    def test_valid_single_source(self):
        """Test valid data config with single source."""
        config = DataConfig(
            sources=[DataSourceConfig(name="algoseek-us-equity-1d-unadjusted", universe=["AAPL", "MSFT"])]
        )
        assert len(config.sources) == 1
        assert config.sources[0].name == "algoseek-us-equity-1d-unadjusted"
        assert len(config.sources[0].universe) == 2


class TestStrategyConfigItem:
    """Tests for StrategyConfigItem."""

    def test_valid_config(self):
        """Test valid strategy config."""
        config = StrategyConfigItem(
            strategy_id="momentum_v1",
            universe=["AAPL", "MSFT"],
            data_sources=["algoseek-us-equity-1d-unadjusted"],
            config={"lookback": 20, "threshold": 0.05},
        )
        assert config.strategy_id == "momentum_v1"
        assert config.universe == ["AAPL", "MSFT"]


class TestRiskPolicyConfig:
    """Tests for RiskPolicyConfig."""

    def test_valid_config(self):
        """Test valid risk policy config."""
        config = RiskPolicyConfig(name="naive", config={"max_pct_position_size": 0.30})
        assert config.name == "naive"
        assert config.config["max_pct_position_size"] == 0.30


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_valid_complete_config(self):
        """Test valid complete backtest config."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_equity=Decimal("1000000"),
            data=DataConfig(
                sources=[DataSourceConfig(name="algoseek-us-equity-1d-unadjusted", universe=["AAPL", "GOOGL", "MSFT"])]
            ),
            strategies=[
                StrategyConfigItem(
                    strategy_id="test",
                    universe=["AAPL", "GOOGL"],
                    data_sources=["algoseek-us-equity-1d-unadjusted"],
                    config={},
                )
            ],
            risk_policy=RiskPolicyConfig(name="naive"),
        )
        assert config.start_date.year == 2020
        assert len(config.all_symbols) == 3

    def test_end_before_start(self):
        """Test end date before start date."""
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            BacktestConfig(
                start_date=datetime(2020, 12, 31),
                end_date=datetime(2020, 1, 1),
                initial_equity=Decimal("1000000"),
                data=DataConfig(sources=[DataSourceConfig(name="algoseek-us-equity-1d-unadjusted", universe=["AAPL"])]),
                strategies=[
                    StrategyConfigItem(
                        strategy_id="test",
                        universe=["AAPL"],
                        data_sources=["algoseek-us-equity-1d-unadjusted"],
                    )
                ],
                risk_policy=RiskPolicyConfig(name="naive"),
            )

    def test_replay_speed_default(self):
        """Test replay_speed defaults to 0.0."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_equity=Decimal("1000000"),
            data=DataConfig(sources=[DataSourceConfig(name="algoseek-us-equity-1d-unadjusted", universe=["AAPL"])]),
            strategies=[
                StrategyConfigItem(
                    strategy_id="test",
                    universe=["AAPL"],
                    data_sources=["algoseek-us-equity-1d-unadjusted"],
                )
            ],
            risk_policy=RiskPolicyConfig(name="naive"),
        )
        assert config.replay_speed == 0.0


class TestLoadBacktestConfig:
    """Tests for load_backtest_config function."""

    def test_load_valid_config(self):
        """Test loading valid YAML config."""
        config_dict = {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "initial_equity": 1000000,
            "data": {
                "sources": [
                    {"name": "algoseek-us-equity-1d-unadjusted", "universe": ["AAPL", "GOOGL"]},
                ]
            },
            "strategies": [
                {
                    "strategy_id": "momentum",
                    "universe": ["AAPL"],
                    "data_sources": ["algoseek-us-equity-1d-unadjusted"],
                }
            ],
            "risk_policy": {"name": "naive"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_backtest_config(temp_path)
            assert config.start_date.year == 2020
            assert len(config.all_symbols) == 2
        finally:
            Path(temp_path).unlink()

    def test_file_not_found(self):
        """Test loading non-existent file."""
        with pytest.raises(ConfigLoadError, match="Config file not found"):
            load_backtest_config("/nonexistent/path/config.yaml")

    def test_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: {")
            temp_path = f.name

        try:
            with pytest.raises(ConfigLoadError, match="Invalid YAML"):
                load_backtest_config(temp_path)
        finally:
            Path(temp_path).unlink()
