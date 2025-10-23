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

from qtrader.engine.config import BacktestConfig, ConfigLoadError, DataConfig, StrategyConfigItem, load_backtest_config


class TestDataConfig:
    """Tests for DataConfig."""

    def test_valid_config(self):
        """Test valid data config."""
        config = DataConfig(dataset="algoseek-us-equity-1d-unadjusted")
        assert config.dataset == "algoseek-us-equity-1d-unadjusted"


class TestStrategyConfigItem:
    """Tests for StrategyConfigItem."""

    def test_valid_config(self):
        """Test valid strategy config."""
        config = StrategyConfigItem(
            path="strategies/momentum.py",
            strategy_id="momentum_v1",
            config={"lookback": 20, "threshold": 0.05},
        )
        assert config.path == "strategies/momentum.py"
        assert config.config["lookback"] == 20

    def test_empty_config(self):
        """Test empty strategy config."""
        config = StrategyConfigItem(
            path="strategies/simple.py",
            strategy_id="simple",
        )
        assert config.config == {}


class TestBacktestConfig:
    """Tests for BacktestConfig (simplified - only run parameters)."""

    def test_valid_complete_config(self):
        """Test valid complete backtest config."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_capital=Decimal("1000000"),
            warmup_bars=100,
            universe=["AAPL", "GOOGL", "MSFT"],
            data=DataConfig(dataset="algoseek-us-equity-1d-unadjusted"),
            strategies=[
                StrategyConfigItem(
                    path="strategies/test.py",
                    strategy_id="test",
                    config={},
                )
            ],
        )
        assert config.start_date.year == 2020
        assert len(config.universe) == 3

    def test_end_before_start(self):
        """Test end date before start date."""
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            BacktestConfig(
                start_date=datetime(2020, 12, 31),
                end_date=datetime(2020, 1, 1),
                initial_capital=Decimal("1000000"),
                universe=["AAPL"],
                data=DataConfig(dataset="algoseek-us-equity-1d-unadjusted"),
                strategies=[StrategyConfigItem(path="test.py", strategy_id="test")],
            )

    def test_negative_warmup(self):
        """Test negative warmup bars."""
        with pytest.raises(ValueError, match="warmup_bars must be non-negative"):
            BacktestConfig(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31),
                initial_capital=Decimal("1000000"),
                warmup_bars=-10,
                universe=["AAPL"],
                data=DataConfig(dataset="algoseek-us-equity-1d-unadjusted"),
                strategies=[StrategyConfigItem(path="test.py", strategy_id="test")],
            )

    def test_empty_universe(self):
        """Test empty universe."""
        with pytest.raises(ValueError, match="universe cannot be empty"):
            BacktestConfig(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31),
                initial_capital=Decimal("1000000"),
                universe=[],
                data=DataConfig(dataset="algoseek-us-equity-1d-unadjusted"),
                strategies=[StrategyConfigItem(path="test.py", strategy_id="test")],
            )

    def test_replay_speed_default(self):
        """Test replay_speed defaults to 0.0 (full speed)."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_capital=Decimal("1000000"),
            universe=["AAPL"],
            data=DataConfig(dataset="algoseek-us-equity-1d-unadjusted"),
            strategies=[StrategyConfigItem(path="test.py", strategy_id="test")],
        )
        assert config.replay_speed == 0.0

    def test_replay_speed_custom(self):
        """Test custom replay_speed."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_capital=Decimal("1000000"),
            universe=["AAPL"],
            replay_speed=1.0,
            data=DataConfig(dataset="algoseek-us-equity-1d-unadjusted"),
            strategies=[StrategyConfigItem(path="test.py", strategy_id="test")],
        )
        assert config.replay_speed == 1.0

    def test_replay_speed_negative(self):
        """Test negative replay_speed is rejected."""
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            BacktestConfig(
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2020, 12, 31),
                initial_capital=Decimal("1000000"),
                universe=["AAPL"],
                replay_speed=-1.0,
                data=DataConfig(dataset="algoseek-us-equity-1d-unadjusted"),
                strategies=[StrategyConfigItem(path="test.py", strategy_id="test")],
            )


class TestLoadBacktestConfig:
    """Tests for load_backtest_config function."""

    def test_load_valid_config(self):
        """Test loading valid YAML config."""
        config_dict = {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "initial_capital": 1000000,
            "warmup_bars": 100,
            "universe": ["AAPL", "GOOGL"],
            "data": {"dataset": "algoseek-us-equity-1d-unadjusted"},
            "strategies": [{"path": "strategies/momentum.py", "strategy_id": "momentum"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            config = load_backtest_config(temp_path)
            assert config.start_date.year == 2020
            assert len(config.universe) == 2
            assert config.initial_capital == Decimal("1000000")
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

    def test_non_dict_config(self):
        """Test loading non-dict YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(["list", "not", "dict"], f)
            temp_path = f.name

        try:
            with pytest.raises(ConfigLoadError, match="must be a YAML dictionary"):
                load_backtest_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_validation_error(self):
        """Test config validation error."""
        config_dict = {
            "start_date": "2020-01-01",
            "end_date": "2019-01-01",  # Before start!
            "initial_capital": 1000000,
            "universe": ["AAPL"],
            "data": {"dataset": "algoseek-us-equity-1d-unadjusted"},
            "strategies": [{"path": "test.py", "strategy_id": "test"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name

        try:
            with pytest.raises(ConfigLoadError, match="Config validation failed"):
                load_backtest_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_with_path_object(self):
        """Test loading with Path object."""
        config_dict = {
            "start_date": "2020-01-01",
            "end_date": "2020-12-31",
            "initial_capital": 1000000,
            "universe": ["AAPL"],
            "data": {"dataset": "algoseek-us-equity-1d-unadjusted"},
            "strategies": [{"path": "test.py", "strategy_id": "test"}],
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = Path(f.name)

        try:
            config = load_backtest_config(temp_path)
            assert config.start_date.year == 2020
        finally:
            temp_path.unlink()
