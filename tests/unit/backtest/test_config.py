"""Tests for backtest configuration loading and validation."""

import tempfile
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from qtrader.backtest.config import (
    BacktestConfig,
    ConcentrationLimitConfig,
    ConfigLoadError,
    DataConfig,
    ExecutionConfig,
    LeverageLimitConfig,
    PortfolioConfig,
    PositionSizingConfig,
    RiskBudgetConfig,
    RiskConfig,
    StrategyConfigItem,
    load_backtest_config,
)


class TestDataConfig:
    """Tests for DataConfig."""

    def test_valid_config(self):
        """Test valid data config."""
        config = DataConfig(source="schwab", data_path="/path/to/data", dataset="schwab-us-equity-1d-adjusted")
        assert config.source == "schwab"
        assert config.data_path == "/path/to/data"
        assert config.dataset == "schwab-us-equity-1d-adjusted"

    def test_invalid_source(self):
        """Test invalid data source."""
        with pytest.raises(ValueError, match="source must be one of"):
            DataConfig(source="invalid", data_path="/path", dataset="invalid-dataset")


class TestPortfolioConfig:
    """Tests for PortfolioConfig."""

    def test_valid_config(self):
        """Test valid portfolio config."""
        config = PortfolioConfig(
            initial_capital=Decimal("1000000"),
            commission_model="fixed",
            commission_rate=0.001,
            slippage_model="fixed",
            slippage_bps=5.0,
        )
        assert config.initial_capital == Decimal("1000000")
        assert config.commission_rate == 0.001

    def test_defaults(self):
        """Test default values."""
        config = PortfolioConfig(initial_capital=Decimal("1000000"))
        assert config.commission_model == "fixed"
        assert config.slippage_model == "fixed"

    def test_invalid_commission_model(self):
        """Test invalid commission model."""
        with pytest.raises(ValueError, match="commission_model must be one of"):
            PortfolioConfig(initial_capital=Decimal("1000000"), commission_model="invalid")


class TestRiskBudgetConfig:
    """Tests for RiskBudgetConfig."""

    def test_valid_budget(self):
        """Test valid budget."""
        budget = RiskBudgetConfig(strategy_id="momentum", capital_weight=0.6)
        assert budget.strategy_id == "momentum"
        assert budget.capital_weight == 0.6

    def test_weight_out_of_range(self):
        """Test capital weight out of range."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            RiskBudgetConfig(strategy_id="test", capital_weight=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            RiskBudgetConfig(strategy_id="test", capital_weight=-0.1)


class TestPositionSizingConfig:
    """Tests for PositionSizingConfig."""

    def test_valid_fraction(self):
        """Test valid sizing fraction."""
        sizing = PositionSizingConfig(fraction=0.03)
        assert sizing.fraction == 0.03

    def test_fraction_out_of_range(self):
        """Test fraction out of range."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            PositionSizingConfig(fraction=1.5)

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            PositionSizingConfig(fraction=0.0)


class TestConcentrationLimitConfig:
    """Tests for ConcentrationLimitConfig."""

    def test_valid_limit(self):
        """Test valid concentration limit."""
        limit = ConcentrationLimitConfig(max_position_pct=0.10)
        assert limit.max_position_pct == 0.10

    def test_limit_out_of_range(self):
        """Test limit out of range."""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            ConcentrationLimitConfig(max_position_pct=1.5)


class TestLeverageLimitConfig:
    """Tests for LeverageLimitConfig."""

    def test_valid_limits(self):
        """Test valid leverage limits."""
        limit = LeverageLimitConfig(max_gross=2.0, max_net=1.0)
        assert limit.max_gross == 2.0
        assert limit.max_net == 1.0

    def test_negative_leverage(self):
        """Test negative leverage."""
        with pytest.raises(ValueError, match="must be non-negative"):
            LeverageLimitConfig(max_gross=-1.0, max_net=1.0)


class TestRiskConfig:
    """Tests for RiskConfig."""

    def test_valid_config(self):
        """Test valid risk config."""
        config = RiskConfig(
            cash_buffer_pct=0.02,
            budgets=[
                RiskBudgetConfig(strategy_id="momentum", capital_weight=0.6),
                RiskBudgetConfig(strategy_id="mean_reversion", capital_weight=0.4),
            ],
            sizing={
                "momentum": PositionSizingConfig(fraction=0.03),
                "mean_reversion": PositionSizingConfig(fraction=0.02),
            },
            concentration=ConcentrationLimitConfig(max_position_pct=0.10),
            leverage=LeverageLimitConfig(max_gross=2.0, max_net=1.0),
        )
        assert len(config.budgets) == 2
        assert config.cash_buffer_pct == 0.02

    def test_budgets_sum_exceeds_one(self):
        """Test budget weights summing to > 1."""
        with pytest.raises(ValueError, match="budget weights sum to"):
            RiskConfig(
                budgets=[
                    RiskBudgetConfig(strategy_id="strat1", capital_weight=0.7),
                    RiskBudgetConfig(strategy_id="strat2", capital_weight=0.5),
                ],
                sizing={
                    "strat1": PositionSizingConfig(fraction=0.03),
                    "strat2": PositionSizingConfig(fraction=0.03),
                },
                concentration=ConcentrationLimitConfig(max_position_pct=0.10),
                leverage=LeverageLimitConfig(max_gross=2.0, max_net=1.0),
            )


class TestExecutionConfig:
    """Tests for ExecutionConfig."""

    def test_valid_config(self):
        """Test valid execution config."""
        config = ExecutionConfig(
            fill_policy="next_bar",
            commission_model="fixed",
            slippage_model="fixed",
        )
        assert config.fill_policy == "next_bar"

    def test_defaults(self):
        """Test default values."""
        config = ExecutionConfig()
        assert config.fill_policy == "next_bar"

    def test_invalid_fill_policy(self):
        """Test invalid fill policy."""
        with pytest.raises(ValueError, match="fill_policy must be one of"):
            ExecutionConfig(fill_policy="invalid")


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
    """Tests for BacktestConfig."""

    def test_valid_complete_config(self):
        """Test valid complete backtest config."""
        config = BacktestConfig(
            start_date=datetime(2020, 1, 1),
            end_date=datetime(2020, 12, 31),
            initial_capital=Decimal("1000000"),
            warmup_bars=100,
            universe=["AAPL", "GOOGL", "MSFT"],
            data=DataConfig(source="schwab", data_path="/data", dataset="schwab-us-equity-1d-adjusted"),
            portfolio=PortfolioConfig(initial_capital=Decimal("1000000")),
            risk=RiskConfig(
                budgets=[RiskBudgetConfig(strategy_id="test", capital_weight=0.5)],
                sizing={"test": PositionSizingConfig(fraction=0.03)},
                concentration=ConcentrationLimitConfig(max_position_pct=0.10),
                leverage=LeverageLimitConfig(max_gross=2.0, max_net=1.0),
            ),
            execution=ExecutionConfig(),
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
                data=DataConfig(source="schwab", data_path="/data", dataset="schwab-us-equity-1d-adjusted"),
                portfolio=PortfolioConfig(initial_capital=Decimal("1000000")),
                risk=RiskConfig(
                    budgets=[RiskBudgetConfig(strategy_id="test", capital_weight=0.5)],
                    sizing={"test": PositionSizingConfig(fraction=0.03)},
                    concentration=ConcentrationLimitConfig(max_position_pct=0.10),
                    leverage=LeverageLimitConfig(max_gross=2.0, max_net=1.0),
                ),
                execution=ExecutionConfig(),
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
                data=DataConfig(source="schwab", data_path="/data", dataset="schwab-us-equity-1d-adjusted"),
                portfolio=PortfolioConfig(initial_capital=Decimal("1000000")),
                risk=RiskConfig(
                    budgets=[RiskBudgetConfig(strategy_id="test", capital_weight=0.5)],
                    sizing={"test": PositionSizingConfig(fraction=0.03)},
                    concentration=ConcentrationLimitConfig(max_position_pct=0.10),
                    leverage=LeverageLimitConfig(max_gross=2.0, max_net=1.0),
                ),
                execution=ExecutionConfig(),
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
                data=DataConfig(source="schwab", data_path="/data", dataset="schwab-us-equity-1d-adjusted"),
                portfolio=PortfolioConfig(initial_capital=Decimal("1000000")),
                risk=RiskConfig(
                    budgets=[RiskBudgetConfig(strategy_id="test", capital_weight=0.5)],
                    sizing={"test": PositionSizingConfig(fraction=0.03)},
                    concentration=ConcentrationLimitConfig(max_position_pct=0.10),
                    leverage=LeverageLimitConfig(max_gross=2.0, max_net=1.0),
                ),
                execution=ExecutionConfig(),
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
            "data": {"source": "schwab", "data_path": "/data", "dataset": "schwab-us-equity-1d-adjusted"},
            "portfolio": {"initial_capital": 1000000},
            "risk": {
                "budgets": [{"strategy_id": "momentum", "capital_weight": 0.6}],
                "sizing": {"momentum": {"fraction": 0.03}},
                "concentration": {"max_position_pct": 0.10},
                "leverage": {"max_gross": 2.0, "max_net": 1.0},
            },
            "execution": {},
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
            "data": {"source": "schwab", "data_path": "/data", "dataset": "schwab-us-equity-1d-adjusted"},
            "portfolio": {"initial_capital": 1000000},
            "risk": {
                "budgets": [{"strategy_id": "test", "capital_weight": 0.5}],
                "sizing": {"test": {"fraction": 0.03}},
                "concentration": {"max_position_pct": 0.10},
                "leverage": {"max_gross": 2.0, "max_net": 1.0},
            },
            "execution": {},
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
            "data": {"source": "schwab", "data_path": "/data", "dataset": "schwab-us-equity-1d-adjusted"},
            "portfolio": {"initial_capital": 1000000},
            "risk": {
                "budgets": [{"strategy_id": "test", "capital_weight": 0.5}],
                "sizing": {"test": {"fraction": 0.03}},
                "concentration": {"max_position_pct": 0.10},
                "leverage": {"max_gross": 2.0, "max_net": 1.0},
            },
            "execution": {},
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
