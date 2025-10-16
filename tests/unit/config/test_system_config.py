"""Unit tests for system configuration management.

Tests configuration loading, merging, environment variable substitution,
and dataclass construction for system-level settings.
"""

import os
from pathlib import Path
from unittest.mock import mock_open, patch

from qtrader.config.system_config import (
    CommissionConfig,
    DataConfig,
    ExecutionConfig,
    FillPolicyConfig,
    LoggingConfig,
    OutputConfig,
    ReportingConfig,
    SlippageConfig,
    SystemConfig,
    _deep_merge,
    _substitute_env_vars,
    get_config,
    reload_config,
)


class TestOutputConfig:
    """Test OutputConfig dataclass."""

    def test_output_config_default_values(self):
        """Test OutputConfig has correct default values."""
        # Arrange & Act
        config = OutputConfig()

        # Assert
        assert config.default_results_dir == "backtest_results"
        assert config.use_timestamps is True
        assert config.timestamp_format == "%Y%m%d_%H%M%S"
        assert config.organize_by_date is False
        assert "metadata" in config.generate_files
        assert config.generate_files["metadata"] is True

    def test_output_config_custom_values(self):
        """Test OutputConfig accepts custom values."""
        # Arrange & Act
        config = OutputConfig(
            default_results_dir="custom_results",
            use_timestamps=False,
            timestamp_format="%Y-%m-%d",
            organize_by_date=True,
            generate_files={"trades": True, "fills": False},
        )

        # Assert
        assert config.default_results_dir == "custom_results"
        assert config.use_timestamps is False
        assert config.timestamp_format == "%Y-%m-%d"
        assert config.organize_by_date is True
        assert config.generate_files == {"trades": True, "fills": False}


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""

    def test_logging_config_default_values(self):
        """Test LoggingConfig has correct default values."""
        # Arrange & Act
        config = LoggingConfig()

        # Assert
        assert config.level == "INFO"
        assert config.log_to_file is False
        assert config.log_dir == "logs"
        assert config.log_file_pattern == "qtrader_{timestamp}.log"
        assert config.structlog_format == "console"
        assert config.include_timestamps is True
        assert config.colorize is True

    def test_logging_config_custom_values(self):
        """Test LoggingConfig accepts custom values."""
        # Arrange & Act
        config = LoggingConfig(
            level="DEBUG",
            log_to_file=True,
            log_dir="custom_logs",
            log_file_pattern="app_{timestamp}.log",
            structlog_format="json",
            include_timestamps=False,
            colorize=False,
        )

        # Assert
        assert config.level == "DEBUG"
        assert config.log_to_file is True
        assert config.log_dir == "custom_logs"
        assert config.log_file_pattern == "app_{timestamp}.log"
        assert config.structlog_format == "json"
        assert config.include_timestamps is False
        assert config.colorize is False


class TestCommissionConfig:
    """Test CommissionConfig dataclass."""

    def test_commission_config_default_values(self):
        """Test CommissionConfig has correct default values."""
        # Arrange & Act
        config = CommissionConfig()

        # Assert
        assert config.model == "per_share"
        assert config.per_share == 0.0005
        assert config.minimum == 1.00
        assert config.maximum is None

    def test_commission_config_custom_values(self):
        """Test CommissionConfig accepts custom values."""
        # Arrange & Act
        config = CommissionConfig(model="percent", per_share=0.001, minimum=2.00, maximum=50.00)

        # Assert
        assert config.model == "percent"
        assert config.per_share == 0.001
        assert config.minimum == 2.00
        assert config.maximum == 50.00


class TestSlippageConfig:
    """Test SlippageConfig dataclass."""

    def test_slippage_config_default_values(self):
        """Test SlippageConfig has correct default values."""
        # Arrange & Act
        config = SlippageConfig()

        # Assert
        assert config.market_order_bps == 5
        assert config.limit_order_bps == 0
        assert config.stop_order_bps == 5
        assert config.mode == "conservative"

    def test_slippage_config_custom_values(self):
        """Test SlippageConfig accepts custom values."""
        # Arrange & Act
        config = SlippageConfig(
            market_order_bps=10,
            limit_order_bps=2,
            stop_order_bps=8,
            mode="aggressive",
        )

        # Assert
        assert config.market_order_bps == 10
        assert config.limit_order_bps == 2
        assert config.stop_order_bps == 8
        assert config.mode == "aggressive"


class TestFillPolicyConfig:
    """Test FillPolicyConfig dataclass."""

    def test_fill_policy_config_default_values(self):
        """Test FillPolicyConfig has correct default values."""
        # Arrange & Act
        config = FillPolicyConfig()

        # Assert
        assert config.limit_mode == "conservative"
        assert config.stop_mode == "conservative"
        assert config.moc_slip_bps == 5

    def test_fill_policy_config_custom_values(self):
        """Test FillPolicyConfig accepts custom values."""
        # Arrange & Act
        config = FillPolicyConfig(limit_mode="aggressive", stop_mode="realistic", moc_slip_bps=10)

        # Assert
        assert config.limit_mode == "aggressive"
        assert config.stop_mode == "realistic"
        assert config.moc_slip_bps == 10


class TestExecutionConfig:
    """Test ExecutionConfig dataclass."""

    def test_execution_config_default_values(self):
        """Test ExecutionConfig has correct default values."""
        # Arrange & Act
        config = ExecutionConfig()

        # Assert
        assert config.queue_bars == 3
        assert isinstance(config.commission, CommissionConfig)
        assert isinstance(config.slippage, SlippageConfig)
        assert isinstance(config.fill_policy, FillPolicyConfig)

    def test_execution_config_custom_values(self):
        """Test ExecutionConfig accepts custom nested configs."""
        # Arrange
        commission = CommissionConfig(model="percent", per_share=0.002)
        slippage = SlippageConfig(market_order_bps=15)
        fill_policy = FillPolicyConfig(limit_mode="aggressive")

        # Act
        config = ExecutionConfig(
            commission=commission,
            slippage=slippage,
            fill_policy=fill_policy,
            queue_bars=5,
        )

        # Assert
        assert config.queue_bars == 5
        assert config.commission.model == "percent"
        assert config.slippage.market_order_bps == 15
        assert config.fill_policy.limit_mode == "aggressive"


class TestDataConfig:
    """Test DataConfig dataclass."""

    def test_data_config_default_values(self):
        """Test DataConfig has correct default values."""
        # Arrange & Act
        config = DataConfig()

        # Assert
        assert config.sources_config == "config/data_sources.yaml"
        assert config.default_mode == "adjusted"
        assert config.default_timezone == "America/New_York"
        assert config.price_decimals == 4
        assert config.validate_on_load is True
        assert config.cache_enabled is False
        assert config.cache_dir == ".cache/qtrader"
        assert config.cache_max_size_mb == 1000

    def test_data_config_custom_values(self):
        """Test DataConfig accepts custom values."""
        # Arrange & Act
        config = DataConfig(
            sources_config="custom/sources.yaml",
            default_mode="total_return",
            default_timezone="UTC",
            price_decimals=2,
            validate_on_load=False,
            cache_enabled=True,
            cache_dir="/tmp/cache",
            cache_max_size_mb=5000,
        )

        # Assert
        assert config.sources_config == "custom/sources.yaml"
        assert config.default_mode == "total_return"
        assert config.default_timezone == "UTC"
        assert config.price_decimals == 2
        assert config.validate_on_load is False
        assert config.cache_enabled is True
        assert config.cache_dir == "/tmp/cache"
        assert config.cache_max_size_mb == 5000


class TestReportingConfig:
    """Test ReportingConfig dataclass."""

    def test_reporting_config_default_values(self):
        """Test ReportingConfig has correct default values."""
        # Arrange & Act
        config = ReportingConfig()

        # Assert
        assert "total_return" in config.metrics
        assert "sharpe_ratio" in config.metrics
        assert "max_drawdown" in config.metrics
        assert config.format_json is True
        assert config.format_csv is True
        assert config.format_html is False
        assert config.format_pdf is False
        assert config.decimal_places == 2
        assert config.include_trade_details is True

    def test_reporting_config_custom_values(self):
        """Test ReportingConfig accepts custom values."""
        # Arrange & Act
        config = ReportingConfig(
            metrics=["total_return", "sharpe_ratio"],
            format_json=False,
            format_csv=False,
            format_html=True,
            format_pdf=True,
            decimal_places=4,
            include_trade_details=False,
        )

        # Assert
        assert config.metrics == ["total_return", "sharpe_ratio"]
        assert config.format_json is False
        assert config.format_csv is False
        assert config.format_html is True
        assert config.format_pdf is True
        assert config.decimal_places == 4
        assert config.include_trade_details is False


class TestSystemConfig:
    """Test SystemConfig dataclass and configuration loading."""

    def test_system_config_default_values(self):
        """Test SystemConfig has correct default nested configs."""
        # Arrange & Act
        config = SystemConfig()

        # Assert
        assert isinstance(config.output, OutputConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.execution, ExecutionConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.reporting, ReportingConfig)

    def test_system_config_load_with_no_files_returns_defaults(self):
        """Test loading config with no files returns default values."""
        # Arrange
        with patch.object(Path, "exists", return_value=False):
            # Act
            config = SystemConfig.load()

            # Assert
            assert config.output.default_results_dir == "backtest_results"
            assert config.logging.level == "INFO"
            assert config.execution.queue_bars == 3
            assert config.data.default_mode == "adjusted"
            assert config.reporting.format_json is True

    def test_system_config_load_from_explicit_path(self):
        """Test loading config from explicit path."""
        # Arrange
        yaml_content = """
output:
  default_results_dir: "custom_output"
logging:
  level: "DEBUG"
"""
        mock_file = mock_open(read_data=yaml_content)

        with patch("builtins.open", mock_file):
            with patch.object(Path, "exists", return_value=True):
                # Act
                config = SystemConfig.load(Path("custom_config.yaml"))

                # Assert
                assert config.output.default_results_dir == "custom_output"
                assert config.logging.level == "DEBUG"

    def test_system_config_load_from_project_config(self):
        """Test loading config from project-relative path."""
        # Arrange
        yaml_content = """
execution:
  queue_bars: 10
data:
  default_timezone: "UTC"
"""
        mock_file = mock_open(read_data=yaml_content)

        with patch("builtins.open", mock_file):
            with patch.object(Path, "exists") as mock_exists:
                # Project config exists, home config doesn't
                mock_exists.side_effect = lambda: True

                # Act
                config = SystemConfig.load()

                # Assert
                assert config.execution.queue_bars == 10
                assert config.data.default_timezone == "UTC"

    def test_system_config_load_merges_multiple_configs(self):
        """Test loading merges project and home configs correctly."""
        # Arrange
        project_yaml = """
output:
  default_results_dir: "project_results"
logging:
  level: "INFO"
"""
        home_yaml = """
logging:
  level: "DEBUG"
  log_to_file: true
"""

        def mock_open_files(path, *args, **kwargs):
            if "qtrader.yaml" in str(path):
                return mock_open(read_data=project_yaml).return_value
            elif "config.yaml" in str(path):
                return mock_open(read_data=home_yaml).return_value
            raise FileNotFoundError()

        with patch("builtins.open", side_effect=mock_open_files):
            with patch.object(Path, "exists", return_value=True):
                # Act
                config = SystemConfig.load()

                # Assert
                # Home config overrides project config for logging.level
                assert config.logging.level == "DEBUG"
                assert config.logging.log_to_file is True
                # Project config value preserved
                assert config.output.default_results_dir == "project_results"

    def test_system_config_from_dict_builds_nested_configs(self):
        """Test _from_dict builds correct nested configuration objects."""
        # Arrange
        config_dict = {
            "output": {"default_results_dir": "test_results", "use_timestamps": False},
            "logging": {"level": "WARNING", "log_to_file": True},
            "execution": {
                "queue_bars": 7,
                "commission": {"model": "percent", "per_share": 0.002},
            },
            "data": {"default_mode": "total_return", "cache": {"enabled": True}},
            "reporting": {
                "decimal_places": 3,
                "formats": {"json": False, "html": True},
            },
        }

        # Act
        config = SystemConfig._from_dict(config_dict)

        # Assert
        assert config.output.default_results_dir == "test_results"
        assert config.output.use_timestamps is False
        assert config.logging.level == "WARNING"
        assert config.logging.log_to_file is True
        assert config.execution.queue_bars == 7
        assert config.execution.commission.model == "percent"
        assert config.data.default_mode == "total_return"
        assert config.data.cache_enabled is True
        assert config.reporting.decimal_places == 3
        assert config.reporting.format_json is False
        assert config.reporting.format_html is True

    def test_system_config_from_dict_with_empty_dict_uses_defaults(self):
        """Test _from_dict with empty dict uses all default values."""
        # Arrange
        config_dict = {}

        # Act
        config = SystemConfig._from_dict(config_dict)

        # Assert
        assert config.output.default_results_dir == "backtest_results"
        assert config.logging.level == "INFO"
        assert config.execution.queue_bars == 3
        assert config.data.default_mode == "adjusted"
        assert config.reporting.format_json is True

    def test_system_config_from_dict_partial_override(self):
        """Test _from_dict with partial config overrides only specified values."""
        # Arrange
        config_dict = {
            "logging": {"level": "ERROR"},  # Only override level
            "execution": {"queue_bars": 15},  # Only override queue_bars
        }

        # Act
        config = SystemConfig._from_dict(config_dict)

        # Assert
        # Overridden values
        assert config.logging.level == "ERROR"
        assert config.execution.queue_bars == 15
        # Default values preserved
        assert config.logging.log_to_file is False
        assert config.execution.commission.model == "per_share"
        assert config.output.default_results_dir == "backtest_results"


class TestDeepMerge:
    """Test _deep_merge utility function."""

    def test_deep_merge_empty_dicts_returns_empty(self):
        """Test merging two empty dicts returns empty dict."""
        # Arrange
        base = {}
        override = {}

        # Act
        result = _deep_merge(base, override)

        # Assert
        assert result == {}

    def test_deep_merge_override_wins(self):
        """Test override dict takes precedence for simple values."""
        # Arrange
        base = {"key1": "value1", "key2": "value2"}
        override = {"key2": "override2", "key3": "value3"}

        # Act
        result = _deep_merge(base, override)

        # Assert
        assert result["key1"] == "value1"
        assert result["key2"] == "override2"
        assert result["key3"] == "value3"

    def test_deep_merge_nested_dicts(self):
        """Test deep merging of nested dictionaries."""
        # Arrange
        base = {"level1": {"level2": {"key1": "value1", "key2": "value2"}, "other": "data"}}
        override = {"level1": {"level2": {"key2": "override2", "key3": "value3"}}}

        # Act
        result = _deep_merge(base, override)

        # Assert
        assert result["level1"]["level2"]["key1"] == "value1"
        assert result["level1"]["level2"]["key2"] == "override2"
        assert result["level1"]["level2"]["key3"] == "value3"
        assert result["level1"]["other"] == "data"

    def test_deep_merge_preserves_base_when_no_override(self):
        """Test base values preserved when override doesn't contain key."""
        # Arrange
        base = {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}
        override = {"f": 5}

        # Act
        result = _deep_merge(base, override)

        # Assert
        assert result["a"] == 1
        assert result["b"] == 2
        assert result["c"]["d"] == 3
        assert result["c"]["e"] == 4
        assert result["f"] == 5

    def test_deep_merge_non_dict_overrides_completely(self):
        """Test non-dict values in override replace base values completely."""
        # Arrange
        base = {"key": {"nested": "value"}}
        override = {"key": "simple_string"}

        # Act
        result = _deep_merge(base, override)

        # Assert
        assert result["key"] == "simple_string"

    def test_deep_merge_list_override(self):
        """Test list values in override replace base lists completely."""
        # Arrange
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}

        # Act
        result = _deep_merge(base, override)

        # Assert
        assert result["items"] == [4, 5]


class TestSubstituteEnvVars:
    """Test _substitute_env_vars utility function."""

    def test_substitute_env_vars_in_simple_string(self):
        """Test substituting environment variable in simple string."""
        # Arrange
        os.environ["TEST_VAR"] = "test_value"

        # Act
        result = _substitute_env_vars("prefix_${TEST_VAR}_suffix")

        # Assert
        assert result == "prefix_test_value_suffix"

        # Cleanup
        del os.environ["TEST_VAR"]

    def test_substitute_env_vars_in_dict(self):
        """Test substituting environment variables in dictionary."""
        # Arrange
        os.environ["DB_HOST"] = "localhost"
        os.environ["DB_PORT"] = "5432"
        config = {"database": {"host": "${DB_HOST}", "port": "${DB_PORT}"}}

        # Act
        result = _substitute_env_vars(config)

        # Assert
        assert result["database"]["host"] == "localhost"
        assert result["database"]["port"] == "5432"

        # Cleanup
        del os.environ["DB_HOST"]
        del os.environ["DB_PORT"]

    def test_substitute_env_vars_in_list(self):
        """Test substituting environment variables in list."""
        # Arrange
        os.environ["ITEM1"] = "value1"
        os.environ["ITEM2"] = "value2"
        config = ["${ITEM1}", "${ITEM2}", "literal"]

        # Act
        result = _substitute_env_vars(config)

        # Assert
        assert result[0] == "value1"
        assert result[1] == "value2"
        assert result[2] == "literal"

        # Cleanup
        del os.environ["ITEM1"]
        del os.environ["ITEM2"]

    def test_substitute_env_vars_missing_var_keeps_placeholder(self):
        """Test missing environment variable keeps placeholder."""
        # Arrange
        config = "value_is_${NONEXISTENT_VAR}"

        # Act
        result = _substitute_env_vars(config)

        # Assert
        assert result == "value_is_${NONEXISTENT_VAR}"

    def test_substitute_env_vars_nested_structure(self):
        """Test substituting in deeply nested structure."""
        # Arrange
        os.environ["LEVEL1"] = "value1"
        os.environ["LEVEL2"] = "value2"
        config = {
            "top": {
                "middle": {
                    "bottom": ["${LEVEL1}", {"key": "${LEVEL2}"}],
                    "sibling": "${LEVEL1}",
                }
            }
        }

        # Act
        result = _substitute_env_vars(config)

        # Assert
        assert result["top"]["middle"]["bottom"][0] == "value1"
        assert result["top"]["middle"]["bottom"][1]["key"] == "value2"
        assert result["top"]["middle"]["sibling"] == "value1"

        # Cleanup
        del os.environ["LEVEL1"]
        del os.environ["LEVEL2"]

    def test_substitute_env_vars_non_string_unchanged(self):
        """Test non-string values pass through unchanged."""
        # Arrange
        config = {"int": 42, "float": 3.14, "bool": True, "none": None}

        # Act
        result = _substitute_env_vars(config)

        # Assert
        assert result["int"] == 42
        assert result["float"] == 3.14
        assert result["bool"] is True
        assert result["none"] is None

    def test_substitute_env_vars_multiple_vars_in_string(self):
        """Test multiple environment variables in single string."""
        # Arrange
        os.environ["VAR1"] = "first"
        os.environ["VAR2"] = "second"
        config = "${VAR1}_middle_${VAR2}"

        # Act
        result = _substitute_env_vars(config)

        # Assert
        assert result == "first_middle_second"

        # Cleanup
        del os.environ["VAR1"]
        del os.environ["VAR2"]


class TestGlobalConfigSingleton:
    """Test global config singleton functions."""

    def test_get_config_returns_system_config(self):
        """Test get_config returns SystemConfig instance."""
        # Arrange
        import qtrader.config.system_config as config_module

        config_module._config = None  # Reset global state

        with patch.object(SystemConfig, "load") as mock_load:
            mock_load.return_value = SystemConfig()

            # Act
            config = get_config()

            # Assert
            assert isinstance(config, SystemConfig)
            mock_load.assert_called_once()

    def test_get_config_caches_config(self):
        """Test get_config caches config and doesn't reload."""
        # Arrange
        import qtrader.config.system_config as config_module

        config_module._config = None  # Reset global state

        with patch.object(SystemConfig, "load") as mock_load:
            mock_load.return_value = SystemConfig()

            # Act
            config1 = get_config()
            config2 = get_config()

            # Assert
            assert config1 is config2
            # Should only load once despite two calls
            assert mock_load.call_count == 1

    def test_reload_config_forces_reload(self):
        """Test reload_config forces config reload."""
        # Arrange
        import qtrader.config.system_config as config_module

        config_module._config = None  # Reset global state

        with patch.object(SystemConfig, "load") as mock_load:
            mock_load.return_value = SystemConfig()

            # Act
            get_config()
            reload_config()

            # Assert
            # Should be called twice: once for get_config, once for reload_config
            assert mock_load.call_count == 2

    def test_reload_config_with_path(self):
        """Test reload_config with explicit path."""
        # Arrange
        import qtrader.config.system_config as config_module

        config_module._config = None  # Reset global state
        test_path = Path("/test/config.yaml")

        with patch.object(SystemConfig, "load") as mock_load:
            mock_load.return_value = SystemConfig()

            # Act
            reload_config(test_path)

            # Assert
            mock_load.assert_called_once_with(test_path)


class TestSystemConfigIntegration:
    """Integration tests for complete configuration loading scenarios."""

    def test_load_config_with_env_vars_substitution(self):
        """Test loading config with environment variable substitution."""
        # Arrange
        os.environ["RESULTS_DIR"] = "/custom/results"
        os.environ["LOG_LEVEL"] = "WARNING"

        yaml_content = """
output:
  default_results_dir: "${RESULTS_DIR}"
logging:
  level: "${LOG_LEVEL}"
"""
        mock_file = mock_open(read_data=yaml_content)

        with patch("builtins.open", mock_file):
            with patch.object(Path, "exists", return_value=True):
                # Act
                config = SystemConfig.load(Path("test.yaml"))

                # Assert
                assert config.output.default_results_dir == "/custom/results"
                assert config.logging.level == "WARNING"

        # Cleanup
        del os.environ["RESULTS_DIR"]
        del os.environ["LOG_LEVEL"]

    def test_load_config_with_complex_nested_structure(self):
        """Test loading config with complex nested structure."""
        # Arrange
        yaml_content = """
output:
  default_results_dir: "results"
  generate_files:
    metadata: true
    trades: true
    fills: false
logging:
  level: "DEBUG"
  structlog:
    format: "json"
    colorize: false
execution:
  queue_bars: 5
  commission:
    model: "percent"
    per_share: 0.001
    minimum: 2.50
  slippage:
    market_order_bps: 10
    mode: "realistic"
  fill_policy:
    limit_mode: "aggressive"
data:
  sources_config: "custom/sources.yaml"
  cache:
    enabled: true
    cache_dir: "/tmp/cache"
    max_size_mb: 2000
reporting:
  metrics:
    - "total_return"
    - "sharpe_ratio"
  decimal_places: 4
  formats:
    json: true
    csv: true
    html: true
"""
        mock_file = mock_open(read_data=yaml_content)

        with patch("builtins.open", mock_file):
            with patch.object(Path, "exists", return_value=True):
                # Act
                config = SystemConfig.load(Path("test.yaml"))

                # Assert - Output
                assert config.output.default_results_dir == "results"
                assert config.output.generate_files["trades"] is True
                assert config.output.generate_files["fills"] is False

                # Assert - Logging
                assert config.logging.level == "DEBUG"
                assert config.logging.structlog_format == "json"
                assert config.logging.colorize is False

                # Assert - Execution
                assert config.execution.queue_bars == 5
                assert config.execution.commission.model == "percent"
                assert config.execution.commission.per_share == 0.001
                assert config.execution.slippage.market_order_bps == 10
                assert config.execution.fill_policy.limit_mode == "aggressive"

                # Assert - Data
                assert config.data.sources_config == "custom/sources.yaml"
                assert config.data.cache_enabled is True
                assert config.data.cache_dir == "/tmp/cache"

                # Assert - Reporting
                assert config.reporting.metrics == ["total_return", "sharpe_ratio"]
                assert config.reporting.decimal_places == 4
                assert config.reporting.format_html is True

    def test_load_config_empty_yaml_uses_defaults(self):
        """Test loading empty YAML file uses all default values."""
        # Arrange
        yaml_content = ""
        mock_file = mock_open(read_data=yaml_content)

        with patch("builtins.open", mock_file):
            with patch.object(Path, "exists", return_value=True):
                # Act
                config = SystemConfig.load(Path("empty.yaml"))

                # Assert - all defaults
                assert config.output.default_results_dir == "backtest_results"
                assert config.logging.level == "INFO"
                assert config.execution.queue_bars == 3
                assert config.data.default_mode == "adjusted"
                assert config.reporting.format_json is True
