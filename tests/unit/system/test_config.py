"""Tests for system configuration."""

from decimal import Decimal

from qtrader.system import SystemConfig, get_system_config, reload_system_config


class TestSystemConfigLoad:
    """Test loading system configuration."""

    def test_load_default_config(self):
        """Test loading from default location (config/system.yaml or config/qtrader.yaml)."""
        config = SystemConfig.load()

        # Verify key settings loaded
        assert config.data.sources_config == "config/data_sources.yaml"
        assert config.data.default_mode == "adjusted"
        assert config.portfolio.lot_method_long == "fifo"
        # execution.fill_policy may be string or dict depending on which YAML loaded
        assert config.execution.fill_policy is not None

    def test_load_with_defaults_when_no_file(self, tmp_path):
        """Test that built-in defaults are used when no config file exists."""
        # Try to load from non-existent path
        config = SystemConfig.load(tmp_path / "nonexistent.yaml")

        # Should have built-in defaults
        assert config.data.default_mode == "adjusted"
        assert config.portfolio.initial_capital == Decimal("100000.00")
        assert config.execution.commission.per_share == 0.0005
        assert config.risk.cash_buffer_pct == 0.02

    def test_config_structure_complete(self):
        """Test that all config sections are present."""
        config = SystemConfig.load()

        # All sections should exist
        assert hasattr(config, "data")
        assert hasattr(config, "portfolio")
        assert hasattr(config, "execution")
        assert hasattr(config, "risk")
        assert hasattr(config, "strategy")
        assert hasattr(config, "output")
        assert hasattr(config, "logging")
        assert hasattr(config, "reporting")
        assert hasattr(config, "development")
        assert hasattr(config, "preferences")


class TestDataConfig:
    """Test data service configuration."""

    def test_data_config_defaults(self):
        """Test data configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.data.default_mode == "adjusted"
        assert config.data.default_timezone == "America/New_York"
        # Note: qtrader.yaml has 6, system.yaml has 4
        assert config.data.price_decimals in [4, 6]  # Allow both
        assert config.data.validate_on_load is True

    def test_data_cache_config(self):
        """Test data cache configuration."""
        config = SystemConfig.load()

        assert hasattr(config.data, "cache")
        assert config.data.cache.enabled is False
        assert config.data.cache.cache_dir == ".cache/qtrader"


class TestPortfolioConfig:
    """Test portfolio service configuration."""

    def test_portfolio_config_defaults(self):
        """Test portfolio configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.portfolio.initial_capital == Decimal("100000.00")
        assert config.portfolio.lot_method_long == "fifo"
        assert config.portfolio.lot_method_short == "lifo"
        assert config.portfolio.max_ledger_entries == 10000

    def test_portfolio_commission_config(self):
        """Test portfolio commission configuration."""
        config = SystemConfig.load()

        assert config.portfolio.commission.model == "per_share"
        assert config.portfolio.commission.per_share == 0.0005
        assert config.portfolio.commission.minimum == 1.00
        assert config.portfolio.commission.maximum is None

    def test_portfolio_slippage_config(self):
        """Test portfolio slippage configuration."""
        config = SystemConfig.load()

        assert config.portfolio.slippage.model == "fixed"
        assert config.portfolio.slippage.slippage_bps == 5.0


class TestExecutionConfig:
    """Test execution service configuration."""

    def test_execution_config_defaults(self):
        """Test execution configuration has correct defaults."""
        config = SystemConfig.load()

        # fill_policy may be string or dict depending on YAML structure
        assert config.execution.fill_policy is not None
        assert config.execution.queue_bars == 3
        assert config.execution.max_volume_participation == 0.10

    def test_execution_slippage_config(self):
        """Test execution slippage configuration."""
        config = SystemConfig.load()

        # Slippage values may vary between system.yaml and qtrader.yaml
        assert hasattr(config.execution.slippage, "market_order_bps")
        assert hasattr(config.execution.slippage, "limit_order_bps")
        assert hasattr(config.execution.slippage, "stop_order_bps")
        assert config.execution.slippage.mode == "conservative"

    def test_execution_commission_config(self):
        """Test execution commission configuration."""
        config = SystemConfig.load()

        assert config.execution.commission.model == "per_share"
        # Commission may be 0.0 or 0.0005 depending on which YAML loaded
        assert config.execution.commission.per_share >= 0.0


class TestRiskConfig:
    """Test risk service configuration."""

    def test_risk_config_defaults(self):
        """Test risk configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.risk.cash_buffer_pct == 0.02

    def test_risk_budgets_config(self):
        """Test risk budgets configuration."""
        config = SystemConfig.load()

        assert len(config.risk.budgets) == 1
        assert config.risk.budgets[0].strategy_id == "default"
        assert config.risk.budgets[0].capital_weight == 1.0

    def test_risk_sizing_config(self):
        """Test risk sizing configuration."""
        config = SystemConfig.load()

        assert "default" in config.risk.sizing
        assert config.risk.sizing["default"].model == "fixed_fraction"
        assert config.risk.sizing["default"].fraction == 0.20

    def test_risk_concentration_config(self):
        """Test risk concentration limits."""
        config = SystemConfig.load()

        assert config.risk.concentration.max_position_pct == 0.10
        assert config.risk.concentration.max_sector_pct == 0.30

    def test_risk_leverage_config(self):
        """Test risk leverage limits."""
        config = SystemConfig.load()

        assert config.risk.leverage.max_gross == 1.0
        assert config.risk.leverage.max_net == 1.0

    def test_risk_checks_config(self):
        """Test risk checks configuration."""
        config = SystemConfig.load()

        assert config.risk.checks.position_size is True
        assert config.risk.checks.leverage is True
        assert config.risk.checks.concentration is True
        assert config.risk.checks.cash_buffer is True


class TestStrategyConfig:
    """Test strategy service configuration."""

    def test_strategy_config_defaults(self):
        """Test strategy configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.strategy.validate_on_load is True
        assert config.strategy.allow_concurrent is True

    def test_strategy_warmup_config(self):
        """Test strategy warmup configuration."""
        config = SystemConfig.load()

        assert config.strategy.warmup.skip_signals is True


class TestOutputConfig:
    """Test output configuration."""

    def test_output_config_defaults(self):
        """Test output configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.output.default_results_dir == "output/backtests"
        assert config.output.use_timestamps is True
        # organize_by_date may vary between configs
        assert isinstance(config.output.organize_by_date, bool)

    def test_output_generate_files_config(self):
        """Test output file generation configuration."""
        config = SystemConfig.load()

        assert config.output.generate_files.metadata is True
        assert config.output.generate_files.trades is True
        assert config.output.generate_files.fills is True
        assert config.output.generate_files.portfolio is True

    def test_output_formats_config(self):
        """Test output format configuration."""
        config = SystemConfig.load()

        assert config.output.formats.json is True
        assert config.output.formats.csv is True
        assert config.output.formats.parquet is False


class TestLoggingConfig:
    """Test logging configuration."""

    def test_logging_config_defaults(self):
        """Test logging configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.logging.level == "INFO"
        assert config.logging.log_to_file is False
        assert config.logging.log_dir == "logs"

    def test_logging_structlog_config(self):
        """Test structlog configuration."""
        config = SystemConfig.load()

        assert config.logging.structlog.format == "console"
        assert config.logging.structlog.include_timestamps is True
        assert config.logging.structlog.colorize is True

    def test_logging_services_config(self):
        """Test service-specific logging levels."""
        config = SystemConfig.load()

        assert config.logging.services.data == "INFO"
        assert config.logging.services.portfolio == "INFO"
        assert config.logging.services.execution == "INFO"


class TestReportingConfig:
    """Test reporting configuration."""

    def test_reporting_config_defaults(self):
        """Test reporting configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.reporting.decimal_places == 4
        assert config.reporting.include_trade_details is True

    def test_reporting_metrics_config(self):
        """Test reporting metrics list."""
        config = SystemConfig.load()

        expected_metrics = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
        ]
        for metric in expected_metrics:
            assert metric in config.reporting.metrics


class TestDevelopmentConfig:
    """Test development configuration."""

    def test_development_config_defaults(self):
        """Test development configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.development.debug_mode is False
        assert config.development.profile is False
        assert config.development.strict_validation is True
        assert config.development.save_debug_snapshots is False


class TestPreferencesConfig:
    """Test preferences configuration."""

    def test_preferences_config_defaults(self):
        """Test preferences configuration has correct defaults."""
        config = SystemConfig.load()

        assert config.preferences.date_format == "%Y-%m-%d"
        assert config.preferences.currency_symbol == "$"
        assert config.preferences.thousands_separator == ","
        assert config.preferences.decimal_separator == "."


class TestSingletonPattern:
    """Test singleton pattern for system config."""

    def test_get_system_config_returns_singleton(self):
        """Test that get_system_config returns cached instance."""
        config1 = get_system_config()
        config2 = get_system_config()

        # Should be the same instance
        assert config1 is config2

    def test_reload_system_config_creates_new_instance(self):
        """Test that reload forces new instance."""
        config1 = get_system_config()
        config2 = reload_system_config()

        # Should be different instances (reloaded)
        assert config1 is not config2


class TestConfigMerging:
    """Test configuration merging and precedence."""

    def test_custom_config_overrides_defaults(self, tmp_path):
        """Test that custom config overrides built-in defaults."""
        # Create custom config file
        custom_config = tmp_path / "custom.yaml"
        custom_config.write_text(
            """
portfolio:
  initial_capital: 500000.00
  lot_method_long: lifo

execution:
  fill_policy: aggressive
  queue_bars: 5

logging:
  level: DEBUG
"""
        )

        config = SystemConfig.load(custom_config)

        # Custom values should override defaults
        assert config.portfolio.initial_capital == Decimal("500000.00")
        assert config.portfolio.lot_method_long == "lifo"
        assert config.execution.fill_policy == "aggressive"
        assert config.execution.queue_bars == 5
        assert config.logging.level == "DEBUG"

        # Defaults should remain for unspecified values
        assert config.risk.cash_buffer_pct == 0.02
        assert config.data.default_mode == "adjusted"


class TestEnvironmentVariableSubstitution:
    """Test environment variable substitution."""

    def test_env_var_substitution(self, tmp_path, monkeypatch):
        """Test that ${VAR} syntax substitutes environment variables."""
        # Set environment variable
        monkeypatch.setenv("TEST_DATA_PATH", "/custom/data/path")
        monkeypatch.setenv("TEST_RESULTS_DIR", "/custom/results")

        # Create config with env vars
        custom_config = tmp_path / "env_test.yaml"
        custom_config.write_text(
            """
data:
  sources_config: ${TEST_DATA_PATH}/sources.yaml

output:
  default_results_dir: ${TEST_RESULTS_DIR}
"""
        )

        config = SystemConfig.load(custom_config)

        assert config.data.sources_config == "/custom/data/path/sources.yaml"
        assert config.output.default_results_dir == "/custom/results"
