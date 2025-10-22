"""
System Configuration Management.

Consolidated configuration for the entire QTrader system.
Philosophy: "In real life, the system is one"

This module provides THE system configuration - a single source of truth
for how QTrader operates. All backtests use this configuration for:
- Commission models
- Slippage models
- Data sources
- Risk limits
- Portfolio accounting
- Output settings
- Logging

What varies per backtest (NOT here):
- Start/end dates
- Universe (symbols)
- Initial capital
- Which strategies to run

Usage:
    >>> from qtrader.system import get_system_config
    >>> config = get_system_config()
    >>> print(config.execution.commission.per_share)
    0.0005
"""

import os
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class DataCacheConfig:
    """Data caching configuration."""

    enabled: bool = False
    cache_dir: str = ".cache/qtrader"
    max_size_mb: int = 1000


@dataclass
class DataConfig:
    """Data service configuration."""

    sources_config: str = "config/data_sources.yaml"
    default_mode: str = "adjusted"
    default_timezone: str = "America/New_York"
    price_decimals: int = 4
    validate_on_load: bool = True
    cache: DataCacheConfig = field(default_factory=DataCacheConfig)


@dataclass
class CommissionConfig:
    """Commission calculation configuration."""

    model: str = "per_share"
    per_share: float = 0.0005
    minimum: float = 1.00
    maximum: Optional[float] = None


@dataclass
class SlippageConfig:
    """Slippage model configuration."""

    model: str = "fixed"
    slippage_bps: float = 5.0


@dataclass
class PortfolioConfig:
    """Portfolio service configuration."""

    initial_capital: Decimal = Decimal("100000.00")
    lot_method_long: str = "fifo"
    lot_method_short: str = "lifo"
    max_ledger_entries: int = 10000
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)


@dataclass
class ExecutionSlippageConfig:
    """Execution slippage configuration."""

    market_order_bps: int = 5
    limit_order_bps: int = 0
    stop_order_bps: int = 5
    moc_slip_bps: int = 5
    mode: str = "conservative"


@dataclass
class ExecutionConfig:
    """Execution service configuration."""

    fill_policy: str = "conservative"
    slippage: ExecutionSlippageConfig = field(default_factory=ExecutionSlippageConfig)
    commission: CommissionConfig = field(default_factory=CommissionConfig)
    queue_bars: int = 3
    max_volume_participation: float = 0.10


@dataclass
class StrategyBudgetConfig:
    """Strategy budget allocation."""

    strategy_id: str
    capital_weight: float


@dataclass
class SizingConfigData:
    """Position sizing configuration."""

    model: str = "fixed_fraction"
    fraction: float = 0.20


@dataclass
class ConcentrationConfig:
    """Concentration limit configuration."""

    max_position_pct: float = 0.10
    max_sector_pct: float = 0.30


@dataclass
class LeverageConfig:
    """Leverage limit configuration."""

    max_gross: float = 1.0
    max_net: float = 1.0


@dataclass
class RiskChecksConfig:
    """Risk checks to perform."""

    position_size: bool = True
    leverage: bool = True
    concentration: bool = True
    cash_buffer: bool = True


@dataclass
class RiskConfig:
    """Risk service configuration."""

    cash_buffer_pct: float = 0.02
    budgets: list[StrategyBudgetConfig] = field(default_factory=list)
    sizing: dict[str, SizingConfigData] = field(default_factory=dict)
    concentration: ConcentrationConfig = field(default_factory=ConcentrationConfig)
    leverage: LeverageConfig = field(default_factory=LeverageConfig)
    checks: RiskChecksConfig = field(default_factory=RiskChecksConfig)


@dataclass
class StrategyWarmupConfig:
    """Strategy warmup configuration."""

    skip_signals: bool = True


@dataclass
class StrategyConfig:
    """Strategy service configuration."""

    validate_on_load: bool = True
    allow_concurrent: bool = True
    warmup: StrategyWarmupConfig = field(default_factory=StrategyWarmupConfig)


@dataclass
class OutputFilesConfig:
    """Output file generation configuration."""

    metadata: bool = True
    trades: bool = True
    fills: bool = True
    portfolio: bool = True
    positions: bool = True
    equity_curve: bool = True
    metrics: bool = True


@dataclass
class OutputFormatsConfig:
    """Output format configuration."""

    json: bool = True
    csv: bool = True
    parquet: bool = False


@dataclass
class OutputConfig:
    """Output and results configuration."""

    default_results_dir: str = "output/backtests"
    use_timestamps: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    organize_by_date: bool = False
    generate_files: OutputFilesConfig = field(default_factory=OutputFilesConfig)
    formats: OutputFormatsConfig = field(default_factory=OutputFormatsConfig)


@dataclass
class LoggingConfig:
    """Logging configuration (maps to log_system.LoggingConfig)."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["console", "json"] = "console"
    timestamp_format: Literal["iso", "compact", "time", "short"] = "compact"
    enable_file: bool = True
    file_path: str | None = None
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "WARNING"
    file_rotation: bool = True
    max_file_size_mb: int = 10
    backup_count: int = 3
    console_width: int = 0

    def to_logger_config(self):
        """Convert to log_system.LoggingConfig for LoggerFactory."""
        from pathlib import Path

        from qtrader.system.log_system import LoggingConfig as LogSystemConfig

        return LogSystemConfig(
            level=self.level,
            format=self.format,
            timestamp_format=self.timestamp_format,
            enable_file=self.enable_file,
            file_path=Path(self.file_path) if self.file_path else None,
            file_level=self.file_level,
            file_rotation=self.file_rotation,
            max_file_size_mb=self.max_file_size_mb,
            backup_count=self.backup_count,
            console_width=self.console_width,
        )


@dataclass
class ReportingConfig:
    """Reporting configuration."""

    metrics: list[str] = field(
        default_factory=lambda: [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "avg_trade_duration",
            "calmar_ratio",
            "sortino_ratio",
            "num_trades",
            "avg_win",
            "avg_loss",
        ]
    )
    decimal_places: int = 4
    include_trade_details: bool = True
    formats: OutputFormatsConfig = field(default_factory=OutputFormatsConfig)


@dataclass
class DevelopmentConfig:
    """Development and debug configuration."""

    debug_mode: bool = False
    profile: bool = False
    strict_validation: bool = True
    save_debug_snapshots: bool = False
    debug_snapshot_interval: int = 100


@dataclass
class PreferencesConfig:
    """User preference configuration."""

    date_format: str = "%Y-%m-%d"
    currency_symbol: str = "$"
    thousands_separator: str = ","
    decimal_separator: str = "."


@dataclass
class SystemConfig:
    """
    Complete system configuration for QTrader.

    This is THE system - a single source of truth for how QTrader operates.
    All backtests use this configuration to ensure consistent, fair comparisons.

    Philosophy: "In real life, the system is one"
    - ONE execution system with ONE commission model
    - ONE data source configuration
    - ONE risk management framework
    - ONE portfolio accounting method

    What's defined here (system-level):
        - Commission models
        - Slippage models
        - Data sources
        - Risk limits
        - Portfolio accounting
        - Logging/output settings

    What varies per backtest (NOT here):
        - Start/end dates
        - Universe (which symbols)
        - Initial capital
        - Which strategies to run

    Example:
        >>> config = SystemConfig.load()
        >>> print(config.execution.commission.per_share)
        0.0005
        >>> print(config.portfolio.lot_method_long)
        fifo
    """

    data: DataConfig = field(default_factory=DataConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)
    preferences: PreferencesConfig = field(default_factory=PreferencesConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "SystemConfig":
        """
        Load system configuration from YAML file.

        Configuration search order:
        1. Explicit config_path if provided
        2. ./config/system.yaml (project-relative)
        3. ./config/qtrader.yaml (legacy fallback)
        4. ~/.qtrader/system.yaml (user home)
        5. Built-in defaults (if no files found)

        Args:
            config_path: Explicit path to config file (overrides search)

        Returns:
            SystemConfig instance with loaded settings

        Example:
            >>> # Load from default location
            >>> config = SystemConfig.load()
            >>>
            >>> # Load from specific file
            >>> config = SystemConfig.load(Path("my_config.yaml"))
        """
        config_dict: dict[str, Any] = {}

        # Search for config files
        search_paths: list[Path] = []

        if config_path:
            search_paths.append(config_path)
        else:
            # Project-relative (new location)
            project_config = Path("config/system.yaml")
            if project_config.exists():
                search_paths.append(project_config)

            # Project-relative (legacy location)
            legacy_config = Path("config/qtrader.yaml")
            if legacy_config.exists():
                search_paths.append(legacy_config)

            # User home
            home_config = Path.home() / ".qtrader" / "system.yaml"
            if home_config.exists():
                search_paths.append(home_config)

        # Load config files (later files override earlier)
        for path in search_paths:
            if path.exists():
                with open(path) as f:
                    loaded = yaml.safe_load(f)
                    if loaded:
                        config_dict = _deep_merge(config_dict, loaded)

        # Substitute environment variables
        config_dict = _substitute_env_vars(config_dict)

        # Build config from dict
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: dict[str, Any]) -> "SystemConfig":
        """Build SystemConfig from nested dictionary."""
        # Data
        data_dict = config_dict.get("data", {})
        cache_dict = data_dict.get("cache", {})
        data = DataConfig(
            sources_config=data_dict.get("sources_config", "config/data_sources.yaml"),
            default_mode=data_dict.get("default_mode", "adjusted"),
            default_timezone=data_dict.get("default_timezone", "America/New_York"),
            price_decimals=data_dict.get("price_decimals", 4),
            validate_on_load=data_dict.get("validate_on_load", True),
            cache=DataCacheConfig(
                enabled=cache_dict.get("enabled", False),
                cache_dir=cache_dict.get("cache_dir", ".cache/qtrader"),
                max_size_mb=cache_dict.get("max_size_mb", 1000),
            ),
        )

        # Portfolio
        portfolio_dict = config_dict.get("portfolio", {})
        portfolio_commission_dict = portfolio_dict.get("commission", {})
        portfolio_slippage_dict = portfolio_dict.get("slippage", {})
        portfolio = PortfolioConfig(
            initial_capital=Decimal(str(portfolio_dict.get("initial_capital", "100000.00"))),
            lot_method_long=portfolio_dict.get("lot_method_long", "fifo"),
            lot_method_short=portfolio_dict.get("lot_method_short", "lifo"),
            max_ledger_entries=portfolio_dict.get("max_ledger_entries", 10000),
            commission=CommissionConfig(
                model=portfolio_commission_dict.get("model", "per_share"),
                per_share=portfolio_commission_dict.get("per_share", 0.0005),
                minimum=portfolio_commission_dict.get("minimum", 1.00),
                maximum=portfolio_commission_dict.get("maximum"),
            ),
            slippage=SlippageConfig(
                model=portfolio_slippage_dict.get("model", "fixed"),
                slippage_bps=portfolio_slippage_dict.get("slippage_bps", 5.0),
            ),
        )

        # Execution
        execution_dict = config_dict.get("execution", {})
        exec_slippage_dict = execution_dict.get("slippage", {})
        exec_commission_dict = execution_dict.get("commission", {})
        execution = ExecutionConfig(
            fill_policy=execution_dict.get("fill_policy", "conservative"),
            slippage=ExecutionSlippageConfig(
                market_order_bps=exec_slippage_dict.get("market_order_bps", 5),
                limit_order_bps=exec_slippage_dict.get("limit_order_bps", 0),
                stop_order_bps=exec_slippage_dict.get("stop_order_bps", 5),
                moc_slip_bps=exec_slippage_dict.get("moc_slip_bps", 5),
                mode=exec_slippage_dict.get("mode", "conservative"),
            ),
            commission=CommissionConfig(
                model=exec_commission_dict.get("model", "per_share"),
                per_share=exec_commission_dict.get("per_share", 0.0005),
                minimum=exec_commission_dict.get("minimum", 1.00),
                maximum=exec_commission_dict.get("maximum"),
            ),
            queue_bars=execution_dict.get("queue_bars", 3),
            max_volume_participation=execution_dict.get("max_volume_participation", 0.10),
        )

        # Risk
        risk_dict = config_dict.get("risk", {})
        budgets_list = risk_dict.get("budgets", [{"strategy_id": "default", "capital_weight": 1.0}])
        budgets = [
            StrategyBudgetConfig(strategy_id=b["strategy_id"], capital_weight=b["capital_weight"]) for b in budgets_list
        ]
        sizing_dict = risk_dict.get("sizing", {"default": {"model": "fixed_fraction", "fraction": 0.20}})
        sizing = {
            sid: SizingConfigData(model=cfg.get("model", "fixed_fraction"), fraction=cfg.get("fraction", 0.20))
            for sid, cfg in sizing_dict.items()
        }
        concentration_dict = risk_dict.get("concentration", {})
        leverage_dict = risk_dict.get("leverage", {})
        checks_dict = risk_dict.get("checks", {})
        risk = RiskConfig(
            cash_buffer_pct=risk_dict.get("cash_buffer_pct", 0.02),
            budgets=budgets,
            sizing=sizing,
            concentration=ConcentrationConfig(
                max_position_pct=concentration_dict.get("max_position_pct", 0.10),
                max_sector_pct=concentration_dict.get("max_sector_pct", 0.30),
            ),
            leverage=LeverageConfig(
                max_gross=leverage_dict.get("max_gross", 1.0),
                max_net=leverage_dict.get("max_net", 1.0),
            ),
            checks=RiskChecksConfig(
                position_size=checks_dict.get("position_size", True),
                leverage=checks_dict.get("leverage", True),
                concentration=checks_dict.get("concentration", True),
                cash_buffer=checks_dict.get("cash_buffer", True),
            ),
        )

        # Strategy
        strategy_dict = config_dict.get("strategy", {})
        warmup_dict = strategy_dict.get("warmup", {})
        strategy = StrategyConfig(
            validate_on_load=strategy_dict.get("validate_on_load", True),
            allow_concurrent=strategy_dict.get("allow_concurrent", True),
            warmup=StrategyWarmupConfig(skip_signals=warmup_dict.get("skip_signals", True)),
        )

        # Output
        output_dict = config_dict.get("output", {})
        gen_files_dict = output_dict.get("generate_files", {})
        output_formats_dict = output_dict.get("formats", {})
        output = OutputConfig(
            default_results_dir=output_dict.get("default_results_dir", "output/backtests"),
            use_timestamps=output_dict.get("use_timestamps", True),
            timestamp_format=output_dict.get("timestamp_format", "%Y%m%d_%H%M%S"),
            organize_by_date=output_dict.get("organize_by_date", False),
            generate_files=OutputFilesConfig(
                metadata=gen_files_dict.get("metadata", True),
                trades=gen_files_dict.get("trades", True),
                fills=gen_files_dict.get("fills", True),
                portfolio=gen_files_dict.get("portfolio", True),
                positions=gen_files_dict.get("positions", True),
                equity_curve=gen_files_dict.get("equity_curve", True),
                metrics=gen_files_dict.get("metrics", True),
            ),
            formats=OutputFormatsConfig(
                json=output_formats_dict.get("json", True),
                csv=output_formats_dict.get("csv", True),
                parquet=output_formats_dict.get("parquet", False),
            ),
        )

        # Logging
        logging_dict = config_dict.get("logging", {})
        logging = LoggingConfig(
            level=logging_dict.get("level", "INFO"),
            format=logging_dict.get("format", "console"),
            timestamp_format=logging_dict.get("timestamp_format", "compact"),
            enable_file=logging_dict.get("enable_file", True),
            file_path=logging_dict.get("file_path"),
            file_level=logging_dict.get("file_level", "WARNING"),
            file_rotation=logging_dict.get("file_rotation", True),
            max_file_size_mb=logging_dict.get("max_file_size_mb", 10),
            backup_count=logging_dict.get("backup_count", 3),
            console_width=logging_dict.get("console_width", 0),
        )

        # Reporting
        reporting_dict = config_dict.get("reporting", {})
        reporting_formats_dict = reporting_dict.get("formats", {})
        reporting = ReportingConfig(
            metrics=reporting_dict.get(
                "metrics",
                [
                    "total_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "win_rate",
                    "profit_factor",
                    "avg_trade_duration",
                    "calmar_ratio",
                    "sortino_ratio",
                    "num_trades",
                    "avg_win",
                    "avg_loss",
                ],
            ),
            decimal_places=reporting_dict.get("decimal_places", 4),
            include_trade_details=reporting_dict.get("include_trade_details", True),
            formats=OutputFormatsConfig(
                json=reporting_formats_dict.get("json", True),
                csv=reporting_formats_dict.get("csv", True),
                parquet=reporting_formats_dict.get("parquet", False),
            ),
        )

        # Development
        dev_dict = config_dict.get("development", {})
        development = DevelopmentConfig(
            debug_mode=dev_dict.get("debug_mode", False),
            profile=dev_dict.get("profile", False),
            strict_validation=dev_dict.get("strict_validation", True),
            save_debug_snapshots=dev_dict.get("save_debug_snapshots", False),
            debug_snapshot_interval=dev_dict.get("debug_snapshot_interval", 100),
        )

        # Preferences
        pref_dict = config_dict.get("preferences", {})
        preferences = PreferencesConfig(
            date_format=pref_dict.get("date_format", "%Y-%m-%d"),
            currency_symbol=pref_dict.get("currency_symbol", "$"),
            thousands_separator=pref_dict.get("thousands_separator", ","),
            decimal_separator=pref_dict.get("decimal_separator", "."),
        )

        return cls(
            data=data,
            portfolio=portfolio,
            execution=execution,
            risk=risk,
            strategy=strategy,
            output=output,
            logging=logging,
            reporting=reporting,
            development=development,
            preferences=preferences,
        )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def _substitute_env_vars(config: Any) -> Any:
    """Recursively substitute ${VAR_NAME} with environment variables."""
    import re

    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        pattern = r"\$\{([^}]+)\}"

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace_var, config)
    else:
        return config


# Global config instance (loaded on first access)
_config: Optional[SystemConfig] = None


def get_system_config(config_path: Optional[Path] = None) -> SystemConfig:
    """
    Get the system configuration singleton.

    Loads config on first access and caches for subsequent calls.
    Use reload_system_config() to force reload.

    Args:
        config_path: Optional explicit path to config file

    Returns:
        SystemConfig instance

    Example:
        >>> config = get_system_config()
        >>> print(config.execution.commission.per_share)
        0.0005
    """
    global _config
    if _config is None or config_path is not None:
        _config = SystemConfig.load(config_path)
    return _config


def reload_system_config(config_path: Optional[Path] = None) -> SystemConfig:
    """
    Reload system configuration from files.

    Forces a fresh load of configuration, clearing the cached singleton.

    Args:
        config_path: Optional explicit path to config file

    Returns:
        Reloaded SystemConfig instance

    Example:
        >>> # Modify config file
        >>> config = reload_system_config()
        >>> # Now uses new settings
    """
    global _config
    _config = SystemConfig.load(config_path)
    return _config
