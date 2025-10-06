"""
System configuration management for QTrader.

Loads system-level configuration from YAML files with precedence:
1. CLI flags (handled by CLI code)
2. Environment variables (${VAR_NAME} substitution)
3. ./config/qtrader.yaml (project-relative)
4. ~/.qtrader/config.yaml (user home)
5. Built-in defaults
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class OutputConfig:
    """Output and results configuration."""

    default_results_dir: str = "backtest_results"
    use_timestamps: bool = True
    timestamp_format: str = "%Y%m%d_%H%M%S"
    generate_files: Dict[str, bool] = field(
        default_factory=lambda: {
            "metadata": True,
            "trades": True,
            "fills": True,
            "portfolio": True,
            "positions": True,
            "equity_curve": True,
        }
    )
    organize_by_date: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    log_to_file: bool = False
    log_dir: str = "logs"
    log_file_pattern: str = "qtrader_{timestamp}.log"
    structlog_format: str = "console"
    include_timestamps: bool = True
    colorize: bool = True


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

    market_order_bps: int = 5
    limit_order_bps: int = 0
    stop_order_bps: int = 5
    mode: str = "conservative"


@dataclass
class FillPolicyConfig:
    """Fill policy configuration."""

    limit_mode: str = "conservative"
    stop_mode: str = "conservative"
    moc_slip_bps: int = 5


@dataclass
class ExecutionConfig:
    """Execution engine configuration."""

    commission: CommissionConfig = field(default_factory=CommissionConfig)
    slippage: SlippageConfig = field(default_factory=SlippageConfig)
    fill_policy: FillPolicyConfig = field(default_factory=FillPolicyConfig)
    queue_bars: int = 3


@dataclass
class DataConfig:
    """Data loading and management configuration."""

    sources_config: str = "config/data_sources.yaml"
    default_mode: str = "adjusted"
    default_timezone: str = "America/New_York"
    price_decimals: int = 4
    validate_on_load: bool = True
    cache_enabled: bool = False
    cache_dir: str = ".cache/qtrader"
    cache_max_size_mb: int = 1000


@dataclass
class ReportingConfig:
    """Reporting configuration."""

    metrics: List[str] = field(
        default_factory=lambda: [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "win_rate",
            "profit_factor",
            "avg_trade_duration",
            "calmar_ratio",
            "sortino_ratio",
        ]
    )
    format_json: bool = True
    format_csv: bool = True
    format_html: bool = False
    format_pdf: bool = False
    decimal_places: int = 2
    include_trade_details: bool = True


@dataclass
class SystemConfig:
    """Complete system configuration."""

    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "SystemConfig":
        """
        Load system configuration with precedence.

        Args:
            config_path: Explicit path to config file (overrides search)

        Returns:
            SystemConfig instance with loaded settings

        Configuration search order:
        1. Explicit config_path if provided
        2. ./config/qtrader.yaml (project-relative)
        3. ~/.qtrader/config.yaml (user home)
        4. Built-in defaults (if no files found)
        """
        # Start with defaults
        config_dict: Dict[str, Any] = {}

        # Search for config files
        search_paths = []

        if config_path:
            search_paths.append(config_path)
        else:
            # Project-relative
            project_config = Path("config/qtrader.yaml")
            if project_config.exists():
                search_paths.append(project_config)

            # User home
            home_config = Path.home() / ".qtrader" / "config.yaml"
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

        # Build config objects from dict
        return cls._from_dict(config_dict)

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> "SystemConfig":
        """Build SystemConfig from nested dictionary."""
        # Output
        output_data = config_dict.get("output", {})
        output = OutputConfig(
            default_results_dir=output_data.get("default_results_dir", "backtest_results"),
            use_timestamps=output_data.get("use_timestamps", True),
            timestamp_format=output_data.get("timestamp_format", "%Y%m%d_%H%M%S"),
            generate_files=output_data.get("generate_files", OutputConfig().generate_files),
            organize_by_date=output_data.get("organize_by_date", False),
        )

        # Logging
        logging_data = config_dict.get("logging", {})
        structlog_data = logging_data.get("structlog", {})
        logging = LoggingConfig(
            level=logging_data.get("level", "INFO"),
            log_to_file=logging_data.get("log_to_file", False),
            log_dir=logging_data.get("log_dir", "logs"),
            log_file_pattern=logging_data.get("log_file_pattern", "qtrader_{timestamp}.log"),
            structlog_format=structlog_data.get("format", "console"),
            include_timestamps=structlog_data.get("include_timestamps", True),
            colorize=structlog_data.get("colorize", True),
        )

        # Execution
        exec_data = config_dict.get("execution", {})
        commission_data = exec_data.get("commission", {})
        slippage_data = exec_data.get("slippage", {})
        fill_policy_data = exec_data.get("fill_policy", {})

        commission = CommissionConfig(
            model=commission_data.get("model", "per_share"),
            per_share=commission_data.get("per_share", 0.0005),
            minimum=commission_data.get("minimum", 1.00),
            maximum=commission_data.get("maximum"),
        )

        slippage = SlippageConfig(
            market_order_bps=slippage_data.get("market_order_bps", 5),
            limit_order_bps=slippage_data.get("limit_order_bps", 0),
            stop_order_bps=slippage_data.get("stop_order_bps", 5),
            mode=slippage_data.get("mode", "conservative"),
        )

        fill_policy = FillPolicyConfig(
            limit_mode=fill_policy_data.get("limit_mode", "conservative"),
            stop_mode=fill_policy_data.get("stop_mode", "conservative"),
            moc_slip_bps=fill_policy_data.get("moc_slip_bps", 5),
        )

        execution = ExecutionConfig(
            commission=commission,
            slippage=slippage,
            fill_policy=fill_policy,
            queue_bars=exec_data.get("queue_bars", 3),
        )

        # Data
        data_dict = config_dict.get("data", {})
        cache_data = data_dict.get("cache", {})
        data = DataConfig(
            sources_config=data_dict.get("sources_config", "config/data_sources.yaml"),
            default_mode=data_dict.get("default_mode", "adjusted"),
            default_timezone=data_dict.get("default_timezone", "America/New_York"),
            price_decimals=data_dict.get("price_decimals", 4),
            validate_on_load=data_dict.get("validate_on_load", True),
            cache_enabled=cache_data.get("enabled", False),
            cache_dir=cache_data.get("cache_dir", ".cache/qtrader"),
            cache_max_size_mb=cache_data.get("max_size_mb", 1000),
        )

        # Reporting
        reporting_dict = config_dict.get("reporting", {})
        formats_data = reporting_dict.get("formats", {})

        reporting = ReportingConfig(
            metrics=reporting_dict.get("metrics", ReportingConfig().metrics),
            format_json=formats_data.get("json", True),
            format_csv=formats_data.get("csv", True),
            format_html=formats_data.get("html", False),
            format_pdf=formats_data.get("pdf", False),
            decimal_places=reporting_dict.get("decimal_places", 2),
            include_trade_details=reporting_dict.get("include_trade_details", True),
        )

        return cls(
            output=output,
            logging=logging,
            execution=execution,
            data=data,
            reporting=reporting,
        )


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
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
    if isinstance(config, dict):
        return {k: _substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [_substitute_env_vars(item) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR_NAME} with environment variable
        import re

        pattern = r"\$\{([^}]+)\}"

        def replace_var(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace_var, config)
    else:
        return config


# Global config instance (loaded on first access)
_config: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """
    Get the system configuration singleton.

    Loads config on first access and caches for subsequent calls.

    Returns:
        SystemConfig instance
    """
    global _config
    if _config is None:
        _config = SystemConfig.load()
    return _config


def reload_config(config_path: Optional[Path] = None) -> SystemConfig:
    """
    Reload system configuration from files.

    Args:
        config_path: Optional explicit path to config file

    Returns:
        Reloaded SystemConfig instance
    """
    global _config
    _config = SystemConfig.load(config_path)
    return _config
