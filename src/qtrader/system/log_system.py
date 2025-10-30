"""Centralized logging configuration for QTrader."""

import logging
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(BaseModel):
    """Configuration for logging system.

    Logging Levels Guide (User-Centric):

    INFO (Default - User-Facing):
    - Backtest started/completed
    - Universe loaded (summary)
    - Strategy signals
    - Orders placed/filled
    - Performance milestones
    - High-level summaries

    DEBUG (Developer Mode):
    - Service initialization details
    - EventBus subscriptions
    - Data loading per-symbol
    - Internal operations
    - Adapter details

    WARNING:
    - Recoverable issues (missing symbols, partial data)
    - Strategy skipped signals

    ERROR:
    - Order rejections
    - Data loading failures
    - Failures affecting results

    CRITICAL:
    - System-level failures

    Recommended Usage:
    - Normal backtesting: level="INFO" (clean, user-friendly output)
    - Development/debugging: level="DEBUG" (verbose, all operations)
    - Production monitoring: level="WARNING" (issues only)

    Timestamp Format Options:
    - "iso": 2025-10-22T20:50:07.288824Z (full ISO format)
    - "compact": 251022-205007.28 (YYMMDD-HHMMSS.ms) - recommended
    - "time": 20:50:07.28 (time only, good for same-day logs)
    - "short": 1022T205007 (MMDDTHHMMSS, very compact)
    """

    level: LogLevel = Field(
        default="INFO",
        description="Minimum log level (INFO=user-friendly, DEBUG=verbose, WARNING=issues only)",
    )
    format: Literal["console", "json"] = Field(
        default="console",
        description="Output format: console, or json",
    )
    timestamp_format: Literal["iso", "compact", "time", "short"] = Field(
        default="compact",
        description="Timestamp format for console output",
    )
    enable_file: bool = Field(
        default=True,
        description="Enable logging to file (WARNING and above by default)",
    )
    file_path: Path | None = Field(
        default=None,
        description="Path to log file (uses logs/qtrader.log if None)",
    )
    file_level: LogLevel = Field(
        default="WARNING",
        description="Minimum log level for file output",
    )
    file_rotation: bool = Field(
        default=True,
        description="Enable log file rotation (when file gets too large)",
    )
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum log file size in MB before rotation",
    )
    backup_count: int = Field(
        default=3,
        description="Number of rotated log files to keep",
    )
    console_width: int = Field(
        default=0,
        description="Maximum console line width (0 = no limit)",
    )


class LoggerFactory:
    """
    Factory for creating and configuring structured loggers.

    Provides centralized configuration for all QTrader logging.
    Call configure() once at application startup, then use get_logger()
    to get configured logger instances throughout the codebase.

    Example:
        # At startup
        config = LoggingConfig(level="DEBUG", enable_file=True, file_path=Path("qtrader.log"))
        LoggerFactory.configure(config)

        # In modules
        logger = LoggerFactory.get_logger()
        logger.info("trading.order_placed", symbol="AAPL", quantity=100)
    """

    _config: LoggingConfig | None = None
    _configured: bool = False

    @classmethod
    def configure(cls, config: LoggingConfig | None = None) -> None:
        """
        Configure the logging system.

        Should be called once at application startup before any logging occurs.

        Args:
            config: LoggingConfig instance. If None, uses default configuration.
        """
        if config is None:
            config = LoggingConfig()

        cls._config = config

        processors = cls._build_common_processors(config.timestamp_format)

        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setLevel(getattr(logging, config.level))
        console_processor: Any
        if config.format == "console":
            console_processor = cls._custom_console_renderer()
        else:
            console_processor = structlog.processors.JSONRenderer()
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=console_processor,
                foreign_pre_chain=processors,
            )
        )

        handlers: list[logging.Handler] = [console_handler]
        root_level = getattr(logging, config.level)

        # Configure file logging if enabled
        if config.enable_file:
            # Use default if file_path not provided
            if config.file_path is None:
                config.file_path = Path("logs/qtrader.log")

            file_handler = cls._configure_file_logging(config, processors)
            handlers.append(file_handler)
            root_level = min(root_level, getattr(logging, config.file_level))

        logging.basicConfig(level=root_level, handlers=handlers, force=True)

        # Configure structlog
        configured_processors = list(processors)
        if config.format == "console":
            configured_processors.extend(
                [
                    structlog.dev.set_exc_info,
                    structlog.processors.ExceptionRenderer(
                        structlog.dev.plain_traceback,  # type: ignore[arg-type]
                    ),
                ]
            )
        else:
            configured_processors.append(structlog.processors.format_exc_info)
        configured_processors.append(structlog.stdlib.ProcessorFormatter.wrap_for_formatter)

        structlog.configure(
            processors=configured_processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        cls._configured = True

    @classmethod
    def _build_common_processors(cls, timestamp_format: str) -> list[Any]:
        """Processors shared by both structlog and stdlib handlers before rendering."""
        return [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            cls._get_timestamper(timestamp_format),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
        ]

    @staticmethod
    def _get_timestamper(fmt: str) -> Any:
        """Get appropriate timestamper based on format with milliseconds."""

        def add_timestamp_processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
            """Add formatted timestamp with milliseconds."""
            now = datetime.now(timezone.utc)
            ms = now.microsecond // 10000  # Get 2-digit milliseconds

            if fmt == "iso":
                event_dict["timestamp"] = now.isoformat()
            elif fmt == "compact":
                # YYMMDD-HHMMSS.ms format
                event_dict["timestamp"] = now.strftime(f"%y%m%d-%H%M%S.{ms:02d}")
            elif fmt == "time":
                # Just time with milliseconds
                event_dict["timestamp"] = now.strftime(f"%H:%M:%S.{ms:02d}")
            elif fmt == "short":
                # MMDDTHHMMSS format (no separators, no ms for brevity)
                event_dict["timestamp"] = now.strftime("%m%dT%H%M%S")
            else:
                event_dict["timestamp"] = now.isoformat()

            return event_dict

        return add_timestamp_processor

    @staticmethod
    def _custom_console_renderer():
        """Custom console renderer with file:line info."""

        def renderer(logger: Any, name: str, event_dict: dict[str, Any]) -> str:
            """Render log with timestamp, level, message, and location."""
            # Extract components
            timestamp = event_dict.pop("timestamp", "")
            level = event_dict.pop("level", "info").upper()
            event = event_dict.pop("event", "")
            filename = event_dict.pop("filename", "")
            lineno = event_dict.pop("lineno", "")
            logger_name = event_dict.pop("logger", "")

            # Color codes
            colors = {
                "DEBUG": "\033[36m",  # Cyan
                "INFO": "\033[32m",  # Green
                "WARNING": "\033[33m",  # Yellow
                "ERROR": "\033[31m",  # Red
                "CRITICAL": "\033[35m",  # Magenta
            }
            reset = "\033[0m"
            gray = "\033[90m"

            # Format level with color
            level_color = colors.get(level, "")
            level_str = f"[{level_color}{level.lower()}{reset}]"

            # Format context (key=value pairs)
            context_parts = []
            for key, value in sorted(event_dict.items()):
                if key.startswith("_"):
                    continue
                context_parts.append(f"{key}={value}")

            context_str = " ".join(context_parts) if context_parts else ""

            # Format location (module.path:lineno)
            if filename and lineno:
                # Remove .py extension
                module_file = Path(filename).stem
                # Combine logger name with filename for full path
                if logger_name and logger_name != "qtrader":
                    location = f"{gray}({logger_name}.{module_file}:{lineno}){reset}"
                else:
                    location = f"{gray}({module_file}:{lineno}){reset}"
            else:
                location = ""

            # Build line: timestamp level event | context (location)
            parts = [timestamp, level_str, event]

            if context_str:
                parts.append(f"{gray}|{reset} {context_str}")

            if location:
                parts.append(location)

            return " ".join(parts)

        return renderer

    @classmethod
    def _configure_file_logging(cls, config: LoggingConfig, pre_chain: list[Any]) -> logging.Handler:
        """Configure file output for logging."""
        file_path = config.file_path
        assert file_path is not None  # Already validated in configure()

        # Create log directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure file handler
        handler: logging.Handler
        if config.file_rotation:
            handler = RotatingFileHandler(
                filename=str(file_path),
                maxBytes=config.max_file_size_mb * 1024 * 1024,
                backupCount=config.backup_count,
                encoding="utf-8",
            )
        else:
            handler = logging.FileHandler(
                filename=str(file_path),
                encoding="utf-8",
            )

        # Set file handler level to the configured file_level
        handler.setLevel(getattr(logging, config.file_level))

        handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(),
                foreign_pre_chain=pre_chain,
            )
        )

        return handler

    @classmethod
    def get_logger(cls, name: str | None = None):
        """
        Get a configured logger instance.

        Args:
            name: Optional logger name. If None, uses the calling module's __name__.

        Returns:
            Configured structlog BoundLogger instance.
        """
        if not cls._configured:
            # Auto-configure with defaults if not explicitly configured
            cls.configure()

        if name is None:
            # Get caller's module name
            import inspect

            frame = inspect.currentframe()
            if frame and frame.f_back:
                name = frame.f_back.f_globals.get("__name__", "qtrader")
            else:
                name = "qtrader"

        return structlog.get_logger(name)

    @classmethod
    def get_config(cls) -> LoggingConfig:
        """Get current logging configuration."""
        if cls._config is None:
            return LoggingConfig()
        return cls._config

    @classmethod
    def is_configured(cls) -> bool:
        """Check if logging has been configured."""
        return cls._configured

    @classmethod
    def reset(cls) -> None:
        """Reset logging configuration (mainly for testing)."""
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
        root_logger.handlers.clear()
        root_logger.setLevel(logging.NOTSET)
        cls._config = None
        cls._configured = False
        # Reset structlog to defaults
        structlog.reset_defaults()
