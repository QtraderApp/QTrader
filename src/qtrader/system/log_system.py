"""Centralized logging configuration for QTrader."""

import logging
import re
import sys
from datetime import datetime, timezone
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

        # Configure standard library logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=getattr(logging, config.level),
        )

        # Build structlog processors
        processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            cls._get_timestamper(config.timestamp_format),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                ]
            ),
        ]

        # Add format-specific processors
        if config.format == "console":
            processors.extend(
                [
                    structlog.dev.set_exc_info,
                    structlog.processors.ExceptionRenderer(
                        structlog.dev.plain_traceback,  # type: ignore[arg-type]
                    ),
                    cls._custom_console_renderer(),
                ]
            )
        else:  # json
            processors.extend(
                [
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ]
            )

        # Configure file logging if enabled (BEFORE structlog setup)
        # This way we can use ProcessorFormatter with a plain renderer
        if config.enable_file:
            # Use default path if not specified
            if config.file_path is None:
                config.file_path = Path("logs/qtrader.log")

            cls._configure_file_logging(config)

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        cls._configured = True

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
    def _configure_file_logging(cls, config: LoggingConfig) -> None:
        """Configure file output for logging."""
        file_path = config.file_path
        assert file_path is not None  # Already validated in configure()

        # Create log directory if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Configure file handler
        handler: logging.Handler
        if config.file_rotation:
            from logging.handlers import RotatingFileHandler

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

        # Use ANSI-stripping formatter for file logging so no color codes are written
        class AnsiStripFormatter(logging.Formatter):
            """Formatter that strips ANSI escape sequences from the message portion.

            Keeps the overall format (timestamp, level, message, module:lineno).
            """

            _ansi_re = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

            def format(self, record: logging.LogRecord) -> str:
                # Ensure message is a string
                try:
                    msg = record.getMessage()
                except Exception:
                    # Fallback to raw message
                    msg = str(record.msg)

                # Strip ANSI sequences
                clean = AnsiStripFormatter._ansi_re.sub("", msg)

                # Temporarily replace record.msg so standard formatting includes clean message
                orig_msg = record.msg
                record.msg = clean
                try:
                    out = super().format(record)
                finally:
                    record.msg = orig_msg

                return out

        handler.setFormatter(
            AnsiStripFormatter(
                fmt="%(asctime)s - %(levelname)s - %(message)s - %(name)s:%(lineno)d", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )

        # Add handler to root logger
        root_logger = logging.getLogger()
        # Set root logger to the minimum of console level and file level
        min_level = min(getattr(logging, config.level), getattr(logging, config.file_level))
        root_logger.setLevel(min_level)
        root_logger.addHandler(handler)

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
        cls._config = None
        cls._configured = False
        # Reset structlog to defaults
        structlog.reset_defaults()
