"""Centralized logging configuration for QTrader."""

import logging
import sys
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LoggingConfig(BaseModel):
    """Configuration for logging system."""

    level: LogLevel = Field(
        default="INFO",
        description="Minimum log level to capture",
    )
    format: Literal["console", "json"] = Field(
        default="console",
        description="Output format: human-readable console or structured JSON",
    )
    enable_file: bool = Field(
        default=False,
        description="Enable logging to file",
    )
    file_path: Path | None = Field(
        default=None,
        description="Path to log file (required if enable_file=True)",
    )
    file_level: LogLevel = Field(
        default="DEBUG",
        description="Minimum log level for file output",
    )
    file_rotation: bool = Field(
        default=True,
        description="Enable log file rotation (when file gets too large)",
    )
    max_file_size_mb: int = Field(
        default=100,
        description="Maximum log file size in MB before rotation",
    )
    backup_count: int = Field(
        default=5,
        description="Number of rotated log files to keep",
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
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
        ]

        # Add format-specific processors
        if config.format == "console":
            processors.extend(
                [
                    structlog.dev.set_exc_info,
                    structlog.processors.ExceptionRenderer(
                        structlog.dev.plain_traceback,  # type: ignore[arg-type]
                    ),
                    structlog.dev.ConsoleRenderer(colors=True),
                ]
            )
        else:  # json
            processors.extend(
                [
                    structlog.processors.format_exc_info,
                    structlog.processors.JSONRenderer(),
                ]
            )

        # Configure structlog
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        # Configure file logging if enabled
        if config.enable_file:
            if config.file_path is None:
                raise ValueError("file_path must be provided when enable_file=True")

            cls._configure_file_logging(config)

        cls._configured = True

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

        # Set file handler level - must be at DEBUG to capture all messages
        # The actual filtering happens at the logger level
        handler.setLevel(logging.DEBUG)

        # Use simple format for file output (structlog already formats the message)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(message)s",
            )
        )

        # Add handler to root logger
        root_logger = logging.getLogger()
        # Set root logger to DEBUG so file handler can capture all levels
        root_logger.setLevel(logging.DEBUG)
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
