"""Reporting service for performance metrics and analysis."""

from qtrader.services.reporting.config import ReportingConfig
from qtrader.services.reporting.formatters import display_performance_report
from qtrader.services.reporting.service import ReportingService
from qtrader.services.reporting.writers import (
    write_drawdowns_parquet,
    write_equity_curve_parquet,
    write_json_report,
    write_returns_parquet,
    write_trades_parquet,
)

__all__ = [
    "ReportingService",
    "ReportingConfig",
    "display_performance_report",
    "write_json_report",
    "write_equity_curve_parquet",
    "write_returns_parquet",
    "write_trades_parquet",
    "write_drawdowns_parquet",
]
