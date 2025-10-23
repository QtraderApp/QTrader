"""
Performance Metrics Library.

All metrics must inherit from BaseMetric and implement:
- compute(): Calculate metric from backtest results
- name: Metric identifier (snake_case)
- display_name: Human-readable name
"""

from qtrader.libraries.performance.base import BacktestResult, BaseMetric

__all__ = [
    "BaseMetric",
    "BacktestResult",
]
