"""
Strategy Service.

Orchestrates multiple external strategy instances, loads strategy files,
and routes events to each strategy.
"""

from qtrader.services.strategy.interface import IStrategyService
from qtrader.services.strategy.service import StrategyService

__all__ = [
    "IStrategyService",
    "StrategyService",
]
