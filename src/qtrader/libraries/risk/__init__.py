"""
Risk Policy Library.

All risk policies must inherit from BaseRiskPolicy and implement:
- evaluate_signal(): Evaluate signal against portfolio state
- calculate_position_size(): Calculate number of shares
- batch_evaluate(): Evaluate multiple signals (optional, for netting)
"""

from qtrader.libraries.risk.base import BaseRiskPolicy, OrderDecision, PortfolioState

__all__ = [
    "BaseRiskPolicy",
    "OrderDecision",
    "PortfolioState",
]
