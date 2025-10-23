"""
RiskService: Event-driven risk management and order approval.

This module provides risk evaluation, position sizing, and limit checking
for trading signals before they are sent to ExecutionService.
"""

from qtrader.services.portfolio_manager.allocator import allocate_capital
from qtrader.services.portfolio_manager.config_loader import ConfigLoadError, load_risk_config
from qtrader.services.portfolio_manager.interface import IRiskService
from qtrader.services.portfolio_manager.limits import (
    LimitViolation,
    check_all_limits,
    check_concentration_limit,
    check_leverage_limits,
)
from qtrader.services.portfolio_manager.models import (
    ConcentrationLimit,
    LeverageLimit,
    OrderBase,
    PortfolioState,
    Position,
    RiskConfig,
    Signal,
    SizingConfig,
    StrategyBudget,
)
from qtrader.services.portfolio_manager.service import RiskService
from qtrader.services.portfolio_manager.sizer import FixedFractionSizer, size_position

__all__ = [
    # Service
    "RiskService",
    "IRiskService",
    # Models
    "Signal",
    "OrderBase",
    "Position",
    "PortfolioState",
    "StrategyBudget",
    "SizingConfig",
    "ConcentrationLimit",
    "LeverageLimit",
    "RiskConfig",
    # Configuration
    "load_risk_config",
    "ConfigLoadError",
    # Capital Allocation
    "allocate_capital",
    # Position Sizing
    "FixedFractionSizer",
    "size_position",
    # Limit Checking
    "LimitViolation",
    "check_concentration_limit",
    "check_leverage_limits",
    "check_all_limits",
]
