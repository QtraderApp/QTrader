"""Execution service for order simulation and fill generation.

This module provides realistic order execution simulation for backtesting.
ExecutionService accepts orders and returns fills without modifying portfolio state,
maintaining clean separation of concerns.

Public API:
    - IExecutionService: Protocol defining the execution service interface
    - ExecutionService: Main implementation
    - Order: Order model with state tracking
    - Fill: Immutable fill result
    - OrderState: Order state enum
    - OrderSide: Buy/Sell enum
    - OrderType: Market/Limit/Stop/MOC enum
    - TimeInForce: DAY/GTC/IOC/FOK enum
    - ExecutionConfig: Configuration for execution service
    - SlippageModel: Slippage model enum
    - ISlippageCalculator: Slippage calculator interface
    - SlippageCalculatorFactory: Factory for creating slippage calculators
"""

from qtrader.services.execution.config import ExecutionConfig, SlippageConfig
from qtrader.services.execution.interface import IExecutionService
from qtrader.services.execution.models import Fill, FillDecision, Order, OrderSide, OrderState, OrderType, TimeInForce
from qtrader.services.execution.service import ExecutionService
from qtrader.services.execution.slippage import ISlippageCalculator, SlippageCalculatorFactory, SlippageModel

__all__ = [
    "ExecutionConfig",
    "ExecutionService",
    "Fill",
    "FillDecision",
    "IExecutionService",
    "ISlippageCalculator",
    "Order",
    "OrderSide",
    "OrderState",
    "OrderType",
    "SlippageCalculatorFactory",
    "SlippageConfig",
    "SlippageModel",
    "TimeInForce",
]
