"""
Event infrastructure for QTrader event-driven architecture.

This module provides the event system for loose coupling between services:
- Event classes: Immutable events representing facts that occurred
- EventBus: Publish/subscribe infrastructure for event distribution

Events enable:
- Services to communicate without direct dependencies
- Multiple consumers per event (one-to-many)
- Audit trail and replay capability
- Deterministic backtesting
"""

from qtrader.events.event_bus import EventBus, IEventBus
from qtrader.events.events import (
    BacktestEndedEvent,
    BacktestStartedEvent,
    BarCloseEvent,
    CashChangedEvent,
    CorporateActionEvent,
    Event,
    FillEvent,
    MarketDataEvent,
    OrderEvent,
    PortfolioStateEvent,
    PositionChangedEvent,
    PriceBarEvent,
    RiskEvaluationTriggerEvent,
    RiskViolationEvent,
    SignalEvent,
    ValuationTriggerEvent,
)

__all__ = [
    # Base
    "Event",
    # Market Data
    "MarketDataEvent",
    "PriceBarEvent",
    "CorporateActionEvent",
    # Trading
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    # Portfolio
    "PositionChangedEvent",
    "CashChangedEvent",
    "PortfolioStateEvent",
    # Risk
    "RiskViolationEvent",
    "RiskEvaluationTriggerEvent",
    "ValuationTriggerEvent",
    # Backtest Control
    "BacktestStartedEvent",
    "BacktestEndedEvent",
    "BarCloseEvent",
    # EventBus
    "IEventBus",
    "EventBus",
]
