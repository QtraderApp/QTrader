"""
Risk Service Interface.

Defines the protocol (interface) for risk management service.
Follows LEGO architecture principles:
- Pure event-driven (no direct API calls)
- Immutable inputs
- No state mutation (publishes events instead)
"""

from typing import Any, Protocol

from qtrader.events.events import RiskEvaluationTriggerEvent, SignalEvent

# Note: PortfolioStateEvent will be added when implementing PortfolioService Phase 2 integration
# For now, RiskService will work with PortfolioState model snapshots


class IRiskService(Protocol):
    """
    Risk service interface (MVP).

    Responsibilities:
    - Subscribe to SignalEvent, PortfolioStateEvent
    - Allocate capital across strategies (fixed risk budgets)
    - Size positions per signal (fixed_fraction model)
    - Enforce portfolio limits (concentration, leverage)
    - Publish OrderApprovedEvent or OrderRejectedEvent per signal

    Does NOT:
    - Execute orders (ExecutionService responsibility)
    - Mutate portfolio (PortfolioService responsibility)
    - Load market data (DataService responsibility)
    - Make strategy decisions (Strategy responsibility)

    Event Flow:
        1. Strategy → SignalEvent
        2. RiskService buffers signals (on_signal)
        3. BacktestEngine → RiskEvaluationTriggerEvent
        4. RiskService evaluates batch (on_risk_evaluation_trigger)
        5. RiskService → OrderApprovedEvent | OrderRejectedEvent

    Pure Functions:
        All methods are pure - same inputs produce same outputs.
        No hidden state or side effects except event publishing.

    Example:
        >>> from qtrader.events.event_bus import EventBus
        >>> from qtrader.services.risk.service import RiskService
        >>> bus = EventBus()
        >>> risk_service = RiskService(config, bus)
        >>> bus.subscribe(SignalEvent, risk_service.on_signal)
        >>> bus.subscribe(RiskEvaluationTriggerEvent, risk_service.on_risk_evaluation_trigger)
        >>> bus.subscribe(PortfolioStateEvent, risk_service.on_portfolio_state)
    """

    def on_signal(self, event: SignalEvent) -> None:
        """
        Handle incoming trading signal.

        Buffers signal for batch evaluation. Signals are processed
        in batches to enable cross-signal logic (e.g., leverage checks
        across all pending orders).

        Args:
            event: Trading signal from strategy

        Side Effects:
            - Buffers signal internally
            - Logs signal receipt (DEBUG level)

        Validation:
            - Ensures event.ts matches current bar (timestamp consistency)
            - Validates signal strength in [-1, 1]

        Example:
            >>> signal = SignalEvent(
            ...     ts=datetime(2020, 1, 2, 16, 0),
            ...     strategy_id="momentum_v1",
            ...     symbol="AAPL",
            ...     side="BUY",
            ...     strength=0.75
            ... )
            >>> risk_service.on_signal(signal)  # Buffered for batch eval
        """
        ...

    def on_risk_evaluation_trigger(self, event: RiskEvaluationTriggerEvent) -> None:
        """
        Evaluate buffered signals and publish orders/rejections.

        Triggered by BacktestEngine at end of bar. Processes all buffered
        signals in batch:
        1. Allocate capital per strategy (fixed budgets)
        2. Size each signal (fixed_fraction)
        3. Check concentration limits (per-symbol)
        4. Check leverage limits (portfolio-wide)
        5. Publish OrderApprovedEvent or OrderRejectedEvent
        6. Clear signal buffer

        Args:
            event: Trigger event with evaluation timestamp

        Side Effects:
            - Publishes OrderApprovedEvent per approved signal
            - Publishes OrderRejectedEvent per rejected signal
            - Clears signal buffer
            - Logs allocation, approvals, rejections (INFO/WARNING)

        Validation:
            - Ensures event.ts matches buffered signal timestamps
            - Validates portfolio state is cached

        Example:
            >>> trigger = RiskEvaluationTriggerEvent(
            ...     ts=datetime(2020, 1, 2, 16, 0)
            ... )
            >>> risk_service.on_risk_evaluation_trigger(trigger)
            # Publishes: OrderApprovedEvent(symbol="AAPL", quantity=500, ...)
            # Publishes: OrderRejectedEvent(symbol="TSLA", reason="Exceeds concentration", ...)
        """
        ...

    def on_portfolio_state(self, event: Any) -> None:
        """
        Update cached portfolio state.

        Caches latest portfolio snapshot for risk checks. RiskService
        never mutates this state - it's read-only input.

        Args:
            event: Portfolio state snapshot from PortfolioService
                  (Type will be PortfolioStateEvent once defined in Phase 2 integration)

        Side Effects:
            - Updates cached portfolio state
            - Logs state update (DEBUG level)

        Validation:
            - Ensures event.ts is not stale (monotonic timestamp)

        Note:
            PortfolioStateEvent will be defined when integrating with
            PortfolioService (Phase 2). For now using Any type.

        Example:
            >>> # Pseudo-code until PortfolioStateEvent exists
            >>> from qtrader.services.risk.models import PortfolioState
            >>> state_data = PortfolioState(
            ...     ts=datetime(2020, 1, 2, 16, 0),
            ...     equity=Decimal("1000000"),
            ...     cash=Decimal("500000"),
            ...     ...
            ... )
            >>> risk_service.on_portfolio_state(state_data)  # Cached for next batch
        """
        ...
