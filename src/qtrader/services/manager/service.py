"""
Risk Service Implementation.

Pure event-driven service for risk management:
- Buffers signals from strategies
- Caches portfolio state from PortfolioService
- Evaluates signals in batches (triggered by BacktestEngine)
- Publishes OrderApprovedEvent or OrderRejectedEvent

Phase 4 MVP: Fixed budgets, fixed_fraction sizing, concentration + leverage limits only.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any

from qtrader.events.event_bus import EventBus
from qtrader.events.events import OrderApprovedEvent, OrderRejectedEvent, RiskEvaluationTriggerEvent, SignalEvent
from qtrader.services.manager.allocator import allocate_capital
from qtrader.services.manager.limits import check_all_limits
from qtrader.services.manager.models import OrderBase, PortfolioState, RiskConfig, Signal
from qtrader.services.manager.sizer import FixedFractionSizer
from qtrader.system import LoggerFactory


class RiskService:
    """
    RiskService: Event-driven risk management.

    Evaluates trading signals, sizes positions, checks limits, and approves/rejects
    orders before sending to ExecutionService.
    """

    def __init__(self, config: RiskConfig, event_bus: EventBus) -> None:
        """
        Initialize RiskService.

        Args:
            config: Risk configuration (budgets, sizing, limits)
            event_bus: Event bus for subscribing/publishing

        Side Effects:
            - Creates logger (risk.service namespace)
            - Initializes empty signal buffer
            - Initializes None portfolio state cache
        """
        self._config = config
        self._event_bus = event_bus
        self._logger = LoggerFactory.get_logger("risk.service")

        # Signal buffer for batch evaluation
        self._signal_buffer: list[SignalEvent] = []

        # Cached portfolio state (latest snapshot)
        self._portfolio_state: PortfolioState | None = None

        # Current timestamp for consistency checks
        self._current_ts: datetime | None = None

        self._logger.info(
            "RiskService initialized",
            extra={
                "num_strategies": len(config.budgets),
                "total_budget_weight": sum(b.capital_weight for b in config.budgets),
                "concentration_limit": config.concentration.max_position_pct,
                "leverage_limit_gross": config.leverage.max_gross,
                "leverage_limit_net": config.leverage.max_net,
            },
        )

        # Subscribe to events
        self._event_bus.subscribe("signal", self.on_signal)  # type: ignore[arg-type]
        self._event_bus.subscribe("risk_evaluation_trigger", self.on_risk_evaluation_trigger)  # type: ignore[arg-type]

    @classmethod
    def from_config(cls, config_dict: dict[str, Any], event_bus: EventBus) -> "RiskService":
        """
        Factory method to create RiskService from configuration dictionary.

        Args:
            config_dict: Risk configuration section from backtest config
            event_bus: Event bus for service communication

        Returns:
            Configured RiskService instance

        Example:
            >>> config_dict = {...}  # Risk config from BacktestConfig
            >>> service = RiskService.from_config(config_dict, event_bus)
        """
        from qtrader.services.manager.models import ConcentrationLimit, LeverageLimit, SizingConfig, StrategyBudget

        # Convert BacktestConfig format to Phase 4 RiskConfig format
        budgets = [
            StrategyBudget(
                strategy_id=b["strategy_id"],
                capital_weight=b["capital_weight"],
            )
            for b in config_dict["budgets"]
        ]

        sizing = {
            strategy_id: SizingConfig(
                model="fixed_fraction",
                fraction=cfg["fraction"],
            )
            for strategy_id, cfg in config_dict["sizing"].items()
        }

        concentration = ConcentrationLimit(max_position_pct=config_dict["concentration"]["max_position_pct"])

        leverage = LeverageLimit(
            max_gross=config_dict["leverage"]["max_gross"],
            max_net=config_dict["leverage"]["max_net"],
        )

        cash_buffer_pct = config_dict.get("cash_buffer_pct", 0.02)

        config = RiskConfig(
            budgets=budgets,
            sizing=sizing,
            concentration=concentration,
            leverage=leverage,
            cash_buffer_pct=cash_buffer_pct,
        )

        return cls(config=config, event_bus=event_bus)

    def on_signal(self, event: SignalEvent) -> None:
        """
        Handle incoming trading signal.

        Buffers signal for batch evaluation. Validates timestamp consistency
        to ensure all events at timestamp ts are from same bar.

        Args:
            event: Trading signal from strategy

        Side Effects:
            - Buffers signal in _signal_buffer
            - Logs signal receipt (DEBUG)
            - Updates _current_ts if first signal of bar

        Raises:
            ValueError: If event.ts is inconsistent with current bar

        Example:
            >>> signal = SignalEvent(
            ...     ts=datetime(2020, 1, 2, 16, 0),
            ...     strategy_id="momentum_v1",
            ...     symbol="AAPL",
            ...     side="BUY",
            ...     strength=0.75
            ... )
            >>> risk_service.on_signal(signal)  # Buffered
        """
        # Timestamp consistency check
        if self._current_ts is None:
            # First signal of the bar
            self._current_ts = event.ts
        elif event.ts != self._current_ts:
            raise ValueError(
                f"Signal timestamp {event.ts} does not match current bar {self._current_ts}. "
                f"All signals must be from the same bar."
            )

        # Validate signal strength
        if not -1.0 <= event.strength <= 1.0:
            self._logger.warning(
                "Invalid signal strength, clamping to [-1, 1]",
                extra={
                    "strategy_id": event.strategy_id,
                    "symbol": event.symbol,
                    "strength": event.strength,
                },
            )
            # Note: We log but don't reject - SignalEvent validation should catch this

        # Buffer signal
        self._signal_buffer.append(event)

        self._logger.debug(
            "Signal received and buffered",
            extra={
                "ts": event.ts.isoformat(),
                "strategy_id": event.strategy_id,
                "symbol": event.symbol,
                "side": event.side,
                "strength": event.strength,
                "buffer_size": len(self._signal_buffer),
            },
        )

    def on_portfolio_state(self, event: Any) -> None:
        """
        Update cached portfolio state.

        Caches latest portfolio snapshot for risk checks. RiskService
        never mutates this state - it's read-only input.

        Args:
            event: Portfolio state snapshot (will be PortfolioStateEvent once defined)

        Side Effects:
            - Updates _portfolio_state cache
            - Logs state update (DEBUG)

        Note:
            Using Any type until PortfolioStateEvent is defined in Phase 2 integration.
            For now, expects event to have PortfolioState attributes.

        Example:
            >>> # Pseudo-code until PortfolioStateEvent exists
            >>> state = PortfolioState(
            ...     ts=datetime(2020, 1, 2, 16, 0),
            ...     equity=Decimal("1000000"),
            ...     cash=Decimal("500000"),
            ...     ...
            ... )
            >>> risk_service.on_portfolio_state(state)
        """
        # For MVP, assume event is PortfolioState directly
        # TODO: Update when PortfolioStateEvent is defined
        if isinstance(event, PortfolioState):
            self._portfolio_state = event
        else:
            # Try to extract PortfolioState from event
            if hasattr(event, "ts") and hasattr(event, "equity"):
                self._portfolio_state = PortfolioState(
                    ts=event.ts,
                    equity=event.equity,
                    cash=event.cash,
                    gross_exposure=event.gross_exposure,
                    net_exposure=event.net_exposure,
                    positions=event.positions,
                )
            else:
                self._logger.error(
                    "Invalid portfolio state event format",
                    extra={"event_type": type(event).__name__},
                )
                return

        self._logger.debug(
            "Portfolio state cached",
            extra={
                "ts": self._portfolio_state.ts.isoformat(),
                "equity": float(self._portfolio_state.equity),
                "cash": float(self._portfolio_state.cash),
                "num_positions": len(self._portfolio_state.positions),
            },
        )

    def on_risk_evaluation_trigger(self, event: RiskEvaluationTriggerEvent) -> None:
        """
        Evaluate buffered signals and publish orders/rejections.

        Triggered by BacktestEngine at end of bar. Processes all buffered
        signals in batch (detailed implementation in Days 12-13).

        Args:
            event: Trigger event with evaluation timestamp

        Side Effects:
            - Publishes OrderApprovedEvent per approved signal
            - Publishes OrderRejectedEvent per rejected signal
            - Clears signal buffer
            - Resets current timestamp
            - Logs batch evaluation (INFO)

        Example:
            >>> trigger = RiskEvaluationTriggerEvent(
            ...     ts=datetime(2020, 1, 2, 16, 0)
            ... )
            >>> risk_service.on_risk_evaluation_trigger(trigger)
        """
        # Timestamp consistency check
        if self._current_ts is not None and event.ts != self._current_ts:
            self._logger.warning(
                "Evaluation timestamp mismatch",
                extra={
                    "trigger_ts": event.ts.isoformat(),
                    "current_ts": self._current_ts.isoformat(),
                },
            )

        num_signals = len(self._signal_buffer)

        if num_signals == 0:
            self._logger.debug(
                "No signals to evaluate",
                extra={"ts": event.ts.isoformat()},
            )
            self._current_ts = None
            return

        if self._portfolio_state is None:
            self._logger.error(
                "Cannot evaluate signals: no portfolio state cached",
                extra={"ts": event.ts.isoformat(), "num_signals": num_signals},
            )
            # Reject all signals
            for signal_event in self._signal_buffer:
                self._publish_rejection(signal_event, event.ts, "No portfolio state available")
            self._signal_buffer.clear()
            self._current_ts = None
            return

        self._logger.info(
            "Starting batch risk evaluation",
            extra={
                "ts": event.ts.isoformat(),
                "num_signals": num_signals,
                "equity": float(self._portfolio_state.equity),
            },
        )

        # Step 1: Allocate capital across strategies
        allocations = allocate_capital(
            budgets=self._config.budgets,
            equity=self._portfolio_state.equity,
            cash_buffer_pct=self._config.cash_buffer_pct,
        )

        self._logger.info(
            "Capital allocated across strategies",
            extra={
                "ts": event.ts.isoformat(),
                "allocations": {strat_id: float(capital) for strat_id, capital in allocations.items()},
            },
        )

        # Step 2: Size positions and check limits for each signal
        approved_orders: list[OrderBase] = []
        rejected_signals: list[tuple[SignalEvent, str]] = []

        for signal_event in self._signal_buffer:
            # Extract price from signal metadata (MVP: strategy provides price)
            price = signal_event.metadata.get("price")
            if price is None:
                self._logger.warning(
                    "Signal missing price in metadata",
                    extra={
                        "strategy_id": signal_event.strategy_id,
                        "symbol": signal_event.symbol,
                    },
                )
                rejected_signals.append((signal_event, "Missing price in signal metadata"))
                continue

            current_price = Decimal(str(price))

            # Convert SignalEvent to Signal model for business logic
            signal = Signal(
                strategy_id=signal_event.strategy_id,
                symbol=signal_event.symbol,
                side=signal_event.side,
                strength=signal_event.strength,
                metadata=signal_event.metadata,
            )

            # Get allocated capital for this strategy
            allocated_capital = allocations.get(signal.strategy_id, Decimal("0"))
            if allocated_capital == 0:
                rejected_signals.append((signal_event, "Strategy has zero allocated capital"))
                continue

            # Get sizing config for this strategy
            sizing_config = self._config.sizing.get(signal.strategy_id)
            if sizing_config is None:
                rejected_signals.append((signal_event, "No sizing config for strategy"))
                continue

            # Size position
            sizer = FixedFractionSizer(
                fraction=Decimal(str(sizing_config.fraction)),
                lot_size=1,  # MVP: default lot size
                min_quantity=0,  # MVP: no minimum
            )

            quantity = sizer.size_position(
                signal=signal,
                allocated_capital=allocated_capital,
                current_price=current_price,
            )

            if quantity == 0:
                rejected_signals.append((signal_event, "Position size rounded to zero"))
                continue

            # Create proposed order
            proposed_order = OrderBase(
                strategy_id=signal.strategy_id,
                symbol=signal.symbol,
                side=signal.side,
                quantity=quantity,
                reason=f"Pending: {quantity} shares sized",
            )

            # Check limits
            violations = check_all_limits(
                order=proposed_order,
                current_positions=list(self._portfolio_state.positions.values()),
                equity=self._portfolio_state.equity,
                current_price=current_price,
                concentration_limit=self._config.concentration,
                leverage_limit=self._config.leverage,
            )

            if violations:
                # Reject due to limit violations
                reasons = [v.message for v in violations]
                rejection_reason = "; ".join(reasons)
                rejected_signals.append((signal_event, rejection_reason))
            else:
                # Approve order
                approved_order = OrderBase(
                    strategy_id=signal.strategy_id,
                    symbol=signal.symbol,
                    side=signal.side,
                    quantity=quantity,
                    reason=(
                        f"Approved: {quantity} shares, "
                        f"fraction={float(sizing_config.fraction):.4f}, "
                        f"allocated_capital={float(allocated_capital):.2f}, "
                        f"price={float(current_price):.2f}"
                    ),
                )
                approved_orders.append(approved_order)

        # Step 3: Publish approvals and rejections
        for order in approved_orders:
            self._publish_approval(order, event.ts)

        for signal_event, reason in rejected_signals:
            self._publish_rejection(signal_event, event.ts, reason)

        # Clear buffer and reset timestamp
        self._signal_buffer.clear()
        self._current_ts = None

        self._logger.info(
            "Batch risk evaluation complete",
            extra={
                "ts": event.ts.isoformat(),
                "num_signals": num_signals,
                "approved": len(approved_orders),
                "rejected": len(rejected_signals),
            },
        )

    def _publish_rejection(self, signal: SignalEvent, ts: datetime, reason: str) -> None:
        """
        Publish OrderRejectedEvent.

        Helper method to publish rejection with consistent format.

        Args:
            signal: Original signal event
            ts: Rejection timestamp
            reason: Detailed rejection reason

        Side Effects:
            - Publishes OrderRejectedEvent to event bus
            - Logs rejection (WARNING)
        """
        rejection = OrderRejectedEvent(
            ts=ts,
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            strength=signal.strength,
            reason=reason,
        )
        self._event_bus.publish(rejection)

        self._logger.warning(
            "Order rejected",
            extra={
                "ts": ts.isoformat(),
                "strategy_id": signal.strategy_id,
                "symbol": signal.symbol,
                "side": signal.side,
                "reason": reason,
            },
        )

    def _publish_approval(self, order: OrderBase, ts: datetime) -> None:
        """
        Publish OrderApprovedEvent.

        Helper method to publish approval with consistent format.

        Args:
            order: Risk-approved order
            ts: Approval timestamp

        Side Effects:
            - Publishes OrderApprovedEvent to event bus
            - Logs approval (INFO)
        """
        approval = OrderApprovedEvent(
            ts=ts,
            strategy_id=order.strategy_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            reason=order.reason,
        )
        self._event_bus.publish(approval)

        self._logger.info(
            "Order approved",
            extra={
                "ts": ts.isoformat(),
                "strategy_id": order.strategy_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": order.quantity,
                "reason": order.reason,
            },
        )
