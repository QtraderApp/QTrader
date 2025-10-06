"""
Risk Manager - Centralized risk management system.

Evaluates trading signals against risk policies, determines position sizing,
and controls portfolio-level constraints (concentration, leverage, cash).
"""

from decimal import Decimal
from typing import NamedTuple

import structlog

from qtrader.models.order import Order, OrderBase, OrderSide, OrderState
from qtrader.models.portfolio import Portfolio
from qtrader.risk.policy import RiskPolicy
from qtrader.risk.signal import Signal, SignalDirection, SignalType
from qtrader.risk.sizing import calculate_position_size

logger = structlog.get_logger()


class RiskDecision(NamedTuple):
    """
    Risk decision for a signal.

    Attributes:
        approved: True if signal approved, False if rejected
        sized_qty: Approved quantity in shares (0 if rejected)
        reason: Explanation (approval reason or rejection reason)
        original_qty: Original calculated quantity before limits
        applied_limits: List of limits that were applied
    """

    approved: bool
    sized_qty: int
    reason: str
    original_qty: int = 0
    applied_limits: list[str] = []


class RiskManager:
    """
    Centralized risk management (portfolio-scoped).

    Flow:
    1. Receive Signal from strategy
    2. Validate against risk policies
    3. Calculate position size
    4. Apply concentration limits
    5. Check cash availability
    6. Return RiskDecision (approved/rejected + sized qty)
    """

    def __init__(self, policy: RiskPolicy, portfolio: Portfolio):
        """
        Initialize risk manager.

        Args:
            policy: Risk policy configuration
            portfolio: Portfolio instance for state queries
        """
        self.policy = policy
        self.portfolio = portfolio
        self.signal_count = 0
        self.approved_count = 0
        self.rejected_count = 0

        # Validate policy
        policy.validate()

        logger.info(
            "risk.manager.initialized",
            sizing_method=policy.sizing_method.value,
            max_position_pct=str(policy.max_position_pct),
            max_positions=policy.max_positions,
            allow_shorting=policy.allow_shorting,
        )

    def evaluate_signal(self, signal: Signal, current_price: Decimal) -> RiskDecision:
        """
        Evaluate signal and determine sized order.

        Args:
            signal: Trading signal from strategy
            current_price: Current market price for symbol

        Returns:
            RiskDecision with approved qty or rejection reason
        """
        self.signal_count += 1

        # Validate signal
        try:
            signal.validate()
        except ValueError as e:
            self.rejected_count += 1
            logger.warning(
                "risk.signal_validation_failed",
                signal_id=signal.signal_id,
                reason=str(e),
            )
            return RiskDecision(
                approved=False,
                sized_qty=0,
                reason=f"Signal validation failed: {e}",
            )

        # Step 1: Validate direction (shorting allowed?)
        if signal.direction == SignalDirection.SHORT and not self.policy.allow_shorting:
            self.rejected_count += 1
            logger.info(
                "risk.short_not_allowed",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
            )
            return RiskDecision(
                approved=False,
                sized_qty=0,
                reason="Shorting not allowed by policy",
            )

        # Step 2: Check portfolio-level constraints (leverage, exposure)
        constraint_check = self._check_portfolio_constraints(signal)
        if not constraint_check[0]:
            self.rejected_count += 1
            return RiskDecision(
                approved=False,
                sized_qty=0,
                reason=constraint_check[1],
            )

        # Step 3: Calculate initial position size
        try:
            sized_qty = calculate_position_size(
                signal=signal,
                policy=self.policy,
                portfolio=self.portfolio,
                current_price=current_price,
            )
        except ValueError as e:
            self.rejected_count += 1
            logger.warning(
                "risk.sizing_failed",
                signal_id=signal.signal_id,
                reason=str(e),
            )
            return RiskDecision(
                approved=False,
                sized_qty=0,
                reason=f"Sizing calculation failed: {e}",
            )

        if sized_qty <= 0:
            self.rejected_count += 1
            return RiskDecision(
                approved=False,
                sized_qty=0,
                reason="Calculated size is zero or negative",
            )

        original_qty = sized_qty
        applied_limits = []

        # Step 4a (optional): Check cash BEFORE concentration adjustment
        # This ensures multi-strategy fairness by rejecting signals that can't afford
        # the calculated size, even if concentration limits would reduce it to fit.
        if self.policy.check_cash_before_concentration:
            pre_concentration_cash_check = self._check_cash_availability(
                signal=signal,
                sized_qty=sized_qty,
                current_price=current_price,
            )
            if not pre_concentration_cash_check[0]:
                self.rejected_count += 1
                logger.info(
                    "risk.insufficient_cash_pre_concentration",
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    sized_qty=sized_qty,
                    reason=pre_concentration_cash_check[1],
                )
                return RiskDecision(
                    approved=False,
                    sized_qty=0,
                    reason=pre_concentration_cash_check[1],
                    original_qty=original_qty,
                    applied_limits=[],
                )

        # Step 4: Apply concentration limits
        sized_qty, limits = self._apply_concentration_limits(
            signal=signal,
            sized_qty=sized_qty,
            current_price=current_price,
        )

        applied_limits.extend(limits)

        if sized_qty <= 0:
            self.rejected_count += 1
            return RiskDecision(
                approved=False,
                sized_qty=0,
                reason=f"Position size reduced to zero by concentration limits: {', '.join(applied_limits)}",
                original_qty=original_qty,
                applied_limits=applied_limits,
            )

        # Step 5: Check cash availability
        cash_check = self._check_cash_availability(
            signal=signal,
            sized_qty=sized_qty,
            current_price=current_price,
        )

        if not cash_check[0]:
            self.rejected_count += 1
            return RiskDecision(
                approved=False,
                sized_qty=0,
                reason=cash_check[1],
                original_qty=original_qty,
                applied_limits=applied_limits,
            )

        # Step 6: Approve signal
        self.approved_count += 1

        logger.info(
            "risk.signal_approved",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            sized_qty=sized_qty,
            original_qty=original_qty,
            applied_limits=applied_limits if applied_limits else None,
        )

        return RiskDecision(
            approved=True,
            sized_qty=sized_qty,
            reason="Signal approved",
            original_qty=original_qty,
            applied_limits=applied_limits,
        )

    def signal_to_order(self, signal: Signal, decision: RiskDecision, current_price: Decimal) -> OrderBase:
        """
        Convert approved signal + decision to sized Order.

        Args:
            signal: Original trading signal
            decision: Approved risk decision with sized quantity
            current_price: Current market price

        Returns:
            Sized Order ready for execution

        Raises:
            ValueError: If signal was rejected
        """
        if not decision.approved:
            raise ValueError(f"Cannot convert rejected signal to order: {decision.reason}")

        # Determine order side
        if signal.signal_type in (SignalType.ENTRY_LONG, SignalType.EXIT_SHORT):
            side = OrderSide.BUY
        else:  # ENTRY_SHORT, EXIT_LONG
            side = OrderSide.SELL

        # Create order
        order = Order(
            order_id=f"ord-{signal.signal_id}",
            strategy_ts=signal.strategy_ts,
            symbol=signal.symbol,
            side=side,
            qty=decision.sized_qty,
            order_type=signal.order_type,
            state=OrderState.SUBMITTED,
            limit_price=signal.limit_price,
            stop_price=signal.stop_price,
            tif=signal.tif,
            signal_price=current_price,  # Store signal price for deviation checks
        )

        logger.debug(
            "risk.signal_to_order",
            signal_id=signal.signal_id,
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
        )

        return order

    def _check_portfolio_constraints(self, signal: Signal) -> tuple[bool, str]:
        """
        Check portfolio-level constraints (leverage, exposure).

        Args:
            signal: Trading signal

        Returns:
            (is_valid, reason) tuple
        """
        # Get current exposure
        equity = self.portfolio.get_equity()
        if equity <= Decimal("0"):
            if self.policy.reject_on_leverage_breach:
                return (False, "Portfolio equity is zero or negative")
            else:
                logger.warning("risk.negative_equity", equity=str(equity))

        # Calculate current gross/net exposure
        # Note: For simplicity, we skip market value calculation here since we don't have current prices.
        # Concentration is enforced at signal evaluation time using current_price parameter.
        # This check is for leverage based on existing positions only.

        # For now, return True as leverage checks require real-time pricing
        # TODO: Implement proper leverage tracking with current market prices
        return (True, "Portfolio constraints satisfied (leverage check TODO)")

    def _apply_concentration_limits(
        self,
        signal: Signal,
        sized_qty: int,
        current_price: Decimal,
    ) -> tuple[int, list[str]]:
        """
        Apply concentration limits to position size.

        Args:
            signal: Trading signal
            sized_qty: Initial calculated size
            current_price: Current market price

        Returns:
            (adjusted_qty, applied_limits) tuple
        """
        applied_limits = []
        adjusted_qty = sized_qty

        # Get equity
        equity = self.portfolio.get_equity()
        if equity <= Decimal("0"):
            return (0, ["zero_equity"])

        # Check max position percentage
        position_value = Decimal(adjusted_qty) * current_price
        position_pct = position_value / equity

        if position_pct > self.policy.max_position_pct:
            # Reduce to max allowed
            max_value = equity * self.policy.max_position_pct
            adjusted_qty = int(max_value / current_price)
            applied_limits.append(f"max_position_pct:{self.policy.max_position_pct:.2%}")

            logger.debug(
                "risk.concentration_limit_applied",
                signal_id=signal.signal_id,
                original_qty=sized_qty,
                adjusted_qty=adjusted_qty,
                limit="max_position_pct",
            )

        # Check max positions count
        if self.policy.max_positions is not None:
            all_positions = self.portfolio.positions.get_all_positions()
            current_positions = len([p for p in all_positions.values() if p.qty != 0])

            # If adding new position (not in portfolio yet)
            existing_position = self.portfolio.positions.get_position(signal.symbol)
            if existing_position is None or existing_position.qty == 0:
                if current_positions >= self.policy.max_positions:
                    if self.policy.reject_on_concentration_breach:
                        return (0, [f"max_positions:{self.policy.max_positions}"])
                    else:
                        applied_limits.append(f"max_positions:{self.policy.max_positions}")

        return (adjusted_qty, applied_limits)

    def _check_cash_availability(
        self,
        signal: Signal,
        sized_qty: int,
        current_price: Decimal,
    ) -> tuple[bool, str]:
        """
        Check if sufficient cash available for order.

        Args:
            signal: Trading signal
            sized_qty: Position size in shares
            current_price: Current market price

        Returns:
            (is_available, reason) tuple
        """
        # Calculate required cash
        position_value = Decimal(sized_qty) * current_price

        # Get current cash
        cash = self.portfolio.cash.get_balance()

        # Calculate minimum required cash (with reserve)
        equity = self.portfolio.get_equity()
        min_cash = equity * self.policy.cash_reserve_pct

        available_cash = cash - min_cash

        if position_value > available_cash:
            if self.policy.reject_on_insufficient_cash:
                return (
                    False,
                    f"Insufficient cash: need {position_value:.2f}, available {available_cash:.2f} (after {self.policy.cash_reserve_pct:.1%} reserve)",
                )
            else:
                logger.warning(
                    "risk.insufficient_cash_warning",
                    signal_id=signal.signal_id,
                    required=str(position_value),
                    available=str(available_cash),
                )

        return (True, "Cash available")

    def get_stats(self) -> dict:
        """
        Get risk manager statistics.

        Returns:
            Dictionary with signal counts and approval rate
        """
        approval_rate = self.approved_count / self.signal_count if self.signal_count > 0 else 0.0

        return {
            "signals_total": self.signal_count,
            "signals_approved": self.approved_count,
            "signals_rejected": self.rejected_count,
            "approval_rate": f"{approval_rate:.1%}",
        }
