"""Execution engine - processes bars and fills orders."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.execution.commission import CommissionCalculator
from qtrader.execution.config import ExecutionConfig
from qtrader.execution.fill_policy import FillDecision, FillPolicy
from qtrader.models.canonical_bar import CanonicalBar
from qtrader.models.order import Fill, OrderBase, OrderSide, OrderState, TimeInForce
from qtrader.models.portfolio import Portfolio

logger = LoggerFactory.get_logger()


class ExecutionEngine:
    """
    Execution engine processes bars and fills orders.

    Event loop:
    1. Receive bar
    2. Update portfolio prices
    3. Evaluate pending orders
    4. Generate fills
    5. Apply fills to portfolio
    6. Update order states
    7. Handle EOD accruals (borrow costs)

    Stage 3 supports:
    - Market orders (fill at next bar open)
    - MOC orders (fill at current bar close with slippage)
    """

    def __init__(
        self,
        portfolio: Portfolio,
        fill_policy: Optional[FillPolicy] = None,
        commission: Optional[CommissionCalculator] = None,
        config: Optional[ExecutionConfig] = None,
    ):
        """
        Initialize execution engine.

        Args:
            portfolio: Portfolio to manage
            fill_policy: Fill policy for order evaluation (created from config if None)
            commission: Commission calculator (created from config if None)
            config: Execution configuration (uses defaults if None)
        """
        self.portfolio = portfolio
        self.config = config or ExecutionConfig()

        # Create fill_policy and commission if not provided
        self.fill_policy = fill_policy or FillPolicy(
            moc_slip_bps=self.config.moc_slip_bps,
            stop_slip_bps=self.config.stop_slip_bps,
            limit_mode=self.config.limit_mode,
            stop_mode=self.config.stop_mode,
        )
        self.commission_calc = commission or CommissionCalculator(
            per_share=self.config.per_share,
            ticket_min=self.config.ticket_min,
        )

        # Apply high participation guardrail
        if self.config.max_participation > Decimal("0.20") and not self.config.allow_high_participation:
            logger.warning(
                "execution_engine.high_participation_clamped",
                requested=float(self.config.max_participation),
                clamped_to=0.20,
                reason="max_participation > 0.20 requires allow_high_participation=True",
            )
            # Clamp to 0.20
            self.config = self.config._replace(max_participation=Decimal("0.20"))

        # Order tracking
        self.pending_orders: Dict[str, OrderBase] = {}  # order_id -> Order
        self.filled_orders: Dict[str, OrderBase] = {}  # order_id -> Order
        self.expired_orders: Dict[str, OrderBase] = {}  # order_id -> Order
        self.all_fills: List[Fill] = []

        # Partial fill tracking (Stage 5)
        self.order_partial_counts: Dict[str, int] = {}  # order_id -> number of partial fills
        self.order_queue_bars: Dict[str, int] = {}  # order_id -> bars in queue

        # Per-bar per-side participation tracking
        self.bar_participation: Dict[tuple, int] = {}  # (bar_ts, symbol, side) -> filled_qty

        # Bar tracking (for Market orders that need next bar)
        self.current_bar: Optional[CanonicalBar] = None
        self.next_bar: Optional[CanonicalBar] = None

        logger.info(
            "execution_engine.initialized",
            max_participation=float(self.config.max_participation),
            queue_bars=self.config.queue_bars,
            allow_high_participation=self.config.allow_high_participation,
        )

    def submit_order(self, order: OrderBase, bar_ts: Optional[datetime] = None) -> None:
        """
        Submit order to execution engine.

        Args:
            order: Order to submit
            bar_ts: Current bar timestamp (uses current_bar.ts if None and current_bar exists)

        Raises:
            ValueError: If order already exists
        """
        if order.order_id in self.pending_orders:
            logger.error(
                "execution_engine.order_duplicate",
                order_id=order.order_id,
                symbol=order.symbol,
                reason="Order ID already exists in pending orders",
            )
            raise ValueError(f"Order {order.order_id} already exists")

        # Determine bar timestamp
        if bar_ts is None:
            if self.current_bar is not None:
                bar_ts = self.current_bar.ts
            elif order.strategy_ts is not None:
                bar_ts = order.strategy_ts
            else:
                raise ValueError("bar_ts must be provided when current_bar is None")

        # Set order to SUBMITTED state if not already
        if order.state != OrderState.SUBMITTED:
            order = order.with_state(OrderState.SUBMITTED)

        # Set submission_bar_ts and remaining_qty for new orders
        if order.submission_bar_ts is None:
            order = order._replace(
                submission_bar_ts=bar_ts,
                remaining_qty=order.qty,  # Initialize remaining_qty to full qty
            )
            logger.debug(
                "execution_engine.order_submission_initialized",
                order_id=order.order_id,
                submission_ts=bar_ts.isoformat(),
                remaining_qty=order.remaining_qty,
            )

        self.pending_orders[order.order_id] = order

        # Initialize tracking for partial fills
        if order.order_id not in self.order_partial_counts:
            self.order_partial_counts[order.order_id] = 0
        if order.order_id not in self.order_queue_bars:
            self.order_queue_bars[order.order_id] = 0

        logger.info(
            "execution_engine.order_submitted",
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side.value,
            qty=order.qty,
            order_type=order.order_type.value,
            tif=order.tif.value,
            limit_price=float(order.limit_price) if order.limit_price else None,
            stop_price=float(order.stop_price) if order.stop_price else None,
        )

    def on_bar(
        self,
        bar: CanonicalBar,
        symbol: str,
        ts: datetime,
        next_bar: Optional[CanonicalBar] = None,
        is_close_only: bool = False,
    ) -> List[Fill]:
        """
        Process bar and generate fills (Phase 4 CanonicalBar architecture).

        Args:
            bar: Current bar to process (CanonicalBar - no symbol/ts fields)
            symbol: Symbol for this bar (from MultiModeBar)
            ts: Timestamp for this bar (parsed from bar.trade_datetime)
            next_bar: Next bar (needed for Market orders)
            is_close_only: If True, skip limit/stop evaluation (malformed OHLC bar)

        Returns:
            List of fills generated on this bar
        """
        self.current_bar = bar
        self.next_bar = next_bar

        # Update portfolio with current prices (convert float to Decimal)
        self.portfolio.update_prices({symbol: Decimal(str(bar.close))})

        logger.debug(
            "execution_engine.on_bar",
            symbol=symbol,
            ts=ts.isoformat(),
            close=float(bar.close),
            pending_orders=len(self.pending_orders),
            is_close_only=is_close_only,
        )

        # Evaluate pending orders for this symbol
        fills = []
        orders_to_remove = []
        orders_to_expire = []

        for order_id, order in list(self.pending_orders.items()):
            # Only evaluate orders for this symbol
            if order.symbol != symbol:
                continue

            # Check DAY order expiration
            # DAY orders expire at end of day (after submission bar has passed)
            # For intraday bars: expires when date changes
            # For daily bars: expires after 1 bar (next day)
            # EXCEPT: PARTIALLY_FILLED orders continue until queue_bars expires them
            if (
                order.tif == TimeInForce.DAY
                and order.submission_bar_ts is not None
                and order.state != OrderState.PARTIALLY_FILLED  # Partials use queue_bars expiration
            ):
                # Get submission date and current date
                submission_date = order.submission_bar_ts.date()
                current_date = ts.date()

                # Expire if we're past the submission date
                if current_date > submission_date:
                    orders_to_expire.append(order_id)
                    updated_order = order.with_state(OrderState.EXPIRED)
                    self.expired_orders[order_id] = updated_order

                    logger.info(
                        "execution_engine.order_expired",
                        order_id=order_id,
                        symbol=order.symbol,
                        order_type=order.order_type.value,
                        submission_date=submission_date.isoformat(),
                        current_date=current_date.isoformat(),
                        reason="DAY order expired (new day)",
                    )
                    continue

            # Evaluate order
            decision = self.fill_policy.evaluate_order(order, bar, next_bar, is_close_only)

            if decision.should_fill:
                # Check fill price deviation safeguard (if enabled)
                if order.signal_price is not None and self.config.max_fill_price_deviation_pct is not None:
                    deviation_check = self._check_fill_price_deviation(
                        order, decision.fill_price, self.config.max_fill_price_deviation_pct
                    )
                    if not deviation_check[0]:
                        # Reject fill due to excessive price deviation
                        logger.warning(
                            "execution_engine.fill_rejected_price_deviation",
                            order_id=order_id,
                            symbol=order.symbol,
                            signal_price=float(order.signal_price),
                            fill_price=float(decision.fill_price),
                            deviation_pct=deviation_check[2],
                            max_allowed_pct=float(self.config.max_fill_price_deviation_pct),
                            reason=deviation_check[1],
                        )
                        # Cancel the order
                        orders_to_remove.append(order_id)
                        updated_order = order.with_state(OrderState.CANCELED)
                        self.expired_orders[order_id] = updated_order
                        continue

                # Calculate participation cap (Stage 5)
                participation_cap = self._calculate_participation_cap(bar, symbol, ts, order.side)

                # Determine fill quantity (may be partial)
                requested_qty = order.remaining_qty
                fill_qty = min(requested_qty, participation_cap)

                if fill_qty == 0:
                    # No capacity to fill on this bar
                    logger.debug(
                        "execution_engine.order_no_participation_cap",
                        order_id=order_id,
                        symbol=order.symbol,
                        requested_qty=requested_qty,
                        participation_cap=participation_cap,
                        reason="Participation cap exhausted for this bar/side",
                    )
                    continue

                # Increment partial index
                self.order_partial_counts[order_id] += 1
                partial_index = self.order_partial_counts[order_id]

                # Generate fill
                fill = self._generate_fill(order, decision, bar, symbol, ts, fill_qty, partial_index)
                fills.append(fill)

                # Apply fill to portfolio
                self._apply_fill(order, fill)

                # Update participation tracking
                self._update_participation(symbol, ts, order.side, fill_qty)

                # Update order with partial fill
                remaining_qty = requested_qty - fill_qty
                updated_order = order.with_partial_fill(fill_qty, decision.fill_price, remaining_qty)

                if updated_order.state == OrderState.FILLED:
                    # Order fully filled
                    self.filled_orders[order_id] = updated_order
                    orders_to_remove.append(order_id)

                    logger.info(
                        "execution_engine.order_filled",
                        order_id=order_id,
                        fill_id=fill.fill_id,
                        symbol=order.symbol,
                        side=order.side.value,
                        qty=fill_qty,
                        total_filled=updated_order.filled_qty,
                        fill_price=float(fill.price),
                        avg_fill_price=float(updated_order.avg_fill_price) if updated_order.avg_fill_price else None,
                        fees=float(fill.fees),
                        slippage_bps=fill.slippage_bps,
                        order_type=order.order_type.value,
                        partial_index=partial_index,
                    )
                else:
                    # Partial fill - check if this is the first partial or a queued one
                    # First partial doesn't increment queue counter, subsequent ones do
                    is_first_partial = self.order_queue_bars[order_id] == 0 and order.filled_qty == 0

                    if not is_first_partial:
                        # This is a queued residual fill - increment counter
                        self.order_queue_bars[order_id] += 1

                    # Update pending order
                    self.pending_orders[order_id] = updated_order

                    # Check if this partial should expire due to queue_bars limit
                    # This happens AFTER the fill is generated and counter is incremented
                    if self.order_queue_bars[order_id] >= self.config.queue_bars:
                        orders_to_expire.append(order_id)
                        expired_order = updated_order.with_state(OrderState.EXPIRED)
                        self.expired_orders[order_id] = expired_order

                        logger.info(
                            "execution_engine.order_expired_queue",
                            order_id=order_id,
                            symbol=order.symbol,
                            reason="queue_bars_exceeded",
                            queue_bars=self.order_queue_bars[order_id],
                            max_queue_bars=self.config.queue_bars,
                            filled_qty=expired_order.filled_qty,
                            remaining_qty=expired_order.remaining_qty,
                        )
                    else:
                        logger.info(
                            "execution_engine.order_partially_filled",
                            order_id=order_id,
                            fill_id=fill.fill_id,
                            symbol=order.symbol,
                            side=order.side.value,
                            filled_qty=fill_qty,
                            remaining_qty=remaining_qty,
                            total_filled=updated_order.filled_qty,
                            fill_price=float(fill.price),
                            avg_fill_price=float(updated_order.avg_fill_price)
                            if updated_order.avg_fill_price
                            else None,
                            fees=float(fill.fees),
                            participation=fill.participation,
                            partial_index=partial_index,
                        )
            else:
                # Order didn't fill - log reason for debugging
                logger.debug(
                    "execution_engine.order_not_filled",
                    order_id=order_id,
                    symbol=order.symbol,
                    order_type=order.order_type.value,
                    remaining_qty=order.remaining_qty,
                    reason=decision.reason or "not evaluated yet",
                    limit_price=float(order.limit_price) if order.limit_price else None,
                    stop_price=float(order.stop_price) if order.stop_price else None,
                    bar_high=float(bar.high),
                    bar_low=float(bar.low),
                    bar_close=float(bar.close),
                    is_close_only=is_close_only,
                )

        # Remove filled orders from pending
        for order_id in orders_to_remove:
            del self.pending_orders[order_id]

        # Remove expired orders from pending
        for order_id in orders_to_expire:
            del self.pending_orders[order_id]

        return fills

    def _generate_fill(
        self,
        order: OrderBase,
        decision: FillDecision,
        bar: CanonicalBar,
        symbol: str,
        ts: datetime,
        fill_qty: int,
        partial_index: int,
    ) -> Fill:
        """
        Generate fill from order and decision (Phase 4 CanonicalBar architecture).

        Args:
            order: Order being filled
            decision: Fill decision with price
            bar: Current bar (CanonicalBar - no symbol/ts fields)
            symbol: Symbol for this bar
            ts: Timestamp for this bar
            fill_qty: Quantity to fill (may be less than order.remaining_qty)
            partial_index: Index of this partial fill (0 for full fills)

        Returns:
            Fill object
        """
        # Calculate commission for this fill slice
        commission_result = self.commission_calc.calculate(fill_qty)

        # Generate unique fill ID
        fill_id = f"fill-{uuid.uuid4().hex[:8]}"

        # Calculate slippage in bps
        if order.order_type.value == "MOC":
            slippage_bps = self.config.moc_slip_bps
        elif order.order_type.value == "STOP":
            slippage_bps = self.config.stop_slip_bps
        else:
            slippage_bps = 0  # Market/Limit orders have no slippage

        # Calculate participation (fill_qty / bar.volume)
        participation = float(fill_qty) / float(bar.volume) if bar.volume > 0 else 0.0

        # Ensure fill_price is Decimal (CanonicalBar prices are float)
        fill_price_decimal = (
            Decimal(str(decision.fill_price)) if not isinstance(decision.fill_price, Decimal) else decision.fill_price
        )

        # Create fill
        fill = Fill(
            fill_id=fill_id,
            order_id=order.order_id,
            execution_ts=ts,
            symbol=order.symbol,
            side=order.side,
            qty=fill_qty,
            price=fill_price_decimal,
            slippage_bps=slippage_bps,
            fees=commission_result.commission,
            participation=participation,
            partial_index=partial_index,
        )

        self.all_fills.append(fill)
        return fill

    def _apply_fill(
        self,
        order: OrderBase,
        fill: Fill,
    ) -> None:
        """
        Apply fill to portfolio (Phase 4 - bar not needed, all info in fill).

        Args:
            order: Order being filled
            fill: Fill details (includes symbol, ts, price, qty, etc.)
        """
        try:
            self.portfolio.apply_fill(
                symbol=fill.symbol,
                side=fill.side,
                qty=fill.qty,
                fill_price=fill.price,
                commission=fill.fees,
                ts=fill.execution_ts,
                order_id=fill.order_id,
                fill_id=fill.fill_id,
            )
        except Exception as e:
            logger.error(
                "execution_engine.fill_application_failed",
                order_id=order.order_id,
                fill_id=fill.fill_id,
                symbol=fill.symbol,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    def _calculate_participation_cap(self, bar: CanonicalBar, symbol: str, ts: datetime, side: OrderSide) -> int:
        """
        Calculate participation cap for this bar and side (Phase 4 CanonicalBar architecture).

        Args:
            bar: Current bar (CanonicalBar)
            symbol: Symbol for this bar
            ts: Timestamp for this bar
            side: Order side (BUY or SELL)

        Returns:
            Maximum shares that can fill on this bar for this side
        """
        # Base cap: max_participation × bar.volume
        total_cap = int(bar.volume * self.config.max_participation)

        # Get already filled quantity for this bar/symbol/side
        key = (ts, symbol, side)
        already_filled = self.bar_participation.get(key, 0)

        # Remaining cap for this side
        remaining_cap = max(0, total_cap - already_filled)

        logger.debug(
            "execution_engine.participation_cap_calculated",
            symbol=symbol,
            side=side.value,
            bar_volume=bar.volume,
            max_participation=float(self.config.max_participation),
            total_cap=total_cap,
            already_filled=already_filled,
            remaining_cap=remaining_cap,
        )

        return remaining_cap

    def _check_fill_price_deviation(
        self, order: OrderBase, fill_price: Decimal, max_deviation_pct: Decimal
    ) -> tuple[bool, str, float]:
        """
        Check if fill price deviates too much from signal price.

        Args:
            order: Order with signal_price
            fill_price: Proposed fill price (may be float from CanonicalBar)
            max_deviation_pct: Maximum allowed deviation (e.g., 0.10 = 10%)

        Returns:
            Tuple of (approved, reason, deviation_pct)
            - approved: True if within tolerance, False if excessive deviation
            - reason: Explanation
            - deviation_pct: Actual deviation as percentage (e.g., 0.15 = 15%)
        """
        if order.signal_price is None:
            return (True, "No signal price to check", 0.0)

        # Ensure fill_price is Decimal for comparison (CanonicalBar uses float)
        if not isinstance(fill_price, Decimal):
            fill_price = Decimal(str(fill_price))

        # Calculate absolute deviation percentage
        deviation = abs(fill_price - order.signal_price) / order.signal_price
        deviation_pct = float(deviation)

        if deviation > max_deviation_pct:
            return (
                False,
                f"Fill price deviates {deviation_pct:.2%} from signal price (max {float(max_deviation_pct):.2%})",
                deviation_pct,
            )

        return (True, "Fill price within tolerance", deviation_pct)

    def _update_participation(self, symbol: str, ts: datetime, side: OrderSide, filled_qty: int) -> None:
        """
        Update participation tracking after a fill (Phase 4 CanonicalBar architecture).

        Args:
            symbol: Symbol for this bar
            ts: Timestamp for this bar
            side: Order side
            filled_qty: Quantity filled
        """
        key = (ts, symbol, side)
        self.bar_participation[key] = self.bar_participation.get(key, 0) + filled_qty

        logger.debug(
            "execution_engine.participation_updated",
            symbol=symbol,
            side=side.value,
            filled_qty=filled_qty,
            total_filled=self.bar_participation[key],
        )

    def on_end_of_day(self, ts: datetime) -> None:
        """
        Handle end-of-day processing.

        - Accrue borrow costs on short positions

        Args:
            ts: End-of-day timestamp
        """
        # Log EOD before borrow cost calculation
        logger.debug(
            "execution_engine.eod",
            ts=ts.isoformat(),
            pending_orders=len(self.pending_orders),
            cash=float(self.portfolio.cash.get_balance()),
        )

        # Accrue borrow costs
        self.portfolio.apply_borrow_cost(
            borrow_rate_annual=self.config.borrow_rate_annual,
            ts=ts,
        )

    def get_pending_orders(self) -> Dict[str, OrderBase]:
        """Get all pending orders."""
        return self.pending_orders.copy()

    def get_filled_orders(self) -> Dict[str, OrderBase]:
        """Get all filled orders."""
        return self.filled_orders.copy()

    def get_all_fills(self) -> List[Fill]:
        """Get all fills generated."""
        return self.all_fills.copy()

    def get_statistics(self) -> Dict:
        """Get execution statistics."""
        return {
            "pending_orders": len(self.pending_orders),
            "filled_orders": len(self.filled_orders),
            "total_fills": len(self.all_fills),
            "portfolio_equity": float(self.portfolio.get_equity()),
            "portfolio_cash": float(self.portfolio.cash.get_balance()),
        }

    def get_orders(self) -> List[OrderBase]:
        """
        Get all orders (pending + filled + expired).

        Returns:
            List of all orders
        """
        all_orders: List[OrderBase] = []
        all_orders.extend(self.pending_orders.values())
        all_orders.extend(self.filled_orders.values())
        all_orders.extend(self.expired_orders.values())
        return all_orders

    def get_fills(self) -> List[Fill]:
        """
        Get all fills.

        Returns:
            List of all fills
        """
        return self.all_fills.copy()
