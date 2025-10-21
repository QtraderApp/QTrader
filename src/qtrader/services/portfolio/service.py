"""Portfolio service implementation.

Main service for portfolio accounting with lot-based position tracking.

Week 1: Basic fills (open positions), cash management, queries
Week 2: Lot accounting (FIFO/LIFO), realized P&L
Week 3: Corporate actions, fees, mark-to-market
Week 4: State management, polish
"""

from datetime import datetime
from decimal import Decimal
from typing import Literal

from qtrader.config import LoggerFactory
from qtrader.services.portfolio.lot_tracker import LotTracker
from qtrader.services.portfolio.models import (
    Ledger,
    LedgerEntry,
    LedgerEntryType,
    Lot,
    LotSide,
    PortfolioConfig,
    PortfolioState,
    Position,
)

logger = LoggerFactory.get_logger()


class PortfolioService:
    """
    Portfolio service implementation.

    Provides lot-based position accounting with complete audit trail.
    Week 1 supports basic fill processing (opens only), cash tracking, and queries.

    Example:
        >>> config = PortfolioConfig(initial_cash=Decimal("100000"))
        >>> portfolio = PortfolioService(config)
        >>>
        >>> # Open long position
        >>> portfolio.apply_fill(
        ...     fill_id="fill_001",
        ...     timestamp=datetime.now(),
        ...     symbol="AAPL",
        ...     side="buy",
        ...     quantity=Decimal("100"),
        ...     price=Decimal("150.00")
        ... )
        >>>
        >>> print(f"Cash: ${portfolio.get_cash()}")
        >>> print(f"Equity: ${portfolio.get_equity()}")
    """

    def __init__(self, config: PortfolioConfig):
        """
        Initialize portfolio service.

        Args:
            config: Portfolio configuration
        """
        self.config = config

        # Core state
        self._cash: Decimal = config.initial_cash
        self._positions: dict[str, Position] = {}
        self._lot_tracker: dict[str, LotTracker] = {}  # symbol → LotTracker

        # Ledger
        self._ledger = Ledger(max_entries=config.max_ledger_entries)

        # Cumulative metrics
        self._cumulative_realized_pnl = Decimal("0")  # Track realized P&L
        self._total_commissions = Decimal("0")
        self._total_borrow_fees = Decimal("0")
        self._total_margin_interest = Decimal("0")
        self._total_dividends_received = Decimal("0")
        self._total_dividends_paid = Decimal("0")

        logger.info(
            "portfolio_service.initialized",
            initial_cash=str(config.initial_cash),
            lot_method_long=config.lot_method_long,
            lot_method_short=config.lot_method_short,
        )

    # ==================== Fill Processing ====================

    def apply_fill(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        side: Literal["buy", "sell"],
        quantity: Decimal,
        price: Decimal,
        commission: Decimal = Decimal("0"),
    ) -> None:
        """
        Apply fill to portfolio.

        Routes to appropriate handler based on existing position:
        - Buy with no short → open/add to long position
        - Sell with no long → open/add to short position
        - Buy with existing short → close short position (LIFO)
        - Sell with existing long → close long position (FIFO)

        Args:
            fill_id: Unique identifier for this fill
            timestamp: When fill occurred
            symbol: Ticker symbol
            side: "buy" or "sell"
            quantity: Number of shares (positive)
            price: Price per share
            commission: Commission paid

        Raises:
            ValueError: If inputs invalid
        """
        # Validate inputs
        self._validate_fill_inputs(fill_id, quantity, price, commission)

        # Determine action
        has_long = self._has_long_position(symbol)
        has_short = self._has_short_position(symbol)
        is_buy = side == "buy"
        is_sell = side == "sell"

        if is_buy and not has_short:
            # Open or add to long position
            self._open_long_position(fill_id, timestamp, symbol, quantity, price, commission)
        elif is_sell and not has_long:
            # Open or add to short position
            self._open_short_position(fill_id, timestamp, symbol, quantity, price, commission)
        elif is_sell and has_long:
            # Close long position (FIFO)
            self._close_long_position(fill_id, timestamp, symbol, quantity, price, commission)
        elif is_buy and has_short:
            # Close short position (LIFO)
            self._close_short_position(fill_id, timestamp, symbol, quantity, price, commission)
        else:
            # Should never reach here
            raise ValueError(f"Invalid state: side={side}, has_long={has_long}, has_short={has_short}")

        logger.info(
            "portfolio_service.fill_applied",
            fill_id=fill_id,
            symbol=symbol,
            side=side,
            quantity=str(quantity),
            price=str(price),
            commission=str(commission),
        )

    def _open_long_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> None:
        """
        Open or add to long position.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to buy
            price: Price per share
            commission: Commission paid
        """
        # Create lot
        lot = Lot(
            lot_id=f"{fill_id}_lot",
            symbol=symbol,
            side=LotSide.LONG,
            quantity=quantity,
            entry_price=price,
            entry_timestamp=timestamp,
            entry_fill_id=fill_id,
            entry_commission=commission,
        )

        # Add to lot tracker
        if symbol not in self._lot_tracker:
            self._lot_tracker[symbol] = LotTracker()
        self._lot_tracker[symbol].add_lot(lot)

        # Update position
        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal("0"),
                lots=[],
                last_updated=timestamp,
            )

        position = self._positions[symbol]
        position.quantity += quantity
        position.lots.append(lot)
        position.total_cost += quantity * price
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (buy = cash out)
        cash_flow = -(quantity * price + commission)
        self._cash += cash_flow
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=quantity,
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            fill_id=fill_id,
            lot_ids=[lot.lot_id],
            description=f"Buy {quantity} {symbol} @ ${price}",
        )
        self._ledger.add_entry(entry)

    def _open_short_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> None:
        """
        Open or add to short position.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to sell short
            price: Price per share
            commission: Commission paid
        """
        # Create lot (negative quantity for short)
        lot = Lot(
            lot_id=f"{fill_id}_lot",
            symbol=symbol,
            side=LotSide.SHORT,
            quantity=-quantity,  # Negative for short
            entry_price=price,
            entry_timestamp=timestamp,
            entry_fill_id=fill_id,
            entry_commission=commission,
        )

        # Add to lot tracker
        if symbol not in self._lot_tracker:
            self._lot_tracker[symbol] = LotTracker()
        self._lot_tracker[symbol].add_lot(lot)

        # Update position
        if symbol not in self._positions:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=Decimal("0"),
                lots=[],
                last_updated=timestamp,
            )

        position = self._positions[symbol]
        position.quantity -= quantity  # Negative for short
        position.lots.append(lot)
        position.total_cost -= quantity * price  # Negative cost for short
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (sell short = cash in)
        cash_flow = quantity * price - commission
        self._cash += cash_flow
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=-quantity,  # Negative for short
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            fill_id=fill_id,
            lot_ids=[lot.lot_id],
            description=f"Sell short {quantity} {symbol} @ ${price}",
        )
        self._ledger.add_entry(entry)

    def _close_long_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> None:
        """
        Close long position using FIFO lot matching.

        Calculates realized P&L for each matched lot, updates position,
        adds cash, creates ledger entries.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to sell (positive)
            price: Sale price per share
            commission: Commission paid

        Raises:
            ValueError: If insufficient long quantity to close
        """
        # Match lots using FIFO
        tracker = self._lot_tracker[symbol]
        matches = tracker.match_close_long(quantity)

        total_realized_pnl = Decimal("0")
        closed_lot_ids: list[str] = []

        # Calculate realized P&L for each matched lot
        for lot, qty_closed in matches:
            # Realized P&L = (exit_price - entry_price) * quantity - total_commissions
            # Both entry and exit commissions are allocated proportionally
            exit_commission = commission * (qty_closed / quantity)
            entry_commission = lot.entry_commission * (qty_closed / lot.quantity)
            total_commissions = entry_commission + exit_commission
            pnl = (price - lot.entry_price) * qty_closed - total_commissions
            total_realized_pnl += pnl
            closed_lot_ids.append(lot.lot_id)

            logger.debug(
                "portfolio_service.lot_closed",
                lot_id=lot.lot_id,
                symbol=symbol,
                side="long",
                quantity=qty_closed,
                entry_price=float(lot.entry_price),
                exit_price=float(price),
                realized_pnl=float(pnl),
            )

        # Update position
        position = self._positions[symbol]
        position.quantity -= quantity

        # Remove closed lots from position.lots
        position.lots = [
            lot for lot in position.lots if not any(lot.lot_id == closed_id for closed_id in closed_lot_ids)
        ]

        # Also remove any "_remaining" versions that got split
        position.lots = [
            lot
            for lot in position.lots
            if not any(lot.lot_id.startswith(f"{closed_id}_remaining") for closed_id in closed_lot_ids)
        ]

        # Add any remaining lots from tracker
        tracker_lots = tracker.get_lots(LotSide.LONG)
        for tracker_lot in tracker_lots:
            if tracker_lot not in position.lots:
                position.lots.append(tracker_lot)

        # Recalculate total_cost and avg_price
        position.total_cost = sum(
            (lot.quantity * lot.entry_price for lot in position.lots),
            start=Decimal("0"),
        )
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (sell = cash in)
        cash_flow = quantity * price - commission
        self._cash += cash_flow
        self._cumulative_realized_pnl += total_realized_pnl
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=-quantity,  # Negative for sell
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            realized_pnl=total_realized_pnl,
            fill_id=fill_id,
            lot_ids=closed_lot_ids,
            description=f"Sell {quantity} {symbol} @ ${price} (FIFO close)",
        )
        self._ledger.add_entry(entry)

        logger.info(
            "portfolio_service.position_closed",
            symbol=symbol,
            side="long",
            quantity=float(quantity),
            price=float(price),
            realized_pnl=float(total_realized_pnl),
            lots_closed=len(matches),
        )

    def _close_short_position(
        self,
        fill_id: str,
        timestamp: datetime,
        symbol: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> None:
        """
        Close short position using LIFO lot matching.

        Calculates realized P&L for each matched lot, updates position,
        deducts cash, creates ledger entries.

        Args:
            fill_id: Fill identifier
            timestamp: Fill time
            symbol: Ticker
            quantity: Shares to buy to cover (positive)
            price: Buy price per share
            commission: Commission paid

        Raises:
            ValueError: If insufficient short quantity to close
        """
        # Match lots using LIFO
        tracker = self._lot_tracker[symbol]
        matches = tracker.match_close_short(quantity)

        total_realized_pnl = Decimal("0")
        closed_lot_ids: list[str] = []

        # Calculate realized P&L for each matched lot
        for lot, qty_closed in matches:
            # Realized P&L = (entry_price - exit_price) * quantity - total_commissions
            # Both entry and exit commissions are allocated proportionally
            # For shorts: profit when buy back at lower price
            exit_commission = commission * (qty_closed / quantity)
            entry_commission = lot.entry_commission * (abs(qty_closed) / abs(lot.quantity))
            total_commissions = entry_commission + exit_commission
            pnl = (lot.entry_price - price) * qty_closed - total_commissions
            total_realized_pnl += pnl
            closed_lot_ids.append(lot.lot_id)

            logger.debug(
                "portfolio_service.lot_closed",
                lot_id=lot.lot_id,
                symbol=symbol,
                side="short",
                quantity=qty_closed,
                entry_price=float(lot.entry_price),
                exit_price=float(price),
                realized_pnl=float(pnl),
            )

        # Update position
        position = self._positions[symbol]
        position.quantity += quantity  # Add back (shorts are negative)

        # Remove closed lots from position.lots
        position.lots = [
            lot for lot in position.lots if not any(lot.lot_id == closed_id for closed_id in closed_lot_ids)
        ]

        # Also remove any "_remaining" versions that got split
        position.lots = [
            lot
            for lot in position.lots
            if not any(lot.lot_id.startswith(f"{closed_id}_remaining") for closed_id in closed_lot_ids)
        ]

        # Add any remaining lots from tracker
        tracker_lots = tracker.get_lots(LotSide.SHORT)
        for tracker_lot in tracker_lots:
            if tracker_lot not in position.lots:
                position.lots.append(tracker_lot)

        # Recalculate total_cost and avg_price
        position.total_cost = sum(
            (lot.quantity * lot.entry_price for lot in position.lots),
            start=Decimal("0"),
        )
        position.avg_price = position.total_cost / position.quantity if position.quantity != 0 else Decimal("0")
        position.last_updated = timestamp

        # Update cash (buy to cover = cash out)
        cash_flow = -(quantity * price + commission)
        self._cash += cash_flow
        self._cumulative_realized_pnl += total_realized_pnl
        self._total_commissions += commission

        # Create ledger entry
        entry = LedgerEntry(
            entry_id=f"{fill_id}_ledger",
            timestamp=timestamp,
            entry_type=LedgerEntryType.FILL,
            symbol=symbol,
            quantity=quantity,  # Positive for buy
            price=price,
            cash_flow=cash_flow,
            commission=commission,
            realized_pnl=total_realized_pnl,
            fill_id=fill_id,
            lot_ids=closed_lot_ids,
            description=f"Buy to cover {quantity} {symbol} @ ${price} (LIFO close)",
        )
        self._ledger.add_entry(entry)

        logger.info(
            "portfolio_service.position_closed",
            symbol=symbol,
            side="short",
            quantity=float(quantity),
            price=float(price),
            realized_pnl=float(total_realized_pnl),
            lots_closed=len(matches),
        )

    # ==================== Market Data ====================

    def update_prices(self, prices: dict[str, Decimal]) -> None:
        """
        Update mark-to-market prices (intraday).

        Args:
            prices: Dict mapping symbol → current price
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_market_value(price)

        logger.debug(
            "portfolio_service.prices_updated",
            count=len(prices),
        )

    def mark_to_market(self, timestamp: datetime) -> None:
        """
        Perform end-of-day mark-to-market valuation.

        Week 1: Price updates only
        Week 3: Adds fee/interest accruals

        Args:
            timestamp: Time of mark
        """
        # Week 1: Just update prices (already done via update_prices)
        # Week 3: Will add borrow fees and margin interest accruals
        logger.info(
            "portfolio_service.mark_to_market",
            timestamp=timestamp.isoformat(),
        )

    # ==================== Corporate Actions ====================

    def process_split(
        self,
        symbol: str,
        split_date: datetime,
        ratio: Decimal,
    ) -> None:
        """
        Process stock split.

        Week 3 implementation.

        Args:
            symbol: Symbol splitting
            split_date: Date of split
            ratio: Split ratio

        Raises:
            NotImplementedError: Week 3
        """
        raise NotImplementedError("Week 3 implementation")

    def process_dividend(
        self,
        symbol: str,
        ex_date: datetime,
        amount_per_share: Decimal,
    ) -> None:
        """
        Process cash dividend.

        Week 3 implementation.

        Args:
            symbol: Symbol paying dividend
            ex_date: Ex-dividend date
            amount_per_share: Dividend per share

        Raises:
            NotImplementedError: Week 3
        """
        raise NotImplementedError("Week 3 implementation")

    # ==================== Queries ====================

    def get_position(self, symbol: str) -> Position | None:
        """Get current position for symbol."""
        position = self._positions.get(symbol)
        if position and position.quantity == 0 and not self.config.keep_position_history:
            return None
        return position

    def get_positions(self) -> dict[str, Position]:
        """Get all current positions."""
        if self.config.keep_position_history:
            return self._positions.copy()
        else:
            # Filter out flat positions
            return {symbol: pos for symbol, pos in self._positions.items() if pos.quantity != 0}

    def get_cash(self) -> Decimal:
        """Get current cash balance."""
        return self._cash

    def get_equity(self) -> Decimal:
        """Calculate total portfolio equity."""
        market_value = sum(pos.market_value for pos in self._positions.values())
        return self._cash + market_value

    def get_state(self) -> PortfolioState:
        """
        Get complete portfolio state snapshot.

        Returns:
            Immutable state snapshot
        """
        # Calculate exposures
        long_exposure = Decimal("0")
        short_exposure = Decimal("0")
        unrealized_pnl = Decimal("0")

        for position in self._positions.values():
            unrealized_pnl += position.unrealized_pnl
            if position.quantity > 0:
                long_exposure += position.market_value
            elif position.quantity < 0:
                short_exposure += abs(position.market_value)

        net_exposure = long_exposure - short_exposure
        gross_exposure = long_exposure + short_exposure
        market_value_total = sum((pos.market_value for pos in self._positions.values()), start=Decimal("0"))
        equity = self._cash + market_value_total

        # Calculate leverage (handle zero equity)
        leverage = gross_exposure / equity if equity > 0 else Decimal("0")

        # Realized P&L from ledger
        realized_pnl = self._calculate_realized_pnl()

        return PortfolioState(
            timestamp=datetime.now(),
            cash=self._cash,
            positions=self._positions.copy(),
            equity=equity,
            market_value=market_value_total,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            total_pnl=realized_pnl + unrealized_pnl,
            long_exposure=long_exposure,
            short_exposure=short_exposure,
            net_exposure=net_exposure,
            gross_exposure=gross_exposure,
            leverage=leverage,
            total_commissions=self._total_commissions,
            total_borrow_fees=self._total_borrow_fees,
            total_margin_interest=self._total_margin_interest,
            total_dividends_received=self._total_dividends_received,
            total_dividends_paid=self._total_dividends_paid,
        )

    def get_ledger(
        self,
        since: datetime | None = None,
        entry_types: list[LedgerEntryType] | None = None,
    ) -> list[LedgerEntry]:
        """Get ledger entries."""
        entries = self._ledger.get_entries(since=since)

        # Filter by type if specified
        if entry_types is not None:
            entries = [e for e in entries if e.entry_type in entry_types]

        return entries

    def get_realized_pnl(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> Decimal:
        """Get realized P&L."""
        entries = self.get_ledger(since=since, entry_types=[LedgerEntryType.FILL])

        if symbol is not None:
            entries = [e for e in entries if e.symbol == symbol]

        # Sum realized P&L from entries
        return sum((e.realized_pnl for e in entries if e.realized_pnl is not None), start=Decimal("0"))

    def get_unrealized_pnl(
        self,
        symbol: str | None = None,
    ) -> Decimal:
        """Get unrealized P&L."""
        if symbol is not None:
            position = self.get_position(symbol)
            return position.unrealized_pnl if position else Decimal("0")

        return sum((pos.unrealized_pnl for pos in self._positions.values()), start=Decimal("0"))

    # ==================== Helper Methods ====================

    def _validate_fill_inputs(
        self,
        fill_id: str,
        quantity: Decimal,
        price: Decimal,
        commission: Decimal,
    ) -> None:
        """
        Validate fill inputs.

        Args:
            fill_id: Fill identifier
            quantity: Quantity
            price: Price
            commission: Commission

        Raises:
            ValueError: If any validation fails
        """
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")

        if price <= 0:
            raise ValueError(f"Price must be positive, got {price}")

        if commission < 0:
            raise ValueError(f"Commission cannot be negative, got {commission}")

        # Check for duplicate fill_id in ledger
        existing_entries = [e for e in self._ledger.get_entries() if e.fill_id == fill_id]
        if existing_entries:
            raise ValueError(f"Fill ID {fill_id} already exists in ledger")

    def _has_long_position(self, symbol: str) -> bool:
        """Check if symbol has long position."""
        if symbol not in self._positions:
            return False
        return self._positions[symbol].quantity > 0

    def _has_short_position(self, symbol: str) -> bool:
        """Check if symbol has short position."""
        if symbol not in self._positions:
            return False
        return self._positions[symbol].quantity < 0

    def _calculate_realized_pnl(self) -> Decimal:
        """Calculate total realized P&L from ledger."""
        entries = self._ledger.get_entries(entry_type=LedgerEntryType.FILL)
        return sum((e.realized_pnl for e in entries if e.realized_pnl is not None), start=Decimal("0"))
