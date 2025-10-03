"""Position tracking with average cost method."""

from decimal import Decimal
from typing import Dict, NamedTuple

from qtrader.config.logging_config import LoggerFactory
from qtrader.models.order import OrderSide

logger = LoggerFactory.get_logger()


class Position(NamedTuple):
    """
    Immutable position snapshot.

    Uses average cost method for position tracking.
    """

    symbol: str
    qty: int  # Positive for long, negative for short, 0 for flat
    avg_price: Decimal  # Average entry price (always positive)
    realized_pnl: Decimal = Decimal("0.0")  # Cumulative realized PnL

    def is_long(self) -> bool:
        """Check if position is long."""
        return self.qty > 0

    def is_short(self) -> bool:
        """Check if position is short."""
        return self.qty < 0

    def is_flat(self) -> bool:
        """Check if position is flat (no shares)."""
        return self.qty == 0

    def market_value(self, current_price: Decimal) -> Decimal:
        """
        Calculate current market value.

        For long: positive value
        For short: negative value
        """
        return Decimal(self.qty) * current_price

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """
        Calculate unrealized PnL.

        For long: (current_price - avg_price) * qty
        For short: (avg_price - current_price) * abs(qty)
        """
        if self.is_flat():
            return Decimal("0.0")

        if self.is_long():
            return (current_price - self.avg_price) * self.qty
        else:
            # For short positions, profit when price goes down
            return (self.avg_price - current_price) * abs(self.qty)


class PositionTracker:
    """
    Track positions with average cost method.

    Handles:
    - Opening new positions
    - Adding to existing positions (averages cost)
    - Reducing positions (realizes PnL)
    - Closing positions (realizes PnL)
    - Flipping positions (long→short or short→long)
    """

    def __init__(self):
        """Initialize empty position tracker."""
        self._positions: Dict[str, Position] = {}
        logger.info("position_tracker.initialized")

    def get_position(self, symbol: str) -> Position:
        """
        Get current position for symbol.

        Returns flat position if symbol not held.
        """
        return self._positions.get(symbol, Position(symbol=symbol, qty=0, avg_price=Decimal("0.0")))

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all non-flat positions."""
        return {symbol: pos for symbol, pos in self._positions.items() if not pos.is_flat()}

    def update_position(self, symbol: str, side: OrderSide, qty: int, price: Decimal) -> Position:
        """
        Update position after fill.

        Args:
            symbol: Symbol to update
            side: BUY or SELL
            qty: Quantity filled (always positive)
            price: Fill price

        Returns:
            Updated Position

        Handles:
        - Opening new position
        - Adding to existing position (averages cost)
        - Reducing position (realizes PnL)
        - Closing position (realizes PnL)
        - Flipping position (realizes PnL + opens opposite position)
        """
        current = self.get_position(symbol)

        # Convert side to signed quantity
        signed_qty = qty if side == OrderSide.BUY else -qty

        logger.debug(
            "position_tracker.update",
            symbol=symbol,
            side=side.value,
            qty=qty,
            price=float(price),
            current_qty=current.qty,
            current_avg_price=float(current.avg_price),
        )

        # Case 1: Opening new position (current flat)
        if current.is_flat():
            new_pos = Position(
                symbol=symbol,
                qty=signed_qty,
                avg_price=price,
                realized_pnl=current.realized_pnl,
            )
            self._positions[symbol] = new_pos
            logger.info(
                "position_tracker.opened",
                symbol=symbol,
                qty=new_pos.qty,
                avg_price=float(new_pos.avg_price),
            )
            return new_pos

        # Case 2: Adding to existing position (same direction)
        if (current.is_long() and side == OrderSide.BUY) or (current.is_short() and side == OrderSide.SELL):
            # Average cost: (old_cost + new_cost) / (old_qty + new_qty)
            old_value = current.avg_price * abs(current.qty)
            new_value = price * qty
            new_qty = current.qty + signed_qty
            new_avg_price = (old_value + new_value) / abs(new_qty)

            new_pos = Position(
                symbol=symbol,
                qty=new_qty,
                avg_price=new_avg_price,
                realized_pnl=current.realized_pnl,
            )
            self._positions[symbol] = new_pos
            logger.info(
                "position_tracker.added",
                symbol=symbol,
                new_qty=new_pos.qty,
                new_avg_price=float(new_pos.avg_price),
            )
            return new_pos

        # Case 3: Reducing or closing or flipping position (opposite direction)
        # Calculate realized PnL for the portion being closed
        close_qty = min(abs(current.qty), qty)
        if current.is_long():
            # Long position being reduced by sell
            realized = (price - current.avg_price) * close_qty
        else:
            # Short position being reduced by buy
            realized = (current.avg_price - price) * close_qty

        new_realized_pnl = current.realized_pnl + realized
        new_qty = current.qty + signed_qty

        logger.info(
            "position_tracker.realized_pnl",
            symbol=symbol,
            close_qty=close_qty,
            realized_pnl=float(realized),
            total_realized_pnl=float(new_realized_pnl),
        )

        # If position flips or closes, reset avg_price
        if new_qty == 0:
            # Position closed
            new_pos = Position(
                symbol=symbol,
                qty=0,
                avg_price=Decimal("0.0"),
                realized_pnl=new_realized_pnl,
            )
            self._positions[symbol] = new_pos
            logger.info("position_tracker.closed", symbol=symbol)
            return new_pos

        elif (current.is_long() and new_qty < 0) or (current.is_short() and new_qty > 0):
            # Position flipped - use price from flip trade as new avg_price
            new_pos = Position(
                symbol=symbol,
                qty=new_qty,
                avg_price=price,
                realized_pnl=new_realized_pnl,
            )
            self._positions[symbol] = new_pos
            logger.info(
                "position_tracker.flipped",
                symbol=symbol,
                new_qty=new_pos.qty,
                new_avg_price=float(new_pos.avg_price),
            )
            return new_pos

        else:
            # Position reduced but not closed - keep same avg_price
            new_pos = Position(
                symbol=symbol,
                qty=new_qty,
                avg_price=current.avg_price,
                realized_pnl=new_realized_pnl,
            )
            self._positions[symbol] = new_pos
            logger.info(
                "position_tracker.reduced",
                symbol=symbol,
                new_qty=new_pos.qty,
                avg_price=float(new_pos.avg_price),
            )
            return new_pos

    def get_total_exposure(self, prices: Dict[str, Decimal]) -> tuple[Decimal, Decimal]:
        """
        Calculate total long and short exposure.

        Args:
            prices: Current prices for each symbol

        Returns:
            (long_exposure, short_exposure) - both positive values
        """
        long_exposure = Decimal("0.0")
        short_exposure = Decimal("0.0")

        for symbol, pos in self._positions.items():
            if pos.is_flat():
                continue

            price = prices.get(symbol)
            if price is None:
                logger.warning("position_tracker.missing_price", symbol=symbol, qty=pos.qty)
                continue

            market_value = pos.market_value(price)
            if pos.is_long():
                long_exposure += market_value
            else:
                short_exposure += abs(market_value)

        return long_exposure, short_exposure

    def get_total_unrealized_pnl(self, prices: Dict[str, Decimal]) -> Decimal:
        """Calculate total unrealized PnL across all positions."""
        total = Decimal("0.0")
        for symbol, pos in self._positions.items():
            if pos.is_flat():
                continue
            price = prices.get(symbol)
            if price is not None:
                total += pos.unrealized_pnl(price)
        return total

    def get_total_realized_pnl(self) -> Decimal:
        """Calculate total realized PnL across all symbols."""
        return sum((pos.realized_pnl for pos in self._positions.values()), Decimal("0.0"))
