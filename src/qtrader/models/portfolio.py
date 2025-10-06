"""Portfolio manager with positions and cash."""

from datetime import datetime
from decimal import Decimal
from typing import Dict, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.models.ledger import CashLedger
from qtrader.models.order import OrderSide
from qtrader.models.position import Position, PositionTracker

logger = LoggerFactory.get_logger()


class Portfolio:
    """
    Unified account state: cash ledger + position tracker.

    Handles atomic operations:
    - Fill order → update cash + update position
    - Pay dividend → debit cash (if short)
    - Accrue borrow cost → debit cash

    Provides unified views:
    - Total equity (cash + unrealized PnL)
    - Margin usage
    - Position summaries
    """

    def __init__(self, initial_cash: Decimal):
        """
        Initialize portfolio with starting cash.

        Args:
            initial_cash: Starting cash balance
        """
        self.cash = CashLedger(initial_cash=initial_cash)
        self.positions = PositionTracker()
        self._current_prices: Dict[str, Decimal] = {}  # Track latest prices
        logger.info("portfolio.initialized", initial_cash=float(initial_cash))

    def apply_fill(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        fill_price: Decimal,
        commission: Decimal,
        ts: datetime,
        order_id: str,
        fill_id: str,
    ) -> None:
        """
        Apply fill atomically: update cash AND position.

        BUY:  cash -= (qty * fill_price + commission), position += qty
        SELL: cash += (qty * fill_price - commission), position -= qty

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            qty: Fill quantity (positive)
            fill_price: Execution price (after slippage)
            commission: Total commission for this fill
            ts: Fill timestamp
            order_id: Order identifier
            fill_id: Fill identifier

        Raises:
            ValueError: If qty <= 0 or commission < 0
        """
        if qty <= 0:
            raise ValueError(f"Fill qty must be > 0, got {qty}")
        if commission < Decimal("0"):
            raise ValueError(f"Commission must be >= 0, got {commission}")

        # Calculate gross value (before commission)
        gross_value = Decimal(qty) * fill_price

        # Calculate net cash impact (including commission)
        if side == OrderSide.BUY:
            # Buying: debit cash for shares + commission
            net_cash_impact = -(gross_value + commission)
        else:  # SELL
            # Selling: credit cash for shares - commission
            net_cash_impact = gross_value - commission

        # Update cash ledger
        if net_cash_impact < Decimal("0"):
            # Check if we have sufficient cash for buy orders
            if side == OrderSide.BUY:
                cash_balance = self.cash.get_balance()
                if cash_balance + net_cash_impact < Decimal("0"):
                    logger.warning(
                        "portfolio.insufficient_cash",
                        symbol=symbol,
                        side=side.value,
                        qty=qty,
                        required_cash=float(abs(net_cash_impact)),
                        available_cash=float(cash_balance),
                        shortfall=float(abs(cash_balance + net_cash_impact)),
                        order_id=order_id,
                    )

            self.cash.debit(
                amount=abs(net_cash_impact),
                timestamp=ts.isoformat(),
                transaction_type="FILL",
                description=f"Fill {fill_id} | {side.value} {qty} {symbol} @ ${fill_price} + ${commission} commission",
            )
        else:
            self.cash.credit(
                amount=net_cash_impact,
                timestamp=ts.isoformat(),
                transaction_type="FILL",
                description=f"Fill {fill_id} | {side.value} {qty} {symbol} @ ${fill_price} - ${commission} commission",
            )

        # Update position tracker (uses side, qty, price)
        old_position = self.positions.get_position(symbol)
        old_qty = old_position.qty if old_position else 0

        self.positions.update_position(symbol, side, qty, fill_price)

        new_position = self.positions.get_position(symbol)
        new_qty = new_position.qty if new_position else 0

        # Log position transitions (especially long→short or short→long)
        if old_qty > 0 and new_qty < 0:
            logger.info(
                "portfolio.position_transition_long_to_short",
                symbol=symbol,
                old_qty=old_qty,
                new_qty=new_qty,
                fill_qty=qty,
                side=side.value,
            )
        elif old_qty < 0 and new_qty > 0:
            logger.info(
                "portfolio.position_transition_short_to_long",
                symbol=symbol,
                old_qty=old_qty,
                new_qty=new_qty,
                fill_qty=qty,
                side=side.value,
            )
        elif old_qty != 0 and new_qty == 0:
            logger.info(
                "portfolio.position_closed",
                symbol=symbol,
                old_qty=old_qty,
                realized_pnl=float(new_position.realized_pnl) if new_position else 0.0,
            )

        # Track current price for this symbol
        self._current_prices[symbol] = fill_price

        logger.info(
            "portfolio.fill_applied",
            symbol=symbol,
            side=side.value,
            qty=qty,
            fill_price=float(fill_price),
            commission=float(commission),
            cash_impact=float(net_cash_impact),
            cash_balance=float(self.cash.get_balance()),
            position_qty=new_qty,
            position_avg_price=float(new_position.avg_price) if new_position else 0.0,
        )

    def apply_short_dividend(
        self,
        symbol: str,
        dividend_per_share: Decimal,
        ts: datetime,
    ) -> None:
        """
        Debit cash for short dividend (only if net short).

        Args:
            symbol: Trading symbol
            dividend_per_share: Dividend amount per share
            ts: Ex-dividend date timestamp
        """
        position = self.positions.get_position(symbol)
        if position and position.qty < 0:
            dividend_owed = abs(position.qty) * dividend_per_share
            self.cash.debit(
                amount=dividend_owed,
                timestamp=ts.isoformat(),
                transaction_type="DIVIDEND",
                description=f"Short dividend on {symbol}: {abs(position.qty)} shares @ ${dividend_per_share}/share",
            )
            logger.info(
                "portfolio.short_dividend",
                symbol=symbol,
                qty=position.qty,
                dividend_per_share=float(dividend_per_share),
                dividend_owed=float(dividend_owed),
            )

    def apply_long_dividend(
        self,
        symbol: str,
        dividend_per_share: Decimal,
        ts: datetime,
    ) -> None:
        """
        Credit cash for long dividend (only if net long).

        Args:
            symbol: Trading symbol
            dividend_per_share: Dividend amount per share
            ts: Ex-dividend date timestamp
        """
        position = self.positions.get_position(symbol)
        if position and position.qty > 0:
            dividend_received = position.qty * dividend_per_share
            self.cash.credit(
                amount=dividend_received,
                timestamp=ts.isoformat(),
                transaction_type="DIVIDEND_RECEIVED",
                description=f"Long dividend on {symbol}: {position.qty} shares @ ${dividend_per_share}/share",
            )
            logger.info(
                "portfolio.long_dividend",
                symbol=symbol,
                qty=position.qty,
                dividend_per_share=float(dividend_per_share),
                dividend_received=float(dividend_received),
            )

    def apply_borrow_cost(
        self,
        borrow_rate_annual: Decimal,
        ts: datetime,
    ) -> None:
        """
        Accrue daily borrow cost on short positions.

        Cost = abs(short_market_value) * (annual_rate / 252)

        Args:
            borrow_rate_annual: Annual borrow rate (e.g., 0.03 for 3%)
            ts: Timestamp for accrual
        """
        # Get all short positions
        all_positions = self.positions.get_all_positions()  # Returns Dict[str, Position]
        short_positions = [pos for symbol, pos in all_positions.items() if pos.qty < 0]

        if not short_positions:
            return

        # Calculate total short market value using current prices
        total_short_mv = Decimal("0.0")
        for pos in short_positions:
            if pos.symbol in self._current_prices:
                price = self._current_prices[pos.symbol]
                total_short_mv += pos.market_value(price)  # Negative for shorts

        abs_short_mv = abs(total_short_mv)

        if abs_short_mv > Decimal("0"):
            # Daily cost = annual_rate / 252 trading days
            daily_cost = abs_short_mv * (borrow_rate_annual / Decimal("252"))

            self.cash.debit(
                amount=daily_cost,
                timestamp=ts.isoformat(),
                transaction_type="BORROW_COST",
                description=f"Borrow cost on ${abs_short_mv:,.2f} short MV @ {borrow_rate_annual:.2%} annual",
            )

            logger.info(
                "portfolio.borrow_cost",
                short_market_value=float(abs_short_mv),
                borrow_rate=float(borrow_rate_annual),
                daily_cost=float(daily_cost),
            )

    def get_equity(self) -> Decimal:
        """
        Calculate total account equity.

        Equity = cash + unrealized PnL

        Returns:
            Total equity value
        """
        unrealized_pnl = self.positions.get_total_unrealized_pnl(self._current_prices)
        return self.cash.get_balance() + unrealized_pnl

    def get_margin_usage(self) -> Decimal:
        """
        Calculate margin used by positions.

        Simple model: margin = abs(long_mv) + abs(short_mv)

        Returns:
            Total margin usage
        """
        long_mv, short_mv = self.positions.get_total_exposure(self._current_prices)
        return long_mv + short_mv  # short_mv is already positive

    def can_afford(self, cost: Decimal, cushion: Decimal = Decimal("0")) -> bool:
        """
        Check if account can afford a transaction.

        Args:
            cost: Transaction cost
            cushion: Optional safety cushion

        Returns:
            True if cash balance >= cost + cushion
        """
        return self.cash.can_afford(cost, cushion)

    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Position or None if flat
        """
        return self.positions.get_position(symbol)

    def update_prices(self, prices: Dict[str, Decimal]) -> None:
        """
        Update current prices for portfolio valuation.

        Args:
            prices: Dictionary of symbol -> current price
        """
        self._current_prices.update(prices)

    def snapshot(self, ts: datetime) -> Dict:
        """
        Complete portfolio snapshot for output files.

        Args:
            ts: Snapshot timestamp

        Returns:
            Dictionary with all portfolio state
        """
        positions_list = []
        all_positions = self.positions.get_all_positions()  # Dict[str, Position]

        for symbol, pos in all_positions.items():
            if pos.qty != 0:  # Only include non-flat positions
                price = self._current_prices.get(symbol, pos.avg_price)  # Fall back to avg_price
                positions_list.append(
                    {
                        "symbol": pos.symbol,
                        "qty": pos.qty,
                        "avg_price": float(pos.avg_price),
                        "market_value": float(pos.market_value(price)),
                        "unrealized_pnl": float(pos.unrealized_pnl(price)),
                    }
                )

        long_mv, short_mv = self.positions.get_total_exposure(self._current_prices)

        return {
            "ts": ts.isoformat(),
            "cash": float(self.cash.get_balance()),
            "equity": float(self.get_equity()),
            "positions": positions_list,
            "long_mv": float(long_mv),
            "short_mv": float(short_mv),
            "total_unrealized_pnl": float(self.positions.get_total_unrealized_pnl(self._current_prices)),
            "total_realized_pnl": float(self.positions.get_total_realized_pnl()),
        }
