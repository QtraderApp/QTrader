"""Context object passed to strategy methods."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from decimal import Decimal

    from qtrader.models import OrderBase, Portfolio
    from qtrader.risk import RiskDecision, RiskManager, Signal


class Context:
    """
    Context object providing strategy interface to engine.

    Phase 2: Provides access to risk management, portfolio state, and order submission.
    """

    def __init__(
        self,
        risk_manager: Optional["RiskManager"] = None,
        portfolio: Optional["Portfolio"] = None,
        current_symbol: Optional[str] = None,
        current_price: Optional["Decimal"] = None,
    ):
        """
        Initialize context with engine components.

        Args:
            risk_manager: RiskManager instance for signal evaluation
            portfolio: Portfolio instance for state queries
            current_symbol: Current symbol being processed
            current_price: Current bar price
        """
        self.risk_manager = risk_manager
        self.portfolio = portfolio
        self.current_symbol = current_symbol
        self.current_price = current_price

    # ========================================================================
    # Risk Management API
    # ========================================================================

    def evaluate_signal(self, signal: "Signal") -> "RiskDecision":
        """
        Evaluate signal through risk manager.

        Args:
            signal: Trading signal to evaluate

        Returns:
            RiskDecision with approval status and sized quantity

        Raises:
            RuntimeError: If risk manager not configured
        """
        if self.risk_manager is None:
            raise RuntimeError("Risk manager not configured")
        if self.current_price is None:
            raise RuntimeError("Current price not set")

        return self.risk_manager.evaluate_signal(signal, self.current_price)

    def signal_to_order(self, signal: "Signal", decision: "RiskDecision") -> "OrderBase":
        """
        Convert approved signal to order.

        Args:
            signal: Trading signal
            decision: Approved risk decision

        Returns:
            Sized order ready for execution

        Raises:
            RuntimeError: If risk manager not configured
            ValueError: If signal was rejected
        """
        if self.risk_manager is None:
            raise RuntimeError("Risk manager not configured")
        if self.current_price is None:
            raise RuntimeError("Current price not set")

        return self.risk_manager.signal_to_order(signal, decision, self.current_price)

    # ========================================================================
    # Portfolio Query API
    # ========================================================================

    def get_equity(self) -> "Decimal":
        """Get current portfolio equity."""
        if self.portfolio is None:
            raise RuntimeError("Portfolio not configured")
        return self.portfolio.get_equity()

    def get_cash(self) -> "Decimal":
        """Get current cash balance."""
        if self.portfolio is None:
            raise RuntimeError("Portfolio not configured")
        return self.portfolio.cash.get_balance()

    def get_position(self, symbol: Optional[str] = None):
        """
        Get current position for symbol.

        Args:
            symbol: Symbol to query (defaults to current_symbol)

        Returns:
            Position object
        """
        if self.portfolio is None:
            raise RuntimeError("Portfolio not configured")

        symbol = symbol or self.current_symbol
        if symbol is None:
            raise RuntimeError("No symbol specified")

        return self.portfolio.positions.get_position(symbol)

    # ========================================================================
    # Legacy Order API (deprecated - use signals instead)
    # ========================================================================

    def buy_market(self, qty: int) -> str:
        """
        Submit market buy order (legacy API).

        DEPRECATED: Use Signal-based API instead for proper risk management.
        """
        raise NotImplementedError("Use Signal-based API instead")

    def sell_market(self, qty: int) -> str:
        """
        Submit market sell order (legacy API).

        DEPRECATED: Use Signal-based API instead for proper risk management.
        """
        raise NotImplementedError("Use Signal-based API instead")
