"""Context object passed to strategy methods."""

from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from decimal import Decimal

    from qtrader.indicators.manager import IndicatorManager
    from qtrader.models import OrderBase, Portfolio
    from qtrader.models.canonical_bar import CanonicalBar
    from qtrader.risk import RiskDecision, RiskManager, Signal


class Context:
    """
    Context object providing strategy interface to engine.

    Phase 2: Provides access to risk management, portfolio state, and order submission.
    Phase 3: Adds indicator support with bar history and crossover tracking.
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
        self.current_date: Optional[datetime] = None

        # Indicator support (Phase 3)
        self._indicator_manager: Optional["IndicatorManager"] = None
        self._bar_history: dict[str, list["CanonicalBar"]] = defaultdict(list)
        self._indicator_tracking: dict[tuple[str, str], Optional[float]] = {}  # (symbol, key) -> value

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
    # Indicator API (Phase 3)
    # ========================================================================

    @property
    def ind(self) -> "IndicatorManager":
        """
        Get indicator manager for computing technical indicators.

        Returns:
            IndicatorManager instance

        Raises:
            RuntimeError: If indicator manager not configured

        Example:
            sma_20 = ctx.ind.sma(symbol, 20)
            rsi = ctx.ind.rsi(symbol, 14)
        """
        if self._indicator_manager is None:
            # Lazy initialization
            from qtrader.indicators.manager import IndicatorManager

            self._indicator_manager = IndicatorManager(self)
        return self._indicator_manager

    def get_bar_history(self, symbol: str, lookback: int) -> list["CanonicalBar"]:
        """
        Get historical bars for symbol.

        Args:
            symbol: Symbol to get bars for
            lookback: Number of bars to return (most recent)

        Returns:
            List of Bar objects (most recent last)

        Example:
            bars = ctx.get_bar_history(symbol, 20)
            closes = [float(b.close) for b in bars]
        """
        history = self._bar_history.get(symbol, [])
        return history[-lookback:] if lookback > 0 else history

    def current_bar(self, symbol: str) -> Optional["CanonicalBar"]:
        """
        Get current bar for symbol.

        Args:
            symbol: Symbol to get current bar for

        Returns:
            Current Bar or None if no bars exist
        """
        history = self._bar_history.get(symbol)
        return history[-1] if history else None

    def _track_indicator(self, symbol: str, key: str, value: Optional[float]) -> None:
        """
        Track indicator value for crossover detection.

        Internal method called by strategies to enable ctx.crossed_above/below helpers.

        Args:
            symbol: Symbol indicator is for
            key: Indicator identifier (e.g., 'sma_20', 'rsi_14')
            value: Indicator value to track
        """
        self._indicator_tracking[(symbol, key)] = value

    def _get_previous_indicator(self, symbol: str, key: str) -> Optional[float]:
        """
        Get previously tracked indicator value.

        Args:
            symbol: Symbol
            key: Indicator identifier

        Returns:
            Previous value or None if not tracked
        """
        return self._indicator_tracking.get((symbol, key))

    def crossed_above(self, symbol: str, key1: str, key2: str) -> bool:
        """
        Check if indicator1 crossed above indicator2.

        Requires values to be tracked via _track_indicator().

        Args:
            symbol: Symbol
            key1: First indicator key
            key2: Second indicator key

        Returns:
            True if crossover occurred, False otherwise

        Example:
            ctx._track_indicator(symbol, 'sma_20', sma_20)
            ctx._track_indicator(symbol, 'sma_50', sma_50)
            if ctx.crossed_above(symbol, 'sma_20', 'sma_50'):
                # Bullish crossover
                pass
        """
        from qtrader.indicators.helpers import crossed_above

        curr1 = self._indicator_tracking.get((symbol, key1))
        curr2 = self._indicator_tracking.get((symbol, key2))
        # Previous values from prior bar (would need to be saved in engine)
        # For now, this is a stub - full implementation needs engine support
        prev1 = self._get_previous_indicator(symbol, f"prev_{key1}")
        prev2 = self._get_previous_indicator(symbol, f"prev_{key2}")

        return crossed_above(curr1, curr2, prev1, prev2)

    def crossed_below(self, symbol: str, key1: str, key2: str) -> bool:
        """
        Check if indicator1 crossed below indicator2.

        Requires values to be tracked via _track_indicator().

        Args:
            symbol: Symbol
            key1: First indicator key
            key2: Second indicator key

        Returns:
            True if crossover occurred, False otherwise
        """
        from qtrader.indicators.helpers import crossed_below

        curr1 = self._indicator_tracking.get((symbol, key1))
        curr2 = self._indicator_tracking.get((symbol, key2))
        prev1 = self._get_previous_indicator(symbol, f"prev_{key1}")
        prev2 = self._get_previous_indicator(symbol, f"prev_{key2}")

        return crossed_below(curr1, curr2, prev1, prev2)

    def crossed_above_threshold(self, symbol: str, key: str, threshold: float) -> bool:
        """
        Check if indicator crossed above threshold.

        Requires value to be tracked via _track_indicator().

        Args:
            symbol: Symbol
            key: Indicator key
            threshold: Threshold value

        Returns:
            True if crossed above threshold, False otherwise

        Example:
            ctx._track_indicator(symbol, 'rsi_14', rsi)
            if ctx.crossed_above_threshold(symbol, 'rsi_14', 30):
                # RSI crossed above 30 (oversold exit)
                pass
        """
        from qtrader.indicators.helpers import crossed_above_threshold

        curr = self._indicator_tracking.get((symbol, key))
        prev = self._get_previous_indicator(symbol, f"prev_{key}")

        return crossed_above_threshold(curr, prev, threshold)

    def crossed_below_threshold(self, symbol: str, key: str, threshold: float) -> bool:
        """
        Check if indicator crossed below threshold.

        Requires value to be tracked via _track_indicator().

        Args:
            symbol: Symbol
            key: Indicator key
            threshold: Threshold value

        Returns:
            True if crossed below threshold, False otherwise
        """
        from qtrader.indicators.helpers import crossed_below_threshold

        curr = self._indicator_tracking.get((symbol, key))
        prev = self._get_previous_indicator(symbol, f"prev_{key}")

        return crossed_below_threshold(curr, prev, threshold)

    def _add_bar_to_history(self, symbol: str, bar: "CanonicalBar") -> None:
        """
        Add bar to history (called by engine).

        Internal method - not for strategy use.

        Args:
            symbol: Symbol for the bar
            bar: Bar to add
        """
        self._bar_history[symbol].append(bar)

    def _save_indicator_state(self) -> None:
        """
        Save current indicator values as previous (called by engine between bars).

        Internal method - not for strategy use.
        """
        # Save current values as "prev_*" for next bar's crossover detection
        for (symbol, key), value in list(self._indicator_tracking.items()):
            if not key.startswith("prev_"):
                self._indicator_tracking[(symbol, f"prev_{key}")] = value

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
