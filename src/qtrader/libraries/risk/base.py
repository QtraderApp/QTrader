"""
Base Risk Policy Abstract Class.

All risk policies must inherit from BaseRiskPolicy and implement the required methods.
Risk policies operate at Portfolio Manager level and decide position sizing and risk limits.

Philosophy:
- Risk policies apply at PORTFOLIO level, not strategy level
- Strategies generate signals ("I want to buy"), risk policies decide size ("buy 100 shares")
- Risk policies have access to full portfolio state
- Risk policies can reject signals based on risk limits
- Multiple strategies → multiple signals → risk policy evaluates all in batch

Registry Name: Derived from class name (e.g., NaiveRiskPolicy → "naive")
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from qtrader.events.events import SignalEvent


@dataclass
class PortfolioState:
    """
    Portfolio state snapshot for risk evaluation.

    Actual implementation in qtrader.services.risk will provide this.
    Provides risk policies access to:
    - current_equity: Total portfolio equity
    - positions: Current positions with P&L
    - cash_available: Available cash
    - leverage: Current leverage ratio
    - drawdown: Current drawdown from peak
    """

    current_equity: Decimal
    cash_available: Decimal
    leverage: float
    drawdown: float
    positions: dict[str, Any]


@dataclass
class OrderDecision:
    """
    Risk policy decision about a signal.

    Attributes:
        approved: Whether signal is approved
        symbol: Symbol to trade
        side: BUY or SELL
        quantity: Number of shares (0 = reject)
        reason: Explanation (for logging)
    """

    approved: bool
    symbol: str
    side: str
    quantity: int
    reason: str


class BaseRiskPolicy(ABC):
    """
    Abstract base class for all risk policies.

    Responsibilities:
    - Evaluate signals against portfolio state
    - Calculate position sizes based on risk parameters
    - Enforce risk limits (max position, drawdown, leverage, etc.)
    - Batch evaluate multiple signals (cross-strategy netting)

    Does NOT:
    - Generate signals (that's strategies)
    - Execute orders (that's ExecutionService)
    - Track positions (that's PortfolioService)

    Flow:
        Strategies → SignalEvent → Portfolio Manager (RiskService):
            1. Batch collect all signals
            2. Get current portfolio state
            3. Apply risk policy to each signal
            4. Calculate position sizes
            5. Check risk limits
            6. Emit OrderEvent or reject
        → ExecutionService → Fill → Portfolio updates

    Example Implementation:
        ```python
        class NaiveRiskPolicy(BaseRiskPolicy):
            def __init__(self, config: NaiveRiskConfig):
                self.max_pct_position = config.max_pct_position_size
                self.max_drawdown = config.max_drawdown

            def evaluate_signal(
                self,
                signal: SignalEvent,
                portfolio: PortfolioState,
                price: Decimal
            ) -> OrderDecision:
                # Check drawdown limit
                if portfolio.drawdown > self.max_drawdown:
                    return OrderDecision(
                        approved=False,
                        symbol=signal.symbol,
                        side=signal.side,
                        quantity=0,
                        reason=f"Drawdown limit exceeded: {portfolio.drawdown:.2%}"
                    )

                # Calculate position size
                quantity = self.calculate_position_size(
                    signal, portfolio, price
                )

                return OrderDecision(
                    approved=True,
                    symbol=signal.symbol,
                    side=signal.side,
                    quantity=quantity,
                    reason=f"Approved {quantity} shares"
                )

            def calculate_position_size(
                self,
                signal: SignalEvent,
                portfolio: PortfolioState,
                price: Decimal
            ) -> int:
                # Max position value = max_pct * equity
                max_value = portfolio.current_equity * Decimal(self.max_pct_position)

                # Scale by signal confidence
                target_value = max_value * Decimal(signal.strength)

                # Convert to shares
                quantity = int(target_value / price)

                return quantity
        ```
    """

    @abstractmethod
    def __init__(self, config: Any):
        """
        Initialize risk policy with configuration.

        Args:
            config: Risk policy configuration object
                   Contains parameters like max_position_size, max_drawdown, etc.

        Example:
            NaiveRiskPolicy(NaiveRiskConfig(
                max_pct_position_size=0.90,
                max_drawdown=0.25
            ))

        Note:
            Store config and initialize any internal state needed.
        """
        pass

    @abstractmethod
    def evaluate_signal(self, signal: SignalEvent, portfolio: PortfolioState, price: Decimal) -> OrderDecision:
        """
        Evaluate a signal against portfolio state and risk limits.

        Args:
            signal: Trading signal from strategy
                   - signal.symbol: Symbol ticker
                   - signal.side: BUY or SELL
                   - signal.strength: Confidence [0, 1]
                   - signal.strategy_id: Source strategy

            portfolio: Current portfolio state
                      - portfolio.current_equity: Total equity
                      - portfolio.cash_available: Available cash
                      - portfolio.drawdown: Current drawdown
                      - portfolio.positions: Current positions

            price: Current market price for sizing calculation

        Returns:
            OrderDecision with approval status and position size

        Decision Logic:
            1. Check risk limits (drawdown, leverage, concentration)
            2. Calculate position size if approved
            3. Return decision with quantity and reason

        Example:
            >>> decision = policy.evaluate_signal(signal, portfolio, price)
            >>> if decision.approved:
            ...     print(f"Buy {decision.quantity} shares")
            ... else:
            ...     print(f"Rejected: {decision.reason}")

        Note:
            - Can reject signals (quantity=0) if risk limits breached
            - Can scale position size based on signal confidence
            - Should log rejection reasons for debugging
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal: SignalEvent, portfolio: PortfolioState, price: Decimal) -> int:
        """
        Calculate position size (number of shares) for approved signal.

        Args:
            signal: Trading signal (contains confidence/strength)
            portfolio: Current portfolio state (for equity-based sizing)
            price: Current market price

        Returns:
            Number of shares to trade (integer, >= 0)

        Sizing Methods:
            - Fixed percentage: position_value = equity * max_pct
            - Confidence-scaled: position_value = equity * max_pct * confidence
            - Volatility-based: position_size = risk_amount / (volatility * price)
            - Kelly criterion: fraction = (p*b - q) / b

        Examples:
            # Fixed percentage (90% max)
            >>> max_value = portfolio.equity * 0.90
            >>> quantity = int(max_value / price)

            # Confidence-scaled
            >>> max_value = portfolio.equity * 0.90
            >>> scaled_value = max_value * signal.strength
            >>> quantity = int(scaled_value / price)

            # Volatility-based
            >>> risk_per_trade = portfolio.equity * 0.02  # 2% risk
            >>> volatility = get_volatility(signal.symbol)  # ATR or std
            >>> quantity = int(risk_per_trade / (volatility * price))

        Note:
            - Returns integer number of shares (can't trade fractional)
            - Should respect cash_available (don't exceed buying power)
            - Can scale by signal confidence (strength field)
        """
        pass

    @abstractmethod
    def batch_evaluate(
        self, signals: list[SignalEvent], portfolio: PortfolioState, prices: dict[str, Decimal]
    ) -> list[OrderDecision]:
        """
        Evaluate multiple signals in batch (for cross-strategy netting).

        Args:
            signals: List of signals from multiple strategies
            portfolio: Current portfolio state
            prices: Current prices {symbol: price}

        Returns:
            List of order decisions (one per signal)

        Batch Processing Benefits:
            1. Cross-strategy netting: A buys + B sells = net position
            2. Portfolio-level limits: Total exposure across all signals
            3. Optimization: Reuse portfolio state for all signals

        Example:
            >>> signals = [
            ...     SignalEvent(strategy="A", symbol="AAPL", side="BUY", strength=0.8),
            ...     SignalEvent(strategy="B", symbol="AAPL", side="SELL", strength=0.6),
            ...     SignalEvent(strategy="C", symbol="MSFT", side="BUY", strength=0.9)
            ... ]
            >>> decisions = policy.batch_evaluate(signals, portfolio, prices)
            >>> # Result: Net AAPL position (A-B), full MSFT buy (C)

        Netting Logic:
            1. Group signals by symbol
            2. Calculate net signal per symbol (buy - sell strength)
            3. Size based on net signal
            4. Avoid wash trades (opposing signals from different strategies)

        Note:
            - Default implementation: call evaluate_signal() for each
            - Advanced: implement cross-strategy netting and optimization
            - Should prevent wash trades (double commissions)
        """
        # Default implementation: evaluate each signal independently
        decisions = []
        for signal in signals:
            price = prices.get(signal.symbol)
            if price is None:
                decisions.append(
                    OrderDecision(
                        approved=False, symbol=signal.symbol, side=signal.side, quantity=0, reason="Price not available"
                    )
                )
            else:
                decisions.append(self.evaluate_signal(signal, portfolio, price))
        return decisions

    @property
    def name(self) -> str:
        """
        Risk policy name for registry and logging.

        Returns:
            Policy name (defaults to snake_case of class name)

        Example:
            NaiveRiskPolicy → "naive"
            VolTargetRiskPolicy → "vol_target"

        Note:
            Override if you want a custom registry name.
        """
        # Convert CamelCase to snake_case
        name = self.__class__.__name__
        if name.endswith("RiskPolicy"):
            name = name[:-10]  # Remove "RiskPolicy" suffix
        elif name.endswith("Policy"):
            name = name[:-6]  # Remove "Policy" suffix

        # Simple conversion (can be overridden)
        return "".join(["_" + c.lower() if c.isupper() else c for c in name]).lstrip("_")
