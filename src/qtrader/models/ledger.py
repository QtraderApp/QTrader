"""Cash ledger with Decimal precision."""

from decimal import Decimal
from typing import List, NamedTuple, Optional

from qtrader.config.logging_config import LoggerFactory

logger = LoggerFactory.get_logger()


class CashTransaction(NamedTuple):
    """Record of a cash transaction."""

    timestamp: str  # ISO format datetime
    transaction_type: str  # FILL, DIVIDEND, BORROW_COST, DEPOSIT, WITHDRAWAL
    amount: Decimal  # Positive for credit, negative for debit
    description: str
    balance_after: Decimal
    strategy_name: Optional[str] = None  # Track which strategy generated this transaction


class CashLedger:
    """
    Cash ledger with Decimal precision.

    Tracks all cash movements:
    - Fill settlements (gross value + fees)
    - Short dividend payments
    - Borrow cost accruals
    - Initial deposits
    """

    def __init__(self, initial_cash: Decimal = Decimal("100000.0")):
        """
        Initialize cash ledger.

        Args:
            initial_cash: Starting cash balance (default 100k)
        """
        if initial_cash < 0:
            raise ValueError(f"initial_cash must be >= 0, got {initial_cash}")

        self._balance = initial_cash
        self._transactions: List[CashTransaction] = []

        # Record initial deposit
        self._transactions.append(
            CashTransaction(
                timestamp="",  # Will be set when engine starts
                transaction_type="DEPOSIT",
                amount=initial_cash,
                description="Initial deposit",
                balance_after=initial_cash,
            )
        )

        logger.info("cash_ledger.initialized", initial_cash=float(initial_cash))

    def get_balance(self) -> Decimal:
        """Get current cash balance."""
        return self._balance

    def debit(
        self,
        amount: Decimal,
        timestamp: str,
        transaction_type: str,
        description: str,
        strategy_name: Optional[str] = None,
    ) -> Decimal:
        """
        Debit (subtract) from cash.

        Args:
            amount: Amount to debit (positive value)
            timestamp: ISO format timestamp
            transaction_type: Type of transaction
            description: Human-readable description
            strategy_name: Optional strategy identifier

        Returns:
            New balance

        Raises:
            ValueError: If amount is negative
        """
        if amount < 0:
            raise ValueError(f"Debit amount must be >= 0, got {amount}")

        self._balance -= amount

        self._transactions.append(
            CashTransaction(
                timestamp=timestamp,
                transaction_type=transaction_type,
                amount=-amount,  # Negative for debit
                description=description,
                balance_after=self._balance,
                strategy_name=strategy_name,
            )
        )

        logger.debug(
            "cash_ledger.debit",
            amount=float(amount),
            type=transaction_type,
            balance=float(self._balance),
            strategy_name=strategy_name,
        )

        return self._balance

    def credit(
        self,
        amount: Decimal,
        timestamp: str,
        transaction_type: str,
        description: str,
        strategy_name: Optional[str] = None,
    ) -> Decimal:
        """
        Credit (add) to cash.

        Args:
            amount: Amount to credit (positive value)
            timestamp: ISO format timestamp
            transaction_type: Type of transaction
            description: Human-readable description
            strategy_name: Optional strategy identifier

        Returns:
            New balance

        Raises:
            ValueError: If amount is negative
        """
        if amount < 0:
            raise ValueError(f"Credit amount must be >= 0, got {amount}")

        self._balance += amount

        self._transactions.append(
            CashTransaction(
                timestamp=timestamp,
                transaction_type=transaction_type,
                amount=amount,  # Positive for credit
                description=description,
                balance_after=self._balance,
                strategy_name=strategy_name,
            )
        )

        logger.debug(
            "cash_ledger.credit",
            amount=float(amount),
            type=transaction_type,
            balance=float(self._balance),
            strategy_name=strategy_name,
        )

        return self._balance

    def can_afford(self, amount: Decimal, cushion: Decimal = Decimal("0.0")) -> bool:
        """
        Check if sufficient cash available.

        Args:
            amount: Amount needed (positive value)
            cushion: Additional buffer to maintain (default 0)

        Returns:
            True if balance >= amount + cushion
        """
        return self._balance >= (amount + cushion)

    def get_transactions(self) -> List[CashTransaction]:
        """Get all transactions (read-only copy)."""
        return list(self._transactions)

    def get_transactions_by_strategy(self, strategy_name: str) -> List[CashTransaction]:
        """
        Get all transactions for a specific strategy.

        Args:
            strategy_name: Strategy identifier

        Returns:
            List of transactions for this strategy
        """
        return [t for t in self._transactions if t.strategy_name == strategy_name]

    def get_strategy_pnl(self, strategy_name: str) -> Decimal:
        """
        Calculate net cash flow for a specific strategy.

        Args:
            strategy_name: Strategy identifier

        Returns:
            Net P&L from this strategy's transactions
        """
        strategy_transactions = self.get_transactions_by_strategy(strategy_name)
        return sum((t.amount for t in strategy_transactions), Decimal("0.0"))

    def get_net_flow(self) -> Decimal:
        """
        Calculate net cash flow (excluding initial deposit).

        Positive = net deposits/profits
        Negative = net withdrawals/losses
        """
        # Skip first transaction (initial deposit)
        return sum((t.amount for t in self._transactions[1:]), Decimal("0.0"))
