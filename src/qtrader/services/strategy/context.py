"""
Strategy Context Implementation.

Provides runtime context to strategies during execution. Strategies use
this to emit signals and query market state (positions, prices, bars).

Architecture:
- Context is strategy's interface to the outside world
- Strategies NEVER directly access services or event bus
- All communication flows through Context

Philosophy:
- Strategies declare INTENT (signals), not orders
- Strategies ask for STATE (positions, prices), don't manage it
- Strategies are STATELESS regarding portfolio (no position tracking)
"""

from decimal import Decimal
from typing import Any, Optional

import structlog

from qtrader.events.event_bus import IEventBus
from qtrader.events.events import SignalEvent
from qtrader.services.strategy.models import SignalIntention

logger = structlog.get_logger()


class Context:
    """
    Strategy execution context.

    Provides strategies with:
    1. Signal emission (emit_signal) - PRIMARY USE
    2. Position queries (get_position) - FUTURE
    3. Price queries (get_price) - FUTURE
    4. Historical bars (get_bars) - FUTURE

    Current Implementation Status:
    - ✅ emit_signal: Fully implemented
    - 🚧 get_position: Stub (returns None)
    - 🚧 get_price: Stub (returns None)
    - 🚧 get_bars: Stub (returns None)

    Usage in Strategy:
        ```python
        class MyStrategy(Strategy):
            def on_bar(self, event: PriceBarEvent, context: Context) -> None:
                if event.is_warmup:
                    return

                # Emit signal
                context.emit_signal(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    intention=SignalIntention.OPEN_LONG,
                    confidence=0.8,
                    reason="Golden cross detected"
                )

                # Query position (future)
                # position = context.get_position(event.symbol)
                # if position and position.quantity > 0:
                #     # Already long, don't add more
                #     return
        ```

    Note:
        - Context is created per strategy instance (not per bar)
        - Context is stateless - doesn't store strategy state
        - Strategies should store their own state (indicators, counters, etc.)
    """

    def __init__(self, strategy_id: str, event_bus: IEventBus):
        """
        Initialize context for a strategy.

        Args:
            strategy_id: Unique strategy identifier (from config.name)
            event_bus: Event bus for publishing signals
        """
        self._strategy_id = strategy_id
        self._event_bus = event_bus
        self._signal_count = 0  # Track emitted signals for logging

        logger.debug(
            "strategy.context.initialized",
            strategy_id=strategy_id,
        )

    def emit_signal(
        self,
        timestamp: str,
        symbol: str,
        intention: SignalIntention | str,
        price: Decimal | float | str,
        confidence: Decimal | float | str,
        reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        stop_loss: Optional[Decimal | float | str] = None,
        take_profit: Optional[Decimal | float | str] = None,
    ) -> SignalEvent:
        """
        Emit a trading signal to the event bus.

        This is the PRIMARY way strategies communicate their trading intent.
        Signals are published to the event bus where other services can consume them.

        Args:
            timestamp: Signal generation time (ISO8601 UTC string, usually bar.timestamp)
            symbol: Instrument symbol to trade
            intention: Trading intention (OPEN_LONG/CLOSE_LONG/OPEN_SHORT/CLOSE_SHORT)
            price: Price at which signal generated (typically current market price)
            confidence: Signal strength [0.0, 1.0]
            reason: Optional human-readable explanation
            metadata: Optional additional context (indicator values, etc.)
            stop_loss: Optional stop loss price
            take_profit: Optional take profit price

        Returns:
            The created SignalEvent instance (for testing/logging)

        Example:
            >>> context.emit_signal(
            ...     timestamp="2024-01-02T16:00:00Z",
            ...     symbol="AAPL",
            ...     intention=SignalIntention.OPEN_LONG,
            ...     price=Decimal("150.25"),
            ...     confidence=0.85,
            ...     reason="SMA crossover: fast > slow",
            ...     metadata={"fast_sma": 150.0, "slow_sma": 148.0},
            ...     stop_loss=Decimal("145.00"),
            ...     take_profit=Decimal("160.00")
            ... )

        Notes:
            - Signal is automatically tagged with strategy_id
            - Signal is published to event bus immediately
            - Signal validates against signal.v1.json schema
            - RiskService decides whether to act on the signal
            - Multiple signals can be emitted per bar (if needed)
        """
        # Convert to Decimal if needed
        if not isinstance(price, Decimal):
            price = Decimal(str(price))
        if not isinstance(confidence, Decimal):
            confidence = Decimal(str(confidence))
        if stop_loss is not None and not isinstance(stop_loss, Decimal):
            stop_loss = Decimal(str(stop_loss))
        if take_profit is not None and not isinstance(take_profit, Decimal):
            take_profit = Decimal(str(take_profit))

        # Create SignalEvent (validates against schema)
        signal = SignalEvent(
            timestamp=timestamp,
            strategy_id=self._strategy_id,
            symbol=symbol,
            intention=intention,
            price=price,
            confidence=confidence,
            reason=reason,
            metadata=metadata,
            stop_loss=stop_loss,
            take_profit=take_profit,
            source_service="strategy_service",
        )

        # Publish to event bus
        self._event_bus.publish(signal)

        # Track for logging
        self._signal_count += 1

        logger.info(
            "strategy.signal.emitted",
            strategy_id=self._strategy_id,
            symbol=symbol,
            intention=signal.intention,
            price=str(signal.price),
            confidence=str(signal.confidence),
            reason=signal.reason,
            total_signals=self._signal_count,
        )

        return signal

    def get_position(self, symbol: str) -> Optional[Any]:
        """
        Get current position for a symbol.

        STUB: Not yet implemented. Returns None.

        Future implementation will query PortfolioService for current position.

        Args:
            symbol: Instrument symbol

        Returns:
            Position object or None if no position
            Currently always returns None (stub)

        Example (future):
            >>> position = context.get_position("AAPL")
            >>> if position:
            ...     print(f"Long {position.quantity} shares at avg ${position.avg_price}")
        """
        logger.debug(
            "strategy.context.get_position.stub",
            strategy_id=self._strategy_id,
            symbol=symbol,
            result="stub_returns_none",
        )
        return None

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.

        STUB: Not yet implemented. Returns None.

        Future implementation will query DataService for latest price.

        Args:
            symbol: Instrument symbol

        Returns:
            Latest price or None if not available
            Currently always returns None (stub)

        Example (future):
            >>> price = context.get_price("AAPL")
            >>> if price and price > self.entry_price * 1.10:
            ...     # Price up 10%, take profit
            ...     context.emit_signal(...)
        """
        logger.debug(
            "strategy.context.get_price.stub",
            strategy_id=self._strategy_id,
            symbol=symbol,
            result="stub_returns_none",
        )
        return None

    def get_bars(self, symbol: str, n: int = 1) -> Optional[list]:
        """
        Get N most recent bars for a symbol.

        STUB: Not yet implemented. Returns None.

        Future implementation will query DataService for historical bars.

        Args:
            symbol: Instrument symbol
            n: Number of bars to retrieve (default 1 = just last bar)

        Returns:
            List of Bar objects or None if not available
            Currently always returns None (stub)

        Example (future):
            >>> bars = context.get_bars("AAPL", n=20)
            >>> if bars:
            ...     prices = [bar.close for bar in bars]
            ...     sma = sum(prices) / len(prices)
        """
        logger.debug(
            "strategy.context.get_bars.stub",
            strategy_id=self._strategy_id,
            symbol=symbol,
            n=n,
            result="stub_returns_none",
        )
        return None

    @property
    def strategy_id(self) -> str:
        """Get the strategy identifier."""
        return self._strategy_id

    @property
    def signal_count(self) -> int:
        """Get total number of signals emitted by this strategy."""
        return self._signal_count
