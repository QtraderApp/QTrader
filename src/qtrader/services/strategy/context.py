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

import uuid
from collections import deque
from decimal import Decimal
from typing import Any, Optional

import structlog

from qtrader.events.event_bus import IEventBus
from qtrader.events.events import PriceBarEvent, SignalEvent
from qtrader.services.strategy.models import SignalIntention

logger = structlog.get_logger()


class Context:
    """
    Strategy execution context.

    Provides strategies with:
    1. Signal emission (emit_signal) - FULLY IMPLEMENTED
    2. Price queries (get_price) - IMPLEMENTED
    3. Historical bars (get_bars) - IMPLEMENTED

    Position Tracking Philosophy:
    - Strategies track their own positions via on_position_filled() events
    - No get_position() method - event-driven architecture
    - Strategies can maintain self.positions = {} if needed
    - Most strategies don't need position state - RiskManager handles it

    Current Implementation Status:
    - ✅ emit_signal: Fully implemented
    - ✅ get_price: Queries cached bars
    - ✅ get_bars: Returns historical bars from cache
    - ❌ get_position: REMOVED - use event-driven position tracking instead

    Usage in Strategy:
        ```python
        class MyStrategy(Strategy):
            def __init__(self, config):
                self.config = config
                self.positions = {}  # Optional: track if needed

            def on_position_filled(self, event: PositionFilledEvent, context: Context) -> None:
                # Optional: track position changes via events
                self.positions[event.symbol] = event.quantity

            def on_bar(self, event: PriceBarEvent, context: Context) -> None:
                # Get historical bars for indicator calculation
                bars = context.get_bars(event.symbol, n=20)
                if bars is None or len(bars) < 20:
                    return  # Self-managed warmup

                prices = [bar.close for bar in bars]
                sma = sum(prices) / len(prices)

                # Get current price
                current_price = context.get_price(event.symbol)

                if current_price and current_price > sma:
                    # Emit signal (don't check positions - RiskManager does that)
                    context.emit_signal(
                        timestamp=event.timestamp,
                        symbol=event.symbol,
                        intention=SignalIntention.OPEN_LONG,
                        confidence=0.8,
                        price=current_price,
                        reason=f"Price {current_price} above SMA {sma}"
                    )
        ```

    Note:
        - Context is created per strategy instance (not per bar)
        - Bar history cached per symbol with configurable max_bars
        - Strategies can track their own state (indicators, positions, etc.)
        - For position tracking: implement on_position_filled() lifecycle method
    """

    def __init__(
        self,
        strategy_id: str,
        event_bus: IEventBus,
        max_bars: int = 500,
    ):
        """
        Initialize context for a strategy.

        Args:
            strategy_id: Unique strategy identifier (from config.name)
            event_bus: Event bus for publishing signals
            max_bars: Maximum bars to cache per symbol (default 500)
        """
        self._strategy_id = strategy_id
        self._event_bus = event_bus
        self._signal_count = 0  # Track emitted signals for logging
        self._max_bars = max_bars

        # Bar cache: {symbol: deque[PriceBarEvent]}
        # Uses deque with maxlen for automatic windowing
        self._bar_cache: dict[str, deque[PriceBarEvent]] = {}

        logger.debug(
            "strategy.context.initialized",
            strategy_id=strategy_id,
            max_bars=max_bars,
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

        # Generate unique signal_id using UUID
        # Format: {strategy_id}-{uuid} for traceability and uniqueness across runs
        signal_id = f"{self._strategy_id}-{uuid.uuid4()}"

        # Create SignalEvent (validates against schema)
        signal = SignalEvent(
            signal_id=signal_id,
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

    def cache_bar(self, event: PriceBarEvent) -> None:
        """
        Cache bar for historical queries.

        Called by StrategyService before on_bar() to maintain rolling window
        of bars per symbol for historical lookups.

        Args:
            event: PriceBarEvent to cache
        """
        symbol = event.symbol

        # Create deque for symbol if first bar
        if symbol not in self._bar_cache:
            self._bar_cache[symbol] = deque(maxlen=self._max_bars)

        # Append bar (deque auto-evicts oldest if at maxlen)
        self._bar_cache[symbol].append(event)

        logger.debug(
            "strategy.context.bar_cached",
            strategy_id=self._strategy_id,
            symbol=symbol,
            timestamp=event.timestamp,
            cached_bars=len(self._bar_cache[symbol]),
        )

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get latest close price for a symbol.

        Returns the close price of the most recent cached bar.
        Used for current market price in strategy logic.

        Args:
            symbol: Instrument symbol

        Returns:
            Latest close price as Decimal, or None if no bars cached

        Example:
            >>> price = context.get_price("AAPL")
            >>> if price:
            ...     print(f"Current AAPL price: ${price}")
            ...
            ...     # Use for signal price
            ...     context.emit_signal(
            ...         timestamp=event.timestamp,
            ...         symbol="AAPL",
            ...         intention=SignalIntention.OPEN_LONG,
            ...         price=price,
            ...         confidence=0.80
            ...     )

        Performance:
            O(1) - fast lookup from cache

        Note:
            Returns close price of most recent bar. For intraday strategies,
            you may want to use event.close directly or get_bars(symbol, 1)[0]
            to access full OHLCV data.
        """
        if symbol not in self._bar_cache or len(self._bar_cache[symbol]) == 0:
            logger.debug(
                "strategy.context.get_price.no_data",
                strategy_id=self._strategy_id,
                symbol=symbol,
            )
            return None

        latest_bar = self._bar_cache[symbol][-1]

        logger.debug(
            "strategy.context.get_price.success",
            strategy_id=self._strategy_id,
            symbol=symbol,
            price=str(latest_bar.close),
            timestamp=latest_bar.timestamp,
        )

        return latest_bar.close

    def get_bars(self, symbol: str, n: int = 1) -> Optional[list[PriceBarEvent]]:
        """
        Get N most recent bars for a symbol.

        Returns historical bars from cache. Used for indicator calculation
        (SMA, RSI, etc.) and pattern detection.

        Args:
            symbol: Instrument symbol
            n: Number of bars to retrieve (default 1 = just last bar)

        Returns:
            List of PriceBarEvent in chronological order (oldest first),
            or None if insufficient bars cached

        Example:
            >>> # Calculate 20-period SMA
            >>> bars = context.get_bars("AAPL", n=20)
            >>> if bars and len(bars) == 20:
            ...     prices = [bar.close for bar in bars]
            ...     sma_20 = sum(prices) / 20
            ...
            ...     current_price = bars[-1].close
            ...     if current_price > sma_20:
            ...         # Price above SMA - bullish signal
            ...         context.emit_signal(...)
            >>>
            >>> # Get last 50 bars for pattern detection
            >>> bars = context.get_bars("AAPL", n=50)
            >>> if bars and len(bars) >= 50:
            ...     highs = [bar.high for bar in bars]
            ...     resistance = max(highs[-20:])  # 20-bar resistance

        Performance:
            O(n) - efficient slice from deque

        Note:
            - Returns bars in chronological order (oldest first, newest last)
            - Returns None if fewer than n bars available
            - Maximum bars cached per symbol: self._max_bars (default 500)
            - For longer histories, increase max_bars in Context initialization
        """
        if symbol not in self._bar_cache or len(self._bar_cache[symbol]) == 0:
            logger.debug(
                "strategy.context.get_bars.no_data",
                strategy_id=self._strategy_id,
                symbol=symbol,
                requested=n,
            )
            return None

        cached_count = len(self._bar_cache[symbol])

        # Return None if insufficient bars
        if cached_count < n:
            logger.debug(
                "strategy.context.get_bars.insufficient",
                strategy_id=self._strategy_id,
                symbol=symbol,
                requested=n,
                available=cached_count,
            )
            return None

        # Get last n bars and convert deque to list
        bars = list(self._bar_cache[symbol])[-n:]

        logger.debug(
            "strategy.context.get_bars.success",
            strategy_id=self._strategy_id,
            symbol=symbol,
            requested=n,
            returned=len(bars),
        )

        return bars

    @property
    def strategy_id(self) -> str:
        """Get the strategy identifier."""
        return self._strategy_id

    @property
    def signal_count(self) -> int:
        """Get total number of signals emitted by this strategy."""
        return self._signal_count
