"""
Event class definitions for QTrader event-driven architecture.

All events are immutable (frozen dataclasses) for deterministic processing
and safe sharing across services. Events represent facts that occurred in
the past and cannot be modified.

Event Types:
- MarketDataEvent: Price updates, corporate actions
- OrderEvent: Order creation by strategy
- FillEvent: Order execution by execution service
- PositionChangedEvent: Position updates by portfolio
- CashChangedEvent: Cash balance updates by portfolio
- RiskViolationEvent: Risk limit breaches
- BacktestControlEvent: Backtest lifecycle events

Phase-Based Event Ordering for Quantitative Correctness:
========================================================
QTrader uses a strict phase-based architecture with barrier events to ensure
causally consistent state and prevent common quant gotchas.

Phase Flow (per timestamp T):
-----------------------------

Phase 1: MarketData
    Events: PriceBarEvent (multiple, one per symbol)
    Published by: DataService
    Consumed by: StrategyService (updates indicators), RiskService (caches prices)
    Guarantees:
        - ALL bars for timestamp T published before Phase 2
        - No signals generated until all data arrived (prevents lookahead)
        - Services update internal state (indicators, caches)

Phase 2: Valuation (BARRIER)
    Event: ValuationTriggerEvent (single, marks barrier)
    Published by: BacktestEngine (after all PriceBarEvent for T)
    Consumed by: PortfolioService
    Guarantees:
        - Creates CONSISTENT SNAPSHOT of portfolio state
        - All market data for T processed before valuation
        - Equity, positions, P&L calculated atomically
        - Risk decisions use single coherent view (no race conditions)

Phase 3: RiskEvaluation (BARRIER)
    Event: RiskEvaluationTriggerEvent (single, marks barrier)
    Published by: BacktestEngine (after valuation complete)
    Consumed by: RiskService
    Guarantees:
        - Processes ALL buffered signals in single batch
        - Cross-strategy netting applied (A buys + B sells = net 0)
        - Position limits checked against consistent portfolio snapshot
        - No wash trades (prevents double commissions)
        - Approved orders published as OrderApprovedEvent

Phase 4: Execution (next cycle at T+1)
    Event: OrderEvent, FillEvent
    Published by: ExecutionService
    Consumed by: PortfolioService (updates positions/cash)
    Guarantees:
        - Fills occur at T+1 prices (realistic slippage model)
        - Fill → Portfolio update → Risk sees new state
        - Strict causal ordering (no stale state)

Barrier Events Explained:
-------------------------
Barrier events (ValuationTriggerEvent, RiskEvaluationTriggerEvent) are
"clock ticks" that synchronize all services to the same logical time.

Without barriers:
    ❌ Strategy A signal arrives before all bars → uses partial data
    ❌ Risk reads portfolio while Fill processing → wrong leverage
    ❌ Strategy A buys + Strategy B sells → 2 fills, double commissions

With barriers:
    ✅ All bars arrive → barrier → strategies see complete data
    ✅ Fill completes → Portfolio updates → barrier → Risk sees new state
    ✅ Signals buffered → barrier → net across strategies → single fill

Quant Gotchas Prevented:
========================

1. Race Conditions:
   Problem: Strategy A and B signals arrive in arbitrary order, Risk uses
            inconsistent snapshots for position limit checks
   Solution: ValuationTriggerEvent = clock tick, all strategies synchronized
   Example: Both strategies evaluate at same portfolio equity value

2. Wash Trades:
   Problem: Strategy A buys 100 AAPL + Strategy B sells 100 AAPL = 2 fills
            → double commissions, tax reporting complexity
   Solution: RiskEvaluationTriggerEvent triggers batch netting across strategies
   Example: A buys 100 + B sells 100 = net 0 (no fill, no commission)

3. Stale Portfolio State:
   Problem: Risk calculates position size while Fill still updating Portfolio
            → wrong leverage, violates limits
   Solution: Strict phase ordering: Fill → Portfolio → Barrier → Risk
   Example: Fill at 09:31 → Portfolio equity updates → 09:32 Risk sees new equity

4. Indicator Lookahead:
   Problem: Strategy generates signal before all bars for timestamp arrive
            → uses partial data, unrealistic backtest
   Solution: All PriceBarEvent published before strategies evaluate
   Example: 500 stocks * 1 bar → all arrive → then strategies compute

5. Temporal Consistency:
   Problem: Different services see different "now" times, calculations drift
   Solution: Barrier events mark logical clock ticks, services sync
   Example: Valuation at 16:00, Risk at 16:00, Execution at 16:00 (next day)

Production Considerations:
==========================

Backtest Mode (synchronous):
    - Single-threaded, deterministic
    - Barriers are function calls (no async coordination needed)
    - Perfect for research: reproducible, debuggable

Production Mode (async):
    - Same phase pattern, async event bus
    - Barriers use async coordination (e.g., CountDownLatch)
    - Bounded queues prevent memory explosion
    - Backpressure policy: drop old bars during market bursts (open/close)
    - State versioning: tag snapshots with clock tick for debugging

Implementation Pattern:
    Backtest: synchronous calls, immediate propagation
    Production: async queues, coordinator tracks barriers
    Both: same phase ordering, same logical correctness

Event Design Principles:
========================

1. Immutability (frozen=True):
   - Events are facts, cannot be modified after creation
   - Safe to share across services without defensive copying
   - Enables event replay for debugging

2. Self-Describing:
   - All events include timestamp, event_id
   - Services can reconstruct state from event log
   - Audit trail for regulatory compliance

3. Separation of Concerns:
   - Events carry data (Bar, Signal, Fill)
   - Services implement business logic
   - Engine orchestrates phases (no business logic)

4. Contracts Integration:
   - Events wrap contract payloads (PriceBarEvent contains Bar from contracts.data)
   - Infrastructure (event_id, timestamp) separate from domain (Bar)
   - Services depend on contracts, not internal models
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal
from uuid import uuid4

from qtrader.contracts.data import Bar


@dataclass(frozen=True)
class Event:
    """
    Base event class for all QTrader events.

    All events are immutable (frozen=True) to ensure deterministic processing
    and prevent accidental modifications. Each event has a unique ID and
    timestamp for tracking and replay.

    Attributes:
        event_id: Unique identifier for this event instance
        timestamp: When the event occurred
        event_type: Type identifier for event routing (set by subclasses)
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""


# ============================================
# Market Data Events
# ============================================


@dataclass(frozen=True)
class MarketDataEvent(Event):
    """
    Base class for market data events.

    Published by: DataService
    Consumed by: Strategy, Analytics, Reporting
    """

    symbol: str = ""


@dataclass(frozen=True)
class PriceBarEvent(MarketDataEvent):
    """
    Price bar received from data feed.

    Published when new OHLCV data arrives. Contains the full Bar
    with all three adjustment modes (unadjusted, adjusted, total_return).

    Published by: DataService
    Consumed by: Strategy (for signals), Analytics, Reporting

    Example:
        >>> event = PriceBarEvent(
        ...     symbol="AAPL",
        ...     bar=bar_instance,
        ...     timestamp=datetime(2020, 1, 2, 16, 0),
        ...     is_warmup=False
        ... )
    """

    event_type: str = "price_bar"
    bar: Bar | None = None
    is_warmup: bool = False  # True during warmup phase, strategies should not generate signals


@dataclass(frozen=True)
class CorporateActionEvent(MarketDataEvent):
    """
    Corporate action occurred (split, dividend, etc.).

    Published when corporate action detected from data feed. Portfolio
    uses this to adjust positions and cash accordingly.

    Published by: DataService (from Algoseek data)
    Consumed by: PortfolioService, Analytics, Reporting

    Attributes:
        symbol: Ticker symbol
        action_type: Type of action ('dividend' or 'split')
        effective_date: When action takes effect
        dividend_amount: Amount per share for dividends
        dividend_type: Cash or stock dividend
        ex_date: Ex-dividend date (for dividends)
        split_ratio: Split ratio (for splits, e.g., 4.0 for 4-for-1)

    Examples:
        >>> # Cash dividend
        >>> event = CorporateActionEvent(
        ...     symbol="AAPL",
        ...     action_type="dividend",
        ...     effective_date=datetime(2020, 2, 7),
        ...     ex_date=datetime(2020, 2, 7),
        ...     dividend_amount=Decimal("0.77"),
        ...     dividend_type="cash"
        ... )
        >>>
        >>> # Stock split (4-for-1)
        >>> event = CorporateActionEvent(
        ...     symbol="AAPL",
        ...     action_type="split",
        ...     effective_date=datetime(2020, 8, 31),
        ...     split_ratio=Decimal("4.0")
        ... )
    """

    event_type: str = "corporate_action"
    action_type: Literal["dividend", "split"] = "dividend"
    effective_date: datetime | None = None

    # For dividends
    dividend_amount: Decimal | None = None
    dividend_type: Literal["cash", "stock"] = "cash"
    ex_date: datetime | None = None

    # For splits
    split_ratio: Decimal | None = None


# ============================================
# Trading Events
# ============================================


@dataclass(frozen=True)
class SignalEvent(Event):
    """
    Trading signal generated by a strategy (Phase 4 spec).

    Published by Strategy when generating a buy/sell signal.
    RiskService subscribes to evaluate and size the signal.

    Published by: Strategy
    Consumed by: RiskService (Phase 4)

    Attributes:
        ts: Timestamp of the signal (bar close time)
        strategy_id: ID of strategy generating signal
        symbol: Instrument symbol
        side: BUY or SELL
        strength: Signal confidence in [-1, 1], where:
            - 1.0 = maximum conviction buy
            - 0.0 = no conviction
            - -1.0 = maximum conviction sell (for short strategies)
        metadata: Optional dict for strategy-specific data (e.g., price, volatility)

    Example:
        >>> event = SignalEvent(
        ...     ts=datetime(2020, 1, 2, 16, 0),
        ...     strategy_id="momentum_v1",
        ...     symbol="AAPL",
        ...     side="BUY",
        ...     strength=0.75
        ... )

    Flow:
        Strategy → SignalEvent → RiskService → OrderApprovedEvent/OrderRejectedEvent
    """

    event_type: str = "signal"
    ts: datetime = field(default_factory=datetime.now)
    strategy_id: str = ""
    symbol: str = ""
    side: Literal["BUY", "SELL"] = "BUY"
    strength: float = 0.0  # [-1, 1]
    metadata: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderEvent(Event):
    """
    Order approved by risk manager.

    Published when RiskService approves a signal and creates an order for
    execution. Orders are validated against risk limits (position size,
    margin requirements, etc.) before being sent to execution.

    Published by: RiskService (Phase 4)
    Consumed by: ExecutionService (Phase 3), Analytics, Reporting

    Attributes:
        order_id: Unique order identifier
        signal_id: Associated signal ID (if from signal)
        symbol: Ticker symbol
        side: Buy or sell
        quantity: Number of shares (validated by risk)
        order_type: Market, limit, etc.
        limit_price: For limit orders
        metadata: Additional order details

    Example:
        >>> order = OrderEvent(
        ...     order_id="ord_123",
        ...     signal_id="sig_123",
        ...     symbol="AAPL",
        ...     side="buy",
        ...     quantity=Decimal("100"),
        ...     order_type="market"
        ... )

    Flow:
        SignalEvent → RiskService validates → OrderEvent → ExecutionService

    Note:
        Full Order model will be defined in Phase 3 (ExecutionService).
        For now, we use a simplified structure.
    """

    event_type: str = "order"
    order_id: str = ""
    signal_id: str = ""  # Link back to signal
    symbol: str = ""
    side: Literal["buy", "sell"] = "buy"
    quantity: Decimal = Decimal("0")
    order_type: Literal["market", "limit"] = "market"
    limit_price: Decimal | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FillEvent(Event):
    """
    Order filled by execution engine.

    Published when execution service successfully fills an order (fully or
    partially). Portfolio service processes fills to update positions.

    Published by: ExecutionService (Phase 3)
    Consumed by: PortfolioService (Phase 2), Analytics, Reporting

    Attributes:
        fill_id: Unique fill identifier
        order_id: Associated order ID
        symbol: Ticker symbol
        side: Buy or sell
        quantity: Shares filled
        price: Fill price per share (including slippage)
        commission: Commission paid (calculated by ExecutionService from system config)
        slippage_bps: Slippage applied in basis points (for audit trail)
        timestamp: When fill occurred
        metadata: Additional fill details

    Note:
        Commission and slippage are calculated by ExecutionService based on
        system configuration. Portfolio records these values but does not
        recalculate them.
    """

    event_type: str = "fill"
    fill_id: str = ""
    order_id: str = ""
    symbol: str = ""
    side: Literal["buy", "sell"] = "buy"
    quantity: Decimal = Decimal("0")
    price: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    slippage_bps: int = 0  # Basis points of slippage applied
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================
# Portfolio Events
# ============================================


@dataclass(frozen=True)
class PositionChangedEvent(Event):
    """
    Position changed (fill applied or corporate action).

    Published when portfolio position changes due to fill execution,
    corporate action, or other event.

    Published by: PortfolioService (Phase 2)
    Consumed by: RiskService (Phase 4), Analytics, Reporting

    Attributes:
        symbol: Ticker symbol
        old_quantity: Previous position quantity (positive=long, negative=short)
        new_quantity: New position quantity
        reason: Why position changed ('fill', 'split', 'dividend')
        metadata: Additional details (e.g., lot info)

    Example:
        >>> event = PositionChangedEvent(
        ...     symbol="AAPL",
        ...     old_quantity=Decimal("0"),
        ...     new_quantity=Decimal("100"),
        ...     reason="fill"
        ... )
    """

    event_type: str = "position_changed"
    symbol: str = ""
    old_quantity: Decimal = Decimal("0")
    new_quantity: Decimal = Decimal("0")
    reason: Literal["fill", "split", "dividend"] = "fill"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CashChangedEvent(Event):
    """
    Cash balance changed.

    Published when portfolio cash changes due to fill, dividend,
    fee accrual, or interest charge.

    Published by: PortfolioService (Phase 2)
    Consumed by: RiskService (Phase 4), Analytics, Reporting

    Attributes:
        old_cash: Previous cash balance
        new_cash: New cash balance
        change_amount: Cash change (new - old)
        reason: Why cash changed
        metadata: Additional details

    Example:
        >>> event = CashChangedEvent(
        ...     old_cash=Decimal("100000"),
        ...     new_cash=Decimal("92500"),
        ...     change_amount=Decimal("-7500"),
        ...     reason="fill_buy"
        ... )
    """

    event_type: str = "cash_changed"
    old_cash: Decimal = Decimal("0")
    new_cash: Decimal = Decimal("0")
    change_amount: Decimal = Decimal("0")
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioStateEvent(Event):
    """
    Current portfolio state snapshot.

    Published by PortfolioService after valuation trigger.
    Contains all metrics needed for risk evaluation.

    Published by: PortfolioService
    Consumed by: RiskService (for risk checks)

    Attributes:
        ts: Timestamp of state snapshot
        total_equity: Total portfolio value (cash + positions)
        cash: Available cash
        positions_value: Total value of all positions
        num_positions: Number of open positions
        gross_exposure: Sum of abs(position_values)
        net_exposure: Sum of position_values (long - short)
        metadata: Additional state information

    Example:
        >>> event = PortfolioStateEvent(
        ...     ts=datetime(2020, 1, 2, 16, 0),
        ...     total_equity=Decimal("105000"),
        ...     cash=Decimal("50000"),
        ...     positions_value=Decimal("55000"),
        ...     num_positions=3,
        ...     gross_exposure=Decimal("55000"),
        ...     net_exposure=Decimal("55000")
        ... )
    """

    event_type: str = "portfolio_state"
    ts: datetime = field(default_factory=datetime.now)
    total_equity: Decimal = Decimal("0")
    cash: Decimal = Decimal("0")
    positions_value: Decimal = Decimal("0")
    num_positions: int = 0
    gross_exposure: Decimal = Decimal("0")
    net_exposure: Decimal = Decimal("0")
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================
# Risk Events
# ============================================


@dataclass(frozen=True)
class RiskViolationEvent(Event):
    """
    Risk limit violated.

    Published when a risk check fails (e.g., position size too large,
    insufficient margin, exceeded loss limit).

    Published by: RiskService (Phase 4)
    Consumed by: ExecutionService (to reject orders), Reporting

    Attributes:
        violation_type: Type of violation
        severity: How severe ('warning', 'error', 'critical')
        message: Human-readable description
        symbol: Affected symbol (if applicable)
        metadata: Additional violation details

    Example:
        >>> event = RiskViolationEvent(
        ...     violation_type="position_limit_exceeded",
        ...     severity="error",
        ...     message="Position size exceeds 10% of portfolio",
        ...     symbol="AAPL"
        ... )
    """

    event_type: str = "risk_violation"
    violation_type: str = ""
    severity: Literal["warning", "error", "critical"] = "error"
    message: str = ""
    symbol: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================
# Backtest Control Events
# ============================================


@dataclass(frozen=True)
class BacktestStartedEvent(Event):
    """
    Backtest started.

    Published at beginning of backtest run. Services can use this to
    initialize state, clear histories, etc.

    Published by: BacktestEngine (Phase 5)
    Consumed by: All services, Analytics, Reporting

    Attributes:
        backtest_id: Unique backtest run identifier
        start_date: Backtest start date
        end_date: Backtest end date
        config: Backtest configuration
    """

    event_type: str = "backtest_started"
    backtest_id: str = ""
    start_date: datetime | None = None
    end_date: datetime | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BacktestEndedEvent(Event):
    """
    Backtest ended.

    Published at end of backtest run. Services can use this to finalize
    calculations, generate reports, clean up resources.

    Published by: BacktestEngine (Phase 5)
    Consumed by: All services, Analytics, Reporting

    Attributes:
        backtest_id: Unique backtest run identifier
        total_bars: Total bars processed
        total_fills: Total fills executed
        duration_seconds: Backtest runtime
        metadata: Additional backtest statistics
    """

    event_type: str = "backtest_ended"
    backtest_id: str = ""
    total_bars: int = 0
    total_fills: int = 0
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BarCloseEvent(Event):
    """
    Bar closed (end of bar processing).

    Published when all processing for current bar is complete. Marks the
    boundary between bars and triggers mark-to-market, fee accrual, etc.

    Published by: BacktestEngine (Phase 5)
    Consumed by: PortfolioService (mark-to-market), Analytics

    Attributes:
        current_time: Current backtest time
        bar_number: Sequential bar number (1, 2, 3, ...)
        metadata: Additional bar details

    Example:
        >>> event = BarCloseEvent(
        ...     current_time=datetime(2020, 1, 2, 16, 0),
        ...     bar_number=1
        ... )
    """

    event_type: str = "bar_close"
    current_time: datetime | None = None
    bar_number: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================
# Risk Management Events (Phase 4)
# ============================================


@dataclass(frozen=True)
class ValuationTriggerEvent(Event):
    """
    Barrier event: trigger portfolio valuation for consistent snapshot.

    This is a CLOCK TICK BARRIER that marks "all market data for timestamp T
    has arrived". PortfolioService uses this to calculate equity, positions,
    and P&L atomically, creating a consistent snapshot for risk decisions.

    Phase Ordering:
        Phase 1: All PriceBarEvent for T published (market data complete)
        Phase 2: ValuationTriggerEvent published (BARRIER)
        Phase 3: PortfolioService calculates metrics using complete data

    Published by: BacktestEngine (after all PriceBarEvent for timestamp T)
    Consumed by: PortfolioService

    Why This Matters (Quant Gotchas):
    ---------------------------------
    Without this barrier:
        ❌ Risk Service calculates position size while bars still arriving
        ❌ Strategy A uses equity $100K, Strategy B uses equity $98K (race)
        ❌ Position limits violated because of inconsistent snapshot

    With this barrier:
        ✅ All services see same portfolio state (equity, positions, P&L)
        ✅ Risk decisions use consistent snapshot (no race conditions)
        ✅ Deterministic: same inputs → same outputs (for research)

    Implementation:
        Backtest: synchronous call, no coordination needed
        Production: async barrier (e.g., CountDownLatch pattern)

    Attributes:
        ts: Timestamp of the valuation (bar close time)

    Example:
        >>> # BacktestEngine publishes after all bars for T
        >>> event = ValuationTriggerEvent(
        ...     ts=datetime(2020, 1, 2, 16, 0)
        ... )
        >>> # PortfolioService calculates equity, all strategies see same value
    """

    event_type: str = "valuation_trigger"
    ts: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class RiskEvaluationTriggerEvent(Event):
    """
    Barrier event: trigger risk evaluation with cross-strategy netting.

    This is a CLOCK TICK BARRIER that marks "portfolio valuation complete,
    now evaluate ALL buffered signals in single batch". RiskService uses this
    to apply cross-strategy netting and check limits against consistent snapshot.

    Phase Ordering:
        Phase 1: All PriceBarEvent for T published (market data complete)
        Phase 2: ValuationTriggerEvent → Portfolio calculates equity
        Phase 3: RiskEvaluationTriggerEvent published (BARRIER)
        Phase 4: RiskService processes ALL signals with netting

    Published by: BacktestEngine (after ValuationTriggerEvent processed)
    Consumed by: RiskService

    Why This Matters (Quant Gotchas):
    ---------------------------------
    Without this barrier:
        ❌ Strategy A buys 100 AAPL + Strategy B sells 100 AAPL = 2 fills
           → double commissions, tax reporting complexity (wash sale)
        ❌ Signals processed individually → no netting opportunity
        ❌ Position limits checked multiple times → race conditions

    With this barrier:
        ✅ All signals buffered → barrier → net across strategies
        ✅ A buys 100 + B sells 100 = net 0 (no fill, no commission)
        ✅ Single batch evaluation → consistent limit checks
        ✅ Audit trail: all signals evaluated together (transparency)

    Cross-Strategy Netting Example:
        Signals buffered:
            - Strategy "momentum_v1": BUY 100 AAPL (strength=0.8)
            - Strategy "mean_revert": SELL 100 AAPL (strength=0.6)

        Without netting:
            → 2 orders: BUY 100 + SELL 100
            → 2 fills: commission on 200 shares
            → Net position: 0 (but paid 2x commissions)

        With netting (this event triggers):
            → Net signal: 0 AAPL (100 - 100)
            → 0 orders, 0 commissions
            → Same net position, lower costs

    Implementation:
        Backtest: synchronous call, processes buffer immediately
        Production: async barrier, bounded buffer (drop if >1000 signals)

    Attributes:
        ts: Timestamp of the evaluation (bar close time)

    Example:
        >>> # RiskService buffers signals during bar processing
        >>> # BacktestEngine publishes after valuation complete
        >>> event = RiskEvaluationTriggerEvent(
        ...     ts=datetime(2020, 1, 2, 16, 0)
        ... )
        >>> # RiskService: net signals, check limits, publish approved orders
    """

    event_type: str = "risk_evaluation_trigger"
    ts: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class OrderApprovedEvent(Event):
    """
    Risk-approved order ready for execution.

    Published by RiskService after successful risk checks.
    ExecutionService subscribes to execute approved orders.

    Attributes:
        ts: Timestamp of approval (bar close time)
        strategy_id: Strategy that generated the signal
        symbol: Instrument symbol
        side: BUY or SELL
        quantity: Number of shares/contracts (risk-determined)
        reason: Audit trail explaining approval decision

    Example:
        >>> event = OrderApprovedEvent(
        ...     ts=datetime(2020, 1, 2, 16, 0),
        ...     strategy_id="momentum_v1",
        ...     symbol="AAPL",
        ...     side="BUY",
        ...     quantity=500,
        ...     reason="Approved: 500 shares, 2% of allocated capital, within limits"
        ... )
    """

    event_type: str = "order_approved"
    ts: datetime = field(default_factory=datetime.now)
    strategy_id: str = ""
    symbol: str = ""
    side: Literal["BUY", "SELL"] = "BUY"
    quantity: int = 0
    reason: str = ""


@dataclass(frozen=True)
class OrderRejectedEvent(Event):
    """
    Signal rejected by RiskService.

    Published by RiskService when signal fails risk checks.
    Includes detailed reason for audit trail.

    Attributes:
        ts: Timestamp of rejection (bar close time)
        strategy_id: Strategy that generated the signal
        symbol: Instrument symbol
        side: BUY or SELL
        strength: Original signal strength
        reason: Detailed rejection reason (e.g., "Rejected: exceeds concentration limit")

    Example:
        >>> event = OrderRejectedEvent(
        ...     ts=datetime(2020, 1, 2, 16, 0),
        ...     strategy_id="momentum_v1",
        ...     symbol="AAPL",
        ...     side="BUY",
        ...     strength=0.75,
        ...     reason="Rejected: would exceed 10% concentration limit (current 8%, proposed 12%)"
        ... )
    """

    event_type: str = "order_rejected"
    ts: datetime = field(default_factory=datetime.now)
    strategy_id: str = ""
    symbol: str = ""
    side: Literal["BUY", "SELL"] = "BUY"
    strength: float = 0.0
    reason: str = ""
