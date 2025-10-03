"""Tests for volume participation and partial fills (Stage 5)."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.execution.commission import CommissionCalculator
from qtrader.execution.config import ExecutionConfig
from qtrader.execution.engine import ExecutionEngine
from qtrader.execution.fill_policy import FillPolicy
from qtrader.models.bar import Bar
from qtrader.models.order import Order, OrderSide, OrderState, OrderType, TimeInForce
from qtrader.models.portfolio import Portfolio


@pytest.fixture
def config_10pct_participation():
    """Config with 10% max participation."""
    return ExecutionConfig(
        per_share=Decimal("0.0005"),
        ticket_min=Decimal("1.00"),
        max_participation=Decimal("0.10"),  # 10% max
        queue_bars=3,  # Keep residuals for 3 bars
        allow_high_participation=False,
    )


@pytest.fixture
def config_high_participation():
    """Config allowing high participation (>20%)."""
    return ExecutionConfig(
        per_share=Decimal("0.0005"),
        ticket_min=Decimal("1.00"),
        max_participation=Decimal("0.25"),  # 25% max
        queue_bars=3,
        allow_high_participation=True,  # Explicitly allow high participation
    )


@pytest.fixture
def portfolio():
    """Portfolio with cash."""
    return Portfolio(initial_cash=Decimal("10000000"))  # $10M for testing large orders


@pytest.fixture
def engine(config_10pct_participation, portfolio):
    """Execution engine with 10% participation cap."""
    policy = FillPolicy(
        limit_mode=config_10pct_participation.limit_mode,
        stop_mode=config_10pct_participation.stop_mode,
    )
    commission = CommissionCalculator(
        per_share=config_10pct_participation.per_share,
        ticket_min=config_10pct_participation.ticket_min,
    )
    return ExecutionEngine(
        portfolio=portfolio,
        fill_policy=policy,
        commission=commission,
        config=config_10pct_participation,
    )


class TestVolumeParticipation:
    """Test volume participation capping."""

    def test_order_fits_within_participation(self, engine):
        """Order quantity < participation cap fills completely."""
        bar = Bar(
            ts=datetime(2023, 1, 3, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=100_000,  # Cap = 10% = 10,000 shares
        )

        next_bar = Bar(
            ts=datetime(2023, 1, 4, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("151.00"),
            high=Decimal("153.00"),
            low=Decimal("150.00"),
            close=Decimal("152.00"),
            volume=100_000,
        )

        # Submit order for 5,000 shares (well below cap)
        order = Order(
            order_id="order_1",
            strategy_ts=bar.ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=5_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=bar.ts,
        )

        engine.submit_order(order, bar.ts)
        engine.evaluate_orders(bar, next_bar)
        engine.end_of_bar(bar)

        # Order should fill completely on next bar
        fills = engine.get_fills()
        assert len(fills) == 1
        assert fills[0].qty == 5_000
        assert fills[0].participation == pytest.approx(0.05)  # 5% of volume

    def test_order_exceeds_participation_cap(self, engine):
        """Order quantity > participation cap results in partial fill."""
        bar = Bar(
            ts=datetime(2023, 1, 3, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=100_000,  # Cap = 10% = 10,000 shares
        )

        next_bar = Bar(
            ts=datetime(2023, 1, 4, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("151.00"),
            high=Decimal("153.00"),
            low=Decimal("150.00"),
            close=Decimal("152.00"),
            volume=100_000,
        )

        # Submit order for 25,000 shares (exceeds cap)
        order = Order(
            order_id="order_1",
            strategy_ts=bar.ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=25_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=bar.ts,
        )

        engine.submit_order(order, bar.ts)
        engine.evaluate_orders(bar, next_bar)
        engine.end_of_bar(bar)

        # Should fill 10,000 shares (cap) on next bar
        fills = engine.get_fills()
        assert len(fills) == 1
        assert fills[0].qty == 10_000
        assert fills[0].participation == pytest.approx(0.10)
        assert fills[0].partial_index == 1

        # Order should be PARTIALLY_FILLED with 15,000 remaining
        orders = engine.get_orders()
        assert len(orders) == 1
        assert orders[0].state == OrderState.PARTIALLY_FILLED
        assert orders[0].filled_qty == 10_000
        assert orders[0].remaining_qty == 15_000


class TestResidualQueuing:
    """Test residual queuing and expiration."""

    def test_residual_fills_over_multiple_bars(self, engine):
        """Residual quantity fills over subsequent bars."""
        bar1 = Bar(
            ts=datetime(2023, 1, 3, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=100_000,  # Cap = 10,000
        )

        bar2 = Bar(
            ts=datetime(2023, 1, 4, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("151.00"),
            high=Decimal("153.00"),
            low=Decimal("150.00"),
            close=Decimal("152.00"),
            volume=100_000,
        )

        bar3 = Bar(
            ts=datetime(2023, 1, 5, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("152.00"),
            high=Decimal("154.00"),
            low=Decimal("151.00"),
            close=Decimal("153.00"),
            volume=100_000,
        )

        # Submit order for 25,000 shares
        order = Order(
            order_id="order_1",
            strategy_ts=bar1.ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=25_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=bar1.ts,
        )

        engine.submit_order(order, bar1.ts)
        engine.evaluate_orders(bar1, bar2)
        engine.end_of_bar(bar1)

        # First fill: 10,000 shares
        assert len(engine.get_fills()) == 1
        assert engine.get_fills()[0].qty == 10_000

        # Bar 2: Another 10,000 shares
        engine.evaluate_orders(bar2, bar3)
        engine.end_of_bar(bar2)

        # Second fill: 10,000 shares
        assert len(engine.get_fills()) == 2
        assert engine.get_fills()[1].qty == 10_000
        assert engine.get_fills()[1].partial_index == 2

        # Bar 3: Final 5,000 shares
        bar4 = Bar(
            ts=datetime(2023, 1, 6, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("153.00"),
            high=Decimal("155.00"),
            low=Decimal("152.00"),
            close=Decimal("154.00"),
            volume=100_000,
        )

        engine.evaluate_orders(bar3, bar4)
        engine.end_of_bar(bar3)

        # Third fill: 5,000 shares
        assert len(engine.get_fills()) == 3
        assert engine.get_fills()[2].qty == 5_000
        assert engine.get_fills()[2].partial_index == 3

        # Order should be FILLED
        orders = engine.get_orders()
        assert orders[0].state == OrderState.FILLED
        assert orders[0].filled_qty == 25_000
        assert orders[0].remaining_qty == 0

    def test_residual_expires_after_queue_bars(self, engine):
        """Residual expires after queue_bars without filling."""
        bars = []
        for day in range(3, 8):  # Create 5 bars
            bar = Bar(
                ts=datetime(2023, 1, day, 16, 0, tzinfo=timezone.utc),
                symbol="AAPL",
                open=Decimal("150.00"),
                high=Decimal("152.00"),
                low=Decimal("149.00"),
                close=Decimal("151.00"),
                volume=100_000,  # Cap = 10,000
            )
            bars.append(bar)

        # Submit order for 50,000 shares
        order = Order(
            order_id="order_1",
            strategy_ts=bars[0].ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=50_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=bars[0].ts,
        )

        engine.submit_order(order, bars[0].ts)
        engine.evaluate_orders(bars[0], bars[1])
        engine.end_of_bar(bars[0])

        # Fill 1: 10,000 shares
        assert len(engine.get_fills()) == 1

        # Bars 2-4: Fill another 30,000 shares over 3 bars (queue_bars=3)
        for i in range(1, 4):
            next_bar = bars[i + 1] if i + 1 < len(bars) else None
            engine.evaluate_orders(bars[i], next_bar)
            engine.end_of_bar(bars[i])

        # Total fills: 40,000 shares (4 bars × 10,000)
        assert len(engine.get_fills()) == 4
        total_filled = sum(f.qty for f in engine.get_fills())
        assert total_filled == 40_000

        # Order should be EXPIRED with 10,000 remaining
        orders = engine.get_orders()
        assert orders[0].state == OrderState.EXPIRED
        assert orders[0].filled_qty == 40_000
        assert orders[0].remaining_qty == 10_000


class TestHighParticipationGuardrail:
    """Test high participation guardrail."""

    def test_warns_and_clamps_high_participation(self, engine):
        """High participation > 20% warns and clamps to 20% when not allowed."""
        # Try to set max_participation to 30% (> 20%)
        config = ExecutionConfig(
            max_participation=Decimal("0.30"),  # 30% > 20% threshold
            allow_high_participation=False,  # Not allowed
            queue_bars=3,
        )

        # Engine should warn and clamp to 0.20
        policy = FillPolicy(limit_mode="conservative", stop_mode="conservative")
        commission = CommissionCalculator(per_share=Decimal("0.0005"), ticket_min=Decimal("1.00"))
        portfolio = Portfolio(initial_cash=Decimal("10000000"))

        engine = ExecutionEngine(
            portfolio=portfolio,
            fill_policy=policy,
            commission=commission,
            config=config,
        )

        # Check that effective participation is clamped
        assert engine.config.max_participation == Decimal("0.20")

    def test_allows_high_participation_when_explicitly_enabled(self, config_high_participation, portfolio):
        """High participation allowed when explicitly enabled."""
        policy = FillPolicy(limit_mode="conservative", stop_mode="conservative")
        commission = CommissionCalculator(per_share=Decimal("0.0005"), ticket_min=Decimal("1.00"))

        engine = ExecutionEngine(
            portfolio=portfolio,
            fill_policy=policy,
            commission=commission,
            config=config_high_participation,
        )

        # Should allow 25% participation
        assert engine.config.max_participation == Decimal("0.25")


class TestPartialFillAccounting:
    """Test accounting for partial fills."""

    def test_avg_fill_price_calculated_correctly(self, engine):
        """Average fill price calculated correctly across partials."""
        # Bar 1: Fill at $150
        bar1 = Bar(
            ts=datetime(2023, 1, 3, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("150.00"),
            volume=100_000,
        )

        bar2 = Bar(
            ts=datetime(2023, 1, 4, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("152.00"),
            high=Decimal("154.00"),
            low=Decimal("151.00"),
            close=Decimal("152.00"),
            volume=100_000,
        )

        bar3 = Bar(
            ts=datetime(2023, 1, 5, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("153.00"),
            high=Decimal("155.00"),
            low=Decimal("152.00"),
            close=Decimal("154.00"),
            volume=100_000,
        )

        order = Order(
            order_id="order_1",
            strategy_ts=bar1.ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=20_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=bar1.ts,
        )

        engine.submit_order(order, bar1.ts)
        engine.evaluate_orders(bar1, bar2)
        engine.end_of_bar(bar1)

        # Bar 2: Fill at $152 (bar2.open)
        engine.evaluate_orders(bar2, bar3)
        engine.end_of_bar(bar2)

        # Average fill price should be: (10k * 152 + 10k * 153) / 20k = 152.50
        orders = engine.get_orders()
        assert orders[0].avg_fill_price == Decimal("152.50")

    def test_commissions_applied_per_partial(self, engine):
        """Commissions applied to each partial fill separately."""
        bar = Bar(
            ts=datetime(2023, 1, 3, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.00"),
            volume=100_000,  # Cap = 10,000
        )

        bar2 = Bar(
            ts=datetime(2023, 1, 4, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("151.00"),
            high=Decimal("153.00"),
            low=Decimal("150.00"),
            close=Decimal("152.00"),
            volume=100_000,
        )

        bar3 = Bar(
            ts=datetime(2023, 1, 5, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("152.00"),
            high=Decimal("154.00"),
            low=Decimal("151.00"),
            close=Decimal("153.00"),
            volume=100_000,
        )

        order = Order(
            order_id="order_1",
            strategy_ts=bar.ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=20_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=bar.ts,
        )

        engine.submit_order(order, bar.ts)
        engine.evaluate_orders(bar, bar2)
        engine.end_of_bar(bar)

        # First partial fill: 10,000 shares
        fills = engine.get_fills()
        assert len(fills) == 1

        # Commission: max(10,000 * 0.0005, 1.00) = max(5.00, 1.00) = 5.00
        assert fills[0].fees == Decimal("5.00")

        # Bar 2
        engine.evaluate_orders(bar2, bar3)
        engine.end_of_bar(bar2)

        # Second partial fill: 10,000 shares
        fills = engine.get_fills()  # Refresh fills list
        assert len(fills) == 2
        assert fills[1].fees == Decimal("5.00")

        # Total fees: $10.00
        total_fees = sum(f.fees for f in fills)
        assert total_fees == Decimal("10.00")
        engine.end_of_bar(bar2)

        # Second partial fill: 10,000 shares
        assert len(engine.get_fills()) == 2
        assert fills[1].fees == Decimal("5.00")

        # Total fees: $10.00
        total_fees = sum(f.fees for f in engine.get_fills())
        assert total_fees == Decimal("10.00")


class TestPartialFillsWithLimitOrders:
    """Test partial fills with limit orders."""

    def test_limit_order_partial_fill(self, engine):
        """Limit order can be partially filled."""
        bar = Bar(
            ts=datetime(2023, 1, 3, 16, 0, tzinfo=timezone.utc),
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("149.00"),  # Touches limit
            low=Decimal("148.00"),
            close=Decimal("151.00"),
            volume=100_000,  # Cap = 10,000
        )

        # Limit buy at $149
        order = Order(
            order_id="order_1",
            strategy_ts=bar.ts,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=25_000,
            order_type=OrderType.LIMIT,
            state=OrderState.SUBMITTED,
            limit_price=Decimal("149.00"),
            tif=TimeInForce.DAY,
            submission_bar_ts=bar.ts,
        )

        engine.submit_order(order, bar.ts)
        engine.evaluate_orders(bar)
        engine.end_of_bar(bar)

        # Should fill 10,000 shares at limit price
        fills = engine.get_fills()
        assert len(fills) == 1
        assert fills[0].qty == 10_000
        assert fills[0].price == Decimal("149.00")

        # Order should be PARTIALLY_FILLED
        orders = engine.get_orders()
        assert orders[0].state == OrderState.PARTIALLY_FILLED
        assert orders[0].remaining_qty == 15_000
