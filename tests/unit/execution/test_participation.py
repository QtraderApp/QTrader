"""Tests for volume participation and partial fills (Stage 5)."""

from datetime import datetime
from decimal import Decimal

import pytest
import pytz

from qtrader.execution.commission import CommissionCalculator
from qtrader.execution.config import ExecutionConfig
from qtrader.execution.engine import ExecutionEngine
from qtrader.execution.fill_policy import FillPolicy
from qtrader.models.bar import Bar
from qtrader.models.order import Order, OrderSide, OrderState, OrderType, TimeInForce
from qtrader.models.portfolio import Portfolio

ET = pytz.timezone("US/Eastern")

# Test timestamps
BAR_TS_1 = datetime(2024, 1, 2, 9, 30, tzinfo=ET)
BAR_TS_2 = datetime(2024, 1, 2, 9, 31, tzinfo=ET)
BAR_TS_3 = datetime(2024, 1, 2, 9, 32, tzinfo=ET)
BAR_TS_4 = datetime(2024, 1, 2, 9, 33, tzinfo=ET)

# Additional timestamps for test_residual_expires_after_queue_bars
BAR_TS_DAY3 = datetime(2023, 1, 3, 16, 0, tzinfo=ET)
BAR_TS_DAY4 = datetime(2023, 1, 4, 16, 0, tzinfo=ET)
BAR_TS_DAY5 = datetime(2023, 1, 5, 16, 0, tzinfo=ET)
BAR_TS_DAY6 = datetime(2023, 1, 6, 16, 0, tzinfo=ET)
BAR_TS_DAY7 = datetime(2023, 1, 7, 16, 0, tzinfo=ET)


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
            trade_datetime=datetime(2023, 1, 3, 16, 0, tzinfo=ET).isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,  # Cap = 10% = 10,000 shares
        )

        next_bar = Bar(
            trade_datetime=datetime(2023, 1, 4, 16, 0, tzinfo=ET).isoformat(),
            open=151.00,
            high=153.00,
            low=150.00,
            close=152.00,
            volume=100_000,
        )

        # Submit order for 5,000 shares (well below cap)
        order = Order(
            order_id="order_1",
            strategy_ts=BAR_TS_1,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=5_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=BAR_TS_1,
        )

        engine.submit_order(order, BAR_TS_1)
        engine.on_bar(bar, symbol="AAPL", ts=BAR_TS_1, next_bar=next_bar)
        # Order should fill completely on next bar
        fills = engine.all_fills
        assert len(fills) == 1
        assert fills[0].qty == 5_000
        assert fills[0].participation == pytest.approx(0.05)  # 5% of volume

    def test_order_exceeds_participation_cap(self, engine):
        """Order quantity > participation cap results in partial fill."""
        bar = Bar(
            trade_datetime=datetime(2023, 1, 3, 16, 0, tzinfo=ET).isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,  # Cap = 10% = 10,000 shares
        )

        next_bar = Bar(
            trade_datetime=datetime(2023, 1, 4, 16, 0, tzinfo=ET).isoformat(),
            open=151.00,
            high=153.00,
            low=150.00,
            close=152.00,
            volume=100_000,
        )

        # Submit order for 25,000 shares (exceeds cap)
        order = Order(
            order_id="order_1",
            strategy_ts=BAR_TS_1,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=25_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=BAR_TS_1,
        )

        engine.submit_order(order, BAR_TS_1)
        engine.on_bar(bar, symbol="AAPL", ts=BAR_TS_1, next_bar=next_bar)
        # Should fill 10,000 shares (cap) on next bar
        fills = engine.all_fills
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
            trade_datetime=datetime(2023, 1, 3, 16, 0, tzinfo=ET).isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,  # Cap = 10,000
        )

        bar2 = Bar(
            trade_datetime=datetime(2023, 1, 4, 16, 0, tzinfo=ET).isoformat(),
            open=151.00,
            high=153.00,
            low=150.00,
            close=152.00,
            volume=100_000,
        )

        bar3 = Bar(
            trade_datetime=datetime(2023, 1, 5, 16, 0, tzinfo=ET).isoformat(),
            open=152.00,
            high=154.00,
            low=151.00,
            close=153.00,
            volume=100_000,
        )

        # Submit order for 25,000 shares
        order = Order(
            order_id="order_1",
            strategy_ts=BAR_TS_1,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=25_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=BAR_TS_1,
        )

        engine.submit_order(order, BAR_TS_1)
        engine.on_bar(bar1, symbol="AAPL", ts=BAR_TS_1, next_bar=bar2)
        # First fill: 10,000 shares
        assert len(engine.all_fills) == 1
        assert engine.all_fills[0].qty == 10_000

        # Bar 2: Another 10,000 shares
        engine.on_bar(bar2, symbol="AAPL", ts=BAR_TS_2, next_bar=bar3)
        # Second fill: 10,000 shares
        assert len(engine.all_fills) == 2
        assert engine.all_fills[1].qty == 10_000
        assert engine.all_fills[1].partial_index == 2

        # Bar 3: Final 5,000 shares
        bar4 = Bar(
            trade_datetime=datetime(2023, 1, 6, 16, 0, tzinfo=ET).isoformat(),
            open=153.00,
            high=155.00,
            low=152.00,
            close=154.00,
            volume=100_000,
        )

        engine.on_bar(bar3, symbol="AAPL", ts=BAR_TS_3, next_bar=bar4)
        # Third fill: 5,000 shares
        assert len(engine.all_fills) == 3
        assert engine.all_fills[2].qty == 5_000
        assert engine.all_fills[2].partial_index == 3

        # Order should be FILLED
        orders = engine.get_orders()
        assert orders[0].state == OrderState.FILLED
        assert orders[0].filled_qty == 25_000
        assert orders[0].remaining_qty == 0

    def test_residual_expires_after_queue_bars(self, engine):
        """Residual expires after queue_bars without filling."""
        # Create 5 bars (days 3-7)
        bar1 = Bar(
            trade_datetime=BAR_TS_DAY3.isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,  # Cap = 10,000
        )
        bar2 = Bar(
            trade_datetime=BAR_TS_DAY4.isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,
        )
        bar3 = Bar(
            trade_datetime=BAR_TS_DAY5.isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,
        )
        bar4 = Bar(
            trade_datetime=BAR_TS_DAY6.isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,
        )
        bar5 = Bar(
            trade_datetime=BAR_TS_DAY7.isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,
        )
        bars = [bar1, bar2, bar3, bar4, bar5]

        # Submit order for 50,000 shares
        order = Order(
            order_id="order_1",
            strategy_ts=BAR_TS_DAY3,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=50_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=BAR_TS_DAY3,
        )

        engine.submit_order(order, BAR_TS_DAY3)
        engine.on_bar(bars[0], symbol="AAPL", ts=BAR_TS_DAY3, next_bar=bars[1])
        # Fill 1: 10,000 shares
        assert len(engine.all_fills) == 1

        # Bars 2-4: Fill another 30,000 shares over 3 bars (queue_bars=3)
        engine.on_bar(bars[1], symbol="AAPL", ts=BAR_TS_DAY4, next_bar=bars[2])
        engine.on_bar(bars[2], symbol="AAPL", ts=BAR_TS_DAY5, next_bar=bars[3])
        engine.on_bar(bars[3], symbol="AAPL", ts=BAR_TS_DAY6, next_bar=bars[4])
        # Total fills: 40,000 shares (4 bars × 10,000)
        assert len(engine.all_fills) == 4
        total_filled = sum(f.qty for f in engine.all_fills)
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
            trade_datetime=datetime(2023, 1, 3, 16, 0, tzinfo=ET).isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=150.00,
            volume=100_000,
        )

        bar2 = Bar(
            trade_datetime=datetime(2023, 1, 4, 16, 0, tzinfo=ET).isoformat(),
            open=152.00,
            high=154.00,
            low=151.00,
            close=152.00,
            volume=100_000,
        )

        bar3 = Bar(
            trade_datetime=datetime(2023, 1, 5, 16, 0, tzinfo=ET).isoformat(),
            open=153.00,
            high=155.00,
            low=152.00,
            close=154.00,
            volume=100_000,
        )

        order = Order(
            order_id="order_1",
            strategy_ts=BAR_TS_1,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=20_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=BAR_TS_1,
        )

        engine.submit_order(order, BAR_TS_1)
        engine.on_bar(bar1, symbol="AAPL", ts=BAR_TS_1, next_bar=bar2)
        # Bar 2: Fill at $152 (bar2.open)
        engine.on_bar(bar2, symbol="AAPL", ts=BAR_TS_2, next_bar=bar3)
        # Average fill price should be: (10k * 152 + 10k * 153) / 20k = 152.50
        orders = engine.get_orders()
        assert orders[0].avg_fill_price == Decimal("152.50")

    def test_commissions_applied_per_partial(self, engine):
        """Commissions applied to each partial fill separately."""
        bar = Bar(
            trade_datetime=datetime(2023, 1, 3, 16, 0, tzinfo=ET).isoformat(),
            open=150.00,
            high=152.00,
            low=149.00,
            close=151.00,
            volume=100_000,  # Cap = 10,000
        )

        bar2 = Bar(
            trade_datetime=datetime(2023, 1, 4, 16, 0, tzinfo=ET).isoformat(),
            open=151.00,
            high=153.00,
            low=150.00,
            close=152.00,
            volume=100_000,
        )

        bar3 = Bar(
            trade_datetime=datetime(2023, 1, 5, 16, 0, tzinfo=ET).isoformat(),
            open=152.00,
            high=154.00,
            low=151.00,
            close=153.00,
            volume=100_000,
        )

        order = Order(
            order_id="order_1",
            strategy_ts=BAR_TS_1,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=20_000,
            order_type=OrderType.MARKET,
            state=OrderState.SUBMITTED,
            tif=TimeInForce.DAY,
            submission_bar_ts=BAR_TS_1,
        )

        engine.submit_order(order, BAR_TS_1)
        engine.on_bar(bar, symbol="AAPL", ts=BAR_TS_1, next_bar=bar2)
        # First partial fill: 10,000 shares
        fills = engine.all_fills
        assert len(fills) == 1

        # Commission: max(10,000 * 0.0005, 1.00) = max(5.00, 1.00) = 5.00
        assert fills[0].fees == Decimal("5.00")

        # Bar 2
        engine.on_bar(bar2, symbol="AAPL", ts=BAR_TS_2, next_bar=bar3)
        # Second partial fill: 10,000 shares
        fills = engine.all_fills  # Refresh fills list
        assert len(fills) == 2
        assert fills[1].fees == Decimal("5.00")

        # Total fees: $10.00
        total_fees = sum(f.fees for f in fills)
        assert total_fees == Decimal("10.00")
        # Second partial fill: 10,000 shares
        assert len(engine.all_fills) == 2
        assert fills[1].fees == Decimal("5.00")

        # Total fees: $10.00
        total_fees = sum(f.fees for f in engine.all_fills)
        assert total_fees == Decimal("10.00")


class TestPartialFillsWithLimitOrders:
    """Test partial fills with limit orders."""

    def test_limit_order_partial_fill(self, engine):
        """Limit order can be partially filled."""
        bar = Bar(
            trade_datetime=datetime(2023, 1, 3, 16, 0, tzinfo=ET).isoformat(),
            open=150.00,
            high=149.00,  # Touches limit
            low=148.00,
            close=151.00,
            volume=100_000,  # Cap = 10,000
        )

        # Limit buy at $149
        order = Order(
            order_id="order_1",
            strategy_ts=BAR_TS_1,
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=25_000,
            order_type=OrderType.LIMIT,
            state=OrderState.SUBMITTED,
            limit_price=Decimal("149.00"),
            tif=TimeInForce.DAY,
            submission_bar_ts=BAR_TS_1,
        )

        engine.submit_order(order, BAR_TS_1)
        engine.on_bar(bar, symbol="AAPL", ts=BAR_TS_1)
        # Should fill 10,000 shares at limit price
        fills = engine.all_fills
        assert len(fills) == 1
        assert fills[0].qty == 10_000
        assert fills[0].price == Decimal("149.00")

        # Order should be PARTIALLY_FILLED
        orders = engine.get_orders()
        assert orders[0].state == OrderState.PARTIALLY_FILLED
        assert orders[0].remaining_qty == 15_000
