"""Integration tests for dividend processing in backtests."""

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.api.strategy import Strategy
from qtrader.execution.config import ExecutionConfig
from qtrader.models.bar import AdjustmentEvent, Bar
from qtrader.models.order import OrderSide
from qtrader.models.portfolio import Portfolio
from qtrader.risk.manager import RiskManager
from qtrader.risk.policy import RiskPolicy


class SimpleStrategy(Strategy):
    """Simple strategy for testing dividend processing."""

    def on_init(self, ctx: Context):
        """Initialize strategy."""
        pass

    def on_bar(self, bar: Bar, ctx: Context):
        """Process bar - just hold positions."""
        pass


class TestBacktestDividends:
    """Test suite for dividend processing in backtests."""

    @pytest.fixture
    def portfolio(self):
        """Create portfolio with initial cash."""
        return Portfolio(initial_cash=Decimal("100000"))

    @pytest.fixture
    def context(self, portfolio):
        """Create context with portfolio."""
        policy = RiskPolicy()
        risk_manager = RiskManager(portfolio=portfolio, policy=policy)
        return Context(risk_manager=risk_manager, portfolio=portfolio)

    @pytest.fixture
    def config(self):
        """Create execution config."""
        return ExecutionConfig(warmup=False)

    @pytest.fixture
    def strategy(self):
        """Create simple strategy."""
        return SimpleStrategy()

    def test_backtest_without_adjustment_events(self, config, strategy, context):
        """Test backtest runs without adjustment events (backward compatibility)."""
        backtest = Backtest(config, strategy)

        # Create simple bar data
        bars = [
            Bar(
                ts=datetime(2023, 2, 10, 9, 30, tzinfo=timezone.utc),
                symbol="AAPL",
                open=Decimal("150.00"),
                high=Decimal("151.00"),
                low=Decimal("149.00"),
                close=Decimal("150.50"),
                volume=1000000,
            )
        ]

        result = backtest.run(
            ctx=context,
            bars=bars,
            symbols=["AAPL"],
            out_dir=Path("/tmp"),
        )

        assert result["total_bars"] == 1
        assert result["trading_bars"] == 1
        assert "dividends" not in result  # No dividend processing

    def test_backtest_with_adjustment_events_no_positions(self, config, strategy, context):
        """Test backtest with adjustment events but no short positions."""
        backtest = Backtest(config, strategy)

        # Create bar data
        ex_date = datetime(2023, 2, 10, 9, 30, tzinfo=timezone.utc)
        bars = [
            Bar(
                ts=ex_date,
                symbol="AAPL",
                open=Decimal("150.00"),
                high=Decimal("151.00"),
                low=Decimal("149.00"),
                close=Decimal("150.50"),
                volume=1000000,
            )
        ]

        # Create adjustment events (timestamp must match bar)
        adjustment_events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=ex_date,  # Same timestamp as bar
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={
                        "close_before": "152.55",
                        "close_after": "152.32",
                    },
                )
            ]
        }

        result = backtest.run(
            ctx=context,
            bars=bars,
            symbols=["AAPL"],
            out_dir=Path("/tmp"),
            adjustment_events=adjustment_events,
        )

        assert result["total_bars"] == 1
        assert result["trading_bars"] == 1
        assert "dividends" in result
        assert result["dividends"]["total_events"] == 1
        assert result["dividends"]["processed_count"] == 0  # No short positions
        assert result["dividends"]["skipped_count"] == 1

    def test_backtest_with_short_position_and_dividend(self, config, strategy, context, portfolio):
        """Test backtest processes dividend on short position."""
        # Create short position before backtest
        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=100,
            fill_price=Decimal("150.00"),
            commission=Decimal("0"),
            ts=datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc),
            order_id="order-1",
            fill_id="fill-1",
        )

        initial_cash = portfolio.cash.get_balance()
        backtest = Backtest(config, strategy)

        # Create bar data (on ex-date)
        bars = [
            Bar(
                ts=datetime(2023, 2, 10, 9, 30, tzinfo=timezone.utc),
                symbol="AAPL",
                open=Decimal("150.00"),
                high=Decimal("151.00"),
                low=Decimal("149.00"),
                close=Decimal("150.50"),
                volume=1000000,
            )
        ]

        # Create adjustment events (ex-date matches bar date)
        adjustment_events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=datetime(2023, 2, 10, 9, 30, tzinfo=timezone.utc),
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={
                        "close_before": "152.55",
                        "close_after": "152.32",
                    },
                )
            ]
        }

        result = backtest.run(
            ctx=context,
            bars=bars,
            symbols=["AAPL"],
            out_dir=Path("/tmp"),
            adjustment_events=adjustment_events,
        )

        assert result["total_bars"] == 1
        assert "dividends" in result
        assert result["dividends"]["processed_count"] == 1
        assert result["dividends"]["skipped_count"] == 0

        # Verify dividend was debited
        # 100 shares * $0.23 = $23.00
        expected_cash = initial_cash - Decimal("23.00")
        assert portfolio.cash.get_balance() == expected_cash

    def test_backtest_multiple_dividends_same_date(self, config, strategy, context, portfolio):
        """Test backtest processes multiple dividends on same date."""
        ts = datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc)

        # Create short positions
        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=100,
            fill_price=Decimal("150.00"),
            commission=Decimal("0"),
            ts=ts,
            order_id="order-1",
            fill_id="fill-1",
        )
        portfolio.apply_fill(
            symbol="MSFT",
            side=OrderSide.SELL,
            qty=50,
            fill_price=Decimal("250.00"),
            commission=Decimal("0"),
            ts=ts,
            order_id="order-2",
            fill_id="fill-2",
        )

        initial_cash = portfolio.cash.get_balance()
        backtest = Backtest(config, strategy)

        # Create bar data
        ex_date = datetime(2023, 2, 10, 9, 30, tzinfo=timezone.utc)
        bars = [
            Bar(
                ts=ex_date,
                symbol="AAPL",
                open=Decimal("150.00"),
                high=Decimal("151.00"),
                low=Decimal("149.00"),
                close=Decimal("150.50"),
                volume=1000000,
            ),
            Bar(
                ts=ex_date,
                symbol="MSFT",
                open=Decimal("250.00"),
                high=Decimal("251.00"),
                low=Decimal("249.00"),
                close=Decimal("250.50"),
                volume=500000,
            ),
        ]

        # Create adjustment events
        adjustment_events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=ex_date,
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={
                        "close_before": "152.55",
                        "close_after": "152.32",
                    },
                )
            ],
            "MSFT": [
                AdjustmentEvent(
                    ts=ex_date,
                    symbol="MSFT",
                    event_type="CashDiv",
                    px_factor=Decimal("1.002481"),
                    vol_factor=Decimal("1.0"),
                    metadata={
                        "close_before": "250.00",
                        "close_after": "249.38",
                    },
                )
            ],
        }

        result = backtest.run(
            ctx=context,
            bars=bars,
            symbols=["AAPL", "MSFT"],
            out_dir=Path("/tmp"),
            adjustment_events=adjustment_events,
        )

        assert "dividends" in result
        assert result["dividends"]["processed_count"] == 2
        assert result["dividends"]["total_symbols"] == 2

        # Verify dividends were debited
        # AAPL: 100 * $0.23 = $23.00
        # MSFT: 50 * $0.62 = $31.00
        # Total: $54.00
        expected_cash = initial_cash - Decimal("54.00")
        assert portfolio.cash.get_balance() == expected_cash

    def test_backtest_dividend_processor_statistics(self, config, strategy, context, portfolio):
        """Test dividend processor statistics are returned correctly."""
        # Create one short, one long position
        ts = datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc)

        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.SELL,
            qty=100,
            fill_price=Decimal("150.00"),
            commission=Decimal("0"),
            ts=ts,
            order_id="order-1",
            fill_id="fill-1",
        )
        portfolio.apply_fill(
            symbol="MSFT",
            side=OrderSide.BUY,
            qty=50,
            fill_price=Decimal("250.00"),
            commission=Decimal("0"),
            ts=ts,
            order_id="order-2",
            fill_id="fill-2",
        )

        backtest = Backtest(config, strategy)

        # Create bar data
        ex_date = datetime(2023, 2, 10, 9, 30, tzinfo=timezone.utc)
        bars = [
            Bar(
                ts=ex_date,
                symbol="AAPL",
                open=Decimal("150.00"),
                high=Decimal("151.00"),
                low=Decimal("149.00"),
                close=Decimal("150.50"),
                volume=1000000,
            ),
        ]

        # Both have dividends, but only AAPL (short) should be processed
        adjustment_events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=ex_date,
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={
                        "close_before": "152.55",
                        "close_after": "152.32",
                    },
                )
            ],
            "MSFT": [
                AdjustmentEvent(
                    ts=ex_date,
                    symbol="MSFT",
                    event_type="CashDiv",
                    px_factor=Decimal("1.002481"),
                    vol_factor=Decimal("1.0"),
                    metadata={
                        "close_before": "250.00",
                        "close_after": "249.38",
                    },
                )
            ],
        }

        result = backtest.run(
            ctx=context,
            bars=bars,
            symbols=["AAPL", "MSFT"],
            out_dir=Path("/tmp"),
            adjustment_events=adjustment_events,
        )

        assert "dividends" in result
        stats = result["dividends"]

        assert stats["total_symbols"] == 2
        assert stats["total_events"] == 2
        assert stats["unique_ex_dates"] == 1
        assert stats["processed_count"] == 1  # Only AAPL (short)
        assert stats["skipped_count"] == 1  # MSFT (long)
        assert stats["success_rate"] == 0.5
