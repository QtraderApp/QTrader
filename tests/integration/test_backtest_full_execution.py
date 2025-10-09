"""
Integration test: Full backtest with signal-to-order execution.

Tests complete trading loop:
- Strategy generates signals
- Risk manager evaluates and sizes
- Orders submitted to execution engine
- Fills applied to portfolio
- Portfolio state updated
"""

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.data import DataLoader, MultiModeBar, PriceSeriesIterator
from qtrader.execution.config import ExecutionConfig
from qtrader.models.portfolio import Portfolio
from qtrader.models.vendors.algoseek.bar import AlgoseekBar
from qtrader.models.vendors.algoseek.price_series import AlgoseekPriceSeries
from qtrader.risk import RiskManager, RiskPolicy, Signal, SignalDirection, SignalType
from qtrader.risk.sizing import SizingMethod


class SimpleBuyStrategy:
    """Strategy that buys on first bar, holds, then exits on last bar."""

    def __init__(self):
        self.bars_seen = 0
        self.total_bars = 0
        self.has_bought = False
        self.has_sold = False

    def on_init(self, ctx: Context) -> None:
        """Initialize strategy."""
        pass

    def on_start(self, ctx: Context) -> None:
        """Called after warmup."""
        pass

    def on_bar(self, bar: MultiModeBar, ctx: Context) -> Optional[List[Signal]]:
        """
        Buy on bar 1, sell on last bar.

        Args:
            bar: MultiModeBar with all adjustment modes
            ctx: Context (contains current_symbol)

        Returns:
            List of signals
        """
        self.bars_seen += 1

        # Use adjusted bar for strategy logic
        adjusted_bar = bar.adjusted
        symbol = bar.symbol

        # Parse timestamp from ISO string
        bar_ts = datetime.fromisoformat(adjusted_bar.trade_datetime)

        # Buy on first bar
        if self.bars_seen == 1 and not self.has_bought:
            self.has_bought = True
            return [
                Signal(
                    signal_id="buy_1",
                    strategy_ts=bar_ts,
                    symbol=symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    direction=SignalDirection.LONG,
                    conviction=Decimal("1.0"),
                )
            ]

        # Sell on last bar (bar 5)
        if self.bars_seen == 5 and not self.has_sold:
            self.has_sold = True
            return [
                Signal(
                    signal_id="sell_1",
                    strategy_ts=bar_ts,
                    symbol=symbol,
                    signal_type=SignalType.EXIT_LONG,
                    direction=SignalDirection.FLAT,
                    conviction=Decimal("1.0"),
                )
            ]

        return None

    def on_fill(self, fill, ctx: Context) -> None:
        """Called after fill."""
        pass

    def on_end(self, ctx: Context) -> None:
        """Called at end."""
        pass


def create_test_iterators(symbol: str, count: int, start_price: float = 100.0) -> Dict[str, PriceSeriesIterator]:
    """
    Create test data iterators with realistic OHLC.

    Args:
        symbol: Symbol for bars
        count: Number of bars to create
        start_price: Starting price

    Returns:
        Dict mapping symbol to PriceSeriesIterator
    """
    vendor_bars = []
    base_date = datetime(2024, 1, 1, 9, 30)
    price = start_price

    for i in range(count):
        open_price = price
        high_price = price + 1.0
        low_price = price - 1.0
        close_price = price + 0.5  # Small gain each bar

        vendor_bar = AlgoseekBar(
            Ticker=symbol,
            TradeDate=base_date + timedelta(days=i),  # datetime, not ISO string
            Open=open_price,
            High=high_price,
            Low=low_price,
            Close=close_price,
            MarketHoursVolume=1000000,
            # No adjustments (unadjusted = adjusted for this test)
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
        )
        vendor_bars.append(vendor_bar)
        price = close_price

    # Create price series and iterator
    loader = DataLoader({})
    price_series = AlgoseekPriceSeries(symbol=symbol, bars=vendor_bars)
    iterator = loader.load_data_from_series(price_series)

    return {symbol: iterator}


class TestBacktestFullExecution:
    """Test full backtest execution with trading."""

    def test_simple_buy_and_sell(self):
        """Test simple buy then sell strategy."""
        # Setup
        strategy = SimpleBuyStrategy()
        config = ExecutionConfig(warmup=False)
        portfolio = Portfolio(initial_cash=Decimal("100000"))

        # Create risk manager with policy
        policy = RiskPolicy(
            sizing_method=SizingMethod.FIXED_VALUE,
            default_position_size=Decimal("1000"),  # $1000 per position
            max_position_pct=Decimal("0.10"),  # 10% max per position
        )
        risk_manager = RiskManager(portfolio=portfolio, policy=policy)

        # Create context with both portfolio and risk manager
        ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

        # Create 5 bars using iterator approach
        data_iterators = create_test_iterators("AAPL", count=5, start_price=100.0)

        # Run backtest
        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, data_iterators, ["AAPL"], out_dir=Path("/tmp"))

        # Verify backtest completed
        assert metadata["total_bars"] == 5
        assert metadata["trading_bars"] == 5

        # Verify fills occurred
        assert metadata["total_fills"] > 0, "Expected at least one fill"

        # Verify strategy received signals
        assert strategy.has_bought, "Strategy should have bought"
        assert strategy.has_sold, "Strategy should have sold"

        # Verify portfolio changed (cash should be different from initial)
        final_cash = Decimal(str(metadata["final_cash"]))

        # Note: Cash might be slightly less due to commissions
        # But we should have traded (not equal to initial cash)
        assert final_cash != Decimal("100000"), "Cash should have changed due to trading"

    def test_rejected_signal_no_cash(self):
        """Test that signals are rejected when insufficient cash."""
        # Setup with minimal cash
        strategy = SimpleBuyStrategy()
        config = ExecutionConfig(warmup=False)
        portfolio = Portfolio(initial_cash=Decimal("10"))  # Very low cash

        # Risk manager with normal policy
        policy = RiskPolicy(
            sizing_method=SizingMethod.FIXED_VALUE,
            default_position_size=Decimal("10000"),  # Would require $10k
            max_position_pct=Decimal("0.10"),
        )
        risk_manager = RiskManager(portfolio=portfolio, policy=policy)
        ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

        # Create bars using iterator approach
        data_iterators = create_test_iterators("AAPL", count=5)

        # Run backtest
        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, data_iterators, ["AAPL"], out_dir=Path("/tmp"))

        # Should complete but with no fills
        assert metadata["total_fills"] == 0, "Should have no fills due to insufficient cash"
        assert metadata["final_cash"] == 10.0, "Cash should be unchanged"

    def test_portfolio_state_after_fill(self):
        """Test portfolio state is correctly updated after fills."""
        # Setup
        strategy = SimpleBuyStrategy()
        config = ExecutionConfig(warmup=False)
        portfolio = Portfolio(initial_cash=Decimal("100000"))

        policy = RiskPolicy(
            sizing_method=SizingMethod.FIXED_VALUE,
            default_position_size=Decimal("5000"),
            max_position_pct=Decimal("0.10"),
        )
        risk_manager = RiskManager(portfolio=portfolio, policy=policy)
        ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

        # Create bars (need one extra bar for sell order to fill)
        data_iterators = create_test_iterators("AAPL", count=6, start_price=100.0)

        # Run backtest
        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, data_iterators, ["AAPL"], out_dir=Path("/tmp"))

        # Verify portfolio has position history
        position = portfolio.positions.get_position("AAPL")

        # After buy and sell, position should be reduced or flat
        # Market sell on bar 5 fills at open of bar 6
        assert metadata["total_fills"] >= 2, "Should have buy and sell fills"

        # Position should have some realized PnL from trading
        # (Could be positive or negative depending on price movement and commissions)
        assert position.realized_pnl != Decimal("0") or position.qty < 49, "Should have traded"

    def test_execution_metadata(self):
        """Test that execution metadata is returned."""
        # Setup
        strategy = SimpleBuyStrategy()
        config = ExecutionConfig(warmup=False)
        portfolio = Portfolio(initial_cash=Decimal("100000"))

        policy = RiskPolicy(max_position_pct=Decimal("0.10"))
        risk_manager = RiskManager(portfolio=portfolio, policy=policy)
        ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

        # Create bars using iterator approach
        data_iterators = create_test_iterators("AAPL", count=5)

        # Run backtest
        backtest = Backtest(config, strategy)
        metadata = backtest.run(ctx, data_iterators, ["AAPL"], out_dir=Path("/tmp"))

        # Verify execution metadata exists
        assert "execution" in metadata, "Should have execution metadata"
        assert "pending_orders" in metadata["execution"]
        assert "filled_orders" in metadata["execution"]

    def test_portfolio_snapshots_created(self):
        """Test that portfolio snapshots are captured."""
        # Setup
        strategy = SimpleBuyStrategy()
        config = ExecutionConfig(warmup=False)
        portfolio = Portfolio(initial_cash=Decimal("100000"))

        policy = RiskPolicy(max_position_pct=Decimal("0.10"))
        risk_manager = RiskManager(portfolio=portfolio, policy=policy)
        ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

        # Create bars using iterator approach
        data_iterators = create_test_iterators("AAPL", count=5)

        # Run backtest
        backtest = Backtest(config, strategy)
        _ = backtest.run(ctx, data_iterators, ["AAPL"], out_dir=Path("/tmp"))

        # Verify snapshots were created
        assert len(backtest.portfolio_snapshots) > 0, "Should have portfolio snapshots"

        # Verify snapshot structure (new 28-field format)
        snapshot = backtest.portfolio_snapshots[0]
        assert "timestamp" in snapshot
        assert "end_cash" in snapshot
        assert "total_value" in snapshot
        assert "num_positions" in snapshot
