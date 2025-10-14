"""Test split accounting with Phase 2 architecture (unadjusted execution)."""

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytz

from qtrader.api.backtest import Backtest
from qtrader.api.context import Context
from qtrader.api.strategy import Strategy
from qtrader.data.iterator import PriceSeriesIterator
from qtrader.execution.config import ExecutionConfig
from qtrader.models.bar import Bar, PriceSeries
from qtrader.models.multi_bar import MultiBar
from qtrader.models.portfolio import Portfolio
from qtrader.risk import RiskManager, RiskPolicy, SizingMethod
from qtrader.risk.signal import Signal, SignalDirection, SignalType

ET = pytz.timezone("US/Eastern")


class SimpleBuyHoldSellStrategy(Strategy):
    """Simple strategy: buy on first bar, hold, sell on last bar."""

    def __init__(self):
        super().__init__()
        self.buy_executed = False
        self.sell_executed = False
        self.bar_count = 0

    def on_bar(self, bar: MultiBar, ctx: Context):
        self.bar_count += 1

        # Buy on first bar
        if self.bar_count == 1 and not self.buy_executed:
            self.buy_executed = True
            return [
                Signal(
                    signal_id=f"buy_{bar.symbol}_{self.bar_count}",
                    strategy_ts=datetime.fromisoformat(bar.trade_datetime),
                    symbol=bar.symbol,
                    signal_type=SignalType.ENTRY_LONG,
                    direction=SignalDirection.LONG,
                    target_qty=1,  # Buy 1 share (pre-split)
                )
            ]

        # Sell on bar 4 (after split on bar 3, with bar 5 available for execution)
        if self.bar_count == 4 and not self.sell_executed:
            self.sell_executed = True
            # Get current position
            position = ctx.portfolio.positions.get_position(bar.symbol)
            if position and not position.is_flat():
                return [
                    Signal(
                        signal_id=f"sell_{bar.symbol}_{self.bar_count}",
                        strategy_ts=datetime.fromisoformat(bar.trade_datetime),
                        symbol=bar.symbol,
                        signal_type=SignalType.EXIT_LONG,
                        direction=SignalDirection.FLAT,
                        target_qty=position.qty,  # Sell all (should be 4 after split)
                    )
                ]

        return []


def test_split_accounting_with_unadjusted_execution():
    """
    Test that split accounting works correctly with unadjusted execution.

    Timeline:
    - Bar 1: Buy 1 share @ $500 (unadjusted)
    - Bar 2: Dividend $0.82 (unadjusted) × 1 share = $0.82
    - Bar 3: Split 4:1 (position should become 4 shares @ $125 cost basis)
    - Bar 4: Hold
    - Bar 5: Sell 4 shares @ $130 (unadjusted post-split)

    Expected:
    - Buy: -$500
    - Dividend: +$0.82
    - Split: 1 → 4 shares, cost basis $500 → $125/share
    - Sell: +$520 (4 × $130)
    - Profit: ~$20 (ignoring small commissions)
    """
    # Create bars simulating AAPL around 4:1 split
    # Unadjusted prices: pre-split $500, post-split $125-130
    # Adjusted prices: all around $125-130 (split-adjusted)

    bars = [
        # Bar 1: Pre-split
        MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-01T16:00:00",
            unadjusted=Bar(
                trade_datetime=datetime(2020, 8, 1, 16, 0).isoformat(),
                open=495,
                high=505,
                low=490,
                close=500,
                volume=1000000,
                dividend=None,
            ),
            adjusted=Bar(
                trade_datetime=datetime(2020, 8, 1, 16, 0).isoformat(),
                open=123.75,
                high=126.25,
                low=122.50,
                close=125.00,
                volume=1000000,
                dividend=None,
            ),
            total_return=Bar(
                trade_datetime=datetime(2020, 8, 1, 16, 0).isoformat(),
                open=123.75,
                high=126.25,
                low=122.50,
                close=125.00,
                volume=1000000,
                dividend=None,
            ),
        ),
        # Bar 2: Dividend (pre-split)
        MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-07T16:00:00",
            unadjusted=Bar(
                trade_datetime=datetime(2020, 8, 7, 16, 0).isoformat(),
                open=498,
                high=508,
                low=495,
                close=502,
                volume=1000000,
                dividend=0.82,  # Unadjusted dividend
            ),
            adjusted=Bar(
                trade_datetime=datetime(2020, 8, 7, 16, 0).isoformat(),
                open=124.50,
                high=127.00,
                low=123.75,
                close=125.50,
                volume=1000000,
                dividend=0.205,  # Split-adjusted dividend
            ),
            total_return=Bar(
                trade_datetime=datetime(2020, 8, 7, 16, 0).isoformat(),
                open=124.50,
                high=127.00,
                low=123.75,
                close=125.50,
                volume=1000000,
                dividend=0.205,
            ),
        ),
        # Bar 3: Split 4:1 happens
        # Unadjusted price drops 4x, adjusted stays consistent
        MultiBar(
            symbol="AAPL",
            trade_datetime="2020-08-31T16:00:00",
            unadjusted=Bar(
                trade_datetime=datetime(2020, 8, 31, 16, 0).isoformat(),
                open=126,  # Post-split price
                high=132,
                low=124,
                close=129,
                volume=4000000,  # Volume also adjusts
                dividend=None,
            ),
            adjusted=Bar(
                trade_datetime=datetime(2020, 8, 31, 16, 0).isoformat(),
                open=126.00,
                high=132.00,
                low=124.00,
                close=129.00,
                volume=4000000,
                dividend=None,
            ),
            total_return=Bar(
                trade_datetime=datetime(2020, 8, 31, 16, 0).isoformat(),
                open=126.00,
                high=132.00,
                low=124.00,
                close=129.00,
                volume=4000000,
                dividend=None,
            ),
        ),
        # Bar 4: Post-split (hold)
        MultiBar(
            symbol="AAPL",
            trade_datetime="2020-09-01T16:00:00",
            unadjusted=Bar(
                trade_datetime=datetime(2020, 9, 1, 16, 0).isoformat(),
                open=128,
                high=134,
                low=127,
                close=131,
                volume=3500000,
                dividend=None,
            ),
            adjusted=Bar(
                trade_datetime=datetime(2020, 9, 1, 16, 0).isoformat(),
                open=128.00,
                high=134.00,
                low=127.00,
                close=131.00,
                volume=3500000,
                dividend=None,
            ),
            total_return=Bar(
                trade_datetime=datetime(2020, 9, 1, 16, 0).isoformat(),
                open=128.00,
                high=134.00,
                low=127.00,
                close=131.00,
                volume=3500000,
                dividend=None,
            ),
        ),
        # Bar 5: Sell (post-split)
        MultiBar(
            symbol="AAPL",
            trade_datetime="2020-09-20T16:00:00",
            unadjusted=Bar(
                trade_datetime=datetime(2020, 9, 20, 16, 0).isoformat(),
                open=129,
                high=135,
                low=128,
                close=130,
                volume=3000000,
                dividend=None,
            ),
            adjusted=Bar(
                trade_datetime=datetime(2020, 9, 20, 16, 0).isoformat(),
                open=129.00,
                high=135.00,
                low=128.00,
                close=130.00,
                volume=3000000,
                dividend=None,
            ),
            total_return=Bar(
                trade_datetime=datetime(2020, 9, 20, 16, 0).isoformat(),
                open=129.00,
                high=135.00,
                low=128.00,
                close=130.00,
                volume=3000000,
                dividend=None,
            ),
        ),
    ]

    # Create iterator by extracting bars from each mode
    unadjusted_bars = [bar.unadjusted for bar in bars]
    adjusted_bars = [bar.adjusted for bar in bars]
    total_return_bars = [bar.total_return for bar in bars]

    series_dict = {
        "unadjusted": PriceSeries(mode="unadjusted", symbol="AAPL", bars=unadjusted_bars),
        "adjusted": PriceSeries(mode="adjusted", symbol="AAPL", bars=adjusted_bars),
        "total_return": PriceSeries(mode="total_return", symbol="AAPL", bars=total_return_bars),
    }

    iterator = PriceSeriesIterator(series_dict)

    # Create portfolio and context
    portfolio = Portfolio(initial_cash=Decimal("10000"))

    # Create risk manager (required for signal processing)
    # Use FIXED_QUANTITY sizing to respect target_qty in EXIT signals
    risk_policy = RiskPolicy(
        sizing_method=SizingMethod.FIXED_QUANTITY,
        default_position_size=Decimal("1"),  # Default 1 share (signals will override with target_qty)
        max_position_pct=Decimal("1.0"),  # Allow 100% concentration
    )
    risk_manager = RiskManager(policy=risk_policy, portfolio=portfolio)

    ctx = Context(portfolio=portfolio, risk_manager=risk_manager)

    # Create strategy and backtest
    strategy = SimpleBuyHoldSellStrategy()
    config = ExecutionConfig()
    backtest = Backtest(config=config, strategy=strategy)

    # Run backtest
    backtest.run(
        ctx=ctx,
        data_iterators={"AAPL": iterator},
        symbols=["AAPL"],
        out_dir=Path("/tmp"),
    )

    # Verify results
    print("\n=== Backtest Results ===")
    print(f"Final cash: ${float(ctx.portfolio.cash.get_balance()):.2f}")
    print(f"Final equity: ${float(ctx.portfolio.get_equity()):.2f}")
    print(f"Total fills: {len(backtest.all_fills)}")

    # Check fills
    fills = backtest.all_fills
    assert len(fills) == 2, f"Expected 2 fills (1 buy + 1 sell), got {len(fills)}"

    # First fill: Buy 1 share @ ~$498 (next bar open)
    buy_fill = fills[0]
    print(f"\nBuy fill: {buy_fill.qty} shares @ ${float(buy_fill.price):.2f}")
    assert buy_fill.qty == 1
    assert abs(float(buy_fill.price) - 498.0) < 1.0  # Next bar open (~$498)

    # Second fill: Sell 4 shares @ ~$129 (after split, next bar open)
    sell_fill = fills[1]
    print(f"Sell fill: {sell_fill.qty} shares @ ${float(sell_fill.price):.2f}")
    assert sell_fill.qty == 4, f"Expected to sell 4 shares (after split), got {sell_fill.qty}"
    assert abs(float(sell_fill.price) - 129.0) < 1.0  # Post-split price

    # Check final position (should be flat)
    final_position = ctx.portfolio.positions.get_position("AAPL")
    print(f"Final position: {final_position.qty if final_position and not final_position.is_flat() else 0} shares")
    assert final_position is None or final_position.is_flat(), "Expected flat position after selling all shares"

    # Calculate P&L
    initial_cash = Decimal("10000")
    final_cash = ctx.portfolio.cash.get_balance()
    profit = final_cash - initial_cash
    print(f"\nP&L: ${float(profit):.2f}")

    # Verify accounting:
    # Buy: -$498 (1 share at unadjusted price)
    # Dividend: +$0.82 (unadjusted, for 1 share)
    # Split: 1 → 4 shares @ $124.50 avg cost
    # Sell: +$516 (4 shares × $129 unadjusted)
    # Gross: $18.82
    # Less commissions: -$2.00 (2 fills × $1)
    # Net: ~$16.82

    # Allow for small variations due to commissions and timing
    assert float(profit) > 15.0, f"Expected profit > $15, got ${float(profit):.2f}"
    assert float(profit) < 20.0, f"Expected profit < $20, got ${float(profit):.2f}"

    print("\n✅ Split accounting test PASSED!")
    print("- Buy at unadjusted price")
    print("- Dividend paid on unadjusted amount")
    print("- Split updated position (1 → 4 shares)")
    print("- Sell at unadjusted post-split price")
    print("- P&L reconciles correctly")


if __name__ == "__main__":
    test_split_accounting_with_unadjusted_execution()
