"""Tests for Position tracking."""

from decimal import Decimal

from qtrader.models.order import OrderSide
from qtrader.models.position import Position, PositionTracker


def test_position_creation():
    """Position should be created correctly."""
    pos = Position(symbol="AAPL", qty=100, avg_price=Decimal("150.00"), realized_pnl=Decimal("0.0"))

    assert pos.symbol == "AAPL"
    assert pos.qty == 100
    assert pos.avg_price == Decimal("150.00")
    assert pos.realized_pnl == Decimal("0.0")


def test_position_is_long():
    """Position should identify long correctly."""
    long_pos = Position("AAPL", 100, Decimal("150.00"))
    assert long_pos.is_long()
    assert not long_pos.is_short()
    assert not long_pos.is_flat()


def test_position_is_short():
    """Position should identify short correctly."""
    short_pos = Position("AAPL", -100, Decimal("150.00"))
    assert short_pos.is_short()
    assert not short_pos.is_long()
    assert not short_pos.is_flat()


def test_position_is_flat():
    """Position should identify flat correctly."""
    flat_pos = Position("AAPL", 0, Decimal("0.0"))
    assert flat_pos.is_flat()
    assert not flat_pos.is_long()
    assert not flat_pos.is_short()


def test_position_market_value_long():
    """Long position market value should be positive."""
    pos = Position("AAPL", 100, Decimal("150.00"))
    # 100 * 155 = 15500
    assert pos.market_value(Decimal("155.00")) == Decimal("15500.00")


def test_position_market_value_short():
    """Short position market value should be negative."""
    pos = Position("AAPL", -100, Decimal("150.00"))
    # -100 * 155 = -15500
    assert pos.market_value(Decimal("155.00")) == Decimal("-15500.00")


def test_position_unrealized_pnl_long_profit():
    """Long position with profit should calculate correctly."""
    pos = Position("AAPL", 100, Decimal("150.00"))
    # (155 - 150) * 100 = 500
    assert pos.unrealized_pnl(Decimal("155.00")) == Decimal("500.00")


def test_position_unrealized_pnl_long_loss():
    """Long position with loss should calculate correctly."""
    pos = Position("AAPL", 100, Decimal("150.00"))
    # (145 - 150) * 100 = -500
    assert pos.unrealized_pnl(Decimal("145.00")) == Decimal("-500.00")


def test_position_unrealized_pnl_short_profit():
    """Short position with profit should calculate correctly."""
    pos = Position("AAPL", -100, Decimal("150.00"))
    # (150 - 145) * 100 = 500 (profit when price goes down)
    assert pos.unrealized_pnl(Decimal("145.00")) == Decimal("500.00")


def test_position_unrealized_pnl_short_loss():
    """Short position with loss should calculate correctly."""
    pos = Position("AAPL", -100, Decimal("150.00"))
    # (150 - 155) * 100 = -500 (loss when price goes up)
    assert pos.unrealized_pnl(Decimal("155.00")) == Decimal("-500.00")


def test_tracker_initialization():
    """Position tracker should initialize empty."""
    tracker = PositionTracker()
    assert len(tracker.get_all_positions()) == 0


def test_tracker_get_nonexistent_position():
    """Getting non-existent position should return flat position."""
    tracker = PositionTracker()
    pos = tracker.get_position("AAPL")

    assert pos.symbol == "AAPL"
    assert pos.qty == 0
    assert pos.avg_price == Decimal("0.0")
    assert pos.is_flat()


def test_tracker_open_long_position():
    """Opening long position should work correctly."""
    tracker = PositionTracker()
    pos = tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))

    assert pos.symbol == "AAPL"
    assert pos.qty == 100
    assert pos.avg_price == Decimal("150.00")
    assert pos.realized_pnl == Decimal("0.0")
    assert pos.is_long()


def test_tracker_open_short_position():
    """Opening short position should work correctly."""
    tracker = PositionTracker()
    pos = tracker.update_position("AAPL", OrderSide.SELL, 100, Decimal("150.00"))

    assert pos.symbol == "AAPL"
    assert pos.qty == -100
    assert pos.avg_price == Decimal("150.00")
    assert pos.realized_pnl == Decimal("0.0")
    assert pos.is_short()


def test_tracker_add_to_long_position():
    """Adding to long position should average cost."""
    tracker = PositionTracker()

    # Open position: 100 @ $150
    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))

    # Add: 50 @ $160
    pos = tracker.update_position("AAPL", OrderSide.BUY, 50, Decimal("160.00"))

    # New qty: 150
    # Avg cost: (100*150 + 50*160) / 150 = 23000 / 150 = 153.333...
    assert pos.qty == 150
    expected_avg = (Decimal("150.00") * 100 + Decimal("160.00") * 50) / 150
    assert pos.avg_price == expected_avg
    assert pos.realized_pnl == Decimal("0.0")  # No close yet


def test_tracker_add_to_short_position():
    """Adding to short position should average cost."""
    tracker = PositionTracker()

    # Open short: -100 @ $150
    tracker.update_position("AAPL", OrderSide.SELL, 100, Decimal("150.00"))

    # Add to short: -50 @ $145
    pos = tracker.update_position("AAPL", OrderSide.SELL, 50, Decimal("145.00"))

    # New qty: -150
    # Avg cost: (100*150 + 50*145) / 150 = 22250 / 150 = 148.333...
    assert pos.qty == -150
    expected_avg = (Decimal("150.00") * 100 + Decimal("145.00") * 50) / 150
    assert pos.avg_price == expected_avg
    assert pos.realized_pnl == Decimal("0.0")


def test_tracker_reduce_long_position():
    """Reducing long position should realize PnL."""
    tracker = PositionTracker()

    # Open: 100 @ $150
    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))

    # Reduce: sell 40 @ $155
    pos = tracker.update_position("AAPL", OrderSide.SELL, 40, Decimal("155.00"))

    # New qty: 60
    # Realized PnL: (155 - 150) * 40 = 200
    assert pos.qty == 60
    assert pos.avg_price == Decimal("150.00")  # Avg price unchanged when reducing
    assert pos.realized_pnl == Decimal("200.00")


def test_tracker_close_long_position():
    """Closing long position should realize PnL and flatten."""
    tracker = PositionTracker()

    # Open: 100 @ $150
    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))

    # Close: sell 100 @ $160
    pos = tracker.update_position("AAPL", OrderSide.SELL, 100, Decimal("160.00"))

    # Realized PnL: (160 - 150) * 100 = 1000
    assert pos.qty == 0
    assert pos.avg_price == Decimal("0.0")
    assert pos.realized_pnl == Decimal("1000.00")
    assert pos.is_flat()


def test_tracker_flip_long_to_short():
    """Flipping from long to short should realize PnL and open short."""
    tracker = PositionTracker()

    # Open long: 100 @ $150
    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))

    # Flip: sell 150 @ $155
    pos = tracker.update_position("AAPL", OrderSide.SELL, 150, Decimal("155.00"))

    # Close 100, open short 50
    # Realized PnL: (155 - 150) * 100 = 500
    # New position: -50 @ $155
    assert pos.qty == -50
    assert pos.avg_price == Decimal("155.00")
    assert pos.realized_pnl == Decimal("500.00")
    assert pos.is_short()


def test_tracker_flip_short_to_long():
    """Flipping from short to long should realize PnL and open long."""
    tracker = PositionTracker()

    # Open short: -100 @ $150
    tracker.update_position("AAPL", OrderSide.SELL, 100, Decimal("150.00"))

    # Flip: buy 150 @ $145
    pos = tracker.update_position("AAPL", OrderSide.BUY, 150, Decimal("145.00"))

    # Close 100, open long 50
    # Realized PnL: (150 - 145) * 100 = 500
    # New position: 50 @ $145
    assert pos.qty == 50
    assert pos.avg_price == Decimal("145.00")
    assert pos.realized_pnl == Decimal("500.00")
    assert pos.is_long()


def test_tracker_close_short_position():
    """Closing short position should realize PnL."""
    tracker = PositionTracker()

    # Open short: -100 @ $150
    tracker.update_position("AAPL", OrderSide.SELL, 100, Decimal("150.00"))

    # Close: buy 100 @ $145
    pos = tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("145.00"))

    # Realized PnL: (150 - 145) * 100 = 500 (profit)
    assert pos.qty == 0
    assert pos.avg_price == Decimal("0.0")
    assert pos.realized_pnl == Decimal("500.00")
    assert pos.is_flat()


def test_tracker_multiple_positions():
    """Tracker should handle multiple symbols."""
    tracker = PositionTracker()

    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))
    tracker.update_position("MSFT", OrderSide.BUY, 50, Decimal("250.00"))
    tracker.update_position("AMZN", OrderSide.SELL, 25, Decimal("3000.00"))

    positions = tracker.get_all_positions()
    assert len(positions) == 3
    assert positions["AAPL"].qty == 100
    assert positions["MSFT"].qty == 50
    assert positions["AMZN"].qty == -25


def test_tracker_total_exposure():
    """Tracker should calculate total exposure correctly."""
    tracker = PositionTracker()

    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))  # Long
    tracker.update_position("MSFT", OrderSide.BUY, 50, Decimal("250.00"))  # Long
    tracker.update_position("AMZN", OrderSide.SELL, 25, Decimal("3000.00"))  # Short

    prices = {
        "AAPL": Decimal("155.00"),  # 100 * 155 = 15500
        "MSFT": Decimal("260.00"),  # 50 * 260 = 13000
        "AMZN": Decimal("2950.00"),  # -25 * 2950 = -73750
    }

    long_exp, short_exp = tracker.get_total_exposure(prices)

    assert long_exp == Decimal("28500.00")  # 15500 + 13000
    assert short_exp == Decimal("73750.00")  # abs(-73750)


def test_tracker_total_unrealized_pnl():
    """Tracker should calculate total unrealized PnL."""
    tracker = PositionTracker()

    # AAPL: long 100 @ 150
    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))
    # MSFT: long 50 @ 250
    tracker.update_position("MSFT", OrderSide.BUY, 50, Decimal("250.00"))
    # AMZN: short -25 @ 3000
    tracker.update_position("AMZN", OrderSide.SELL, 25, Decimal("3000.00"))

    prices = {
        "AAPL": Decimal("155.00"),  # (155-150) * 100 = 500
        "MSFT": Decimal("245.00"),  # (245-250) * 50 = -250
        "AMZN": Decimal("2900.00"),  # (3000-2900) * 25 = 2500
    }

    total_upnl = tracker.get_total_unrealized_pnl(prices)
    assert total_upnl == Decimal("2750.00")  # 500 - 250 + 2500


def test_tracker_total_realized_pnl():
    """Tracker should track total realized PnL."""
    tracker = PositionTracker()

    # Open AAPL: 100 @ 150
    tracker.update_position("AAPL", OrderSide.BUY, 100, Decimal("150.00"))
    # Close AAPL: sell 100 @ 160 (profit 1000)
    tracker.update_position("AAPL", OrderSide.SELL, 100, Decimal("160.00"))

    # Open MSFT: 50 @ 250
    tracker.update_position("MSFT", OrderSide.BUY, 50, Decimal("250.00"))
    # Close MSFT: sell 50 @ 240 (loss 500)
    tracker.update_position("MSFT", OrderSide.SELL, 50, Decimal("240.00"))

    total_rpnl = tracker.get_total_realized_pnl()
    assert total_rpnl == Decimal("500.00")  # 1000 - 500
