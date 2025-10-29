"""Unit tests for LotTracker - FIFO/LIFO matching logic."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.services.portfolio.lot_tracker import LotTracker
from qtrader.services.portfolio.models import Lot, LotSide


@pytest.fixture
def timestamp() -> datetime:
    """Standard timestamp for tests."""
    return datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc)


class TestFIFOMatching:
    """Test FIFO matching for long positions."""

    def test_match_full_close_single_lot(self, timestamp: datetime) -> None:
        """Test closing entire position with one lot."""
        tracker = LotTracker()

        # Add one lot: 100 @ $150
        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.LONG,
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        tracker.add_lot(lot1)

        # Close entire position
        matches = tracker.match_close_long(Decimal("100"))

        assert len(matches) == 1
        assert matches[0][0].lot_id == "lot_001"
        assert matches[0][1] == Decimal("100")

        # No lots remaining
        assert len(tracker.get_lots(LotSide.LONG)) == 0

    def test_match_partial_close_single_lot(self, timestamp: datetime) -> None:
        """Test partial close leaving remainder."""
        tracker = LotTracker()

        # Add one lot: 100 @ $150
        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.LONG,
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        tracker.add_lot(lot1)

        # Close 60 shares
        matches = tracker.match_close_long(Decimal("60"))

        assert len(matches) == 1
        assert matches[0][0].lot_id == "lot_001"
        assert matches[0][1] == Decimal("60")

        # One lot remaining with 40 shares
        remaining = tracker.get_lots(LotSide.LONG)
        assert len(remaining) == 1
        assert remaining[0].quantity == Decimal("40")
        assert remaining[0].entry_price == Decimal("150.00")  # Same price
        assert remaining[0].lot_id == "lot_001_remaining"

    def test_match_fifo_order_multiple_lots(self, timestamp: datetime) -> None:
        """Test FIFO matching closes oldest first."""
        tracker = LotTracker()

        # Add three lots (different times)
        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.LONG,
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        lot2 = Lot(
            lot_id="lot_002",
            symbol="AAPL",
            side=LotSide.LONG,
            quantity=Decimal("50"),
            entry_price=Decimal("155.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_002",
        )
        lot3 = Lot(
            lot_id="lot_003",
            symbol="AAPL",
            side=LotSide.LONG,
            quantity=Decimal("75"),
            entry_price=Decimal("160.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_003",
        )

        tracker.add_lot(lot1)
        tracker.add_lot(lot2)
        tracker.add_lot(lot3)

        # Close 130 shares (should close lot1 fully + lot2 partially)
        matches = tracker.match_close_long(Decimal("130"))

        assert len(matches) == 2
        # First match: lot1 (100 shares, oldest)
        assert matches[0][0].lot_id == "lot_001"
        assert matches[0][1] == Decimal("100")
        # Second match: lot2 (30 shares, partial)
        assert matches[1][0].lot_id == "lot_002"
        assert matches[1][1] == Decimal("30")

        # Remaining: lot2_remaining (20) + lot3 (75)
        remaining = tracker.get_lots(LotSide.LONG)
        assert len(remaining) == 2
        assert remaining[0].quantity == Decimal("20")  # lot2 remainder
        assert remaining[1].quantity == Decimal("75")  # lot3 untouched

    def test_match_insufficient_quantity(self, timestamp: datetime) -> None:
        """Test error when closing more than available."""
        tracker = LotTracker()

        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.LONG,
            quantity=Decimal("100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        tracker.add_lot(lot1)

        # Try to close more than available
        with pytest.raises(ValueError, match="Insufficient long quantity"):
            tracker.match_close_long(Decimal("150"))

    def test_match_zero_quantity(self) -> None:
        """Test error when closing zero quantity."""
        tracker = LotTracker()

        with pytest.raises(ValueError, match="must be positive"):
            tracker.match_close_long(Decimal("0"))

    def test_match_negative_quantity(self) -> None:
        """Test error when closing negative quantity."""
        tracker = LotTracker()

        with pytest.raises(ValueError, match="must be positive"):
            tracker.match_close_long(Decimal("-50"))


class TestLIFOMatching:
    """Test LIFO matching for short positions."""

    def test_match_full_close_single_lot(self, timestamp: datetime) -> None:
        """Test closing entire short position with one lot."""
        tracker = LotTracker()

        # Add one lot: -100 @ $150
        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.SHORT,
            quantity=Decimal("-100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        tracker.add_lot(lot1)

        # Close entire position
        matches = tracker.match_close_short(Decimal("100"))

        assert len(matches) == 1
        assert matches[0][0].lot_id == "lot_001"
        assert matches[0][1] == Decimal("100")  # Returns positive qty

        # No lots remaining
        assert len(tracker.get_lots(LotSide.SHORT)) == 0

    def test_match_partial_close_single_lot(self, timestamp: datetime) -> None:
        """Test partial close leaving remainder."""
        tracker = LotTracker()

        # Add one lot: -100 @ $150
        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.SHORT,
            quantity=Decimal("-100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        tracker.add_lot(lot1)

        # Close 60 shares
        matches = tracker.match_close_short(Decimal("60"))

        assert len(matches) == 1
        assert matches[0][0].lot_id == "lot_001"
        assert matches[0][1] == Decimal("60")

        # One lot remaining with -40 shares
        remaining = tracker.get_lots(LotSide.SHORT)
        assert len(remaining) == 1
        assert remaining[0].quantity == Decimal("-40")
        assert remaining[0].entry_price == Decimal("150.00")
        assert remaining[0].lot_id == "lot_001_remaining"

    def test_match_lifo_order_multiple_lots(self, timestamp: datetime) -> None:
        """Test LIFO matching closes newest first."""
        tracker = LotTracker()

        # Add three lots (in order)
        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.SHORT,
            quantity=Decimal("-100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        lot2 = Lot(
            lot_id="lot_002",
            symbol="AAPL",
            side=LotSide.SHORT,
            quantity=Decimal("-50"),
            entry_price=Decimal("155.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_002",
        )
        lot3 = Lot(
            lot_id="lot_003",
            symbol="AAPL",
            side=LotSide.SHORT,
            quantity=Decimal("-75"),
            entry_price=Decimal("160.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_003",
        )

        tracker.add_lot(lot1)
        tracker.add_lot(lot2)
        tracker.add_lot(lot3)

        # Close 100 shares (should close lot3 fully + lot2 partially)
        matches = tracker.match_close_short(Decimal("100"))

        assert len(matches) == 2
        # First match: lot3 (75 shares, newest)
        assert matches[0][0].lot_id == "lot_003"
        assert matches[0][1] == Decimal("75")
        # Second match: lot2 (25 shares, partial)
        assert matches[1][0].lot_id == "lot_002"
        assert matches[1][1] == Decimal("25")

        # Remaining: lot1 (-100) + lot2_remaining (-25)
        remaining = tracker.get_lots(LotSide.SHORT)
        assert len(remaining) == 2
        assert remaining[0].quantity == Decimal("-100")  # lot1 untouched
        assert remaining[1].quantity == Decimal("-25")  # lot2 remainder

    def test_match_insufficient_quantity(self, timestamp: datetime) -> None:
        """Test error when closing more than available."""
        tracker = LotTracker()

        lot1 = Lot(
            lot_id="lot_001",
            symbol="AAPL",
            side=LotSide.SHORT,
            quantity=Decimal("-100"),
            entry_price=Decimal("150.00"),
            entry_timestamp=timestamp,
            entry_fill_id="fill_001",
        )
        tracker.add_lot(lot1)

        # Try to close more than available
        with pytest.raises(ValueError, match="Insufficient short quantity"):
            tracker.match_close_short(Decimal("150"))
