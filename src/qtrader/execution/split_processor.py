"""Stock Split Processor - handles corporate actions affecting position quantity."""

from decimal import Decimal
from typing import Any, Dict

import structlog

from qtrader.models.position import Position, PositionTracker

logger = structlog.get_logger(__name__)


class SplitProcessor:
    """Process stock split corporate actions."""

    def __init__(self, position_tracker: PositionTracker) -> None:
        """Initialize split processor with position tracker."""
        self.position_tracker = position_tracker
        self.processed_count = 0
        self.skipped_count = 0
        logger.info("split_processor.initialized")

    def process_split(
        self,
        symbol: str,
        adjustment_factor: Decimal,
        current_price: Decimal,
    ) -> Dict[str, Any]:
        """
        Process a stock split event.

        Args:
            symbol: Stock symbol
            adjustment_factor: AlgoSeek adjustment factor (0.25 = 4:1 split)
            current_price: Current price after split

        Returns:
            Dict with processing results
        """
        position = self.position_tracker.get_position(symbol)

        if position.is_flat():
            self.skipped_count += 1
            return {"processed": False, "reason": "No position", "symbol": symbol}

        # Calculate split ratio
        split_ratio = Decimal("1") / adjustment_factor

        # Capture pre-split values
        old_qty = position.qty
        old_avg_cost = position.avg_price
        pre_split_price = current_price / split_ratio
        old_market_value = position.market_value(pre_split_price)

        # Calculate new values
        new_qty = int(old_qty * split_ratio)
        new_avg_cost = old_avg_cost / split_ratio

        # Create new position
        new_position = Position(
            symbol=symbol,
            qty=new_qty,
            avg_price=new_avg_cost,
            realized_pnl=position.realized_pnl,
        )

        # Replace position
        self.position_tracker._positions[symbol] = new_position
        new_market_value = new_position.market_value(current_price)

        self.processed_count += 1

        logger.info(
            "split_processor.split_applied",
            symbol=symbol,
            split_ratio=float(split_ratio),
            old_qty=old_qty,
            new_qty=new_qty,
            old_avg_cost=float(old_avg_cost),
            new_avg_cost=float(new_avg_cost),
        )

        return {
            "processed": True,
            "symbol": symbol,
            "split_ratio": split_ratio,
            "old_qty": old_qty,
            "new_qty": new_qty,
            "old_avg_cost": old_avg_cost,
            "new_avg_cost": new_avg_cost,
            "old_market_value": old_market_value,
            "new_market_value": new_market_value,
        }

    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return {
            "processed": self.processed_count,
            "skipped": self.skipped_count,
        }
