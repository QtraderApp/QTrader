"""
Dividend processor for handling ex-date processing during backtests.

This module processes dividend events and applies them to both long and short positions:
- Long positions: Credit cash (dividend income)
- Short positions: Debit cash (dividend cost)
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from qtrader.config.logging_config import LoggerFactory
from qtrader.execution.dividend_calculator import DividendCalculator
from qtrader.models.bar import AdjustmentEvent
from qtrader.models.portfolio import Portfolio

logger = LoggerFactory.get_logger(__name__)


class DividendProcessor:
    """
    Process dividend events during backtest execution.

    Responsibilities:
    - Index adjustment events by ex-date
    - Calculate dividend amounts from adjustment factors
    - Apply dividends to long positions (credit cash)
    - Apply dividends to short positions (debit cash)
    - Track dividend processing statistics

    Example:
        >>> events = {"AAPL": [adjustment_event1, adjustment_event2]}
        >>> processor = DividendProcessor(portfolio, events)
        >>> results = processor.process_ex_date(datetime(2023, 2, 10))
        >>> # Processes all dividend events for Feb 10, 2023
    """

    def __init__(
        self,
        portfolio: Portfolio,
        adjustment_events: Dict[str, List[AdjustmentEvent]],
    ):
        """
        Initialize dividend processor with portfolio and events.

        Args:
            portfolio: Portfolio instance to apply dividends to
            adjustment_events: Dict mapping symbol -> list of adjustment events
        """
        self.portfolio = portfolio
        self.adjustment_events = adjustment_events
        self.events_by_date = self._index_by_date(adjustment_events)
        self.processed_count = 0
        self.skipped_count = 0

        logger.info(
            "dividend_processor.initialized",
            total_symbols=len(adjustment_events),
            total_events=sum(len(events) for events in adjustment_events.values()),
            unique_dates=len(self.events_by_date),
        )

    def _index_by_date(
        self, adjustment_events: Dict[str, List[AdjustmentEvent]]
    ) -> Dict[datetime, List[AdjustmentEvent]]:
        """
        Index adjustment events by ex-date for fast lookup.

        Args:
            adjustment_events: Dict mapping symbol -> list of adjustment events

        Returns:
            Dict mapping ex-date -> list of events on that date
        """
        indexed: Dict[datetime, List[AdjustmentEvent]] = {}

        for symbol, events in adjustment_events.items():
            for event in events:
                # Only process cash dividend events
                if event.event_type.lower() in ("cashdiv", "cash_div", "dividend"):
                    date_key = event.ts
                    if date_key not in indexed:
                        indexed[date_key] = []
                    indexed[date_key].append(event)

        logger.debug(
            "dividend_processor.indexed",
            unique_dates=len(indexed),
            total_dividend_events=sum(len(events) for events in indexed.values()),
        )

        return indexed

    def process_ex_date(self, ts: datetime, close_prices: Optional[Dict[str, Decimal]] = None) -> List[Dict[str, Any]]:
        """
        Process dividend ex-dates for current bar.

        For each dividend event on this date:
        1. Calculate dividend amount from adjustment factors
        2. Check if portfolio has long or short position
        3. Apply dividend credit to long positions or debit to short positions
        4. Track processing results

        Args:
            ts: Current bar timestamp (ex-date)
            close_prices: Optional dict of symbol -> close price (day before ex-date)
                         If not provided, will extract from adjustment event metadata

        Returns:
            List of processed dividend events with results
        """
        events = self.events_by_date.get(ts, [])
        if not events:
            return []

        results = []

        for event in events:
            result = self._process_single_event(event, close_prices)
            results.append(result)

            if result["processed"]:
                self.processed_count += 1
            else:
                self.skipped_count += 1

        logger.debug(
            "dividend_processor.ex_date_processed",
            date=ts.isoformat(),
            events_count=len(events),
            processed=sum(1 for r in results if r["processed"]),
            skipped=sum(1 for r in results if not r["processed"]),
        )

        return results

    def _process_single_event(
        self,
        event: AdjustmentEvent,
        close_prices: Optional[Dict[str, Decimal]] = None,
    ) -> Dict[str, Any]:
        """
        Process a single dividend event.

        Args:
            event: Adjustment event to process
            close_prices: Optional dict of symbol -> close price

        Returns:
            Dict with processing results
        """
        result = {
            "symbol": event.symbol,
            "ts": event.ts,
            "event_type": event.event_type,
            "processed": False,
            "dividend_amount": None,
            "position_qty": None,
            "total_debit": None,
            "reason": None,
        }

        # Check if portfolio has position (long or short)
        position = self.portfolio.get_position(event.symbol)
        if not position or position.qty == 0:
            result["reason"] = "no_position"
            logger.debug(
                "dividend_processor.skip_no_position",
                symbol=event.symbol,
                date=event.ts.isoformat(),
            )
            return result

        result["position_qty"] = position.qty

        # Calculate dividend amount
        dividend_amount = self._calculate_dividend(event, close_prices)
        if dividend_amount is None or dividend_amount <= Decimal("0"):
            result["reason"] = "invalid_dividend"
            logger.warning(
                "dividend_processor.invalid_dividend",
                symbol=event.symbol,
                date=event.ts.isoformat(),
                calculated_amount=float(dividend_amount) if dividend_amount else None,
            )
            return result

        result["dividend_amount"] = dividend_amount

        # Apply dividend based on position direction
        try:
            if position.qty < 0:
                # SHORT: Pay dividend (debit)
                self.portfolio.apply_short_dividend(
                    symbol=event.symbol,
                    dividend_per_share=dividend_amount,
                    ts=event.ts,
                )
                result["processed"] = True
                result["total_debit"] = abs(position.qty) * dividend_amount
                result["reason"] = "success_short"

                logger.info(
                    "dividend_processor.applied_short",
                    symbol=event.symbol,
                    date=event.ts.isoformat(),
                    dividend_per_share=float(dividend_amount),
                    position_qty=position.qty,
                    total_debit=float(result["total_debit"]),
                )
            elif position.qty > 0:
                # LONG: Receive dividend (credit)
                self.portfolio.apply_long_dividend(
                    symbol=event.symbol,
                    dividend_per_share=dividend_amount,
                    ts=event.ts,
                )
                result["processed"] = True
                result["total_credit"] = position.qty * dividend_amount
                result["reason"] = "success_long"

                logger.info(
                    "dividend_processor.applied_long",
                    symbol=event.symbol,
                    date=event.ts.isoformat(),
                    dividend_per_share=float(dividend_amount),
                    position_qty=position.qty,
                    total_credit=float(result["total_credit"]),
                )

        except Exception as e:
            result["reason"] = f"error: {str(e)}"
            logger.error(
                "dividend_processor.error",
                symbol=event.symbol,
                date=event.ts.isoformat(),
                error=str(e),
            )

        return result

    def _calculate_dividend(
        self,
        event: AdjustmentEvent,
        close_prices: Optional[Dict[str, Decimal]] = None,
    ) -> Optional[Decimal]:
        """
        Calculate dividend amount from adjustment event.

        Tries multiple methods to find close prices:
        1. close_prices dict (if provided)
        2. event.metadata['close_before']
        3. event.metadata['close_after'] with px_factor

        Args:
            event: Adjustment event with price factors
            close_prices: Optional dict of symbol -> close price (day before ex-date)

        Returns:
            Dividend amount per share, or None if cannot calculate
        """
        symbol = event.symbol

        # Try to get close_before from provided dict
        close_before = None
        if close_prices and symbol in close_prices:
            close_before = close_prices[symbol]

        # Try to get from event metadata
        if close_before is None and "close_before" in event.metadata:
            close_before = Decimal(str(event.metadata["close_before"]))

        # Try to get close_after from metadata
        close_after = None
        if "close_after" in event.metadata:
            close_after = Decimal(str(event.metadata["close_after"]))

        # Need both prices to calculate
        if close_before is None or close_after is None:
            logger.warning(
                "dividend_processor.missing_prices",
                symbol=symbol,
                has_close_before=close_before is not None,
                has_close_after=close_after is not None,
            )
            return None

        # Calculate using DividendCalculator
        return DividendCalculator.calculate_from_factors(
            close_before=close_before,
            close_after=close_after,
            cumulative_price_factor=event.px_factor,
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get dividend processing statistics.

        Returns:
            Dict with processing statistics
        """
        return {
            "total_symbols": len(self.adjustment_events),
            "total_events": sum(len(events) for events in self.adjustment_events.values()),
            "unique_ex_dates": len(self.events_by_date),
            "processed_count": self.processed_count,
            "skipped_count": self.skipped_count,
            "success_rate": (
                self.processed_count / (self.processed_count + self.skipped_count)
                if (self.processed_count + self.skipped_count) > 0
                else 0.0
            ),
        }
