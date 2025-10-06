"""Unit tests for DividendProcessor."""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from qtrader.execution.dividend_processor import DividendProcessor
from qtrader.models.bar import AdjustmentEvent
from qtrader.models.order import OrderSide
from qtrader.models.portfolio import Portfolio


class TestDividendProcessor:
    """Test suite for DividendProcessor."""

    @pytest.fixture
    def portfolio(self):
        """Create a portfolio for testing."""
        return Portfolio(initial_cash=Decimal("100000"))

    @pytest.fixture
    def sample_dividend_event(self):
        """Create a sample dividend adjustment event."""
        return AdjustmentEvent(
            ts=datetime(2023, 2, 10),
            symbol="AAPL",
            event_type="CashDiv",
            px_factor=Decimal("1.001508"),
            vol_factor=Decimal("1.0"),
            metadata={
                "close_before": "152.55",
                "close_after": "152.32",
                "amount": "0.23",
            },
        )

    def test_processor_initialization(self, portfolio, sample_dividend_event):
        """Test DividendProcessor initialization."""
        events = {"AAPL": [sample_dividend_event]}
        processor = DividendProcessor(portfolio, events)

        assert processor.portfolio == portfolio
        assert processor.adjustment_events == events
        assert len(processor.events_by_date) == 1
        assert processor.processed_count == 0
        assert processor.skipped_count == 0

    def test_index_by_date_single_event(self, portfolio, sample_dividend_event):
        """Test event indexing by date."""
        events = {"AAPL": [sample_dividend_event]}
        processor = DividendProcessor(portfolio, events)

        indexed = processor.events_by_date
        assert datetime(2023, 2, 10) in indexed
        assert len(indexed[datetime(2023, 2, 10)]) == 1
        assert indexed[datetime(2023, 2, 10)][0].symbol == "AAPL"

    def test_index_by_date_multiple_symbols_same_date(self, portfolio):
        """Test indexing when multiple symbols have events on same date."""
        date = datetime(2023, 2, 10)
        events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=date,
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "152.55", "close_after": "152.32"},
                )
            ],
            "MSFT": [
                AdjustmentEvent(
                    ts=date,
                    symbol="MSFT",
                    event_type="CashDiv",
                    px_factor=Decimal("1.002481"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "250.00", "close_after": "249.38"},
                )
            ],
        }

        processor = DividendProcessor(portfolio, events)
        indexed = processor.events_by_date

        assert len(indexed[date]) == 2
        symbols = {event.symbol for event in indexed[date]}
        assert symbols == {"AAPL", "MSFT"}

    def test_index_by_date_filters_non_dividend_events(self, portfolio):
        """Test that non-dividend events are filtered out."""
        events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=datetime(2023, 2, 10),
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={},
                ),
                AdjustmentEvent(
                    ts=datetime(2023, 3, 15),
                    symbol="AAPL",
                    event_type="Split",
                    px_factor=Decimal("2.0"),
                    vol_factor=Decimal("0.5"),
                    metadata={},
                ),
            ]
        }

        processor = DividendProcessor(portfolio, events)
        indexed = processor.events_by_date

        # Only dividend should be indexed
        assert len(indexed) == 1
        assert datetime(2023, 2, 10) in indexed
        assert datetime(2023, 3, 15) not in indexed

    def test_process_ex_date_no_events(self, portfolio):
        """Test processing when no events exist for date."""
        events = {}
        processor = DividendProcessor(portfolio, events)

        results = processor.process_ex_date(datetime(2023, 2, 10))
        assert results == []

    def test_process_ex_date_no_position(self, portfolio, sample_dividend_event):
        """Test processing when portfolio has no position."""
        events = {"AAPL": [sample_dividend_event]}
        processor = DividendProcessor(portfolio, events)

        results = processor.process_ex_date(datetime(2023, 2, 10))

        assert len(results) == 1
        assert results[0]["symbol"] == "AAPL"
        assert results[0]["processed"] is False
        assert results[0]["reason"] == "no_position"
        assert processor.skipped_count == 1
        assert processor.processed_count == 0

    def test_process_ex_date_long_position_processed(self, portfolio, sample_dividend_event):
        """Test that long positions receive dividend credit."""
        # Add long position
        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            fill_price=Decimal("150.00"),
            commission=Decimal("0"),
            ts=datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc),
            order_id="order-1",
            fill_id="fill-1",
        )

        initial_cash = portfolio.cash.get_balance()

        events = {"AAPL": [sample_dividend_event]}
        processor = DividendProcessor(portfolio, events)

        results = processor.process_ex_date(datetime(2023, 2, 10))

        assert len(results) == 1
        assert results[0]["processed"] is True
        assert results[0]["reason"] == "success_long"
        assert results[0]["position_qty"] == Decimal("100")
        assert results[0]["dividend_amount"] == Decimal("0.23")
        assert results[0]["total_credit"] == Decimal("23.00")  # 100 shares * $0.23

        # Check cash was credited
        assert portfolio.cash.get_balance() == initial_cash + Decimal("23.00")
        assert processor.processed_count == 1

    def test_process_ex_date_short_position_processed(self, portfolio, sample_dividend_event):
        """Test dividend processing for short position."""
        # Add short position
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

        events = {"AAPL": [sample_dividend_event]}
        processor = DividendProcessor(portfolio, events)

        results = processor.process_ex_date(datetime(2023, 2, 10))

        assert len(results) == 1
        assert results[0]["processed"] is True
        assert results[0]["reason"] == "success_short"
        assert results[0]["position_qty"] == Decimal("-100")
        assert results[0]["dividend_amount"] == Decimal("0.23")
        assert results[0]["total_debit"] == Decimal("23.00")  # 100 shares * $0.23

        # Check cash was debited
        assert portfolio.cash.get_balance() == initial_cash - Decimal("23.00")
        assert processor.processed_count == 1
        assert processor.skipped_count == 0

    def test_process_multiple_short_positions_same_date(self, portfolio):
        """Test processing multiple dividends on same date."""
        date = datetime(2023, 2, 10)
        ts = datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc)

        # Add short positions
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

        events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=date,
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "152.55", "close_after": "152.32"},
                )
            ],
            "MSFT": [
                AdjustmentEvent(
                    ts=date,
                    symbol="MSFT",
                    event_type="CashDiv",
                    px_factor=Decimal("1.002481"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "250.00", "close_after": "249.38"},
                )
            ],
        }

        processor = DividendProcessor(portfolio, events)
        results = processor.process_ex_date(date)

        assert len(results) == 2
        assert all(r["processed"] for r in results)

        # AAPL: 100 shares * $0.23 = $23.00
        # MSFT: 50 shares * $0.62 = $31.00
        # Total: $54.00
        expected_debit = Decimal("23.00") + Decimal("31.00")
        assert portfolio.cash.get_balance() == initial_cash - expected_debit

    def test_calculate_dividend_with_metadata_prices(self, portfolio):
        """Test dividend calculation using prices from metadata."""
        event = AdjustmentEvent(
            ts=datetime(2023, 2, 10),
            symbol="AAPL",
            event_type="CashDiv",
            px_factor=Decimal("1.001508"),
            vol_factor=Decimal("1.0"),
            metadata={
                "close_before": "152.55",
                "close_after": "152.32",
            },
        )

        processor = DividendProcessor(portfolio, {})
        dividend = processor._calculate_dividend(event)

        assert dividend == Decimal("0.23")

    def test_calculate_dividend_with_close_prices_dict(self, portfolio):
        """Test dividend calculation using provided close_prices dict."""
        event = AdjustmentEvent(
            ts=datetime(2023, 2, 10),
            symbol="AAPL",
            event_type="CashDiv",
            px_factor=Decimal("1.001508"),
            vol_factor=Decimal("1.0"),
            metadata={
                "close_after": "152.32",
            },
        )

        close_prices = {"AAPL": Decimal("152.55")}
        processor = DividendProcessor(portfolio, {})
        dividend = processor._calculate_dividend(event, close_prices)

        assert dividend == Decimal("0.23")

    def test_calculate_dividend_missing_prices(self, portfolio):
        """Test dividend calculation fails gracefully when prices missing."""
        event = AdjustmentEvent(
            ts=datetime(2023, 2, 10),
            symbol="AAPL",
            event_type="CashDiv",
            px_factor=Decimal("1.001508"),
            vol_factor=Decimal("1.0"),
            metadata={},  # No price data
        )

        processor = DividendProcessor(portfolio, {})
        dividend = processor._calculate_dividend(event)

        assert dividend is None

    def test_get_stats_empty(self, portfolio):
        """Test statistics for empty processor."""
        processor = DividendProcessor(portfolio, {})
        stats = processor.get_stats()

        assert stats["total_symbols"] == 0
        assert stats["total_events"] == 0
        assert stats["unique_ex_dates"] == 0
        assert stats["processed_count"] == 0
        assert stats["skipped_count"] == 0
        assert stats["success_rate"] == 0.0

    def test_get_stats_after_processing(self, portfolio, sample_dividend_event):
        """Test statistics after processing some events."""
        # Add one short position (will be processed)
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

        # Add another event for symbol with no position (will be skipped)
        events = {
            "AAPL": [sample_dividend_event],
            "MSFT": [
                AdjustmentEvent(
                    ts=datetime(2023, 2, 10),
                    symbol="MSFT",
                    event_type="CashDiv",
                    px_factor=Decimal("1.002481"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "250.00", "close_after": "249.38"},
                )
            ],
        }

        processor = DividendProcessor(portfolio, events)
        processor.process_ex_date(datetime(2023, 2, 10))

        stats = processor.get_stats()

        assert stats["total_symbols"] == 2
        assert stats["total_events"] == 2
        assert stats["unique_ex_dates"] == 1
        assert stats["processed_count"] == 1
        assert stats["skipped_count"] == 1
        assert stats["success_rate"] == 0.5

    def test_process_handles_invalid_dividend_amount(self, portfolio):
        """Test processing when calculated dividend is invalid."""
        # Add short position
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

        # Event with invalid price factor (would result in negative/zero dividend)
        event = AdjustmentEvent(
            ts=datetime(2023, 2, 10),
            symbol="AAPL",
            event_type="CashDiv",
            px_factor=Decimal("0.99"),  # Price went up, not down
            vol_factor=Decimal("1.0"),
            metadata={"close_before": "152.00", "close_after": "153.00"},
        )

        events = {"AAPL": [event]}
        processor = DividendProcessor(portfolio, events)

        results = processor.process_ex_date(datetime(2023, 2, 10))

        assert len(results) == 1
        assert results[0]["processed"] is False
        assert results[0]["reason"] == "invalid_dividend"
        assert processor.skipped_count == 1

    def test_process_different_event_type_names(self, portfolio):
        """Test that various dividend event type names are recognized."""
        date = datetime(2023, 2, 10)
        ts = datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc)

        # Add short position
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

        # Test different event type variations
        event_types = ["CashDiv", "cash_div", "Dividend"]

        for event_type in event_types:
            event = AdjustmentEvent(
                ts=date,
                symbol="AAPL",
                event_type=event_type,
                px_factor=Decimal("1.001508"),
                vol_factor=Decimal("1.0"),
                metadata={"close_before": "152.55", "close_after": "152.32"},
            )

            processor = DividendProcessor(portfolio, {"AAPL": [event]})
            assert len(processor.events_by_date) == 1
            assert date in processor.events_by_date

    def test_process_ex_date_maintains_counts(self, portfolio):
        """Test that processing maintains accurate counts across multiple calls."""
        date1 = datetime(2023, 2, 10)
        date2 = datetime(2023, 5, 15)
        ts = datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc)

        # Add short position
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

        events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=date1,
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "152.55", "close_after": "152.32"},
                ),
                AdjustmentEvent(
                    ts=date2,
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.001508"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "152.55", "close_after": "152.32"},
                ),
            ],
            "MSFT": [
                AdjustmentEvent(
                    ts=date1,
                    symbol="MSFT",
                    event_type="CashDiv",
                    px_factor=Decimal("1.002481"),
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "250.00", "close_after": "249.38"},
                )
            ],
        }

        processor = DividendProcessor(portfolio, events)

        # Process first date (1 processed, 1 skipped)
        results1 = processor.process_ex_date(date1)
        assert len(results1) == 2
        assert processor.processed_count == 1
        assert processor.skipped_count == 1

        # Process second date (1 processed)
        results2 = processor.process_ex_date(date2)
        assert len(results2) == 1
        assert processor.processed_count == 2
        assert processor.skipped_count == 1

    def test_process_long_position_receives_dividend(self, portfolio):
        """Test that long positions receive dividend credits."""
        # Add long position
        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=200,
            fill_price=Decimal("150.00"),
            commission=Decimal("0"),
            ts=datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc),
            order_id="order-1",
            fill_id="fill-1",
        )

        initial_cash = portfolio.cash.get_balance()

        # Dividend event
        event = AdjustmentEvent(
            ts=datetime(2023, 2, 10),
            symbol="AAPL",
            event_type="CashDiv",
            px_factor=Decimal("1.001508"),
            vol_factor=Decimal("1.0"),
            metadata={"close_before": "152.55", "close_after": "152.32"},
        )

        events = {"AAPL": [event]}
        processor = DividendProcessor(portfolio, events)

        results = processor.process_ex_date(datetime(2023, 2, 10))

        assert len(results) == 1
        assert results[0]["processed"] is True
        assert results[0]["reason"] == "success_long"
        assert results[0]["position_qty"] == 200
        assert "total_credit" in results[0]

        # Cash should increase (dividend received)
        assert portfolio.cash.get_balance() > initial_cash
        assert processor.processed_count == 1
        assert processor.skipped_count == 0

    def test_process_mixed_portfolio_long_and_short(self, portfolio):
        """Test processing dividends for mixed long/short positions."""
        ts = datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc)

        # Add long AAPL position
        portfolio.apply_fill(
            symbol="AAPL",
            side=OrderSide.BUY,
            qty=100,
            fill_price=Decimal("180.00"),
            commission=Decimal("0"),
            ts=ts,
            order_id="order-1",
            fill_id="fill-1",
        )

        # Add short MSFT position
        portfolio.apply_fill(
            symbol="MSFT",
            side=OrderSide.SELL,
            qty=50,
            fill_price=Decimal("400.00"),
            commission=Decimal("0"),
            ts=ts,
            order_id="order-2",
            fill_id="fill-2",
        )

        initial_cash = portfolio.cash.get_balance()

        # Same ex-date for both
        date = datetime(2023, 2, 10)
        events = {
            "AAPL": [
                AdjustmentEvent(
                    ts=date,
                    symbol="AAPL",
                    event_type="CashDiv",
                    px_factor=Decimal("1.0025"),  # ~$0.45/share
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "180.00", "close_after": "179.55"},
                )
            ],
            "MSFT": [
                AdjustmentEvent(
                    ts=date,
                    symbol="MSFT",
                    event_type="CashDiv",
                    px_factor=Decimal("1.00125"),  # ~$0.50/share
                    vol_factor=Decimal("1.0"),
                    metadata={"close_before": "400.00", "close_after": "399.50"},
                )
            ],
        }

        processor = DividendProcessor(portfolio, events)
        results = processor.process_ex_date(date)

        assert len(results) == 2
        assert processor.processed_count == 2
        assert processor.skipped_count == 0

        # Find AAPL and MSFT results
        aapl_result = next(r for r in results if r["symbol"] == "AAPL")
        msft_result = next(r for r in results if r["symbol"] == "MSFT")

        # AAPL (long) should receive credit
        assert aapl_result["processed"] is True
        assert aapl_result["reason"] == "success_long"
        assert "total_credit" in aapl_result

        # MSFT (short) should pay debit
        assert msft_result["processed"] is True
        assert msft_result["reason"] == "success_short"
        assert "total_debit" in msft_result

        # Net cash change: +AAPL credit - MSFT debit
        # Should be positive if AAPL dividend > MSFT dividend
        final_cash = portfolio.cash.get_balance()
        assert final_cash != initial_cash  # Cash should have changed

    def test_process_long_position_logging(self, portfolio):
        """Test that long dividend processing includes correct log fields."""
        # Add long position
        portfolio.apply_fill(
            symbol="MSFT",
            side=OrderSide.BUY,
            qty=50,
            fill_price=Decimal("400.00"),
            commission=Decimal("0"),
            ts=datetime(2023, 2, 9, 9, 30, tzinfo=timezone.utc),
            order_id="order-1",
            fill_id="fill-1",
        )

        event = AdjustmentEvent(
            ts=datetime(2023, 2, 10),
            symbol="MSFT",
            event_type="CashDiv",
            px_factor=Decimal("1.00125"),
            vol_factor=Decimal("1.0"),
            metadata={"close_before": "400.00", "close_after": "399.50"},
        )

        events = {"MSFT": [event]}
        processor = DividendProcessor(portfolio, events)

        results = processor.process_ex_date(datetime(2023, 2, 10))

        assert len(results) == 1
        result = results[0]
        assert result["symbol"] == "MSFT"
        assert result["processed"] is True
        assert result["position_qty"] == 50  # Long position
        assert result["dividend_amount"] is not None
        assert result["total_credit"] > Decimal("0")
        assert result["reason"] == "success_long"
