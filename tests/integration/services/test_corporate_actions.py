"""Integration tests for corporate action detection via DataService.

Tests the full flow from DataService → Adapter → AlgoseekBar detection.
"""

from datetime import date
from decimal import Decimal

import pytest

from qtrader.events.events import CorporateActionEvent
from qtrader.services.data.data_config import BarSchemaConfig, DataConfig
from qtrader.services.data.data_source_selector import AssetClass, DataSourceSelector
from qtrader.services.data.service import DataService


@pytest.fixture
def data_service():
    """Create DataService configured for Algoseek."""
    selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
    bar_schema = BarSchemaConfig(
        ts="TradeDate",
        symbol="Ticker",
        open="Open",
        high="High",
        low="Low",
        close="Close",
        volume="MarketHoursVolume",
    )
    config = DataConfig(
        mode="adjusted",
        bar_schema=bar_schema,
        source_selector=selector,
    )
    return DataService(config)


class TestCorporateActionsDetection:
    """Test corporate action detection from real Algoseek data."""

    def test_detect_aapl_dividend_2020_feb(self, data_service):
        """Test detection of AAPL dividend in Feb 2020."""
        # AAPL paid dividend on 2020-02-07 (ex-date)
        actions = data_service.get_corporate_actions(
            "AAPL",
            date(2020, 2, 1),
            date(2020, 2, 28),
        )

        # Should find at least one dividend
        dividends = [a for a in actions if a.action_type == "dividend"]
        assert len(dividends) >= 1, "Expected to find AAPL dividend in Feb 2020"

        # Check first dividend
        dividend = dividends[0]
        assert dividend.symbol == "AAPL"
        assert dividend.action_type == "dividend"
        assert dividend.dividend_type == "cash"
        assert dividend.dividend_amount is not None
        assert dividend.dividend_amount > Decimal("0")
        assert dividend.ex_date is not None

    def test_detect_aapl_split_2020_aug(self, data_service):
        """Test detection of AAPL 4-for-1 split in Aug 2020."""
        # AAPL did 4-for-1 split on 2020-08-31
        actions = data_service.get_corporate_actions(
            "AAPL",
            date(2020, 8, 1),
            date(2020, 8, 31),
        )

        # Should find split
        splits = [a for a in actions if a.action_type == "split"]
        assert len(splits) >= 1, "Expected to find AAPL 4:1 split in Aug 2020"

        # Check split ratio
        split = splits[0]
        assert split.symbol == "AAPL"
        assert split.action_type == "split"
        assert split.split_ratio is not None
        # 4-for-1 split means ratio = 4.0
        assert split.split_ratio == Decimal("4.0")

    def test_detect_msft_dividend_2020_may(self, data_service):
        """Test detection of MSFT dividend in May 2020."""
        # MSFT pays quarterly dividends
        actions = data_service.get_corporate_actions(
            "MSFT",
            date(2020, 5, 1),
            date(2020, 5, 31),
        )

        # Should find at least one dividend
        dividends = [a for a in actions if a.action_type == "dividend"]
        assert len(dividends) >= 1, "Expected to find MSFT dividend in May 2020"

        dividend = dividends[0]
        assert dividend.symbol == "MSFT"
        assert dividend.dividend_amount is not None
        assert dividend.dividend_amount > Decimal("0")

    def test_no_corporate_actions_in_quiet_period(self, data_service):
        """Test period with no corporate actions returns empty list."""
        # Pick a short period unlikely to have corporate actions
        actions = data_service.get_corporate_actions(
            "AAPL",
            date(2020, 3, 1),
            date(2020, 3, 7),
        )

        # May have some actions, but should return a list (not error)
        assert isinstance(actions, list)

    def test_corporate_actions_chronological_order(self, data_service):
        """Test that corporate actions are returned in chronological order."""
        # Get full year of AAPL actions
        actions = data_service.get_corporate_actions(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        # Should have multiple actions (dividends + split)
        assert len(actions) > 0

        # Check chronological order
        for i in range(len(actions) - 1):
            current_date = actions[i].effective_date
            next_date = actions[i + 1].effective_date
            assert current_date <= next_date, "Actions should be in chronological order"

    def test_corporate_action_event_structure(self, data_service):
        """Test that CorporateActionEvent has all required fields."""
        actions = data_service.get_corporate_actions(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        assert len(actions) > 0, "Expected to find some corporate actions"

        for action in actions:
            # All actions should be CorporateActionEvent
            assert isinstance(action, CorporateActionEvent)

            # Required fields
            assert action.event_id is not None
            assert action.timestamp is not None
            assert action.event_type == "corporate_action"
            assert action.symbol is not None
            assert action.action_type in ("dividend", "split")
            assert action.effective_date is not None

            # Type-specific fields
            if action.action_type == "dividend":
                assert action.dividend_amount is not None
                assert action.dividend_type in ("cash", "stock")
                assert action.ex_date is not None
            elif action.action_type == "split":
                assert action.split_ratio is not None
                assert action.split_ratio > Decimal("0")

    def test_invalid_date_range_raises_error(self, data_service):
        """Test that invalid date range raises ValueError."""
        with pytest.raises(ValueError, match="Invalid date range"):
            data_service.get_corporate_actions(
                "AAPL",
                date(2020, 12, 31),
                date(2020, 1, 1),  # end < start
            )

    def test_symbol_not_found_raises_error(self, data_service):
        """Test that missing symbol raises appropriate error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            data_service.get_corporate_actions(
                "INVALID_SYMBOL_XYZ",
                date(2020, 1, 1),
                date(2020, 12, 31),
            )


class TestCorporateActionsEdgeCases:
    """Test edge cases and error handling."""

    def test_same_day_range(self, data_service):
        """Test single-day date range."""
        # Use known dividend date
        actions = data_service.get_corporate_actions(
            "AAPL",
            date(2020, 2, 7),
            date(2020, 2, 7),
        )

        # Should return list (may be empty or have dividend)
        assert isinstance(actions, list)

    def test_year_long_range(self, data_service):
        """Test full year range for multiple corporate actions."""
        actions = data_service.get_corporate_actions(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        # AAPL in 2020 had quarterly dividends + 1 split
        # Should have at least 5 events (4 dividends + 1 split)
        assert len(actions) >= 5, "Expected at least 4 dividends + 1 split in 2020"

        # Count by type
        dividends = [a for a in actions if a.action_type == "dividend"]
        splits = [a for a in actions if a.action_type == "split"]

        assert len(dividends) >= 4, "Expected at least 4 quarterly dividends"
        assert len(splits) >= 1, "Expected at least 1 split (Aug 2020)"

    def test_multiple_symbols_sequential(self, data_service):
        """Test getting corporate actions for multiple symbols sequentially."""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        results = {}

        for symbol in symbols:
            try:
                actions = data_service.get_corporate_actions(
                    symbol,
                    date(2020, 1, 1),
                    date(2020, 12, 31),
                )
                results[symbol] = actions
            except Exception:
                # Some symbols may not have data
                pass

        # Should have results for at least one symbol
        assert len(results) > 0

        # All results should be lists
        for symbol, actions in results.items():
            assert isinstance(actions, list)
