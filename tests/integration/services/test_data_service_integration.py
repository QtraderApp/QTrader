"""Integration tests for DataService.

Tests DataService with real data files to ensure proper integration
with DataLoader, adapters, and file system.
"""

from datetime import date
from pathlib import Path

import pytest

from qtrader.services import DataService
from qtrader.services.data.config import BarSchemaConfig, DataConfig
from qtrader.services.data.loaders.iterator import PriceSeriesIterator
from qtrader.services.data.models import Instrument, MultiBar
from qtrader.services.data.source_selector import AssetClass, DataSourceSelector


@pytest.fixture
def test_data_path() -> Path:
    """Path to test data."""
    return Path("data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample")


@pytest.fixture
def bar_schema() -> BarSchemaConfig:
    """Standard bar schema for Algoseek data."""
    return BarSchemaConfig(
        ts="trade_datetime",
        symbol="symbol",
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
    )


@pytest.fixture
def data_config(bar_schema: BarSchemaConfig, test_data_path: Path) -> DataConfig:
    """Data configuration for integration tests."""
    selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
    return DataConfig(
        mode="adjusted",
        frequency="1d",
        timezone="America/New_York",
        source_selector=selector,
        bar_schema=bar_schema,
    )


@pytest.fixture
def data_service(data_config: DataConfig) -> DataService:
    """Create DataService for integration tests."""
    return DataService(data_config)


class TestDataServiceIntegration:
    """Integration tests with real data."""

    def test_load_symbol_real_data(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test loading real data for a symbol."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        # Load AAPL data
        iterator = data_service.load_symbol(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 1, 31),
        )

        assert isinstance(iterator, PriceSeriesIterator)

        # Consume iterator and collect bars
        bars = list(iterator)

        # Should have data for January 2020
        assert len(bars) > 0
        assert all(isinstance(bar, MultiBar) for bar in bars)

        # Verify all bars are for AAPL
        assert all(bar.symbol == "AAPL" for bar in bars)

        # Verify all bars are in January 2020
        for bar in bars:
            assert bar.trade_datetime.year == 2020
            assert bar.trade_datetime.month == 1

        # Verify adjustment modes exist
        first_bar = bars[0]
        assert first_bar.unadjusted is not None
        assert first_bar.adjusted is not None
        assert first_bar.total_return is not None

        # Verify OHLC data
        assert first_bar.adjusted.open > 0
        assert first_bar.adjusted.high > 0
        assert first_bar.adjusted.low > 0
        assert first_bar.adjusted.close > 0
        assert first_bar.adjusted.volume >= 0

    def test_load_universe_real_data(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test loading multiple symbols."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        # Load universe
        symbols = ["AAPL", "MSFT"]
        iterators = data_service.load_universe(
            symbols,
            date(2020, 1, 1),
            date(2020, 1, 31),
        )

        assert isinstance(iterators, dict)

        # Check we got iterators for both symbols
        # (may be less if data not available for all)
        assert len(iterators) <= len(symbols)
        assert all(isinstance(it, PriceSeriesIterator) for it in iterators.values())

        # Verify each symbol's data
        for symbol, iterator in iterators.items():
            bars = list(iterator)
            assert len(bars) > 0
            assert all(bar.symbol == symbol for bar in bars)

    def test_get_instrument(self, data_service: DataService) -> None:
        """Test getting instrument metadata."""
        instrument = data_service.get_instrument("AAPL")

        assert instrument.symbol == "AAPL"
        # New minimal API: Instrument only has symbol, frequency, metadata
        assert isinstance(instrument, Instrument)

    def test_load_symbol_date_validation(self, data_service: DataService) -> None:
        """Test date range validation."""
        with pytest.raises(ValueError, match="Invalid date range"):
            data_service.load_symbol(
                "AAPL",
                date(2020, 12, 31),
                date(2020, 1, 1),  # End before start
            )

    def test_load_symbol_missing_data(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test loading symbol with no data files."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        # Try to load a symbol that definitely doesn't exist
        with pytest.raises((FileNotFoundError, ValueError)):
            iterator = data_service.load_symbol(
                "NOTASYMBOL12345",
                date(2020, 1, 1),
                date(2020, 1, 31),
            )
            # Try to consume iterator (error may be lazy)
            list(iterator)

    def test_iterator_is_consumable_once(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test that iterator can only be consumed once."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        iterator = data_service.load_symbol(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 1, 31),
        )

        # First consumption
        bars1 = list(iterator)
        assert len(bars1) > 0

        # Second consumption should yield nothing (iterator exhausted)
        bars2 = list(iterator)
        assert len(bars2) == 0

    def test_multibar_adjustment_modes(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test that MultiBar contains all adjustment modes."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        iterator = data_service.load_symbol(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 1, 31),
        )

        # Get first bar
        first_bar = next(iterator)

        # Verify all three modes present
        assert first_bar.unadjusted is not None
        assert first_bar.adjusted is not None
        assert first_bar.total_return is not None

        # Verify they're all valid bars
        for mode_name in ["unadjusted", "adjusted", "total_return"]:
            bar = first_bar.get_bar(mode_name)  # type: ignore
            assert bar.open > 0
            assert bar.high >= bar.low
            assert bar.close > 0

    def test_bars_are_chronologically_ordered(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test that bars are returned in chronological order."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        iterator = data_service.load_symbol(
            "AAPL",
            date(2020, 1, 1),
            date(2020, 1, 31),
        )

        bars = list(iterator)

        # Verify chronological order
        for i in range(1, len(bars)):
            assert bars[i].trade_datetime > bars[i - 1].trade_datetime


class TestDataServiceWithConfiguration:
    """Test DataService with different configurations."""

    def test_different_modes(
        self,
        bar_schema: BarSchemaConfig,
        test_data_path: Path,
    ) -> None:
        """Test loading data with different adjustment modes in config."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        # Test with each mode
        for mode in ["unadjusted", "adjusted", "total_return"]:
            selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
            config = DataConfig(
                mode=mode,
                frequency="1d",
                timezone="America/New_York",
                source_selector=selector,
                bar_schema=bar_schema,
            )

            service = DataService(config)
            iterator = service.load_symbol(
                "AAPL",
                date(2020, 1, 1),
                date(2020, 1, 10),
            )

            # Should load successfully regardless of mode
            # (mode affects configuration, but all modes are always loaded)
            bars = list(iterator)
            assert len(bars) > 0


class TestDataServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_day_range(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test loading data for a single day."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        # Load single trading day
        iterator = data_service.load_symbol(
            "AAPL",
            date(2020, 1, 2),
            date(2020, 1, 2),
        )

        bars = list(iterator)

        # Should have exactly one bar (or zero if not a trading day)
        assert len(bars) in [0, 1]

        if len(bars) == 1:
            assert bars[0].trade_datetime.date() == date(2020, 1, 2)

    def test_weekend_range(
        self,
        data_service: DataService,
        test_data_path: Path,
    ) -> None:
        """Test loading data for weekend (no trading days)."""
        # Skip if test data not available
        if not test_data_path.exists():
            pytest.skip(f"Test data not found at {test_data_path}")

        # Load weekend (Jan 4-5, 2020 was Saturday-Sunday)
        iterator = data_service.load_symbol(
            "AAPL",
            date(2020, 1, 4),
            date(2020, 1, 5),
        )

        bars = list(iterator)

        # Should have no bars (no trading on weekends)
        assert len(bars) == 0

    def test_empty_symbol_list(
        self,
        data_service: DataService,
    ) -> None:
        """Test loading empty universe."""
        iterators = data_service.load_universe(
            [],
            date(2020, 1, 1),
            date(2020, 1, 31),
        )

        assert isinstance(iterators, dict)
        assert len(iterators) == 0
