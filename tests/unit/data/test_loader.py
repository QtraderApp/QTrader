"""Tests for DataLoader service."""

import datetime

import pytest

from qtrader.models.vendors.algoseek import AlgoseekBar, AlgoseekPriceSeries
from qtrader.services.data.loaders.iterator import PriceSeriesIterator
from qtrader.services.data.loaders.loader import DataLoader


@pytest.fixture
def sample_algoseek_bars():
    """Create sample Algoseek bars for testing."""
    return [
        AlgoseekBar(
            TradeDate=datetime.datetime(2020, 8, 7),
            Ticker="AAPL",
            Open=100.0,
            High=105.0,
            Low=99.0,
            Close=104.0,
            MarketHoursVolume=1000000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
            AdjustmentFactor=0.00789423076923076,  # Dividend adjustment
            AdjustmentReason="CashDiv",
        ),
        AlgoseekBar(
            TradeDate=datetime.datetime(2020, 8, 10),
            Ticker="AAPL",
            Open=104.0,
            High=108.0,
            Low=103.0,
            Close=107.0,
            MarketHoursVolume=1100000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=1.0,
            AdjustmentFactor=None,
            AdjustmentReason=None,
        ),
        AlgoseekBar(
            TradeDate=datetime.datetime(2020, 8, 31),
            Ticker="AAPL",
            Open=427.0,
            High=437.0,
            Low=425.0,
            Close=436.0,
            MarketHoursVolume=4800000,
            CumulativePriceFactor=1.0,
            CumulativeVolumeFactor=0.25,  # 4:1 split
            AdjustmentFactor=4.0,
            AdjustmentReason="Subdiv",
        ),
    ]


class TestDataLoaderCreation:
    """Test DataLoader creation."""

    def test_create_loader(self):
        """Test creating a DataLoader."""
        # Arrange
        config = {"data_path": "/path/to/data"}

        # Act
        loader = DataLoader(config)

        # Assert
        assert loader.config == config

    def test_create_loader_empty_config(self):
        """Test creating loader with empty config."""
        # Arrange & Act
        loader = DataLoader({})

        # Assert
        assert loader.config == {}


class TestDataLoaderFromSeries:
    """Test load_data_from_series method (available in Phase 2)."""

    def test_load_from_vendor_series(self, sample_algoseek_bars):
        """Test loading from pre-built vendor series."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act
        iterator = loader.load_data_from_series(vendor_series)

        # Assert: Returns iterator
        assert isinstance(iterator, PriceSeriesIterator)
        assert iterator.symbol == "AAPL"

    def test_load_from_series_yields_multi_mode_bars(self, sample_algoseek_bars):
        """Test that iterator yields MultiBar instances."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act
        iterator = loader.load_data_from_series(vendor_series)
        first_bar = next(iterator)

        # Assert: MultiBar with all three modes
        assert first_bar.symbol == "AAPL"
        assert first_bar.unadjusted is not None
        assert first_bar.adjusted is not None
        assert first_bar.total_return is not None

    def test_load_from_series_all_modes_present(self, sample_algoseek_bars):
        """Test that all three adjustment modes are available."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act
        iterator = loader.load_data_from_series(vendor_series)
        bars = list(iterator)

        # Assert: All bars have all modes
        for bar in bars:
            assert bar.unadjusted.open > 0
            assert bar.adjusted.open > 0
            assert bar.total_return.open > 0

    def test_load_from_series_correct_length(self, sample_algoseek_bars):
        """Test iterator has correct number of bars."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act
        iterator = loader.load_data_from_series(vendor_series)
        bars = list(iterator)

        # Assert
        assert len(bars) == 3  # Same as input

    def test_load_from_series_dividend_preserved(self, sample_algoseek_bars):
        """Test that multi-mode bars are created (dividend details tested in canonical_bar tests)."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act
        iterator = loader.load_data_from_series(vendor_series)
        first_bar = next(iterator)

        # Assert: All three modes exist
        assert first_bar.unadjusted is not None
        assert first_bar.adjusted is not None
        assert first_bar.total_return is not None
        # Dividend transformation tested in AlgoseekPriceSeries tests

    def test_load_from_series_split_adjustment(self, sample_algoseek_bars):
        """Test that all three modes are available (split logic tested in AlgoseekPriceSeries)."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act
        iterator = loader.load_data_from_series(vendor_series)
        bars = list(iterator)
        split_bar = bars[2]  # Third bar has split

        # Assert: All modes exist and are accessible
        assert split_bar.unadjusted.open == 427.0  # Actual traded
        assert split_bar.adjusted.open > 0  # Adjusted exists
        assert split_bar.total_return.open > 0  # Total return exists
        # Actual split adjustment logic tested in AlgoseekPriceSeries tests

    def test_load_from_series_empty(self):
        """Test loading empty series."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=[])

        # Act
        iterator = loader.load_data_from_series(vendor_series)
        bars = list(iterator)

        # Assert
        assert len(bars) == 0


class TestDataLoaderFromAdapter:
    """Test load_data method (Phase 3 integration)."""

    def test_load_data_requires_adapter_config(self):
        """Test that load_data requires adapter configuration."""
        # Arrange: Loader without adapter config
        loader = DataLoader({})

        # Act & Assert: Should raise ValueError for missing adapter config
        with pytest.raises(ValueError, match="Adapter configuration missing"):
            loader.load_data("AAPL", "2020-01-01", "2020-12-31")


class TestDataLoaderUseCases:
    """Test real-world usage patterns."""

    def test_golden_data_workflow(self, sample_algoseek_bars):
        """Test workflow with golden test data."""
        # Arrange: Golden data loaded
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act: Load and process
        iterator = loader.load_data_from_series(vendor_series)

        # Simulate strategy using adjusted mode
        strategy_closes = []
        for multi_bar in iterator:
            strategy_bar = multi_bar.adjusted
            strategy_closes.append(strategy_bar.close)

        # Assert: Strategy gets adjusted prices
        assert len(strategy_closes) == 3
        assert all(close > 0 for close in strategy_closes)  # All valid prices
        # Actual adjustment values tested in AlgoseekPriceSeries tests    def test_execution_and_performance_workflow(self, sample_algoseek_bars):
        """Test execution uses unadjusted, performance uses total_return."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)

        # Act: Process bars for different components
        iterator = loader.load_data_from_series(vendor_series)

        results = []
        for multi_bar in iterator:
            # Execution uses unadjusted (realistic fills)
            exec_bar = multi_bar.unadjusted
            fill_price = exec_bar.close
            commission = fill_price * 100 * 0.001

            # Performance uses total_return (includes dividends)
            perf_bar = multi_bar.total_return

            results.append(
                {
                    "exec_close": fill_price,
                    "commission": commission,
                    "perf_close": perf_bar.close,
                }
            )

        # Assert: Execution sees actual prices, all modes accessible
        assert results[0]["exec_close"] == 104.0  # Actual traded
        assert results[0]["commission"] == pytest.approx(10.4)  # On actual price
        assert results[0]["perf_close"] > 0  # Total return exists
        # Specific transformation values tested in AlgoseekPriceSeries tests

    def test_multi_pass_with_reset(self, sample_algoseek_bars):
        """Test multiple passes over same data."""
        # Arrange
        loader = DataLoader({})
        vendor_series = AlgoseekPriceSeries(symbol="AAPL", bars=sample_algoseek_bars)
        iterator = loader.load_data_from_series(vendor_series)

        # Act: First pass
        first_pass = [bar.adjusted.close for bar in iterator]

        # Reset and second pass
        iterator.reset()
        second_pass = [bar.adjusted.close for bar in iterator]

        # Assert: Same results
        assert first_pass == second_pass
        assert len(first_pass) == 3
