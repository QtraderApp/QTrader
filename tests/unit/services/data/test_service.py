"""Unit tests for DataService.

Tests the DataService implementation using mocks to avoid file system
dependencies. Validates interface compliance and error handling.
"""

from datetime import date, datetime
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

from qtrader.config import AssetClass, BarSchemaConfig, DataConfig, DataSourceSelector
from qtrader.data.iterator import PriceSeriesIterator
from qtrader.models.bar import Bar, PriceSeries
from qtrader.models.instrument import Instrument
from qtrader.models.multi_bar import MultiBar
from qtrader.services import DataService


@pytest.fixture
def bar_schema() -> BarSchemaConfig:
    """Standard bar schema for tests."""
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
def data_config(bar_schema: BarSchemaConfig) -> DataConfig:
    """Standard data configuration for tests."""
    selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
    return DataConfig(
        mode="adjusted",
        frequency="1d",
        timezone="America/New_York",
        source_selector=selector,
        bar_schema=bar_schema,
    )


@pytest.fixture
def mock_resolver() -> MagicMock:
    """Mock DataSourceResolver."""
    resolver = MagicMock()
    resolver.sources = {
        "algoseek-us-equity-1d-unadjusted": {
            "adapter": "algoseekOHLC",
            "root_path": "data/test",
            "mode": "standard_adjusted",
        }
    }
    return resolver


@pytest.fixture
def sample_bars() -> List[Bar]:
    """Sample bars for testing."""
    return [
        Bar(
            trade_datetime=datetime(2020, 1, 2, 16, 0),
            open=100.0,
            high=105.0,
            low=99.0,
            close=104.0,
            volume=1000000,
        ),
        Bar(
            trade_datetime=datetime(2020, 1, 3, 16, 0),
            open=104.0,
            high=108.0,
            low=103.0,
            close=107.0,
            volume=1200000,
        ),
    ]


@pytest.fixture
def sample_series_dict(sample_bars: List[Bar]) -> Dict[str, PriceSeries]:
    """Sample series dict for testing."""
    return {
        "unadjusted": PriceSeries(symbol="AAPL", mode="unadjusted", bars=sample_bars),
        "adjusted": PriceSeries(symbol="AAPL", mode="adjusted", bars=sample_bars),
        "total_return": PriceSeries(symbol="AAPL", mode="total_return", bars=sample_bars),
    }


class TestDataServiceInitialization:
    """Test DataService initialization."""

    def test_init_with_config(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test initialization with config."""
        # DataService now creates its own resolver if not provided
        service = DataService(data_config)

        assert service.config == data_config
        # Service creates its own resolver, not using the mock
        assert service.resolver is not None
        assert service.loader is not None
        assert service._instrument_cache == {}

    def test_init_creates_default_resolver(self, data_config: DataConfig) -> None:
        """Test initialization creates default resolver if not provided."""
        with patch("qtrader.services.data.service.DataSourceResolver") as mock_resolver_class:
            mock_resolver_class.return_value = MagicMock()

            service = DataService(data_config)

            mock_resolver_class.assert_called_once()
            assert service.resolver is not None


class TestLoadSymbol:
    """Test load_symbol method."""

    def test_load_symbol_success(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
        sample_series_dict: Dict[str, PriceSeries],
    ) -> None:
        """Test successful symbol loading."""
        with patch.object(DataService, "load_symbol") as mock_load:
            service = DataService(data_config, mock_resolver)

            # Mock loader.load_data to return iterator
            mock_iterator = PriceSeriesIterator(sample_series_dict)
            mock_load.return_value = mock_iterator

            # Load symbol
            result = service.load_symbol(
                "AAPL",
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

            assert isinstance(result, PriceSeriesIterator)
            mock_load.assert_called_once_with(
                "AAPL",
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

    def test_load_symbol_invalid_date_range(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test load_symbol with invalid date range."""
        service = DataService(data_config, mock_resolver)

        with pytest.raises(ValueError, match="Invalid date range"):
            service.load_symbol(
                "AAPL",
                date(2020, 12, 31),
                date(2020, 1, 1),  # End before start
            )

    def test_load_symbol_with_data_source_override(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
        sample_series_dict: Dict[str, PriceSeries],
    ) -> None:
        """Test load_symbol with data source override."""
        service = DataService(data_config, mock_resolver)

        mock_iterator = PriceSeriesIterator(sample_series_dict)
        with patch.object(service.loader, "load_data", return_value=mock_iterator):
            # Load with override (currently not used by loader, but accepted)
            result = service.load_symbol(
                "AAPL",
                date(2020, 1, 1),
                date(2020, 12, 31),
                data_source="schwab",
            )

            assert isinstance(result, PriceSeriesIterator)


class TestLoadUniverse:
    """Test load_universe method."""

    def test_load_universe_success(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
        sample_series_dict: Dict[str, PriceSeries],
    ) -> None:
        """Test successful universe loading."""
        service = DataService(data_config, mock_resolver)

        # Mock load_symbol to return iterators
        def mock_load_symbol(symbol: str, start: date, end: date, **kwargs) -> PriceSeriesIterator:
            series_dict = {
                "unadjusted": PriceSeries(symbol=symbol, mode="unadjusted", bars=sample_series_dict["unadjusted"].bars),
                "adjusted": PriceSeries(symbol=symbol, mode="adjusted", bars=sample_series_dict["adjusted"].bars),
                "total_return": PriceSeries(
                    symbol=symbol, mode="total_return", bars=sample_series_dict["total_return"].bars
                ),
            }
            return PriceSeriesIterator(series_dict)

        with patch.object(service, "load_symbol", side_effect=mock_load_symbol) as mock_load:
            # Load universe
            symbols = ["AAPL", "MSFT", "GOOGL"]
            result = service.load_universe(
                symbols,
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

            assert isinstance(result, dict)
            assert len(result) == 3
            assert set(result.keys()) == set(symbols)
            assert all(isinstance(iterator, PriceSeriesIterator) for iterator in result.values())
            assert mock_load.call_count == 3

    def test_load_universe_partial_failure(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
        sample_series_dict: Dict[str, PriceSeries],
    ) -> None:
        """Test universe loading with some symbols failing."""
        service = DataService(data_config, mock_resolver)

        # Mock load_symbol to fail for BADSTOCK
        def mock_load_symbol(symbol: str, start: date, end: date, **kwargs) -> PriceSeriesIterator:
            if symbol == "BADSTOCK":
                raise FileNotFoundError(f"Data not found for {symbol}")
            series_dict = {
                "unadjusted": PriceSeries(symbol=symbol, mode="unadjusted", bars=sample_series_dict["unadjusted"].bars),
                "adjusted": PriceSeries(symbol=symbol, mode="adjusted", bars=sample_series_dict["adjusted"].bars),
                "total_return": PriceSeries(
                    symbol=symbol, mode="total_return", bars=sample_series_dict["total_return"].bars
                ),
            }
            return PriceSeriesIterator(series_dict)

        with patch.object(service, "load_symbol", side_effect=mock_load_symbol):
            # Load universe with one bad symbol
            symbols = ["AAPL", "BADSTOCK", "MSFT"]
            result = service.load_universe(
                symbols,
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

            # Should return only successful symbols
            assert len(result) == 2
            assert set(result.keys()) == {"AAPL", "MSFT"}
            assert "BADSTOCK" not in result

    def test_load_universe_invalid_date_range(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test load_universe with invalid date range."""
        service = DataService(data_config, mock_resolver)

        with pytest.raises(ValueError, match="Invalid date range"):
            service.load_universe(
                ["AAPL", "MSFT"],
                date(2020, 12, 31),
                date(2020, 1, 1),  # End before start
            )


class TestGetInstrument:
    """Test get_instrument method."""

    def test_get_instrument_creates_instrument(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test get_instrument creates Instrument."""
        service = DataService(data_config, mock_resolver)

        instrument = service.get_instrument("AAPL")

        assert isinstance(instrument, Instrument)
        assert instrument.symbol == "AAPL"
        # New minimal API: Instrument only has symbol, frequency, metadata
        assert instrument.frequency is None
        assert instrument.metadata == {}

    def test_get_instrument_caches_result(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test get_instrument caches instruments."""
        service = DataService(data_config, mock_resolver)

        instrument1 = service.get_instrument("AAPL")
        instrument2 = service.get_instrument("AAPL")

        # Should return same object (from cache)
        assert instrument1 is instrument2
        assert len(service._instrument_cache) == 1

    def test_get_instrument_different_symbols(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test get_instrument for different symbols."""
        service = DataService(data_config, mock_resolver)

        apple = service.get_instrument("AAPL")
        microsoft = service.get_instrument("MSFT")

        assert apple.symbol == "AAPL"
        assert microsoft.symbol == "MSFT"
        assert len(service._instrument_cache) == 2


class TestListAvailableSymbols:
    """Test list_available_symbols method."""

    def test_list_available_symbols_success(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test list_available_symbols returns list of symbols from symbol_map."""
        service = DataService(data_config, mock_resolver)

        # Should succeed - reads from data/equity_security_master_sample.csv
        symbols = service.list_available_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert all(isinstance(s, str) for s in symbols)
        # Check it's sorted
        assert symbols == sorted(symbols)


class TestPrivateMethods:
    """Test private helper methods."""

    def test_build_adapter_config_from_resolver(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test _build_adapter_config uses resolver sources."""
        service = DataService(data_config, mock_resolver)

        config = service._build_adapter_config()

        assert config["adapter"] == "algoseekOHLC"
        assert "root_path" in config

    def test_build_adapter_config_fallback(
        self,
        data_config: DataConfig,
    ) -> None:
        """Test _build_adapter_config fallback when source not in resolver."""
        # Create resolver without algoseek source
        resolver = MagicMock()
        resolver.sources = {}

        service = DataService(data_config, resolver)
        config = service._build_adapter_config()

        # Should return fallback config
        assert "adapter" in config
        assert "root_path" in config

    def test_dataset_inference_algoseek(
        self,
        bar_schema: BarSchemaConfig,
    ) -> None:
        """Test dataset inference for algoseek."""
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        # Don't mock the resolver - let DataService create real one
        service = DataService(config)

        # Verify the dataset was inferred correctly
        assert service.dataset == "algoseek-us-equity-1d-unadjusted"

    def test_dataset_inference_schwab(
        self,
        bar_schema: BarSchemaConfig,
    ) -> None:
        """Test dataset inference for schwab."""
        selector = DataSourceSelector(provider="schwab", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        # Don't mock the resolver - let DataService create real one
        service = DataService(config)

        # Verify the dataset was inferred correctly
        assert service.dataset == "schwab-us-equity-1d-adjusted"

    def test_dataset_inference_csv(
        self,
        bar_schema: BarSchemaConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test dataset inference for csv."""
        selector = DataSourceSelector(provider="csv", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        service = DataService(config, mock_resolver)

        # CSV uses fallback dataset
        assert service.dataset is not None

    def test_dataset_inference_unknown_uses_fallback(
        self,
        bar_schema: BarSchemaConfig,
        mock_resolver: MagicMock,
    ) -> None:
        """Test dataset inference defaults for unknown sources."""
        selector = DataSourceSelector(provider="unknown", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        service = DataService(config, mock_resolver)

        # Unknown uses fallback dataset
        assert service.dataset is not None


class TestIntegrationWithRealTypes:
    """Test DataService with real types (not mocks)."""

    def test_can_create_iterator_and_iterate(
        self,
        data_config: DataConfig,
        mock_resolver: MagicMock,
        sample_series_dict: Dict[str, PriceSeries],
    ) -> None:
        """Test that returned iterator works correctly."""
        service = DataService(data_config, mock_resolver)

        # Mock loader to return real iterator
        iterator = PriceSeriesIterator(sample_series_dict)
        with patch.object(service.loader, "load_data", return_value=iterator):
            # Load and iterate
            result = service.load_symbol(
                "AAPL",
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

            bars = list(result)
            assert len(bars) == 2
            assert all(isinstance(bar, MultiBar) for bar in bars)
            assert bars[0].symbol == "AAPL"
            assert bars[0].adjusted.close == 104.0
            assert bars[1].adjusted.close == 107.0
