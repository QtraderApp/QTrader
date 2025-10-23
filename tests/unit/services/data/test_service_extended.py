"""Additional unit tests for DataService edge cases.

Tests additional scenarios, error handling, and edge cases to improve coverage.
"""

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from qtrader.services import DataService
from qtrader.services.data.config import BarSchemaConfig, DataConfig
from qtrader.services.data.loaders.iterator import PriceSeriesIterator
from qtrader.services.data.source_selector import AssetClass, DataSourceSelector


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


class TestDataServiceEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_init_with_explicit_dataset(self, data_config: DataConfig):
        """Test initialization with explicit dataset parameter."""
        # Arrange & Act
        service = DataService(data_config, dataset="algoseek-us-equity-1d-unadjusted")

        # Assert
        assert service.dataset == "algoseek-us-equity-1d-unadjusted"
        assert service.config == data_config

    def test_init_with_custom_resolver(self, data_config: DataConfig):
        """Test initialization with custom resolver."""
        # Arrange
        custom_resolver = MagicMock()
        custom_resolver.sources = {
            "custom-source": {
                "adapter": "customAdapter",
                "root_path": "/custom/path",
            }
        }

        # Act
        service = DataService(data_config, resolver=custom_resolver)

        # Assert
        assert service.resolver is custom_resolver

    def test_load_symbol_with_same_start_and_end_date(self, data_config: DataConfig):
        """Test load_symbol with start_date == end_date."""
        # Arrange
        service = DataService(data_config)
        mock_iterator = MagicMock(spec=PriceSeriesIterator)

        with patch.object(service.loader, "load_data", return_value=mock_iterator):
            # Act
            result = service.load_symbol(
                "AAPL",
                date(2020, 1, 1),
                date(2020, 1, 1),  # Same date
            )

            # Assert - Should work (single day)
            assert result is not None

    def test_load_universe_with_empty_list(self, data_config: DataConfig):
        """Test load_universe with empty symbols list."""
        # Arrange
        service = DataService(data_config)

        # Act
        result = service.load_universe(
            [],  # Empty list
            date(2020, 1, 1),
            date(2020, 12, 31),
        )

        # Assert - Should return empty dict
        assert result == {}

    def test_load_universe_with_single_symbol(self, data_config: DataConfig):
        """Test load_universe with single symbol."""
        # Arrange
        service = DataService(data_config)
        mock_iterator = MagicMock(spec=PriceSeriesIterator)

        with patch.object(service.loader, "load_data", return_value=mock_iterator):
            # Act
            result = service.load_universe(
                ["AAPL"],  # Single symbol
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

            # Assert
            assert len(result) == 1
            assert "AAPL" in result

    def test_load_universe_all_symbols_fail(self, data_config: DataConfig):
        """Test load_universe when all symbols fail to load."""
        # Arrange
        service = DataService(data_config)

        def mock_load_symbol_fails(symbol, start, end, **kwargs):
            raise FileNotFoundError(f"No data for {symbol}")

        with patch.object(service, "load_symbol", side_effect=mock_load_symbol_fails):
            # Act
            result = service.load_universe(
                ["BAD1", "BAD2", "BAD3"],
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

            # Assert - Should return empty dict
            assert result == {}

    def test_get_instrument_cache_isolation(self, data_config: DataConfig):
        """Test that instrument cache doesn't mix up symbols."""
        # Arrange
        service = DataService(data_config)

        # Act
        aapl1 = service.get_instrument("AAPL")
        msft = service.get_instrument("MSFT")
        aapl2 = service.get_instrument("AAPL")

        # Assert
        assert aapl1 is aapl2  # Same object from cache
        assert aapl1 is not msft  # Different symbols get different objects
        assert aapl1.symbol == "AAPL"
        assert msft.symbol == "MSFT"

    def test_get_instrument_case_sensitivity(self, data_config: DataConfig):
        """Test that symbol case is preserved."""
        # Arrange
        service = DataService(data_config)

        # Act
        upper = service.get_instrument("AAPL")
        lower = service.get_instrument("aapl")

        # Assert - Symbols are case-sensitive
        assert upper.symbol == "AAPL"
        assert lower.symbol == "aapl"
        assert upper is not lower  # Different cache entries


class TestDataServiceErrorHandling:
    """Test error handling and validation."""

    def test_load_symbol_with_file_not_found(self, data_config: DataConfig):
        """Test load_symbol handles FileNotFoundError."""
        # Arrange
        service = DataService(data_config)

        with patch.object(service.loader, "load_data") as mock_load:
            mock_load.side_effect = FileNotFoundError("Data file not found")

            # Act & Assert
            with pytest.raises(FileNotFoundError):
                service.load_symbol(
                    "BADSTOCK",
                    date(2020, 1, 1),
                    date(2020, 12, 31),
                )

    def test_load_universe_with_value_error(self, data_config: DataConfig):
        """Test load_universe handles ValueError for individual symbols."""
        # Arrange
        service = DataService(data_config)
        mock_iterator = MagicMock(spec=PriceSeriesIterator)

        def mock_load_symbol(symbol, start, end, **kwargs):
            if symbol == "INVALID":
                raise ValueError("Invalid symbol")
            return mock_iterator

        with patch.object(service, "load_symbol", side_effect=mock_load_symbol):
            # Act
            result = service.load_universe(
                ["AAPL", "INVALID", "MSFT"],
                date(2020, 1, 1),
                date(2020, 12, 31),
            )

            # Assert - Should skip invalid symbol
            assert len(result) == 2
            assert "INVALID" not in result

    def test_list_available_symbols_with_data_source_param(self, data_config: DataConfig):
        """Test list_available_symbols with data_source parameter."""
        # Arrange
        service = DataService(data_config)

        # Act - data_source param currently unused (reserved for future)
        symbols = service.list_available_symbols(data_source="algoseek")

        # Assert - Should return list of symbols
        assert isinstance(symbols, list)
        assert len(symbols) > 0


class TestDataServiceConfigurationVariants:
    """Test various configuration scenarios."""

    def test_service_with_adjusted_mode(self, bar_schema: BarSchemaConfig):
        """Test service with adjusted mode."""
        # Arrange
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )

        # Act
        service = DataService(config)

        # Assert
        assert service.config.mode == "adjusted"

    def test_service_with_unadjusted_mode(self, bar_schema: BarSchemaConfig):
        """Test service with unadjusted mode."""
        # Arrange
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="unadjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )

        # Act
        service = DataService(config)

        # Assert
        assert service.config.mode == "unadjusted"

    def test_service_with_total_return_mode(self, bar_schema: BarSchemaConfig):
        """Test service with total_return mode."""
        # Arrange
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="total_return",
            bar_schema=bar_schema,
            source_selector=selector,
        )

        # Act
        service = DataService(config)

        # Assert
        assert service.config.mode == "total_return"

    def test_service_with_different_frequencies(self, bar_schema: BarSchemaConfig):
        """Test service with different frequency settings."""
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)

        for freq in ["1d", "1h", "1min"]:
            config = DataConfig(
                mode="adjusted",
                frequency=freq,
                bar_schema=bar_schema,
                source_selector=selector,
            )
            service = DataService(config)
            assert service.config.frequency == freq


class TestDataServiceBuildAdapterConfig:
    """Test _build_adapter_config method variations."""

    def test_build_adapter_config_with_matching_selector(self, data_config: DataConfig):
        """Test adapter config when selector matches source."""
        # Arrange
        resolver = MagicMock()
        resolver.sources = {
            "algoseek-test": {
                "adapter": "algoseekOHLC",
                "root_path": "/data/algoseek",
                "provider": "algoseek",
                "asset_class": "equity",
            }
        }
        service = DataService(data_config, resolver=resolver)

        # Act
        config = service._build_adapter_config()

        # Assert
        assert config["adapter"] == "algoseekOHLC"
        assert config["root_path"] == "/data/algoseek"

    def test_build_adapter_config_with_no_matching_sources(self, bar_schema: BarSchemaConfig):
        """Test adapter config fallback when no sources match."""
        # Arrange
        selector = DataSourceSelector(provider="unknown", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        resolver = MagicMock()
        resolver.sources = {}  # Empty sources
        service = DataService(config, resolver=resolver)

        # Act
        adapter_config = service._build_adapter_config()

        # Assert - Should use fallback
        assert "adapter" in adapter_config
        assert "root_path" in adapter_config

    def test_build_adapter_config_handles_exception(self, data_config: DataConfig):
        """Test adapter config handles exceptions gracefully."""
        # Arrange
        resolver = MagicMock()
        resolver.sources = MagicMock(side_effect=Exception("Resolver error"))
        service = DataService(data_config, resolver=resolver)

        # Act
        config = service._build_adapter_config()

        # Assert - Should use fallback despite exception
        assert "adapter" in config


class TestDataServiceInferDataset:
    """Test _infer_dataset_from_selector method."""

    def test_infer_dataset_finds_matching_source(self, bar_schema: BarSchemaConfig):
        """Test dataset inference finds matching source."""
        # Arrange
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        service = DataService(config)

        # Act - Service should have inferred dataset during init
        # Assert
        assert service.dataset is not None
        assert "algoseek" in service.dataset.lower()

    def test_infer_dataset_by_provider_fallback(self, bar_schema: BarSchemaConfig):
        """Test dataset inference falls back to provider matching."""
        # Arrange
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        service = DataService(config)

        # Act & Assert
        assert service.dataset is not None
        assert "algoseek" in service.dataset.lower()

    def test_infer_dataset_returns_none_for_unknown(self, bar_schema: BarSchemaConfig):
        """Test dataset inference returns None for unknown provider."""
        # Arrange
        selector = DataSourceSelector(provider="totally_unknown_provider", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        resolver = MagicMock()
        resolver.sources = {}  # No sources
        service = DataService(config, resolver=resolver)

        # Act & Assert
        # Should have logged warning and attempted inference
        assert service.dataset is None or service.dataset == ""

    def test_infer_dataset_handles_exception(self, bar_schema: BarSchemaConfig):
        """Test dataset inference handles exceptions."""
        # Arrange
        selector = DataSourceSelector(provider="algoseek", asset_class=AssetClass.EQUITY)
        config = DataConfig(
            mode="adjusted",
            bar_schema=bar_schema,
            source_selector=selector,
        )
        resolver = MagicMock()
        resolver.sources.items = MagicMock(side_effect=Exception("Resolver error"))
        service = DataService(config, resolver=resolver)

        # Act - Should handle exception and possibly return None
        result = service._infer_dataset_from_selector(selector)

        # Assert - Should return None on exception
        assert result is None


class TestDataServiceLogging:
    """Test that service logs appropriately."""

    def test_initialization_logs_warning_without_dataset(self, data_config: DataConfig):
        """Test initialization logs warning when dataset not provided."""
        # This test verifies logging behavior (already covered by existing tests)
        # but documents the expected warning
        with patch("qtrader.services.data.service.logger") as mock_logger:
            DataService(data_config)  # No explicit dataset

            # Should log warning about deprecated usage
            assert mock_logger.warning.called

    def test_load_symbol_logs_operations(self, data_config: DataConfig):
        """Test load_symbol logs operations."""
        # Arrange
        service = DataService(data_config)
        mock_iterator = MagicMock(spec=PriceSeriesIterator)

        with patch("qtrader.services.data.service.logger") as mock_logger:
            with patch.object(service.loader, "load_data", return_value=mock_iterator):
                # Act
                service.load_symbol("AAPL", date(2020, 1, 1), date(2020, 12, 31))

                # Assert - Now logs at DEBUG level for per-symbol operations
                assert mock_logger.debug.called

    def test_load_universe_logs_failures(self, data_config: DataConfig):
        """Test load_universe logs symbol failures."""
        # Arrange
        service = DataService(data_config)

        def mock_load_symbol(symbol, start, end, **kwargs):
            if symbol == "BAD":
                raise FileNotFoundError()
            return MagicMock(spec=PriceSeriesIterator)

        with patch("qtrader.services.data.service.logger") as mock_logger:
            with patch.object(service, "load_symbol", side_effect=mock_load_symbol):
                # Act
                service.load_universe(
                    ["AAPL", "BAD", "MSFT"],
                    date(2020, 1, 1),
                    date(2020, 12, 31),
                )

                # Assert - Should log warnings for failures
                assert mock_logger.warning.called
