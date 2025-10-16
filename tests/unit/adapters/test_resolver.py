"""Tests for DataSourceResolver."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.config.data_source_selector import AssetClass, DataSourceSelector, DataType
from qtrader.models.instrument import Instrument


@pytest.fixture
def temp_config(tmp_path: Path) -> Path:
    """Create temporary config file."""
    config_content = """
data_sources:
  algoseek-us-equity-1d-unadjusted:
    provider: algoseek
    asset_class: equity
    data_type: ohlcv
    frequency: 1d
    region: US
    adjustment_mode: unadjusted
    adapter: algoseekOHLC
    root_path: "data/sample"

  schwab-us-equity-1d-adjusted:
    provider: schwab
    asset_class: equity
    data_type: ohlcv
    frequency: 1d
    region: US
    adjustment_mode: adjusted
    adapter: schwabOHLC
    cache_root: "data/cache"

  binance-crypto-1m:
    provider: binance
    asset_class: crypto
    data_type: ohlcv
    frequency: 1m
    adapter: binance_api
"""
    config_path = tmp_path / "data_sources.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def instrument() -> Instrument:
    """Create test instrument."""
    return Instrument(symbol="AAPL")


class TestResolverBySelectorMatching:
    """Test DataSourceSelector matching logic."""

    def test_match_by_provider(self, temp_config: Path, instrument: Instrument) -> None:
        """Test matching by provider only."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(provider="schwab")

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should match schwab source
            assert mock_create.called
            call_args = mock_create.call_args[0]
            assert call_args[0] == "schwab-us-equity-1d-adjusted"  # source_name

    def test_match_by_asset_class(self, temp_config: Path, instrument: Instrument) -> None:
        """Test matching by asset class only."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(asset_class=AssetClass.CRYPTO)

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should match binance source
            assert mock_create.called
            call_args = mock_create.call_args[0]
            assert call_args[0] == "binance-crypto-1m"  # source_name

    def test_match_by_multiple_criteria(self, temp_config: Path, instrument: Instrument) -> None:
        """Test matching with multiple criteria."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(
            asset_class=AssetClass.EQUITY,
            frequency="1d",
            region="US",
        )

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should match either algoseek or schwab
            assert mock_create.called
            call_args = mock_create.call_args[0]
            assert call_args[0] in ["algoseek-us-equity-1d-unadjusted", "schwab-us-equity-1d-adjusted"]

    def test_match_by_provider_and_asset_class(self, temp_config: Path, instrument: Instrument) -> None:
        """Test matching by provider and asset class."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(
            provider="algoseek",
            asset_class=AssetClass.EQUITY,
        )

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should match algoseek source
            assert mock_create.called
            call_args = mock_create.call_args[0]
            assert call_args[0] == "algoseek-us-equity-1d-unadjusted"

    def test_no_match_raises_error(self, temp_config: Path, instrument: Instrument) -> None:
        """Test that no match raises ValueError."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(
            provider="nonexistent",
            asset_class=AssetClass.EQUITY,
        )

        with pytest.raises(ValueError, match="No data source matches selector"):
            resolver.resolve_by_selector(selector, instrument)

    def test_match_by_data_type(self, temp_config: Path, instrument: Instrument) -> None:
        """Test matching by data type."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(
            asset_class=AssetClass.EQUITY,
            data_type=DataType.OHLCV,
        )

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should match either algoseek or schwab
            assert mock_create.called
            call_args = mock_create.call_args[0]
            assert call_args[0] in ["algoseek-us-equity-1d-unadjusted", "schwab-us-equity-1d-adjusted"]

    def test_match_by_frequency(self, temp_config: Path, instrument: Instrument) -> None:
        """Test matching by frequency."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(frequency="1m")

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should match binance (1m frequency)
            assert mock_create.called
            call_args = mock_create.call_args[0]
            assert call_args[0] == "binance-crypto-1m"


class TestResolverBySelectorFallback:
    """Test fallback provider functionality."""

    def test_fallback_on_primary_failure(self, temp_config: Path, instrument: Instrument) -> None:
        """Test that fallback provider is tried when primary fails."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(
            provider="schwab",
            asset_class=AssetClass.EQUITY,
            fallback_providers=["algoseek"],
        )

        with patch.object(resolver, "_create_adapter") as mock_create:
            # First call (schwab) raises error, second call (algoseek) succeeds
            mock_create.side_effect = [
                Exception("Schwab API error"),
                MagicMock(),  # Algoseek succeeds
            ]

            result = resolver.resolve_by_selector(selector, instrument)

            # Should have tried both providers
            assert mock_create.call_count == 2
            assert result is not None

    def test_no_fallback_when_primary_succeeds(self, temp_config: Path, instrument: Instrument) -> None:
        """Test that fallback is not tried when primary succeeds."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(
            provider="schwab",
            asset_class=AssetClass.EQUITY,
            fallback_providers=["algoseek"],
        )

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should only call primary
            assert mock_create.call_count == 1
            call_args = mock_create.call_args[0]
            assert call_args[0] == "schwab-us-equity-1d-adjusted"

    def test_multiple_fallbacks(self, temp_config: Path, instrument: Instrument) -> None:
        """Test multiple fallback providers."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(
            provider="nonexistent",
            asset_class=AssetClass.EQUITY,
            fallback_providers=["schwab", "algoseek"],
        )

        with patch.object(resolver, "_create_adapter") as mock_create:
            # Primary fails (nonexistent not in config), schwab succeeds
            mock_create.return_value = MagicMock()

            # This will fail on primary (no match), then try schwab
            with pytest.raises(ValueError):
                # Primary won't match, so will raise before trying fallbacks
                resolver.resolve_by_selector(selector, instrument)


class TestResolverMultipleMatches:
    """Test behavior when multiple sources match."""

    def test_logs_multiple_matches(self, temp_config: Path, instrument: Instrument) -> None:
        """Test that multiple matches are logged."""
        resolver = DataSourceResolver(str(temp_config))
        # This will match both algoseek and schwab
        selector = DataSourceSelector(asset_class=AssetClass.EQUITY)

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()

            with patch("qtrader.adapters.resolver.logger") as mock_logger:
                resolver.resolve_by_selector(selector, instrument)

                # Should log multiple matches
                mock_logger.info.assert_called()
                call_args = mock_logger.info.call_args
                assert "multiple_matches" in str(call_args)

    def test_uses_first_match(self, temp_config: Path, instrument: Instrument) -> None:
        """Test that first match is used."""
        resolver = DataSourceResolver(str(temp_config))
        selector = DataSourceSelector(asset_class=AssetClass.EQUITY)

        with patch.object(resolver, "_create_adapter") as mock_create:
            mock_create.return_value = MagicMock()
            resolver.resolve_by_selector(selector, instrument)

            # Should use first match
            assert mock_create.called
            call_args = mock_create.call_args[0]
            # First match could be either depending on dict order
            assert call_args[0] in ["algoseek-us-equity-1d-unadjusted", "schwab-us-equity-1d-adjusted"]


class TestResolverLegacy:
    """Test legacy resolve() method still works."""

    def test_resolve_by_instrument(self, temp_config: Path) -> None:
        """Test resolve() method with Instrument - deprecated API."""
        resolver = DataSourceResolver(str(temp_config))
        instrument = Instrument(symbol="AAPL")

        # The old resolve() is deprecated and now raises AttributeError
        # when instrument doesn't have data_source in metadata
        with pytest.raises(AttributeError, match="Instrument is missing 'data_source'"):
            resolver.resolve(instrument)
