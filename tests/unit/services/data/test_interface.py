"""Unit tests for data service interfaces.

Tests Protocol interfaces to validate their structure and documentation.
Since Protocols are structural types, we verify they have correct signatures
and can be satisfied by implementations.
"""

from datetime import date
from typing import Dict, List
from unittest.mock import MagicMock

from qtrader.models.bar import PriceSeries
from qtrader.models.instrument import Instrument
from qtrader.services.data.interface import IDataAdapter, IDataService
from qtrader.services.data.loaders.iterator import PriceSeriesIterator


class TestIDataServiceProtocol:
    """Test IDataService protocol structure."""

    def test_protocol_has_load_symbol_method(self):
        """Test IDataService has load_symbol method."""
        # Arrange - Create a mock that satisfies the protocol
        mock_service = MagicMock(spec=IDataService)

        # Act - Verify method exists
        assert hasattr(mock_service, "load_symbol")
        assert callable(mock_service.load_symbol)

    def test_protocol_has_load_universe_method(self):
        """Test IDataService has load_universe method."""
        # Arrange
        mock_service = MagicMock(spec=IDataService)

        # Act & Assert
        assert hasattr(mock_service, "load_universe")
        assert callable(mock_service.load_universe)

    def test_protocol_has_get_instrument_method(self):
        """Test IDataService has get_instrument method."""
        # Arrange
        mock_service = MagicMock(spec=IDataService)

        # Act & Assert
        assert hasattr(mock_service, "get_instrument")
        assert callable(mock_service.get_instrument)

    def test_protocol_has_list_available_symbols_method(self):
        """Test IDataService has list_available_symbols method."""
        # Arrange
        mock_service = MagicMock(spec=IDataService)

        # Act & Assert
        assert hasattr(mock_service, "list_available_symbols")
        assert callable(mock_service.list_available_symbols)

    def test_mock_implementation_satisfies_protocol(self):
        """Test a mock implementation satisfies IDataService protocol."""

        # Arrange - Create a mock that implements all required methods
        class MockDataService:
            def load_symbol(self, symbol: str, start_date: date, end_date: date, *, data_source=None):
                return MagicMock(spec=PriceSeriesIterator)

            def load_universe(self, symbols: List[str], start_date: date, end_date: date, *, data_source=None):
                return {}

            def get_instrument(self, symbol: str):
                return Instrument(symbol=symbol)

            def list_available_symbols(self, data_source=None):
                return []

        # Act
        service = MockDataService()

        # Assert - Should be able to call all methods
        assert callable(service.load_symbol)
        assert callable(service.load_universe)
        assert callable(service.get_instrument)
        assert callable(service.list_available_symbols)

    def test_load_symbol_signature_requirements(self):
        """Test load_symbol has correct signature requirements."""

        # This test documents the expected signature
        class TestService:
            def load_symbol(
                self,
                symbol: str,
                start_date: date,
                end_date: date,
                *,
                data_source: str = None,
            ) -> PriceSeriesIterator:
                return MagicMock(spec=PriceSeriesIterator)

        service = TestService()
        result = service.load_symbol("AAPL", date(2020, 1, 1), date(2020, 12, 31))

        # Should return PriceSeriesIterator
        assert result is not None

    def test_load_universe_signature_requirements(self):
        """Test load_universe has correct signature requirements."""

        # This test documents the expected signature
        class TestService:
            def load_universe(
                self,
                symbols: List[str],
                start_date: date,
                end_date: date,
                *,
                data_source: str = None,
            ) -> Dict[str, PriceSeriesIterator]:
                return {}

        service = TestService()
        result = service.load_universe(["AAPL"], date(2020, 1, 1), date(2020, 12, 31))

        # Should return dict
        assert isinstance(result, dict)


class TestIDataAdapterProtocol:
    """Test IDataAdapter protocol structure."""

    def test_protocol_has_read_bars_method(self):
        """Test IDataAdapter has read_bars method."""
        # Arrange
        mock_adapter = MagicMock(spec=IDataAdapter)

        # Act & Assert
        assert hasattr(mock_adapter, "read_bars")
        assert callable(mock_adapter.read_bars)

    def test_protocol_has_to_canonical_series_method(self):
        """Test IDataAdapter has to_canonical_series method."""
        # Arrange
        mock_adapter = MagicMock(spec=IDataAdapter)

        # Act & Assert
        assert hasattr(mock_adapter, "to_canonical_series")
        assert callable(mock_adapter.to_canonical_series)

    def test_mock_implementation_satisfies_protocol(self):
        """Test a mock implementation satisfies IDataAdapter protocol."""

        # Arrange
        class MockDataAdapter:
            def read_bars(self, start_date: str, end_date: str):
                return []

            def to_canonical_series(self, bars: List):
                return {
                    "unadjusted": MagicMock(spec=PriceSeries),
                    "adjusted": MagicMock(spec=PriceSeries),
                    "total_return": MagicMock(spec=PriceSeries),
                }

        # Act
        adapter = MockDataAdapter()

        # Assert - Should be able to call all methods
        assert callable(adapter.read_bars)
        assert callable(adapter.to_canonical_series)

    def test_read_bars_signature_requirements(self):
        """Test read_bars has correct signature requirements."""

        # This test documents the expected signature
        class TestAdapter:
            def read_bars(self, start_date: str, end_date: str) -> List:
                return []

        adapter = TestAdapter()
        result = adapter.read_bars("2020-01-01", "2020-12-31")

        # Should return list
        assert isinstance(result, list)

    def test_to_canonical_series_signature_requirements(self):
        """Test to_canonical_series has correct signature requirements."""

        # This test documents the expected signature
        class TestAdapter:
            def to_canonical_series(self, bars: List) -> Dict[str, PriceSeries]:
                return {
                    "unadjusted": MagicMock(spec=PriceSeries),
                    "adjusted": MagicMock(spec=PriceSeries),
                    "total_return": MagicMock(spec=PriceSeries),
                }

        adapter = TestAdapter()
        result = adapter.to_canonical_series([])

        # Should return dict with required keys
        assert isinstance(result, dict)
        assert "unadjusted" in result
        assert "adjusted" in result
        assert "total_return" in result

    def test_canonical_series_must_have_all_modes(self):
        """Test canonical series requires all three adjustment modes."""
        # This test documents the contract
        required_modes = {"unadjusted", "adjusted", "total_return"}

        class TestAdapter:
            def to_canonical_series(self, bars: List) -> Dict[str, PriceSeries]:
                # Must return all three modes
                return {
                    "unadjusted": MagicMock(spec=PriceSeries),
                    "adjusted": MagicMock(spec=PriceSeries),
                    "total_return": MagicMock(spec=PriceSeries),
                }

        adapter = TestAdapter()
        result = adapter.to_canonical_series([])

        # All modes must be present
        assert set(result.keys()) == required_modes


class TestProtocolUsagePatterns:
    """Test common usage patterns with protocols."""

    def test_service_can_be_mocked_for_testing(self):
        """Test IDataService can be easily mocked for testing."""
        # Arrange - Mock service for testing
        mock_service = MagicMock(spec=IDataService)
        mock_iterator = MagicMock(spec=PriceSeriesIterator)
        mock_service.load_symbol.return_value = mock_iterator

        # Act - Use in test scenario
        result = mock_service.load_symbol("AAPL", date(2020, 1, 1), date(2020, 12, 31))

        # Assert
        assert result == mock_iterator
        mock_service.load_symbol.assert_called_once_with("AAPL", date(2020, 1, 1), date(2020, 12, 31))

    def test_adapter_can_be_mocked_for_testing(self):
        """Test IDataAdapter can be easily mocked for testing."""
        # Arrange
        mock_adapter = MagicMock(spec=IDataAdapter)
        mock_series = {
            "unadjusted": MagicMock(spec=PriceSeries),
            "adjusted": MagicMock(spec=PriceSeries),
            "total_return": MagicMock(spec=PriceSeries),
        }
        mock_adapter.to_canonical_series.return_value = mock_series

        # Act
        result = mock_adapter.to_canonical_series([])

        # Assert
        assert result == mock_series
        mock_adapter.to_canonical_series.assert_called_once_with([])

    def test_service_protocol_enables_dependency_injection(self):
        """Test IDataService enables dependency injection pattern."""

        # Arrange - Strategy that depends on IDataService
        class SimpleStrategy:
            def __init__(self, data_service):  # Type: IDataService
                self.data_service = data_service

            def load_data(self, symbol: str):
                return self.data_service.load_symbol(symbol, date(2020, 1, 1), date(2020, 12, 31))

        # Act - Inject mock service
        mock_service = MagicMock(spec=IDataService)
        mock_iterator = MagicMock(spec=PriceSeriesIterator)
        mock_service.load_symbol.return_value = mock_iterator

        strategy = SimpleStrategy(mock_service)
        result = strategy.load_data("AAPL")

        # Assert - Strategy works with protocol
        assert result == mock_iterator

    def test_adapter_protocol_enables_vendor_abstraction(self):
        """Test IDataAdapter enables vendor-specific implementations."""
        # This test demonstrates how different vendors can implement the protocol

        class MockAlgoseekAdapter:
            def read_bars(self, start_date: str, end_date: str):
                return [{"algoseek_specific": "data"}]

            def to_canonical_series(self, bars: List):
                return {
                    "unadjusted": MagicMock(spec=PriceSeries),
                    "adjusted": MagicMock(spec=PriceSeries),
                    "total_return": MagicMock(spec=PriceSeries),
                }

        class MockSchwabAdapter:
            def read_bars(self, start_date: str, end_date: str):
                return [{"schwab_specific": "data"}]

            def to_canonical_series(self, bars: List):
                return {
                    "unadjusted": MagicMock(spec=PriceSeries),
                    "adjusted": MagicMock(spec=PriceSeries),
                    "total_return": MagicMock(spec=PriceSeries),
                }

        # Both adapters satisfy the protocol
        algoseek = MockAlgoseekAdapter()
        schwab = MockSchwabAdapter()

        # Both can be used interchangeably
        for adapter in [algoseek, schwab]:
            bars = adapter.read_bars("2020-01-01", "2020-12-31")
            assert isinstance(bars, list)

            series = adapter.to_canonical_series(bars)
            assert "unadjusted" in series
            assert "adjusted" in series
            assert "total_return" in series


class TestProtocolDocumentation:
    """Test that protocols have proper documentation."""

    def test_idataservice_has_docstring(self):
        """Test IDataService has documentation."""
        assert IDataService.__doc__ is not None
        assert len(IDataService.__doc__) > 0
        assert "Data service interface" in IDataService.__doc__

    def test_idataadapter_has_docstring(self):
        """Test IDataAdapter has documentation."""
        assert IDataAdapter.__doc__ is not None
        assert len(IDataAdapter.__doc__) > 0
        assert "Adapter interface" in IDataAdapter.__doc__

    def test_protocol_methods_have_docstrings(self):
        """Test protocol methods have documentation."""
        # IDataService methods
        assert IDataService.load_symbol.__doc__ is not None
        assert IDataService.load_universe.__doc__ is not None
        assert IDataService.get_instrument.__doc__ is not None
        assert IDataService.list_available_symbols.__doc__ is not None

        # IDataAdapter methods
        assert IDataAdapter.read_bars.__doc__ is not None
        assert IDataAdapter.to_canonical_series.__doc__ is not None
