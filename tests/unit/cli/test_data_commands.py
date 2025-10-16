"""Unit tests for CLI data commands.

Tests the thin CLI orchestration layer without executing real commands.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from qtrader.cli.commands.data import data_group
from qtrader.models import Bar, MultiBar


class TestRawDataCommand:
    """Test the raw data command."""

    def test_raw_data_requires_options(self):
        """Test that raw data command requires dataset and symbol."""
        runner = CliRunner()
        result = runner.invoke(data_group, ["raw"])

        # Should fail without required arguments
        assert result.exit_code != 0

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_success(self, mock_data_service_class):
        """Test successful raw data display."""
        runner = CliRunner()

        # Mock the DataService
        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service

        # Mock load_symbol to return empty iterator
        mock_service.load_symbol.return_value = iter([])

        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-12-31",
            ],
        )

        # Should succeed
        assert result.exit_code == 0
        mock_service.load_symbol.assert_called_once()

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_with_source(self, mock_data_service_class):
        """Test raw data with source parameter."""
        runner = CliRunner()

        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service
        mock_service.load_symbol.return_value = iter([])

        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-12-31",
                "--source",
                "schwab",
            ],
        )

        assert result.exit_code == 0


class TestUpdateDatasetCommand:
    """Test the update dataset command."""

    def test_update_requires_dataset(self):
        """Test that update command requires dataset."""
        runner = CliRunner()
        result = runner.invoke(data_group, ["update"])

        # Should fail without required dataset argument
        assert result.exit_code != 0

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_dry_run(self, mock_update_service_class):
        """Test update in dry-run mode."""
        runner = CliRunner()

        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service

        # Mock get_symbols_to_update
        mock_service.get_symbols_to_update.return_value = (["AAPL"], "1 symbol")

        # Mock update_symbols to return empty iterator
        mock_service.update_symbols.return_value = iter([])

        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset", "--dry-run"])

        assert result.exit_code == 0
        mock_service.get_symbols_to_update.assert_called_once()
        mock_service.update_symbols.assert_called_once()

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_with_explicit_symbols(self, mock_update_service_class):
        """Test update with explicit symbol list."""
        runner = CliRunner()

        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = (["AAPL", "MSFT"], "2 symbols")
        mock_service.update_symbols.return_value = iter([])

        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset", "--symbols", "AAPL,MSFT"])

        assert result.exit_code == 0
        # Should be called with explicit symbols
        mock_service.get_symbols_to_update.assert_called_once_with(["AAPL", "MSFT"])

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_verbose_mode(self, mock_update_service_class):
        """Test update in verbose mode."""
        runner = CliRunner()

        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = (["AAPL"], "1 symbol")
        mock_service.update_symbols.return_value = iter([])

        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset", "--verbose"])

        assert result.exit_code == 0
        # Should pass verbose=True to update_symbols
        call_kwargs = mock_service.update_symbols.call_args[1]
        assert call_kwargs["verbose"] is True


class TestCacheInfoCommand:
    """Test the cache info command."""

    def test_cache_info_requires_dataset(self):
        """Test that cache-info command requires dataset."""
        runner = CliRunner()
        result = runner.invoke(data_group, ["cache-info"])

        # Should fail without required dataset argument
        assert result.exit_code != 0

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_cache_info_success(self, mock_update_service_class):
        """Test successful cache info display."""
        runner = CliRunner()

        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service

        # Mock get_cache_root
        mock_cache_root = MagicMock()
        mock_cache_root.exists.return_value = True
        mock_service.get_cache_root.return_value = mock_cache_root

        # Mock scan_cached_symbols
        mock_service.scan_cached_symbols.return_value = ["AAPL", "MSFT"]

        # Mock read_symbol_metadata for each symbol
        mock_service.read_symbol_metadata.side_effect = [
            {
                "symbol": "AAPL",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "row_count": "252",
                "last_update": "2020-12-31",
                "error": False,
            },
            {
                "symbol": "MSFT",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "row_count": "252",
                "last_update": "2020-12-31",
                "error": False,
            },
        ]

        result = runner.invoke(data_group, ["cache-info", "--dataset", "test-dataset"])

        assert result.exit_code == 0
        mock_service.scan_cached_symbols.assert_called_once()
        assert mock_service.read_symbol_metadata.call_count == 2

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_cache_info_no_cache(self, mock_update_service_class):
        """Test cache info with no cache directory."""
        runner = CliRunner()

        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service

        # Mock get_cache_root to return None
        mock_service.get_cache_root.return_value = None

        result = runner.invoke(data_group, ["cache-info", "--dataset", "test-dataset"])

        # Should succeed but show no cache message
        assert result.exit_code == 0

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_cache_info_empty_cache(self, mock_update_service_class):
        """Test cache info with empty cache."""
        runner = CliRunner()

        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service

        # Mock get_cache_root
        mock_cache_root = MagicMock()
        mock_cache_root.exists.return_value = True
        mock_service.get_cache_root.return_value = mock_cache_root

        # Mock scan_cached_symbols to return empty list
        mock_service.scan_cached_symbols.return_value = []

        result = runner.invoke(data_group, ["cache-info", "--dataset", "test-dataset"])

        # Should succeed but show empty cache message
        assert result.exit_code == 0


class TestDataGroupIntegration:
    """Integration tests for the data command group."""

    def test_data_group_help(self):
        """Test data group shows help."""
        runner = CliRunner()
        result = runner.invoke(data_group, ["--help"])

        assert result.exit_code == 0
        assert "data" in result.output.lower()
        assert "raw" in result.output.lower()
        assert "update" in result.output.lower()
        assert "cache-info" in result.output.lower()

    def test_invalid_command(self):
        """Test invalid command shows error."""
        runner = CliRunner()
        result = runner.invoke(data_group, ["invalid-command"])

        assert result.exit_code != 0


class TestRawDataCommandEdgeCases:
    """Test edge cases and error handling for raw data command."""

    def test_raw_data_invalid_date_format_returns_error(self):
        """Test that invalid date format is handled gracefully."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-13-45",  # Invalid date
                "--end-date",
                "2020-12-31",
            ],
        )

        # Assert
        assert result.exit_code == 1
        assert "Invalid date format" in result.output

    def test_raw_data_malformed_date_returns_error(self):
        """Test that malformed date string is handled."""
        # Arrange
        runner = CliRunner()

        # Act
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "not-a-date",
                "--end-date",
                "2020-12-31",
            ],
        )

        # Assert
        assert result.exit_code == 1
        assert "Invalid date format" in result.output

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_file_not_found_error_shows_helpful_message(self, mock_data_service_class):
        """Test FileNotFoundError displays helpful message."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service
        mock_service.load_symbol.side_effect = FileNotFoundError("Data file not found")

        # Act
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-12-31",
            ],
        )

        # Assert
        assert result.exit_code == 1
        assert "Data file not found" in result.output
        assert "Make sure the data files exist" in result.output

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_value_error_displays_error_message(self, mock_data_service_class):
        """Test ValueError displays error message."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service
        mock_service.load_symbol.side_effect = ValueError("Invalid configuration")

        # Act
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-12-31",
            ],
        )

        # Assert
        assert result.exit_code == 1
        assert "Invalid configuration" in result.output

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_generic_exception_shows_traceback(self, mock_data_service_class):
        """Test generic exception displays traceback."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service
        mock_service.load_symbol.side_effect = RuntimeError("Unexpected error")

        # Act
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-12-31",
            ],
        )

        # Assert
        assert result.exit_code == 1
        assert "Unexpected error" in result.output

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_displays_bars_with_correct_format(self, mock_data_service_class):
        """Test that bars are displayed with correct formatting."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service

        # Create mock bars
        mock_bar = Bar(
            trade_datetime=datetime(2020, 1, 2, 16, 0),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
            dividend=None,
        )
        mock_multi_bar = MultiBar(
            symbol="AAPL",
            trade_datetime=datetime(2020, 1, 2, 16, 0),
            unadjusted=mock_bar,
            adjusted=mock_bar,
            total_return=mock_bar,
        )
        mock_service.load_symbol.return_value = iter([mock_multi_bar])

        # Act - provide input to skip the interactive prompt
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-01-31",
            ],
            input="\n",  # Simulate pressing Enter
        )

        # Assert
        assert result.exit_code == 0
        assert "Loaded 1 bars" in result.output
        assert "End of data" in result.output


class TestUpdateDatasetCommandEdgeCases:
    """Test edge cases and error handling for update dataset command."""

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_no_symbols_found_shows_message(self, mock_update_service_class):
        """Test update with no symbols found shows helpful message."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = ([], "no symbols")

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset"])

        # Assert
        assert result.exit_code == 0
        assert "No symbols found to update" in result.output
        assert "universe.csv" in result.output

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_with_errors_displays_error_list(self, mock_update_service_class):
        """Test update with errors displays error list."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = (["AAPL", "MSFT"], "2 symbols")

        # Create mock results with one success and one failure
        mock_result_success = MagicMock()
        mock_result_success.symbol = "AAPL"
        mock_result_success.success = True
        mock_result_success.bars_added = 10
        mock_result_success.error = None

        mock_result_failure = MagicMock()
        mock_result_failure.symbol = "MSFT"
        mock_result_failure.success = False
        mock_result_failure.bars_added = 0
        mock_result_failure.error = "API error"

        mock_service.update_symbols.return_value = iter([mock_result_success, mock_result_failure])
        mock_service.get_cache_metadata.return_value = ("2020-01-01", "2020-12-31", 252)

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset"])

        # Assert
        assert result.exit_code == 0
        assert "Successful: 1/2" in result.output
        assert "Errors (1)" in result.output
        assert "MSFT: API error" in result.output

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_dry_run_shows_warning_message(self, mock_update_service_class):
        """Test dry run shows appropriate warning."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = (["AAPL"], "1 symbol")

        mock_result = MagicMock()
        mock_result.symbol = "AAPL"
        mock_result.success = True
        mock_result.bars_added = 5
        mock_result.error = None

        mock_service.update_symbols.return_value = iter([mock_result])
        mock_service.get_cache_metadata.return_value = ("2020-01-01", "2020-12-31", 252)

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset", "--dry-run"])

        # Assert
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "This was a dry run" in result.output

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_handles_value_error(self, mock_update_service_class):
        """Test update handles ValueError gracefully."""
        # Arrange
        runner = CliRunner()
        mock_update_service_class.side_effect = ValueError("Invalid dataset")

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "invalid-dataset"])

        # Assert
        assert result.exit_code == 1
        assert "Invalid dataset" in result.output

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_handles_generic_exception(self, mock_update_service_class):
        """Test update handles generic exceptions with traceback."""
        # Arrange
        runner = CliRunner()
        mock_update_service_class.side_effect = RuntimeError("Unexpected error")

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset"])

        # Assert
        assert result.exit_code == 1
        assert "Unexpected error" in result.output

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_displays_statistics_correctly(self, mock_update_service_class):
        """Test update displays correct statistics."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = (["AAPL", "MSFT", "TSLA"], "3 symbols")

        # Create mock results - all successful
        results = []
        for symbol, bars in [("AAPL", 10), ("MSFT", 15), ("TSLA", 20)]:
            mock_result = MagicMock()
            mock_result.symbol = symbol
            mock_result.success = True
            mock_result.bars_added = bars
            mock_result.error = None
            results.append(mock_result)

        mock_service.update_symbols.return_value = iter(results)
        mock_service.get_cache_metadata.return_value = ("2020-01-01", "2020-12-31", 252)

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset"])

        # Assert
        assert result.exit_code == 0
        assert "Successful: 3/3" in result.output
        assert "Total bars added: 45" in result.output  # 10 + 15 + 20


class TestCacheInfoCommandEdgeCases:
    """Test edge cases and error handling for cache-info command."""

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_cache_info_cache_exists_false_shows_message(self, mock_update_service_class):
        """Test cache info when cache exists() returns False."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service

        mock_cache_root = MagicMock()
        mock_cache_root.exists.return_value = False
        mock_service.get_cache_root.return_value = mock_cache_root

        # Act
        result = runner.invoke(data_group, ["cache-info", "--dataset", "test-dataset"])

        # Assert
        assert result.exit_code == 0
        assert "No cache found" in result.output

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_cache_info_displays_multiple_symbols(self, mock_update_service_class):
        """Test cache info displays multiple symbols correctly."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service

        mock_cache_root = MagicMock()
        mock_cache_root.exists.return_value = True
        mock_service.get_cache_root.return_value = mock_cache_root
        mock_service.scan_cached_symbols.return_value = ["AAPL", "MSFT", "TSLA"]

        # Mock metadata for each symbol
        metadata_list = [
            {
                "symbol": "AAPL",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "row_count": "252",
                "last_update": "2020-12-31",
            },
            {
                "symbol": "MSFT",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "row_count": "252",
                "last_update": "2020-12-31",
            },
            {
                "symbol": "TSLA",
                "start_date": "2020-01-01",
                "end_date": "2020-12-31",
                "row_count": "252",
                "last_update": "2020-12-31",
            },
        ]
        mock_service.read_symbol_metadata.side_effect = metadata_list

        # Act
        result = runner.invoke(data_group, ["cache-info", "--dataset", "test-dataset"])

        # Assert
        assert result.exit_code == 0
        assert "Cached symbols: 3" in result.output
        assert mock_service.read_symbol_metadata.call_count == 3

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_cache_info_handles_value_error(self, mock_update_service_class):
        """Test cache info handles ValueError gracefully."""
        # Arrange
        runner = CliRunner()
        mock_update_service_class.side_effect = ValueError("Invalid dataset configuration")

        # Act
        result = runner.invoke(data_group, ["cache-info", "--dataset", "invalid-dataset"])

        # Assert
        assert result.exit_code == 1
        assert "Invalid dataset configuration" in result.output

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_cache_info_handles_generic_exception(self, mock_update_service_class):
        """Test cache info handles generic exceptions."""
        # Arrange
        runner = CliRunner()
        mock_update_service_class.side_effect = RuntimeError("Unexpected error")

        # Act
        result = runner.invoke(data_group, ["cache-info", "--dataset", "test-dataset"])

        # Assert
        assert result.exit_code == 1
        assert "Unexpected error" in result.output


class TestRawDataCommandInteractive:
    """Test interactive features of raw data command."""

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_multiple_bars_display_correctly(self, mock_data_service_class):
        """Test displaying multiple bars interactively."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service

        # Create multiple mock bars
        mock_bars = []
        for i in range(3):
            mock_bar = Bar(
                trade_datetime=datetime(2020, 1, 2 + i, 16, 0),
                open=100.0 + i,
                high=105.0 + i,
                low=99.0 + i,
                close=103.0 + i,
                volume=1000000,
                dividend=None,
            )
            mock_multi_bar = MultiBar(
                symbol="AAPL",
                trade_datetime=datetime(2020, 1, 2 + i, 16, 0),
                unadjusted=mock_bar,
                adjusted=mock_bar,
                total_return=mock_bar,
            )
            mock_bars.append(mock_multi_bar)

        mock_service.load_symbol.return_value = iter(mock_bars)

        # Act - Provide enter key presses to progress through all bars
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-01-31",
            ],
            input="\n\n\n",  # Press Enter to progress through bars
        )

        # Assert
        assert result.exit_code == 0
        assert "Loaded 3 bars" in result.output
        assert "End of data" in result.output

    @patch("qtrader.cli.commands.data.DataService")
    def test_raw_data_with_dividend_displays_correctly(self, mock_data_service_class):
        """Test bar with dividend displays correctly."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_data_service_class.return_value = mock_service

        mock_bar = Bar(
            trade_datetime=datetime(2020, 1, 2, 16, 0),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
            dividend=None,
        )
        mock_multi_bar = MultiBar(
            symbol="AAPL",
            trade_datetime=datetime(2020, 1, 2, 16, 0),
            unadjusted=mock_bar,
            adjusted=mock_bar,
            total_return=mock_bar,
        )
        mock_service.load_symbol.return_value = iter([mock_multi_bar])

        # Act
        result = runner.invoke(
            data_group,
            [
                "raw",
                "--symbol",
                "AAPL",
                "--start-date",
                "2020-01-01",
                "--end-date",
                "2020-01-31",
            ],
        )

        # Assert
        assert result.exit_code == 0
        assert "Loaded 1 bars" in result.output


class TestUpdateDatasetSymbolParsing:
    """Test symbol list parsing in update command."""

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_parses_comma_separated_symbols_with_spaces(self, mock_update_service_class):
        """Test symbols with spaces are parsed correctly."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = (["AAPL", "MSFT", "TSLA"], "3 symbols")
        mock_service.update_symbols.return_value = iter([])

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset", "--symbols", "AAPL, MSFT , TSLA"])

        # Assert
        assert result.exit_code == 0
        # Verify symbols were trimmed
        call_args = mock_service.get_symbols_to_update.call_args[0][0]
        assert call_args == ["AAPL", "MSFT", "TSLA"]

    @patch("qtrader.cli.commands.data.UpdateService")
    def test_update_with_single_symbol(self, mock_update_service_class):
        """Test update with single symbol."""
        # Arrange
        runner = CliRunner()
        mock_service = MagicMock()
        mock_update_service_class.return_value = mock_service
        mock_service.get_symbols_to_update.return_value = (["AAPL"], "1 symbol")
        mock_service.update_symbols.return_value = iter([])

        # Act
        result = runner.invoke(data_group, ["update", "--dataset", "test-dataset", "--symbols", "AAPL"])

        # Assert
        assert result.exit_code == 0
        call_args = mock_service.get_symbols_to_update.call_args[0][0]
        assert call_args == ["AAPL"]


class TestListDatasetsCommand:
    """Test the list datasets command."""

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_success(self, mock_resolver_class):
        """Test successful dataset listing."""
        # Arrange
        runner = CliRunner()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.config_path = "/path/to/data_sources.yaml"
        mock_resolver.list_sources.return_value = [
            "schwab-us-equity-1d-adjusted",
            "algoseek-us-equity-1d-unadjusted",
        ]
        mock_resolver.get_source_config.side_effect = [
            {
                "provider": "schwab",
                "adapter": "schwabOHLC",
                "asset_class": "equity",
            },
            {
                "provider": "algoseek",
                "adapter": "algoseekOHLC",
                "asset_class": "equity",
            },
        ]

        # Act
        result = runner.invoke(data_group, ["list"])

        # Assert
        assert result.exit_code == 0
        assert "Found 2 configured dataset(s)" in result.output
        assert "schwab-us-equity-1d-adjusted" in result.output
        assert "algoseek-us-equity-1d-unadjusted" in result.output
        assert "schwab" in result.output
        assert "algoseek" in result.output

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_verbose(self, mock_resolver_class):
        """Test dataset listing with verbose flag."""
        # Arrange
        runner = CliRunner()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.config_path = "/path/to/data_sources.yaml"
        mock_resolver.list_sources.return_value = ["schwab-us-equity-1d-adjusted"]
        mock_resolver.get_source_config.return_value = {
            "provider": "schwab",
            "adapter": "schwabOHLC",
            "asset_class": "equity",
            "frequency": "1d",
            "cache_root": "/cache/path",
        }

        # Act
        result = runner.invoke(data_group, ["list", "--verbose"])

        # Assert
        assert result.exit_code == 0
        assert "schwab-us-equity-1d-adjusted" in result.output
        assert "1d" in result.output  # Frequency shown in verbose
        assert "✓" in result.output or "✗" in result.output  # Cache status

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_no_cache(self, mock_resolver_class):
        """Test dataset listing shows no cache indicator."""
        # Arrange
        runner = CliRunner()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.config_path = "/path/to/data_sources.yaml"
        mock_resolver.list_sources.return_value = ["algoseek-us-equity-1d"]
        mock_resolver.get_source_config.return_value = {
            "provider": "algoseek",
            "adapter": "algoseekOHLC",
            "asset_class": "equity",
            "frequency": "1d",
            # No cache_root
        }

        # Act
        result = runner.invoke(data_group, ["list", "--verbose"])

        # Assert
        assert result.exit_code == 0
        assert "algoseek-us-equity-1d" in result.output

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_empty(self, mock_resolver_class):
        """Test dataset listing with no configured datasets."""
        # Arrange
        runner = CliRunner()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.list_sources.return_value = []

        # Act
        result = runner.invoke(data_group, ["list"])

        # Assert
        assert result.exit_code == 0
        assert "No datasets configured" in result.output

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_config_not_found(self, mock_resolver_class):
        """Test dataset listing when config file not found."""
        # Arrange
        runner = CliRunner()
        mock_resolver_class.side_effect = FileNotFoundError("data_sources.yaml not found")

        # Act
        result = runner.invoke(data_group, ["list"])

        # Assert
        assert result.exit_code == 1
        assert "data_sources.yaml not found" in result.output

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_handles_missing_fields(self, mock_resolver_class):
        """Test dataset listing handles missing configuration fields gracefully."""
        # Arrange
        runner = CliRunner()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.config_path = "/path/to/data_sources.yaml"
        mock_resolver.list_sources.return_value = ["incomplete-dataset"]
        mock_resolver.get_source_config.return_value = {
            "adapter": "someAdapter",
            # Missing provider, asset_class, etc.
        }

        # Act
        result = runner.invoke(data_group, ["list"])

        # Assert
        assert result.exit_code == 0
        assert "incomplete-dataset" in result.output
        assert "N/A" in result.output  # Should show N/A for missing fields

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_sorted_output(self, mock_resolver_class):
        """Test that datasets are displayed in sorted order."""
        # Arrange
        runner = CliRunner()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.config_path = "/path/to/data_sources.yaml"
        # Return datasets in random order
        mock_resolver.list_sources.return_value = [
            "zebra-dataset",
            "alpha-dataset",
            "beta-dataset",
        ]
        mock_resolver.get_source_config.return_value = {
            "provider": "test",
            "adapter": "testAdapter",
            "asset_class": "equity",
        }

        # Act
        result = runner.invoke(data_group, ["list"])

        # Assert
        assert result.exit_code == 0
        # All datasets should appear in output (order checked by position)
        alpha_pos = result.output.find("alpha-dataset")
        beta_pos = result.output.find("beta-dataset")
        zebra_pos = result.output.find("zebra-dataset")
        # Should be in alphabetical order
        assert alpha_pos < beta_pos < zebra_pos

    @patch("qtrader.cli.commands.data.DataSourceResolver")
    def test_list_datasets_exception_handling(self, mock_resolver_class):
        """Test dataset listing handles unexpected exceptions."""
        # Arrange
        runner = CliRunner()
        mock_resolver = MagicMock()
        mock_resolver_class.return_value = mock_resolver
        mock_resolver.list_sources.side_effect = Exception("Unexpected error")

        # Act
        result = runner.invoke(data_group, ["list"])

        # Assert
        assert result.exit_code == 1
        assert "Unexpected error" in result.output
