"""Unit tests for CLI data commands.

Tests the thin CLI orchestration layer without executing real commands.
"""

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from qtrader.cli.commands.data import data_group


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
