"""
Data source resolver.

Maps logical Instrument specifications to physical data adapters using
external configuration (data_sources.yaml). Enables environment-specific
configuration without changing strategy code.
"""

import os
from pathlib import Path
from typing import Any, Dict

import structlog
import yaml

from qtrader.adapters.base import DataAdapter
from qtrader.models.instrument import Instrument

logger = structlog.get_logger()


class DataSourceResolver:
    """
    Resolves logical Instrument to physical data adapter.

    Loads data source configuration from YAML and instantiates appropriate
    adapters. Supports environment variable substitution (${VAR}).

    Configuration file locations (checked in order):
    1. Path provided to __init__
    2. ./config/data_sources.yaml (project-relative)
    3. ~/.qtrader/data_sources.yaml (user home)

    Example data_sources.yaml:
        data_sources:
          algoseek:
            adapter: algoseekOHLC
            root_path: "data/us-equity-daily-ohlc-standard-adjusted-secid-all-parquet-sample"
            mode: standard_adjusted
            path_template: "{root_path}/SecId={secid}/*.parquet"
            symbol_map: "data/equity_security_master_sample.csv"

          csv_samples:
            adapter: csv
            root_path: "data/csv"

          database:
            adapter: postgres_adapter
            connection_string: "${DB_CONNECTION_STRING}"
            schema: "market_data"

          iqfeed:
            adapter: iqfeed_api
            api_key: "${IQFEED_API_KEY}"

    Usage:
        >>> resolver = DataSourceResolver()
        >>> instrument = Instrument("AAPL", InstrumentType.EQUITY, DataSource.ALGOSEEK)
        >>> adapter = resolver.resolve(instrument)
        >>> config = DataConfig(bar_schema=bar_schema)
        >>> bars = adapter.read_bars(config)
    """

    def __init__(self, config_path: str | None = None):
        """
        Initialize resolver.

        Args:
            config_path: Path to data_sources.yaml. If None, searches default locations.

        Raises:
            FileNotFoundError: If config file not found in any location.
            ValueError: If config format is invalid.
        """
        self.config_path = self._find_config(config_path)
        self.sources = self._load_config(self.config_path)
        self._adapter_cache: Dict[str, type] = {}

    def _find_config(self, config_path: str | None) -> Path:
        """
        Find data sources config file.

        Args:
            config_path: Explicit path or None to search defaults.

        Returns:
            Path to config file.

        Raises:
            FileNotFoundError: If no config file found.
        """
        if config_path:
            path = Path(config_path).expanduser()
            if path.exists():
                return path
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Try default locations
        candidates = [
            Path("config/data_sources.yaml"),  # Project-relative
            Path.home() / ".qtrader" / "data_sources.yaml",  # User home
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(f"data_sources.yaml not found in any location: {candidates}")

    def _load_config(self, path: Path) -> Dict[str, Dict[str, Any]]:
        """
        Load and validate configuration.

        Args:
            path: Path to YAML config file.

        Returns:
            Dict mapping data source name to configuration.

        Raises:
            ValueError: If config format is invalid.
        """
        with open(path) as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict) or "data_sources" not in config:
            raise ValueError(f"Invalid config format in {path}. Expected 'data_sources' key.")

        sources = config["data_sources"]

        # Validate each source has required 'adapter' field
        for source_name, source_config in sources.items():
            if "adapter" not in source_config:
                raise ValueError(f"Data source '{source_name}' missing required 'adapter' field")

        return dict(sources)

    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Substitute environment variables in config values.

        Replaces ${VAR_NAME} with environment variable value.

        Args:
            config: Configuration dict (possibly nested).

        Returns:
            Config with environment variables substituted.

        Raises:
            KeyError: If referenced environment variable not found.
        """
        result: Dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                var_name = value[2:-1]
                result[key] = os.environ[var_name]
            elif isinstance(value, dict):
                result[key] = self._substitute_env_vars(value)
            else:
                result[key] = value
        return result

    def _get_adapter_class(self, adapter_name: str) -> type:
        """
        Get adapter class by name.

        Args:
            adapter_name: Adapter identifier (e.g., "algoseekOHLC", "csv").

        Returns:
            Adapter class.

        Raises:
            ValueError: If adapter not found.
        """
        if adapter_name in self._adapter_cache:
            return self._adapter_cache[adapter_name]

        # Map adapter names to classes
        adapter_map = {
            "algoseekOHLC": "qtrader.adapters.algoseek.AlgoseekOHLCAdapter",
            "csv": "qtrader.adapters.csv_adapter.CSVAdapter",
            # Add more adapters as needed
        }

        if adapter_name not in adapter_map:
            raise ValueError(f"Unknown adapter: {adapter_name}. Available: {list(adapter_map.keys())}")

        # Dynamic import
        module_path, class_name = adapter_map[adapter_name].rsplit(".", 1)
        module = __import__(module_path, fromlist=[class_name])
        adapter_class: type = getattr(module, class_name)

        self._adapter_cache[adapter_name] = adapter_class
        return adapter_class

    def resolve(self, instrument: Instrument) -> DataAdapter:
        """
        Resolve instrument to data adapter.

        Args:
            instrument: Logical instrument specification.

        Returns:
            Instantiated data adapter.

        Raises:
            KeyError: If data source not configured.
            ValueError: If adapter cannot be loaded.
        """
        source_name = instrument.data_source.value

        if source_name not in self.sources:
            raise KeyError(f"Data source '{source_name}' not configured. Available: {list(self.sources.keys())}")

        source_config = self.sources[source_name].copy()

        # Substitute environment variables
        source_config = self._substitute_env_vars(source_config)

        # Get adapter class
        adapter_name = source_config.pop("adapter")
        adapter_class = self._get_adapter_class(adapter_name)

        # Instantiate adapter with config and instrument
        adapter: DataAdapter = adapter_class(source_config, instrument)
        return adapter

    def list_sources(self) -> list[str]:
        """Get list of configured data sources."""
        return list(self.sources.keys())

    def get_source_config(self, source_name: str) -> Dict[str, Any]:
        """
        Get configuration for a data source.

        Args:
            source_name: Data source name.

        Returns:
            Configuration dict (with env vars substituted).

        Raises:
            KeyError: If source not configured.
        """
        if source_name not in self.sources:
            raise KeyError(f"Data source '{source_name}' not configured")

        config = self.sources[source_name].copy()
        return self._substitute_env_vars(config)
