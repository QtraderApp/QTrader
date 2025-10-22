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

from qtrader.models.instrument import Instrument
from qtrader.services.data.source_selector import DataSourceSelector

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

        Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.
        Nested dicts are processed recursively.

        Args:
            config: Configuration dict (possibly nested).

        Returns:
            Config with environment variables substituted.

        Raises:
            KeyError: If referenced environment variable not found and no default.
        """
        result: Dict[str, Any] = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract variable name and optional default
                var_expr = value[2:-1]  # Remove ${ and }

                if ":-" in var_expr:
                    # Handle ${VAR:-default} syntax
                    var_name, default_value = var_expr.split(":-", 1)
                    result[key] = os.environ.get(var_name, default_value)
                else:
                    # Handle ${VAR} syntax (no default)
                    result[key] = os.environ[var_expr]
            elif isinstance(value, dict):
                result[key] = self._substitute_env_vars(value)
            else:
                result[key] = value
        return result

    def _get_adapter_class(self, adapter_name: str) -> type:
        """
        Get adapter class by name.

        Args:
            adapter_name: Adapter identifier (e.g., "algoseekOHLC").

        Returns:
            Adapter class.

        Raises:
            ValueError: If adapter not found.
        """
        if adapter_name in self._adapter_cache:
            return self._adapter_cache[adapter_name]

        # Map adapter names to classes
        adapter_map = {
            "algoseekOHLC": "qtrader.services.data.adapters.algoseek.AlgoseekOHLCVendorAdapter",
            "schwabOHLC": "qtrader.services.data.adapters.schwab.SchwabOHLCAdapter",
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

    def resolve_by_selector(self, selector: DataSourceSelector, instrument: Instrument):
        """
        Resolve data source using DataSourceSelector.

        Finds the best matching data source based on selector criteria
        (provider, asset_class, frequency, etc.). Supports fallback providers.

        Args:
            selector: Structured selector with matching criteria.
            instrument: Instrument to load data for.

        Returns:
            Instantiated data adapter.

        Raises:
            ValueError: If no matching source found.
            KeyError: If adapter cannot be loaded.

        Examples:
            >>> # Match by provider
            >>> selector = DataSourceSelector(provider="schwab", asset_class=AssetClass.EQUITY)
            >>> adapter = resolver.resolve_by_selector(selector, instrument)
            >>>
            >>> # Match any equity source with fallback
            >>> selector = DataSourceSelector(
            ...     asset_class=AssetClass.EQUITY,
            ...     fallback_providers=["schwab", "algoseek"]
            ... )
            >>> adapter = resolver.resolve_by_selector(selector, instrument)
        """
        # Find matching sources
        matches = []
        for source_name, source_config in self.sources.items():
            if selector.matches(source_config):
                matches.append((source_name, source_config))

        if not matches:
            available = list(self.sources.keys())
            raise ValueError(
                f"No data source matches selector: {selector.to_tag()}\n"
                f"Available sources: {available}\n"
                f"Selector criteria: provider={selector.provider}, "
                f"asset_class={selector.asset_class}, frequency={selector.frequency}"
            )

        # Use first match (or implement priority logic later)
        if len(matches) > 1:
            logger.info(
                "resolver.multiple_matches",
                selector=selector.to_tag(),
                matches=[m[0] for m in matches],
                selected=matches[0][0],
            )

        source_name, source_config = matches[0]

        # Try primary source
        try:
            return self._create_adapter(source_name, source_config, instrument)
        except Exception as e:
            logger.warning(
                "resolver.primary_failed",
                source=source_name,
                error=str(e),
            )

            # Try fallback providers
            for fallback_provider in selector.fallback_providers:
                try:
                    fallback_selector = DataSourceSelector(
                        provider=fallback_provider,
                        asset_class=selector.asset_class,
                        data_type=selector.data_type,
                        frequency=selector.frequency,
                        exchange=selector.exchange,
                        region=selector.region,
                    )
                    logger.info(
                        "resolver.trying_fallback",
                        fallback=fallback_provider,
                    )
                    return self.resolve_by_selector(fallback_selector, instrument)
                except Exception:
                    continue

            # No fallbacks worked, raise original error
            raise

    def _create_adapter(self, source_name: str, source_config: Dict[str, Any], instrument: Instrument):
        """
        Create adapter instance from config.

        Args:
            source_name: Name of data source.
            source_config: Source configuration dict.
            instrument: Instrument to load data for.

        Returns:
            Instantiated adapter.

        Raises:
            ValueError: If adapter cannot be loaded.
        """
        # Make a copy to avoid modifying original
        config = source_config.copy()

        # Substitute environment variables
        config = self._substitute_env_vars(config)

        # Get adapter class
        adapter_name = config.pop("adapter")
        adapter_class = self._get_adapter_class(adapter_name)

        # Instantiate adapter with config and instrument
        adapter = adapter_class(config, instrument)

        logger.info(
            "resolver.adapter_created",
            source=source_name,
            adapter=adapter_name,
            instrument=instrument.symbol,
        )

        return adapter

    def resolve_by_dataset(self, dataset: str, instrument: Instrument):
        """
        Resolve instrument to data adapter using explicit dataset name.

        This is the preferred method - it's explicit about which dataset to use,
        avoiding ambiguity and inference. Dataset config is the single source
        of truth for provider, asset type, frequency, etc.

        Args:
            dataset: Dataset name from data_sources.yaml (e.g., "schwab-us-equity-1d-adjusted")
            instrument: Instrument with symbol (and optional overrides)

        Returns:
            Instantiated data adapter.

        Raises:
            KeyError: If dataset not configured.
            ValueError: If adapter cannot be loaded.

        Examples:
            >>> resolver = DataSourceResolver()
            >>> instrument = Instrument("AAPL")
            >>> adapter = resolver.resolve_by_dataset("schwab-us-equity-1d-adjusted", instrument)
            >>> bars = adapter.read_bars(start_date="2024-01-01", end_date="2024-12-31")
        """
        if dataset not in self.sources:
            raise KeyError(f"Dataset '{dataset}' not configured. Available: {list(self.sources.keys())}")

        source_config = self.sources[dataset]
        return self._create_adapter(dataset, source_config, instrument)

    def resolve(self, instrument: Instrument):
        """
        DEPRECATED: Resolve instrument to data adapter by inferring dataset.

        This method is deprecated because it requires instrument to specify
        data_source, which duplicates what the config already knows.

        Use resolve_by_dataset() instead with explicit dataset name.

        Args:
            instrument: Instrument specification (should have data_source for backward compat)

        Returns:
            Instantiated data adapter.

        Raises:
            KeyError: If data source not configured.
            ValueError: If adapter cannot be loaded.
            AttributeError: If instrument doesn't have data_source (new Instrument API)
        """
        logger.warning(
            "resolve() is deprecated. Use resolve_by_dataset() with explicit dataset name. "
            "This avoids duplication between config and instrument metadata."
        )

        # For backward compatibility with OLD Instrument API that had data_source field
        # New Instrument only has (symbol, frequency, metadata)
        # So this will fail gracefully with clear error message

        try:
            # Try to get data_source from metadata (temporary backward compat hack)
            if "data_source" in instrument.metadata:
                source_name = instrument.metadata["data_source"]
            else:
                raise AttributeError(
                    "Instrument is missing 'data_source'. "
                    "New Instrument API only has (symbol, frequency, metadata). "
                    "Use resolve_by_dataset(dataset, instrument) instead. "
                    "Example: resolver.resolve_by_dataset('schwab-us-equity-1d-adjusted', Instrument('AAPL'))"
                )
        except AttributeError:
            raise AttributeError(
                "Instrument is missing 'data_source'. "
                "New Instrument API only has (symbol, frequency, metadata). "
                "Use resolve_by_dataset(dataset, instrument) instead. "
                "Example: resolver.resolve_by_dataset('schwab-us-equity-1d-adjusted', Instrument('AAPL'))"
            )

        if source_name not in self.sources:
            # Backward compatibility: Try to find source by provider name
            # E.g., "algoseek" → "algoseek-us-equity-1d-unadjusted"
            matching_sources = [name for name, config in self.sources.items() if config.get("provider") == source_name]

            if not matching_sources:
                raise KeyError(f"Data source '{source_name}' not configured. Available: {list(self.sources.keys())}")

            # Use first matching source
            if len(matching_sources) > 1:
                logger.warning(
                    f"Multiple sources match provider '{source_name}': {matching_sources}. "
                    f"Using first match: {matching_sources[0]}"
                )
            source_name = matching_sources[0]

        source_config = self.sources[source_name]
        return self._create_adapter(source_name, source_config, instrument)

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
