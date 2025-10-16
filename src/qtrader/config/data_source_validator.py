"""Validation for data_sources.yaml configuration.

Validates:
- Source naming conventions
- Required metadata fields
- Metadata field values
- Duplicate configurations
"""

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


class DataSourceValidationError(Exception):
    """Raised when data source configuration is invalid."""

    pass


class DataSourceValidator:
    """Validates data_sources.yaml configuration."""

    # Naming convention pattern: <provider>-<region>-<asset>-<freq>[-variant]
    # Examples: algoseek-us-equity-1d-unadjusted, binance-crypto-1m
    NAME_PATTERN = re.compile(
        r"^[a-z0-9]+(?:-[a-z0-9]+)*$"  # lowercase alphanumeric with hyphens
    )

    # Frequency pattern: 1m, 5m, 15m, 1h, 1d, etc.
    FREQUENCY_PATTERN = re.compile(r"^\d+[mhd]$")

    # Required metadata fields
    REQUIRED_METADATA = ["provider", "asset_class", "data_type", "frequency"]

    # Valid values for metadata fields
    VALID_ASSET_CLASSES = [
        "equity",
        "futures",
        "options",
        "crypto",
        "forex",
        "fixed_income",
    ]
    VALID_DATA_TYPES = ["ohlcv", "trades", "quotes", "greeks", "fundamentals"]
    VALID_ADJUSTMENT_MODES = ["unadjusted", "adjusted", "split_adjusted"]

    @classmethod
    def validate_file(cls, config_path: str | Path) -> None:
        """
        Validate entire data_sources.yaml file.

        Args:
            config_path: Path to data_sources.yaml file

        Raises:
            DataSourceValidationError: If validation fails
            FileNotFoundError: If file not found
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        if not config or "data_sources" not in config:
            raise DataSourceValidationError("Missing 'data_sources' key in config")

        sources = config["data_sources"]
        if not sources:
            raise DataSourceValidationError("No data sources defined")

        errors: List[str] = []

        for source_name, source_config in sources.items():
            try:
                cls.validate_source(source_name, source_config)
            except DataSourceValidationError as e:
                errors.append(f"Source '{source_name}': {e}")

        # Check for duplicate configurations
        duplicate_errors = cls._check_duplicates(sources)
        errors.extend(duplicate_errors)

        if errors:
            error_msg = "\n".join([f"  - {e}" for e in errors])
            raise DataSourceValidationError(f"Data source validation failed:\n{error_msg}")

    @classmethod
    def validate_source(cls, source_name: str, source_config: Dict[str, Any]) -> None:
        """
        Validate a single data source configuration.

        Args:
            source_name: Name of the data source
            source_config: Configuration dict

        Raises:
            DataSourceValidationError: If validation fails
        """
        errors: List[str] = []

        # Validate naming convention
        if not cls.NAME_PATTERN.match(source_name):
            errors.append(
                f"Invalid name format: '{source_name}'. "
                "Use lowercase alphanumeric with hyphens, e.g., 'provider-region-asset-freq'"
            )

        # Check required metadata fields
        missing_fields = [field for field in cls.REQUIRED_METADATA if field not in source_config]
        if missing_fields:
            errors.append(f"Missing required metadata fields: {missing_fields}")

        # Validate metadata field values
        if "asset_class" in source_config:
            if source_config["asset_class"] not in cls.VALID_ASSET_CLASSES:
                errors.append(
                    f"Invalid asset_class: '{source_config['asset_class']}'. Valid values: {cls.VALID_ASSET_CLASSES}"
                )

        if "data_type" in source_config:
            if source_config["data_type"] not in cls.VALID_DATA_TYPES:
                errors.append(
                    f"Invalid data_type: '{source_config['data_type']}'. Valid values: {cls.VALID_DATA_TYPES}"
                )

        if "adjustment_mode" in source_config:
            if source_config["adjustment_mode"] not in cls.VALID_ADJUSTMENT_MODES:
                errors.append(
                    f"Invalid adjustment_mode: '{source_config['adjustment_mode']}'. "
                    f"Valid values: {cls.VALID_ADJUSTMENT_MODES}"
                )

        if "frequency" in source_config:
            freq = source_config["frequency"]
            if not cls.FREQUENCY_PATTERN.match(str(freq)):
                errors.append(f"Invalid frequency format: '{freq}'. Use format like: 1m, 5m, 15m, 1h, 1d")

        # Check for adapter field
        if "adapter" not in source_config:
            errors.append("Missing 'adapter' field in configuration")

        if errors:
            raise DataSourceValidationError("; ".join(errors))

    @classmethod
    def _check_duplicates(cls, sources: Dict[str, Dict[str, Any]]) -> List[str]:
        """
        Check for duplicate configurations.

        Returns configurations that have identical metadata but different names.

        Args:
            sources: Dict of source configurations

        Returns:
            List of error messages for duplicates found
        """
        errors: List[str] = []
        seen_configs: Dict[str, List[str]] = {}

        for source_name, source_config in sources.items():
            # Create signature from metadata
            signature = (
                source_config.get("provider", ""),
                source_config.get("asset_class", ""),
                source_config.get("data_type", ""),
                source_config.get("frequency", ""),
                source_config.get("region", ""),
                source_config.get("adjustment_mode", ""),
            )

            # Convert to string for dict key
            sig_str = str(signature)

            if sig_str in seen_configs:
                seen_configs[sig_str].append(source_name)
            else:
                seen_configs[sig_str] = [source_name]

        # Report duplicates
        for sig_str, source_names in seen_configs.items():
            if len(source_names) > 1:
                errors.append(f"Duplicate configurations found: {source_names}. These sources have identical metadata.")

        return errors


def validate_data_sources(config_path: str | Path) -> None:
    """
    Convenience function to validate data_sources.yaml.

    Args:
        config_path: Path to data_sources.yaml file

    Raises:
        DataSourceValidationError: If validation fails
        FileNotFoundError: If file not found

    Example:
        >>> from qtrader.config.data_source_validator import validate_data_sources
        >>> validate_data_sources("config/data_sources.yaml")
    """
    DataSourceValidator.validate_file(config_path)
