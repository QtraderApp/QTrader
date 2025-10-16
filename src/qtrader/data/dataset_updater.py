"""
Generic dataset updater.

Provides adapter-agnostic functionality to update cached data from any
source (Schwab, Algoseek, future providers). Scans cache directories,
detects which symbols need updates, and calls adapter-specific update
methods.

Key features:
- Works with any adapter that supports incremental updates
- Scans cache for symbols requiring updates
- Batch updates with progress tracking
- Dry-run mode for planning
- Verbose logging for debugging
"""

from datetime import date
from pathlib import Path
from typing import Iterator, List, Optional

import structlog

from qtrader.adapters.resolver import DataSourceResolver
from qtrader.models.instrument import Instrument

logger = structlog.get_logger()


class DatasetUpdateResult:
    """
    Result of updating a single symbol.

    Attributes:
        symbol: Stock symbol
        success: Whether update succeeded
        bars_added: Number of new bars added (0 if error or already up-to-date)
        start_date: First date in update range
        end_date: Last date in update range
        error: Error message if failed
    """

    def __init__(
        self,
        symbol: str,
        success: bool,
        bars_added: int = 0,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        error: Optional[str] = None,
    ):
        """
        Initialize update result.

        Args:
            symbol: Stock symbol
            success: Whether update succeeded
            bars_added: Number of new bars added
            start_date: First date in update range
            end_date: Last date in update range
            error: Error message if failed
        """
        self.symbol = symbol
        self.success = success
        self.bars_added = bars_added
        self.start_date = start_date
        self.end_date = end_date
        self.error = error

    def __repr__(self) -> str:
        """String representation."""
        if self.success:
            if self.bars_added == 0:
                return f"DatasetUpdateResult(symbol={self.symbol}, already_current=True)"
            return (
                f"DatasetUpdateResult(symbol={self.symbol}, bars_added={self.bars_added}, "
                f"range={self.start_date} to {self.end_date})"
            )
        return f"DatasetUpdateResult(symbol={self.symbol}, error={self.error})"


class DatasetUpdater:
    """
    Generic dataset updater for any data source.

    Handles incremental updates of cached data across all supported adapters.
    Works by:
    1. Resolving dataset configuration from data_sources.yaml
    2. Instantiating appropriate adapter
    3. Scanning cache directory for symbols
    4. Calling adapter's update_to_latest() if available
    5. Falling back to full refetch if adapter doesn't support incremental

    Example:
        >>> # Update all symbols in Schwab dataset
        >>> updater = DatasetUpdater("schwab-us-equity-1d-adjusted")
        >>> results = list(updater.update_all(dry_run=False, verbose=True))
        >>>
        >>> # Update specific symbols
        >>> results = list(updater.update_symbols(["AAPL", "TSLA"], dry_run=False))
        >>>
        >>> # Check what would be updated (dry run)
        >>> results = list(updater.update_all(dry_run=True))
        >>> for result in results:
        ...     print(f"{result.symbol}: {result.bars_added} bars needed")

    Attributes:
        dataset_name: Dataset identifier from data_sources.yaml
        resolver: Data source resolver for adapter instantiation
        adapter: Instantiated data adapter
        adapter_config: Adapter configuration from YAML
    """

    def __init__(
        self,
        dataset_name: str,
        config_path: Optional[str] = None,
    ):
        """
        Initialize dataset updater.

        Args:
            dataset_name: Dataset identifier from data_sources.yaml
                         (e.g., "schwab-us-equity-1d-adjusted")
            config_path: Optional path to data_sources.yaml. If None, uses default.

        Raises:
            ValueError: If dataset not found in configuration
            ValueError: If adapter doesn't support updates
        """
        self.dataset_name = dataset_name
        self.resolver = DataSourceResolver(config_path)

        # Get dataset configuration
        if dataset_name not in self.resolver.sources:
            available = list(self.resolver.sources.keys())
            raise ValueError(f"Dataset '{dataset_name}' not found. Available: {available}")

        self.adapter_config = self.resolver.sources[dataset_name]
        adapter_name = self.adapter_config["adapter"]

        # Get adapter class (but don't instantiate yet - we need symbol-specific instruments)
        self.adapter_class = self.resolver._get_adapter_class(adapter_name)

        # Verify adapter class supports updates (has update_to_latest method)
        if not hasattr(self.adapter_class, "update_to_latest"):
            raise ValueError(
                f"Adapter '{adapter_name}' does not support incremental updates. "
                f"Add update_to_latest() method to enable dataset updates."
            )

        logger.info(
            "dataset_updater.initialized",
            dataset=dataset_name,
            adapter=adapter_name,
            cache_enabled=self._supports_caching(),
        )

    def _supports_caching(self) -> bool:
        """
        Check if adapter supports caching.

        Returns:
            True if adapter class has cache_root in its signature
        """
        # Check config for cache_root (indicates caching support)
        return "cache_root" in self.adapter_config

    def _get_cache_root(self) -> Optional[Path]:
        """
        Get cache root directory.

        Returns:
            Path to cache root or None if caching not supported
        """
        if not self._supports_caching():
            return None
        return Path(self.adapter_config["cache_root"])

    def _scan_cached_symbols(self) -> List[str]:
        """
        Scan cache directory for symbols.

        Returns:
            List of symbols with cached data (empty if caching not supported)
        """
        cache_root = self._get_cache_root()
        if not cache_root or not cache_root.exists():
            logger.warning(
                "dataset_updater.no_cache",
                dataset=self.dataset_name,
                cache_root=str(cache_root) if cache_root else None,
            )
            return []

        # Each symbol has a subdirectory with data.parquet
        symbols = []
        for symbol_dir in cache_root.iterdir():
            if symbol_dir.is_dir() and (symbol_dir / "data.parquet").exists():
                symbols.append(symbol_dir.name)

        logger.info(
            "dataset_updater.cache_scan",
            dataset=self.dataset_name,
            symbols_found=len(symbols),
        )
        return sorted(symbols)

    def update_symbol(
        self,
        symbol: str,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> DatasetUpdateResult:
        """
        Update a single symbol to latest available data.

        Creates a symbol-specific adapter and calls its update_to_latest() method.
        If adapter supports incremental updates, only fetches new bars.

        Args:
            symbol: Stock symbol to update
            dry_run: If True, only check what would be updated (no API calls)
            verbose: Enable detailed logging

        Returns:
            Update result with bars added, date range, or error

        Example:
            >>> updater = DatasetUpdater("schwab-us-equity-1d-adjusted")
            >>> result = updater.update_symbol("AAPL", dry_run=False)
            >>> if result.success:
            ...     print(f"Added {result.bars_added} bars")
            ... else:
            ...     print(f"Error: {result.error}")
        """
        try:
            if verbose:
                logger.info(
                    "dataset_updater.update_symbol.start",
                    symbol=symbol,
                    dataset=self.dataset_name,
                    dry_run=dry_run,
                )

            # Create instrument for this symbol
            instrument = Instrument(symbol=symbol)

            # Create adapter instance for this symbol
            adapter = self.resolver.resolve_by_dataset(self.dataset_name, instrument)

            # Call adapter's update method
            # Adapters should return (bars_added, start_date, end_date)
            result = adapter.update_to_latest(dry_run=dry_run)

            # Parse adapter result
            # Expected format: (bars_added: int, start_date: date | None, end_date: date | None)
            if isinstance(result, tuple) and len(result) == 3:
                bars_added, start_date, end_date = result
            else:
                # Fallback for adapters that don't return structured result
                bars_added = result if isinstance(result, int) else 0
                start_date = None
                end_date = None

            if verbose:
                logger.info(
                    "dataset_updater.update_symbol.success",
                    symbol=symbol,
                    bars_added=bars_added,
                    start_date=str(start_date) if start_date else None,
                    end_date=str(end_date) if end_date else None,
                )

            return DatasetUpdateResult(
                symbol=symbol,
                success=True,
                bars_added=bars_added,
                start_date=start_date,
                end_date=end_date,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                "dataset_updater.update_symbol.error",
                symbol=symbol,
                dataset=self.dataset_name,
                error=error_msg,
            )
            return DatasetUpdateResult(
                symbol=symbol,
                success=False,
                error=error_msg,
            )

    def update_symbols(
        self,
        symbols: List[str],
        dry_run: bool = False,
        verbose: bool = False,
    ) -> Iterator[DatasetUpdateResult]:
        """
        Update multiple symbols.

        Yields results as each symbol is processed (not batched).

        Args:
            symbols: List of symbols to update
            dry_run: If True, only check what would be updated
            verbose: Enable detailed logging

        Yields:
            Update result for each symbol

        Example:
            >>> updater = DatasetUpdater("schwab-us-equity-1d-adjusted")
            >>> symbols = ["AAPL", "TSLA", "NVDA"]
            >>> for result in updater.update_symbols(symbols, verbose=True):
            ...     print(f"{result.symbol}: {result.bars_added} bars")
        """
        logger.info(
            "dataset_updater.update_symbols.start",
            dataset=self.dataset_name,
            symbol_count=len(symbols),
            dry_run=dry_run,
        )

        for symbol in symbols:
            yield self.update_symbol(
                symbol=symbol,
                dry_run=dry_run,
                verbose=verbose,
            )

    def update_all(
        self,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> Iterator[DatasetUpdateResult]:
        """
        Update all symbols in cache.

        Scans cache directory for symbols and updates each one.

        Args:
            dry_run: If True, only check what would be updated
            verbose: Enable detailed logging

        Yields:
            Update result for each symbol

        Example:
            >>> updater = DatasetUpdater("schwab-us-equity-1d-adjusted")
            >>> results = list(updater.update_all(dry_run=False))
            >>> successful = [r for r in results if r.success]
            >>> print(f"Updated {len(successful)} symbols")
        """
        symbols = self._scan_cached_symbols()

        if not symbols:
            logger.warning(
                "dataset_updater.update_all.no_symbols",
                dataset=self.dataset_name,
            )
            return

        logger.info(
            "dataset_updater.update_all.start",
            dataset=self.dataset_name,
            symbol_count=len(symbols),
            dry_run=dry_run,
        )

        yield from self.update_symbols(
            symbols=symbols,
            dry_run=dry_run,
            verbose=verbose,
        )
