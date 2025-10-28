"""
Library Registry System - Auto-Discovery and Validation

Provides automatic discovery and registration of library components:
- Indicators (built-in and custom)
- Strategies (built-in and custom)
- Risk Policies (built-in and custom)
- Metrics (built-in and custom)

Philosophy:
- "Convention over configuration" - just create a class that inherits from BaseXYZ
- Auto-discover components from buildin/ and custom library paths
- Validate ABC compliance at registration time
- Provide clean API for component lookup by name

Usage:
    # Auto-discover all indicators
    indicator_registry = IndicatorRegistry()
    indicator_registry.discover()

    # List available indicators
    print(indicator_registry.list_names())

    # Get indicator class by name
    SMA = indicator_registry.get("sma")
    indicator = SMA(period=20)
"""

import importlib
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar

from qtrader.libraries.indicators.base import BaseIndicator
from qtrader.libraries.strategies.base import Strategy, StrategyConfig
from qtrader.libraries.strategies.loader import StrategyLoader, StrategyLoadError

# Type variable for generic registry
T = TypeVar("T")


class RegistryError(Exception):
    """Base exception for registry errors."""

    pass


class ComponentNotFoundError(RegistryError):
    """Component not found in registry."""

    pass


class DuplicateComponentError(RegistryError):
    """Component already registered with this name."""

    pass


class InvalidComponentError(RegistryError):
    """Component does not meet requirements (ABC compliance, etc.)."""

    pass


class BaseRegistry(Generic[T]):
    """
    Base registry for auto-discovery and validation of library components.

    Responsibilities:
    - Scan directories for Python modules
    - Import and inspect classes
    - Validate ABC compliance
    - Register components by name
    - Provide lookup API

    Type Parameters:
        T: The base class type (e.g., BaseIndicator)
    """

    def __init__(self, base_class: Type[T], component_type: str):
        """
        Initialize registry.

        Args:
            base_class: The ABC base class (e.g., BaseIndicator)
            component_type: Human-readable component type (e.g., "indicator")
        """
        self.base_class = base_class
        self.component_type = component_type
        self._registry: dict[str, Type[T]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        component_class: Type[T],
        metadata: dict[str, Any] | None = None,
        allow_override: bool = False,
    ) -> None:
        """
        Register a component class.

        Args:
            name: Component name (registry key)
            component_class: The component class
            metadata: Optional metadata (source path, description, etc.)
            allow_override: Allow replacing existing component

        Raises:
            InvalidComponentError: If component doesn't inherit from base class
            DuplicateComponentError: If name already registered (and not allow_override)
        """
        # Validate inheritance
        if not issubclass(component_class, self.base_class):
            raise InvalidComponentError(f"{component_class.__name__} does not inherit from {self.base_class.__name__}")

        # Check for duplicates
        if name in self._registry and not allow_override:
            raise DuplicateComponentError(
                f"{self.component_type} '{name}' already registered "
                f"({self._registry[name].__module__}.{self._registry[name].__name__})"
            )

        # Register
        self._registry[name] = component_class
        self._metadata[name] = metadata or {}

    def get(self, name: str) -> Type[T]:
        """
        Get component class by name.

        Args:
            name: Component name

        Returns:
            Component class

        Raises:
            ComponentNotFoundError: If name not in registry
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise ComponentNotFoundError(f"{self.component_type} '{name}' not found. Available: {available}")

        return self._registry[name]

    def list_names(self) -> list[str]:
        """
        List all registered component names.

        Returns:
            Sorted list of component names
        """
        return sorted(self._registry.keys())

    def list_components(self) -> dict[str, Type[T]]:
        """
        Get all registered components.

        Returns:
            Dict mapping names to component classes
        """
        return dict(self._registry)

    def get_metadata(self, name: str) -> dict[str, Any]:
        """
        Get metadata for a component.

        Args:
            name: Component name

        Returns:
            Metadata dict

        Raises:
            ComponentNotFoundError: If name not in registry
        """
        if name not in self._metadata:
            raise ComponentNotFoundError(f"{self.component_type} '{name}' not found")

        return dict(self._metadata[name])

    def discover_from_module(
        self,
        module_path: Path,
        source_type: str = "unknown",
        name_transform: Callable[[str], str] | None = None,
    ) -> int:
        """
        Discover and register components from a Python module file.

        Args:
            module_path: Path to Python module (.py file)
            source_type: Source identifier ("buildin", "custom", etc.)
            name_transform: Optional function to transform class name to registry name

        Returns:
            Number of components registered from this module

        Example:
            registry.discover_from_module(
                Path("indicators/moving_averages.py"),
                source_type="buildin",
                name_transform=lambda name: name.lower()
            )
        """
        if not module_path.exists() or not module_path.is_file():
            return 0

        if module_path.name.startswith("_"):
            # Skip __init__.py, __pycache__, etc.
            return 0

        count = 0

        try:
            # Import module dynamically
            module_name = f"qtrader.registry.discovered.{module_path.stem}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)

            if spec is None or spec.loader is None:
                return 0

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Scan for classes that inherit from base_class
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip the base class itself
                if obj is self.base_class:
                    continue

                # Skip ABC classes
                if inspect.isabstract(obj):
                    continue

                # Check if it inherits from base class
                if not issubclass(obj, self.base_class):
                    continue

                # Transform name if function provided
                registry_name = name_transform(name) if name_transform else name.lower()

                # Register
                metadata = {
                    "source_type": source_type,
                    "module_path": str(module_path),
                    "class_name": name,
                    "module_name": module.__name__,
                }

                self.register(registry_name, obj, metadata, allow_override=False)
                count += 1

        except Exception as e:
            # Log but don't fail - some modules might have import issues
            # In production, use proper logging
            print(f"Warning: Failed to discover from {module_path}: {e}")

        return count

    def discover_from_directory(
        self,
        directory: Path,
        source_type: str = "unknown",
        recursive: bool = True,
        name_transform: Callable[[str], str] | None = None,
    ) -> int:
        """
        Discover and register components from a directory.

        Args:
            directory: Path to directory containing Python modules
            source_type: Source identifier ("buildin", "custom", etc.)
            recursive: Scan subdirectories recursively
            name_transform: Optional function to transform class name to registry name

        Returns:
            Total number of components registered

        Example:
            registry.discover_from_directory(
                Path("src/qtrader/libraries/indicators/buildin"),
                source_type="buildin",
                name_transform=lambda name: name.lower()
            )
        """
        if not directory.exists() or not directory.is_dir():
            return 0

        count = 0

        # Scan for .py files
        pattern = "**/*.py" if recursive else "*.py"
        for module_path in directory.glob(pattern):
            count += self.discover_from_module(module_path, source_type, name_transform)

        return count

    def clear(self) -> None:
        """Clear all registered components."""
        self._registry.clear()
        self._metadata.clear()

    def __len__(self) -> int:
        """Return number of registered components."""
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        """Check if component is registered."""
        return name in self._registry

    def __repr__(self) -> str:
        """String representation."""
        return f"<{self.__class__.__name__} {self.component_type}s={len(self)}>"


class IndicatorRegistry(BaseRegistry[BaseIndicator]):
    """
    Registry for technical indicators.

    Auto-discovers indicators from:
    - Built-in: src/qtrader/libraries/indicators/buildin/
    - Custom: my_library/indicators/ (from system config)

    Usage:
        registry = IndicatorRegistry()
        registry.discover()

        # List available
        print(registry.list_names())  # ['sma', 'ema', 'bollinger_bands', ...]

        # Get indicator class
        SMA = registry.get("sma")
        indicator = SMA(period=20)
    """

    def __init__(self):
        """Initialize indicator registry."""
        super().__init__(BaseIndicator, "indicator")

    def discover(
        self,
        buildin_path: Path | None = None,
        custom_paths: list[Path] | None = None,
    ) -> dict[str, int]:
        """
        Auto-discover indicators from built-in and custom paths.

        Args:
            buildin_path: Path to built-in indicators (default: auto-detect)
            custom_paths: Paths to custom indicator libraries (default: from system config)

        Returns:
            Dict with counts: {"buildin": X, "custom": Y}

        Example:
            counts = registry.discover()
            print(f"Found {counts['buildin']} built-in, {counts['custom']} custom")
        """
        counts = {"buildin": 0, "custom": 0}

        # Auto-detect built-in path if not provided
        if buildin_path is None:
            # Assume registry.py is in src/qtrader/libraries/
            buildin_path = Path(__file__).parent / "indicators" / "buildin"

        # Discover built-in indicators
        if buildin_path.exists():
            counts["buildin"] = self.discover_from_directory(
                buildin_path,
                source_type="buildin",
                recursive=True,
                name_transform=lambda name: name.lower(),
            )

        # Discover custom indicators
        if custom_paths:
            for custom_path in custom_paths:
                if custom_path.exists():
                    counts["custom"] += self.discover_from_directory(
                        custom_path,
                        source_type="custom",
                        recursive=True,
                        name_transform=lambda name: name.lower(),
                    )

        return counts


def get_indicator_registry() -> IndicatorRegistry:
    """
    Get singleton indicator registry instance.

    Returns:
        Global indicator registry

    Usage:
        registry = get_indicator_registry()
        SMA = registry.get("sma")
    """
    # For now, create new instance
    # In production, implement proper singleton pattern
    return IndicatorRegistry()


class StrategyRegistry(BaseRegistry[Strategy]):
    """
    Registry for trading strategies.

    Auto-discovers strategies from:
    - Built-in: src/qtrader/libraries/strategies/buildin/
    - Custom: my_library/strategies/ (from system config)

    Usage:
        registry = StrategyRegistry()
        registry.discover()

        # List available
        print(registry.list_names())  # ['bollinger_breakout', 'mean_reversion', ...]

        # Get strategy class
        BollingerBreakout = registry.get("bollinger_breakout")
        config = BollingerBreakoutConfig(bb_period=20)
        strategy = BollingerBreakout(config)
    """

    def __init__(self):
        """Initialize strategy registry."""
        super().__init__(Strategy, "strategy")
        self._configs: dict[str, StrategyConfig] = {}  # Store configs alongside classes
        self._loader = StrategyLoader()

    def load_from_directory(
        self,
        directory: Path,
        recursive: bool = False,
    ) -> dict[str, tuple[type[Strategy], StrategyConfig]]:
        """
        Load strategies using StrategyLoader (new approach).

        This is the recommended way to load strategies as it properly
        extracts StrategyConfig instances from the same file.

        Args:
            directory: Path to directory containing strategy .py files
            recursive: If True, search subdirectories

        Returns:
            Dict mapping strategy name to (StrategyClass, config)

        Raises:
            StrategyLoadError: If directory doesn't exist or loading fails

        Example:
            registry = StrategyRegistry()
            strategies = registry.load_from_directory(Path("my_library/strategies"))

            for name, (cls, config) in strategies.items():
                print(f"Loaded: {name} - {config.display_name}")
        """
        strategies: dict[str, tuple[type[Strategy], StrategyConfig]] = self._loader.load_from_directory(
            directory, recursive
        )

        # Register each strategy in the base registry
        for name, (strategy_class, config) in strategies.items():
            metadata = {
                "source_type": "custom",
                "class_name": strategy_class.__name__,
                "warmup_bars": config.warmup_bars,
                "display_name": config.display_name,
                "description": config.description,
            }

            self.register(name, strategy_class, metadata, allow_override=False)
            self._configs[name] = config

        return strategies

    def get_strategy_class(self, name: str) -> type[Strategy]:
        """
        Get strategy class by name.

        Args:
            name: Strategy name (from config.name)

        Returns:
            Strategy class

        Raises:
            ComponentNotFoundError: If strategy not found
        """
        return self.get(name)

    def get_strategy_config(self, name: str) -> StrategyConfig:
        """
        Get strategy config by name.

        Args:
            name: Strategy name (from config.name)

        Returns:
            StrategyConfig instance

        Raises:
            ComponentNotFoundError: If strategy not found
        """
        if name not in self._configs:
            raise ComponentNotFoundError(f"Strategy '{name}' not found or has no config")

        return self._configs[name]

    def list_strategies(self) -> dict[str, dict[str, Any]]:
        """
        List all registered strategies with their metadata.

        Returns:
            Dict mapping strategy names to metadata

        Example:
            strategies = registry.list_strategies()
            for name, info in strategies.items():
                print(f"{name}: {info['display_name']} (warmup: {info['warmup_bars']})")
        """
        result = {}
        for name in self.list_names():
            metadata = self.get_metadata(name)
            if name in self._configs:
                config = self._configs[name]
                metadata.update(
                    {
                        "display_name": config.display_name,
                        "warmup_bars": config.warmup_bars,
                        "description": config.description,
                    }
                )
            result[name] = metadata

        return result

    def discover(
        self,
        buildin_path: Path | None = None,
        custom_paths: list[Path] | None = None,
    ) -> dict[str, int]:
        """
        Auto-discover strategies from built-in and custom paths.

        Args:
            buildin_path: Path to built-in strategies (default: auto-detect)
            custom_paths: Paths to custom strategy libraries (default: from system config)

        Returns:
            Dict with counts: {"buildin": X, "custom": Y}

        Example:
            counts = registry.discover()
            print(f"Found {counts['buildin']} built-in, {counts['custom']} custom")
        """
        counts = {"buildin": 0, "custom": 0}

        # Auto-detect built-in path if not provided
        if buildin_path is None:
            # Assume registry.py is in src/qtrader/libraries/
            buildin_path = Path(__file__).parent / "strategies" / "buildin"

        # Discover built-in strategies
        if buildin_path.exists():
            counts["buildin"] = self.discover_from_directory(
                buildin_path,
                source_type="buildin",
                recursive=True,
                name_transform=lambda name: name.lower(),
            )

        # Discover custom strategies using StrategyLoader
        if custom_paths:
            for custom_path in custom_paths:
                try:
                    strategies = self.load_from_directory(custom_path, recursive=True)
                    counts["custom"] += len(strategies)
                except StrategyLoadError as e:
                    # Log error but continue with other paths
                    print(f"Warning: Failed to load strategies from {custom_path}: {e}")

        return counts


def get_strategy_registry() -> StrategyRegistry:
    """
    Get singleton strategy registry instance.

    Returns:
        Global strategy registry

    Usage:
        registry = get_strategy_registry()
        BollingerBreakout = registry.get("bollinger_breakout")
    """
    # For now, create new instance
    # In production, implement proper singleton pattern
    return StrategyRegistry()
