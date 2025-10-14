"""QTrader CLI - Clean implementation."""

import importlib.util
import inspect
import sys
from typing import Any, Dict

import click


def _load_module(path: str):
    """Load Python module from file path."""
    spec = importlib.util.spec_from_file_location("strategy", path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["strategy"] = module
    spec.loader.exec_module(module)
    return module


def _find_strategy_class(module):
    """Find Strategy class (has on_bar method)."""
    classes = [
        (n, o)
        for n, o in inspect.getmembers(module, inspect.isclass)
        if hasattr(o, "on_bar") and o.__module__ == module.__name__
    ]
    if not classes:
        raise ValueError("No Strategy class found")
    if len(classes) > 1:
        raise ValueError(f"Multiple classes found: {[n for n, _ in classes]}")
    return classes[0][1]


def _apply_overrides(config: Dict[str, Any], overrides: tuple) -> Dict[str, Any]:
    """Apply --set overrides to config."""
    import ast

    config = dict(config)
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid: {override}")
        key, val = override.split("=", 1)
        try:
            config[key.strip()] = ast.literal_eval(val.strip())
        except (ValueError, SyntaxError):
            config[key.strip()] = val.strip()
    return config


@click.group()
@click.version_option(version="0.1.0")
def main():
    """QTrader - Quantitative Trading Backtest System"""
    pass


if __name__ == "__main__":
    main()
