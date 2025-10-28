"""
Strategy Library.

All strategies must inherit from Strategy and implement:
- on_bar(): Process price bars and generate signals
- warmup_bars_required(): Declare warmup needs

All strategy configs must inherit from StrategyConfig:
- Define tunable parameters (periods, thresholds, etc.)
- Separate PROCESS (strategy code) from PARAMETERS (config values)
"""

from qtrader.libraries.strategies.base import Context, Strategy, StrategyConfig

__all__ = [
    "Strategy",
    "StrategyConfig",
    "Context",
]
