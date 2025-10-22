"""
System configuration package.

Provides consolidated system-level configuration for all services.
Philosophy: "In real life, the system is one" - one configuration for the entire system.
"""

from qtrader.system.config import SystemConfig, get_system_config, reload_system_config

__all__ = [
    "SystemConfig",
    "get_system_config",
    "reload_system_config",
]
