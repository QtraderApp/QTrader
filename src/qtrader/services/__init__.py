"""QTrader services package.

This package contains service implementations following the lego architecture
pattern. Each service is independently testable and communicates via Protocol
interfaces using dependency injection.
"""

from qtrader.services.data import DataService, IDataAdapter, IDataService

__all__: list[str] = [
    "DataService",
    "IDataService",
    "IDataAdapter",
]
