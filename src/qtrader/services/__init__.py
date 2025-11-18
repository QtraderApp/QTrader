"""QTrader services package.

This package contains service implementations following the lego architecture
pattern. Each service is independently testable and communicates via Protocol
interfaces using dependency injection.
"""

from qtrader.services.data import DataService, IDataService
from qtrader.services.data.update_service import UpdateService

__all__: list[str] = [
    "DataService",
    "IDataService",
    "UpdateService",
]
