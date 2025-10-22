"""
Service Contracts - Public APIs for QTrader Services.

This package defines the PUBLIC contracts (APIs) for all QTrader services.
Each service has its own contract module defining the data models and events
it publishes.

Contracts are VERSIONED and follow semantic versioning:
- MAJOR: Breaking changes (remove fields, change types)
- MINOR: Backward-compatible additions (new optional fields, new events)
- PATCH: Bug fixes, documentation

Services must maintain backward compatibility within major version.

Contract Modules:
- data: DataService contract (price bars, corporate actions, quotes, news)
- portfolio: PortfolioService contract (positions, cash, valuations)
- execution: ExecutionService contract (orders, fills, commissions)
- risk: RiskService contract (limits, approvals, rejections)

Design Principles:
- Single Responsibility: Each contract owns ONE service's API
- Immutability: All models frozen (frozen=True)
- Validation: Pydantic models with strict validation
- Documentation: Clear docstrings stating publisher and consumers
"""

from qtrader.contracts.data import CONTRACT_VERSION as DATA_CONTRACT_VERSION
from qtrader.contracts.data import (
    AdjustmentMode,
    Bar,
    CorporateAction,
    CorporateActionType,
    DataSource,
    Instrument,
    InstrumentType,
    MultiBar,
    PriceSeries,
)

__all__ = [
    # Version
    "DATA_CONTRACT_VERSION",
    # Data Contract
    "Bar",
    "PriceSeries",
    "MultiBar",
    "AdjustmentMode",
    "Instrument",
    "InstrumentType",
    "DataSource",
    "CorporateAction",
    "CorporateActionType",
]
