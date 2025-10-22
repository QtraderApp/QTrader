"""
ExecutionService Contract - Published Events and Data Models.

This module defines the PUBLIC API of ExecutionService. All data structures
published by ExecutionService are defined here.

CONTRACT: ExecutionService v1.0.0

Published Data Models:
- Order: Order specification
- Fill: Order execution result
- Commission: Commission calculation

Published By: ExecutionService
Consumed By: PortfolioService, Analytics, Reporting

Design Principles:
- Immutability: All models frozen=True
- Validation: Pydantic strict validation
- Decimal precision: Use Decimal for prices and quantities
"""

CONTRACT_VERSION = "1.0.0"

# TODO: Define execution contract models here when refactoring execution service
# For now, execution models remain in their current location
