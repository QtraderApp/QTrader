"""
PortfolioService Contract - Published Events and Data Models.

This module defines the PUBLIC API of PortfolioService. All data structures
published by PortfolioService are defined here.

CONTRACT: PortfolioService v1.0.0

Published Data Models:
- Position: Current position in an instrument
- PortfolioState: Complete portfolio snapshot
- Transaction: Trade execution record

Published By: PortfolioService
Consumed By: RiskService, Analytics, Reporting

Design Principles:
- Immutability: All models frozen=True
- Validation: Pydantic strict validation
- Decimal precision: Use Decimal for monetary values
"""

CONTRACT_VERSION = "1.0.0"

# TODO: Define portfolio contract models here when implementing portfolio service
# For now, portfolio models remain in their current location
