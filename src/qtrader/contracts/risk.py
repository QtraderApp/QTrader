"""
RiskService Contract - Published Events and Data Models.

This module defines the PUBLIC API of RiskService. All data structures
published by RiskService are defined here.

CONTRACT: RiskService v1.0.0

Published Data Models:
- RiskLimits: Risk limit configuration
- OrderApproval: Approved order with sizing
- OrderRejection: Rejected signal with reason

Published By: RiskService
Consumed By: ExecutionService, Analytics, Reporting

Design Principles:
- Immutability: All models frozen=True
- Validation: Pydantic strict validation
- Audit trail: Clear rejection reasons
"""

CONTRACT_VERSION = "1.0.0"

# TODO: Define risk contract models here when refactoring risk service
# For now, risk models remain in their current location
