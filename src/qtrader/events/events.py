"""
Event system for QTrader - Pydantic models validated against JSON Schema contracts.

Architecture:
    JSON Schema (*.v1.json) ← Source of truth (cross-language contract)
         ↓
    Pydantic Event ← Python implementation with automatic validation
         ↓
    Event Bus

All events inherit from Event base class which validates against JSON Schema.
Control events (barriers, lifecycle) don't require schemas.
"""

import json
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, ClassVar, Optional

import jsonschema
from pydantic import BaseModel, Field, model_validator

# ============================================
# Base Event
# ============================================


class Event(BaseModel):
    """
    Base class for all QTrader events.

    Validates against JSON Schema contracts automatically.
    Each subclass specifies its schema base name via SCHEMA_BASE class variable.
    Schema filename is constructed as: {SCHEMA_BASE}.v{event_version}.json

    Example:
        SCHEMA_BASE = "bar"
        event_version = "1"
        → Loads: bar.v1.json

        When you change event_version to "2":
        → Loads: bar.v2.json
    """

    # Class variable: override in subclasses if schema validation needed
    SCHEMA_BASE: ClassVar[Optional[str]] = None

    # Envelope fields (not validated against domain schema)
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = "base"
    event_version: str = Field(default="1", description="Version of the event schema")
    occurred_at: datetime = Field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    source_service: str = "unknown"

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_vs_schema(self) -> "Event":
        """Validate event against its JSON Schema contract."""
        if self.SCHEMA_BASE is None:
            # Base Event or control events don't need schema validation
            return self

        # Construct schema filename from base + version
        schema_file = f"{self.SCHEMA_BASE}.v{self.event_version}.json"
        schema_path = Path(__file__).parent.parent / "contracts" / "schemas" / schema_file

        # Load schema
        try:
            with open(schema_path) as f:
                schema = json.load(f)
        except FileNotFoundError:
            raise ValueError(f"Schema file not found: {schema_file}. Expected at: {schema_path}")

        # Extract domain data (exclude envelope fields)
        domain_data = self.model_dump(
            exclude={
                "event_id",
                "event_type",
                "event_version",
                "occurred_at",
                "correlation_id",
                "causation_id",
                "source_service",
            }
        )

        # Validate
        try:
            jsonschema.validate(domain_data, schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"{self.__class__.__name__} violates JSON Schema '{schema_file}': {e.message}")

        return self


# ============================================
# Market Data Events
# ============================================


class PriceBarEvent(Event):
    """Price bar event - validates against bar.v{version}.json"""

    SCHEMA_BASE: ClassVar[Optional[str]] = "bar"  # ← Schema filename auto-constructed

    event_type: str = "price_bar"

    # Domain fields (from bar.v1.json schema)
    symbol: str
    asset_class: str
    interval: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted: bool
    cumulative_price_factor: Decimal
    cumulative_volume_factor: Decimal
    price_adjustment_factor: Optional[Decimal] = None
    volume_adjustment_factor: Optional[Decimal] = None
    adjustment_reason: Optional[str] = None
    price_currency: str = "USD"
    price_scale: int = 2
    source: str
    trace_id: Optional[str] = None


class CorporateActionEvent(Event):
    """Corporate action event - validates against corporate_action schema"""

    SCHEMA_BASE: ClassVar[Optional[str]] = "corporate_action"
    event_type: str = "corporate_action"

    # Domain fields (from corporate_action.v1.json schema)
    symbol: str
    action_type: str  # "split" | "dividend" | "merger"
    announcement_date: Optional[datetime] = None
    ex_date: Optional[datetime] = None
    effective_date: datetime
    split_from: Optional[int] = None
    split_to: Optional[int] = None
    split_ratio: Optional[Decimal] = None
    dividend_amount: Optional[Decimal] = None
    dividend_currency: Optional[str] = None
    dividend_type: Optional[str] = None
    price_adjustment_factor: Optional[Decimal] = None
    volume_adjustment_factor: Optional[Decimal] = None


# Barrier events don't need schema validation
class ValuationTriggerEvent(Event):
    """Barrier event - no schema needed"""

    event_type: str = "valuation_trigger"


class RiskEvaluationTriggerEvent(Event):
    """Barrier event - no schema needed"""

    event_type: str = "risk_evaluation_trigger"


class BarCloseEvent(Event):
    """Barrier event - no schema needed"""

    event_type: str = "bar_close"


class BacktestStartedEvent(Event):
    """Lifecycle event - no schema needed"""

    event_type: str = "backtest_started"
    config: dict[str, Any] = Field(default_factory=dict)


class BacktestEndedEvent(Event):
    """Lifecycle event - no schema needed"""

    event_type: str = "backtest_ended"
    success: bool = True
    error_message: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
