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

Design Principles:
- Envelope and payload validated separately
- Schemas cached and pre-compiled for performance
- UTC timezone-aware timestamps (RFC3339 with Z)
- Field types aligned to wire contract (strings for decimals)
- event_type matches schema base name
"""

import json
from datetime import datetime, timezone
from decimal import Decimal
from functools import lru_cache
from importlib import resources
from typing import Any, ClassVar, Optional
from uuid import uuid4

import jsonschema
from jsonschema import Draft202012Validator, FormatChecker
from pydantic import BaseModel, Field, field_serializer, field_validator, model_validator

# ============================================
# Constants
# ============================================

# Reserved envelope field names (excluded from payload validation)
RESERVED_ENVELOPE_KEYS = {
    "event_id",
    "event_type",
    "event_version",
    "occurred_at",
    "correlation_id",
    "causation_id",
    "source_service",
}

# Schema package path (single source of truth for imports)
SCHEMA_PACKAGE = "qtrader.contracts.schemas"

# JavaScript-safe integer limit (2^53 - 1)
JS_SAFE_INTEGER_MAX = 9007199254740991


# ============================================
# Schema Loading & Caching
# ============================================


@lru_cache(maxsize=128)
def load_and_compile_schema(schema_name: str) -> Draft202012Validator:
    """
    Load and compile JSON Schema validator with caching.

    Uses importlib.resources for package-safe loading (works with wheels).

    Args:
        schema_name: Schema filename (e.g., "bar.v1.json")

    Returns:
        Pre-compiled validator with format checker

    Raises:
        FileNotFoundError: If schema file doesn't exist
    """
    try:
        schema_file = resources.files(SCHEMA_PACKAGE).joinpath(schema_name)
        with schema_file.open("r", encoding="utf-8") as f:
            schema = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Schema not found: {schema_name} in package {SCHEMA_PACKAGE}")

    # Pre-compile validator with format checker for uuid, date-time, etc.
    return Draft202012Validator(schema, format_checker=FormatChecker())


@lru_cache(maxsize=8)
def load_envelope_schema() -> Draft202012Validator:
    """Load and compile envelope schema validator."""
    return load_and_compile_schema("envelope.v1.json")


# ============================================
# Base Event Classes
# ============================================


class BaseEvent(BaseModel):
    """
    Base for all events - provides envelope fields only.
    All events (including control/lifecycle) validate envelope.
    """

    # Envelope fields (validated against envelope.v1.json)
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str = "base"
    event_version: int = Field(default=1, description="Schema major version")
    occurred_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="UTC timestamp RFC3339"
    )
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    source_service: str = "unknown"

    model_config = {"frozen": True}

    @field_validator("occurred_at", mode="before")
    @classmethod
    def ensure_utc(cls, v: Any) -> datetime:
        """Ensure timestamp is UTC timezone-aware."""
        if isinstance(v, str):
            # Parse ISO string
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        elif isinstance(v, datetime):
            dt = v
        else:
            raise ValueError(f"Cannot parse datetime from {type(v)}: {v}")

        # Convert to UTC if needed
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)

        return dt

    @field_serializer("occurred_at")
    def _serialize_occurred_at(self, v: datetime) -> str:
        """Serialize datetime to RFC3339 with Z suffix."""
        return v.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @model_validator(mode="after")
    def _validate_envelope(self) -> "BaseEvent":
        """
        Validate envelope fields against envelope.v1.json.

        All events (ValidatedEvent and ControlEvent) inherit this validation.

        Drops None values from optional fields before validation to avoid
        schema validation errors when fields are nullable.
        """
        envelope_validator = load_envelope_schema()

        # Build envelope data, excluding None values for optional fields
        envelope_data = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_version": self.event_version,
            "occurred_at": self.occurred_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_service": self.source_service,
        }

        # Only include optional fields if they have values
        if self.correlation_id is not None:
            envelope_data["correlation_id"] = self.correlation_id
        if self.causation_id is not None:
            envelope_data["causation_id"] = self.causation_id

        try:
            envelope_validator.validate(envelope_data)
        except jsonschema.ValidationError as e:
            raise ValueError(
                f"{self.__class__.__name__} envelope validation failed (envelope.v1.json): {e.message}\n"
                f"Path: {list(e.path)}\n"
                f"Schema path: {list(e.schema_path)}"
            )

        return self


class ValidatedEvent(BaseEvent):
    """
    Base for domain events that require JSON Schema validation.

    Validates payload against domain-specific schema (e.g., bar.v1.json).
    Envelope validation inherited from BaseEvent.
    """

    # Class variable: override in subclasses (must match event_type)
    SCHEMA_BASE: ClassVar[Optional[str]] = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "ValidatedEvent":
        """
        Validate payload fields against {SCHEMA_BASE}.v{event_version}.json.

        Envelope validation already done by BaseEvent._validate_envelope().

        Raises:
            ValueError: If validation fails, with full error context
        """
        # Skip if no schema specified (shouldn't happen for ValidatedEvent)
        if self.SCHEMA_BASE is None:
            raise ValueError(f"{self.__class__.__name__} must specify SCHEMA_BASE")

        # Verify event_type matches SCHEMA_BASE (fixed: removed no-op replace)
        if self.event_type != self.SCHEMA_BASE:
            raise ValueError(
                f"{self.__class__.__name__}: event_type '{self.event_type}' must equal SCHEMA_BASE '{self.SCHEMA_BASE}'"
            )

        # Serialize to dict for validation
        data = self.model_dump()

        # Extract payload (everything except envelope fields)
        payload_data = {k: v for k, v in data.items() if k not in RESERVED_ENVELOPE_KEYS}

        schema_file = f"{self.SCHEMA_BASE}.v{self.event_version}.json"

        try:
            payload_validator = load_and_compile_schema(schema_file)
            payload_validator.validate(payload_data)
        except FileNotFoundError as e:
            raise ValueError(f"{self.__class__.__name__}: Schema not found: {schema_file}") from e
        except jsonschema.ValidationError as e:
            raise ValueError(
                f"{self.__class__.__name__} payload validation failed against {schema_file}: {e.message}\n"
                f"Path: {list(e.path)}\n"
                f"Schema path: {list(e.schema_path)}\n"
                f"Failed value: {e.instance}"
            )

        return self


class ControlEvent(BaseEvent):
    """
    Base for control/lifecycle events that don't require payload validation.

    Only validates envelope, skips payload schema validation.
    Use for: barriers, lifecycle events, coordination signals.
    """

    pass


# ============================================
# Market Data Events
# ============================================


class PriceBarEvent(ValidatedEvent):
    """
    Price bar event - validates against bar.v{version}.json.

    Wire format uses strings for decimals to avoid floating point issues.
    Pydantic auto-converts to Decimal for Python domain use.
    """

    SCHEMA_BASE: ClassVar[Optional[str]] = "bar"
    event_type: str = "bar"  # Must match SCHEMA_BASE

    # Domain fields (wire uses strings, Pydantic converts)
    symbol: str
    asset_class: str = "equity"
    interval: str = "1D"
    timestamp: str  # ISO8601 string on wire (UTC RFC3339)
    timestamp_local: Optional[str] = None  # RFC3339 with offset (e.g., 09:30:00-05:00)
    timezone: Optional[str] = None  # IANA timezone (e.g., America/New_York)
    open: Decimal  # String on wire, Decimal in Python
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    adjusted: bool = False
    cumulative_price_factor: Decimal
    cumulative_volume_factor: Decimal
    price_adjustment_factor: Optional[Decimal] = None
    volume_adjustment_factor: Optional[Decimal] = None
    adjustment_reason: Optional[str] = None
    price_currency: str = "USD"
    price_scale: int = 2
    source: str
    trace_id: Optional[str] = None

    @field_serializer(
        "open",
        "high",
        "low",
        "close",
        "cumulative_price_factor",
        "cumulative_volume_factor",
        "price_adjustment_factor",
        "volume_adjustment_factor",
    )
    def _serialize_decimal(self, v: Optional[Decimal]) -> Optional[str]:
        """Serialize Decimal to string for wire format."""
        return format(v, "f") if v is not None else None

    @field_serializer("volume")
    def _serialize_volume(self, v: Optional[int]) -> Optional[int | str]:
        """
        Serialize volume as int if within JavaScript safe integer range,
        otherwise as string to prevent precision loss in JS consumers.

        JavaScript safe integer: 2^53 - 1 = 9,007,199,254,740,991
        """
        if v is None:
            return None
        if v > JS_SAFE_INTEGER_MAX:
            return str(v)
        return v


class CorporateActionEvent(ValidatedEvent):
    """
    Corporate action event - validates against corporate_action.v{version}.json.
    """

    SCHEMA_BASE: ClassVar[Optional[str]] = "corporate_action"
    event_type: str = "corporate_action"

    # Required domain fields (per schema)
    symbol: str
    asset_class: str = "equity"  # Default to equity
    action_type: str  # "split" | "dividend" | "merger" | etc.
    announcement_date: str  # ISO8601 date (YYYY-MM-DD)
    ex_date: str  # ISO8601 date
    effective_date: str  # ISO8601 date
    source: str  # Data source

    # Optional fields
    record_date: Optional[str] = None
    payment_date: Optional[str] = None
    split_from: Optional[int] = None
    split_to: Optional[int] = None
    split_ratio: Optional[Decimal] = None
    dividend_amount: Optional[Decimal] = None
    dividend_currency: Optional[str] = None
    dividend_type: Optional[str] = None
    price_adjustment_factor: Optional[Decimal] = None
    volume_adjustment_factor: Optional[Decimal] = None
    new_symbol: Optional[str] = None
    source_reference: Optional[str] = None
    notes: Optional[str] = None
    trace_id: Optional[str] = None

    @field_serializer(
        "split_ratio",
        "dividend_amount",
        "price_adjustment_factor",
        "volume_adjustment_factor",
    )
    def _serialize_decimal(self, v: Optional[Decimal]) -> Optional[str]:
        """Serialize Decimal to string for wire format."""
        return format(v, "f") if v is not None else None


# ============================================
# Control Events (No Payload Validation)
# ============================================


class ValuationTriggerEvent(ControlEvent):
    """Barrier event - triggers portfolio valuation."""

    event_type: str = "valuation_trigger"


class RiskEvaluationTriggerEvent(ControlEvent):
    """Barrier event - triggers risk evaluation."""

    event_type: str = "risk_evaluation_trigger"


class BarCloseEvent(ControlEvent):
    """Barrier event - marks end of bar processing."""

    event_type: str = "bar_close"


class BacktestStartedEvent(ControlEvent):
    """Lifecycle event - backtest started."""

    event_type: str = "backtest_started"
    config: dict[str, Any] = Field(default_factory=dict)


class BacktestEndedEvent(ControlEvent):
    """Lifecycle event - backtest ended."""

    event_type: str = "backtest_ended"
    success: bool = True
    error_message: str = ""
    stats: dict[str, Any] = Field(default_factory=dict)
