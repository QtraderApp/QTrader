"""
Event store implementations for QTrader.

Provides append-only persistence layers that can record the full stream of
`BaseEvent` objects emitted by the system. The default in-memory store is
optimized for fast backtests, while the SQLite backend offers durable storage
for audit and replay workflows.
"""

from __future__ import annotations

import json
import sqlite3
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Type

from qtrader.events.events import BaseEvent
from qtrader.system import LoggerFactory

logger = LoggerFactory.get_logger()

EventClass = Type[BaseEvent]


def _collect_event_classes() -> Dict[str, EventClass]:
    """Discover all subclasses of BaseEvent and map them by event_type."""
    registry: Dict[str, EventClass] = {}
    stack: List[Type[BaseEvent]] = list(BaseEvent.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        event_type = getattr(cls, "event_type", None)
        if isinstance(event_type, str):
            registry[event_type] = cls
    return registry


_EVENT_CLASS_REGISTRY = _collect_event_classes()


def register_event_class(event_cls: EventClass) -> None:
    """
    Register additional event classes for deserialization.

    Useful when events are defined dynamically or in optional packages.
    """
    if not issubclass(event_cls, BaseEvent):
        raise TypeError("event_cls must inherit from BaseEvent")
    event_type = getattr(event_cls, "event_type", None)
    if not isinstance(event_type, str):
        raise ValueError(f"{event_cls.__name__} missing event_type string literal")
    _EVENT_CLASS_REGISTRY[event_type] = event_cls
    logger.debug("event_store.event_registered", event_type=event_type, cls=event_cls.__name__)


def resolve_event_class(event_type: str) -> EventClass:
    """Resolve the concrete event class for a stored event_type."""
    try:
        return _EVENT_CLASS_REGISTRY[event_type]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise LookupError(f"Unknown event_type '{event_type}'. Register the class via register_event_class().") from exc


class EventStore(ABC):
    """
    Abstract base class for append-only event storage.

    Implementations must provide durable, ordered persistence of events and
    support querying by id, correlation id, and type.
    """

    @abstractmethod
    def append(self, event: BaseEvent) -> None:
        """Record event (append-only)."""

    @abstractmethod
    def get_by_id(self, event_id: str) -> Optional[BaseEvent]:
        """Retrieve event by unique identifier."""

    @abstractmethod
    def get_by_correlation_id(self, correlation_id: str) -> List[BaseEvent]:
        """Return all events that share a correlation id."""

    @abstractmethod
    def get_by_type(
        self,
        event_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[BaseEvent]:
        """Return events of a specific type filtered by optional time window."""

    @abstractmethod
    def get_all(self, limit: Optional[int] = None) -> List[BaseEvent]:
        """Return all events (or the first N when limit is provided)."""

    @abstractmethod
    def count(self) -> int:
        """Return total number of persisted events."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all events (testing utility)."""

    def close(self) -> None:
        """Optional cleanup hook for stores managing external resources."""
        # Default is a no-op; override when needed.


class InMemoryEventStore(EventStore):
    """Fast append-only store backed by Python data structures."""

    def __init__(self) -> None:
        self._events: List[BaseEvent] = []
        self._by_id: Dict[str, BaseEvent] = {}
        self._by_correlation: Dict[str, List[BaseEvent]] = defaultdict(list)
        self._by_type: Dict[str, List[BaseEvent]] = defaultdict(list)
        logger.debug("event_store.memory_initialized")

    def append(self, event: BaseEvent) -> None:
        if event.event_id in self._by_id:
            raise ValueError(f"Duplicate event_id detected: {event.event_id}")

        self._events.append(event)
        self._by_id[event.event_id] = event
        if event.correlation_id:
            self._by_correlation[event.correlation_id].append(event)
        self._by_type[event.event_type].append(event)
        logger.debug(
            "event_store.memory_append",
            event_type=event.event_type,
            event_id=event.event_id,
            total=len(self._events),
        )

    def get_by_id(self, event_id: str) -> Optional[BaseEvent]:
        return self._by_id.get(event_id)

    def get_by_correlation_id(self, correlation_id: str) -> List[BaseEvent]:
        events = list(self._by_correlation.get(correlation_id, []))
        return sorted(events, key=lambda e: e.occurred_at)

    def get_by_type(
        self,
        event_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[BaseEvent]:
        events = list(self._by_type.get(event_type, []))
        if start_time is not None:
            events = [evt for evt in events if evt.occurred_at >= start_time]
        if end_time is not None:
            events = [evt for evt in events if evt.occurred_at <= end_time]
        return sorted(events, key=lambda e: e.occurred_at)

    def get_all(self, limit: Optional[int] = None) -> List[BaseEvent]:
        if limit is None:
            return list(self._events)
        return list(self._events[: max(limit, 0)])

    def count(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        self._events.clear()
        self._by_id.clear()
        self._by_correlation.clear()
        self._by_type.clear()
        logger.debug("event_store.memory_cleared")


class SQLiteEventStore(EventStore):
    """Durable event store backed by SQLite."""

    def __init__(self, db_path: str | Path) -> None:
        path_str = str(db_path)
        self._disk_path: Optional[Path] = None if path_str == ":memory:" else Path(path_str)
        if self._disk_path is not None:
            self._disk_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(path_str, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        logger.debug("event_store.sqlite_initialized", path=path_str)

    def _create_schema(self) -> None:
        with self._conn:  # auto-commit schema changes
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    event_version INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    correlation_id TEXT,
                    causation_id TEXT,
                    source_service TEXT,
                    payload TEXT NOT NULL
                )
            """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_corr ON events(correlation_id)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_events_occurred_at ON events(occurred_at)")

    @staticmethod
    def _serialize_event(event: BaseEvent) -> str:
        """
        Serialize event to JSON.

        Uses Pydantic's JSON-compatible dump to preserve Decimal precision
        and datetime formatting.
        """
        payload = event.model_dump(mode="json")
        return json.dumps(payload, separators=(",", ":"))

    @staticmethod
    def _deserialize_event(payload: str, event_type: str) -> BaseEvent:
        data = json.loads(payload)
        event_cls = resolve_event_class(event_type)
        return event_cls.model_validate(data)

    def append(self, event: BaseEvent) -> None:
        payload = self._serialize_event(event)
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO events (
                        event_id,
                        event_type,
                        event_version,
                        occurred_at,
                        correlation_id,
                        causation_id,
                        source_service,
                        payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        event.event_id,
                        event.event_type,
                        event.event_version,
                        event.occurred_at.isoformat(),
                        event.correlation_id,
                        event.causation_id,
                        event.source_service,
                        payload,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"Duplicate event_id detected: {event.event_id}") from exc
        logger.debug(
            "event_store.sqlite_append",
            event_type=event.event_type,
            event_id=event.event_id,
        )

    def get_by_id(self, event_id: str) -> Optional[BaseEvent]:
        cursor = self._conn.execute(
            "SELECT payload, event_type FROM events WHERE event_id = ?",
            (event_id,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return self._deserialize_event(row["payload"], row["event_type"])

    def get_by_correlation_id(self, correlation_id: str) -> List[BaseEvent]:
        cursor = self._conn.execute(
            """
            SELECT payload, event_type
            FROM events
            WHERE correlation_id = ?
            ORDER BY occurred_at
        """,
            (correlation_id,),
        )
        return [self._deserialize_event(row["payload"], row["event_type"]) for row in cursor.fetchall()]

    def get_by_type(
        self,
        event_type: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[BaseEvent]:
        query = ["SELECT payload, event_type FROM events WHERE event_type = ?"]
        params: List[object] = [event_type]
        if start_time is not None:
            query.append("AND occurred_at >= ?")
            params.append(start_time.isoformat())
        if end_time is not None:
            query.append("AND occurred_at <= ?")
            params.append(end_time.isoformat())
        query.append("ORDER BY occurred_at")
        cursor = self._conn.execute(" ".join(query), tuple(params))
        return [self._deserialize_event(row["payload"], row["event_type"]) for row in cursor.fetchall()]

    def get_all(self, limit: Optional[int] = None) -> List[BaseEvent]:
        sql = "SELECT payload, event_type FROM events ORDER BY occurred_at"
        if limit is not None:
            sql += " LIMIT ?"
            cursor = self._conn.execute(sql, (max(limit, 0),))
        else:
            cursor = self._conn.execute(sql)
        return [self._deserialize_event(row["payload"], row["event_type"]) for row in cursor.fetchall()]

    def count(self) -> int:
        cursor = self._conn.execute("SELECT COUNT(*) AS total FROM events")
        row = cursor.fetchone()
        return int(row["total"]) if row else 0

    def clear(self) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM events")
        logger.debug("event_store.sqlite_cleared")

    def close(self) -> None:
        self._conn.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "EventStore",
    "InMemoryEventStore",
    "SQLiteEventStore",
    "register_event_class",
    "resolve_event_class",
]
