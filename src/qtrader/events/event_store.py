# """
# Event Store - Event persistence and replay functionality.

# Provides interfaces and implementations for recording all events that flow
# through the system. Critical for:
# - Regulatory compliance (audit trail)
# - Debugging (replay specific scenarios)
# - Testing (record production data, replay in dev)
# - Analytics (query event stream for performance analysis)

# Implementations:
# - InMemoryEventStore: Fast in-memory storage for backtesting
# - SQLiteEventStore: Persistent storage for production audit trail
# - ParquetEventStore: Columnar storage for analytical queries (future)
# """

# import json
# import sqlite3
# from abc import ABC, abstractmethod
# from datetime import datetime
# from pathlib import Path
# from typing import Any, List, Optional

# from qtrader.events.events import Event


# class EventStore(ABC):
#     """
#     Abstract base class for event storage.

#     Event stores are APPEND-ONLY (events are immutable facts).
#     No update or delete operations allowed.
#     """

#     @abstractmethod
#     def append(self, event: Event) -> None:
#         """
#         Record event (append-only).

#         Args:
#             event: Event to record

#         Raises:
#             ValueError: If event_id already exists (duplicate)
#         """
#         pass

#     @abstractmethod
#     def get_by_id(self, event_id: str) -> Optional[Event]:
#         """
#         Retrieve event by unique ID.

#         Args:
#             event_id: Event identifier

#         Returns:
#             Event if found, None otherwise
#         """
#         pass

#     @abstractmethod
#     def get_by_correlation_id(self, correlation_id: str) -> List[Event]:
#         """
#         Get all events in a workflow (same correlation_id).

#         Useful for:
#         - Debugging entire backtest run
#         - Tracing signal → order → fill chains
#         - Replay specific workflows

#         Args:
#             correlation_id: Workflow identifier

#         Returns:
#             List of events (chronologically ordered by created_at)
#         """
#         pass

#     @abstractmethod
#     def get_by_type(
#         self, event_type: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
#     ) -> List[Event]:
#         """
#         Get events by type and optional time range.

#         Args:
#             event_type: Event type (e.g., "signal", "fill")
#             start_time: Start of time range (inclusive)
#             end_time: End of time range (inclusive)

#         Returns:
#             List of events (chronologically ordered)
#         """
#         pass

#     @abstractmethod
#     def get_all(self, limit: Optional[int] = None) -> List[Event]:
#         """
#         Get all events (or first N if limit specified).

#         Args:
#             limit: Maximum number of events to return

#         Returns:
#             List of events (chronologically ordered)
#         """
#         pass

#     @abstractmethod
#     def count(self) -> int:
#         """
#         Get total number of events stored.

#         Returns:
#             Event count
#         """
#         pass

#     @abstractmethod
#     def clear(self) -> None:
#         """
#         Clear all events (for testing only).

#         WARNING: This is a destructive operation.
#         """
#         pass


# class InMemoryEventStore(EventStore):
#     """
#     In-memory event store for backtesting.

#     Fast, no I/O overhead. Events lost when process ends.
#     Ideal for backtests and unit tests.

#     Attributes:
#         _events: List of all events (append-only)
#         _by_id: Index by event_id for fast lookup
#         _by_correlation: Index by correlation_id for workflow queries
#         _by_type: Index by event_type for type queries
#     """

#     def __init__(self):
#         """Initialize empty event store."""
#         self._events: List[Event] = []
#         self._by_id: dict[str, Event] = {}
#         self._by_correlation: dict[str, List[Event]] = {}
#         self._by_type: dict[str, List[Event]] = {}

#     def append(self, event: Event) -> None:
#         """Record event in memory."""
#         if event.event_id in self._by_id:
#             raise ValueError(f"Duplicate event_id: {event.event_id}")

#         # Append to main list
#         self._events.append(event)

#         # Index by event_id
#         self._by_id[event.event_id] = event

#         # Index by correlation_id
#         if event.correlation_id:
#             self._by_correlation.setdefault(event.correlation_id, []).append(event)

#         # Index by event_type
#         if event.event_type:
#             self._by_type.setdefault(event.event_type, []).append(event)

#     def get_by_id(self, event_id: str) -> Optional[Event]:
#         """Retrieve event by ID."""
#         return self._by_id.get(event_id)

#     def get_by_correlation_id(self, correlation_id: str) -> List[Event]:
#         """Get all events in workflow."""
#         events = self._by_correlation.get(correlation_id, [])
#         # Sort by created_at for chronological order
#         return sorted(events, key=lambda e: e.created_at)

#     def get_by_type(
#         self, event_type: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
#     ) -> List[Event]:
#         """Get events by type and time range."""
#         events = self._by_type.get(event_type, [])

#         # Filter by time range
#         if start_time:
#             events = [e for e in events if e.timestamp >= start_time]
#         if end_time:
#             events = [e for e in events if e.timestamp <= end_time]

#         # Sort chronologically
#         return sorted(events, key=lambda e: e.timestamp)

#     def get_all(self, limit: Optional[int] = None) -> List[Event]:
#         """Get all events."""
#         if limit:
#             return self._events[:limit]
#         return self._events.copy()

#     def count(self) -> int:
#         """Get total event count."""
#         return len(self._events)

#     def clear(self) -> None:
#         """Clear all events."""
#         self._events.clear()
#         self._by_id.clear()
#         self._by_correlation.clear()
#         self._by_type.clear()


# class SQLiteEventStore(EventStore):
#     """
#     SQLite event store for persistent audit trail.

#     Events stored in SQLite database for:
#     - Production audit trail (regulatory compliance)
#     - Long-term event history
#     - Cross-session replay

#     Schema:
#         CREATE TABLE events (
#             event_id TEXT PRIMARY KEY,
#             event_type TEXT NOT NULL,
#             timestamp TEXT NOT NULL,
#             created_at TEXT NOT NULL,
#             correlation_id TEXT,
#             causation_id TEXT,
#             source_service TEXT,
#             payload TEXT NOT NULL
#         );

#     Notes:
#         - Events serialized as JSON in payload column
#         - Indexes on correlation_id, event_type, timestamp for fast queries
#         - Append-only (no updates or deletes)
#     """

#     def __init__(self, db_path: str | Path):
#         """
#         Initialize SQLite event store.

#         Args:
#             db_path: Path to SQLite database file
#         """
#         self.db_path = Path(db_path)
#         self.db_path.parent.mkdir(parents=True, exist_ok=True)
#         self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
#         self._create_schema()

#     def _create_schema(self) -> None:
#         """Create events table and indexes."""
#         cursor = self._conn.cursor()

#         # Create events table
#         cursor.execute(
#             """
#             CREATE TABLE IF NOT EXISTS events (
#                 event_id TEXT PRIMARY KEY,
#                 event_type TEXT NOT NULL,
#                 timestamp TEXT NOT NULL,
#                 created_at TEXT NOT NULL,
#                 correlation_id TEXT,
#                 causation_id TEXT,
#                 source_service TEXT,
#                 payload TEXT NOT NULL
#             )
#         """
#         )

#         # Create indexes for fast queries
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlation_id ON events(correlation_id)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_event_type ON events(event_type)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)")
#         cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON events(created_at)")

#         self._conn.commit()

#     def _serialize_event(self, event: Event) -> str:
#         """
#         Serialize event to JSON.

#         Args:
#             event: Event to serialize

#         Returns:
#             JSON string
#         """
#         # Convert event to dict (dataclass)
#         event_dict = event.__dict__.copy()

#         # Convert datetime objects to ISO strings
#         for key, value in event_dict.items():
#             if isinstance(value, datetime):
#                 event_dict[key] = value.isoformat()

#         return json.dumps(event_dict)

#     def _deserialize_event(self, payload: str, event_type: str) -> Event:
#         """
#         Deserialize event from JSON.

#         Args:
#             payload: JSON string
#             event_type: Event type for reconstruction

#         Returns:
#             Event instance

#         Note:
#             This is a simplified deserializer. Full implementation would
#             need to reconstruct specific event subclasses based on event_type.
#         """
#         event_dict = json.loads(payload)

#         # Convert ISO strings back to datetime
#         for key in ["timestamp", "created_at"]:
#             if key in event_dict and isinstance(event_dict[key], str):
#                 event_dict[key] = datetime.fromisoformat(event_dict[key])

#         # Reconstruct Event (base class only for now)
#         # TODO: Reconstruct specific subclasses based on event_type
#         return Event(**event_dict)

#     def append(self, event: Event) -> None:
#         """Record event in SQLite database."""
#         cursor = self._conn.cursor()

#         payload = self._serialize_event(event)

#         try:
#             cursor.execute(
#                 """
#                 INSERT INTO events (
#                     event_id, event_type, timestamp, created_at,
#                     correlation_id, causation_id, source_service, payload
#                 ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
#             """,
#                 (
#                     event.event_id,
#                     event.event_type,
#                     event.timestamp.isoformat(),
#                     event.created_at.isoformat(),
#                     event.correlation_id,
#                     event.causation_id,
#                     event.source_service,
#                     payload,
#                 ),
#             )
#             self._conn.commit()
#         except sqlite3.IntegrityError as e:
#             raise ValueError(f"Duplicate event_id: {event.event_id}") from e

#     def get_by_id(self, event_id: str) -> Optional[Event]:
#         """Retrieve event by ID from database."""
#         cursor = self._conn.cursor()
#         cursor.execute("SELECT payload, event_type FROM events WHERE event_id = ?", (event_id,))
#         row = cursor.fetchone()

#         if row:
#             payload, event_type = row
#             return self._deserialize_event(payload, event_type)
#         return None

#     def get_by_correlation_id(self, correlation_id: str) -> List[Event]:
#         """Get all events in workflow from database."""
#         cursor = self._conn.cursor()
#         cursor.execute(
#             "SELECT payload, event_type FROM events WHERE correlation_id = ? ORDER BY created_at",
#             (correlation_id,),
#         )

#         events = []
#         for payload, event_type in cursor.fetchall():
#             events.append(self._deserialize_event(payload, event_type))
#         return events

#     def get_by_type(
#         self, event_type: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
#     ) -> List[Event]:
#         """Get events by type and time range from database."""
#         cursor = self._conn.cursor()

#         query = "SELECT payload, event_type FROM events WHERE event_type = ?"
#         params: List[Any] = [event_type]

#         if start_time:
#             query += " AND timestamp >= ?"
#             params.append(start_time.isoformat())

#         if end_time:
#             query += " AND timestamp <= ?"
#             params.append(end_time.isoformat())

#         query += " ORDER BY timestamp"

#         cursor.execute(query, params)

#         events = []
#         for payload, evt_type in cursor.fetchall():
#             events.append(self._deserialize_event(payload, evt_type))
#         return events

#     def get_all(self, limit: Optional[int] = None) -> List[Event]:
#         """Get all events from database."""
#         cursor = self._conn.cursor()

#         query = "SELECT payload, event_type FROM events ORDER BY created_at"
#         if limit:
#             query += f" LIMIT {limit}"

#         cursor.execute(query)

#         events = []
#         for payload, event_type in cursor.fetchall():
#             events.append(self._deserialize_event(payload, event_type))
#         return events

#     def count(self) -> int:
#         """Get total event count from database."""
#         cursor = self._conn.cursor()
#         cursor.execute("SELECT COUNT(*) FROM events")
#         result = cursor.fetchone()
#         return int(result[0]) if result else 0

#     def clear(self) -> None:
#         """Clear all events from database."""
#         cursor = self._conn.cursor()
#         cursor.execute("DELETE FROM events")
#         self._conn.commit()

#     def close(self) -> None:
#         """Close database connection."""
#         self._conn.close()

#     def __del__(self):
#         """Cleanup: close database on garbage collection."""
#         if hasattr(self, "_conn"):
#             self._conn.close()


# # ============================================
# # Public API
# # ============================================

# __all__ = [
#     "EventStore",
#     "InMemoryEventStore",
#     "SQLiteEventStore",
# ]
