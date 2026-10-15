"""Durable local human-support queue for M7 infrastructure fallbacks."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


QueueStatus = Literal["queued", "claimed", "resolved"]


class QueueStateError(RuntimeError):
    """Raised when an operator attempts an invalid queue lifecycle transition."""


@dataclass(frozen=True)
class QueueItem:
    queue_id: str
    case_id: str
    customer_ref_hash: str
    channel: str
    failed_operation: str
    failure_category: str
    retry_count: int
    circuit_state: str
    status: QueueStatus
    created_at: float
    claimed_at: float | None
    resolved_at: float | None
    operator: str | None
    resolution_note: str | None

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


class HumanQueue:
    """Stores one operational fallback handoff per M6 case ID."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time) -> None:
        self.path = Path(path)
        self.clock = clock

    def enqueue(
        self,
        *,
        case_id: str,
        customer_ref: str,
        channel: str,
        failed_operation: str,
        failure_category: str,
        retry_count: int,
        circuit_state: str,
    ) -> QueueItem:
        now = self.clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_schema(connection)
            existing = connection.execute(
                "SELECT * FROM m7_human_queue WHERE case_id=?", (case_id,)
            ).fetchone()
            if existing is not None:
                connection.commit()
                return self._item(existing)
            cursor = connection.execute(
                "INSERT INTO m7_human_queue (case_id, customer_ref_hash, channel, failed_operation, "
                "failure_category, retry_count, circuit_state, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?)",
                (
                    case_id,
                    self._customer_hash(connection, customer_ref),
                    channel,
                    failed_operation,
                    failure_category,
                    retry_count,
                    circuit_state,
                    now,
                ),
            )
            queue_id = f"QUEUE-{int(cursor.lastrowid):06d}"
            connection.execute("UPDATE m7_human_queue SET queue_id=? WHERE id=?", (queue_id, cursor.lastrowid))
            row = connection.execute("SELECT * FROM m7_human_queue WHERE id=?", (cursor.lastrowid,)).fetchone()
            connection.commit()
            assert row is not None
            return self._item(row)

    def list(self, status: QueueStatus | None = None) -> list[QueueItem]:
        with self._connection() as connection:
            self._ensure_schema(connection)
            if status is None:
                rows = connection.execute("SELECT * FROM m7_human_queue ORDER BY id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM m7_human_queue WHERE status=? ORDER BY id", (status,)
                ).fetchall()
            return [self._item(row) for row in rows]

    def claim(self, queue_id: str, *, operator: str) -> QueueItem:
        return self._transition(queue_id, operator=operator, note=None, target="claimed")

    def resolve(self, queue_id: str, *, operator: str, note: str) -> QueueItem:
        if not note.strip():
            raise QueueStateError("A non-empty resolution note is required.")
        return self._transition(queue_id, operator=operator, note=note.strip(), target="resolved")

    def _transition(self, queue_id: str, *, operator: str, note: str | None, target: QueueStatus) -> QueueItem:
        if not operator.strip():
            raise QueueStateError("Operator name is required.")
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            self._ensure_schema(connection)
            row = connection.execute("SELECT * FROM m7_human_queue WHERE queue_id=?", (queue_id,)).fetchone()
            if row is None:
                connection.commit()
                raise QueueStateError(f"Queue item {queue_id!r} was not found.")
            current = str(row["status"])
            if target == "claimed":
                if current != "queued":
                    connection.commit()
                    raise QueueStateError(f"Only queued items can be claimed; {queue_id!r} is {current}.")
                connection.execute(
                    "UPDATE m7_human_queue SET status='claimed', claimed_at=?, operator=? WHERE queue_id=?",
                    (self.clock(), operator.strip(), queue_id),
                )
            else:
                if current != "claimed":
                    connection.commit()
                    raise QueueStateError(f"Only claimed items can be resolved; {queue_id!r} is {current}.")
                connection.execute(
                    "UPDATE m7_human_queue SET status='resolved', resolved_at=?, operator=?, resolution_note=? "
                    "WHERE queue_id=?",
                    (self.clock(), operator.strip(), note, queue_id),
                )
            updated = connection.execute("SELECT * FROM m7_human_queue WHERE queue_id=?", (queue_id,)).fetchone()
            connection.commit()
            assert updated is not None
            return self._item(updated)

    @staticmethod
    def _item(row: sqlite3.Row) -> QueueItem:
        return QueueItem(
            queue_id=str(row["queue_id"]), case_id=str(row["case_id"]),
            customer_ref_hash=str(row["customer_ref_hash"]), channel=str(row["channel"]),
            failed_operation=str(row["failed_operation"]), failure_category=str(row["failure_category"]),
            retry_count=int(row["retry_count"]), circuit_state=str(row["circuit_state"]),
            status=str(row["status"]), created_at=float(row["created_at"]),
            claimed_at=float(row["claimed_at"]) if row["claimed_at"] is not None else None,
            resolved_at=float(row["resolved_at"]) if row["resolved_at"] is not None else None,
            operator=str(row["operator"]) if row["operator"] is not None else None,
            resolution_note=str(row["resolution_note"]) if row["resolution_note"] is not None else None,
        )

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS m7_human_queue ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, queue_id TEXT UNIQUE, case_id TEXT UNIQUE NOT NULL, "
            "customer_ref_hash TEXT NOT NULL, channel TEXT NOT NULL, failed_operation TEXT NOT NULL, "
            "failure_category TEXT NOT NULL, retry_count INTEGER NOT NULL, circuit_state TEXT NOT NULL, "
            "status TEXT NOT NULL, created_at REAL NOT NULL, claimed_at REAL, resolved_at REAL, "
            "operator TEXT, resolution_note TEXT)"
        )

    @staticmethod
    def _customer_hash(connection: sqlite3.Connection, customer_ref: str) -> str:
        """Use a database-local HMAC key so queue exports cannot enumerate customer IDs."""

        connection.execute(
            "CREATE TABLE IF NOT EXISTS m7_runtime_secrets (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        row = connection.execute(
            "SELECT value FROM m7_runtime_secrets WHERE name='queue_customer_hash'"
        ).fetchone()
        if row is None:
            secret = secrets.token_hex(32)
            connection.execute(
                "INSERT INTO m7_runtime_secrets (name, value) VALUES ('queue_customer_hash', ?)", (secret,)
            )
        else:
            secret = str(row["value"])
        return hmac.new(secret.encode("utf-8"), customer_ref.encode("utf-8"), hashlib.sha256).hexdigest()

    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection
