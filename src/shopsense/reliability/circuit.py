"""A SQLite-backed, cross-process circuit breaker for the order-system MCP boundary."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


CircuitState = Literal["closed", "open", "half_open"]


class CircuitOpenError(RuntimeError):
    """Raised when a logical order-system call is intentionally short-circuited."""

    def __init__(self, state: CircuitState) -> None:
        super().__init__(f"Order-system circuit breaker is {state}.")
        self.state = state


@dataclass(frozen=True)
class CircuitLease:
    probe: bool


class CircuitBreaker:
    """Persist one dependency's health state and reserve a single half-open probe."""

    def __init__(
        self,
        path: Path,
        *,
        dependency: str = "shopsense-order-mcp",
        failure_threshold: int = 3,
        reset_timeout_seconds: float = 30.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least one")
        if reset_timeout_seconds <= 0:
            raise ValueError("reset_timeout_seconds must be positive")
        self.path = Path(path)
        self.dependency = dependency
        self.failure_threshold = failure_threshold
        self.reset_timeout_seconds = reset_timeout_seconds
        self.clock = clock

    def acquire(self) -> CircuitLease:
        """Permit one logical call, or short-circuit it when the dependency is unhealthy."""

        now = self.clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection)
            state = str(row["state"])
            if state == "closed":
                connection.commit()
                return CircuitLease(probe=False)
            if state == "open":
                opened_at = float(row["opened_at"]) if row["opened_at"] is not None else now
                if now - opened_at < self.reset_timeout_seconds:
                    connection.commit()
                    raise CircuitOpenError("open")
                connection.execute(
                    "UPDATE m7_circuit_breakers SET state='half_open', probe_started_at=?, updated_at=? "
                    "WHERE dependency=?",
                    (now, now, self.dependency),
                )
                connection.commit()
                return CircuitLease(probe=True)
            probe_started = float(row["probe_started_at"]) if row["probe_started_at"] is not None else now
            if now - probe_started < self.reset_timeout_seconds:
                connection.commit()
                raise CircuitOpenError("half_open")
            connection.execute(
                "UPDATE m7_circuit_breakers SET probe_started_at=?, updated_at=? WHERE dependency=?",
                (now, now, self.dependency),
            )
            connection.commit()
            return CircuitLease(probe=True)

    def record_success(self, lease: CircuitLease) -> None:
        now = self.clock()
        with self._connection() as connection:
            connection.execute(
                "UPDATE m7_circuit_breakers SET state='closed', consecutive_failures=0, opened_at=NULL, "
                "probe_started_at=NULL, updated_at=? WHERE dependency=?",
                (now, self.dependency),
            )

    def record_failure(self, lease: CircuitLease) -> CircuitState:
        now = self.clock()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = self._row(connection)
            failures = int(row["consecutive_failures"]) + 1
            if lease.probe or failures >= self.failure_threshold:
                state: CircuitState = "open"
                connection.execute(
                    "UPDATE m7_circuit_breakers SET state='open', consecutive_failures=?, opened_at=?, "
                    "probe_started_at=NULL, updated_at=? WHERE dependency=?",
                    (failures, now, now, self.dependency),
                )
            else:
                state = "closed"
                connection.execute(
                    "UPDATE m7_circuit_breakers SET consecutive_failures=?, updated_at=? WHERE dependency=?",
                    (failures, now, self.dependency),
                )
            connection.commit()
            return state

    def snapshot(self) -> dict[str, object]:
        with self._connection() as connection:
            row = self._row(connection)
            return {
                "dependency": self.dependency,
                "state": row["state"],
                "consecutive_failures": int(row["consecutive_failures"]),
                "opened_at": row["opened_at"],
                "probe_started_at": row["probe_started_at"],
            }

    def _row(self, connection: sqlite3.Connection) -> sqlite3.Row:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS m7_circuit_breakers ("
            "dependency TEXT PRIMARY KEY, state TEXT NOT NULL, consecutive_failures INTEGER NOT NULL, "
            "opened_at REAL, probe_started_at REAL, updated_at REAL NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO m7_circuit_breakers "
            "(dependency, state, consecutive_failures, opened_at, probe_started_at, updated_at) "
            "VALUES (?, 'closed', 0, NULL, NULL, ?)",
            (self.dependency, self.clock()),
        )
        row = connection.execute(
            "SELECT * FROM m7_circuit_breakers WHERE dependency=?", (self.dependency,)
        ).fetchone()
        assert row is not None
        return row

    def _connection(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection
