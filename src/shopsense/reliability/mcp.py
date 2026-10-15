"""M7 retry and circuit-breaker protection around read-only MCP operations."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .circuit import CircuitBreaker, CircuitOpenError
from .tracing import NoopTracer, TraceRecorder


class MCPClient(Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class OrderSystemFailure(RuntimeError):
    """A safe failure description suitable for a fallback queue record."""

    category: str
    failed_operation: str
    retry_count: int
    circuit_state: str

    def __str__(self) -> str:
        return f"Order-system operation {self.failed_operation} failed: {self.category}."


class ResilientMCPToolClient:
    """Retry transient reads, count logical failures, and protect the shared dependency."""

    def __init__(
        self,
        client: MCPClient,
        breaker: CircuitBreaker,
        *,
        tracer: TraceRecorder | None = None,
        max_attempts: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        random_fn: Callable[[], float] = random.random,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self.client = client
        self.breaker = breaker
        self.tracer = tracer or NoopTracer()
        self.max_attempts = max_attempts
        self.sleep_fn = sleep_fn
        self.random_fn = random_fn

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            with self.tracer.span("shopsense.m7.circuit_breaker", {"operation": name, "event": "acquire"}):
                lease = self.breaker.acquire()
        except CircuitOpenError as exc:
            raise OrderSystemFailure("circuit_open", name, 0, exc.state) from exc
        attempts = 1 if lease.probe else self.max_attempts
        for attempt in range(1, attempts + 1):
            try:
                with self.tracer.span("shopsense.m7.mcp", {"operation": name, "attempt": attempt, "probe": lease.probe}):
                    result = self.client.call(name, arguments)
                if not isinstance(result, dict) or not isinstance(result.get("found"), bool):
                    state = self._record_failure(name, lease, attempt, "malformed_response")
                    raise OrderSystemFailure("malformed_response", name, attempt, state)
                with self.tracer.span("shopsense.m7.circuit_breaker", {"operation": name, "event": "success", "probe": lease.probe}):
                    self.breaker.record_success(lease)
                return result
            except OrderSystemFailure:
                raise
            except Exception as exc:
                if _is_malformed(exc):
                    state = self._record_failure(name, lease, attempt, "malformed_response")
                    raise OrderSystemFailure("malformed_response", name, attempt, state) from exc
                if not _is_transient(exc):
                    raise
                if attempt < attempts:
                    with self.tracer.span("shopsense.m7.mcp_retry", {"operation": name, "next_attempt": attempt + 1}):
                        self.sleep_fn(_backoff(attempt, self.random_fn))
                    continue
                state = self._record_failure(name, lease, attempt, "retry_exhausted")
                raise OrderSystemFailure("retry_exhausted", name, attempt, state) from exc
        raise AssertionError("MCP retry loop must return or raise")

    def _record_failure(self, name: str, lease: Any, attempt: int, category: str) -> str:
        with self.tracer.span(
            "shopsense.m7.circuit_breaker",
            {"operation": name, "event": "failure", "attempt": attempt, "category": category, "probe": lease.probe},
        ):
            return self.breaker.record_failure(lease)


def _backoff(attempt: int, random_fn: Callable[[], float]) -> float:
    """Small bounded exponential delay plus jitter; unit tests inject a no-op sleeper."""

    return min(1.0, 0.1 * (2 ** (attempt - 1))) + min(0.05, max(0.0, random_fn()) * 0.05)


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    message = str(exc).lower()
    return any(token in message for token in ("timeout", "timed out", "connection", "transport", "temporarily unavailable", "server failed"))


def _is_malformed(exc: Exception) -> bool:
    """Treat the stdio client's parse/schema failures as one logical outage."""

    message = str(exc).lower()
    return "mcp tool returned non-json" in message or "mcp tool returned an unexpected result" in message
