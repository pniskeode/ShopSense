from __future__ import annotations

from contextlib import contextmanager

import pytest

from shopsense.reliability import CircuitBreaker, CircuitOpenError, FaultInjectingMCPToolClient, OrderSystemFailure, ResilientMCPToolClient


class FlakyClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def call(self, name, arguments):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RecordingTracer:
    def __init__(self) -> None:
        self.names: list[str] = []

    @contextmanager
    def span(self, name: str, metadata=None):
        self.names.append(name)
        yield


def test_retries_transient_reads_but_not_malformed_response(tmp_path) -> None:
    breaker = CircuitBreaker(tmp_path / "m7.sqlite")
    flaky = FlakyClient([ConnectionError("temporary"), {"found": True}])
    client = ResilientMCPToolClient(flaky, breaker, sleep_fn=lambda _: None, random_fn=lambda: 0)
    assert client.call("lookup_order", {}) == {"found": True}
    assert flaky.calls == 2

    malformed = FlakyClient([{"found": "yes"}, {"found": True}])
    client = ResilientMCPToolClient(malformed, CircuitBreaker(tmp_path / "other.sqlite"), sleep_fn=lambda _: None)
    with pytest.raises(OrderSystemFailure, match="malformed_response"):
        client.call("lookup_order", {})
    assert malformed.calls == 1

    non_json = FlakyClient([RuntimeError("M6 MCP tool returned non-JSON output")])
    breaker = CircuitBreaker(tmp_path / "non-json.sqlite")
    with pytest.raises(OrderSystemFailure, match="malformed_response"):
        ResilientMCPToolClient(non_json, breaker, sleep_fn=lambda _: None).call("lookup_order", {})
    assert breaker.snapshot()["consecutive_failures"] == 1


def test_retry_and_breaker_transitions_have_separate_safe_trace_spans(tmp_path) -> None:
    tracer = RecordingTracer()
    client = ResilientMCPToolClient(
        FlakyClient([ConnectionError("temporary"), {"found": True}]),
        CircuitBreaker(tmp_path / "traced.sqlite"), tracer=tracer, sleep_fn=lambda _: None, random_fn=lambda: 0,
    )

    client.call("lookup_order", {})

    assert "shopsense.m7.mcp" in tracer.names
    assert "shopsense.m7.mcp_retry" in tracer.names
    assert "shopsense.m7.circuit_breaker" in tracer.names


def test_breaker_opens_after_three_logical_failures_then_allows_one_probe(tmp_path) -> None:
    now = [0.0]
    breaker = CircuitBreaker(tmp_path / "m7.sqlite", clock=lambda: now[0], reset_timeout_seconds=30)
    for _ in range(3):
        with pytest.raises(OrderSystemFailure):
            ResilientMCPToolClient(FlakyClient([ConnectionError("down")] * 3), breaker, sleep_fn=lambda _: None).call("lookup_order", {})
    assert breaker.snapshot()["state"] == "open"
    with pytest.raises(CircuitOpenError):
        breaker.acquire()
    now[0] = 31.0
    probe = breaker.acquire()
    assert probe.probe is True
    with pytest.raises(CircuitOpenError):
        breaker.acquire()
    breaker.record_success(probe)
    assert breaker.snapshot()["state"] == "closed"


def test_two_breaker_instances_share_one_half_open_probe(tmp_path) -> None:
    now = [0.0]
    path = tmp_path / "shared.sqlite"
    first = CircuitBreaker(path, clock=lambda: now[0], reset_timeout_seconds=30)
    second = CircuitBreaker(path, clock=lambda: now[0], reset_timeout_seconds=30)
    for _ in range(3):
        with pytest.raises(OrderSystemFailure):
            ResilientMCPToolClient(FlakyClient([ConnectionError("down")] * 3), first, sleep_fn=lambda _: None).call("lookup_order", {})
    now[0] = 31.0
    assert second.acquire().probe is True
    with pytest.raises(CircuitOpenError):
        first.acquire()


def test_seeded_fault_injection_is_explicit_and_never_a_cli_default(tmp_path) -> None:
    source = FlakyClient([{"found": True}])
    with pytest.raises(ValueError, match="test-only"):
        FaultInjectingMCPToolClient(source, mode="outage", seed=7)
    injected = FaultInjectingMCPToolClient(source, mode="malformed", seed=7, enabled_for_tests=True)
    with pytest.raises(OrderSystemFailure, match="malformed_response"):
        ResilientMCPToolClient(injected, CircuitBreaker(tmp_path / "fault.sqlite"), sleep_fn=lambda _: None).call("lookup_order", {})
