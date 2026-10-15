from __future__ import annotations

from contextlib import contextmanager

from shopsense.reliability.tracing import LangfuseTracer


class RecordingClient:
    def __init__(self) -> None:
        self.inputs: list[dict] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs):
        self.inputs.append(kwargs["input"])
        yield

    def flush(self) -> None:
        return None


class BrokenClient:
    def start_as_current_observation(self, **kwargs):
        raise RuntimeError("telemetry offline")

    def flush(self) -> None:
        raise RuntimeError("telemetry offline")


def test_trace_metadata_hashes_customer_and_never_contains_message_text() -> None:
    client = RecordingClient()
    tracer = LangfuseTracer(client, pii_salt="trace-salt")
    with tracer.run(case_id="m6:chat:CHAT-safe", channel="chat", customer_ref="KW-C-11111"):
        with tracer.span("shopsense.m7.mcp", {"operation": "lookup_order", "attempt": 1}):
            pass
    rendered = repr(client.inputs)
    assert "KW-C-11111" not in rendered
    assert "raw customer message" not in rendered
    assert "lookup_order" in rendered


def test_trace_failures_are_fail_open() -> None:
    tracer = LangfuseTracer(BrokenClient(), pii_salt="trace-salt")
    with tracer.run(case_id="m6:chat:CHAT-safe", channel="chat", customer_ref="KW-C-11111"):
        with tracer.span("shopsense.m7.mcp"):
            pass
    tracer.flush()
