from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from shopsense.llm import LiteLLMClient
from shopsense.workflow.m6 import JsonM6Model, _fallback_triage


def test_policy_worker_requests_plain_text_not_json_mode() -> None:
    received: dict[str, Any] = {}

    def completion(**kwargs: Any) -> Any:
        received.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content="Returns are eligible. [SOURCE:returns:test:0]"
        ))])

    model = JsonM6Model(LiteLLMClient(model="test-model", completion_fn=completion))
    answer = model.answer_policy("Can I return this?", "[SOURCE:returns:test:0] Returns are eligible.")

    assert answer == "Returns are eligible. [SOURCE:returns:test:0]"
    assert "response_format" not in received


def test_fallback_triage_routes_general_delivery_timing_to_policy_rag() -> None:
    decision = _fallback_triage("When will standard delivery arrive?")

    assert decision.intent == "policy"
    assert decision.request_kind == "policy"
    assert decision.needs_policy is True
