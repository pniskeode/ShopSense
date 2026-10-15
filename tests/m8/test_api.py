from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from shopsense.api import create_app
from shopsense.m8 import M8ChatResult


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, str | None]] = []

    def handle_chat(self, *, customer_ref: str | None, message: str, message_id: str | None = None) -> M8ChatResult:
        self.calls.append({"customer_ref": customer_ref, "message": message, "message_id": message_id})
        return M8ChatResult(
            case_id=f"m6:chat:{message_id or 'GENERATED'}",
            message_id=message_id or "GENERATED",
            workflow_status="auto_resolved",
            message="Safe response.",
            policy_citations=["returns-policy:test:0"],  # type: ignore[arg-type]
            guardrail_outcome="allow",
        )


def test_health_is_available_without_runtime_configuration() -> None:
    client = TestClient(create_app(lambda: FakeRuntime()))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "shopsense-m8"}


def test_chat_returns_only_customer_safe_fields() -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(lambda: runtime))

    response = client.post(
        "/chat",
        json={"customer_ref": "KW-C-TEST", "question": "What is the return window?", "message_id": "M8-API"},
    )

    assert response.status_code == 200
    record = response.json()
    assert record["input"] == {
        "customer_ref": "KW-C-TEST",
        "question": "What is the return window?",
        "message_id": "M8-API",
    }
    output = record["output"]
    assert output["evaluation_run_id"] == "chat-M8-API"
    assert output["case_id"] == "m6:chat:M8-API"
    assert output["expected"] == "Safe response."
    assert output["must_cite"] == ["Kartway Returns and Refunds Policy"]
    assert output["must_not_contain"] == []
    assert output["expected_route"] == "auto_resolved"
    assert "milestones" not in output
    assert "message_id" not in output
    assert not any(key.endswith("_o") for key in output)
    assert not any("match" in key or key in {"review_reasons", "overall_result"} for key in output)
    assert "flagged_for_abuse" not in str(record)
    assert runtime.calls == [{"customer_ref": "KW-C-TEST", "message": "What is the return window?", "message_id": "M8-API"}]


def test_malformed_api_input_is_422() -> None:
    client = TestClient(create_app(lambda: FakeRuntime()))

    response = client.post("/chat", json={"customer_ref": "KW-C-TEST"})

    assert response.status_code == 422


def test_blank_message_is_422_before_runtime_initialisation() -> None:
    client = TestClient(create_app(lambda: (_ for _ in ()).throw(AssertionError("runtime must not initialise"))))

    response = client.post("/chat", json={"customer_ref": "   ", "question": "  "})

    assert response.status_code == 422


def test_customer_ref_is_optional_for_a_general_policy_message() -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(lambda: runtime))

    response = client.post("/chat", json={"question": "What is the return window?", "message_id": "M8-PUBLIC"})

    assert response.status_code == 200
    assert runtime.calls == [{"customer_ref": None, "message": "What is the return window?", "message_id": "M8-PUBLIC"}]


def test_chat_accepts_a_bare_question_and_returns_the_agreed_input_output_record() -> None:
    runtime = FakeRuntime()
    client = TestClient(create_app(runtime_factory=lambda: runtime))

    response = client.post("/chat", json={"question": "What is the return window?"})

    assert response.status_code == 200
    record = response.json()
    assert record["input"] == {"question": "What is the return window?"}
    assert record["output"]["case_id"] == "m6:chat:GENERATED"
    assert record["output"]["expected"] == "Safe response."
    assert not any(key.endswith("_o") for key in record["output"])
    assert "overall_result" not in record["output"]


def test_chat_rejects_golden_expectation_fields() -> None:
    case = {
        "id": "SHOPSENSE-EV-901",
        "question": "What is the return window?",
        "expected": "A grounded answer.",
        "category": "factual",
        "must_cite": [],
        "must_not_contain": [],
        "expected_route": "auto",
    }
    client = TestClient(create_app(lambda: FakeRuntime()))

    response = client.post("/chat", json=case)

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "id"]


def test_separate_evaluation_routes_are_not_exposed() -> None:
    client = TestClient(create_app(lambda: FakeRuntime()))

    assert client.post("/evaluate", json={"question": "What is the return window?"}).status_code == 404
    assert client.post("/evaluate/golden", json={}).status_code == 404
