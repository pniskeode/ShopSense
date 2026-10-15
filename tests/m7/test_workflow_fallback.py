from __future__ import annotations

from typing import Any

from shopsense.reliability import CircuitBreaker, HumanQueue, ResilientMCPToolClient
from shopsense.workflow.m6 import IncomingMessage, M6Workflow, ReviewerNote, TriageDecision


class OrderModel:
    def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
        return TriageDecision(intent="order", order_ref="KW-O-TEST")

    def choose_route(self, legal_routes: set[str], state: dict[str, Any]) -> str:
        return sorted(legal_routes)[0]

    def review(self, triage: dict[str, Any], operational: dict[str, Any], policy: dict[str, Any]) -> ReviewerNote:
        return ReviewerNote()

    def answer_policy(self, question: str, evidence: str) -> str:
        return ""


class DownMCP:
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("order system temporarily unavailable")


class MissingOrderMCP:
    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"found": False, "error": "order_not_found", "message": "Order was not found."}


class UnusedPolicy:
    def search_policy_documents(self, query: str, **kwargs: Any) -> list[object]:
        return []

    def format_context(self, hits: list[object]) -> str:
        return ""


class BrokenQueue:
    def enqueue(self, **kwargs):
        raise OSError("local queue is unavailable")


def test_mcp_outage_creates_one_durable_fallback_not_human_refund_approval(tmp_path) -> None:
    runtime = tmp_path / "m7.sqlite"
    workflow = M6Workflow(
        policy_service=UnusedPolicy(),  # type: ignore[arg-type]
        model=OrderModel(),
        tool_client=ResilientMCPToolClient(
            DownMCP(), CircuitBreaker(runtime), sleep_fn=lambda _: None, random_fn=lambda: 0
        ),
        fallback_queue=HumanQueue(runtime),
        checkpoint_path=tmp_path / "m6.sqlite",
    )
    result = workflow.run_message(IncomingMessage.chat("KW-C-TEST", "Where is KW-O-TEST?", "CHAT-DOWN"))

    assert result.workflow_status == "human_support_escalation"
    assert result.reliability["used_fallback"] is True
    assert result.reliability["retry_count"] == 3
    queue = HumanQueue(runtime)
    assert len(queue.list("queued")) == 1
    repeated = workflow.run_message(IncomingMessage.chat("KW-C-TEST", "Where is KW-O-TEST?", "CHAT-DOWN"))
    assert repeated == result
    assert len(queue.list("queued")) == 1


def test_order_only_mcp_outage_still_creates_a_durable_fallback(tmp_path) -> None:
    runtime = tmp_path / "m7.sqlite"
    workflow = M6Workflow(
        policy_service=UnusedPolicy(),  # type: ignore[arg-type]
        model=OrderModel(),
        tool_client=ResilientMCPToolClient(
            DownMCP(), CircuitBreaker(runtime), sleep_fn=lambda _: None, random_fn=lambda: 0
        ),
        fallback_queue=HumanQueue(runtime),
        checkpoint_path=tmp_path / "m6.sqlite",
    )

    result = workflow.run_message(IncomingMessage.chat(None, "Where is KW-O-TEST?", "CHAT-ORDER-ONLY-DOWN"))

    assert result.workflow_status == "human_support_escalation"
    assert result.reliability["queue_id"].startswith("QUEUE-")
    assert len(HumanQueue(runtime).list("queued")) == 1


def test_queue_storage_failure_still_returns_safe_human_escalation(tmp_path) -> None:
    workflow = M6Workflow(
        policy_service=UnusedPolicy(),  # type: ignore[arg-type]
        model=OrderModel(),
        tool_client=ResilientMCPToolClient(
            DownMCP(), CircuitBreaker(tmp_path / "m7.sqlite"), sleep_fn=lambda _: None, random_fn=lambda: 0
        ),
        fallback_queue=BrokenQueue(),  # type: ignore[arg-type]
        checkpoint_path=tmp_path / "m6.sqlite",
    )
    result = workflow.run_message(IncomingMessage.chat("KW-C-TEST", "Where is KW-O-TEST?", "CHAT-QUEUE-DOWN"))

    assert result.workflow_status == "human_support_escalation"
    assert result.reliability["queue_id"] is None
    assert result.reliability["queue_write_failed"] is True


def test_known_missing_order_never_creates_an_m7_fallback_item(tmp_path) -> None:
    runtime = tmp_path / "m7.sqlite"
    workflow = M6Workflow(
        policy_service=UnusedPolicy(),  # type: ignore[arg-type]
        model=OrderModel(),
        tool_client=ResilientMCPToolClient(MissingOrderMCP(), CircuitBreaker(runtime), sleep_fn=lambda _: None),
        fallback_queue=HumanQueue(runtime),
        checkpoint_path=tmp_path / "m6.sqlite",
    )

    result = workflow.run_message(IncomingMessage.chat("KW-C-TEST", "Where is KW-O-TEST?", "CHAT-MISSING"))

    assert result.workflow_status == "not_found"
    assert result.reliability == {}
    assert HumanQueue(runtime).list() == []
