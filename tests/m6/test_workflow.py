from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shopsense.policy import PolicyHit
from shopsense.workflow.m6 import IncomingMessage, M6Workflow, M6WorkflowError, ReviewerNote, TriageDecision


pytest.importorskip("langgraph")
pytest.importorskip("langgraph.checkpoint.sqlite")


class FakeModel:
    def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
        if "refund" in message.body.lower():
            return TriageDecision(intent="refund", order_ref="KW-O-TEST", needs_policy=True)
        if "angry" in message.body.lower():
            return TriageDecision(intent="other", sentiment="angry")
        return TriageDecision(intent="order", order_ref="KW-O-TEST")

    def choose_route(self, legal_routes: set[str], state: dict[str, Any]) -> str:
        return sorted(legal_routes)[0]

    def review(self, triage: dict[str, Any], operational: dict[str, Any], policy: dict[str, Any]) -> ReviewerNote:
        return ReviewerNote(note="Reviewed independently.")

    def answer_policy(self, question: str, evidence: str) -> str:
        return "The automated cap is ₹2,000. [SOURCE:refund-authority:test:0]"


class FakeMCP:
    def __init__(self, approved_amount: int = 1_000) -> None:
        self.approved_amount = approved_amount
        self.calls = 0
        self.arguments: list[dict[str, Any]] = []

    def call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.arguments.append(dict(arguments))
        if arguments["order_ref"] == "KW-O-999999":
            return {"found": False, "error": "order_not_found", "message": "Order was not found."}
        if name == "lookup_order":
            return {
                "found": True,
                "order_ref": "KW-O-TEST",
                "product": {
                    "title": "Test item", "category": "Electronics", "sku": "KW-SKU-TEST",
                    "price_inr": 1_999, "warranty_months": 12,
                },
                "quantity": 2,
                "order_value_inr": 3_998,
                "placed_at": "2026-10-01T10:00",
                "status": "shipped",
                "delivery_promise_days": 4,
                "actual_delivery_days": None,
            }
        if name == "track_shipment":
            return {
                "found": True,
                "order_ref": "KW-O-TEST",
                "tracking_id": "KW-TRK-TEST",
                "carrier": "Test Carrier",
                "last_scan": "in_transit",
                "last_scan_at": "2026-10-02T11:00",
                "scan_location": "Test City",
            }
        if name == "refund_or_replace":
            return {
                "found": True,
                "order_ref": "KW-O-TEST",
                "refunds": [{
                    "refund_id": "KW-RF-TEST", "order_ref": "KW-O-TEST", "amount_inr": self.approved_amount,
                    "reason_code": "late_delivery", "status": "approved", "requested_on": "2026-10-01",
                }],
                "summary": {
                    "processed_amount_inr": 0, "approved_amount_inr": self.approved_amount,
                    "requested_amount_inr": 0, "rejected_amount_inr": 0,
                },
            }
        if name == "get_refund_amount":
            return {
                "found": True, "order_ref": "KW-O-TEST", "amount_inr": self.approved_amount,
                "refunds": [],
                "summary": {
                    "processed_amount_inr": 0, "approved_amount_inr": self.approved_amount,
                    "requested_amount_inr": 0, "rejected_amount_inr": 0,
                },
            }
        return {"found": True}


class FakePolicyService:
    def search_policy_documents(self, query: str, **kwargs: Any) -> list[PolicyHit]:
        if kwargs.get("policy_slug") == "refund-authority":
            text = "Refunds up to ₹2,000 may be approved automatically."
        else:
            text = "Electronics returns are eligible according to the applicable policy."
        return [PolicyHit(text=text, score=1.0, metadata={"citation_id": "refund-authority:test:0"})]

    def format_context(self, hits: list[PolicyHit]) -> str:
        return "\n".join(f"[SOURCE:{hit.metadata['citation_id']}] {hit.text}" for hit in hits)


class UnavailablePolicyService(FakePolicyService):
    def search_policy_documents(self, query: str, **kwargs: Any) -> list[PolicyHit]:
        raise RuntimeError("Qdrant is unavailable")


def _workflow(tmp_path: Path, mcp: FakeMCP) -> M6Workflow:
    return M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=mcp,
        model=FakeModel(),
        checkpoint_path=tmp_path / "m6.sqlite",
    )


def test_high_refund_pauses_then_resumes_without_changing_source_data(tmp_path: Path) -> None:
    mcp = FakeMCP(approved_amount=2_001)
    workflow = _workflow(tmp_path, mcp)

    paused = workflow.run_message(IncomingMessage.chat("KW-C-TEST", "Please check my refund for KW-O-TEST", "CHAT-HIGH"))

    assert paused.workflow_status == "awaiting_human_approval"
    assert paused.case_id == "m6:chat:CHAT-HIGH"
    assert "Approved refund amount exceeds the policy cap of ₹2000" in paused.route_reasons
    approved = workflow.resume(paused.case_id, "approve")
    assert approved.workflow_status == "approved_for_payout"
    assert "remain unchanged" in approved.message


def test_duplicate_message_is_idempotent(tmp_path: Path) -> None:
    mcp = FakeMCP()
    workflow = _workflow(tmp_path, mcp)
    message = IncomingMessage.chat("KW-C-TEST", "Where is order KW-O-TEST?", "CHAT-DUPLICATE")

    first = workflow.run_message(message)
    calls_after_first = mcp.calls
    second = workflow.run_message(message)

    assert first.workflow_status == "auto_resolved"
    assert second == first
    assert mcp.calls == calls_after_first


def test_order_only_chat_uses_mcp_without_a_customer_reference(tmp_path: Path) -> None:
    mcp = FakeMCP()
    workflow = _workflow(tmp_path, mcp)

    result = workflow.run_message(IncomingMessage.chat(None, "Where is order KW-O-TEST?", "CHAT-ORDER-ONLY"))

    assert result.workflow_status == "auto_resolved"
    assert mcp.calls == 1
    assert mcp.arguments == [{"order_ref": "KW-O-TEST"}]
    assert result.message == "\n".join([
        "Order details for KW-O-TEST:",
        "- Product: Test item",
        "- Category: Electronics",
        "- SKU: KW-SKU-TEST",
        "- Product price: ₹1,999",
        "- Warranty: 12 months",
        "- Quantity: 2",
        "- Stored order value: ₹3,998",
        "- Placed at: 2026-10-01T10:00",
        "- Status: shipped",
        "- Delivery promise: 4 days",
    ])


def test_order_summary_is_short_only_when_the_customer_explicitly_requests_it(tmp_path: Path) -> None:
    result = _workflow(tmp_path, FakeMCP()).run_message(
        IncomingMessage.chat(None, "Summarize order KW-O-TEST.", "CHAT-ORDER-SUMMARY")
    )

    assert result.workflow_status == "auto_resolved"
    assert result.message == "Order KW-O-TEST for Test item is currently shipped."


def test_shipment_lookup_returns_all_safe_tracking_fields_by_default(tmp_path: Path) -> None:
    class ShipmentModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(
                intent="shipment", request_kind="operational", requires_order_lookup=True, order_ref="KW-O-TEST"
            )

    workflow = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(), model=ShipmentModel(), checkpoint_path=tmp_path / "m6.sqlite",
    )
    result = workflow.run_message(IncomingMessage.chat(None, "Track shipment for KW-O-TEST.", "CHAT-SHIPMENT-FULL"))

    assert result.workflow_status == "auto_resolved"
    assert result.message == "\n".join([
        "Shipment details for order KW-O-TEST:",
        "- Tracking ID: KW-TRK-TEST",
        "- Carrier: Test Carrier",
        "- Latest status: in_transit",
        "- Latest scan location: Test City",
        "- Latest scan time: 2026-10-02T11:00",
    ])


def test_refund_status_returns_all_records_without_policy_or_approval(tmp_path: Path) -> None:
    mcp = FakeMCP(approved_amount=2_001)
    result = _workflow(tmp_path, mcp).run_message(
        IncomingMessage.chat(None, "Refund status of order KW-O-TEST?", "CHAT-REFUND-STATUS")
    )

    assert result.workflow_status == "auto_resolved"
    assert result.policy_citations == ()
    assert result.route_reasons == ()
    assert result.message == "\n".join([
        "Refund details for order KW-O-TEST:",
        "- Refund ID: KW-RF-TEST; Status: approved; Amount: ₹2,001; Reason: late_delivery; Requested on: 2026-10-01",
        "Status totals:",
        "- Processed (already refunded): ₹0",
        "- Approved (awaiting payout): ₹2,001",
        "- Requested: ₹0",
        "- Rejected: ₹0",
    ])
    assert [call["order_ref"] for call in mcp.arguments] == ["KW-O-TEST", "KW-O-TEST", "KW-O-TEST"]


def test_explicit_refund_status_overrides_a_mistaken_policy_model_route(tmp_path: Path) -> None:
    class MistakenPolicyModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="policy", request_kind="policy", needs_policy=True)

    mcp = FakeMCP(approved_amount=2_001)
    workflow = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=mcp, model=MistakenPolicyModel(), checkpoint_path=tmp_path / "m6.sqlite",
    )
    result = workflow.run_message(
        IncomingMessage.chat(None, "Refund status of order KW-O-000001?", "CHAT-REFUND-STATUS-OVERRIDE")
    )

    assert result.workflow_status == "auto_resolved"
    assert result.policy_citations == ()
    assert "Refund details for order KW-O-TEST" in result.message


def test_refund_amount_query_overrides_a_mistaken_policy_model_route(tmp_path: Path) -> None:
    class MistakenPolicyModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="policy", request_kind="policy", needs_policy=True)

    mcp = FakeMCP(approved_amount=2_001)
    workflow = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=mcp, model=MistakenPolicyModel(), checkpoint_path=tmp_path / "m6.sqlite",
    )
    result = workflow.run_message(
        IncomingMessage.chat(None, "What is the approved refund amount for order KW-O-000001?", "CHAT-REFUND-AMOUNT-OVERRIDE")
    )

    assert result.workflow_status == "auto_resolved"
    assert result.policy_citations == ()
    assert "Approved (awaiting payout): ₹2,001" in result.message
    assert [call["order_ref"] for call in mcp.arguments] == ["KW-O-000001", "KW-O-000001", "KW-O-000001"]


def test_refund_status_summary_is_short_only_when_explicitly_requested(tmp_path: Path) -> None:
    result = _workflow(tmp_path, FakeMCP()).run_message(
        IncomingMessage.chat(None, "Briefly summarize refund status of order KW-O-TEST.", "CHAT-REFUND-STATUS-SUMMARY")
    )

    assert result.workflow_status == "auto_resolved"
    assert result.message == "Refund status for order KW-O-TEST: approved amount awaiting payout is ₹1,000."


def test_angry_non_refund_routes_to_human_support_not_refund_approval(tmp_path: Path) -> None:
    result = _workflow(tmp_path, FakeMCP()).run_message(
        IncomingMessage.email({"message_id": "EMAIL-ANGRY", "customer_ref": "KW-C-TEST", "body": "I am angry and this is the worst service."})
    )

    assert result.workflow_status == "human_support_escalation"
    assert result.case_id == "m6:email:EMAIL-ANGRY"


def test_unknown_order_is_a_safe_not_found_response(tmp_path: Path) -> None:
    result = _workflow(tmp_path, FakeMCP()).run_message(
        IncomingMessage.chat("KW-C-TEST", "Where is order KW-O-999999?", "CHAT-MISSING")
    )

    assert result.workflow_status == "not_found"
    assert "couldn’t find or verify" in result.message


def test_explicit_refund_signal_cannot_be_downgraded_by_triage_model(tmp_path: Path) -> None:
    class UnsafeTriageModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="other")

    workflow = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(approved_amount=2_001),
        model=UnsafeTriageModel(),
        checkpoint_path=tmp_path / "m6.sqlite",
    )

    result = workflow.run_message(IncomingMessage.chat("KW-C-TEST", "I need a refund for KW-O-123456", "CHAT-GUARD"))

    assert result.workflow_status == "awaiting_human_approval"


def test_refund_with_unavailable_policy_evidence_never_auto_resolves(tmp_path: Path) -> None:
    workflow = M6Workflow(
        policy_service=UnavailablePolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(),
        model=FakeModel(),
        checkpoint_path=tmp_path / "m6.sqlite",
    )

    result = workflow.run_message(IncomingMessage.chat("KW-C-TEST", "Please check my refund for KW-O-TEST", "CHAT-POLICY-DOWN"))

    assert result.workflow_status == "human_support_escalation"
    assert "can’t confirm the applicable refund policy" in result.message


def test_angry_refund_without_an_order_reference_needs_the_reference_before_review(tmp_path: Path) -> None:
    class NoOrderRefundModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="refund", sentiment="angry", needs_policy=True)

    workflow = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(approved_amount=9_000),
        model=NoOrderRefundModel(),
        checkpoint_path=tmp_path / "m6.sqlite",
    )
    result = workflow.run_message(
        IncomingMessage.chat("KW-C-TEST", "I am angry. I need a refund of ₹9000.", "CHAT-NO-ORDER")
    )

    assert result.workflow_status == "needs_order_reference"
    assert "Please provide the order reference" in result.message


def test_general_policy_question_goes_to_rag_without_an_mcp_lookup(tmp_path: Path) -> None:
    class PolicyModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="policy", request_kind="policy", needs_policy=True)

        def answer_policy(self, question: str, evidence: str) -> str:
            return "Electronics returns are eligible. [SOURCE:refund-authority:test:0]"

    mcp = FakeMCP()
    workflow = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=mcp, model=PolicyModel(), checkpoint_path=tmp_path / "m6.sqlite",
    )

    result = workflow.run_message(IncomingMessage.chat(None, "What is the return window for electronics?", "CHAT-POLICY"))

    assert result.workflow_status == "auto_resolved"
    assert result.policy_citations == ("refund-authority:test:0",)
    assert mcp.calls == 0


def test_policy_worker_repairs_one_uncited_draft_before_returning_it(tmp_path: Path) -> None:
    class RepairModel(FakeModel):
        def __init__(self) -> None:
            self.repairs = 0

        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="policy", request_kind="policy", needs_policy=True)

        def answer_policy(self, question: str, evidence: str) -> str:
            return "Electronics returns are eligible."

        def repair_policy_answer(self, question: str, evidence: str, draft: str) -> str:
            self.repairs += 1
            return "Electronics returns are eligible. [SOURCE:refund-authority:test:0]"

    model, mcp = RepairModel(), FakeMCP()
    workflow = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=mcp, model=model, checkpoint_path=tmp_path / "m6.sqlite",
    )

    result = workflow.run_message(IncomingMessage.chat(None, "What is the return policy?", "CHAT-REPAIR"))

    assert result.workflow_status == "auto_resolved"
    assert model.repairs == 1
    assert mcp.calls == 0


def test_general_policy_comparison_overrides_an_operational_model_route(tmp_path: Path) -> None:
    class MisclassifiedComparisonModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(
                intent="refund", request_kind="operational", requires_order_lookup=True, needs_policy=True,
            )

        def answer_policy(self, question: str, evidence: str) -> str:
            return "The applicable return window is in the policy. [SOURCE:refund-authority:test:0]"

    mcp = FakeMCP()
    result = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=mcp, model=MisclassifiedComparisonModel(), checkpoint_path=tmp_path / "m6.sqlite",
    ).run_message(IncomingMessage.chat(
        None,
        "I bought a blender 40 days ago and want to return it. The site says 30 days but a friend said apparel gets 45, so surely kitchen does too?",
        "CHAT-POLICY-COMPARISON",
    ))

    assert result.workflow_status == "auto_resolved"
    assert mcp.calls == 0


def test_order_reference_clarification_never_exposes_internal_policy_citations(tmp_path: Path) -> None:
    class OperationalRefundModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="refund", request_kind="operational", requires_order_lookup=True, needs_policy=True)

    result = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(), model=OperationalRefundModel(), checkpoint_path=tmp_path / "m6.sqlite",
    ).run_message(IncomingMessage.chat(None, "I need a refund.", "CHAT-ORDER-REFERENCE"))

    assert result.workflow_status == "needs_order_reference"
    assert result.policy_citations == ()


def test_policy_repair_retries_once_after_a_transient_provider_failure(tmp_path: Path) -> None:
    class TransientRepairModel(FakeModel):
        def __init__(self) -> None:
            self.repairs = 0

        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="policy", request_kind="policy", needs_policy=True)

        def answer_policy(self, question: str, evidence: str) -> str:
            return "Electronics returns are eligible."

        def repair_policy_answer(self, question: str, evidence: str, draft: str) -> str:
            self.repairs += 1
            if self.repairs == 1:
                raise RuntimeError("temporary provider failure")
            return "Electronics returns are eligible. [SOURCE:refund-authority:test:0]"

    model = TransientRepairModel()
    result = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(), model=model, checkpoint_path=tmp_path / "m6.sqlite",
    ).run_message(IncomingMessage.chat(None, "What is the return policy?", "CHAT-REPAIR-RETRY"))

    assert result.workflow_status == "auto_resolved"
    assert model.repairs == 2


def test_policy_scope_preserves_broad_discovery_and_adds_model_selected_sources(tmp_path: Path) -> None:
    class ScopedPolicyService(FakePolicyService):
        def __init__(self) -> None:
            self.requested_slugs: list[str | None] = []

        def available_policy_slugs(self) -> tuple[str, ...]:
            return ("category-home-kitchen", "returns-policy", "category-apparel")

        def search_policy_documents(self, query: str, **kwargs: Any) -> list[PolicyHit]:
            slug = kwargs.get("policy_slug")
            self.requested_slugs.append(slug)
            if slug is None:
                return [
                    PolicyHit(
                        text="Home & Kitchen returns have a 45-day window.", score=1.0,
                        metadata={"citation_id": "category-home-kitchen:test:0"},
                    ),
                    PolicyHit(
                        text="The general return window is 30 days unless overridden.", score=0.9,
                        metadata={"citation_id": "returns-policy:test:0"},
                    ),
                ]
            texts = {
                "category-home-kitchen": ("Home & Kitchen returns have a 45-day window.", "category-home-kitchen:test:0"),
                "returns-policy": ("The general return window is 30 days unless overridden.", "returns-policy:test:0"),
                "category-apparel": ("Apparel returns have a 45-day window.", "category-apparel:test:0"),
            }
            text, citation_id = texts[str(slug)]
            return [PolicyHit(text=text, score=1.0, metadata={"citation_id": citation_id})]

    class ScopedModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(
                intent="policy", request_kind="policy", needs_policy=True,
                policy_slugs=["category-home-kitchen", "returns-policy"],
            )

        def answer_policy(self, question: str, evidence: str) -> str:
            return (
                "A blender is covered by the Home & Kitchen 45-day return window, which overrides the general rule. "
                "[SOURCE:category-home-kitchen:test:0] [SOURCE:returns-policy:test:0]"
            )

    service = ScopedPolicyService()
    workflow = M6Workflow(
        policy_service=service,  # type: ignore[arg-type]
        tool_client=FakeMCP(), model=ScopedModel(), checkpoint_path=tmp_path / "m6.sqlite",
    )

    result = workflow.run_message(IncomingMessage.chat(None, "Can I return my blender after 40 days?", "CHAT-SCOPE"))

    assert result.workflow_status == "auto_resolved"
    assert result.policy_citations == ("category-home-kitchen:test:0", "returns-policy:test:0")
    assert service.requested_slugs == [None, "category-home-kitchen", "returns-policy"]


def test_policy_scope_does_not_hide_broadly_discovered_category_when_triage_is_wrong(tmp_path: Path) -> None:
    class DiscoveryPolicyService(FakePolicyService):
        def available_policy_slugs(self) -> tuple[str, ...]:
            return ("category-home-kitchen", "category-apparel")

        def search_policy_documents(self, query: str, **kwargs: Any) -> list[PolicyHit]:
            slug = kwargs.get("policy_slug")
            if slug is None:
                return [
                    PolicyHit(
                        text="Blenders are Home & Kitchen items with a 45-day return window.", score=1.0,
                        metadata={"citation_id": "category-home-kitchen:test:0"},
                    )
                ]
            if slug == "category-apparel":
                return [
                    PolicyHit(
                        text="Apparel returns have a 45-day return window.", score=1.0,
                        metadata={"citation_id": "category-apparel:test:0"},
                    )
                ]
            return []

    class ComparisonModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            # The customer's Apparel comparison leads triage astray.  Broad
            # discovery must still retain the Blender/Home & Kitchen passage.
            return TriageDecision(intent="policy", request_kind="policy", needs_policy=True, policy_slugs=["category-apparel"])

        def answer_policy(self, question: str, evidence: str) -> str:
            assert "Blenders are Home & Kitchen" in evidence
            return "A blender follows the Home & Kitchen 45-day return window. [SOURCE:category-home-kitchen:test:0]"

    result = M6Workflow(
        policy_service=DiscoveryPolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(), model=ComparisonModel(), checkpoint_path=tmp_path / "m6.sqlite",
    ).run_message(IncomingMessage.chat(None, "Can I return a blender after 40 days? Apparel has 45 days.", "CHAT-DISCOVERY"))

    assert result.workflow_status == "auto_resolved"
    assert result.policy_citations == ("category-home-kitchen:test:0",)


def test_abstained_policy_response_never_exposes_retrieved_citations(tmp_path: Path) -> None:
    class AbstainingModel(FakeModel):
        def triage(self, message: IncomingMessage, memory_context: str) -> TriageDecision:
            return TriageDecision(intent="policy", request_kind="policy", needs_policy=True)

        def answer_policy(self, question: str, evidence: str) -> str:
            return "I can’t confirm that policy detail from the available ShopSense policy sources."

    result = M6Workflow(
        policy_service=FakePolicyService(),  # type: ignore[arg-type]
        tool_client=FakeMCP(), model=AbstainingModel(), checkpoint_path=tmp_path / "m6.sqlite",
    ).run_message(IncomingMessage.chat(None, "What is the shipping cost?", "CHAT-ABSTAIN"))

    assert result.workflow_status == "needs_clarification"
    assert result.policy_citations == ()


def test_message_id_cannot_be_reused_for_a_different_customer_message(tmp_path: Path) -> None:
    mcp = FakeMCP()
    workflow = _workflow(tmp_path, mcp)
    workflow.run_message(IncomingMessage.chat("KW-C-TEST", "Where is order KW-O-TEST?", "CHAT-COLLIDE"))
    calls_after_first = mcp.calls

    with pytest.raises(M6WorkflowError, match="already bound"):
        workflow.run_message(IncomingMessage.chat("KW-C-OTHER", "Please refund KW-O-TEST", "CHAT-COLLIDE"))

    assert mcp.calls == calls_after_first
