from __future__ import annotations

import json
from pathlib import Path

import pytest

import shopsense.workflow.m5 as m5
from shopsense.workflow.m5 import M5Workflow, RefundPolicyDecision


pytest.importorskip("langgraph")
pytest.importorskip("langgraph.checkpoint.sqlite")


class FakePolicyProvider:
    """Credential-free M4 policy boundary used by workflow tests."""

    def __init__(self, *, cap: int = 2_000, eligible: bool | None = None) -> None:
        self.cap = cap
        self.eligible = eligible
        self.categories: list[str | None] = []

    def check(self, *, category: str | None) -> RefundPolicyDecision:
        self.categories.append(category)
        return RefundPolicyDecision(
            automated_cap_inr=self.cap,
            citation_ids=("refund-authority:policy:markdown:0",),
            eligible=self.eligible,
            eligibility_reason="The retrieved policy marks this category as non-returnable."
            if self.eligible is False
            else None,
        )


def _write_m1_intake(path: Path, *records: dict[str, object]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def _m1_record(
    record_id: str,
    *,
    intent: str = "refund_request",
    category: str = "Electronics",
    sentiment: str = "calm",
    claimed_amount_inr: int | None = 1_000,
    requires_human: bool = False,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "ticket_id": f"KW-T-{record_id}",
        "raw_text": "Please help with my order.",
        "ground_truth": {
            "intent": intent,
            "category": category,
            "sentiment": sentiment,
            "claimed_amount_inr": claimed_amount_inr,
            "requires_human": requires_human,
        },
    }


def _workflow(tmp_path: Path, provider: FakePolicyProvider) -> M5Workflow:
    intake_path = tmp_path / "records.jsonl"
    return M5Workflow(
        policy_provider=provider,
        intake_path=intake_path,
        checkpoint_path=tmp_path / "m5.sqlite",
    )


def test_m1_low_risk_case_auto_resolves_with_policy_evidence(tmp_path: Path) -> None:
    provider = FakePolicyProvider()
    workflow = _workflow(tmp_path, provider)
    _write_m1_intake(workflow.intake_path, _m1_record("SHOPSENSE-AUTO"))

    result = workflow.run_m1_record("SHOPSENSE-AUTO")

    assert result.workflow_status == "auto_resolved"
    assert result.route_reasons == ("M1 scenario is low risk",)
    assert result.policy_cap_inr == 2_000
    assert result.policy_citations == ("refund-authority:policy:markdown:0",)
    assert provider.categories == ["Electronics"]
    assert result.action_result["mode"] == "m1_scenario"


def test_m1_non_returnable_category_is_policy_denied(tmp_path: Path) -> None:
    provider = FakePolicyProvider(eligible=False)
    workflow = _workflow(tmp_path, provider)
    _write_m1_intake(workflow.intake_path, _m1_record("SHOPSENSE-DENIED", category="Groceries"))

    result = workflow.run_m1_record("SHOPSENSE-DENIED")

    assert result.workflow_status == "policy_denied"
    assert "non-returnable" in result.message
    assert result.policy_citations == ("refund-authority:policy:markdown:0",)


def test_human_approval_checkpoint_survives_a_new_workflow_instance(tmp_path: Path) -> None:
    provider = FakePolicyProvider()
    workflow = _workflow(tmp_path, provider)
    _write_m1_intake(
        workflow.intake_path,
        _m1_record("SHOPSENSE-APPROVE", requires_human=True),
    )

    paused = workflow.run_m1_record("SHOPSENSE-APPROVE")

    assert paused.workflow_status == "awaiting_human_approval"
    assert paused.case_id == "m1:SHOPSENSE-APPROVE"
    assert workflow.checkpoint_path.exists()

    restarted = M5Workflow(
        policy_provider=provider,
        intake_path=workflow.intake_path,
        checkpoint_path=workflow.checkpoint_path,
    )
    completed = restarted.resume(paused.case_id, "approve")

    assert completed.workflow_status == "approved_for_payout"
    assert completed.message == "Human approval was recorded. The source refund records remain unchanged."
    with pytest.raises(m5.M5WorkflowError, match="not awaiting human approval"):
        restarted.resume(paused.case_id, "approve")


def test_human_rejection_is_persisted_without_changing_source_data(tmp_path: Path) -> None:
    provider = FakePolicyProvider()
    workflow = _workflow(tmp_path, provider)
    _write_m1_intake(
        workflow.intake_path,
        _m1_record("SHOPSENSE-REJECT", sentiment="angry"),
    )

    paused = workflow.run_m1_record("SHOPSENSE-REJECT")
    result = workflow.resume(paused.case_id, "reject")

    assert paused.workflow_status == "awaiting_human_approval"
    assert paused.route_reasons == ("M1 sentiment is angry",)
    assert result.workflow_status == "rejected"
    assert "source refund records remain unchanged" in result.message


def test_m2_amount_over_cap_pauses_and_preserves_refund_status_totals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = FakePolicyProvider()
    workflow = _workflow(tmp_path, provider)
    refund = {
        "refund_id": "KW-R-1",
        "amount_inr": 2_001,
        "status": "approved",
    }
    monkeypatch.setattr(m5, "lookup_order", lambda order_ref: {"found": True, "order_ref": order_ref})
    monkeypatch.setattr(
        m5,
        "refund_or_replace",
        lambda order_ref, action: {
            "found": True,
            "refunds": [refund],
            "summary": {
                "requested_amount_inr": 0,
                "approved_amount_inr": 2_001,
                "processed_amount_inr": 0,
                "rejected_amount_inr": 0,
            },
        },
    )
    monkeypatch.setattr(
        m5,
        "get_refund_amount",
        lambda order_ref: {
            "found": True,
            "amount_inr": 2_001,
            "refunds": [refund],
            "calculation": "approved total",
        },
    )

    result = workflow.run_m2_order("KW-O-TEST")

    assert result.workflow_status == "awaiting_human_approval"
    assert result.route_reasons == ("M2 approved amount exceeds the policy cap of ₹2000",)
    assert result.source_refund_statuses == {
        "approved": 2_001,
        "processed": 0,
        "rejected": 0,
        "requested": 0,
    }
