"""Read-only M5 refund-approval workflow built with LangGraph.

M1 intake scenarios and M2 mock-refund records are separate sources.  The
workflow never joins or changes them.  SQLite is used only for LangGraph
checkpoints so a human approval can resume after a process restart.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol, TypedDict

from shopsense.policy import PolicyUnavailableError
from shopsense.policy.service import PolicyHit, PolicyService, create_m5_policy_service
from shopsense.tools import get_refund_amount, lookup_order, refund_or_replace


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INTAKE_PATH = PROJECT_ROOT / "data" / "shopsense" / "intake" / "records.jsonl"
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "runtime" / "shopsense_m5.sqlite"
REFUND_AUTHORITY_SLUG = "refund-authority"
KNOWN_REFUND_STATUSES = {"requested", "approved", "processed", "rejected"}


class M5WorkflowError(RuntimeError):
    """Raised for invalid M5 inputs or unavailable runtime dependencies."""


@dataclass(frozen=True)
class RefundPolicyDecision:
    """Evidence-backed routing policy obtained from the M4 retrieval service."""

    automated_cap_inr: int
    citation_ids: tuple[str, ...]
    eligible: bool | None
    eligibility_reason: str | None


class RefundPolicyProvider(Protocol):
    """Small boundary that keeps M5 policy routing testable and grounded."""

    def check(self, *, category: str | None) -> RefundPolicyDecision: ...


class M4RefundPolicyProvider:
    """Retrieve the refund cap and category eligibility from the M4 corpus."""

    def __init__(self, service_factory: Callable[[], PolicyService] = create_m5_policy_service) -> None:
        self._service_factory = service_factory
        self._service: PolicyService | None = None

    @property
    def service(self) -> PolicyService:
        if self._service is None:
            self._service = self._service_factory()
        return self._service

    def check(self, *, category: str | None) -> RefundPolicyDecision:
        authority_hits = self.service.search_policy_documents(
            "What is the automated refund approval cap?",
            limit=3,
            policy_slug=REFUND_AUTHORITY_SLUG,
        )
        cap = _automated_refund_cap(authority_hits)
        citations = _citation_ids(authority_hits)

        if not category:
            return RefundPolicyDecision(cap, citations, None, None)

        category_hits = self.service.search_policy_documents(
            f"Is a {category} item eligible for a return or refund?",
            limit=3,
        )
        category_text = "\n".join(hit.text for hit in category_hits)
        category_citations = _citation_ids(category_hits)
        if re.search(r"\bnon[- ]returnable\b|\bnot returnable\b", category_text, re.IGNORECASE):
            return RefundPolicyDecision(
                cap,
                tuple(dict.fromkeys((*citations, *category_citations))),
                False,
                f"The retrieved policy marks {category} as non-returnable.",
            )
        return RefundPolicyDecision(
            cap,
            tuple(dict.fromkeys((*citations, *category_citations))),
            None,
            None,
        )


@dataclass(frozen=True)
class M5CaseResult:
    """Customer-safe result returned by a completed or paused M5 case."""

    case_id: str
    source: str
    workflow_status: str
    route_reasons: tuple[str, ...]
    policy_cap_inr: int | None
    policy_citations: tuple[str, ...]
    action_result: dict[str, Any]
    source_refund_statuses: dict[str, int]
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "source": self.source,
            "workflow_status": self.workflow_status,
            "route_reasons": list(self.route_reasons),
            "policy_cap_inr": self.policy_cap_inr,
            "policy_citations": list(self.policy_citations),
            "action_result": self.action_result,
            "source_refund_statuses": self.source_refund_statuses,
            "message": self.message,
        }


class WorkflowState(TypedDict, total=False):
    """Primitive-only state so durable SQLite checkpoints are safe to restore."""

    case_id: str
    source: str
    record_id: str
    order_ref: str
    m1_record: dict[str, Any]
    triage: dict[str, Any]
    policy_cap_inr: int
    policy_citations: list[str]
    policy_eligible: bool | None
    policy_reason: str
    policy_error: str
    action_result: dict[str, Any]
    source_refund_statuses: dict[str, int]
    validation_errors: list[str]
    route: str
    route_reasons: list[str]
    human_decision: str
    workflow_status: str
    message: str


class M5Workflow:
    """Run and resume read-only M5 cases using a durable LangGraph workflow."""

    def __init__(
        self,
        *,
        policy_provider: RefundPolicyProvider | None = None,
        intake_path: Path = DEFAULT_INTAKE_PATH,
        checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    ) -> None:
        self.policy_provider = policy_provider or M4RefundPolicyProvider()
        self.intake_path = Path(intake_path)
        self.checkpoint_path = Path(checkpoint_path)

    def run_m1_record(self, record_id: str) -> M5CaseResult:
        record = _load_m1_record(self.intake_path, record_id)
        initial_state: WorkflowState = {
            "case_id": f"m1:{record['record_id']}",
            "source": "m1",
            "record_id": record["record_id"],
            "m1_record": record,
        }
        return self._invoke(initial_state)

    def run_m2_order(self, order_ref: str) -> M5CaseResult:
        normalized = order_ref.strip() if isinstance(order_ref, str) else ""
        if not normalized:
            raise M5WorkflowError("M2 order reference is required")
        initial_state: WorkflowState = {
            "case_id": f"m2:{normalized}",
            "source": "m2",
            "order_ref": normalized,
        }
        return self._invoke(initial_state)

    def resume(self, case_id: str, decision: str) -> M5CaseResult:
        normalized_decision = decision.strip().lower() if isinstance(decision, str) else ""
        if normalized_decision not in {"approve", "reject"}:
            raise M5WorkflowError("M5 decision must be approve or reject")
        config = _config(case_id)
        with self._graph_session() as graph:
            snapshot = graph.get_state(config)
            if not snapshot.values:
                raise M5WorkflowError(f"No M5 checkpoint exists for case {case_id!r}")
            if "human_approval" not in snapshot.next:
                raise M5WorkflowError(f"Case {case_id!r} is not awaiting human approval")
            from langgraph.types import Command

            graph.invoke(Command(resume={"decision": normalized_decision}), config)
            return _result_from_snapshot(graph.get_state(config))

    def _invoke(self, state: WorkflowState) -> M5CaseResult:
        config = _config(state["case_id"])
        with self._graph_session() as graph:
            graph.invoke(state, config)
            return _result_from_snapshot(graph.get_state(config))

    @contextmanager
    def _graph_session(self) -> Iterator[Any]:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise M5WorkflowError(
                "M5 requires LangGraph and its SQLite checkpoint package. Install the project dependencies."
            ) from exc

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.checkpoint_path, check_same_thread=False)
        try:
            saver = SqliteSaver(connection)
            saver.setup()
            yield _build_graph(self.policy_provider, saver)
        finally:
            connection.close()


def _build_graph(policy_provider: RefundPolicyProvider, checkpointer: Any) -> Any:
    """Build the LAB B-style graph with its irreversible work after approval."""

    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
    except ImportError as exc:
        raise M5WorkflowError(
            "M5 requires LangGraph and its SQLite checkpoint package. Install the project dependencies."
        ) from exc

    def triage_node(state: WorkflowState) -> dict[str, Any]:
        if state["source"] == "m1":
            ground_truth = state["m1_record"]["ground_truth"]
            return {
                "triage": {
                    "intent": ground_truth["intent"],
                    "sentiment": ground_truth["sentiment"],
                    "requires_human": ground_truth["requires_human"],
                    "claimed_amount_inr": ground_truth["claimed_amount_inr"],
                    "category": ground_truth.get("category"),
                }
            }
        return {"triage": {"intent": "refund_status", "category": None}}

    def policy_check_node(state: WorkflowState) -> dict[str, Any]:
        try:
            category = state["triage"].get("category")
            is_return_or_refund = state["source"] == "m1" and state["triage"].get("intent") in {
                "refund_request",
                "return_request",
            }
            decision = policy_provider.check(
                category=category if is_return_or_refund and isinstance(category, str) else None
            )
        except (M5WorkflowError, PolicyUnavailableError, ValueError) as exc:
            return {"policy_error": str(exc), "policy_citations": []}
        return {
            "policy_cap_inr": decision.automated_cap_inr,
            "policy_citations": list(decision.citation_ids),
            "policy_eligible": decision.eligible,
            "policy_reason": decision.eligibility_reason or "",
        }

    def action_execution_node(state: WorkflowState) -> dict[str, Any]:
        if state["source"] == "m1":
            triage = state["triage"]
            return {
                "action_result": {
                    "mode": "m1_scenario",
                    "intent": triage["intent"],
                    "claimed_amount_inr": triage["claimed_amount_inr"],
                    "message": "M1 scenario values were read as supplied; no source data was changed.",
                },
                "source_refund_statuses": {},
                "validation_errors": [],
            }

        order_ref = state["order_ref"]
        order = lookup_order(order_ref)
        refund_rows = refund_or_replace(order_ref, "refund")
        amount = get_refund_amount(order_ref)
        validation_errors = _validate_m2_refund_data(order, refund_rows, amount)
        summary = refund_rows.get("summary", {}) if isinstance(refund_rows, dict) else {}
        source_statuses = {
            status: int(summary.get(f"{status}_amount_inr", 0))
            for status in sorted(KNOWN_REFUND_STATUSES)
        }
        return {
            "action_result": {
                "mode": "m2_read_only",
                "order": order,
                "refunds": amount.get("refunds", []) if isinstance(amount, dict) else [],
                "approved_amount_inr": amount.get("amount_inr") if isinstance(amount, dict) else None,
                "calculation": amount.get("calculation", "") if isinstance(amount, dict) else "",
                "message": "M2 mock data was read as supplied; no refund status or amount was changed.",
            },
            "source_refund_statuses": source_statuses,
            "validation_errors": validation_errors,
        }

    def route_node(state: WorkflowState) -> dict[str, Any]:
        if state.get("policy_error"):
            return {"route": "failed", "route_reasons": [state["policy_error"]]}
        if state.get("policy_eligible") is False:
            return {"route": "policy_denied", "route_reasons": [state.get("policy_reason", "Policy denied the request")]}
        if state["source"] == "m1":
            triage = state["triage"]
            reasons: list[str] = []
            if triage["requires_human"]:
                reasons.append("M1 requires_human is true")
            if triage["sentiment"] == "angry":
                reasons.append("M1 sentiment is angry")
            claimed_amount = triage["claimed_amount_inr"]
            if isinstance(claimed_amount, int) and claimed_amount > state["policy_cap_inr"]:
                reasons.append(f"M1 claimed amount exceeds the policy cap of ₹{state['policy_cap_inr']}")
            return {"route": "human" if reasons else "auto", "route_reasons": reasons or ["M1 scenario is low risk"]}

        validation_errors = state.get("validation_errors", [])
        if validation_errors:
            return {"route": "human", "route_reasons": validation_errors}
        approved_amount = state["action_result"].get("approved_amount_inr")
        if isinstance(approved_amount, int) and approved_amount > state["policy_cap_inr"]:
            return {
                "route": "human",
                "route_reasons": [f"M2 approved amount exceeds the policy cap of ₹{state['policy_cap_inr']}"],
            }
        return {"route": "auto", "route_reasons": ["M2 approved amount is within the policy cap"]}

    def human_approval_node(state: WorkflowState) -> dict[str, Any]:
        decision = interrupt(
            {
                "case_id": state["case_id"],
                "source": state["source"],
                "route_reasons": state["route_reasons"],
                "policy_cap_inr": state.get("policy_cap_inr"),
                "policy_citations": state.get("policy_citations", []),
                "action_result": state.get("action_result", {}),
                "question": "Approve or reject this ShopSense M5 case?",
            }
        )
        return {"human_decision": str(decision.get("decision", "")).lower()}

    def finalize_node(state: WorkflowState) -> dict[str, Any]:
        route = state.get("route", "failed")
        if route == "failed":
            return {"workflow_status": "failed", "message": state["route_reasons"][0]}
        if route == "policy_denied":
            return {"workflow_status": "policy_denied", "message": state["route_reasons"][0]}
        if route == "human":
            if state.get("human_decision") == "approve":
                return {
                    "workflow_status": "approved_for_payout",
                    "message": "Human approval was recorded. The source refund records remain unchanged.",
                }
            return {
                "workflow_status": "rejected",
                "message": "Human rejection was recorded. The source refund records remain unchanged.",
            }
        return {
            "workflow_status": "auto_resolved",
            "message": "The low-risk workflow completed using read-only source data; no refund record was changed.",
        }

    graph = StateGraph(WorkflowState)
    graph.add_node("triage", triage_node)
    graph.add_node("policy_check", policy_check_node)
    graph.add_node("action_execution", action_execution_node)
    graph.add_node("route", route_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "triage")
    graph.add_edge("triage", "policy_check")
    graph.add_edge("policy_check", "action_execution")
    graph.add_edge("action_execution", "route")
    graph.add_conditional_edges(
        "route",
        lambda state: state["route"],
        {"auto": "finalize", "human": "human_approval", "policy_denied": "finalize", "failed": "finalize"},
    )
    graph.add_edge("human_approval", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def _load_m1_record(path: Path, record_id: str) -> dict[str, Any]:
    if not path.exists():
        raise M5WorkflowError(f"M1 intake file was not found: {path}")
    normalized = record_id.strip() if isinstance(record_id, str) else ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("record_id") == normalized:
            _validate_m1_record(record)
            return record
    raise M5WorkflowError(f"M1 record {normalized!r} was not found")


def _validate_m1_record(record: dict[str, Any]) -> None:
    ground_truth = record.get("ground_truth")
    if not isinstance(ground_truth, dict):
        raise M5WorkflowError("M1 record ground_truth must be an object")
    required_strings = ("record_id", "ticket_id", "raw_text")
    for field in required_strings:
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise M5WorkflowError(f"M1 record field {field!r} must be a non-empty string")
    if not isinstance(ground_truth.get("intent"), str):
        raise M5WorkflowError("M1 ground_truth.intent must be a string")
    if not isinstance(ground_truth.get("sentiment"), str):
        raise M5WorkflowError("M1 ground_truth.sentiment must be a string")
    if not isinstance(ground_truth.get("requires_human"), bool):
        raise M5WorkflowError("M1 ground_truth.requires_human must be a boolean")
    claimed_amount = ground_truth.get("claimed_amount_inr")
    if claimed_amount is not None and (not isinstance(claimed_amount, int) or claimed_amount < 0):
        raise M5WorkflowError("M1 ground_truth.claimed_amount_inr must be a non-negative integer or null")


def _validate_m2_refund_data(
    order: dict[str, Any], refund_rows: dict[str, Any], amount: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if not order.get("found"):
        errors.append("M2 order was not found")
    if not refund_rows.get("found") or not amount.get("found"):
        errors.append("M2 refund records were not found")
        return errors
    for row in refund_rows.get("refunds", []):
        if row.get("status") not in KNOWN_REFUND_STATUSES:
            errors.append(f"Unknown M2 refund status: {row.get('status')!r}")
        if not isinstance(row.get("amount_inr"), int) or row["amount_inr"] <= 0:
            errors.append(f"Invalid M2 refund amount for {row.get('refund_id', 'unknown refund')}")
    return errors


def _automated_refund_cap(hits: list[PolicyHit]) -> int:
    evidence = "\n".join(hit.text for hit in hits)
    matches = re.findall(r"refunds?\s+up\s+to\s+₹\s*([\d,]+)", evidence, re.IGNORECASE)
    if not matches:
        raise M5WorkflowError("Refund-authority policy did not provide an automated approval cap")
    return int(matches[0].replace(",", ""))


def _citation_ids(hits: list[PolicyHit]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(hit.metadata["citation_id"])
            for hit in hits
            if isinstance(hit.metadata.get("citation_id"), str) and hit.metadata["citation_id"]
        )
    )


def _config(case_id: str) -> dict[str, dict[str, str]]:
    if not isinstance(case_id, str) or not case_id.strip():
        raise M5WorkflowError("M5 case ID is required")
    return {"configurable": {"thread_id": case_id.strip()}}


def _result_from_snapshot(snapshot: Any) -> M5CaseResult:
    state: dict[str, Any] = dict(snapshot.values)
    waiting = "human_approval" in snapshot.next
    return M5CaseResult(
        case_id=str(state["case_id"]),
        source=str(state["source"]),
        workflow_status="awaiting_human_approval" if waiting else str(state.get("workflow_status", "failed")),
        route_reasons=tuple(str(reason) for reason in state.get("route_reasons", [])),
        policy_cap_inr=state.get("policy_cap_inr"),
        policy_citations=tuple(str(citation) for citation in state.get("policy_citations", [])),
        action_result=dict(state.get("action_result", {})),
        source_refund_statuses={
            str(key): int(value) for key, value in dict(state.get("source_refund_statuses", {})).items()
        },
        message=(
            "Case is paused for human approval. Resume it with the same case ID and approve or reject."
            if waiting
            else str(state.get("message", "M5 workflow did not produce a final message."))
        ),
    )
