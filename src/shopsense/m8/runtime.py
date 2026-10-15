"""M8 live-chat boundary around the existing M6/M7 workflow."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ValidationError

from shopsense.llm import LiteLLMClient, configured_model
from shopsense.memory import create_m8_summary_model, create_supermemory_service
from shopsense.policy import PolicyHit, create_m8_policy_service
from shopsense.policy.grounding import is_safe_policy_abstention
from shopsense.reliability import CircuitBreaker, HumanQueue, NoopTracer, ResilientMCPToolClient, TraceRecorder, create_m7_tracer
from shopsense.workflow import IncomingMessage, M6CaseResult, M6Workflow
from shopsense.workflow.m5 import M4RefundPolicyProvider, RefundPolicyProvider
from shopsense.workflow.m6 import JsonM6Model, StdioMCPToolClient

from .guardrails import GuardrailAssessment, M8Guardrails


PROJECT_ROOT = Path(__file__).resolve().parents[3]
M8_RUNTIME_PATH = PROJECT_ROOT / "runtime" / "shopsense_m8.sqlite"
ESCALATION_POLICY_SLUG = "escalation-tone"
PRIVACY_POLICY_SLUG = "fraud-abuse"


class M8RuntimeError(RuntimeError):
    """Raised when M8 cannot safely initialise its live workflow dependencies."""


class M8IdempotencyError(M8RuntimeError):
    """A caller reused a message ID for a different blocked chat request."""


class ScopeDecision(BaseModel):
    """A bounded routing decision made before the M6 workflow begins."""

    scope: Literal["policy", "operational", "customer_memory", "unsupported"]
    reason: str = ""


class ScopeClassifier(Protocol):
    """Classify against declared ShopSense capabilities, without answering."""

    def classify(self, question: str) -> ScopeDecision: ...


class M8ScopeClassifier:
    """Use the M8 model only to select a declared capability scope.

    The source inventory comes from the live policy-service manifest plus the
    existing M2/M3 capability contracts.  A provider or schema failure is
    intentionally non-authoritative: M8 falls back to its existing M6 route.
    """

    def __init__(self, client: LiteLLMClient, policy_service: Any) -> None:
        self.client = client
        self.policy_service = policy_service

    def classify(self, question: str) -> ScopeDecision:
        slugs = self._policy_slugs()
        prompt = (
            "Classify this ShopSense request; do not answer it. Return JSON with exactly `scope` and `reason`. "
            "Allowed scope values are policy, operational, customer_memory, unsupported.\n"
            "Available capabilities:\n"
            f"- Policy: returns, shipping, warranty, and refund-policy documents with these manifest source IDs: {slugs or '(none)'}.\n"
            "- Operational: verified order details, shipment tracking, and existing refund records.\n"
            "- Customer memory: a verified customer's past tickets and explicit preferences.\n"
            "No product catalogue, inventory, personalised recommendation capability, or company-wide corporate-profile "
            "documents are available. A category-specific seller compliance clause is not a company-wide stance.\n"
            "Choose unsupported when none of the available sources can answer the request.\n"
            f"Question: {question}"
        )
        try:
            return ScopeDecision.model_validate_json(self.client.complete(prompt))
        except (RuntimeError, ValidationError, ValueError) as exc:
            raise M8RuntimeError(f"M8 scope classifier returned an invalid decision: {exc}") from exc

    def _policy_slugs(self) -> str:
        available = getattr(self.policy_service, "available_policy_slugs", None)
        if not callable(available):
            return ""
        try:
            values = available()
        except Exception:
            return ""
        return ", ".join(str(value) for value in values if isinstance(value, str) and value.strip())


@dataclass(frozen=True)
class M8ChatResult:
    """Customer-safe FastAPI response; excludes logs, tool values, and internal flags."""

    case_id: str
    message_id: str
    workflow_status: str
    message: str
    policy_citations: tuple[str, ...]
    guardrail_outcome: str
    # Evaluation-only evidence. It is deliberately excluded from ``as_dict`` so
    # the public FastAPI contract remains customer-safe and unchanged.
    audit: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "message_id": self.message_id,
            "workflow_status": self.workflow_status,
            "message": self.message,
            "policy_citations": list(self.policy_citations),
            "guardrail_outcome": self.guardrail_outcome,
        }


class GuardrailCheckpoint:
    """Persist safe short-circuit responses without retaining raw customer text."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def save_or_load(
        self, *, case_id: str, customer_ref: str | None, body: str, channel: str, result: M8ChatResult
    ) -> M8ChatResult:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fingerprint = _fingerprint(customer_ref=customer_ref, body=body, channel=channel)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS m8_guardrail_checkpoints (
                case_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, response_json TEXT NOT NULL
                )"""
            )
            row = connection.execute(
                "SELECT fingerprint, response_json FROM m8_guardrail_checkpoints WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO m8_guardrail_checkpoints(case_id, fingerprint, response_json) VALUES (?, ?, ?)",
                    (case_id, fingerprint, json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True)),
                )
                return result
        saved_fingerprint, saved_response = str(row[0]), str(row[1])
        if saved_fingerprint != fingerprint:
            raise M8IdempotencyError("M8 message ID is already bound to a different customer message. Use a new message ID.")
        payload = json.loads(saved_response)
        return M8ChatResult(
            case_id=str(payload["case_id"]),
            message_id=str(payload["message_id"]),
            workflow_status=str(payload["workflow_status"]),
            message=str(payload["message"]),
            policy_citations=tuple(str(item) for item in payload.get("policy_citations", [])),
            guardrail_outcome=str(payload["guardrail_outcome"]),
        )


class M8Runtime:
    """Run validated chat messages, blocking unsafe paths before M6/MCP can act."""

    def __init__(
        self,
        workflow: M6Workflow,
        guardrails: M8Guardrails,
        *,
        tracer: TraceRecorder | None = None,
        guardrail_checkpoint: GuardrailCheckpoint | None = None,
        scope_classifier: ScopeClassifier | None = None,
    ) -> None:
        self.workflow = workflow
        self.guardrails = guardrails
        self.tracer = tracer or NoopTracer()
        self.guardrail_checkpoint = guardrail_checkpoint
        self.scope_classifier = scope_classifier

    def handle_chat(self, *, customer_ref: str | None, message: str, message_id: str | None = None) -> M8ChatResult:
        """Answer a public policy question anonymously or a verified customer request in M6."""

        body = message.strip() if isinstance(message, str) else ""
        normalized_customer = customer_ref.strip() if isinstance(customer_ref, str) and customer_ref.strip() else None
        normalized_message_id = message_id.strip() if isinstance(message_id, str) and message_id.strip() else f"CHAT-{uuid.uuid4().hex}"
        case_id = f"m6:chat:{normalized_message_id}" if normalized_customer else f"m8:anonymous:{normalized_message_id}"
        assessment = self.guardrails.assess(body)
        assessment = self._enrich_guardrail_evidence(assessment, body)
        if assessment.blocked:
            with self.tracer.run(case_id=case_id, channel="chat", customer_ref=normalized_customer or "anonymous"):
                with self.tracer.span("shopsense.m8.guardrail", {"outcome": assessment.outcome}):
                    pass
            blocked_result = M8ChatResult(
                case_id=case_id,
                message_id=normalized_message_id,
                workflow_status="human_support_escalation",
                message=assessment.message,
                policy_citations=assessment.policy_citations,
                guardrail_outcome=assessment.outcome,
            )
            return (
                self.guardrail_checkpoint.save_or_load(
                    case_id=case_id,
                    customer_ref=normalized_customer,
                    body=body,
                    channel="chat",
                    result=blocked_result,
                )
                if self.guardrail_checkpoint is not None
                else blocked_result
            )

        scope = self._classify_scope(body)
        if scope is not None and scope.scope == "unsupported":
            refused = M8ChatResult(
                case_id=case_id,
                message_id=normalized_message_id,
                workflow_status="needs_clarification",
                message=(
                    "I can help with ShopSense orders, shipments, refunds, customer preferences, and return, "
                    "shipping, or warranty policies. I can’t provide that information or service from the "
                    "available ShopSense sources."
                ),
                policy_citations=(),
                guardrail_outcome="allow",
                audit={"m8_scope": "unsupported", "m6_called": False},
            )
            return (
                self.guardrail_checkpoint.save_or_load(
                    case_id=case_id,
                    customer_ref=normalized_customer,
                    body=body,
                    channel="chat",
                    result=refused,
                )
                if self.guardrail_checkpoint is not None
                else refused
            )

        incoming = IncomingMessage.chat(normalized_customer, body, normalized_message_id)
        result = self.workflow.run_message(incoming)
        post_check = self.guardrails.validate_policy_response(
            question=incoming.body,
            workflow_status=result.workflow_status,
            message=result.message,
            policy_citations=result.policy_citations,
        )
        if post_check.outcome == "policy_abstained":
            return M8ChatResult(
                case_id=result.case_id,
                message_id=incoming.message_id,
                workflow_status="needs_clarification",
                message=post_check.message,
                policy_citations=(),
                guardrail_outcome=post_check.outcome,
                audit=_m6_audit(result),
            )
        if result.workflow_status == "needs_clarification" and is_safe_policy_abstention(result.message):
            return M8ChatResult(
                case_id=result.case_id,
                message_id=incoming.message_id,
                workflow_status="needs_clarification",
                message=result.message,
                policy_citations=(),
                guardrail_outcome="policy_abstained",
                audit=_m6_audit(result),
            )
        return _from_m6(result, incoming.message_id)

    def _classify_scope(self, question: str) -> ScopeDecision | None:
        """Return a valid scope decision, or preserve the established route on failure."""

        if self.scope_classifier is None:
            return None
        try:
            decision = self.scope_classifier.classify(question)
        except Exception:
            return None
        with self.tracer.span("shopsense.m8.scope", {"scope": decision.scope}):
            pass
        return decision

    def _enrich_guardrail_evidence(
        self, assessment: GuardrailAssessment, question: str
    ) -> GuardrailAssessment:
        """Attach read-only policy evidence to an already-blocked safe route.

        Guardrails remain authoritative: retrieval can only explain a human
        handoff. It cannot invoke M6/MCP, permit an action, or alter the route.
        """

        slug = {
            "refund_cap_escalated": ESCALATION_POLICY_SLUG,
            "product_safety_escalated": ESCALATION_POLICY_SLUG,
            "privacy_escalated": PRIVACY_POLICY_SLUG,
        }.get(assessment.outcome)
        if slug is None:
            return assessment

        try:
            hits = self.workflow.policy_service.search_policy_documents(
                question, limit=3, policy_slug=slug
            )
        except Exception:
            return assessment

        citations = tuple(dict.fromkeys((*assessment.policy_citations, *_citations(hits))))
        if citations == assessment.policy_citations:
            return assessment
        return replace(
            assessment,
            message=(
                "This refund request exceeds the automated approval limit, so I’m routing it for human review. "
                "Your reported support and escalation concern will be included in that review."
                if assessment.outcome == "refund_cap_escalated" else assessment.message
            ),
            policy_citations=citations,
        )


def create_m8_runtime() -> M8Runtime:
    """Construct the production M8 boundary using M8-specific model configuration."""

    tracer = create_m7_tracer()
    try:
        policy_service = create_m8_policy_service()
        policy_provider: RefundPolicyProvider = M4RefundPolicyProvider(service_factory=lambda: policy_service)
        m8_client = LiteLLMClient(model=configured_model("SHOPSENSE_M8_MODEL"))
        workflow = M6Workflow(
            policy_service=policy_service,
            memory_service=create_supermemory_service(),
            summary_model=create_m8_summary_model(),
            tool_client=ResilientMCPToolClient(
                StdioMCPToolClient(), CircuitBreaker(M8_RUNTIME_PATH), tracer=tracer
            ),
            model=JsonM6Model(m8_client),
            checkpoint_path=M8_RUNTIME_PATH,
            fallback_queue=HumanQueue(M8_RUNTIME_PATH),
            tracer=tracer,
        )
    except Exception as exc:
        tracer.flush()
        raise M8RuntimeError(f"M8 runtime is unavailable: {exc}") from exc
    return M8Runtime(
        workflow,
        M8Guardrails(policy_provider),
        tracer=tracer,
        guardrail_checkpoint=GuardrailCheckpoint(M8_RUNTIME_PATH),
        scope_classifier=M8ScopeClassifier(m8_client, policy_service),
    )


def create_m8_evaluation_runtime(runtime_path: Path) -> M8Runtime:
    """Construct an isolated M8 runtime for read-only golden evaluation.

    Evaluation intentionally has no Supermemory service, summary model, or live
    M8 checkpoint path. It can read the shared policy index and emit Langfuse
    traces, but cannot write customer memory or alter live M5/M7/M8 state.
    """

    isolated_path = Path(runtime_path)
    tracer = create_m7_tracer()
    try:
        policy_service = create_m8_policy_service()
        policy_provider: RefundPolicyProvider = M4RefundPolicyProvider(service_factory=lambda: policy_service)
        m8_client = LiteLLMClient(model=configured_model("SHOPSENSE_M8_MODEL"))
        workflow = M6Workflow(
            policy_service=policy_service,
            memory_service=None,
            summary_model=None,
            tool_client=ResilientMCPToolClient(
                StdioMCPToolClient(), CircuitBreaker(isolated_path), tracer=tracer
            ),
            model=JsonM6Model(m8_client),
            checkpoint_path=isolated_path,
            fallback_queue=HumanQueue(isolated_path),
            tracer=tracer,
        )
    except Exception as exc:
        tracer.flush()
        raise M8RuntimeError(f"M8 evaluation runtime is unavailable: {exc}") from exc
    return M8Runtime(
        workflow,
        M8Guardrails(policy_provider),
        tracer=tracer,
        guardrail_checkpoint=GuardrailCheckpoint(isolated_path),
        scope_classifier=M8ScopeClassifier(m8_client, policy_service),
    )


def _from_m6(result: M6CaseResult, message_id: str) -> M8ChatResult:
    return M8ChatResult(
        case_id=result.case_id,
        message_id=message_id,
        workflow_status=result.workflow_status,
        message=result.message,
        policy_citations=result.policy_citations,
        guardrail_outcome="allow",
        audit=_m6_audit(result),
    )


def _m6_audit(result: M6CaseResult) -> dict[str, Any]:
    """Return compact execution evidence for the private evaluation surface."""

    return {
        "m6_called": True,
        "m6_log": list(result.log),
        "m7_reliability": dict(result.reliability),
    }


def _fingerprint(*, customer_ref: str | None, body: str, channel: str) -> str:
    """Bind an idempotency key without persisting the customer reference or message body."""

    value = f"{customer_ref or ''}\n{body}\n{channel}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _citations(hits: list[PolicyHit]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(hit.metadata["citation_id"])
            for hit in hits
            if isinstance(hit.metadata.get("citation_id"), str) and hit.metadata["citation_id"]
        )
    )


