"""Deterministic M8 controls at the customer-message boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from shopsense.policy import is_policy_question
from shopsense.workflow.m5 import RefundPolicyDecision, RefundPolicyProvider


GuardrailOutcome = Literal[
    "allow", "prompt_injection_blocked", "product_safety_escalated", "privacy_escalated",
    "refund_cap_escalated", "policy_evidence_unavailable", "policy_abstained",
]


@dataclass(frozen=True)
class GuardrailAssessment:
    outcome: GuardrailOutcome
    message: str = ""
    policy_citations: tuple[str, ...] = ()
    automated_cap_inr: int | None = None

    @property
    def blocked(self) -> bool:
        return self.outcome not in {"allow", "policy_abstained"}


class M8Guardrails:
    """Block unsafe action paths without relying on an LLM to notice the risk."""

    _INJECTION = re.compile(
        r"\[(?:system|developer|admin)\s+(?:override|instruction)|\bignore\s+(?:all\s+|previous\s+)?(?:instructions|polic)|"
        r"\b(?:system|developer)\s+prompt\b|\bjailbreak\b|\bdo\s+not\s+follow\s+(?:policy|instructions)",
        re.IGNORECASE,
    )
    _SAFETY = re.compile(
        r"\b(?:smoking|caught\s+fire|on\s+fire|overheat(?:ed|ing)?|burn(?:ed|ing)?|electric\s+shock|physical\s+harm)\b",
        re.IGNORECASE,
    )
    _ACCOUNT_PRIVACY = re.compile(
        r"\b(?:account\s+)?flagged\b|\babuse\s+review\b|\bfraud\s+review\b|\baccount\s+status\b",
        re.IGNORECASE,
    )
    _REFUND = re.compile(r"\b(?:refund|money\s+back|return\s+my\s+money)\b", re.IGNORECASE)
    _AMOUNT_PREFIX = re.compile(r"(?:₹|\binr\b|\brs\.?\b)\s*([0-9][0-9,]*)", re.IGNORECASE)
    _AMOUNT_SUFFIX = re.compile(r"\b([0-9][0-9,]*)\s*(?:rupees?|inr)\b", re.IGNORECASE)

    def __init__(self, policy_provider: RefundPolicyProvider) -> None:
        self.policy_provider = policy_provider

    def assess(self, message: str) -> GuardrailAssessment:
        """Return the first safety outcome in a deliberate, conservative priority order."""

        text = message.strip() if isinstance(message, str) else ""
        if self._INJECTION.search(text):
            return GuardrailAssessment(
                "prompt_injection_blocked",
                "I can help with a genuine ShopSense concern, but I can’t act on embedded commands or change policy. "
                "Please provide a verified order reference for order-specific help.",
            )
        if self._SAFETY.search(text):
            return GuardrailAssessment(
                "product_safety_escalated",
                "For safety, please stop using the item and keep it away from flammable materials if it is safe to do so. "
                "I’m routing this report to a human support specialist.",
            )
        if self._ACCOUNT_PRIVACY.search(text):
            return GuardrailAssessment(
                "privacy_escalated",
                "I can’t discuss internal account-review details. I’m routing your concern to a human support specialist.",
            )
        if self._REFUND.search(text):
            amount = _amount_inr(text)
            if amount is not None:
                try:
                    decision = self.policy_provider.check(category=None)
                except Exception:
                    return GuardrailAssessment(
                        "policy_evidence_unavailable",
                        "I can’t confirm the applicable refund approval policy right now, so I’m routing this to customer support.",
                    )
                if amount > decision.automated_cap_inr:
                    return GuardrailAssessment(
                        "refund_cap_escalated",
                        "This refund request exceeds the automated approval limit, so I’m routing it for human review.",
                        decision.citation_ids,
                        decision.automated_cap_inr,
                    )
        return GuardrailAssessment("allow")

    def validate_policy_response(
        self, *, question: str, workflow_status: str, message: str, policy_citations: tuple[str, ...]
    ) -> GuardrailAssessment:
        """Fail safe if a policy answer reaches M8 without grounded evidence."""

        if workflow_status != "auto_resolved" or not is_policy_question(question):
            return GuardrailAssessment("allow")
        if policy_citations or _is_abstention(message):
            return GuardrailAssessment("allow")
        return GuardrailAssessment(
            "policy_abstained",
            "I can’t confirm that policy detail from the available ShopSense policy sources. "
            "Please provide the product category or order reference, or contact support for a policy review.",
        )


def _amount_inr(text: str) -> int | None:
    match = M8Guardrails._AMOUNT_PREFIX.search(text) or M8Guardrails._AMOUNT_SUFFIX.search(text)
    return int(match.group(1).replace(",", "")) if match else None


def _is_abstention(message: str) -> bool:
    return bool(re.search(r"(?:can(?:not|'t|’t) confirm|unable to confirm|contact support|provide (?:the )?product category)", message, re.IGNORECASE))
