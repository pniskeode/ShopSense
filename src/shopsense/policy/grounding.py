"""M4 response guard: policy claims must be traceable to evidence or tool data."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .service import PolicyHit


SOURCE_PATTERN = re.compile(r"\[SOURCE:([a-z0-9:-]+)\]", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?:₹|\$)?\d+(?:[,.]\d+)?(?:\s*(?:%|business\s+days?|days?|hours?|years?|months?))?", re.IGNORECASE)


@dataclass(frozen=True)
class GroundingReport:
    """Outcome of the deterministic runtime/evaluation grounding checks."""

    grounded: bool
    errors: tuple[str, ...]
    citation_ids: tuple[str, ...]


def validate_answer_grounding(
    answer: str,
    *,
    policy_hits: Iterable[PolicyHit],
    tool_results: Iterable[dict[str, Any]],
    policy_question: bool,
    customer_message: str = "",
) -> GroundingReport:
    """Reject unsupported citations and numbers before a customer sees them.

    The guard deliberately checks only objective, high-risk claims. It does not
    pretend to fully understand natural language; a failed or absent citation
    fails safe rather than allowing a return window or monetary amount through.
    """

    answer = answer.strip()
    hits = list(policy_hits)
    abstained = is_safe_policy_abstention(answer)
    valid_citations = {str(hit.metadata.get("citation_id")) for hit in hits if hit.metadata.get("citation_id")}
    cited = tuple(match.group(1) for match in SOURCE_PATTERN.finditer(answer))
    errors: list[str] = []
    if policy_question:
        if not hits and not abstained:
            errors.append("No policy evidence was retrieved for this policy question")
        if hits and not cited and not abstained:
            errors.append("A policy answer must cite at least one retrieved source")
    unknown = sorted({citation for citation in cited if citation not in valid_citations})
    if unknown:
        errors.append(f"Answer cited evidence that was not retrieved: {', '.join(unknown)}")

    evidence = "\n".join(hit.text for hit in hits)
    tool_text = "\n".join(json.dumps(result, ensure_ascii=False, sort_keys=True) for result in tool_results)
    answer_without_citations = SOURCE_PATTERN.sub("", answer)
    for numeric_claim in NUMBER_PATTERN.findall(answer_without_citations):
        normalized = _normalise_number(numeric_claim)
        if not normalized or normalized in {"1", "2", "3", "4", "5", "6"}:
            # Enumeration/bullet digits do not make a policy claim.
            continue
        if (
            normalized not in _normalise_number(evidence)
            and normalized not in _normalise_number(tool_text)
            and not _is_explicitly_attributed_customer_number(answer_without_citations, numeric_claim, customer_message)
        ):
            errors.append(f"Unsupported numeric claim: {numeric_claim.strip()}")

    return GroundingReport(not errors, tuple(errors), cited)


def safe_policy_abstention() -> str:
    return (
        "I can’t confirm that policy detail from the available ShopSense policy sources. "
        "Please provide the product category or order reference, or contact support for a policy review."
    )


def _normalise_number(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    return (
        normalized.replace("businessdays", "businessday")
        .replace("days", "day")
        .replace("hours", "hour")
        .replace("years", "year")
        .replace("months", "month")
    )


def _is_explicitly_attributed_customer_number(answer: str, numeric_claim: str, customer_message: str) -> bool:
    """Permit a customer's stated scenario fact without treating it as policy.

    A question may say that a purchase was made ``40 days`` ago.  That number
    is not policy evidence, but a helpful answer can repeat it as long as it is
    clearly attributed (for example, ``You mentioned 40 days``).  The policy
    rule itself must still be separately supported by retrieved evidence.
    """

    if not customer_message or _normalise_number(numeric_claim) not in _normalise_number(customer_message):
        return False
    escaped = re.escape(numeric_claim.strip())
    attributed_patterns = (
        rf"\byou\s+(?:said|mentioned|reported|stated)\b[^.!?]{{0,48}}\b{escaped}\b",
        rf"\b{escaped}\b[^.!?]{{0,48}}\byou\s+(?:said|mentioned|reported|stated)\b",
    )
    return any(re.search(pattern, answer, re.IGNORECASE) for pattern in attributed_patterns)


def is_safe_policy_abstention(answer: str) -> bool:
    """Recognise the shared safe fallback before it is treated as an answer."""

    normalized = answer.lower()
    return bool(
        re.search(
            r"(?:can(?:not|'t|’t) confirm|unable to confirm|cannot determine|need (?:an )?order reference|"
            r"provide (?:the )?product category|contact support)",
            normalized,
        )
    )
