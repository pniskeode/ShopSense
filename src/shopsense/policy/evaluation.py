"""Local, reproducible M4 policy-retrieval and groundedness evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from .grounding import validate_answer_grounding
from .service import DEFAULT_CORPUS_ROOT, PolicyHit, PolicyService


DEFAULT_GOLDEN_SET = DEFAULT_CORPUS_ROOT.parent / "evaluation" / "m4_golden_set.json"


@dataclass(frozen=True)
class GoldenCase:
    case_id: str
    query: str
    expected_policy_slugs: tuple[str, ...]
    expected_phrases: tuple[str, ...]
    should_abstain: bool = False
    prompt_injection: bool = False


@dataclass(frozen=True)
class CaseEvaluation:
    case_id: str
    retrieval_hit: bool
    reciprocal_rank: float
    grounded: bool | None
    errors: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationReport:
    total_cases: int
    retrieval_cases: int
    precision_at_3: float
    recall_at_3: float
    mrr: float
    grounded_answers: int
    answered_cases: int
    cases: tuple[CaseEvaluation, ...]

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def load_golden_set(path: Path = DEFAULT_GOLDEN_SET) -> list[GoldenCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 20:
        raise ValueError("The M4 golden set must contain exactly 20 cases")
    cases = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("Each golden-set entry must be an object")
        cases.append(
            GoldenCase(
                case_id=_required(entry, "case_id"),
                query=_required(entry, "query"),
                expected_policy_slugs=tuple(_string_list(entry, "expected_policy_slugs")),
                expected_phrases=tuple(_string_list(entry, "expected_phrases")),
                should_abstain=bool(entry.get("should_abstain", False)),
                prompt_injection=bool(entry.get("prompt_injection", False)),
            )
        )
    return cases


def evaluate_policy_rag(
    service: PolicyService,
    *,
    answer_fn: Callable[[str], tuple[str, list[dict[str, Any]]]] | None = None,
    golden_set_path: Path = DEFAULT_GOLDEN_SET,
) -> EvaluationReport:
    """Measure retrieval and, when supplied, runtime answer grounding over 20 cases."""

    cases = load_golden_set(golden_set_path)
    evaluations: list[CaseEvaluation] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    reciprocal_ranks: list[float] = []
    grounded_answers = 0
    answered_cases = 0

    for case in cases:
        hits = service.search_policy_documents(case.query, limit=3)
        retrieved_slugs = [str(hit.metadata.get("policy_slug", "")) for hit in hits]
        expected = set(case.expected_policy_slugs)
        matching_ranks = [rank for rank, slug in enumerate(retrieved_slugs, start=1) if slug in expected]
        matching_slugs = set(retrieved_slugs).intersection(expected)
        matching_count = len(matching_slugs)
        # The Day 2 definition measures source coverage, rather than a binary
        # hit.  A multi-source case with one of two required policy sources
        # retrieved must therefore contribute 0.5 Recall@3, not 1.0.
        precision_values.append(matching_count / 3)
        recall_values.append(matching_count / len(expected))
        reciprocal_rank = 1.0 / matching_ranks[0] if matching_ranks else 0.0
        reciprocal_ranks.append(reciprocal_rank)

        grounded: bool | None = None
        errors: tuple[str, ...] = ()
        if answer_fn is not None:
            answer, tool_results = answer_fn(case.query)
            report = validate_answer_grounding(
                answer,
                policy_hits=hits,
                tool_results=tool_results,
                policy_question=True,
                customer_message=case.query,
            )
            phrase_errors = tuple(
                f"Expected policy phrase missing: {phrase}"
                for phrase in case.expected_phrases
                if not case.should_abstain and phrase.lower() not in answer.lower()
            )
            grounded = report.grounded and not phrase_errors
            errors = (*report.errors, *phrase_errors)
            answered_cases += 1
            grounded_answers += int(grounded)
        evaluations.append(
            CaseEvaluation(
                case_id=case.case_id,
                retrieval_hit=bool(matching_count),
                reciprocal_rank=reciprocal_rank,
                grounded=grounded,
                errors=errors,
            )
        )
    total = len(cases)
    return EvaluationReport(
        total_cases=total,
        retrieval_cases=total,
        precision_at_3=sum(precision_values) / total,
        recall_at_3=sum(recall_values) / total,
        mrr=sum(reciprocal_ranks) / total,
        grounded_answers=grounded_answers,
        answered_cases=answered_cases,
        cases=tuple(evaluations),
    )


def _required(entry: dict[str, Any], field: str) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Golden-set field {field!r} must be a non-empty string")
    return value.strip()


def _string_list(entry: dict[str, Any], field: str) -> list[str]:
    value = entry.get(field)
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"Golden-set field {field!r} must be a non-empty string list")
    return [item.strip() for item in value]
