"""Read the supplied M8 20-ticket golden-set source without changing it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CAPSTONE_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_M8_GOLDEN_SET = (
    CAPSTONE_ROOT / "data-toolkit" / "capstone-data-toolkit" / "data" / "shopsense" / "eval" / "golden_set.json"
)


@dataclass(frozen=True)
class M8GoldenCase:
    """The upstream evaluation record, retained exactly as authored."""

    case_id: str
    question: str
    expected: str
    category: str
    expected_route: str
    must_cite: tuple[str, ...]
    must_not_contain: tuple[str, ...]


def load_m8_golden_set(path: Path = DEFAULT_M8_GOLDEN_SET) -> list[M8GoldenCase]:
    """Load and validate the supplied 20-case M8 golden source file."""

    source = Path(path)
    if not source.is_file():
        raise ValueError(f"M8 golden set was not found: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"M8 golden set is not valid JSON: {source}") from exc
    if not isinstance(raw, list):
        raise ValueError("M8 golden set must be a JSON array")
    cases = [_case(item, index) for index, item in enumerate(raw, start=1)]
    if len(cases) != 20:
        raise ValueError(f"M8 golden set must contain exactly 20 cases; found {len(cases)}")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("M8 golden-set case IDs must be unique")
    if not any(case.category == "injection" for case in cases):
        raise ValueError("M8 golden set must include a prompt-injection case")
    if not any(case.category == "guardrail" for case in cases):
        raise ValueError("M8 golden set must include guardrail cases")
    return cases


def _case(value: Any, index: int) -> M8GoldenCase:
    if not isinstance(value, dict):
        raise ValueError(f"M8 golden-set item {index} must be an object")
    required = ("id", "question", "expected", "category", "expected_route")
    fields: dict[str, str] = {}
    for field in required:
        item = value.get(field)
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"M8 golden-set item {index} field {field!r} must be a non-empty string")
        fields[field] = item.strip()
    return M8GoldenCase(
        case_id=fields["id"],
        question=fields["question"],
        expected=fields["expected"],
        category=fields["category"],
        expected_route=fields["expected_route"],
        must_cite=_string_list(value.get("must_cite"), index, "must_cite"),
        must_not_contain=_string_list(value.get("must_not_contain"), index, "must_not_contain"),
    )


def _string_list(value: Any, index: int, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"M8 golden-set item {index} field {field!r} must be a list of strings")
    return tuple(item for item in value if item)
