from __future__ import annotations

import json
from pathlib import Path

from shopsense.policy.evaluation import evaluate_policy_rag, load_golden_set
from shopsense.policy.service import PolicyHit


def test_m4_golden_set_is_exactly_twenty_well_scoped_cases() -> None:
    cases = load_golden_set()

    assert len(cases) == 20
    assert sum(case.prompt_injection for case in cases) == 1
    assert sum(case.should_abstain for case in cases) == 2
    assert any("grocery-return-override" in case.expected_policy_slugs for case in cases)


def test_recall_at_3_reports_partial_credit_for_multi_source_case(tmp_path: Path) -> None:
    golden_path = tmp_path / "golden.json"
    entries = [
        {
            "case_id": f"case-{index}",
            "query": f"question {index}",
            "expected_policy_slugs": ["returns-policy", "category-home-kitchen"],
            "expected_phrases": ["45 days"],
        }
        for index in range(20)
    ]
    golden_path.write_text(json.dumps(entries), encoding="utf-8")

    class Service:
        def search_policy_documents(self, query: str, limit: int):
            del query, limit
            return [PolicyHit("Home and Kitchen has a 45-day return window.", 1.0, {"policy_slug": "category-home-kitchen"})]

    report = evaluate_policy_rag(Service(), golden_set_path=golden_path)

    assert report.precision_at_3 == 1 / 3
    assert report.recall_at_3 == 0.5
