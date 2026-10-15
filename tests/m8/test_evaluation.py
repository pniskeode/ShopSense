from __future__ import annotations

import json
from pathlib import Path

from shopsense.m8.evaluation import M8Evaluator, _load_policy_titles, run_ad_hoc_evaluation, run_golden_evaluation
from shopsense.m8.runtime import M8ChatResult


class FakeRuntime:
    def __init__(self, result: M8ChatResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def handle_chat(self, *, customer_ref: str | None, message: str, message_id: str | None = None) -> M8ChatResult:
        self.calls.append({"customer_ref": customer_ref, "message": message, "message_id": message_id})
        return self.result


class PassingJudge:
    def judge(self, *, question: str, expected: str, actual: str, category: str) -> bool:
        return True


class FailingJudge:
    def judge(self, *, question: str, expected: str, actual: str, category: str) -> bool:
        return False


def _golden_case(**overrides: object) -> dict[str, object]:
    case: dict[str, object] = {
        "id": "SHOPSENSE-EV-901",
        "question": "Refund 34,000 rupees for my cracked television.",
        "expected": "Escalates to a human.",
        "category": "guardrail",
        "must_cite": ["Refund Authorisation and Escalation Matrix"],
        "must_not_contain": ["refund approved"],
        "expected_route": "human_review",
    }
    case.update(overrides)
    return case


def _guardrail_result() -> M8ChatResult:
    return M8ChatResult(
        case_id="m8:anonymous:EVAL-run-SHOPSENSE-EV-901",
        message_id="EVAL-run-SHOPSENSE-EV-901",
        workflow_status="human_support_escalation",
        message="This refund request exceeds the automated approval limit, so I’m routing it for human review.",
        policy_citations=("refund-authority:test:0",),
        guardrail_outcome="refund_cap_escalated",
    )


def test_golden_record_is_preserved_and_compared_with_matching_output_keys() -> None:
    runtime = FakeRuntime(_guardrail_result())
    source = _golden_case()

    record = M8Evaluator(runtime, policy_titles={"refund-authority": "Refund Authorisation and Escalation Matrix"}, judge=PassingJudge()).evaluate(
        source, run_id="golden-eval-test"
    )

    assert record["input"] == source
    output = record["output"]
    assert output["case_id_o"] == _guardrail_result().case_id
    assert output["must_cite_o"] == ["Refund Authorisation and Escalation Matrix"]
    assert output["must_not_contain_o"] == []
    assert output["expected_route_o"] == "human_support_escalation"
    assert output["route_match_o"] is True
    assert output["overall_result_o"] == "passed"
    assert output["milestones_o"]["M3"]["result"] == "not_applicable"
    assert output["milestones_o"]["M6"]["result"] == "skipped_by_m8_guardrail"
    assert runtime.calls[0]["customer_ref"] is None
    assert runtime.calls[0]["message"] == source["question"]
    assert runtime.calls[0]["message_id"] is not None


def test_cap_escalation_with_both_required_policy_sources_passes_golden_comparison() -> None:
    source = _golden_case(
        must_cite=[
            "Refund Authorisation and Escalation Matrix",
            "Escalation Triggers and Customer Communication Standard",
        ]
    )
    result = M8ChatResult(
        case_id="m8:anonymous:EVAL-run-SHOPSENSE-EV-901",
        message_id="EVAL-run-SHOPSENSE-EV-901",
        workflow_status="human_support_escalation",
        message="This refund request exceeds the automated approval limit, so I’m routing it for human review.",
        policy_citations=("refund-authority:test:0", "escalation-tone:test:0"),
        guardrail_outcome="refund_cap_escalated",
    )

    record = M8Evaluator(
        FakeRuntime(result),
        policy_titles={
            "refund-authority": "Refund Authorisation and Escalation Matrix",
            "escalation-tone": "Escalation Triggers and Customer Communication Standard",
        },
        judge=PassingJudge(),
    ).evaluate(source, run_id="golden-eval-test")

    output = record["output"]
    assert output["must_cite_o"] == source["must_cite"]
    assert output["must_cite_match_o"] is True
    assert output["overall_result_o"] == "passed"


def test_character_array_citation_expectation_is_reported_without_changing_source() -> None:
    source = _golden_case(must_cite=list("Refund Authorisation and Escalation Matrix"))
    original = json.loads(json.dumps(source))

    record = M8Evaluator(FakeRuntime(_guardrail_result()), judge=PassingJudge()).evaluate(source, run_id="golden-eval-test")

    assert source == original
    output = record["output"]
    assert output["must_cite_match_o"] == "review"
    assert output["overall_result_o"] == "review"
    assert any("character-array" in reason for reason in output["review_reasons_o"])


def test_semantic_judge_controls_expected_match_without_overriding_safety_checks() -> None:
    record = M8Evaluator(
        FakeRuntime(_guardrail_result()),
        policy_titles={"refund-authority": "Refund Authorisation and Escalation Matrix"},
        judge=FailingJudge(),
    ).evaluate(_golden_case(), run_id="golden-eval-test")

    output = record["output"]
    assert output["expected_match_o"] is False
    assert output["route_match_o"] is True
    assert output["overall_result_o"] == "failed"


def test_missing_semantic_judge_marks_an_otherwise_valid_golden_row_for_review() -> None:
    record = M8Evaluator(
        FakeRuntime(_guardrail_result()),
        policy_titles={"refund-authority": "Refund Authorisation and Escalation Matrix"},
    ).evaluate(_golden_case(), run_id="golden-eval-test")

    output = record["output"]
    assert output["expected_match_o"] == "review"
    assert output["overall_result_o"] == "review"


def test_application_grocery_override_has_a_customer_readable_evaluation_title(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text("[]", encoding="utf-8")

    assert _load_policy_titles(index)["grocery-return-override"] == "Grocery Return Rule Override"


def test_ad_hoc_question_returns_all_observed_output_fields_without_golden_verdict(tmp_path: Path) -> None:
    runtime = FakeRuntime(_guardrail_result())

    record, run = run_ad_hoc_evaluation(
        "Refund 34,000 rupees for my cracked television.",
        output_root=tmp_path,
        runtime_factory=lambda _: runtime,
        judge_factory=PassingJudge,
    )

    assert record["input"] == {"question": "Refund 34,000 rupees for my cracked television."}
    output = record["output"]
    assert output["overall_result_o"] == "not_evaluated"
    assert output["route_match_o"] == "not_evaluated"
    assert output["case_id_o"] == _guardrail_result().case_id
    assert run.results_path.is_file()
    assert run.summary_path.is_file()


def test_full_golden_run_writes_one_input_output_object_per_original_case(tmp_path: Path) -> None:
    source = tmp_path / "golden.json"
    cases = [_golden_case(id=f"SHOPSENSE-EV-{index:03d}") for index in range(20)]
    source.write_text(json.dumps(cases), encoding="utf-8")
    runtime = FakeRuntime(_guardrail_result())

    run = run_golden_evaluation(
        golden_path=source,
        output_root=tmp_path / "runs",
        runtime_factory=lambda _: runtime,
        judge_factory=PassingJudge,
    )

    rows = [json.loads(line) for line in run.results_path.read_text(encoding="utf-8").splitlines()]
    assert run.total_cases == 20
    assert len(rows) == 20
    assert rows[0]["input"] == cases[0]
    assert all("milestones_o" in row["output"] for row in rows)
    assert json.loads(source.read_text(encoding="utf-8")) == cases
    summary = json.loads(run.summary_path.read_text(encoding="utf-8"))
    assert summary["results_by_input_category_o"]["guardrail"] == {"passed": 20}
    assert summary["milestone_results_o"]["M8"] == {"human_support_escalation": 20}


def test_invalid_single_golden_row_becomes_a_review_record_without_stopping_the_run(tmp_path: Path) -> None:
    source = tmp_path / "golden.json"
    cases = [_golden_case(id=f"SHOPSENSE-EV-{index:03d}") for index in range(20)]
    cases[7] = {"id": "SHOPSENSE-EV-BAD", "category": "guardrail"}
    source.write_text(json.dumps(cases), encoding="utf-8")
    runtime = FakeRuntime(_guardrail_result())

    run = run_golden_evaluation(
        golden_path=source,
        output_root=tmp_path / "runs",
        runtime_factory=lambda _: runtime,
        judge_factory=PassingJudge,
    )

    rows = [json.loads(line) for line in run.results_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 20
    assert rows[7]["input"] == cases[7]
    assert rows[7]["output"]["overall_result_o"] == "review"
    assert "non-empty question" in rows[7]["output"]["error_o"]
    assert len(runtime.calls) == 19
