import json
from pathlib import Path

from shopsense.intake.batch import retry_failed_records, run_batch
from shopsense.llm.client import LiteLLMClient

from .helpers import llm_response


class FakeCompletion:
    def __init__(self) -> None:
        self.index = 0

    def __call__(self, **kwargs):
        self.index += 1
        return llm_response(
            json.dumps(
                {
                    "issue_type": "track_order",
                    "order_id": "KW-O-1",
                    "sentiment": "calm",
                    "urgency": "medium",
                }
            )
        )


def test_batch_writes_one_result_per_input(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "results.jsonl"
    records = [
        {"record_id": "R-1", "ticket_id": "T-1", "raw_text": "Where is my order?"},
        {"record_id": "R-2", "ticket_id": "T-2", "raw_text": "Please help."},
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    stats = run_batch(
        input_path,
        output_path,
        LiteLLMClient(model="test-model", completion_fn=FakeCompletion()),
        delay_seconds=0,
    )

    results = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert stats.total == stats.successes == 2
    assert stats.failures == 0
    assert len(results) == len(records)
    assert [result["record_id"] for result in results] == ["R-1", "R-2"]
    assert all(result["status"] == "success" for result in results)
    assert all(set(result) == {"record_id", "ticket_id", "status", "support_ticket"} for result in results)


def test_retry_failed_records_updates_only_failed_entries(tmp_path: Path) -> None:
    input_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "results.jsonl"
    input_records = [
        {"record_id": "R-1", "ticket_id": "T-1", "raw_text": "Where is my order?"},
        {"record_id": "R-2", "ticket_id": "T-2", "raw_text": "Please help."},
    ]
    input_path.write_text("\n".join(json.dumps(record) for record in input_records) + "\n", encoding="utf-8")
    output_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "record_id": record["record_id"],
                    "ticket_id": record["ticket_id"],
                    "status": "failed" if record["record_id"] == "R-1" else "success",
                    "support_ticket": None
                    if record["record_id"] == "R-1"
                    else {
                        "issue_type": "track_order",
                        "order_id": "KW-O-1",
                        "sentiment": "calm",
                        "urgency": "medium",
                    },
                }
            )
            for record in input_records
        )
        + "\n",
        encoding="utf-8",
    )

    stats = retry_failed_records(
        input_path,
        output_path,
        LiteLLMClient(model="test-model", completion_fn=FakeCompletion()),
        delay_seconds=0,
    )

    results = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert stats == type(stats)(attempted=1, recovered=1, remaining_failed=0)
    assert all(result["status"] == "success" for result in results)
    assert all("errors" not in result for result in results)
