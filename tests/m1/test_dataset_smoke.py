import json
from pathlib import Path


DATASET = Path(__file__).parents[2] / "data" / "shopsense" / "intake" / "records.jsonl"


def test_generated_dataset_shape_and_observed_vocabularies() -> None:
    records = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(records) == 200
    assert {"ticket_id", "raw_text", "ground_truth", "record_id"}.issubset(records[0])
    assert all(isinstance(record["raw_text"], str) and record["raw_text"] for record in records)

    intents = {record["ground_truth"]["intent"] for record in records}
    sentiments = {record["ground_truth"]["sentiment"] for record in records}
    assert intents == {
        "account_issue",
        "cancel_order",
        "complaint",
        "damaged_delivery",
        "delivery_delay",
        "missing_item",
        "payment_issue",
        "product_information",
        "product_safety",
        "refund_request",
        "return_request",
        "track_order",
        "warranty_claim",
        "wrong_item",
    }
    assert sentiments == {"angry", "calm", "distressed", "frustrated"}

