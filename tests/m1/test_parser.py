import json

from shopsense.intake.parser import parse_support_ticket
from shopsense.llm.client import LiteLLMClient

from .helpers import llm_response


class FakeCompletion:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return llm_response(next(self.responses))


def test_valid_json_succeeds_without_repair() -> None:
    fake = FakeCompletion(
        ['{"issue_type":"track_order","order_id":"KW-O-1","sentiment":"calm","urgency":"medium"}']
    )
    ticket, errors = parse_support_ticket(
        "Where is KW-O-1?", LiteLLMClient(model="test-model", completion_fn=fake)
    )

    assert ticket is not None
    assert ticket.order_id == "KW-O-1"
    assert errors == []
    assert len(fake.calls) == 1
    assert fake.calls[0]["response_format"] == {"type": "json_object"}
    assert "Where is KW-O-1?" in fake.calls[0]["messages"][0]["content"]


def test_schema_failure_uses_repair_and_succeeds() -> None:
    fake = FakeCompletion(
        [
            '{"issue_type":"track_order","order_id":null,"sentiment":"calm"}',
            '{"issue_type":"track_order","order_id":null,"sentiment":"calm","urgency":"low"}',
        ]
    )
    ticket, errors = parse_support_ticket(
        "Can you help?", LiteLLMClient(model="test-model", completion_fn=fake)
    )

    assert ticket is not None
    assert len(errors) == 1
    assert len(fake.calls) == 2
    repair_prompt = fake.calls[1]["messages"][0]["content"]
    assert "Can you help?" in repair_prompt
    assert "Previous JSON attempt" in repair_prompt
    assert "ValidationError" not in repair_prompt or "validation" in repair_prompt.lower()
    assert fake.calls[1]["max_tokens"] == 250


def test_malformed_json_uses_repair_and_succeeds() -> None:
    fake = FakeCompletion(
        [
            "not json",
            '{"issue_type":"product_safety","order_id":null,"sentiment":"calm","urgency":"critical"}',
        ]
    )
    ticket, errors = parse_support_ticket(
        "The product caught fire.", LiteLLMClient(model="test-model", completion_fn=fake)
    )

    assert ticket is not None
    assert errors[0].startswith("JSONDecodeError:")
    assert len(fake.calls) == 2


def test_repair_loop_is_bounded_and_returns_failed_result() -> None:
    invalid = json.dumps(
        {"issue_type": "not_valid", "order_id": None, "sentiment": "calm", "urgency": "low"}
    )
    fake = FakeCompletion([invalid, invalid, invalid, invalid])
    ticket, errors = parse_support_ticket(
        "This cannot be classified.",
        LiteLLMClient(model="test-model", completion_fn=fake),
        max_retries=2,
    )

    assert ticket is None
    assert len(errors) == 3
    assert len(fake.calls) == 3
