import pytest
from pydantic import ValidationError

from shopsense.models.support_ticket import SupportTicket


def test_support_ticket_accepts_complete_output() -> None:
    ticket = SupportTicket(
        issue_type="track_order",
        order_id="KW-O-67890",
        sentiment="calm",
        urgency="medium",
    )
    assert ticket.model_dump(mode="json") == {
        "issue_type": "track_order",
        "order_id": "KW-O-67890",
        "sentiment": "calm",
        "urgency": "medium",
    }


def test_support_ticket_requires_order_id_key_but_accepts_null() -> None:
    ticket = SupportTicket(
        issue_type="damaged_delivery",
        order_id=None,
        sentiment="angry",
        urgency="high",
    )
    assert ticket.order_id is None

    with pytest.raises(ValidationError):
        SupportTicket(
            issue_type="damaged_delivery",
            sentiment="angry",
            urgency="high",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("issue_type", "made_up_issue"),
        ("sentiment", "joyful"),
        ("urgency", "ASAP"),
    ],
)
def test_controlled_vocabularies_reject_unknown_values(field: str, value: str) -> None:
    payload = {
        "issue_type": "track_order",
        "order_id": None,
        "sentiment": "calm",
        "urgency": "low",
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        SupportTicket(**payload)


def test_support_ticket_rejects_extra_keys() -> None:
    with pytest.raises(ValidationError):
        SupportTicket(
            issue_type="track_order",
            order_id=None,
            sentiment="calm",
            urgency="low",
            requires_human=False,
        )
