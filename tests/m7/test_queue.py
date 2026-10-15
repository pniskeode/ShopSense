from __future__ import annotations

import pytest

from shopsense.reliability import HumanQueue, QueueStateError


def test_queue_is_idempotent_and_requires_manual_lifecycle(tmp_path) -> None:
    queue = HumanQueue(tmp_path / "m7.sqlite", clock=lambda: 123.0)
    first = queue.enqueue(
        case_id="m6:chat:CHAT-1", customer_ref="KW-C-11111", channel="chat",
        failed_operation="lookup_order", failure_category="retry_exhausted", retry_count=3, circuit_state="closed",
    )
    duplicate = queue.enqueue(
        case_id="m6:chat:CHAT-1", customer_ref="different", channel="email",
        failed_operation="track_shipment", failure_category="circuit_open", retry_count=0, circuit_state="open",
    )

    assert first.queue_id == duplicate.queue_id
    assert first.customer_ref_hash != "KW-C-11111"
    assert len(first.customer_ref_hash) == 64
    assert len(queue.list("queued")) == 1
    with pytest.raises(QueueStateError):
        queue.resolve(first.queue_id, operator="Asha", note="done")
    claimed = queue.claim(first.queue_id, operator="Asha")
    resolved = queue.resolve(first.queue_id, operator="Asha", note="Verified in order system")
    assert claimed.status == "claimed"
    assert resolved.status == "resolved"
