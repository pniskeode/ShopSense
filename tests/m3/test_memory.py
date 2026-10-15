from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from shopsense.memory.bootstrap import bootstrap_historical_memory, build_historical_tickets
from shopsense.memory.service import (
    CustomerMemoryService,
    MemoryHit,
    SupermemoryBackend,
    extract_explicit_preferences,
)

from .helpers import FakeMemoryBackend


def test_historical_ticket_builder_joins_every_successful_m1_record() -> None:
    tickets = build_historical_tickets(
        intake_path=Path("data/shopsense/intake/records.jsonl"),
        parsed_path=Path("data/shopsense/m1_results/parsed_records.jsonl"),
    )

    assert len(tickets) == 200
    assert all(ticket["customer_ref"] for ticket in tickets)
    assert all(ticket["issue_type"] and ticket["urgency"] for ticket in tickets)


def test_customer_memory_is_strictly_isolated_by_customer_ref(tmp_path) -> None:
    backend = FakeMemoryBackend()
    service = CustomerMemoryService(backend)
    service.record_interaction("KW-C-A", "Where is my package?", "It is in transit.")
    service.record_interaction("KW-C-B", "I need a refund.", "A refund is pending.")

    hits = service.recall_customer_memory("KW-C-A", "package")

    assert len(hits) == 1
    assert "transit" in hits[0].text
    assert "refund" not in hits[0].text


def test_only_explicit_preferences_are_saved_and_new_version_supersedes_old(tmp_path) -> None:
    backend = FakeMemoryBackend()
    service = CustomerMemoryService(backend)

    assert service.remember_explicit_preferences("KW-C-1", "I am angry about this delay.").stored == 0
    assert service.remember_explicit_preferences("KW-C-1", "I prefer email updates.").stored == 1
    assert service.remember_explicit_preferences("KW-C-1", "I prefer SMS updates.").stored == 1

    hits = service.recall_customer_memory("KW-C-1", "How should I contact this customer?", k=5)

    assert len(hits) == 1
    assert "SMS updates" in hits[0].text
    assert hits[0].metadata["preference_version"] == 2


def test_latest_preference_is_recalled_after_restart_from_supermemory() -> None:
    class HistoricalOnlyBackend(FakeMemoryBackend):
        def search(self, query: str, customer_ref: str, limit: int) -> list[MemoryHit]:
            del query, customer_ref, limit
            return [MemoryHit("episodic", "Past delivery-delay ticket.", 0.9, {})]

    backend = HistoricalOnlyBackend()
    first_session = CustomerMemoryService(backend)
    first_session.remember_explicit_preferences(
        "KW-C-11111", "My preference is email or message; phone calls only in an emergency."
    )

    restarted_session = CustomerMemoryService(backend)
    hits = restarted_session.recall_customer_memory("KW-C-11111", "What are my preferences?")

    assert hits[0].kind == "semantic"
    assert "phone calls only in an emergency" in hits[0].text
    assert hits[0].metadata["source"] == "explicit_preference"
    assert any(hit.kind == "episodic" for hit in hits)


def test_supermemory_search_uses_documents_only_when_no_extracted_memory_exists() -> None:
    class Search:
        def memories(self, **kwargs):
            self.memory_kwargs = kwargs
            return SimpleNamespace(
                results=[SimpleNamespace(memory="Memory result", similarity=0.4, metadata={"type": "episodic"})]
            )

        def documents(self, **kwargs):
            self.document_kwargs = kwargs
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        chunks=[SimpleNamespace(content="Document result")],
                        score=0.9,
                        metadata={"type": "semantic"},
                    )
                ]
            )

    backend = object.__new__(SupermemoryBackend)
    backend.client = SimpleNamespace(search=Search())

    hits = backend.search("preference", "KW-C-1", limit=2)

    assert [hit.text for hit in hits] == ["Memory result"]
    assert not hasattr(backend.client.search, "document_kwargs")


def test_supermemory_search_uses_document_fallback_when_extracted_memory_is_empty() -> None:
    class Search:
        def memories(self, **kwargs):
            self.memory_kwargs = kwargs
            return SimpleNamespace(results=[])

        def documents(self, **kwargs):
            self.document_kwargs = kwargs
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        chunks=[SimpleNamespace(content="Document fallback")],
                        score=0.9,
                        metadata={"type": "episodic"},
                    )
                ]
            )

    backend = object.__new__(SupermemoryBackend)
    backend.client = SimpleNamespace(search=Search())

    hits = backend.search("delivery", "KW-C-1", limit=2)

    assert [hit.text for hit in hits] == ["Document fallback"]
    assert backend.client.search.document_kwargs["container_tags"] == ["KW-C-1"]


def test_supermemory_preference_search_falls_back_to_explicit_preference_documents() -> None:
    class Search:
        def memories(self, **kwargs):
            self.memory_kwargs = kwargs
            return SimpleNamespace(results=[])

        def documents(self, **kwargs):
            self.document_kwargs = kwargs
            return SimpleNamespace(
                results=[
                    SimpleNamespace(
                        chunks=[SimpleNamespace(content="Customer explicitly prefers: email updates.")],
                        score=0.9,
                        metadata={
                            "type": "semantic",
                            "source": "explicit_preference",
                            "preference_key": "communication",
                            "preference_version": 1,
                        },
                    ),
                    SimpleNamespace(
                        chunks=[SimpleNamespace(content="Past support ticket")],
                        score=1.0,
                        metadata={"type": "episodic", "source": "m1_history"},
                    ),
                ]
            )

    backend = object.__new__(SupermemoryBackend)
    backend.client = SimpleNamespace(search=Search())

    hits = backend.search_preferences("KW-C-1")

    assert [hit.text for hit in hits] == ["Customer explicitly prefers: email updates."]
    assert backend.client.search.document_kwargs["container_tags"] == ["KW-C-1"]


def test_preference_write_polls_extracted_memory_without_sleeping_when_ready() -> None:
    backend = FakeMemoryBackend()
    sleeps: list[float] = []
    service = CustomerMemoryService(backend, sleep_fn=sleeps.append)

    result = service.remember_explicit_preferences("KW-C-1", "I prefer email updates.")

    assert result.stored == 1
    assert result.warnings == ()
    assert sleeps == []


def test_preference_write_warns_after_bounded_extraction_poll() -> None:
    class DelayedBackend(FakeMemoryBackend):
        def search_extracted_preferences(self, customer_ref: str) -> list[MemoryHit]:
            del customer_ref
            return []

    backend = DelayedBackend()
    sleeps: list[float] = []
    service = CustomerMemoryService(
        backend,
        preference_poll_attempts=3,
        preference_poll_interval_seconds=2,
        sleep_fn=sleeps.append,
    )

    result = service.remember_explicit_preferences("KW-C-1", "I prefer email updates.")

    assert result.stored == 1
    assert len(result.warnings) == 1
    assert sleeps == [2, 2]


def test_explicit_preference_extractor_does_not_infer_from_sentiment() -> None:
    assert extract_explicit_preferences("This has been a frustrating delivery.") == []
    preferences = extract_explicit_preferences("My preference is a refund instead of a replacement.")
    assert preferences[0].key == "resolution"
    assert "refund instead of a replacement" in preferences[0].statement


def test_explicit_contact_instruction_is_stored_without_guessing_from_sentiment() -> None:
    preferences = extract_explicit_preferences("Contact me by email only. Please do not call me.")

    assert len(preferences) == 2
    assert all(preference.key == "communication" for preference in preferences)
    assert "email only" in preferences[0].statement
    assert "do not call me" in preferences[1].statement


def test_historical_bootstrap_is_idempotent(tmp_path) -> None:
    backend = FakeMemoryBackend()
    service = CustomerMemoryService(backend)
    state_path = tmp_path / "bootstrap.json"

    first = bootstrap_historical_memory(service, state_path=state_path)
    second = bootstrap_historical_memory(service, state_path=state_path)

    assert first.total_successful_tickets == 200
    assert first.stored_tickets == 200
    assert second.stored_tickets == 0
    assert second.skipped_tickets == 200


def test_supermemory_backend_omits_none_metadata_values(monkeypatch) -> None:
    class Client:
        def add(self, **kwargs):
            self.kwargs = kwargs

    backend = object.__new__(SupermemoryBackend)
    backend.client = Client()

    backend.add("ticket", "KW-C-1", {"order_ref": None, "ticket_id": "KW-T-1"})

    assert backend.client.kwargs["metadata"] == {"ticket_id": "KW-T-1"}
