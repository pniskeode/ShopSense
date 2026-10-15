from __future__ import annotations

from typing import Any

from shopsense.memory.service import MemoryHit


class FakeMemoryBackend:
    def __init__(self) -> None:
        self.records: list[tuple[str, str, dict[str, Any]]] = []

    def add(self, content: str, customer_ref: str, metadata: dict[str, Any]) -> None:
        self.records.append((customer_ref, content, dict(metadata)))

    def search(self, query: str, customer_ref: str, limit: int) -> list[MemoryHit]:
        del query
        return [
            MemoryHit(
                kind=str(metadata.get("type", "unknown")),
                text=content,
                score=1.0,
                metadata=metadata,
            )
            for stored_customer, content, metadata in reversed(self.records)
            if stored_customer == customer_ref
        ][:limit]

    def search_preferences(self, customer_ref: str) -> list[MemoryHit]:
        return [
            MemoryHit(
                kind="semantic",
                text=content,
                score=1.0,
                metadata=metadata,
            )
            for stored_customer, content, metadata in reversed(self.records)
            if stored_customer == customer_ref and metadata.get("source") == "explicit_preference"
        ]

    def search_extracted_preferences(self, customer_ref: str) -> list[MemoryHit]:
        return self.search_preferences(customer_ref)


class RecordingMemoryService:
    def __init__(self) -> None:
        self.recalled: list[tuple[str, str]] = []
        self.interactions: list[tuple[str, str, str]] = []
        self.preference_texts: list[tuple[str, str]] = []

    def recall_customer_memory(self, customer_ref: str, question: str):
        self.recalled.append((customer_ref, question))
        return [MemoryHit("episodic", "Past ticket: delivery was delayed.", 0.9, {})]

    def format_context(self, customer_ref: str, hits):
        del hits
        return f"M3 CUSTOMER MEMORY\nVerified memory namespace: {customer_ref}\n- past ticket"

    def record_interaction(self, customer_ref: str, question: str, answer: str) -> None:
        self.interactions.append((customer_ref, question, answer))

    def remember_explicit_preferences(self, customer_ref: str, text: str) -> int:
        self.preference_texts.append((customer_ref, text))
        return 0
