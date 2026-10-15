"""Idempotent bootstrap of M1 ticket history into persistent customer memory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .service import CustomerMemoryService


def _project_data_path(*parts: str) -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "shopsense" / Path(*parts)


DEFAULT_INTAKE_PATH = _project_data_path("intake", "records.jsonl")
DEFAULT_PARSED_PATH = _project_data_path("m1_results", "parsed_records.jsonl")
DEFAULT_BOOTSTRAP_STATE_PATH = _project_data_path("memory", "bootstrap_state.json")


@dataclass(frozen=True)
class BootstrapStats:
    total_successful_tickets: int
    stored_tickets: int
    skipped_tickets: int
    stored_preferences: int


class BootstrapState:
    """A local idempotency ledger; it never contains ticket text or credentials."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def contains(self, fingerprint: str) -> bool:
        return fingerprint in self._load()

    def add(self, fingerprint: str) -> None:
        state = self._load()
        state.add(fingerprint)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(sorted(state), stream, ensure_ascii=False, indent=2)
            temporary_path = Path(stream.name)
        try:
            os.replace(temporary_path, self.path)
        except PermissionError:
            # Windows security tooling can temporarily lock a newly created
            # file. The idempotency ledger is tiny, so a direct-write fallback
            # is preferable to making bootstrap unusable.
            self.path.write_text(
                json.dumps(sorted(state), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass

    def _load(self) -> set[str]:
        if not self.path.exists():
            return set()
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
            raise ValueError(f"Invalid bootstrap state file: {self.path}")
        return set(payload)


def build_historical_tickets(intake_path: Path, parsed_path: Path) -> list[dict[str, Any]]:
    """Join M1 source records and successful parsed results by their stable record id."""

    intake_rows = _read_jsonl(intake_path)
    parsed_by_id = {row.get("record_id"): row for row in _read_jsonl(parsed_path)}
    tickets: list[dict[str, Any]] = []
    for source in intake_rows:
        parsed = parsed_by_id.get(source.get("record_id"))
        if not parsed or parsed.get("status") != "success":
            continue
        ticket = parsed.get("support_ticket")
        if not isinstance(ticket, dict):
            continue
        customer_ref = source.get("customer_ref")
        if not isinstance(customer_ref, str) or not customer_ref.strip():
            raise ValueError(f"Missing customer_ref for historical ticket {source.get('record_id')}")

        record = {
            "record_id": source["record_id"],
            "ticket_id": source["ticket_id"],
            "customer_ref": customer_ref,
            "order_ref": source.get("order_ref"),
            "received_at": source["received_at"],
            "raw_text": source["raw_text"],
            "issue_type": ticket["issue_type"],
            "sentiment": ticket["sentiment"],
            "urgency": ticket["urgency"],
        }
        record["fingerprint"] = _fingerprint(record)
        tickets.append(record)
    return sorted(tickets, key=lambda item: (item["received_at"], item["record_id"]))


def bootstrap_historical_memory(
    service: CustomerMemoryService,
    *,
    intake_path: Path = DEFAULT_INTAKE_PATH,
    parsed_path: Path = DEFAULT_PARSED_PATH,
    state_path: Path = DEFAULT_BOOTSTRAP_STATE_PATH,
) -> BootstrapStats:
    """Persist every successful M1 ticket once, retaining customer isolation."""

    state = BootstrapState(state_path)
    tickets = build_historical_tickets(intake_path, parsed_path)
    stored_tickets = skipped_tickets = stored_preferences = 0
    for ticket in tickets:
        if state.contains(ticket["fingerprint"]):
            skipped_tickets += 1
            continue
        service.store_historical_ticket(ticket)
        preference_result = service.remember_explicit_preferences(ticket["customer_ref"], ticket["raw_text"])
        stored_preferences += preference_result.stored
        state.add(ticket["fingerprint"])
        stored_tickets += 1
    return BootstrapStats(
        total_successful_tickets=len(tickets),
        stored_tickets=stored_tickets,
        skipped_tickets=skipped_tickets,
        stored_preferences=stored_preferences,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            rows.append(row)
    return rows


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
