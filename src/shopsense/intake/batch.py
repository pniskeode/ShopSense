"""Streaming JSONL batch runner for ShopSense M1."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from shopsense.intake.parser import parse_support_ticket
from shopsense.llm.client import LiteLLMClient, ModelConfigurationError


@dataclass(frozen=True)
class BatchStats:
    """Summary returned after writing the batch result artifact."""

    total: int
    successes: int
    failures: int


@dataclass(frozen=True)
class RetryStats:
    """Summary returned after retrying existing failed result records."""

    attempted: int
    recovered: int
    remaining_failed: int


def iter_jsonl_records(path: Path) -> Iterator[dict]:
    """Yield one decoded source record at a time."""

    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object on line {line_number} of {path}")
            yield record


def run_batch(
    input_path: Path,
    output_path: Path,
    client: LiteLLMClient,
    *,
    max_retries: int = 2,
    repair_max_tokens: int = 250,
    delay_seconds: float = 5.0,
) -> BatchStats:
    """Process all records and write exactly one result record per input record."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = successes = failures = 0

    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in iter_jsonl_records(input_path):
            total += 1
            record_id = record.get("record_id")
            ticket_id = record.get("ticket_id")
            raw_text = record.get("raw_text")

            if not isinstance(raw_text, str):
                ticket = None
                _errors = ["ValueError: record raw_text must be a string"]
            else:
                ticket, _errors = parse_support_ticket(
                    raw_text,
                    client,
                    max_retries=max_retries,
                    repair_max_tokens=repair_max_tokens,
                )

            status = "success" if ticket is not None else "failed"
            if status == "success":
                successes += 1
            else:
                failures += 1

            result = {
                "record_id": record_id,
                "ticket_id": ticket_id,
                "status": status,
                "support_ticket": ticket.model_dump(mode="json") if ticket else None,
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")

            if delay_seconds > 0:
                time.sleep(delay_seconds)

    return BatchStats(total=total, successes=successes, failures=failures)


def retry_failed_records(
    input_path: Path,
    output_path: Path,
    client: LiteLLMClient,
    *,
    max_retries: int = 2,
    repair_max_tokens: int = 250,
    delay_seconds: float = 5.0,
) -> RetryStats:
    """Retry only records currently marked failed in an existing result file."""

    source_records = {
        record.get("record_id"): record for record in iter_jsonl_records(input_path)
    }
    result_records = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_records = [record for record in result_records if record.get("status") == "failed"]
    recovered = 0

    for index, result in enumerate(failed_records):
        source = source_records.get(result.get("record_id"))
        raw_text = source.get("raw_text") if source else None
        if isinstance(raw_text, str):
            ticket, _errors = parse_support_ticket(
                raw_text,
                client,
                max_retries=max_retries,
                repair_max_tokens=repair_max_tokens,
            )
            if ticket is not None:
                result["status"] = "success"
                result["support_ticket"] = ticket.model_dump(mode="json")
                recovered += 1
        if delay_seconds > 0 and index < len(failed_records) - 1:
            time.sleep(delay_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=output_path.parent,
            prefix=f"{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            for result in result_records:
                temporary.write(json.dumps(result, ensure_ascii=False) + "\n")
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path:
            try:
                os.unlink(temporary_path)
            except FileNotFoundError:
                pass

    return RetryStats(
        attempted=len(failed_records),
        recovered=recovered,
        remaining_failed=len(failed_records) - recovered,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ShopSense M1 structured intake")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/shopsense/intake/records.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/shopsense/m1_results/parsed_records.jsonl"),
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="retry only records currently marked failed in the output artifact",
    )
    args = parser.parse_args()

    try:
        client = LiteLLMClient()
    except ModelConfigurationError as exc:
        parser.error(str(exc))

    if args.retry_failed:
        stats = retry_failed_records(
            args.input,
            args.output,
            client,
            max_retries=args.max_retries,
            delay_seconds=args.delay_seconds,
        )
        print(
            f"Retried {stats.attempted} failed records: "
            f"{stats.recovered} recovered, {stats.remaining_failed} still failed"
        )
    else:
        stats = run_batch(
            args.input,
            args.output,
            client,
            max_retries=args.max_retries,
            delay_seconds=args.delay_seconds,
        )
        print(f"Processed {stats.total} records: {stats.successes} success, {stats.failures} failed")


if __name__ == "__main__":
    main()
