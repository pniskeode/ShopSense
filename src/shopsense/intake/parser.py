"""JSON parsing, Pydantic validation, and bounded repair for M1."""

import json
from typing import Any

from pydantic import ValidationError

from shopsense.llm.client import DEFAULT_REPAIR_MAX_TOKENS, LiteLLMClient
from shopsense.models.support_ticket import SupportTicket
from shopsense.prompts.support_ticket import build_extraction_prompt, build_repair_prompt


def _format_error(error: Exception) -> str:
    """Keep error type visible in the persisted per-record diagnostics."""

    return f"{type(error).__name__}: {error}"


def parse_support_ticket(
    raw_text: str,
    client: LiteLLMClient,
    *,
    max_retries: int = 2,
    repair_max_tokens: int = DEFAULT_REPAIR_MAX_TOKENS,
) -> tuple[SupportTicket | None, list[str]]:
    """Parse one raw ticket, allowing at most ``max_retries`` repair calls.

    The Colab pattern uses ``max_retries=2``: one initial call plus two repair calls.
    """

    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")

    errors: list[str] = []
    try:
        raw = client.complete(build_extraction_prompt(raw_text))
    except Exception as exc:  # provider/library failures must remain record-scoped
        errors.append(_format_error(exc))
        return None, errors

    for attempt in range(max_retries + 1):
        try:
            data: Any = json.loads(raw)
            ticket = SupportTicket.model_validate(data)
            return ticket, errors
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as exc:
            errors.append(_format_error(exc))
            if attempt == max_retries:
                return None, errors

            repair_prompt = build_repair_prompt(
                raw_text=raw_text,
                previous_json=raw,
                error=str(exc),
            )
            try:
                raw = client.complete(repair_prompt, max_tokens=repair_max_tokens)
            except Exception as repair_exc:  # preserve the original and repair-call errors
                errors.append(_format_error(repair_exc))
                return None, errors

    raise AssertionError("bounded repair loop exited unexpectedly")

