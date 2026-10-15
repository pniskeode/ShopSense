"""Safe automatic message identifiers for simulated email."""

from __future__ import annotations

import hashlib
import hmac
import os
import re


class CaseIdConfigurationError(RuntimeError):
    """Raised when automatic simulated-email identity cannot be created safely."""


def _normalise(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def derive_email_message_id(
    customer_ref: str,
    subject: str | None,
    body: str,
    *,
    salt: str | None = None,
) -> str:
    """Return a deterministic but non-guessable message ID for simulated email.

    Real email integrations must supply their provider's immutable message ID.
    This fallback gives the command-line simulation repeat-safe identity without
    exposing a plain hash of customer text.
    """

    secret = salt if salt is not None else os.getenv("SHOPSENSE_CASE_ID_SALT", "")
    if not secret.strip():
        raise CaseIdConfigurationError(
            "SHOPSENSE_CASE_ID_SALT is required when a simulated email omits message_id."
        )
    canonical = "\x1f".join((_normalise(customer_ref), _normalise(subject), _normalise(body)))
    digest = hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"EMAIL-{digest[:32]}"
