from __future__ import annotations

import pytest

from shopsense.reliability import CaseIdConfigurationError, derive_email_message_id
from shopsense.workflow.m6 import IncomingMessage


def test_simulated_email_without_provider_id_derives_a_stable_hmac_id(monkeypatch) -> None:
    monkeypatch.setenv("SHOPSENSE_CASE_ID_SALT", "unit-test-salt")
    first = IncomingMessage.email({"customer_ref": "KW-C-11111", "subject": "  Help ", "body": " My order "})
    second = IncomingMessage.email({"customer_ref": "KW-C-11111", "subject": "Help", "body": "My order"})

    assert first.message_id == second.message_id
    assert first.message_id.startswith("EMAIL-")
    assert "KW-C" not in first.message_id


def test_derived_email_id_requires_a_secret_salt(monkeypatch) -> None:
    monkeypatch.delenv("SHOPSENSE_CASE_ID_SALT", raising=False)
    with pytest.raises(CaseIdConfigurationError):
        derive_email_message_id("KW-C-11111", "Help", "My order")
