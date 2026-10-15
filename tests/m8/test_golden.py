from __future__ import annotations

from shopsense.m8 import load_m8_golden_set


def test_m8_uses_the_supplied_twenty_case_golden_source() -> None:
    cases = load_m8_golden_set()

    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    assert any(case.category == "injection" for case in cases)
    assert any(case.category == "guardrail" for case in cases)
