from __future__ import annotations

from shopsense.policy.grounding import validate_answer_grounding
from shopsense.policy.service import PolicyHit
from shopsense.agent.runner import ShopSenseAgent
from tests.m2.helpers import FakeChatModel, model_response


def _hit() -> PolicyHit:
    return PolicyHit(
        "Electronics have a 15-day return window.",
        1.0,
        {"citation_id": "category-electronics:policy:markdown:0"},
        "electronics",
    )


def test_grounding_accepts_supported_policy_number_and_exact_citation() -> None:
    report = validate_answer_grounding(
        "Electronics can be returned within 15 days [SOURCE:category-electronics:policy:markdown:0].",
        policy_hits=[_hit()],
        tool_results=[],
        policy_question=True,
    )

    assert report.grounded


def test_grounding_rejects_invented_return_window_even_with_a_real_citation() -> None:
    report = validate_answer_grounding(
        "Electronics can be returned within 90 days [SOURCE:category-electronics:policy:markdown:0].",
        policy_hits=[_hit()],
        tool_results=[],
        policy_question=True,
    )

    assert not report.grounded
    assert "Unsupported numeric claim: 90 days" in report.errors


def test_grounding_allows_an_explicitly_attributed_customer_timeline() -> None:
    report = validate_answer_grounding(
        "You mentioned 40 days. Electronics can be returned within 15 days [SOURCE:category-electronics:policy:markdown:0].",
        policy_hits=[_hit()],
        tool_results=[],
        policy_question=True,
        customer_message="I bought it 40 days ago.",
    )

    assert report.grounded


def test_grounding_does_not_allow_an_unattributed_customer_number_as_policy() -> None:
    report = validate_answer_grounding(
        "The return window is 90 days [SOURCE:category-electronics:policy:markdown:0].",
        policy_hits=[_hit()],
        tool_results=[],
        policy_question=True,
        customer_message="My friend said the window is 90 days.",
    )

    assert not report.grounded
    assert "Unsupported numeric claim: 90 days" in report.errors


def test_grounding_accepts_actual_refund_number_from_a_tool_result() -> None:
    report = validate_answer_grounding(
        "The processed total is ₹1200.",
        policy_hits=[],
        tool_results=[{"processed_total": 1200}],
        policy_question=False,
    )

    assert report.grounded


def test_grounding_allows_a_safe_abstention_without_a_policy_citation() -> None:
    report = validate_answer_grounding(
        "I can’t confirm the applicable return window without the product category.",
        policy_hits=[_hit()],
        tool_results=[],
        policy_question=True,
    )

    assert report.grounded


def test_agent_does_not_guess_a_policy_when_m4_retrieval_is_not_configured() -> None:
    model = FakeChatModel([model_response("Electronics have a 90-day return window.")])

    result = ShopSenseAgent(model).run("What is the electronics return policy?")

    assert "not configured" in result.answer
    assert model.calls == []


class _PolicyService:
    def search_policy_documents(self, query: str) -> list[PolicyHit]:
        return [_hit()]

    def format_context(self, hits: list[PolicyHit]) -> str:
        return "Citation: [SOURCE:category-electronics:policy:markdown:0]\nElectronics have a 15-day return window."


def test_m4_policy_path_does_not_bind_m2_operational_tools() -> None:
    model = FakeChatModel(
        [model_response("Electronics can be returned within 15 days [SOURCE:category-electronics:policy:markdown:0].")]
    )
    agent = ShopSenseAgent(model, policy_service=_PolicyService())  # type: ignore[arg-type]

    result = agent.run("What is the electronics return policy?")

    assert "15 days" in result.answer
    # The initial binding configures the M2 path. M4 must not add another tool binding.
    assert model.bind_count == 1
