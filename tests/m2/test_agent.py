from .helpers import FakeChatModel, model_response
from shopsense.agent.runner import MAX_ITERATIONS, ShopSenseAgent


def test_agent_executes_tool_then_answers_from_result() -> None:
    fake = FakeChatModel(
        [
            model_response(tool_calls=[("call-1", "lookup_order", {"order_ref": "KW-O-000000"})]),
            model_response("Your order KW-O-000000 is delivered."),
        ]
    )
    result = ShopSenseAgent(fake).run("Tell me about order KW-O-000000")

    assert result.answer == "Your order KW-O-000000 is delivered."
    assert result.iterations == 2
    assert [call.name for call in result.tool_calls] == ["lookup_order"]
    assert any(message.type == "tool" for message in fake.calls[1])


def test_agent_supports_multiple_tools_in_one_iteration() -> None:
    fake = FakeChatModel(
        [
            model_response(
                tool_calls=[
                    ("call-1", "lookup_order", {"order_ref": "KW-O-000000"}),
                    ("call-2", "track_shipment", {"order_ref": "KW-O-000000"}),
                ]
            ),
            model_response("The order is delivered and its shipment was scanned by Bluedart."),
        ]
    )
    result = ShopSenseAgent(fake).run("Where is order KW-O-000000?")

    assert result.answer.startswith("The order is delivered")
    assert [call.name for call in result.tool_calls] == ["lookup_order", "track_shipment"]


def test_agent_can_answer_without_tools() -> None:
    fake = FakeChatModel([model_response("I can help with ShopSense orders and refunds.")])
    result = ShopSenseAgent(fake).run("What can you help me with?")
    assert result.answer.startswith("I can help")
    assert result.tool_calls == []


def test_agent_feeds_unexpected_tool_error_back_to_model() -> None:
    fake = FakeChatModel(
        [
            model_response(tool_calls=[("call-1", "unknown_tool", {})]),
            model_response("I could not retrieve that information."),
        ]
    )
    result = ShopSenseAgent(fake).run("Check something")
    assert result.answer == "I could not retrieve that information."
    tool_messages = [message for message in fake.calls[1] if message.type == "tool"]
    assert '"error": "tool_error"' in tool_messages[0].content


def test_agent_stops_at_iteration_cap() -> None:
    fake = FakeChatModel(
        [model_response(tool_calls=[(f"call-{i}", "lookup_order", {"order_ref": "KW-O-000000"})]) for i in range(MAX_ITERATIONS)]
    )
    result = ShopSenseAgent(fake).run("Keep checking")
    assert result.iterations == MAX_ITERATIONS
    assert "six-step tool limit" in result.answer
    assert len(fake.calls) == MAX_ITERATIONS


def test_empty_question_is_handled_without_model_call() -> None:
    fake = FakeChatModel([])
    result = ShopSenseAgent(fake).run("   ")
    assert "provide a ShopSense" in result.answer
    assert fake.calls == []
