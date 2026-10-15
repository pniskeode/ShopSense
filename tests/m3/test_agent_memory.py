from shopsense.agent.runner import ShopSenseAgent

from tests.m2.helpers import FakeChatModel, model_response

from .helpers import RecordingMemoryService


def test_agent_adds_customer_memory_context_and_persists_completed_turn() -> None:
    model = FakeChatModel([model_response("Your delivery is still in transit.")])
    memory = RecordingMemoryService()
    agent = ShopSenseAgent(model, memory_service=memory)

    result = agent.run("Where is my delivery?", customer_ref="KW-C-11111")

    assert result.answer == "Your delivery is still in transit."
    assert memory.recalled == [("KW-C-11111", "Where is my delivery?")]
    assert memory.interactions == [
        ("KW-C-11111", "Where is my delivery?", "Your delivery is still in transit.")
    ]
    assert "Verified memory namespace: KW-C-11111" in model.calls[0][0].content
