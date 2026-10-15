from shopsense.memory.rolling import RollingConversation


def test_rolling_summary_keeps_recent_turns_and_preserves_important_early_fact() -> None:
    conversation = RollingConversation(recent_keep=4, token_budget=80)
    conversation.add("user", "My name is Alex and I have a pet parrot named Kiwi.")
    conversation.add("assistant", "Nice to meet you, Alex.")
    for number in range(4):
        conversation.add("user", f"Explain order tracking in detail, part {number}. " * 5)
        conversation.add("assistant", "Detailed tracking explanation. " * 8)

    calls = []

    def summarize(turns, previous_summary):
        calls.append((turns, previous_summary))
        return "Alex has a pet parrot named Kiwi and asked about order tracking."

    assert conversation.compact(summarize) is True
    assert "Kiwi" in conversation.summary
    assert len(conversation.turns) == 4
    assert len(calls) == 1
