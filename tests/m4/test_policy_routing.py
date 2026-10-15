from shopsense.policy.service import is_policy_question


def test_natural_shipping_policy_questions_cannot_bypass_m4_retrieval() -> None:
    assert is_policy_question("When will standard delivery arrive?")
    assert is_policy_question("What does shipping normally take?")
    assert is_policy_question("How quickly will my package reach me?")


def test_live_status_and_customer_memory_questions_stay_on_their_authoritative_paths() -> None:
    assert not is_policy_question("Where is my order KW-O-12345?")
    assert not is_policy_question("What are my contact preferences?")
    assert not is_policy_question("What can you help me with?")
    assert not is_policy_question("I am angry and this is the worst service.")
    assert not is_policy_question("Can you recommend a product for my specific needs?")
