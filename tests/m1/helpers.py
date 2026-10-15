from types import SimpleNamespace


def llm_response(content: str) -> SimpleNamespace:
    """Build the small response shape consumed by LiteLLMClient."""

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )

