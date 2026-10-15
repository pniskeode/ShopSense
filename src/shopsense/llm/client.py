"""Small provider-agnostic LiteLLM client used by M1."""

from collections.abc import Callable
from collections.abc import Mapping, Sequence
import os
from typing import Any

DEFAULT_REPAIR_MAX_TOKENS = 250


class ModelConfigurationError(RuntimeError):
    """Raised when a required model name is absent from environment configuration."""


def configured_model(environment_name: str) -> str:
    """Load a required model setting from .env."""

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    model = os.getenv(environment_name, "").strip()
    if not model:
        raise ModelConfigurationError(
            f"{environment_name} is missing. Add it to .env before running ShopSense."
        )
    return model


class LiteLLMClient:
    """Wrap LiteLLM while keeping model/provider details out of intake code.

    ``completion_fn`` is injectable so deterministic tests never need a live provider.
    """

    def __init__(
        self,
        model: str | None = None,
        timeout: float | None = 60.0,
        completion_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model or configured_model("SHOPSENSE_M1_MODEL")
        self.timeout = timeout
        if completion_fn is None:
            from litellm import completion

            completion_fn = completion
        self._completion = completion_fn

    def complete(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Return the provider response text in JSON mode."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        return self._content_from_response(self._completion(**kwargs))

    def complete_text(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Return ordinary text without requesting provider JSON mode.

        Structured extraction uses :meth:`complete`; grounded customer-facing
        replies must remain plain text so citation markers are preserved exactly.
        """

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return self._content_from_response(self._completion(**kwargs))

    @staticmethod
    def _content_from_response(response: Any) -> str:
        """Extract one non-empty assistant response from a LiteLLM response."""

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            raise RuntimeError("LLM response did not contain choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM response content was empty")
        return content

    def chat(self, messages: Sequence[Mapping[str, Any]], tools: list[dict[str, Any]]) -> Any:
        """Run one tool-capable chat iteration and return the raw provider response."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "tools": tools,
            "tool_choice": "auto",
        }
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        return self._completion(**kwargs)

    def chat_model(self) -> Any:
        """Create the LangChain ChatLiteLLM model used by the M2 agent."""

        from langchain_litellm import ChatLiteLLM

        kwargs: dict[str, Any] = {"model": self.model}
        if self.timeout is not None:
            kwargs["request_timeout"] = self.timeout
        return ChatLiteLLM(**kwargs)
