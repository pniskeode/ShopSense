"""Local FastAPI packaging for the ShopSense M8 demo chat endpoint."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from shopsense.m8 import M8IdempotencyError, M8Runtime, M8RuntimeError, create_m8_runtime
from shopsense.m8.evaluation import format_live_chat_record


class ChatRequest(BaseModel):
    """One live chat request; golden expectations are evaluator-only data."""

    model_config = ConfigDict(extra="forbid")

    customer_ref: Annotated[str | None, Field(default=None, max_length=80)] = None
    question: Annotated[str, Field(min_length=1, max_length=12_000)]
    message_id: Annotated[str | None, Field(default=None, min_length=1, max_length=120)] = None

    @field_validator("question", "message_id", mode="before")
    @classmethod
    def strip_non_empty_values(cls, value: object) -> object:
        if value is None:
            return value
        if not isinstance(value, str):
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("customer_ref", mode="before")
    @classmethod
    def strip_optional_customer_ref(cls, value: object) -> object:
        if value is None:
            return None
        if not isinstance(value, str):
            return value
        return value.strip() or None


def create_app(
    runtime_factory: Callable[[], M8Runtime] = create_m8_runtime,
):
    """Create the local app without opening external connections until `/chat` is called."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise RuntimeError("FastAPI is required for M8. Run `python -m pip install -e .`.") from exc

    app = FastAPI(title="ShopSense M8 Demo API", version="0.1.0", description="Local, policy-grounded ShopSense demo chat API.")
    app.state.runtime = None

    def runtime() -> M8Runtime:
        if app.state.runtime is None:
            app.state.runtime = runtime_factory()
        return app.state.runtime

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "shopsense-m8"}

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        raw_input = request.model_dump(mode="json", exclude_none=True)
        try:
            result = runtime().handle_chat(
                customer_ref=request.customer_ref,
                message=request.question,
                message_id=request.message_id,
            )
        except M8IdempotencyError as exc:
            raise HTTPException(status_code=409, detail="This message ID is already associated with another request.") from exc
        except M8RuntimeError as exc:
            raise HTTPException(status_code=503, detail="ShopSense is temporarily unavailable. Please try again later.") from exc
        except Exception as exc:
            raise HTTPException(status_code=503, detail="ShopSense could not process this request safely.") from exc
        return format_live_chat_record(raw_input, result)

    return app


app = create_app()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Uvicorn is required for M8. Run `python -m pip install -e .`.") from exc
    uvicorn.run("shopsense.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
